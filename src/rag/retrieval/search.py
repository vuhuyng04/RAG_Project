"""Retrieval configurations.

Each `RetrievalConfig` differs from the baseline in as few variables as
possible, so an A/B delta is attributable. The eval matrix runs
(config x corpus state) and every number is tagged with both.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import models

from rag.clients import embed, get_qdrant
from rag.config import State, get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    top_k: int = 5
    fetch_k: int = 20
    use_rerank: bool = False
    use_budget_filter: bool = False
    use_hybrid: bool = False
    # None means "never abstain", which is what the legacy system did: it
    # returned exactly top_k neighbours for every query including
    # "có freeship không", and the prompt then upsold them.
    score_threshold: float | None = None
    description: str = ""


# The A/B ladder. Each step adds exactly one mechanism.
CONFIGS: dict[str, RetrievalConfig] = {
    "legacy": RetrievalConfig(
        name="legacy",
        top_k=5,
        score_threshold=None,
        description="Reproduces the original app: plain dense top-5, no threshold, no rerank.",
    ),
    "dense": RetrievalConfig(
        name="dense",
        top_k=5,
        score_threshold=None,
        description="Dense retrieval on the new pipeline. Isolates the corpus fix.",
    ),
    "dense_threshold": RetrievalConfig(
        name="dense_threshold",
        top_k=5,
        score_threshold=0.60,
        description="Dense + abstention threshold. Isolates the effect of refusing to answer.",
    ),
    "dense_rerank": RetrievalConfig(
        name="dense_rerank",
        top_k=5,
        fetch_k=20,
        use_rerank=True,
        score_threshold=0.60,
        description="Fetch 20, cross-encoder rerank to 5. Isolates reranking.",
    ),
    "dense_budget": RetrievalConfig(
        name="dense_budget",
        top_k=5,
        use_budget_filter=True,
        score_threshold=None,
        description=(
            "Dense + budget filter, no rerank. The control cell for hybrid_budget: "
            "without it, a hybrid_budget gain cannot be attributed to hybrid rather "
            "than to the filter."
        ),
    ),
    "hybrid": RetrievalConfig(
        name="hybrid",
        top_k=5,
        fetch_k=20,
        use_hybrid=True,
        score_threshold=None,
        description="BM25 sparse + dense, fused with RRF. Isolates lexical matching.",
    ),
    "hybrid_budget": RetrievalConfig(
        name="hybrid_budget",
        top_k=5,
        fetch_k=20,
        use_hybrid=True,
        use_budget_filter=True,
        score_threshold=None,
        description="Hybrid + budget filter. No reranker (it cost 10x latency for no gain).",
    ),
    "full": RetrievalConfig(
        name="full",
        top_k=5,
        fetch_k=20,
        use_rerank=True,
        use_budget_filter=True,
        score_threshold=0.60,
        description="Rerank + budget filter from the parsed price payload index.",
    ),
}


# ---------------------------------------------------------------------------
# Budget extraction
# ---------------------------------------------------------------------------

_BUDGET_RE = re.compile(
    r"(?:dưới|duoi|tầm|tam|khoảng|khoang|under|<=?)\s*"
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(k|nghìn|nghin|ngàn|ngan|tr|triệu|trieu|m)\b",
    re.IGNORECASE,
)
_BARE_K_RE = re.compile(r"\b(\d{2,4})\s*k\b", re.IGNORECASE)


def extract_budget_vnd(question: str) -> int | None:
    """Parse a budget from natural Vietnamese.

    Handles 'dưới 500k', 'tầm 1 triệu', 'khoảng 1.5tr' and the bare 'hoa khai
    trương 500k' form that shows up constantly in the harvested real queries.
    """
    m = _BUDGET_RE.search(question)
    if m:
        amount = float(m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        if unit in {"k", "nghìn", "nghin", "ngàn", "ngan"}:
            return int(amount * 1_000)
        return int(amount * 1_000_000)

    m = _BARE_K_RE.search(question)
    if m:
        return int(m.group(1)) * 1_000
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    url: str
    title: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
    retrieved_by: str = "dense"


@dataclass
class SearchResult:
    hits: list[Hit]
    abstained: bool
    latency_ms: float
    budget_applied: int | None = None
    timings: dict[str, float] = field(default_factory=dict)


def search(
    question: str,
    state: State,
    config: RetrievalConfig,
) -> SearchResult:
    s = get_settings()
    client = get_qdrant()
    collection = s.collection_for(state)

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    vector = embed(question)
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000
    if not vector:
        return SearchResult([], abstained=True, latency_ms=0.0, timings=timings)

    query_filter = None
    budget = None
    if config.use_budget_filter:
        budget = extract_budget_vnd(question)
        if budget:
            # price_vnd is None for products with no parseable price; those are
            # excluded rather than assumed affordable.
            query_filter = models.Filter(
                must=[models.FieldCondition(key="price_vnd", range=models.Range(lte=budget))]
            )

    limit = config.fetch_k if (config.use_rerank or config.use_hybrid) else config.top_k
    t0 = time.perf_counter()

    if config.use_hybrid:
        from rag.retrieval.sparse import SPARSE_VECTOR_NAME, encode_query

        # Reciprocal Rank Fusion is done server-side: Qdrant runs both
        # prefetches and merges by rank. Fusing on rank rather than score is
        # what makes it safe to combine cosine similarity with BM25 term
        # weights, which are not on a comparable scale.
        points = client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=vector, limit=limit, filter=query_filter),
                models.Prefetch(
                    query=encode_query(question),
                    using=SPARSE_VECTOR_NAME,
                    limit=limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points
        retrieved_by = "hybrid_rrf"
    else:
        points = client.query_points(
            collection,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points
        retrieved_by = "dense"

    timings["search_ms"] = (time.perf_counter() - t0) * 1000

    hits = [
        Hit(
            url=(p.payload or {}).get("url", ""),
            title=(p.payload or {}).get("title") or "",
            score=p.score,
            payload=p.payload or {},
            retrieved_by=retrieved_by,
        )
        for p in points
    ]

    if config.use_hybrid:
        # RRF produces many exact ties — a document at rank r in one list and
        # absent from the other always scores 1/(60+r), so whole groups share a
        # score and their relative order comes out of Qdrant's internal
        # ordering. Measured across three identical runs that left recall@5
        # stable at 0.417 while MRR@5 moved 0.611 -> 0.694 and nDCG@5 moved
        # 0.517 -> 0.541: rank-sensitive metrics were not reproducible.
        #
        # Breaking ties on URL is arbitrary but *stable*, which is what a
        # committed evaluation number requires. Dense retrieval needs no such
        # fix — its float scores rarely tie.
        hits.sort(key=lambda h: (-h.score, h.url))

    if config.use_rerank and hits:
        t0 = time.perf_counter()
        hits = _rerank(question, hits)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # Threshold is applied to the dense cosine score, not the reranker score:
    # the two are on different scales and calibrating one threshold across both
    # would silently change meaning between configs.
    #
    # It is also why the hybrid configs carry `score_threshold=None`. RRF
    # returns a rank-derived score (roughly 1/(60+rank), so ~0.016 at best),
    # nowhere near the 0.60 cosine threshold — applying it would abstain on
    # every single query. Abstention for hybrid needs its own calibration
    # against RRF scores, which is left for when the golden set is large enough
    # to calibrate on without overfitting (docs/decisions.md D10).
    abstained = False
    if config.score_threshold is not None:
        kept = [h for h in hits if h.score >= config.score_threshold]
        if not kept:
            abstained = True
        hits = kept

    hits = hits[: config.top_k]
    latency = (time.perf_counter() - t_start) * 1000
    return SearchResult(
        hits=hits,
        abstained=abstained or not hits,
        latency_ms=latency,
        budget_applied=budget,
        timings=timings,
    )


def _rerank(question: str, hits: list[Hit]) -> list[Hit]:
    from rag.clients import get_reranker

    model = get_reranker()
    pairs = [(question, f"{h.title}. {(h.payload.get('description') or '')[:400]}") for h in hits]
    scores = model.predict(pairs)
    for h, sc in zip(hits, scores, strict=True):
        h.rerank_score = float(sc)
        h.retrieved_by = "dense+rerank"
    return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
