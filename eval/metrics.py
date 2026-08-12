"""Deterministic retrieval metrics.

No LLM calls, so these run on every cell of the evaluation matrix for free and
are the metrics the headline claims rest on. RAGAS covers the generation half
separately and only on a subset (free-tier budget, see eval/README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Share of gold documents present in the top-k.

    Set-based rather than hit-or-miss: a query with 4 acceptable products
    should not score 1.0 for finding only one of them.
    """
    if not gold:
        return 0.0
    top = retrieved[:k]
    return len(gold & set(top)) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold document appears in the top-k."""
    if not gold:
        return 0.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    for rank, url in enumerate(retrieved[:k], 1):
        if url in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, url in enumerate(retrieved[:k], 1) if url in gold)
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(round(pct / 100 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


@dataclass
class QueryOutcome:
    """One query's result under one configuration."""

    query_id: str
    question: str
    gold_urls: set[str]
    retrieved_urls: list[str]
    answerable: bool
    abstained: bool
    latency_ms: float
    slices: dict[str, str] = field(default_factory=dict)

    # Gold documents that are not in the collection being evaluated at all.
    # Non-empty in the corrupt state by construction — corruption deletes and
    # mangles documents on purpose.
    missing_golds: set[str] = field(default_factory=set)

    @property
    def present_golds(self) -> set[str]:
        return self.gold_urls - self.missing_golds


def aggregate(outcomes: list[QueryOutcome], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Aggregate metrics, reporting retrieval both raw and gold-restricted.

    The two views answer different questions and conflating them is how the
    three-state experiment would lose its meaning:

    * **raw** counts every labelled gold, including ones corruption removed
      from the index. It is the honest end-to-end number — a user does not care
      why the answer is missing.
    * **restricted** counts only golds that still exist in the collection. It
      isolates "did the retriever find what was there" from "was it there".

    In the clean state the two are identical. Where they diverge, the gap is
    exactly the damage corruption did to the corpus rather than to the
    retriever.
    """
    answerable = [o for o in outcomes if o.answerable and o.gold_urls]
    unanswerable = [o for o in outcomes if not o.answerable]

    def _retrieval_block(items: list[QueryOutcome], restricted: bool) -> dict:
        if not items:
            return {}
        block: dict[str, float] = {}
        for k in ks:
            golds = [(o.present_golds if restricted else o.gold_urls) for o in items]
            pairs = [(o.retrieved_urls, g) for o, g in zip(items, golds, strict=True) if g]
            if not pairs:
                continue
            n = len(pairs)
            block[f"recall@{k}"] = round(sum(recall_at_k(r, g, k) for r, g in pairs) / n, 4)
            block[f"hit@{k}"] = round(sum(hit_at_k(r, g, k) for r, g in pairs) / n, 4)
            block[f"mrr@{k}"] = round(sum(mrr_at_k(r, g, k) for r, g in pairs) / n, 4)
            block[f"ndcg@{k}"] = round(sum(ndcg_at_k(r, g, k) for r, g in pairs) / n, 4)
        block["n"] = len(items)
        return block

    # Abstention: the system should stay silent on unanswerable queries and
    # answer the answerable ones. Reported as precision/recall over "abstained".
    tp = sum(1 for o in unanswerable if o.abstained)
    fp = sum(1 for o in outcomes if o.answerable and o.abstained)
    fn = sum(1 for o in unanswerable if not o.abstained)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    latencies = [o.latency_ms for o in outcomes]
    n_missing = sum(1 for o in outcomes if o.missing_golds)

    result = {
        "n_queries": len(outcomes),
        "retrieval_raw": _retrieval_block(answerable, restricted=False),
        "retrieval_restricted_to_present_golds": _retrieval_block(answerable, restricted=True),
        "gold_integrity": {
            "queries_with_missing_golds": n_missing,
            "total_missing_gold_docs": sum(len(o.missing_golds) for o in outcomes),
        },
        "abstention": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_unanswerable": len(unanswerable),
            "false_abstentions": fp,
        },
        "latency_ms": {
            "p50": round(median(latencies), 1) if latencies else 0.0,
            "p95": round(percentile(latencies, 95), 1),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        },
    }

    # Per-slice breakdown. An averaged headline hides that synthetic queries are
    # the easy slice; the sliced view is what belongs on a CV.
    slices: dict[str, dict[str, dict]] = {}
    for key in {k for o in outcomes for k in o.slices}:
        buckets: dict[str, list[QueryOutcome]] = {}
        for o in outcomes:
            if key in o.slices:
                buckets.setdefault(o.slices[key], []).append(o)
        slices[key] = {
            value: _retrieval_block(
                [o for o in items if o.answerable and o.gold_urls], restricted=False
            )
            for value, items in sorted(buckets.items())
        }
    result["slices"] = slices
    return result
