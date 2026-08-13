"""Run deterministic retrieval evaluation over (config x corpus state).

    uv run python -m eval.run_retrieval                       # full matrix
    uv run python -m eval.run_retrieval --states baseline clean --configs baseline dense

No LLM calls, so the whole matrix is free and repeatable — these are the metrics
the headline claims rest on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime

from eval.metrics import QueryOutcome, aggregate
from rag.clients import get_qdrant
from rag.config import DATASET_DIR, RESULTS_DIR, State, get_settings
from rag.evalset.label import load_golden
from rag.evalset.schema import GoldenQuery
from rag.retrieval.search import CONFIGS, RetrievalConfig, search

log = logging.getLogger("run_retrieval")

# Which retrieval config is meaningful for which corpus state. Crossing them is
# possible but muddles attribution, so the default matrix pairs them
# deliberately.
DEFAULT_MATRIX: list[tuple[str, str]] = [
    ("baseline", "baseline"),  # control: naive concatenation, unfiltered corpus
    ("dense", "clean"),  # isolates the ingestion design
    ("dense_threshold", "clean"),  # isolates abstention
    ("dense_budget", "clean"),  # isolates budget filtering
    ("dense_rerank", "clean"),  # isolates reranking
    ("hybrid", "clean"),  # isolates lexical matching
    ("full", "clean"),  # all mechanisms combined
]


def collection_urls(state: State) -> set[str]:
    """Every URL present in a state's collection.

    Used to separate "the retriever missed it" from "it is not in the index",
    which is the distinction the corrupt/repaired comparison exists to make.
    """
    s = get_settings()
    client = get_qdrant()
    name = s.collection_for(state)
    if not client.collection_exists(name):
        return set()

    urls: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name, limit=512, offset=offset, with_payload=["url"], with_vectors=False
        )
        urls.update((p.payload or {}).get("url", "") for p in points)
        if offset is None:
            break
    urls.discard("")
    return urls


def evaluate(
    golden: list[GoldenQuery], state: State, config: RetrievalConfig
) -> tuple[dict, list[dict]]:
    present = collection_urls(state)
    outcomes: list[QueryOutcome] = []
    records: list[dict] = []

    for gq in golden:
        result = search(gq.question, state, config)
        retrieved = [h.url for h in result.hits]
        gold = set(gq.gold_urls)

        outcomes.append(
            QueryOutcome(
                query_id=gq.id,
                question=gq.question,
                gold_urls=gold,
                retrieved_urls=retrieved,
                answerable=gq.answerable,
                abstained=result.abstained,
                latency_ms=result.latency_ms,
                missing_golds=gold - present if present else set(),
                slices=gq.slice_keys(),
            )
        )
        records.append(
            {
                "id": gq.id,
                "question": gq.question,
                "intent": gq.intent.value,
                "answerable": gq.answerable,
                "gold_urls": gq.gold_urls,
                "retrieved": [
                    {"url": h.url, "title": h.title, "score": round(h.score, 4)}
                    for h in result.hits
                ],
                "abstained": result.abstained,
                "latency_ms": round(result.latency_ms, 1),
                "budget_applied": result.budget_applied,
            }
        )

    return aggregate(outcomes), records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--golden", default=str(DATASET_DIR / "golden.jsonl"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    golden = load_golden(args.golden)
    n_unreviewed = sum(1 for g in golden if not g.reviewed)
    log.info(
        "Loaded %d golden queries (%d answerable)",
        len(golden),
        sum(1 for g in golden if g.answerable),
    )
    if n_unreviewed:
        log.warning(
            "%d/%d queries are reviewed=false — numbers are PROVISIONAL until a human pass.",
            n_unreviewed,
            len(golden),
        )

    if args.configs or args.states:
        pairs = [(c, s) for c in (args.configs or CONFIGS) for s in (args.states or ["clean"])]
    else:
        pairs = DEFAULT_MATRIX

    summary: dict[str, dict] = {}
    for config_name, state_name in pairs:
        config = CONFIGS[config_name]
        state = State(state_name)
        cell = f"{config_name}@{state_name}"
        log.info("--- %s ---", cell)

        report, records = evaluate(golden, state, config)
        report["config"] = config_name
        report["state"] = state_name
        report["config_description"] = config.description
        report["golden_reviewed"] = n_unreviewed == 0
        report["generated_at"] = datetime.now(UTC).isoformat()
        summary[cell] = report

        out = RESULTS_DIR / f"retrieval_{config_name}_{state_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (RESULTS_DIR / "per_query").mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "per_query" / f"{config_name}_{state_name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        raw = report["retrieval_raw"]
        log.info(
            "  recall@5=%.3f  mrr@5=%.3f  ndcg@5=%.3f  abstain_f1=%.3f  p95=%.0fms  (n=%s)",
            raw.get("recall@5", 0),
            raw.get("mrr@5", 0),
            raw.get("ndcg@5", 0),
            report["abstention"]["f1"],
            report["latency_ms"]["p95"],
            raw.get("n", 0),
        )

    (RESULTS_DIR / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote %d cells -> %s", len(summary), RESULTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
