"""Build one corpus state: crawl (once, cached) -> transform -> index.

    uv run python -m scripts.build_state --state clean
    uv run python -m scripts.build_state --state legacy
    uv run python -m scripts.build_state --state corrupt --seed 42
    uv run python -m scripts.build_state --state repaired

The crawl is cached to data/raw/products.jsonl and reused by every state, so all
four are built from byte-identical raw data.
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag.config import DATA_DIR, RESULTS_DIR, STATES_DIR, State, get_settings
from rag.ingest.clean import clean_products
from rag.ingest.crawl import crawl_sync
from rag.ingest.index import builders_for, index_state
from rag.ingest.legacy import build_legacy_records
from rag.ingest.quality import assert_clean, quality_report, save_report
from rag.ingest.store import load_products, save_products

RAW_PATH = DATA_DIR / "raw" / "products.jsonl"

log = logging.getLogger("build_state")


def get_raw(refresh: bool = False):
    """Crawl once and cache. Every state derives from this same snapshot."""
    if RAW_PATH.exists() and not refresh:
        products = load_products(RAW_PATH)
        log.info("Loaded %d cached raw records from %s", len(products), RAW_PATH)
        return products
    log.info("Crawling %s ...", get_settings().crawl_base_url)
    products = crawl_sync()
    save_products(products, RAW_PATH)
    return products


def build(state: State, seed: int, refresh: bool, no_index: bool) -> None:
    raw = get_raw(refresh)

    if state is State.LEGACY:
        # Deliberately unfiltered: the notebook indexed whatever the sitemaps
        # returned, archive pages included.
        records = build_legacy_records(raw)

    elif state is State.CLEAN:
        records, stats = clean_products(raw)
        log.info("Clean filter stats: %s", stats)

    elif state is State.CORRUPT:
        from rag.ingest.corrupt import corrupt_products, save_manifest

        clean_records, _ = clean_products(raw)
        records, manifest = corrupt_products(clean_records, seed=seed)
        save_manifest(manifest, STATES_DIR / "corruption_manifest.json")

    elif state is State.REPAIRED:
        from rag.ingest.corrupt import load_manifest
        from rag.ingest.repair import repair_products, score_detection

        corrupt_records = load_products(STATES_DIR / "corrupt.jsonl")
        manifest = load_manifest(STATES_DIR / "corruption_manifest.json")
        records, detected = repair_products(corrupt_records)
        scores = score_detection(detected, manifest)
        save_report(scores, RESULTS_DIR / "repair_detection.json")
        log.info(
            "Repair detection: macro_f1=%.3f overall_f1=%.3f",
            scores["macro_f1"],
            scores["overall_any_defect"]["f1"],
        )
        for label, s in scores["per_defect"].items():
            log.info(
                "   %-18s P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)",
                label,
                s["precision"],
                s["recall"],
                s["f1"],
                s["tp"],
                s["fp"],
                s["fn"],
            )

    else:  # pragma: no cover
        raise ValueError(state)

    save_products(records, STATES_DIR / f"{state.value}.jsonl")

    # Profile with the same text builder this state indexes with, otherwise
    # legacy would be reported using the clean builder and look identical.
    text_fn, _ = builders_for(state)
    report = quality_report(records, state.value, text_fn=text_fn)
    save_report(report, RESULTS_DIR / f"quality_{state.value}.json")
    log.info(
        "state=%s docs=%d boilerplate=%.1f%% price_parse=%.1f%% junk=%d nan_token=%d",
        state.value,
        report["documents"],
        report["boilerplate_ratio_embed_text"] * 100,
        report["price_parse_rate"] * 100,
        report["junk_embed_text"],
        report["embed_text_containing_nan_token"],
    )

    if state is State.CLEAN:
        assert_clean(report)
        log.info("Clean state passed the quality gate.")

    if no_index:
        log.info("--no-index set; skipping Qdrant upsert.")
        return

    result = index_state(records, state)
    log.info("Indexed: %s", result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=[s.value for s in State])
    parser.add_argument("--seed", type=int, default=42, help="corruption seed")
    parser.add_argument("--refresh", action="store_true", help="re-crawl instead of using cache")
    parser.add_argument("--no-index", action="store_true", help="build records but skip Qdrant")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    build(State(args.state), args.seed, args.refresh, args.no_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
