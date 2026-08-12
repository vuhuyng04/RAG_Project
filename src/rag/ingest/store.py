"""Persist crawled records to disk as JSONL.

The crawl is the slow, network-bound, non-deterministic step. Freezing its
output means every downstream state (legacy, clean, corrupt, repaired) is built
from byte-identical raw data — which is what makes the comparison a controlled
experiment rather than four different crawls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag.ingest.schema import Product

log = logging.getLogger(__name__)


def save_products(products: list[Product], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in products:
            fh.write(p.model_dump_json() + "\n")
    log.info("Wrote %d records -> %s", len(products), path)


def load_products(path: Path) -> list[Product]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python -m scripts.build_state --state clean"
        )
    with path.open("r", encoding="utf-8") as fh:
        return [Product(**json.loads(line)) for line in fh if line.strip()]
