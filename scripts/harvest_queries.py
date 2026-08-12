"""Harvest raw candidate queries -> data/evalset/raw_queries.json.

    uv run python -m scripts.harvest_queries

Cached so the (network-bound, non-deterministic) harvest happens once and every
downstream labelling run sees identical input.
"""

from __future__ import annotations

import json
import logging
import sys

from rag.config import DATA_DIR
from rag.evalset.harvest import SIDEBAR_QUERIES, harvest_autocomplete

OUT = DATA_DIR / "evalset" / "raw_queries.json"

log = logging.getLogger("harvest")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    autocomplete = harvest_autocomplete()
    payload = {
        "autocomplete": autocomplete,
        "app_sidebar": list(SIDEBAR_QUERIES),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info(
        "Wrote %d autocomplete + %d sidebar queries -> %s",
        len(autocomplete),
        len(SIDEBAR_QUERIES),
        OUT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
