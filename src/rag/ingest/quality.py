"""Data-quality and freshness reporting, runnable against any corpus state.

The point is to make corpus health a number rather than an impression, so the
clean/corrupt/repaired comparison has something concrete to move.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.ingest.clean import boilerplate_ratio, build_embed_text
from rag.ingest.schema import Product

log = logging.getLogger(__name__)

# Anything embedding to one of these is a poisoned document — the exact failure
# the legacy pipeline shipped.
JUNK_TEXTS = {"nan", "none", "null", ""}


def _repeated_passage_rate(texts: list[str], threshold: float = 0.30) -> float:
    """Share of documents built substantially from corpus-wide repeated passages.

    Uses the same shingle detector as `repair.py`, so the quality report and the
    repair scoring cannot disagree about what boilerplate is.
    """
    from rag.ingest.repair import _shingles, detect_repeated_passages

    repeated = detect_repeated_passages(texts)
    if not repeated:
        return 0.0
    flagged = 0
    counted = 0
    for text in texts:
        shingles = _shingles(text)
        if not shingles:
            continue
        counted += 1
        if len(shingles & repeated) / len(shingles) >= threshold:
            flagged += 1
    return flagged / counted if counted else 0.0


def quality_report(
    products: list[Product],
    state: str,
    text_fn: Callable[[Product], str] | None = None,
) -> dict[str, Any]:
    """Profile a corpus state.

    `text_fn` must be the embed-text builder that this state actually indexes
    with. Defaulting every state to the clean builder would report identical
    numbers for legacy and clean and silently void the comparison — the legacy
    state's whole point is that its embedded text is different.
    """
    n = len(products)
    if n == 0:
        return {"state": state, "documents": 0}

    text_fn = text_fn or build_embed_text
    embed_texts = [text_fn(p) for p in products]

    null_counts = {
        "title": sum(1 for p in products if not p.title),
        "description": sum(1 for p in products if not p.description),
        "price_raw": sum(1 for p in products if not p.price_raw),
        "price_vnd": sum(1 for p in products if p.price_vnd is None),
        "image": sum(1 for p in products if not p.image),
    }

    url_counts = Counter(p.url for p in products)
    hash_counts = Counter(p.content_hash for p in products)

    junk = sum(1 for t in embed_texts if t.strip().lower() in JUNK_TEXTS)
    contains_nan_token = sum(1 for t in embed_texts if "nan" in t.lower().split())

    ages = [(datetime.now(UTC) - p.crawled_at.astimezone(UTC)).days for p in products]

    report = {
        "state": state,
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": n,
        "null_rate": {k: round(v / n, 4) for k, v in null_counts.items()},
        "null_count": null_counts,
        # The headline corpus-health metric: share of embedded tokens that
        # appear in >=60% of documents and therefore carry no signal.
        "boilerplate_ratio_embed_text": round(boilerplate_ratio(embed_texts), 4),
        # Share of documents containing a passage that recurs verbatim across
        # the corpus. Reported alongside the token-frequency ratio above because
        # the two answer different questions and the first one alone is
        # misleading: a 60%-document-frequency threshold cannot see boilerplate
        # injected into a *minority* of documents, so the corrupt state scored
        # *lower* than clean despite 25% of it being flooded on purpose.
        "repeated_passage_doc_rate": round(_repeated_passage_rate(embed_texts), 4),
        "boilerplate_ratio_promo": round(boilerplate_ratio([p.promo for p in products]), 4),
        "boilerplate_ratio_description": round(
            boilerplate_ratio([p.description for p in products]), 4
        ),
        "duplicate_urls": sum(c - 1 for c in url_counts.values() if c > 1),
        "duplicate_content_hashes": sum(c - 1 for c in hash_counts.values() if c > 1),
        "non_product_urls": sum(1 for p in products if not p.is_product),
        "price_parse_rate": round(1 - null_counts["price_vnd"] / n, 4),
        "junk_embed_text": junk,
        "embed_text_containing_nan_token": contains_nan_token,
        "freshness": {
            "oldest_days": max(ages) if ages else None,
            "newest_days": min(ages) if ages else None,
            "median_days": sorted(ages)[len(ages) // 2] if ages else None,
        },
    }
    return report


def assert_clean(report: dict[str, Any]) -> None:
    """Hard gate for the clean state.

    The literal-"nan" defect is the one that reached production last time. If it
    ever reappears in a state that claims to be clean, fail the build rather
    than let it silently into the index.
    """
    problems = []
    if report.get("junk_embed_text", 0) > 0:
        problems.append(f"{report['junk_embed_text']} documents embed to junk text")
    if report.get("embed_text_containing_nan_token", 0) > 0:
        problems.append(
            f"{report['embed_text_containing_nan_token']} documents contain a literal 'nan' token"
        )
    if report.get("non_product_urls", 0) > 0:
        problems.append(f"{report['non_product_urls']} non-product URLs present")
    if report.get("duplicate_urls", 0) > 0:
        problems.append(f"{report['duplicate_urls']} duplicate URLs")
    if problems:
        raise AssertionError("Clean state failed quality gate: " + "; ".join(problems))


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote quality report -> %s", path)
