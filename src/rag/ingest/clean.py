"""Build the text that actually gets embedded.

The most consequential module in the pipeline. Everything downstream — hybrid
search, reranking, prompt design — is bounded by what happens here, because no
retrieval strategy can separate documents whose embeddings encode mostly the
same shared text.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from rag.ingest.schema import Product, normalize_text

log = logging.getLogger(__name__)

# A token appearing in at least this share of documents carries no
# discriminative signal for this corpus and is a boilerplate candidate.
BOILERPLATE_DOC_FREQ = 0.60

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def detect_boilerplate_tokens(
    texts: list[str], threshold: float = BOILERPLATE_DOC_FREQ
) -> set[str]:
    """Tokens present in >= `threshold` of the documents.

    Measured on this corpus: the short description is 96.3% such tokens and the
    description tab is 100%, while the meta description is 15.3%. Used both to
    report a data-quality number and to strip residual boilerplate.
    """
    doc_sets = [set(_tokens(t)) for t in texts if t]
    if not doc_sets:
        return set()
    counts: Counter[str] = Counter()
    for tokens in doc_sets:
        counts.update(tokens)
    cutoff = threshold * len(doc_sets)
    return {tok for tok, n in counts.items() if n >= cutoff}


def boilerplate_ratio(texts: list[str], threshold: float = BOILERPLATE_DOC_FREQ) -> float:
    """Share of all tokens that are boilerplate. A corpus-health metric."""
    shared = detect_boilerplate_tokens(texts, threshold)
    doc_sets = [set(_tokens(t)) for t in texts if t]
    total = sum(len(d) for d in doc_sets)
    if not total:
        return 0.0
    return sum(len(d & shared) for d in doc_sets) / total


def build_embed_text(product: Product, *, include_price_band: bool | None = None) -> str:
    """Compose the string sent to the embedding model.

    Deliberately excludes `promo` (byte-identical across the catalogue) and the
    image URL (appending 'xem ảnh tại <url>' to every document contributes
    tokens that match every query equally).

    The price band is behind a flag. It plausibly helps budget-phrased queries
    ('tầm 1 triệu') by giving them something to align with, but six bands across
    ~660 products means ~110 products share the phrase verbatim — a milder form
    of the very problem being fixed here. Whether it earns its place is an
    empirical question the eval matrix answers, so it is not assumed either way.
    """
    if include_price_band is None:
        from rag.config import get_settings

        include_price_band = get_settings().embed_price_band

    parts = [product.title]
    if include_price_band:
        parts.append(product.price_band())
    parts.append(product.description)
    return normalize_text(" . ".join(p for p in parts if p))


def clean_products(products: list[Product]) -> tuple[list[Product], dict[str, int]]:
    """Filter and deduplicate crawled records.

    Returns the kept products and a breakdown of what was dropped and why, so
    the numbers land in the quality report instead of vanishing.
    """
    stats = {
        "input": len(products),
        "dropped_non_product_url": 0,
        "dropped_no_title": 0,
        "dropped_duplicate": 0,
        "missing_price": 0,
        "missing_description": 0,
    }

    kept: list[Product] = []
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()

    for p in products:
        if not p.is_product:
            stats["dropped_non_product_url"] += 1
            continue
        # A record with no title is not a usable product. Letting nulls through
        # and stringifying them later yields the literal "nan" in the index.
        if not p.title:
            stats["dropped_no_title"] += 1
            continue
        if p.url in seen_urls or p.content_hash in seen_hashes:
            stats["dropped_duplicate"] += 1
            continue

        seen_urls.add(p.url)
        seen_hashes.add(p.content_hash)
        if p.price_vnd is None:
            stats["missing_price"] += 1
        if not p.description:
            stats["missing_description"] += 1
        kept.append(p)

    stats["kept"] = len(kept)
    log.info("Clean: %s", stats)
    return kept, stats
