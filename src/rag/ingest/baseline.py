"""Naive concatenation baseline.

The control condition for the ingestion experiment. It builds one embedding
string by concatenating every available field — title, price, description,
promotional copy, image URL — and applies no URL filtering and no null handling.

This is a plausible first implementation, not a straw man: concatenating
everything is what most quick RAG pipelines do, and on a catalogue with genuinely
distinct product copy it works acceptably. On *this* catalogue it fails in two
specific, measurable ways, which is precisely why it is worth measuring rather
than assuming:

1. The store's promotional text is byte-identical across products and the
   descriptions are near-identical marketing boilerplate, so most of every
   embedding string is text shared with every other document.
2. Missing fields propagate: a null anywhere in the concatenation poisons the
   whole string, and stringifying it yields the literal token ``"nan"`` — which
   then gets embedded and indexed as though it were product copy.

Both effects are quantified in `eval/results/quality_baseline.json`.

Built from the **same frozen crawl** as every other corpus state, so the
comparison isolates the pipeline and holds the raw data constant.
"""

from __future__ import annotations

import logging

from rag.ingest.schema import Product

log = logging.getLogger(__name__)


def baseline_embed_text(product: Product) -> str:
    """Concatenate every field into one embedding string.

    Null handling is deliberately absent. A missing value in any field collapses
    the whole record to ``"nan"``, mirroring how a dataframe pipeline behaves
    when a concatenation meets a null and the result is later stringified.
    """
    columns = [product.title, product.price_raw, product.description, product.promo]
    if any(not c for c in columns):
        return "nan"

    return (
        f"{product.title} giá {product.price_raw}"
        f" mô tả sản phẩm: {product.description}"
        f" khuyến mãi: {product.promo}"
        f" xem ảnh tại {product.image}"
    )


def baseline_payload(product: Product) -> dict[str, object]:
    """Payload without the parsed price field.

    Price stays a formatted string (``"1.950.000₫"``), which is what makes
    budget filtering impossible under this configuration — the single largest
    measured difference in the retrieval A/B ladder.
    """
    return {
        "url": product.url,
        "title": product.title,
        "price": product.price_raw,
        "description": product.description,
        "promo": product.promo,
        "image": product.image,
    }


def build_baseline_records(products: list[Product]) -> list[Product]:
    """Pass everything through unfiltered.

    No URL filtering, so catalogue archive and cart pages are indexed alongside
    products. `baseline_embed_text` does the rest.
    """
    n_nan = sum(1 for p in products if baseline_embed_text(p) == "nan")
    n_non_product = sum(1 for p in products if not p.is_product)
    log.info(
        "Baseline state: %d records, %d embed as literal 'nan', %d non-product URLs kept",
        len(products),
        n_nan,
        n_non_product,
    )
    return list(products)
