"""Faithful reproduction of the original notebook's indexing logic.

This is the "before" column. It is a reproduction rather than a snapshot
because the cluster the notebook wrote to is gone (see docs/decisions.md D3),
and building it from the *same crawl* as the clean state makes the comparison a
controlled experiment: raw data held constant, only the pipeline varies.

Do not "improve" anything in here. Every wart is load-bearing evidence.
Reference: craw_data.ipynb cells 14-16.
"""

from __future__ import annotations

import logging

from rag.ingest.schema import Product

log = logging.getLogger(__name__)


def legacy_embed_text(product: Product) -> str:
    """Reproduce cell-14's content string, bugs included.

    The original:

        df['content'] = df['title'] + ' giá ' + df['price']
                      + ' mô tả sản phẩm: ' + df['description']
                      + ' khuyêt mãi: ' + df['khuyen_mai']
                      + ' xem ảnh tại ' + df['image'].fillna('')

    Three defects are reproduced deliberately:

    1. Only `image` got `.fillna('')`. Any NaN in another column made the whole
       concatenation NaN, and cell-15's `.astype(str)` then turned it into the
       literal string "nan", which was embedded and upserted. Those points were
       live and retrievable.
    2. `khuyen_mai` (byte-identical across all products) and the boilerplate
       description are embedded, which is what collapsed the cosine band.
    3. The typo 'khuyêt mãi' is kept — it was in the embedded text.
    """
    # Emulate pandas NaN propagation: in the original, a missing value in any of
    # these columns poisoned the entire row.
    columns = [product.title, product.price_raw, product.description, product.promo]
    if any(not c for c in columns):
        return "nan"

    return (
        f"{product.title} giá {product.price_raw}"
        f" mô tả sản phẩm: {product.description}"
        f" khuyêt mãi: {product.promo}"
        f" xem ảnh tại {product.image}"
    )


def legacy_payload(product: Product) -> dict[str, object]:
    """Reproduce cell-16's payload, including the always-None `id`.

    The notebook wrote `int(row.get("id")) if not pd.isna(row.get("id")) else None`
    against a DataFrame that had no `id` column, so every one of the 741 points
    carried `id: None`.
    """
    return {
        "url": product.url,
        "title": product.title,
        "price": product.price_raw,
        "description": product.description,
        "khuyen_mai": product.promo,
        "image": product.image,
        "id": None,
    }


def build_legacy_records(products: list[Product]) -> list[Product]:
    """Legacy indexed everything the sitemaps returned — no filtering at all.

    Passed through unchanged; `legacy_embed_text` does the damage.
    """
    n_nan = sum(1 for p in products if legacy_embed_text(p) == "nan")
    n_non_product = sum(1 for p in products if not p.is_product)
    log.info(
        "Legacy state: %d records, %d will embed as literal 'nan', %d non-product URLs kept",
        len(products),
        n_nan,
        n_non_product,
    )
    return list(products)
