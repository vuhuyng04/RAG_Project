"""Crawl product pages from the WooCommerce storefront.

Extraction targets were chosen from measured evidence, not guesswork. Sampling
10 random products and measuring the fraction of tokens shared by >=60% of
documents gave:

    product-short-description   96.3% shared   (5/10 unique)
    #tab-description           100.0% shared   (2/10 unique)
    meta[name=description]      15.3% shared  (10/10 unique)
    title                       27.0% shared  (10/10 unique)

The legacy pipeline embedded the first two. That is the mechanical cause of the
collapsed 0.68-0.71 cosine band. `meta[name=description]` is the discriminative
product copy and is what `clean.py` embeds.

Reproduce with: uv run python -m scripts.probe_boilerplate
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup

from rag.config import get_settings
from rag.ingest.schema import Product, is_product_url

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_LOC_RE = re.compile(r"<loc>(.*?)</loc>", re.DOTALL)


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------


async def fetch_sitemap_urls(client: httpx.AsyncClient, base_url: str) -> list[str]:
    """Collect every URL listed in the product sitemaps, unfiltered.

    Returned *unfiltered* on purpose. The legacy state has to see the same raw
    sitemap output the notebook saw — including the shop archive page
    /cua-hang/, which it indexed as a product with a missing title. Filtering
    here would make the legacy reproduction quietly better than the thing it is
    supposed to reproduce, and the "before" column would understate the problem.

    `clean.clean_products()` applies `is_product_url`; `legacy` does not.
    """
    index = await client.get(f"{base_url}/sitemap.xml")
    index.raise_for_status()
    children = [u for u in _LOC_RE.findall(index.text) if "product-sitemap" in u]
    log.info("Found %d product sitemaps", len(children))

    urls: list[str] = []
    for child in children:
        resp = await client.get(child)
        resp.raise_for_status()
        urls.extend(_LOC_RE.findall(resp.text))

    # Deduplicate while preserving order; sitemaps overlap in practice.
    seen: set[str] = set()
    unique = [u for u in urls if not (u in seen or seen.add(u))]
    n_non_product = sum(1 for u in unique if not is_product_url(u))
    log.info(
        "Sitemap: %d urls -> %d unique (%d are non-product pages, kept for the legacy state)",
        len(urls),
        len(unique),
        n_non_product,
    )
    return unique


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------


def _text(node) -> str:
    return node.get_text(" ", strip=True) if node else ""


def extract_price(soup: BeautifulSoup) -> str:
    """Return the *current* price, honouring WooCommerce sale markup.

    On sale, WooCommerce renders `<del>original</del><ins>current</ins>` and the
    combined text reads
    "1.400.000 ₫ Giá gốc là: 1.400.000₫. 1.200.000 ₫ Giá hiện tại là: ...".
    Taking the first amount yields the struck-through original, which would make
    every budget filter wrong on discounted items, so `<ins>` wins when present.
    """
    on_sale = soup.select_one(
        "p.price ins .woocommerce-Price-amount, span.price ins .woocommerce-Price-amount"
    )
    if on_sale:
        return _text(on_sale)
    regular = soup.select_one(
        "p.price .woocommerce-Price-amount, span.price .woocommerce-Price-amount"
    )
    return _text(regular)


def extract_image(soup: BeautifulSoup) -> str:
    """Prefer the gallery image.

    og:image on this site points at a site-wide banner
    (/wp-content/uploads/2023/12/banner-1_0.jpg) and is also a relative path, so
    it is a last resort rather than a first choice.
    """
    node = soup.select_one(".woocommerce-product-gallery img, img.wp-post-image")
    if node:
        src = node.get("data-large_image") or node.get("src") or ""
        if src.startswith("http"):
            return src
    og = soup.select_one("meta[property='og:image']")
    src = og.get("content", "") if og else ""
    return src if src.startswith("http") else ""


def parse_product(url: str, html: str) -> Product:
    soup = BeautifulSoup(html, "lxml")

    title = _text(soup.select_one("h1.product_title")) or _text(soup.select_one("h1"))

    # The discriminative copy. Falls back to the (boilerplate) short description
    # only when the meta tag is absent, so nothing is silently dropped.
    meta = soup.select_one("meta[name=description]")
    description = (meta.get("content", "") if meta else "") or _text(
        soup.select_one("div.product-short-description")
    )

    # Kept for display, deliberately NOT embedded: measured at 96-100% shared
    # tokens across products.
    promo = _text(soup.select_one("div.product-short-description"))

    return Product(
        url=url,
        title=title,
        price_raw=extract_price(soup),
        description=description,
        promo=promo,
        image=extract_image(soup),
        crawled_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _fetch_one(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> Product | None:
    async with sem:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return parse_product(url, resp.text)
        except Exception as exc:
            log.warning("Failed to crawl %s: %s: %s", url, type(exc).__name__, exc)
            return None


async def crawl_all(urls: list[str] | None = None, concurrency: int | None = None) -> list[Product]:
    s = get_settings()
    concurrency = concurrency or s.crawl_concurrency

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        follow_redirects=True,
        limits=limits,
    ) as client:
        if urls is None:
            urls = await fetch_sitemap_urls(client, s.crawl_base_url)

        sem = asyncio.Semaphore(concurrency)
        tasks = [_fetch_one(client, u, sem) for u in urls]

        products: list[Product] = []
        for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            if result is not None:
                products.append(result)
            if idx % 50 == 0:
                log.info("Crawled %d/%d", idx, len(tasks))

    log.info("Crawl complete: %d/%d pages parsed", len(products), len(urls))
    return products


def crawl_sync(urls: list[str] | None = None) -> list[Product]:
    return asyncio.run(crawl_all(urls))
