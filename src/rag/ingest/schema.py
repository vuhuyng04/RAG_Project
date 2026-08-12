"""The `Product` record — single source of truth for every stage.

Crawl, clean, corrupt, repair and index all speak this type. Corruption and
repair operate on *validated records*, never on raw HTML (see plan §3), which
is what makes the corruption manifest meaningful as ground truth.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Non-product URL detection
# ---------------------------------------------------------------------------

# Feeding raw sitemap URLs straight into a crawler indexes pages like
# https://hoatuoimymy.com/cua-hang/ (the WooCommerce shop archive) as though
# they were products. These are the page families that are not.
NON_PRODUCT_SEGMENTS: frozenset[str] = frozenset(
    {
        "cua-hang",  # shop archive
        "danh-muc",  # category
        "danh-muc-san-pham",
        "product-category",
        "shop",
        "page",
        "gio-hang",  # cart
        "thanh-toan",  # checkout
        "tai-khoan",  # account
        "lien-he",  # contact
        "gioi-thieu",  # about
        "tin-tuc",  # news
        "blog",
        "tag",
        "author",
        "feed",
        "wp-content",
        "wp-json",
    }
)


def is_product_url(url: str) -> bool:
    """True if `url` looks like an individual product page.

    WooCommerce product permalinks on this site are flat: /<product-slug>/.
    Anything with a known archive/system segment, a query string, or an empty
    path is not a product.
    """
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url)
    if parsed.query:
        return False
    segments = [s for s in parsed.path.split("/") if s]
    if not segments:
        return False
    if any(s.lower() in NON_PRODUCT_SEGMENTS for s in segments):
        return False
    # Pagination like /gio-hoa/page/2/ is caught above; a bare numeric tail is
    # also archive-ish rather than a product slug.
    return not segments[-1].isdigit()


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------

_PRICE_CLEAN_RE = re.compile(r"[^\d]")
_PRICE_TOKEN_RE = re.compile(r"\d[\d.,\s]*")

# Below this a "price" is almost certainly a stray number (a product code, a
# percentage) rather than VND. The cheapest real items on this site are ~100k.
MIN_PLAUSIBLE_VND = 10_000
MAX_PLAUSIBLE_VND = 500_000_000


def parse_price_vnd(raw: str | None) -> int | None:
    """Parse a Vietnamese price string into an integer VND amount.

    Returns the **smallest plausible amount** in the string. Handles
    '1.950.000₫', '1,950,000 đ' and variable-product ranges like
    '950.000 - 1.500.000₫'; returns None for 'Liên hệ', blanks and values
    outside the plausible band.

    Taking the minimum is the safe rule for the two multi-amount cases that
    occur here: a range should filter on its lower bound, and a sale string
    ('1.400.000 ₫ Giá gốc là... 1.200.000 ₫ Giá hiện tại là...') should filter
    on the discounted price, which is the lower one. `extract_price()` already
    isolates the correct amount at the selector level (see docs/decisions.md
    D5); this stays as a safety net for strings that slip through.

    Storing price as a formatted string makes budget filtering ('dưới 500k')
    impossible. Parsing it to an integer is what unlocks the payload index.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    matches = _PRICE_TOKEN_RE.findall(text)
    if not matches:
        return None

    candidates = []
    for token in matches:
        digits = _PRICE_CLEAN_RE.sub("", token)
        if not digits:
            continue
        value = int(digits)
        if MIN_PLAUSIBLE_VND <= value <= MAX_PLAUSIBLE_VND:
            candidates.append(value)
    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    """Collapse whitespace and normalise Vietnamese diacritics to NFC.

    Vietnamese text on the web mixes NFC and NFD, which makes otherwise
    identical strings hash differently and breaks duplicate detection.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    # Guard against a null having been stringified upstream, which is how the
    # literal token "nan" ends up embedded as if it were product copy.
    if value.strip().lower() in {"nan", "none", "null", "<na>"}:
        return ""
    value = unicodedata.normalize("NFC", value)
    return _WS_RE.sub(" ", value).strip()


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class Product(BaseModel):
    """One crawled product.

    `embed_text` is deliberately NOT the concatenation of every field.
    Concatenating title + price + description + promo + image URL means that
    when `promo` is byte-identical across the catalogue and `description` is
    near-identical boilerplate, most of every embedded string is text shared
    with every other document, and cosine scores collapse into a narrow band.
    Display metadata lives in the payload; only discriminative text is embedded.
    """

    url: str
    title: str = ""
    price_raw: str = ""
    price_vnd: int | None = None
    description: str = ""
    promo: str = ""
    image: str = ""
    crawled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""

    @field_validator("title", "price_raw", "description", "promo", mode="before")
    @classmethod
    def _normalize(cls, v: object) -> str:
        return normalize_text(v if isinstance(v, str) or v is None else str(v))

    @field_validator("image", "url", mode="before")
    @classmethod
    def _normalize_url(cls, v: object) -> str:
        return normalize_text(v if isinstance(v, str) or v is None else str(v)).replace(" ", "")

    def model_post_init(self, _ctx: object) -> None:
        if self.price_vnd is None and self.price_raw:
            object.__setattr__(self, "price_vnd", parse_price_vnd(self.price_raw))
        if not self.content_hash:
            object.__setattr__(self, "content_hash", self.compute_hash())

    def compute_hash(self) -> str:
        """Stable hash of the substantive fields.

        Excludes `crawled_at` so that re-crawling unchanged pages produces the
        same hash — that is what makes incremental re-indexing possible instead
        of re-embedding all 741 products on every refresh.
        """
        parts = [self.url, self.title, self.price_raw, self.description, self.promo, self.image]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]

    @property
    def is_product(self) -> bool:
        return is_product_url(self.url)

    def price_band(self) -> str:
        """Coarse price bucket, in words, for the embedded text.

        A bare number embeds poorly; 'khoảng 1 triệu' is closer to how a
        customer phrases a budget and gives the dense retriever something to
        match against queries like 'tầm 1 triệu'.
        """
        v = self.price_vnd
        if v is None:
            return ""
        if v < 300_000:
            return "giá rẻ dưới 300 nghìn"
        if v < 600_000:
            return "tầm giá 300 đến 600 nghìn"
        if v < 1_000_000:
            return "tầm giá 600 nghìn đến 1 triệu"
        if v < 2_000_000:
            return "tầm giá 1 đến 2 triệu"
        if v < 5_000_000:
            return "tầm giá 2 đến 5 triệu"
        return "cao cấp trên 5 triệu"

    def to_payload(self) -> dict[str, object]:
        """Qdrant payload — everything needed for display, filtering, freshness."""
        return {
            "url": self.url,
            "title": self.title,
            "price_raw": self.price_raw,
            "price_vnd": self.price_vnd,
            "description": self.description,
            "promo": self.promo,
            "image": self.image,
            "crawled_at": self.crawled_at.isoformat(),
            "content_hash": self.content_hash,
        }
