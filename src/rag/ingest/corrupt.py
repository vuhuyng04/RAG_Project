"""Deliberate, seeded corpus corruption.

Every corruptor models a failure mode that **actually occurs in web-scraped
product catalogues**, so the experiment measures repair against realistic
damage rather than invented noise:

| corruptor          | real defect it reproduces                                |
|--------------------|----------------------------------------------------------|
| nan_injection      | cell-14 missed `.fillna()`; `.astype(str)` wrote "nan"    |
| boilerplate_flood  | `khuyen_mai` byte-identical across all products           |
| non_product_pages  | `/cua-hang/` (shop archive) indexed as row 0              |
| price_corruption   | price stored as a formatted string, unparseable           |
| truncation         | short/empty descriptions from failed selector fallbacks    |
| mojibake           | Vietnamese UTF-8 decoded as latin-1                       |
| duplication        | overlapping sitemaps returning the same product twice     |

The **manifest** is the point of this module. It records exactly which document
received which corruption, which turns repair into a measurable detection task
with precision/recall/F1 — rather than a vague "it looks better afterwards".

Corruption operates on *validated `Product` records*, never on raw HTML, so the
pipeline is a pure function `clean -> corrupt(seed) -> repair` and the same seed
always yields the same manifest.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag.ingest.schema import Product

log = logging.getLogger(__name__)

# The shared promotional passage that makes catalogue documents look alike.
BOILERPLATE = (
    "Mẫu hoa sang trọng, tinh tế. Freeship nội thành. Tư vấn nhiệt tình. "
    "Nhận thiết kế lẵng hoa theo ngân sách của khách hàng. "
    "Dịch vụ: Giao Gấp Trong Vòng 40 phút. Hoa Tươi My My luôn là lựa chọn "
    "tốt nhất của những tín đồ yêu thích hoa."
)

NON_PRODUCT_PAGES = [
    ("https://hoatuoimymy.com/cua-hang/", "Sản phẩm Archive - Shop Hoa Tươi My My"),
    ("https://hoatuoimymy.com/danh-muc/hoa-khai-truong/", "Hoa Khai Trương Archive"),
    ("https://hoatuoimymy.com/gio-hang/", "Giỏ hàng"),
    ("https://hoatuoimymy.com/thanh-toan/", "Thanh toán"),
    ("https://hoatuoimymy.com/tai-khoan/", "Tài khoản"),
]


class Corruption:
    # This corruptor originally injected the literal string "nan" into a field.
    # That turned out to be **impossible to persist**: `Product`'s validator
    # runs `normalize_text`, which maps "nan"/"none"/"null" to "", so the
    # damage was erased on the save/reload round-trip and the detector scored
    # 0/39.
    #
    # That is a genuine result rather than a workaround: the schema eliminates
    # this defect class by construction instead of detecting it after the fact.
    # The BASELINE state still exhibits it directly, since it bypasses the
    # schema. What remains testable downstream is the symptom that *does*
    # survive validation — a required field going empty — so that is what this
    # corruptor injects.
    MISSING_FIELD = "missing_field"
    BOILERPLATE = "boilerplate_flood"
    NON_PRODUCT = "non_product_page"
    PRICE = "price_corruption"
    TRUNCATION = "truncation"
    MOJIBAKE = "mojibake"
    DUPLICATE = "duplication"

    ALL = (MISSING_FIELD, BOILERPLATE, NON_PRODUCT, PRICE, TRUNCATION, MOJIBAKE, DUPLICATE)


@dataclass
class Manifest:
    """Ground truth for the repair evaluation."""

    seed: int
    rates: dict[str, float]
    # url -> list of corruption labels applied
    corrupted: dict[str, list[str]] = field(default_factory=dict)
    n_clean_input: int = 0
    n_corrupt_output: int = 0

    def mark(self, url: str, label: str) -> None:
        self.corrupted.setdefault(url, []).append(label)

    def urls_with(self, label: str) -> set[str]:
        return {u for u, labels in self.corrupted.items() if label in labels}

    @property
    def all_corrupted_urls(self) -> set[str]:
        return set(self.corrupted)

    def summary(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for labels in self.corrupted.values():
            counts.update(labels)
        return dict(counts)


def _mojibake(text: str) -> str:
    """Simulate UTF-8 bytes decoded as latin-1 — the classic Vietnamese mangling."""
    try:
        return text.encode("utf-8").decode("latin-1")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def corrupt_products(
    products: list[Product],
    *,
    seed: int = 42,
    rates: dict[str, float] | None = None,
) -> tuple[list[Product], Manifest]:
    """Apply seeded corruption. Returns damaged records plus the manifest.

    Rates are per-corruptor shares of the corpus. They overlap deliberately: a
    document can receive more than one kind of damage, which is what real
    pipeline failures look like.
    """
    rates = rates or {
        Corruption.MISSING_FIELD: 0.08,
        Corruption.BOILERPLATE: 0.25,
        Corruption.PRICE: 0.10,
        Corruption.TRUNCATION: 0.08,
        Corruption.MOJIBAKE: 0.06,
        Corruption.DUPLICATE: 0.05,
    }
    rng = random.Random(seed)
    manifest = Manifest(seed=seed, rates=dict(rates), n_clean_input=len(products))

    # Work on copies so the clean state on disk is never mutated.
    out = [p.model_copy(deep=True) for p in products]
    n = len(out)

    def pick(rate: float) -> list[int]:
        k = round(rate * n)
        return rng.sample(range(n), min(k, n))

    # --- field-level damage -------------------------------------------------
    for i in pick(rates.get(Corruption.MISSING_FIELD, 0)):
        p = out[i]
        # What a NaN column actually leaves behind once the schema has
        # normalised it: an empty required field.
        target = rng.choice(["description", "title", "price_raw"])
        setattr(p, target, "")
        if target == "price_raw":
            p.price_vnd = None
        manifest.mark(p.url, Corruption.MISSING_FIELD)

    for i in pick(rates.get(Corruption.BOILERPLATE, 0)):
        p = out[i]
        p.description = f"{BOILERPLATE} {p.description}".strip()
        manifest.mark(p.url, Corruption.BOILERPLATE)

    for i in pick(rates.get(Corruption.PRICE, 0)):
        p = out[i]
        p.price_raw = rng.choice(["Liên hệ", "Giá: --", "", "0₫"])
        p.price_vnd = None
        manifest.mark(p.url, Corruption.PRICE)

    for i in pick(rates.get(Corruption.TRUNCATION, 0)):
        p = out[i]
        p.description = p.description[: rng.randint(0, 25)]
        manifest.mark(p.url, Corruption.TRUNCATION)

    for i in pick(rates.get(Corruption.MOJIBAKE, 0)):
        p = out[i]
        p.title = _mojibake(p.title)
        p.description = _mojibake(p.description)
        manifest.mark(p.url, Corruption.MOJIBAKE)

    # --- record-level damage ------------------------------------------------
    # Duplicates enter with a URL variant (trailing slash / tracking param),
    # which is how canonicalisation failures actually manifest — an identical
    # URL would simply collapse on the deterministic point id.
    for i in pick(rates.get(Corruption.DUPLICATE, 0)):
        original = out[i]
        clone = original.model_copy(deep=True)
        clone.url = original.url.rstrip("/") + rng.choice(["//", "/?utm_source=google", "/?ref=fb"])
        out.append(clone)
        manifest.mark(clone.url, Corruption.DUPLICATE)

    for url, title in NON_PRODUCT_PAGES:
        out.append(
            Product(
                url=url,
                title=title,
                description="Danh mục sản phẩm của Shop Hoa Tươi My My.",
                promo="",
                price_raw="",
                image="",
            )
        )
        manifest.mark(url, Corruption.NON_PRODUCT)

    rng.shuffle(out)
    manifest.n_corrupt_output = len(out)

    log.info(
        "Corrupted %d -> %d records (seed=%d): %s",
        len(products),
        len(out),
        seed,
        manifest.summary(),
    )
    return out, manifest


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": manifest.seed,
        "rates": manifest.rates,
        "n_clean_input": manifest.n_clean_input,
        "n_corrupt_output": manifest.n_corrupt_output,
        "summary": manifest.summary(),
        "corrupted": manifest.corrupted,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote corruption manifest -> %s", path)


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = Manifest(
        seed=data["seed"],
        rates=data.get("rates", {}),
        n_clean_input=data.get("n_clean_input", 0),
        n_corrupt_output=data.get("n_corrupt_output", 0),
    )
    m.corrupted = data.get("corrupted", {})
    return m
