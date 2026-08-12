"""Corruption/repair round-trip tests.

The detectors are scored against the manifest, so a silently broken detector
shows up as a bad F1 rather than an exception. These tests pin the properties
that make that scoring trustworthy in the first place: determinism, isolation
of the clean input, and the specific detector bugs that were found and fixed.
"""

from __future__ import annotations

import pytest

from rag.ingest.corrupt import Corruption, corrupt_products
from rag.ingest.repair import (
    _canonical_url,
    demojibake,
    detect_repeated_passages,
    looks_mojibake,
    repair_products,
    score_detection,
)
from rag.ingest.schema import Product

# Descriptions must be genuinely unique per document. Two earlier fixtures were
# rejected for the same reason: one used a single template with the index
# substituted, the next cycled phrases with modulo so every 6th product shared a
# clause. Both were themselves boilerplate, so the repeated-passage detector
# flagged them correctly and the resulting "false positives" looked like a
# detector bug. Real product copy is unique prose; the fixture has to be too.
_VOCAB = [
    "hồng",
    "cúc",
    "ly",
    "lan",
    "tulip",
    "sen",
    "mai",
    "đào",
    "hướng dương",
    "cẩm chướng",
    "đỏ",
    "vàng",
    "trắng",
    "tím",
    "cam",
    "xanh",
    "pastel",
    "rực rỡ",
    "dịu dàng",
    "sang trọng",
    "giỏ",
    "bó",
    "kệ",
    "chậu",
    "lẵng",
    "hộp",
    "thiệp",
    "nơ",
    "voan",
    "kraft",
    "sinh nhật",
    "khai trương",
    "kỷ niệm",
    "tốt nghiệp",
    "tân gia",
    "cảm ơn",
]


def _unique_description(i: int) -> str:
    """Deterministic description with provably no shingle shared between docs.

    Picking words from a 36-word vocabulary was not enough: with 14 words per
    document and 120 documents, 8-gram windows collide by pigeonhole and the
    detector flagged 64 uncorrupted records. Interleaving a per-document nonce
    guarantees every 8-token window contains a token unique to that document.
    """
    words = [_VOCAB[(i * 7 + j * 3) % len(_VOCAB)] for j in range(14)]
    return " ".join(f"{w} n{i}t{j}" for j, w in enumerate(words)) + "."


def make_products(n: int = 60) -> list[Product]:
    return [
        Product(
            url=f"https://hoatuoimymy.com/bo-hoa-m{i}/",
            title=f"Bó Hoa M{i} {_VOCAB[i % len(_VOCAB)]}",
            price_raw=f"{(i % 9 + 1) * 100}.000₫",
            description=_unique_description(i),
            promo="Freeship nội thành.",
            image=f"https://hoatuoimymy.com/img/{i}.jpg",
        )
        for i in range(n)
    ]


def test_corruption_is_deterministic() -> None:
    products = make_products()
    a, ma = corrupt_products(products, seed=42)
    b, mb = corrupt_products(products, seed=42)
    assert ma.corrupted == mb.corrupted
    assert [p.url for p in a] == [p.url for p in b]


def test_different_seeds_differ() -> None:
    products = make_products()
    _, ma = corrupt_products(products, seed=1)
    _, mb = corrupt_products(products, seed=2)
    assert ma.corrupted != mb.corrupted


def test_clean_input_is_not_mutated() -> None:
    """Corruption must not touch the clean state — it is the control group."""
    products = make_products()
    before = [(p.title, p.description, p.price_raw) for p in products]
    corrupt_products(products, seed=42)
    after = [(p.title, p.description, p.price_raw) for p in products]
    assert before == after


def test_repair_detects_most_damage() -> None:
    products = make_products(120)
    corrupted, manifest = corrupt_products(products, seed=42)
    _, detection = repair_products(corrupted)
    scores = score_detection(detection, manifest)
    # Guards against a detector silently regressing to noise. The measured
    # macro F1 on the real corpus is ~0.91; this floor is deliberately loose so
    # the test tracks breakage, not corpus-specific tuning.
    assert scores["macro_f1"] > 0.6, scores["per_defect"]
    assert scores["overall_any_defect"]["f1"] > 0.8


def test_non_product_pages_always_dropped() -> None:
    products = make_products()
    corrupted, manifest = corrupt_products(products, seed=7)
    kept, _ = repair_products(corrupted)
    kept_urls = {p.url for p in kept}
    assert not (kept_urls & manifest.urls_with(Corruption.NON_PRODUCT))


def test_duplicate_repair_keeps_canonical_url() -> None:
    """Regression: the shuffle used to make repair drop the good copy.

    Duplicates enter as tracking-param variants. Dropping whichever copy came
    first meant discarding the canonical record and reporting the duplicate at
    the wrong URL (P=0.52). Canonical URLs must win.
    """
    canonical = "https://hoatuoimymy.com/bo-hoa-m1/"
    base = Product(
        url=canonical,
        title="Bó Hoa M1",
        price_raw="500.000₫",
        description="Mô tả riêng biệt cho bó hoa này.",
    )
    clone = base.model_copy(deep=True)
    clone.url = canonical.rstrip("/") + "/?utm_source=google"

    kept, detection = repair_products([clone, base])
    assert [p.url for p in kept] == [canonical]
    assert clone.url in detection.urls_with(Corruption.DUPLICATE)


def test_tracking_param_clone_is_not_called_non_product() -> None:
    """Regression: '?utm_source=' made is_product_url() reject a real product."""
    canonical = "https://hoatuoimymy.com/bo-hoa-m1/"
    base = Product(
        url=canonical, title="Bó Hoa M1", price_raw="500.000₫", description="Mô tả riêng biệt."
    )
    clone = base.model_copy(deep=True)
    clone.url = canonical.rstrip("/") + "/?ref=fb"

    _, detection = repair_products([base, clone])
    assert clone.url not in detection.urls_with(Corruption.NON_PRODUCT)


def test_repeated_passage_detection_beats_token_frequency() -> None:
    """Regression: flooding a minority of documents was undetectable.

    Token document-frequency with a 60% threshold scored recall 0.017, because
    contaminating 25% of the corpus never pushes a token past 60%. Shingles
    catch it at any rate above their own threshold.
    """
    shared = "mẫu hoa sang trọng tinh tế freeship nội thành tư vấn nhiệt tình luôn sẵn sàng"
    texts = [f"{shared} sản phẩm {i}" for i in range(25)] + [
        f"mô tả hoàn toàn riêng biệt số {i} không giống ai" for i in range(75)
    ]
    repeated = detect_repeated_passages(texts)
    assert repeated, "no repeated passages found at 25% contamination"


@pytest.mark.parametrize("text", ["Hoa Chia Buồn M55", "Giỏ Hoa Sinh Nhật", "Kệ Hoa Khai Trương"])
def test_mojibake_roundtrip(text: str) -> None:
    broken = text.encode("utf-8").decode("latin-1")
    assert looks_mojibake(broken)
    assert demojibake(broken) == text
    assert not looks_mojibake(text)


def test_canonical_url_strips_tracking() -> None:
    assert _canonical_url("https://x.com/a/?utm_source=g") == "https://x.com/a"
    assert _canonical_url("https://x.com/a//") == "https://x.com/a"
    assert _canonical_url("https://x.com/a/") == "https://x.com/a"
