"""Regression tests for keyword matching.

Every case here is a real bug that shipped: diacritic-folded substring matching
tagged 'có freeship không' with flower_type='hồng' because 'khong' contains
'hong'. Cheap to test, and a silently mislabelled facet corrupts the sliced
metrics without ever raising.
"""

from __future__ import annotations

import pytest

from rag.evalset.prefilter import dedupe, facets, provisional_intent
from rag.evalset.schema import Intent


@pytest.mark.parametrize(
    "query,field",
    [
        # 'khong' contains 'hong' (hồng/rose)
        ("có freeship không", "flower_type"),
        # 'khong' contains 'co' (cô/teacher)
        ("có freeship không", "recipient"),
        # 'lang' (lẵng) contains 'lan' (orchid)
        ("lẵng hoa 500k", "flower_type"),
        # 'phong' (phòng) contains 'hong'
        ("giỏ hoa quả hải phòng", "flower_type"),
        # 'ket' (kết) contains 'ke' (kệ/stand)
        ("shop hoa tươi quận 7 cam kết đổi trả", "form_factor"),
        # 'con' contains 'co'
        ("hoa tặng sinh nhật con gái", "recipient"),
        # 'bình dương' is a province, not the 'bình' vase form factor
        ("hoa khai trương bình dương", "form_factor"),
        # 'có giao' folds to ["co","giao"], identical to 'cô giáo' (teacher)
        ("shop có giao nhanh không", "recipient"),
    ],
)
def test_no_substring_false_positives(query: str, field: str) -> None:
    assert facets(query)[field] is None, f"{field} falsely matched in {query!r}"


@pytest.mark.parametrize(
    "query,field,expected",
    [
        ("bó hoa sinh nhật cho nam", "recipient", "nam"),
        ("bó hoa sinh nhật mẹ", "recipient", "mẹ"),
        ("hoa lan hồ điệp đà lạt", "flower_type", "lan hồ điệp"),  # longest key wins
        ("giỏ hoa khai trương", "form_factor", "giỏ"),
        ("hoa hồng tặng người yêu", "flower_type", "hồng"),
        ("hoa khai trương 500k", "occasion", "khai trương"),
    ],
)
def test_true_positives_still_match(query: str, field: str, expected: str) -> None:
    assert facets(query)[field] == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hoa khai trương 500k", Intent.BUDGET_SEARCH),
        ("hoa chia buồn đà nẵng", Intent.OUT_OF_CATALOGUE),
        # Cầu Giấy is a Hanoi district — out of catalogue, not off-topic.
        ("hoa khai trương cầu giấy", Intent.OUT_OF_CATALOGUE),
        # ...but paper flowers really are off-topic.
        ("cách làm hoa giấy", Intent.OFF_TOPIC),
        ("hoa tặng mẹ lớp 4", Intent.OFF_TOPIC),
        ("có freeship không", Intent.POLICY),
        # An HCM district must not be read as out of catalogue.
        ("shop hoa tươi gò vấp", Intent.PRODUCT_SEARCH),
    ],
)
def test_provisional_intent(query: str, expected: Intent) -> None:
    assert provisional_intent(query) is expected


def test_budget_parsing() -> None:
    assert facets("hoa khai trương 500k")["budget_max_vnd"] == 500_000
    assert facets("hoa tặng mẹ dưới 1 triệu")["budget_max_vnd"] == 1_000_000
    assert facets("bó hoa sinh nhật đẹp")["budget_max_vnd"] is None


def test_dedupe_removes_near_duplicates() -> None:
    queries = ["bó hoa sinh nhật", "bó hoa sinh nhật", "kệ hoa khai trương"]
    assert len(dedupe(queries)) == 2


def test_dedupe_keeps_distinct_queries() -> None:
    queries = ["hoa chia buồn đà nẵng", "hoa khai trương 500k", "hoa cưới cầm tay"]
    assert len(dedupe(queries)) == 3
