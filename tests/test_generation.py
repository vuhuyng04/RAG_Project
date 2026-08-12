"""Tests for citation validation and the condensation gate.

Both are deterministic and run without touching the API, which matters under a
20-request/day quota: the parts of generation that can be tested for free
should be.
"""

from __future__ import annotations

import pytest

from rag.generation.citation import (
    citation_validity_rate,
    strip_invalid_citations,
    validate_citations,
)
from rag.generation.condense import needs_condensation
from rag.generation.prompt import CONTEXT_CLOSE, CONTEXT_OPEN, build_prompt
from rag.retrieval.search import Hit


def make_hits(n: int = 3) -> list[Hit]:
    return [
        Hit(
            url=f"https://hoatuoimymy.com/bo-hoa-m{i}/",
            title=f"Bó Hoa M{i}",
            score=0.8 - i * 0.01,
            payload={
                "url": f"https://hoatuoimymy.com/bo-hoa-m{i}/",
                "title": f"Bó Hoa M{i}",
                "price_raw": f"{i}00.000₫",
                "description": "Mô tả sản phẩm.",
            },
        )
        for i in range(1, n + 1)
    ]


# --- citation validation ---------------------------------------------------


def test_valid_citations_accepted() -> None:
    r = validate_citations("Bó hoa này giá 500.000₫ [1], còn mẫu khác [2].", n_context=3)
    assert r.cited == [1, 2]
    assert r.invalid == []
    assert r.unused == [3]
    assert r.is_valid


def test_fabricated_index_detected() -> None:
    """The failure that matters most: a citation to a source that never existed."""
    r = validate_citations("Sản phẩm giá 500.000₫ [7].", n_context=3)
    assert r.invalid == [7]
    assert not r.is_valid


def test_uncited_price_claim_detected() -> None:
    r = validate_citations("Bó hoa này có giá 500.000₫ rất hợp lý.", n_context=3)
    assert r.uncited_claims
    assert not r.is_valid


def test_pleasantries_do_not_need_citation() -> None:
    r = validate_citations(
        "Chào bạn! Cảm ơn bạn đã quan tâm. Bạn cần tư vấn thêm gì ạ?", n_context=3
    )
    assert r.uncited_claims == []
    assert r.is_valid


@pytest.mark.parametrize(
    "sentence",
    [
        # Observed false positives from a real generated answer. Neither asserts
        # anything checkable, so flagging them made uncited_claim_rate noise.
        "Chào bạn, Hoa Tươi My My có một số mẫu kệ hoa khai trương rất đẹp đây ạ:",
        "Bạn muốn tham khảo thêm thông tin chi tiết về mẫu nào không ạ?",
        "Bạn cần mình tư vấn thêm về loại hoa nào không?",
    ],
)
def test_lead_ins_and_questions_are_not_claims(sentence: str) -> None:
    assert validate_citations(sentence, n_context=3).uncited_claims == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Mẫu này có giá 900.000₫ và rất phù hợp.",
        "Hoa Khai Trương M440 là lựa chọn tốt nhất.",
    ],
)
def test_concrete_assertions_still_require_citation(sentence: str) -> None:
    assert validate_citations(sentence, n_context=3).uncited_claims


@pytest.mark.parametrize(
    "sentence",
    [
        # Attribute hallucinations — the claims a RAG system most often invents.
        # An earlier, price-only rule missed all three, which made the metric
        # near-vacuous: it could barely fail.
        "Mẫu này dùng hoa hướng dương và lan hồ điệp trắng, phù hợp cho khai trương.",
        "Kệ hoa này cao khoảng hai mét và có thể giao trong ngày.",
        "Sản phẩm được thiết kế hai tầng với tông đỏ chủ đạo.",
    ],
)
def test_attribute_claims_caught_by_descriptive_tier(sentence: str) -> None:
    r = validate_citations(sentence, n_context=3)
    assert r.uncited_descriptive, "attribute hallucination slipped through both tiers"


@pytest.mark.parametrize(
    "sentence",
    [
        "Chào bạn, cảm ơn bạn đã quan tâm đến shop.",
        "Mình rất vui được tư vấn cho bạn.",
        "Chúc bạn một ngày tốt lành!",
        "Bạn cho mình biết thêm nhu cầu nhé.",
    ],
)
def test_descriptive_tier_does_not_flag_pleasantries(sentence: str) -> None:
    r = validate_citations(sentence, n_context=3)
    assert not r.uncited_claims and not r.uncited_descriptive


def test_citation_scope_is_the_block_not_the_sentence() -> None:
    """A marker at the end of a bullet grounds the whole bullet.

    Verbatim from a real generated answer. Every item is correctly cited, but a
    sentence-scoped check split each bullet on its full stop and flagged the
    first half — which names the product code but carries no marker — producing
    5 false positives on 5 correct bullets.
    """
    answer = (
        "Chào bạn, Hoa Tươi My My có một số mẫu hoa chia buồn trang trọng đây ạ:\n"
        "- **Hoa Chia Buồn M20**: Vòng hoa chia buồn mang ý nghĩa sâu sắc, "
        "với màu trắng, tím, vàng. Giá 1.100.000 đ [1].\n"
        "- **Hoa Chia Buồn M54**: Vòng hoa sang trọng, biểu tượng của lòng tôn "
        "kính và sự vĩnh cửu. Giá 800.000 đ [2].\n"
        "Bạn muốn tìm hiểu thêm về mẫu nào không ạ?"
    )
    r = validate_citations(answer, n_context=5)
    assert r.cited == [1, 2]
    assert r.uncited_claims == [], r.uncited_claims
    assert r.is_valid


def test_uncited_block_is_still_caught() -> None:
    """Block scoping must not let a genuinely uncited bullet through."""
    answer = (
        "- **Hoa Chia Buồn M20**: mẫu này giá 1.100.000 đ [1].\n"
        "- **Hoa Chia Buồn M99**: mẫu này giá 2.000.000 đ.\n"
    )
    r = validate_citations(answer, n_context=2)
    assert r.uncited_claims, "an entirely uncited bullet must be flagged"
    assert not r.is_valid


def test_abstention_answer_is_valid_without_citations() -> None:
    """No context means nothing to cite — that must not count as a violation."""
    r = validate_citations("Mình chưa tìm thấy sản phẩm phù hợp ạ.", n_context=0)
    assert r.is_valid
    assert r.cited == []


def test_strip_invalid_keeps_valid() -> None:
    out = strip_invalid_citations("Mẫu A [1] và mẫu B [9].", n_context=3)
    assert "[1]" in out
    assert "[9]" not in out


def test_validity_rate_ignores_abstentions() -> None:
    reports = [
        validate_citations("Giá 100.000₫ [1].", 2),  # valid
        validate_citations("Giá 200.000₫ [5].", 2),  # invalid index
        validate_citations("Chưa tìm thấy ạ.", 0),  # abstention, excluded
    ]
    stats = citation_validity_rate(reports)
    assert stats["n"] == 2
    assert stats["validity_rate"] == 0.5
    assert stats["invalid_index_rate"] == 0.5


# --- prompt ----------------------------------------------------------------


def test_prompt_delimits_untrusted_context() -> None:
    prompt = build_prompt("hoa sinh nhật", make_hits(2))
    assert CONTEXT_OPEN in prompt and CONTEXT_CLOSE in prompt
    assert "[1]" in prompt and "[2]" in prompt
    # The retrieved text must be declared as data, not instructions.
    assert "DỮ LIỆU" in prompt


def test_empty_hits_uses_abstain_prompt() -> None:
    prompt = build_prompt("có freeship không", [])
    assert CONTEXT_OPEN not in prompt
    assert "KHÔNG tìm thấy" in prompt
    assert "KHÔNG gợi ý" in prompt


# --- condensation gate -----------------------------------------------------


@pytest.mark.parametrize(
    "question,history,expected",
    [
        # No history: nothing to condense against.
        ("hoa tặng mẹ", [], False),
        # Self-contained despite history — must not waste a call or risk mangling.
        ("bó hoa sinh nhật cho mẹ giá rẻ", [("hoa khai trương", "...")], False),
        # Bare comparative with no subject: needs the antecedent.
        ("cái nào rẻ hơn?", [("hoa sinh nhật", "...")], True),
        # Very short follow-up.
        ("còn màu trắng?", [("hoa lan hồ điệp", "...")], True),
        # Back-reference but supplies its own noun — retrievable as-is.
        ("còn bó hoa hồng nào khác không", [("hoa sinh nhật", "...")], False),
    ],
)
def test_condensation_gate(question, history, expected) -> None:
    assert needs_condensation(question, history) is expected
