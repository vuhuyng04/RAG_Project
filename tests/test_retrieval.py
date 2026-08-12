"""Tests for budget parsing and retrieval configuration.

Pure-function tests only — no Qdrant, no network — so they run in CI without
credentials.
"""

from __future__ import annotations

import pytest

from rag.retrieval.search import CONFIGS, extract_budget_vnd


@pytest.mark.parametrize(
    "question,expected",
    [
        # Bare-number form, which dominates the harvested real queries
        # ("hoa khai trương 500k" was the single most common budget phrasing).
        ("hoa khai trương 500k", 500_000),
        ("bó hoa 300k", 300_000),
        # Explicit qualifiers.
        ("hoa tặng mẹ dưới 1 triệu", 1_000_000),
        ("kệ hoa khai trương tầm 2 triệu", 2_000_000),
        ("hoa sinh nhật khoảng 800 nghìn", 800_000),
        ("hoa cưới dưới 1.5 triệu", 1_500_000),
        # No budget mentioned.
        ("bó hoa sinh nhật đẹp", None),
        ("hoa chia buồn trang trọng", None),
    ],
)
def test_extract_budget(question: str, expected: int | None) -> None:
    assert extract_budget_vnd(question) == expected


def test_configs_differ_by_one_variable_at_a_time() -> None:
    """The A/B ladder only means something if each rung adds one mechanism."""
    dense = CONFIGS["dense"]
    threshold = CONFIGS["dense_threshold"]
    rerank = CONFIGS["dense_rerank"]

    # dense -> dense_threshold: only abstention is added.
    assert threshold.use_rerank == dense.use_rerank
    assert threshold.use_budget_filter == dense.use_budget_filter
    assert dense.score_threshold is None and threshold.score_threshold is not None

    # dense_threshold -> dense_rerank: only reranking is added.
    assert rerank.score_threshold == threshold.score_threshold
    assert rerank.use_budget_filter == threshold.use_budget_filter
    assert rerank.use_rerank and not threshold.use_rerank


def test_baseline_config_never_abstains() -> None:
    """The control must have no abstention mechanism at all.

    Returning top-5 for every query, including 'có freeship không', is the
    behaviour the abstention metric exists to measure against.
    """
    baseline = CONFIGS["baseline"]
    assert baseline.score_threshold is None
    assert not baseline.use_rerank
    assert not baseline.use_budget_filter
    assert baseline.top_k == 5


def test_hybrid_configs_do_not_carry_a_cosine_threshold() -> None:
    """RRF scores are rank-derived (~1/60), not cosine.

    Applying the 0.60 cosine threshold to a hybrid config would abstain on
    every query — a silent, total failure that returns no error.
    """
    for name in ("hybrid", "hybrid_budget"):
        assert CONFIGS[name].use_hybrid
        assert CONFIGS[name].score_threshold is None, name


def test_all_configs_have_descriptions() -> None:
    """Descriptions land in eval/results/ and explain what each cell measured."""
    for name, cfg in CONFIGS.items():
        assert cfg.description, f"{name} has no description"
        assert cfg.name == name


def test_every_config_has_a_vietnamese_ui_label() -> None:
    """The UI must not fall back to the English experiment rationale.

    `RetrievalConfig.description` is developer-facing and was rendering inside a
    Vietnamese interface. Adding a config without a matching entry in
    CONFIG_LABELS_VI would silently reintroduce that.
    """
    import ast
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "chatbot.py"
    match = re.search(r"CONFIG_LABELS_VI = (\{.*?\n\})", src.read_text(encoding="utf-8"), re.DOTALL)
    assert match, "CONFIG_LABELS_VI not found in app/chatbot.py"
    labels = ast.literal_eval(match.group(1))

    assert set(labels) == set(CONFIGS), (
        f"missing: {set(CONFIGS) - set(labels)}, stale: {set(labels) - set(CONFIGS)}"
    )
    assert all(v.strip() for v in labels.values())
