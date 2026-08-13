"""Every number quoted in the README must exist in a committed result file.

This is the project's stated acceptance gate, so it is enforced rather than
trusted. Without it, a metric can drift after a re-run and the README silently
becomes fiction — which is exactly the failure mode the project is about.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "eval" / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _readme() -> str:
    return README.read_text(encoding="utf-8")


# --- corpus quality --------------------------------------------------------


@pytest.mark.parametrize(
    "state,key,fmt",
    [
        ("baseline", "repeated_passage_doc_rate", "{:.1%}"),
        ("clean", "repeated_passage_doc_rate", "{:.1%}"),
        ("corrupt", "repeated_passage_doc_rate", "{:.1%}"),
        ("repaired", "repeated_passage_doc_rate", "{:.1%}"),
        ("baseline", "junk_embed_text", "{}"),
        ("baseline", "documents", "{}"),
        ("clean", "documents", "{}"),
    ],
)
def test_quality_numbers_appear_in_readme(state: str, key: str, fmt: str) -> None:
    report = _load(f"quality_{state}.json")
    value = fmt.format(report[key])
    assert value in _readme(), f"{state}.{key} = {value} not found in README"


# --- repair detection ------------------------------------------------------


def test_macro_f1_matches() -> None:
    report = _load("repair_detection.json")
    assert f"{report['macro_f1']:.3f}" in _readme()


def test_every_per_defect_f1_matches() -> None:
    report = _load("repair_detection.json")
    text = _readme()
    for label, scores in report["per_defect"].items():
        assert f"{scores['f1']:.3f}" in text, f"{label} F1 {scores['f1']:.3f} missing"


# --- retrieval -------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        "baseline",
        "dense",
        "dense_threshold",
        "dense_budget",
        "hybrid",
        "hybrid_budget",
        "dense_rerank",
        "full",
    ],
)
def test_retrieval_recall_matches(config: str) -> None:
    report = _load(f"retrieval_{config}_clean.json")
    recall = report["retrieval_raw"]["recall@5"]
    assert f"{recall:.3f}" in _readme(), f"{config} recall@5 {recall:.3f} missing"


def test_repair_recovery_percentage_is_computed_not_asserted() -> None:
    """The 80.0% recovery figure must follow from the three committed states."""
    clean = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    corrupt = _load("retrieval_dense_corrupt.json")["retrieval_raw"]["recall@5"]
    repaired = _load("retrieval_dense_repaired.json")["retrieval_raw"]["recall@5"]
    recovery = (repaired - corrupt) / (clean - corrupt)
    assert f"{recovery:.1%}" in _readme(), f"recovery {recovery:.1%} not in README"


def test_budget_filter_delta_matches() -> None:
    dense = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    budget = _load("retrieval_dense_budget_clean.json")["retrieval_raw"]["recall@5"]
    assert f"{budget - dense:.3f}" in _readme()


# --- honesty guards --------------------------------------------------------


def test_readme_states_the_sample_size_limitation() -> None:
    """The n=12 caveat must survive future edits — it bounds every claim."""
    text = _readme()
    n = _load("retrieval_dense_clean.json")["retrieval_raw"]["n"]
    assert f"n = {n}" in text or f"n={n}" in text
    assert "provisional" in text.lower()


def test_readme_keeps_the_negative_results_section() -> None:
    """The section must survive, and must carry the claims the data supports.

    Deliberately does *not* pin a hybrid direction: that finding reversed
    between n=12 and n=14, and a test asserting yesterday's conclusion would
    have forced the README to keep repeating it.
    """
    text = _readme().lower()
    assert "negative results" in text
    for claim in [
        "does not currently beat the naive baseline",
        "spread is not a quality metric",
        "reranking costs",
    ]:
        assert claim in text, f"missing negative result: {claim}"


def test_readme_reports_the_sample_volatility() -> None:
    """The instability is the headline caveat and must be quantified, not vague."""
    text = _readme()
    n12 = 0.504  # dense@clean at n=12, from the previous committed run
    n14 = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    assert f"{n12 - n14:.3f}" in text, (
        f"README should state the {n12 - n14:.3f} swing that two extra queries caused"
    )


def test_no_placeholder_numbers_left() -> None:
    """Catch un-filled templates like '__%' or 'TODO' before they ship."""
    text = _readme()
    assert "__" not in text
    assert not re.search(r"\bTODO\b|\bTBD\b|\bXX\b", text)
