"""The CV bullets must stay consistent with the committed evidence.

Written after an actual slip: the first draft claimed an "80-query golden set"
when the file on disk holds 33. A CV is the highest-stakes place for a number to
be wrong, and it is the furthest from the data — so it gets the strictest gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "docs" / "cv-bullets.md"
RESULTS = ROOT / "eval" / "results"
GOLDEN = ROOT / "eval" / "dataset" / "golden.jsonl"


def _cv() -> str:
    return CV.read_text(encoding="utf-8")


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _golden() -> list[dict]:
    return [
        json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_no_overstated_golden_set_size() -> None:
    """Any 'N-query golden set' phrasing must match the file on disk."""
    actual = len(_golden())
    for match in re.finditer(r"(\d+)[- ]query golden set", _cv(), re.IGNORECASE):
        assert int(match.group(1)) == actual, (
            f"CV claims a {match.group(1)}-query golden set; the file has {actual}"
        )


def test_headline_numbers_match_results() -> None:
    text = _cv()
    repair = _load("repair_detection.json")
    legacy = _load("quality_legacy.json")

    assert f"{repair['macro_f1']:.3f}" in text
    assert f"{legacy['repeated_passage_doc_rate']:.0%}" in text
    assert str(legacy["junk_embed_text"]) in text
    assert str(legacy["documents"]) in text


def test_budget_delta_matches() -> None:
    dense = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    budget = _load("retrieval_dense_budget_clean.json")["retrieval_raw"]["recall@5"]
    assert f"{budget - dense:.3f}" in _cv()


def test_hybrid_penalty_matches() -> None:
    dense = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    hybrid = _load("retrieval_hybrid_clean.json")["retrieval_raw"]["recall@5"]
    assert f"{dense - hybrid:.3f}" in _cv()


def test_forbidden_claims_section_survives() -> None:
    """The 'what not to claim' table is the point of the document."""
    text = _cv()
    assert "What NOT to claim" in text
    for must_mention in ["RAGAS", "cosine separation", "Hybrid search improved"]:
        assert must_mention.lower() in text.lower()


def test_sample_size_caveat_present() -> None:
    n = _load("retrieval_dense_clean.json")["retrieval_raw"]["n"]
    text = _cv()
    assert f"n = {n}" in text or f"n={n}" in text
    assert "provisional" in text.lower()


def test_does_not_claim_clean_beats_legacy() -> None:
    """Guard the single most tempting unsupported claim.

    dense@clean = 0.504 vs dense@legacy = 0.520 — the clean pipeline does not
    currently win on retrieval, and the CV must not imply otherwise.
    """
    clean = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    legacy = _load("retrieval_dense_legacy.json")["retrieval_raw"]["recall@5"]
    if clean <= legacy:
        text = _cv().lower()
        assert "does not currently beat legacy" in text, (
            "clean does not beat legacy on Recall@5; the CV must say so explicitly"
        )
