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
    baseline = _load("quality_baseline.json")

    assert f"{repair['macro_f1']:.3f}" in text
    assert f"{baseline['repeated_passage_doc_rate']:.0%}" in text
    assert str(baseline["junk_embed_text"]) in text
    assert str(baseline["documents"]) in text


def test_budget_delta_matches() -> None:
    dense = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    budget = _load("retrieval_dense_budget_clean.json")["retrieval_raw"]["recall@5"]
    assert f"{budget - dense:.3f}" in _cv()


def test_no_hybrid_direction_claimed_while_the_pairings_disagree() -> None:
    """Hybrid may only be claimed to help or hurt if both pairings agree.

    At n=12 both pairings lost 0.087 and the CV stated "BM25 hybrid hurts". At
    n=14 one pairing gained and the other lost, so no direction is supportable.
    This guard follows the data instead of pinning yesterday's conclusion.
    """
    dense = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    hybrid = _load("retrieval_hybrid_clean.json")["retrieval_raw"]["recall@5"]
    dense_b = _load("retrieval_dense_budget_clean.json")["retrieval_raw"]["recall@5"]
    hybrid_b = _load("retrieval_hybrid_budget_clean.json")["retrieval_raw"]["recall@5"]

    unfiltered = hybrid - dense
    filtered = hybrid_b - dense_b
    agree = (unfiltered > 0) == (filtered > 0)

    text = _cv().lower()
    if not agree:
        assert "reversed at n=" in text or "pairings" in text, (
            "the two hybrid pairings disagree; the CV must say the result is unstable "
            "rather than assert a direction"
        )
    else:
        direction = "hurts" if unfiltered < 0 else "helps"
        assert direction in text, f"pairings agree that hybrid {direction}; say so"


def test_forbidden_claims_section_survives() -> None:
    """The 'what not to claim' table is the point of the document."""
    text = _cv().lower()
    assert "what not to claim" in text
    for must_mention in ["ragas", "cosine separation", "hybrid", "reranking"]:
        assert must_mention in text, f"missing forbidden-claim entry: {must_mention}"


def test_sample_size_caveat_present() -> None:
    n = _load("retrieval_dense_clean.json")["retrieval_raw"]["n"]
    text = _cv()
    assert f"n = {n}" in text or f"n={n}" in text
    assert "provisional" in text.lower()


def test_does_not_claim_clean_beats_baseline() -> None:
    """Guard the single most tempting unsupported claim.

    The validated pipeline does not currently out-retrieve the naive
    concatenation baseline, and the CV must not imply otherwise. If a larger
    golden set later reverses this, the guard relaxes on its own.
    """
    clean = _load("retrieval_dense_clean.json")["retrieval_raw"]["recall@5"]
    baseline = _load("retrieval_dense_baseline.json")["retrieval_raw"]["recall@5"]
    if clean <= baseline:
        text = _cv().lower()
        assert "does not currently beat the naive baseline" in text, (
            "clean does not out-retrieve the baseline; the CV must say so explicitly"
        )
