"""The golden-set sample must be monotone in `--target`.

At 20 LLM requests/day/model the set grows a handful of queries per run, so
labelling only accumulates if raising the target *adds* queries rather than
reshuffling them. The first version shuffled the combined list at the end, which
made ordering depend on its length: a second run judged a different first-N and
spent 19 calls for 1 cache hit.
"""

from __future__ import annotations

from scripts.build_golden import TARGET_MIX, sample_balanced

BUCKETS = {intent.value: [f"{intent.value} query {i}" for i in range(40)] for intent in TARGET_MIX}


def test_sampling_is_deterministic() -> None:
    assert sample_balanced(BUCKETS, 40, seed=42) == sample_balanced(BUCKETS, 40, seed=42)


def test_raising_the_target_only_adds_queries() -> None:
    """target=80 must be a strict superset of target=40 — the whole point."""
    small = {q for q, _ in sample_balanced(BUCKETS, 40, seed=42)}
    large = {q for q, _ in sample_balanced(BUCKETS, 80, seed=42)}
    assert small <= large, f"lost {len(small - large)} previously-sampled queries"
    assert len(large) > len(small)


def test_growth_is_monotone_at_every_step() -> None:
    previous: set[str] = set()
    for target in (10, 20, 40, 60, 80):
        current = {q for q, _ in sample_balanced(BUCKETS, target, seed=42)}
        assert previous <= current, f"target={target} dropped earlier queries"
        previous = current


def test_one_bucket_does_not_perturb_another() -> None:
    """Per-intent RNGs: a bucket changing size must not reshuffle its neighbours.

    Sharing one RNG across intents meant editing the prefilter for one intent
    silently resampled all the others, invalidating their cached judgements.
    """
    baseline = dict(_by_intent(sample_balanced(BUCKETS, 40, seed=42)))

    grown = dict(BUCKETS)
    grown["product_search"] = grown["product_search"] + [f"extra {i}" for i in range(20)]
    after = dict(_by_intent(sample_balanced(grown, 40, seed=42)))

    for intent, queries in baseline.items():
        if intent == "product_search":
            continue
        assert after[intent] == queries, f"{intent} was perturbed by an unrelated bucket"


def test_different_seeds_give_different_samples() -> None:
    a = {q for q, _ in sample_balanced(BUCKETS, 40, seed=1)}
    b = {q for q, _ in sample_balanced(BUCKETS, 40, seed=2)}
    assert a != b


def _by_intent(picked):
    out: dict[str, list[str]] = {}
    for question, intent in picked:
        out.setdefault(intent.value, []).append(question)
    return out
