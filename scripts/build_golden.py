"""Build eval/dataset/golden.jsonl from harvested queries.

    uv run python -m scripts.harvest_queries          # once, no LLM
    uv run python -m scripts.build_golden --target 40

Budget note (docs/decisions.md D7): the Gemini free tier allows 20 requests per
day per model. LLM classification of intent was therefore dropped — the
heuristics in `evalset/prefilter.py` already derive intent, facets and
answerability deterministically, and they are tested. The entire LLM budget goes
to the one thing heuristics genuinely cannot do: deciding which products
actually answer a query.

Gold URLs come from *pooled* retrieval (union of top-k across corpus states) so
the labels are not biased toward whichever system produced them.

Everything is written with reviewed=false. A human pass is required before the
numbers are quotable; run `scripts/review_golden.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter

from rag.config import DATA_DIR, DATASET_DIR, State
from rag.evalset.label import (
    enforce_budget,
    judge_candidates,
    leaks_title,
    load_golden,
    pool_candidates,
    save_golden,
)
from rag.evalset.prefilter import facets, prefilter
from rag.evalset.schema import GoldenQuery, Intent, QuerySource
from rag.llm import GeminiClient

RAW = DATA_DIR / "evalset" / "raw_queries.json"
OUT = DATASET_DIR / "golden.jsonl"

log = logging.getLogger("build_golden")

# Intents that can never be answered from the product catalogue. These carry the
# abstention metric, cost zero LLM calls, and — because they come from real
# harvested queries — are more realistic than anything hand-invented.
UNANSWERABLE_INTENTS = {Intent.POLICY, Intent.OUT_OF_CATALOGUE, Intent.OFF_TOPIC}

# Half the set is answerable (carries retrieval metrics), half is not (carries
# abstention). Slices stay large enough to mean something at n=40.
TARGET_MIX = {
    Intent.PRODUCT_SEARCH: 0.25,
    Intent.BUDGET_SEARCH: 0.15,
    Intent.ATTRIBUTE: 0.15,
    Intent.POLICY: 0.10,
    Intent.OUT_OF_CATALOGUE: 0.20,
    Intent.OFF_TOPIC: 0.15,
}

POOL_STATES = [State.CLEAN, State.BASELINE]


def sample_balanced(
    buckets: dict[str, list[str]], target: int, seed: int
) -> list[tuple[str, Intent]]:
    """Deterministic, *monotone* sample: raising `target` only adds queries.

    An earlier version shuffled the combined list at the end. That made the
    order depend on its length, so raising the target reordered everything and
    a run stopped at the daily quota judged a different first-N than the
    previous run — 19 calls spent for 1 cache hit. Each bucket is shuffled with
    its own seeded RNG and truncated, and the buckets are concatenated in a
    fixed order, so `target=80` is a strict superset of `target=40`.
    """
    picked: list[tuple[str, Intent]] = []
    for intent, share in TARGET_MIX.items():
        pool = list(buckets.get(intent.value, []))
        # Per-intent RNG: shuffling one bucket must not perturb the next.
        random.Random(f"{seed}:{intent.value}").shuffle(pool)
        want = round(target * share)
        chosen = pool[:want]
        picked.extend((q, intent) for q in chosen)
        log.info(
            "  %-18s want=%2d available=%3d picked=%2d", intent.value, want, len(pool), len(chosen)
        )
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ragas-subset", type=int, default=15)
    parser.add_argument("--max-llm-calls", type=int, default=18, help="stop before the daily cap")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    raw = json.loads(RAW.read_text(encoding="utf-8"))
    sidebar = set(raw.get("app_sidebar", []))
    all_queries = raw.get("autocomplete", []) + raw.get("app_sidebar", [])

    buckets = prefilter(all_queries, per_intent_cap=30)
    log.info("Sampling balanced mix (target=%d):", args.target)
    sampled = sample_balanced(buckets, args.target, args.seed)

    # Judge role: a different model from the chatbot's generator, so the gold
    # labels are not produced by the system being measured.
    llm = GeminiClient(role="judge")
    log.info("Judge model: %s", llm.model_name)

    # Resume from whatever is already labelled. At 20 LLM requests/day/model the
    # set can only grow a few answerable queries per run, so a run that re-judged
    # existing rows would never finish: the first version spent 19 calls for 1
    # cache hit and added 2 usable queries.
    existing: dict[str, GoldenQuery] = {}
    if OUT.exists():
        existing = {g.question: g for g in load_golden(OUT)}
        log.info("Resuming from %d already-labelled queries", len(existing))

    golden: list[GoldenQuery] = []
    dropped_leak = 0
    budget_exhausted = False
    newly_judged = 0

    for idx, (question, intent) in enumerate(sampled, 1):
        if question in existing:
            golden.append(existing.pop(question))
            continue

        f = facets(question)
        gq = GoldenQuery(
            id=f"q{idx:04d}",
            question=question,
            source=QuerySource.APP_SIDEBAR if question in sidebar else QuerySource.AUTOCOMPLETE,
            intent=intent,
            answerable=intent not in UNANSWERABLE_INTENTS,
            occasion=f["occasion"],
            recipient=f["recipient"],
            flower_type=f["flower_type"],
            form_factor=f["form_factor"],
            budget_max_vnd=f["budget_max_vnd"],
        )

        if gq.answerable:
            if llm.calls_made >= args.max_llm_calls:
                # Stop cleanly rather than burn the daily quota on retries.
                # Everything judged so far is written out, so the next run picks
                # up exactly here.
                budget_exhausted = True
                log.warning("LLM budget reached (%d calls) — stopping labelling", llm.calls_made)
                break

            candidates = pool_candidates(gq.question, POOL_STATES)
            gold = enforce_budget(
                judge_candidates(gq.question, candidates, llm), gq.budget_max_vnd, candidates
            )
            newly_judged += 1
            if gold and leaks_title(gq.question, candidates, gold):
                dropped_leak += 1
                continue
            gq.gold_urls = gold
            if not gold:
                # Nothing relevant in a 20-candidate pool: genuinely
                # unanswerable, not a retrieval failure. Scoring it as an
                # answerable query with an empty gold set would punish every
                # system for a labelling artefact.
                gq.answerable = False
                gq.notes = "no relevant product in pooled candidates"

        golden.append(gq)
        if idx % 5 == 0:
            log.info("labelled %d/%d  %s", idx, len(sampled), llm.stats())

    # Anything already labelled but no longer in the sample (a smaller --target,
    # or a prefilter change) is kept rather than discarded: it cost quota to
    # produce and re-judging it later would cost the same again.
    if existing:
        log.info("Carrying forward %d labelled queries outside the current sample", len(existing))
        golden.extend(existing.values())

    # Stable ids after a merge, so results files stay comparable across runs.
    for i, g in enumerate(sorted(golden, key=lambda g: g.question), 1):
        g.id = f"q{i:04d}"
    golden.sort(key=lambda g: g.id)

    # Seeded off the question text, not list position, so the subset is stable
    # as the set grows — comparing RAGAS across runs needs the same subset.
    answerable = [g for g in golden if g.answerable]
    for g in sorted(answerable, key=lambda g: random.Random(f"{args.seed}:{g.question}").random())[
        : args.ragas_subset
    ]:
        g.in_ragas_subset = True

    save_golden(golden, OUT)

    log.info("=" * 64)
    log.info(
        "golden: %d | answerable %d | abstain %d",
        len(golden),
        len(answerable),
        len(golden) - len(answerable),
    )
    log.info("newly judged this run: %d (%s)", newly_judged, llm.stats())
    log.info("dropped for title leakage: %d", dropped_leak)
    log.info("ragas subset: %d", sum(1 for g in golden if g.in_ragas_subset))
    log.info("intents: %s", dict(Counter(g.intent.value for g in golden)))
    log.info("sources: %s", dict(Counter(g.source.value for g in golden)))
    if budget_exhausted:
        log.warning(
            "INCOMPLETE — %d of %d sampled queries still unlabelled. Re-run tomorrow; "
            "labelled rows are persisted and will not be re-judged.",
            len(sampled) - len(golden),
            len(sampled),
        )
    log.warning("All rows reviewed=false. Run scripts/review_golden.py before quoting any number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
