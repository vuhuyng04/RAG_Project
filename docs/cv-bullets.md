# CV bullets

Every figure traces to a committed file under `eval/results/` and is enforced by
`tests/test_cv_claims.py` and `tests/test_readme_numbers.py`. Nothing here is an
estimate.

**Read "What NOT to claim" before using these.** Several obvious, attractive
claims are *not* supported by the data, and an interviewer who opens the repo
will find that out.

---

## Framing

Written as a project, in the present tense of what was built: **"Built"**,
**"Designed"**. The naive concatenation configuration is described as a
**baseline** — which is exactly what it is technically: embedding every field
concatenated together, the obvious first approach, kept as the experimental
control.

One constraint: do not claim "greenfield" or "from scratch" in any medium where
the repository's commit history is visible alongside it. Simply not narrating
the project's development is normal; asserting a history that contradicts the
record is not. Nothing below requires either.

---

## Recommended — 3 bullets

> **RAG Chatbot for Vietnamese E-commerce Product Q&A** · [github](https://github.com/vuhuyng04/RAG_Project) · 2025–26
>
> - Built a Vietnamese-language RAG system over a **483-product** e-commerce
>   catalogue, designed around a measurement that shaped the architecture:
>   product marketing copy is near-identical across the catalogue, so a naive
>   concatenate-every-field baseline leaves **56% of documents built from
>   verbatim-repeated passages** and null-propagation indexes **52 of 484
>   (10.7%) as the literal string `"nan"`**. Ingestion built on a validated
>   schema with discriminative-field selection holds both at **0%**.
> - Designed a **four-state corpus experiment** (baseline → clean → seeded
>   corruption → repaired) in which a corruption manifest serves as ground
>   truth, making data repair a *measurable* task rather than a judgement call:
>   **macro F1 0.911** across seven defect types. Scoring against the manifest
>   exposed four detector bugs invisible without ground truth, lifting macro F1
>   from **0.530 → 0.911**.
> - Built a deterministic evaluation harness over a golden set mined from
>   **897 real search-engine autocomplete queries** and ran a single-variable
>   A/B ladder across 8 retrieval configs. Parsed-price budget filtering gained
>   **+0.272 Recall@5**; cross-encoder reranking cost **~10× p95 latency for no
>   gain**. Growing the labelled set by two queries moved Recall@5 by **0.187**
>   and reversed one A/B conclusion — so the harness reports every retrieval
>   figure as provisional rather than as a result.

## Shorter — 2 bullets

> - Built a Vietnamese e-commerce RAG pipeline whose ingestion design follows
>   from measurement: catalogue-wide boilerplate leaves a naive concatenation
>   baseline with **56% of documents built from repeated passages** and **10.7%
>   indexed as literal `"nan"`**; schema validation with discriminative-field
>   selection holds both at **0%**.
> - Made data quality measurable via a **seeded corruption/repair experiment**
>   scored against a ground-truth manifest (**macro F1 0.911**) plus a
>   single-variable A/B ladder over 8 retrieval configs — surfacing that budget
>   filtering gains **+0.272 Recall@5**, that reranking costs ~10× latency for
>   nothing, and that two additional eval queries move Recall@5 by 0.187, which
>   the write-up reports rather than hides.

## One-liner

> Built an instrumented Vietnamese e-commerce RAG system: measured boilerplate
> contamination affecting **56% of documents** under a naive baseline, designed
> a corruption/repair experiment with manifest ground truth (**macro F1 0.911**),
> and A/B-tested 8 retrieval configs on a self-labelled golden set.

## Tech stack line

> Python · Qdrant (dense + BM25 sparse, RRF) · sentence-transformers
> (gte-multilingual) · Gemini · Streamlit · uv · pytest · ruff

---

## What NOT to claim

| Do not claim | Why |
|---|---|
| "Improved Recall@5 by X% over the baseline" | **`dense` scores 0.317 on clean vs 0.476 on the baseline.** The validated pipeline **does not currently beat the naive baseline** on retrieval, and the gap *widened* with a larger sample. (D9) |
| "BM25 hybrid hurts retrieval" | Held at n=12 (−0.087 in both pairings), **reversed at n=14** (+0.010 in one). Do not state it in either direction. (D10) |
| "Improved cosine separation" | Score spread is not a quality metric — when all five results are correct, flat scores are a success. Measured directly, clean scores *slightly worse*. (D7) |
| "Reranking improved relevance" | Recall@5 0.313 vs 0.317 for plain dense, at ~10× the latency. (D10) |
| Any RAGAS metric | Not run. The free tier allows 20 requests/day/model; the matrix needs ~1000–1500. |
| "Reduced hallucination by X%" | Citation validity is measured, but there is no before/after hallucination rate — the baseline has no citations to validate. |
| "80% of lost quality recovered" | That was the n=12 figure. It is **68.9%** at n=14 and will move again. Quote the detection F1 instead — it is measured against the full manifest, not a sample. |

## Caveats to state if asked

- **n = 14 answerable queries.** Two extra queries moved Recall@5 by 0.187 and
  flipped the sign of one A/B result. Every retrieval figure is provisional.
- **The corpus-quality and repair-detection numbers are not sampled** — computed
  over the full corpus (483–512 documents) and against a complete corruption
  manifest. They are the sturdy half of the project.
- **Latency figures are not sample estimates** — per-query timings, so the ~10×
  reranking cost stands independently of sample size.
- **Golden labels are LLM-assigned** over a pooled candidate set with
  deterministic budget and leakage backstops; not expert annotations, and not
  yet human-reviewed.

## Questions this project answers well

- *"How do you know your retrieval improved?"* → On retrieval it isn't proven,
  and the data currently points the other way. Here is the sample size that
  makes it unprovable, and here is what moving from 12 to 14 queries did to
  every number. What *is* proven is the corpus repair, scored against ground
  truth.
- *"How did you build the eval set without production data?"* → Mined real
  autocomplete queries rather than inventing them; the out-of-scope slice comes
  from genuine out-of-area and off-topic searches. Provenance recorded per row.
- *"What went wrong?"* → Four detector bugs found by manifest scoring; three
  citation-validator revisions, one of which made the metric nearly unable to
  fail; a hybrid conclusion that reversed on two extra queries after looking
  consistent across two cells; a rank-metric reproducibility bug that only
  appeared on the third identical run; and a golden-set sampler that reshuffled
  on every run, so a day's LLM quota bought one cache hit.
