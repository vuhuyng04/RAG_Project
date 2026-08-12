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
control. Describing it that way is accurate.

One constraint worth stating: do not claim "greenfield" or "from scratch" in
any medium where the repository's commit history is visible alongside it. Simply
not narrating the project's development is normal; asserting a history that
contradicts the record is not. Nothing below requires either.

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
>   **macro F1 0.911** across seven defect types, recovering **80% of the
>   Recall@5** lost to corruption. Scoring against the manifest exposed four
>   detector bugs invisible without ground truth, lifting macro F1 from
>   **0.530 → 0.911**.
> - Built a deterministic evaluation harness over a golden set mined from
>   **897 real search-engine autocomplete queries** (Recall@k, MRR, nDCG,
>   abstention F1, citation validity, staged latency) and ran a single-variable
>   A/B ladder across 8 retrieval configs: parsed-price budget filtering gained
>   **+0.150 Recall@5**, while **BM25 hybrid cost −0.087** and cross-encoder
>   reranking cost **~10× p95 latency for no gain** — both reported as negative
>   results.

## Shorter — 2 bullets

> - Built a Vietnamese e-commerce RAG pipeline whose ingestion design follows
>   from measurement: catalogue-wide boilerplate leaves a naive concatenation
>   baseline with **56% of documents built from repeated passages** and **10.7%
>   indexed as literal `"nan"`**; schema validation with discriminative-field
>   selection holds both at **0%**.
> - Made data quality measurable via a **seeded corruption/repair experiment**
>   scored against a ground-truth manifest (**macro F1 0.911**, 80% of lost
>   Recall@5 recovered) plus a single-variable A/B ladder over 8 retrieval
>   configs — reporting negative results (BM25 hybrid −0.087 Recall@5,
>   reranking ~10× latency for no gain) alongside the wins.

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

These are tempting and unsupported. Using them will not survive a repo read.

| Do not claim | Why |
|---|---|
| "Improved Recall@5 by X% over the baseline" | **`dense` scores 0.504 on clean vs 0.514 on the baseline.** The validated pipeline **does not currently beat the naive baseline** on retrieval. The corpus-health numbers are solid; the retrieval improvement is unproven. (D9) |
| "Improved cosine separation" | Score spread is not a quality metric — when all five results are correct, flat scores are a success. Measured directly, clean scores *slightly worse*. (D7) |
| "Hybrid search improved retrieval" | It measured **worse** in both pairings (−0.087). (D10) |
| "Reranking improved relevance" | Recall@5 dropped 0.504 → 0.405 at ~10× the latency. (D10) |
| Any RAGAS metric | Not run. The free tier allows 20 requests/day/model; the matrix needs ~1000–1500. |
| "Reduced hallucination by X%" | Citation validity is measured, but there is no before/after hallucination rate — the baseline has no citations to validate. |

## Caveats to state if asked

- **n = 12 answerable queries.** Labelling stopped mid-run at the daily LLM
  quota. One query changing outcome moves Recall@5 by ~0.08, so every retrieval
  figure is provisional. The corpus-quality and repair-detection numbers are
  **not** subject to this — they are computed over the full corpus (483–512
  documents) and against a complete manifest.
- **Golden labels are LLM-assigned** over a pooled candidate set with
  deterministic budget and leakage backstops; not expert annotations, and not
  yet human-reviewed.
- **Latency figures are not sample estimates** — they are per-query timings, so
  the ~10× reranking cost stands independently of the sample size.

## Questions this project answers well

- *"How do you know your retrieval improved?"* → On retrieval it isn't proven
  yet, and here is the exact sample size that makes it unprovable. What *is*
  proven is the corpus repair, and here is the ground truth it was scored
  against.
- *"How did you build the eval set without production data?"* → Mined real
  autocomplete queries rather than inventing them; the out-of-scope slice comes
  from genuine out-of-area and off-topic searches. Provenance is recorded per
  row.
- *"What went wrong?"* → Four detector bugs found by manifest scoring; three
  citation-validator revisions, one of which made the metric nearly unable to
  fail; a hybrid-retrieval conclusion that reversed once the missing control
  cell was added; and a rank-metric reproducibility bug that only appeared on
  the third identical run.
