# CV bullets

Every figure below traces to a committed file under `eval/results/` and is
enforced by `tests/test_readme_numbers.py`. Nothing here is an estimate.

**Read the "what not to claim" section before using these.** Several obvious,
attractive claims are *not* supported by the current data, and an interviewer
who opens the repo will find that out.

---

## Recommended — 3 bullets

> **RAG Chatbot for Vietnamese E-commerce Product Q&A** · [github](https://github.com/vuhuyng04/RAG_Project) · 2025–26
>
> - Diagnosed why a working RAG chatbot could not distinguish products: the
>   indexing pipeline embedded marketing boilerplate identical across the
>   catalogue, leaving **56% of documents built from verbatim-repeated passages**
>   and **52 of 484 (10.7%) embedded as the literal string `"nan"`**. Rebuilt
>   ingestion around a validated schema and discriminative-field selection —
>   both defects to **0%**.
> - Designed a **four-state corpus experiment** (legacy → clean → seeded
>   corruption → repaired) where a corruption manifest serves as ground truth,
>   making data repair a *measurable* task: **macro F1 0.911** across seven
>   defect types, recovering **80% of the Recall@5** lost to corruption. The
>   manifest exposed four detector bugs invisible without ground truth,
>   lifting macro F1 from **0.530 → 0.911**.
> - Built a deterministic evaluation harness over a **golden set mined from
>   897 real search-engine autocomplete queries** (Recall@k, MRR, nDCG,
>   abstention F1, citation validity, staged latency) and ran a
>   single-variable A/B ladder
>   across 8 retrieval configs: parsed-price budget filtering gained
>   **+0.150 Recall@5**, while **BM25 hybrid cost −0.087** and cross-encoder
>   reranking cost **~10× p95 latency for no gain** — both reported as negative
>   results.

## Shorter — 2 bullets

> - Rebuilt a Vietnamese e-commerce RAG pipeline after diagnosing that
>   catalogue-wide boilerplate left **56% of documents built from repeated
>   passages** and **10.7% indexed as literal `"nan"`**; both reduced to **0%**
>   via schema validation and discriminative-field selection.
> - Made data quality measurable with a **seeded corruption/repair experiment**
>   scored against a ground-truth manifest (**macro F1 0.911**, 80% of lost
>   Recall@5 recovered) plus a single-variable A/B ladder over 8 retrieval
>   configs — reporting negative results (BM25 hybrid −0.087 Recall@5,
>   reranking ~10× latency for no gain) alongside the wins.

## One-liner

> Rebuilt and instrumented a Vietnamese e-commerce RAG chatbot: diagnosed
> boilerplate contamination affecting **56% of documents**, designed a
> corruption/repair experiment with manifest ground truth (**macro F1 0.911**),
> and A/B-tested 8 retrieval configs on a self-labelled golden set.

---

## What NOT to claim

These are tempting and unsupported. Using them will not survive a repo read.

| Do not claim | Why |
|---|---|
| "Improved Recall@5 by X% over the original" | **`dense` scores 0.504 on the clean corpus vs 0.520 on legacy.** The clean pipeline does not currently beat legacy on retrieval. The corpus-health numbers are solid; the retrieval improvement is unproven. (D10) |
| "Improved cosine separation / fixed the 0.02 gap" | Score spread is not a quality metric — when all five results are correct, flat scores are a success. Measured directly, clean scores *slightly worse* than legacy. (D6) |
| "Hybrid search improved retrieval" | It measured **worse** in both pairings (−0.087). (D13) |
| "Reranking improved relevance" | Recall@5 dropped 0.504 → 0.405 at ~10× the latency. (D11) |
| Any RAGAS metric | Not run. The free tier allows 20 requests/day/model; the matrix needs ~1000–1500. |
| "Reduced hallucination by X%" | Citation validity is measured, but no before/after hallucination rate exists — the legacy system had no citations to validate. |

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

## Questions this project is designed to answer well

- *"How do you know your retrieval improved?"* → The honest answer is that on
  retrieval, it isn't proven yet, and here is the exact sample size that makes
  it unprovable. What *is* proven is the corpus repair, and here is the ground
  truth it was scored against.
- *"How did you build the eval set without production data?"* → Mined real
  autocomplete queries rather than inventing them; the out-of-scope slice comes
  from genuine out-of-area and off-topic searches. Provenance is recorded per
  row.
- *"What went wrong?"* → Four detector bugs found by manifest scoring, three
  citation-validator revisions (including one that made the metric nearly
  unable to fail), and a hybrid-retrieval result that reversed once the missing
  control cell was added.
