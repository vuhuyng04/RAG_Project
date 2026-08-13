# Evaluation methodology

Every number reported anywhere in this repository must be reproducible by a
command listed here, and must correspond to a committed file under
`eval/results/`. Numbers that cannot be traced that way do not go in the README
and do not go on a CV.

---

## 1. Where the queries come from

This project evaluates a **publicly crawled third-party catalogue**. It has no
production traffic, no support tickets and no site search logs, and nothing here
pretends otherwise. Query provenance is recorded per row so a reader can weight
the evidence themselves.

| source | n | what it is |
|---|---|---|
| `autocomplete` | most | Real aggregated user queries mined from search-engine autocomplete for Vietnamese flower-buying seeds, expanded one level. These are strings people actually typed. |
| `app_sidebar` | 8 | Questions written from domain knowledge of what customers ask, covering intents autocomplete does not surface. |
| `synthetic` | 0 | LLM-generated from product records. Not used: prone to echoing document wording, which inflates retrieval scores. |

Autocomplete is the closest legitimate proxy for production traffic available
here, and it produced query types that would not have been invented:

- budget phrasings — `hoa khai trương 500k`, `hoa chia buồn giá rẻ`
- recipient distinctions — `bó hoa sinh nhật cho nam` / `cho nữ`
- **out-of-area requests** — `shop hoa tươi cần thơ`, `hoa chia buồn đà nẵng`
  (the shop serves Ho Chi Minh City only)
- **genuinely off-topic** — `hoa tặng mẹ lớp 4`, `hoa tặng mẹ bài đọc`
  (Vietnamese school essay assignments)

The last two groups make the unanswerable slice realistic rather than invented.

```bash
uv run python -m scripts.harvest_queries      # 897 raw queries, no LLM
```

## 2. From 897 raw queries to a labelled set

**Intent, facets and answerability are derived deterministically**
(`src/rag/evalset/prefilter.py`), not by an LLM. An LLM classification stage was
built first and then dropped: the free tier allows 20 requests/day/model
(`docs/decisions.md` D4), and the heuristics do the job under test. The whole
LLM budget goes to gold-URL judging, which heuristics cannot do.

Building them surfaced a bug class worth knowing about: matching keywords
against **diacritic-folded Vietnamese with substring containment** produces
false positives that are invisible until you look. `không` folds to `khong`,
which contains `hong` (hồng/rose), so `có freeship không` was tagged with
`flower_type=hồng`. Seven such cases are now regression tests in
`tests/test_prefilter.py`. The fix is whole-token phrase matching plus stripping
place names before facet extraction.

**Gold URLs come from pooled retrieval.** Candidates are the union of top-k from
every indexed corpus state, deduplicated by URL, then judged. Pooling is the
standard TREC construction and matters specifically here: labelling only what
the *clean* index returns would bake that system's behaviour into the ground
truth and make it unbeatable by construction.

Two deterministic backstops run after the judge:

- **budget enforcement** — golds priced above a stated budget are dropped. LLMs
  are unreliable at numeric comparison and a wrong gold silently corrupts every
  downstream Recall figure.
- **title-leakage rejection** — a query overlapping ≥80% of a gold product's
  title tokens is discarded rather than counted as an easy win.

The judge is `gemini-3.6-flash`, **deliberately a different model from the
chatbot's generator** (`gemini-2.5-flash`). A model grading its own output
inflates faithfulness and relevancy; the per-model quota also means the two
roles draw on separate daily budgets.

```bash
uv run python -m scripts.build_golden --target 80
```

> **Current status: the golden set is incomplete.** 35 queries (14 answerable)
> of a 77-query sample. Labelling advances a few queries per run against the
> 20-request/day/model quota, and **resumes**: already-labelled rows are loaded
> and skipped, so a run only spends quota on new ones. Every row is
> `reviewed=false`. Retrieval numbers below are **provisional** and the harness
> prints a warning saying so.
>
> The sample is *monotone* in `--target` — raising it adds queries without
> reshuffling existing ones — which is what makes daily progress accumulate.
> An earlier version shuffled the combined list at the end, so each run judged a
> different first-N and a full day's quota bought exactly one cache hit.
> `tests/test_golden_sampling.py` pins the property.

## 3. Metrics

### Deterministic — no LLM, run on every matrix cell

| metric | notes |
|---|---|
| Recall@1/5/10 | set-based: a query with 4 acceptable products does not score 1.0 for finding one |
| Hit@k, MRR@5, nDCG@5 | binary relevance |
| Abstention precision / recall / F1 | over the unanswerable slice |
| `invalid_index_rate` | citations pointing at sources that do not exist — purely structural |
| `uncited_price_or_code_claim_rate` | narrow, reliable |
| `uncited_descriptive_claim_rate` | vocabulary heuristic, lower precision, excluded from `is_valid` |
| latency p50/p95 | split by stage: embed / search / rerank / LLM |

Retrieval metrics are computed **on returned point URLs only**, independent of
prompt rendering. Folding presentation fixes into them would make the corpus
delta unattributable.

**Recall is reported twice**, and the pair is the point of the three-state
experiment:

- **raw** — counts every labelled gold, including ones corruption removed from
  the index. The honest end-to-end number; a user does not care *why* an answer
  is missing.
- **restricted to present golds** — counts only golds still in the collection.
  Isolates "did the retriever find it" from "was it there at all".

They are identical in the clean state. Where they diverge, the gap is the damage
corruption did to the corpus rather than to the retriever.

Results are also **sliced** by source, intent, answerability and
budget-constraint. An averaged headline hides which slice is carrying it.

### LLM-judged (RAGAS) — subset only

Faithfulness, Answer Relevancy, Context Recall, Context Precision. Restricted to
a fixed seeded subset, identical across configs — comparing different subsets
would be meaningless. Blocked on quota; see §5.

## 4. Running it

```bash
uv run python -m eval.run_retrieval                     # default matrix
uv run python -m eval.run_retrieval --configs dense hybrid --states clean baseline
```

Writes one JSON per cell plus per-query records to `eval/results/`.

## 5. Known limitations

Stated plainly because they bound every claim made from this data.

1. **n=14 answerable queries.** Growing the set from 12 to 14 moved Recall@5 by
   0.187 and reversed one A/B conclusion outright. No clean-vs-baseline
   conclusion is supportable at this size, and the current data in fact shows
   the baseline *ahead* (`docs/decisions.md` D9).
2. **No human review pass yet.** All rows are `reviewed=false`.
3. **Gold labels are LLM-assigned** over a pooled candidate set, with
   deterministic budget and leakage backstops. They are not expert annotations.
4. **RAGAS has not been run.** The free tier is 20 requests/day/model; the full
   matrix needs ~1000-1500 calls.
5. **One shop, one language, one domain.** Nothing here generalises to other
   catalogues without re-measurement.
6. **BM25 has no Vietnamese analyser.** fastembed rejects
   `language="vietnamese"` and falls back to English stemming and stopwords.

## 6. Negative results

Kept deliberately. A repository that only reports what worked is not evidence of
measurement.

- **Cosine score spread is not a quality metric.** When all five results are
  correct, flat scores are a success. Measured directly, clean scores *slightly
  worse* on spread than the baseline. (D7)
- **Clean does not currently out-retrieve the naive baseline** on Recall@5 /
  MRR@5 / nDCG@5 at n=14, and the gap widened with the larger sample. (D9)
- **The BM25 hybrid result reversed** between n=12 (-0.087 in both pairings)
  and n=14 (+0.010 in one, -0.062 in the other). No direction is claimed. (D10)
- **Cross-encoder reranking cost ~10x p95 latency for no Recall@5 gain.** This
  one held across both sample sizes. (D10)
- **Tuning was stopped deliberately.** One embed-text variant was tried; its
  metrics moved in opposite directions, i.e. noise. Continuing to search for a
  variant that reverses the result would be selecting a configuration by
  overfitting the evaluation set. (D9)
