# Engineering decisions

Short log of choices that are not obvious from the code, and what forced them.
Each entry records the evidence, so a reader (or future me) does not have to
re-derive it.

---

## D1 — `transformers` is pinned below 5.0

**Verified on this machine, 2026-08-12.**

`uv` initially resolved `transformers==5.15.0` / `sentence-transformers==5.7.0`.
Both `Alibaba-NLP/gte-multilingual-base` and `-reranker-base` **loaded without
error** and then crashed at inference:

```
IndexError: index 4752079519744 is out of bounds for dimension 0 with size 8
```

The index is a garbage pointer-sized value, thrown from inside the models'
custom architecture code (they require `trust_remote_code=True`), which was
written against the `transformers` 4.x internals. The failure is silent at load
time, which makes it particularly nasty: a naive setup looks fine until the
first query.

Pinned to `transformers>=4.44,<5` (resolves to 4.57.6) and
`sentence-transformers>=3.0,<5` (resolves to 4.1.0). Both models then work:

```
embedder    dim=768  norm=1.0  max_seq_length=8192
reranker    ("hoa sinh nhật", "Giỏ Hoa Sinh Nhật Vip M155") -> 0.6059
            ("hoa sinh nhật", "Hoa Khai Trương Rực Rỡ")     -> 0.5309
```

The original project pinned `transformers==4.43.1`, so this is consistent with
what it was actually developed against — it just was never written down.

---

## D2 — Gemini model is pinned to `gemini-2.5-flash`, not an alias

The original code used `gemini-2.0-flash`. As of 2026-08 that model is retired
and the API returns:

```
404 This model models/gemini-2.0-flash is no longer available.
```

`gemini-flash-latest` and `gemini-flash-lite-latest` both work, but an **alias
is disqualifying for this project**: the whole point is committed, reproducible
eval numbers, and an alias silently swaps the model underneath them. A result
in `eval/results/` must correspond to a known model.

`gemini-2.5-flash` is GA, multilingual, and cheap. Recorded in
`src/rag/config.py` with the reason inline.

---

## D3 — The `legacy` state is rebuilt, not snapshotted

The plan originally called for snapshotting the deployed `RAG_Huy` collection
to preserve a "before" column, since `products.csv` was never committed and the
data existed only in Qdrant Cloud.

That turned out to be impossible: the provided cluster
(`eu-west-1-0.aws`) is **empty — zero collections**. The notebook's cached
output references a different cluster (`europe-west3-0.gcp`), which is no longer
reachable with the current credentials.

Replacement, which is better anyway: `State.LEGACY` **reproduces the notebook's
indexing logic in code** — every field concatenated into one string (boilerplate
promo text included), the missing-`fillna` bug that writes literal `"nan"`, and
no non-product URL filtering — and builds it from the **same crawl** as
`State.CLEAN`.

Why this is stronger than a snapshot:

- Raw data is held constant, so the clean-vs-legacy delta is attributable to the
  pipeline alone rather than confounded by two different crawls.
- Anyone who clones the repo can rebuild every state and reproduce every number.
  A snapshot would have been a frozen artifact nobody could regenerate.

Cost: it measures a faithful reproduction of the old pipeline, not the exact
bytes that were live at `vuhuy-rag.streamlit.app`. Stated in the README rather
than glossed over.

---

## D4 — Crawl4AI was evaluated and dropped in favour of httpx + BeautifulSoup

The plan called for Crawl4AI, chiefly for its markdown/Pruning content filter as
a way to strip the repeated nav/promo boilerplate that wrecks the embeddings.

Probing the actual site first made that unnecessary. Sampling 10 random products
and measuring the fraction of tokens shared by >=60% of documents:

| field | shared tokens | unique values |
|---|---|---|
| `div.product-short-description` | **96.3%** | 5/10 |
| `#tab-description` | **100.0%** | 2/10 |
| `meta[name=description]` | **15.3%** | 10/10 |
| `h1.product_title` | 27.0% | 10/10 |

The site already exposes clean, per-product, human-written copy in
`meta[name=description]`. There is no boilerplate to strip — there is a
*correct field to select*. A content filter would at best approximate what a
one-line CSS selector gets exactly.

Against that, Crawl4AI costs a Playwright + Chromium install (~400MB) to render
pages that are static server-rendered HTML: everything needed is present in the
initial response, verified by fetching with plain `httpx`.

So: `httpx.AsyncClient` + `lxml`-backed BeautifulSoup, bounded by a semaphore.
Reproduce the measurement with `uv run python -m scripts.probe_boilerplate`.

This is a deliberate deviation from the approved plan, recorded here rather than
made silently.

---

## D5 — Sale prices must be read from `<ins>`, not the first amount

WooCommerce renders a discounted product as `<del>original</del><ins>current</ins>`.
The combined text of `p.price` reads:

```
1.400.000 ₫ Giá gốc là: 1.400.000₫. 1.200.000 ₫ Giá hiện tại là: 1.200.000₫.
```

Naively parsing the first number yields **1.400.000** — the struck-through
original — which would make every budget filter wrong on exactly the products a
price-sensitive customer is most likely to ask about. `extract_price()` prefers
`p.price ins .woocommerce-Price-amount` and falls back to the regular amount.

---

## D6 — "Cosine score spread" is NOT a quality metric. Do not put it on a CV.

The original diagnosis leaned on a number from the notebook's cached output: the
top hit beat an unrelated product by only **0.02**, with all five results inside
a flat 0.68-0.71 band. That was read as "retrieval cannot tell products apart."

Having built both states, this was measured directly — mean spread between the
top-1 and 5th result, across 6 representative Vietnamese queries:

| state | mean spread (top1 − top5) |
|---|---|
| legacy | 0.0252 |
| clean  | **0.0226** |

**The clean corpus has a *slightly smaller* spread.** The metric moved the wrong
way, and it was right to check rather than assume.

The reason is that spread does not measure what it appears to. For
`hoa chia buồn đám tang`, clean returns five "Hoa Chia Buồn" products scoring
0.842/0.840/0.838/0.827/0.827 — a tight band, because **all five are correct**.
Flat scores across five right answers is a success, not a failure. Spread
conflates "the model is undecided" with "there are many good matches."

Consequences, which are binding:

1. No claim about score separation goes in the README or the CV. The 0.02 figure
   is a symptom that motivated the investigation, not evidence of improvement.
2. The only defensible retrieval claim is **Recall@k / MRR / nDCG against the
   labelled golden dataset**. Until that exists, "clean is better than legacy" is
   unproven — the corpus-health numbers below are real, but they measure the
   corpus, not the retrieval.

What *is* established at this point, from `eval/results/quality_*.json`:

| metric | legacy | clean |
|---|---|---|
| documents | 484 | 483 |
| boilerplate ratio in embedded text | 49.6% | **20.9%** |
| documents embedding to literal `"nan"` | **52** | **0** |
| non-product URLs indexed | 1 | 0 |
| price parse rate | 100% | 100% |

52 of 484 legacy documents (10.7%) are unretrievable junk, and that is a real,
reproduced defect. Whether removing them improves answer quality is the next
measurement, not an assumption.

---

## D7 — The Gemini free tier is 20 requests/day/model, not 1500

Discovered by hitting it, 2026-08-12:

```
429 ResourceExhausted
Quota exceeded for metric: generativelanguage.googleapis.com/
  generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash
quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
quota_value: 20
```

The plan assumed ~1500 requests/day and built a whole cost strategy (subset
selection, caching, multi-day spreading) on that figure. The real figure is
**20 per day, per model, per project** — roughly two orders of magnitude less.

Probing confirmed the quota is per-model: after `gemini-2.5-flash` was
exhausted, `gemini-2.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.6-flash` and
others still answered. That yields ~160 calls/day in aggregate, but **rotating
judge models across an evaluation is not acceptable** — the judge is part of the
measuring instrument, and changing it between cells makes the cells
incomparable. Consistency beats volume here.

Consequences:

* Golden-set construction (classify + judge) needs ~45-110 calls depending on
  batching. Feasible on free tier over several days, or in one session with
  billing.
* The RAGAS axis needs roughly **1000-1500 calls** (4 metrics x 40 queries x 6
  matrix cells, each metric costing several calls internally). This is **not
  feasible on the free tier at any batching**, and the disk cache does not help
  a first run.

Mitigations already built regardless of which path is taken:

* `src/rag/llm.py` content-addressed disk cache — a re-run costs zero calls, so
  only genuinely new work consumes quota.
* Token-bucket throttle + exponential backoff on 429.
* Batched classification (20 queries/call) rather than one call per query.

Pre-filtering the 897 harvested queries down to a few hundred candidates before
classification is a further reduction and should be done regardless: classifying
all 897 was wasteful even under the assumed quota.

---

## D8 — Generator and judge are different models

`Settings.gemini_model` (`gemini-2.5-flash`) is what the chatbot answers with.
`Settings.judge_model` (`gemini-3.6-flash`) is what assigns gold labels and will
run RAGAS. `GeminiClient(role="judge")` selects the latter.

Two independent reasons:

1. **Methodology.** A model grading its own output inflates faithfulness and
   answer-relevancy. The judge should not be the system under test.
2. **Quota.** The free-tier limit is per-model (D7), so separating the roles
   doubles the daily budget for free instead of having dataset construction and
   answer generation compete for the same 20 calls.

The model name is part of the LLM disk-cache key, so a cached response from one
model is never silently reused for the other.

---

## D9 — LLM classification of query intent was dropped

The plan had an LLM classify all 897 harvested queries into intent + facets.
Measured against the free-tier budget that is ~45 calls — more than two days of
quota — spent on something the deterministic heuristics in
`evalset/prefilter.py` already do, and which is now covered by 24 passing tests.

The whole LLM budget goes to gold-URL judging instead: deciding which products
actually answer a query is the one part heuristics cannot do.

Building those heuristics surfaced a bug class worth recording, because it would
have silently corrupted every sliced metric. Matching keywords against
diacritic-folded Vietnamese with plain substring containment produces false
positives that look absurd once seen:

| query | wrong facet | why |
|---|---|---|
| `có freeship không` | flower_type = hồng | `không` → `khong`, contains `hong` |
| `có freeship không` | recipient = thầy cô | `có` → `co`, matches key `cô` |
| `shop có giao nhanh không` | recipient = thầy cô | `có giao` → `co giao` = `cô giáo` |
| `lẵng hoa 500k` | flower_type = lan | `lẵng` → `lang`, contains `lan` |
| `giỏ hoa quả hải phòng` | flower_type = hồng | `phòng` → `phong`, contains `hong` |
| `hoa khai trương bình dương` | form_factor = bình | province name read as a vase |
| `hoa khai trương cầu giấy` | intent = off_topic | `giấy` read as paper flowers |

Fixes: whole-token phrase matching instead of substring; place names stripped
before facet extraction; ambiguous single-syllable keys (`cô`, `thầy`, `cô giáo`)
removed entirely; `giấy` narrowed to `hoa giấy`. Each row above is now a
regression test in `tests/test_prefilter.py`.

---

## D10 — First retrieval numbers do NOT support "clean beats legacy" (n=12)

Measured 2026-08-12, `dense` config across all four states, golden set of 33
queries of which **only 12 are answerable**:

| state | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| legacy | **0.520** | **0.875** | **0.750** |
| clean | 0.504 | 0.683 | 0.671 |
| corrupt | 0.296 | 0.500 | 0.452 |
| repaired | 0.463 | 0.600 | 0.582 |

Legacy beats clean on every metric. Two things follow, and neither is
negotiable:

**1. No clean-vs-legacy claim may be made at this sample size.** A 0.016 gap on
n=12 is noise. One query changing outcome moves recall@5 by ~0.08. The golden
set has to grow before this comparison means anything; it stopped at 33 because
the free-tier quota ran out mid-labelling (D7), not because 33 was the target.

**2. Tuning stopped deliberately.** The obvious next move was to hunt for a
variant that reverses the result. One was tried — rebuilding `clean` without the
price band in the embedded text (`EMBED_PRICE_BAND=false`):

| variant | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| clean, with price band | 0.504 | 0.683 | 0.671 |
| clean, without | 0.452 | **0.781** | 0.605 |

The metrics move in opposite directions, which is what noise looks like.
Continuing to try variants against 12 queries would be selecting a
configuration by overfitting the evaluation set — precisely the failure this
project exists to demonstrate awareness of. Further embed-text experiments are
deferred until the golden set is large enough, and any variant chosen this way
would have to be reported as such.

### What the data *does* support

The corrupt/repaired contrast is large enough to read even at n=12:

* corrupt 0.296 -> repaired 0.463, against a clean baseline of 0.504
* recovery = (0.463 - 0.296) / (0.504 - 0.296) = **80.3% of the lost retrieval
  quality restored**

Combined with the detection scores against the corruption manifest
(macro F1 **0.911**, overall F1 **0.972**, `eval/results/repair_detection.json`),
the corruption/repair axis is currently the only part of this project with
numbers worth quoting — and notably it is the part that does not depend on the
LLM quota at all.

A plausible explanation for legacy holding up, worth testing once n is
adequate: legacy embeds the promo boilerplate, which is flower-vocabulary-rich
("hoa tươi", "sang trọng", "freeship"). Against *generic* queries like
"hoa tặng mẹ" that shared vocabulary may act as a weak topical prior rather than
pure noise. If true, the clean pipeline's advantage should appear on *specific*
queries and disappear on generic ones — a slice comparison the harness already
supports and which would be a far more interesting finding than a single
averaged number.

---

## D11 — Cross-encoder reranking costs 10x latency and did not help (n=12)

`eval/results/retrieval_*_clean.json`, all on the clean corpus:

| config | recall@5 | mrr@5 | ndcg@5 | abstain F1 | p95 |
|---|---|---|---|---|---|
| dense | **0.504** | 0.683 | 0.671 | 0.000 | **306 ms** |
| dense_threshold | **0.504** | 0.683 | 0.671 | 0.250 | 354 ms |
| dense_rerank | 0.405 | 0.646 | 0.614 | 0.250 | 2 958 ms |
| full (+ budget filter) | 0.493 | **0.750** | **0.708** | **0.320** | 3 985 ms |

Reranking with `gte-multilingual-reranker-base` on CPU costs roughly **10x**
p95 latency (306 ms -> 2 958 ms; ~2 s for 20 pairs, measured warm) and **lowered**
recall@5. It is not carrying its weight here.

The budget filter does earn its place: `full` recovers most of the recall the
reranker lost while giving the best MRR@5, nDCG@5 and abstention F1 of any
config. That is consistent with the harvested query distribution, where budget
phrasings ("hoa khai trương 500k") are common and real.

Abstention is the clearest monotone win: F1 0.000 -> 0.250 -> 0.320 as the
threshold and then the filter are added. The 0.000 baseline is literal — the
original system never abstained on anything.

Caveats, both binding: n=12 answerable queries, so the quality columns are
indicative only. Latency is *not* subject to that caveat — it is per-query
timing, not an estimate from a sample, so the 10x figure stands on its own.

---

## D12 — The citation validator was nearly vacuous, and was measured to prove it

The first version flagged any sentence containing "giá", "mẫu", "sản phẩm" or a
form-factor noun. On a real generated answer that produced two false positives,
including a question ("Bạn muốn tham khảo thêm ... không ạ?").

Narrowing it to "currency amount or product code" fixed those — and quietly
broke the metric in the other direction. Tested against five realistic
hallucinated sentences, the narrowed rule caught **2/5**. It was blind to
exactly what a RAG system most often invents:

```
Mẫu này dùng hoa hướng dương và lan hồ điệp trắng.     -> missed
Kệ hoa này cao khoảng hai mét, giao trong ngày.        -> missed
Sản phẩm được thiết kế hai tầng với tông đỏ chủ đạo.   -> missed
```

A metric that cannot fail is not evidence, and `citation validity` was slated
for the CV. Resolution: two tiers, reported separately.

* `invalid_index_rate` — citations pointing at sources that do not exist. Purely
  structural, no heuristics. Quotable without caveat.
* `uncited_price_or_code_claim_rate` — narrow, reliable.
* `uncited_descriptive_claim_rate` — attribute-vocabulary heuristic. Lower
  precision, so it is reported but deliberately excluded from `is_valid`.

Re-measured: **5/5** hallucinated sentences caught, **0/5** false positives on
benign pleasantries. Every case is a test in `tests/test_generation.py`.

Attribute-level grounding is not fully decidable without a judge — that is what
RAGAS Faithfulness is for, and why this validator complements it rather than
replacing it.

---

## D13 — BM25 hybrid hurts; the budget filter is what actually helps

Full A/B ladder on the clean corpus. Each rung adds exactly one mechanism.

| config | recall@5 | mrr@5 | ndcg@5 | abstain F1 | p95 |
|---|---|---|---|---|---|
| legacy | 0.504 | 0.683 | 0.671 | 0.000 | 604 ms |
| dense | 0.504 | 0.683 | 0.671 | 0.000 | 1 113 ms |
| dense_threshold | 0.504 | 0.683 | 0.671 | 0.250 | 476 ms |
| **dense_budget** | **0.654** | **0.917** | **0.878** | 0.091 | **582 ms** |
| hybrid | 0.417 | 0.694 | 0.545 | 0.000 | 932 ms |
| hybrid_budget | 0.567 | 0.903 | 0.732 | 0.091 | 538 ms |
| dense_rerank | 0.405 | 0.646 | 0.614 | 0.250 | 5 894 ms |
| full | 0.493 | 0.750 | 0.708 | **0.320** | 5 425 ms |

**BM25 hybrid consistently hurts**, in both pairings:

* dense 0.504 -> hybrid 0.417 (**-0.087**)
* dense_budget 0.654 -> hybrid_budget 0.567 (**-0.087**)

The likely cause is stated in `retrieval/sparse.py`: fastembed has no Vietnamese
analyser, so BM25 runs an English stemmer and English stopwords over Vietnamese
text, surfacing weak lexical matches that RRF then promotes into the top-5.

### The control cell that changed the conclusion

`dense_budget` was **added after the first run**, which had only `dense`,
`hybrid` and `hybrid_budget`. On that evidence `hybrid_budget` (0.567) was the
top scorer and the natural write-up was "hybrid retrieval improved recall".

That would have been wrong. Adding the control shows `dense_budget` (0.654)
beats it, and that the entire gain belongs to the **budget filter** — parsing
`price_vnd` into an integer payload index and filtering on it — which the legacy
pipeline could not do at all because price was stored as a formatted string.

The lesson generalises past this project: an A/B ladder missing one rung will
attribute a gain to whichever mechanism happened to be bundled with it.

### The best configuration is also nearly the cheapest

`dense_budget` wins on all three quality metrics at 582 ms p95, an order of
magnitude below the reranking configs (~5.5 s). `full` retains one advantage —
the best abstention F1 (0.320) — because it is the only config combining a
cosine threshold with the filter. A `dense_budget_threshold` rung would settle
whether reranking contributes anything there; it is the obvious next cell.

All figures are n=12 answerable queries and remain provisional (D10).

---

## D14 — Hybrid retrieval was not reproducible until ties were broken explicitly

Three identical `hybrid` runs, same seed, same data:

| run | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| 1 | 0.417 | 0.611 | 0.517 |
| 2 | 0.417 | 0.611 | 0.522 |
| 3 | 0.417 | **0.694** | **0.541** |

Dense retrieval was bit-identical across the same runs. The difference is RRF:
a document at rank *r* in one prefetch list and absent from the other always
scores exactly 1/(60+*r*), so large groups of results share a score and their
relative order falls out of Qdrant's internal ordering. Set-based recall@5 was
unaffected; every rank-sensitive metric drifted.

An evaluation number that changes between runs cannot be committed, so hybrid
results are now sorted by `(-score, url)`. The tie-break is arbitrary but
stable. Verified: three consecutive runs now return identical values to four
decimal places.

This bug was invisible in a single run — it only appears when a result is
reproduced, which is why the harness is run repeatedly rather than once.

---

## D15 — Three UI bugs that only surfaced by looking at the running app

Health checks and unit tests passed throughout. None of these would have been
caught without opening a browser and clicking.

**1. Abstention rendered as if it were an answer.** When the model correctly
refused ("thông tin về chính sách freeship không có trong danh sách sản phẩm"),
the page still displayed a prominent grid of five flower photos beneath it, each
labelled "not used" — visually identical to the broken legacy behaviour the
abstention feature exists to replace. Now, when nothing is cited, the sources
collapse into a closed expander: still inspectable, no longer presented as an
answer.

**2. `import ui` broke the deployed entry point, not the local one.** Streamlit
puts a script's own directory on `sys.path`, so `app/chatbot.py` imported its UI
module fine when run directly. The root `chatbot.py` shim — which is what
Streamlit Cloud executes — loads it via `runpy`, where `app/` is *not* on the
path, and the whole app died with `ModuleNotFoundError`. `app/chatbot.py` now
adds its own directory. `tests/test_app_entrypoint.py` runs the shim path in a
subprocess so this cannot regress silently.

**3. Citation scope is the block, not the sentence.** Observed on a real answer
with five bullets, every one correctly cited:

```
- **Hoa Chia Buồn M20**: Vòng hoa ... lòng thành kính. Giá 1.100.000 đ [1].
```

Splitting on the full stop leaves `...M20: Vòng hoa ... lòng thành kính.` with
no marker, and the sentence-scoped validator reported **5 uncited claims out of
5 correctly cited bullets**. A marker at the end of a list item grounds that
item, so the check now scopes to the block.

This is the third correction to `uncited_claim_rate` (see D12). The pattern is
worth stating: a validator is itself a measuring instrument, and it needs its
own false-positive and false-negative testing before its output is quotable.
After this fix the same answer reports one flag — a genuinely ungrounded closing
sentence — at the low-confidence descriptive tier, which is correct.

---

## D16 — Collection naming

Collections are `flowers_{legacy,clean,corrupt,repaired}` via
`Settings.collection_for()`. The old name `RAG_Huy` described the author, not
the data; these describe the corpus state, which is what the eval matrix
indexes on.
