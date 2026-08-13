# Engineering decisions

Choices that are not obvious from the code, and the evidence that forced them.
Each entry records how it was measured, so nobody has to re-derive it.

---

## D1 — `transformers` is pinned below 5.0

`uv` resolves `transformers==5.15` by default. Both
`Alibaba-NLP/gte-multilingual-base` and `-reranker-base` **load without error**
under it and then crash at inference:

```
IndexError: index 4752079519744 is out of bounds for dimension 0 with size 8
```

The index is a garbage pointer-sized value thrown from inside the models' custom
architecture code (they require `trust_remote_code=True`), which targets the 4.x
internals. The failure is silent at load time, which makes it particularly
nasty: a naive setup looks healthy until the first query.

Pinned `transformers>=4.44,<5` and `sentence-transformers>=3.0,<5`. Verified:

```
embedder    dim=768  norm=1.0  max_seq_length=8192
reranker    ("hoa sinh nhật", "Giỏ Hoa Sinh Nhật Vip M155") -> 0.6059
            ("hoa sinh nhật", "Hoa Khai Trương Rực Rỡ")     -> 0.5309
```

## D2 — The Gemini model is pinned, never aliased

`gemini-flash-latest` works, but an alias is disqualifying here: the project's
whole premise is committed, reproducible evaluation numbers, and an alias
silently swaps the model underneath them. A figure in `eval/results/` must
correspond to a known model.

`gemini-2.5-flash` is GA, multilingual and cheap. Worth knowing that
`gemini-2.0-flash` is already retired and returns 404 — pinned models need an
occasional liveness check.

## D3 — Generator and judge are different models

`gemini-2.5-flash` answers; `gemini-3.6-flash` assigns gold labels and will run
RAGAS. Two independent reasons:

1. **Methodology.** A model grading its own output inflates faithfulness and
   answer relevancy. The judge should not be the system under test.
2. **Quota.** The free-tier limit is per-model (D4), so separating the roles
   doubles the daily budget instead of having dataset construction compete with
   answer generation for the same 20 calls.

The model name is part of the LLM disk-cache key, so a cached response from one
is never silently reused for the other.

## D4 — The Gemini free tier is 20 requests/day/model

Discovered by hitting it:

```
429 ResourceExhausted
quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
quota_value: 20
```

Roughly two orders of magnitude below the commonly cited figure. Probing
confirmed it is per-model — after `gemini-2.5-flash` was exhausted, other models
still answered — but **rotating judge models across an evaluation is not
acceptable**: the judge is part of the measuring instrument, and changing it
between cells makes the cells incomparable.

Consequences and mitigations:

- Golden-set construction needs ~45 calls; feasible across a few days.
- The RAGAS axis needs ~1000–1500 calls (4 metrics × 40 queries × 6 matrix
  cells). **Not feasible on the free tier at any batching**, and a disk cache
  does not help a first run.
- `src/rag/llm.py` provides a content-addressed disk cache (re-runs cost zero
  calls), a token-bucket throttle, and exponential backoff on 429.
- Query classification was moved off the LLM entirely — see D8.

## D5 — Crawl4AI was evaluated and dropped

Crawl4AI was the obvious choice for its markdown/Pruning content filter, as a
way to strip repeated nav and promo text.

Probing the site first made it unnecessary. Sampling 10 random products and
measuring the fraction of tokens shared by ≥60% of documents:

| field | shared tokens | unique values |
|---|---|---|
| `div.product-short-description` | **96.3%** | 5/10 |
| `#tab-description` | **100.0%** | 2/10 |
| `meta[name=description]` | **15.3%** | 10/10 |
| `h1.product_title` | 27.0% | 10/10 |

The site already exposes clean, per-product, human-written copy in
`meta[name=description]`. There is no boilerplate to strip — there is a
**correct field to select**. A content filter would at best approximate what a
one-line CSS selector gets exactly.

Against that, Crawl4AI costs a Playwright + Chromium install (~400MB) to render
pages that are static server-rendered HTML. So: `httpx.AsyncClient` + `lxml`.
Reproduce with `uv run python -m scripts.probe_boilerplate`.

## D6 — Sale prices must be read from `<ins>`, not the first amount

WooCommerce renders a discounted product as `<del>original</del><ins>current</ins>`.
The combined text of `p.price` reads:

```
1.400.000 ₫ Giá gốc là: 1.400.000₫. 1.200.000 ₫ Giá hiện tại là: 1.200.000₫.
```

Parsing the first number yields **1.400.000** — the struck-through original —
making every budget filter wrong on exactly the products a price-sensitive
customer is most likely to ask about.

## D7 — Cosine score spread is NOT a quality metric

Tempting, because a flat score band looks like a model that cannot decide.
Measured directly — mean spread between top-1 and 5th result across 6
representative Vietnamese queries:

| corpus | mean spread (top1 − top5) |
|---|---|
| baseline | 0.0252 |
| clean | **0.0226** |

The clean corpus has a *slightly smaller* spread. Spread does not measure what
it appears to: for `hoa chia buồn đám tang`, clean returns five "Hoa Chia Buồn"
products scoring 0.842/0.840/0.838/0.827/0.827 — a tight band, because **all
five are correct**. Flat scores across five right answers is a success. Spread
conflates "the model is undecided" with "there are many good matches".

Binding consequence: no claim about score separation appears anywhere. The only
defensible retrieval claims are Recall@k / MRR / nDCG against the labelled
golden set.

## D8 — LLM classification of query intent was dropped

Classifying all 897 harvested queries costs ~45 calls — more than two days of
quota (D4) — for something deterministic heuristics do, now covered by 24
passing tests. The whole LLM budget goes to gold-URL judging instead: deciding
which products actually answer a query is the one part heuristics cannot do.

Building those heuristics surfaced a bug class worth recording, because it would
have silently corrupted every sliced metric. Matching keywords against
**diacritic-folded Vietnamese with substring containment** produces false
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

Fixes: whole-token phrase matching; place names stripped before facet
extraction; ambiguous single-syllable keys removed; `giấy` narrowed to
`hoa giấy`. Every row is a regression test in `tests/test_prefilter.py`.

## D9 — The validated pipeline does not currently out-retrieve the baseline

Measured with the `dense` config across corpus states, n=14 answerable queries:

| state | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| baseline | **0.476** | **0.893** | **0.676** |
| clean | 0.317 | 0.524 | 0.455 |
| corrupt | 0.223 | 0.429 | 0.339 |
| repaired | 0.288 | 0.464 | 0.391 |

### How unstable this is, measured

The golden set grew from 12 to 14 answerable queries. **Two queries.** What that
did to the `dense` config on the clean corpus:

| metric | n=12 | n=14 | Δ |
|---|---|---|---|
| recall@5 | 0.504 | 0.317 | **−0.187** |
| mrr@5 | 0.683 | 0.524 | −0.159 |
| repair recovery | 80.0% | 68.9% | −11.1pp |

A 14% increase in sample size moved the headline metric by 37% of its own
value. This is not a subtle caveat — it is the single most important fact about
every retrieval number in this repository, and it is now measured rather than
asserted. Any conclusion drawn from these figures today would very likely
reverse at n=40.

The corpus-quality and repair-detection figures are unaffected: they are
computed over the full corpus (483–512 documents) and against a complete
corruption manifest, not sampled.

Two things follow, neither negotiable:

**1. No clean-vs-baseline claim at this sample size.** The gap widened from
0.010 to 0.159 when two queries were added — in the same direction, but the
magnitude is clearly not stable. The golden set has to grow before this
comparison means anything either way.

**2. Tuning stopped deliberately.** The obvious next move was hunting for a
variant that reverses the result. One was tried — rebuilding `clean` without the
price band in the embedded text:

| variant | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| with price band | 0.504 | 0.683 | 0.671 |
| without | 0.452 | **0.781** | 0.605 |

The metrics move in opposite directions, which is what noise looks like.
Continuing would be selecting a configuration by overfitting the evaluation set
— precisely the failure this project exists to guard against. Further embed-text
experiments are deferred until the golden set is adequate, and any variant
chosen this way would have to be reported as such.

**What the data does support:** corrupt 0.223 → repaired 0.288 against a clean
baseline of 0.317, i.e. **68.9% of the lost retrieval quality restored**. That
axis does not depend on the LLM quota at all — though the ratio itself is
computed from three sampled numbers and moved 11 points when the sample grew, so
the *detection* F1 scores (measured against the full manifest) are the sturdier
half of the repair story.

A hypothesis worth testing once n is adequate: the baseline embeds promo
boilerplate, which is flower-vocabulary-rich. Against *generic* queries that
shared vocabulary may act as a weak topical prior rather than pure noise. If so,
the validated pipeline should win on *specific* queries and lose on generic ones
— a slice comparison the harness already supports, and a more interesting
finding than a single averaged number.

## D10 — BM25 hybrid hurts; the budget filter is what helps

Full A/B ladder on the clean corpus. Each rung adds exactly one mechanism.

| config | recall@5 | mrr@5 | ndcg@5 | abstain F1 | p95 |
|---|---|---|---|---|---|
| baseline | 0.317 | 0.524 | 0.455 | 0.000 | 613 ms |
| dense | 0.317 | 0.524 | 0.455 | 0.000 | 553 ms |
| dense_threshold | 0.317 | 0.524 | 0.455 | 0.250 | 607 ms |
| **dense_budget** | **0.589** | **0.809** | **0.732** | 0.000 | **620 ms** |
| hybrid | 0.327 | 0.502 | 0.379 | 0.000 | 582 ms |
| hybrid_budget | 0.527 | 0.752 | 0.610 | 0.000 | 659 ms |
| dense_rerank | 0.313 | 0.482 | 0.438 | 0.250 | 6 469 ms |
| full | 0.532 | 0.667 | 0.626 | **0.250** | 8 012 ms |

**The BM25 hybrid result did not survive a larger sample.** At n=12 it lost
0.087 Recall@5 in *both* pairings — consistent enough to look like a real
effect, and it was written up as one. At n=14:

| pairing | n=12 | n=14 |
|---|---|---|
| dense → hybrid | −0.087 | **+0.010** |
| dense_budget → hybrid_budget | −0.087 | −0.062 |

The sign flipped in one pairing. The mechanism proposed for it — fastembed has
no Vietnamese analyser (`language="vietnamese"` raises), so BM25 applies English
stemming and stopwords to Vietnamese text — remains plausible, but the data no
longer supports the conclusion it was invented to explain.

Kept here as a worked example of the failure mode this project is about: a
consistent-looking result across two cells of one small sample is still one
small sample.

### The control cell that changed the conclusion

`dense_budget` was **added after the first run**, which had only `dense`,
`hybrid` and `hybrid_budget`. On that evidence `hybrid_budget` was the top
scorer and the natural write-up was "hybrid retrieval improved recall".

That would have been wrong. The control shows `dense_budget` (0.589) beats it
(0.527), and the gain belongs to the **budget filter** — parsing `price_vnd`
into an integer payload index and filtering on it.

The lesson generalises: an A/B ladder missing one rung will attribute a gain to
whichever mechanism happened to be bundled with it.

Unlike the hybrid finding above, this one **strengthened** as the sample grew:
the budget filter's advantage over plain dense went from +0.150 to +0.272, the
same sign and a larger magnitude. It is the only retrieval conclusion here that
has been stable across both sample sizes.

The best configuration is also nearly the cheapest — `dense_budget` wins on all
three quality metrics at 620 ms p95, an order of magnitude below the reranking
configs (6 469 ms). `full` retains an abstention advantage, being the only
config combining a cosine threshold with the filter.

## D11 — Hybrid retrieval was not reproducible until ties were broken explicitly

Three identical `hybrid` runs, same seed, same data:

| run | recall@5 | mrr@5 | ndcg@5 |
|---|---|---|---|
| 1 | 0.417 | 0.611 | 0.517 |
| 2 | 0.417 | 0.611 | 0.522 |
| 3 | 0.417 | **0.694** | **0.541** |

Dense retrieval was bit-identical across the same runs. RRF is the difference: a
document at rank *r* in one prefetch list and absent from the other always
scores exactly 1/(60+*r*), so large groups share a score and their relative
order falls out of Qdrant's internal ordering. Set-based recall@5 was unaffected;
every rank-sensitive metric drifted.

An evaluation number that changes between runs cannot be committed, so hybrid
results are sorted by `(-score, url)`. The tie-break is arbitrary but stable.
Verified across three consecutive runs.

This bug is invisible in a single run — it only appears when a result is
reproduced, which is why the harness is run repeatedly rather than once.

## D12 — The citation validator needed three revisions before it was quotable

**v1** flagged any sentence containing "giá", "mẫu", "sản phẩm" or a form-factor
noun. On a real answer that produced two false positives, including a question.

**v2** narrowed to "currency amount or product code" — and quietly broke the
metric in the other direction. Tested against five realistic hallucinated
sentences it caught **2/5**, blind to exactly what a RAG system most often
invents:

```
Mẫu này dùng hoa hướng dương và lan hồ điệp trắng.     -> missed
Kệ hoa này cao khoảng hai mét, giao trong ngày.        -> missed
Sản phẩm được thiết kế hai tầng với tông đỏ chủ đạo.   -> missed
```

**v3** scopes citations to the **block**, not the sentence. Models place one
marker at the end of a bullet covering the whole item:

```
- **Hoa Chia Buồn M20**: Vòng hoa ... lòng thành kính. Giá 1.100.000đ [1].
```

Splitting on the full stop leaves the first half unmarked, and v2 reported
**5 uncited claims across 5 correctly cited bullets**.

Final design: two tiers reported separately — `invalid_index_rate` (purely
structural, quotable without caveat), `uncited_price_or_code_claim_rate`
(narrow, reliable), and `uncited_descriptive_claim_rate` (vocabulary heuristic,
lower precision, deliberately excluded from `is_valid`). Re-measured: **5/5**
hallucinated sentences caught, **0/5** false positives on benign pleasantries.

**A validator is itself a measuring instrument, and needs its own
false-positive and false-negative testing before its output is quotable.**

## D13 — Three UI bugs that only surfaced by looking at the running app

Health checks and unit tests passed throughout. None of these would have been
caught without opening a browser and clicking.

**1. Abstention rendered as if it were an answer.** When the model correctly
refused ("thông tin về chính sách freeship không có trong danh sách sản phẩm"),
the page still displayed a prominent grid of five flower photos beneath it, each
labelled "not used" — visually indistinguishable from the failure mode
abstention exists to prevent. Now, when nothing is cited, sources collapse into
a closed expander: still inspectable, no longer presented as an answer.

**2. `import ui` broke the deployed entry point, not the local one.** Streamlit
puts a script's own directory on `sys.path`, so `app/chatbot.py` imported its UI
module fine when run directly. The root `chatbot.py` shim — which is what
Streamlit Cloud executes — loads it via `runpy`, where `app/` is *not* on the
path, and the app died with `ModuleNotFoundError`.
`tests/test_app_entrypoint.py` runs the shim path in a subprocess so this cannot
regress silently.

**3. Ragged card layout.** One `st.columns(3)` for a whole list stacks each
column independently, so cards land at different heights whenever product photos
differ in aspect ratio. Fixed with a fresh row of columns per three cards plus a
4:3 media frame using `object-fit: cover`.

## D14 — Corpus quality needs two boilerplate metrics, not one

The token-document-frequency ratio (share of tokens appearing in ≥60% of
documents) cannot see boilerplate injected into a *minority* of documents.
Flooding 25% of the corpus never pushes any single token past a 60% threshold —
so the corrupt state scored **17.1%**, *lower* than clean's 20.9%, despite being
deliberately contaminated.

The same blind spot broke the repair detector, which scored recall **0.017** on
`boilerplate_flood` until it was rewritten around repeated 8-gram shingles.

Both the quality report and the detector now use the shingle method, so they
cannot disagree about what boilerplate is. The doc-frequency ratio is retained
alongside it because the two answer different questions:

| metric | baseline | clean | corrupt | repaired |
|---|---|---|---|---|
| Boilerplate tokens (doc-freq ≥60%) | 49.6% | 20.9% | 17.1% | 16.2% |
| Docs built from repeated passages | **56.0%** | **0.0%** | 19.7% | 1.3% |

## D15 — The demo serves one configuration; experiment controls live behind `?lab=1`

The app briefly exposed both a retrieval-strategy selector (8 options) and a
corpus-state selector (4 options) in its main sidebar. That is an evaluation
console wearing a product's clothes, and it fails on two counts:

1. **Nothing a user can act on.** No customer shopping for flowers can choose
   between `hybrid_budget` and `dense_rerank`. Presenting the choice implies the
   answer quality is their responsibility.
2. **A corpus state that must never be served.** `corrupt` is a deliberately
   damaged index built for the repair experiment. Offering it in a dropdown next
   to `clean` is one mis-click away from a real answer grounded in known-bad
   data.

The demo now serves `dense_budget` on `clean` — the measured-best configuration
(D10) and also the cheapest of the strong ones. Everything else moved behind
`?lab=1`, which additionally reveals retrieval scores on the cards and the
per-stage latency bar. Selecting a non-production corpus in lab mode raises a
visible warning.

The same split applies inside a turn. A customer sees how their question was
interpreted, which filters ran, and whether every statement is sourced — trust
signals. Stage timings, dense/rerank scores and RRF details are diagnostics and
stay in lab mode. Citation *errors* are the exception and surface in both: a
user has more right to know an answer cited something non-existent than an
engineer does.

`tests/test_app_modes.py` parses the app's AST to assert every `selectbox` sits
inside the `if lab:` branch, and cross-checks the production default against
`eval/results/` so the demo cannot silently drift away from the configuration
the numbers describe.

## D16 — Collection naming

Collections are `flowers_{baseline,clean,corrupt,repaired}` via
`Settings.collection_for()`. Names describe the corpus state, which is what the
evaluation matrix indexes on.
