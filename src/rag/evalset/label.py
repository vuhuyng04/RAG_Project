"""Classify harvested queries and assign gold labels.

Two stages, deliberately separated:

1. **Classify** — intent + structured facets + answerability, in batches. Cheap.
2. **Label** — for answerable queries, build a candidate pool by *pooled
   retrieval* (union of top-k from every indexed state) and have the judge mark
   each candidate relevant or not.

Pooling is the standard TREC construction and matters here for a specific
reason: labelling only what the *clean* index returns would bake that system's
behaviour into the ground truth and make it unbeatable by construction. Taking
the union across states keeps the labels system-agnostic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rag.clients import embed, get_qdrant
from rag.config import State, get_settings
from rag.evalset.schema import GoldenQuery, Intent, QuerySource
from rag.llm import GeminiClient

log = logging.getLogger(__name__)

CLASSIFY_BATCH = 20
POOL_PER_STATE = 10

# The shop's actual service area, needed to judge out-of-catalogue queries.
# Taken from the site's own delivery copy ("Freeship nội thành").
SHOP_CONTEXT = """\
Cửa hàng: Hoa Tươi My My — shop hoa tươi trực tuyến tại TP. Hồ Chí Minh.
Danh mục: bó hoa, giỏ hoa, kệ hoa khai trương, hoa chia buồn, hoa cưới,
chậu lan hồ điệp. Giao hàng nội thành TP.HCM, freeship với đơn từ 400K.
Cửa hàng KHÔNG có chi nhánh ở tỉnh/thành khác."""

_CLASSIFY_PROMPT = """\
Bạn đang xây dựng bộ đánh giá cho chatbot bán hoa. Dưới đây là thông tin cửa hàng:

{shop}

Với MỖI truy vấn tìm kiếm bên dưới (là truy vấn thật của người dùng), hãy phân loại.

intent phải là một trong:
- product_search: tìm sản phẩm hoa cụ thể mà cửa hàng có thể có
- budget_search: tìm sản phẩm kèm ràng buộc giá (vd "hoa khai trương 500k")
- attribute: hỏi về đặc điểm sản phẩm (màu, loại hoa, kích thước)
- policy: hỏi chính sách/dịch vụ (giao hàng, freeship, giờ mở cửa, đặt theo yêu cầu)
- out_of_catalogue: hỏi hoa ở tỉnh/thành KHÁC TP.HCM, hoặc shop khác, hoặc sản phẩm cửa hàng không bán
- off_topic: không liên quan việc mua hoa (bài văn, ý nghĩa hoa học đường, tên gọi...)

answerable = true CHỈ KHI có thể trả lời bằng danh mục sản phẩm hoa của cửa hàng này.
policy, out_of_catalogue và off_topic đều phải có answerable = false.

Trả về JSON: {{"results": [{{"i": <chỉ số>, "intent": "...", "answerable": true/false,
"occasion": <dịp hoặc null>, "recipient": <người nhận hoặc null>,
"flower_type": <loại hoa hoặc null>, "form_factor": <"bó"/"giỏ"/"kệ"/"chậu" hoặc null>,
"budget_max_vnd": <số nguyên VND hoặc null>}}]}}

Truy vấn:
{queries}"""


def classify_queries(queries: list[str], llm: GeminiClient) -> list[dict[str, Any]]:
    """Batch-classify. ~20 queries/call keeps free-tier usage in the tens."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(queries), CLASSIFY_BATCH):
        batch = queries[start : start + CLASSIFY_BATCH]
        listing = "\n".join(f"{i}. {q}" for i, q in enumerate(batch))
        prompt = _CLASSIFY_PROMPT.format(shop=SHOP_CONTEXT, queries=listing)
        try:
            data = llm.generate_json(prompt)
            results = data.get("results", []) if isinstance(data, dict) else data
        except Exception as exc:
            log.warning("classification batch %d failed: %s", start, exc)
            continue

        for item in results:
            idx = item.get("i")
            if not isinstance(idx, int) or not 0 <= idx < len(batch):
                continue
            out.append({**item, "question": batch[idx]})
        log.info("classified %d/%d", min(start + CLASSIFY_BATCH, len(queries)), len(queries))
    return out


# ---------------------------------------------------------------------------
# Gold-URL assignment via pooled retrieval + judging
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
Khách hàng của shop hoa tìm kiếm: "{question}"

Dưới đây là các sản phẩm ứng viên. Với MỖI sản phẩm, quyết định nó có thực sự
phù hợp để trả lời truy vấn này không.

Tiêu chí "phù hợp" (relevant = true):
- Đúng loại/dịp/người nhận mà khách hỏi
- Nếu khách nêu ngân sách, giá sản phẩm phải nằm trong ngân sách đó
- Nếu khách nêu loại hoa hoặc dáng (bó/giỏ/kệ/chậu), sản phẩm phải khớp

Nghiêm khắc. Sản phẩm chỉ "cùng là hoa" thì KHÔNG phải là phù hợp.

Ứng viên:
{candidates}

Trả JSON: {{"results": [{{"i": <chỉ số>, "relevant": true/false}}]}}"""


def pool_candidates(question: str, states: list[State], per_state: int = POOL_PER_STATE):
    """Union of top-k across states, keyed by URL.

    Deduplicated by URL because the same product exists in every collection
    under a different embedding.
    """
    s = get_settings()
    client = get_qdrant()
    vector = embed(question)
    pooled: dict[str, dict[str, Any]] = {}

    for state in states:
        collection = s.collection_for(state)
        if not client.collection_exists(collection):
            continue
        hits = client.query_points(
            collection, query=vector, limit=per_state, with_payload=True
        ).points
        for h in hits:
            payload = h.payload or {}
            url = payload.get("url")
            if not url or url in pooled:
                continue
            pooled[url] = {
                "url": url,
                "title": payload.get("title") or "",
                "price_vnd": payload.get("price_vnd"),
                "price_raw": payload.get("price_raw") or payload.get("price") or "",
                "description": (payload.get("description") or "")[:220],
            }
    return list(pooled.values())


def judge_candidates(
    question: str, candidates: list[dict[str, Any]], llm: GeminiClient
) -> list[str]:
    """Return the URLs the judge marks relevant."""
    if not candidates:
        return []

    listing = "\n".join(
        f"{i}. {c['title']} | giá: {c['price_raw'] or c['price_vnd']} | {c['description']}"
        for i, c in enumerate(candidates)
    )
    prompt = _JUDGE_PROMPT.format(question=question, candidates=listing)
    try:
        data = llm.generate_json(prompt)
        results = data.get("results", []) if isinstance(data, dict) else data
    except Exception as exc:
        log.warning("judging failed for %r: %s", question, exc)
        return []

    gold: list[str] = []
    for item in results:
        idx = item.get("i")
        if isinstance(idx, int) and 0 <= idx < len(candidates) and item.get("relevant"):
            gold.append(candidates[idx]["url"])
    return gold


def enforce_budget(gold_urls: list[str], budget: int | None, candidates) -> list[str]:
    """Drop golds above a stated budget.

    The judge is asked to respect budget but LLMs are unreliable at numeric
    comparison, and a wrong gold here silently corrupts every Recall number
    downstream. Cheap deterministic backstop.
    """
    if budget is None:
        return gold_urls
    by_url = {c["url"]: c for c in candidates}
    kept = []
    for url in gold_urls:
        price = (by_url.get(url) or {}).get("price_vnd")
        if price is None or price <= budget:
            kept.append(url)
    return kept


def leaks_title(question: str, candidates, gold_urls: list[str]) -> bool:
    """True if the query copies a gold product's title nearly verbatim.

    LLM-generated queries tend to echo document wording, which makes retrieval
    look far better than it is. Such queries are dropped from the synthetic
    slice rather than silently inflating the headline number.
    """
    q = question.lower()
    by_url = {c["url"]: c for c in candidates}
    for url in gold_urls:
        title = (by_url.get(url) or {}).get("title", "").lower()
        if not title:
            continue
        tokens = [t for t in title.split() if len(t) > 2]
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in q) / len(tokens)
        if overlap >= 0.8:
            return True
    return False


def to_golden(record: dict[str, Any], idx: int, source: QuerySource) -> GoldenQuery:
    try:
        intent = Intent(record.get("intent", "product_search"))
    except ValueError:
        intent = Intent.PRODUCT_SEARCH
    return GoldenQuery(
        id=f"q{idx:04d}",
        question=record["question"],
        answerable=bool(record.get("answerable", True)),
        source=source,
        intent=intent,
        occasion=record.get("occasion"),
        recipient=record.get("recipient"),
        flower_type=record.get("flower_type"),
        form_factor=record.get("form_factor"),
        budget_max_vnd=record.get("budget_max_vnd"),
    )


def save_golden(queries: list[GoldenQuery], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(q.model_dump_json() + "\n" for q in queries)
    log.info("Wrote %d golden queries -> %s", len(queries), path)


def load_golden(path) -> list[GoldenQuery]:
    with open(path, encoding="utf-8") as fh:
        return [GoldenQuery(**json.loads(line)) for line in fh if line.strip()]
