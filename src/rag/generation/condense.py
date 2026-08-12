"""Turn a follow-up question into a standalone query.

The original app kept `st.session_state.messages` purely for display and never
sent it to the model — single-turn RAG behind a chat UI. "Cái nào rẻ hơn?" had
no antecedent and retrieved on the word "rẻ" alone.

Condensation rewrites the follow-up using the history *before* retrieval.
The alternative — embedding the whole transcript — dilutes the query vector with
earlier turns and costs precision, which matters more here than usual because
the corpus is 480 near-neighbours in one narrow domain.

A heuristic gate decides whether the rewrite is needed at all, so
self-contained questions cost zero LLM calls. That is not only a saving under a
20-request/day quota (docs/decisions.md D7) — it also avoids the rewrite
mangling a question that was already fine.
"""

from __future__ import annotations

import logging
import re

from rag.llm import GeminiClient

log = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 4

# Markers that a question leans on something said earlier: pronouns, bare
# comparatives, ellipsis.
_DEPENDENT_MARKERS = re.compile(
    r"\b(cái|cái đó|nó|này|đó|kia|vậy|thế|còn|thêm|khác|"
    r"rẻ hơn|đắt hơn|to hơn|nhỏ hơn|đẹp hơn|loại nào|mẫu nào|"
    r"còn gì|sao|thì sao|được không|ok không)\b",
    re.IGNORECASE,
)
_FLOWER_NOUN = re.compile(r"\b(hoa|bó|giỏ|kệ|chậu|lẵng|lan|hồng|tulip)\b", re.IGNORECASE)

_PROMPT = """\
Viết lại câu hỏi cuối của khách thành MỘT câu hỏi độc lập, đầy đủ ngữ cảnh,
để dùng cho tìm kiếm sản phẩm hoa.

QUY TẮC:
- Chỉ dùng thông tin có trong lịch sử. KHÔNG thêm chi tiết mới.
- Giữ nguyên mọi ràng buộc giá, loại hoa, dịp, người nhận đã nêu trước đó.
- Trả về DUY NHẤT câu hỏi đã viết lại, không giải thích, không dấu ngoặc kép.

LỊCH SỬ:
{history}

CÂU HỎI CUỐI: {question}

CÂU HỎI ĐỘC LẬP:"""


def needs_condensation(question: str, history: list[tuple[str, str]]) -> bool:
    """Cheap gate: does this question actually depend on earlier turns?"""
    if not history:
        return False
    q = question.strip()
    # Very short questions are almost always follow-ups ("còn màu trắng?").
    if len(q.split()) <= 3:
        return True
    # Contains a back-reference and lacks its own subject noun.
    return bool(_DEPENDENT_MARKERS.search(q)) and not _FLOWER_NOUN.search(q)


def format_history(history: list[tuple[str, str]], max_turns: int = MAX_HISTORY_TURNS) -> str:
    recent = history[-max_turns:]
    return "\n".join(f"Khách: {u}\nTrợ lý: {a[:200]}" for u, a in recent)


def condense(
    question: str,
    history: list[tuple[str, str]],
    llm: GeminiClient | None = None,
) -> tuple[str, bool]:
    """Return (query_to_retrieve_with, was_condensed)."""
    if not needs_condensation(question, history):
        return question, False

    llm = llm or GeminiClient()
    prompt = _PROMPT.format(history=format_history(history), question=question)
    try:
        rewritten = llm.generate(prompt, temperature=0.0).strip().strip('"').split("\n")[0]
    except Exception as exc:
        # A failed rewrite must not break the turn — retrieving on the raw
        # question is degraded, not broken.
        log.warning("Condensation failed, using raw question: %s", exc)
        return question, False

    if not rewritten or len(rewritten) > 300:
        return question, False

    log.info("Condensed %r -> %r", question, rewritten)
    return rewritten, True
