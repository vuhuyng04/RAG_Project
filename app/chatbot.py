"""Streamlit UI.

Thin by design: every retrieval and generation decision lives in `src/rag/` so
the app and the evaluation harness exercise identical code. The original
version had the pipeline inline, which is why the notebook and the app drifted
apart.
"""

from __future__ import annotations

import logging
import sys
from html import escape as html_escape
from pathlib import Path

import requests
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
# `app/` itself must be on the path too. Streamlit adds the script's directory
# automatically when it runs this file directly, but the root `chatbot.py` shim
# — which is what Streamlit Cloud executes — loads it via runpy, and then
# `import ui` fails with ModuleNotFoundError. Doing it here keeps both entry
# points working.
sys.path.insert(0, str(_APP_DIR))

from ui import (
    CSS,
    card_html,
    format_vnd,
    render_citations,
    timing_bar_html,
    trace_row,
)

from rag.config import State, get_settings
from rag.generation.answer import answer_question
from rag.retrieval.search import CONFIGS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

st.set_page_config(page_title="Hoa Tươi My My — Trợ lý tư vấn", page_icon="🌸", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]

# User-facing config descriptions. `RetrievalConfig.description` is the
# developer-facing English rationale that ends up in eval/results/ — keeping the
# two separate stops experiment notes leaking into the UI.
CONFIG_LABELS_VI = {
    "legacy": "Tái hiện hệ thống gốc: dense top-5, không ngưỡng, không lọc.",
    "dense": "Truy hồi dense trên dữ liệu đã làm sạch.",
    "dense_threshold": "Dense + ngưỡng từ chối khi không đủ phù hợp.",
    "dense_budget": "Dense + lọc theo ngân sách. Cấu hình đo tốt nhất và nhẹ nhất.",
    "hybrid": "BM25 + dense, trộn bằng RRF. Đo được là kém hơn dense thuần.",
    "hybrid_budget": "Hybrid + lọc ngân sách.",
    "dense_rerank": "Lấy 20 rồi rerank còn 5. Chậm hơn ~10 lần, chất lượng không tăng.",
    "full": "Rerank + lọc ngân sách + ngưỡng. Từ chối tốt nhất nhưng nặng.",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_image(url: str) -> bytes | None:
    """Fetch a product image once.

    The original downloaded every image on every rerun with no timeout, so each
    Streamlit interaction re-fetched the whole grid.
    """
    if not url or not url.startswith("http"):
        return None
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        log.warning("Image fetch failed for %s: %s", url, exc)
        return None


def _render_grid(hits, cited: set[int]) -> None:
    """Lay cards out row by row.

    One `st.columns(3)` for the whole list stacks each column independently, so
    cards land at ragged heights whenever the product photos differ in aspect
    ratio — which they always do. A fresh row of columns per three cards keeps
    them aligned; the fixed 4:3 media frame in the card CSS does the rest.
    """
    for start in range(0, len(hits), 3):
        row = hits[start : start + 3]
        cols = st.columns(3, gap="small")
        for offset_in_row, hit in enumerate(row):
            index = start + offset_in_row + 1
            # Images are inlined as data URIs so the whole card can be one HTML
            # block; st.image cannot participate in a CSS-controlled layout.
            hit.payload["_image_bytes"] = load_image(hit.payload.get("image", ""))
            with cols[offset_in_row]:
                st.markdown(card_html(hit, index, index in cited), unsafe_allow_html=True)


def render_sources(answer) -> None:
    """Product cards numbered to match the [n] chips in the answer."""
    if not answer.hits:
        return

    cited = set(answer.citations.cited)

    # When the answer cites nothing — because it refused, or grounded itself out
    # of the retrieved set — a prominent grid of five flower photos under the
    # words "we don't have that information" reads exactly like the old broken
    # behaviour it replaced. Demote it to a collapsed expander: still fully
    # inspectable, no longer presented as an answer.
    if not cited:
        with st.expander(f"Đã truy hồi {len(answer.hits)} sản phẩm — không dùng cái nào"):
            st.caption(
                "Hệ thống có tìm thấy các sản phẩm này nhưng câu trả lời không "
                "dựa trên chúng. Hiển thị để bạn kiểm chứng."
            )
            _render_grid(answer.hits, cited)
        return

    st.markdown(
        f'<div class="src-head"><h4>Nguồn tham chiếu</h4>'
        f"<span>{len(cited)}/{len(answer.hits)} sản phẩm được dùng trong câu trả lời</span></div>",
        unsafe_allow_html=True,
    )
    _render_grid(answer.hits, cited)


def render_answer(answer) -> None:
    """Answer body with citation markers rendered as chips."""
    st.markdown(render_citations(answer.text, len(answer.hits)), unsafe_allow_html=True)


def render_trace(answer) -> None:
    """Why these results — the panel that makes the system inspectable."""
    with st.expander("Chi tiết truy hồi"):
        rows = []
        if answer.was_condensed:
            rows.append(
                trace_row(
                    "Câu hỏi viết lại",
                    f"<code>{html_escape(answer.query_used)}</code>"
                    "<br><span class='tag'>bổ sung ngữ cảnh từ lịch sử hội thoại</span>",
                )
            )
        else:
            rows.append(
                trace_row("Truy vấn tìm kiếm", f"<code>{html_escape(answer.query_used)}</code>")
            )

        if answer.budget_applied:
            rows.append(trace_row("Lọc ngân sách", f"≤ {format_vnd(answer.budget_applied)}"))

        rows.append(
            trace_row(
                "Sản phẩm truy hồi", f"{len(answer.hits)} · dùng {len(answer.citations.cited)}"
            )
        )
        rows.append(trace_row("Tổng thời gian", f"{answer.latency_ms:.0f} ms"))
        st.markdown("".join(rows), unsafe_allow_html=True)

        if answer.timings:
            st.markdown(timing_bar_html(answer.timings), unsafe_allow_html=True)

        c = answer.citations
        if answer.abstained:
            st.warning("Không sản phẩm nào vượt ngưỡng phù hợp — hệ thống từ chối trả lời.")
        if c.invalid:
            st.error(
                f"Mô hình trích dẫn nguồn không tồn tại: {c.invalid}. "
                "Đã đánh dấu gạch ngang trong câu trả lời."
            )
        if c.uncited_claims:
            st.warning(
                f"{len(c.uncited_claims)} khẳng định về giá hoặc mã sản phẩm không kèm trích dẫn."
            )
        if c.uncited_descriptive:
            st.info(
                f"{len(c.uncited_descriptive)} mô tả đặc điểm không kèm trích dẫn "
                "(phát hiện bằng heuristic, độ chính xác thấp hơn)."
            )
        if c.is_valid and c.has_citations and not c.uncited_descriptive:
            st.success("Mọi trích dẫn hợp lệ và mọi khẳng định đều có nguồn.")


def main() -> None:
    settings = get_settings()
    st.markdown(CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        # (question, answer) pairs actually fed to condensation — unlike the
        # original, where session state was display-only and never reached the
        # model.
        st.session_state.history = []

    with st.sidebar:
        logo = REPO_ROOT / "logo.png"
        if logo.exists():
            st.image(str(logo), width="stretch")
        st.markdown("### 🌸 Hoa Tươi My My")
        st.caption("Trợ lý tư vấn sản phẩm — RAG có kiểm chứng")

        st.divider()
        st.markdown("**Cấu hình**")
        # Default is dense_budget, not `full`, for two independent reasons:
        #   1. It measured best on recall@5 / MRR@5 / nDCG@5 (docs/decisions.md D13).
        #   2. `full` and `dense_rerank` load a ~1.2GB cross-encoder on top of the
        #      embedding model, which will OOM on Streamlit Community Cloud's
        #      ~1GB limit. They stay selectable for local A/B work.
        config_name = st.selectbox(
            "Chiến lược truy hồi",
            list(CONFIGS),
            index=list(CONFIGS).index("dense_budget"),
            help=(
                "Mỗi cấu hình chỉ khác nhau một cơ chế, để so sánh A/B được. "
                "Các cấu hình có rerank cần nhiều RAM và có thể không chạy được trên bản demo."
            ),
        )
        # RetrievalConfig.description is developer-facing English that explains
        # the experiment design; it belongs in eval/results/, not in a Vietnamese
        # user interface.
        st.caption(CONFIG_LABELS_VI.get(config_name, ""))

        state_name = st.selectbox(
            "Trạng thái dữ liệu",
            [s.value for s in State],
            index=[s.value for s in State].index("clean"),
            help="legacy = pipeline gốc; corrupt/repaired dùng cho thí nghiệm sửa lỗi dữ liệu.",
        )

        st.divider()
        st.markdown("**Thử nhanh**")
        samples = [
            "hoa tặng mẹ ngày 20/10",
            "kệ hoa khai trương dưới 1 triệu",
            "bó hoa sinh nhật cho nữ",
            "hoa chia buồn trang trọng",
            "có freeship không",
            "hoa cưới cần thơ",
        ]
        clicked = None
        for s in samples:
            if st.button(s, width="stretch", key=f"s_{s}"):
                clicked = s

        st.divider()
        if st.button("🗑️ Xoá hội thoại", width="stretch"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        st.caption(f"Model: `{settings.gemini_model}`")
        st.caption(f"Embedding: `{settings.embedding_model.split('/')[-1]}`")

    st.title("🌸 Trợ lý tư vấn Hoa Tươi My My")
    st.caption(
        "Trả lời có trích dẫn nguồn, nhớ ngữ cảnh hội thoại, và **từ chối trả lời** "
        "khi không tìm được sản phẩm phù hợp."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("answer") is not None:
                render_answer(msg["answer"])
                render_sources(msg["answer"])
                render_trace(msg["answer"])
            else:
                st.markdown(msg["content"])

    question = clicked or st.chat_input("Bạn cần tìm hoa cho dịp nào?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm sản phẩm phù hợp..."):
            try:
                answer = answer_question(
                    question,
                    history=st.session_state.history,
                    state=State(state_name),
                    config=config_name,
                )
            except Exception as exc:
                log.exception("Turn failed")
                st.error(f"Lỗi: {exc}")
                return

        render_answer(answer)
        render_sources(answer)
        render_trace(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )
    st.session_state.history.append((question, answer.text))


if __name__ == "__main__":
    main()
