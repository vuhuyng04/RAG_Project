"""Streamlit UI.

Thin by design: every retrieval and generation decision lives in `src/rag/` so
the app and the evaluation harness exercise identical code. Inlining pipeline
logic here would let the measured system and the served system diverge.
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

# What the demo serves. `dense_budget` measured best on Recall@5 / MRR@5 /
# nDCG@5 (docs/decisions.md D10) and is also the cheapest of the strong configs:
# `full` and `dense_rerank` load a ~1.2GB cross-encoder on top of the embedding
# model for a *worse* Recall@5 and ~10x the p95 latency.
PRODUCTION_CONFIG = "dense_budget"
PRODUCTION_STATE = "clean"

SAMPLE_QUERIES = (
    "hoa tặng mẹ ngày 20/10",
    "kệ hoa khai trương dưới 1 triệu",
    "bó hoa sinh nhật cho nữ",
    "hoa chia buồn trang trọng",
    "có freeship không",
    "hoa cưới cần thơ",
)


def lab_mode_enabled() -> bool:
    """Experiment controls are opt-in via ?lab=1."""
    try:
        return st.query_params.get("lab") in {"1", "true", "yes"}
    except Exception:
        return False


# User-facing config descriptions. `RetrievalConfig.description` is the
# developer-facing English rationale that ends up in eval/results/ — keeping the
# two separate stops experiment notes leaking into the UI.
CONFIG_LABELS_VI = {
    "baseline": "Tái hiện hệ thống gốc: dense top-5, không ngưỡng, không lọc.",
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

    Without caching, Streamlit re-fetches the whole grid on every interaction;
    without a timeout, one slow host blocks the render.
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


def _render_grid(hits, cited: set[int], lab: bool = False) -> None:
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
                st.markdown(card_html(hit, index, index in cited, lab=lab), unsafe_allow_html=True)


def render_sources(answer, lab: bool = False) -> None:
    """Product cards numbered to match the [n] chips in the answer."""
    if not answer.hits:
        return

    cited = set(answer.citations.cited)

    # When the answer cites nothing — because it refused, or grounded itself out
    # of the retrieved set — a prominent grid of five flower photos under the
    # words "we don't have that information" reads as a recommendation the system
    # explicitly declined to make. Demote it to a collapsed expander: still fully
    # inspectable, no longer presented as an answer.
    if not cited:
        with st.expander(f"Đã truy hồi {len(answer.hits)} sản phẩm — không dùng cái nào"):
            st.caption(
                "Hệ thống có tìm thấy các sản phẩm này nhưng câu trả lời không "
                "dựa trên chúng. Hiển thị để bạn kiểm chứng."
            )
            _render_grid(answer.hits, cited, lab)
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


def render_trace(answer, lab: bool = False) -> None:
    """Why these results.

    Two audiences, so two levels. A customer needs to know the answer is
    grounded and how their question was interpreted — that is a trust signal.
    Stage timings, RRF details and score distributions are diagnostics, shown
    only in lab mode.
    """
    c = answer.citations

    # Correctness problems surface at both levels: a customer has more right to
    # know the answer cited something that does not exist than an engineer does.
    if c.invalid:
        st.error(
            f"Câu trả lời trích dẫn nguồn không tồn tại ({c.invalid}) — đã đánh dấu gạch ngang."
        )

    with st.expander("Vì sao có kết quả này"):
        rows = []
        if answer.was_condensed:
            rows.append(
                trace_row(
                    "Hiểu câu hỏi thành",
                    f"<code>{html_escape(answer.query_used)}</code>"
                    "<br><span class='tag'>bổ sung ngữ cảnh từ hội thoại trước</span>",
                )
            )
        if answer.budget_applied:
            rows.append(trace_row("Lọc theo ngân sách", f"≤ {format_vnd(answer.budget_applied)}"))
        rows.append(
            trace_row(
                "Sản phẩm tham chiếu",
                f"{len(c.cited)} trong {len(answer.hits)} sản phẩm tìm được",
            )
        )
        rows.append(trace_row("Thời gian phản hồi", f"{answer.latency_ms / 1000:.1f} giây"))
        st.markdown("".join(rows), unsafe_allow_html=True)

        if answer.abstained:
            st.warning("Không sản phẩm nào đủ phù hợp — hệ thống không đưa ra gợi ý.")
        elif c.is_valid and c.has_citations and not c.uncited_descriptive:
            st.success("Mọi thông tin trong câu trả lời đều dẫn nguồn từ sản phẩm bên dưới.")

        if not lab:
            return

        st.divider()
        st.caption("Chẩn đoán (chế độ thí nghiệm)")
        if answer.timings:
            st.markdown(timing_bar_html(answer.timings), unsafe_allow_html=True)
        if c.uncited_claims:
            st.warning(
                f"{len(c.uncited_claims)} khẳng định về giá hoặc mã sản phẩm không kèm trích dẫn."
            )
        if c.uncited_descriptive:
            st.info(
                f"{len(c.uncited_descriptive)} mô tả đặc điểm không kèm trích dẫn "
                "(heuristic từ vựng, độ chính xác thấp hơn)."
            )


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

    lab = lab_mode_enabled()
    config_name, state_name = PRODUCTION_CONFIG, PRODUCTION_STATE

    with st.sidebar:
        logo = REPO_ROOT / "logo.png"
        if logo.exists():
            st.image(str(logo), width="stretch")
        st.markdown("### 🌸 Hoa Tươi My My")
        st.caption("Trợ lý tư vấn sản phẩm")

        st.divider()
        st.markdown("**Thử nhanh**")
        clicked = None
        for s in SAMPLE_QUERIES:
            if st.button(s, width="stretch", key=f"s_{s}"):
                clicked = s

        st.divider()
        if st.button("Xoá hội thoại", width="stretch"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        # Experiment controls are hidden by default. Retrieval strategies and
        # corpus states are evaluation apparatus, not product features: an end
        # user has no way to choose between `hybrid_budget` and `dense_rerank`,
        # and `corrupt` is a deliberately damaged index that must never serve a
        # real answer. They stay reachable at ?lab=1 so the A/B work can be
        # demonstrated without cluttering the product.
        if lab:
            st.divider()
            st.markdown("**🔬 Chế độ thí nghiệm**")
            config_name = st.selectbox(
                "Chiến lược truy hồi",
                list(CONFIGS),
                index=list(CONFIGS).index(PRODUCTION_CONFIG),
                help="Mỗi cấu hình chỉ khác nhau một cơ chế, để so sánh A/B được.",
            )
            st.caption(CONFIG_LABELS_VI.get(config_name, ""))

            state_name = st.selectbox(
                "Trạng thái dữ liệu",
                [s.value for s in State],
                index=[s.value for s in State].index(PRODUCTION_STATE),
                help="baseline = nối thẳng mọi field; corrupt/repaired là dữ liệu hỏng có chủ đích.",
            )
            if state_name != PRODUCTION_STATE:
                st.warning(f"Đang dùng dữ liệu `{state_name}`, không phải dữ liệu sản xuất.")

        st.divider()
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
                render_sources(msg["answer"], lab)
                render_trace(msg["answer"], lab)
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
        render_sources(answer, lab)
        render_trace(answer, lab)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "answer": answer}
    )
    st.session_state.history.append((question, answer.text))


if __name__ == "__main__":
    main()
