"""Presentation layer for the Streamlit app.

Kept separate from `chatbot.py` so the page file stays about flow and this file
stays about appearance.

Design notes, since they are choices rather than defaults:

* **Cards are single HTML blocks**, not stacks of `st.image` + `st.caption`.
  Streamlit's primitives cannot constrain image aspect ratio, so product photos
  ranging from 3:4 to 16:9 produced ragged rows. A fixed 4:3 frame with
  `object-fit: cover` makes every card the same shape.
* **Citations are chips, not bracketed digits.** `[1]` inline in body text is
  indistinguishable from ordinary content; a superscript pill reads as a
  reference and ties visually to the numbered card.
* **Cited and unused sources differ structurally**, via an accent border and
  reduced emphasis, rather than by an emoji the eye has to parse.
* **Colours are semi-transparent overlays on `currentColor`**, so the whole
  thing works in Streamlit's light and dark themes without a second palette.
"""

from __future__ import annotations

import base64
import html

ACCENT = "#d6336c"  # single accent, matched to the shop's floral branding

CSS = f"""
<style>
  :root {{
    --card-radius: 14px;
    --accent: {ACCENT};
    --hairline: color-mix(in srgb, currentColor 14%, transparent);
    --surface: color-mix(in srgb, currentColor 4%, transparent);
    --muted: color-mix(in srgb, currentColor 62%, transparent);
  }}

  /* ---------- citation chips ---------- */
  .cite {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.15rem;
    height: 1.15rem;
    padding: 0 0.32rem;
    /* Left margin only. A symmetric margin pushed the chip away from the
       punctuation that follows it, rendering "950.000đ ¹ ." with a visible gap
       before the full stop. */
    margin: 0 0 0 0.1rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--accent) 14%, transparent);
    color: var(--accent);
    font-size: 0.68rem;
    font-weight: 650;
    line-height: 1;
    vertical-align: super;
    letter-spacing: 0.01em;
  }}
  .cite-bad {{
    background: color-mix(in srgb, #e03131 16%, transparent);
    color: #e03131;
    text-decoration: line-through;
  }}

  /* ---------- source cards ---------- */
  .src-head {{
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin: 1.35rem 0 0.65rem;
  }}
  .src-head h4 {{
    margin: 0;
    font-size: 0.95rem;
    font-weight: 650;
    letter-spacing: 0.01em;
  }}
  .src-head span {{ font-size: 0.78rem; color: var(--muted); }}

  .card {{
    border: 1px solid var(--hairline);
    border-radius: var(--card-radius);
    overflow: hidden;
    background: var(--surface);
    height: 100%;
    display: flex;
    flex-direction: column;
    transition: border-color .15s ease, transform .15s ease;
  }}
  .card:hover {{ transform: translateY(-2px); }}
  .card.is-cited {{ border-color: color-mix(in srgb, var(--accent) 45%, transparent); }}
  .card.is-unused {{ opacity: 0.62; }}

  .card-media {{
    position: relative;
    width: 100%;
    padding-top: 75%;              /* fixed 4:3 frame */
    background: var(--hairline);
  }}
  .card-media img {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;             /* crop, never letterbox */
    display: block;
  }}
  .card-media .rank {{
    position: absolute;
    top: 0.5rem;
    left: 0.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.16rem 0.5rem;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.62);
    backdrop-filter: blur(6px);
    color: #fff;
    font-size: 0.7rem;
    font-weight: 620;
  }}
  .card-media .rank.unused {{ background: rgba(0, 0, 0, 0.42); font-weight: 500; }}

  .card-body {{
    padding: 0.7rem 0.8rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.42rem;
    flex: 1;
  }}
  .card-title {{
    font-size: 0.87rem;
    font-weight: 620;
    line-height: 1.32;
    display: -webkit-box;
    -webkit-line-clamp: 2;         /* keep every card the same height */
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.3em;
  }}
  .card-price {{
    font-size: 1.02rem;
    font-weight: 680;
    color: var(--accent);
    letter-spacing: -0.01em;
  }}
  .card-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: auto;
  }}
  .tag {{
    font-size: 0.68rem;
    padding: 0.14rem 0.42rem;
    border-radius: 6px;
    background: var(--hairline);
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .card-link {{
    font-size: 0.75rem;
    color: var(--accent);
    text-decoration: none;
    font-weight: 560;
  }}
  .card-link:hover {{ text-decoration: underline; }}

  /* ---------- trace panel ---------- */
  .trace-row {{
    display: flex;
    gap: 0.6rem;
    padding: 0.32rem 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--hairline);
  }}
  .trace-row:last-child {{ border-bottom: none; }}
  .trace-key {{ color: var(--muted); min-width: 9.5rem; }}
  .trace-val {{ font-weight: 560; }}
  .trace-val code {{
    background: var(--hairline);
    padding: 0.08rem 0.35rem;
    border-radius: 5px;
    font-size: 0.8rem;
  }}
  .timing-bar {{
    display: flex;
    height: 6px;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.45rem;
    background: var(--hairline);
  }}
  .timing-bar span {{ display: block; height: 100%; }}
</style>
"""

# Stage colours for the latency bar, in pipeline order.
_STAGE_COLORS = {
    "embed_ms": "#4c6ef5",
    "search_ms": "#12b886",
    "rerank_ms": "#fab005",
    "llm_ms": ACCENT,
}
_STAGE_LABELS = {
    "embed_ms": "nhúng",
    "search_ms": "tìm kiếm",
    "rerank_ms": "xếp lại",
    "llm_ms": "sinh câu trả lời",
}


def format_vnd(value: object) -> str:
    """1950000 -> '1.950.000₫'. Falls back to the raw string."""
    if isinstance(value, (int, float)) and value:
        return f"{int(value):,}".replace(",", ".") + "₫"
    text = str(value or "").strip()
    return text or "Liên hệ"


def render_citations(text: str, n_context: int) -> str:
    """Turn `[n]` markers into superscript chips.

    Indices outside the supplied context are marked as broken rather than
    dropped silently — the reader should see that the model referenced
    something that does not exist.
    """
    import re

    def _sub(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_context:
            return f'<span class="cite">{n}</span>'
        return f'<span class="cite cite-bad">{n}</span>'

    return re.sub(r"\[(\d{1,2})\]", _sub, text)


def card_html(hit, index: int, cited: bool, *, lab: bool = False) -> str:
    """One product card as a self-contained HTML block.

    `lab` adds retrieval scores. They are diagnostic values with no meaning to a
    customer — "dense 0.873" answers a question nobody shopping for flowers is
    asking — so the product view shows only the freshness date.
    """
    payload = hit.payload or {}
    title = html.escape(hit.title or "Không có tên")
    price = html.escape(format_vnd(payload.get("price_vnd") or payload.get("price_raw")))
    url = html.escape(payload.get("url") or "")

    img = payload.get("_image_bytes")
    if img:
        src = "data:image/jpeg;base64," + base64.b64encode(img).decode()
        media = f'<img src="{src}" alt="{title}" loading="lazy">'
    else:
        media = ""

    tags = []
    if lab:
        tags.append(f'<span class="tag">dense {hit.score:.3f}</span>')
        if hit.rerank_score is not None:
            tags.append(f'<span class="tag">rerank {hit.rerank_score:.3f}</span>')
    if payload.get("crawled_at"):
        tags.append(f'<span class="tag">cập nhật {str(payload["crawled_at"])[:10]}</span>')

    rank_cls = "rank" if cited else "rank unused"
    rank_label = f"[{index}] đã dùng" if cited else f"[{index}]"
    link = f'<a class="card-link" href="{url}" target="_blank">Xem sản phẩm →</a>' if url else ""

    return f"""
    <div class="card {"is-cited" if cited else "is-unused"}">
      <div class="card-media">
        {media}
        <span class="{rank_cls}">{html.escape(rank_label)}</span>
      </div>
      <div class="card-body">
        <div class="card-title">{title}</div>
        <div class="card-price">{price}</div>
        <div class="card-meta">{"".join(tags)}</div>
        {link}
      </div>
    </div>
    """


def timing_bar_html(timings: dict[str, float]) -> str:
    """Proportional latency breakdown.

    A bar makes it obvious at a glance that generation dominates — which is the
    thing worth knowing, and which a list of four numbers hides.
    """
    total = sum(timings.values()) or 1.0
    segments = "".join(
        f'<span style="width:{v / total * 100:.2f}%;background:{_STAGE_COLORS.get(k, "#868e96")}"></span>'
        for k, v in timings.items()
        if v > 0
    )
    legend = " · ".join(
        f"{_STAGE_LABELS.get(k, k.replace('_ms', ''))} {v:.0f}ms"
        for k, v in timings.items()
        if v > 0
    )
    return f'<div class="timing-bar">{segments}</div><div class="tag" style="margin-top:.4rem">{html.escape(legend)}</div>'


def trace_row(key: str, value_html: str) -> str:
    return f'<div class="trace-row"><span class="trace-key">{html.escape(key)}</span><span class="trace-val">{value_html}</span></div>'
