"""End-to-end turn: condense -> retrieve -> generate -> validate citations.

Shared by the Streamlit app and the evaluation harness so both exercise exactly
the same path. If they diverged, the measured numbers would not describe the
deployed system.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from rag.config import State
from rag.generation.citation import CitationReport, strip_invalid_citations, validate_citations
from rag.generation.condense import condense
from rag.generation.prompt import build_prompt
from rag.llm import GeminiClient
from rag.retrieval.search import CONFIGS, Hit, RetrievalConfig, search

log = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    abstained: bool
    citations: CitationReport
    query_used: str
    was_condensed: bool
    budget_applied: int | None = None
    timings: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "abstained": self.abstained,
            "query_used": self.query_used,
            "was_condensed": self.was_condensed,
            "budget_applied": self.budget_applied,
            "citations": self.citations.to_dict(),
            "sources": [
                {"n": i, "url": h.url, "title": h.title, "score": round(h.score, 4)}
                for i, h in enumerate(self.hits, 1)
            ],
            "latency_ms": round(self.latency_ms, 1),
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
        }


def answer_question(
    question: str,
    *,
    history: list[tuple[str, str]] | None = None,
    state: State | str = State.CLEAN,
    config: RetrievalConfig | str = "full",
    llm: GeminiClient | None = None,
) -> Answer:
    t0 = time.perf_counter()
    history = history or []
    config = CONFIGS[config] if isinstance(config, str) else config
    state = State(state)
    llm = llm or GeminiClient()

    query, was_condensed = condense(question, history, llm)

    result = search(query, state, config)
    timings = dict(result.timings)

    prompt = build_prompt(
        question,
        result.hits,
        history="" if was_condensed else _short_history(history),
    )

    t_llm = time.perf_counter()
    try:
        text = llm.generate(prompt, temperature=0.3, use_cache=False)
    except Exception as exc:
        log.error("Generation failed: %s", exc)
        text = "Xin lỗi, hệ thống đang gặp sự cố. Bạn vui lòng thử lại sau ít phút."
    timings["llm_ms"] = (time.perf_counter() - t_llm) * 1000

    citations = validate_citations(text, len(result.hits))
    if citations.invalid:
        # Never render a fabricated source marker as if it were real.
        log.warning("Answer cited non-existent sources %s", citations.invalid)
        text = strip_invalid_citations(text, len(result.hits))

    return Answer(
        text=text,
        hits=result.hits,
        abstained=result.abstained,
        citations=citations,
        query_used=query,
        was_condensed=was_condensed,
        budget_applied=result.budget_applied,
        timings=timings,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def answer_question_stream(
    question: str,
    *,
    history: list[tuple[str, str]] | None = None,
    state: State | str = State.CLEAN,
    config: RetrievalConfig | str = "full",
    llm: GeminiClient | None = None,
):
    """Same pipeline, but yields answer text as it arrives.

    Yields `str` chunks, then a final `Answer` as the last item. Citation
    validation needs the complete text, so it runs after the stream closes —
    which means a chunk can briefly display a citation that validation later
    strikes through. That is the right trade: the alternative is withholding the
    entire answer for several seconds to guarantee markup that is correct on
    ~99% of turns.
    """
    t0 = time.perf_counter()
    history = history or []
    config = CONFIGS[config] if isinstance(config, str) else config
    state = State(state)
    llm = llm or GeminiClient()

    query, was_condensed = condense(question, history, llm)
    result = search(query, state, config)
    timings = dict(result.timings)

    prompt = build_prompt(
        question,
        result.hits,
        history="" if was_condensed else _short_history(history),
    )

    t_llm = time.perf_counter()
    parts: list[str] = []
    try:
        for chunk in llm.stream(prompt):
            parts.append(chunk)
            yield chunk
    except Exception as exc:
        log.error("Streaming generation failed: %s", exc)
        if not parts:
            fallback = "Xin lỗi, hệ thống đang gặp sự cố. Bạn vui lòng thử lại sau ít phút."
            parts.append(fallback)
            yield fallback
    timings["llm_ms"] = (time.perf_counter() - t_llm) * 1000

    text = "".join(parts)
    citations = validate_citations(text, len(result.hits))
    if citations.invalid:
        log.warning("Answer cited non-existent sources %s", citations.invalid)
        text = strip_invalid_citations(text, len(result.hits))

    yield Answer(
        text=text,
        hits=result.hits,
        abstained=result.abstained,
        citations=citations,
        query_used=query,
        was_condensed=was_condensed,
        budget_applied=result.budget_applied,
        timings=timings,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def _short_history(history: list[tuple[str, str]], max_turns: int = 2) -> str:
    if not history:
        return ""
    return "\n".join(f"Khách: {u}\nTrợ lý: {a[:160]}" for u, a in history[-max_turns:])
