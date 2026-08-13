"""Contract tests for the streaming answer path.

The app consumes `answer_question_stream` as "strings until an Answer arrives".
If that contract slips — an Answer never yielded, or a non-string mixed in — the
UI shows a partial reply with no sources and no error. Stubbed end to end, so
these run without network or quota.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.config import State
from rag.generation import answer as answer_mod
from rag.generation.answer import Answer, answer_question_stream
from rag.retrieval.search import Hit, SearchResult


@dataclass
class StubLLM:
    """Stands in for GeminiClient. `chunks` is what the model "produces"."""

    chunks: tuple[str, ...] = ("Bó Hoa M1 ", "giá 500.000₫ [1].")
    raise_after: int | None = None

    def stream(self, prompt: str, *, temperature: float = 0.3):
        for i, chunk in enumerate(self.chunks):
            if self.raise_after is not None and i == self.raise_after:
                raise RuntimeError("upstream died mid-stream")
            yield chunk

    def generate(self, prompt: str, **kw):  # used by condense()
        return "standalone question"


def _hits(n: int = 2) -> list[Hit]:
    return [
        Hit(
            url=f"https://hoatuoimymy.com/bo-hoa-m{i}/",
            title=f"Bó Hoa M{i}",
            score=0.8,
            payload={"url": f"https://hoatuoimymy.com/bo-hoa-m{i}/", "title": f"Bó Hoa M{i}"},
        )
        for i in range(1, n + 1)
    ]


@pytest.fixture
def stub_search(monkeypatch):
    def _search(question, state, config):
        return SearchResult(hits=_hits(), abstained=False, latency_ms=1.0, timings={})

    monkeypatch.setattr(answer_mod, "search", _search)


def _consume(gen) -> tuple[list[str], Answer | None]:
    chunks: list[str] = []
    final: Answer | None = None
    for item in gen:
        if isinstance(item, str):
            chunks.append(item)
        else:
            final = item
    return chunks, final


def test_yields_text_then_a_single_answer(stub_search) -> None:
    chunks, final = _consume(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=StubLLM())
    )
    assert chunks == ["Bó Hoa M1 ", "giá 500.000₫ [1]."]
    assert isinstance(final, Answer)
    assert final.text == "".join(chunks)


def test_answer_is_always_last(stub_search) -> None:
    """The app relies on the Answer terminating the stream."""
    items = list(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=StubLLM())
    )
    assert isinstance(items[-1], Answer)
    assert all(isinstance(i, str) for i in items[:-1])


def test_citations_are_validated_after_the_stream_closes(stub_search) -> None:
    """A fabricated index must be struck from the final text, not the chunks."""
    llm = StubLLM(chunks=("Sản phẩm này giá 500.000₫ [9].",))
    chunks, final = _consume(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=llm)
    )
    assert "[9]" in chunks[0], "the raw chunk is shown as produced"
    assert final.citations.invalid == [9]
    assert "[9]" not in final.text, "validation strips it from the committed answer"


def test_mid_stream_failure_still_yields_an_answer(stub_search) -> None:
    """A dropped connection must not leave the UI without a terminating Answer."""
    llm = StubLLM(chunks=("Bó hoa ", "này ", "rất đẹp."), raise_after=2)
    chunks, final = _consume(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=llm)
    )
    assert chunks == ["Bó hoa ", "này "]
    assert isinstance(final, Answer)
    assert final.text == "Bó hoa này "


def test_total_failure_yields_a_user_facing_message(stub_search) -> None:
    llm = StubLLM(chunks=("anything",), raise_after=0)
    chunks, final = _consume(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=llm)
    )
    assert chunks and "sự cố" in chunks[0]
    assert isinstance(final, Answer)


def test_timings_include_generation(stub_search) -> None:
    _, final = _consume(
        answer_question_stream("hoa sinh nhật", state=State.CLEAN, config="dense", llm=StubLLM())
    )
    assert "llm_ms" in final.timings
    assert final.latency_ms > 0
