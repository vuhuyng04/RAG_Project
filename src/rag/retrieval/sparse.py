"""BM25 sparse vectors for hybrid retrieval.

fastembed's BM25 does **not** support Vietnamese: `SparseTextEmbedding(
"Qdrant/bm25", language="vietnamese")` raises "vietnamese language is not
supported". It falls back to the English analyser, which means no Vietnamese
stopword removal and an English stemmer applied to Vietnamese tokens.

That is less damaging than it sounds, and it was checked rather than assumed.
Vietnamese is analytic — words do not inflect — so whitespace tokenisation
loses far less than it would for a fusional language. A smoke test with
"kệ hoa khai trương M362" against three products gave term overlaps of
[5, 1, 1], correctly ranking the intended document first.

Whether it earns its place in the pipeline is settled by the eval matrix, not
by this reasoning. If hybrid does not beat dense, that negative result is
reported (plan §4.4).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import models

log = logging.getLogger(__name__)

SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_bm25():
    from fastembed import SparseTextEmbedding

    log.info("Loading sparse model %s (English analyser — see module docstring)", BM25_MODEL)
    return SparseTextEmbedding(BM25_MODEL)


def _to_qdrant(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )


def encode_documents(texts: list[str]) -> list[models.SparseVector]:
    """Document-side BM25 (term frequency weighted)."""
    return [_to_qdrant(e) for e in get_bm25().embed(texts)]


def encode_query(text: str) -> models.SparseVector:
    """Query-side BM25.

    `query_embed` differs from `embed`: query terms are not TF-weighted, which
    is what BM25 expects. Using the document encoder for queries is a common
    and silent mistake.
    """
    return _to_qdrant(next(iter(get_bm25().query_embed([text]))))
