"""Shared, cached client construction.

The app and the eval harness must use *identical* clients and models, otherwise
before/after numbers are not comparable. Both import from here.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from rag.config import get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder, SentenceTransformer

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    from qdrant_client import QdrantClient

    s = get_settings()
    s.require("qdrant_endpoint", "qdrant_api_key")
    log.info("Connecting to Qdrant at %s", s.qdrant_endpoint)
    return QdrantClient(url=s.qdrant_endpoint, api_key=s.qdrant_api_key, timeout=60)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    log.info("Loading embedding model %s (CPU)", s.embedding_model)
    return SentenceTransformer(s.embedding_model, trust_remote_code=True, device="cpu")


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    s = get_settings()
    log.info("Loading reranker %s (CPU)", s.reranker_model)
    return CrossEncoder(s.reranker_model, trust_remote_code=True, device="cpu")


@lru_cache(maxsize=2)
def get_gemini(model_name: str | None = None):
    """Generator by default; pass `judge_model` for evaluation calls."""
    import google.generativeai as genai

    s = get_settings()
    s.require("gemini_api_key")
    genai.configure(api_key=s.gemini_api_key)
    return genai.GenerativeModel(model_name or s.gemini_model)


def embed(text: str) -> list[float]:
    """Embed a single query string.

    Returns [] for blank input so callers can short-circuit instead of sending
    a meaningless vector to Qdrant.
    """
    if not text or not text.strip():
        return []
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )
    return [v.tolist() for v in vectors]
