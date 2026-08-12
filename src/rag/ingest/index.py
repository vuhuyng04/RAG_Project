"""Create Qdrant collections and upsert one corpus state into one collection."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from qdrant_client import models

from rag.clients import embed_batch, get_qdrant
from rag.config import State, get_settings
from rag.ingest.clean import build_embed_text
from rag.ingest.legacy import legacy_embed_text, legacy_payload
from rag.ingest.schema import Product
from rag.retrieval.sparse import SPARSE_VECTOR_NAME, encode_documents

log = logging.getLogger(__name__)

UPSERT_BATCH = 64


def point_id(url: str) -> str:
    """Deterministic id so re-running updates points instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def builders_for(state: State) -> tuple[Callable[[Product], str], Callable[[Product], dict]]:
    """The (embed_text, payload) pair a state indexes with.

    Single source of truth. Indexing and quality reporting both go through here
    so a state can never be profiled with a different text builder than the one
    it was indexed with.
    """
    if State(state) is State.LEGACY:
        return legacy_embed_text, legacy_payload
    return build_embed_text, Product.to_payload


def ensure_collection(name: str, *, recreate: bool = False) -> None:
    client = get_qdrant()
    s = get_settings()

    if recreate and client.collection_exists(name):
        log.warning("Deleting existing collection %s", name)
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=s.embedding_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            # Named sparse vector alongside the unnamed dense one, so hybrid
            # search is a query-time choice rather than a separate collection.
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        log.info("Created collection %s (dim=%d, cosine, +bm25 sparse)", name, s.embedding_dim)

    # Payload indexes. price_vnd unlocks budget filtering ("dưới 500k"), which
    # the legacy pipeline could not do at all because price was a formatted
    # string. The title full-text index is the fallback lexical path if BM25
    # sparse vectors tokenize Vietnamese poorly (plan §4.4).
    for field, schema in (
        ("price_vnd", models.PayloadSchemaType.INTEGER),
        ("url", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception as exc:  # already exists
            log.debug("payload index %s on %s: %s", field, name, exc)

    try:
        client.create_payload_index(
            name,
            field_name="title",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.MULTILINGUAL,
                lowercase=True,
            ),
        )
    except Exception as exc:
        log.debug("text index on %s: %s", name, exc)


def index_state(
    products: list[Product],
    state: State,
    *,
    recreate: bool = True,
) -> dict[str, object]:
    """Embed and upsert `products` into the collection for `state`.

    The legacy state uses the notebook's text and payload builders; every other
    state uses the cleaned ones. This is the single place the two pipelines
    diverge, which keeps the comparison honest.
    """
    s = get_settings()
    collection = s.collection_for(state)
    ensure_collection(collection, recreate=recreate)

    text_fn, payload_fn = builders_for(state)
    texts = [text_fn(p) for p in products]
    log.info("Embedding %d documents for state=%s ...", len(texts), state.value)
    vectors = embed_batch(texts)

    log.info("Building BM25 sparse vectors ...")
    sparse_vectors = encode_documents(texts)

    client = get_qdrant()
    total = 0
    for start in range(0, len(products), UPSERT_BATCH):
        chunk = products[start : start + UPSERT_BATCH]
        chunk_vecs = vectors[start : start + UPSERT_BATCH]
        chunk_sparse = sparse_vectors[start : start + UPSERT_BATCH]
        chunk_texts = texts[start : start + UPSERT_BATCH]
        points = [
            models.PointStruct(
                id=point_id(p.url),
                vector={"": v, SPARSE_VECTOR_NAME: sv},
                # embed_text is stored so the eval harness can show exactly what
                # was indexed without re-deriving it.
                payload={**payload_fn(p), "embed_text": t},
            )
            for p, v, sv, t in zip(chunk, chunk_vecs, chunk_sparse, chunk_texts, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        total += len(points)
        log.info("  upserted %d/%d", total, len(products))

    info = client.get_collection(collection)
    log.info("State %s -> %s: %d points", state.value, collection, info.points_count)
    return {
        "state": state.value,
        "collection": collection,
        "points": info.points_count,
        "documents": len(products),
    }
