"""
Hybrid retrieval logic (dense bge-m3 + sparse BM25, fused via Qdrant's
native RRF). Shared by the /retrieve and /ask endpoints in main.py.
"""

from __future__ import annotations

from typing import Optional

import ollama
from langdetect import detect, LangDetectException
from qdrant_client import models

from . import setup_collection as cfg
from .schemas import RetrievedChunk


def detect_language(text: str) -> Optional[str]:
    """
    Best-effort language detection on the raw query string.
    Returns None if detection fails (e.g. query too short/ambiguous) —
    callers should treat this as 'unknown', not as an error.
    """
    try:
        return detect(text)
    except LangDetectException:
        return None


def embed_dense(text: str) -> list[float]:
    """Embed the query with bge-m3 via the local Ollama instance."""
    response = ollama.embed(model=cfg.DENSE_MODEL_NAME, input=text)
    embeddings = response.get("embeddings") if isinstance(response, dict) else response.embeddings
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding for the query")
    return embeddings[0]


def embed_sparse(text: str) -> models.SparseVector:
    """Embed the query with BM25 via FastEmbed (same model used at ingestion)."""
    raw = next(iter(cfg.bm25_model.embed([text])))
    return models.SparseVector(indices=raw.indices.tolist(), values=raw.values.tolist())


def to_retrieved_chunk(point: models.ScoredPoint) -> RetrievedChunk:
    """Map a Qdrant scored point back into the API's response schema."""
    payload = point.payload or {}

    def first_or_none(value):
        # category/project/etc. are stored as lists (for MatchAny support);
        # unwrap to a single value for the response.
        if isinstance(value, list):
            return value[0] if value else None
        return value

    return RetrievedChunk(
        chunk_id   = payload.get("chunk_id", str(point.id)),
        score      = point.score,
        title      = payload.get("title"),
        text       = payload.get("text"),
        project    = first_or_none(payload.get("project")),
        category   = first_or_none(payload.get("category")),
        wine_type  = payload.get("wine_type"),
        language   = payload.get("language"),
        chunk_type = payload.get("chunk_type"),
        wine_name  = payload.get("wine_name"),
        vintage    = payload.get("vintage"),
    )


def do_retrieve(
    query: str,
    limit: int,
    project: Optional[str] = None,
    category: Optional[str] = None,
    wine_type: Optional[str] = None,
    language: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> tuple[Optional[str], list[models.ScoredPoint]]:
    """
    Core hybrid retrieval logic, shared by /retrieve and /ask.

    Returns (detected_language, scored_points). Raises plain Python
    exceptions on failure — callers (the HTTP endpoints in main.py) are
    responsible for translating those into HTTPException with the
    right status code, so this function stays usable outside FastAPI
    too (scripts, tests, future endpoints).
    """
    if cfg.startup_error is not None:
        raise RuntimeError(f"API not initialised: {cfg.startup_error}")
    if cfg.qdrant_client is None or cfg.bm25_model is None:
        raise RuntimeError("API not fully initialised yet")

    detected_lang = detect_language(query)

    dense_vector  = embed_dense(query)
    sparse_vector = embed_sparse(query)

    conditions = []
    field_values = {
        "project": project, "category": category, "wine_type": wine_type,
        "language": language, "chunk_type": chunk_type,
    }
    for field, value in field_values.items():
        if value is not None:
            conditions.append(models.FieldCondition(key=field, match=models.MatchAny(any=[value])))
    query_filter = models.Filter(must=conditions) if conditions else None

    response = cfg.qdrant_client.query_points(
        collection_name=cfg.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector,  using=cfg.DENSE_VECTOR_NAME,  limit=cfg.PREFETCH_LIMIT, filter=query_filter),
            models.Prefetch(query=sparse_vector, using=cfg.SPARSE_VECTOR_NAME, limit=cfg.PREFETCH_LIMIT, filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )

    return detected_lang, response.points