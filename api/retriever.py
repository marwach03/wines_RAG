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


def rerank_points(query: str, points: list[models.ScoredPoint], limit: int) -> list[models.ScoredPoint]:
    """
    Re-score the RRF-fused candidates with a cross-encoder reranker
    (jina-reranker-v2-base-multilingual, IT/EN/FR-capable), then return
    only the top `limit` by reranker score.

    Cross-encoders score a (query, document) pair jointly rather than
    comparing independently-computed embeddings, which generally gives
    more accurate relevance ordering than the first-stage RRF fusion —
    at the cost of being slower, which is why it only runs on the
    already-narrowed candidate pool, not the whole collection.

    Points with no text in their payload are kept at the end, unscored
    (a cross-encoder has nothing to score them against) rather than
    dropped — being filtered out silently here would be confusing
    compared to /retrieve's normal behaviour.
    """
    if not points:
        return points

    scorable = [p for p in points if (p.payload or {}).get("text")]
    unscorable = [p for p in points if not (p.payload or {}).get("text")]

    if not scorable:
        return points[:limit]

    documents = [p.payload["text"] for p in scorable]
    scores = list(cfg.reranker_model.rerank(query, documents))

    for point, score in zip(scorable, scores):
        point.score = float(score)  # overwrite the RRF score so the
                                      # returned order and the displayed
                                      # score stay consistent — without
                                      # this, callers would see chunks
                                      # sorted by reranker score but
                                      # labelled with their stale RRF score

    reranked = sorted(scorable, key=lambda p: p.score, reverse=True)
    return (reranked + unscorable)[:limit]


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

    Two stages:
      1. Hybrid first-stage retrieval: dense (bge-m3) + sparse (BM25)
         candidates, fused with Qdrant's native RRF — pulls more
         candidates than `limit` so the reranker has a real pool to
         work with (see RERANK_CANDIDATE_MULTIPLIER/_MIN).
      2. Cross-encoder reranking (jina-reranker-v2-base-multilingual)
         re-scores those candidates against the raw query text and
         keeps only the top `limit`.

    Returns (detected_language, scored_points). Raises plain Python
    exceptions on failure — callers (the HTTP endpoints in main.py) are
    responsible for translating those into HTTPException with the
    right status code, so this function stays usable outside FastAPI
    too (scripts, tests, future endpoints).
    """
    if cfg.startup_error is not None:
        raise RuntimeError(f"API not initialised: {cfg.startup_error}")
    if cfg.qdrant_client is None or cfg.bm25_model is None or cfg.reranker_model is None:
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

    rerank_pool_size = max(limit * cfg.RERANK_CANDIDATE_MULTIPLIER, cfg.RERANK_CANDIDATE_MIN)

    response = cfg.qdrant_client.query_points(
        collection_name=cfg.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vector,  using=cfg.DENSE_VECTOR_NAME,  limit=cfg.PREFETCH_LIMIT, filter=query_filter),
            models.Prefetch(query=sparse_vector, using=cfg.SPARSE_VECTOR_NAME, limit=cfg.PREFETCH_LIMIT, filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=rerank_pool_size,
    )

    reranked_points = rerank_points(query, response.points, limit)

    return detected_lang, reranked_points