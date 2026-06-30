"""
Qdrant collection setup and shared client state for the API.

This mirrors the collection definition in wines_qdrant_ingest.py
(same collection name, same dense/sparse vector config) — kept as a
separate, intentionally duplicated copy here so the API module
doesn't import the standalone ingestion script as a dependency.
If you change the vector config in one place, mirror it in the other.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# ─── CONFIG ───────────────────────────────────────────────────────────────────
COLLECTION_NAME      = "mastroberardino_wines"
QDRANT_HOST           = "localhost"
QDRANT_PORT            = 6333

DENSE_VECTOR_NAME      = "dense"
SPARSE_VECTOR_NAME     = "sparse"
DENSE_DIM               = 1024            # bge-m3 dense embedding size

DENSE_MODEL_NAME        = "bge-m3"        # via Ollama, same model used for ingestion
BM25_MODEL_NAME          = "Qdrant/bm25"  # via FastEmbed, same model used for ingestion
GENERATION_MODEL_NAME     = "llama3.2:1b" # via Ollama, 100% local — no Groq; 1b chosen for low-RAM (8GB) CPU machines
RERANKER_MODEL_NAME        = "jinaai/jina-reranker-v2-base-multilingual"  # via FastEmbed; chosen over
                                          # bge-reranker-base because it's explicitly multilingual
                                          # (IT/EN/FR), unlike bge-reranker-base/-large which are
                                          # optimised for Chinese/English. ~1.1GB, fine on 8GB RAM CPU.

DEFAULT_LIMIT            = 5
PREFETCH_LIMIT            = 50   # candidates pulled per branch (dense/sparse) before RRF fusion
RERANK_CANDIDATE_MULTIPLIER = 4  # how many extra candidates to pull from RRF before reranking,
                                  # relative to the final limit (e.g. limit=5 -> rerank top 20)
RERANK_CANDIDATE_MIN       = 20  # floor on the above, so small `limit` values still give the
                                  # reranker a meaningful pool to work with

# Fields the caller is allowed to filter on. Mirrors the chunk metadata
# produced by wines_chunking.py / wines_embedding.py.
FILTERABLE_FIELDS = {"project", "category", "wine_type", "language", "chunk_type"}


# ─── SHARED CLIENT STATE ──────────────────────────────────────────────────────
# Initialised once at API startup (see main.py's startup event) and read by
# retriever.py / generator.py. Kept as module-level globals here so every
# module imports the same single instance rather than creating its own.

qdrant_client: Optional[QdrantClient] = None
bm25_model: Optional[SparseTextEmbedding] = None
reranker_model: Optional["TextCrossEncoder"] = None
startup_error: Optional[str] = None


def load_clients() -> None:
    """
    Initialise the Qdrant client, the BM25 sparse model, and the
    cross-encoder reranker once, at process startup, rather than on
    every request. The Ollama client is stateless (a thin HTTP wrapper)
    so it doesn't need explicit initialisation here.

    Failures here (Qdrant unreachable, models not downloadable yet,
    etc.) are captured rather than raised, so the API process still comes
    up and /health can report what's wrong instead of the whole server
    failing to start.
    """
    global qdrant_client, bm25_model, reranker_model, startup_error
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
        reranker_model = TextCrossEncoder(model_name=RERANKER_MODEL_NAME)
    except Exception as e:
        startup_error = str(e)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the hybrid collection (dense + sparse named vectors) if it
    doesn't exist yet, or recreate it from scratch if requested.

    Not called automatically by the API — ingestion (wines_qdrant_ingest.py)
    is expected to have already created the collection. This is provided so
    the API side can also (re)create it standalone if needed, e.g. for
    local dev/testing without running the full ingestion pipeline first.
    """
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )