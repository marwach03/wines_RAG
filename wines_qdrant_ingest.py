"""
╔══════════════════════════════════════════════════════════════════╗
║  WINES QDRANT INGESTION PIPELINE                                 ║
║  Input  : output/wines_embeddings.json                          ║
║           (chunk_id + dense vector (bge-m3, 1024d) + payload)    ║
║  Output : Qdrant collection 'mastroberardino_wines'              ║
║           - named vector 'dense'  : bge-m3 embedding (1024d)     ║
║           - named vector 'sparse' : BM25 sparse vector           ║
║             (computed client-side via FastEmbed, IDF modifier    ║
║             enabled on the collection so Qdrant can score it)    ║
╚══════════════════════════════════════════════════════════════════╝

Same hybrid setup as the NIS2 RAG pipeline:
  - dense (bge-m3)  : semantic / multilingual similarity
  - sparse (BM25)   : exact keyword matching
  - fusion at query time via Qdrant's native RRF (see query example
    at the bottom of this file / wines_query.py if you build one)

BM25 sparse vectors are computed CLIENT-SIDE with FastEmbed
("Qdrant/bm25" model) rather than relying on Qdrant's server-side
BM25 inference (available since Qdrant 1.15.2). This keeps the
script working regardless of which Qdrant server version is
running locally.

Usage:
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant   # if not already running
  python wines_qdrant_ingest.py
  python wines_qdrant_ingest.py --recreate   # drop & rebuild the collection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).parent
DEFAULT_INPUT      = SCRIPT_DIR / "output" / "wines_embeddings.json"
COLLECTION_NAME    = "mastroberardino_wines"

QDRANT_HOST        = "localhost"
QDRANT_PORT         = 6333

DENSE_VECTOR_NAME   = "dense"
SPARSE_VECTOR_NAME  = "sparse"
DENSE_DIM            = 1024     # bge-m3 dense embedding size
BM25_MODEL_NAME      = "Qdrant/bm25"

UPSERT_BATCH_SIZE    = 64

# Payload fields that should be turned into native Python lists before
# upsert, so Qdrant's MatchAny filter works on them (same fix applied on
# the NIS2 pipeline for the 'topic' field — comma-joined strings break
# MatchAny silently).
LIST_FIELDS = {"category"}  # category can hold a single string today;
                             # kept as a list-compatible field for future
                             # multi-category chunks without a payload migration


# ─── LOADING ──────────────────────────────────────────────────────────────────

def load_records(input_file: Path) -> list[dict]:
    """Load the embeddings + payload produced by wines_embedding.py."""
    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        print("  Run wines_embedding.py first to produce it.", file=sys.stderr)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        print(f"ERROR: expected a JSON list of records in {input_file}", file=sys.stderr)
        sys.exit(1)

    return records


# ─── COLLECTION SETUP ─────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """
    Create the hybrid collection (dense + sparse named vectors) if it
    doesn't exist yet, or recreate it from scratch if --recreate is set.
    """
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        print(f"  Dropping existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        print(f"  Collection '{COLLECTION_NAME}' already exists, reusing it.")
        return

    print(f"  Creating collection '{COLLECTION_NAME}'...")
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


# ─── SPARSE EMBEDDING ─────────────────────────────────────────────────────────

def compute_sparse_vectors(texts: list[str], bm25_model: SparseTextEmbedding) -> list[models.SparseVector]:
    """
    Compute BM25 sparse vectors for a batch of texts via FastEmbed.
    Returns one models.SparseVector per input text, same order.
    """
    raw = list(bm25_model.embed(texts))
    return [
        models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in raw
    ]


# ─── PAYLOAD NORMALISATION ────────────────────────────────────────────────────

def normalize_payload(payload: dict) -> dict:
    """
    Ensure filterable fields are native Python lists rather than scalars,
    so Qdrant's MatchAny filter works correctly (mirrors the NIS2 fix for
    the 'topic' field: comma-joined strings silently break MatchAny).
    """
    payload = dict(payload)
    for field in LIST_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            payload[field] = [value]
        elif isinstance(value, list):
            pass  # already correct
        else:
            payload[field] = [value]
    return payload


# ─── INGESTION ────────────────────────────────────────────────────────────────

def ingest(
    client: QdrantClient,
    records: list[dict],
    bm25_model: SparseTextEmbedding,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> int:
    """
    Upsert every record into the collection. Each point gets:
      - a deterministic integer id derived from its position (Qdrant
        accepts int or UUID ids; we use the running index since
        chunk_id strings are kept in the payload for lookups/filters)
      - a 'dense' named vector (already computed, from wines_embedding.py)
      - a 'sparse' named vector (computed here, client-side, via FastEmbed)
      - the full original payload, with list-type fields normalised
    """
    total = len(records)
    upserted = 0
    skipped: list[str] = []

    print(f"\n[1/2] Upserting {total} points into '{COLLECTION_NAME}' "
          f"(batch size = {batch_size})...")

    for start in range(0, total, batch_size):
        batch = records[start: start + batch_size]

        valid_batch = []
        for r in batch:
            if not r.get("vector") or len(r["vector"]) != DENSE_DIM:
                skipped.append(r.get("chunk_id", "<unknown>"))
                continue
            text = r.get("payload", {}).get("text", "")
            if not text or not text.strip():
                skipped.append(r.get("chunk_id", "<unknown>"))
                continue
            valid_batch.append(r)

        if not valid_batch:
            continue

        texts          = [r["payload"]["text"] for r in valid_batch]
        sparse_vectors = compute_sparse_vectors(texts, bm25_model)

        points = []
        for idx, (record, sparse_vec) in enumerate(zip(valid_batch, sparse_vectors)):
            payload = normalize_payload(record["payload"])
            points.append(
                models.PointStruct(
                    id=start + idx,
                    vector={
                        DENSE_VECTOR_NAME:  record["vector"],
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        upserted += len(points)

        done = min(start + batch_size, total)
        print(f"  ... {done}/{total} records processed ({upserted} upserted so far)")

    if skipped:
        print(f"\n  WARN: skipped {len(skipped)} record(s) (missing/invalid vector or text):")
        for cid in skipped:
            print(f"    - {cid}")

    return upserted


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(
    input_file: Path,
    host: str,
    port: int,
    recreate: bool,
    batch_size: int,
) -> None:
    records = load_records(input_file)
    print(f"  Loaded {len(records)} records from {input_file}")

    client = QdrantClient(host=host, port=port)

    print("\n[setup] Ensuring collection exists...")
    ensure_collection(client, recreate=recreate)

    print("\n[setup] Loading BM25 sparse model (FastEmbed, first run downloads it)...")
    bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

    upserted = ingest(client, records, bm25_model, batch_size=batch_size)

    print(f"\n[2/2] Done. {upserted} points upserted into '{COLLECTION_NAME}'.")
    count = client.count(COLLECTION_NAME, exact=True).count
    print(f"  Collection now holds {count} points total.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest wines embeddings into a hybrid Qdrant collection "
                     "(dense bge-m3 + sparse BM25)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                         help="Path to wines_embeddings.json")
    parser.add_argument("--host", default=QDRANT_HOST,
                         help=f"Qdrant host (default: {QDRANT_HOST})")
    parser.add_argument("--port", type=int, default=QDRANT_PORT,
                         help=f"Qdrant REST port (default: {QDRANT_PORT})")
    parser.add_argument("--recreate", action="store_true",
                         help="Drop and recreate the collection before ingesting")
    parser.add_argument("--batch-size", type=int, default=UPSERT_BATCH_SIZE,
                         help=f"Points per upsert call (default: {UPSERT_BATCH_SIZE})")
    args = parser.parse_args()

    run(
        input_file = Path(args.input),
        host       = args.host,
        port       = args.port,
        recreate   = args.recreate,
        batch_size = args.batch_size,
    )


if __name__ == "__main__":
    main()