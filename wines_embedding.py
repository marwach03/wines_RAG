"""
╔══════════════════════════════════════════════════════════════════╗
║  WINES EMBEDDING PIPELINE                                        ║
║  Input  : output/wines_chunks_all.json                          ║
║           (merged stilema + mastroberardino chunks)              ║
║  Model  : bge-m3 (1024d, multilingual) via Ollama, local         ║
║  Output : output/wines_embeddings.json                          ║
║           one record per chunk: chunk_id + vector + metadata,    ║
║           ready to upsert into Qdrant.                           ║
╚══════════════════════════════════════════════════════════════════╝

Same model choice as the NIS2 RAG pipeline: bge-m3 was benchmarked
against LaBSE and Cohere there and won on multilingual Hit@3
(100% vs 83.3% for LaBSE). Reused as-is here, no re-benchmarking.

Usage:
  ollama pull bge-m3                       # one-time, if not already pulled
  python wines_embedding.py
  python wines_embedding.py --input output/wines_chunks_all.json \
                             --output output/wines_embeddings.json \
                             --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ollama

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
DEFAULT_INPUT  = SCRIPT_DIR / "output" / "wines_chunks_all.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "wines_embeddings.json"

MODEL_NAME     = "bge-m3"
EXPECTED_DIM   = 1024   # bge-m3 produces 1024-dimensional dense vectors
DEFAULT_BATCH  = 16     # chunks per Ollama call; keeps payloads reasonable


# ─── LOADING ──────────────────────────────────────────────────────────────────

def load_chunks(input_file: Path) -> list[dict]:
    """Load the merged chunk list produced by wines_chunking.py."""
    if not input_file.exists():
        print(f"ERROR: input file not found: {input_file}", file=sys.stderr)
        print("  Run wines_chunking.py first to produce it.", file=sys.stderr)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        print(f"ERROR: expected a JSON list of chunks in {input_file}", file=sys.stderr)
        sys.exit(1)

    return chunks


# ─── MODEL AVAILABILITY ───────────────────────────────────────────────────────

def ensure_model_available(model_name: str) -> None:
    """
    Check that the embedding model is pulled in the local Ollama instance.
    Fails fast with a clear instruction rather than letting every embed
    call error out one by one.
    """
    try:
        local_models = ollama.list()
    except Exception as e:
        print(f"ERROR: could not reach the local Ollama server: {e}", file=sys.stderr)
        print("  Make sure Ollama is running (e.g. `ollama serve`).", file=sys.stderr)
        sys.exit(1)

    names = []
    for m in local_models.get("models", []):
        # ollama-python returns dicts or Model objects depending on version
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            names.append(name)

    if not any(n.startswith(model_name) for n in names):
        print(f"ERROR: model '{model_name}' is not pulled locally.", file=sys.stderr)
        print(f"  Run: ollama pull {model_name}", file=sys.stderr)
        sys.exit(1)


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], model_name: str = MODEL_NAME) -> list[list[float]]:
    """
    Call Ollama's /api/embed endpoint (via the ollama-python client) with a
    batch of texts. Returns one dense vector per input text, in the same
    order. Raises on failure — caller decides whether to retry.
    """
    response = ollama.embed(model=model_name, input=texts)
    embeddings = response.get("embeddings") if isinstance(response, dict) else response.embeddings
    if embeddings is None or len(embeddings) != len(texts):
        raise RuntimeError(
            f"Unexpected embeddings response: got {len(embeddings) if embeddings else 0} "
            f"vectors for {len(texts)} inputs"
        )
    return embeddings


def embed_chunks(
    chunks: list[dict],
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH,
    max_retries: int = 3,
) -> list[dict]:
    """
    Embed every chunk's `text` field in batches.

    Returns a list of records, each shaped for direct Qdrant ingestion:
      {
        "chunk_id": ...,
        "vector":   [float, ...]   (1024d, bge-m3 dense embedding),
        "payload":  { ...all original chunk metadata except 'text' is kept,
                       plus 'text' itself, so Qdrant can return it directly }
      }

    Chunks with empty/missing text are skipped with a warning — embedding
    an empty string is meaningless and would pollute the vector store.
    """
    records: list[dict] = []
    skipped: list[str] = []

    total = len(chunks)
    print(f"\n[1/3] Embedding {total} chunks with '{model_name}' "
          f"(batch size = {batch_size})...")

    for start in range(0, total, batch_size):
        batch = chunks[start: start + batch_size]

        # Filter out chunks with no usable text up front
        valid_batch = [c for c in batch if c.get("text") and c["text"].strip()]
        for c in batch:
            if c not in valid_batch:
                skipped.append(c.get("chunk_id", "<unknown>"))

        if not valid_batch:
            continue

        texts = [c["text"] for c in valid_batch]

        # Retry loop: local Ollama calls can transiently fail (model loading,
        # OOM under concurrent use, etc.)
        for attempt in range(1, max_retries + 1):
            try:
                vectors = embed_batch(texts, model_name)
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"  ERROR: batch starting at chunk {start} failed "
                          f"after {max_retries} attempts: {e}", file=sys.stderr)
                    raise
                wait = 2 ** attempt
                print(f"  WARN: batch starting at chunk {start} failed "
                      f"(attempt {attempt}/{max_retries}): {e} — retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)

        for chunk, vector in zip(valid_batch, vectors):
            if len(vector) != EXPECTED_DIM:
                print(f"  WARN: chunk {chunk.get('chunk_id')} produced a "
                      f"{len(vector)}-dim vector, expected {EXPECTED_DIM}",
                      file=sys.stderr)

            payload = {k: v for k, v in chunk.items()}  # keep all original fields, including text

            records.append({
                "chunk_id": chunk["chunk_id"],
                "vector":   vector,
                "payload":  payload,
            })

        done = min(start + batch_size, total)
        print(f"  ... {done}/{total} chunks embedded")

    if skipped:
        print(f"\n  WARN: skipped {len(skipped)} chunk(s) with empty text:")
        for cid in skipped:
            print(f"    - {cid}")

    return records


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(input_file: Path, output_file: Path, model_name: str, batch_size: int) -> list[dict]:
    chunks = load_chunks(input_file)
    print(f"  Loaded {len(chunks)} chunks from {input_file}")

    ensure_model_available(model_name)

    records = embed_chunks(chunks, model_name=model_name, batch_size=batch_size)

    print(f"\n[2/3] Embedded {len(records)}/{len(chunks)} chunks successfully")
    if records:
        dims = {len(r["vector"]) for r in records}
        print(f"  Vector dimensions seen: {sorted(dims)}")

    print("\n[3/3] Writing output...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  → {output_file}")

    return records


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate bge-m3 embeddings (via local Ollama) for wines chunks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                         help="Path to the merged chunks JSON (wines_chunks_all.json)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                         help="Path to write the embeddings JSON")
    parser.add_argument("--model", default=MODEL_NAME,
                         help=f"Ollama embedding model name (default: {MODEL_NAME})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                         help=f"Chunks per Ollama call (default: {DEFAULT_BATCH})")
    args = parser.parse_args()

    run(
        input_file  = Path(args.input),
        output_file = Path(args.output),
        model_name  = args.model,
        batch_size  = args.batch_size,
    )


if __name__ == "__main__":
    main()