"""
╔══════════════════════════════════════════════════════════════════╗
║  WINES RETRIEVAL + GENERATION API                                ║
║  Collection : mastroberardino_wines (Qdrant, hybrid dense+sparse)║
║  Endpoints  : POST /retrieve, POST /ask, GET /health              ║
╚══════════════════════════════════════════════════════════════════╝

Same shape as the NIS2 retrieval API, plus a generation step:
  - /retrieve : hybrid search only (dense bge-m3 + sparse BM25 via
    Qdrant's native RRF fusion). Returns raw chunks. Useful for
    debugging/evaluating retrieval quality on its own.
  - /ask      : /retrieve under the hood, then feeds the retrieved
    chunks to a local LLM (Llama via Ollama, 100% local — no Groq)
    which writes the final answer IN THE QUESTION'S LANGUAGE, even
    though the source chunks are only available in IT/EN. This is
    where the translation actually happens: the LLM reads IT/EN
    context and answers in whatever language the question was in.
    

Both endpoints auto-detect the query's language (langdetect) and
support the same optional metadata filters.

Module layout:
  schemas.py          - Pydantic request/response models
  setup_collection.py - Qdrant config, shared clients, collection setup
  retriever.py         - hybrid dense+sparse retrieval logic
  generator.py         - local LLM answer generation
  main.py (this file)  - FastAPI app + routes only

Run from the wines_RAG/ root:
  uvicorn api.main:app --reload --port 8000

Example requests:
  curl -X POST http://localhost:8000/retrieve \
       -H "Content-Type: application/json" \
       -d '{"query": "vino rosso strutturato per carni rosse", "limit": 5}'

  curl -X POST http://localhost:8000/ask \
       -H "Content-Type: application/json" \
       -d '{"query": "quel vin rouge structuré recommandes-tu avec une viande rouge ?", "limit": 5}'
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import setup_collection as cfg
from .schemas import RetrieveRequest, RetrieveResponse, AskRequest, AskResponse
from .retriever import do_retrieve, to_retrieved_chunk
from .generator import build_context_block, generate_answer


# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Wines Retrieval API",
    description="Hybrid dense+sparse retrieval over the Mastroberardino wines catalogue",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    """Initialise the shared Qdrant + BM25 clients once at process startup."""
    cfg.load_clients()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Basic healthcheck: confirms the API is up and Qdrant is reachable."""
    if cfg.startup_error is not None:
        raise HTTPException(status_code=503, detail=f"Startup failed: {cfg.startup_error}")

    if cfg.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not initialised")

    try:
        exists = cfg.qdrant_client.collection_exists(cfg.COLLECTION_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")

    if not exists:
        raise HTTPException(
            status_code=503,
            detail=f"Collection '{cfg.COLLECTION_NAME}' does not exist. "
                   f"Run wines_qdrant_ingest.py first.",
        )

    count = cfg.qdrant_client.count(cfg.COLLECTION_NAME, exact=True).count
    return {"status": "ok", "collection": cfg.COLLECTION_NAME, "points": count}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Hybrid retrieval only: dense (bge-m3) + sparse (BM25) candidates,
    fused with Qdrant's native RRF. Returns raw chunks, no generation —
    useful for debugging/evaluating retrieval quality on its own.
    """
    try:
        detected_lang, points = do_retrieve(
            query=request.query, limit=request.limit,
            project=request.project, category=request.category,
            wine_type=request.wine_type, language=request.language,
            chunk_type=request.chunk_type,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")

    results = [to_retrieved_chunk(p) for p in points]
    return RetrieveResponse(query=request.query, detected_language=detected_lang, results=results)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Retrieval + generation: runs the same hybrid retrieval as /retrieve,
    then asks a local Llama model (via Ollama) to write an answer in the
    SAME language as the question — even though the source chunks are
    only available in Italian/English. This is the endpoint that does
    the "ask in French, get French back" translation behaviour.
    """
    try:
        detected_lang, points = do_retrieve(
            query=request.query, limit=request.limit,
            project=request.project, category=request.category,
            wine_type=request.wine_type, language=request.language,
            chunk_type=request.chunk_type,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")

    sources = [to_retrieved_chunk(p) for p in points]

    if not sources:
        # Nothing retrieved: don't call the LLM with an empty context,
        # it would just hallucinate. Be explicit about it instead.
        return AskResponse(
            query=request.query,
            detected_language=detected_lang,
            answer="No relevant information was found in the catalogue for this question.",
            sources=[],
        )

    context_block = build_context_block(sources)

    try:
        answer = generate_answer(request.query, context_block)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation failed (Ollama/{cfg.GENERATION_MODEL_NAME}): {e}")

    return AskResponse(
        query=request.query,
        detected_language=detected_lang,
        answer=answer,
        sources=sources,
    )