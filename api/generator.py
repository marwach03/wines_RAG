"""
Answer generation via a local Llama model (Ollama, 100% local — no Groq).
Takes retrieved chunks (Italian/English source text) and writes the final
answer in whatever language the question was asked in.
"""

from __future__ import annotations

import ollama

from . import setup_collection as cfg
from .schemas import RetrievedChunk


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Render the retrieved chunks as a single text block to inject into
    the generation prompt. Each source is numbered so the model can
    refer back to it, and so the caller can sanity-check which chunks
    fed which part of the answer.
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = c.title or c.wine_name or c.chunk_id
        parts.append(f"[Source {i} — {label} ({c.language or '?'})]\n{c.text or ''}")
    return "\n\n".join(parts)


def generate_answer(query: str, context_block: str) -> str:
    """
    Generate the final answer with a local Llama model via Ollama.

    The system prompt explicitly instructs the model to answer in the
    SAME language as the question, regardless of what language the
    retrieved source chunks are written in (IT/EN only, today) — this
    is where the "ask in French, get a French answer" behaviour
    actually happens; the vector store itself never translates anything.
    """
    system_prompt = (
        "You are a knowledgeable assistant for the Mastroberardino wine catalogue. "
        "You will be given a question and several source excerpts (in Italian and/or "
        "English) retrieved from the catalogue. Answer the question using ONLY the "
        "information in the sources. "
        "VERY IMPORTANT: always answer in the SAME language as the question, "
        "translating naturally from the sources if needed — never answer in the "
        "source language if the question was asked in a different one. "
        "If the sources don't contain enough information to answer, say so honestly "
        "in the question's language instead of making things up."
    )

    user_prompt = f"Question: {query}\n\nSources:\n{context_block}"

    response = ollama.chat(
        model=cfg.GENERATION_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    message = response.get("message") if isinstance(response, dict) else response.message
    content = message.get("content") if isinstance(message, dict) else message.content
    if not content:
        raise RuntimeError("Ollama returned an empty generation response")
    return content