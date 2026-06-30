"""
Answer generation via a local Llama model (Ollama, 100% local — no Groq).
Takes retrieved chunks (Italian/English source text) and writes the final
answer in whatever language the question was asked in.
"""

from __future__ import annotations

import ollama

from . import setup_collection as cfg
from .schemas import RetrievedChunk


def reorder_for_lost_in_the_middle(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Mitigate the "lost in the middle" effect (Liu et al., 2023): LLMs tend
    to under-use information placed in the middle of a long context,
    favouring what's at the very start or the very end.

    Chunks arrive already sorted by relevance (best first, from Qdrant's
    RRF fusion). This re-orders them so the most relevant chunks end up
    at BOTH ends of the context block, and the least relevant ones are
    pushed to the middle — the position where a small model is most
    likely to under-use them anyway.

    Example with 5 chunks ranked [1, 2, 3, 4, 5] by relevance (1 = best):
      front gets odd ranks in order : 1, 3, 5
      back  gets even ranks, reversed: 4, 2
      final layout: [1, 3, 5, 4, 2]
                     ^best        ^2nd-best
      i.e. rank 1 opens the context, rank 2 closes it, rank 5 (weakest)
      sits in the dead centre.

    With <= 2 chunks this is a no-op (nothing to reorder).
    """
    if len(chunks) <= 2:
        return list(chunks)

    front: list[RetrievedChunk] = []
    back: list[RetrievedChunk] = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            front.append(chunk)
        else:
            back.append(chunk)

    return front + list(reversed(back))


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Render the retrieved chunks as a single text block to inject into
    the generation prompt. Each source is numbered so the model can
    refer back to it, and so the caller can sanity-check which chunks
    fed which part of the answer.

    Chunks are re-ordered first (see reorder_for_lost_in_the_middle) so
    the most relevant sources sit at the start and end of the block
    rather than being diluted in the middle.
    """
    ordered = reorder_for_lost_in_the_middle(chunks)
    parts = []
    for i, c in enumerate(ordered, start=1):
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
        "You are a strict but fluent retrieval-based assistant for the Mastroberardino wine catalogue.\n\n"

        "TASK:\n"
        "Answer the question using ONLY the provided sources.\n\n"

        "HARD RULES:\n"
        "- Do NOT add any external knowledge.\n"
        "- Do NOT invent, infer, or complete missing information.\n"
        "- Do NOT add refusal sentences unless the entire answer is missing.\n"
        "- Every factual statement must be grounded in the sources.\n\n"

        "SOURCE USAGE:\n"
        "- Use only information explicitly present in the sources for the requested wine/vintage.\n"
        "- Ignore unrelated wines or vintages.\n"
        "- Do not mix different sources if they refer to different products.\n\n"

        "TRANSLATION RULE (IMPORTANT):\n"
        "- You MAY translate the source content into the user's language.\n"
        "- Translation must be faithful and natural.\n"
        "- You may slightly rephrase ONLY for grammar, not meaning.\n"
        "- Do NOT add new adjectives, interpretations, or restructuring of facts.\n\n"

        "OUTPUT STYLE:\n"
        "- Produce one coherent answer (no bullet fragmentation unless present in sources).\n"
        "- Do not repeat sentences like 'Information not available in the sources'.\n"
        "- If no relevant information exists at all, say so once, clearly.\n\n"

        "LANGUAGE RULE:\n"
        "- Always respond in the language of the user question.\n"
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


# ─── DIAGNOSTIC: "lost in the middle" empirical test ─────────────────────────

def test_lost_in_the_middle(
    n_filler_chunks: int = 4,
    needle_position: int = 2,
    filler_text: str = (
        "Questo vino presenta un colore rosso rubino intenso con riflessi granati. "
        "Il profumo è caratterizzato da note di frutta rossa matura e leggere "
        "speziature, con un finale persistente e tannini ben integrati."
    ),
) -> dict:
    """
    Manual diagnostic for the "lost in the middle" effect on the current
    GENERATION_MODEL_NAME.

    Builds a fake context of `n_filler_chunks` generic, look-alike wine
    description chunks, plants ONE distinctive "needle" fact (a made-up
    serving temperature that can't be guessed or hallucinated correctly
    by chance) at `needle_position` (0 = very first chunk, len-1 = very
    last chunk, anything in between = the middle), then asks the model a
    question that can ONLY be answered correctly by reading that needle.

    Returns a dict with the raw answer and whether the needle value was
    found in it, so you can compare needle_position=0 (front, easy),
    needle_position=middle (hard, the effect we're worried about), and
    needle_position=last (back, also easy) to see if the model's recall
    actually drops in the middle on llama3.2:1b (or whichever model is
    configured) with YOUR real chunk sizes.

    This does NOT touch Qdrant — it's a synthetic, fully offline test of
    the generation step alone. Run it from a Python shell:

        from api.generator import test_lost_in_the_middle
        test_lost_in_the_middle(n_filler_chunks=4, needle_position=0)
        test_lost_in_the_middle(n_filler_chunks=4, needle_position=2)
        test_lost_in_the_middle(n_filler_chunks=4, needle_position=4)

    Compare the three `needle_found` results.
    """
    total_chunks = n_filler_chunks + 1
    if not (0 <= needle_position < total_chunks):
        raise ValueError(
            f"needle_position must be between 0 and {total_chunks - 1} "
            f"(got {needle_position}) for {n_filler_chunks} filler chunks + 1 needle"
        )

    needle_temperature = "14.5°C"  # arbitrary, distinctive — not a plausible default guess
    needle_wine_name    = "Vendemmia Segreta"  # fake wine name, won't collide with real catalogue entries

    needle_chunk = RetrievedChunk(
        chunk_id   = "synthetic_needle",
        score      = 1.0,
        title      = needle_wine_name,
        text       = (
            f"[{needle_wine_name}]\n\n{filler_text} "
            f"La temperatura di servizio consigliata per questo vino è {needle_temperature}."
        ),
        wine_name  = needle_wine_name,
        language   = "it",
        chunk_type = "wine_sheet",
    )

    fake_chunks: list[RetrievedChunk] = []
    for i in range(total_chunks):
        if i == needle_position:
            fake_chunks.append(needle_chunk)
        else:
            fake_chunks.append(RetrievedChunk(
                chunk_id   = f"synthetic_filler_{i}",
                score      = 1.0,
                title      = f"Vino Filler {i}",
                text       = f"[Vino Filler {i}]\n\n{filler_text}",
                wine_name  = f"Vino Filler {i}",
                language   = "it",
                chunk_type = "wine_sheet",
            ))

    # IMPORTANT: bypass build_context_block's own reordering here — we want
    # to control the EXACT position of the needle in the raw context fed to
    # the model, to isolate the model's positional recall from our own
    # mitigation logic. To test the mitigation itself, call
    # build_context_block(fake_chunks) instead of this raw join.
    parts = []
    for i, c in enumerate(fake_chunks, start=1):
        label = c.title or c.wine_name or c.chunk_id
        parts.append(f"[Source {i} — {label} ({c.language or '?'})]\n{c.text or ''}")
    raw_context_block = "\n\n".join(parts)

    question = f"Qual è la temperatura di servizio consigliata per il vino '{needle_wine_name}'?"

    answer = generate_answer(question, raw_context_block)
    needle_found = needle_temperature in answer

    return {
        "model":              cfg.GENERATION_MODEL_NAME,
        "n_filler_chunks":    n_filler_chunks,
        "needle_position":    needle_position,
        "total_chunks":       total_chunks,
        "expected_value":     needle_temperature,
        "answer":             answer,
        "needle_found":       needle_found,
    }