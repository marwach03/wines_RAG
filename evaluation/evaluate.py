"""
╔══════════════════════════════════════════════════════════════════╗
║         WINES RAG PIPELINE  —  EVALUATION                       ║
║  Step : Évaluation rigoureuse avec LLM-judge                    ║
╚══════════════════════════════════════════════════════════════════╝

Same evaluation structure as the NIS2 RAG pipeline's evaluation script,
adapted to the wines domain and to this project's 100%-local constraint:

  RAGAS was considered but skipped: it depends on tiktoken, which has
  no Python 3.14 wheels yet, plus the full langchain stack — too many
  fragile transitive dependencies for this machine (already a source
  of prior installation issues here with fastembed/onnxruntime).

  LLM-judge backend (only mode — no Groq, no RAGAS fallback):
    Faithfulness      : is the answer supported by the retrieved sources?
    Answer Relevancy  : does the answer address the question asked?
  Both scored 0.0-1.0 by a LOCAL judge model (llama3.2:1b via Ollama,
  same model as generation) using structured JSON output.

  Domain note: NIS2's Citation Accuracy (regex on cited "Art. X") and
  Refusal Accuracy (does the model correctly refuse out-of-scope
  compliance questions) are deliberately NOT reused here — both are
  specific to a legal/compliance RAG and have no equivalent concept in
  a wine catalogue.

Retrieval metrics (mathematical — exact, no LLM needed)
─────────────────────────────────────────────────────────
  Hit@K, MRR

Requirements:
    pip install httpx ollama rich

API must be running : uvicorn api.main:app --port 8000

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import ollama
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR        = Path(__file__).parent
API_BASE           = "http://localhost:8000"
DATASET_PATH         = SCRIPT_DIR / "golden_dataset.json"
DEFAULT_OUTPUT         = SCRIPT_DIR / "output" / "evaluation_report.json"

REQUEST_TIMEOUT      = 600  # generous: CPU-only Ollama (llama3.2:1b on 8GB RAM) can be slow
DEFAULT_K              = 5

JUDGE_MODEL_NAME         = "llama3.2:1b"  # same local model as generation, 100% local

# ── Thresholds ────────────────────────────────────────────────────
# Deliberately more moderate than NIS2's (90%/80%-class targets):
# this pipeline runs llama3.2:1b on an 8GB-RAM CPU-only machine, a
# known, accepted hardware constraint — these thresholds reflect a
# realistic bar for that setup rather than a generic RAG standard.
THRESHOLD_HIT_AT_K              = 0.60
THRESHOLD_MRR                    = 0.50
THRESHOLD_FAITHFULNESS            = 0.50
THRESHOLD_ANSWER_RELEVANCY         = 0.50


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

def load_dataset(path: Path, lang_filter: Optional[str] = None, id_filter: Optional[str] = None) -> list[dict]:
    if not path.exists():
        console.print(f"[red]ERROR: golden dataset not found: {path}[/red]")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        console.print(f"[red]ERROR: expected a JSON list in {path}[/red]")
        sys.exit(1)
    if lang_filter:
        data = [s for s in data if s.get("language") == lang_filter]
    if id_filter:
        data = [s for s in data if s.get("id") == id_filter]
    return data


# ══════════════════════════════════════════════════════════════════
# API CALLS
# ══════════════════════════════════════════════════════════════════

def call_retrieve(client: httpx.Client, entry: dict, k: int) -> dict:
    payload = {"query": entry["query"], "limit": k, **entry.get("filters", {})}
    response = client.post(f"{API_BASE}/retrieve", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def call_ask(client: httpx.Client, entry: dict, k: int) -> dict:
    payload = {"query": entry["query"], "limit": k, **entry.get("filters", {})}
    response = client.post(f"{API_BASE}/ask", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ══════════════════════════════════════════════════════════════════
# 1 — RETRIEVAL METRICS  (mathematical — exact)
# ══════════════════════════════════════════════════════════════════

def compute_hit_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> bool:
    """True if at least one expected chunk appears anywhere in the retrieved list."""
    expected_set = set(expected_chunk_ids)
    return any(cid in expected_set for cid in retrieved_chunk_ids)


def compute_reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """1 / rank of the first retrieved chunk in the expected set (0.0 if none found)."""
    expected_set = set(expected_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in expected_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(samples: list[dict], k: int) -> dict:
    console.print("\n[bold cyan]━━━ 1. RETRIEVAL METRICS (exact) ━━━[/bold cyan]")

    eval_s = [s for s in samples if s.get("expected_chunk_ids")]
    skipped = len(samples) - len(eval_s)
    console.print(f"  Samples with ground truth : {len(eval_s)}"
                  + (f"  [dim]({skipped} skipped, no expected_chunk_ids)[/dim]" if skipped else ""))

    hits, rrs, per_sample = [], [], []

    with httpx.Client() as client:
        for s in eval_s:
            try:
                response = call_retrieve(client, s, k)
                retrieved = [r["chunk_id"] for r in response.get("results", [])]
            except Exception as exc:
                console.print(f"  [red]ERROR[/red] {s['id']}: {exc}")
                continue

            expected = s["expected_chunk_ids"]
            hit = compute_hit_at_k(retrieved, expected)
            rr  = compute_reciprocal_rank(retrieved, expected)

            hits.append(hit)
            rrs.append(rr)

            console.print(
                f"  [dim]{s['id']:10}[/dim]  {s['language'].upper()}  "
                f"Hit@{k}={'[green]✓[/green]' if hit else '[red]✗[/red]'}  "
                f"RR={rr:.2f}  top3={retrieved[:3]}"
            )
            per_sample.append({
                "id": s["id"], "language": s["language"],
                f"hit@{k}": hit, "reciprocal_rank": round(rr, 3),
                "retrieved_chunk_ids": retrieved, "expected_chunk_ids": expected,
            })

    n = len(hits) or 1
    metrics = {
        f"hit@{k}":     round(sum(hits) / n, 3) if hits else None,
        "mrr":            round(sum(rrs) / n, 3) if rrs else None,
        "n_samples":        len(hits),
        "per_sample":         per_sample,
    }

    table = Table(title="Retrieval Metrics (exact)", box=box.ROUNDED)
    table.add_column("Metric",    style="cyan", min_width=16)
    table.add_column("Score",     justify="right")
    table.add_column("Threshold", style="dim", justify="right")
    table.add_column("Status",    justify="center")
    for name, score, thr, lbl in [
        (f"Hit@{k}", metrics[f"hit@{k}"], THRESHOLD_HIT_AT_K, f"≥ {THRESHOLD_HIT_AT_K:.0%}"),
        ("MRR",      metrics["mrr"],      THRESHOLD_MRR,      f"≥ {THRESHOLD_MRR}"),
    ]:
        if score is None:
            table.add_row(name, "N/A", lbl, "⬜")
            continue
        table.add_row(name, f"{score:.3f}", lbl,
                      "[green]✓ PASS[/green]" if score >= thr else "[red]✗ FAIL[/red]")
    console.print(table)
    return metrics


# ══════════════════════════════════════════════════════════════════
# 2 — GENERATION METRICS  (LLM-judge — local llama3.2:1b via Ollama)
# ══════════════════════════════════════════════════════════════════

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score":         {"type": "number"},
        "justification": {"type": "string"},
    },
    "required": ["score", "justification"],
}


def _run_judge(prompt: str) -> dict:
    """
    Shared call to the local judge model with structured JSON output.
    Returns {"score": float, "justification": str}. On any failure
    (model error, malformed JSON, missing fields), returns score=None
    with the error as justification — callers must treat None as "could
    not be judged", never silently as 0.
    """
    try:
        response = ollama.chat(
            model=JUDGE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=JUDGE_SCHEMA,
        )
        message = response.get("message") if isinstance(response, dict) else response.message
        content = message.get("content") if isinstance(message, dict) else message.content
        parsed = json.loads(content)
        score = max(0.0, min(1.0, float(parsed["score"])))  # clamp, judges occasionally drift outside [0,1]
        return {"score": score, "justification": parsed.get("justification", "")}
    except Exception as e:
        return {"score": None, "justification": f"Judge call failed: {e}"}


def judge_faithfulness(question: str, answer: str, sources_text: str) -> dict:
    """Is `answer` actually supported by `sources_text`, or does it hallucinate?"""
    prompt = (
        "You are evaluating a RAG system's answer for faithfulness to its sources.\n"
        "Score from 0.0 to 1.0 how well the ANSWER is supported by the SOURCES — "
        "1.0 means every claim in the answer is grounded in the sources, "
        "0.0 means the answer states things that contradict or are absent from the sources.\n"
        "Respond ONLY with JSON: {\"score\": <float 0-1>, \"justification\": \"<short reason>\"}\n\n"
        f"QUESTION:\n{question}\n\nSOURCES:\n{sources_text}\n\nANSWER:\n{answer}"
    )
    return _run_judge(prompt)


def judge_answer_relevancy(question: str, answer: str) -> dict:
    """Does `answer` actually address `question`, regardless of factual correctness?"""
    prompt = (
        "You are evaluating a RAG system's answer for relevancy to the question asked.\n"
        "Score from 0.0 to 1.0 how directly the ANSWER addresses the QUESTION — "
        "1.0 means it directly and completely answers what was asked, "
        "0.0 means it's off-topic or doesn't address the question at all. "
        "Do NOT judge factual correctness here, only relevancy.\n"
        "Respond ONLY with JSON: {\"score\": <float 0-1>, \"justification\": \"<short reason>\"}\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    )
    return _run_judge(prompt)


def build_sources_text(sources: list[dict]) -> str:
    """Flatten the /ask response's `sources` list into plain text for the judge prompt."""
    parts = []
    for s in sources:
        label = s.get("title") or s.get("wine_name") or s.get("chunk_id")
        parts.append(f"[{label}]\n{s.get('text', '')}")
    return "\n\n".join(parts)


def evaluate_generation(samples: list[dict], k: int) -> dict:
    console.print(f"\n[bold cyan]━━━ 2. GENERATION METRICS (LLM-judge — {JUDGE_MODEL_NAME}, local) ━━━[/bold cyan]")

    faith_s, relev_s, per_sample = [], [], []

    with httpx.Client() as client:
        for s in samples:
            try:
                ask_response = call_ask(client, s, k)
                answer  = ask_response.get("answer", "")
                sources = ask_response.get("sources", [])
            except Exception as exc:
                console.print(f"  [red]ERROR[/red] {s['id']}: {exc}")
                continue

            if not sources:
                console.print(f"  [yellow]SKIP[/yellow] {s['id']}: no sources retrieved, nothing to judge")
                continue

            sources_text = build_sources_text(sources)
            faith = judge_faithfulness(s["query"], answer, sources_text)
            relev = judge_answer_relevancy(s["query"], answer)

            f_score, r_score = faith["score"], relev["score"]
            if f_score is not None:
                faith_s.append(f_score)
            if r_score is not None:
                relev_s.append(r_score)

            f_col = "green" if (f_score is not None and f_score >= THRESHOLD_FAITHFULNESS) else "red"
            r_col = "green" if (r_score is not None and r_score >= THRESHOLD_ANSWER_RELEVANCY) else "red"
            f_disp = f"{f_score:.2f}" if f_score is not None else "N/A"
            r_disp = f"{r_score:.2f}" if r_score is not None else "N/A"

            console.print(
                f"  [dim]{s['id']:10}[/dim]  {s['language'].upper()}  "
                f"faith=[{f_col}]{f_disp}[/{f_col}]  rel=[{r_col}]{r_disp}[/{r_col}]"
            )
            per_sample.append({
                "id": s["id"], "language": s["language"],
                "faithfulness": faith, "answer_relevancy": relev,
                "answer_preview": answer[:200],
            })

    def avg(values: list[float]) -> Optional[float]:
        return round(sum(values) / len(values), 3) if values else None

    metrics = {
        "faithfulness":      avg(faith_s),
        "answer_relevancy":   avg(relev_s),
        "n_judged_faithfulness": len(faith_s),
        "n_judged_relevancy":     len(relev_s),
        "per_sample":               per_sample,
    }

    table = Table(title="Generation Metrics (LLM-judge, local)", box=box.ROUNDED)
    table.add_column("Metric",          style="cyan", min_width=18)
    table.add_column("Score",           justify="right")
    table.add_column("Threshold",       style="dim", justify="right")
    table.add_column("Status",          justify="center")
    table.add_column("Description",     style="dim")
    for name, score, thr, desc in [
        ("Faithfulness",     metrics["faithfulness"],     THRESHOLD_FAITHFULNESS,     "Claims supported by sources"),
        ("Answer Relevancy", metrics["answer_relevancy"], THRESHOLD_ANSWER_RELEVANCY, "Response addresses the question"),
    ]:
        if score is None:
            table.add_row(name, "N/A", f"≥ {thr}", "⬜", desc)
            continue
        table.add_row(name, f"{score:.3f}", f"≥ {thr}",
                      "[green]✓ PASS[/green]" if score >= thr else "[red]✗ FAIL[/red]", desc)
    console.print(table)
    return metrics


# ══════════════════════════════════════════════════════════════════
# 3 — END-TO-END SUMMARY
# ══════════════════════════════════════════════════════════════════

def print_summary(retrieval: dict, generation: dict, k: int) -> dict:
    console.print("\n[bold cyan]━━━ 3. END-TO-END SUMMARY ━━━[/bold cyan]")

    table = Table(title=f"Wines RAG Pipeline — Full Evaluation  [judge: {JUDGE_MODEL_NAME}, local]",
                  box=box.DOUBLE_EDGE)
    table.add_column("Layer",     style="bold cyan", min_width=18)
    table.add_column("Metric",    style="white",     min_width=20)
    table.add_column("Score",     justify="right")
    table.add_column("Method",    style="dim",       justify="center")
    table.add_column("Status",    justify="center")

    def row(layer, metric, score, thr, method):
        if score is None:
            table.add_row(layer, metric, "N/A", method, "⬜")
            return
        ok = score >= thr
        table.add_row(layer, metric, f"{score:.3f}", method,
                      "[green]✓ PASS[/green]" if ok else "[red]✗ FAIL[/red]")

    row("Retrieval",  f"Hit@{k}",           retrieval.get(f"hit@{k}"),         THRESHOLD_HIT_AT_K,         "exact")
    row("Retrieval",  "MRR",                retrieval.get("mrr"),              THRESHOLD_MRR,              "exact")
    row("Generation", "Faithfulness",       generation.get("faithfulness"),    THRESHOLD_FAITHFULNESS,     "LLM-judge")
    row("Generation", "Answer Relevancy",   generation.get("answer_relevancy"),THRESHOLD_ANSWER_RELEVANCY, "LLM-judge")

    console.print(table)

    return {
        "retrieval":  retrieval,
        "generation": generation,
        "k":           k,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Wines RAG — Evaluation")
    parser.add_argument("--golden",  default=str(DATASET_PATH), help="Path to golden_dataset.json")
    parser.add_argument("--mode",    choices=["all", "retrieval", "generation"], default="all")
    parser.add_argument("--lang",    choices=["it", "en", "fr"], default=None)
    parser.add_argument("--id",      default=None)
    parser.add_argument("--k",       type=int, default=DEFAULT_K)
    parser.add_argument("--output",  default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    console.print("\n[bold white]╔══════════════════════════════════════════════╗[/bold white]")
    console.print("[bold white]║   WINES RAG — EVALUATION                     ║[/bold white]")
    console.print("[bold white]╚══════════════════════════════════════════════╝[/bold white]")

    console.print(f"\n  Judge : [bold yellow]{JUDGE_MODEL_NAME}[/bold yellow] (Ollama, local — no Groq, no RAGAS)")

    try:
        h = httpx.get(f"{API_BASE}/health", timeout=10).json()
        color = "green" if h.get("status") == "ok" else "yellow"
        console.print(f"  API   : [{color}]{h.get('status')}[/{color}]  |  "
                      f"Collection: {h.get('collection')}  |  Points: {h.get('points')}")
    except Exception as exc:
        console.print(f"[red]Cannot reach API: {exc}[/red]")
        console.print("  → uvicorn api.main:app --port 8000")
        return

    samples = load_dataset(Path(args.golden), args.lang, args.id)
    console.print(f"  Dataset : {len(samples)} samples\n")

    retrieval_r, generation_r = {}, {}

    if args.mode in ("all", "retrieval"):
        retrieval_r = evaluate_retrieval(samples, args.k)

    if args.mode in ("all", "generation"):
        generation_r = evaluate_generation(samples, args.k)

    if args.mode == "all":
        summary = print_summary(retrieval_r, generation_r, args.k)
    else:
        summary = {
            "retrieval":  retrieval_r,
            "generation": generation_r,
            "k":           args.k,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        console.print(f"\n  [green]Saved → {out}[/green]")

    console.print("\n[bold green]Evaluation complete.[/bold green]\n")


if __name__ == "__main__":
    main()