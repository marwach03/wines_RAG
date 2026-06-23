"""
╔══════════════════════════════════════════════════════════════════╗
║  STILEMA CHUNKING PIPELINE                                       ║
║  Input  : stilema_it.pdf  (77 pages, single-page, ~454pts wide) ║
║  Output : output/stilema_chunks.json                            ║
╚══════════════════════════════════════════════════════════════════╝

Category read strategy:
  Each wine sheet's SECOND page has the category label printed in the
  top-right corner (e.g. "ICON", "CRU", "HERITAGE").
  Additionally, standalone marker pages (a single word centred, ≤ 6 spans)
  precede each category group. Both signals are used: the marker page sets
  the tracker, the per-page label confirms/overrides it.

Wine sheet structure (2 pages per wine):
  Page A (odd internal index after marker):
    - Wine name : large font (~27pt), non-bold, top-left
    - Subtitle  : medium font (~14pt), below name
    - Intro paragraph : body text
    - PROFILO DEL VINO section with labelled fields

  Page B (even internal index):
    - CARATTERI SENSORIALI section
    - Category label : top-right corner, ~10pt, non-bold (e.g. "ICON")
    - "Raccontato da Piero Mastroberardino" : QR code caption, skip
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF


# ─── PATHS ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DEFAULT_PDF = SCRIPT_DIR / "data" / "stilema_it.pdf"
OUTPUT_DIR  = SCRIPT_DIR / "output"

# ─── CATEGORY LABELS ──────────────────────────────────────────────────────────
# These appear both as standalone marker pages and in top-right of page B
CATEGORY_LABELS = {
    "icon", "cru", "heritage", "smart",
    "passiti", "passiti da muffa nobile", "olio evo bio",
}

# Short aliases for metadata
CATEGORY_CANONICAL = {
    "icon":                    "ICON",
    "cru":                     "CRU",
    "heritage":                "HERITAGE",
    "smart":                   "SMART",
    "passiti":                 "PASSITI",
    "passiti da muffa nobile": "PASSITI",
    "olio evo bio":            "OLIO EVO BIO",
}

# ─── FONT SIZE THRESHOLDS ─────────────────────────────────────────────────────
SZ_WINE_NAME   = 20.0   # Large wine name title  (e.g. "STILÈMA" at ~27pt,
                         #                             "RADICI" at ~27pt)
SZ_WINE_SUB    = 12.0   # Subtitle denomination  (e.g. "FIANO DI AVELLINO DOCG")
SZ_BODY        = 8.5    # Normal body text

# ─── NOISE PATTERNS ───────────────────────────────────────────────────────────
# Spans to ignore unconditionally
NOISE_RE = re.compile(
    r"^(raccontato da piero|mastroberardino\.com|"
    r"©\s*mastroberardino|page\s*\d+|\d+\s*/\s*\d+|"
    r"incoming@|wine\s*shop|public\s*relations|enoteca@|pr@|"
    r"via\s+re\s+manfredi|atripalda|www\.|"
    r"ministero|politiche\s+agricole|consiglio\s+per|"
    r"catalogoviti\.|sperimentazione)$",
    re.IGNORECASE,
)

# ─── TEXT HELPERS ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", s).strip()


def clean(text: str) -> str:
    """Collapse whitespace only."""
    return re.sub(r"\s+", " ", text).strip()


def is_noise(text: str) -> bool:
    return bool(NOISE_RE.match(text.strip()))


def is_category_label(text: str) -> bool:
    return normalize(text) in CATEGORY_LABELS


def slug(text: str) -> str:
    s = normalize(text)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def guess_wine_type(wine_name: str, subtitle: str) -> str:
    """
    Infer wine colour from name + denomination.
    Defaults to 'red' for Aglianico/Taurasi, 'white' otherwise,
    'rosato' for rosato denominations, 'passito' for passiti,
    'oil' for olive oil.
    """
    combined = normalize(wine_name + " " + subtitle)
    if "olio" in combined or "evo" in combined:
        return "oil"
    if "passito" in combined:
        return "passito"
    if "rosato" in combined:
        return "rosato"
    if any(k in combined for k in ("taurasi", "aglianico", "rosso")):
        return "red"
    return "white"


# ─── SPAN EXTRACTION ──────────────────────────────────────────────────────────

def extract_spans(page: fitz.Page) -> list[dict]:
    """
    Return all text spans from a page, sorted top-to-bottom then left-to-right.
    Each span: {text, x0, y0, x1, sz, bold}
    """
    spans = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                t = clean(s.get("text", ""))
                if not t:
                    continue
                spans.append({
                    "text": t,
                    "x0":   round(s["origin"][0]),
                    "y0":   round(s["origin"][1]),
                    "x1":   round(s["bbox"][2]),
                    "sz":   round(s["size"], 1),
                    "bold": bool(s["flags"] & 2**4),
                })
    spans.sort(key=lambda s: (s["y0"], s["x0"]))
    return spans


# ─── PAGE CLASSIFIER ──────────────────────────────────────────────────────────

def classify_page(spans: list[dict], page_width: float) -> str:
    """
    Classify a single page into one of:
      'empty'           — no text (photo / blank)
      'cover'           — cover / back cover
      'category_marker' — standalone category label page
      'history'         — narrative text (Progetto Stilèma, Le scelte, etc.)
      'wine_a'          — first page of a wine sheet (name + PROFILO)
      'wine_b'          — second page of a wine sheet (CARATTERI SENSORIALI)
    """
    if not spans:
        return "empty"

    texts_norm = {normalize(s["text"]) for s in spans}
    max_sz     = max(s["sz"] for s in spans)

    # ── Category marker page ─────────────────────────────────────────────────
    # A page whose only meaningful content is a single category label.
    # Heuristic: ≤ 8 spans, at least one matches a category label.
    meaningful = [s for s in spans if not is_noise(s["text"])]
    if len(meaningful) <= 8:
        cat_spans = [s for s in meaningful if is_category_label(s["text"])]
        if cat_spans:
            return "category_marker"

    # ── Cover / back cover ───────────────────────────────────────────────────
    # The front cover has "I VINI" and "mastroberardino.com" but no wine data.
    if "i vini" in texts_norm and len(meaningful) <= 10:
        return "cover"
    if len(meaningful) <= 4:
        return "empty"

    # ── Wine sheet page B (CARATTERI SENSORIALI) ─────────────────────────────
    if "caratteri sensoriali" in texts_norm:
        return "wine_b"

    # ── Wine sheet page A (PROFILO DEL VINO or PROFILO DEL PASSITO) ──────────
    profilo_labels = {"profilo del vino", "profilo del passito", "profilo dell olio"}
    if texts_norm & profilo_labels:
        return "wine_a"

    # ── History / narrative ───────────────────────────────────────────────────
    # Large-font section title (> 20pt) and substantial body text.
    if max_sz >= 20.0:
        body_spans = [s for s in spans if s["sz"] < 15.0 and not is_noise(s["text"])]
        if len(body_spans) >= 5:
            return "history"

    # ── Fallback: treat as history if enough body text ───────────────────────
    body_spans = [s for s in spans if s["sz"] < 15.0 and not is_noise(s["text"])]
    if len(body_spans) >= 8:
        return "history"

    return "empty"


# ─── CATEGORY EXTRACTION FROM PAGE ───────────────────────────────────────────

def extract_category_from_page(spans: list[dict], page_width: float) -> str | None:
    """
    Read the category label from the top-right corner of a wine_b page,
    or from a category_marker page.
    Returns the canonical category string, or None if not found.
    """
    for s in spans:
        if is_category_label(s["text"]):
            norm = normalize(s["text"])
            return CATEGORY_CANONICAL.get(norm)
    return None


# ─── HISTORY PARSER ───────────────────────────────────────────────────────────

def parse_history_pages(pages_spans: list[list[dict]], page_width: float) -> list[dict]:
    """
    Merge consecutive history pages into section-based chunks.

    Two-column layout: left column (x0 < mid_x) is read entirely before
    right column (x0 >= mid_x) to preserve natural reading order.

    A new chunk begins when a large title span (sz >= 20pt) is encountered.
    """
    mid_x = page_width / 2
    chunks: list[dict] = []

    current_title = "Il Progetto Stilèma"
    current_lines: list[str] = []
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx, current_lines
        text = " ".join(current_lines).strip()
        if len(text) > 80:
            chunk_idx += 1
            chunks.append({
                "chunk_id":   f"it_stilema_history_{chunk_idx:02d}",
                "chunk_type": "history",
                "language":   "it",
                "title":      current_title,
                "text":       f"[{current_title}]\n\n{text}",
                "wine_name":  None,
                "vintage":    None,
                "category":   "history",
                "wine_type":  None,
                "project":    "stilema",
                "fields":     {},
            })
        current_lines.clear()

    for spans in pages_spans:
        left  = sorted([s for s in spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
        right = sorted([s for s in spans if s["x0"] >= mid_x], key=lambda s: s["y0"])

        for s in (left + right):
            t = s["text"]
            if is_noise(t) or is_category_label(t):
                continue
            if s["sz"] >= 20.0 and not s["bold"]:
                # Large section title → new chunk boundary
                flush()
                current_title = t
            elif s["sz"] >= 12.0 and s["bold"]:
                # Sub-section title (e.g. "Le scelte in vigna") → new chunk
                flush()
                current_title = t
            elif s["sz"] >= 10.0 and not s["bold"] and len(t) > 30:
                # Italic/large drop-cap or pull quote → keep as body
                current_lines.append(t)
            elif s["sz"] >= SZ_BODY:
                current_lines.append(t)

    flush()
    return chunks


# ─── WINE SHEET PARSER ────────────────────────────────────────────────────────

def parse_wine_pair(
    page_a_spans: list[dict],
    page_b_spans: list[dict],
    page_width:   float,
    category:     str,
    chunk_idx:    int,
) -> dict:
    """
    Build one chunk from the two pages of a wine sheet.

    Page A provides:
      - wine_name  (largest font span, typically sz ~27pt)
      - subtitle   (second large span, sz ~14pt)
      - intro      (body paragraph before PROFILO)
      - profilo    (labelled technical fields)

    Page B provides:
      - caratteri sensoriali (colour, aroma, taste)
      - food pairings
      - ageing potential
      - serving temperature
      - category label (top-right, used to confirm/override tracker)

    The full text is assembled as:
      [Wine Name Subtitle]
      <intro>
      <profilo fields>
      <caratteri sensoriali>
      <pairings / serving>
    """
    mid_x = page_width / 2

    # ── 1. Extract wine name + subtitle from page A ───────────────────────────
    # The wine name is the span with the largest font size on page A.
    # The subtitle is the next largest that is NOT the same span.
    meaningful_a = [s for s in page_a_spans
                    if not is_noise(s["text"]) and not is_category_label(s["text"])
                    and s["sz"] >= SZ_WINE_SUB]

    # Sort by font size descending to find title candidates
    by_size = sorted(meaningful_a, key=lambda s: -s["sz"])

    wine_name = ""
    subtitle  = ""

    if by_size:
        wine_name = by_size[0]["text"]

    # Subtitle: next large span that is on the same or next y-row and
    # clearly a denomination (contains DOCG/DOC/IGT/DOP or is directly below)
    for s in by_size[1:]:
        if s["text"] != wine_name and s["sz"] >= SZ_WINE_SUB:
            subtitle = s["text"]
            break

    # ── 2. Collect body text from page A (intro + profilo fields) ────────────
    # Read left column first, then right column for 2-column layout.
    left_a  = sorted([s for s in page_a_spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
    right_a = sorted([s for s in page_a_spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
    ordered_a = left_a + right_a

    body_a_lines: list[str] = []
    in_profilo   = False
    skip_set     = {normalize(wine_name), normalize(subtitle)}

    for s in ordered_a:
        t    = s["text"]
        norm = normalize(t)

        if is_noise(t) or is_category_label(t):
            continue
        if norm in skip_set:
            continue
        if norm in ("profilo del vino", "profilo del passito", "profilo dell olio"):
            in_profilo = True
            continue  # skip the section header itself
        if s["sz"] < 6.5:
            continue  # micro bar-chart labels

        body_a_lines.append(t)

    # ── 3. Collect body text from page B (caratteri sensoriali + pairings) ────
    # Also extract category label from top-right corner.
    cat_from_page_b = extract_category_from_page(page_b_spans, page_width)
    if cat_from_page_b:
        category = cat_from_page_b  # page-level label takes priority

    left_b  = sorted([s for s in page_b_spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
    right_b = sorted([s for s in page_b_spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
    ordered_b = left_b + right_b

    body_b_lines: list[str] = []
    for s in ordered_b:
        t    = s["text"]
        norm = normalize(t)

        if is_noise(t) or is_category_label(t):
            continue
        if norm == "caratteri sensoriali":
            continue  # skip section header
        if s["sz"] < 6.5:
            continue

        body_b_lines.append(t)

    # ── 4. Assemble full text ─────────────────────────────────────────────────
    wine_full = f"{wine_name} {subtitle}".strip() if subtitle else wine_name
    header    = f"[{wine_full}]"
    body      = " ".join(body_a_lines + body_b_lines).strip()
    full_text = f"{header}\n\n{body}"

    # ── 5. Extract vintage if present ─────────────────────────────────────────
    vintage: int | None = None
    year_match = re.search(r"\b(19|20)\d{2}\b", wine_full + " " + body[:200])
    if year_match:
        vintage = int(year_match.group())

    return {
        "chunk_id":   f"it_wine_{chunk_idx:03d}_{slug(wine_full)}",
        "chunk_type": "wine_sheet",
        "language":   "it",
        "title":      wine_full,
        "text":       full_text,
        "wine_name":  wine_name,
        "vintage":    vintage,
        "category":   category,
        "wine_type":  guess_wine_type(wine_name, subtitle),
        "project":    "stilema",
        "fields":     {},
    }


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(pdf_path: Path, output_file: Path) -> list[dict]:
    print(f"\n[1/4] Opening {pdf_path.name}...")
    doc        = fitz.open(str(pdf_path))
    n          = len(doc)
    page_width = doc[0].rect.width
    print(f"  {n} pages, width={page_width:.0f}pts")

    # Extract all spans
    all_spans = [extract_spans(doc[i]) for i in range(n)]
    doc.close()

    # Classify every page
    classifications = [classify_page(all_spans[i], page_width) for i in range(n)]

    print("\n[2/4] Page classifications:")
    for i, cls in enumerate(classifications):
        if cls not in ("empty",):
            print(f"  p{i+1:02d} → {cls}")

    # ── Build page groups ─────────────────────────────────────────────────────
    # Walk pages in order; track current category; pair wine_a + wine_b pages.
    chunks:           list[dict] = []
    history_buffer:   list[list[dict]] = []
    current_category: str  = "ICON"  # default before any marker
    wine_a_pending:   list[dict] | None = None
    wine_chunk_idx    = 0

    def flush_history():
        nonlocal history_buffer
        if history_buffer:
            h = parse_history_pages(history_buffer, page_width)
            chunks.extend(h)
            print(f"  → {len(h)} history chunks")
        history_buffer = []

    for i, cls in enumerate(classifications):
        spans = all_spans[i]

        if cls == "empty" or cls == "cover":
            # Flush any pending history before skipping
            flush_history()
            wine_a_pending = None
            continue

        elif cls == "category_marker":
            flush_history()
            wine_a_pending = None
            cat = extract_category_from_page(spans, page_width)
            if cat:
                current_category = cat
                print(f"  p{i+1:02d} → category marker: {current_category}")

        elif cls == "history":
            # Flush pending wine_a if orphaned
            if wine_a_pending is not None:
                print(f"  WARN p{i+1}: found history while wine_a pending — dropping wine_a")
                wine_a_pending = None
            history_buffer.append(spans)

        elif cls == "wine_a":
            flush_history()
            # If previous wine_a was orphaned (no page B), drop it
            if wine_a_pending is not None:
                print(f"  WARN p{i+1}: consecutive wine_a pages — dropping previous")
            wine_a_pending = spans

        elif cls == "wine_b":
            flush_history()
            if wine_a_pending is None:
                # wine_b without wine_a: try to build a partial chunk from page B only
                print(f"  WARN p{i+1}: wine_b without preceding wine_a — building partial chunk")
                wine_a_pending = []  # empty page A
            wine_chunk_idx += 1
            chunk = parse_wine_pair(
                wine_a_pending, spans,
                page_width, current_category, wine_chunk_idx,
            )
            chunks.append(chunk)
            wine_a_pending = None

    # Flush anything remaining
    flush_history()
    if wine_a_pending is not None:
        print("  WARN: trailing wine_a with no page B — discarded")

    # ── Deduplicate by chunk_id ───────────────────────────────────────────────
    seen:   set[str]   = set()
    unique: list[dict] = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    print(f"\n[3/4] Total chunks: {len(unique)}")
    types = {}
    cats  = {}
    for c in unique:
        types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
        cat = c.get("category") or "-"
        cats[cat] = cats.get(cat, 0) + 1
    print(f"  By type     : {types}")
    print(f"  By category : {cats}")

    print("\n[4/4] Writing output...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"  → {output_file}")

    print("\nChunk IDs:")
    for c in unique:
        print(f"  {c['chunk_id']}")

    return unique


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Stilema chunking pipeline")
    p.add_argument("--pdf",    default=str(DEFAULT_PDF),
                   help="Path to stilema_it.pdf")
    p.add_argument("--output", default=str(OUTPUT_DIR / "stilema_chunks.json"),
                   help="Output JSON path")
    p.add_argument("--debug",  action="store_true",
                   help="Print span-level details for each page and exit")
    p.add_argument("--page",   type=int, default=None,
                   help="Single page number to inspect with --debug (1-based)")
    args = p.parse_args()

    if args.debug:
        doc = fitz.open(args.pdf)
        pw  = doc[0].rect.width
        pages = [args.page - 1] if args.page else range(min(10, len(doc)))
        for i in pages:
            if i >= len(doc):
                continue
            spans = extract_spans(doc[i])
            cls   = classify_page(spans, pw)
            print(f"\n== PAGE {i+1} [{cls}] ==")
            print(f"  {'x0':>5} {'y0':>5} {'sz':>5}  B  text")
            print("  " + "-" * 65)
            for s in spans[:60]:
                b = "B" if s["bold"] else " "
                print(f"  {s['x0']:>5} {s['y0']:>5} {s['sz']:>5.1f}  {b}  {s['text'][:60]}")
        doc.close()
        return

    run(Path(args.pdf), Path(args.output))


if __name__ == "__main__":
    main()