"""
╔══════════════════════════════════════════════════════════════════╗
║  WINES CHUNKING PIPELINE — Mastroberardino                       ║
║  Input  : mastro_it.pdf (28 pages, single)                       ║
║           mastro_en.pdf (15 pages, double logical pages)         ║
║  Output : output/wines_chunks_all.json                           ║
╚══════════════════════════════════════════════════════════════════╝

Chunks produced:
  - 1 chunk  "chronology"
  - N chunks "history"
  - 1 chunk per wine sheet (title + vintage + harvest description)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / "output"

# Font size thresholds — detected via --debug mode on the actual PDFs
SZ_COVER_TITLE   = 20.0   
SZ_SECTION_TITLE = 30.0   
SZ_WINE_TITLE    = 12.0   
SZ_VENDEMMIA     = 9.5    
SZ_CHRONO        = 7.5    

# Weather / climate block labels (normalised, no diacritics) — IT + EN
# These appear as bold labels inside a visual infographic block on each wine
# sheet page and must be excluded from all chunk text
METEO_LABELS_NORM = {
    # Italian
    "piovosita", "temperature medie", "periodo di vendemmia",
    "bassa", "elevata", "precoce", "tardivo",
    # English
    "rainfall", "average temperatures", "harvest period",
    "low", "heavy", "high", "early", "late",
}

# Navigation / footer patterns — page numbers, restaurant names, URLs
NAV_PATTERNS = re.compile(
    r"^(taurasi\s+\d{4}[-–]\d{4}|a century of|mastroberardino\.com|"
    r"page\s*\d+|\d+\s*/\s*\d+|ai fiori|the langham)$",
    re.IGNORECASE,
)

# Width of one logical page in the English double-page PDF
HALF_PAGE_WIDTH = 539.0


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase + strip diacritics for label matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def clean(text: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def is_meteo(text: str) -> bool:
    """Return True if the span belongs to the weather infographic block."""
    return normalize(text) in METEO_LABELS_NORM


def is_nav(text: str) -> bool:
    """Return True if the span is a navigation label, footer, or page number."""
    return bool(NAV_PATTERNS.match(text.strip()))


def slug(text: str) -> str:
    """Build a URL-safe identifier string from a wine title."""
    s = normalize(text)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


# ─── SPAN EXTRACTION ──────────────────────────────────────────────────────────

def extract_spans(page: fitz.Page) -> list[dict]:
    """
    Extract all text spans from a PDF page with position and style metadata.
    Image blocks (type != 0) are automatically skipped.
    Returns spans sorted top-to-bottom, left-to-right.
    """
    spans = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = clean(span.get("text", ""))
                if not text:
                    continue
                bold = bool(span["flags"] & 2**4)
                spans.append({
                    "text": text,
                    "x0":   round(span["origin"][0]),
                    "y0":   round(span["origin"][1]),
                    "sz":   round(span["size"], 1),
                    "bold": bold,
                })
    spans.sort(key=lambda s: (s["y0"], s["x0"]))
    return spans


# ─── DOUBLE-PAGE HANDLING (mastro_en.pdf) ─────────────────────────────────────

def split_double_page(spans: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split a double-page scan into two independent logical pages.
    Left  page : spans with x0 <  539 (coordinates kept as-is)
    Right page : spans with x0 >= 539 (x0 remapped to x0 - 539)
    """
    left, right = [], []
    for s in spans:
        if s["x0"] < HALF_PAGE_WIDTH:
            left.append(s)
        else:
            # Remap x0 so the right page behaves like a standalone single page
            right.append({**s, "x0": s["x0"] - int(HALF_PAGE_WIDTH)})
    return left, right


def is_double_page(spans: list[dict]) -> bool:
    """
    Detect whether a physical page contains two logical pages side by side.
    Triggered when any span has x0 >= 539pts (right logical page boundary).
    """
    return any(s["x0"] >= HALF_PAGE_WIDTH for s in spans)


# ─── PAGE CLASSIFIER ──────────────────────────────────────────────────────────

def classify_page(spans: list[dict], lang: str) -> str:
    """
    Assign a structural type to each logical page before text extraction.

    Returns one of:
      'cover'      -- title page (large TAURASI font), skipped
      'empty'      -- no text spans (image-only page), skipped
      'chrono'     -- vintage chronology table (small bold spans, 3+ columns)
      'history'    -- narrative text (company history)
      'wine_sheet' -- structured technical sheet for one wine

    Detection is based purely on measurable font properties:
      - font size thresholds for title detection
      - bold flag + x-position distribution for chronology tables
      - harvest label presence for wine sheet confirmation
    """
    if not spans:
        return "empty"

    max_sz = max(s["sz"] for s in spans)
    texts  = {normalize(s["text"]) for s in spans}

    # Cover page: very large "TAURASI" title
    if max_sz >= SZ_COVER_TITLE and "taurasi" in texts:
        return "cover"

    # History / narrative: contains a very large section title (38pt+)
    if max_sz >= SZ_SECTION_TITLE:
        return "history"

    # Chronology table: many small bold spans distributed across 3+ x-columns
    # Each column is separated by more than 50pts
    small_bold = [s for s in spans if s["sz"] <= 8.5 and s["bold"]]
    if len(small_bold) >= 4:
        x_vals = sorted({s["x0"] for s in small_bold})
        cols = [x_vals[0]]
        for x in x_vals[1:]:
            if x - cols[-1] > 50:
                cols.append(x)
        if len(cols) >= 3:
            return "chrono"

    # Wine technical sheet: bold title at 14pt+ and harvest label present
    harvest_label  = "the harvest" if lang == "en" else "la vendemmia"
    has_wine_title = any(s["sz"] >= SZ_WINE_TITLE and s["bold"] for s in spans)
    has_harvest    = harvest_label in texts

    if has_wine_title and has_harvest:
        return "wine_sheet"
    # Some sheets span two pages; the second page has no harvest label
    if has_wine_title:
        return "wine_sheet"

    return "history"


# ─── CHRONOLOGY PARSER ────────────────────────────────────────────────────────

def parse_chrono_pages(pages_spans: list[list[dict]], lang: str) -> dict:
    """
    Merge all chronology pages into a single chunk.

    Spans are grouped by y0 proximity (within 5pts = same table row),
    then joined with a pipe separator to reconstruct the table structure.
    Navigation labels and weather labels are excluded.
    """
    lines = []
    for spans in pages_spans:
        rows: dict[int, list[str]] = {}
        for s in spans:
            key = s["y0"] // 5 * 5  # group spans on the same horizontal line
            rows.setdefault(key, []).append(s["text"])
        for y in sorted(rows):
            line = " | ".join(rows[y])
            if line and not is_meteo(line) and not is_nav(line):
                lines.append(line)

    text  = "\n".join(lines)
    title = ("Chronology Taurasi Mastroberardino" if lang == "en"
             else "Cronologia Taurasi Mastroberardino")
    label = ("[Chronology Taurasi Mastroberardino]" if lang == "en"
             else "[Cronologia dei vini Taurasi Mastroberardino]")
    return {
        "chunk_id":   f"{lang}_chrono_taurasi",
        "chunk_type": "chronology",
        "language":   lang,
        "title":      title,
        "text":       f"{label}\n\n{text}",
        "wine_name":  None,
        "vintage":    None,
        "category":   "chronology",
        "wine_type":  "red",
        "fields":     {},
    }


# ─── HISTORY PARSER ───────────────────────────────────────────────────────────

def parse_history_pages(pages_spans: list[list[dict]], page_width: float,
                        lang: str) -> list[dict]:
    """
    Parse narrative pages into section-based chunks.

    Two-column layout reading order:
      1. All spans with x0 < mid_x (left column), sorted top-to-bottom
      2. All spans with x0 >= mid_x (right column), sorted top-to-bottom
    This prevents the text from left and right columns from being interleaved.

    A new chunk is created at each section title:
      - Font size >= 30pt  → major title 
      - Font size >= 11pt bold → sub-section title
    """
    mid_x = page_width / 2
    chunks: list[dict] = []
    current_title = ("Mastroberardino History" if lang == "en"
                     else "Storia Mastroberardino")
    current_lines: list[str] = []
    chunk_idx = 0

    def flush():
        """Save the accumulated lines as a new chunk if content is long enough."""
        nonlocal chunk_idx, current_lines
        text = " ".join(current_lines).strip()
        if len(text) > 80:
            chunk_idx += 1
            chunks.append({
                "chunk_id":   f"{lang}_history_{chunk_idx:02d}",
                "chunk_type": "history",
                "language":   lang,
                "title":      current_title,
                "text":       f"[{current_title}]\n\n{text}",
                "wine_name":  None,
                "vintage":    None,
                "category":   "history",
                "wine_type":  None,
                "fields":     {},
            })
        current_lines.clear()

    for spans in pages_spans:
        # Read left column entirely before right column
        left  = sorted([s for s in spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
        right = sorted([s for s in spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
        ordered = left + right

        for s in ordered:
            text = s["text"]
            if is_meteo(text) or is_nav(text):
                continue
            if s["sz"] >= SZ_SECTION_TITLE:
                # Major section title → save current chunk, start a new one
                flush()
                current_title = text
            elif s["sz"] >= 11.0 and s["bold"]:
                # Sub-section title → same behaviour
                flush()
                current_title = text
            else:
                current_lines.append(text)

    flush()
    return chunks


# ─── WINE SHEET PARSER ────────────────────────────────────────────────────────

def parse_wine_sheet_pages(pages_spans: list[list[dict]], page_width: float,
                           lang: str) -> list[dict]:
    """
    Parse wine technical sheet pages into one chunk per wine.

    State machine (per span):
      1. Bold span >= 14pt, not a 4-digit year  → new wine title, flush previous
      2. Non-bold span >= 14pt matching ^\d{4}  → store as vintage year
      3. "La vendemmia" / "The harvest" label   → section marker, skip
      4. Bold label in meteo_section_labels     → enter weather block, skip
      5. Any span while in_meteo_block          → skip
      6. Spans < 6.5pt                          → skip (bar chart micro-labels)
      7. Everything else                        → append to body text

    The weather infographic block (Piovosità, Temperature medie, BASSA/ELEVATA)
    is excluded by detecting its opening bold label, then ignoring all content
    until a new wine title resets the in_meteo_block flag.
    """
    mid_x = page_width / 2
    chunks: list[dict] = []

    current_wine: str | None  = None
    current_year: str | None  = None
    current_lines: list[str]  = []
    chunk_idx = 0

    # Bold labels that mark the start of the weather infographic block
    meteo_section_labels = {
        "piovosita", "temperature medie", "periodo di vendemmia",
        "rainfall", "average temperatures", "harvest period",
    }
    harvest_labels = {"la vendemmia", "the harvest"}

    def flush():
        """Finalise the current wine chunk and reset the state machine."""
        nonlocal chunk_idx, current_wine, current_year, current_lines
        if not current_wine:
            current_lines = []
            return
        text = " ".join(current_lines).strip()
        if len(text) >= 30:
            chunk_idx += 1
            wine_full = (f"{current_wine} {current_year}".strip()
                         if current_year else current_wine)
            chunks.append({
                "chunk_id":   f"{lang}_wine_{chunk_idx:03d}_{slug(wine_full)}",
                "chunk_type": "wine_sheet",
                "language":   lang,
                "title":      wine_full,
                "text":       f"[{wine_full}]\n\n{text}",
                "wine_name":  current_wine,
                "vintage":    _extract_year(current_year),
                "category":   _guess_category(current_wine),
                "wine_type":  "red",
                "fields":     {},
            })
        current_lines = []
        current_wine  = None
        current_year  = None

    for spans in pages_spans:
        # Reset weather block flag at every new page boundary
        in_meteo_block = False

        # Read left column entirely before right column (two-column layout)
        left  = sorted([s for s in spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
        right = sorted([s for s in spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
        ordered = left + right

        for s in ordered:
            text = s["text"]
            norm = normalize(text)

            # Always skip navigation / footer spans
            if is_nav(text):
                continue

            # Detect start of weather infographic block (bold label ~10pt)
            if s["sz"] >= 9.5 and s["bold"] and norm in meteo_section_labels:
                in_meteo_block = True
                continue

            # Skip all content inside the weather block
            if in_meteo_block:
                continue

            # New wine title: bold span >= 14pt, not a bare 4-digit year
            if s["sz"] >= SZ_WINE_TITLE and s["bold"] and not re.match(r"^\d{4}$", text):
                if current_wine:
                    flush()
                current_wine   = text
                in_meteo_block = False
                continue

            # Vintage year: non-bold span >= 14pt matching a year pattern
            if (current_wine and not current_year
                    and s["sz"] >= SZ_WINE_TITLE
                    and re.match(r"^\d{4}", text)):
                current_year = text
                continue

            # Harvest section label → marks start of body text, skip the label itself
            if norm in harvest_labels:
                continue

            # Skip bar-chart scale micro-labels (< 6.5pt)
            if s["sz"] < 6.5:
                continue

            # Append to body text if we are inside a wine record
            if current_wine and text:
                current_lines.append(text)

    flush()
    return chunks


# ─── METADATA HELPERS ─────────────────────────────────────────────────────────

def _extract_year(s: str | None) -> int | None:
    """Extract a 4-digit year integer from a string, or return None."""
    if not s:
        return None
    m = re.search(r"\d{4}", s)
    return int(m.group()) if m else None


def _guess_category(wine_name: str) -> str:
    """
    Infer the Mastroberardino wine category from the wine name.
    Used for mastro_it and mastro_en only.
    """
    n = normalize(wine_name)
    if "radici" in n:
        return "Radici"
    if "centotrenta" in n or "130" in n:
        return "Centotrenta"
    if "fondatore" in n or "antonio" in n:
        return "Special Edition"
    return "Taurasi"


# ─── DEBUG MODE ───────────────────────────────────────────────────────────────

def debug_pdf(pdf_path: Path, page_limit: int = 6, single_page: int | None = None):
    """
    Print span details for a PDF to inspect font sizes and column positions.
    Run this first on any new PDF before writing or adjusting the chunking logic.

    Usage:
      python wines_chunking.py --debug --pdf data/mastro_it.pdf
      python wines_chunking.py --debug --pdf data/mastro_it.pdf --page 5
    """
    doc = fitz.open(str(pdf_path))
    pw  = doc[0].rect.width
    print(f"\n  PDF   : {pdf_path}")
    print(f"  Pages : {len(doc)}")
    print(f"  Width : {pw:.0f} pts  Height: {doc[0].rect.height:.0f} pts")
    print(f"  Mid-X : {pw/2:.0f} pts  (estimated column split)")

    pages = [single_page] if single_page is not None else range(min(page_limit, len(doc)))
    for i in pages:
        if i >= len(doc):
            continue
        page   = doc[i]
        spans  = extract_spans(page)
        double = is_double_page(spans)
        cls    = classify_page(spans, lang="?")
        print(f"\n  == PAGE {i+1} [{cls}]{'  [DOUBLE]' if double else ''} ==")
        print(f"  {'x0':>6} {'y0':>6} {'sz':>5}  B  text")
        print("  " + "-" * 70)
        for s in spans[:50]:
            b = "B" if s["bold"] else " "
            print(f"  {s['x0']:>6} {s['y0']:>6} {s['sz']:>5.1f}  {b}  {s['text'][:60]}")
    doc.close()


# ─── SINGLE PDF RUNNER ────────────────────────────────────────────────────────

def run_single_pdf(pdf_path: Path, lang: str) -> list[dict]:
    """
    Process one PDF file and return its list of chunks.

    Double-page detection:
      If any span has x0 >= 539pts, the PDF is treated as double-page format.
      Each physical page is split into two logical pages before classification.
    """
    print(f"\n  Opening {pdf_path.name}  [lang={lang}]...")
    doc     = fitz.open(str(pdf_path))
    n_pages = len(doc)
    pw_raw  = doc[0].rect.width
    print(f"  {n_pages} physical pages, width={pw_raw:.0f}pts")

    # Extract all spans from physical pages, then close the file
    raw_spans = [extract_spans(doc[i]) for i in range(n_pages)]
    doc.close()

    # Check if the PDF uses a double-page physical layout
    double = any(is_double_page(s) for s in raw_spans)
    print(f"  Double-page format: {double}")

    # Build the list of logical pages and set the effective page width
    if double:
        logical_pages: list[list[dict]] = []
        for spans in raw_spans:
            left, right = split_double_page(spans)
            if left:
                logical_pages.append(left)
            if right:
                logical_pages.append(right)
        page_width = HALF_PAGE_WIDTH
    else:
        logical_pages = raw_spans
        page_width    = pw_raw

    print(f"  Logical pages: {len(logical_pages)}")

    # Classify each logical page
    classifications = [classify_page(s, lang) for s in logical_pages]
    for i, cls in enumerate(classifications):
        print(f"    lp{i+1:02d} -> {cls}")

    # Group pages by type
    chrono_spans : list[list[dict]] = []
    history_spans: list[list[dict]] = []
    wine_spans   : list[list[dict]] = []

    for i, cls in enumerate(classifications):
        s = logical_pages[i]
        if cls == "chrono":
            chrono_spans.append(s)
        elif cls == "history":
            history_spans.append(s)
        elif cls == "wine_sheet":
            wine_spans.append(s)

    chunks: list[dict] = []

    if chrono_spans:
        chunks.append(parse_chrono_pages(chrono_spans, lang))
        print(f"  -> 1 chronology chunk")

    if history_spans:
        h = parse_history_pages(history_spans, page_width, lang)
        chunks.extend(h)
        print(f"  -> {len(h)} history chunks")

    if wine_spans:
        w = parse_wine_sheet_pages(wine_spans, page_width, lang)
        chunks.extend(w)
        print(f"  -> {len(w)} wine sheet chunks")

    return chunks


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(pdf_it: Path | None, pdf_en: Path | None, output_file: Path) -> list[dict]:
    """
    Run the full chunking pipeline across both PDFs and write the output.
    Results from both PDFs are merged and deduplicated by chunk_id.
    """
    print("\n[1/4] Parsing PDFs...")
    all_chunks: list[dict] = []

    if pdf_it and pdf_it.exists():
        all_chunks.extend(run_single_pdf(pdf_it, lang="it"))
    elif pdf_it:
        print(f"  WARN: {pdf_it} not found, skipped")

    if pdf_en and pdf_en.exists():
        all_chunks.extend(run_single_pdf(pdf_en, lang="en"))
    elif pdf_en:
        print(f"  WARN: {pdf_en} not found, skipped")

    # Deduplicate: the same wine can appear in both IT and EN catalogues
    seen:   set[str]   = set()
    unique: list[dict] = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    print(f"\n[2/4] Total chunks: {len(unique)}")

    # Print distribution summary for quick validation
    types: dict[str, int] = {}
    langs: dict[str, int] = {}
    for c in unique:
        types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1
        langs[c["language"]]   = langs.get(c["language"],   0) + 1
    print(f"  By type     : {types}")
    print(f"  By language : {langs}")

    print("\n[3/4] Writing output...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"  -> {output_file}")

    print("\n[4/4] Chunk IDs:")
    for c in unique:
        print(f"  {c['chunk_id']}")

    return unique


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    default_it     = str(SCRIPT_DIR / "data" / "mastro_it.pdf")
    default_en     = str(SCRIPT_DIR / "data" / "mastro_en.pdf")
    default_output = str(OUTPUT_DIR  / "wines_chunks_all.json")

    parser = argparse.ArgumentParser(
        description="Mastroberardino wine chunking pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--pdf-it", default=default_it,
                        help="Italian catalogue PDF (mastro_it.pdf)")
    parser.add_argument("--pdf-en", default=default_en,
                        help="English catalogue PDF (mastro_en.pdf)")
    parser.add_argument("--output", default=default_output,
                        help="Output JSON path")
    parser.add_argument("--debug",  action="store_true",
                        help="Print span details for inspection (no chunking)")
    parser.add_argument("--pdf",    default=None,
                        help="PDF to inspect with --debug (default: mastro_it.pdf)")
    parser.add_argument("--page",   type=int, default=None,
                        help="Single page number for --debug (1-based)")
    args = parser.parse_args()

    if args.debug:
        pdf      = Path(args.pdf) if args.pdf else Path(default_it)
        page_idx = (args.page - 1) if args.page else None
        debug_pdf(pdf, page_limit=8, single_page=page_idx)
    else:
        run(
            pdf_it      = Path(args.pdf_it),
            pdf_en      = Path(args.pdf_en),
            output_file = Path(args.output),
        )


if __name__ == "__main__":
    main()