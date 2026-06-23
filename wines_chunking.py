"""
╔══════════════════════════════════════════════════════════════════╗
║  WINES CHUNKING PIPELINE — merged entry point                    ║
║                                                                  ║
║  Two independent PDF families, two independent pipelines,        ║
║  one CLI. Select which one runs with --source.                   ║
║                                                                  ║
║  --source stilema                                                ║
║      Input  : stilema_it.pdf  (77 pages, single-page, ~454pts)   ║
║      Output : output/stilema_chunks.json                         ║
║      Wine sheets span 2 pages (A: profile / B: sensory).         ║
║      Category tracked via marker pages + top-right page label.   ║
║                                                                  ║
║  --source mastro                                                 ║
║      Input  : mastro_it.pdf (28p, single) + mastro_en.pdf        ║
║               (15p, double logical pages, 539pt half-width)      ║
║      Output : output/wines_chunks_all.json                       ║
║      Wine sheets span 1 logical page, parsed via a state         ║
║      machine. Includes a chronology table chunk.                 ║
║                                                                  ║
║  The two pipelines do NOT share page classification or parsing   ║
║  logic — their source documents have different structures        ║
║  (font thresholds, section labels, page layout). Only generic    ║
║  text helpers (normalize/clean/slug) are shared.                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS (identical behaviour in both original scripts)
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent


def normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", s).strip()


def clean(text: str) -> str:
    """Collapse whitespace only."""
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    """Build a URL-safe identifier string from a title."""
    s = normalize(text)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def extract_spans(page: fitz.Page) -> list[dict]:
    """
    Return all text spans from a page, sorted top-to-bottom then left-to-right.
    Each span: {text, x0, y0, x1, sz, bold}
    Shared by both pipelines — span extraction logic is identical.
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


# ═══════════════════════════════════════════════════════════════════════════
#  STILEMA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class Stilema:
    """
    Namespace holding everything specific to the Stilèma pipeline.
    Input  : stilema_it.pdf (77 pages, single-page, ~454pts wide)
    Output : output/stilema_chunks.json
    """

    DEFAULT_PDF = SCRIPT_DIR / "data" / "stilema_it.pdf"
    OUTPUT_DIR  = SCRIPT_DIR / "output"

    # Category labels — appear both as standalone marker pages and in the
    # top-right corner of page B
    CATEGORY_LABELS = {
        "icon", "cru", "heritage", "smart",
        "passiti", "passiti da muffa nobile", "olio evo bio",
    }

    CATEGORY_CANONICAL = {
        "icon":                    "ICON",
        "cru":                     "CRU",
        "heritage":                "HERITAGE",
        "smart":                   "SMART",
        "passiti":                 "PASSITI",
        "passiti da muffa nobile": "PASSITI",
        "olio evo bio":            "OLIO EVO BIO",
    }

    # Font size thresholds
    SZ_WINE_NAME = 20.0   # Large wine name title (e.g. "STILÈMA" at ~27pt)
    SZ_WINE_SUB  = 12.0   # Subtitle denomination (e.g. "FIANO DI AVELLINO DOCG")
    SZ_BODY      = 8.5    # Normal body text

    NOISE_RE = re.compile(
        r"^(raccontato da piero|mastroberardino\.com|"
        r"©\s*mastroberardino|page\s*\d+|\d+\s*/\s*\d+|"
        r"incoming@|wine\s*shop|public\s*relations|enoteca@|pr@|"
        r"via\s+re\s+manfredi|atripalda|www\.|"
        r"ministero|politiche\s+agricole|consiglio\s+per|"
        r"catalogoviti\.|sperimentazione)$",
        re.IGNORECASE,
    )

    # ─── text helpers ──────────────────────────────────────────────────────

    @classmethod
    def is_noise(cls, text: str) -> bool:
        return bool(cls.NOISE_RE.match(text.strip()))

    @classmethod
    def is_category_label(cls, text: str) -> bool:
        return normalize(text) in cls.CATEGORY_LABELS

    @staticmethod
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

    # ─── page classifier ───────────────────────────────────────────────────

    @classmethod
    def classify_page(cls, spans: list[dict], page_width: float) -> str:
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

        # Category marker page: ≤ 8 meaningful spans, at least one is a category label
        meaningful = [s for s in spans if not cls.is_noise(s["text"])]
        if len(meaningful) <= 8:
            cat_spans = [s for s in meaningful if cls.is_category_label(s["text"])]
            if cat_spans:
                return "category_marker"

        # Cover / back cover
        if "i vini" in texts_norm and len(meaningful) <= 10:
            return "cover"
        if len(meaningful) <= 4:
            return "empty"

        # Wine sheet page B (CARATTERI SENSORIALI)
        if "caratteri sensoriali" in texts_norm:
            return "wine_b"

        # Wine sheet page A (PROFILO DEL VINO / PASSITO / OLIO)
        profilo_labels = {"profilo del vino", "profilo del passito", "profilo dell olio"}
        if texts_norm & profilo_labels:
            return "wine_a"

        # History / narrative: large-font section title + substantial body text
        if max_sz >= 20.0:
            body_spans = [s for s in spans if s["sz"] < 15.0 and not cls.is_noise(s["text"])]
            if len(body_spans) >= 5:
                return "history"

        # Fallback: treat as history if enough body text
        body_spans = [s for s in spans if s["sz"] < 15.0 and not cls.is_noise(s["text"])]
        if len(body_spans) >= 8:
            return "history"

        return "empty"

    @classmethod
    def extract_category_from_page(cls, spans: list[dict], page_width: float) -> str | None:
        """
        Read the category label from the top-right corner of a wine_b page,
        or from a category_marker page.
        """
        for s in spans:
            if cls.is_category_label(s["text"]):
                norm = normalize(s["text"])
                return cls.CATEGORY_CANONICAL.get(norm)
        return None

    # ─── history parser ─────────────────────────────────────────────────────

    @classmethod
    def parse_history_pages(cls, pages_spans: list[list[dict]], page_width: float) -> list[dict]:
        """
        Merge consecutive history pages into section-based chunks.
        Two-column layout: left column read entirely before right column.
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
                if cls.is_noise(t) or cls.is_category_label(t):
                    continue
                if s["sz"] >= 20.0 and not s["bold"]:
                    flush()
                    current_title = t
                elif s["sz"] >= 12.0 and s["bold"]:
                    flush()
                    current_title = t
                elif s["sz"] >= 10.0 and not s["bold"] and len(t) > 30:
                    current_lines.append(t)
                elif s["sz"] >= cls.SZ_BODY:
                    current_lines.append(t)

        flush()
        return chunks

    # ─── wine sheet parser ──────────────────────────────────────────────────

    @classmethod
    def parse_wine_pair(
        cls,
        page_a_spans: list[dict],
        page_b_spans: list[dict],
        page_width:   float,
        category:     str,
        chunk_idx:    int,
    ) -> dict:
        """
        Build one chunk from the two pages of a wine sheet.
        Page A: wine_name, subtitle, intro, profilo fields.
        Page B: caratteri sensoriali, pairings, category label (top-right).
        """
        mid_x = page_width / 2

        # 1. Extract wine name + subtitle from page A
        meaningful_a = [s for s in page_a_spans
                         if not cls.is_noise(s["text"]) and not cls.is_category_label(s["text"])
                         and s["sz"] >= cls.SZ_WINE_SUB]

        by_size = sorted(meaningful_a, key=lambda s: -s["sz"])

        wine_name = ""
        subtitle  = ""

        if by_size:
            wine_name = by_size[0]["text"]

        for s in by_size[1:]:
            if s["text"] != wine_name and s["sz"] >= cls.SZ_WINE_SUB:
                subtitle = s["text"]
                break

        # 2. Collect body text from page A (intro + profilo fields)
        left_a  = sorted([s for s in page_a_spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
        right_a = sorted([s for s in page_a_spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
        ordered_a = left_a + right_a

        body_a_lines: list[str] = []
        skip_set = {normalize(wine_name), normalize(subtitle)}

        for s in ordered_a:
            t    = s["text"]
            norm = normalize(t)

            if cls.is_noise(t) or cls.is_category_label(t):
                continue
            if norm in skip_set:
                continue
            if norm in ("profilo del vino", "profilo del passito", "profilo dell olio"):
                continue  # skip the section header itself
            if s["sz"] < 6.5:
                continue  # micro bar-chart labels

            body_a_lines.append(t)

        # 3. Collect body text from page B (caratteri sensoriali + pairings)
        cat_from_page_b = cls.extract_category_from_page(page_b_spans, page_width)
        if cat_from_page_b:
            category = cat_from_page_b  # page-level label takes priority

        left_b  = sorted([s for s in page_b_spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
        right_b = sorted([s for s in page_b_spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
        ordered_b = left_b + right_b

        body_b_lines: list[str] = []
        for s in ordered_b:
            t    = s["text"]
            norm = normalize(t)

            if cls.is_noise(t) or cls.is_category_label(t):
                continue
            if norm == "caratteri sensoriali":
                continue  # skip section header
            if s["sz"] < 6.5:
                continue

            body_b_lines.append(t)

        # 4. Assemble full text
        wine_full = f"{wine_name} {subtitle}".strip() if subtitle else wine_name
        header    = f"[{wine_full}]"
        body      = " ".join(body_a_lines + body_b_lines).strip()
        full_text = f"{header}\n\n{body}"

        # 5. Extract vintage if present
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
            "wine_type":  cls.guess_wine_type(wine_name, subtitle),
            "project":    "stilema",
            "fields":     {},
        }

    # ─── main pipeline ──────────────────────────────────────────────────────

    @classmethod
    def run(cls, pdf_path: Path, output_file: Path) -> list[dict]:
        print(f"\n[1/4] Opening {pdf_path.name}...")
        doc        = fitz.open(str(pdf_path))
        n          = len(doc)
        page_width = doc[0].rect.width
        print(f"  {n} pages, width={page_width:.0f}pts")

        all_spans = [extract_spans(doc[i]) for i in range(n)]
        doc.close()

        classifications = [cls.classify_page(all_spans[i], page_width) for i in range(n)]

        print("\n[2/4] Page classifications:")
        for i, c in enumerate(classifications):
            if c not in ("empty",):
                print(f"  p{i+1:02d} → {c}")

        chunks:           list[dict] = []
        history_buffer:   list[list[dict]] = []
        current_category: str  = "ICON"  # default before any marker
        wine_a_pending:   list[dict] | None = None
        wine_chunk_idx    = 0

        def flush_history():
            nonlocal history_buffer
            if history_buffer:
                h = cls.parse_history_pages(history_buffer, page_width)
                chunks.extend(h)
                print(f"  → {len(h)} history chunks")
            history_buffer = []

        for i, c in enumerate(classifications):
            spans = all_spans[i]

            if c == "empty" or c == "cover":
                flush_history()
                wine_a_pending = None
                continue

            elif c == "category_marker":
                flush_history()
                wine_a_pending = None
                cat = cls.extract_category_from_page(spans, page_width)
                if cat:
                    current_category = cat
                    print(f"  p{i+1:02d} → category marker: {current_category}")

            elif c == "history":
                if wine_a_pending is not None:
                    print(f"  WARN p{i+1}: found history while wine_a pending — dropping wine_a")
                    wine_a_pending = None
                history_buffer.append(spans)

            elif c == "wine_a":
                flush_history()
                if wine_a_pending is not None:
                    print(f"  WARN p{i+1}: consecutive wine_a pages — dropping previous")
                wine_a_pending = spans

            elif c == "wine_b":
                flush_history()
                if wine_a_pending is None:
                    print(f"  WARN p{i+1}: wine_b without preceding wine_a — building partial chunk")
                    wine_a_pending = []
                wine_chunk_idx += 1
                chunk = cls.parse_wine_pair(
                    wine_a_pending, spans,
                    page_width, current_category, wine_chunk_idx,
                )
                chunks.append(chunk)
                wine_a_pending = None

        flush_history()
        if wine_a_pending is not None:
            print("  WARN: trailing wine_a with no page B — discarded")

        # Deduplicate by chunk_id
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

    @classmethod
    def debug(cls, pdf_path: str, page: int | None):
        doc = fitz.open(pdf_path)
        pw  = doc[0].rect.width
        pages = [page - 1] if page else range(min(10, len(doc)))
        for i in pages:
            if i >= len(doc):
                continue
            spans = extract_spans(doc[i])
            c     = cls.classify_page(spans, pw)
            print(f"\n== PAGE {i+1} [{c}] ==")
            print(f"  {'x0':>5} {'y0':>5} {'sz':>5}  B  text")
            print("  " + "-" * 65)
            for s in spans[:60]:
                b = "B" if s["bold"] else " "
                print(f"  {s['x0']:>5} {s['y0']:>5} {s['sz']:>5.1f}  {b}  {s['text'][:60]}")
        doc.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MASTROBERARDINO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class Mastro:
    """
    Namespace holding everything specific to the Mastroberardino pipeline.
    Input  : mastro_it.pdf (28 pages, single)
             mastro_en.pdf (15 pages, double logical pages)
    Output : output/wines_chunks_all.json
    """

    OUTPUT_DIR = SCRIPT_DIR / "output"

    # Font size thresholds — detected via --debug mode on the actual PDFs
    SZ_COVER_TITLE   = 20.0
    SZ_SECTION_TITLE = 30.0
    SZ_WINE_TITLE    = 12.0
    SZ_VENDEMMIA     = 9.5
    SZ_CHRONO        = 7.5

    # Weather / climate block labels (normalised, no diacritics) — IT + EN
    METEO_LABELS_NORM = {
        "piovosita", "temperature medie", "periodo di vendemmia",
        "bassa", "elevata", "precoce", "tardivo",
        "rainfall", "average temperatures", "harvest period",
        "low", "heavy", "high", "early", "late",
    }

    NAV_PATTERNS = re.compile(
        r"^(taurasi\s+\d{4}[-–]\d{4}|a century of|mastroberardino\.com|"
        r"page\s*\d+|\d+\s*/\s*\d+|ai fiori|the langham)$",
        re.IGNORECASE,
    )

    # Width of one logical page in the English double-page PDF
    HALF_PAGE_WIDTH = 539.0

    # ─── text helpers ──────────────────────────────────────────────────────

    @classmethod
    def is_meteo(cls, text: str) -> bool:
        return normalize(text) in cls.METEO_LABELS_NORM

    @classmethod
    def is_nav(cls, text: str) -> bool:
        return bool(cls.NAV_PATTERNS.match(text.strip()))

    # ─── double-page handling (mastro_en.pdf) ──────────────────────────────

    @classmethod
    def split_double_page(cls, spans: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Split a double-page scan into two independent logical pages.
        Left  page : spans with x0 <  539 (coordinates kept as-is)
        Right page : spans with x0 >= 539 (x0 remapped to x0 - 539)
        """
        left, right = [], []
        for s in spans:
            if s["x0"] < cls.HALF_PAGE_WIDTH:
                left.append(s)
            else:
                right.append({**s, "x0": s["x0"] - int(cls.HALF_PAGE_WIDTH)})
        return left, right

    @classmethod
    def is_double_page(cls, spans: list[dict]) -> bool:
        return any(s["x0"] >= cls.HALF_PAGE_WIDTH for s in spans)

    # ─── page classifier ───────────────────────────────────────────────────

    @classmethod
    def classify_page(cls, spans: list[dict], lang: str) -> str:
        """
        Assign a structural type to each logical page:
          'cover'      -- title page (large TAURASI font), skipped
          'empty'      -- no text spans (image-only page), skipped
          'chrono'     -- vintage chronology table
          'history'    -- narrative text (company history)
          'wine_sheet' -- structured technical sheet for one wine
        """
        if not spans:
            return "empty"

        max_sz = max(s["sz"] for s in spans)
        texts  = {normalize(s["text"]) for s in spans}

        # Cover page: very large "TAURASI" title
        if max_sz >= cls.SZ_COVER_TITLE and "taurasi" in texts:
            return "cover"

        # History / narrative: contains a very large section title (38pt+)
        if max_sz >= cls.SZ_SECTION_TITLE:
            return "history"

        # Chronology table: many small bold spans distributed across 3+ x-columns
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
        has_wine_title = any(s["sz"] >= cls.SZ_WINE_TITLE and s["bold"] for s in spans)
        has_harvest    = harvest_label in texts

        if has_wine_title and has_harvest:
            return "wine_sheet"
        if has_wine_title:
            return "wine_sheet"

        return "history"

    # ─── chronology parser ──────────────────────────────────────────────────

    @classmethod
    def parse_chrono_pages(cls, pages_spans: list[list[dict]], lang: str) -> dict:
        """
        Merge all chronology pages into a single chunk.
        Spans grouped by y0 proximity (within 5pts = same table row).
        """
        lines = []
        for spans in pages_spans:
            rows: dict[int, list[str]] = {}
            for s in spans:
                key = s["y0"] // 5 * 5
                rows.setdefault(key, []).append(s["text"])
            for y in sorted(rows):
                line = " | ".join(rows[y])
                if line and not cls.is_meteo(line) and not cls.is_nav(line):
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
            "project":    "mastro",
            "fields":     {},
        }

    # ─── history parser ─────────────────────────────────────────────────────

    @classmethod
    def parse_history_pages(cls, pages_spans: list[list[dict]], page_width: float,
                             lang: str) -> list[dict]:
        """
        Parse narrative pages into section-based chunks.
        Two-column reading order: left column, then right column.
        New chunk at each section title (sz >= 30pt, or sz >= 11pt bold).
        """
        mid_x = page_width / 2
        chunks: list[dict] = []
        current_title = ("Mastroberardino History" if lang == "en"
                          else "Storia Mastroberardino")
        current_lines: list[str] = []
        chunk_idx = 0

        def flush():
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
            left  = sorted([s for s in spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
            right = sorted([s for s in spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
            ordered = left + right

            for s in ordered:
                text = s["text"]
                if cls.is_meteo(text) or cls.is_nav(text):
                    continue
                if s["sz"] >= cls.SZ_SECTION_TITLE:
                    flush()
                    current_title = text
                elif s["sz"] >= 11.0 and s["bold"]:
                    flush()
                    current_title = text
                else:
                    current_lines.append(text)

        flush()
        return chunks

    # ─── wine sheet parser ──────────────────────────────────────────────────

    @classmethod
    def parse_wine_sheet_pages(cls, pages_spans: list[list[dict]], page_width: float,
                                lang: str) -> list[dict]:
        """
        Parse wine technical sheet pages into one chunk per wine via a
        state machine. Excludes the weather infographic block and
        bar-chart micro-labels.
        """
        mid_x = page_width / 2
        chunks: list[dict] = []

        current_wine: str | None  = None
        current_year: str | None  = None
        current_lines: list[str]  = []
        chunk_idx = 0

        meteo_section_labels = {
            "piovosita", "temperature medie", "periodo di vendemmia",
            "rainfall", "average temperatures", "harvest period",
        }
        harvest_labels = {"la vendemmia", "the harvest"}

        def flush():
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
                    "vintage":    cls._extract_year(current_year),
                    "category":   cls._guess_category(current_wine),
                    "wine_type":  "red",
                    "fields":     {},
                })
            current_lines = []
            current_wine  = None
            current_year  = None

        for spans in pages_spans:
            in_meteo_block = False

            left  = sorted([s for s in spans if s["x0"] < mid_x],  key=lambda s: s["y0"])
            right = sorted([s for s in spans if s["x0"] >= mid_x], key=lambda s: s["y0"])
            ordered = left + right

            for s in ordered:
                text = s["text"]
                norm = normalize(text)

                if cls.is_nav(text):
                    continue

                if s["sz"] >= 9.5 and s["bold"] and norm in meteo_section_labels:
                    in_meteo_block = True
                    continue

                if in_meteo_block:
                    continue

                if s["sz"] >= cls.SZ_WINE_TITLE and s["bold"] and not re.match(r"^\d{4}$", text):
                    if current_wine:
                        flush()
                    current_wine   = text
                    in_meteo_block = False
                    continue

                if (current_wine and not current_year
                        and s["sz"] >= cls.SZ_WINE_TITLE
                        and re.match(r"^\d{4}", text)):
                    current_year = text
                    continue

                if norm in harvest_labels:
                    continue

                if s["sz"] < 6.5:
                    continue

                if current_wine and text:
                    current_lines.append(text)

        flush()
        return chunks

    # ─── metadata helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_year(s: str | None) -> int | None:
        if not s:
            return None
        m = re.search(r"\d{4}", s)
        return int(m.group()) if m else None

    @staticmethod
    def _guess_category(wine_name: str) -> str:
        n = normalize(wine_name)
        if "radici" in n:
            return "Radici"
        if "centotrenta" in n or "130" in n:
            return "Centotrenta"
        if "fondatore" in n or "antonio" in n:
            return "Special Edition"
        return "Taurasi"

    # ─── debug mode ─────────────────────────────────────────────────────────

    @classmethod
    def debug(cls, pdf_path: Path, page_limit: int = 6, single_page: int | None = None):
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
            double = cls.is_double_page(spans)
            c      = cls.classify_page(spans, lang="?")
            print(f"\n  == PAGE {i+1} [{c}]{'  [DOUBLE]' if double else ''} ==")
            print(f"  {'x0':>6} {'y0':>6} {'sz':>5}  B  text")
            print("  " + "-" * 70)
            for s in spans[:50]:
                b = "B" if s["bold"] else " "
                print(f"  {s['x0']:>6} {s['y0']:>6} {s['sz']:>5.1f}  {b}  {s['text'][:60]}")
        doc.close()

    # ─── single PDF runner ──────────────────────────────────────────────────

    @classmethod
    def run_single_pdf(cls, pdf_path: Path, lang: str) -> list[dict]:
        """
        Process one PDF file and return its list of chunks.
        If any span has x0 >= 539pts, the PDF is treated as double-page
        format and each physical page is split into two logical pages.
        """
        print(f"\n  Opening {pdf_path.name}  [lang={lang}]...")
        doc     = fitz.open(str(pdf_path))
        n_pages = len(doc)
        pw_raw  = doc[0].rect.width
        print(f"  {n_pages} physical pages, width={pw_raw:.0f}pts")

        raw_spans = [extract_spans(doc[i]) for i in range(n_pages)]
        doc.close()

        double = any(cls.is_double_page(s) for s in raw_spans)
        print(f"  Double-page format: {double}")

        if double:
            logical_pages: list[list[dict]] = []
            for spans in raw_spans:
                left, right = cls.split_double_page(spans)
                if left:
                    logical_pages.append(left)
                if right:
                    logical_pages.append(right)
            page_width = cls.HALF_PAGE_WIDTH
        else:
            logical_pages = raw_spans
            page_width    = pw_raw

        print(f"  Logical pages: {len(logical_pages)}")

        classifications = [cls.classify_page(s, lang) for s in logical_pages]
        for i, c in enumerate(classifications):
            print(f"    lp{i+1:02d} -> {c}")

        chrono_spans : list[list[dict]] = []
        history_spans: list[list[dict]] = []
        wine_spans   : list[list[dict]] = []

        for i, c in enumerate(classifications):
            s = logical_pages[i]
            if c == "chrono":
                chrono_spans.append(s)
            elif c == "history":
                history_spans.append(s)
            elif c == "wine_sheet":
                wine_spans.append(s)

        chunks: list[dict] = []

        if chrono_spans:
            chunks.append(cls.parse_chrono_pages(chrono_spans, lang))
            print(f"  -> 1 chronology chunk")

        if history_spans:
            h = cls.parse_history_pages(history_spans, page_width, lang)
            chunks.extend(h)
            print(f"  -> {len(h)} history chunks")

        if wine_spans:
            w = cls.parse_wine_sheet_pages(wine_spans, page_width, lang)
            chunks.extend(w)
            print(f"  -> {len(w)} wine sheet chunks")

        return chunks

    # ─── main pipeline ──────────────────────────────────────────────────────

    @classmethod
    def run(cls, pdf_it: Path | None, pdf_en: Path | None, output_file: Path) -> list[dict]:
        """
        Run the full chunking pipeline across both PDFs and write the output.
        Results from both PDFs are merged and deduplicated by chunk_id.
        """
        print("\n[1/4] Parsing PDFs...")
        all_chunks: list[dict] = []

        if pdf_it and pdf_it.exists():
            all_chunks.extend(cls.run_single_pdf(pdf_it, lang="it"))
        elif pdf_it:
            print(f"  WARN: {pdf_it} not found, skipped")

        if pdf_en and pdf_en.exists():
            all_chunks.extend(cls.run_single_pdf(pdf_en, lang="en"))
        elif pdf_en:
            print(f"  WARN: {pdf_en} not found, skipped")

        seen:   set[str]   = set()
        unique: list[dict] = []
        for c in all_chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                unique.append(c)

        print(f"\n[2/4] Total chunks: {len(unique)}")

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


# ═══════════════════════════════════════════════════════════════════════════
#  MERGE HELPER — used only when no --source is given (both pipelines run)
# ═══════════════════════════════════════════════════════════════════════════

def _tag_and_prefix(chunks: list[dict], id_prefix: str, project: str) -> list[dict]:
    """
    Prefix each chunk_id with id_prefix (so stilema and mastro chunks can be
    merged into a single flat list without chunk_id collisions — both
    pipelines independently generate ids like 'it_wine_001_...') and stamp
    every chunk with its source project name.
    """
    tagged = []
    for c in chunks:
        c = dict(c)  # shallow copy, don't mutate the original
        c["chunk_id"] = f"{id_prefix}_{c['chunk_id']}"
        c["project"]  = project
        tagged.append(c)
    return tagged


def _write_merged(chunks: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n  → {output_file}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI — single entry point, dispatches on --source
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Wines chunking pipeline (merged: stilema + mastro)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["stilema", "mastro"], default=None,
        help="Which pipeline to run:\n"
             "  stilema -> stilema_it.pdf (single source, category-based)\n"
             "  mastro  -> mastro_it.pdf + mastro_en.pdf (chrono + history + wine sheets)\n"
             "If omitted, BOTH pipelines run and their chunks are merged into\n"
             "ONE flat list in a single output file (default: wines_chunks_all.json).",
    )

    # stilema args
    parser.add_argument("--pdf", default=str(Stilema.DEFAULT_PDF),
                         help="[stilema] Path to stilema_it.pdf")
    parser.add_argument("--output", default=None,
                         help="Output JSON path (defaults depend on --source / merged mode)")

    # mastro args
    parser.add_argument("--pdf-it", default=str(SCRIPT_DIR / "data" / "mastro_it.pdf"),
                         help="[mastro] Italian catalogue PDF (mastro_it.pdf)")
    parser.add_argument("--pdf-en", default=str(SCRIPT_DIR / "data" / "mastro_en.pdf"),
                         help="[mastro] English catalogue PDF (mastro_en.pdf)")

    # shared debug args
    parser.add_argument("--debug", action="store_true",
                         help="Print span-level details for inspection (no chunking)")
    parser.add_argument("--page", type=int, default=None,
                         help="Single page number to inspect with --debug (1-based)")

    args = parser.parse_args()

    if args.source is None:
        # No --source given: run BOTH pipelines and merge into one flat list.
        if args.debug:
            parser.error("--debug requires an explicit --source (stilema or mastro)")

        print("\n" + "═" * 78)
        print("  RUNNING STILEMA PIPELINE")
        print("═" * 78)
        # Write to a throwaway temp path; we only need the in-memory list here.
        stilema_chunks = Stilema.run(Path(args.pdf), Stilema.OUTPUT_DIR / "stilema_chunks.json")

        print("\n" + "═" * 78)
        print("  RUNNING MASTRO PIPELINE")
        print("═" * 78)
        mastro_chunks = Mastro.run(
            pdf_it      = Path(args.pdf_it),
            pdf_en      = Path(args.pdf_en),
            output_file = Mastro.OUTPUT_DIR / "mastro_chunks.json",
        )

        print("\n" + "═" * 78)
        print("  MERGING INTO A SINGLE FLAT LIST")
        print("═" * 78)
        merged = (
            _tag_and_prefix(stilema_chunks, id_prefix="stilema", project="stilema")
            + _tag_and_prefix(mastro_chunks, id_prefix="mastro", project="mastroberardino")
        )

        # Deduplicate by the now-prefixed chunk_id (should be a no-op, but
        # keeps the same safety guarantee as each individual pipeline).
        seen:   set[str]   = set()
        unique: list[dict] = []
        for c in merged:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                unique.append(c)

        types: dict[str, int] = {}
        projects: dict[str, int] = {}
        for c in unique:
            types[c["chunk_type"]]   = types.get(c["chunk_type"], 0) + 1
            projects[c["project"]]   = projects.get(c["project"], 0) + 1
        print(f"  Total merged chunks : {len(unique)}")
        print(f"  By type             : {types}")
        print(f"  By project          : {projects}")

        merged_output = Path(args.output) if args.output else (SCRIPT_DIR / "output" / "wines_chunks_all.json")
        _write_merged(unique, merged_output)
        return

    if args.source == "stilema":
        if args.debug:
            Stilema.debug(args.pdf, args.page)
            return
        output = Path(args.output) if args.output else (Stilema.OUTPUT_DIR / "stilema_chunks.json")
        Stilema.run(Path(args.pdf), output)

    elif args.source == "mastro":
        if args.debug:
            # --pdf, if explicitly given, overrides which file --debug inspects;
            # otherwise default to the Italian catalogue
            pdf      = Path(args.pdf) if args.pdf != str(Stilema.DEFAULT_PDF) else Path(args.pdf_it)
            page_idx = (args.page - 1) if args.page else None
            Mastro.debug(pdf, page_limit=8, single_page=page_idx)
            return
        output = Path(args.output) if args.output else (Mastro.OUTPUT_DIR / "mastro_chunks.json")
        Mastro.run(
            pdf_it      = Path(args.pdf_it),
            pdf_en      = Path(args.pdf_en),
            output_file = output,
        )


if __name__ == "__main__":
    main()