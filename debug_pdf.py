"""
Quick PDF structure debugger — run this before the chunking pipeline.
Usage: python debug_pdf.py data/mastro_it.pdf
       python debug_pdf.py data/mastro_it.pdf --page 5
"""
import sys
import argparse
import fitz

def debug(pdf_path: str, target_page: int | None, n_pages: int) -> None:
    doc = fitz.open(pdf_path)
    print(f"\n  PDF   : {pdf_path}")
    print(f"  Pages : {len(doc)}")
    print(f"  Width : {doc[0].rect.width:.0f} pts  "
          f"Height: {doc[0].rect.height:.0f} pts")
    print(f"  Mid-X : {doc[0].rect.width/2:.0f} pts  (column split estimate)\n")

    pages = (
        [(target_page - 1, doc[target_page - 1])]
        if target_page
        else [(i, doc[i]) for i in range(min(n_pages, len(doc)))]
    )

    for idx, page in pages:
        print(f"\n  ══ PAGE {idx+1} ══")
        print(f"  {'x0':>5}  {'y0':>5}  {'sz':>5}  {'B':1}  {'font':<35}  {'color'}")
        print(f"  {'─'*72}")
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                lt = " ".join(s["text"].strip() for s in line["spans"]).strip()
                if not lt:
                    continue
                f = line["spans"][0]
                b = "B" if ("Bold" in f["font"] or "bold" in f["font"].lower()) else " "
                x0 = round(line["bbox"][0])
                y0 = round(line["bbox"][1])
                sz = round(f["size"], 1)
                color = f.get("color", 0)
                print(f"  {x0:>5}  {y0:>5}  {sz:>5}  {b}  {f['font']:<35}  #{color:06x}")
    doc.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--page",    type=int, default=None)
    p.add_argument("--n-pages", type=int, default=6)
    args = p.parse_args()
    debug(args.pdf, args.page, args.n_pages)