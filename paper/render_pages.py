"""Render main.pdf pages to PNG for visual proof-reading (needs PyMuPDF; base conda python).

    python paper/render_pages.py [dpi] [first-last]
"""
import sys
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / "_pages"; OUT.mkdir(exist_ok=True)
dpi = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = sys.argv[2] if len(sys.argv) > 2 else None
doc = fitz.open(HERE / "main.pdf")
pages = range(len(doc))
if rng:
    a, b = rng.split("-"); pages = range(int(a) - 1, int(b))
for i in pages:
    pix = doc[i].get_pixmap(dpi=dpi)
    pix.save(OUT / f"p{i + 1:02d}.png")
print(f"{len(doc)} pages; wrote {OUT}")
