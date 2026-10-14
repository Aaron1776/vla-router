"""Render docs/CONCEPT.md to a print-ready PDF with headless Google Chrome.

Usage (from the repo root):
    python scripts/build_concept_pdf.py [--out docs/CONCEPT.pdf] [--font-pt 10]

Needs the `markdown` package and Google Chrome. Prints the page count when done.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import tempfile

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.55in 0.6in; }
html { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: FONT_PTpt; line-height: 1.28; color: #111; }
body { margin: 0; }
h1 { font-size: 1.45em; margin: 0 0 0.25em; }
h2 { font-size: 1.12em; margin: 0.75em 0 0.25em; }
p { margin: 0 0 0.45em; }
ol, ul { margin: 0 0 0.45em; padding-left: 1.4em; }
li { margin: 0 0 0.15em; }
code { font-family: Menlo, monospace; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; margin: 0.25em 0 0.5em; font-size: 0.93em; }
th, td { border: 0.5pt solid #999; padding: 0.15em 0.4em; text-align: left; vertical-align: top; }
th { background: #eee; }
.refs ol { font-size: 0.86em; }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="docs/CONCEPT.md")
    parser.add_argument("--out", default="docs/CONCEPT.pdf")
    parser.add_argument("--font-pt", type=float, default=10.0)
    args = parser.parse_args()

    body = markdown.markdown(pathlib.Path(args.src).read_text(encoding="utf-8"), extensions=["tables"])
    # Smaller type for the reference list only.
    body = re.sub(r"(<h2>References</h2>\s*)(<ol>.*?</ol>)", r'\1<div class="refs">\2</div>', body, flags=re.S)
    html = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Concept Document</title>'
        f"<style>{CSS.replace('FONT_PT', str(args.font_pt))}</style></head><body>{body}</body></html>"
    )

    out = pathlib.Path(args.out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "concept.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()],
            check=True,
            capture_output=True,
        )
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    print(f"wrote {out} ({pages} pages)")


if __name__ == "__main__":
    main()
