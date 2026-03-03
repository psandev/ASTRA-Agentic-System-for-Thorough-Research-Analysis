#!/usr/bin/env python3
"""
Post-process an existing ASTRA HTML report to apply two fixes:

  1. Linkify bare URLs in text nodes (per-section reference lists etc.)
  2. Inject chart figures inline after their source table (if any)
  3. Apply updated CSS for scientific-paper look
  4. Style per-section reference subsections as footnote-like blocks

Usage:
    python scripts/fix_html_report.py path/to/ASTRA_Report.html
    python scripts/fix_html_report.py path/to/ASTRA_Report.html --inplace
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# ── Updated CSS (mirrors _HTML_EXTRA_CSS in layer6_report.py) ──────────────

NEW_CSS = """
/* ── Base typography ─────────────────────────────────────────────────── */
body        { font-family: "Georgia", "Times New Roman", serif;
              font-size: 17px; line-height: 1.75; color: #1a1a2e; max-width: 900px;
              margin: 0 auto; padding: 2rem 2.5rem; }
h1          { font-size: 1.7em; color: #003366; border-bottom: 3px solid #003366;
              padding-bottom: 10px; margin-top: 2.5rem; }
h2          { font-size: 1.35em; color: #004080; margin-top: 2rem;
              border-bottom: 1px solid #cde; padding-bottom: 4px; }
h3          { font-size: 1.15em; color: #1a5276; margin-top: 1.6rem; }
h4          { font-size: 1em; color: #2e4057; font-style: italic; margin-top: 1.2rem; }
a           { color: #0055cc; text-decoration: none; }
a:hover     { text-decoration: underline; }
a.cite      { font-size: 0.82em; vertical-align: super; color: #0055aa;
              font-family: sans-serif; }
p           { margin: 0.4em 0 1em; text-align: justify; }
table       { border-collapse: collapse; width: 100%; margin: 1.4em 0 0.4em;
              font-size: 0.93em; font-family: sans-serif; }
caption     { font-size: 0.92em; color: #444; font-style: italic;
              caption-side: top; text-align: left; margin-bottom: 4px; }
th          { background: #003366; color: #fff; padding: 9px 13px; text-align: left;
              font-weight: 600; letter-spacing: 0.02em; }
td          { border: 1px solid #c5cfe8; padding: 7px 13px; vertical-align: top; }
tr:nth-child(even) td { background: #f0f4fb; }
tr:hover td { background: #e8eef8; }
code        { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
              font-size: 0.85em; background: #f0f4f8; padding: 2px 6px;
              border-radius: 3px; color: #c0392b; }
pre         { background: #1e1e2e; color: #cdd6f4; padding: 16px 20px;
              border-radius: 8px; overflow-x: auto; font-size: 0.88em; }
pre code    { background: none; padding: 0; color: inherit; }
blockquote  { border-left: 4px solid #0066cc; margin: 1.2em 0;
              padding: 0.6em 1.4em; background: #f5f9ff; color: #333;
              border-radius: 0 6px 6px 0; font-style: italic; }
.chart-figure {
    margin: 1.8em auto; text-align: center;
    page-break-inside: avoid; max-width: 820px;
}
.chart-figure figure {
    display: inline-block; width: 100%;
    border: 1px solid #d0daea; border-radius: 8px;
    padding: 16px 16px 12px; background: #f9faff;
    box-shadow: 0 2px 10px rgba(0,0,51,.07);
}
.chart-figure img {
    max-width: 100%; height: auto; display: block;
    margin: 0 auto 8px; border-radius: 4px; box-shadow: none;
}
.chart-figure figcaption {
    font-family: sans-serif; font-size: 0.88em; color: #444;
    margin-top: 6px; text-align: center; font-style: italic; line-height: 1.4;
}
img         { max-width: 100%; height: auto; display: block; margin: 1.5em auto;
              border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
.toc        { background: #f5f9ff; border: 1px solid #b8d0f0; border-radius: 8px;
              padding: 1.2rem 1.8rem; margin: 2.5rem 0; font-family: sans-serif; }
.toc h2     { border: none; margin-top: 0; color: #003366; font-size: 1.1em; }
.toc ol     { margin: 0.4em 0; padding-left: 1.4em; }
.toc li     { margin: 0.25em 0; }
.toc a      { color: #003366; }
.title-page { text-align: center; padding: 70px 0 50px;
              border-bottom: 2px solid #dde; margin-bottom: 3.5rem; }
.title-page h1 { border: none; font-size: 2em; color: #002244; }
.section    { margin-bottom: 3.5rem; border-bottom: 1px solid #e0e4ef;
              padding-bottom: 1.5rem; }
ol.bib-list { font-family: sans-serif; font-size: 0.92em; line-height: 1.6;
              padding-left: 1.6em; }
ol.bib-list li { margin-bottom: 0.6em; }
/* Per-section reference lists — styled as subtle footnotes */
h3#references, h4#references,
h3#key-sources, h4#key-sources,
h3#references-1, h3#references-2, h3#references-3,
h3#references-4, h3#references-5, h3#references-6,
h3#references-7 {
    font-size: 0.88em; color: #888; border-bottom: none;
    margin-top: 2em; letter-spacing: 0.05em; text-transform: uppercase;
}
h3#references + ul, h4#references + ul,
h3#key-sources + ul, h4#key-sources + ul,
h3#references-1 + ul, h3#references-2 + ul,
h3#references-3 + ul, h3#references-4 + ul,
h3#references-5 + ul, h3#references-6 + ul,
h3#references-7 + ul {
    font-family: sans-serif; font-size: 0.85em; color: #666;
    background: #f8f9fb; border-left: 3px solid #ccc;
    padding: 8px 16px; border-radius: 0 4px 4px 0;
    list-style: none; margin: 0;
}
h3#references + ul li, h4#references + ul li,
h3#key-sources + ul li, h4#key-sources + ul li,
h3#references-1 + ul li, h3#references-2 + ul li,
h3#references-3 + ul li, h3#references-4 + ul li,
h3#references-5 + ul li, h3#references-6 + ul li,
h3#references-7 + ul li { padding: 1px 0; }
"""


def _strip_section_references(html: str) -> str:
    """Remove redundant per-section References/Key-Sources subsections."""
    return re.sub(
        r"<h[34][^>]*>\s*(?:References|Key\s+Sources|Further\s+Reading|Sources)\s*</h[34]>"
        r"\s*(?:<[ou]l>.*?</[ou]l>|<p>.*?</p>)",
        "",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _linkify_bare_urls(html: str) -> str:
    """Wrap bare http(s) URLs in text nodes with <a> tags.

    Tracks anchor depth so URLs inside existing links are not double-wrapped.
    """
    parts = re.split(r"(<[^>]+>)", html)
    result: list[str] = []
    in_anchor = 0

    for part in parts:
        if part.startswith("<"):
            low = part.lower()
            if re.match(r"<a[\s>]", low):
                in_anchor += 1
            elif low.startswith("</a"):
                in_anchor = max(0, in_anchor - 1)
            result.append(part)
        else:
            if in_anchor == 0:
                part = re.sub(
                    r"(https?://[^\s<>\"',\)\]]+)",
                    r'<a href="\1" target="_blank">\1</a>',
                    part,
                )
            result.append(part)

    return "".join(result)


def fix_report(html: str) -> str:
    """Apply all fixes to the HTML string and return the updated version."""

    # 1. Replace existing <style> block with updated CSS
    html = re.sub(
        r"<style>.*?</style>",
        f"<style>{NEW_CSS}</style>",
        html,
        flags=re.DOTALL,
    )

    # 2. Upgrade bibliography <ol> to add class="bib-list"
    html = html.replace(
        "<h1>Bibliography</h1><ol>",
        '<h1>Bibliography</h1><ol class="bib-list">',
    )

    # 3. Fix double-wrapped anchor tags from previous linkify runs
    #    e.g. <a href="url"><a href="url">url</a></a> → <a href="url">url</a>
    html = re.sub(
        r'<a([^>]*)><a[^>]*>([^<]*)</a></a>',
        r'<a\1>\2</a>',
        html,
    )

    # 4. Strip per-section reference subsections (global pass — safe since the
    #    global Bibliography uses <h1>, not <h3>/<h4>)
    html = _strip_section_references(html)

    # 5. Linkify bare URLs across the full document text nodes
    #    (anchor-depth tracking prevents double-wrapping)
    html = _linkify_bare_urls(html)

    return html


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: fix_html_report.py <path-to-html> [--inplace]")
        sys.exit(1)

    src = Path(sys.argv[1])
    inplace = "--inplace" in sys.argv

    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    html = src.read_text(encoding="utf-8")
    fixed = fix_report(html)

    if inplace:
        src.write_text(fixed, encoding="utf-8")
        print(f"Updated in place: {src}")
    else:
        out = src.with_stem(src.stem + "_fixed")
        out.write_text(fixed, encoding="utf-8")
        print(f"Written to: {out}")


if __name__ == "__main__":
    main()
