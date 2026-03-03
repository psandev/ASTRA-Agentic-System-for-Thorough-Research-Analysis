# ASTRA Ingestion Flow — Current State (March 2026)

Reference document. No changes planned.

---

## PDF Sources

```
PDF source
  ├── text  ──────────────────────────────────────────→ chunk_text()
  │   Docling → markdown (reading order, OCR)                ↓
  │   PyMuPDF fallback (born-digital, <5 pages)        FAISS + BM25
  │   tables included as degraded pipe-table text            ↓
  │   formulas extracted as LaTeX inline               hybrid_retrieve()
  │                                                          ↓
  │                                                     bge_rerank()
  │                                                          ↓
  │                                                    write_section() [LLM]
  │                                                          ↓
  │                                                   markdown sections
  │
  ├── figures/charts  ────────────────────────────────────────────────────┐
  │   Docling (2nd pass, generate_picture_images=True)                    │
  │   → PNG saved to session/sources/figures/                             │
  │   → size filter (min px) + _is_valid_image() corrupt check            │
  │   → Qwen2.5-VL-7B describes → {title, description, key_insight}       │
  │   → RAG chunk built: "[FIGURE] title \n description \n key_insight"   │
  │   → FAISS + BM25 (chunk_type="figure", image_path in metadata)  ✓    │
  │                  ↓ figure_search() at write time                      │
  │                    cosine threshold 0.28                              │
  │                    dedup: each figure used once across all sections   │
  │                  ↓                                                    │
  │             inline in final document as base64 <img>             ✓   │
  │                                                                       │
  ├── tables  ─────────────────────────────────────────────────────────── │ ──┐
  │   Docling text path: exported as markdown pipe tables             ✓   │   │
  │   Docling visual path: _render_table_png() → matplotlib PNG           │   │
  │   → same VLM + FAISS pipeline as figures above                   ✓   │   │
  │   → embedded in report as figure if relevance score ≥ 0.28       ✓   │   │
  │                                                                       │   │
  └── formulas  ──────────────────────────────────────────────────────────┘   │
      Docling formula_enrich → LaTeX strings                                  │
      Embedded inline in markdown text as $...$ / $$...$$              ✓     │
                                                                              │
                                            Final HTML / PDF / DOCX  ←───────┘
```

---

## HTML / Web Sources

```
HTML source
  ├── text  ──────────────────────────────────────────→ chunk_text()
  │   Jina Reader → clean markdown (primary)                 ↓
  │   Firecrawl → structured markdown (alternative)    FAISS + BM25
  │   Trafilatura → plain text (fallback)                    ↓
  │   BeautifulSoup → stripped text (last resort)      hybrid_retrieve()
  │   tables: pipe-table if Jina/Firecrawl + well-formed HTML  ↓
  │            plain text if Trafilatura/BS4 (structure lost)  ↓
  │                                                     bge_rerank()
  │                                                          ↓
  │                                                    write_section() [LLM]
  │                                                          ↓
  │                                                   markdown sections
  │
  ├── images/figures  ──────────────────────────────────────────────────── ┐
  │   BeautifulSoup scrapes <figure> + <img> from raw_html                 │
  │   Priority 1: <figure><figcaption> elements                            │
  │   Priority 2: <img> in article body                                    │
  │   → download PNG/JPG (15s timeout)                                     │
  │   → size + aspect ratio filter + _is_valid_image() corrupt check       │
  │   → content-hash dedup across all sources                              │
  │   → Qwen2.5-VL-7B describes → {title, description, key_insight}        │
  │   → RAG chunk built → FAISS + BM25 (chunk_type="figure")          ✓   │
  │                  ↓ figure_search() at write time                       │
  │             inline in final document as base64 <img>              ✓   │
  │                                                                        │
  └── tables  ─────────────────────────────────────────────────────────── ─┘
      Rendered to PNG via matplotlib (pd.read_html + _render_dataframe_png) ✓
      → VLM described + FAISS+BM25 indexed (chunk_type="figure",
        figure_type="table", extraction_method="web_table")              ✓
      → embedded in report if relevance score ≥ 0.28                    ✓
      Note: text path (Jina/Firecrawl pipe-table) ALSO preserved         △
      Gate: ASTRA_HTML_TABLE_VISION=false disables this path entirely
```

---

## Auto-generated Charts (separate from visual pipeline)

```
write_section() → markdown output
      ↓
_try_generate_chart() scans for pipe tables with numeric columns
      ↓ found              ↓ not found
generate_chart_matplotlib  skip silently
→ bar chart PNG saved to session/charts/
→ embedded in report via base64 <img>

Note: the original markdown table ALSO renders as a native <table> in HTML.
Both appear in the final report — the table as text, the chart as image.
```

---

## Summary in plain language

### PDF

**Text:** Docling parses the PDF with GPU acceleration, preserving reading order and structure. The full document comes out as markdown including inline LaTeX formulas. This text is chunked and indexed in FAISS and BM25 for retrieval.

**Images and charts:** Docling runs a second pass with image extraction enabled. Every figure and chart is saved as a PNG, filtered for size and integrity, then sent to Qwen2.5-VL-7B which writes a natural-language description including a key insight. That description becomes a RAG chunk stored in FAISS alongside the text chunks — but the image file path is kept in the metadata. At report-writing time, `figure_search()` matches the section topic against figure descriptions by cosine similarity; figures above the threshold are embedded directly into the final document as base64 images.

**Tables:** PDFs get both paths. The text path gives a markdown pipe-table via Docling's text export — usable for retrieval but structure is sometimes degraded. The visual path renders the table to a PNG via matplotlib, which then goes through the same VLM description and FAISS indexing as figures. So PDF tables can appear as embedded images in the report if they are relevant to a section.

**Formulas:** Docling extracts LaTeX strings which are embedded inline in the markdown as `$...$` and `$$...$$`. They render with MathJax in the HTML output.

### HTML

**Text:** The article is fetched through Jina Reader or Firecrawl, which return clean markdown. If those fail, Trafilatura or BeautifulSoup extract plain text. The result is chunked and indexed the same way as PDF text.

**Images:** BeautifulSoup parses the raw HTML and collects `<img>` tags, prioritising `<figure><figcaption>` elements. Images are downloaded, filtered for size and aspect ratio, checked for corruption, deduplicated by content hash, then described by Qwen2.5-VL-7B and indexed in FAISS exactly like PDF figures. They can appear embedded in the final report.

**Tables:** BeautifulSoup parses `<table>` elements from the raw HTML and filters out layout/navigation tables via `_is_data_table()`. Real data tables are parsed to DataFrames with `pd.read_html()` and rendered to styled PNG snapshots via `_render_dataframe_png()` (same blue-header, alternating-row style as PDF tables). Each PNG goes through the same VLM description and FAISS+BM25 indexing pipeline as figures — and can appear embedded in the final report if it scores above the relevance threshold. The text path (Jina/Firecrawl pipe-table markdown) is preserved alongside, so simple tables have two retrieval representations. This feature is gated by `ASTRA_HTML_TABLE_VISION` (default `true`).

**Graphs/charts in HTML:** Only if they are rasterized images (`<img>` tags pointing to PNG/JPG). SVG charts embedded inline and JavaScript-rendered charts (D3, Chart.js, Plotly) are invisible to the pipeline and silently skipped.
