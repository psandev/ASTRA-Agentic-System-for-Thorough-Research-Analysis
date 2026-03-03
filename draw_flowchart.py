"""Generate the ASTRA system architecture flowchart and save as PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ── Canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 34))
ax.set_xlim(0, 24)
ax.set_ylim(0, 34)
ax.axis("off")
BG = "#0d1117"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "ui":           "#0f2d4a",   # navy  – Gradio UI
    "node":         "#0f2e1a",   # green – main pipeline nodes
    "crawl":        "#1a2a0a",   # olive – crawlers
    "doc":          "#2a1a0a",   # brown – doc processing
    "rag":          "#0a1a2e",   # blue  – RAG / index
    "vision":       "#2a0a2e",   # purple – vision / VLM
    "draft":        "#0a2a1a",   # teal-green – writer
    "judge":        "#2e0a1a",   # rose  – quality
    "decision":     "#2e2200",   # amber – decision
    "refine":       "#2a1500",   # burnt orange – refiner
    "assemble":     "#0a1f2e",   # steel – assembler
    "output":       "#001f1f",   # dark teal – output
    "callout":      "#151520",   # very dark – sidebars
    "text":         "#e6e6e6",
    "subtext":      "#999999",
    "dim":          "#666666",
    # borders
    "b_ui":         "#3a9fff",
    "b_node":       "#3aff88",
    "b_crawl":      "#88cc44",
    "b_doc":        "#ff9944",
    "b_rag":        "#4488ff",
    "b_vision":     "#cc66ff",
    "b_draft":      "#44ffcc",
    "b_judge":      "#ff4488",
    "b_dec":        "#ffcc00",
    "b_refine":     "#ff8844",
    "b_assemble":   "#44ccff",
    "b_output":     "#00ffcc",
    "b_callout":    "#444466",
    # arrows
    "arr":          "#556677",
    "arr_yes":      "#3aff88",
    "arr_no":       "#ff4444",
    "arr_side":     "#888866",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", color=C["node"],
        border=C["b_node"], fontsize=10, subfontsize=7.5, pad=0.09):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=color, edgecolor=border, linewidth=2.0, zorder=3,
    )
    ax.add_patch(rect)
    label_y = y + (0.15 if sublabel else 0)
    ax.text(x, label_y, label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=C["text"], zorder=4, multialignment="center")
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color=C["subtext"], zorder=4, multialignment="center",
                linespacing=1.35)


def diamond(ax, x, y, w, h, label, color=C["decision"],
            border=C["b_dec"], fontsize=8.5):
    dx, dy = w / 2, h / 2
    xs = [x, x + dx, x, x - dx, x]
    ys = [y + dy, y, y - dy, y, y + dy]
    ax.fill(xs, ys, color=color, zorder=3)
    ax.plot(xs, ys, color=border, linewidth=2.2, zorder=4)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=C["text"], zorder=5, multialignment="center")


def arr(ax, x1, y1, x2, y2, color=C["arr"], label="", lw=1.8, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=2)
    if label:
        mx = (x1 + x2) / 2 + (0.35 if rad > 0 else (-0.35 if rad < 0 else 0.15))
        my = (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=8, color=color,
                fontweight="bold", zorder=5, ha="left" if label[0] != " " else "right")


def hline(ax, y, x1=1.5, x2=22.5, color="#222233", lw=0.6):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=1, ls="--")


def phase(ax, y, text):
    ax.text(0.15, y, text, fontsize=6, color="#445566", ha="left", va="center",
            style="italic", multialignment="left", zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# Y coordinates  (higher = higher on canvas)
# ═══════════════════════════════════════════════════════════════════════════
CX = 11.5   # main column centre

Y = {
    "title":  33.1,
    "ui":     31.9,
    "node1":  30.4,
    "crawl":  28.5,
    "proc":   26.5,
    "rag":    24.6,
    "draft":  22.7,
    "judge":  20.7,
    "dec":    19.0,
    "refine": 17.3,
    "assem":  16.9,
    "output": 15.1,
    "legend": 11.8,
    "footer": 0.45,
}

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(CX, Y["title"], "ASTRA — System Architecture",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="#ffffff", zorder=5)
ax.text(CX, Y["title"] - 0.6,
        "Agentic System for Thorough Research & Analysis  ·  8-node LangGraph Pipeline",
        ha="center", va="center", fontsize=10.5, color="#888888",
        style="italic", zorder=5)

# ── Phase labels ─────────────────────────────────────────────────────────────
phase_defs = [
    (Y["ui"],     "UI"),
    (Y["node1"],  "NODE 1\nQuery"),
    (Y["crawl"],  "NODE 2\nCrawl"),
    (Y["proc"],   "NODE 3\nParse"),
    (Y["rag"],    "NODE 4\nIndex"),
    (Y["draft"],  "NODE 5\nDraft"),
    (Y["judge"],  "NODE 6\nEval"),
    (Y["refine"], "NODE 7\nRefine"),
    (Y["assem"],  "NODE 8\nAssem."),
    (Y["output"], "OUTPUT"),
]
for py, ptxt in phase_defs:
    phase(ax, py, ptxt)

# ── Divider lines ─────────────────────────────────────────────────────────────
for yd in [Y["ui"] - 0.75, Y["node1"] - 0.75, Y["crawl"] - 0.9,
           Y["proc"] - 0.9, Y["rag"] - 0.9, Y["draft"] - 0.9,
           Y["judge"] - 0.9, Y["dec"] - 1.0, Y["output"] - 0.85]:
    hline(ax, yd)

# ════════════════════════════════════════════════════════════════════════════
# NODE 0 — Gradio UI
# ════════════════════════════════════════════════════════════════════════════
box(ax, CX, Y["ui"], 7.5, 0.80,
    "Gradio Web UI  (0.0.0.0:7860)",
    "User enters research query  ·  Live streaming progress  ·  HTML / PDF / MD downloads  ·  LangSmith trace link",
    color=C["ui"], border=C["b_ui"], fontsize=11, subfontsize=8)

# ════════════════════════════════════════════════════════════════════════════
# NODE 1 — Query Intelligence
# ════════════════════════════════════════════════════════════════════════════
arr(ax, CX, Y["ui"] - 0.40, CX, Y["node1"] + 0.52)
box(ax, CX, Y["node1"], 8.0, 0.90,
    "NODE 1  ·  Query Intelligence  (Qwen3-32B-AWQ)",
    "query_expand()  →  sub-queries · section outline · source priorities\n"
    "plan_research()  →  research plan passed to state",
    color=C["node"], border=C["b_node"], fontsize=10.5, subfontsize=8)

# ════════════════════════════════════════════════════════════════════════════
# NODE 2 — Parallel Crawlers (fan-out row)
# ════════════════════════════════════════════════════════════════════════════
CXWS = [4.0, 8.1, 14.2, 18.4]
crawl_labels = [
    ("Web Crawl",     "Tavily (primary)\nDDG · Jina · Firecrawl\nMedium · Substack"),
    ("Academic",      "arXiv · OpenAlex\nSemantic Scholar\nPubMed · Papers-w-Code"),
    ("Specialist",    "HuggingFace Blog\nResearch Blogs\nAI news aggregators"),
    ("Code / Repos",  "GitHub search\nstars-sorted top 5\nper sub-query"),
]
for cx_s in CXWS:
    ax.annotate("", xy=(cx_s, Y["crawl"] + 0.46),
                xytext=(CX, Y["node1"] - 0.45),
                arrowprops=dict(arrowstyle="-|>", color=C["b_crawl"],
                                lw=1.3, connectionstyle="arc3,rad=0.0"), zorder=2)
for cx_s, (lbl, sub) in zip(CXWS, crawl_labels):
    box(ax, cx_s, Y["crawl"], 3.7, 0.84, lbl, sub,
        color=C["crawl"], border=C["b_crawl"], fontsize=9, subfontsize=7)
ax.text(CX, Y["crawl"] - 0.70,
        "ThreadPoolExecutor (max 12 workers)  ·  Tavily→DDG fallback  ·  deduplicate_sources() by URL + content hash",
        ha="center", va="center", fontsize=7.2, color="#667755", style="italic")

# ════════════════════════════════════════════════════════════════════════════
# NODE 3 — Document Processing
# ════════════════════════════════════════════════════════════════════════════
for cx_s in CXWS:
    ax.annotate("", xy=(CX, Y["proc"] + 0.56),
                xytext=(cx_s, Y["crawl"] - 0.42),
                arrowprops=dict(arrowstyle="-|>", color=C["b_doc"],
                                lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=2)

doc_cxs = [6.8, 11.5, 16.2]
doc_labels = [
    ("PDF Processor",
     "Docling (GPU, CUDA)\ntables · figures · formulas\nPyMuPDF fallback"),
    ("Web Extractor",
     "Jina · Firecrawl\nmarkdown → paragraphs\nBS4 table extraction"),
    ("Visual Intel\n(layer3_vision)",
     "Figure extraction (Docling)\nQwen2.5-VL-7B ← localhost:8001\n[Type][Title][Desc][Key Insight]"),
]
doc_borders = [C["b_doc"], C["b_doc"], C["b_vision"]]
doc_colors  = [C["doc"],   C["doc"],   C["vision"]]
for cx_s, (lbl, sub), bc, fc in zip(doc_cxs, doc_labels, doc_borders, doc_colors):
    box(ax, cx_s, Y["proc"], 4.2, 0.90, lbl, sub,
        color=fc, border=bc, fontsize=8.5, subfontsize=7)

ax.text(CX, Y["proc"] - 0.72,
        "All figures saved as PNG  ·  .json sidecar with VLM description  ·  content-hash dedup  ·  size filter (≥ 150 px)",
        ha="center", va="center", fontsize=7.2, color="#886644", style="italic")

# ════════════════════════════════════════════════════════════════════════════
# NODE 4 — Index Knowledge
# ════════════════════════════════════════════════════════════════════════════
for cx_s in doc_cxs:
    ax.annotate("", xy=(CX, Y["rag"] + 0.58),
                xytext=(cx_s, Y["proc"] - 0.45),
                arrowprops=dict(arrowstyle="-|>", color=C["b_rag"],
                                lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=2)

rag_cxs = [6.2, 11.5, 16.8]
rag_labels = [
    ("BAAI/bge-m3  (1024d)",
     "CPU · batch=128\nL2-normalized IP\n512-tok chunks, 64 overlap"),
    ("FAISS  +  BM25",
     "IndexFlatIP dense index\nBM25Okapi sparse index\nRRF fusion  (0.6 / 0.4)"),
    ("LightRAG  (optional)",
     "Knowledge graph\nmulti-hop relational\nqueries across docs"),
]
for cx_s, (lbl, sub) in zip(rag_cxs, rag_labels):
    box(ax, cx_s, Y["rag"], 4.4, 0.90, lbl, sub,
        color=C["rag"], border=C["b_rag"], fontsize=8.5, subfontsize=7)

ax.text(CX, Y["rag"] - 0.72,
        "Text chunks  +  figure chunks (chunk_type='figure', image_path in metadata)  indexed together in same FAISS collection",
        ha="center", va="center", fontsize=7.2, color="#446688", style="italic")

# ════════════════════════════════════════════════════════════════════════════
# NODE 5 — Draft Report
# ════════════════════════════════════════════════════════════════════════════
for cx_s in rag_cxs:
    ax.annotate("", xy=(CX, Y["draft"] + 0.60),
                xytext=(cx_s, Y["rag"] - 0.45),
                arrowprops=dict(arrowstyle="-|>", color=C["b_draft"],
                                lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=2)

box(ax, CX, Y["draft"], 8.8, 0.96,
    "NODE 5  ·  Draft Report  (Qwen3-32B-AWQ, temp=0.7)",
    "per section: hybrid_retrieve() → bge_rerank (bge-reranker-v2-m3) → write_section()\n"
    "figure_search(): FAISS figure-only  cosine ≥ ASTRA_FIGURE_RELEVANCE_THRESHOLD\n"
    "_try_generate_chart(): pipe-table → matplotlib/plotly PNG  (auto-injected after </table>)",
    color=C["draft"], border=C["b_draft"], fontsize=10, subfontsize=7.8)

# figure_search sidebar
box(ax, 21.0, Y["draft"] - 0.05, 3.2, 0.80,
    "figure_search()",
    "figures compete only\nagainst each other\nthreshold rejects off-topic",
    color=C["callout"], border=C["b_vision"], fontsize=7.5, subfontsize=6.5)
arr(ax, 15.9, Y["draft"] - 0.05, 19.4, Y["draft"] - 0.05,
    color=C["b_vision"], lw=1.2)

# ════════════════════════════════════════════════════════════════════════════
# NODE 6 — Evaluate Quality
# ════════════════════════════════════════════════════════════════════════════
arr(ax, CX, Y["draft"] - 0.48, CX, Y["judge"] + 0.66)
box(ax, CX, Y["judge"], 8.8, 1.08,
    "NODE 6  ·  Evaluate Quality  (Qwen3-32B-AWQ, temp=0)",
    "evaluate_section()  ×  N sections   —   LLM judge with structured output\n"
    "factual_accuracy  ·  citation_faithfulness  ·  completeness\n"
    "coherence  ·  visual_richness  ·  relevance     (all scored 0.0 – 1.0)",
    color=C["judge"], border=C["b_judge"], fontsize=10, subfontsize=8)

# Thresholds sidebar
box(ax, 21.2, Y["judge"] + 0.05, 3.4, 1.00,
    "Thresholds",
    "factual_accuracy  ≥ 0.70\ncitation_faith.   ≥ 0.80\ncompleteness      ≥ 0.60\ncoherence         ≥ 0.70\nvisual_richness   ≥ 0.50\nrelevance         ≥ 0.80",
    color=C["callout"], border=C["b_judge"], fontsize=7, subfontsize=6.5)
arr(ax, 15.9, Y["judge"] + 0.05, 19.5, Y["judge"] + 0.05,
    color=C["b_judge"], lw=1.2)

# ════════════════════════════════════════════════════════════════════════════
# Decision Diamond
# ════════════════════════════════════════════════════════════════════════════
arr(ax, CX, Y["judge"] - 0.54, CX, Y["dec"] + 0.65)
diamond(ax, CX, Y["dec"], 6.4, 1.18,
        "All sections PASS thresholds?\nOR max iterations reached?\n(ASTRA_EVAL_MAX_ITERATIONS = 3)")

# ════════════════════════════════════════════════════════════════════════════
# NO → NODE 7 Refiner  (left branch)
# ════════════════════════════════════════════════════════════════════════════
refine_x = 3.8
arr(ax, CX - 3.2, Y["dec"], refine_x + 2.2, Y["refine"] + 0.44,
    color=C["arr_no"], label="NO", lw=2.0)
box(ax, refine_x, Y["refine"], 5.6, 0.82,
    "NODE 7  ·  Refine",
    "gap_analysis()  →  targeted sub-queries\ntrigger_reresearch()  →  re-crawl + re-index\nupdate_knowledge_base()  →  re-write sections",
    color=C["refine"], border=C["b_refine"], fontsize=9.5, subfontsize=7.5)

# Loop back to Draft
ax.annotate("", xy=(CX - 5.4, Y["draft"] + 0.0),
            xytext=(refine_x - 2.8, Y["refine"] + 0.15),
            arrowprops=dict(arrowstyle="-|>", color=C["arr_no"], lw=1.8,
                            connectionstyle="arc3,rad=-0.35"), zorder=2)
ax.text(0.6, (Y["refine"] + Y["draft"]) / 2 + 0.3,
        "re-index\n& re-draft",
        fontsize=7.5, color=C["arr_no"], ha="center", fontweight="bold")

# ════════════════════════════════════════════════════════════════════════════
# YES → NODE 8 Assemble Report
# ════════════════════════════════════════════════════════════════════════════
arr(ax, CX, Y["dec"] - 0.59, CX, Y["assem"] + 0.62,
    color=C["arr_yes"], label=" YES", lw=2.0)
box(ax, CX, Y["assem"], 9.0, 1.00,
    "NODE 8  ·  Assemble Report",
    "build_html()  →  self-contained HTML  ·  MathJax  ·  base64 images  ·  active links\n"
    "build_pdf()   →  weasyprint HTML→PDF  ·  A4  ·  page numbers\n"
    "build_markdown()  →  .md with inline chart references",
    color=C["assemble"], border=C["b_assemble"], fontsize=10, subfontsize=8)

# ════════════════════════════════════════════════════════════════════════════
# Final Outputs
# ════════════════════════════════════════════════════════════════════════════
arr(ax, CX, Y["assem"] - 0.50, CX, Y["output"] + 0.54, color=C["b_output"])
box(ax, CX, Y["output"], 9.4, 0.90,
    "Session Output  ·  data/sessions/<slug>_<timestamp>/",
    "reports/ASTRA_Report.html  (primary)  ·  ASTRA_Report.pdf  ·  ASTRA_Report.md\n"
    "charts/ (PNG + SVG)  ·  sources/pdfs/  ·  sources/figures/ (PNG + .json)  ·  logs/session.log",
    color=C["output"], border=C["b_output"], fontsize=10, subfontsize=8)

# ════════════════════════════════════════════════════════════════════════════
# LangSmith callout (right side, mid-height)
# ════════════════════════════════════════════════════════════════════════════
ls_y = (Y["draft"] + Y["judge"]) / 2
box(ax, 21.4, ls_y, 4.0, 1.60,
    "LangSmith Tracing",
    "LANGCHAIN_TRACING_V2=true\nProject: ASTRA-Research-Agent\n\n@traced_node() on every node\nAll LLM calls auto-traced\nTrace URL surfaced in Gradio UI",
    color=C["callout"], border="#8866ff", fontsize=8, subfontsize=7)
arr(ax, 15.9, Y["judge"] - 0.35, 19.4, ls_y + 0.25,
    color="#8866ff", lw=1.1)

# ════════════════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════════════════
hline(ax, Y["legend"] + 0.70, x1=0.5, x2=23.5, color="#333344", lw=1.0)
ax.text(CX, Y["legend"] + 0.40, "Legend", ha="center",
        fontsize=9, color="#cccccc", fontweight="bold")

legend_items = [
    (C["ui"],       C["b_ui"],       "Gradio UI  /  Final Output"),
    (C["node"],     C["b_node"],     "Main Pipeline Node  (Qwen3-32B-AWQ)"),
    (C["crawl"],    C["b_crawl"],    "Web / Academic / Specialist Crawlers"),
    (C["doc"],      C["b_doc"],      "Document Parsing  (Docling + PyMuPDF)"),
    (C["vision"],   C["b_vision"],   "Visual Intelligence  (Qwen2.5-VL-7B)"),
    (C["rag"],      C["b_rag"],      "RAG Index  (FAISS + BM25 + LightRAG)"),
    (C["draft"],    C["b_draft"],    "Section Writer  /  Chart Generator"),
    (C["judge"],    C["b_judge"],    "LLM-as-Judge  Quality Evaluator"),
    (C["decision"], C["b_dec"],      "Decision / Routing Point"),
    (C["refine"],   C["b_refine"],   "Iterative Refiner  (max 3 loops)"),
    (C["assemble"], C["b_assemble"], "Report Assembler  (HTML + PDF + MD)"),
    (C["output"],   C["b_output"],   "Session Output Folder"),
]

# 2-column legend
n_cols = 2
n_per_col = (len(legend_items) + 1) // 2
col_xs = [3.5, 13.5]
row_h = 0.48

for idx, (fc, ec, label) in enumerate(legend_items):
    col = idx // n_per_col
    row = idx % n_per_col
    lx = col_xs[col]
    ly = Y["legend"] - 0.05 - row * row_h
    rect = FancyBboxPatch((lx, ly - 0.14), 0.52, 0.36,
                          boxstyle="round,pad=0.04",
                          facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.68, ly + 0.04, label,
            fontsize=8, color=C["text"], va="center")

# Arrow legend below colour swatches
arr_y = Y["legend"] - n_per_col * row_h - 0.25
ax.annotate("", xy=(col_xs[0] + 0.52, arr_y),
            xytext=(col_xs[0], arr_y),
            arrowprops=dict(arrowstyle="-|>", color=C["arr_yes"], lw=1.6))
ax.text(col_xs[0] + 0.68, arr_y, "All pass → assemble report",
        fontsize=8, color=C["arr_yes"], va="center")

ax.annotate("", xy=(col_xs[1] + 0.52, arr_y),
            xytext=(col_xs[1], arr_y),
            arrowprops=dict(arrowstyle="-|>", color=C["arr_no"], lw=1.6))
ax.text(col_xs[1] + 0.68, arr_y, "Gaps found → refine (loop ≤ 3×)",
        fontsize=8, color=C["arr_no"], va="center")

# ── Footer ────────────────────────────────────────────────────────────────────
hline(ax, 1.2, x1=0.5, x2=23.5, color="#222233", lw=0.8)
ax.text(CX, Y["footer"],
        "ASTRA  ·  Multi-Agent Deep Research Pipeline  ·  "
        "LangGraph 8-node FSM  ·  vLLM (Qwen3-32B + Qwen2.5-VL-7B)  ·  "
        "BAAI/bge-m3  ·  FAISS + BM25  ·  Gradio  ·  weasyprint",
        ha="center", va="center", fontsize=8, color="#445566", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
import os
out = "/mnt/data1/peter/repos/ASTRA_OS/docs/astra_architecture.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout(pad=0.1)
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
