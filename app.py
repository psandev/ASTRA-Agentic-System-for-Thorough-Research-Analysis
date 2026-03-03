"""
ASTRA Gradio UI — Research interface for the multi-agent pipeline.

Two-Phase Design:
  Phase 1 — "Analyze Query": runs only Layer 1 (enrich + expand).
             Shows enriched query, section outline, clarifying questions.
             User can edit the plan before running.
  Phase 2 — "Run Research": takes Phase 1 outputs and runs the full pipeline.

Features:
  - Real-time progress tracking via status updates
  - Download links for PDF + Markdown output
  - LangSmith trace link per session
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator

import gradio as gr
from loguru import logger

# Load env before anything else
from dotenv import load_dotenv
load_dotenv()

from astra.config import get_config
from astra.graph import build_graph, run_query_analysis
from astra.state import new_session
from astra.utils.tracing import setup_langsmith


# ─── Setup ────────────────────────────────────────────────────────────────────

cfg = get_config()
setup_langsmith()

# Pre-compile the graph (loads models) lazily
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Status messages ──────────────────────────────────────────────────────────

STATUS_MESSAGES = {
    "running": "🔍 Analyzing query and building research plan...",
    "crawling": "🌐 Crawling web, academic, and repository sources...",
    "processing": "📄 Extracting content from documents...",
    "indexing": "🗂️ Building knowledge base (FAISS + BM25)...",
    "drafting": "✍️ Writing report sections with RAG retrieval...",
    "evaluating": "⚖️ Evaluating quality with LLM Judge...",
    "refining": "🔄 Running targeted refinement for failing sections...",
    "assembling": "📋 Assembling final PDF and Markdown report...",
    "done": "✅ Research complete!",
    "error": "❌ Error occurred",
}


# ─── Phase 1: Query Analysis ──────────────────────────────────────────────────

def analyze_query_streaming(
    query: str,
) -> Generator:
    """
    Run Layer 1 only (enrich + expand) and yield results.

    Yields: (enriched_query, section_outline_text, cq1_update, cq2_update)
    where cq*_update is a gr.update with the question as label, empty value.
    """
    if not query.strip():
        yield ("", "", gr.update(label="Clarifying Question 1 (optional answer)", value=""),
               gr.update(label="Clarifying Question 2 (optional answer)", value=""))
        return

    yield ("⏳ Analyzing query...", "",
           gr.update(label="Clarifying Question 1 (optional answer)", value=""),
           gr.update(label="Clarifying Question 2 (optional answer)", value=""))

    result_container: dict = {}
    error_container: dict = {}

    def _run():
        try:
            result_container.update(run_query_analysis(query))
        except Exception as e:
            error_container["error"] = str(e)
            logger.exception(f"[UI] Query analysis failed: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=120)

    if error_container.get("error"):
        yield (f"❌ {error_container['error']}", "",
               gr.update(label="Clarifying Question 1 (optional answer)", value=""),
               gr.update(label="Clarifying Question 2 (optional answer)", value=""))
        return

    enriched = result_container.get("enriched_query", query)
    outline = result_container.get("section_outline", [])
    cqs = result_container.get("clarifying_questions") or []

    outline_text = "\n".join(outline)
    cq1_label = f"Your answer (Q1: {cqs[0]})" if len(cqs) > 0 else "Clarifying Question 1 (optional answer)"
    cq2_label = f"Your answer (Q2: {cqs[1]})" if len(cqs) > 1 else "Clarifying Question 2 (optional answer)"

    yield (
        enriched,
        outline_text,
        gr.update(label=cq1_label, value=""),
        gr.update(label=cq2_label, value=""),
    )


# ─── Phase 2: Full Research Pipeline (streaming) ─────────────────────────────

def run_research_streaming(
    query: str,
    enriched_query: str,
    section_outline_text: str,
    cq1_answer: str,
    cq2_answer: str,
) -> Generator[tuple[str, str, str, str, str], None, None]:
    """
    Run the full ASTRA pipeline, feeding in Phase 1 results.

    Yields: (status_text, progress_log, pdf_path, md_path, trace_url)
    """
    effective_query = enriched_query.strip() if enriched_query.strip() else query.strip()
    if not effective_query:
        yield ("⚠️ Please enter a research query.", "", None, None, "")
        return

    # Parse section outline override
    custom_outline: list[str] = []
    if section_outline_text.strip():
        custom_outline = [
            line.strip() for line in section_outline_text.strip().splitlines()
            if line.strip()
        ]

    # Build clarification dict from Phase 1 question answers
    clarif_dict: dict = {}
    if cq1_answer.strip():
        clarif_dict["clarification_1"] = cq1_answer.strip()
    if cq2_answer.strip():
        clarif_dict["clarification_2"] = cq2_answer.strip()

    session = new_session(effective_query)
    session_id = session["session_id"]

    # Inject Phase 1 section outline if user provided/edited one
    if custom_outline:
        session["section_outline"] = custom_outline

    if clarif_dict:
        session["user_clarifications"] = clarif_dict

    # Also store the original query so the pipeline knows the true origin
    session["original_query"] = query.strip() if query.strip() else effective_query

    trace_url = "https://smith.langchain.com/projects/ASTRA-Research-Agent"
    log_lines: list[str] = []

    # Create per-session log directory
    from astra.config import get_config as _gc
    _cfg = _gc()
    session_dir = _cfg.create_session_dir(effective_query)
    session["session_output_dir"] = str(session_dir)

    from astra.utils.rich_log import add_session_log
    add_session_log(str(session_dir / "logs" / "session.log"))

    yield (
        f"🚀 Starting research session `{session_id[:8]}`...",
        f"Session ID: {session_id}\nQuery: {effective_query}\n",
        None,
        None,
        trace_url,
    )

    result_container: dict = {}
    error_container: dict = {}
    progress_container: list[str] = []

    def _run():
        try:
            graph = _get_graph()
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": cfg.astra_langgraph_recursion_limit,
            }

            for chunk in graph.stream(session, config=config):
                for node_name, node_output in chunk.items():
                    status = node_output.get("status", "running")
                    msg = STATUS_MESSAGES.get(status, f"Processing: {node_name}")
                    progress_container.append(
                        f"[{time.strftime('%H:%M:%S')}] {node_name}: {msg}"
                    )

            result_container.update(graph.get_state(config).values)
        except Exception as e:
            error_container["error"] = str(e)
            logger.exception(f"[UI] Research failed: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    last_log_len = 0
    last_yield_time = time.time()
    current_status = "🔄 Processing..."
    while thread.is_alive():
        if len(progress_container) > last_log_len:
            new_lines = progress_container[last_log_len:]
            log_lines.extend(new_lines)
            last_log_len = len(progress_container)
            current_status = new_lines[-1] if new_lines else current_status
            last_yield_time = time.time()
            yield (
                current_status,
                "\n".join(log_lines[-30:]),
                None,
                None,
                trace_url,
            )
        elif time.time() - last_yield_time >= 10:
            # Heartbeat: keep the streaming connection alive
            last_yield_time = time.time()
            yield (
                current_status,
                "\n".join(log_lines[-30:]),
                None,
                None,
                trace_url,
            )
        time.sleep(1)

    if error_container.get("error"):
        error_msg = error_container["error"]
        yield (
            f"❌ Error: {error_msg}",
            "\n".join(log_lines) + f"\n\nERROR: {error_msg}",
            None,
            None,
            trace_url,
        )
        return

    final_state = result_container
    pdf_path = final_state.get("final_pdf_path", "")
    md_path = final_state.get("final_md_path", "")
    word_count = final_state.get("final_word_count", 0)

    log_lines.extend(progress_container[last_log_len:])
    log_lines.append(f"\n✅ Complete! {word_count:,} words")
    if pdf_path:
        log_lines.append(f"📄 PDF: {pdf_path}")
    if md_path:
        log_lines.append(f"📝 Markdown: {md_path}")

    yield (
        f"✅ Research complete! {word_count:,} words in "
        f"{len(final_state.get('draft_sections', {}))} sections.",
        "\n".join(log_lines),
        pdf_path if pdf_path and Path(pdf_path).exists() else None,
        md_path if md_path and Path(md_path).exists() else None,
        trace_url,
    )


# ─── Gradio UI definition ─────────────────────────────────────────────────────

_UI_CSS = """
.title { text-align: center; }
.subtitle { text-align: center; color: #666; }
.status-box { font-family: monospace; font-size: 0.9em; }
.phase-header { font-weight: bold; color: #333; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ASTRA — Deep Research Agent") as demo:

        # Header
        gr.Markdown(
            """
            # 🔭 ASTRA — Agentic System for Thorough Research & Analysis
            **Autonomous deep research powered by Qwen3-32B + LangGraph + 7-layer multi-agent pipeline**

            *Sources: DuckDuckGo · Tavily · arXiv · Semantic Scholar · OpenAlex · GitHub · PubMed · Wikipedia*
            """,
            elem_classes=["title"],
        )

        with gr.Row():
            with gr.Column(scale=3):

                # ── Phase 1: Query Input ──────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### 📝 Phase 1 — Query Analysis")
                    query_input = gr.Textbox(
                        label="Enter your research topic or question",
                        placeholder=(
                            "e.g., 'What are the latest advances in "
                            "multi-agent LLM systems for scientific research?'"
                        ),
                        lines=3,
                        max_lines=6,
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze Query", variant="secondary", size="lg"
                    )

                # ── Phase 1 Results (shown after analysis) ────────────────────
                with gr.Group(visible=True) as phase1_results:
                    gr.Markdown("### 🧠 Query Analysis Results")
                    enriched_query_box = gr.Textbox(
                        label="Enriched Query (editable — this is what ASTRA will research)",
                        lines=3,
                        interactive=True,
                        placeholder="Enriched query will appear here after analysis...",
                    )
                    section_outline_box = gr.Textbox(
                        label="Section Outline (editable — one section title per line)",
                        lines=8,
                        interactive=True,
                        placeholder=(
                            "Section outline will appear here.\n"
                            "Edit to customize the report structure."
                        ),
                    )
                    with gr.Row():
                        cq1_box = gr.Textbox(
                            label="Clarifying Question 1 (optional answer)",
                            lines=2,
                            interactive=True,
                            placeholder="Clarifying question from ASTRA will appear here...",
                        )
                        cq2_box = gr.Textbox(
                            label="Clarifying Question 2 (optional answer)",
                            lines=2,
                            interactive=True,
                            placeholder="Clarifying question from ASTRA will appear here...",
                        )

                # ── Phase 2: Run Research ─────────────────────────────────────
                with gr.Row():
                    run_btn = gr.Button(
                        "🚀 Run Research", variant="primary", size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

                # Example queries
                gr.Markdown("#### 💡 Example Queries")
                gr.Examples(
                    examples=[
                        ["What are the state-of-the-art techniques in agentic RAG systems?"],
                        ["Survey of open-source LLM quantization methods for inference optimization"],
                        ["Recent advances in multimodal document understanding and extraction"],
                        ["Graph neural networks for drug discovery: methods and benchmarks"],
                    ],
                    inputs=[query_input],
                )

            with gr.Column(scale=2):
                # Status panel
                with gr.Group():
                    gr.Markdown("### 📊 Research Progress")
                    status_display = gr.Textbox(
                        label="Current Status",
                        value="Waiting for query...",
                        interactive=False,
                        elem_classes=["status-box"],
                    )
                    progress_log = gr.Textbox(
                        label="Pipeline Log",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                        elem_classes=["status-box"],
                    )

        # Output panel
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Downloads")
                pdf_download = gr.File(label="Download Report (PDF)", visible=True)
                md_download = gr.File(label="Download Report (Markdown)", visible=True)

            with gr.Column():
                gr.Markdown("### 🔍 Observability")
                trace_link = gr.Textbox(
                    label="LangSmith Trace URL",
                    interactive=False,
                    placeholder="Trace URL will appear here...",
                )
                gr.Markdown(
                    """
                    **LangSmith Tracing** is enabled for all pipeline runs.
                    Every node, tool call, and token is traced automatically.

                    Project: `ASTRA-Research-Agent`
                    """
                )

        # Pipeline info
        with gr.Accordion("ℹ️ ASTRA Pipeline Architecture", open=False):
            gr.Markdown(
                """
                | Layer | Component | Tools |
                |-------|-----------|-------|
                | 1 | Query Intelligence | enrich_query, query_expand, plan_research |
                | 2 | Multi-Source Crawlers | DuckDuckGo, Tavily, Jina, Firecrawl, arXiv, S2, OpenAlex, PubMed, GitHub, Wikipedia |
                | 3 | Document Processor | Docling (GPU), PyMuPDF, Qwen2.5-VL |
                | 4 | Agentic RAG Engine | FAISS, BM25, LightRAG, bge-m3, bge-reranker |
                | 5 | LLM Judge / QA | evaluate_section, deepeval, flag_gaps |
                | 6 | Report Writer | write_section, Plotly, Matplotlib, weasyprint PDF |
                | 7 | Refinement Loop | gap_analysis, re-research, check_convergence |

                **Models:** Qwen3-32B-AWQ (vLLM:8000) · Qwen2.5-VL-7B (vLLM:8001) · BAAI/bge-m3 · BAAI/bge-reranker-v2-m3
                """
            )

        # ── Event handlers ────────────────────────────────────────────────────

        # Phase 1: Analyze button
        analyze_btn.click(
            fn=analyze_query_streaming,
            inputs=[query_input],
            outputs=[enriched_query_box, section_outline_box, cq1_box, cq2_box],
            show_progress="minimal",
        )

        # Phase 2: Run Research button
        run_btn.click(
            fn=run_research_streaming,
            inputs=[
                query_input,
                enriched_query_box,
                section_outline_box,
                cq1_box,
                cq2_box,
            ],
            outputs=[status_display, progress_log, pdf_download, md_download, trace_link],
            show_progress="minimal",
        )

        # Clear button
        def _clear():
            return (
                "",  # query_input
                "",  # enriched_query_box
                "",  # section_outline_box
                gr.update(label="Clarifying Question 1 (optional answer)", value=""),
                gr.update(label="Clarifying Question 2 (optional answer)", value=""),
                "Waiting for query...",  # status_display
                "",  # progress_log
                None,  # pdf_download
                None,  # md_download
                "",   # trace_link
            )

        clear_btn.click(
            fn=_clear,
            outputs=[
                query_input,
                enriched_query_box,
                section_outline_box,
                cq1_box,
                cq2_box,
                status_display,
                progress_log,
                pdf_download,
                md_download,
                trace_link,
            ],
        )

    return demo


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting ASTRA Gradio UI...")
    demo = build_ui()
    demo.queue(max_size=5)
    demo.launch(
        server_name=cfg.astra_gradio_host,
        server_port=cfg.astra_gradio_port,
        share=cfg.astra_gradio_share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_UI_CSS,
        allowed_paths=[cfg.astra_sessions_dir, cfg.astra_output_dir],
    )
