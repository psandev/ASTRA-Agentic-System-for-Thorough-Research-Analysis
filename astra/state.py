"""
ASTRA LangGraph State — TypedDict for the research session state.
All nodes read/write this shared state object.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional
from typing_extensions import TypedDict, NotRequired


class ResearchSource(TypedDict):
    url: str
    title: str
    snippet: str
    source_type: str          # "web" | "arxiv" | "s2" | "github" | "pubmed" | "openalex"
    raw_content: NotRequired[str]
    pdf_url: NotRequired[str]
    year: NotRequired[int]
    citation_count: NotRequired[int]
    authors: NotRequired[list[str]]


class ProcessedDocument(TypedDict):
    source_url: str
    markdown: str
    tables: list[dict]
    figures: list[dict]
    formulas: list[str]
    chunks: list[dict]        # [{id, text, metadata}]
    page_count: int
    processing_method: str    # "docling" | "pymupdf" | "jina"


class EvaluationResult(TypedDict):
    section_title: str
    factual_accuracy: float
    citation_faithfulness: float
    completeness: float
    coherence: float
    visual_richness: float
    relevance: float
    overall_score: float
    critique: str
    needs_refinement: bool
    gap_queries: list[str]


class ReportSection(TypedDict):
    title: str
    markdown: str
    citations: list[dict]
    word_count: int
    chart_paths: NotRequired[list[str]]
    table_data: NotRequired[list[dict]]


class AstraState(TypedDict):
    """Main LangGraph state object — persists across all nodes."""

    # ── Session metadata ──────────────────────────────────────────────────────
    session_id: str
    iteration: int                          # refinement loop counter
    status: str                             # "running" | "evaluating" | "refining" | "done" | "error"
    error_message: NotRequired[str]

    # ── Layer 1: Query Intelligence ──────────────────────────────────────────
    original_query: str
    enriched_query: NotRequired[str]          # rewritten, more precise query from enrich_query
    expertise_level: NotRequired[str]         # "beginner" | "intermediate" | "expert"
    purpose: NotRequired[str]                 # "academic" | "professional" | "learning" | "building" | "policy"
    implicit_needs: NotRequired[list[str]]    # inferred implicit requirements
    clarifying_questions: NotRequired[list[str]]
    user_clarifications: NotRequired[dict[str, str]]
    sub_queries: list[str]
    section_outline: list[str]
    source_priorities: dict[str, Any]
    research_plan: dict[str, Any]

    # ── Layer 2: Crawled Sources ─────────────────────────────────────────────
    collected_sources: list[ResearchSource]
    downloaded_pdf_paths: list[str]

    # ── Layer 3: Processed Documents + Visual Intelligence ───────────────────
    processed_documents: list[ProcessedDocument]
    extracted_figures: NotRequired[list[dict]]   # {image_path, caption, description, figure_type, source_url, ...}
    figure_chunks: NotRequired[list[dict]]        # RAG text chunks derived from figure descriptions

    # ── Layer 4: Knowledge Base (stored as session IDs for FAISS/BM25) ───────
    kb_collection: str                      # e.g. "astra_<session_id[:8]>"
    kb_chunk_count: int
    all_chunks: list[dict]                  # [{id, text, metadata}]

    # ── Layer 5/6: Draft Report ───────────────────────────────────────────────
    draft_sections: dict[str, ReportSection]
    evaluation_results: dict[str, EvaluationResult]

    # ── Layer 7: Refinement ───────────────────────────────────────────────────
    gap_sections: list[str]                 # section titles needing refinement
    converged: bool

    # ── Session output folder ──────────────────────────────────────────────────
    session_output_dir: NotRequired[str]   # e.g. .../sessions/hybrid_rag_1234567/

    # ── Final Output ──────────────────────────────────────────────────────────
    final_pdf_path: NotRequired[str]
    final_html_path: NotRequired[str]
    final_md_path: NotRequired[str]
    final_word_count: NotRequired[int]


def new_session(query: str) -> AstraState:
    """Create a fresh state for a new research session."""
    session_id = str(uuid.uuid4())
    return AstraState(
        session_id=session_id,
        iteration=0,
        status="running",
        original_query=query,
        sub_queries=[],
        section_outline=[],
        source_priorities={},
        research_plan={},
        collected_sources=[],
        downloaded_pdf_paths=[],
        processed_documents=[],
        kb_collection=f"astra_{session_id[:8]}",
        kb_chunk_count=0,
        all_chunks=[],
        draft_sections={},
        evaluation_results={},
        gap_sections=[],
        converged=False,
    )
