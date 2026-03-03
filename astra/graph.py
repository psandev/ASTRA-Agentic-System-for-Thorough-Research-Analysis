"""
ASTRA LangGraph Workflow — 8-step research pipeline.

Nodes:
  1. query_intelligence  → Layer 1: expand & plan
  2. parallel_crawl      → Layer 2: multi-source crawling
  3. process_documents   → Layer 3: document extraction
  4. index_knowledge     → Layer 4: FAISS + BM25 indexing
  5. draft_report        → Layer 5+6: RAG-retrieval + section writing
  6. evaluate_quality    → Layer 5: LLM judge scoring
  7. refine_if_needed    → Layer 7: conditional refinement
  8. assemble_report     → Layer 6: DOCX + MD assembly

Conditional edge: evaluate_quality → refine_if_needed | assemble_report
Refinement edge:  refine_if_needed → evaluate_quality (loop ≤ 3 times)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from astra.config import get_config
from astra.state import AstraState, new_session
from astra.utils.tracing import setup_langsmith, traced_node


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1: Query Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def node_query_intelligence(state: AstraState) -> dict:
    """Layer 1 — Enrich query, then expand into research plan."""
    from astra.tools.layer1_query import enrich_query, query_expand

    session_id = state["session_id"]
    query = state["original_query"]
    clarifications = state.get("user_clarifications")

    logger.info(f"[Node 1] Query Intelligence: '{query[:80]}'")

    # Step 1a: Enrich the query (infer expertise, purpose, implicit needs)
    enrichment = enrich_query.invoke({"query": query})
    enriched_query = enrichment.get("enriched_query", query)
    enrichment_notes = enrichment.get("enrichment_notes", [])
    implicit_needs = enrichment.get("implicit_needs", [])

    logger.info(
        f"[Node 1] Enriched query: '{enriched_query[:80]}' "
        f"(expertise={enrichment.get('expertise_level')}, "
        f"purpose={enrichment.get('purpose')})"
    )

    # Step 1b: Expand the enriched query into sub-queries + section outline
    # Pass enrichment_notes as clarifications to guide section generation
    expand_clarifications = dict(clarifications) if clarifications else {}
    if enrichment_notes:
        expand_clarifications["enrichment_notes"] = "; ".join(enrichment_notes)
    if implicit_needs:
        expand_clarifications["implicit_needs"] = "; ".join(implicit_needs)

    expanded = query_expand.invoke(
        {
            "query": enriched_query,
            "user_clarifications": expand_clarifications or None,
        }
    )

    # Step 1c: Check if clarification is needed
    clarifying_questions = expanded.get("clarifying_questions")

    # Step 1d: Cap sections and sub-queries for performance
    cfg = get_config()
    sub_queries = expanded.get("sub_queries", [enriched_query])[:cfg.astra_max_sub_queries]
    section_outline = expanded.get("section_outline", [])[:cfg.astra_max_sections]

    return {
        "enriched_query": enriched_query,
        "expertise_level": enrichment.get("expertise_level", "intermediate"),
        "purpose": enrichment.get("purpose", "academic"),
        "implicit_needs": implicit_needs,
        "sub_queries": sub_queries,
        "section_outline": section_outline,
        "source_priorities": expanded.get("source_priorities", {}),
        "research_plan": expanded,
        "clarifying_questions": clarifying_questions,
        "status": "crawling",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2: Parallel Crawling
# ─────────────────────────────────────────────────────────────────────────────

def node_parallel_crawl(state: AstraState) -> dict:
    """Layer 2 — Launch parallel crawler agents for all source types."""
    from astra.tools.layer2_crawlers import (
        arxiv_search,
        duckduckgo_search,
        tavily_search,
        github_search,
        jina_fetch_url,
        openalex_search,
        semantic_scholar_search,
        medium_search,
        substack_search,
        papers_with_code_search,
        huggingface_blog_search,
        research_blogs_search,
        wikipedia_search,
        deduplicate_sources,
    )

    session_id = state["session_id"]
    sub_queries = state.get("sub_queries", [state["original_query"]])
    source_priorities = state.get("source_priorities", {})

    logger.info(
        f"[Node 2] Parallel crawl: {len(sub_queries)} queries "
        f"across web + academic + repos"
    )

    cfg = get_config()
    max_workers = min(cfg.astra_max_concurrent_crawlers, 12)

    web_results: list[dict] = []
    academic_results: list[dict] = []
    repo_results: list[dict] = []

    def crawl_web_single(query: str) -> list[dict]:
        """Tavily search (primary) with DDG sequential fallback."""
        try:
            results = tavily_search.invoke({"query": query, "max_results": 10})
            if results:
                return results
        except Exception as e:
            logger.warning(f"[Node 2] Tavily failed for '{query[:40]}': {e}")
        # Fallback to DDG (sequential, no Jina enrichment to avoid blocking)
        try:
            return duckduckgo_search.invoke({"query": query, "max_results": 10})
        except Exception as e:
            logger.warning(f"[Node 2] DDG failed for '{query[:40]}': {e}")
            return []

    def crawl_academic(query: str) -> list[dict]:
        results = []
        try:
            results.extend(arxiv_search.invoke({"query": query, "max_results": 10}))
        except Exception as e:
            logger.warning(f"[Node 2] arXiv failed: {e}")
        try:
            results.extend(openalex_search.invoke({"query": query, "max_results": 5}))
        except Exception as e:
            logger.warning(f"[Node 2] OpenAlex failed: {e}")
        return results

    def crawl_repos(query: str) -> list[dict]:
        try:
            return github_search.invoke({"query": query, "max_results": 5})
        except Exception as e:
            logger.warning(f"[Node 2] GitHub failed: {e}")
            return []

    # ── Web: run DDG queries sequentially to avoid rate-limit blocks ──────────
    # Use only the main query + up to 1 more sub-query for web
    web_queries = [state["original_query"]] + sub_queries[:1]
    for q in web_queries:
        try:
            web_results.extend(crawl_web_single(q))
        except Exception as e:
            logger.warning(f"[Node 2] Web crawl error: {e}")

    # ── Academic + Repos: run in parallel (arXiv/OpenAlex/GitHub handle rate limits) ──
    specialist_results: list[dict] = []

    def crawl_specialist(query: str) -> list[dict]:
        """Medium, Substack, Papers With Code, HF Blog, research blogs, Wikipedia."""
        out: list[dict] = []
        try:
            out.extend(medium_search.invoke({"query": query, "max_results": 5}))
        except Exception:
            pass
        try:
            out.extend(substack_search.invoke({"query": query, "max_results": 4}))
        except Exception:
            pass
        try:
            out.extend(papers_with_code_search.invoke({"query": query, "max_results": 5}))
        except Exception:
            pass
        try:
            out.extend(huggingface_blog_search.invoke({"query": query, "max_results": 4}))
        except Exception:
            pass
        try:
            out.extend(research_blogs_search.invoke({"query": query, "max_results": 5}))
        except Exception:
            pass
        try:
            out.extend(wikipedia_search.invoke({"query": query, "max_results": 5}))
        except Exception:
            pass
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        acad_futures = [executor.submit(crawl_academic, q) for q in sub_queries[:5]]
        repo_futures = [executor.submit(crawl_repos, q) for q in sub_queries[:2]]
        # Only run specialist crawl for main query to avoid rate limits
        spec_future = executor.submit(crawl_specialist, state["original_query"])

        for f in acad_futures:
            try:
                academic_results.extend(f.result(timeout=60))
            except Exception as e:
                logger.warning(f"[Node 2] Academic crawler error: {e}")

        for f in repo_futures:
            try:
                repo_results.extend(f.result(timeout=30))
            except Exception as e:
                logger.warning(f"[Node 2] Repo crawler error: {e}")

        try:
            specialist_results.extend(spec_future.result(timeout=60))
        except Exception as e:
            logger.warning(f"[Node 2] Specialist crawler error: {e}")

    logger.info(
        f"[Node 2] Specialist sources: {len(specialist_results)} "
        f"(Medium/Substack/PWC/HFBlog/Blogs)"
    )

    # Aggregate and deduplicate
    all_results = web_results + academic_results + repo_results + specialist_results
    unique_results = deduplicate_sources(all_results)

    logger.info(
        f"[Node 2] Crawl complete: {len(unique_results)} unique sources "
        f"(web={len(web_results)}, academic={len(academic_results)}, "
        f"repos={len(repo_results)})"
    )

    return {
        "collected_sources": unique_results,
        "status": "processing",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3: Document Processing
# ─────────────────────────────────────────────────────────────────────────────

def node_process_documents(state: AstraState) -> dict:
    """Layer 3 — Extract text/tables/figures from all sources."""
    import httpx as _httpx
    from astra.tools.layer3_docs import batch_parse_pdfs, _pymupdf_parse
    from astra.tools.layer3_vision import process_visual_sources

    session_id = state["session_id"]
    sources = state.get("collected_sources", [])
    pdf_paths = state.get("downloaded_pdf_paths", [])
    session_output_dir = state.get("session_output_dir", "")

    logger.info(
        f"[Node 3] Processing: {len(pdf_paths)} PDFs + "
        f"{len(sources)} web sources"
    )

    cfg = get_config()
    processed: list[dict] = []

    # Process PDFs in batch
    if pdf_paths:
        parsed_pdfs = batch_parse_pdfs(pdf_paths, max_workers=8)
        for doc in parsed_pdfs:
            processed.append(
                {
                    "source_url": doc.get("path", ""),
                    "markdown": doc.get("markdown", ""),
                    "tables": doc.get("tables", []),
                    "figures": doc.get("figures", []),
                    "formulas": doc.get("formulas", []),
                    "chunks": [],
                    "page_count": doc.get("page_count", 0),
                    "processing_method": doc.get("metadata", {}).get("tool", "docling"),
                }
            )

    # Process web sources — also fetch raw HTML for a subset to enable image scraping
    web_html_fetched = 0
    for source in sources:
        raw = source.get("raw_content", "")
        if not raw:
            raw = source.get("snippet", "")
        if not raw:
            continue

        raw_html = ""
        src_url = source.get("url", "")
        source_type = source.get("source_type", "web")
        # Fetch raw HTML for the first 8 content-rich web sources
        if (
            web_html_fetched < 8
            and source_type in ("web", "medium", "substack", "huggingface", "blog")
            and src_url.startswith("http")
            and not src_url.endswith(".pdf")
        ):
            try:
                r = _httpx.get(
                    src_url, timeout=10, follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ASTRA-Bot/1.0)"},
                )
                if "html" in r.headers.get("content-type", ""):
                    raw_html = r.text
                    web_html_fetched += 1
            except Exception:
                pass

        processed.append(
            {
                "source_url": src_url,
                "markdown": raw,
                "raw_html": raw_html,
                "tables": [],
                "figures": [],
                "formulas": [],
                "chunks": [],
                "page_count": 1,
                "processing_method": "jina",
            }
        )

    logger.info(
        f"[Node 3] Processed {len(processed)} documents "
        f"({web_html_fetched} with raw HTML for image scraping)"
    )

    # ── Visual Intelligence: extract figures from PDFs and web articles ───────
    extracted_figures: list[dict] = []
    figure_chunks: list[dict] = []
    if session_output_dir and cfg.astra_vision_enabled:
        try:
            extracted_figures, figure_chunks = process_visual_sources(
                sources=sources,
                processed_docs=processed,
                session_dir=session_output_dir,
            )
            logger.info(
                f"[Node 3] Vision pipeline: {len(extracted_figures)} figures, "
                f"{len(figure_chunks)} RAG chunks"
            )
        except Exception as e:
            logger.warning(f"[Node 3] Vision pipeline failed (non-fatal): {e}")

    return {
        "processed_documents": processed,
        "extracted_figures": extracted_figures,
        "figure_chunks": figure_chunks,
        "status": "indexing",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4: Knowledge Indexing
# ─────────────────────────────────────────────────────────────────────────────

def node_index_knowledge(state: AstraState) -> dict:
    """Layer 4 — Chunk all documents and build FAISS + BM25 indices."""
    from astra.tools.layer4_rag import build_index, chunk_text

    session_id = state["session_id"]
    collection = state.get("kb_collection", f"astra_{session_id[:8]}")
    documents = state.get("processed_documents", [])
    figure_chunks = state.get("figure_chunks", [])

    logger.info(
        f"[Node 4] Indexing {len(documents)} documents → collection='{collection}'"
    )

    cfg = get_config()
    all_chunks: list[dict] = []

    for doc in documents:
        text = doc.get("markdown", "")
        source_url = doc.get("source_url", "")
        if not text or len(text) < 50:
            continue

        chunks = chunk_text(
            text,
            source_url=source_url,
            chunk_size=cfg.astra_chunk_size,
            chunk_overlap=cfg.astra_chunk_overlap,
        )
        all_chunks.extend(chunks)

    # Include figure RAG chunks (VLM descriptions) from the vision pipeline
    if figure_chunks:
        all_chunks.extend(figure_chunks)
        logger.info(f"[Node 4] +{len(figure_chunks)} figure RAG chunks added to index")

    # Build FAISS + BM25 index
    if all_chunks:
        build_index(all_chunks, collection)
        logger.info(
            f"[Node 4] Index complete: {len(all_chunks)} chunks in '{collection}'"
        )
    else:
        logger.warning("[Node 4] No chunks to index")

    return {
        "all_chunks": all_chunks,
        "kb_chunk_count": len(all_chunks),
        "status": "drafting",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5: Report Drafting
# ─────────────────────────────────────────────────────────────────────────────

def _try_generate_chart(
    section_title: str,
    markdown: str,
    charts_dir: str,
) -> str:
    """
    Detect the first Markdown table with numeric data and generate a bar chart.
    Returns the PNG path if successful, empty string otherwise.
    """
    import os

    lines = markdown.split("\n")
    header: list[str] = []
    data_rows: list[list[str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if data_rows:
                break
            continue
        if re.match(r"^\|[-| :]+\|$", stripped):
            continue  # Skip separator lines
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if not header:
            header = cells
        else:
            data_rows.append(cells)

    if not header or len(data_rows) < 2 or len(header) < 2:
        return ""

    # Find a column with numeric values
    x_data: list[str] = []
    y_data: list[float] = []
    y_label = ""

    for col_idx in range(1, len(header)):
        nums: list[float] = []
        labels: list[str] = []
        for row in data_rows:
            if col_idx >= len(row):
                continue
            raw = row[col_idx].replace("%", "").replace(",", "").strip()
            try:
                nums.append(float(raw))
                labels.append(row[0])
            except ValueError:
                break
        if len(nums) >= 2:
            x_data = labels
            y_data = nums
            y_label = header[col_idx]
            break

    if not x_data:
        return ""

    try:
        from astra.tools.layer6_report import generate_chart_matplotlib
        safe = re.sub(r"[^\w\-]", "_", section_title)[:40]
        out_base = os.path.join(charts_dir, f"chart_{safe}")
        result = generate_chart_matplotlib.invoke({
            "chart_type": "bar",
            "data": {
                "x": x_data,
                "y": y_data,
                "x_label": header[0],
                "y_label": y_label,
            },
            "title": f"{section_title} — {y_label}",
            "output_path": out_base,
        })
        return result.get("png_path", "")
    except Exception as e:
        logger.warning(f"[Node 5] Chart generation skipped for '{section_title}': {e}")
        return ""


def node_draft_report(state: AstraState) -> dict:
    """Layer 5+6 — RAG-retrieve + write each section in parallel; auto-generate charts."""
    from concurrent.futures import as_completed
    from astra.tools.layer4_rag import hybrid_retrieve, bge_rerank, figure_search as _figure_search, preload_models
    from astra.tools.layer6_report import write_section

    # Pre-warm embedding + reranker models in the main thread BEFORE spawning workers.
    # Without this, all 3 workers race on _get_reranker_model() simultaneously,
    # causing concurrent AutoModelForSequenceClassification.from_pretrained() calls
    # that corrupt the global singleton and produce meta-tensor errors at inference.
    preload_models()

    session_id = state["session_id"]
    section_outline = state.get("section_outline", [])
    collection = state.get("kb_collection", f"astra_{session_id[:8]}")
    sub_queries = state.get("sub_queries", [])
    session_output_dir = state.get("session_output_dir", "")
    charts_dir = (
        str(Path(session_output_dir) / "charts")
        if session_output_dir
        else str(get_config().get_chart_path(""))
    )

    if not section_outline:
        section_outline = [
            "Executive Summary",
            "Introduction & Background",
            "Main Analysis",
            "Key Findings",
            "Conclusions & Future Directions",
        ]

    cfg = get_config()
    logger.info(f"[Node 5] Drafting {len(section_outline)} sections (parallel, 3 workers)")

    def _write_one(args: tuple) -> tuple:
        i, section_title = args
        section_query = sub_queries[i] if i < len(sub_queries) else state["original_query"]

        # Retrieve relevant chunks
        candidates = hybrid_retrieve.invoke(
            {
                "query": section_query,
                "top_k": cfg.astra_retrieval_top_k,
                "collection": collection,
            }
        )

        # Rerank — ALWAYS rerank after hybrid_retrieve
        if candidates:
            reranked = bge_rerank.invoke(
                {
                    "query": section_query,
                    "candidates": candidates,
                    "top_k": cfg.astra_retrieval_rerank_top_k,
                }
            )
        else:
            reranked = []

        # Write section
        section_outline_hint = (
            f"Cover: {section_query}. "
            f"This is section {i+1} of {len(section_outline)}."
        )
        written = write_section.invoke(
            {
                "section_title": section_title,
                "section_outline": section_outline_hint,
                "retrieved_chunks": reranked if reranked else candidates,
            }
        )
        section_md = written.get("markdown", "")
        logger.info(
            f"[Node 5] Section {i+1}/{len(section_outline)}: '{section_title}' "
            f"— {written.get('word_count', 0)} words"
        )

        # Auto-generate chart if section contains a data table
        chart_path = _try_generate_chart(section_title, section_md, charts_dir)
        chart_paths = [chart_path] if chart_path else []
        if chart_path:
            logger.info(f"[Node 5] Chart generated for '{section_title}': {chart_path}")

        # Collect figure candidates (dedup applied after all sections finish)
        fig_candidates: list[dict] = []
        fig_results = _figure_search(
            section_query,
            top_k=4,
            collection=collection,
            min_score=cfg.astra_figure_relevance_threshold,
        )
        for fchunk in fig_results:
            meta = fchunk.get("metadata", {})
            img_path = meta.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue
            try:
                if Path(img_path).stat().st_size < 8_000:
                    continue
            except OSError:
                continue
            fig_candidates.append({
                "path": img_path,
                "caption": meta.get("caption", ""),
                "title": meta.get("title", ""),
                "description": fchunk.get("text", ""),
                "score": fchunk.get("score", 0.0),
            })

        return i, section_title, {
            "title": section_title,
            "markdown": section_md,
            "citations": written.get("citations", []),
            "word_count": written.get("word_count", 0),
            "chart_paths": chart_paths,
            "_fig_candidates": fig_candidates,
        }

    # Run 3 sections concurrently — vLLM batches concurrent requests efficiently
    indexed_outline = list(enumerate(section_outline))
    raw_results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_write_one, args): args for args in indexed_outline}
        for fut in as_completed(futures):
            try:
                raw_results.append(fut.result())
            except Exception as e:
                _i, _title = futures[fut]
                logger.error(f"[Node 5] Section '{_title}' failed: {e}")

    # Sort by original index so figure dedup follows document order
    raw_results.sort(key=lambda x: x[0])

    draft_sections: dict = {}
    _used_figure_paths: set[str] = set()
    for _i, section_title, data in raw_results:
        source_figures: list[dict] = []
        for fig in data.pop("_fig_candidates", []):
            img_path = fig["path"]
            if img_path not in _used_figure_paths:
                _used_figure_paths.add(img_path)
                source_figures.append(fig)
        if source_figures:
            logger.info(
                f"[Node 5] {len(source_figures)} figure(s) matched for '{section_title}' "
                f"(scores: {[round(s['score'], 2) for s in source_figures]})"
            )
        data["source_figures"] = source_figures
        draft_sections[section_title] = data

    total_words = sum(s.get("word_count", 0) for s in draft_sections.values())
    chart_count = sum(1 for s in draft_sections.values() if s.get("chart_paths"))
    logger.info(
        f"[Node 5] Draft complete: {len(draft_sections)} sections, "
        f"{total_words} total words, {chart_count} charts generated"
    )

    return {
        "draft_sections": draft_sections,
        "status": "evaluating",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6: Quality Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def node_evaluate_quality(state: AstraState) -> dict:
    """Layer 5 — Evaluate every section with the LLM judge (parallel, 4 workers).

    On iteration > 0, sections that already passed (needs_refinement=False) and
    whose markdown content has not changed since the previous evaluation are
    served from the cached result without calling the judge LLM.  This avoids
    re-scoring unchanged sections in every refinement loop — typically saving
    5–6 judge calls (≈20 000 tokens) per session.
    """
    from concurrent.futures import as_completed
    from astra.tools.layer4_rag import preload_models
    from astra.tools.layer5_judge import evaluate_section

    preload_models()  # ensure models are loaded before workers start

    draft_sections = state.get("draft_sections", {})
    sub_queries = state.get("sub_queries", [])
    all_chunks = state.get("all_chunks", [])
    iteration = state.get("iteration", 0)
    prev_eval = state.get("evaluation_results", {})  # results from previous iteration

    logger.info(
        f"[Node 6] Evaluating {len(draft_sections)} sections "
        f"(parallel, 4 workers, iteration {iteration})"
    )

    evaluation_results: dict = {}
    gap_sections: list[str] = []
    section_titles = list(draft_sections.keys())

    # ── Partition: cached vs needs fresh evaluation ───────────────────────────
    to_evaluate: list[tuple[int, str, str]] = []  # (idx, title, content_hash)
    n_cached = 0

    for i, title in enumerate(section_titles):
        content = draft_sections[title].get("markdown", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        prev = prev_eval.get(title, {})

        if (
            iteration > 0                               # not the first pass
            and prev                                    # has a previous result
            and not prev.get("needs_refinement", True)  # previously passed
            and prev.get("_content_hash") == content_hash  # content unchanged
        ):
            # Section passed last round and was not rewritten — reuse result
            evaluation_results[title] = prev
            n_cached += 1
            logger.info(
                f"[Node 6] '{title}': cache hit (hash={content_hash}) — skipping LLM eval"
            )
        else:
            to_evaluate.append((i, title, content_hash))

    if n_cached:
        logger.info(
            f"[Node 6] {n_cached}/{len(draft_sections)} sections served from cache"
        )

    # ── Evaluate remaining sections in parallel ───────────────────────────────
    def _evaluate_one(args: tuple) -> tuple:
        i, section_title, content_hash = args
        section = draft_sections[section_title]
        content = section.get("markdown", "")
        sub_query = sub_queries[i] if i < len(sub_queries) else state["original_query"]
        source_texts = [
            c["text"] for c in all_chunks[:20]
            if any(
                word.lower() in c["text"].lower()
                for word in sub_query.split()[:5]
            )
        ][:10]
        eval_result = evaluate_section.invoke(
            {
                "section_title": section_title,
                "section_content": content,
                "source_chunks": source_texts,
                "sub_query": sub_query,
            }
        )
        # Attach hash so the next iteration can detect unchanged sections
        eval_result["_content_hash"] = content_hash
        return section_title, eval_result

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(_evaluate_one, args): args[1]
            for args in to_evaluate
        }
        for fut in as_completed(futures):
            try:
                section_title, eval_result = fut.result()
                evaluation_results[section_title] = eval_result
                if eval_result.get("needs_refinement", False):
                    gap_sections.append(section_title)
            except Exception as e:
                section_title = futures[fut]
                logger.error(f"[Node 6] Evaluation failed for '{section_title}': {e}")

    logger.info(
        f"[Node 6] Evaluation complete: {len(gap_sections)}/{len(draft_sections)} "
        f"sections need refinement "
        f"({len(to_evaluate) - len(gap_sections)} passed, {n_cached} from cache)"
    )

    return {
        "evaluation_results": evaluation_results,
        "gap_sections": gap_sections,
        "status": "refining" if gap_sections else "assembling",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7: Iterative Refinement
# ─────────────────────────────────────────────────────────────────────────────

def node_refine(state: AstraState) -> dict:
    """Layer 7 — Targeted re-research and re-generation for failing sections."""
    from astra.tools.layer7_refinement import (
        gap_analysis,
        trigger_reresearch,
        update_knowledge_base,
        check_convergence,
    )
    from astra.tools.layer4_rag import hybrid_retrieve, bge_rerank
    from astra.tools.layer6_report import write_section

    session_id = state["session_id"]
    iteration = state.get("iteration", 0) + 1
    evaluation_results = state.get("evaluation_results", {})
    research_plan = state.get("research_plan", {})
    collection = state.get("kb_collection", f"astra_{session_id[:8]}")
    cfg = get_config()

    logger.info(f"[Node 7] Refinement iteration {iteration}/{cfg.eval_max_iterations}")

    # Check convergence
    current_scores = {
        k: v.get("overall_score", 0.0) for k, v in evaluation_results.items()
    }
    previous_scores = research_plan.get(f"scores_iteration_{iteration - 1}", {})

    convergence = check_convergence.invoke(
        {
            "current_scores": current_scores,
            "previous_scores": previous_scores or None,
            "iteration": iteration,
            "max_iterations": cfg.eval_max_iterations,
        }
    )

    if convergence["converged"]:
        logger.info(f"[Node 7] Converged: {convergence['reason']}")
        return {"iteration": iteration, "converged": True, "status": "assembling"}

    # Gap analysis
    gap_result = gap_analysis.invoke(
        {"evaluation_results": evaluation_results, "research_plan": research_plan}
    )
    gaps = gap_result.get("gaps", [])

    updated_sections = dict(state.get("draft_sections", {}))

    # ── Phase 1: Batch re-research (one pass for all gaps) ───────────────────
    # Collect and deduplicate gap queries across all failing sections
    all_gap_queries: list[str] = []
    seen_queries: set[str] = set()
    for gap in gaps:
        for q in gap.get("gap_queries", []):
            if q and q not in seen_queries:
                seen_queries.add(q)
                all_gap_queries.append(q)

    if all_gap_queries:
        logger.info(
            f"[Node 7] Batch re-research: {len(all_gap_queries)} unique queries "
            f"for {len(gaps)} failing sections"
        )
        reresearch_result = trigger_reresearch.invoke(
            {
                "gap_queries": all_gap_queries,
                "collection": collection,
                "max_results_per_query": 8,
            }
        )
        new_sources = reresearch_result.get("new_sources", [])
        if new_sources:
            new_docs = [
                {
                    "markdown": s.get("raw_content", s.get("snippet", s.get("abstract", ""))),
                    "url": s.get("url", ""),
                }
                for s in new_sources
                if s.get("raw_content") or s.get("snippet") or s.get("abstract")
            ]
            # Single KB update for all sections
            update_knowledge_base.invoke(
                {"new_documents": new_docs, "collection": collection}
            )
            logger.info(
                f"[Node 7] KB updated with {len(new_docs)} new docs "
                f"({len(new_sources)} raw sources)"
            )

    # ── Phase 2: Re-generate each failing section from the refreshed KB ──────
    from concurrent.futures import as_completed as _as_completed
    from astra.tools.layer4_rag import figure_search as _figure_search, preload_models

    preload_models()  # ensure models are ready before workers start

    # Pre-populate used figure paths from sections that are NOT being rewritten.
    _rewrite_titles = {gap["section"] for gap in gaps if gap.get("gap_queries")}
    _used_figure_paths: set[str] = set()
    for _sec_title, _sec_data in updated_sections.items():
        if _sec_title not in _rewrite_titles:
            for sfig in _sec_data.get("source_figures", []):
                if sfig.get("path"):
                    _used_figure_paths.add(sfig["path"])

    fig_threshold = cfg.astra_figure_relevance_threshold

    def _rewrite_one(gap: dict) -> tuple:
        section_title = gap["section"]
        gap_queries = gap.get("gap_queries", [])
        if not gap_queries:
            return section_title, None

        sub_query = gap_queries[0]
        candidates = hybrid_retrieve.invoke(
            {"query": sub_query, "top_k": 25, "collection": collection}
        )
        reranked = bge_rerank.invoke(
            {"query": sub_query, "candidates": candidates, "top_k": 12}
        ) if candidates else []

        rewritten = write_section.invoke(
            {
                "section_title": section_title,
                "section_outline": f"Refined coverage of: {', '.join(gap_queries)}. "
                                   f"Critique: {gap.get('critique', '')}",
                "retrieved_chunks": reranked if reranked else candidates,
            }
        )

        # Collect figure candidates (dedup applied after all rewrites finish)
        fig_candidates: list[dict] = []
        for fchunk in _figure_search(
            sub_query, top_k=4, collection=collection, min_score=fig_threshold
        ):
            meta = fchunk.get("metadata", {})
            img_path = meta.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue
            try:
                if Path(img_path).stat().st_size < 8_000:
                    continue
            except OSError:
                continue
            fig_candidates.append({
                "path": img_path,
                "caption": meta.get("caption", ""),
                "title": meta.get("title", ""),
                "description": fchunk.get("text", ""),
                "score": fchunk.get("score", 0.0),
            })

        return section_title, {
            "rewritten": rewritten,
            "gap": gap,
            "fig_candidates": fig_candidates,
        }

    active_gaps = [g for g in gaps if g.get("gap_queries")]
    rewrite_results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_rewrite_one, gap): gap["section"] for gap in active_gaps}
        for fut in _as_completed(futures):
            try:
                rewrite_results.append(fut.result())
            except Exception as e:
                logger.error(f"[Node 7] Rewrite failed for '{futures[fut]}': {e}")

    # Apply figure dedup and store results
    for section_title, data in rewrite_results:
        if data is None:
            continue
        rewritten = data["rewritten"]
        source_figures: list[dict] = []
        for fig in data["fig_candidates"]:
            img_path = fig["path"]
            if img_path not in _used_figure_paths:
                _used_figure_paths.add(img_path)
                source_figures.append(fig)
        fig_note = f" — {len(source_figures)} figure(s) re-matched" if source_figures else ""
        logger.info(
            f"[Node 7] Rewrote '{section_title}' "
            f"({rewritten.get('word_count', 0)} words){fig_note}"
        )
        updated_sections[section_title] = {
            "title": section_title,
            "markdown": rewritten.get("markdown", ""),
            "citations": rewritten.get("citations", []),
            "word_count": rewritten.get("word_count", 0),
            "chart_paths": updated_sections.get(section_title, {}).get("chart_paths", []),
            "source_figures": source_figures,
        }

    # Store current scores in research plan for convergence tracking
    research_plan[f"scores_iteration_{iteration}"] = current_scores

    return {
        "draft_sections": updated_sections,
        "iteration": iteration,
        "converged": False,
        "research_plan": research_plan,
        "status": "evaluating",
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 8: Document Assembly
# ─────────────────────────────────────────────────────────────────────────────

def node_assemble_report(state: AstraState) -> dict:
    """Layer 6 — Assemble final HTML + PDF + Markdown report in session folder."""
    from astra.tools.layer6_report import build_pdf, build_html, build_markdown

    session_id = state["session_id"]
    draft_sections = state.get("draft_sections", {})
    session_output_dir = state.get("session_output_dir", "")

    logger.info(f"[Node 8] Assembling report: {len(draft_sections)} sections")

    cfg = get_config()

    # Determine reports directory
    if session_output_dir:
        reports_dir = Path(session_output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        base_stem = reports_dir / "ASTRA_Report"
    else:
        import time
        ts = int(time.time())
        base_stem = Path(cfg.get_output_path(f"ASTRA_Report_{ts}"))

    # Build bibliography from collected sources
    collected_sources = state.get("collected_sources", [])
    bibliography: list[dict] = []
    seen_urls: set[str] = set()
    for src in collected_sources:
        url = src.get("url", src.get("pdf_url", ""))
        if url and url not in seen_urls:
            seen_urls.add(url)
            bibliography.append(
                {
                    "key": f"Source {len(bibliography)+1}",
                    "title": src.get("title", url),
                    "authors": src.get("authors", ["Unknown"]),
                    "year": src.get("year", "n.d."),
                    "url": url,
                }
            )

    sections_list = list(draft_sections.values())
    total_words = sum(s.get("word_count", 0) for s in sections_list)

    # Source figures are now placed inline within their sections by build_html
    # (matched per-section via RAG chunk retrieval in node_draft_report).
    # No separate appendix of figures is needed.
    report_figures: list[dict] = []

    # Build HTML (primary — with MathJax and active links)
    html_result = build_html.invoke(
        {
            "sections": sections_list,
            "figures": report_figures,
            "bibliography": bibliography,
            "output_path": str(base_stem) + ".html",
            "include_toc": True,
        }
    )

    # Build PDF
    pdf_result = build_pdf.invoke(
        {
            "sections": sections_list,
            "figures": report_figures,
            "bibliography": bibliography,
            "output_path": str(base_stem) + ".pdf",
            "include_toc": True,
        }
    )

    # Build Markdown
    md_result = build_markdown.invoke(
        {
            "sections": sections_list,
            "bibliography": bibliography,
            "output_path": str(base_stem) + ".md",
        }
    )

    logger.info(
        f"[Node 8] Report assembled: {total_words} words | "
        f"HTML={html_result.get('html_path', '')} | "
        f"PDF={pdf_result.get('pdf_path', '')} | "
        f"MD={md_result.get('md_path', '')}"
    )

    return {
        "final_html_path": html_result.get("html_path", ""),
        "final_pdf_path": pdf_result.get("pdf_path", ""),
        "final_md_path": md_result.get("md_path", ""),
        "final_word_count": total_words,
        "status": "done",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_evaluation(
    state: AstraState,
) -> Literal["refine", "assemble_report"]:
    """Decide whether to refine or assemble after quality evaluation."""
    cfg = get_config()
    gap_sections = state.get("gap_sections", [])
    iteration = state.get("iteration", 0)
    converged = state.get("converged", False)

    if converged or not gap_sections or iteration >= cfg.eval_max_iterations:
        return "assemble_report"
    return "refine"


def _route_after_refine(
    state: AstraState,
) -> Literal["evaluate_quality", "assemble_report"]:
    """After refinement, re-evaluate or go straight to assembly if converged."""
    converged = state.get("converged", False)
    cfg = get_config()
    iteration = state.get("iteration", 0)

    if converged or iteration >= cfg.eval_max_iterations:
        return "assemble_report"
    return "evaluate_quality"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the ASTRA LangGraph workflow."""
    setup_langsmith()

    cfg = get_config()

    graph = StateGraph(AstraState)

    # Register nodes
    graph.add_node("query_intelligence", node_query_intelligence)
    graph.add_node("parallel_crawl", node_parallel_crawl)
    graph.add_node("process_documents", node_process_documents)
    graph.add_node("index_knowledge", node_index_knowledge)
    graph.add_node("draft_report", node_draft_report)
    graph.add_node("evaluate_quality", node_evaluate_quality)
    graph.add_node("refine", node_refine)
    graph.add_node("assemble_report", node_assemble_report)

    # Linear edges
    graph.add_edge(START, "query_intelligence")
    graph.add_edge("query_intelligence", "parallel_crawl")
    graph.add_edge("parallel_crawl", "process_documents")
    graph.add_edge("process_documents", "index_knowledge")
    graph.add_edge("index_knowledge", "draft_report")
    graph.add_edge("draft_report", "evaluate_quality")

    # Conditional: evaluate → refine OR assemble
    graph.add_conditional_edges(
        "evaluate_quality",
        _route_after_evaluation,
        {"refine": "refine", "assemble_report": "assemble_report"},
    )

    # Conditional: refine → re-evaluate OR assemble
    graph.add_conditional_edges(
        "refine",
        _route_after_refine,
        {"evaluate_quality": "evaluate_quality", "assemble_report": "assemble_report"},
    )

    graph.add_edge("assemble_report", END)

    # Compile with checkpointing (try SQLite external package, fall back to MemorySaver)
    checkpointer = None
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver

        checkpoint_dir = Path(cfg.astra_langgraph_checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        import sqlite3
        conn = sqlite3.connect(cfg.astra_langgraph_sqlite_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("[Graph] Using SQLite checkpointer")
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            logger.info("[Graph] Using MemorySaver checkpointer (SQLite not installed)")
        except Exception as e:
            logger.warning(f"[Graph] No checkpointer available ({e}) — compiling without")

    compiled = graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()
    logger.info("[Graph] Compiled successfully")

    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for programmatic use
# ─────────────────────────────────────────────────────────────────────────────

def run_query_analysis(
    query: str,
    clarifications: dict | None = None,
) -> dict:
    """
    Run only the query intelligence step (Layer 1: enrich + expand + plan).
    Used by the two-phase Gradio UI "Analyze Query" button.

    Returns a dict with:
      enriched_query, expertise_level, purpose, implicit_needs,
      section_outline, sub_queries, clarifying_questions
    """
    from astra.tools.layer1_query import enrich_query, query_expand

    logger.info(f"[ASTRA] run_query_analysis: '{query[:80]}'")

    enrichment = enrich_query.invoke({"query": query})
    enriched_query = enrichment.get("enriched_query", query)
    enrichment_notes = enrichment.get("enrichment_notes", [])
    implicit_needs = enrichment.get("implicit_needs", [])

    expand_clarifications = dict(clarifications) if clarifications else {}
    if enrichment_notes:
        expand_clarifications["enrichment_notes"] = "; ".join(enrichment_notes)
    if implicit_needs:
        expand_clarifications["implicit_needs"] = "; ".join(implicit_needs)

    expanded = query_expand.invoke(
        {
            "query": enriched_query,
            "user_clarifications": expand_clarifications or None,
        }
    )

    return {
        "enriched_query": enriched_query,
        "original_query": query,
        "expertise_level": enrichment.get("expertise_level", "intermediate"),
        "purpose": enrichment.get("purpose", "academic"),
        "implicit_needs": implicit_needs,
        "enrichment_notes": enrichment_notes,
        "section_outline": expanded.get("section_outline", []),
        "sub_queries": expanded.get("sub_queries", []),
        "clarifying_questions": expanded.get("clarifying_questions") or [],
    }


def run_research(query: str, clarifications: dict | None = None) -> AstraState:
    """
    Execute a full ASTRA research session.

    Args:
        query: The user's research query.
        clarifications: Optional answers to clarifying questions.

    Returns:
        Final AstraState with report paths.
    """
    from astra.utils.rich_log import add_session_log, print_evaluation_scores

    cfg = get_config()

    # Create per-query session output directory
    session_dir = cfg.create_session_dir(query)
    session_dir_str = str(session_dir)

    # Add per-session log file
    session_log = str(session_dir / "logs" / "session.log")
    add_session_log(session_log)

    graph = build_graph()
    initial_state = new_session(query)
    initial_state["session_output_dir"] = session_dir_str
    if clarifications:
        initial_state["user_clarifications"] = clarifications

    config = {
        "configurable": {"thread_id": initial_state["session_id"]},
        "recursion_limit": cfg.astra_langgraph_recursion_limit,
    }

    logger.info(f"[ASTRA] Starting research session: {initial_state['session_id']}")
    logger.info(f"[ASTRA] Query: '{query}'")
    logger.info(f"[ASTRA] Session dir: {session_dir_str}")

    start_time = time.time()
    final_state = graph.invoke(initial_state, config=config)
    elapsed = time.time() - start_time

    # Print evaluation scores as Rich table
    if final_state.get("evaluation_results"):
        print_evaluation_scores(final_state["evaluation_results"])

    logger.info(
        f"[ASTRA] Research complete in {elapsed:.1f}s — "
        f"words={final_state.get('final_word_count', 0)}, "
        f"html={final_state.get('final_html_path', '')}"
    )
    return final_state
