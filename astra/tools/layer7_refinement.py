"""
ASTRA Layer 7 — Refinement Meta-Loop
Action paradigm: HYBRID (JSON for scores, code for targeted crawl)

Tools:
  gap_analysis, trigger_reresearch, update_knowledge_base, check_convergence

Controls the iterative self-refinement loop (max 3 iterations).
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool
from loguru import logger

from astra.config import get_config


# ─── gap_analysis ─────────────────────────────────────────────────────────────

@tool
def gap_analysis(
    evaluation_results: dict[str, dict],
    research_plan: dict,
) -> dict:
    """
    Layer 7: Identify sections needing refinement based on judge scores.

    Args:
        evaluation_results: Dict of section_title → evaluation scores.
        research_plan: The original research plan from Layer 1.

    Returns:
        {gaps, should_iterate, iteration_number}
    """
    cfg = get_config()
    # visual_richness excluded: figure injection is decoupled from LLM rewriting;
    # triggering re-research for low visual_richness wastes compute with no benefit.
    thresholds = {
        "factual_accuracy": cfg.eval_min_factual_accuracy,
        "citation_faithfulness": cfg.eval_min_citation_faithfulness,
        "completeness": cfg.eval_min_completeness,
        "coherence": cfg.eval_min_coherence,
        "relevance": cfg.eval_min_relevance,
    }

    gaps: list[dict] = []

    for section_title, scores in evaluation_results.items():
        failing = {
            dim: scores.get(dim, 0.0)
            for dim, threshold in thresholds.items()
            if scores.get(dim, 0.0) < threshold
        }
        if failing:
            gap_queries = scores.get("gap_queries", [])
            if not gap_queries:
                # Generate gap queries from research plan sub-queries
                sub_queries = research_plan.get("sub_queries", [])
                # Match section to sub-query by title similarity
                gap_queries = [
                    q for q in sub_queries
                    if any(
                        word.lower() in section_title.lower()
                        for word in q.split()[:3]
                    )
                ][:3]

                if not gap_queries:
                    gap_queries = sub_queries[:2]

            gaps.append(
                {
                    "section": section_title,
                    "score": scores.get("overall_score", 0.0),
                    "gap_queries": gap_queries,
                    "failing_dimensions": list(failing.keys()),
                    "critique": scores.get("critique", ""),
                }
            )

    should_iterate = len(gaps) > 0
    current_iteration = research_plan.get("iteration", 0)

    logger.info(
        f"[Layer 7] Gap analysis: {len(gaps)} sections need refinement "
        f"(should_iterate={should_iterate})"
    )

    return {
        "gaps": gaps,
        "should_iterate": should_iterate,
        "iteration_number": current_iteration + 1,
    }


# ─── trigger_reresearch ───────────────────────────────────────────────────────

@tool
def trigger_reresearch(
    gap_queries: list[str],
    collection: str = "default",
    max_results_per_query: int = 10,
) -> dict:
    """
    Layer 7: Execute targeted re-research for identified gaps.
    Does NOT run a full re-crawl — only searches for gap_queries.

    Args:
        gap_queries: List of specific queries to address gaps.
        collection: Collection name to add new chunks to.
        max_results_per_query: Max sources per gap query.

    Returns:
        {new_sources_count, new_chunks_count}
    """
    from astra.tools.layer2_crawlers import duckduckgo_search, arxiv_search

    logger.info(f"[Layer 7] Re-research for {len(gap_queries)} gap queries")

    new_sources: list[dict] = []

    for query in gap_queries:
        # Web search for fresh sources
        web_results = duckduckgo_search.invoke(
            {"query": query, "max_results": max_results_per_query}
        )
        new_sources.extend(web_results)

        # Academic search for supporting evidence
        acad_results = arxiv_search.invoke(
            {"query": query, "max_results": 5}
        )
        new_sources.extend(acad_results)

    logger.info(f"[Layer 7] Re-research collected {len(new_sources)} new sources")
    return {
        "new_sources_count": len(new_sources),
        "new_sources": new_sources,
        "new_chunks_count": 0,  # Will be populated after processing
    }


# ─── update_knowledge_base ────────────────────────────────────────────────────

@tool
def update_knowledge_base(
    new_documents: list[dict],
    collection: str = "default",
) -> dict:
    """
    Layer 7: Add new document chunks to the existing knowledge base.
    Called after trigger_reresearch to enrich the RAG index.

    Args:
        new_documents: List of processed document dicts with markdown content.
        collection: Collection name to update.

    Returns:
        {added_chunks, total_chunks}
    """
    from astra.tools.layer4_rag import _get_or_create_store, chunk_text, build_index_incremental

    logger.info(f"[Layer 7] Updating knowledge base: {len(new_documents)} new docs")

    store = _get_or_create_store(collection)
    existing_count = len(store.get("chunks", []))

    new_chunks: list[dict] = []
    for doc in new_documents:
        markdown = doc.get("markdown", doc.get("text", ""))
        source_url = doc.get("url", doc.get("source_url", ""))
        if not markdown:
            continue
        chunks = chunk_text(markdown, source_url)
        new_chunks.extend(chunks)

    if new_chunks:
        # Incremental update: only embed new chunks, append to live FAISS index.
        # Avoids re-embedding the full existing KB (~1000 chunks × 80s) on every call.
        build_index_incremental(new_chunks, collection)

    total = existing_count + len(new_chunks)
    logger.info(
        f"[Layer 7] KB updated: +{len(new_chunks)} chunks (total={total})"
    )
    return {
        "added_chunks": len(new_chunks),
        "total_chunks": total,
    }


# ─── check_convergence ────────────────────────────────────────────────────────

@tool
def check_convergence(
    current_scores: dict,
    previous_scores: Optional[dict] = None,
    iteration: int = 1,
    max_iterations: int = 3,
) -> dict:
    """
    Layer 7: Determine if another refinement iteration would yield marginal gains.

    Terminates if:
      - All scores ≥ thresholds (optimal)
      - max_iterations reached
      - Score improvement < 2% from previous iteration

    Args:
        current_scores: Dict of section_title → overall_score.
        previous_scores: Scores from the previous iteration.
        iteration: Current iteration number.
        max_iterations: Maximum allowed iterations.

    Returns:
        {converged, reason, recommendation}
    """
    cfg = get_config()

    # Check 1: max iterations
    if iteration >= max_iterations:
        logger.info(f"[Layer 7] Convergence: max iterations ({max_iterations}) reached")
        return {
            "converged": True,
            "reason": f"Max iterations ({max_iterations}) reached",
            "recommendation": "Proceed to document assembly with current draft",
        }

    # Check 2: all scores above threshold
    all_scores = list(current_scores.values())
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        if avg_score >= 0.80:
            logger.info(f"[Layer 7] Convergence: all scores above threshold (avg={avg_score:.2f})")
            return {
                "converged": True,
                "reason": f"All section scores above threshold (avg={avg_score:.2f})",
                "recommendation": "Report quality is high — proceed to assembly",
            }

    # Check 3: marginal improvement
    if previous_scores:
        prev_values = list(previous_scores.values())
        curr_values = list(current_scores.values())

        if prev_values and curr_values:
            prev_avg = sum(prev_values) / len(prev_values)
            curr_avg = sum(curr_values) / len(curr_values)
            improvement = curr_avg - prev_avg
            min_improvement = cfg.astra_min_refine_improvement

            if improvement < min_improvement:
                logger.info(
                    f"[Layer 7] Convergence: marginal improvement "
                    f"({improvement:.3f} < {min_improvement})"
                )
                return {
                    "converged": True,
                    "reason": f"Marginal improvement of {improvement:.3f} — diminishing returns",
                    "recommendation": "Further iterations unlikely to improve quality significantly",
                }

    # Check 4: any section still failing
    failing_count = sum(1 for s in current_scores.values() if s < 0.65)

    logger.info(
        f"[Layer 7] Not converged: iteration {iteration}/{max_iterations}, "
        f"{failing_count} sections still below threshold"
    )
    return {
        "converged": False,
        "reason": f"Iteration {iteration}/{max_iterations} — {failing_count} sections need improvement",
        "recommendation": "Continue with targeted re-research for failing sections",
    }
