"""
ASTRA Orchestrator Agent — ReAct agent with access to all ASTRA tools.

The orchestrator is Qwen3-32B-AWQ (via vLLM) with tool-calling capability.
It follows the Thought-Action-Observation (TAO) loop for atomic operations
and the Thought-Code-Observation (TCO) loop for multi-step code actions.

System prompt is loaded from ASTRA_OS_Claude_prompt.txt.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from astra.config import get_config

# ─── Load ASTRA system prompt ─────────────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load the ASTRA orchestrator system prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "ASTRA_OS_Claude_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    # Fallback to brief inline prompt
    return """You are ASTRA, an autonomous deep research intelligence platform.
You orchestrate multi-agent workflows to produce comprehensive research reports.
Always think step-by-step before acting."""


# ─── LLM Factory ──────────────────────────────────────────────────────────────

# Qwen3 models expose a thinking mode via chat template kwargs.
# We disable it for all ASTRA calls so <think> tokens don't bleed into
# section text or structured JSON outputs.
_QWEN3_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def get_main_llm(temperature: float | None = None) -> BaseChatModel:
    """Create the main Qwen3-32B-AWQ LLM via vLLM OpenAI-compatible API."""
    cfg = get_config()
    return ChatOpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        model=cfg.astra_main_model,
        temperature=temperature if temperature is not None else cfg.astra_main_temperature,
        max_tokens=cfg.astra_main_max_tokens,
        streaming=True,
        extra_body=_QWEN3_NO_THINK,
    )


def get_judge_llm() -> BaseChatModel:
    """Create the deterministic judge LLM (temperature=0.0)."""
    cfg = get_config()
    return ChatOpenAI(
        base_url=cfg.astra_judge_base_url,
        api_key=cfg.vllm_api_key,
        model=cfg.astra_judge_model,
        temperature=cfg.astra_judge_temperature,
        max_tokens=cfg.astra_judge_max_tokens,
        extra_body=_QWEN3_NO_THINK,
    )


def get_planner_llm() -> BaseChatModel:
    """Create the planner LLM (temperature=0.3 for structured output)."""
    cfg = get_config()
    return ChatOpenAI(
        base_url=cfg.vllm_base_url,
        api_key=cfg.vllm_api_key,
        model=cfg.astra_main_model,
        temperature=cfg.astra_main_temperature_structured,
        max_tokens=cfg.astra_main_max_tokens,
        extra_body=_QWEN3_NO_THINK,
    )


# ─── All tools registry ───────────────────────────────────────────────────────

def get_all_tools() -> list:
    """Return all ASTRA tools for the orchestrator agent."""
    from astra.tools.layer1_query import query_expand, plan_research
    from astra.tools.layer2_crawlers import (
        duckduckgo_search,
        jina_fetch_url,
        firecrawl_scrape,
        firecrawl_crawl,
        arxiv_search,
        semantic_scholar_search,
        openalex_search,
        pubmed_search,
        github_search,
        searxng_search,
        wikipedia_search,
    )
    from astra.tools.layer3_docs import (
        docling_parse_pdf,
        pymupdf_extract,
        parse_document,
        qwen_vl_describe_chart,
    )
    from astra.tools.layer4_rag import (
        embed_chunks,
        faiss_search,
        bm25_search,
        hybrid_retrieve,
        bge_rerank,
        lightrag_query,
    )
    from astra.tools.layer5_judge import (
        evaluate_section,
        score_citations,
        deepeval_run,
        flag_gaps,
    )
    from astra.tools.layer6_report import (
        write_section,
        generate_chart_plotly,
        generate_chart_matplotlib,
        render_mermaid,
        build_docx,
        build_markdown,
        embed_figure,
    )
    from astra.tools.layer7_refinement import (
        gap_analysis,
        trigger_reresearch,
        update_knowledge_base,
        check_convergence,
    )

    return [
        # Layer 1
        query_expand, plan_research,
        # Layer 2
        duckduckgo_search, jina_fetch_url, firecrawl_scrape, firecrawl_crawl,
        arxiv_search, semantic_scholar_search, openalex_search, pubmed_search,
        github_search, searxng_search, wikipedia_search,
        # Layer 3
        docling_parse_pdf, pymupdf_extract, parse_document, qwen_vl_describe_chart,
        # Layer 4
        embed_chunks, faiss_search, bm25_search, hybrid_retrieve, bge_rerank,
        lightrag_query,
        # Layer 5
        evaluate_section, score_citations, deepeval_run, flag_gaps,
        # Layer 6
        write_section, generate_chart_plotly, generate_chart_matplotlib,
        render_mermaid, build_docx, build_markdown, embed_figure,
        # Layer 7
        gap_analysis, trigger_reresearch, update_knowledge_base, check_convergence,
    ]


def create_orchestrator_agent():
    """
    Create a ReAct orchestrator agent with all ASTRA tools bound.
    Uses the full ASTRA system prompt.
    """
    from langgraph.prebuilt import create_react_agent

    llm = get_main_llm()
    tools = get_all_tools()
    system_prompt = _load_system_prompt()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )
    logger.info(f"[Orchestrator] ReAct agent created with {len(tools)} tools")
    return agent
