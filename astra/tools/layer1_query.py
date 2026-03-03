"""
ASTRA Layer 1 — Query Intelligence Agent
Tools: query_expand, plan_research
Action paradigm: JSON tool-call (reasoning-first)

Analyzes user queries, generates research plans with sub-queries, section
outlines, and source priorities.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from langchain_core.tools import tool
from loguru import logger
from openai import OpenAI

from astra.config import get_config

# ─── System prompts ─────────────────────────────────────────────────────────

_ENRICH_SYSTEM = """You are ASTRA's Query Enrichment Agent.

Your job: analyze the user's raw research query and produce a richer, more precise version.

Infer:
1. expertise_level: "beginner" | "intermediate" | "expert" — based on vocabulary and phrasing
2. purpose: "academic" | "professional" | "learning" | "building" | "policy"
3. temporal_scope: "latest" | "historical" | "timeless"
4. implicit_needs: what the user likely needs but didn't state (e.g., code examples, benchmarks, cost analysis)
5. enriched_query: a rewritten, more precise, complete version of the original query
6. enrichment_notes: 2–5 brief notes to guide section planning and source selection

Return ONLY valid JSON:
{
  "enriched_query": "<rewritten precise query>",
  "original_query": "<original unchanged>",
  "expertise_level": "<beginner|intermediate|expert>",
  "purpose": "<academic|professional|learning|building|policy>",
  "temporal_scope": "<latest|historical|timeless>",
  "implicit_needs": ["<need1>", "<need2>", ...],
  "enrichment_notes": ["<note1>", "<note2>", ...]
}"""

_EXPAND_SYSTEM = """You are the Query Intelligence Agent for ASTRA.

Your job: transform a user query into a structured research plan.

Analyze the query for:
1. Missing dimensions (timeframe, domain, depth, geography, scope)
2. Natural sub-components that can be researched independently
3. Appropriate source types (web, academic, code repos)

Return ONLY valid JSON matching this schema:
{
  "sub_queries": ["<query1>", "<query2>", ...],  // 5-15 specific sub-queries
  "section_outline": ["<title1>", "<title2>", ...],  // report sections
  "source_priorities": {
    "web": <0-1 float>,
    "arxiv": <0-1 float>,
    "semantic_scholar": <0-1 float>,
    "github": <0-1 float>,
    "pubmed": <0-1 float>,
    "openalex": <0-1 float>
  },
  "clarifying_questions": ["<q1>", "<q2>"] or null  // max 3, null if clear
}"""

_PLAN_SYSTEM = """You are the Research Planner for ASTRA.

Convert an expanded research plan into a sequenced execution plan.
Assign each sub-query to crawler agents and specify search parameters.

Return ONLY valid JSON:
{
  "session_id": "<uuid>",
  "phases": [
    {
      "phase": 1,
      "name": "Initial Broad Search",
      "parallel_tasks": [
        {"agent": "web_crawler", "queries": ["..."], "max_results": 10},
        {"agent": "academic_crawler", "queries": ["..."], "databases": ["arxiv", "s2"]},
        {"agent": "repo_crawler", "queries": ["..."], "language": null}
      ]
    }
  ],
  "total_sub_queries": <int>,
  "estimated_sources": <int>
}"""


def _call_llm(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Call the Qwen3-32B-AWQ orchestrator via vLLM OpenAI-compatible API."""
    cfg = get_config()
    client = OpenAI(base_url=cfg.vllm_base_url, api_key=cfg.vllm_api_key)

    response = client.chat.completions.create(
        model=cfg.astra_main_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response (handles markdown code fences)."""
    raw = raw.strip()
    # Strip ```json ... ``` fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )
    return json.loads(raw)


# ─── Tool implementations ────────────────────────────────────────────────────

@tool
def enrich_query(query: str) -> dict:
    """
    Layer 1: Pre-process a user query to infer context, rewrite it precisely,
    and surface implicit research needs before query expansion.

    Args:
        query: The raw user research query.

    Returns:
        dict with enriched_query, expertise_level, purpose, implicit_needs,
        enrichment_notes, and original_query.
    """
    logger.info(f"[Layer 1] enrich_query: '{query[:80]}'")

    _passthrough = {
        "enriched_query": query,
        "original_query": query,
        "expertise_level": "intermediate",
        "purpose": "academic",
        "temporal_scope": "latest",
        "implicit_needs": [],
        "enrichment_notes": [],
    }

    try:
        raw = _call_llm(
            system_prompt=_ENRICH_SYSTEM,
            user_message=f"Research query to enrich:\n{query}",
            temperature=0.3,
            max_tokens=1024,
        )
        result = _parse_json_response(raw)
        # Ensure original_query is always set
        result.setdefault("original_query", query)
        result.setdefault("enriched_query", query)
        result.setdefault("implicit_needs", [])
        result.setdefault("enrichment_notes", [])
        logger.info(
            f"[Layer 1] enrich_query: expertise={result.get('expertise_level')}, "
            f"purpose={result.get('purpose')}, "
            f"implicit_needs={result.get('implicit_needs')}"
        )
        return result
    except Exception as e:
        logger.warning(f"[Layer 1] enrich_query failed ({e}), using passthrough")
        return _passthrough


@tool
def query_expand(query: str, user_clarifications: Optional[dict] = None) -> dict:
    """
    Layer 1: Analyze a user query and generate a structured research plan.

    Args:
        query: The user's research query.
        user_clarifications: Optional dict of answers to clarifying questions.

    Returns:
        dict with sub_queries, section_outline, source_priorities,
        and optional clarifying_questions.
    """
    cfg = get_config()
    logger.info(f"[Layer 1] query_expand: '{query[:80]}...'")

    user_msg = f"Research query: {query}"
    if user_clarifications:
        user_msg += f"\n\nUser clarifications: {json.dumps(user_clarifications, indent=2)}"

    try:
        raw = _call_llm(
            system_prompt=_EXPAND_SYSTEM,
            user_message=user_msg,
            temperature=0.3,
        )
        result = _parse_json_response(raw)
        logger.info(
            f"[Layer 1] Expanded to {len(result.get('sub_queries', []))} sub-queries, "
            f"{len(result.get('section_outline', []))} sections"
        )
        return result
    except Exception as e:
        logger.error(f"[Layer 1] query_expand failed: {e}")
        # Return a minimal fallback plan
        return {
            "sub_queries": [query],
            "section_outline": [
                "Executive Summary",
                "Introduction",
                "Main Analysis",
                "Findings",
                "Conclusions",
                "Bibliography",
            ],
            "source_priorities": {
                "web": 0.8,
                "arxiv": 0.6,
                "semantic_scholar": 0.6,
                "github": 0.4,
                "pubmed": 0.3,
                "openalex": 0.5,
            },
            "clarifying_questions": None,
        }


@tool
def plan_research(research_plan: dict) -> dict:
    """
    Layer 1: Convert an expanded query plan into a sequenced LangGraph state.

    Args:
        research_plan: Output from query_expand.

    Returns:
        dict with langgraph_state representing the session execution plan.
    """
    logger.info("[Layer 1] plan_research: building execution plan")
    cfg = get_config()

    sub_queries = research_plan.get("sub_queries", [])
    source_priorities = research_plan.get("source_priorities", {})

    user_msg = (
        f"Research plan to sequence:\n{json.dumps(research_plan, indent=2)}"
    )

    try:
        raw = _call_llm(
            system_prompt=_PLAN_SYSTEM,
            user_message=user_msg,
            temperature=0.1,
        )
        plan = _parse_json_response(raw)
    except Exception as e:
        logger.warning(f"[Layer 1] plan_research LLM failed ({e}), using default plan")
        plan = _build_default_plan(sub_queries, source_priorities)

    # Merge the section outline into the plan
    plan["section_outline"] = research_plan.get("section_outline", [])
    plan["sub_queries"] = sub_queries
    plan["source_priorities"] = source_priorities

    return {"langgraph_state": plan}


def _build_default_plan(
    sub_queries: list[str],
    source_priorities: dict[str, float],
) -> dict:
    """Build a default execution plan when the LLM is unavailable."""
    session_id = str(uuid.uuid4())
    web_qs = sub_queries
    acad_qs = sub_queries

    return {
        "session_id": session_id,
        "phases": [
            {
                "phase": 1,
                "name": "Broad Discovery",
                "parallel_tasks": [
                    {
                        "agent": "web_crawler",
                        "queries": web_qs[:5],
                        "max_results": 10,
                    },
                    {
                        "agent": "academic_crawler",
                        "queries": acad_qs[:5],
                        "databases": ["arxiv", "s2", "openalex"],
                    },
                    {
                        "agent": "repo_crawler",
                        "queries": sub_queries[:3],
                        "language": None,
                    },
                ],
            },
            {
                "phase": 2,
                "name": "Deep Dive",
                "parallel_tasks": [
                    {
                        "agent": "web_crawler",
                        "queries": web_qs[5:],
                        "max_results": 10,
                    },
                    {
                        "agent": "academic_crawler",
                        "queries": acad_qs[5:],
                        "databases": ["arxiv", "s2", "pubmed"],
                    },
                ],
            },
        ],
        "total_sub_queries": len(sub_queries),
        "estimated_sources": len(sub_queries) * 8,
    }
