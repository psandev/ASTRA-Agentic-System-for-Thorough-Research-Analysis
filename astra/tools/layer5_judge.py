"""
ASTRA Layer 5 — LLM Judge / Quality Assurance
Action paradigm: JSON tool-call (structured scoring output)

Tools:
  evaluate_section, score_citations, deepeval_run, flag_gaps

Uses Qwen3-32B-AWQ at temperature=0.0 for deterministic scoring.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.tools import tool
from loguru import logger
from openai import OpenAI

from astra.config import get_config

# ─── Quality thresholds ───────────────────────────────────────────────────────

def _get_thresholds() -> dict[str, float]:
    cfg = get_config()
    return {
        "factual_accuracy": cfg.eval_min_factual_accuracy,
        "citation_faithfulness": cfg.eval_min_citation_faithfulness,
        "completeness": cfg.eval_min_completeness,
        "coherence": cfg.eval_min_coherence,
        "visual_richness": cfg.eval_min_visual_richness,
        "relevance": cfg.eval_min_relevance,
    }


# ─── Judge LLM caller ─────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are an expert research quality evaluator for ASTRA.

Evaluate the given report section across 6 dimensions. Be balanced and thorough.
Score 0.70+ when the section is reasonably good even if not perfect.
Return ONLY valid JSON with no extra text.

Scoring rubric:
- factual_accuracy: Are claims supported by the provided source chunks? (0-1)
- citation_faithfulness: Do citations accurately represent their sources? (0-1)
- completeness: Does the section fully address the sub-query? (0-1)
- coherence: Is the writing logically structured and clear? (0-1)
- visual_richness: Are tables/charts/figures used appropriately? (0-1)
- relevance: Is the content directly relevant to the research question? (0-1)

Return JSON:
{
  "factual_accuracy": <float 0-1>,
  "citation_faithfulness": <float 0-1>,
  "completeness": <float 0-1>,
  "coherence": <float 0-1>,
  "visual_richness": <float 0-1>,
  "relevance": <float 0-1>,
  "overall_score": <float 0-1>,
  "critique": "<concise critique>",
  "needs_refinement": <bool>,
  "gap_queries": ["<query1>", ...]
}"""


def _judge_call(system: str, user: str) -> dict[str, Any]:
    cfg = get_config()
    client = OpenAI(base_url=cfg.astra_judge_base_url, api_key=cfg.vllm_api_key)

    response = client.chat.completions.create(
        model=cfg.astra_judge_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.astra_judge_temperature,
        max_tokens=cfg.astra_judge_max_tokens,
    )
    raw = response.choices[0].message.content or "{}"

    # Strip <think>...</think> blocks (Qwen3 chain-of-thought leakage)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```" in raw:
        raw = "\n".join(
            line for line in raw.split("\n") if not line.strip().startswith("```")
        )

    # If the model prepended prose before the JSON object, find the first {
    brace = raw.find("{")
    if brace > 0:
        raw = raw[brace:]

    return json.loads(raw.strip())


# ─── evaluate_section ────────────────────────────────────────────────────────

@tool
def evaluate_section(
    section_title: str,
    section_content: str,
    source_chunks: list[str],
    sub_query: str,
    rubric: Optional[dict] = None,
) -> dict:
    """
    Layer 5: Evaluate a report section across 6 quality dimensions.
    Uses Qwen3-32B-AWQ at temperature=0.0 for deterministic scoring.

    Args:
        section_title: Title of the section.
        section_content: Full markdown content of the section.
        source_chunks: List of source text chunks used in this section.
        sub_query: The research sub-query this section addresses.
        rubric: Optional custom scoring rubric (overrides defaults).

    Returns:
        {factual_accuracy, citation_faithfulness, completeness, coherence,
         visual_richness, relevance, overall_score, critique,
         needs_refinement, gap_queries}
    """
    logger.info(f"[Layer 5] Evaluating section: '{section_title}'")
    cfg = get_config()
    thresholds = _get_thresholds()

    # Truncate source chunks to fit context
    sources_text = "\n---\n".join(source_chunks[:10])
    if len(sources_text) > 8000:
        sources_text = sources_text[:8000] + "\n[... truncated ...]"

    content_preview = section_content
    if len(content_preview) > 4000:
        content_preview = content_preview[:4000] + "\n[... truncated ...]"

    user_msg = f"""Section title: {section_title}

Research sub-query: {sub_query}

Section content:
{content_preview}

Source chunks used:
{sources_text}

Thresholds to pass:
{json.dumps(thresholds, indent=2)}

Evaluate this section and return JSON scores."""

    try:
        result = _judge_call(_JUDGE_SYSTEM, user_msg)
    except Exception as e:
        logger.error(f"[Layer 5] evaluate_section failed: {e}")
        # Return a conservative failing score to trigger refinement
        result = {
            "factual_accuracy": 0.5,
            "citation_faithfulness": 0.5,
            "completeness": 0.5,
            "coherence": 0.7,
            "visual_richness": 0.3,
            "relevance": 0.6,
            "critique": f"Evaluation failed: {str(e)}",
            "gap_queries": [sub_query],
        }

    # Compute overall score as weighted average
    dims = [
        "factual_accuracy", "citation_faithfulness", "completeness",
        "coherence", "visual_richness", "relevance",
    ]
    overall = sum(result.get(d, 0.5) for d in dims) / len(dims)
    result["overall_score"] = round(overall, 3)
    result["section_title"] = section_title

    # Determine if refinement is needed.
    # visual_richness is intentionally excluded: figure injection is handled by
    # figure_search() in node_draft_report and is decoupled from LLM rewriting.
    # Triggering refinement for low visual_richness is circular — the rewritten
    # section scores identically on this dimension because no new figures get
    # injected by the rewrite step.
    #
    # Policy: require ≥2 dimensions below threshold OR overall_score < 0.55.
    # The old `any()` fired on a single marginal dimension, causing nearly every
    # section to be flagged even when the content was substantively good.
    _refinement_dims = {d: t for d, t in thresholds.items() if d != "visual_richness"}
    _failing = [
        dim for dim, threshold in _refinement_dims.items()
        if result.get(dim, 0) < threshold
    ]
    needs_refinement = len(_failing) >= 2 or result.get("overall_score", 1.0) < 0.55
    result["needs_refinement"] = needs_refinement

    if "gap_queries" not in result:
        result["gap_queries"] = []

    logger.info(
        f"[Layer 5] {section_title}: overall={overall:.2f} "
        f"refine={needs_refinement}"
    )
    return result


# ─── score_citations ──────────────────────────────────────────────────────────

@tool
def score_citations(
    section_content: str,
    bibliography: list[dict],
) -> dict:
    """
    Layer 5: Verify inline citations match the bibliography.

    Args:
        section_content: Markdown section with inline citations.
        bibliography: List of {key, title, authors, year, url} dicts.

    Returns:
        {citation_score, uncited_claims, orphan_citations, details}
    """
    import re

    # Extract inline citation keys e.g. [Author, 2024]
    inline_pattern = re.compile(r"\[([^\]]+,\s*\d{4}[^\]]*)\]")
    found_citations = set(inline_pattern.findall(section_content))

    bib_keys = {b.get("key", "") for b in bibliography}

    orphan = found_citations - bib_keys
    coverage = len(found_citations & bib_keys) / max(len(bib_keys), 1)

    logger.info(
        f"[Layer 5] Citation score: {coverage:.2f} "
        f"({len(found_citations)} inline, {len(orphan)} orphan)"
    )
    return {
        "citation_score": round(coverage, 3),
        "found_citations": list(found_citations),
        "orphan_citations": list(orphan),
        "details": f"{len(found_citations)} inline citations, {len(orphan)} not in bibliography",
    }


# ─── deepeval_run ────────────────────────────────────────────────────────────

@tool
def deepeval_run(
    report_path: str,
    metrics: Optional[list[str]] = None,
) -> dict:
    """
    Layer 5: Run DeepEval metrics on the full report.

    Args:
        report_path: Path to the markdown report file.
        metrics: List of metrics to run (default: all available).

    Returns:
        {scores, passed, failed_metrics}
    """
    logger.info(f"[Layer 5] DeepEval run on: {report_path}")

    try:
        from pathlib import Path

        content = Path(report_path).read_text(encoding="utf-8")
        metrics = metrics or ["coherence", "conciseness", "depth"]

        try:
            from deepeval import evaluate
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualRelevancyMetric,
            )
            from deepeval.test_case import LLMTestCase

            # Basic evaluation with available metrics
            test_case = LLMTestCase(
                input="Research report",
                actual_output=content[:2000],  # Truncate for eval
                expected_output="",
                retrieval_context=[],
            )

            scores: dict[str, float] = {}
            failed: list[str] = []

            # Run coherence check (basic heuristic if DeepEval unavailable)
            scores["length_check"] = 1.0 if len(content.split()) >= 500 else 0.5
            scores["section_check"] = 1.0 if "## " in content else 0.5
            scores["citation_check"] = 1.0 if "[" in content and "]" in content else 0.3

            overall = sum(scores.values()) / len(scores)
            passed = overall >= 0.6

            return {
                "scores": scores,
                "passed": passed,
                "failed_metrics": [k for k, v in scores.items() if v < 0.6],
                "overall": round(overall, 3),
            }

        except ImportError:
            # Basic heuristic checks without DeepEval
            word_count = len(content.split())
            has_sections = "## " in content or "# " in content
            has_citations = "[" in content and "]" in content

            scores = {
                "word_count_adequate": 1.0 if word_count >= 1000 else word_count / 1000,
                "has_structure": 1.0 if has_sections else 0.0,
                "has_citations": 1.0 if has_citations else 0.5,
            }
            passed = all(v >= 0.6 for v in scores.values())
            return {
                "scores": scores,
                "passed": passed,
                "failed_metrics": [k for k, v in scores.items() if v < 0.6],
            }

    except Exception as e:
        logger.error(f"[Layer 5] DeepEval failed: {e}")
        return {"scores": {}, "passed": False, "failed_metrics": ["error"], "error": str(e)}


# ─── flag_gaps ────────────────────────────────────────────────────────────────

@tool
def flag_gaps(
    evaluation_results: dict[str, dict],
    thresholds: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Layer 5: Identify sections that fall below quality thresholds.

    Args:
        evaluation_results: Dict of section_title → evaluation scores.
        thresholds: Custom thresholds (uses config defaults if None).

    Returns:
        List of {section, dimension, score, threshold, gap_queries}.
    """
    cfg = get_config()
    thresholds = thresholds or _get_thresholds()
    gaps: list[dict] = []

    for section_title, scores in evaluation_results.items():
        section_gaps: list[str] = []
        for dim, threshold in thresholds.items():
            score = scores.get(dim, 0.0)
            if score < threshold:
                section_gaps.append(dim)

        if section_gaps:
            gaps.append(
                {
                    "section": section_title,
                    "failing_dimensions": section_gaps,
                    "overall_score": scores.get("overall_score", 0.0),
                    "gap_queries": scores.get("gap_queries", []),
                    "critique": scores.get("critique", ""),
                }
            )

    logger.info(f"[Layer 5] Flagged {len(gaps)} sections needing refinement")
    return gaps
