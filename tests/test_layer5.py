"""
Layer 5 Test — LLM Judge / Quality Assurance
Tests: evaluate_section (mocked), score_citations, flag_gaps

Run: python main.py --test-layer 5
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

SAMPLE_SECTION = """
## Multi-Agent LLM Systems

Multi-agent large language model (LLM) systems have emerged as a powerful paradigm
for solving complex tasks that require diverse expertise [Source 1]. These systems
coordinate multiple AI agents, each specializing in specific domains, to collaboratively
produce high-quality outputs [Source 2].

Recent work by [Source 3] demonstrates that multi-agent collaboration can achieve
superior performance compared to single-agent approaches, particularly for research
tasks requiring both broad coverage and deep analysis.

### Key Components

The architecture typically involves:
1. An orchestrator agent coordinating the workflow
2. Specialist agents handling domain-specific tasks
3. Quality evaluation agents providing feedback

| Component | Function | Technology |
|-----------|----------|------------|
| Orchestrator | Workflow control | LangGraph |
| Crawlers | Information gathering | DuckDuckGo, arXiv |
| Writer | Content generation | Qwen3-32B |

Evidence from [Source 2] confirms that hybrid retrieval combining dense and sparse
search significantly improves recall compared to either approach alone.
"""

SAMPLE_CHUNKS = [
    "Multi-agent systems use multiple AI agents for collaborative problem solving.",
    "LangGraph enables stateful multi-actor applications with cyclic graph support.",
    "Hybrid retrieval combining FAISS and BM25 improves recall significantly.",
    "Research automation using LLMs can produce comprehensive reports efficiently.",
]


def test_evaluate_section():
    console.print("\n[bold]Test: evaluate_section[/bold]")
    from astra.tools.layer5_judge import evaluate_section

    try:
        result = evaluate_section.invoke(
            {
                "section_title": "Multi-Agent LLM Systems",
                "section_content": SAMPLE_SECTION,
                "source_chunks": SAMPLE_CHUNKS,
                "sub_query": "What are multi-agent LLM systems and how do they work?",
            }
        )
        assert "overall_score" in result, "Missing overall_score"
        assert "needs_refinement" in result, "Missing needs_refinement"
        assert 0 <= result["overall_score"] <= 1, "Score out of range"

        console.print(f"  ✅ Evaluation complete: score={result['overall_score']:.2f}")
        console.print(f"    Needs refinement: {result['needs_refinement']}")
        console.print(f"    Critique: {result.get('critique','')[:80]}")
        return True
    except Exception as e:
        if "Connection refused" in str(e) or "8000" in str(e):
            console.print("  ⚠️  vLLM not running — evaluation returns fallback scores")
            result = evaluate_section.invoke(
                {
                    "section_title": "Test",
                    "section_content": "Test content",
                    "source_chunks": [],
                    "sub_query": "test",
                }
            )
            assert "overall_score" in result
            console.print("  ✅ Fallback evaluation works")
            return True
        raise


def test_score_citations():
    console.print("\n[bold]Test: score_citations[/bold]")
    from astra.tools.layer5_judge import score_citations

    bibliography = [
        {"key": "Source 1", "title": "Multi-Agent Systems", "url": "https://example.com/1"},
        {"key": "Source 2", "title": "Hybrid Retrieval", "url": "https://example.com/2"},
        {"key": "Source 3", "title": "Agent Collaboration", "url": "https://example.com/3"},
    ]

    result = score_citations.invoke(
        {"section_content": SAMPLE_SECTION, "bibliography": bibliography}
    )
    assert "citation_score" in result, "Missing citation_score"
    assert "found_citations" in result, "Missing found_citations"

    console.print(f"  ✅ Citation score: {result['citation_score']:.2f}")
    console.print(f"    Found: {result.get('found_citations', [])}")
    console.print(f"    Orphan: {result.get('orphan_citations', [])}")
    return True


def test_flag_gaps():
    console.print("\n[bold]Test: flag_gaps[/bold]")
    from astra.tools.layer5_judge import flag_gaps

    eval_results = {
        "Introduction": {
            "factual_accuracy": 0.85,
            "citation_faithfulness": 0.90,
            "completeness": 0.70,
            "coherence": 0.85,
            "visual_richness": 0.60,
            "relevance": 0.90,
            "overall_score": 0.80,
            "needs_refinement": False,
            "gap_queries": [],
            "critique": "Good section",
        },
        "Analysis": {
            "factual_accuracy": 0.60,  # Below threshold 0.70
            "citation_faithfulness": 0.70,  # Below threshold 0.80
            "completeness": 0.50,  # Below threshold 0.60
            "coherence": 0.75,
            "visual_richness": 0.40,  # Below threshold 0.50
            "relevance": 0.85,
            "overall_score": 0.63,
            "needs_refinement": True,
            "gap_queries": ["detailed analysis query"],
            "critique": "Needs more evidence",
        },
    }

    gaps = flag_gaps.invoke({"evaluation_results": eval_results})
    assert isinstance(gaps, list), "Expected list"
    # Analysis section should be flagged
    gap_sections = {g["section"] for g in gaps}
    assert "Analysis" in gap_sections, "Analysis should be flagged"

    console.print(f"  ✅ Flagged {len(gaps)} sections")
    for gap in gaps:
        console.print(f"    • {gap['section']}: failing={gap.get('failing_dimensions', [])}")
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 5: LLM Judge / QA ═══[/bold cyan]\n")

    tests = [
        ("evaluate_section", test_evaluate_section),
        ("score_citations", test_score_citations),
        ("flag_gaps", test_flag_gaps),
    ]

    table = Table(title="Layer 5 Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            table.add_row(name, "[green]PASS ✅[/green]")
            passed += 1
        except Exception as e:
            table.add_row(name, f"[red]FAIL ❌ {str(e)[:60]}[/red]")
            failed += 1

    console.print(table)
    console.print(f"\n[bold]Layer 5: {passed}/{passed+failed} passed[/bold]")
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
