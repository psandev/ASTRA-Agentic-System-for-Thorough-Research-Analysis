"""
Layer 7 Test — Refinement Meta-Loop
Tests: gap_analysis, check_convergence, update_knowledge_base

Run: python main.py --test-layer 7
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def test_gap_analysis():
    console.print("\n[bold]Test: gap_analysis[/bold]")
    from astra.tools.layer7_refinement import gap_analysis

    eval_results = {
        "Introduction": {
            "factual_accuracy": 0.85,
            "citation_faithfulness": 0.90,
            "completeness": 0.75,
            "coherence": 0.85,
            "visual_richness": 0.65,
            "relevance": 0.90,
            "overall_score": 0.82,
            "needs_refinement": False,
            "gap_queries": [],
            "critique": "Well written",
        },
        "Methods": {
            "factual_accuracy": 0.55,  # Below 0.70
            "citation_faithfulness": 0.65,  # Below 0.80
            "completeness": 0.45,  # Below 0.60
            "coherence": 0.70,
            "visual_richness": 0.35,  # Below 0.50
            "relevance": 0.75,
            "overall_score": 0.58,
            "needs_refinement": True,
            "gap_queries": ["detailed methodology", "experimental setup"],
            "critique": "Lacks methodological detail",
        },
    }

    research_plan = {
        "sub_queries": ["AI agents overview", "methodology for LLM agents", "experimental results"],
        "iteration": 0,
    }

    result = gap_analysis.invoke(
        {"evaluation_results": eval_results, "research_plan": research_plan}
    )

    assert "gaps" in result, "Missing gaps"
    assert "should_iterate" in result, "Missing should_iterate"
    assert result["should_iterate"] is True, "Should iterate for failing sections"

    gap_sections = {g["section"] for g in result["gaps"]}
    assert "Methods" in gap_sections, "Methods should be in gaps"

    console.print(f"  ✅ gap_analysis: {len(result['gaps'])} gaps found")
    for gap in result["gaps"]:
        console.print(f"    • '{gap['section']}': {gap.get('failing_dimensions', [])}")
    return True


def test_check_convergence_max_iter():
    console.print("\n[bold]Test: check_convergence — max iterations[/bold]")
    from astra.tools.layer7_refinement import check_convergence

    result = check_convergence.invoke(
        {
            "current_scores": {"A": 0.6, "B": 0.5},
            "previous_scores": None,
            "iteration": 3,
            "max_iterations": 3,
        }
    )
    assert result["converged"] is True, "Should converge at max iterations"
    console.print(f"  ✅ Max iterations: converged={result['converged']} — {result['reason']}")
    return True


def test_check_convergence_high_scores():
    console.print("\n[bold]Test: check_convergence — all scores high[/bold]")
    from astra.tools.layer7_refinement import check_convergence

    result = check_convergence.invoke(
        {
            "current_scores": {"A": 0.92, "B": 0.88, "C": 0.95},
            "previous_scores": None,
            "iteration": 1,
            "max_iterations": 3,
        }
    )
    assert result["converged"] is True, "Should converge with high scores"
    console.print(f"  ✅ High scores: converged={result['converged']} — {result['reason']}")
    return True


def test_check_convergence_marginal_gain():
    console.print("\n[bold]Test: check_convergence — marginal improvement[/bold]")
    from astra.tools.layer7_refinement import check_convergence

    result = check_convergence.invoke(
        {
            "current_scores": {"A": 0.71, "B": 0.72},
            "previous_scores": {"A": 0.70, "B": 0.71},
            "iteration": 2,
            "max_iterations": 3,
        }
    )
    assert result["converged"] is True, "Should converge with marginal improvement"
    console.print(f"  ✅ Marginal gain: converged={result['converged']}")
    return True


def test_check_convergence_not_converged():
    console.print("\n[bold]Test: check_convergence — not converged[/bold]")
    from astra.tools.layer7_refinement import check_convergence

    result = check_convergence.invoke(
        {
            "current_scores": {"A": 0.55, "B": 0.48, "C": 0.60},
            "previous_scores": None,
            "iteration": 1,
            "max_iterations": 3,
        }
    )
    assert result["converged"] is False, "Should NOT converge with low scores"
    console.print(f"  ✅ Not converged: converged={result['converged']} — {result['reason']}")
    return True


def test_update_knowledge_base():
    console.print("\n[bold]Test: update_knowledge_base[/bold]")
    from astra.tools.layer7_refinement import update_knowledge_base
    from astra.tools.layer4_rag import _INDEX_STORE, build_index

    # Pre-populate the collection
    initial_chunks = [
        {"id": "c1", "text": "Initial document chunk one.", "metadata": {}},
        {"id": "c2", "text": "Initial document chunk two.", "metadata": {}},
    ]
    build_index(initial_chunks, "refine_test")

    # Update with new docs
    new_docs = [
        {"markdown": "New research findings about AI agents and multi-hop reasoning.", "url": "https://example.com/new"},
        {"markdown": "Updated methodology section with additional details and citations.", "url": ""},
    ]

    result = update_knowledge_base.invoke(
        {"new_documents": new_docs, "collection": "refine_test"}
    )
    assert "added_chunks" in result, "Missing added_chunks"
    assert result["added_chunks"] > 0, "Expected new chunks to be added"

    console.print(
        f"  ✅ KB updated: +{result['added_chunks']} new chunks "
        f"(total={result['total_chunks']})"
    )
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 7: Refinement Meta-Loop ═══[/bold cyan]\n")

    tests = [
        ("gap_analysis", test_gap_analysis),
        ("check_convergence — max iter", test_check_convergence_max_iter),
        ("check_convergence — high scores", test_check_convergence_high_scores),
        ("check_convergence — marginal gain", test_check_convergence_marginal_gain),
        ("check_convergence — not converged", test_check_convergence_not_converged),
        ("update_knowledge_base", test_update_knowledge_base),
    ]

    table = Table(title="Layer 7 Test Results")
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
            console.print_exception(max_frames=3)

    console.print(table)
    console.print(f"\n[bold]Layer 7: {passed}/{passed+failed} passed[/bold]")
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
