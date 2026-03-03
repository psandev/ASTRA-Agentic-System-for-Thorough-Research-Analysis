"""
Layer 2 Test — Multi-Source Crawlers
Tests: DuckDuckGo, Jina, arXiv, Semantic Scholar, OpenAlex, GitHub

Run: python main.py --test-layer 2
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def test_duckduckgo():
    console.print("\n[bold]Test: DuckDuckGo search[/bold]")
    from astra.tools.layer2_crawlers import duckduckgo_search

    results = duckduckgo_search.invoke(
        {"query": "LangGraph multi-agent LLM 2025", "max_results": 5}
    )
    assert isinstance(results, list), "Expected list"
    if len(results) == 0:
        console.print("  ⚠️  DuckDuckGo returned 0 results (possible rate limit) — skipping assertions")
        return True
    for r in results[:3]:
        assert "url" in r, "Missing url"
        assert "title" in r, "Missing title"

    console.print(f"  ✅ Got {len(results)} results")
    for r in results[:3]:
        console.print(f"    • [{r.get('source_type','web')}] {r.get('title','')[:60]}")
    return True


def test_arxiv():
    console.print("\n[bold]Test: arXiv search[/bold]")
    from astra.tools.layer2_crawlers import arxiv_search

    results = arxiv_search.invoke(
        {"query": "large language model agents tool use", "max_results": 5}
    )
    assert isinstance(results, list), "Expected list"

    console.print(f"  ✅ Got {len(results)} papers")
    for p in results[:3]:
        console.print(f"    • {p.get('title','')[:60]} ({p.get('published','')})")
    return True


def test_semantic_scholar():
    console.print("\n[bold]Test: Semantic Scholar search[/bold]")
    from astra.tools.layer2_crawlers import semantic_scholar_search

    results = semantic_scholar_search.invoke(
        {"query": "retrieval augmented generation survey", "max_results": 5}
    )
    assert isinstance(results, list), "Expected list"

    console.print(f"  ✅ Got {len(results)} papers")
    for p in results[:3]:
        console.print(
            f"    • {p.get('title','')[:50]} "
            f"(cited={p.get('citation_count',0)})"
        )
    return True


def test_openalex():
    console.print("\n[bold]Test: OpenAlex search[/bold]")
    from astra.tools.layer2_crawlers import openalex_search

    results = openalex_search.invoke(
        {"query": "transformer neural network NLP", "max_results": 5}
    )
    assert isinstance(results, list), "Expected list"

    console.print(f"  ✅ Got {len(results)} works")
    for w in results[:3]:
        console.print(
            f"    • {w.get('title','')[:50]} "
            f"({w.get('year','')}, cited={w.get('cited_by_count',0)})"
        )
    return True


def test_github():
    console.print("\n[bold]Test: GitHub search[/bold]")
    from astra.tools.layer2_crawlers import github_search

    results = github_search.invoke(
        {"query": "langgraph multi-agent", "max_results": 5}
    )
    assert isinstance(results, list), "Expected list"

    console.print(f"  ✅ Got {len(results)} repos")
    for r in results[:3]:
        console.print(
            f"    • {r.get('full_name','')[:40]} "
            f"⭐{r.get('stars',0):,}"
        )
    return True


def test_jina():
    console.print("\n[bold]Test: Jina URL fetch[/bold]")
    from astra.tools.layer2_crawlers import jina_fetch_url

    result = jina_fetch_url.invoke(
        {"url": "https://lilianweng.github.io/posts/2023-06-23-agent/", "timeout_seconds": 30}
    )
    assert "markdown" in result, "Missing markdown"
    assert len(result.get("markdown", "")) > 100, "Markdown too short"

    console.print(f"  ✅ Fetched {len(result.get('markdown',''))} chars")
    console.print(f"    Title: {result.get('title','')[:60]}")
    return True


def test_deduplication():
    console.print("\n[bold]Test: Source deduplication[/bold]")
    from astra.tools.layer2_crawlers import deduplicate_sources

    sources = [
        {"url": "https://example.com/a", "title": "A"},
        {"url": "https://example.com/a", "title": "A (dup)"},
        {"url": "https://example.com/b", "title": "B"},
        {"url": "", "title": "No URL"},
    ]
    result = deduplicate_sources(sources)
    assert len(result) == 3, f"Expected 3, got {len(result)}"
    console.print(f"  ✅ Deduplicated 4 → 3 sources")
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 2: Multi-Source Crawlers ═══[/bold cyan]\n")

    tests = [
        ("DuckDuckGo Search", test_duckduckgo),
        ("arXiv Search", test_arxiv),
        ("Semantic Scholar", test_semantic_scholar),
        ("OpenAlex", test_openalex),
        ("GitHub Search", test_github),
        ("Jina URL Fetch", test_jina),
        ("Deduplication", test_deduplication),
    ]

    table = Table(title="Layer 2 Test Results", show_header=True)
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            table.add_row(name, "[green]PASS ✅[/green]")
            passed += 1
        except Exception as e:
            table.add_row(name, f"[red]FAIL ❌ — {str(e)[:60]}[/red]")
            failed += 1
            console.print_exception(max_frames=3)

    console.print("\n")
    console.print(table)
    console.print(
        f"\n[bold]Layer 2 Summary: {passed}/{passed+failed} tests passed[/bold]"
    )
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
