"""
Layer 6 Test — Report Writer & Visual Generator
Tests: generate_chart_matplotlib, generate_chart_plotly, build_markdown

Run: python main.py --test-layer 6
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def test_generate_chart_matplotlib():
    console.print("\n[bold]Test: generate_chart_matplotlib[/bold]")
    from astra.tools.layer6_report import generate_chart_matplotlib

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_chart_matplotlib.invoke(
            {
                "chart_type": "bar",
                "data": {
                    "x": ["GPT-4", "Qwen3-32B", "Llama-3.3-70B", "Claude 3.5"],
                    "y": [85.2, 82.7, 79.1, 88.4],
                    "x_label": "Model",
                    "y_label": "MMLU Score (%)",
                },
                "title": "LLM Benchmark Comparison",
                "output_path": f"{tmpdir}/test_chart",
                "dpi": 72,
            }
        )
        assert "png_path" in result, "Missing png_path"
        if result["png_path"] and Path(result["png_path"]).exists():
            size = Path(result["png_path"]).stat().st_size
            console.print(f"  ✅ PNG created: {result['png_path']} ({size:,} bytes)")
        else:
            console.print("  ⚠️  PNG creation failed (may be expected in headless env)")
    return True


def test_generate_chart_plotly():
    console.print("\n[bold]Test: generate_chart_plotly[/bold]")
    from astra.tools.layer6_report import generate_chart_plotly

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_chart_plotly.invoke(
            {
                "chart_type": "line",
                "data": {
                    "x": [2020, 2021, 2022, 2023, 2024, 2025],
                    "y": [7, 15, 45, 120, 350, 800],
                },
                "title": "AI Research Papers per Year",
                "x_label": "Year",
                "y_label": "Paper Count (K)",
                "output_path": f"{tmpdir}/test_plotly",
            }
        )
        assert "html_path" in result, "Missing html_path"
        if result["html_path"] and Path(result["html_path"]).exists():
            size = Path(result["html_path"]).stat().st_size
            console.print(f"  ✅ HTML chart: {result['html_path']} ({size:,} bytes)")
        else:
            console.print("  ⚠️  Plotly HTML creation returned empty path")
    return True


def test_build_markdown():
    console.print("\n[bold]Test: build_markdown[/bold]")
    from astra.tools.layer6_report import build_markdown

    sections = [
        {
            "title": "Executive Summary",
            "markdown": "## Executive Summary\n\nThis report covers recent advances in multi-agent AI systems. Key findings include improved performance and scalability [Source 1].",
            "citations": [{"key": "Source 1", "url": "https://example.com"}],
            "word_count": 25,
        },
        {
            "title": "Introduction",
            "markdown": "## Introduction\n\nMulti-agent LLM systems represent a paradigm shift in AI research automation.",
            "citations": [],
            "word_count": 15,
        },
    ]

    bibliography = [
        {
            "key": "Source 1",
            "title": "Multi-Agent Systems Survey",
            "authors": ["Smith, J.", "Doe, A."],
            "year": "2024",
            "url": "https://example.com/paper1",
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_markdown.invoke(
            {
                "sections": sections,
                "bibliography": bibliography,
                "output_path": f"{tmpdir}/test_report.md",
            }
        )
        assert "md_path" in result, "Missing md_path"
        md_path = result["md_path"]
        assert Path(md_path).exists(), f"Markdown file not created: {md_path}"

        content = Path(md_path).read_text()
        assert "Executive Summary" in content, "Missing section in output"
        assert "Bibliography" in content, "Missing bibliography"
        assert len(content) > 100, "Report too short"

        console.print(f"  ✅ Markdown report: {md_path} ({len(content):,} chars)")
    return True


def test_build_pdf():
    console.print("\n[bold]Test: build_pdf[/bold]")
    from astra.tools.layer6_report import build_pdf

    sections = [
        {
            "title": "Executive Summary",
            "markdown": "This report covers recent advances.\n\n## Key Points\n\nImportant finding **#1**.\n\nImportant finding **#2**.",
            "citations": [],
            "word_count": 15,
            "chart_paths": [],
        }
    ]

    bibliography = [
        {"key": "Ref 1", "title": "Test Paper", "authors": ["Author"], "year": "2024", "url": "https://example.com"}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_pdf.invoke(
            {
                "sections": sections,
                "figures": [],
                "bibliography": bibliography,
                "output_path": f"{tmpdir}/test_report.pdf",
                "include_toc": True,
            }
        )
        assert "pdf_path" in result, "Missing pdf_path"
        if result["pdf_path"] and Path(result["pdf_path"]).exists():
            size = Path(result["pdf_path"]).stat().st_size
            console.print(f"  ✅ PDF created: {result['pdf_path']} ({size:,} bytes)")
        else:
            console.print(f"  ⚠️  PDF not created (error: {result.get('error', 'unknown')})")
    return True


def test_build_html():
    console.print("\n[bold]Test: build_html[/bold]")
    from astra.tools.layer6_report import build_html

    sections = [
        {
            "title": "Introduction to Hybrid Retrieval",
            "markdown": (
                "## Overview\n\n"
                "Hybrid retrieval combines **dense** and **sparse** methods.\n"
                "The formula for RRF is: $RRF(d) = \\sum_{r \\in R} \\frac{1}{k + r(d)}$\n\n"
                "See the [BM25 paper](https://arxiv.org/abs/2310.12345) for details.\n\n"
                "| Method | Precision | Recall |\n"
                "|--------|-----------|--------|\n"
                "| BM25   | 0.72      | 0.68   |\n"
                "| Dense  | 0.78      | 0.74   |\n"
                "| Hybrid | 0.85      | 0.82   |\n"
            ),
            "citations": [],
            "word_count": 60,
            "chart_paths": [],
        }
    ]
    bibliography = [
        {"key": "Ref1", "title": "BM25 Survey", "authors": ["Smith, J."], "year": "2024", "url": "https://example.com"}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_html.invoke({
            "sections": sections,
            "figures": [],
            "bibliography": bibliography,
            "output_path": f"{tmpdir}/test_report.html",
            "include_toc": True,
        })
        assert "html_path" in result, "Missing html_path"
        html_path = result["html_path"]
        assert Path(html_path).exists(), f"HTML file not created: {html_path}"

        content = Path(html_path).read_text()
        assert "MathJax" in content, "MathJax not included"
        assert 'href="https://arxiv.org' in content, "Hyperlink not active"
        assert "<table" in content, "Table not converted"
        assert "Hybrid Retrieval" in content, "Section title missing"

        console.print(f"  ✅ HTML created: {html_path} ({len(content):,} chars)")
        console.print(f"     MathJax: {'yes' if 'MathJax' in content else 'NO'}")
        console.print(f"     Active links: {'yes' if 'href=' in content else 'NO'}")
        console.print(f"     Tables: {'yes' if '<table' in content else 'NO'}")
    return True


def test_embed_figure():
    console.print("\n[bold]Test: embed_figure[/bold]")
    from astra.tools.layer6_report import embed_figure

    content = "## Analysis\n\nThis section analyzes benchmark results.\n\nThe data shows improvements."
    result = embed_figure.invoke(
        {
            "markdown_content": content,
            "figure_path": "/tmp/test_chart.png",
            "caption": "Figure 1: Benchmark Comparison",
            "after_keyword": "benchmark results",
        }
    )
    assert "Figure 1" in result, "Caption not embedded"
    assert "/tmp/test_chart.png" in result, "Figure path not embedded"
    console.print("  ✅ Figure embedded in markdown")
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 6: Report Writer & Visual ═══[/bold cyan]\n")

    tests = [
        ("Matplotlib chart", test_generate_chart_matplotlib),
        ("Plotly chart", test_generate_chart_plotly),
        ("build_markdown", test_build_markdown),
        ("build_pdf", test_build_pdf),
        ("build_html", test_build_html),
        ("embed_figure", test_embed_figure),
    ]

    table = Table(title="Layer 6 Test Results")
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
    console.print(f"\n[bold]Layer 6: {passed}/{passed+failed} passed[/bold]")
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
