"""
Layer 3 Test — Document Processing
Tests: PyMuPDF extraction, parse_document routing, chunk parsing

Run: python main.py --test-layer 3
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _create_sample_pdf() -> str:
    """Create a minimal test PDF using reportlab or fpdf2."""
    try:
        from reportlab.pdfgen import canvas
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        c = canvas.Canvas(tmp.name)
        c.setFont("Helvetica", 12)
        c.drawString(72, 700, "ASTRA Test Document")
        c.drawString(72, 680, "This is a sample PDF for testing document parsing.")
        c.drawString(72, 660, "It contains text that should be extracted by PyMuPDF.")
        c.drawString(72, 640, "The extraction pipeline should handle this correctly.")
        c.save()
        return tmp.name
    except ImportError:
        pass

    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt="ASTRA Test Document", ln=True)
        pdf.cell(200, 10, txt="This is a test PDF for document parsing.", ln=True)
        pdf.cell(200, 10, txt="Layer 3 document processing verification.", ln=True)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.output(tmp.name)
        return tmp.name
    except ImportError:
        pass

    return ""


def test_pymupdf_extract():
    console.print("\n[bold]Test: PyMuPDF text extraction[/bold]")
    from astra.tools.layer3_docs import pymupdf_extract

    pdf_path = _create_sample_pdf()
    if not pdf_path:
        console.print("  ⚠️  No PDF creator available — testing with fallback logic")
        result = pymupdf_extract.invoke({"file_path": "/nonexistent.pdf"})
        assert "page_count" in result
        console.print("  ✅ Error handling works correctly")
        return True

    try:
        result = pymupdf_extract.invoke({"file_path": pdf_path})
        assert "markdown" in result or "text" in result, "Missing text/markdown"
        assert result.get("page_count", 0) >= 1, "Expected pages"

        text = result.get("markdown", result.get("text", ""))
        console.print(f"  ✅ Extracted {len(text)} chars from {result.get('page_count')}p PDF")
        console.print(f"    Preview: {text[:100].strip()}")
        return True
    finally:
        Path(pdf_path).unlink(missing_ok=True)


def test_chunk_text():
    console.print("\n[bold]Test: Text chunking[/bold]")
    from astra.tools.layer4_rag import chunk_text

    long_text = """
    # Introduction

    This is the first paragraph of a long document about artificial intelligence
    and machine learning. It contains many sentences that should be properly
    chunked by the semantic text splitter.

    # Background

    The background section discusses the history of neural networks and deep learning.
    Starting from the perceptron in the 1950s, AI has come a long way. Modern
    transformer architectures have revolutionized the field.

    # Methods

    Our approach uses a combination of retrieval-augmented generation and
    knowledge graph reasoning. The hybrid retrieval system combines dense
    FAISS search with sparse BM25 for optimal performance.

    # Results

    Experiments show significant improvements over baseline methods. The
    system achieves state-of-the-art performance on multiple benchmarks.
    Ablation studies confirm the importance of each component.

    # Conclusion

    We have presented a novel approach to multi-agent research systems.
    Future work will explore additional improvements to the pipeline.
    """ * 3  # Repeat to create ~3000 char document

    chunks = chunk_text(long_text, source_url="https://example.com/test")
    assert len(chunks) > 0, "Expected chunks"
    for chunk in chunks:
        assert "id" in chunk, "Missing chunk id"
        assert "text" in chunk, "Missing chunk text"
        assert "metadata" in chunk, "Missing metadata"

    console.print(f"  ✅ Created {len(chunks)} chunks from {len(long_text)} chars")
    console.print(f"    Avg chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")
    return True


def test_parse_routing():
    console.print("\n[bold]Test: parse_document routing logic[/bold]")
    from astra.tools.layer3_docs import parse_document

    # Test with nonexistent file — should handle gracefully
    result = parse_document.invoke({"file_path": "/tmp/nonexistent_astra_test.pdf"})
    assert "markdown" in result or "text" in result or "error" in result.get("metadata", {})
    console.print("  ✅ Routing handles missing files gracefully")
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 3: Document Processing ═══[/bold cyan]\n")

    tests = [
        ("PyMuPDF extraction", test_pymupdf_extract),
        ("Text chunking", test_chunk_text),
        ("Parse routing", test_parse_routing),
    ]

    table = Table(title="Layer 3 Test Results")
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
    console.print(f"\n[bold]Layer 3: {passed}/{passed+failed} passed[/bold]")
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
