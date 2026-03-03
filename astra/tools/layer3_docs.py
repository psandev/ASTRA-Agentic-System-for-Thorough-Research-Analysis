"""
ASTRA Layer 3 — Multimodal Document Processor
Action paradigm: CODE-ACTION (loops, file I/O, data transform)

Tools:
  docling_parse_pdf, pymupdf_extract, extract_tables,
  classify_figures, qwen_vl_describe_chart

Primary: Docling (IBM, MIT license), GPU-accelerated.
Fallback: PyMuPDF for born-digital PDFs < 5 pages.
"""
from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import tool
from loguru import logger

from astra.config import get_config


# ─── Docling ─────────────────────────────────────────────────────────────────

@tool
def docling_parse_pdf(
    file_path: str,
    device: str = "cuda",
    table_mode: str = "accurate",
    ocr_enabled: bool = True,
    picture_classify: bool = True,
    formula_enrich: bool = True,
    export_format: str = "markdown",
) -> dict:
    """
    Layer 3: Parse a PDF/DOCX/HTML with Docling (IBM, MIT license), GPU-accelerated.

    Extracts: text with reading order, tables (HTML→JSON), figures with
    captions, formulas (LaTeX), code blocks.

    Args:
        file_path: Path to the document.
        device: "cuda" or "cpu".
        table_mode: "accurate" or "fast".
        ocr_enabled: Enable OCR for scanned pages.
        picture_classify: Classify figure types.
        formula_enrich: Extract LaTeX formulas.
        export_format: "markdown" or "json".

    Returns:
        {markdown, tables, figures, formulas, metadata, page_count}
    """
    cfg = get_config()
    if not cfg.astra_docling_enabled:
        logger.info("[Layer 3] Docling disabled — using PyMuPDF")
        return pymupdf_extract.invoke({"file_path": file_path})  # type: ignore

    logger.info(f"[Layer 3] Docling parsing: {file_path} device={device}")

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr_enabled
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = (table_mode == "accurate")

        # Set picture/formula options via attribute check (API varies by Docling version)
        for attr, val in [
            ("do_picture_classifier", picture_classify),
            ("generate_picture_images", picture_classify),
            ("do_formula_enrichment", formula_enrich),
        ]:
            try:
                if hasattr(pipeline_options, attr):
                    setattr(pipeline_options, attr, val)
            except Exception:
                pass

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(file_path)
        doc = result.document

        # Export markdown
        markdown = doc.export_to_markdown()

        # Extract tables
        tables = []
        for table in doc.tables:
            try:
                table_data = table.export_to_dataframe().to_dict(orient="records")
                tables.append(
                    {"html": str(table), "json": table_data}
                )
            except Exception:
                tables.append({"html": str(table), "json": []})

        # Extract figures
        figures = []
        for pic in doc.pictures:
            figures.append(
                {
                    "caption": str(pic.caption_text(doc)) if hasattr(pic, "caption_text") else "",
                    "type": str(getattr(pic, "annotations", [{}])[0].get("predicted_class", "figure")) if hasattr(pic, "annotations") and pic.annotations else "figure",
                    "path": "",
                }
            )

        # Extract formulas
        formulas = []
        try:
            for item in doc.texts:
                if hasattr(item, "label") and "formula" in str(item.label).lower():
                    formulas.append(str(item.text))
        except Exception:
            pass

        # Page count
        page_count = len(doc.pages) if hasattr(doc, "pages") else 0

        logger.info(
            f"[Layer 3] Docling: {page_count}p, {len(tables)} tables, "
            f"{len(figures)} figs, {len(formulas)} formulas"
        )
        return {
            "markdown": markdown,
            "tables": tables,
            "figures": figures,
            "formulas": formulas,
            "metadata": {"source": file_path, "tool": "docling"},
            "page_count": page_count,
        }

    except ImportError:
        logger.warning("[Layer 3] Docling not installed — falling back to PyMuPDF")
        return _pymupdf_parse(file_path)
    except Exception as e:
        logger.error(f"[Layer 3] Docling failed: {e} — falling back to PyMuPDF")
        if "OOM" in str(e) or "CUDA" in str(e):
            logger.info("[Layer 3] GPU OOM — retrying Docling on CPU")
            try:
                return docling_parse_pdf.invoke(  # type: ignore
                    {"file_path": file_path, "device": "cpu"}
                )
            except Exception:
                pass
        return _pymupdf_parse(file_path)


# ─── PyMuPDF ─────────────────────────────────────────────────────────────────

def _pymupdf_parse(file_path: str, extract_images: bool = False) -> dict:
    """Internal PyMuPDF parser (no @tool decorator — called by fallback logic)."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages = []
        full_text = []

        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            pages.append({"page_num": i + 1, "text": page_text})
            full_text.append(page_text)

        metadata = doc.metadata or {}
        page_count = len(doc)
        doc.close()

        logger.info(f"[Layer 3] PyMuPDF: {page_count} pages from {file_path}")
        return {
            "markdown": "\n\n".join(full_text),
            "tables": [],
            "figures": [],
            "formulas": [],
            "metadata": {**metadata, "source": file_path, "tool": "pymupdf"},
            "page_count": page_count,
            "pages": pages,
        }
    except ImportError:
        logger.error("[Layer 3] PyMuPDF not installed")
        return {
            "markdown": "",
            "tables": [],
            "figures": [],
            "formulas": [],
            "metadata": {"source": file_path, "tool": "none", "error": "no parser available"},
            "page_count": 0,
        }
    except Exception as e:
        logger.error(f"[Layer 3] PyMuPDF failed: {e}")
        return {
            "markdown": "",
            "tables": [],
            "figures": [],
            "formulas": [],
            "metadata": {"source": file_path, "tool": "pymupdf", "error": str(e)},
            "page_count": 0,
        }


@tool
def pymupdf_extract(file_path: str, extract_images: bool = False) -> dict:
    """
    Layer 3: Fast PDF extraction via PyMuPDF. Best for born-digital PDFs < 5 pages.

    Args:
        file_path: Path to the PDF.
        extract_images: Whether to extract embedded images.

    Returns:
        {text, pages, metadata}
    """
    cfg = get_config()
    p = Path(file_path)
    if not p.exists():
        return {
            "text": "",
            "pages": [],
            "markdown": "",
            "tables": [],
            "figures": [],
            "formulas": [],
            "metadata": {"error": f"File not found: {file_path}"},
            "page_count": 0,
        }

    result = _pymupdf_parse(file_path, extract_images)
    result["text"] = result["markdown"]
    return result


# ─── Smart dispatcher (decides Docling vs PyMuPDF) ───────────────────────────

@tool
def parse_document(file_path: str) -> dict:
    """
    Layer 3: Smart document parser — chooses Docling or PyMuPDF automatically.

    Rules:
      - If page_count ≤ ASTRA_PYMUPDF_FALLBACK_THRESHOLD_PAGES and no OCR needed
        → pymupdf_extract (fast path)
      - Otherwise → docling_parse_pdf (GPU-accelerated)

    Args:
        file_path: Path to the document file.

    Returns:
        Parsed document dict.
    """
    cfg = get_config()

    # Quick page count check via PyMuPDF (instant)
    try:
        import fitz
        doc = fitz.open(file_path)
        page_count = len(doc)
        # Check if any page has no extractable text (likely scanned)
        has_text = any(
            len(doc[i].get_text("text").strip()) > 50
            for i in range(min(3, page_count))
        )
        doc.close()
    except Exception:
        page_count = 999
        has_text = False

    if page_count <= cfg.astra_pymupdf_fallback_pages and has_text:
        logger.info(
            f"[Layer 3] Fast path (PyMuPDF): {file_path} ({page_count}p, text=True)"
        )
        return pymupdf_extract.invoke({"file_path": file_path})
    else:
        logger.info(
            f"[Layer 3] Full parse (Docling): {file_path} ({page_count}p, text={has_text})"
        )
        return docling_parse_pdf.invoke(
            {
                "file_path": file_path,
                "device": cfg.astra_docling_device,
                "table_mode": cfg.astra_docling_table_mode,
                "ocr_enabled": cfg.astra_docling_ocr_enabled,
                "picture_classify": cfg.astra_docling_picture_classify,
                "formula_enrich": cfg.astra_docling_formula_enrich,
            }
        )


# ─── Qwen-VL Vision ──────────────────────────────────────────────────────────

@tool
def qwen_vl_describe_chart(
    image_path: str,
    prompt: str = (
        "Describe this chart in detail. Extract all numerical data, axis labels, "
        "legend entries, and key trends as structured JSON."
    ),
) -> dict:
    """
    Layer 3: Describe/extract data from charts using Qwen2.5-VL-7B-Instruct.

    Args:
        image_path: Path to the chart/figure image.
        prompt: Instruction prompt for the vision model.

    Returns:
        {description, extracted_data, chart_type, confidence}
    """
    cfg = get_config()
    if not cfg.astra_vision_enabled:
        return {
            "description": "Vision model disabled",
            "extracted_data": None,
            "chart_type": "unknown",
            "confidence": 0.0,
        }

    logger.info(f"[Layer 3] Qwen-VL describing: {image_path}")

    try:
        # Encode image as base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")

        from openai import OpenAI
        client = OpenAI(
            base_url=cfg.astra_vision_base_url,
            api_key=cfg.astra_vision_api_key,
        )

        response = client.chat.completions.create(
            model=cfg.astra_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=cfg.astra_vision_temperature,
            max_tokens=cfg.astra_vision_max_tokens,
        )

        content = response.choices[0].message.content or ""

        # Try to parse extracted_data from JSON blocks
        extracted_data = None
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                extracted_data = json.loads(json_str)
            except Exception:
                pass
        elif "{" in content:
            try:
                start = content.index("{")
                end = content.rindex("}") + 1
                extracted_data = json.loads(content[start:end])
            except Exception:
                pass

        return {
            "description": content,
            "extracted_data": extracted_data,
            "chart_type": extracted_data.get("chart_type", "unknown")
            if isinstance(extracted_data, dict)
            else "unknown",
            "confidence": 0.9 if extracted_data else 0.5,
        }

    except Exception as e:
        logger.error(f"[Layer 3] Qwen-VL failed: {e}")
        return {
            "description": f"Vision analysis failed: {str(e)}",
            "extracted_data": None,
            "chart_type": "unknown",
            "confidence": 0.0,
        }


# ─── Batch processing helper ─────────────────────────────────────────────────

def batch_parse_pdfs(
    pdf_paths: list[str],
    max_workers: int = 8,
) -> list[dict]:
    """
    Process multiple PDFs in parallel using ThreadPoolExecutor.
    Used in code-action blocks for bulk document processing.

    Returns:
        List of parsed document dicts, with content_hash for deduplication.
    """
    logger.info(f"[Layer 3] Batch processing {len(pdf_paths)} PDFs (workers={max_workers})")

    def parse_single(path: str) -> dict:
        result = parse_document.invoke({"file_path": path})
        result["path"] = path
        content_preview = result.get("markdown", "")[:500]
        result["content_hash"] = hashlib.md5(content_preview.encode()).hexdigest()
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parsed = list(executor.map(parse_single, pdf_paths))

    # Deduplicate by content hash
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    for doc in parsed:
        h = doc.get("content_hash", "")
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)

    logger.info(
        f"[Layer 3] Batch complete: {len(unique)} unique docs "
        f"(deduped {len(parsed) - len(unique)})"
    )
    return unique
