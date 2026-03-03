"""
ASTRA Layer 3 — Visual Intelligence Pipeline

Handles: PDF downloading, figure/table/chart extraction (Docling + PyMuPDF),
web article image scraping, Vision LLM description (Qwen2.5-VL), and
multi-modal RAG chunk generation.

Design philosophy:
  - Every image extracted from a source article is saved as a PNG snapshot
    to session/sources/figures/ with a sidecar .json metadata file
  - Each figure is described by the Vision LLM and stored as a text RAG chunk
    so it participates in hybrid retrieval alongside regular text chunks
  - Figures referenced in the report include real source images (not generated)

Improvements over GAIK VisionRagParser:
  - Handles web articles (HTML) in addition to PDFs
  - Smart image filtering: size, aspect ratio, URL pattern blacklisting
  - Separate table-to-image renderer (pandas + matplotlib)
  - Parallel VLM inference with rate limiting
  - Standalone RAG chunks per figure (not page-level appending)
  - Content-hash deduplication across sources
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from astra.config import get_config


# ─── Vision prompt ────────────────────────────────────────────────────────────

_FIGURE_VISION_PROMPT = """\
Analyze this image extracted from a research article or web page.

If this is a CHART, GRAPH, or DATA VISUALIZATION:
1. State the title/subtitle if visible
2. Identify the chart type (bar, line, scatter, heatmap, etc.)
3. Describe the axes, units, and key data points or trends
4. State the main finding or insight being communicated

If this is a TABLE:
1. Describe what the table compares or shows
2. Extract the most important values or patterns
3. State the main conclusion from the data

If this is a DIAGRAM, ARCHITECTURE, or FLOWCHART:
1. Identify the diagram type
2. Describe the main components and their relationships
3. State the key concept being illustrated

If this is a SCREENSHOT, PHOTO, or ILLUSTRATION:
1. Briefly describe what is shown
2. State its relevance to the document context

Respond in this EXACT format (fill in all fields):
[Type]: <chart|table|diagram|architecture|photo|screenshot|figure>
[Title]: <visible title, or "Untitled">
[Description]: <2-4 sentence description of content>
[Key Insight]: <1 sentence insight useful for answering questions about the topic>

Be precise and concise. Focus on information retrievable for research questions.\
"""

# URL patterns that indicate non-content images (icons, logos, tracking pixels)
_SKIP_URL_PATTERNS = [
    "avatar", "logo", "icon", "badge", "button", "sprite", "emoji",
    "pixel", "1x1", "tracking", "ads", "banner", "profile", "author",
    "social", "share", "like", "heart", "thumb", "favicon", "spinner",
    "loading", "placeholder", "blank", "spacer", "separator",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _url_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _is_valid_image(path: str) -> bool:
    """
    Return True only if the image file is non-corrupt and fully readable.
    Uses Pillow verify() + load() to catch truncated or damaged files.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        with Image.open(path) as img:
            img.verify()           # detects structural corruption
        with Image.open(path) as img:
            img.load()             # catches truncation that verify() misses
        return True
    except (Exception,):           # UnidentifiedImageError, OSError, SyntaxError, etc.
        return False


def _file_hash(path: str) -> str:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except Exception:
        return ""


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an image file."""
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    return b64, mime


def _pil_from_docling(img_obj):
    """Extract a PIL.Image from various Docling image object wrappers."""
    if img_obj is None:
        return None
    if hasattr(img_obj, "save"):          # already a PIL image
        return img_obj
    for attr in ("pil_image", "image", "_image"):
        pil = getattr(img_obj, attr, None)
        if pil is not None and hasattr(pil, "save"):
            return pil
    if hasattr(img_obj, "to_pil"):
        try:
            return img_obj.to_pil()
        except Exception:
            pass
    return None


def _save_metadata(fig: dict, figures_dir: Path) -> None:
    """Write a .json sidecar next to the PNG snapshot."""
    try:
        img_path = fig.get("image_path", "")
        if not img_path:
            return
        meta_path = Path(img_path).with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {k: v for k, v in fig.items() if k != "image_path"},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


# ─── 1. PDF Downloading ───────────────────────────────────────────────────────

def download_pdfs_batch(
    sources: list[dict],
    output_dir: str,
    max_workers: int = 4,
    timeout: int = 15,
) -> list[str]:
    """
    Download PDFs from sources that expose a pdf_url.
    Handles arXiv, OpenAlex OA, Semantic Scholar open-access PDFs.

    Returns list of local PDF paths (cached + newly downloaded).
    """
    cfg = get_config()
    if not cfg.astra_pdf_download_enabled:
        logger.info("[Layer 3 Vision] PDF download disabled (ASTRA_PDF_DOWNLOAD_ENABLED=false)")
        return []

    pdf_dir = Path(output_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Optional[str], str]] = []  # (url_or_None, local_path)
    for src in sources:
        pdf_url = src.get("pdf_url", "")
        if not pdf_url or not pdf_url.startswith("http"):
            continue
        # Normalise arXiv abs → pdf URL
        if "arxiv.org/abs/" in pdf_url:
            arxiv_id = pdf_url.split("/abs/")[-1].strip("/").split("v")[0]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        safe = _url_hash(pdf_url) + ".pdf"
        local = pdf_dir / safe
        if local.exists() and local.stat().st_size > 4096:
            jobs.append((None, str(local)))   # already cached
        else:
            jobs.append((pdf_url, str(local)))

    if not jobs:
        return []

    def _dl(url: Optional[str], local: str) -> Optional[str]:
        if url is None:
            return local
        try:
            r = httpx.get(
                url, timeout=timeout, follow_redirects=True,
                headers={"User-Agent": "ASTRA-Research-Bot/1.0 (academic use)"},
            )
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "pdf" not in ct and r.content[:4] != b"%PDF":
                logger.warning(f"[Layer 3 Vision] Not a PDF response from {url[:60]}")
                return None
            Path(local).write_bytes(r.content)
            logger.info(f"[Layer 3 Vision] PDF saved: {Path(local).name} ({len(r.content)//1024}KB)")
            return local
        except Exception as e:
            logger.warning(f"[Layer 3 Vision] PDF download failed {url[:60]}: {e}")
            return None

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = {exe.submit(_dl, u, p): p for u, p in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)

    logger.info(f"[Layer 3 Vision] PDFs ready: {len(results)}/{len(jobs)}")
    return results


# ─── 2a. Figure Extraction from PDFs — Docling ───────────────────────────────

def _extract_pdf_figures_docling(
    pdf_path: str,
    figures_dir: Path,
    source_url: str,
    min_size: int,
) -> list[dict]:
    """Extract figures from PDF using Docling with generate_picture_images=True."""
    figures: list[dict] = []

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        opts = PdfPipelineOptions()
        opts.do_ocr = False                      # skip OCR for speed
        opts.do_table_structure = True
        opts.generate_picture_images = True       # ← key flag for image extraction
        opts.generate_page_images = False
        opts.do_formula_enrichment = False

        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        result = conv.convert(pdf_path)
        doc = result.document

        stem = _url_hash(pdf_path)

        # ── Pictures / Charts / Figures ──────────────────────────────────────
        for idx, entry in enumerate(doc.iterate_items()):
            item = entry[0] if isinstance(entry, tuple) else entry
            if not hasattr(item, "label"):
                continue
            label = str(item.label).lower()
            if "picture" not in label and "figure" not in label:
                continue
            if not (hasattr(item, "image") and item.image):
                continue

            page_num = None
            if hasattr(item, "prov") and item.prov:
                page_num = item.prov[0].page_no

            caption = ""
            if hasattr(item, "caption_text"):
                try:
                    caption = str(item.caption_text(doc) or "")[:400]
                except Exception:
                    pass

            pil = _pil_from_docling(item.image)
            if pil is None:
                continue
            w, h = pil.size
            if w < min_size or h < min_size:
                continue

            img_path = figures_dir / f"fig_{stem}_{idx:03d}.png"
            pil.save(str(img_path), format="PNG")

            if not _is_valid_image(str(img_path)):
                logger.warning(
                    f"[Layer 3 Vision] Skipping corrupt/truncated image: {img_path.name}"
                )
                img_path.unlink(missing_ok=True)
                continue

            figures.append({
                "image_path": str(img_path),
                "caption": caption,
                "page": page_num,
                "source_url": source_url,
                "pdf_path": pdf_path,
                "figure_type": "figure",
                "width": w,
                "height": h,
                "extraction_method": "docling",
            })

        # ── Tables → PNG via matplotlib ───────────────────────────────────────
        for t_idx, table in enumerate(doc.tables):
            try:
                img_path = _render_table_png(table, figures_dir, stem, t_idx)
                if img_path:
                    figures.append({
                        "image_path": img_path,
                        "caption": f"Table {t_idx + 1}",
                        "page": None,
                        "source_url": source_url,
                        "pdf_path": pdf_path,
                        "figure_type": "table",
                        "extraction_method": "docling",
                    })
            except Exception:
                pass

        logger.info(
            f"[Layer 3 Vision] Docling: {len(figures)} items from {Path(pdf_path).name}"
        )
        return figures

    except ImportError:
        logger.warning("[Layer 3 Vision] Docling not installed, falling back to PyMuPDF")
        return []
    except Exception as e:
        logger.warning(f"[Layer 3 Vision] Docling extraction failed: {e}")
        return []


def _render_dataframe_png(df, figures_dir: Path, stem: str, idx: int) -> str:
    """Render a pandas DataFrame to a styled PNG table using matplotlib."""
    try:
        if df.empty or len(df) < 2 or len(df.columns) < 2:
            return ""

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows, n_cols = len(df), len(df.columns)
        fig_w = max(6, min(14, n_cols * 1.8))
        fig_h = max(2, min(10, n_rows * 0.45 + 1.2))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=list(df.columns),
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)

        # Header row styling
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_facecolor("#003366")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")

        # Alternating row colours
        for i in range(1, n_rows + 1):
            for j in range(n_cols):
                if i % 2 == 0:
                    tbl[i, j].set_facecolor("#f0f4fb")

        plt.tight_layout()
        out = str(figures_dir / f"table_{stem}_{idx:03d}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out
    except Exception as e:
        logger.debug(f"[Layer 3 Vision] Table render failed: {e}")
        return ""


def _render_table_png(table, figures_dir: Path, stem: str, idx: int) -> str:
    """Render a Docling table to PNG (thin wrapper over _render_dataframe_png)."""
    try:
        df = table.export_to_dataframe()
    except Exception:
        return ""
    return _render_dataframe_png(df, figures_dir, stem, idx)


# ─── 2b. Figure Extraction from PDFs — PyMuPDF fallback ─────────────────────

def _extract_pdf_figures_pymupdf(
    pdf_path: str,
    figures_dir: Path,
    source_url: str,
    min_size: int,
) -> list[dict]:
    """Extract images from PDF using PyMuPDF (raw pixel extraction)."""
    figures: list[dict] = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        stem = _url_hash(pdf_path)

        for page_idx, page in enumerate(doc):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    w = base_img.get("width", 0)
                    h = base_img.get("height", 0)
                    if w < min_size or h < min_size:
                        continue
                    if w / max(h, 1) > 10 or h / max(w, 1) > 10:
                        continue   # banner-like strip

                    img_bytes = base_img["image"]
                    img_ext = base_img.get("ext", "png")

                    # Normalise to PNG
                    from PIL import Image
                    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                    img_path = figures_dir / f"fig_{stem}_{page_idx:03d}_{img_idx:02d}.png"
                    pil.save(str(img_path), format="PNG")

                    figures.append({
                        "image_path": str(img_path),
                        "caption": "",
                        "page": page_idx + 1,
                        "source_url": source_url,
                        "pdf_path": pdf_path,
                        "figure_type": "figure",
                        "width": w,
                        "height": h,
                        "extraction_method": "pymupdf",
                    })
                except Exception:
                    continue

        doc.close()
        logger.info(
            f"[Layer 3 Vision] PyMuPDF: {len(figures)} images from {Path(pdf_path).name}"
        )
    except ImportError:
        logger.error("[Layer 3 Vision] PyMuPDF not installed")
    except Exception as e:
        logger.error(f"[Layer 3 Vision] PyMuPDF extraction failed: {e}")

    return figures


# ─── 2c. Figure Extraction from PDF — dispatcher ─────────────────────────────

def extract_figures_from_pdf(
    pdf_path: str,
    figures_dir: str,
    source_url: str = "",
) -> list[dict]:
    """
    Extract figures, charts, and tables from a PDF.
    Tries Docling first (structured extraction with captions),
    falls back to PyMuPDF (raw pixel extraction).
    """
    cfg = get_config()
    min_sz = cfg.astra_vision_min_image_size
    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    figs: list[dict] = []
    if cfg.astra_docling_enabled:
        figs = _extract_pdf_figures_docling(pdf_path, fdir, source_url, min_sz)

    if not figs:
        figs = _extract_pdf_figures_pymupdf(pdf_path, fdir, source_url, min_sz)

    return figs


# ─── 3. Figure Extraction from Web Articles ───────────────────────────────────

def extract_figures_from_web(
    url: str,
    html_content: str,
    figures_dir: str,
    max_images: int = 6,
) -> list[dict]:
    """
    Scrape and download images from a web article's HTML.

    Filtering strategy:
    - Skip data URIs (embedded blobs, usually icons)
    - Skip SVGs in inline context (usually icons)
    - Skip URLs matching known non-content patterns
    - Require minimum pixel dimensions after download
    - Skip very elongated strips (aspect ratio >8:1)
    - Prefer <figure>/<figcaption> elements first

    Returns list of figure dicts with local image_path.
    """
    cfg = get_config()
    min_sz = cfg.astra_vision_min_image_size

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("[Layer 3 Vision] BeautifulSoup not installed — skipping web images")
        return []

    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    soup = BeautifulSoup(html_content, "html.parser")

    # Collect (img_tag, caption, priority) in priority order
    candidates: list[tuple] = []

    # Priority 1: <figure> elements with captions
    for fig_el in soup.find_all("figure"):
        img = fig_el.find("img")
        if not img:
            continue
        cap_el = fig_el.find("figcaption")
        cap = cap_el.get_text(strip=True)[:300] if cap_el else ""
        candidates.append((img, cap, 1))

    # Priority 2: images in article body
    for img in soup.find_all("img"):
        if img.parent and img.parent.name == "figure":
            continue
        candidates.append((img, img.get("alt", "")[:200], 2))

    figures: list[dict] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    for img_tag, caption, _priority in candidates:
        if len(figures) >= max_images:
            break

        # Resolve src
        src = (
            img_tag.get("src") or img_tag.get("data-src") or
            img_tag.get("data-lazy-src") or img_tag.get("data-original") or ""
        )
        if not src:
            continue
        if src.startswith("data:"):
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = base + src
        elif not src.startswith("http"):
            src = urljoin(url, src)

        if src in seen_urls:
            continue
        seen_urls.add(src)

        # Skip non-content URL patterns
        src_lower = src.lower()
        if any(p in src_lower for p in _SKIP_URL_PATTERNS):
            continue
        # Skip SVGs in non-figure context
        if src_lower.endswith(".svg") and _priority > 1:
            continue

        # Download image
        try:
            r = httpx.get(
                src, timeout=12, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ASTRA-Bot/1.0)"},
            )
            r.raise_for_status()
            if "image" not in r.headers.get("content-type", ""):
                continue

            # Dimension + dedup check
            from PIL import Image
            img_obj = Image.open(BytesIO(r.content)).convert("RGB")
            w, h = img_obj.size

            if w < min_sz or h < min_sz:
                continue
            if w / max(h, 1) > 8 or h / max(w, 1) > 8:
                continue   # banner strip

            img_bytes_png = BytesIO()
            img_obj.save(img_bytes_png, format="PNG")
            content_hash = hashlib.md5(img_bytes_png.getvalue()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            img_path = fdir / f"fig_web_{_url_hash(src)}.png"
            img_path.write_bytes(img_bytes_png.getvalue())

            if not _is_valid_image(str(img_path)):
                logger.warning(
                    f"[Layer 3 Vision] Skipping corrupt/truncated web image: {img_path.name}"
                )
                img_path.unlink(missing_ok=True)
                continue

            figures.append({
                "image_path": str(img_path),
                "caption": caption or img_tag.get("alt", ""),
                "page": None,
                "source_url": url,
                "figure_type": "figure",
                "width": w,
                "height": h,
                "alt_text": img_tag.get("alt", ""),
                "extraction_method": "web_scrape",
            })

        except Exception as e:
            logger.debug(f"[Layer 3 Vision] Image download failed {src[:60]}: {e}")
            continue

    logger.info(f"[Layer 3 Vision] Web scrape: {len(figures)} images from {url[:70]}")
    return figures


# ─── 3b. Table Extraction from Web Articles ───────────────────────────────────

def _is_data_table(tag) -> bool:
    """Return True if a BeautifulSoup <table> tag looks like a real data table."""
    # Skip tables inside navigation/structural containers
    for parent in tag.parents:
        if getattr(parent, "name", None) in ("nav", "header", "footer", "aside", "form"):
            return False
    rows = tag.find_all("tr")
    if len(rows) < 2:
        return False
    first_cells = rows[0].find_all(["td", "th"])
    if len(first_cells) < 2 or len(first_cells) > 20:
        return False   # too narrow = likely layout; too wide = calendar/spreadsheet
    all_cells = tag.find_all(["td", "th"])
    non_empty = sum(1 for c in all_cells if c.get_text(strip=True))
    return non_empty >= 4


def _extract_web_tables(
    url: str,
    html_content: str,
    figures_dir: str,
    max_tables: int = 5,
) -> list[dict]:
    """
    Extract <table> elements from raw HTML, render to PNG, return figure dicts.
    Uses same dict schema as extract_figures_from_web() for pipeline compatibility.
    """
    import pandas as pd
    from bs4 import BeautifulSoup
    from PIL import Image

    fdir = Path(figures_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(html_content, "html.parser")

    results: list[dict] = []
    stem = _url_hash(url)

    for idx, tbl_tag in enumerate(soup.find_all("table")):
        if len(results) >= max_tables:
            break
        if not _is_data_table(tbl_tag):
            continue

        # Caption: <caption> element > preceding heading > fallback
        cap_el = tbl_tag.find("caption")
        caption = cap_el.get_text(strip=True) if cap_el else ""
        if not caption:
            prev = tbl_tag.find_previous(["h2", "h3", "h4"])
            if prev:
                caption = prev.get_text(strip=True)[:200]
        if not caption:
            caption = f"Table {idx + 1}"

        # Parse to DataFrame
        try:
            dfs = pd.read_html(str(tbl_tag))
        except Exception:
            continue
        if not dfs:
            continue
        df = dfs[0]
        if df.empty or len(df) < 2 or len(df.columns) < 2:
            continue

        # Render to PNG
        out_path = _render_dataframe_png(df, fdir, stem, idx)
        if not out_path:
            continue
        if not _is_valid_image(out_path):
            Path(out_path).unlink(missing_ok=True)
            continue

        try:
            with Image.open(out_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        results.append({
            "image_path": out_path,
            "caption": caption,
            "page": None,
            "source_url": url,
            "figure_type": "table",
            "width": w,
            "height": h,
            "alt_text": caption,
            "extraction_method": "web_table",
        })

    logger.info(f"[Layer 3 Vision] Web table extraction: {len(results)} tables from {url[:70]}")
    return results


# ─── 4. Vision LLM Description ───────────────────────────────────────────────

def describe_figure(
    image_path: str,
    caption: str = "",
    source_url: str = "",
    context: str = "",
) -> dict:
    """
    Send an image to Qwen2.5-VL (port 8001) for structured description.

    Returns:
        {figure_type, title, description, key_insight, confidence, raw_response}
    """
    cfg = get_config()

    empty = {
        "figure_type": "figure", "title": "",
        "description": caption or "No description available",
        "key_insight": caption, "confidence": 0.0, "raw_response": "",
    }

    if not cfg.astra_vision_enabled:
        return {**empty, "description": caption or "Vision disabled"}

    if not Path(image_path).exists():
        logger.warning(f"[Layer 3 Vision] Image missing: {image_path}")
        return empty

    try:
        b64, mime = _image_to_base64(image_path)

        prompt = _FIGURE_VISION_PROMPT
        if caption:
            prompt += f"\n\nCaption/alt-text from source: '{caption}'"
        if context:
            prompt += f"\nDocument context: {context[:300]}"

        from openai import OpenAI
        client = OpenAI(
            base_url=cfg.astra_vision_base_url,
            api_key=cfg.astra_vision_api_key,
        )
        response = client.chat.completions.create(
            model=cfg.astra_vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=cfg.astra_vision_temperature,
            max_tokens=cfg.astra_vision_max_tokens,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_vision_response(raw)
        parsed["raw_response"] = raw
        logger.debug(
            f"[Layer 3 Vision] Described {Path(image_path).name} → "
            f"{parsed['figure_type']}: {parsed['title'][:50]}"
        )
        return parsed

    except Exception as e:
        logger.error(f"[Layer 3 Vision] VLM failed for {image_path}: {e}")
        return {**empty, "confidence": 0.0}


def _parse_vision_response(content: str) -> dict:
    """Parse the structured [Field]: value response from the vision model."""
    result = {
        "figure_type": "figure",
        "title": "",
        "description": content,
        "key_insight": "",
        "confidence": 0.8,
    }
    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("[Type]:"):
            result["figure_type"] = s[7:].strip().lower()
        elif s.startswith("[Title]:"):
            result["title"] = s[8:].strip()
        elif s.startswith("[Description]:"):
            result["description"] = s[14:].strip()
        elif s.startswith("[Key Insight]:"):
            result["key_insight"] = s[14:].strip()

    # Normalise type vocabulary
    ft = result["figure_type"]
    if any(t in ft for t in ["chart", "graph", "bar", "line", "scatter", "plot", "histogram"]):
        result["figure_type"] = "chart"
    elif "table" in ft:
        result["figure_type"] = "table"
    elif any(t in ft for t in ["diagram", "architecture", "flow", "network", "pipeline"]):
        result["figure_type"] = "diagram"
    elif any(t in ft for t in ["screenshot", "screen"]):
        result["figure_type"] = "screenshot"
    elif any(t in ft for t in ["photo", "photograph", "image"]):
        result["figure_type"] = "photo"
    else:
        result["figure_type"] = "figure"

    return result


# ─── 5. RAG chunk builder ─────────────────────────────────────────────────────

def _build_figure_rag_chunk(fig: dict) -> dict:
    """
    Build a rich text RAG chunk from an enriched figure dict.
    The chunk participates in FAISS+BM25 hybrid retrieval.
    """
    ft = fig.get("figure_type", "figure").upper()
    title = fig.get("title", "")
    description = fig.get("description", "")
    key_insight = fig.get("key_insight", "")
    caption = fig.get("caption", "")
    source = fig.get("source_url", "")
    img_path = fig.get("image_path", "")

    parts: list[str] = []
    parts.append(f"[{ft}] {title}" if title and title.lower() != "untitled" else f"[{ft}]")
    if caption and caption.lower() != title.lower():
        parts.append(f"Caption: {caption}")
    if description:
        parts.append(description)
    if key_insight and key_insight != description:
        parts.append(f"Key insight: {key_insight}")
    if source:
        parts.append(f"Source: {source}")

    text = "\n".join(parts)

    return {
        "id": _url_hash(img_path + text[:30]),
        "text": text,
        "metadata": {
            "source_url": source,
            "chunk_type": "figure",
            "figure_type": fig.get("figure_type", "figure"),
            "image_path": img_path,
            "caption": caption,
            "title": title,
            "chunk_index": 0,          # will be overwritten by caller
        },
    }


# ─── 6. Main Orchestrator ─────────────────────────────────────────────────────

def process_visual_sources(
    sources: list[dict],
    processed_docs: list[dict],
    session_dir: str,
) -> tuple[list[dict], list[dict]]:
    """
    Full visual intelligence pipeline for one research session.

    Steps:
      1. Download PDFs from collected sources (arXiv, OA, S2)
      2. Extract figures/tables from PDFs via Docling (+ PyMuPDF fallback)
      3. Extract images from web articles via HTML scraping
      4. Deduplicate by content hash
      5. Describe all figures with Qwen2.5-VL (parallel, rate-limited)
      6. Save PNG snapshots + .json sidecar metadata files
      7. Build RAG text chunks from descriptions
      8. Write master figures_metadata.json index

    Returns:
      (figures_for_report, figure_rag_chunks)
      - figures_for_report: enriched dicts including image_path + description
      - figure_rag_chunks: text chunks ready for FAISS+BM25 indexing
    """
    cfg = get_config()
    figures_dir = Path(session_dir) / "sources" / "figures"
    pdfs_dir = Path(session_dir) / "sources" / "pdfs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    all_raw: list[dict] = []

    # ── Step 1: Download PDFs ──────────────────────────────────────────────────
    logger.info("[Layer 3 Vision] Step 1: Downloading PDFs from sources...")
    pdf_paths = download_pdfs_batch(sources, str(pdfs_dir), max_workers=4)

    # ── Step 2: Extract from PDFs ─────────────────────────────────────────────
    logger.info(f"[Layer 3 Vision] Step 2: Extracting figures from {len(pdf_paths)} PDFs...")
    for pdf_path in pdf_paths:
        # Resolve back to original source URL
        pdf_name = Path(pdf_path).name
        source_url = next(
            (s.get("url", "") for s in sources
             if _url_hash(s.get("pdf_url", "")) + ".pdf" == pdf_name),
            pdf_path,
        )
        figs = extract_figures_from_pdf(pdf_path, str(figures_dir), source_url)
        all_raw.extend(figs)
        if len(all_raw) >= cfg.astra_vision_max_figures * 2:
            break

    # ── Step 3: Extract from web articles ────────────────────────────────────
    logger.info("[Layer 3 Vision] Step 3: Extracting images from web articles...")
    web_docs = [
        d for d in processed_docs
        if d.get("processing_method") in ("jina", "firecrawl")
        and len(d.get("markdown", "")) > 300
    ][:12]   # cap at 12 web pages

    for doc in web_docs:
        if len(all_raw) >= cfg.astra_vision_max_figures * 2:
            break
        src_url = doc.get("source_url", "")
        # Use raw_html if stored, otherwise we can't scrape images reliably
        raw_html = doc.get("raw_html", "")
        if not raw_html:
            continue
        try:
            figs = extract_figures_from_web(
                src_url, raw_html, str(figures_dir), max_images=5
            )
            all_raw.extend(figs)
        except Exception as e:
            logger.warning(f"[Layer 3 Vision] Web image scrape failed for {src_url}: {e}")

    # ── Step 3b: Extract HTML tables from web articles ────────────────────────
    if cfg.astra_html_table_vision:
        logger.info("[Layer 3 Vision] Step 3b: Extracting HTML tables from web articles...")
        for doc in web_docs:
            if len(all_raw) >= cfg.astra_vision_max_figures * 2:
                break
            src_url = doc.get("source_url", "")
            raw_html = doc.get("raw_html", "")
            if not raw_html:
                continue
            try:
                tbls = _extract_web_tables(src_url, raw_html, str(figures_dir), max_tables=5)
                all_raw.extend(tbls)
            except Exception as e:
                logger.warning(f"[Layer 3 Vision] Web table extraction failed for {src_url}: {e}")

    logger.info(f"[Layer 3 Vision] Raw figures collected: {len(all_raw)}")

    # ── Step 4: Deduplicate by content hash ──────────────────────────────────
    seen: set[str] = set()
    unique: list[dict] = []
    for fig in all_raw:
        p = fig.get("image_path", "")
        if not p or not Path(p).exists():
            continue
        h = _file_hash(p)
        if h and h not in seen:
            seen.add(h)
            unique.append(fig)

    logger.info(f"[Layer 3 Vision] After dedup: {len(unique)} unique figures")

    if not unique:
        logger.info("[Layer 3 Vision] No figures found — skipping vision description")
        return [], []

    # ── Step 5: Vision LLM description (parallel, 2 workers) ─────────────────
    to_describe = unique[:cfg.astra_vision_max_figures]
    logger.info(f"[Layer 3 Vision] Step 5: Describing {len(to_describe)} figures with VLM...")

    enriched_figures: list[dict] = []

    def _describe_one(fig: dict) -> dict:
        desc = describe_figure(
            image_path=fig["image_path"],
            caption=fig.get("caption", ""),
            source_url=fig.get("source_url", ""),
        )
        return {**fig, **desc}

    # Vision LLM is sequential by nature; use 2 workers max to avoid OOM
    with ThreadPoolExecutor(max_workers=2) as exe:
        futures = [exe.submit(_describe_one, f) for f in to_describe]
        for fut in as_completed(futures):
            try:
                enriched_figures.append(fut.result())
            except Exception as e:
                logger.warning(f"[Layer 3 Vision] Figure description error: {e}")

    # ── Step 6: Save metadata sidecars ───────────────────────────────────────
    for fig in enriched_figures:
        _save_metadata(fig, figures_dir)

    # ── Step 7: Build RAG chunks ──────────────────────────────────────────────
    figure_chunks: list[dict] = []
    for i, fig in enumerate(enriched_figures):
        chunk = _build_figure_rag_chunk(fig)
        chunk["metadata"]["chunk_index"] = i
        figure_chunks.append(chunk)

    # ── Step 8: Write master index ────────────────────────────────────────────
    index_path = figures_dir / "figures_index.json"
    index_path.write_text(
        json.dumps(enriched_figures, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(
        f"[Layer 3 Vision] Complete: {len(enriched_figures)} figures described, "
        f"{len(figure_chunks)} RAG chunks — index: {index_path}"
    )
    return enriched_figures, figure_chunks
