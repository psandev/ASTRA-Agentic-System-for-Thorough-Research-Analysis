"""
ASTRA Configuration — loads all settings from .env via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AstraConfig(BaseSettings):
    """All ASTRA settings mapped from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── vLLM / Main LLM ───────────────────────────────────────────────────────
    vllm_base_url: str = Field("http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field("not-needed", alias="VLLM_API_KEY")
    astra_main_model: str = Field("astra-main", alias="ASTRA_MAIN_MODEL_ALIAS")
    astra_main_max_tokens: int = Field(4096, alias="ASTRA_MAIN_MAX_TOKENS")
    astra_main_temperature: float = Field(0.7, alias="ASTRA_MAIN_TEMPERATURE")
    astra_main_temperature_structured: float = Field(
        0.1, alias="ASTRA_MAIN_TEMPERATURE_STRUCTURED"
    )

    # ── Judge LLM ─────────────────────────────────────────────────────────────
    astra_judge_model: str = Field("astra-main", alias="ASTRA_JUDGE_MODEL_ALIAS")
    astra_judge_base_url: str = Field(
        "http://localhost:8000/v1", alias="ASTRA_JUDGE_BASE_URL"
    )
    astra_judge_temperature: float = Field(0.0, alias="ASTRA_JUDGE_TEMPERATURE")
    astra_judge_max_tokens: int = Field(2048, alias="ASTRA_JUDGE_MAX_TOKENS")

    # ── Vision LLM ────────────────────────────────────────────────────────────
    astra_vision_base_url: str = Field(
        "http://localhost:8001/v1", alias="ASTRA_VISION_BASE_URL"
    )
    astra_vision_api_key: str = Field("not-needed", alias="ASTRA_VISION_API_KEY")
    astra_vision_model: str = Field("astra-vision", alias="ASTRA_VISION_MODEL_ALIAS")
    astra_vision_enabled: bool = Field(True, alias="ASTRA_VISION_ENABLED")
    astra_vision_temperature: float = Field(0.1, alias="ASTRA_VISION_TEMPERATURE")
    astra_vision_max_tokens: int = Field(1024, alias="ASTRA_VISION_MAX_TOKENS")

    # ── Embeddings & Reranking ─────────────────────────────────────────────────
    astra_embedding_model: str = Field("BAAI/bge-m3", alias="ASTRA_EMBEDDING_MODEL")
    astra_embedding_device: str = Field("cpu", alias="ASTRA_EMBEDDING_DEVICE")
    astra_embedding_batch_size: int = Field(128, alias="ASTRA_EMBEDDING_BATCH_SIZE")
    astra_embedding_dim: int = Field(1024, alias="ASTRA_EMBEDDING_DIM")
    astra_reranker_model: str = Field(
        "BAAI/bge-reranker-v2-m3", alias="ASTRA_RERANKER_MODEL"
    )
    astra_reranker_device: str = Field("cpu", alias="ASTRA_RERANKER_DEVICE")
    astra_reranker_batch_size: int = Field(32, alias="ASTRA_RERANKER_BATCH_SIZE")
    astra_reranker_top_k: int = Field(10, alias="ASTRA_RERANKER_TOP_K")

    # ── Docling ───────────────────────────────────────────────────────────────
    astra_docling_enabled: bool = Field(True, alias="ASTRA_DOCLING_ENABLED")
    astra_docling_device: str = Field("cuda", alias="ASTRA_DOCLING_DEVICE")
    astra_docling_ocr_enabled: bool = Field(True, alias="ASTRA_DOCLING_OCR_ENABLED")
    astra_docling_table_mode: str = Field("accurate", alias="ASTRA_DOCLING_TABLE_MODE")
    astra_docling_picture_classify: bool = Field(
        True, alias="ASTRA_DOCLING_PICTURE_CLASSIFY"
    )
    astra_docling_formula_enrich: bool = Field(
        True, alias="ASTRA_DOCLING_FORMULA_ENRICH"
    )
    astra_pymupdf_fallback_pages: int = Field(
        5, alias="ASTRA_PYMUPDF_FALLBACK_THRESHOLD_PAGES"
    )

    # ── Search ────────────────────────────────────────────────────────────────
    astra_duckduckgo_max_results: int = Field(
        10, alias="ASTRA_DUCKDUCKGO_MAX_RESULTS"
    )
    astra_jina_base_url: str = Field("https://r.jina.ai", alias="ASTRA_JINA_BASE_URL")
    astra_jina_enabled: bool = Field(True, alias="ASTRA_JINA_ENABLED")
    astra_firecrawl_base_url: str = Field(
        "http://localhost:3002", alias="ASTRA_FIRECRAWL_BASE_URL"
    )
    astra_firecrawl_enabled: bool = Field(True, alias="ASTRA_FIRECRAWL_ENABLED")
    astra_firecrawl_api_key: str = Field(
        "local-no-key-needed", alias="ASTRA_FIRECRAWL_API_KEY"
    )
    astra_searxng_enabled: bool = Field(False, alias="ASTRA_SEARXNG_ENABLED")
    astra_searxng_base_url: str = Field(
        "http://localhost:8080", alias="ASTRA_SEARXNG_BASE_URL"
    )

    # ── Academic APIs ──────────────────────────────────────────────────────────
    astra_arxiv_max_results: int = Field(20, alias="ASTRA_ARXIV_MAX_RESULTS")
    astra_s2_base_url: str = Field(
        "https://api.semanticscholar.org/graph/v1", alias="ASTRA_S2_BASE_URL"
    )
    astra_s2_api_key: Optional[str] = Field(None, alias="ASTRA_S2_API_KEY")
    astra_s2_max_results: int = Field(20, alias="ASTRA_S2_MAX_RESULTS")
    astra_openalex_base_url: str = Field(
        "https://api.openalex.org", alias="ASTRA_OPENALEX_BASE_URL"
    )
    astra_openalex_email: str = Field("", alias="ASTRA_OPENALEX_EMAIL")
    astra_openalex_max_results: int = Field(20, alias="ASTRA_OPENALEX_MAX_RESULTS")
    astra_pubmed_base_url: str = Field(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", alias="ASTRA_PUBMED_BASE_URL"
    )
    astra_pubmed_max_results: int = Field(15, alias="ASTRA_PUBMED_MAX_RESULTS")
    astra_github_token: Optional[str] = Field(None, alias="ASTRA_GITHUB_TOKEN")
    astra_github_max_results: int = Field(10, alias="ASTRA_GITHUB_MAX_RESULTS")

    # ── Vector Stores & RAG ───────────────────────────────────────────────────
    astra_chroma_persist_dir: str = Field(
        "./data/chroma_db", alias="ASTRA_CHROMA_PERSIST_DIR"
    )
    astra_faiss_index_type: str = Field(
        "IndexFlatIP", alias="ASTRA_FAISS_INDEX_TYPE"
    )
    astra_retrieval_dense_weight: float = Field(
        0.6, alias="ASTRA_RETRIEVAL_DENSE_WEIGHT"
    )
    astra_retrieval_sparse_weight: float = Field(
        0.4, alias="ASTRA_RETRIEVAL_SPARSE_WEIGHT"
    )
    astra_retrieval_top_k: int = Field(20, alias="ASTRA_RETRIEVAL_TOP_K")
    astra_retrieval_rerank_top_k: int = Field(10, alias="ASTRA_RETRIEVAL_RERANK_TOP_K")
    # Minimum bge-m3 cosine similarity a scraped figure must score against a
    # section query to be placed inline in that section.  Figures below this
    # threshold are off-topic and are silently excluded.  0.28 is calibrated
    # to reject clearly unrelated figures while accepting relevant ones even
    # when VLM descriptions are empty (caption-only matching).
    astra_figure_relevance_threshold: float = Field(
        0.28, alias="ASTRA_FIGURE_RELEVANCE_THRESHOLD"
    )
    astra_lightrag_enabled: bool = Field(True, alias="ASTRA_LIGHTRAG_ENABLED")
    astra_lightrag_working_dir: str = Field(
        "./data/lightrag", alias="ASTRA_LIGHTRAG_WORKING_DIR"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    astra_chunk_size: int = Field(512, alias="ASTRA_CHUNK_SIZE")
    astra_chunk_overlap: int = Field(64, alias="ASTRA_CHUNK_OVERLAP")

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(True, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(
        "ASTRA-Research-Agent", alias="LANGCHAIN_PROJECT"
    )
    langchain_api_key: Optional[str] = Field(None, alias="LANGCHAIN_API_KEY")

    # ── LangGraph ─────────────────────────────────────────────────────────────
    astra_langgraph_recursion_limit: int = Field(
        100, alias="ASTRA_LANGGRAPH_RECURSION_LIMIT"
    )
    astra_langgraph_checkpoint_dir: str = Field(
        "./data/checkpoints", alias="ASTRA_LANGGRAPH_CHECKPOINT_DIR"
    )
    astra_langgraph_sqlite_path: str = Field(
        "./data/checkpoints/astra_state.db", alias="ASTRA_LANGGRAPH_SQLITE_PATH"
    )

    # ── Output ────────────────────────────────────────────────────────────────
    astra_output_dir: str = Field("./output/reports", alias="ASTRA_OUTPUT_DIR")
    astra_docx_template: str = Field(
        "./templates/astra_report_template.docx", alias="ASTRA_DOCX_TEMPLATE"
    )
    astra_citation_style: str = Field("apa", alias="ASTRA_CITATION_STYLE")

    # ── Evaluation Thresholds ─────────────────────────────────────────────────
    eval_min_factual_accuracy: float = Field(
        0.70, alias="ASTRA_EVAL_MIN_FACTUAL_ACCURACY"
    )
    eval_min_citation_faithfulness: float = Field(
        0.80, alias="ASTRA_EVAL_MIN_CITATION_FAITHFULNESS"
    )
    eval_min_completeness: float = Field(0.60, alias="ASTRA_EVAL_MIN_COMPLETENESS")
    eval_min_coherence: float = Field(0.70, alias="ASTRA_EVAL_MIN_COHERENCE")
    eval_min_visual_richness: float = Field(0.50, alias="ASTRA_EVAL_MIN_VISUAL_RICHNESS")
    eval_min_relevance: float = Field(0.80, alias="ASTRA_EVAL_MIN_RELEVANCE")
    eval_max_iterations: int = Field(3, alias="ASTRA_EVAL_MAX_ITERATIONS")

    # ── Charts ────────────────────────────────────────────────────────────────
    astra_chart_output_dir: str = Field(
        "./output/charts", alias="ASTRA_CHART_OUTPUT_DIR"
    )
    astra_matplotlib_style: str = Field(
        "seaborn-v0_8-darkgrid", alias="ASTRA_MATPLOTLIB_STYLE"
    )
    astra_chart_dpi: int = Field(150, alias="ASTRA_CHART_DPI")

    # ── UI ────────────────────────────────────────────────────────────────────
    astra_gradio_host: str = Field("0.0.0.0", alias="ASTRA_GRADIO_HOST")
    astra_gradio_port: int = Field(7860, alias="ASTRA_GRADIO_PORT")
    astra_gradio_share: bool = Field(False, alias="ASTRA_GRADIO_SHARE")
    astra_gradio_theme: str = Field("soft", alias="ASTRA_GRADIO_THEME")

    # ── Sandbox ───────────────────────────────────────────────────────────────
    astra_sandbox_timeout_seconds: int = Field(
        120, alias="ASTRA_SANDBOX_TIMEOUT_SECONDS"
    )
    astra_sandbox_memory_limit: str = Field("4g", alias="ASTRA_SANDBOX_MEMORY_LIMIT")
    astra_sandbox_max_retries: int = Field(3, alias="ASTRA_SANDBOX_MAX_RETRIES")

    # ── Performance ───────────────────────────────────────────────────────────
    astra_max_concurrent_crawlers: int = Field(
        12, alias="ASTRA_MAX_CONCURRENT_CRAWLERS"
    )
    astra_num_cpu_workers: int = Field(16, alias="ASTRA_NUM_CPU_WORKERS")
    astra_max_sections: int = Field(8, alias="ASTRA_MAX_SECTIONS")
    astra_max_sub_queries: int = Field(8, alias="ASTRA_MAX_SUB_QUERIES")
    astra_min_refine_improvement: float = Field(0.05, alias="ASTRA_MIN_REFINE_IMPROVEMENT")

    # ── Logging ───────────────────────────────────────────────────────────────
    astra_log_level: str = Field("INFO", alias="ASTRA_LOG_LEVEL")
    astra_log_file: str = Field(
        "./data/logs/astra.log", alias="ASTRA_LOG_FILE"
    )
    astra_data_dir: str = Field("./data", alias="ASTRA_DATA_DIR")
    astra_temp_dir: str = Field("./data/temp", alias="ASTRA_TEMP_DIR")
    astra_deepeval_results_folder: str = Field(
        "./data/evaluation", alias="DEEPEVAL_RESULTS_FOLDER"
    )

    # ── Per-query session folders ──────────────────────────────────────────────
    astra_sessions_dir: str = Field(
        "./data/sessions", alias="ASTRA_SESSIONS_DIR"
    )

    # ── Tavily search ─────────────────────────────────────────────────────────
    tavily_api_key: Optional[str] = Field(None, alias="TAVILY_API_KEY")

    # ── Visual Intelligence ────────────────────────────────────────────────────
    astra_pdf_download_enabled: bool = Field(True, alias="ASTRA_PDF_DOWNLOAD_ENABLED")
    astra_vision_max_figures: int = Field(30, alias="ASTRA_VISION_MAX_FIGURES")
    astra_vision_min_image_size: int = Field(150, alias="ASTRA_VISION_MIN_IMAGE_SIZE")
    astra_html_table_vision: bool = Field(True, alias="ASTRA_HTML_TABLE_VISION")

    # ── Subscription scrapers (optional session cookies) ──────────────────────
    # Medium: copy `sid` cookie value from browser DevTools after logging in
    astra_medium_session_cookie: Optional[str] = Field(None, alias="ASTRA_MEDIUM_SESSION_COOKIE")
    # Substack: copy `substack.sid` cookie value from browser DevTools
    astra_substack_session_cookie: Optional[str] = Field(None, alias="ASTRA_SUBSTACK_SESSION_COOKIE")

    def create_session_dir(self, query: str) -> Path:
        """Create a per-query session directory: <sessions_dir>/<slug>_<timestamp>/"""
        import re
        import time

        stopwords = {
            "a", "an", "the", "in", "of", "for", "on", "with", "and", "or",
            "to", "that", "is", "are", "was", "how", "why", "what", "vs",
        }
        words = re.sub(r"[^\w\s]", "", query.lower()).split()
        meaningful = [w for w in words if w not in stopwords and len(w) > 2][:4]
        slug = "_".join(meaningful) if meaningful else "astra_research"
        ts = int(time.time())
        dir_name = f"{slug}_{ts}"

        session_dir = Path(self.astra_sessions_dir) / dir_name
        for sub in ("reports", "charts", "logs", "sources"):
            (session_dir / sub).mkdir(parents=True, exist_ok=True)

        return session_dir

    def get_output_path(self, filename: str) -> Path:
        p = Path(self.astra_output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def get_temp_path(self, filename: str) -> Path:
        p = Path(self.astra_temp_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def get_chart_path(self, filename: str) -> Path:
        p = Path(self.astra_chart_output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / filename


@lru_cache(maxsize=1)
def get_config() -> AstraConfig:
    """Cached singleton config loader."""
    return AstraConfig()
