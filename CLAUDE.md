# CLAUDE.md

ASTRA: multi-agent LangGraph pipeline → research report (HTML/PDF/MD) via web crawl, RAG, LLM drafting, iterative eval.

- **Env**: `conda activate astra_os_312` (Python 3.12)
- **Sessions**: `/home/access/peter/trd/ASTRA_OS_data/sessions/<slug>_<timestamp>/`

## Run
```bash
python main.py                    # Gradio UI
python main.py --query "topic"    # CLI
python main.py --test-layer N     # N=1–7, test layer in isolation
python -m pytest tests/           # tests
```
vLLM required at `localhost:8000` (main) and `localhost:8001` (vision) for layers 1, 5, 6.

## Pipeline
`query_intelligence → parallel_crawl → process_documents → index_knowledge → draft_report → evaluate_quality → [refine ≤3×] → assemble_report`

Only branch: after `evaluate_quality` — failing sections → `node_refine`, else → `node_assemble_report`.

## Key Files
| Path | Purpose |
|------|---------|
| `astra/config.py` | `AstraConfig` pydantic settings; `get_config()` singleton; `create_session_dir()` |
| `astra/state.py` | `AstraState` TypedDict; `new_session()` factory |
| `astra/graph.py` | All 8 node functions + `build_graph()` + `run_research()` |
| `astra/agents/orchestrator.py` | `get_main_llm`, `get_judge_llm`, `get_all_tools()` |
| `astra/tools/layer1_query.py` | `enrich_query`, `query_expand` |
| `astra/tools/layer2_crawlers.py` | 14 crawlers: tavily, ddg, jina, arxiv, s2, openalex, pubmed, github, wikipedia… |
| `astra/tools/layer3_docs.py` | `docling_parse_pdf`, `pymupdf_extract` |
| `astra/tools/layer3_vision.py` | Figure/table extraction, VLM description, `process_visual_sources()` |
| `astra/tools/layer4_rag.py` | `chunk_text`, `embed_chunks`, `build_index`, `hybrid_retrieve`, `bge_rerank` |
| `astra/tools/layer5_judge.py` | `evaluate_section`, `flag_gaps` |
| `astra/tools/layer6_report.py` | `write_section`, chart generation, `build_html`, `build_pdf` |
| `astra/tools/layer7_refinement.py` | `gap_analysis`, `trigger_reresearch`, `check_convergence` |
| `app.py` / `main.py` | Gradio UI / CLI |

## Models
- **Main + Judge**: `Qwen3-32B-AWQ` @ `localhost:8000` alias `astra-main` (judge: temp=0)
- **Vision**: `Qwen2.5-VL-7B` @ `localhost:8001` alias `astra-vision`
- **Embeddings**: `BAAI/bge-m3` (CPU, 1024-dim) | **Reranker**: `BAAI/bge-reranker-v2-m3` (CPU)
- **Stores**: FAISS (in-memory) + BM25 per session

## Conventions
- Always `get_config()` — never instantiate `AstraConfig` directly.
- Session path in `state["session_output_dir"]`; propagated to all output tools.
- Layer tools = pure functions. Nodes in `graph.py` orchestrate + handle fallbacks.
- Chart auto-generation fires when section markdown has a pipe-table with numeric data.
- Tavily = primary web search; DDG = sequential fallback (parallel causes rate-limit hangs).
- HTML is primary output (MathJax, base64 images, active links); PDF via weasyprint.

## Known Issues
- `duckduckgo-search` renamed to `ddgs` — `requirements.txt`: `ddgs>=0.7.0`; fallback import in layer2.
- `langgraph.checkpoint.sqlite` not bundled — `pip install langgraph-checkpoint-sqlite` for SQLite.
- Docling `do_picture_classifier` field renamed — handled with `hasattr` check.
- Vision port 8001 down → `describe_figure()` silently returns confidence=0.0 (caption-only).
- Off-topic figures filtered by cosine similarity gate `ASTRA_FIGURE_RELEVANCE_THRESHOLD=0.28`.
