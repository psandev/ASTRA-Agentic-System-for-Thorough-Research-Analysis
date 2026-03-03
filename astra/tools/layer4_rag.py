"""
ASTRA Layer 4 — Agentic RAG Engine
Action paradigm: CODE-ACTION (FAISS, BM25, LightRAG, reranking)

Tools:
  embed_chunks, faiss_search, bm25_search, hybrid_retrieve,
  bge_rerank, lightrag_query

Embedding:  BAAI/bge-m3 (CPU, 1024-dim)
Reranker:   BAAI/bge-reranker-v2-m3 (CPU)
Dense:      FAISS IndexFlatIP (in-memory)
Sparse:     rank-bm25
Graph:      LightRAG (optional)
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
from langchain_core.tools import tool
from loguru import logger

from astra.config import get_config

# ─── Global in-memory index store ────────────────────────────────────────────
# Keyed by collection name → {faiss_index, bm25_index, chunks}

_INDEX_STORE: dict[str, dict] = {}


def _get_or_create_store(collection: str) -> dict:
    if collection not in _INDEX_STORE:
        _INDEX_STORE[collection] = {
            "faiss_index": None,
            "bm25_index": None,
            "chunks": [],
            "embeddings": None,
        }
    return _INDEX_STORE[collection]


# ─── Lazy-loaded models ───────────────────────────────────────────────────────

import threading

_embedding_model = None
_reranker_model = None
_model_load_lock = threading.Lock()  # serialize lazy model loading (prevents concurrent from_pretrained)
_reranker_lock = threading.Lock()    # serialize reranker inference (not thread-safe for .predict())


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _model_load_lock:
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            cfg = get_config()
            logger.info(f"[Layer 4] Loading embedding model: {cfg.astra_embedding_model}")
            _embedding_model = SentenceTransformer(
                cfg.astra_embedding_model,
                device=cfg.astra_embedding_device,
            )
    return _embedding_model


def _get_reranker_model():
    """
    Load the reranker once and cache it.

    Uses CrossEncoder (sentence-transformers) as the primary backend because it:
    - Does NOT call model.to(cuda) on every inference call (avoids meta-tensor errors)
    - Loads directly on the specified device without accelerate/device_map
    - Is safe for CPU-only operation even when CUDA is visible

    FlagReranker is skipped as primary because compute_score_single_gpu() unconditionally
    does self.model.to(target_devices[0]) which resolves to cuda:0 via CUDA_VISIBLE_DEVICES,
    causing "Cannot copy out of meta tensor" when the model was loaded on CPU via accelerate.
    """
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    with _model_load_lock:
        if _reranker_model is None:
            cfg = get_config()
            device = cfg.astra_reranker_device  # "cpu" by default
            logger.info(
                f"[Layer 4] Loading reranker: {cfg.astra_reranker_model} (device={device})"
            )
            try:
                from sentence_transformers import CrossEncoder
                _reranker_model = CrossEncoder(cfg.astra_reranker_model, device=device)
                logger.info(f"[Layer 4] Reranker loaded via CrossEncoder on {device}")
            except Exception as e:
                logger.warning(f"[Layer 4] CrossEncoder failed ({e}), reranking disabled")
                _reranker_model = None
            # Never fall back to FlagReranker: its compute_score_single_gpu()
            # unconditionally calls self.model.to(target_devices[0]) which resolves
            # to cuda:0 (via CUDA_VISIBLE_DEVICES) and raises NotImplementedError
            # when the model was loaded with meta tensors via accelerate.
    return _reranker_model


def preload_models() -> None:
    """
    Eagerly load embedding and reranker models in the calling thread.
    Call this BEFORE spawning any ThreadPoolExecutor to prevent concurrent from_pretrained races.
    """
    _get_embedding_model()
    _get_reranker_model()


def build_index_incremental(new_chunks: list[dict], collection: str) -> int:
    """
    Add new chunks to an existing FAISS+BM25 index without re-embedding existing chunks.

    Unlike build_index(), this only embeds `new_chunks`, then appends the vectors
    to the live FAISS index and rebuilds BM25 from all chunks (BM25 is CPU-only,
    rebuilds in milliseconds). Use this in the refinement loop to avoid a full
    re-embedding pass over the entire knowledge base on every iteration.

    Returns the number of chunks added.
    """
    from rank_bm25 import BM25Okapi

    store = _get_or_create_store(collection)

    if not new_chunks:
        return 0

    if store.get("faiss_index") is None:
        # No existing index — build from scratch
        build_index(new_chunks, collection)
        return len(new_chunks)

    # Embed only the new chunks
    texts = [c["text"] for c in new_chunks]
    emb_result = embed_chunks.invoke({"texts": texts, "batch_size": 128})
    new_vecs = np.array(emb_result["embeddings"], dtype="float32")

    # Append to live FAISS index (IndexFlatIP.add() is incremental)
    store["faiss_index"].add(new_vecs)

    # Extend stored chunk list and embedding matrix
    store["chunks"].extend(new_chunks)
    existing_embs = store.get("embeddings")
    store["embeddings"] = (
        np.vstack([existing_embs, new_vecs]) if existing_embs is not None else new_vecs
    )

    # Rebuild BM25 from all chunks (tokenisation only, no neural inference — fast)
    all_texts = [c["text"] for c in store["chunks"]]
    store["bm25_index"] = BM25Okapi([t.lower().split() for t in all_texts])

    total = len(store["chunks"])
    logger.info(
        f"[Layer 4] Incremental index update: +{len(new_chunks)} chunks "
        f"(total={total} in '{collection}')"
    )
    return len(new_chunks)


# ─── Chunking helper ─────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source_url: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Semantic text chunker: splits text into overlapping token windows.
    Respects paragraph/section boundaries where possible.
    """
    cfg = get_config()
    chunk_size = chunk_size or cfg.astra_chunk_size
    overlap = chunk_overlap or cfg.astra_chunk_overlap

    # Split on paragraph boundaries first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[dict] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Approximate token count (1 token ≈ 4 chars)
        para_tokens = len(para) // 4

        if current_len + para_tokens > chunk_size and current_chunk:
            chunk_text_str = "\n\n".join(current_chunk)
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text_str,
                    "metadata": {
                        "source_url": source_url,
                        "chunk_index": len(chunks),
                    },
                }
            )
            # Overlap: keep last portion
            overlap_tokens = 0
            overlap_chunks: list[str] = []
            for c in reversed(current_chunk):
                if overlap_tokens + len(c) // 4 <= overlap:
                    overlap_chunks.insert(0, c)
                    overlap_tokens += len(c) // 4
                else:
                    break
            current_chunk = overlap_chunks
            current_len = overlap_tokens

        current_chunk.append(para)
        current_len += para_tokens

    if current_chunk:
        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "text": "\n\n".join(current_chunk),
                "metadata": {
                    "source_url": source_url,
                    "chunk_index": len(chunks),
                },
            }
        )

    return chunks


# ─── embed_chunks ─────────────────────────────────────────────────────────────

@tool
def embed_chunks(
    texts: list[str],
    batch_size: int = 128,
    normalize: bool = True,
) -> dict:
    """
    Layer 4: Generate dense embeddings using BAAI/bge-m3 on CPU.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.
        normalize: L2-normalize output vectors.

    Returns:
        {embeddings: list[list[float]], dim: int}
    """
    logger.info(f"[Layer 4] Embedding {len(texts)} chunks (batch={batch_size})")
    model = _get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(texts) > 50,
    )
    return {
        "embeddings": embeddings.tolist(),
        "dim": embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings[0]),
    }


# ─── Index builder ────────────────────────────────────────────────────────────

def build_index(chunks: list[dict], collection: str) -> None:
    """
    Build FAISS + BM25 indices from a list of chunk dicts.
    Called from the index_knowledge graph node.
    """
    import faiss
    from rank_bm25 import BM25Okapi

    store = _get_or_create_store(collection)
    store["chunks"] = chunks

    texts = [c["text"] for c in chunks]
    if not texts:
        logger.warning(f"[Layer 4] No chunks to index in collection '{collection}'")
        return

    # Build dense FAISS index
    emb_result = embed_chunks.invoke({"texts": texts, "batch_size": 128})
    vecs = np.array(emb_result["embeddings"], dtype="float32")
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    store["faiss_index"] = index
    store["embeddings"] = vecs

    # Build sparse BM25 index
    tokenized = [text.lower().split() for text in texts]
    store["bm25_index"] = BM25Okapi(tokenized)

    logger.info(
        f"[Layer 4] Index built: collection='{collection}' "
        f"chunks={len(chunks)} dim={dim}"
    )


# ─── faiss_search ─────────────────────────────────────────────────────────────

@tool
def faiss_search(
    query: str,
    top_k: int = 20,
    collection: str = "default",
    threshold: Optional[float] = None,
) -> list[dict]:
    """
    Layer 4: Dense vector similarity search using FAISS.

    Args:
        query: Query string.
        top_k: Number of results to return.
        collection: Name of the index collection.
        threshold: Minimum similarity score (0-1).

    Returns:
        List of {chunk_id, text, score, metadata}.
    """
    store = _get_or_create_store(collection)
    index = store.get("faiss_index")
    chunks = store.get("chunks", [])

    if index is None or not chunks:
        logger.warning(f"[Layer 4] FAISS: no index for collection '{collection}'")
        return []

    model = _get_embedding_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = index.search(q_vec, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if threshold and float(score) < threshold:
            continue
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk.get("id", str(idx)),
                "text": chunk.get("text", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return results


# ─── bm25_search ──────────────────────────────────────────────────────────────

@tool
def bm25_search(
    query: str,
    top_k: int = 20,
    collection: str = "default",
) -> list[dict]:
    """
    Layer 4: Sparse keyword retrieval using BM25 (rank-bm25).

    Args:
        query: Query string.
        top_k: Number of results.
        collection: Collection name.

    Returns:
        List of {chunk_id, text, score, metadata}.
    """
    store = _get_or_create_store(collection)
    bm25 = store.get("bm25_index")
    chunks = store.get("chunks", [])

    if bm25 is None or not chunks:
        logger.warning(f"[Layer 4] BM25: no index for collection '{collection}'")
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk.get("id", str(idx)),
                "text": chunk.get("text", ""),
                "score": float(scores[idx]),
                "metadata": chunk.get("metadata", {}),
            }
        )

    return results


# ─── hybrid_retrieve ──────────────────────────────────────────────────────────

@tool
def hybrid_retrieve(
    query: str,
    top_k: int = 20,
    collection: str = "default",
    dense_weight: float = 0.6,
) -> list[dict]:
    """
    Layer 4: Hybrid retrieval combining FAISS dense + BM25 sparse via RRF.

    Uses Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings.

    Args:
        query: Query string.
        top_k: Final number of results.
        collection: Collection name.
        dense_weight: Weight for dense results in fusion (0-1).

    Returns:
        List of {chunk_id, text, rrf_score, metadata}.
    """
    cfg = get_config()
    dense_weight = dense_weight or cfg.astra_retrieval_dense_weight
    sparse_weight = 1.0 - dense_weight

    logger.debug(f"[Layer 4] Hybrid retrieve: '{query[:60]}' collection={collection}")

    # Fetch more candidates for fusion
    candidate_k = min(top_k * 3, 60)

    dense_results = faiss_search.invoke(
        {"query": query, "top_k": candidate_k, "collection": collection}
    )
    sparse_results = bm25_search.invoke(
        {"query": query, "top_k": candidate_k, "collection": collection}
    )

    # RRF fusion
    rrf_k = 60  # standard RRF constant
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (rrf_k + rank))
        chunk_map[cid] = result

    for rank, result in enumerate(sparse_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * (1.0 / (rrf_k + rank))
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "chunk_id": cid,
            "text": chunk_map[cid]["text"],
            "rrf_score": float(rrf_score),
            "metadata": chunk_map[cid]["metadata"],
        }
        for cid, rrf_score in ranked
        if cid in chunk_map
    ]


# ─── bge_rerank ───────────────────────────────────────────────────────────────

@tool
def bge_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Layer 4: Cross-encoder reranking using BAAI/bge-reranker-v2-m3 on CPU.
    Always call after hybrid_retrieve for final precision boost.

    Args:
        query: Query string.
        candidates: List of candidate chunks from hybrid_retrieve.
        top_k: Number of top results after reranking.

    Returns:
        List of {chunk_id, text, rerank_score, metadata}.
    """
    if not candidates:
        return []

    logger.debug(f"[Layer 4] Reranking {len(candidates)} candidates → top {top_k}")
    reranker = _get_reranker_model()

    if reranker is None:
        # Reranker unavailable — return top-k candidates ordered by retrieval score
        logger.warning("[Layer 4] Reranker unavailable, returning candidates by retrieval score")
        return [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "rerank_score": c.get("rrf_score", c.get("score", 0.0)),
                "metadata": c.get("metadata", {}),
            }
            for c in candidates[:top_k]
        ]

    pairs = [(query, c["text"]) for c in candidates]

    # Serialize reranker calls — CrossEncoder.predict is not thread-safe on
    # the same model singleton when called from parallel workers.
    with _reranker_lock:
        scores = reranker.predict(pairs)  # CrossEncoder API

    if not isinstance(scores, list):
        scores = scores.tolist()

    # Sort by reranking score
    scored = sorted(
        zip(candidates, scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": c.get("chunk_id", ""),
            "text": c.get("text", ""),
            "rerank_score": float(score),
            "metadata": c.get("metadata", {}),
        }
        for c, score in scored
    ]


# ─── figure_search ────────────────────────────────────────────────────────────

def figure_search(
    query: str,
    top_k: int = 5,
    collection: str = "default",
    min_score: float = 0.0,
) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant figure chunks for a query.

    Unlike hybrid_retrieve, this function ONLY returns chunks whose metadata
    has chunk_type == 'figure', so scraped figures compete only against each
    other — not against the vastly more numerous text chunks.

    Args:
        query:      Section query string.
        top_k:      Maximum figures to return.
        collection: FAISS/BM25 index collection name.
        min_score:  Minimum cosine similarity threshold (0–1).  Figures below
                    this score are off-topic and should be excluded.

    Returns:
        List of {chunk_id, text, score, metadata} dicts.
    """
    store = _get_or_create_store(collection)
    index = store.get("faiss_index")
    chunks = store.get("chunks", [])

    if index is None or not chunks:
        return []

    # Build a map: original FAISS index → chunk, for figure chunks only
    figure_idx_map: dict[int, dict] = {
        i: c
        for i, c in enumerate(chunks)
        if c.get("metadata", {}).get("chunk_type") == "figure"
    }
    if not figure_idx_map:
        return []

    model = _get_embedding_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # Search the full index broadly, then post-filter to figure chunks
    n_candidates = min(len(chunks), max(top_k * 20, 100))
    scores, indices = index.search(q_vec, n_candidates)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx not in figure_idx_map:
            continue
        if float(score) < min_score:
            continue
        chunk = figure_idx_map[idx]
        results.append(
            {
                "chunk_id": chunk.get("id", str(idx)),
                "text": chunk.get("text", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
        if len(results) >= top_k:
            break

    logger.debug(
        f"[Layer 4] figure_search: '{query[:50]}' → {len(results)} figures "
        f"(threshold={min_score}, total_figure_chunks={len(figure_idx_map)})"
    )
    return results


# ─── lightrag_query ───────────────────────────────────────────────────────────

@tool
def lightrag_query(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
) -> dict:
    """
    Layer 4: Knowledge graph + vector hybrid query using LightRAG.
    Best for multi-hop relational reasoning across documents.

    Args:
        query: Complex query requiring entity relationships.
        mode: "hybrid" | "global" | "local" | "naive".
        top_k: Number of results.

    Returns:
        {answer, entities, relations, source_chunks}
    """
    cfg = get_config()
    if not cfg.astra_lightrag_enabled:
        logger.info("[Layer 4] LightRAG disabled, using hybrid_retrieve fallback")
        fallback = hybrid_retrieve.invoke({"query": query, "top_k": top_k})
        return {
            "answer": "\n\n".join(c["text"] for c in fallback[:3]),
            "entities": [],
            "relations": [],
            "source_chunks": fallback,
        }

    logger.info(f"[Layer 4] LightRAG query mode={mode}: '{query[:60]}'")

    try:
        import asyncio
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from functools import partial

        cfg_obj = get_config()
        working_dir = cfg_obj.astra_lightrag_working_dir

        from pathlib import Path
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=cfg_obj.vllm_base_url,
                api_key=cfg_obj.vllm_api_key,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = await client.chat.completions.create(
                model=cfg_obj.astra_main_model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""

        async def embed_func(texts):
            model = _get_embedding_model()
            return model.encode(texts, normalize_embeddings=True).tolist()

        async def run_query():
            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                embedding_func=openai_embed if False else embed_func,
            )
            result = await rag.aquery(query, param=QueryParam(mode=mode))
            return result

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_query())
        loop.close()

        return {
            "answer": str(result),
            "entities": [],
            "relations": [],
            "source_chunks": [],
        }

    except ImportError:
        logger.warning("[Layer 4] LightRAG not installed, using hybrid_retrieve")
        fallback = hybrid_retrieve.invoke({"query": query, "top_k": top_k})
        return {
            "answer": "\n\n".join(c["text"] for c in fallback[:3]),
            "entities": [],
            "relations": [],
            "source_chunks": fallback,
        }
    except Exception as e:
        logger.error(f"[Layer 4] LightRAG failed: {e}")
        fallback = hybrid_retrieve.invoke({"query": query, "top_k": top_k})
        return {
            "answer": "\n\n".join(c["text"] for c in fallback[:3]),
            "entities": [],
            "relations": [],
            "source_chunks": fallback,
        }
