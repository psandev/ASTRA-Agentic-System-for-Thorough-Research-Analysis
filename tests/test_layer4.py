"""
Layer 4 Test — Agentic RAG Engine
Tests: embed_chunks, faiss_search, bm25_search, hybrid_retrieve, bge_rerank

Run: python main.py --test-layer 4
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

SAMPLE_CHUNKS = [
    {
        "id": "chunk_001",
        "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                "It extends LangChain with cyclic graph support and enables complex agent workflows.",
        "metadata": {"source_url": "https://langchain.com/langgraph"},
    },
    {
        "id": "chunk_002",
        "text": "Retrieval-Augmented Generation (RAG) combines dense retrieval with generative models. "
                "FAISS enables fast approximate nearest neighbor search for semantic retrieval.",
        "metadata": {"source_url": "https://arxiv.org/abs/2005.11401"},
    },
    {
        "id": "chunk_003",
        "text": "BM25 is a bag-of-words retrieval function that ranks documents based on query term frequency. "
                "It is particularly effective for keyword-based sparse retrieval.",
        "metadata": {"source_url": "https://en.wikipedia.org/wiki/Okapi_BM25"},
    },
    {
        "id": "chunk_004",
        "text": "Qwen3-32B is a state-of-the-art language model with 32 billion parameters. "
                "It achieves excellent performance on reasoning and instruction following benchmarks.",
        "metadata": {"source_url": "https://huggingface.co/Qwen/Qwen3-32B-AWQ"},
    },
    {
        "id": "chunk_005",
        "text": "Multi-agent systems use multiple AI agents collaborating to solve complex tasks. "
                "Each agent specializes in a specific capability and communicates via messages.",
        "metadata": {"source_url": "https://arxiv.org/abs/2308.00352"},
    },
]


def test_embed_chunks():
    console.print("\n[bold]Test: embed_chunks (BAAI/bge-m3)[/bold]")
    from astra.tools.layer4_rag import embed_chunks

    texts = [c["text"] for c in SAMPLE_CHUNKS]
    result = embed_chunks.invoke({"texts": texts, "batch_size": 32})

    assert "embeddings" in result, "Missing embeddings"
    assert "dim" in result, "Missing dim"
    assert len(result["embeddings"]) == len(texts), "Wrong number of embeddings"
    assert result["dim"] > 0, "Invalid dimension"

    console.print(f"  ✅ Embedded {len(texts)} texts → dim={result['dim']}")
    return True


def test_build_index():
    console.print("\n[bold]Test: Build FAISS + BM25 index[/bold]")
    from astra.tools.layer4_rag import build_index, _INDEX_STORE

    build_index(SAMPLE_CHUNKS, "test_collection")

    store = _INDEX_STORE.get("test_collection", {})
    assert store.get("faiss_index") is not None, "FAISS index not built"
    assert store.get("bm25_index") is not None, "BM25 index not built"
    assert len(store.get("chunks", [])) == len(SAMPLE_CHUNKS), "Wrong chunk count"

    console.print(f"  ✅ Built index: {len(SAMPLE_CHUNKS)} chunks")
    return True


def test_faiss_search():
    console.print("\n[bold]Test: FAISS dense search[/bold]")
    from astra.tools.layer4_rag import faiss_search, build_index

    # Ensure index exists
    build_index(SAMPLE_CHUNKS, "test_collection")

    results = faiss_search.invoke(
        {"query": "multi-agent LLM workflow", "top_k": 3, "collection": "test_collection"}
    )
    assert isinstance(results, list), "Expected list"
    assert len(results) > 0, "Expected results"
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r

    console.print(f"  ✅ FAISS returned {len(results)} results")
    for r in results[:2]:
        console.print(f"    • [{r['score']:.3f}] {r['text'][:60]}")
    return True


def test_bm25_search():
    console.print("\n[bold]Test: BM25 sparse search[/bold]")
    from astra.tools.layer4_rag import bm25_search, build_index

    build_index(SAMPLE_CHUNKS, "test_collection")

    results = bm25_search.invoke(
        {"query": "retrieval augmented generation FAISS", "top_k": 3, "collection": "test_collection"}
    )
    assert isinstance(results, list), "Expected list"

    console.print(f"  ✅ BM25 returned {len(results)} results")
    for r in results[:2]:
        console.print(f"    • [{r['score']:.3f}] {r['text'][:60]}")
    return True


def test_hybrid_retrieve():
    console.print("\n[bold]Test: Hybrid retrieval (FAISS + BM25 → RRF)[/bold]")
    from astra.tools.layer4_rag import hybrid_retrieve, build_index

    build_index(SAMPLE_CHUNKS, "test_collection")

    results = hybrid_retrieve.invoke(
        {
            "query": "language model retrieval search",
            "top_k": 3,
            "collection": "test_collection",
            "dense_weight": 0.6,
        }
    )
    assert isinstance(results, list), "Expected list"
    assert len(results) > 0, "Expected results"
    for r in results:
        assert "rrf_score" in r, "Missing rrf_score"

    console.print(f"  ✅ Hybrid returned {len(results)} results")
    for r in results[:2]:
        console.print(f"    • [rrf={r['rrf_score']:.4f}] {r['text'][:60]}")
    return True


def test_bge_rerank():
    console.print("\n[bold]Test: bge-reranker cross-encoder[/bold]")
    from astra.tools.layer4_rag import bge_rerank, hybrid_retrieve, build_index

    build_index(SAMPLE_CHUNKS, "test_collection")

    candidates = hybrid_retrieve.invoke(
        {"query": "LLM agent workflow", "top_k": 5, "collection": "test_collection"}
    )
    if not candidates:
        console.print("  ⚠️  No candidates for reranking — skipping")
        return True

    reranked = bge_rerank.invoke(
        {"query": "LLM agent workflow", "candidates": candidates, "top_k": 3}
    )
    assert isinstance(reranked, list), "Expected list"
    for r in reranked:
        assert "rerank_score" in r, "Missing rerank_score"

    console.print(f"  ✅ Reranked {len(candidates)} → {len(reranked)} results")
    for r in reranked[:2]:
        console.print(f"    • [score={r['rerank_score']:.4f}] {r['text'][:60]}")
    return True


def run_tests():
    console.print("\n[bold cyan]═══ Layer 4: Agentic RAG Engine ═══[/bold cyan]\n")
    console.print("[yellow]Note: First run loads BAAI/bge-m3 and bge-reranker models (~1GB)...[/yellow]\n")

    tests = [
        ("embed_chunks (bge-m3)", test_embed_chunks),
        ("Build FAISS + BM25 index", test_build_index),
        ("FAISS dense search", test_faiss_search),
        ("BM25 sparse search", test_bm25_search),
        ("Hybrid retrieval (RRF)", test_hybrid_retrieve),
        ("bge-reranker rerank", test_bge_rerank),
    ]

    table = Table(title="Layer 4 Test Results")
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
    console.print(f"\n[bold]Layer 4: {passed}/{passed+failed} passed[/bold]")
    return failed == 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_tests()
