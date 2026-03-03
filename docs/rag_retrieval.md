# ASTRA RAG — Embedding, Indexing and Retrieval (March 2026)

Full pipeline from raw text to final ranked chunks passed to the LLM writer.

---

## The Three-Stage Funnel

```
All collected text (1000+ chunks)
        │
        ▼
Stage 1 — RECALL
  FAISS dense search  ──┐
                        ├──→ RRF merge  →  ~60 candidates
  BM25 sparse search  ──┘

        ▼
Stage 2 — COMBINE
  Reciprocal Rank Fusion
  weighted 60% dense / 40% sparse
  → ~20 merged candidates

        ▼
Stage 3 — PRECISION
  bge-reranker-v2-m3 (CrossEncoder)
  reads query + chunk as a pair
  → top 10 chunks  →  write_section() [LLM]
```

Each stage trades speed for precision. FAISS and BM25 are fast but imprecise.
The reranker is slow (cross-encoder, not dot product) but precise.

---

## Step 0 — Chunking (`chunk_text`)

Before anything is indexed, raw markdown text is split into overlapping windows.

**Algorithm:**
1. Split on double newlines (`\n\n`) — paragraph boundaries
2. Accumulate paragraphs until the window reaches `chunk_size` tokens (approx: chars ÷ 4)
3. When the window is full, emit the chunk, then carry forward the last `chunk_overlap`
   tokens as the start of the next chunk

**Parameters (from `.env`):**
```
ASTRA_CHUNK_SIZE=512      # tokens per chunk
ASTRA_CHUNK_OVERLAP=64    # tokens carried forward to next chunk
```

**Why overlap?** A sentence that straddles a chunk boundary would be missing context
in both neighbours without it. Overlap ensures every sentence appears in full in at
least one chunk.

**Output per chunk:**
```python
{
  "id": "uuid",
  "text": "...",
  "metadata": {
    "source_url": "https://...",
    "chunk_index": 3,
  }
}
```

Figure chunks from the visual pipeline follow the same structure but add
`chunk_type: "figure"` and `image_path` to metadata.

---

## Step 1 — Embedding (`embed_chunks`, `build_index`)

### Model
**BAAI/bge-m3** running on CPU, embedding dimension 1024.
bge-m3 is a multilingual model trained for retrieval — it produces normalized vectors
where cosine similarity equals dot product (so `IndexFlatIP` in FAISS is correct).

### What happens
```python
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
embeddings = model.encode(
    texts,
    batch_size=128,
    normalize_embeddings=True,   # L2-normalize → dot product = cosine similarity
)
# shape: (n_chunks, 1024)  dtype: float32
```

All chunk texts are encoded in batches of 128. For a typical session (~1000 chunks)
this takes 60–90 seconds on CPU. The model is loaded once and cached globally for
the session lifetime.

### FAISS index construction
```python
import faiss
index = faiss.IndexFlatIP(dim)   # Inner Product = cosine on L2-normalized vectors
index.add(vecs)                   # all vectors added at once
```

`IndexFlatIP` is an exact (brute-force) search — no approximation, no clustering.
For ~1000 chunks this is fast enough (~5ms per query). At 100k+ chunks an
approximate index (IVF, HNSW) would be needed.

### BM25 index construction
```python
from rank_bm25 import BM25Okapi
tokenized = [text.lower().split() for text in texts]
bm25 = BM25Okapi(tokenized)
```

BM25 requires no neural inference — just word frequency counts. Rebuilds in
milliseconds even for 1000+ chunks.

### Incremental update (`build_index_incremental`)
Used during the refinement loop (Layer 7) when new documents are fetched for gap-filling:

```
New chunks arrive (e.g. 50 from re-research)
  ↓
embed only the new chunks  (not the existing 1000+)
  ↓
faiss_index.add(new_vecs)         ← FAISS IndexFlatIP supports incremental .add()
  ↓
BM25Okapi([all chunks])           ← BM25 rebuilt from scratch (all chunks, fast)
```

This avoids re-embedding the full KB on every refinement iteration (~80s saved per call).

---

## Step 2 — Dense Retrieval (`faiss_search`)

At query time (once per section, at write time):

```python
q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
scores, indices = index.search(q_vec, top_k)
```

FAISS computes the dot product between the query vector and every stored vector,
returns the top-k by score. Score range is 0–1 (cosine similarity on normalized vectors).

**What it finds well:** paraphrases, synonyms, conceptually related content.
**What it misses:** exact model names, version numbers, acronyms, numeric values —
terms that have neighbours in embedding space that dilute the signal.

---

## Step 3 — Sparse Retrieval (`bm25_search`)

```python
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)
```

BM25-Okapi score for a document D and query Q:

```
score(D, Q) = Σ IDF(qᵢ) · f(qᵢ, D) · (k₁ + 1)
                          ─────────────────────────
                          f(qᵢ, D) + k₁·(1 - b + b·|D|/avgdl)

where:
  IDF(qᵢ)    = log((N - df + 0.5) / (df + 0.5))   — rare terms score higher
  f(qᵢ, D)   = term frequency in document D
  |D|/avgdl  = document length normalisation
  k₁ = 1.5, b = 0.75  (BM25Okapi defaults)
```

Chunks with zero keyword overlap score exactly 0 and are excluded.

**What it finds well:** exact model names, metric abbreviations, author names, numbers.
**What it misses:** paraphrases — a chunk using "attention complexity" scores zero
for a query containing "computational cost".

---

## Step 4 — RRF Merge (`hybrid_retrieve`)

Neither FAISS nor BM25 scores are on the same scale, so raw score addition would
be meaningless. Instead, only **rank positions** are used — Reciprocal Rank Fusion:

```
RRF score = dense_weight  · 1/(60 + rank_in_dense)
          + sparse_weight · 1/(60 + rank_in_sparse)
```

**Parameters:**
```
dense_weight  = 0.6   (ASTRA_RETRIEVAL_DENSE_WEIGHT)
sparse_weight = 0.4
rrf_k         = 60    (standard constant — dampens the influence of top-1 vs top-2)
```

**Example with three chunks:**

| Chunk | FAISS rank | BM25 rank | RRF score |
|-------|-----------|-----------|-----------|
| A | 1 | — | 0.6 × 1/61 = 0.00984 |
| B | 2 | 1 | 0.6 × 1/62 + 0.4 × 1/61 = 0.00968 + 0.00656 = **0.01624** |
| C | — | 2 | 0.4 × 1/62 = 0.00645 |

Chunk B wins despite not ranking #1 in either system, because it appears in both.
Chunk A ranked #1 in FAISS but missing from BM25 is overtaken by B.

This is the core property of RRF: **chunks corroborated by both retrieval methods
are promoted above chunks that only one method found.**

The constant 60 in the denominator is the "smoothing constant" — it prevents the
#1 ranked item from having a disproportionately large advantage over #2.

---

## Step 5 — Cross-encoder Reranking (`bge_rerank`)

### Why reranking is needed
Both FAISS and BM25 encode the query and document **independently**. The query vector
is computed once; each document vector was computed at index time. They never see each
other — similarity is just a dot product between two standalone vectors.

A cross-encoder reads the query and each candidate **as a single concatenated input**:

```
[CLS] query text [SEP] candidate chunk text [SEP]
```

This lets the model perform full attention between query tokens and document tokens —
it can notice "the query asks for accuracy on MMLU, and this chunk reports it for a
different benchmark" in a way a bi-encoder cannot.

### Model
**BAAI/bge-reranker-v2-m3** loaded as a `CrossEncoder` (sentence-transformers).
Runs on CPU. Outputs a single relevance score per (query, chunk) pair.

```python
pairs = [(query, chunk["text"]) for chunk in candidates]
scores = reranker.predict(pairs)   # shape: (n_candidates,)
```

**Thread safety:** `reranker.predict()` is not thread-safe on the shared singleton.
A `_reranker_lock` serializes all reranker calls from parallel section workers.

### Cost
Cross-encoder inference is O(n) in the number of candidates (each pair is a separate
forward pass). With ~20 candidates this takes ~300–800ms on CPU — acceptable because
it runs once per section, not per chunk.

### What it corrects
- False positives from FAISS: semantically similar chunks that don't actually answer
  the query (e.g. "attention mechanism" retrieved for a query about "attention deficit")
- False positives from BM25: keyword matches with wrong context
  (e.g. "GPT-4" appears in a chunk comparing it unfavourably — BM25 scores it high,
  cross-encoder scores it low because the content contradicts the query intent)

---

## Step 6 — Figure Search (`figure_search`)

Runs in parallel with the text retrieval pipeline, FAISS-only (no BM25, no reranking):

```python
# Only searches chunks where metadata["chunk_type"] == "figure"
results = [r for r in faiss_search(query, top_k) if r["metadata"].get("chunk_type") == "figure"]
results = [r for r in results if r["score"] >= min_score]  # threshold: 0.28
```

No BM25 for figures because figure chunk text is a natural-language description
written by the VLM — exact keyword matching is less reliable than semantic similarity
on generated prose. No reranking because the figure pool is small (typically 20–60
per session) and direct cosine similarity is sufficient.

---

## Full Flow for One Section

```
section_query = "transformer attention computational complexity"
collection    = "session_abc123"

1. hybrid_retrieve(query, top_k=20, collection)
   ├── faiss_search(query, top_k=60)   → 60 dense candidates
   └── bm25_search(query, top_k=60)    → 60 sparse candidates
       ↓
   RRF merge (k=60, dense_weight=0.6)  → 20 fused candidates

2. bge_rerank(query, candidates=20, top_k=10)
   → 10 cross-encoder ranked chunks

3. figure_search(query, top_k=4, min_score=0.28)
   → 0–4 relevant figure chunks (with image_path in metadata)

4. write_section(
       section_title="Attention Complexity",
       retrieved_chunks=10 text chunks,
       figures=figure chunks
   )
   → markdown section with inline citations + figure embeds
```

---

## Key Parameters (`.env`)

| Variable | Default | Effect |
|----------|---------|--------|
| `ASTRA_CHUNK_SIZE` | 512 | Tokens per chunk |
| `ASTRA_CHUNK_OVERLAP` | 64 | Overlap between consecutive chunks |
| `ASTRA_EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense embedding model |
| `ASTRA_EMBEDDING_DEVICE` | `cpu` | Device for bge-m3 |
| `ASTRA_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `ASTRA_RERANKER_DEVICE` | `cpu` | Device for reranker |
| `ASTRA_RETRIEVAL_DENSE_WEIGHT` | `0.6` | FAISS weight in RRF |
| `ASTRA_RETRIEVAL_RERANK_TOP_K` | `10` | Chunks passed to LLM after reranking |
| `ASTRA_FIGURE_RELEVANCE_THRESHOLD` | `0.28` | Min cosine score for figure inclusion |
