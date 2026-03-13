"""Generate all 9 Jupyter notebooks for the Retrieval module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "02-retrieval"))
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def write_notebook(filename: str, cells: list) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.cells = cells
    path = os.path.join(OUTPUT_DIR, filename)
    nbformat.write(nb, path)
    print(f"  Created {path}")


def md(source: str):
    return nbformat.v4.new_markdown_cell(source)


def code(source: str):
    return nbformat.v4.new_code_cell(source)


# ---------------------------------------------------------------------------
# 01 - Dense Retriever
# ---------------------------------------------------------------------------
def nb_dense_retriever():
    write_notebook("01_dense_retriever.ipynb", [
        md(
            "# Dense Retriever\n"
            "> Retrieve context items by embedding similarity using a vector store.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `DenseRetriever`, `InMemoryVectorStore`, `InMemoryContextStore`, `QueryBundle` |\n"
            "| Difficulty | Beginner |\n"
            "\n"
            "`DenseRetriever` embeds queries and documents into a shared vector space and\n"
            "retrieves the closest matches from an `InMemoryVectorStore`.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import DenseRetriever\n"
            "from anchor.storage import InMemoryVectorStore, InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md(
            "## Create Stores and Embedding Function\n"
            "We use in-memory stores and a simple hash-based embedding for demonstration."
        ),
        code(
            "store = InMemoryVectorStore()\n"
            "ctx_store = InMemoryContextStore()\n"
            "\n"
            "def embed_fn(text: str) -> list[float]:\n"
            "    \"\"\"Deterministic pseudo-embedding for demonstration.\"\"\"\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "vec = embed_fn(\"hello world\")\n"
            "print(f\"Embedding dimension: {len(vec)}\")\n"
            "print(f\"First 5 values: {vec[:5]}\")"
        ),
        md(
            "## Initialize the Dense Retriever\n"
            "Wire up the vector store, context store, and embedding function."
        ),
        code(
            "retriever = DenseRetriever(\n"
            "    vector_store=store,\n"
            "    context_store=ctx_store,\n"
            "    embed_fn=embed_fn,\n"
            ")\n"
            "\n"
            "print(f\"Retriever type: {type(retriever).__name__}\")"
        ),
        md(
            "## Index Documents\n"
            "Create `ContextItem` objects and index them into the retriever."
        ),
        code(
            "topics = [\"python\", \"rust\", \"go\", \"javascript\", \"typescript\"]\n"
            "\n"
            "items = [\n"
            "    ContextItem(\n"
            "        id=f\"doc-{i}\",\n"
            "        content=f\"Document about {topic}: {topic} is a popular programming language.\",\n"
            "        source=SourceType.RETRIEVAL,\n"
            "        score=0.0,\n"
            "        priority=5,\n"
            "        token_count=10,\n"
            "    )\n"
            "    for i, topic in enumerate(topics)\n"
            "]\n"
            "\n"
            "retriever.index(items)\n"
            "\n"
            "print(f\"Indexed {len(items)} documents:\")\n"
            "for item in items:\n"
            "    print(f\"  {item.id}: {item.content[:50]}\")"
        ),
        md(
            "## Retrieve by Query\n"
            "Wrap the query in a `QueryBundle` and call `.retrieve()`."
        ),
        code(
            "query = QueryBundle(query_str=\"programming languages\")\n"
            "results = retriever.retrieve(query, top_k=3)\n"
            "\n"
            "print(f\"Query: '{query.query_str}'\")\n"
            "print(f\"Top {len(results)} results:\\n\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:60]}\")"
        ),
        md(
            "## Vary top_k\n"
            "Adjust how many results are returned."
        ),
        code(
            "for k in [1, 3, 5]:\n"
            "    results = retriever.retrieve(query, top_k=k)\n"
            "    print(f\"top_k={k}: returned {len(results)} results\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `DenseRetriever` uses embedding similarity for semantic search\n"
            "- Provide `vector_store`, `context_store`, and `embed_fn` at construction\n"
            "- `.index(items)` adds `ContextItem` objects to the vector store\n"
            "- `.retrieve(query, top_k=N)` returns the N closest matches\n"
            "- Swap `InMemoryVectorStore` for a persistent backend in production\n"
            "\n"
            "**Next:** [Sparse Retriever](02_sparse_retriever.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Sparse Retriever
# ---------------------------------------------------------------------------
def nb_sparse_retriever():
    write_notebook("02_sparse_retriever.ipynb", [
        md(
            "# Sparse Retriever\n"
            "> BM25-based keyword retrieval for exact term matching.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `SparseRetriever`, `InMemoryContextStore`, `QueryBundle` |\n"
            "| Difficulty | Beginner |\n"
            "\n"
            "`SparseRetriever` uses the BM25 algorithm for term-frequency-based retrieval.\n"
            "It excels at exact keyword matching where dense embeddings may miss specific terms.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import SparseRetriever\n"
            "from anchor.storage import InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md(
            "## Create the Sparse Retriever\n"
            "Only a context store is needed -- no embedding function required."
        ),
        code(
            "ctx_store = InMemoryContextStore()\n"
            "retriever = SparseRetriever(context_store=ctx_store)\n"
            "\n"
            "print(f\"Retriever type: {type(retriever).__name__}\")"
        ),
        md("## Index Documents"),
        code(
            "items = [\n"
            "    ContextItem(id=\"bm25-1\", content=\"Python is great for data science and machine learning.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=12),\n"
            "    ContextItem(id=\"bm25-2\", content=\"Rust provides memory safety without a garbage collector.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=10),\n"
            "    ContextItem(id=\"bm25-3\", content=\"Go was designed at Google for systems programming.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=10),\n"
            "    ContextItem(id=\"bm25-4\", content=\"JavaScript powers the interactive web and Node.js servers.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=11),\n"
            "    ContextItem(id=\"bm25-5\", content=\"Machine learning models need large datasets for training.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=10),\n"
            "]\n"
            "\n"
            "retriever.index(items)\n"
            "print(f\"Indexed {len(items)} documents for BM25\")"
        ),
        md(
            "## Retrieve with Keyword Query\n"
            "BM25 scores documents based on term frequency and inverse document frequency."
        ),
        code(
            "query = QueryBundle(query_str=\"machine learning\")\n"
            "results = retriever.retrieve(query, top_k=3)\n"
            "\n"
            "print(f\"Query: '{query.query_str}'\")\n"
            "print(f\"Results ({len(results)}):\\n\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:60]}\")"
        ),
        md("## Compare Queries\nDifferent keywords surface different documents."),
        code(
            "queries = [\"memory safety\", \"data science\", \"Google systems\", \"web servers\"]\n"
            "\n"
            "for q in queries:\n"
            "    results = retriever.retrieve(QueryBundle(query_str=q), top_k=2)\n"
            "    top = results[0].content[:50] if results else \"(none)\"\n"
            "    print(f\"  '{q}' -> {top}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `SparseRetriever` implements BM25 keyword-based retrieval\n"
            "- No embedding function needed -- operates on raw token frequencies\n"
            "- Best for queries with specific, distinctive terms\n"
            "- Complements `DenseRetriever` -- combine them in a `HybridRetriever`\n"
            "\n"
            "**Next:** [Hybrid Retriever](03_hybrid_retriever.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Hybrid Retriever
# ---------------------------------------------------------------------------
def nb_hybrid_retriever():
    write_notebook("03_hybrid_retriever.ipynb", [
        md(
            "# Hybrid Retriever\n"
            "> Fuse dense and sparse retrieval with Reciprocal Rank Fusion (RRF).\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `HybridRetriever`, `DenseRetriever`, `SparseRetriever` |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "`HybridRetriever` combines a dense retriever (semantic) with a sparse retriever\n"
            "(keyword) and fuses their results using RRF weighted by configurable weights.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import DenseRetriever, SparseRetriever, HybridRetriever\n"
            "from anchor.storage import InMemoryVectorStore, InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md("## Build Dense and Sparse Retrievers"),
        code(
            "# Shared stores and embedding function\n"
            "vector_store = InMemoryVectorStore()\n"
            "ctx_store = InMemoryContextStore()\n"
            "\n"
            "def embed_fn(text: str) -> list[float]:\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "dense = DenseRetriever(\n"
            "    vector_store=vector_store,\n"
            "    context_store=ctx_store,\n"
            "    embed_fn=embed_fn,\n"
            ")\n"
            "\n"
            "sparse = SparseRetriever(context_store=ctx_store)\n"
            "\n"
            "print(\"Dense and sparse retrievers created.\")"
        ),
        md("## Index Shared Documents"),
        code(
            "docs = [\n"
            "    (\"h-1\", \"Python is widely used for data science and machine learning.\"),\n"
            "    (\"h-2\", \"Rust ensures memory safety through its ownership model.\"),\n"
            "    (\"h-3\", \"Go is a statically typed language designed for concurrency.\"),\n"
            "    (\"h-4\", \"Machine learning requires large datasets and GPU compute.\"),\n"
            "    (\"h-5\", \"Data pipelines often use Python with Apache Spark.\"),\n"
            "    (\"h-6\", \"The ownership model in Rust prevents data races at compile time.\"),\n"
            "]\n"
            "\n"
            "items = [\n"
            "    ContextItem(id=did, content=content, source=SourceType.RETRIEVAL,\n"
            "               score=0.0, priority=5, token_count=15)\n"
            "    for did, content in docs\n"
            "]\n"
            "\n"
            "dense.index(items)\n"
            "sparse.index(items)\n"
            "print(f\"Indexed {len(items)} documents in both retrievers.\")"
        ),
        md(
            "## Create the Hybrid Retriever\n"
            "Weights control the balance: `(0.7, 0.3)` means 70% dense, 30% sparse."
        ),
        code(
            "hybrid = HybridRetriever(\n"
            "    dense=dense,\n"
            "    sparse=sparse,\n"
            "    weights=(0.7, 0.3),\n"
            ")\n"
            "\n"
            "print(f\"Hybrid weights: dense={0.7}, sparse={0.3}\")"
        ),
        md("## Retrieve with RRF Fusion"),
        code(
            "query = QueryBundle(query_str=\"machine learning data\")\n"
            "results = hybrid.retrieve(query, top_k=4)\n"
            "\n"
            "print(f\"Query: '{query.query_str}'\")\n"
            "print(f\"Fused results ({len(results)}):\\n\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:60]}\")"
        ),
        md("## Experiment with Weights\nShift the balance toward sparse or dense."),
        code(
            "for w_dense, w_sparse in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]:\n"
            "    h = HybridRetriever(dense=dense, sparse=sparse, weights=(w_dense, w_sparse))\n"
            "    results = h.retrieve(query, top_k=3)\n"
            "    top_id = results[0].id if results else \"(none)\"\n"
            "    print(f\"  weights=({w_dense}, {w_sparse}) -> top result: {top_id}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `HybridRetriever` fuses dense (semantic) and sparse (keyword) results via RRF\n"
            "- `weights=(dense, sparse)` controls the fusion balance\n"
            "- Both sub-retrievers share the same indexed `ContextItem` objects\n"
            "- Hybrid retrieval is more robust than either method alone\n"
            "- Tune weights based on your query distribution\n"
            "\n"
            "**Next:** [Scored Memory Retriever](04_scored_memory_retriever.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Scored Memory Retriever
# ---------------------------------------------------------------------------
def nb_scored_memory_retriever():
    write_notebook("04_scored_memory_retriever.ipynb", [
        md(
            "# Scored Memory Retriever\n"
            "> Retrieve persistent memory entries by embedding similarity with score decay.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `ScoredMemoryRetriever`, `MemoryRetrieverAdapter`, `InMemoryEntryStore` |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "`ScoredMemoryRetriever` searches a persistent entry store using embeddings.\n"
            "`MemoryRetrieverAdapter` wraps it to conform to the standard `Retriever` protocol.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import ScoredMemoryRetriever, MemoryRetrieverAdapter\n"
            "from anchor.storage import InMemoryEntryStore\n"
            "from anchor.models import MemoryEntry, QueryBundle\n"
            "from datetime import datetime, timedelta, timezone"
        ),
        md("## Populate the Entry Store\nCreate memory entries with varying ages."),
        code(
            "entry_store = InMemoryEntryStore()\n"
            "now = datetime.now(timezone.utc)\n"
            "\n"
            "entries = [\n"
            "    MemoryEntry(entry_id=\"mem-1\", content=\"User prefers Python for scripting tasks.\",\n"
            "               created_at=now - timedelta(hours=1)),\n"
            "    MemoryEntry(entry_id=\"mem-2\", content=\"User is working on a Rust side project.\",\n"
            "               created_at=now - timedelta(hours=24)),\n"
            "    MemoryEntry(entry_id=\"mem-3\", content=\"User asked about Go concurrency patterns.\",\n"
            "               created_at=now - timedelta(hours=72)),\n"
            "    MemoryEntry(entry_id=\"mem-4\", content=\"User needs help with Python data pipelines.\",\n"
            "               created_at=now - timedelta(hours=2)),\n"
            "    MemoryEntry(entry_id=\"mem-5\", content=\"User mentioned interest in TypeScript migration.\",\n"
            "               created_at=now - timedelta(hours=168)),\n"
            "]\n"
            "\n"
            "for entry in entries:\n"
            "    entry_store.put(entry)\n"
            "\n"
            "print(f\"Entry store populated with {len(entries)} memories.\")"
        ),
        md("## Create the Scored Memory Retriever"),
        code(
            "def embed_fn(text: str) -> list[float]:\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "scored = ScoredMemoryRetriever(\n"
            "    store=entry_store,\n"
            "    embed_fn=embed_fn,\n"
            ")\n"
            "\n"
            "print(f\"Retriever type: {type(scored).__name__}\")"
        ),
        md("## Retrieve Memories by Query"),
        code(
            "query = QueryBundle(query_str=\"Python scripting\")\n"
            "results = scored.retrieve(query, top_k=3)\n"
            "\n"
            "print(f\"Query: '{query.query_str}'\")\n"
            "print(f\"Results ({len(results)}):\\n\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:60]}\")"
        ),
        md(
            "## Wrap with MemoryRetrieverAdapter\n"
            "The adapter makes `ScoredMemoryRetriever` conform to the standard `Retriever`\n"
            "protocol, allowing it to be used in pipelines alongside dense/sparse retrievers."
        ),
        code(
            "adapter = MemoryRetrieverAdapter(retriever=scored)\n"
            "\n"
            "# The adapter uses the same retrieve interface\n"
            "results = adapter.retrieve(QueryBundle(query_str=\"Rust project\"), top_k=2)\n"
            "\n"
            "print(f\"Adapter type: {type(adapter).__name__}\")\n"
            "print(f\"Results via adapter ({len(results)}):\\n\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:60]}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `ScoredMemoryRetriever` searches persistent memory entries by embedding similarity\n"
            "- Entries with recency and relevance combine for final scores\n"
            "- `MemoryRetrieverAdapter` wraps it to fit the standard `Retriever` protocol\n"
            "- Use the adapter to include memory retrieval in hybrid pipelines\n"
            "\n"
            "**Next:** [Async Retrievers](05_async_retrievers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Async Retrievers
# ---------------------------------------------------------------------------
def nb_async_retrievers():
    write_notebook("05_async_retrievers.ipynb", [
        md(
            "# Async Retrievers\n"
            "> Run retrieval concurrently with async/await for I/O-heavy workloads.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `AsyncDenseRetriever`, `AsyncHybridRetriever` |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "`AsyncDenseRetriever` and `AsyncHybridRetriever` expose `await aretrieve()`\n"
            "for non-blocking retrieval in async applications.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "import asyncio\n"
            "from anchor.retrieval import AsyncDenseRetriever, AsyncHybridRetriever\n"
            "from anchor.storage import InMemoryVectorStore, InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md("## Shared Infrastructure"),
        code(
            "vector_store = InMemoryVectorStore()\n"
            "ctx_store = InMemoryContextStore()\n"
            "\n"
            "def embed_fn(text: str) -> list[float]:\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "items = [\n"
            "    ContextItem(id=f\"async-{i}\", content=f\"Async doc {i}: {topic} programming guide.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=10)\n"
            "    for i, topic in enumerate([\"python\", \"rust\", \"go\", \"java\", \"kotlin\"])\n"
            "]\n"
            "\n"
            "print(f\"Prepared {len(items)} items for async retrieval.\")"
        ),
        md("## AsyncDenseRetriever\nIndex and retrieve using `await aretrieve()`."),
        code(
            "async_dense = AsyncDenseRetriever(\n"
            "    vector_store=vector_store,\n"
            "    context_store=ctx_store,\n"
            "    embed_fn=embed_fn,\n"
            ")\n"
            "\n"
            "# Index is typically synchronous\n"
            "async_dense.index(items)\n"
            "\n"
            "print(f\"Indexed {len(items)} items in AsyncDenseRetriever.\")"
        ),
        md("## Run an Async Query"),
        code(
            "async def run_dense_query():\n"
            "    query = QueryBundle(query_str=\"programming guide\")\n"
            "    results = await async_dense.aretrieve(query, top_k=3)\n"
            "    print(f\"Async dense results ({len(results)}):\")\n"
            "    for i, item in enumerate(results):\n"
            "        print(f\"  [{i+1}] {item.content[:50]}\")\n"
            "    return results\n"
            "\n"
            "# In a notebook, use 'await' directly. Here we use asyncio.run().\n"
            "results = asyncio.run(run_dense_query())"
        ),
        md(
            "## AsyncHybridRetriever\n"
            "Combine async dense and sparse retrieval with concurrent execution."
        ),
        code(
            "from anchor.retrieval import SparseRetriever\n"
            "\n"
            "sparse = SparseRetriever(context_store=ctx_store)\n"
            "sparse.index(items)\n"
            "\n"
            "async_hybrid = AsyncHybridRetriever(\n"
            "    dense=async_dense,\n"
            "    sparse=sparse,\n"
            "    weights=(0.6, 0.4),\n"
            ")\n"
            "\n"
            "print(\"AsyncHybridRetriever created with weights (0.6, 0.4).\")"
        ),
        md("## Concurrent Retrieval\nRun multiple queries in parallel using `asyncio.gather`."),
        code(
            "async def run_concurrent_queries():\n"
            "    queries = [\n"
            "        QueryBundle(query_str=\"python programming\"),\n"
            "        QueryBundle(query_str=\"rust guide\"),\n"
            "        QueryBundle(query_str=\"concurrent systems\"),\n"
            "    ]\n"
            "\n"
            "    tasks = [async_hybrid.aretrieve(q, top_k=2) for q in queries]\n"
            "    all_results = await asyncio.gather(*tasks)\n"
            "\n"
            "    for q, results in zip(queries, all_results):\n"
            "        top = results[0].content[:40] if results else \"(none)\"\n"
            "        print(f\"  '{q.query_str}' -> {top}\")\n"
            "\n"
            "asyncio.run(run_concurrent_queries())"
        ),
        md(
            "## Key Takeaways\n"
            "- `AsyncDenseRetriever` and `AsyncHybridRetriever` use `await aretrieve()`\n"
            "- Indexing remains synchronous; retrieval is async for I/O concurrency\n"
            "- Use `asyncio.gather` to run multiple queries in parallel\n"
            "- Ideal for web servers and applications with concurrent request handling\n"
            "\n"
            "**Next:** [Late Interaction Retriever](06_late_interaction.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Late Interaction Retriever
# ---------------------------------------------------------------------------
def nb_late_interaction():
    write_notebook("06_late_interaction.ipynb", [
        md(
            "# Late Interaction Retriever\n"
            "> Token-level interaction scoring for fine-grained retrieval (ColBERT-style).\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `LateInteractionRetriever`, `LateInteractionScorer`, "
            "`MaxSimScorer`, `SharedSpaceRetriever`, `CrossModalEncoder` |\n"
            "| Difficulty | Advanced |\n"
            "\n"
            "Late interaction models compute fine-grained token-to-token similarities\n"
            "between queries and documents, enabling more precise matching than single-vector\n"
            "approaches.\n"
            "\n"
            "**Time:** ~10 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import (\n"
            "    LateInteractionRetriever,\n"
            "    LateInteractionScorer,\n"
            "    MaxSimScorer,\n"
            "    SharedSpaceRetriever,\n"
            "    CrossModalEncoder,\n"
            ")\n"
            "from anchor.storage import InMemoryVectorStore, InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle\n"
            "import random"
        ),
        md(
            "## Mock Multi-Vector Encoder\n"
            "Late interaction requires a per-token encoder that returns a matrix\n"
            "(one vector per token) rather than a single vector."
        ),
        code(
            "def mock_token_encoder(text: str) -> list[list[float]]:\n"
            "    \"\"\"Return a list of token-level embeddings (mock).\"\"\"\n"
            "    random.seed(hash(text) % 10000)\n"
            "    tokens = text.split()[:20]  # cap at 20 tokens\n"
            "    dim = 32\n"
            "    return [[random.random() for _ in range(dim)] for _ in tokens]\n"
            "\n"
            "sample = mock_token_encoder(\"hello world test\")\n"
            "print(f\"Tokens: 3, Vectors: {len(sample)}, Dim: {len(sample[0])}\")"
        ),
        md(
            "## MaxSim Scorer\n"
            "`MaxSimScorer` computes the maximum similarity between each query token\n"
            "and all document tokens, then sums the scores (ColBERT-style MaxSim)."
        ),
        code(
            "scorer = MaxSimScorer()\n"
            "\n"
            "q_vecs = mock_token_encoder(\"what is Python\")\n"
            "d_vecs = mock_token_encoder(\"Python is a popular programming language\")\n"
            "\n"
            "score = scorer.score(q_vecs, d_vecs)\n"
            "print(f\"MaxSim score: {score:.4f}\")"
        ),
        md(
            "## Late Interaction Scorer\n"
            "Wraps a token encoder and a similarity scorer into a single scoring interface."
        ),
        code(
            "li_scorer = LateInteractionScorer(\n"
            "    encoder=mock_token_encoder,\n"
            "    scorer=scorer,\n"
            ")\n"
            "\n"
            "score = li_scorer.score_pair(\"what is Python\", \"Python is a popular language\")\n"
            "print(f\"Late interaction score: {score:.4f}\")"
        ),
        md("## LateInteractionRetriever\nCombine late interaction scoring with a retriever."),
        code(
            "vector_store = InMemoryVectorStore()\n"
            "ctx_store = InMemoryContextStore()\n"
            "\n"
            "# Single-vector embed_fn for initial candidate retrieval\n"
            "def embed_fn(text: str) -> list[float]:\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "li_retriever = LateInteractionRetriever(\n"
            "    vector_store=vector_store,\n"
            "    context_store=ctx_store,\n"
            "    embed_fn=embed_fn,\n"
            "    scorer=li_scorer,\n"
            ")\n"
            "\n"
            "items = [\n"
            "    ContextItem(id=f\"li-{i}\", content=text, source=SourceType.RETRIEVAL,\n"
            "               score=0.0, priority=5, token_count=12)\n"
            "    for i, text in enumerate([\n"
            "        \"Python is great for data analysis and scripting.\",\n"
            "        \"Rust has a steep learning curve but excellent safety.\",\n"
            "        \"Go routines make concurrent programming straightforward.\",\n"
            "        \"JavaScript dominates frontend web development.\",\n"
            "    ])\n"
            "]\n"
            "\n"
            "li_retriever.index(items)\n"
            "print(f\"Indexed {len(items)} items in LateInteractionRetriever.\")"
        ),
        code(
            "results = li_retriever.retrieve(QueryBundle(query_str=\"data scripting\"), top_k=3)\n"
            "\n"
            "print(\"Late interaction retrieval results:\")\n"
            "for i, item in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={item.score:.4f}  {item.content[:50]}\")"
        ),
        md(
            "## SharedSpaceRetriever and CrossModalEncoder\n"
            "For cross-modal retrieval (e.g., text-to-image), a `CrossModalEncoder`\n"
            "projects different modalities into a shared embedding space."
        ),
        code(
            "# Mock cross-modal encoder: maps any modality to the same vector space\n"
            "cross_encoder = CrossModalEncoder(\n"
            "    text_encoder=embed_fn,\n"
            "    image_encoder=embed_fn,  # In practice, a vision model\n"
            ")\n"
            "\n"
            "shared_retriever = SharedSpaceRetriever(\n"
            "    vector_store=InMemoryVectorStore(),\n"
            "    context_store=InMemoryContextStore(),\n"
            "    encoder=cross_encoder,\n"
            ")\n"
            "\n"
            "print(f\"SharedSpaceRetriever type: {type(shared_retriever).__name__}\")\n"
            "print(f\"CrossModalEncoder type: {type(cross_encoder).__name__}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MaxSimScorer` implements ColBERT-style token-level MaxSim scoring\n"
            "- `LateInteractionScorer` pairs a token encoder with a similarity scorer\n"
            "- `LateInteractionRetriever` uses coarse candidates + fine-grained reranking\n"
            "- `CrossModalEncoder` and `SharedSpaceRetriever` enable cross-modal search\n"
            "- Late interaction improves precision over single-vector retrieval\n"
            "\n"
            "**Next:** [Rerankers](07_rerankers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 07 - Rerankers
# ---------------------------------------------------------------------------
def nb_rerankers():
    write_notebook("07_rerankers.ipynb", [
        md(
            "# Rerankers\n"
            "> Score, reorder, and pipeline multiple reranking strategies.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `ScoreReranker`, `CrossEncoderReranker`, `FlashRankReranker`, "
            "`CohereReranker`, `RoundRobinReranker`, `RerankerPipeline`, `AsyncCrossEncoderReranker` |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "Rerankers refine initial retrieval results by applying more expensive scoring\n"
            "models. Anchor provides several strategies that can be combined into a pipeline.\n"
            "\n"
            "**Time:** ~10 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import (\n"
            "    ScoreReranker,\n"
            "    CrossEncoderReranker,\n"
            "    FlashRankReranker,\n"
            "    CohereReranker,\n"
            "    RoundRobinReranker,\n"
            "    RerankerPipeline,\n"
            "    AsyncCrossEncoderReranker,\n"
            ")\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md("## Create Sample Results\nSimulate initial retrieval results to rerank."),
        code(
            "def make_results():\n"
            "    \"\"\"Create mock retrieval results with varying scores.\"\"\"\n"
            "    data = [\n"
            "        (\"r-1\", \"Python is great for data science.\", 0.85),\n"
            "        (\"r-2\", \"Rust ensures memory safety.\", 0.72),\n"
            "        (\"r-3\", \"Go is designed for concurrency.\", 0.68),\n"
            "        (\"r-4\", \"Python data pipelines with Pandas.\", 0.91),\n"
            "        (\"r-5\", \"JavaScript powers the web.\", 0.55),\n"
            "        (\"r-6\", \"Machine learning with Python and TensorFlow.\", 0.88),\n"
            "    ]\n"
            "    return [\n"
            "        ContextItem(id=did, content=text, source=SourceType.RETRIEVAL,\n"
            "                   score=score, priority=5, token_count=10)\n"
            "        for did, text, score in data\n"
            "    ]\n"
            "\n"
            "results = make_results()\n"
            "print(\"Initial results (by original score):\")\n"
            "for r in results:\n"
            "    print(f\"  {r.id}: score={r.score:.2f}  {r.content[:40]}\")"
        ),
        md(
            "## ScoreReranker\n"
            "Simply re-sorts results by their existing scores (useful as a baseline or\n"
            "normalization step)."
        ),
        code(
            "score_reranker = ScoreReranker()\n"
            "query = QueryBundle(query_str=\"Python data science\")\n"
            "\n"
            "reranked = score_reranker.rerank(query, make_results(), top_k=4)\n"
            "\n"
            "print(\"ScoreReranker results:\")\n"
            "for i, r in enumerate(reranked):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")"
        ),
        md(
            "## CrossEncoderReranker (Mocked)\n"
            "In production, this calls a cross-encoder model (e.g., `sentence-transformers`).\n"
            "Here we mock it with a simple scoring function."
        ),
        code(
            "# Mock cross-encoder scoring function\n"
            "def mock_cross_encoder(query: str, document: str) -> float:\n"
            "    \"\"\"Higher score if query terms appear in document.\"\"\"\n"
            "    query_terms = set(query.lower().split())\n"
            "    doc_terms = set(document.lower().split())\n"
            "    overlap = len(query_terms & doc_terms)\n"
            "    return overlap / max(len(query_terms), 1)\n"
            "\n"
            "cross_reranker = CrossEncoderReranker(score_fn=mock_cross_encoder)\n"
            "reranked = cross_reranker.rerank(query, make_results(), top_k=4)\n"
            "\n"
            "print(\"CrossEncoderReranker results:\")\n"
            "for i, r in enumerate(reranked):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")"
        ),
        md(
            "## FlashRankReranker (Mocked)\n"
            "FlashRank provides fast, lightweight reranking. We mock the scoring here."
        ),
        code(
            "# Mock FlashRank scoring\n"
            "def mock_flashrank(query: str, document: str) -> float:\n"
            "    \"\"\"Simple length-normalized overlap score.\"\"\"\n"
            "    q_words = set(query.lower().split())\n"
            "    d_words = set(document.lower().split())\n"
            "    return len(q_words & d_words) / max(len(d_words), 1)\n"
            "\n"
            "flash_reranker = FlashRankReranker(score_fn=mock_flashrank)\n"
            "reranked = flash_reranker.rerank(query, make_results(), top_k=4)\n"
            "\n"
            "print(\"FlashRankReranker results:\")\n"
            "for i, r in enumerate(reranked):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")"
        ),
        md(
            "## CohereReranker (Mocked)\n"
            "The Cohere reranker calls the Cohere Rerank API. We mock it for demonstration."
        ),
        code(
            "# Mock Cohere reranking\n"
            "def mock_cohere(query: str, document: str) -> float:\n"
            "    \"\"\"Simulate Cohere-style relevance scoring.\"\"\"\n"
            "    q_set = set(query.lower().split())\n"
            "    d_set = set(document.lower().split())\n"
            "    overlap = len(q_set & d_set)\n"
            "    return min(overlap * 0.3, 1.0)\n"
            "\n"
            "cohere_reranker = CohereReranker(score_fn=mock_cohere)\n"
            "reranked = cohere_reranker.rerank(query, make_results(), top_k=4)\n"
            "\n"
            "print(\"CohereReranker results:\")\n"
            "for i, r in enumerate(reranked):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")"
        ),
        md(
            "## RoundRobinReranker\n"
            "Interleaves results from multiple result sets in a round-robin fashion.\n"
            "Useful for combining results from different retrieval strategies."
        ),
        code(
            "rr_reranker = RoundRobinReranker()\n"
            "\n"
            "# Simulate two different result sets\n"
            "set_a = make_results()[:3]  # First 3\n"
            "set_b = make_results()[3:]  # Last 3\n"
            "\n"
            "interleaved = rr_reranker.interleave([set_a, set_b], top_k=5)\n"
            "\n"
            "print(\"RoundRobinReranker interleaved results:\")\n"
            "for i, r in enumerate(interleaved):\n"
            "    print(f\"  [{i+1}] {r.id}  {r.content[:40]}\")"
        ),
        md(
            "## RerankerPipeline\n"
            "Chain multiple rerankers into a sequential pipeline. Each stage refines\n"
            "the output of the previous one."
        ),
        code(
            "pipeline = RerankerPipeline(\n"
            "    stages=[\n"
            "        score_reranker,       # Stage 1: sort by score\n"
            "        cross_reranker,       # Stage 2: cross-encoder refinement\n"
            "    ]\n"
            ")\n"
            "\n"
            "final = pipeline.rerank(query, make_results(), top_k=3)\n"
            "\n"
            "print(\"RerankerPipeline results:\")\n"
            "for i, r in enumerate(final):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")"
        ),
        md(
            "## AsyncCrossEncoderReranker (Mocked)\n"
            "For async applications, use `AsyncCrossEncoderReranker` with `await arerank()`."
        ),
        code(
            "import asyncio\n"
            "\n"
            "async def mock_async_score(query: str, document: str) -> float:\n"
            "    return mock_cross_encoder(query, document)\n"
            "\n"
            "async_reranker = AsyncCrossEncoderReranker(score_fn=mock_async_score)\n"
            "\n"
            "async def run_async_rerank():\n"
            "    results = await async_reranker.arerank(query, make_results(), top_k=3)\n"
            "    print(\"AsyncCrossEncoderReranker results:\")\n"
            "    for i, r in enumerate(results):\n"
            "        print(f\"  [{i+1}] score={r.score:.4f}  {r.content[:40]}\")\n"
            "\n"
            "asyncio.run(run_async_rerank())"
        ),
        md(
            "## Key Takeaways\n"
            "- `ScoreReranker` sorts by existing scores (baseline)\n"
            "- `CrossEncoderReranker` applies a cross-encoder model for pairwise scoring\n"
            "- `FlashRankReranker` and `CohereReranker` integrate external reranking services\n"
            "- `RoundRobinReranker` interleaves results from multiple sources\n"
            "- `RerankerPipeline` chains stages sequentially for multi-pass refinement\n"
            "- `AsyncCrossEncoderReranker` provides async support via `await arerank()`\n"
            "\n"
            "**Next:** [Routers](08_routers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 08 - Routers
# ---------------------------------------------------------------------------
def nb_routers():
    write_notebook("08_routers.ipynb", [
        md(
            "# Routers\n"
            "> Route queries to the right retriever based on rules, keywords, or metadata.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `CallbackRouter`, `KeywordRouter`, `MetadataRouter`, `RoutedRetriever` |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "Routers decide which retriever should handle a query. This enables specialized\n"
            "retrieval paths for different query types.\n"
            "\n"
            "**Time:** ~8 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.retrieval import (\n"
            "    CallbackRouter,\n"
            "    KeywordRouter,\n"
            "    MetadataRouter,\n"
            "    RoutedRetriever,\n"
            "    DenseRetriever,\n"
            "    SparseRetriever,\n"
            ")\n"
            "from anchor.storage import InMemoryVectorStore, InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType, QueryBundle"
        ),
        md("## Create Two Specialized Retrievers\nOne for code topics, one for general docs."),
        code(
            "def embed_fn(text: str) -> list[float]:\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "# Code retriever\n"
            "code_store = InMemoryVectorStore()\n"
            "code_ctx = InMemoryContextStore()\n"
            "code_retriever = DenseRetriever(\n"
            "    vector_store=code_store, context_store=code_ctx, embed_fn=embed_fn,\n"
            ")\n"
            "code_items = [\n"
            "    ContextItem(id=\"code-1\", content=\"def hello(): print('hello world')\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=8),\n"
            "    ContextItem(id=\"code-2\", content=\"async def fetch(url): return await get(url)\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=10),\n"
            "]\n"
            "code_retriever.index(code_items)\n"
            "\n"
            "# Docs retriever\n"
            "docs_store = InMemoryVectorStore()\n"
            "docs_ctx = InMemoryContextStore()\n"
            "docs_retriever = DenseRetriever(\n"
            "    vector_store=docs_store, context_store=docs_ctx, embed_fn=embed_fn,\n"
            ")\n"
            "docs_items = [\n"
            "    ContextItem(id=\"doc-1\", content=\"Anchor is a context management framework.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=8),\n"
            "    ContextItem(id=\"doc-2\", content=\"Installation: pip install astro-anchor\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=7),\n"
            "]\n"
            "docs_retriever.index(docs_items)\n"
            "\n"
            "print(\"Code retriever: 2 items indexed.\")\n"
            "print(\"Docs retriever: 2 items indexed.\")"
        ),
        md(
            "## CallbackRouter\n"
            "Route based on a user-defined callback function."
        ),
        code(
            "def route_fn(query: QueryBundle) -> str:\n"
            "    \"\"\"Route code-related queries to the code retriever.\"\"\"\n"
            "    code_terms = {\"def\", \"async\", \"function\", \"class\", \"code\", \"implement\"}\n"
            "    query_terms = set(query.query_str.lower().split())\n"
            "    if query_terms & code_terms:\n"
            "        return \"code\"\n"
            "    return \"docs\"\n"
            "\n"
            "callback_router = CallbackRouter(route_fn=route_fn)\n"
            "\n"
            "# Test routing decisions\n"
            "for q in [\"implement a function\", \"how to install anchor\", \"async code example\"]:\n"
            "    route = callback_router.route(QueryBundle(query_str=q))\n"
            "    print(f\"  '{q}' -> route: {route}\")"
        ),
        md(
            "## KeywordRouter\n"
            "Route based on keyword matching with configurable keyword-to-route mappings."
        ),
        code(
            "keyword_router = KeywordRouter(\n"
            "    routes={\n"
            "        \"code\": [\"def\", \"class\", \"function\", \"implement\", \"code\"],\n"
            "        \"docs\": [\"install\", \"guide\", \"tutorial\", \"documentation\"],\n"
            "    },\n"
            "    default_route=\"docs\",\n"
            ")\n"
            "\n"
            "for q in [\"implement a class\", \"installation guide\", \"random query\"]:\n"
            "    route = keyword_router.route(QueryBundle(query_str=q))\n"
            "    print(f\"  '{q}' -> route: {route}\")"
        ),
        md(
            "## MetadataRouter\n"
            "Route based on metadata attached to the `QueryBundle`."
        ),
        code(
            "metadata_router = MetadataRouter(\n"
            "    metadata_key=\"domain\",\n"
            "    default_route=\"docs\",\n"
            ")\n"
            "\n"
            "q_code = QueryBundle(query_str=\"show me examples\", metadata={\"domain\": \"code\"})\n"
            "q_docs = QueryBundle(query_str=\"show me examples\", metadata={\"domain\": \"docs\"})\n"
            "q_none = QueryBundle(query_str=\"show me examples\")\n"
            "\n"
            "print(f\"  domain='code'  -> route: {metadata_router.route(q_code)}\")\n"
            "print(f\"  domain='docs'  -> route: {metadata_router.route(q_docs)}\")\n"
            "print(f\"  domain=None    -> route: {metadata_router.route(q_none)} (default)\")"
        ),
        md(
            "## RoutedRetriever\n"
            "Combine a router with a mapping of route names to retrievers."
        ),
        code(
            "routed = RoutedRetriever(\n"
            "    router=keyword_router,\n"
            "    retrievers={\n"
            "        \"code\": code_retriever,\n"
            "        \"docs\": docs_retriever,\n"
            "    },\n"
            ")\n"
            "\n"
            "# Query routed to 'code' retriever\n"
            "results = routed.retrieve(QueryBundle(query_str=\"implement a function\"), top_k=2)\n"
            "print(\"Query: 'implement a function' (routed to: code)\")\n"
            "for r in results:\n"
            "    print(f\"  {r.id}: {r.content[:50]}\")\n"
            "\n"
            "print()\n"
            "\n"
            "# Query routed to 'docs' retriever\n"
            "results = routed.retrieve(QueryBundle(query_str=\"installation guide\"), top_k=2)\n"
            "print(\"Query: 'installation guide' (routed to: docs)\")\n"
            "for r in results:\n"
            "    print(f\"  {r.id}: {r.content[:50]}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `CallbackRouter` routes via a user-defined function for maximum flexibility\n"
            "- `KeywordRouter` routes by matching query terms against keyword lists\n"
            "- `MetadataRouter` routes based on `QueryBundle.metadata` fields\n"
            "- `RoutedRetriever` wires a router to a dict of retrievers\n"
            "- Routing enables specialized retrieval paths without complex conditionals\n"
            "\n"
            "**Next:** [Custom Retriever](09_custom_retriever.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 09 - Custom Retriever
# ---------------------------------------------------------------------------
def nb_custom_retriever():
    write_notebook("09_custom_retriever.ipynb", [
        md(
            "# Custom Retriever\n"
            "> Implement the `Retriever` protocol using PEP 544 structural subtyping.\n"
            "\n"
            "| Module | `anchor.retrieval` |\n"
            "|--------|--------------------|\n"
            "| Key classes | `Retriever` protocol |\n"
            "| Difficulty | Intermediate |\n"
            "\n"
            "Anchor's `Retriever` is a PEP 544 `Protocol` class. Any object with matching\n"
            "`retrieve()` and `index()` methods is a valid retriever -- no inheritance required.\n"
            "\n"
            "**Time:** ~8 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.models import ContextItem, SourceType, QueryBundle\n"
            "from typing import Protocol, runtime_checkable"
        ),
        md(
            "## The Retriever Protocol\n"
            "Here is what the protocol looks like (simplified). Your class just needs\n"
            "to implement `retrieve()` and `index()`."
        ),
        code(
            "# This is the protocol defined by Anchor (shown for reference)\n"
            "#\n"
            "# @runtime_checkable\n"
            "# class Retriever(Protocol):\n"
            "#     def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...\n"
            "#     def index(self, items: list[ContextItem]) -> None: ...\n"
            "\n"
            "print(\"Retriever protocol: retrieve(query, top_k) -> list[ContextItem]\")\n"
            "print(\"                     index(items) -> None\")"
        ),
        md(
            "## Implement a Custom Retriever\n"
            "We build a simple retriever that matches documents by substring overlap."
        ),
        code(
            "class SubstringRetriever:\n"
            "    \"\"\"A custom retriever that scores documents by substring overlap with the query.\"\"\"\n"
            "\n"
            "    def __init__(self):\n"
            "        self._items: list[ContextItem] = []\n"
            "\n"
            "    def index(self, items: list[ContextItem]) -> None:\n"
            "        \"\"\"Add items to the internal store.\"\"\"\n"
            "        self._items.extend(items)\n"
            "\n"
            "    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:\n"
            "        \"\"\"Score items by word overlap and return the top-k.\"\"\"\n"
            "        query_words = set(query.query_str.lower().split())\n"
            "\n"
            "        scored = []\n"
            "        for item in self._items:\n"
            "            doc_words = set(item.content.lower().split())\n"
            "            overlap = len(query_words & doc_words)\n"
            "            score = overlap / max(len(query_words), 1)\n"
            "            scored.append((score, item))\n"
            "\n"
            "        scored.sort(key=lambda x: x[0], reverse=True)\n"
            "\n"
            "        results = []\n"
            "        for score, item in scored[:top_k]:\n"
            "            # Return a copy with the updated score\n"
            "            results.append(ContextItem(\n"
            "                id=item.id,\n"
            "                content=item.content,\n"
            "                source=item.source,\n"
            "                score=score,\n"
            "                priority=item.priority,\n"
            "                token_count=item.token_count,\n"
            "            ))\n"
            "        return results\n"
            "\n"
            "print(\"SubstringRetriever class defined.\")"
        ),
        md("## Verify Protocol Conformance\nUse `isinstance()` with the runtime-checkable protocol."),
        code(
            "from anchor.retrieval import Retriever\n"
            "\n"
            "my_retriever = SubstringRetriever()\n"
            "\n"
            "print(f\"Is SubstringRetriever a Retriever? {isinstance(my_retriever, Retriever)}\")"
        ),
        md("## Index and Retrieve"),
        code(
            "items = [\n"
            "    ContextItem(id=\"custom-1\", content=\"Python is great for data science.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=8),\n"
            "    ContextItem(id=\"custom-2\", content=\"Rust ensures memory safety.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=6),\n"
            "    ContextItem(id=\"custom-3\", content=\"Data pipelines process large datasets.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=7),\n"
            "    ContextItem(id=\"custom-4\", content=\"Go is designed for concurrent programming.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.0, priority=5, token_count=8),\n"
            "]\n"
            "\n"
            "my_retriever.index(items)\n"
            "print(f\"Indexed {len(items)} items.\")"
        ),
        code(
            "query = QueryBundle(query_str=\"data science pipelines\")\n"
            "results = my_retriever.retrieve(query, top_k=3)\n"
            "\n"
            "print(f\"Query: '{query.query_str}'\")\n"
            "print(f\"Results ({len(results)}):\\n\")\n"
            "for i, r in enumerate(results):\n"
            "    print(f\"  [{i+1}] score={r.score:.4f}  {r.content}\")"
        ),
        md(
            "## Use the Custom Retriever in a RoutedRetriever\n"
            "Since it satisfies the protocol, it plugs into any Anchor component\n"
            "that accepts a `Retriever`."
        ),
        code(
            "from anchor.retrieval import RoutedRetriever, KeywordRouter\n"
            "\n"
            "router = KeywordRouter(\n"
            "    routes={\"custom\": [\"data\", \"science\", \"pipeline\"]},\n"
            "    default_route=\"custom\",\n"
            ")\n"
            "\n"
            "routed = RoutedRetriever(\n"
            "    router=router,\n"
            "    retrievers={\"custom\": my_retriever},\n"
            ")\n"
            "\n"
            "results = routed.retrieve(QueryBundle(query_str=\"data pipeline\"), top_k=2)\n"
            "\n"
            "print(\"Routed through RoutedRetriever:\")\n"
            "for r in results:\n"
            "    print(f\"  {r.id}: score={r.score:.4f}  {r.content[:50]}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- Anchor uses PEP 544 structural subtyping -- no base class inheritance needed\n"
            "- Implement `retrieve(query, top_k)` and `index(items)` to satisfy the protocol\n"
            "- Use `isinstance(obj, Retriever)` to verify conformance at runtime\n"
            "- Custom retrievers plug seamlessly into `RoutedRetriever`, `HybridRetriever`, etc.\n"
            "- This pattern encourages composition over inheritance\n"
            "\n"
            "**Back to:** [Retrieval README](README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Retrieval notebooks in {OUTPUT_DIR}/\n")

    nb_dense_retriever()
    nb_sparse_retriever()
    nb_hybrid_retriever()
    nb_scored_memory_retriever()
    nb_async_retrievers()
    nb_late_interaction()
    nb_rerankers()
    nb_routers()
    nb_custom_retriever()

    print(f"\nDone. 9 notebooks created.")


if __name__ == "__main__":
    main()
