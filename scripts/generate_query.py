"""Generate all 6 Jupyter notebooks for the Query Transformation module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "05-query"))
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
# 01 - HyDE Transformer
# ---------------------------------------------------------------------------
def nb_hyde_transformer():
    write_notebook("01_hyde_transformer.ipynb", [
        md(
            "# HyDE Transformer\n"
            "> Expand queries with hypothetical document embeddings for better retrieval.\n"
            "\n"
            "`HyDETransformer` generates a hypothetical document that *would* answer the\n"
            "query, then uses that document's embedding for retrieval instead of the\n"
            "original query. This bridges the vocabulary gap between short queries and\n"
            "long documents.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query import HyDETransformer\n"
            "from anchor.models import QueryBundle"
        ),
        md(
            "## Define a Generation Function\n"
            "HyDE needs a function that takes a query string and returns a hypothetical\n"
            "document. In production this would call an LLM; here we use a mock."
        ),
        code(
            "def mock_generate(query: str) -> str:\n"
            "    \"\"\"Simulate an LLM generating a hypothetical answer document.\"\"\"\n"
            "    return (\n"
            "        f\"A hypothetical document about {query} would discuss the fundamentals \"\n"
            "        f\"of the topic, covering key concepts, common patterns, and practical \"\n"
            "        f\"applications. It would explain how {query} relates to modern software \"\n"
            "        f\"engineering practices and provide concrete examples.\"\n"
            "    )\n"
            "\n"
            "# Test the generation function\n"
            "sample = mock_generate(\"context engineering\")\n"
            "print(f\"Generated document ({len(sample)} chars):\")\n"
            "print(f\"  {sample[:80]}...\")"
        ),
        md(
            "## Create the HyDE Transformer\n"
            "Pass the generation function to `HyDETransformer`. The transformer will\n"
            "call it on each query to produce a hypothetical document."
        ),
        code(
            "hyde = HyDETransformer(generate_fn=mock_generate)\n"
            "\n"
            "print(f\"Transformer: {type(hyde).__name__}\")\n"
            "print(f\"Generate fn: {hyde.generate_fn.__name__}\")"
        ),
        md(
            "## Transform a Query\n"
            "Wrap the query string in a `QueryBundle`, then call `transform()`. The\n"
            "result is a list of `QueryBundle` objects -- the original plus the\n"
            "hypothetical document version."
        ),
        code(
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "print(f\"Original query: {query.query_str}\")\n"
            "\n"
            "expanded = hyde.transform(query)\n"
            "\n"
            "print(f\"\\nExpanded to {len(expanded)} queries:\\n\")\n"
            "for i, q in enumerate(expanded):\n"
            "    print(f\"  [{i}] ({len(q.query_str)} chars): {q.query_str[:70]}...\")"
        ),
        md(
            "## Try Different Queries\n"
            "HyDE works best when queries are short and would benefit from expansion\n"
            "into richer document-like text."
        ),
        code(
            "test_queries = [\n"
            "    \"What is context engineering?\",\n"
            "    \"How does RAG work?\",\n"
            "    \"vector database comparison\",\n"
            "    \"memory management in LLM applications\",\n"
            "]\n"
            "\n"
            "for q_str in test_queries:\n"
            "    q = QueryBundle(query_str=q_str)\n"
            "    results = hyde.transform(q)\n"
            "    hypo = results[-1].query_str  # last one is the hypothetical doc\n"
            "    print(f\"Query: {q_str}\")\n"
            "    print(f\"  HyDE: {hypo[:65]}...\")\n"
            "    print()"
        ),
        md(
            "## When to Use HyDE\n"
            "HyDE shines when there is a vocabulary mismatch between user queries and\n"
            "stored documents."
        ),
        code(
            "# Compare query lengths before and after HyDE\n"
            "queries = [\"RAG\", \"How to build a chatbot\", \"Explain transformer attention\"]\n"
            "\n"
            "print(f\"{'Query':<30} {'Original':>10} {'HyDE':>10} {'Expansion':>10}\")\n"
            "print(\"-\" * 65)\n"
            "for q_str in queries:\n"
            "    q = QueryBundle(query_str=q_str)\n"
            "    results = hyde.transform(q)\n"
            "    orig_len = len(q_str)\n"
            "    hyde_len = len(results[-1].query_str)\n"
            "    ratio = hyde_len / orig_len\n"
            "    print(f\"{q_str:<30} {orig_len:>10} {hyde_len:>10} {ratio:>9.1f}x\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `HyDETransformer` generates a hypothetical answer document for each query\n"
            "- The hypothetical document is used for embedding-based retrieval\n"
            "- Bridges the vocabulary gap between short queries and long documents\n"
            "- Requires a generation function (LLM call in production, mock for testing)\n"
            "- Best for short, keyword-style queries that need expansion\n"
            "\n"
            "**Next:** [Multi-Query Expansion](02_multi_query.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Multi-Query Expansion
# ---------------------------------------------------------------------------
def nb_multi_query():
    write_notebook("02_multi_query.ipynb", [
        md(
            "# Multi-Query Expansion\n"
            "> Generate multiple query variations to improve retrieval recall.\n"
            "\n"
            "`MultiQueryTransformer` produces several rephrased versions of the\n"
            "original query. Each variation is used for retrieval independently,\n"
            "and results are merged. This captures different facets of the user's\n"
            "intent.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query import MultiQueryTransformer\n"
            "from anchor.models import QueryBundle"
        ),
        md(
            "## Define a Multi-Query Generator\n"
            "The generator takes a query string and the number of variations to\n"
            "produce. In production this would be an LLM call."
        ),
        code(
            "def mock_generate(query: str, num: int) -> list:\n"
            "    \"\"\"Generate num query variations from the original query.\"\"\"\n"
            "    variations = [\n"
            "        f\"{query} - explained simply\",\n"
            "        f\"What are the key concepts of {query.lower()}?\",\n"
            "        f\"Practical guide to {query.lower()}\",\n"
            "        f\"{query} best practices and patterns\",\n"
            "        f\"How does {query.lower()} work in practice?\",\n"
            "    ]\n"
            "    return variations[:num]\n"
            "\n"
            "# Test the generator\n"
            "results = mock_generate(\"context engineering\", 3)\n"
            "print(f\"Generated {len(results)} variations:\")\n"
            "for i, v in enumerate(results):\n"
            "    print(f\"  [{i}] {v}\")"
        ),
        md(
            "## Create the Multi-Query Transformer\n"
            "Set `num_queries` to control how many variations are generated per query."
        ),
        code(
            "multi = MultiQueryTransformer(generate_fn=mock_generate, num_queries=3)\n"
            "\n"
            "print(f\"Transformer: {type(multi).__name__}\")\n"
            "print(f\"Num queries: {multi.num_queries}\")"
        ),
        md(
            "## Transform a Query\n"
            "`transform()` returns a list of `QueryBundle` objects, one for each\n"
            "generated variation."
        ),
        code(
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "print(f\"Original: {query.query_str}\\n\")\n"
            "\n"
            "expanded = multi.transform(query)\n"
            "\n"
            "print(f\"Expanded to {len(expanded)} queries:\")\n"
            "for i, q in enumerate(expanded):\n"
            "    print(f\"  [{i}] {q.query_str}\")"
        ),
        md(
            "## Vary the Number of Queries\n"
            "More variations improve recall but increase retrieval cost."
        ),
        code(
            "for n in [2, 3, 5]:\n"
            "    transformer = MultiQueryTransformer(generate_fn=mock_generate, num_queries=n)\n"
            "    results = transformer.transform(query)\n"
            "    print(f\"num_queries={n}: {len(results)} queries generated\")\n"
            "    for q in results:\n"
            "        print(f\"    {q.query_str}\")\n"
            "    print()"
        ),
        md(
            "## Multi-Query Retrieval Pattern\n"
            "In a full pipeline, each variation retrieves independently and results\n"
            "are merged with deduplication."
        ),
        code(
            "# Simulate multi-query retrieval\n"
            "def mock_retrieve(query_str: str) -> list:\n"
            "    \"\"\"Return mock document IDs based on query content.\"\"\"\n"
            "    base = hash(query_str) % 100\n"
            "    return [f\"doc-{base + i}\" for i in range(3)]\n"
            "\n"
            "expanded = multi.transform(query)\n"
            "\n"
            "all_results = []\n"
            "for q in expanded:\n"
            "    docs = mock_retrieve(q.query_str)\n"
            "    all_results.extend(docs)\n"
            "    print(f\"  Query: {q.query_str[:50]}...\")\n"
            "    print(f\"    Retrieved: {docs}\")\n"
            "\n"
            "# Deduplicate\n"
            "unique_docs = list(dict.fromkeys(all_results))\n"
            "print(f\"\\nTotal retrieved: {len(all_results)}\")\n"
            "print(f\"After dedup:     {len(unique_docs)}\")\n"
            "print(f\"Unique docs:     {unique_docs}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MultiQueryTransformer` generates multiple phrasings of a single query\n"
            "- Each variation retrieves independently, improving recall\n"
            "- Results are merged and deduplicated across all variations\n"
            "- `num_queries` controls the recall-vs-cost trade-off\n"
            "- The generation function can be swapped between mock and real LLM calls\n"
            "\n"
            "**Next:** [Query Decomposition](03_decomposition.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Query Decomposition
# ---------------------------------------------------------------------------
def nb_decomposition():
    write_notebook("03_decomposition.ipynb", [
        md(
            "# Query Decomposition\n"
            "> Break complex queries into simpler sub-questions for targeted retrieval.\n"
            "\n"
            "`DecompositionTransformer` splits a multi-part query into focused\n"
            "sub-questions. Each sub-question retrieves independently, and results\n"
            "are combined to answer the original complex query.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query import DecompositionTransformer\n"
            "from anchor.models import QueryBundle"
        ),
        md(
            "## Define a Decomposition Function\n"
            "The decompose function takes a query string and returns a list of\n"
            "sub-question strings. In production this calls an LLM."
        ),
        code(
            "def mock_decompose(query: str) -> list:\n"
            "    \"\"\"Break a query into sub-questions based on its key terms.\"\"\"\n"
            "    words = query.split()[:3]\n"
            "    return [f\"Sub-question {i}: {part}\" for i, part in enumerate(words)]\n"
            "\n"
            "# Test the decomposition\n"
            "subs = mock_decompose(\"What is context engineering?\")\n"
            "print(f\"Decomposed into {len(subs)} sub-questions:\")\n"
            "for s in subs:\n"
            "    print(f\"  {s}\")"
        ),
        md(
            "## Create the Decomposition Transformer\n"
            "Pass the decompose function to `DecompositionTransformer`."
        ),
        code(
            "decomp = DecompositionTransformer(decompose_fn=mock_decompose)\n"
            "\n"
            "print(f\"Transformer: {type(decomp).__name__}\")"
        ),
        md(
            "## Transform a Query\n"
            "`transform()` returns a list of `QueryBundle` objects, one per\n"
            "sub-question."
        ),
        code(
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "print(f\"Original: {query.query_str}\\n\")\n"
            "\n"
            "sub_queries = decomp.transform(query)\n"
            "\n"
            "print(f\"Decomposed into {len(sub_queries)} sub-queries:\")\n"
            "for i, sq in enumerate(sub_queries):\n"
            "    print(f\"  [{i}] {sq.query_str}\")"
        ),
        md(
            "## Realistic Decomposition Example\n"
            "A more realistic decompose function would produce meaningful sub-questions."
        ),
        code(
            "def realistic_decompose(query: str) -> list:\n"
            "    \"\"\"Simulate LLM decomposition of complex queries.\"\"\"\n"
            "    decompositions = {\n"
            "        \"Compare RAG and fine-tuning for production LLM apps\": [\n"
            "            \"What is retrieval-augmented generation (RAG)?\",\n"
            "            \"What is LLM fine-tuning?\",\n"
            "            \"What are the trade-offs between RAG and fine-tuning?\",\n"
            "            \"Which approach works better for production applications?\",\n"
            "        ],\n"
            "        \"How do vector databases handle scaling and consistency?\": [\n"
            "            \"How do vector databases handle horizontal scaling?\",\n"
            "            \"What consistency models do vector databases use?\",\n"
            "            \"What are the trade-offs between scaling and consistency?\",\n"
            "        ],\n"
            "    }\n"
            "    return decompositions.get(query, [f\"Sub: {query}\"])\n"
            "\n"
            "realistic_decomp = DecompositionTransformer(decompose_fn=realistic_decompose)\n"
            "\n"
            "complex_query = QueryBundle(\n"
            "    query_str=\"Compare RAG and fine-tuning for production LLM apps\"\n"
            ")\n"
            "subs = realistic_decomp.transform(complex_query)\n"
            "\n"
            "print(f\"Original: {complex_query.query_str}\\n\")\n"
            "print(f\"Sub-questions:\")\n"
            "for i, sq in enumerate(subs):\n"
            "    print(f\"  {i + 1}. {sq.query_str}\")"
        ),
        md(
            "## Decomposition Retrieval Pattern\n"
            "Each sub-question retrieves independently. Results are aggregated to\n"
            "build a comprehensive answer."
        ),
        code(
            "def mock_retrieve(query_str: str) -> list:\n"
            "    base = hash(query_str) % 100\n"
            "    return [f\"doc-{base + i}\" for i in range(2)]\n"
            "\n"
            "all_docs = []\n"
            "for sq in subs:\n"
            "    docs = mock_retrieve(sq.query_str)\n"
            "    all_docs.extend(docs)\n"
            "    print(f\"  Q: {sq.query_str[:50]}\")\n"
            "    print(f\"    -> {docs}\")\n"
            "\n"
            "unique = list(dict.fromkeys(all_docs))\n"
            "print(f\"\\nTotal: {len(all_docs)} docs, {len(unique)} unique\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `DecompositionTransformer` breaks complex queries into focused sub-questions\n"
            "- Each sub-question retrieves independently for targeted results\n"
            "- Results are aggregated and deduplicated across sub-questions\n"
            "- Best for multi-part or comparison queries\n"
            "- The decompose function can use an LLM or rule-based logic\n"
            "\n"
            "**Next:** [Step-Back Prompting](04_step_back.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Step-Back Prompting
# ---------------------------------------------------------------------------
def nb_step_back():
    write_notebook("04_step_back.ipynb", [
        md(
            "# Step-Back Prompting\n"
            "> Abstract a query to retrieve broader, principle-level context.\n"
            "\n"
            "`StepBackTransformer` generates a more general version of the query\n"
            "that retrieves foundational knowledge. The original and step-back\n"
            "queries retrieve together, combining specific and general context.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query import StepBackTransformer\n"
            "from anchor.models import QueryBundle"
        ),
        md(
            "## Define a Step-Back Function\n"
            "The function takes a specific query and returns a more general version.\n"
            "In production this would be an LLM call."
        ),
        code(
            "def mock_step_back(query: str) -> str:\n"
            "    \"\"\"Generate a broader, more abstract version of the query.\"\"\"\n"
            "    return f\"What are the general principles behind {query}?\"\n"
            "\n"
            "# Test the step-back function\n"
            "original = \"How does Anchor handle token budget overflow?\"\n"
            "stepped = mock_step_back(original)\n"
            "\n"
            "print(f\"Original:   {original}\")\n"
            "print(f\"Step-back:  {stepped}\")"
        ),
        md(
            "## Create the Step-Back Transformer\n"
            "Pass the generation function to `StepBackTransformer`."
        ),
        code(
            "step_back = StepBackTransformer(generate_fn=mock_step_back)\n"
            "\n"
            "print(f\"Transformer: {type(step_back).__name__}\")"
        ),
        md(
            "## Transform a Query\n"
            "`transform()` returns a list containing both the original and the\n"
            "abstracted query."
        ),
        code(
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "print(f\"Original: {query.query_str}\\n\")\n"
            "\n"
            "abstracted = step_back.transform(query)\n"
            "\n"
            "print(f\"Transformed to {len(abstracted)} queries:\")\n"
            "for i, q in enumerate(abstracted):\n"
            "    print(f\"  [{i}] {q.query_str}\")"
        ),
        md(
            "## Realistic Step-Back Examples\n"
            "Step-back prompting works best when the original query is very specific\n"
            "and would benefit from broader context."
        ),
        code(
            "def realistic_step_back(query: str) -> str:\n"
            "    \"\"\"Simulate LLM-generated step-back queries.\"\"\"\n"
            "    step_backs = {\n"
            "        \"Why does my FAISS index return wrong results for cosine similarity?\": (\n"
            "            \"How do vector similarity search algorithms work?\"\n"
            "        ),\n"
            "        \"How do I configure Anchor's sliding window memory with 4096 tokens?\": (\n"
            "            \"What are the different memory management strategies for LLM context?\"\n"
            "        ),\n"
            "        \"Why is my RAG pipeline returning irrelevant chunks?\": (\n"
            "            \"What factors affect retrieval quality in RAG systems?\"\n"
            "        ),\n"
            "    }\n"
            "    return step_backs.get(query, f\"What are the general principles behind {query}?\")\n"
            "\n"
            "realistic = StepBackTransformer(generate_fn=realistic_step_back)\n"
            "\n"
            "examples = [\n"
            "    \"Why does my FAISS index return wrong results for cosine similarity?\",\n"
            "    \"How do I configure Anchor's sliding window memory with 4096 tokens?\",\n"
            "    \"Why is my RAG pipeline returning irrelevant chunks?\",\n"
            "]\n"
            "\n"
            "for q_str in examples:\n"
            "    q = QueryBundle(query_str=q_str)\n"
            "    results = realistic.transform(q)\n"
            "    print(f\"Specific:  {q_str}\")\n"
            "    print(f\"Step-back: {results[-1].query_str}\")\n"
            "    print()"
        ),
        md(
            "## Dual Retrieval Pattern\n"
            "Use both the original and step-back queries for retrieval, then merge."
        ),
        code(
            "def mock_retrieve(query_str: str) -> list:\n"
            "    base = hash(query_str) % 100\n"
            "    return [f\"doc-{base + i}\" for i in range(3)]\n"
            "\n"
            "query = QueryBundle(query_str=\"Why is my RAG pipeline returning irrelevant chunks?\")\n"
            "results = realistic.transform(query)\n"
            "\n"
            "print(\"Dual retrieval:\\n\")\n"
            "all_docs = []\n"
            "for q in results:\n"
            "    docs = mock_retrieve(q.query_str)\n"
            "    all_docs.extend(docs)\n"
            "    label = \"Original\" if q == results[0] else \"Step-back\"\n"
            "    print(f\"  {label}: {q.query_str[:55]}\")\n"
            "    print(f\"    -> {docs}\")\n"
            "\n"
            "unique = list(dict.fromkeys(all_docs))\n"
            "print(f\"\\nTotal: {len(all_docs)} docs, {len(unique)} unique\")\n"
            "print(f\"Unique docs: {unique}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `StepBackTransformer` generates a broader version of specific queries\n"
            "- Both original and step-back queries retrieve together\n"
            "- Combines specific answers with foundational context\n"
            "- Best for highly specific or troubleshooting queries\n"
            "- The abstraction level depends on the generation function\n"
            "\n"
            "**Next:** [Query Classifiers](05_classifiers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Query Classifiers
# ---------------------------------------------------------------------------
def nb_classifiers():
    write_notebook("05_classifiers.ipynb", [
        md(
            "# Query Classifiers\n"
            "> Route queries to different retrieval strategies based on their type.\n"
            "\n"
            "Anchor provides three classifiers: `KeywordClassifier` (rule-based),\n"
            "`CallbackClassifier` (custom function), and `EmbeddingClassifier`\n"
            "(semantic similarity). Use them to choose the right retrieval path\n"
            "for each query.\n"
            "\n"
            "**Time:** ~8 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query.classifiers import (\n"
            "    KeywordClassifier,\n"
            "    CallbackClassifier,\n"
            "    EmbeddingClassifier,\n"
            ")"
        ),
        md(
            "## 1. KeywordClassifier\n"
            "Matches query text against keyword lists for each category. Simple,\n"
            "fast, and interpretable."
        ),
        code(
            "keyword = KeywordClassifier(\n"
            "    keywords={\n"
            "        \"technical\": [\"API\", \"code\", \"function\", \"error\", \"debug\"],\n"
            "        \"general\": [\"what\", \"how\", \"explain\", \"overview\", \"introduction\"],\n"
            "        \"tutorial\": [\"step-by-step\", \"guide\", \"tutorial\", \"example\"],\n"
            "    }\n"
            ")\n"
            "\n"
            "print(f\"Classifier: {type(keyword).__name__}\")\n"
            "print(f\"Categories: {list(keyword.keywords.keys())}\")"
        ),
        code(
            "# Classify several queries\n"
            "test_queries = [\n"
            "    \"How does the API work?\",\n"
            "    \"Explain the architecture overview\",\n"
            "    \"Step-by-step guide to building a pipeline\",\n"
            "    \"Debug this function error\",\n"
            "    \"What is Anchor?\",\n"
            "]\n"
            "\n"
            "print(f\"{'Query':<50} {'Category':>10}\")\n"
            "print(\"-\" * 62)\n"
            "for q in test_queries:\n"
            "    result = keyword.classify(q)\n"
            "    print(f\"{q:<50} {result:>10}\")"
        ),
        md(
            "## 2. CallbackClassifier\n"
            "Delegates classification to a user-defined function. Maximum\n"
            "flexibility for custom logic."
        ),
        code(
            "def custom_classify(query: str) -> str:\n"
            "    \"\"\"Classify based on query characteristics.\"\"\"\n"
            "    q = query.lower()\n"
            "    if any(kw in q for kw in [\"api\", \"code\", \"function\", \"class\"]):\n"
            "        return \"technical\"\n"
            "    elif any(kw in q for kw in [\"compare\", \"vs\", \"difference\"]):\n"
            "        return \"comparison\"\n"
            "    elif query.endswith(\"?\"):\n"
            "        return \"question\"\n"
            "    else:\n"
            "        return \"general\"\n"
            "\n"
            "callback = CallbackClassifier(callback_fn=custom_classify)\n"
            "\n"
            "print(f\"Classifier: {type(callback).__name__}\")"
        ),
        code(
            "test_queries = [\n"
            "    \"How does the API handle errors?\",\n"
            "    \"Compare RAG vs fine-tuning\",\n"
            "    \"Is Anchor production-ready?\",\n"
            "    \"Anchor context pipeline overview\",\n"
            "]\n"
            "\n"
            "print(f\"{'Query':<45} {'Category':>12}\")\n"
            "print(\"-\" * 59)\n"
            "for q in test_queries:\n"
            "    result = callback.classify(q)\n"
            "    print(f\"{q:<45} {result:>12}\")"
        ),
        md(
            "## 3. EmbeddingClassifier\n"
            "Uses embedding similarity to classify queries. Each category has\n"
            "example texts that define its embedding centroid."
        ),
        code(
            "# Mock embedding function (deterministic hash-based)\n"
            "def embed_fn(text: str) -> list:\n"
            "    \"\"\"Produce a fixed-length vector from text.\"\"\"\n"
            "    padded = text[:64].ljust(64)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "embedding = EmbeddingClassifier(\n"
            "    categories={\n"
            "        \"technical\": [\"API reference\", \"code example\", \"function signature\"],\n"
            "        \"general\": [\"overview\", \"introduction\", \"getting started\"],\n"
            "        \"troubleshooting\": [\"error message\", \"bug fix\", \"debugging\"],\n"
            "    },\n"
            "    embed_fn=embed_fn,\n"
            ")\n"
            "\n"
            "print(f\"Classifier: {type(embedding).__name__}\")\n"
            "print(f\"Categories: {list(embedding.categories.keys())}\")\n"
            "print(f\"Embedding dim: {len(embed_fn('test'))}\")"
        ),
        code(
            "test_queries = [\n"
            "    \"Show me the API reference\",\n"
            "    \"Getting started with Anchor\",\n"
            "    \"How to fix this error\",\n"
            "    \"Code example for retrieval\",\n"
            "]\n"
            "\n"
            "print(f\"{'Query':<40} {'Category':>16}\")\n"
            "print(\"-\" * 58)\n"
            "for q in test_queries:\n"
            "    result = embedding.classify(q)\n"
            "    print(f\"{q:<40} {result:>16}\")"
        ),
        md(
            "## Routing Pattern\n"
            "Use classification results to route queries to different retrieval\n"
            "strategies."
        ),
        code(
            "# Define retrieval strategies per category\n"
            "strategies = {\n"
            "    \"technical\": \"Use code-aware retriever with AST parsing\",\n"
            "    \"general\": \"Use standard dense retrieval with reranking\",\n"
            "    \"comparison\": \"Use multi-query expansion then merge results\",\n"
            "    \"question\": \"Use HyDE for hypothetical document generation\",\n"
            "    \"troubleshooting\": \"Use step-back prompting for broader context\",\n"
            "}\n"
            "\n"
            "queries = [\n"
            "    \"How does the API handle rate limiting?\",\n"
            "    \"Compare vector and keyword search\",\n"
            "    \"Is caching enabled by default?\",\n"
            "    \"Anchor framework overview\",\n"
            "]\n"
            "\n"
            "print(\"Query routing:\\n\")\n"
            "for q in queries:\n"
            "    category = callback.classify(q)\n"
            "    strategy = strategies.get(category, \"Default retrieval\")\n"
            "    print(f\"  Query:    {q}\")\n"
            "    print(f\"  Category: {category}\")\n"
            "    print(f\"  Strategy: {strategy}\")\n"
            "    print()"
        ),
        md(
            "## Classifier Comparison"),
        code(
            "# Compare all three classifiers on the same queries\n"
            "comparison_queries = [\n"
            "    \"How does the API work?\",\n"
            "    \"Getting started with Anchor\",\n"
            "    \"Debug this error in the pipeline\",\n"
            "]\n"
            "\n"
            "print(f\"{'Query':<38} {'Keyword':>12} {'Callback':>12} {'Embedding':>12}\")\n"
            "print(\"-\" * 76)\n"
            "for q in comparison_queries:\n"
            "    kw_result = keyword.classify(q)\n"
            "    cb_result = callback.classify(q)\n"
            "    em_result = embedding.classify(q)\n"
            "    print(f\"{q:<38} {kw_result:>12} {cb_result:>12} {em_result:>12}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- **KeywordClassifier**: rule-based, fast, good for well-defined categories\n"
            "- **CallbackClassifier**: maximum flexibility with custom Python logic\n"
            "- **EmbeddingClassifier**: semantic matching, adapts to query phrasing\n"
            "- Use classifiers to route queries to the best retrieval strategy\n"
            "- Classifiers can be composed: use keyword first, fall back to embedding\n"
            "\n"
            "**Next:** [Query Pipeline](06_query_pipeline.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Query Pipeline
# ---------------------------------------------------------------------------
def nb_query_pipeline():
    write_notebook("06_query_pipeline.ipynb", [
        md(
            "# Query Pipeline\n"
            "> Chain multiple query transformers into a sequential pipeline.\n"
            "\n"
            "`QueryTransformPipeline` runs a sequence of transformers in order.\n"
            "This notebook also covers `ContextualQueryTransformer` and\n"
            "`ConversationRewriter` for adding context and rewriting queries\n"
            "from conversation history.\n"
            "\n"
            "**Time:** ~8 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.query import (\n"
            "    QueryTransformPipeline,\n"
            "    ContextualQueryTransformer,\n"
            "    ConversationRewriter,\n"
            ")\n"
            "from anchor.models import QueryBundle"
        ),
        md(
            "## ContextualQueryTransformer\n"
            "Adds surrounding context (e.g., system prompt, user profile) to the query\n"
            "before retrieval."
        ),
        code(
            "context_fn = lambda q: QueryBundle(\n"
            "    query_str=f\"Given the context: {q.query_str}\"\n"
            ")\n"
            "\n"
            "contextual = ContextualQueryTransformer(context_fn=context_fn)\n"
            "\n"
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "result = contextual.transform(query)\n"
            "\n"
            "print(f\"Original:    {query.query_str}\")\n"
            "print(f\"Contextual:  {result[0].query_str}\")"
        ),
        md(
            "## ConversationRewriter\n"
            "Rewrites a follow-up query using conversation history to make it\n"
            "self-contained for retrieval."
        ),
        code(
            "rewrite_fn = lambda q: QueryBundle(\n"
            "    query_str=f\"Rewritten: {q.query_str}\"\n"
            ")\n"
            "\n"
            "rewriter = ConversationRewriter(rewrite_fn=rewrite_fn)\n"
            "\n"
            "follow_up = QueryBundle(query_str=\"How does it handle memory?\")\n"
            "result = rewriter.transform(follow_up)\n"
            "\n"
            "print(f\"Follow-up:   {follow_up.query_str}\")\n"
            "print(f\"Rewritten:   {result[0].query_str}\")"
        ),
        md(
            "## Realistic Conversation Rewriting\n"
            "In practice, the rewriter uses conversation history to resolve pronouns\n"
            "and implicit references."
        ),
        code(
            "# Simulate conversation-aware rewriting\n"
            "conversation_history = [\n"
            "    {\"role\": \"user\", \"content\": \"Tell me about Anchor's memory module\"},\n"
            "    {\"role\": \"assistant\", \"content\": \"The memory module manages conversation...\"},\n"
            "]\n"
            "\n"
            "def history_aware_rewrite(q: QueryBundle) -> QueryBundle:\n"
            "    \"\"\"Rewrite using conversation history for context.\"\"\"\n"
            "    last_topic = conversation_history[0][\"content\"]\n"
            "    return QueryBundle(\n"
            "        query_str=f\"Regarding {last_topic}: {q.query_str}\"\n"
            "    )\n"
            "\n"
            "smart_rewriter = ConversationRewriter(rewrite_fn=history_aware_rewrite)\n"
            "\n"
            "# \"it\" refers to the memory module from conversation history\n"
            "ambiguous = QueryBundle(query_str=\"How does it handle eviction?\")\n"
            "resolved = smart_rewriter.transform(ambiguous)\n"
            "\n"
            "print(f\"Ambiguous: {ambiguous.query_str}\")\n"
            "print(f\"Resolved:  {resolved[0].query_str}\")"
        ),
        md(
            "## Build a Query Pipeline\n"
            "`QueryTransformPipeline` chains transformers sequentially. The output\n"
            "of each transformer feeds into the next."
        ),
        code(
            "pipeline = QueryTransformPipeline(\n"
            "    transformers=[contextual, rewriter]\n"
            ")\n"
            "\n"
            "print(f\"Pipeline: {type(pipeline).__name__}\")\n"
            "print(f\"Stages:   {len(pipeline.transformers)}\")\n"
            "for i, t in enumerate(pipeline.transformers):\n"
            "    print(f\"  [{i}] {type(t).__name__}\")"
        ),
        md(
            "## Run the Pipeline\n"
            "`transform()` passes the query through each stage in order."
        ),
        code(
            "query = QueryBundle(query_str=\"What is context engineering?\")\n"
            "print(f\"Input: {query.query_str}\\n\")\n"
            "\n"
            "result = pipeline.transform(query)\n"
            "\n"
            "print(f\"Pipeline produced {len(result)} queries:\")\n"
            "for i, q in enumerate(result):\n"
            "    print(f\"  [{i}] {q.query_str}\")"
        ),
        md(
            "## Multi-Stage Pipeline Example\n"
            "Combine context injection, rewriting, and expansion in a single\n"
            "pipeline."
        ),
        code(
            "from anchor.query import HyDETransformer\n"
            "\n"
            "# Stage 1: Add context\n"
            "add_context = ContextualQueryTransformer(\n"
            "    context_fn=lambda q: QueryBundle(\n"
            "        query_str=f\"[User: developer] {q.query_str}\"\n"
            "    )\n"
            ")\n"
            "\n"
            "# Stage 2: Rewrite for clarity\n"
            "clarify = ConversationRewriter(\n"
            "    rewrite_fn=lambda q: QueryBundle(\n"
            "        query_str=q.query_str.replace(\"it\", \"Anchor framework\")\n"
            "    )\n"
            ")\n"
            "\n"
            "# Stage 3: HyDE expansion\n"
            "hyde = HyDETransformer(\n"
            "    generate_fn=lambda q: f\"A document about {q} would cover...\"\n"
            ")\n"
            "\n"
            "full_pipeline = QueryTransformPipeline(\n"
            "    transformers=[add_context, clarify, hyde]\n"
            ")\n"
            "\n"
            "print(f\"Full pipeline stages:\")\n"
            "for i, t in enumerate(full_pipeline.transformers):\n"
            "    print(f\"  [{i}] {type(t).__name__}\")"
        ),
        code(
            "# Run the full pipeline\n"
            "query = QueryBundle(query_str=\"How does it handle memory?\")\n"
            "print(f\"Input: {query.query_str}\\n\")\n"
            "\n"
            "results = full_pipeline.transform(query)\n"
            "\n"
            "print(f\"Pipeline output ({len(results)} queries):\")\n"
            "for i, q in enumerate(results):\n"
            "    label = q.query_str[:70]\n"
            "    suffix = \"...\" if len(q.query_str) > 70 else \"\"\n"
            "    print(f\"  [{i}] {label}{suffix}\")"
        ),
        md(
            "## Pipeline Inspection\n"
            "Track how the query transforms through each stage."
        ),
        code(
            "# Step through the pipeline manually to see each transformation\n"
            "query = QueryBundle(query_str=\"How does it handle memory?\")\n"
            "print(f\"Stage 0 (input): {query.query_str}\\n\")\n"
            "\n"
            "current = [query]\n"
            "for i, transformer in enumerate(full_pipeline.transformers):\n"
            "    stage_name = type(transformer).__name__\n"
            "    # Apply transformer to first query in current list\n"
            "    current = transformer.transform(current[0])\n"
            "    print(f\"Stage {i + 1} ({stage_name}):\")\n"
            "    for q in current:\n"
            "        print(f\"  -> {q.query_str[:70]}\")\n"
            "    print()"
        ),
        md(
            "## Key Takeaways\n"
            "- `QueryTransformPipeline` chains transformers in sequence\n"
            "- `ContextualQueryTransformer` injects external context into queries\n"
            "- `ConversationRewriter` resolves ambiguity from conversation history\n"
            "- Pipelines enable composable query processing workflows\n"
            "- Step through the pipeline manually to debug transformations\n"
            "- Combine context, rewriting, and expansion for production-grade query handling\n"
            "\n"
            "**Back to:** [Query README](README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Query Transformation notebooks in {OUTPUT_DIR}/\n")

    nb_hyde_transformer()
    nb_multi_query()
    nb_decomposition()
    nb_step_back()
    nb_classifiers()
    nb_query_pipeline()

    print(f"\nDone. 6 notebooks created.")


if __name__ == "__main__":
    main()
