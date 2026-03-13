"""Generate all 5 Jupyter notebooks for the Ingestion module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "04-ingestion"))
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
# 01 - Document Ingester
# ---------------------------------------------------------------------------
def nb_document_ingester():
    write_notebook("01_document_ingester.ipynb", [
        md(
            "# Document Ingester\n"
            "> Chunk and ingest documents into context items for the Anchor pipeline.\n"
            "\n"
            "`DocumentIngester` takes raw text, runs it through a chunker, and produces\n"
            "a list of `ContextItem` objects annotated with source type, priority, and\n"
            "user-supplied metadata.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.ingestion import DocumentIngester\n"
            "from anchor.ingestion.chunkers import RecursiveCharacterChunker\n"
            "from anchor.models import SourceType"
        ),
        md(
            "## Configure the Chunker\n"
            "`RecursiveCharacterChunker` splits text using a hierarchy of separators\n"
            "(paragraphs, sentences, words) to produce chunks that respect natural\n"
            "boundaries."
        ),
        code(
            "chunker = RecursiveCharacterChunker(chunk_size=256, overlap=50)\n"
            "\n"
            "print(f\"Chunk size: {chunker.chunk_size}\")\n"
            "print(f\"Overlap:    {chunker.overlap}\")"
        ),
        md(
            "## Create the Ingester\n"
            "`source_type` tags every item produced by this ingester. `priority` controls\n"
            "where items land in the context window (lower = closer to the system prompt)."
        ),
        code(
            "ingester = DocumentIngester(\n"
            "    chunker=chunker,\n"
            "    source_type=SourceType.RETRIEVAL,\n"
            "    priority=5,\n"
            ")\n"
            "\n"
            "print(f\"Source type: {ingester.source_type}\")\n"
            "print(f\"Priority:    {ingester.priority}\")"
        ),
        md(
            "## Prepare a Sample Document\n"
            "We use a multi-paragraph mock document so the chunker has enough text\n"
            "to produce several chunks."
        ),
        code(
            "text = (\n"
            "    \"Anchor is a framework for building context-aware AI applications. \"\n"
            "    \"It provides tools for memory management, retrieval-augmented generation, \"\n"
            "    \"and intelligent context assembly. The framework is designed to be modular, \"\n"
            "    \"allowing developers to pick and choose the components they need.\\n\\n\"\n"
            "    \"The ingestion module handles converting raw documents into structured \"\n"
            "    \"context items. It supports multiple chunking strategies, each optimized \"\n"
            "    \"for different content types. Text is split into overlapping chunks to \"\n"
            "    \"preserve context at boundaries.\\n\\n\"\n"
            "    \"Once ingested, context items flow through the pipeline where they are \"\n"
            "    \"scored, filtered, and assembled into the final prompt. Priority values \"\n"
            "    \"determine ordering, while token budgets control how much context fits \"\n"
            "    \"into the model's context window.\\n\\n\"\n"
            "    \"Anchor supports various source types including retrieval results, \"\n"
            "    \"conversation history, tool outputs, and system instructions. Each source \"\n"
            "    \"type can be configured with different priority levels and token budgets \"\n"
            "    \"to ensure the most relevant information reaches the model.\"\n"
            ")\n"
            "\n"
            "print(f\"Document length: {len(text)} characters\")"
        ),
        md(
            "## Ingest the Document\n"
            "`ingest_text()` chunks the text and wraps each chunk in a `ContextItem`.\n"
            "Pass optional `metadata` to attach to every item."
        ),
        code(
            "items = ingester.ingest_text(text, metadata={\"source\": \"readme.md\"})\n"
            "\n"
            "print(f\"Generated {len(items)} context items\\n\")\n"
            "for i, item in enumerate(items):\n"
            "    content_preview = str(item.content)[:70]\n"
            "    print(f\"  [{i}] priority={item.priority}  \"\n"
            "          f\"type={item.source_type.value}\")\n"
            "    print(f\"      {content_preview}...\")\n"
            "    print()"
        ),
        md(
            "## Inspect Metadata\n"
            "Each context item carries the metadata you passed to `ingest_text()`."
        ),
        code(
            "first_item = items[0]\n"
            "\n"
            "print(f\"Content length: {len(str(first_item.content))} chars\")\n"
            "print(f\"Source type:    {first_item.source_type}\")\n"
            "print(f\"Priority:       {first_item.priority}\")\n"
            "print(f\"Metadata:       {first_item.metadata}\")"
        ),
        md(
            "## Ingest Multiple Documents\n"
            "Call `ingest_text()` for each document. Metadata lets you track provenance."
        ),
        code(
            "docs = [\n"
            "    (\"First document about Python basics and syntax.\", {\"source\": \"python.md\"}),\n"
            "    (\"Second document covering JavaScript and web APIs.\", {\"source\": \"js.md\"}),\n"
            "    (\"Third document on database design and SQL patterns.\", {\"source\": \"sql.md\"}),\n"
            "]\n"
            "\n"
            "all_items = []\n"
            "for content, meta in docs:\n"
            "    batch = ingester.ingest_text(content, metadata=meta)\n"
            "    all_items.extend(batch)\n"
            "    print(f\"  {meta['source']}: {len(batch)} items\")\n"
            "\n"
            "print(f\"\\nTotal items across all documents: {len(all_items)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `DocumentIngester` pairs a chunker with source-type and priority settings\n"
            "- `ingest_text()` returns a list of `ContextItem` objects ready for the pipeline\n"
            "- Metadata flows through to every item for provenance tracking\n"
            "- Swap chunkers to change how documents are split without changing ingestion logic\n"
            "\n"
            "**Next:** [Chunking Strategies](02_chunking_strategies.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Chunking Strategies
# ---------------------------------------------------------------------------
def nb_chunking_strategies():
    write_notebook("02_chunking_strategies.ipynb", [
        md(
            "# Chunking Strategies\n"
            "> Compare six built-in chunkers and learn when to use each one.\n"
            "\n"
            "Anchor ships with chunkers optimized for different content types: plain text,\n"
            "semantic boundaries, sentences, source code, and tabular data.\n"
            "\n"
            "**Time:** ~10 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.ingestion.chunkers import (\n"
            "    FixedSizeChunker,\n"
            "    RecursiveCharacterChunker,\n"
            "    SemanticChunker,\n"
            "    SentenceChunker,\n"
            "    CodeChunker,\n"
            "    TableAwareChunker,\n"
            ")"
        ),
        md(
            "## Sample Texts\n"
            "We prepare different content types to demonstrate each chunker's strengths."
        ),
        code(
            "plain_text = (\n"
            "    \"Anchor is a framework for building context-aware AI applications. \"\n"
            "    \"It provides tools for memory management, retrieval-augmented generation, \"\n"
            "    \"and intelligent context assembly. The framework is designed to be modular \"\n"
            "    \"and extensible. Developers can pick and choose the components they need. \"\n"
            "    \"The ingestion module handles converting raw documents into structured \"\n"
            "    \"context items. It supports multiple chunking strategies optimized for \"\n"
            "    \"different content types. Text is split into overlapping chunks to preserve \"\n"
            "    \"context at boundaries. Once ingested, context items flow through the \"\n"
            "    \"pipeline where they are scored, filtered, and assembled into the prompt.\"\n"
            ")\n"
            "\n"
            "code_text = '''def fibonacci(n):\n"
            "    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
            "\n"
            "def factorial(n):\n"
            "    \"\"\"Return n factorial.\"\"\"\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
            "\n"
            "def is_prime(n):\n"
            "    \"\"\"Check if n is a prime number.\"\"\"\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
            "'''\n"
            "\n"
            "table_text = (\n"
            "    \"| Language | Year | Typing   |\\n\"\n"
            "    \"|----------|------|----------|\\n\"\n"
            "    \"| Python   | 1991 | Dynamic  |\\n\"\n"
            "    \"| Java     | 1995 | Static   |\\n\"\n"
            "    \"| Rust     | 2010 | Static   |\\n\"\n"
            "    \"| Go       | 2009 | Static   |\\n\"\n"
            "    \"\\n\"\n"
            "    \"These languages represent different design philosophies. \"\n"
            "    \"Python favors readability. Java emphasizes enterprise patterns. \"\n"
            "    \"Rust prioritizes memory safety. Go targets simplicity and concurrency.\"\n"
            ")\n"
            "\n"
            "print(f\"Plain text: {len(plain_text)} chars\")\n"
            "print(f\"Code text:  {len(code_text)} chars\")\n"
            "print(f\"Table text: {len(table_text)} chars\")"
        ),
        md(
            "## 1. FixedSizeChunker\n"
            "Splits text into chunks of exactly `chunk_size` characters with a sliding\n"
            "overlap. Simple and predictable."
        ),
        code(
            "fixed = FixedSizeChunker(chunk_size=200, overlap=50)\n"
            "chunks = fixed.chunk(plain_text)\n"
            "\n"
            "print(f\"FixedSizeChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars): {chunk[:60]}...\")"
        ),
        md(
            "## 2. RecursiveCharacterChunker\n"
            "Tries to split on paragraph breaks first, then sentences, then words.\n"
            "Produces more natural chunk boundaries than fixed-size."
        ),
        code(
            "recursive = RecursiveCharacterChunker(chunk_size=200, overlap=50)\n"
            "chunks = recursive.chunk(plain_text)\n"
            "\n"
            "print(f\"RecursiveCharacterChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars): {chunk[:60]}...\")"
        ),
        md(
            "## 3. SemanticChunker\n"
            "Groups sentences whose embeddings are above a similarity threshold.\n"
            "Requires an embedding function."
        ),
        code(
            "# Mock embedding function for demonstration\n"
            "embed_fn = lambda text: [hash(c) % 100 / 100.0 for c in text[:64].ljust(64)]\n"
            "\n"
            "semantic = SemanticChunker(\n"
            "    embed_fn=embed_fn,\n"
            "    threshold=0.5,\n"
            "    chunk_size=200,\n"
            ")\n"
            "chunks = semantic.chunk(plain_text)\n"
            "\n"
            "print(f\"SemanticChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars): {chunk[:60]}...\")"
        ),
        md(
            "## 4. SentenceChunker\n"
            "Splits on sentence boundaries. The `overlap` parameter controls how many\n"
            "trailing sentences from the previous chunk are prepended to the next."
        ),
        code(
            "sentence = SentenceChunker(chunk_size=200, overlap=1)\n"
            "chunks = sentence.chunk(plain_text)\n"
            "\n"
            "print(f\"SentenceChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars): {chunk[:60]}...\")"
        ),
        md(
            "## 5. CodeChunker\n"
            "Splits source code at function/class boundaries. Keeps logical units\n"
            "together."
        ),
        code(
            "code_chunker = CodeChunker(chunk_size=200, overlap=50)\n"
            "chunks = code_chunker.chunk(code_text)\n"
            "\n"
            "print(f\"CodeChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    lines = chunk.strip().split('\\n')\n"
            "    print(f\"  [{i}] ({len(chunk)} chars, {len(lines)} lines):\")\n"
            "    for line in lines[:3]:\n"
            "        print(f\"      {line}\")\n"
            "    if len(lines) > 3:\n"
            "        print(f\"      ... (+{len(lines) - 3} more lines)\")\n"
            "    print()"
        ),
        md(
            "## 6. TableAwareChunker\n"
            "Detects tables (Markdown or delimited) and keeps them intact rather\n"
            "than splitting mid-row."
        ),
        code(
            "table_chunker = TableAwareChunker(chunk_size=200, overlap=100)\n"
            "chunks = table_chunker.chunk(table_text)\n"
            "\n"
            "print(f\"TableAwareChunker: {len(chunks)} chunks\\n\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars):\")\n"
            "    for line in chunk.strip().split('\\n')[:5]:\n"
            "        print(f\"      {line}\")\n"
            "    print()"
        ),
        md("## Side-by-Side Comparison on Plain Text"),
        code(
            "chunkers = {\n"
            "    \"FixedSize\":     FixedSizeChunker(chunk_size=200, overlap=50),\n"
            "    \"Recursive\":     RecursiveCharacterChunker(chunk_size=200, overlap=50),\n"
            "    \"Semantic\":      SemanticChunker(embed_fn=embed_fn, threshold=0.5, chunk_size=200),\n"
            "    \"Sentence\":      SentenceChunker(chunk_size=200, overlap=1),\n"
            "    \"Code\":          CodeChunker(chunk_size=200, overlap=50),\n"
            "    \"TableAware\":    TableAwareChunker(chunk_size=200, overlap=100),\n"
            "}\n"
            "\n"
            "print(f\"{'Chunker':<16} {'Chunks':>6}  {'Avg Size':>8}\")\n"
            "print(\"-\" * 34)\n"
            "for name, chunker in chunkers.items():\n"
            "    chunks = chunker.chunk(plain_text)\n"
            "    avg = sum(len(c) for c in chunks) / len(chunks) if chunks else 0\n"
            "    print(f\"{name:<16} {len(chunks):>6}  {avg:>8.0f}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- **FixedSizeChunker**: predictable sizing, may break mid-word\n"
            "- **RecursiveCharacterChunker**: best general-purpose choice for prose\n"
            "- **SemanticChunker**: groups semantically related sentences (needs embeddings)\n"
            "- **SentenceChunker**: respects sentence boundaries, good for Q&A\n"
            "- **CodeChunker**: preserves function/class boundaries in source code\n"
            "- **TableAwareChunker**: keeps tables intact, ideal for structured data\n"
            "- All chunkers share the same `.chunk(text) -> list[str]` interface\n"
            "\n"
            "**Next:** [Parsers](03_parsers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Parsers
# ---------------------------------------------------------------------------
def nb_parsers():
    write_notebook("03_parsers.ipynb", [
        md(
            "# Parsers\n"
            "> Convert raw file bytes into plain text for chunking and ingestion.\n"
            "\n"
            "Anchor parsers implement a simple interface: `parse(content: bytes) -> str`.\n"
            "Each parser declares which file extensions it supports via\n"
            "`supported_extensions`.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.ingestion.parsers import PlainTextParser, MarkdownParser, HTMLParser"
        ),
        md(
            "## PlainTextParser\n"
            "The simplest parser -- decodes bytes to a UTF-8 string with optional\n"
            "whitespace normalization."
        ),
        code(
            "text_parser = PlainTextParser()\n"
            "\n"
            "print(f\"Supported extensions: {text_parser.supported_extensions}\")\n"
            "\n"
            "raw = b\"Hello, world!\\nThis is a plain text document.\\n\"\n"
            "result = text_parser.parse(raw)\n"
            "\n"
            "print(f\"\\nInput bytes: {len(raw)} bytes\")\n"
            "print(f\"Output text: {repr(result)}\")"
        ),
        md(
            "## MarkdownParser\n"
            "Strips Markdown formatting (headings, bold, links, etc.) and returns\n"
            "clean text suitable for chunking."
        ),
        code(
            "md_parser = MarkdownParser()\n"
            "\n"
            "print(f\"Supported extensions: {md_parser.supported_extensions}\")\n"
            "\n"
            "markdown_content = b\"\"\"# Project README\n"
            "\n"
            "## Overview\n"
            "\n"
            "This is a **sample** project that demonstrates the `Anchor` framework.\n"
            "\n"
            "### Features\n"
            "\n"
            "- Context-aware AI pipelines\n"
            "- Modular [architecture](https://example.com)\n"
            "- Built-in memory management\n"
            "\n"
            "## Installation\n"
            "\n"
            "```bash\n"
            "pip install anchor\n"
            "```\n"
            "\n"
            "See the [docs](https://docs.example.com) for more details.\n"
            "\"\"\"\n"
            "\n"
            "result = md_parser.parse(markdown_content)\n"
            "\n"
            "print(f\"\\nInput bytes:  {len(markdown_content)} bytes\")\n"
            "print(f\"Output chars: {len(result)} chars\")\n"
            "print(f\"\\nParsed output:\")\n"
            "print(result)"
        ),
        md(
            "## HTMLParser\n"
            "Extracts visible text from HTML documents, stripping tags, scripts,\n"
            "and style blocks."
        ),
        code(
            "html_parser = HTMLParser()\n"
            "\n"
            "print(f\"Supported extensions: {html_parser.supported_extensions}\")\n"
            "\n"
            "html_content = b\"\"\"<!DOCTYPE html>\n"
            "<html>\n"
            "<head><title>Sample Page</title></head>\n"
            "<body>\n"
            "  <h1>Welcome to Anchor</h1>\n"
            "  <p>Anchor is a <strong>context-aware</strong> AI framework.</p>\n"
            "  <ul>\n"
            "    <li>Memory management</li>\n"
            "    <li>Retrieval-augmented generation</li>\n"
            "    <li>Intelligent context assembly</li>\n"
            "  </ul>\n"
            "  <script>console.log('ignored');</script>\n"
            "  <footer><p>Copyright 2024</p></footer>\n"
            "</body>\n"
            "</html>\n"
            "\"\"\"\n"
            "\n"
            "result = html_parser.parse(html_content)\n"
            "\n"
            "print(f\"\\nInput bytes:  {len(html_content)} bytes\")\n"
            "print(f\"Output chars: {len(result)} chars\")\n"
            "print(f\"\\nParsed output:\")\n"
            "print(result)"
        ),
        md(
            "## Parser Selection Pattern\n"
            "Match a file extension to the right parser at runtime."
        ),
        code(
            "# Build a registry of parsers\n"
            "parsers = [PlainTextParser(), MarkdownParser(), HTMLParser()]\n"
            "\n"
            "registry = {}\n"
            "for parser in parsers:\n"
            "    for ext in parser.supported_extensions:\n"
            "        registry[ext] = parser\n"
            "\n"
            "print(\"Parser registry:\")\n"
            "for ext, parser in sorted(registry.items()):\n"
            "    print(f\"  {ext:<6} -> {type(parser).__name__}\")"
        ),
        code(
            "# Lookup by extension\n"
            "def parse_file(filename: str, content: bytes) -> str:\n"
            "    ext = '.' + filename.rsplit('.', 1)[-1] if '.' in filename else ''\n"
            "    parser = registry.get(ext)\n"
            "    if parser is None:\n"
            "        raise ValueError(f\"No parser for extension: {ext}\")\n"
            "    return parser.parse(content)\n"
            "\n"
            "# Test with different file types\n"
            "test_files = [\n"
            "    (\"notes.txt\", b\"Plain text notes.\"),\n"
            "    (\"readme.md\", b\"# Heading\\nSome **bold** text.\"),\n"
            "    (\"index.html\", b\"<p>Hello <em>world</em></p>\"),\n"
            "]\n"
            "\n"
            "for filename, content in test_files:\n"
            "    parsed = parse_file(filename, content)\n"
            "    print(f\"  {filename:<14} -> {repr(parsed)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- All parsers implement `parse(content: bytes) -> str`\n"
            "- `supported_extensions` declares which file types a parser handles\n"
            "- **PlainTextParser**: UTF-8 decode with whitespace cleanup\n"
            "- **MarkdownParser**: strips formatting, returns clean text\n"
            "- **HTMLParser**: extracts visible text, ignores scripts/styles\n"
            "- Build a registry dict to dispatch by file extension at runtime\n"
            "\n"
            "**Next:** [Metadata Enrichment](04_metadata_enrichment.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Metadata Enrichment
# ---------------------------------------------------------------------------
def nb_metadata_enrichment():
    write_notebook("04_metadata_enrichment.ipynb", [
        md(
            "# Metadata Enrichment\n"
            "> Attach document IDs, chunk IDs, and custom metadata to ingested chunks.\n"
            "\n"
            "Anchor provides ID generation utilities and a `MetadataEnricher` that lets\n"
            "you inject arbitrary metadata into chunks during ingestion.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.ingestion import DocumentIngester, MetadataEnricher\n"
            "from anchor.ingestion.metadata import (\n"
            "    generate_doc_id,\n"
            "    generate_chunk_id,\n"
            "    extract_chunk_metadata,\n"
            ")\n"
            "from anchor.ingestion.chunkers import RecursiveCharacterChunker\n"
            "from anchor.models import SourceType"
        ),
        md(
            "## Document & Chunk ID Generation\n"
            "`generate_doc_id` creates a content-addressable hash for a document.\n"
            "`generate_chunk_id` extends it with a chunk index for uniqueness."
        ),
        code(
            "content = \"Anchor is a framework for building context-aware AI applications.\"\n"
            "\n"
            "doc_id = generate_doc_id(content)\n"
            "print(f\"Document ID: {doc_id}\")\n"
            "\n"
            "# Generate IDs for individual chunks\n"
            "for i in range(3):\n"
            "    chunk_id = generate_chunk_id(doc_id, index=i)\n"
            "    print(f\"  Chunk {i} ID: {chunk_id}\")"
        ),
        md(
            "## Content-Addressable Properties\n"
            "The same content always produces the same document ID, making deduplication\n"
            "straightforward."
        ),
        code(
            "# Same content -> same ID\n"
            "id_a = generate_doc_id(\"Hello, world!\")\n"
            "id_b = generate_doc_id(\"Hello, world!\")\n"
            "id_c = generate_doc_id(\"Hello, world?\")\n"
            "\n"
            "print(f\"Same content:  {id_a == id_b}\")  # True\n"
            "print(f\"Diff content:  {id_a == id_c}\")  # False\n"
            "print(f\"\\nID A: {id_a}\")\n"
            "print(f\"ID B: {id_b}\")\n"
            "print(f\"ID C: {id_c}\")"
        ),
        md(
            "## Extract Chunk Metadata\n"
            "`extract_chunk_metadata` computes basic statistics about a chunk (word count,\n"
            "character count, etc.)."
        ),
        code(
            "chunk_text = (\n"
            "    \"Anchor provides tools for memory management, retrieval-augmented \"\n"
            "    \"generation, and intelligent context assembly.\"\n"
            ")\n"
            "\n"
            "meta = extract_chunk_metadata(chunk_text)\n"
            "\n"
            "print(\"Chunk metadata:\")\n"
            "for key, value in meta.items():\n"
            "    print(f\"  {key}: {value}\")"
        ),
        md(
            "## Custom Enrichment Function\n"
            "A `MetadataEnricher` wraps a user-defined function that receives chunks\n"
            "and document-level metadata, and returns enriched metadata dicts."
        ),
        code(
            "def enrich(chunks, doc_metadata):\n"
            "    \"\"\"Add word count and merge with document metadata.\"\"\"\n"
            "    return [\n"
            "        {\n"
            "            \"chunk\": c,\n"
            "            \"word_count\": len(c.split()),\n"
            "            **doc_metadata,\n"
            "        }\n"
            "        for c in chunks\n"
            "    ]\n"
            "\n"
            "enricher = MetadataEnricher(enrichment_fn=enrich)\n"
            "print(f\"Enricher ready: {type(enricher).__name__}\")"
        ),
        md("## Apply Enrichment to Chunks"),
        code(
            "# Chunk a sample document\n"
            "chunker = RecursiveCharacterChunker(chunk_size=120, overlap=20)\n"
            "sample_text = (\n"
            "    \"Anchor is a modular framework. It handles memory, retrieval, \"\n"
            "    \"and context assembly. Developers can pick components they need. \"\n"
            "    \"The ingestion module converts raw documents into context items.\"\n"
            ")\n"
            "chunks = chunker.chunk(sample_text)\n"
            "\n"
            "# Enrich with document-level metadata\n"
            "doc_meta = {\"source\": \"overview.md\", \"version\": \"1.0\"}\n"
            "enriched = enricher.enrichment_fn(chunks, doc_meta)\n"
            "\n"
            "print(f\"Enriched {len(enriched)} chunks:\\n\")\n"
            "for i, item in enumerate(enriched):\n"
            "    print(f\"  [{i}] word_count={item['word_count']}  \"\n"
            "          f\"source={item['source']}  version={item['version']}\")\n"
            "    print(f\"      {item['chunk'][:60]}...\")\n"
            "    print()"
        ),
        md(
            "## Full Pipeline: IDs + Enrichment\n"
            "Combine ID generation with enrichment for a complete metadata workflow."
        ),
        code(
            "full_text = (\n"
            "    \"The context pipeline scores and filters items. Priority values \"\n"
            "    \"determine ordering. Token budgets control how much context fits. \"\n"
            "    \"Anchor supports retrieval results, conversation history, and tools.\"\n"
            ")\n"
            "\n"
            "doc_id = generate_doc_id(full_text)\n"
            "chunks = chunker.chunk(full_text)\n"
            "\n"
            "print(f\"Document ID: {doc_id}\")\n"
            "print(f\"Chunks: {len(chunks)}\\n\")\n"
            "\n"
            "for i, chunk in enumerate(chunks):\n"
            "    chunk_id = generate_chunk_id(doc_id, index=i)\n"
            "    meta = extract_chunk_metadata(chunk)\n"
            "    print(f\"  Chunk {i}: id={chunk_id[:16]}...\")\n"
            "    for k, v in meta.items():\n"
            "        print(f\"    {k}: {v}\")\n"
            "    print()"
        ),
        md(
            "## Key Takeaways\n"
            "- `generate_doc_id()` creates deterministic, content-addressable document IDs\n"
            "- `generate_chunk_id()` extends the doc ID with a chunk index\n"
            "- `extract_chunk_metadata()` computes word count and other chunk statistics\n"
            "- `MetadataEnricher` wraps a custom function to inject metadata during ingestion\n"
            "- Combine IDs and enrichment for a complete provenance-tracking workflow\n"
            "\n"
            "**Next:** [Parent-Child Chunks](05_parent_child_chunks.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Parent-Child Chunks
# ---------------------------------------------------------------------------
def nb_parent_child_chunks():
    write_notebook("05_parent_child_chunks.ipynb", [
        md(
            "# Parent-Child Chunks\n"
            "> Split documents into large parent chunks and small child chunks for\n"
            "> fine-grained retrieval with full-context expansion.\n"
            "\n"
            "`ParentChildChunker` creates a two-level hierarchy: large \"parent\" chunks\n"
            "for context, and small \"child\" chunks for precise matching. At retrieval\n"
            "time, `ParentExpander` maps matched children back to their parent.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.ingestion.chunkers import ParentChildChunker\n"
            "from anchor.ingestion import ParentExpander"
        ),
        md(
            "## Prepare a Document\n"
            "We use a multi-section document to see how parent and child chunks align."
        ),
        code(
            "document = (\n"
            "    \"Anchor is a framework for building context-aware AI applications. \"\n"
            "    \"It provides tools for memory management, retrieval-augmented generation, \"\n"
            "    \"and intelligent context assembly. The framework is designed to be modular, \"\n"
            "    \"allowing developers to pick and choose the components they need. \"\n"
            "    \"Each component can be used independently or combined into full pipelines.\\n\\n\"\n"
            "    \"The ingestion module handles converting raw documents into structured \"\n"
            "    \"context items. It supports multiple chunking strategies, each optimized \"\n"
            "    \"for different content types. Text is split into overlapping chunks to \"\n"
            "    \"preserve context at chunk boundaries. Parsers convert raw bytes into text \"\n"
            "    \"before chunking begins.\\n\\n\"\n"
            "    \"The retrieval module searches a vector store for relevant context items. \"\n"
            "    \"It supports dense retrieval, sparse retrieval, and hybrid approaches. \"\n"
            "    \"Results are scored and ranked before being added to the context window. \"\n"
            "    \"Rerankers can further refine the ordering of retrieved items.\\n\\n\"\n"
            "    \"The memory module manages conversation history and long-term knowledge. \"\n"
            "    \"It supports sliding windows, summary buffers, and graph-based memory. \"\n"
            "    \"Eviction policies control which turns are dropped when the token budget \"\n"
            "    \"is exceeded. Decay strategies model how memories fade over time.\"\n"
            ")\n"
            "\n"
            "print(f\"Document length: {len(document)} characters\")"
        ),
        md(
            "## Create the Parent-Child Chunker\n"
            "Parent chunks are large (500 chars), child chunks are small (100 chars).\n"
            "The overlap ensures no information is lost at boundaries."
        ),
        code(
            "chunker = ParentChildChunker(\n"
            "    parent_chunk_size=500,\n"
            "    child_chunk_size=100,\n"
            "    overlap=20,\n"
            ")\n"
            "\n"
            "print(f\"Parent chunk size: {chunker.parent_chunk_size}\")\n"
            "print(f\"Child chunk size:  {chunker.child_chunk_size}\")\n"
            "print(f\"Overlap:           {chunker.overlap}\")"
        ),
        md(
            "## Chunk the Document\n"
            "The chunker returns a structure containing both parent and child chunks\n"
            "with their mapping."
        ),
        code(
            "chunks = chunker.chunk(document)\n"
            "\n"
            "print(f\"Total chunk objects returned: {len(chunks)}\")\n"
            "print(f\"\\nChunk details:\")\n"
            "for i, chunk in enumerate(chunks):\n"
            "    print(f\"  [{i}] ({len(chunk)} chars): {chunk[:50]}...\")"
        ),
        md(
            "## Examine the Hierarchy\n"
            "Let's manually create parent and child chunks to understand the hierarchy."
        ),
        code(
            "# Simulate parent/child relationship\n"
            "parent_size = 500\n"
            "child_size = 100\n"
            "\n"
            "# Create parent chunks\n"
            "parents = []\n"
            "for start in range(0, len(document), parent_size):\n"
            "    parents.append(document[start:start + parent_size])\n"
            "\n"
            "print(f\"Parent chunks: {len(parents)}\")\n"
            "for i, parent in enumerate(parents):\n"
            "    print(f\"  Parent [{i}] ({len(parent)} chars): {parent[:50]}...\")\n"
            "\n"
            "print()\n"
            "\n"
            "# Create child chunks from first parent\n"
            "first_parent = parents[0]\n"
            "children = []\n"
            "for start in range(0, len(first_parent), child_size - 20):\n"
            "    children.append(first_parent[start:start + child_size])\n"
            "\n"
            "print(f\"Children of Parent [0]: {len(children)}\")\n"
            "for i, child in enumerate(children):\n"
            "    print(f\"  Child [{i}] ({len(child)} chars): {child[:40]}...\")"
        ),
        md(
            "## Parent Expander Concept\n"
            "`ParentExpander` maps child chunk matches back to their parent chunk\n"
            "for full-context retrieval. In production, you provide a child retriever\n"
            "and a parent store."
        ),
        code(
            "# Simulate the parent expansion workflow\n"
            "\n"
            "# 1. Build a mock parent store (parent_id -> parent_text)\n"
            "parent_store = {}\n"
            "child_to_parent = {}\n"
            "\n"
            "for p_idx, parent in enumerate(parents):\n"
            "    parent_id = f\"parent-{p_idx}\"\n"
            "    parent_store[parent_id] = parent\n"
            "    # Create children for this parent\n"
            "    for c_idx in range(0, len(parent), child_size - 20):\n"
            "        child_text = parent[c_idx:c_idx + child_size]\n"
            "        child_id = f\"child-{p_idx}-{c_idx}\"\n"
            "        child_to_parent[child_id] = parent_id\n"
            "\n"
            "print(f\"Parent store: {len(parent_store)} parents\")\n"
            "print(f\"Child-to-parent map: {len(child_to_parent)} children\")"
        ),
        code(
            "# 2. Simulate a child match and expand to parent\n"
            "matched_child_id = list(child_to_parent.keys())[3]  # pick one\n"
            "matched_parent_id = child_to_parent[matched_child_id]\n"
            "expanded_text = parent_store[matched_parent_id]\n"
            "\n"
            "print(f\"Matched child:  {matched_child_id}\")\n"
            "print(f\"Parent ID:      {matched_parent_id}\")\n"
            "print(f\"Expanded text:  ({len(expanded_text)} chars)\")\n"
            "print(f\"  {expanded_text[:80]}...\")"
        ),
        md(
            "## ParentExpander API\n"
            "In a real application, `ParentExpander` wraps this lookup pattern."
        ),
        code(
            "# ParentExpander usage pattern:\n"
            "#\n"
            "#   expander = ParentExpander(\n"
            "#       retriever=child_retriever,   # retrieves matching child chunks\n"
            "#       parent_store=parent_store,    # maps child -> parent for expansion\n"
            "#   )\n"
            "#\n"
            "#   # At query time:\n"
            "#   results = expander.retrieve(query=\"What is Anchor?\")\n"
            "#   # Returns parent chunks that contain matching children\n"
            "\n"
            "print(\"ParentExpander workflow:\")\n"
            "print(\"  1. Query arrives\")\n"
            "print(\"  2. Child retriever finds matching small chunks\")\n"
            "print(\"  3. ParentExpander maps children to parent chunks\")\n"
            "print(\"  4. Full parent context is returned for the pipeline\")\n"
            "print()\n"
            "print(\"Benefits:\")\n"
            "print(\"  - Small child chunks give precise similarity matching\")\n"
            "print(\"  - Large parent chunks provide full surrounding context\")\n"
            "print(\"  - Best of both worlds: precision + context\")"
        ),
        md(
            "## Choosing Chunk Sizes\n"
            "The ratio between parent and child sizes controls the trade-off\n"
            "between precision and context."
        ),
        code(
            "configs = [\n"
            "    {\"parent\": 256,  \"child\": 64,  \"overlap\": 10},\n"
            "    {\"parent\": 512,  \"child\": 128, \"overlap\": 20},\n"
            "    {\"parent\": 1024, \"child\": 256, \"overlap\": 50},\n"
            "    {\"parent\": 2048, \"child\": 512, \"overlap\": 100},\n"
            "]\n"
            "\n"
            "print(f\"{'Parent':>8} {'Child':>8} {'Overlap':>8} {'Ratio':>8} {'Use Case'}\")\n"
            "print(\"-\" * 60)\n"
            "for cfg in configs:\n"
            "    ratio = cfg['parent'] / cfg['child']\n"
            "    use_case = (\n"
            "        \"FAQ / short docs\" if cfg['parent'] <= 256\n"
            "        else \"General prose\" if cfg['parent'] <= 512\n"
            "        else \"Technical docs\" if cfg['parent'] <= 1024\n"
            "        else \"Long-form content\"\n"
            "    )\n"
            "    print(f\"{cfg['parent']:>8} {cfg['child']:>8} {cfg['overlap']:>8} \"\n"
            "          f\"{ratio:>7.1f}x {use_case}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `ParentChildChunker` creates a two-level chunk hierarchy\n"
            "- Small child chunks enable precise similarity matching\n"
            "- Large parent chunks provide full surrounding context\n"
            "- `ParentExpander` maps matched children back to their parent at retrieval time\n"
            "- Tune the parent/child size ratio based on your content type and query patterns\n"
            "\n"
            "**Back to:** [Ingestion README](README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Ingestion notebooks in {OUTPUT_DIR}/\n")

    nb_document_ingester()
    nb_chunking_strategies()
    nb_parsers()
    nb_metadata_enrichment()
    nb_parent_child_chunks()

    print(f"\nDone. 5 notebooks created.")


if __name__ == "__main__":
    main()
