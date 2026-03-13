"""Generate all 6 Jupyter notebooks for the Storage module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "12-storage"))
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
# 01 - Vector Store
# ---------------------------------------------------------------------------
def nb_vector_store():
    write_notebook("01_vector_store.ipynb", [
        md(
            "# Vector Store\n"
            "> Store and search embeddings with metadata.\n"
            "\n"
            "`InMemoryVectorStore` provides a lightweight, dictionary-backed vector store\n"
            "for adding embeddings, running similarity searches, and managing entries.\n"
            "Use it for prototyping or swap in a production backend via the protocol.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.storage import InMemoryVectorStore"
        ),
        md(
            "## Create a Vector Store\n"
            "No configuration required for the in-memory implementation."
        ),
        code(
            "store = InMemoryVectorStore()\n"
            "\n"
            "print(f\"Type: {type(store).__name__}\")"
        ),
        md(
            "## Add Embeddings\n"
            "Each embedding has an ID, a vector, and optional metadata."
        ),
        code(
            "store.add_embedding(\n"
            "    id=\"doc-1\",\n"
            "    embedding=[0.1, 0.2, 0.3, 0.4, 0.5],\n"
            "    metadata={\"source\": \"wiki\", \"topic\": \"python\"},\n"
            ")\n"
            "\n"
            "store.add_embedding(\n"
            "    id=\"doc-2\",\n"
            "    embedding=[0.5, 0.4, 0.3, 0.2, 0.1],\n"
            "    metadata={\"source\": \"docs\", \"topic\": \"rust\"},\n"
            ")\n"
            "\n"
            "store.add_embedding(\n"
            "    id=\"doc-3\",\n"
            "    embedding=[0.1, 0.2, 0.3, 0.4, 0.6],\n"
            "    metadata={\"source\": \"wiki\", \"topic\": \"python\"},\n"
            ")\n"
            "\n"
            "print(\"Added 3 embeddings\")"
        ),
        md(
            "## Similarity Search\n"
            "Pass a query embedding and retrieve the top-k closest matches."
        ),
        code(
            "results = store.search(\n"
            "    query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],\n"
            "    top_k=5,\n"
            ")\n"
            "\n"
            "print(f\"Results: {len(results)}\")\n"
            "for r in results:\n"
            "    print(f\"  id={r['id']}  score={r.get('score', 'N/A')}  \"\n"
            "          f\"meta={r.get('metadata', {})}\")"
        ),
        md(
            "## Delete and Clear\n"
            "Remove individual entries or wipe the store."
        ),
        code(
            "store.delete(\"doc-1\")\n"
            "print(\"Deleted 'doc-1'\")\n"
            "\n"
            "# Verify removal\n"
            "results_after = store.search(\n"
            "    query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],\n"
            "    top_k=5,\n"
            ")\n"
            "print(f\"Results after delete: {len(results_after)}\")\n"
            "\n"
            "# Clear everything\n"
            "store.clear()\n"
            "print(\"\\nStore cleared\")\n"
            "results_empty = store.search(\n"
            "    query_embedding=[0.1, 0.2, 0.3],\n"
            "    top_k=5,\n"
            ")\n"
            "print(f\"Results after clear: {len(results_empty)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryVectorStore` stores embeddings with optional metadata\n"
            "- `.search()` returns the top-k nearest neighbors\n"
            "- `.delete()` removes by ID; `.clear()` wipes the store\n"
            "- Swap to a production backend (FAISS, Pinecone, etc.) via the `VectorStore` protocol\n"
            "\n"
            "**Next:** [Document Store](02_document_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Document Store
# ---------------------------------------------------------------------------
def nb_document_store():
    write_notebook("02_document_store.ipynb", [
        md(
            "# Document Store\n"
            "> Store, retrieve, and manage text documents with metadata.\n"
            "\n"
            "`InMemoryDocumentStore` provides a simple key-value store for text documents.\n"
            "Each document has an ID, content string, and optional metadata dictionary.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.storage import InMemoryDocumentStore"
        ),
        md(
            "## Create a Document Store"
        ),
        code(
            "store = InMemoryDocumentStore()\n"
            "\n"
            "print(f\"Type: {type(store).__name__}\")"
        ),
        md(
            "## Add Documents\n"
            "Each document requires an ID, content, and optional metadata."
        ),
        code(
            "store.add(\n"
            "    doc_id=\"doc-1\",\n"
            "    content=\"Python is a high-level programming language.\",\n"
            "    metadata={\"author\": \"Alice\", \"category\": \"programming\"},\n"
            ")\n"
            "\n"
            "store.add(\n"
            "    doc_id=\"doc-2\",\n"
            "    content=\"Rust focuses on safety and performance.\",\n"
            "    metadata={\"author\": \"Bob\", \"category\": \"programming\"},\n"
            ")\n"
            "\n"
            "store.add(\n"
            "    doc_id=\"doc-3\",\n"
            "    content=\"Machine learning is a subset of artificial intelligence.\",\n"
            "    metadata={\"author\": \"Carol\", \"category\": \"AI\"},\n"
            ")\n"
            "\n"
            "print(\"Added 3 documents\")"
        ),
        md(
            "## Retrieve a Single Document\n"
            "`.get()` returns the document by ID."
        ),
        code(
            "doc = store.get(\"doc-1\")\n"
            "\n"
            "print(f\"ID:       {doc['id']}\")\n"
            "print(f\"Content:  {doc['content']}\")\n"
            "print(f\"Metadata: {doc['metadata']}\")"
        ),
        md(
            "## List All Documents\n"
            "`.get_all()` returns every stored document."
        ),
        code(
            "all_docs = store.get_all()\n"
            "\n"
            "print(f\"Total documents: {len(all_docs)}\\n\")\n"
            "for d in all_docs:\n"
            "    print(f\"  [{d['id']}] {d['content'][:50]}...\")\n"
            "    print(f\"           author={d['metadata'].get('author')}\")"
        ),
        md(
            "## Delete a Document"
        ),
        code(
            "store.delete(\"doc-1\")\n"
            "print(\"Deleted 'doc-1'\")\n"
            "\n"
            "remaining = store.get_all()\n"
            "print(f\"Remaining: {len(remaining)} documents\")\n"
            "for d in remaining:\n"
            "    print(f\"  [{d['id']}] {d['content'][:50]}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryDocumentStore` is a simple ID-keyed text store\n"
            "- `.add()` stores content with metadata; `.get()` retrieves by ID\n"
            "- `.get_all()` lists every document; `.delete()` removes one\n"
            "- Ideal for prototyping before moving to a persistent backend\n"
            "\n"
            "**Next:** [Context Store](03_context_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Context Store
# ---------------------------------------------------------------------------
def nb_context_store():
    write_notebook("03_context_store.ipynb", [
        md(
            "# Context Store\n"
            "> Manage `ContextItem` objects with scores, priorities, and source types.\n"
            "\n"
            "`InMemoryContextStore` holds `ContextItem` instances \u2014 the core unit of\n"
            "context in Anchor.  Each item carries content, a source type, relevance\n"
            "score, priority, and token count.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.storage import InMemoryContextStore\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md(
            "## Create a Context Store"
        ),
        code(
            "store = InMemoryContextStore()\n"
            "\n"
            "print(f\"Type: {type(store).__name__}\")"
        ),
        md(
            "## Add Context Items\n"
            "Build `ContextItem` objects and store them."
        ),
        code(
            "item1 = ContextItem(\n"
            "    id=\"ctx-1\",\n"
            "    content=\"Python was created by Guido van Rossum.\",\n"
            "    source=SourceType.RETRIEVAL,\n"
            "    score=0.95,\n"
            "    priority=5,\n"
            "    token_count=8,\n"
            ")\n"
            "\n"
            "item2 = ContextItem(\n"
            "    id=\"ctx-2\",\n"
            "    content=\"The user prefers concise answers.\",\n"
            "    source=SourceType.MEMORY,\n"
            "    score=0.80,\n"
            "    priority=7,\n"
            "    token_count=6,\n"
            ")\n"
            "\n"
            "item3 = ContextItem(\n"
            "    id=\"ctx-3\",\n"
            "    content=\"Always respond in English.\",\n"
            "    source=SourceType.SYSTEM,\n"
            "    score=1.0,\n"
            "    priority=10,\n"
            "    token_count=4,\n"
            ")\n"
            "\n"
            "store.add(item1)\n"
            "store.add(item2)\n"
            "store.add(item3)\n"
            "\n"
            "print(\"Added 3 context items\")"
        ),
        md(
            "## Retrieve a Single Item"
        ),
        code(
            "retrieved = store.get(\"ctx-1\")\n"
            "\n"
            "print(f\"ID:       {retrieved.id}\")\n"
            "print(f\"Content:  {retrieved.content}\")\n"
            "print(f\"Source:   {retrieved.source.value}\")\n"
            "print(f\"Score:    {retrieved.score}\")\n"
            "print(f\"Priority: {retrieved.priority}\")\n"
            "print(f\"Tokens:   {retrieved.token_count}\")"
        ),
        md(
            "## List All Items"
        ),
        code(
            "all_items = store.get_all()\n"
            "\n"
            "print(f\"Total items: {len(all_items)}\\n\")\n"
            "for item in all_items:\n"
            "    print(f\"  [{item.id}] source={item.source.value:10s}  \"\n"
            "          f\"score={item.score:.2f}  priority={item.priority}  \"\n"
            "          f\"tokens={item.token_count}\")\n"
            "    print(f\"           content='{item.content}'\")"
        ),
        md(
            "## Sort by Priority and Score\n"
            "Context items are plain objects \u2014 sort them for pipeline ordering."
        ),
        code(
            "sorted_items = sorted(all_items, key=lambda x: (-x.priority, -x.score))\n"
            "\n"
            "print(\"Items sorted by priority (desc), then score (desc):\\n\")\n"
            "for item in sorted_items:\n"
            "    print(f\"  [{item.id}] priority={item.priority}  \"\n"
            "          f\"score={item.score:.2f}  '{item.content[:40]}'\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryContextStore` holds `ContextItem` objects\n"
            "- Each item has content, source type, score, priority, and token count\n"
            "- `.add()`, `.get()`, and `.get_all()` cover basic CRUD\n"
            "- Items can be sorted by priority/score for pipeline ordering\n"
            "\n"
            "**Next:** [Entry Store](04_entry_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Entry Store
# ---------------------------------------------------------------------------
def nb_entry_store():
    write_notebook("04_entry_store.ipynb", [
        md(
            "# Entry Store\n"
            "> Manage `MemoryEntry` objects for long-term memory.\n"
            "\n"
            "`InMemoryEntryStore` stores `MemoryEntry` instances with CRUD operations.\n"
            "Memory entries represent facts, preferences, or other persistent knowledge\n"
            "that should survive across sessions.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.storage import InMemoryEntryStore\n"
            "from anchor.models import MemoryEntry"
        ),
        md(
            "## Create an Entry Store"
        ),
        code(
            "store = InMemoryEntryStore()\n"
            "\n"
            "print(f\"Type: {type(store).__name__}\")"
        ),
        md(
            "## Add Memory Entries\n"
            "Each entry carries an ID, content, relevance score, and optional metadata."
        ),
        code(
            "entry1 = MemoryEntry(\n"
            "    id=\"mem-1\",\n"
            "    content=\"User prefers dark mode interfaces.\",\n"
            "    relevance_score=0.8,\n"
            "    metadata={\"category\": \"preference\"},\n"
            ")\n"
            "\n"
            "entry2 = MemoryEntry(\n"
            "    id=\"mem-2\",\n"
            "    content=\"User is a senior Python developer.\",\n"
            "    relevance_score=0.9,\n"
            "    metadata={\"category\": \"profile\"},\n"
            ")\n"
            "\n"
            "entry3 = MemoryEntry(\n"
            "    id=\"mem-3\",\n"
            "    content=\"Last project used FastAPI and PostgreSQL.\",\n"
            "    relevance_score=0.6,\n"
            "    metadata={\"category\": \"history\"},\n"
            ")\n"
            "\n"
            "store.add(entry1)\n"
            "store.add(entry2)\n"
            "store.add(entry3)\n"
            "\n"
            "print(\"Added 3 memory entries\")"
        ),
        md(
            "## List All Entries\n"
            "`.list_all()` returns every stored entry."
        ),
        code(
            "all_entries = store.list_all()\n"
            "\n"
            "print(f\"Total entries: {len(all_entries)}\\n\")\n"
            "for e in all_entries:\n"
            "    print(f\"  [{e.id}] score={e.relevance_score:.1f}  \"\n"
            "          f\"cat={e.metadata.get('category', 'N/A')}\")\n"
            "    print(f\"          '{e.content}'\")"
        ),
        md(
            "## Update an Entry\n"
            "Modify an entry and call `.update()` to persist the change."
        ),
        code(
            "# Update the relevance score\n"
            "entry1.relevance_score = 0.95\n"
            "entry1.content = \"User strongly prefers dark mode interfaces.\"\n"
            "\n"
            "store.update(entry1)\n"
            "print(\"Updated 'mem-1'\")\n"
            "\n"
            "# Verify\n"
            "updated = [e for e in store.list_all() if e.id == \"mem-1\"][0]\n"
            "print(f\"  New score:   {updated.relevance_score}\")\n"
            "print(f\"  New content: '{updated.content}'\")"
        ),
        md(
            "## Delete an Entry"
        ),
        code(
            "store.delete(\"mem-3\")\n"
            "print(\"Deleted 'mem-3'\")\n"
            "\n"
            "remaining = store.list_all()\n"
            "print(f\"Remaining: {len(remaining)} entries\")\n"
            "for e in remaining:\n"
            "    print(f\"  [{e.id}] '{e.content[:50]}'\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryEntryStore` manages `MemoryEntry` objects\n"
            "- `.add()`, `.update()`, `.list_all()`, `.delete()` cover full CRUD\n"
            "- Entries carry relevance scores for prioritized retrieval\n"
            "- Use metadata to categorize entries (preference, profile, history, etc.)\n"
            "\n"
            "**Next:** [JSON File Store](05_json_file_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - JSON File Store
# ---------------------------------------------------------------------------
def nb_json_file_store():
    write_notebook("05_json_file_store.ipynb", [
        md(
            "# JSON File Store\n"
            "> Persist memory entries to disk as JSON.\n"
            "\n"
            "`JsonFileMemoryStore` extends the entry store concept with file-based\n"
            "persistence.  Entries are saved to and loaded from a JSON file, making it\n"
            "easy to maintain memory across application restarts.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "import os\n"
            "import tempfile\n"
            "from anchor.storage import JsonFileMemoryStore\n"
            "from anchor.models import MemoryEntry"
        ),
        md(
            "## Create a Store with a File Path\n"
            "We use a temporary file so this notebook is self-contained."
        ),
        code(
            "tmp_dir = tempfile.mkdtemp()\n"
            "file_path = os.path.join(tmp_dir, \"test_memory.json\")\n"
            "\n"
            "store = JsonFileMemoryStore(file_path=file_path)\n"
            "\n"
            "print(f\"File path: {file_path}\")\n"
            "print(f\"File exists: {os.path.exists(file_path)}\")"
        ),
        md(
            "## Add Entries and Save to Disk"
        ),
        code(
            "entry1 = MemoryEntry(\n"
            "    id=\"mem-1\",\n"
            "    content=\"User prefers concise code examples.\",\n"
            "    relevance_score=0.85,\n"
            "    metadata={\"source\": \"conversation\"},\n"
            ")\n"
            "\n"
            "entry2 = MemoryEntry(\n"
            "    id=\"mem-2\",\n"
            "    content=\"Primary language is Python 3.12.\",\n"
            "    relevance_score=0.9,\n"
            "    metadata={\"source\": \"profile\"},\n"
            ")\n"
            "\n"
            "store.add(entry1)\n"
            "store.add(entry2)\n"
            "\n"
            "# Persist to file\n"
            "store.save()\n"
            "\n"
            "print(f\"Saved {len(store.list_all())} entries\")\n"
            "print(f\"File exists: {os.path.exists(file_path)}\")\n"
            "print(f\"File size:   {os.path.getsize(file_path)} bytes\")"
        ),
        md(
            "## Load from Disk\n"
            "Create a new store instance pointing to the same file and load data."
        ),
        code(
            "loaded = JsonFileMemoryStore(file_path=file_path)\n"
            "loaded.load()\n"
            "\n"
            "entries = loaded.list_all()\n"
            "print(f\"Loaded {len(entries)} entries from disk\\n\")\n"
            "for e in entries:\n"
            "    print(f\"  [{e.id}] score={e.relevance_score}  '{e.content}'\")"
        ),
        md(
            "## Modify and Re-Save\n"
            "Updates are persisted by calling `.save()` again."
        ),
        code(
            "new_entry = MemoryEntry(\n"
            "    id=\"mem-3\",\n"
            "    content=\"Frequently works with async/await patterns.\",\n"
            "    relevance_score=0.7,\n"
            "    metadata={\"source\": \"observation\"},\n"
            ")\n"
            "\n"
            "loaded.add(new_entry)\n"
            "loaded.save()\n"
            "\n"
            "print(f\"Now {len(loaded.list_all())} entries on disk\")\n"
            "print(f\"File size: {os.path.getsize(file_path)} bytes\")"
        ),
        md(
            "## Cleanup\n"
            "Remove the temporary file."
        ),
        code(
            "os.remove(file_path)\n"
            "os.rmdir(tmp_dir)\n"
            "print(f\"Cleaned up temporary file: {file_path}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `JsonFileMemoryStore` persists memory entries as JSON on disk\n"
            "- `.save()` writes current state; `.load()` restores from file\n"
            "- Combine with `MemoryEntry` for simple, file-based persistence\n"
            "- Ideal for local development and small-scale deployments\n"
            "\n"
            "**Next:** [Custom Store](06_custom_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Custom Store
# ---------------------------------------------------------------------------
def nb_custom_store():
    write_notebook("06_custom_store.ipynb", [
        md(
            "# Custom Store\n"
            "> Implement the storage protocols with your own backends.\n"
            "\n"
            "Anchor defines three storage protocols: `ContextStore`, `VectorStore`, and\n"
            "`MemoryEntryStore`.  Any class that implements the required methods can be\n"
            "used as a drop-in replacement \u2014 no inheritance needed.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.protocols import ContextStore, VectorStore, MemoryEntryStore"
        ),
        md(
            "## Protocol Overview\n"
            "Each protocol requires a small set of methods:\n"
            "\n"
            "| Protocol | Methods |\n"
            "|---|---|\n"
            "| `ContextStore` | `add()`, `get()`, `get_all()` |\n"
            "| `VectorStore` | `add_embedding()`, `search()`, `delete()`, `clear()` |\n"
            "| `MemoryEntryStore` | `add()`, `update()`, `delete()`, `list_all()` |"
        ),
        md(
            "## Implement a Custom ContextStore\n"
            "A minimal dictionary-backed context store."
        ),
        code(
            "class DictContextStore:\n"
            "    \"\"\"Simple dict-backed ContextStore.\"\"\"\n"
            "\n"
            "    def __init__(self):\n"
            "        self._items = {}\n"
            "\n"
            "    def add(self, item):\n"
            "        self._items[item.id] = item\n"
            "        print(f\"  Added context item '{item.id}'\")\n"
            "\n"
            "    def get(self, item_id: str):\n"
            "        return self._items.get(item_id)\n"
            "\n"
            "    def get_all(self):\n"
            "        return list(self._items.values())\n"
            "\n"
            "\n"
            "ctx_store = DictContextStore()\n"
            "print(f\"Satisfies ContextStore: {isinstance(ctx_store, ContextStore)}\")"
        ),
        md(
            "## Implement a Custom VectorStore\n"
            "A stub that stores vectors in a list."
        ),
        code(
            "class ListVectorStore:\n"
            "    \"\"\"List-backed VectorStore for demonstration.\"\"\"\n"
            "\n"
            "    def __init__(self):\n"
            "        self._vectors = []\n"
            "\n"
            "    def add_embedding(self, id: str, embedding: list, metadata: dict = None):\n"
            "        self._vectors.append({\"id\": id, \"embedding\": embedding, \"metadata\": metadata or {}})\n"
            "        print(f\"  Added embedding '{id}' (dim={len(embedding)})\")\n"
            "\n"
            "    def search(self, query_embedding: list, top_k: int = 5) -> list:\n"
            "        # Simplified: return all entries (real impl would compute similarity)\n"
            "        print(f\"  Searching top-{top_k} among {len(self._vectors)} vectors\")\n"
            "        return self._vectors[:top_k]\n"
            "\n"
            "    def delete(self, id: str):\n"
            "        self._vectors = [v for v in self._vectors if v[\"id\"] != id]\n"
            "        print(f\"  Deleted '{id}'\")\n"
            "\n"
            "    def clear(self):\n"
            "        self._vectors.clear()\n"
            "        print(\"  Cleared all vectors\")\n"
            "\n"
            "\n"
            "vec_store = ListVectorStore()\n"
            "print(f\"Satisfies VectorStore: {isinstance(vec_store, VectorStore)}\")"
        ),
        md(
            "## Implement a Custom MemoryEntryStore\n"
            "A dictionary-backed store for memory entries."
        ),
        code(
            "class DictEntryStore:\n"
            "    \"\"\"Dict-backed MemoryEntryStore.\"\"\"\n"
            "\n"
            "    def __init__(self):\n"
            "        self._entries = {}\n"
            "\n"
            "    def add(self, entry):\n"
            "        self._entries[entry.id] = entry\n"
            "        print(f\"  Added entry '{entry.id}'\")\n"
            "\n"
            "    def update(self, entry):\n"
            "        self._entries[entry.id] = entry\n"
            "        print(f\"  Updated entry '{entry.id}'\")\n"
            "\n"
            "    def delete(self, entry_id: str):\n"
            "        self._entries.pop(entry_id, None)\n"
            "        print(f\"  Deleted entry '{entry_id}'\")\n"
            "\n"
            "    def list_all(self):\n"
            "        return list(self._entries.values())\n"
            "\n"
            "\n"
            "entry_store = DictEntryStore()\n"
            "print(f\"Satisfies MemoryEntryStore: {isinstance(entry_store, MemoryEntryStore)}\")"
        ),
        md(
            "## Exercise All Three Stores\n"
            "Quick round-trip to confirm each store works."
        ),
        code(
            "from anchor.models import ContextItem, SourceType, MemoryEntry\n"
            "\n"
            "# ContextStore\n"
            "print(\"=== ContextStore ===\")\n"
            "ctx_item = ContextItem(\n"
            "    id=\"c1\", content=\"test context\",\n"
            "    source=SourceType.RETRIEVAL, score=0.9, priority=5, token_count=2,\n"
            ")\n"
            "ctx_store.add(ctx_item)\n"
            "print(f\"  get('c1'): {ctx_store.get('c1').content}\")\n"
            "print(f\"  get_all(): {len(ctx_store.get_all())} items\")\n"
            "\n"
            "# VectorStore\n"
            "print(\"\\n=== VectorStore ===\")\n"
            "vec_store.add_embedding(id=\"v1\", embedding=[0.1, 0.2], metadata={\"tag\": \"demo\"})\n"
            "results = vec_store.search(query_embedding=[0.1, 0.2], top_k=3)\n"
            "print(f\"  Results: {len(results)}\")\n"
            "vec_store.delete(\"v1\")\n"
            "vec_store.clear()\n"
            "\n"
            "# MemoryEntryStore\n"
            "print(\"\\n=== MemoryEntryStore ===\")\n"
            "mem = MemoryEntry(\n"
            "    id=\"m1\", content=\"test memory\",\n"
            "    relevance_score=0.7, metadata={},\n"
            ")\n"
            "entry_store.add(mem)\n"
            "mem.content = \"updated memory\"\n"
            "entry_store.update(mem)\n"
            "print(f\"  list_all(): {len(entry_store.list_all())} entries\")\n"
            "entry_store.delete(\"m1\")\n"
            "print(f\"  After delete: {len(entry_store.list_all())} entries\")"
        ),
        md(
            "## Key Takeaways\n"
            "- Three storage protocols: `ContextStore`, `VectorStore`, `MemoryEntryStore`\n"
            "- All are structural (duck-typed) \u2014 implement the methods, skip the base class\n"
            "- `isinstance()` confirms conformance at runtime\n"
            "- Custom backends can wrap databases, cloud APIs, or any persistence layer\n"
            "\n"
            "**Previous:** [JSON File Store](05_json_file_store.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Storage notebooks in {OUTPUT_DIR} ...")
    nb_vector_store()
    nb_document_store()
    nb_context_store()
    nb_entry_store()
    nb_json_file_store()
    nb_custom_store()
    print("Done \u2714")


if __name__ == "__main__":
    main()
