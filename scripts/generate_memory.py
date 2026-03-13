"""Generate all 8 Jupyter notebooks for the Memory module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "01-memory"))
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
# 01 - Sliding Window Memory
# ---------------------------------------------------------------------------
def nb_sliding_window():
    write_notebook("01_sliding_window.ipynb", [
        md(
            "# Sliding Window Memory\n"
            "> Keep the last N tokens of conversation, automatically evicting the oldest turns.\n"
            "\n"
            "`SlidingWindowMemory` is the simplest conversation memory strategy: it keeps a\n"
            "fixed-size window of recent turns and drops older ones when the token limit is\n"
            "exceeded.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory import SlidingWindowMemory"
        ),
        md(
            "## Create a Sliding Window\n"
            "We set a small token budget so we can observe eviction behavior quickly."
        ),
        code(
            "memory = SlidingWindowMemory(max_tokens=2048)\n"
            "\n"
            "print(f\"Max tokens: {memory.max_tokens}\")\n"
            "print(f\"Current turns: {len(memory.turns)}\")\n"
            "print(f\"Current tokens: {memory.total_tokens}\")"
        ),
        md(
            "## Add Conversation Turns\n"
            "Each turn can carry optional metadata (timestamps, tool IDs, etc.)."
        ),
        code(
            "# Add several rounds of conversation\n"
            "memory.add_turn(\"user\", \"Hello!\", metadata={\"timestamp\": \"2024-01-01\"})\n"
            "memory.add_turn(\"assistant\", \"Hi there! How can I help?\")\n"
            "memory.add_turn(\"user\", \"What is the capital of France?\")\n"
            "memory.add_turn(\"assistant\", \"The capital of France is Paris.\")\n"
            "\n"
            "print(f\"Turns: {len(memory.turns)}\")\n"
            "print(f\"Total tokens: {memory.total_tokens}\")"
        ),
        md(
            "## Inspect Stored Turns\n"
            "The `.turns` property exposes `ConversationTurn` objects."
        ),
        code(
            "for i, turn in enumerate(memory.turns):\n"
            "    meta = f\"  meta={turn.metadata}\" if turn.metadata else \"\"\n"
            "    print(f\"[{i}] {turn.role}: {turn.content[:60]}{meta}\")"
        ),
        md(
            "## Export as Context Items\n"
            "`.as_context_items()` produces a `list[ContextItem]` ready for the pipeline."
        ),
        code(
            "items = memory.as_context_items()\n"
            "\n"
            "print(f\"Context items produced: {len(items)}\")\n"
            "for item in items:\n"
            "    print(f\"  - {item.source_type.value}: {str(item.content)[:60]}\")"
        ),
        md(
            "## Observe Eviction\n"
            "Push many turns to exceed `max_tokens` and watch the window slide."
        ),
        code(
            "# Use a tiny window to force eviction\n"
            "small = SlidingWindowMemory(max_tokens=150)\n"
            "\n"
            "for i in range(10):\n"
            "    small.add_turn(\"user\", f\"Message number {i} with some padding text.\")\n"
            "    small.add_turn(\"assistant\", f\"Reply to message {i}.\")\n"
            "\n"
            "print(f\"Turns remaining: {len(small.turns)}\")\n"
            "print(f\"Tokens used: {small.total_tokens} / {small.max_tokens}\")\n"
            "print(\"\\nSurviving turns:\")\n"
            "for turn in small.turns:\n"
            "    print(f\"  {turn.role}: {turn.content}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `SlidingWindowMemory` maintains a FIFO window capped by token count\n"
            "- `.add_turn()` accepts `role`, `content`, and optional `metadata`\n"
            "- `.as_context_items()` bridges memory to the context pipeline\n"
            "- Oldest turns are evicted first when the budget is exceeded\n"
            "\n"
            "**Next:** [Summary Buffer](02_summary_buffer.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Summary Buffer Memory
# ---------------------------------------------------------------------------
def nb_summary_buffer():
    write_notebook("02_summary_buffer.ipynb", [
        md(
            "# Summary Buffer Memory\n"
            "> Progressively summarize older conversation turns to fit more history in fewer tokens.\n"
            "\n"
            "`SummaryBufferMemory` keeps recent turns verbatim and compresses older turns\n"
            "into a running summary using a caller-supplied `compact_fn`.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory import SummaryBufferMemory"
        ),
        md(
            "## Define a Compaction Function\n"
            "In production you would call an LLM here. For this recipe we use a simple mock."
        ),
        code(
            "def mock_compact(turns):\n"
            "    \"\"\"Concatenate turn previews as a stand-in for LLM summarization.\"\"\"\n"
            "    return \"Summary: \" + \" | \".join(t.content[:20] for t in turns)\n"
            "\n"
            "print(\"Compaction function ready.\")"
        ),
        md(
            "## Create the Memory\n"
            "`summary_priority` controls where the summary appears in the context window\n"
            "(lower = closer to the system prompt)."
        ),
        code(
            "memory = SummaryBufferMemory(\n"
            "    max_tokens=1024,\n"
            "    compact_fn=mock_compact,\n"
            "    summary_priority=6,\n"
            ")\n"
            "\n"
            "print(f\"Max tokens: {memory.max_tokens}\")"
        ),
        md(
            "## Add Messages Until Compaction Triggers\n"
            "We add enough messages to push the memory over its token limit, forcing\n"
            "the buffer to compact older turns into a summary."
        ),
        code(
            "messages = [\n"
            "    (\"user\", \"Tell me about the history of Python.\"),\n"
            "    (\"assistant\", \"Python was created by Guido van Rossum and released in 1991. \"\n"
            "     \"It emphasizes code readability and allows programmers to express concepts \"\n"
            "     \"in fewer lines of code than languages like C++ or Java.\"),\n"
            "    (\"user\", \"What about its name? Why Python?\"),\n"
            "    (\"assistant\", \"The name comes from Monty Python's Flying Circus, the BBC \"\n"
            "     \"comedy series. Guido was a fan of the show and wanted a short, unique, \"\n"
            "     \"and slightly mysterious name.\"),\n"
            "    (\"user\", \"What version is current?\"),\n"
            "    (\"assistant\", \"Python 3.12 is the latest stable release. Python 2 reached \"\n"
            "     \"end of life in January 2020.\"),\n"
            "    (\"user\", \"Tell me about type hints.\"),\n"
            "    (\"assistant\", \"Type hints were introduced in PEP 484. They let you annotate \"\n"
            "     \"function signatures and variables with expected types without changing \"\n"
            "     \"runtime behavior.\"),\n"
            "    (\"user\", \"And async support?\"),\n"
            "    (\"assistant\", \"Python added async/await syntax in 3.5 via PEP 492. The \"\n"
            "     \"asyncio library provides the event loop infrastructure.\"),\n"
            "]\n"
            "\n"
            "for role, content in messages:\n"
            "    memory.add_message(role, content)\n"
            "\n"
            "print(f\"Messages added: {len(messages)}\")"
        ),
        md("## Retrieve Context Items\n"
           "The returned items include both the summary and the recent verbatim turns."
        ),
        code(
            "items = memory.as_context_items()\n"
            "\n"
            "print(f\"Context items: {len(items)}\\n\")\n"
            "for item in items:\n"
            "    preview = str(item.content)[:80]\n"
            "    print(f\"  [{item.source_type.value}] {preview}...\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `SummaryBufferMemory` blends verbatim recent turns with a compressed summary\n"
            "- Supply a `compact_fn` (typically an LLM call) to control how summaries are made\n"
            "- `summary_priority` determines where the summary is placed in the context window\n"
            "- This strategy retains more conversational history than a pure sliding window\n"
            "\n"
            "**Next:** [Graph Memory](03_graph_memory.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Graph Memory
# ---------------------------------------------------------------------------
def nb_graph_memory():
    write_notebook("03_graph_memory.ipynb", [
        md(
            "# Graph Memory\n"
            "> Track entities and relationships across conversations.\n"
            "\n"
            "`SimpleGraphMemory` provides a lightweight in-memory entity graph. You can\n"
            "store entities with metadata, create typed relationships between them, and\n"
            "link entities to specific memory entry IDs for retrieval.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory import SimpleGraphMemory"
        ),
        md("## Create Entities\n"
           "Entities are nodes in the graph. Each has an ID and optional metadata."
        ),
        code(
            "graph = SimpleGraphMemory()\n"
            "\n"
            "graph.add_entity(\"alice\", metadata={\"role\": \"engineer\", \"team\": \"backend\"})\n"
            "graph.add_entity(\"bob\", metadata={\"role\": \"designer\", \"team\": \"frontend\"})\n"
            "graph.add_entity(\"project-x\", metadata={\"type\": \"project\", \"status\": \"active\"})\n"
            "graph.add_entity(\"project-y\", metadata={\"type\": \"project\", \"status\": \"planned\"})\n"
            "\n"
            "print(\"Entities added: alice, bob, project-x, project-y\")"
        ),
        md("## Create Relationships\n"
           "Relationships are directed edges with a type label."
        ),
        code(
            "graph.add_relationship(\"alice\", \"works_on\", \"project-x\")\n"
            "graph.add_relationship(\"bob\", \"works_on\", \"project-x\")\n"
            "graph.add_relationship(\"alice\", \"mentors\", \"bob\")\n"
            "graph.add_relationship(\"alice\", \"leads\", \"project-y\")\n"
            "\n"
            "print(\"Relationships:\")\n"
            "print(\"  alice --works_on--> project-x\")\n"
            "print(\"  bob   --works_on--> project-x\")\n"
            "print(\"  alice --mentors---> bob\")\n"
            "print(\"  alice --leads-----> project-y\")"
        ),
        md("## Link Memory Entries to Entities\n"
           "Associate memory entry IDs with entities so you can retrieve relevant memories."
        ),
        code(
            "graph.link_memory(\"alice\", \"mem-001\")\n"
            "graph.link_memory(\"alice\", \"mem-002\")\n"
            "graph.link_memory(\"project-x\", \"mem-003\")\n"
            "\n"
            "print(\"Linked memories:\")\n"
            "print(\"  alice     -> mem-001, mem-002\")\n"
            "print(\"  project-x -> mem-003\")"
        ),
        md("## Traverse the Graph\n"
           "`get_related_entities` performs a breadth-first traversal up to `max_depth`."
        ),
        code(
            "related = graph.get_related_entities(\"alice\", max_depth=2)\n"
            "\n"
            "print(f\"Entities related to 'alice' (depth <= 2): {len(related)}\")\n"
            "for entity_id in related:\n"
            "    print(f\"  - {entity_id}\")"
        ),
        md("## Retrieve Related Memory IDs\n"
           "Collect all memory IDs linked to entities within the traversal radius."
        ),
        code(
            "memory_ids = graph.get_related_memory_ids(\"alice\", max_depth=2)\n"
            "\n"
            "print(f\"Memory IDs related to 'alice': {memory_ids}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `SimpleGraphMemory` stores entities, relationships, and memory links\n"
            "- Use `add_relationship()` for directed, typed edges between entities\n"
            "- `link_memory()` associates entries with entities for retrieval\n"
            "- `get_related_entities()` traverses the graph up to a specified depth\n"
            "- Graph memory shines when conversations involve multiple interacting concepts\n"
            "\n"
            "**Next:** [Memory Manager](04_memory_manager.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Memory Manager
# ---------------------------------------------------------------------------
def nb_memory_manager():
    write_notebook("04_memory_manager.ipynb", [
        md(
            "# Memory Manager\n"
            "> Unified facade for conversation memory and persistent storage.\n"
            "\n"
            "`MemoryManager` combines short-term conversation memory with a persistent\n"
            "entry store. It handles adding messages, building context items, and recalling\n"
            "relevant entries from long-term storage.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory import MemoryManager, SlidingWindowMemory\n"
            "from anchor.storage import InMemoryEntryStore"
        ),
        md("## Initialize the Manager\n"
           "Provide a token budget and a persistent store backend."
        ),
        code(
            "store = InMemoryEntryStore()\n"
            "\n"
            "manager = MemoryManager(\n"
            "    conversation_tokens=4096,\n"
            "    persistent_store=store,\n"
            ")\n"
            "\n"
            "print(f\"Conversation token budget: 4096\")\n"
            "print(f\"Persistent store type: {type(store).__name__}\")"
        ),
        md("## Add Conversation Messages"),
        code(
            "manager.add_message(\"user\", \"Tell me about Python\")\n"
            "manager.add_message(\"assistant\", \"Python is a high-level programming language \"\n"
            "    \"known for its readability and versatility.\")\n"
            "manager.add_message(\"user\", \"What about its package manager?\")\n"
            "manager.add_message(\"assistant\", \"pip is the standard package manager for \"\n"
            "    \"Python. It installs packages from PyPI.\")\n"
            "\n"
            "print(\"Added 4 messages to conversation memory.\")"
        ),
        md("## Get Context Items\n"
           "Combine conversation turns and any relevant persistent entries into context items."
        ),
        code(
            "context_items = manager.get_context_items()\n"
            "\n"
            "print(f\"Total context items: {len(context_items)}\")\n"
            "for item in context_items:\n"
            "    print(f\"  [{item.source_type.value}] {str(item.content)[:70]}\")"
        ),
        md("## Recall from Persistent Storage\n"
           "Search the persistent store for entries matching a query."
        ),
        code(
            "recalled = manager.recall(\"Python\")\n"
            "\n"
            "print(f\"Recalled entries: {len(recalled) if recalled else 0}\")\n"
            "if recalled:\n"
            "    for entry in recalled:\n"
            "        print(f\"  - {str(entry)[:80]}\")\n"
            "else:\n"
            "    print(\"  (No persistent entries stored yet -- \"\n"
            "          \"recall returns results from long-term storage.)\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MemoryManager` is the recommended high-level interface for memory\n"
            "- It combines conversation (short-term) and persistent (long-term) memory\n"
            "- `.get_context_items()` produces pipeline-ready context items\n"
            "- `.recall(query)` searches the persistent store for relevant memories\n"
            "- Swap `InMemoryEntryStore` for a vector DB store in production\n"
            "\n"
            "**Next:** [Eviction Policies](05_eviction_policies.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Eviction Policies
# ---------------------------------------------------------------------------
def nb_eviction_policies():
    write_notebook("05_eviction_policies.ipynb", [
        md(
            "# Eviction Policies\n"
            "> Control which turns are removed when conversation memory overflows.\n"
            "\n"
            "When a `SlidingWindowMemory` exceeds its token budget it needs an eviction\n"
            "policy to decide which turns to drop. Anchor ships three built-in policies:\n"
            "**FIFO**, **Importance**, and **Paired**.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory import SlidingWindowMemory\n"
            "from anchor.memory.eviction import (\n"
            "    FIFOEviction,\n"
            "    ImportanceEviction,\n"
            "    PairedEviction,\n"
            ")"
        ),
        md(
            "## Helper: Fill a Memory\n"
            "We will reuse this function to populate memory objects for each policy."
        ),
        code(
            "def fill_memory(memory, n=8):\n"
            "    \"\"\"Add n user/assistant turn pairs.\"\"\"\n"
            "    for i in range(n):\n"
            "        memory.add_turn(\"user\", f\"User message {i}: \" + \"x\" * 30)\n"
            "        memory.add_turn(\"assistant\", f\"Reply {i}: \" + \"y\" * 30)\n"
            "    return memory\n"
            "\n"
            "print(\"Helper ready.\")"
        ),
        md(
            "## FIFO Eviction\n"
            "The default: oldest turns are evicted first."
        ),
        code(
            "fifo = FIFOEviction()\n"
            "memory_fifo = fill_memory(SlidingWindowMemory(max_tokens=512, eviction_policy=fifo))\n"
            "\n"
            "print(f\"Turns remaining: {len(memory_fifo.turns)}\")\n"
            "print(f\"Tokens: {memory_fifo.total_tokens} / {memory_fifo.max_tokens}\")\n"
            "print(\"\\nSurviving turns (newest):\")\n"
            "for t in memory_fifo.turns:\n"
            "    print(f\"  {t.role}: {t.content[:50]}\")"
        ),
        md(
            "## Importance-Based Eviction\n"
            "Provide a scoring function; turns with the **lowest** importance are evicted first."
        ),
        code(
            "# Longer content = more important (toy heuristic)\n"
            "importance_fn = lambda turn: len(turn.content) / 100\n"
            "importance = ImportanceEviction(importance_fn=importance_fn)\n"
            "\n"
            "memory_imp = fill_memory(\n"
            "    SlidingWindowMemory(max_tokens=512, eviction_policy=importance)\n"
            ")\n"
            "\n"
            "print(f\"Turns remaining: {len(memory_imp.turns)}\")\n"
            "print(f\"Tokens: {memory_imp.total_tokens} / {memory_imp.max_tokens}\")\n"
            "print(\"\\nSurviving turns (most important):\")\n"
            "for t in memory_imp.turns:\n"
            "    print(f\"  {t.role}: {t.content[:50]}\")"
        ),
        md(
            "## Paired Eviction\n"
            "Evict user + assistant turns as a pair so you never leave an orphaned message."
        ),
        code(
            "paired = PairedEviction()\n"
            "memory_paired = fill_memory(\n"
            "    SlidingWindowMemory(max_tokens=512, eviction_policy=paired)\n"
            ")\n"
            "\n"
            "print(f\"Turns remaining: {len(memory_paired.turns)}\")\n"
            "print(f\"Tokens: {memory_paired.total_tokens} / {memory_paired.max_tokens}\")\n"
            "print(\"\\nSurviving paired turns:\")\n"
            "for t in memory_paired.turns:\n"
            "    print(f\"  {t.role}: {t.content[:50]}\")"
        ),
        md("## Compare Side-by-Side"),
        code(
            "print(f\"{'Policy':<20} {'Remaining Turns':>15} {'Tokens Used':>12}\")\n"
            "print(\"-\" * 50)\n"
            "for name, mem in [\n"
            "    (\"FIFO\", memory_fifo),\n"
            "    (\"Importance\", memory_imp),\n"
            "    (\"Paired\", memory_paired),\n"
            "]:\n"
            "    print(f\"{name:<20} {len(mem.turns):>15} {mem.total_tokens:>12}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- **FIFOEviction**: simple oldest-first removal (the default)\n"
            "- **ImportanceEviction**: uses a scoring function to keep high-value turns\n"
            "- **PairedEviction**: preserves user/assistant turn pairs\n"
            "- Pass the policy via `eviction_policy=` to `SlidingWindowMemory`\n"
            "\n"
            "**Next:** [Decay Strategies](06_decay_strategies.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Decay Strategies
# ---------------------------------------------------------------------------
def nb_decay_strategies():
    write_notebook("06_decay_strategies.ipynb", [
        md(
            "# Decay Strategies\n"
            "> Model how memory fades over time using decay curves and recency scoring.\n"
            "\n"
            "Anchor provides two decay models for persistent memory entries and two\n"
            "recency scorers for conversation turns.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory.decay import EbbinghausDecay, LinearDecay\n"
            "from anchor.memory.recency import ExponentialRecencyScorer, LinearRecencyScorer"
        ),
        md(
            "## Ebbinghaus Forgetting Curve\n"
            "Models the classic exponential forgetting curve. Repeated access (reinforcement)\n"
            "slows the decay."
        ),
        code(
            "ebbinghaus = EbbinghausDecay(\n"
            "    base_strength=1.0,\n"
            "    reinforcement_factor=0.5,\n"
            ")\n"
            "\n"
            "# Simulate retention at various hours since last access\n"
            "print(f\"{'Hours':>6}  {'Retention':>10}\")\n"
            "print(\"-\" * 20)\n"
            "for hours in [0, 1, 6, 24, 72, 168]:\n"
            "    # Approximate by computing decay manually for demonstration\n"
            "    import math\n"
            "    retention = math.exp(-hours / (24 * ebbinghaus.base_strength))\n"
            "    print(f\"{hours:>6}  {retention:>10.4f}\")"
        ),
        md(
            "## Linear Decay\n"
            "Retention decreases linearly, reaching 0.5 at `half_life_hours`."
        ),
        code(
            "linear = LinearDecay(half_life_hours=168.0)  # 1 week\n"
            "\n"
            "print(f\"Half-life: {linear.half_life_hours} hours (1 week)\")\n"
            "print(f\"\\n{'Hours':>6}  {'Retention':>10}\")\n"
            "print(\"-\" * 20)\n"
            "for hours in [0, 24, 72, 168, 336, 504]:\n"
            "    retention = max(0.0, 1.0 - (hours / (2 * linear.half_life_hours)))\n"
            "    print(f\"{hours:>6}  {retention:>10.4f}\")"
        ),
        md(
            "## Exponential Recency Scorer\n"
            "Scores conversation turns by position. Recent turns score higher with\n"
            "exponential drop-off."
        ),
        code(
            "exp_scorer = ExponentialRecencyScorer(decay_rate=2.0)\n"
            "\n"
            "total_turns = 10\n"
            "print(f\"Exponential scorer (decay_rate={exp_scorer.decay_rate})\")\n"
            "print(f\"\\n{'Position':>8}  {'Score':>8}\")\n"
            "print(\"-\" * 20)\n"
            "for pos in range(total_turns):\n"
            "    import math\n"
            "    # position 0 = most recent\n"
            "    score = math.exp(-exp_scorer.decay_rate * pos / total_turns)\n"
            "    print(f\"{pos:>8}  {score:>8.4f}\")"
        ),
        md(
            "## Linear Recency Scorer\n"
            "Scores decrease linearly from 1.0 (most recent) to near 0.0 (oldest)."
        ),
        code(
            "lin_scorer = LinearRecencyScorer()\n"
            "\n"
            "total_turns = 10\n"
            "print(\"Linear recency scorer\")\n"
            "print(f\"\\n{'Position':>8}  {'Score':>8}\")\n"
            "print(\"-\" * 20)\n"
            "for pos in range(total_turns):\n"
            "    score = 1.0 - (pos / total_turns)\n"
            "    print(f\"{pos:>8}  {score:>8.4f}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- **EbbinghausDecay**: exponential forgetting with reinforcement support\n"
            "- **LinearDecay**: simpler linear drop-off controlled by `half_life_hours`\n"
            "- **ExponentialRecencyScorer** / **LinearRecencyScorer**: score conversation\n"
            "  turns by position for priority-aware context building\n"
            "- Decay strategies feed into garbage collection (next notebook)\n"
            "\n"
            "**Next:** [Consolidation](07_consolidation.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 07 - Consolidation
# ---------------------------------------------------------------------------
def nb_consolidation():
    write_notebook("07_consolidation.ipynb", [
        md(
            "# Memory Consolidation\n"
            "> Merge similar memories to reduce redundancy and keep storage efficient.\n"
            "\n"
            "`SimilarityConsolidator` compares a new memory entry against existing ones\n"
            "using embeddings. When similarity exceeds a threshold the entries are merged.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory.consolidation import SimilarityConsolidator\n"
            "from anchor.models import MemoryEntry"
        ),
        md(
            "## Mock Embedding Function\n"
            "In production, use a real embedding model. Here we create a deterministic\n"
            "hash-based mock."
        ),
        code(
            "def mock_embed(text: str) -> list[float]:\n"
            "    \"\"\"Deterministic pseudo-embedding for demonstration.\"\"\"\n"
            "    padded = text[:128].ljust(128)\n"
            "    return [hash(c) % 100 / 100.0 for c in padded]\n"
            "\n"
            "# Quick test\n"
            "vec = mock_embed(\"hello world\")\n"
            "print(f\"Embedding dimension: {len(vec)}\")\n"
            "print(f\"First 5 values: {vec[:5]}\")"
        ),
        md("## Create the Consolidator"),
        code(
            "consolidator = SimilarityConsolidator(\n"
            "    embed_fn=mock_embed,\n"
            "    similarity_threshold=0.85,\n"
            "    max_cache_size=1000,\n"
            ")\n"
            "\n"
            "print(f\"Similarity threshold: {consolidator.similarity_threshold}\")\n"
            "print(f\"Max cache size: {consolidator.max_cache_size}\")"
        ),
        md(
            "## Prepare Memory Entries\n"
            "Create a set of existing entries and a new entry to consolidate against them."
        ),
        code(
            "from datetime import datetime, timezone\n"
            "\n"
            "existing_entries = [\n"
            "    MemoryEntry(\n"
            "        entry_id=\"e-001\",\n"
            "        content=\"Python was released in 1991 by Guido van Rossum.\",\n"
            "        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"e-002\",\n"
            "        content=\"JavaScript is the most popular web programming language.\",\n"
            "        created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"e-003\",\n"
            "        content=\"Rust provides memory safety without garbage collection.\",\n"
            "        created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),\n"
            "    ),\n"
            "]\n"
            "\n"
            "# New entry similar to e-001\n"
            "new_entry = MemoryEntry(\n"
            "    entry_id=\"e-new\",\n"
            "    content=\"Python was first released in 1991 by Guido van Rossum.\",\n"
            "    created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),\n"
            ")\n"
            "\n"
            "print(f\"Existing entries: {len(existing_entries)}\")\n"
            "print(f\"New entry: '{new_entry.content[:50]}...'\")"
        ),
        md("## Run Consolidation\n"
           "The consolidator returns a list of `(MemoryOperation, MemoryEntry | None)` tuples."
        ),
        code(
            "operations = consolidator.consolidate(new_entry, existing_entries)\n"
            "\n"
            "print(f\"Operations returned: {len(operations)}\\n\")\n"
            "for op, entry in operations:\n"
            "    entry_info = entry.entry_id if entry else \"N/A\"\n"
            "    content_preview = entry.content[:60] if entry else \"\"\n"
            "    print(f\"  Operation: {op}\")\n"
            "    print(f\"  Entry:     {entry_info}\")\n"
            "    if content_preview:\n"
            "        print(f\"  Content:   {content_preview}...\")\n"
            "    print()"
        ),
        md(
            "## Key Takeaways\n"
            "- `SimilarityConsolidator` deduplicates memories based on embedding similarity\n"
            "- Supply your own `embed_fn` to control how content is vectorized\n"
            "- `similarity_threshold` (0-1) controls how aggressive merging is\n"
            "- Consolidation returns operations you can apply to your storage backend\n"
            "- This prevents unbounded growth of near-duplicate memories\n"
            "\n"
            "**Next:** [Garbage Collection](08_garbage_collection.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 08 - Garbage Collection
# ---------------------------------------------------------------------------
def nb_garbage_collection():
    write_notebook("08_garbage_collection.ipynb", [
        md(
            "# Memory Garbage Collection\n"
            "> Two-phase cleanup: prune expired entries, then prune decayed entries.\n"
            "\n"
            "`MemoryGarbageCollector` removes stale memories in two phases:\n"
            "1. **Expired** -- entries past their TTL\n"
            "2. **Decayed** -- entries whose retention score has fallen below a threshold\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.memory.gc import MemoryGarbageCollector, GCStats\n"
            "from anchor.memory.decay import LinearDecay\n"
            "from anchor.storage import InMemoryEntryStore\n"
            "from anchor.models import MemoryEntry\n"
            "from datetime import datetime, timedelta, timezone"
        ),
        md("## Populate a Store with Entries\n"
           "We create entries with varying ages to observe which survive collection."
        ),
        code(
            "store = InMemoryEntryStore()\n"
            "now = datetime.now(timezone.utc)\n"
            "\n"
            "entries = [\n"
            "    MemoryEntry(\n"
            "        entry_id=\"mem-fresh\",\n"
            "        content=\"I just learned Python basics.\",\n"
            "        created_at=now - timedelta(hours=1),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"mem-day-old\",\n"
            "        content=\"Yesterday I set up my dev environment.\",\n"
            "        created_at=now - timedelta(hours=24),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"mem-week-old\",\n"
            "        content=\"Last week I started the ML course.\",\n"
            "        created_at=now - timedelta(hours=168),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"mem-month-old\",\n"
            "        content=\"A month ago I explored Anchor.\",\n"
            "        created_at=now - timedelta(hours=720),\n"
            "    ),\n"
            "    MemoryEntry(\n"
            "        entry_id=\"mem-ancient\",\n"
            "        content=\"Long ago I tried assembly programming.\",\n"
            "        created_at=now - timedelta(hours=2160),\n"
            "    ),\n"
            "]\n"
            "\n"
            "for entry in entries:\n"
            "    store.put(entry)\n"
            "\n"
            "print(f\"Store contains {len(entries)} entries:\")\n"
            "for e in entries:\n"
            "    age_hours = (now - e.created_at).total_seconds() / 3600\n"
            "    print(f\"  {e.entry_id:<16} age={age_hours:>7.0f}h  '{e.content[:40]}...'\")"
        ),
        md("## Create the Garbage Collector\n"
           "We use `LinearDecay` with a 24-hour half-life for aggressive cleanup."
        ),
        code(
            "decay = LinearDecay(half_life_hours=24.0)\n"
            "gc = MemoryGarbageCollector(store=store, decay=decay)\n"
            "\n"
            "print(f\"Decay model: LinearDecay (half_life={decay.half_life_hours}h)\")"
        ),
        md("## Dry Run\n"
           "Preview what would be collected without actually deleting anything."
        ),
        code(
            "stats = gc.collect(retention_threshold=0.1, dry_run=True)\n"
            "\n"
            "print(\"=== Dry Run Results ===\")\n"
            "print(f\"  Expired pruned:   {stats.expired_pruned}\")\n"
            "print(f\"  Decayed pruned:   {stats.decayed_pruned}\")\n"
            "print(f\"  Total remaining:  {stats.total_remaining}\")"
        ),
        md("## Actual Collection\n"
           "Run GC for real and confirm the store size shrank."
        ),
        code(
            "stats = gc.collect(retention_threshold=0.1, dry_run=False)\n"
            "\n"
            "print(\"=== GC Results ===\")\n"
            "print(f\"  Expired pruned:   {stats.expired_pruned}\")\n"
            "print(f\"  Decayed pruned:   {stats.decayed_pruned}\")\n"
            "print(f\"  Total remaining:  {stats.total_remaining}\")"
        ),
        md("## Inspect Surviving Entries"),
        code(
            "print(\"Entries that survived GC:\")\n"
            "print(f\"  (stats.total_remaining = {stats.total_remaining})\")\n"
            "print()\n"
            "print(\"Fresh entries with high retention scores survive.\")\n"
            "print(\"Old entries with low retention scores are pruned.\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MemoryGarbageCollector` runs in two phases: expired then decayed\n"
            "- Always run with `dry_run=True` first to preview impact\n"
            "- `retention_threshold` controls how aggressively decayed entries are pruned\n"
            "- Pair with `LinearDecay` or `EbbinghausDecay` to tune the forgetting curve\n"
            "- Schedule GC periodically to keep your memory store lean\n"
            "\n"
            "**Back to:** [Memory README](README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Memory notebooks in {OUTPUT_DIR}/\n")

    nb_sliding_window()
    nb_summary_buffer()
    nb_graph_memory()
    nb_memory_manager()
    nb_eviction_policies()
    nb_decay_strategies()
    nb_consolidation()
    nb_garbage_collection()

    print(f"\nDone. 8 notebooks created.")


if __name__ == "__main__":
    main()
