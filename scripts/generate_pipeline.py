"""Generate the 7 Pipeline module notebooks for the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "00-pipeline"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

KERNEL_META = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def make_nb(cells):
    """Create a notebook with standard kernel metadata."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = KERNEL_META
    nb.cells = cells
    return nb


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


def write_nb(nb, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    nbformat.write(nb, path)
    print(f"  Created {path}")


# ---------------------------------------------------------------------------
# 01 - Basic Pipeline
# ---------------------------------------------------------------------------
def nb_01():
    return make_nb([
        md(
            "# Recipe 01 — Basic Pipeline\n"
            "> Build a complete context window from scratch in under 20 lines.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline` |\n"
            "| **Key classes** | `ContextPipeline`, `QueryBundle`, `SlidingWindowMemory`, `GenericTextFormatter` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "from anchor.pipeline import ContextPipeline\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType\n"
            "from anchor.formatters import GenericTextFormatter\n"
            "from anchor.memory import SlidingWindowMemory"
        ),
        md("## 1 — Create the pipeline\n"
           "A `ContextPipeline` manages the full lifecycle of building a context window.\n"
           "Set `max_tokens` to cap the total budget available for context."),
        code(
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "print(f\"Pipeline created with {pipeline.max_tokens} token budget\")"
        ),
        md("## 2 — Add a system prompt\n"
           "The system prompt is always placed first and counts against the token budget."),
        code(
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "print(\"System prompt added\")"
        ),
        md("## 3 — Attach a formatter\n"
           "`GenericTextFormatter` converts context items into plain-text blocks\n"
           "suitable for any LLM."),
        code(
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "print(\"Formatter attached\")"
        ),
        md("## 4 — Wire up conversation memory\n"
           "`SlidingWindowMemory` keeps the most recent turns that fit within its\n"
           "own token budget."),
        code(
            "memory = SlidingWindowMemory(max_tokens=2048)\n"
            "memory.add_turn(\"user\", \"What is context engineering?\")\n"
            "memory.add_turn(\"assistant\", \"Context engineering is the practice of \"\n"
            "                \"curating the right information into an LLM's context window.\")\n"
            "pipeline.with_memory(memory)\n"
            "print(f\"Memory has {len(memory.get_turns())} turns\")"
        ),
        md("## 5 — Build the context window\n"
           "Call `pipeline.build()` with a `QueryBundle` to assemble the final window."),
        code(
            "query = QueryBundle(query_str=\"Tell me more about it\")\n"
            "result = pipeline.build(query)\n"
            "\n"
            "print(f\"Items in window : {len(result.window.items)}\")\n"
            "print(f\"Tokens used     : {result.window.used_tokens}\")\n"
            "print(f\"Utilization     : {result.window.utilization:.1%}\")"
        ),
        md("## 6 — Inspect the formatted output\n"
           "The `formatted_output` string is what you would pass to your LLM."),
        code(
            "output = result.formatted_output\n"
            "preview = output[:300] if isinstance(output, str) else str(output)[:300]\n"
            "print(preview)"
        ),
        md(
            "## Key Takeaways\n"
            "- `ContextPipeline(max_tokens=N)` is the entry point for every context build.\n"
            "- Attach a **formatter**, **memory**, and optional **steps** before calling `build()`.\n"
            "- `result.window` gives token-level visibility into what was included.\n"
            "\n"
            "**Next:** [Built-in Steps](02_builtin_steps.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Built-in Steps
# ---------------------------------------------------------------------------
def nb_02():
    return make_nb([
        md(
            "# Recipe 02 — Built-in Steps\n"
            "> Explore the ready-made pipeline steps that ship with Anchor.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline.steps` |\n"
            "| **Key classes** | `retriever_step`, `filter_step`, `reranker_step`, `query_transform_step` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "from anchor.pipeline import ContextPipeline\n"
            "from anchor.pipeline.steps import (\n"
            "    retriever_step,\n"
            "    filter_step,\n"
            "    postprocessor_step,\n"
            "    reranker_step,\n"
            "    auto_promotion_step,\n"
            "    graph_retrieval_step,\n"
            "    classified_retriever_step,\n"
            "    query_transform_step,\n"
            ")\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType"
        ),
        md("## 1 — Mock helpers\n"
           "We create lightweight mocks so every cell runs without external services."),
        code(
            "# --- mock retriever ---\n"
            "class MockRetriever:\n"
            "    def retrieve(self, query, top_k=5):\n"
            "        return [\n"
            "            ContextItem(\n"
            "                id=f\"doc-{i}\",\n"
            "                content=f\"Document {i} about {query.query_str}\",\n"
            "                source=SourceType.RETRIEVAL,\n"
            "                score=round(0.95 - i * 0.1, 2),\n"
            "                token_count=15,\n"
            "            )\n"
            "            for i in range(top_k)\n"
            "        ]\n"
            "\n"
            "# --- mock reranker ---\n"
            "class MockReranker:\n"
            "    def rerank(self, items, query, top_k=3):\n"
            "        scored = sorted(items, key=lambda x: x.score or 0, reverse=True)\n"
            "        return scored[:top_k]\n"
            "\n"
            "retriever = MockRetriever()\n"
            "reranker  = MockReranker()\n"
            "print(\"Mocks ready\")"
        ),
        md("## 2 — `retriever_step`\n"
           "Wraps any retriever into a pipeline step. The step calls\n"
           "`retriever.retrieve(query, top_k=...)` and feeds items downstream."),
        code(
            "step = retriever_step(name=\"retriever\", retriever=retriever, top_k=10)\n"
            "print(f\"Step name: {step.name}\")"
        ),
        md("## 3 — `filter_step`\n"
           "Accepts a predicate function `(items, query) -> items` to prune results."),
        code(
            "step = filter_step(\n"
            "    name=\"quality_filter\",\n"
            "    filter_fn=lambda items, q: [i for i in items if (i.score or 0) > 0.5],\n"
            ")\n"
            "print(f\"Step name: {step.name}\")"
        ),
        md("## 4 — `reranker_step`\n"
           "Re-scores items with a reranker model and keeps only `top_k`."),
        code(
            "step = reranker_step(name=\"rerank\", reranker=reranker, top_k=5)\n"
            "print(f\"Step name: {step.name}\")"
        ),
        md("## 5 — `query_transform_step`\n"
           "Rewrites or expands the query before retrieval."),
        code(
            "class MockTransformer:\n"
            "    def transform(self, query):\n"
            "        return QueryBundle(query_str=f\"expanded: {query.query_str}\")\n"
            "\n"
            "step = query_transform_step(name=\"expand\", transformer=MockTransformer())\n"
            "print(f\"Step name: {step.name}\")"
        ),
        md("## 6 — Assemble a full pipeline\n"
           "Chain multiple built-in steps into a single pipeline."),
        code(
            "from anchor.formatters import GenericTextFormatter\n"
            "\n"
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "\n"
            "pipeline.add_step(retriever_step(\"docs\", retriever=retriever, top_k=10))\n"
            "pipeline.add_step(filter_step(\n"
            "    \"quality\",\n"
            "    filter_fn=lambda items, q: [i for i in items if (i.score or 0) > 0.5],\n"
            "))\n"
            "pipeline.add_step(reranker_step(\"rerank\", reranker=reranker, top_k=5))\n"
            "\n"
            "query = QueryBundle(query_str=\"context engineering best practices\")\n"
            "result = pipeline.build(query)\n"
            "\n"
            "print(f\"Items included : {len(result.window.items)}\")\n"
            "print(f\"Token usage    : {result.window.used_tokens}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- Built-in step factories (`retriever_step`, `filter_step`, etc.) cover the most common patterns.\n"
            "- Steps compose left-to-right: each step receives items from the previous one.\n"
            "- Use mock objects during development to iterate quickly.\n"
            "\n"
            "**Next:** [Custom Steps](03_custom_steps.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Custom Steps
# ---------------------------------------------------------------------------
def nb_03():
    return make_nb([
        md(
            "# Recipe 03 — Custom Steps\n"
            "> Write your own pipeline step to implement any transformation logic.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline` |\n"
            "| **Key classes** | `PipelineStep`, `ContextPipeline` |\n"
            "| **Difficulty** | Intermediate |"
        ),
        code(
            "from anchor.pipeline import ContextPipeline, PipelineStep\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType\n"
            "from anchor.formatters import GenericTextFormatter"
        ),
        md("## 1 — Define a custom step function\n"
           "A step function takes `(items, query, **kwargs)` and returns a filtered\n"
           "or transformed list of `ContextItem` objects."),
        code(
            "def score_filter(items, query, **kwargs):\n"
            "    \"\"\"Keep only items with score above 0.3.\"\"\"\n"
            "    return [item for item in items if (item.score or 0) > 0.3]\n"
            "\n"
            "print(f\"Function name: {score_filter.__name__}\")"
        ),
        md("## 2 — Wrap it in a `PipelineStep`\n"
           "Give the step a name so diagnostics can track it."),
        code(
            "step = PipelineStep(name=\"score_filter\", fn=score_filter)\n"
            "print(f\"Step: {step.name}\")"
        ),
        md("## 3 — Create sample data\n"
           "We generate items with varying scores to see the filter in action."),
        code(
            "items = [\n"
            "    ContextItem(id=\"a\", content=\"High quality result\",\n"
            "               source=SourceType.RETRIEVAL, score=0.9, token_count=8),\n"
            "    ContextItem(id=\"b\", content=\"Medium quality result\",\n"
            "               source=SourceType.RETRIEVAL, score=0.5, token_count=8),\n"
            "    ContextItem(id=\"c\", content=\"Low quality result\",\n"
            "               source=SourceType.RETRIEVAL, score=0.1, token_count=8),\n"
            "]\n"
            "print(f\"Created {len(items)} test items\")"
        ),
        md("## 4 — A content-dedup step\n"
           "Custom steps can implement any logic. Here we deduplicate by content hash."),
        code(
            "import hashlib\n"
            "\n"
            "def dedup_step(items, query, **kwargs):\n"
            "    \"\"\"Remove duplicate items based on content hash.\"\"\"\n"
            "    seen = set()\n"
            "    unique = []\n"
            "    for item in items:\n"
            "        h = hashlib.md5(item.content.encode()).hexdigest()\n"
            "        if h not in seen:\n"
            "            seen.add(h)\n"
            "            unique.append(item)\n"
            "    return unique\n"
            "\n"
            "print(\"Dedup step defined\")"
        ),
        md("## 5 — A priority-boost step\n"
           "Boost priority for items whose content matches a keyword."),
        code(
            "def keyword_boost(items, query, **kwargs):\n"
            "    \"\"\"Boost priority when content contains the query string.\"\"\"\n"
            "    boosted = []\n"
            "    for item in items:\n"
            "        if query.query_str.lower() in item.content.lower():\n"
            "            boosted.append(ContextItem(\n"
            "                id=item.id, content=item.content,\n"
            "                source=item.source, score=item.score,\n"
            "                priority=(item.priority or 0) + 10,\n"
            "                token_count=item.token_count,\n"
            "            ))\n"
            "        else:\n"
            "            boosted.append(item)\n"
            "    return boosted\n"
            "\n"
            "print(\"Keyword boost step defined\")"
        ),
        md("## 6 — Plug custom steps into a pipeline\n"
           "Custom steps compose just like built-in ones."),
        code(
            "from anchor.pipeline.steps import retriever_step\n"
            "\n"
            "# Mock retriever returning our sample items\n"
            "class SampleRetriever:\n"
            "    def retrieve(self, query, top_k=5):\n"
            "        return [\n"
            "            ContextItem(id=f\"doc-{i}\", content=f\"Doc about {query.query_str} #{i}\",\n"
            "                        source=SourceType.RETRIEVAL, score=round(0.9 - i*0.2, 2),\n"
            "                        token_count=12)\n"
            "            for i in range(top_k)\n"
            "        ]\n"
            "\n"
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "\n"
            "pipeline.add_step(retriever_step(\"fetch\", retriever=SampleRetriever(), top_k=5))\n"
            "pipeline.add_step(PipelineStep(name=\"score_filter\", fn=score_filter))\n"
            "pipeline.add_step(PipelineStep(name=\"dedup\", fn=dedup_step))\n"
            "pipeline.add_step(PipelineStep(name=\"boost\", fn=keyword_boost))\n"
            "\n"
            "query = QueryBundle(query_str=\"Doc\")\n"
            "result = pipeline.build(query)\n"
            "\n"
            "print(f\"Items after all steps: {len(result.window.items)}\")\n"
            "for item in result.window.items:\n"
            "    print(f\"  {item.id}  score={item.score}  priority={item.priority}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- Any `(items, query, **kwargs) -> list[ContextItem]` function is a valid step.\n"
            "- Wrap it with `PipelineStep(name=..., fn=...)` and call `pipeline.add_step()`.\n"
            "- Custom steps compose seamlessly with built-in steps.\n"
            "\n"
            "**Next:** [Async Pipelines](04_async_pipelines.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Async Pipelines
# ---------------------------------------------------------------------------
def nb_04():
    return make_nb([
        md(
            "# Recipe 04 — Async Pipelines\n"
            "> Run pipeline builds asynchronously for I/O-bound workloads.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline` |\n"
            "| **Key classes** | `ContextPipeline.abuild` |\n"
            "| **Difficulty** | Intermediate |"
        ),
        code(
            "import asyncio\n"
            "from anchor.pipeline import ContextPipeline\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType\n"
            "from anchor.formatters import GenericTextFormatter\n"
            "from anchor.memory import SlidingWindowMemory"
        ),
        md("## 1 — Why async?\n"
           "When pipeline steps call external services (vector DBs, reranker APIs),\n"
           "an async build avoids blocking the event loop and enables concurrency."),
        md("## 2 — Set up the pipeline (same as sync)\n"
           "Configuration is identical — only the final `build` call changes."),
        code(
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "\n"
            "memory = SlidingWindowMemory(max_tokens=1024)\n"
            "memory.add_turn(\"user\", \"What is async programming?\")\n"
            "memory.add_turn(\"assistant\", \"Async programming lets you run I/O tasks concurrently.\")\n"
            "pipeline.with_memory(memory)\n"
            "\n"
            "print(\"Pipeline configured\")"
        ),
        md("## 3 — Build asynchronously\n"
           "Use `await pipeline.abuild(query)` inside an async function.\n"
           "In a Jupyter notebook you can `await` directly at the top level."),
        code(
            "async def build_context():\n"
            "    query = QueryBundle(query_str=\"Tell me more\")\n"
            "    result = await pipeline.abuild(query)\n"
            "    return result\n"
            "\n"
            "# In a notebook you can simply: result = await pipeline.abuild(query)\n"
            "# Outside notebooks use asyncio.run():\n"
            "result = asyncio.run(build_context())\n"
            "\n"
            "print(f\"Items   : {len(result.window.items)}\")\n"
            "print(f\"Tokens  : {result.window.used_tokens}\")\n"
            "print(f\"Utilization: {result.window.utilization:.1%}\")"
        ),
        md("## 4 — Parallel builds\n"
           "You can build multiple context windows concurrently with `asyncio.gather`."),
        code(
            "async def parallel_builds():\n"
            "    queries = [\n"
            "        QueryBundle(query_str=\"What is RAG?\"),\n"
            "        QueryBundle(query_str=\"Explain embeddings\"),\n"
            "        QueryBundle(query_str=\"Token budgeting tips\"),\n"
            "    ]\n"
            "    results = await asyncio.gather(\n"
            "        *(pipeline.abuild(q) for q in queries)\n"
            "    )\n"
            "    return results\n"
            "\n"
            "results = asyncio.run(parallel_builds())\n"
            "\n"
            "for i, r in enumerate(results):\n"
            "    print(f\"Build {i}: {r.window.used_tokens} tokens, \"\n"
            "          f\"{len(r.window.items)} items\")"
        ),
        md("## 5 — Mixing sync and async steps\n"
           "Anchor automatically wraps synchronous step functions when running\n"
           "inside `abuild`, so you can mix both freely."),
        code(
            "from anchor.pipeline import PipelineStep\n"
            "\n"
            "# Sync step — works inside abuild too\n"
            "def tag_items(items, query, **kwargs):\n"
            "    for item in items:\n"
            "        item.content = f\"[tagged] {item.content}\"\n"
            "    return items\n"
            "\n"
            "pipeline.add_step(PipelineStep(name=\"tagger\", fn=tag_items))\n"
            "\n"
            "result = asyncio.run(build_context())\n"
            "print(f\"Items after async+sync mix: {len(result.window.items)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `await pipeline.abuild(query)` is the async counterpart of `build()`.\n"
            "- Use `asyncio.gather` to build multiple windows in parallel.\n"
            "- Sync step functions work inside `abuild` without changes.\n"
            "\n"
            "**Next:** [Pipeline Diagnostics](05_pipeline_diagnostics.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Pipeline Diagnostics
# ---------------------------------------------------------------------------
def nb_05():
    return make_nb([
        md(
            "# Recipe 05 — Pipeline Diagnostics\n"
            "> Understand exactly what happened during a context build.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline` |\n"
            "| **Key classes** | `PipelineDiagnostics`, `ContextWindow` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "from anchor.pipeline import ContextPipeline, PipelineStep\n"
            "from anchor.pipeline.steps import retriever_step, filter_step\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType\n"
            "from anchor.formatters import GenericTextFormatter\n"
            "from anchor.memory import SlidingWindowMemory"
        ),
        md("## 1 — Build a pipeline with several steps\n"
           "We need a multi-step pipeline to generate interesting diagnostics."),
        code(
            "class MockRetriever:\n"
            "    def retrieve(self, query, top_k=5):\n"
            "        return [\n"
            "            ContextItem(\n"
            "                id=f\"doc-{i}\", content=f\"Result {i}\",\n"
            "                source=SourceType.RETRIEVAL,\n"
            "                score=round(0.95 - i * 0.15, 2),\n"
            "                token_count=10,\n"
            "            )\n"
            "            for i in range(top_k)\n"
            "        ]\n"
            "\n"
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "\n"
            "memory = SlidingWindowMemory(max_tokens=1024)\n"
            "memory.add_turn(\"user\", \"previous question\")\n"
            "pipeline.with_memory(memory)\n"
            "\n"
            "pipeline.add_step(retriever_step(\"fetch\", retriever=MockRetriever(), top_k=8))\n"
            "pipeline.add_step(filter_step(\n"
            "    \"quality\",\n"
            "    filter_fn=lambda items, q: [i for i in items if (i.score or 0) > 0.4],\n"
            "))\n"
            "\n"
            "query = QueryBundle(query_str=\"diagnostics demo\")\n"
            "result = pipeline.build(query)\n"
            "print(\"Pipeline built\")"
        ),
        md("## 2 — Access `result.diagnostics`\n"
           "`diagnostics` is a dict-like `PipelineDiagnostics` object summarizing the\n"
           "entire build."),
        code(
            "diag = result.diagnostics\n"
            "print(f\"Steps executed          : {diag.get('steps')}\")\n"
            "print(f\"Total items considered  : {diag.get('total_items_considered')}\")\n"
            "print(f\"Items included          : {diag.get('items_included')}\")\n"
            "print(f\"Items overflowed        : {diag.get('items_overflow')}\")\n"
            "print(f\"Memory items            : {diag.get('memory_items')}\")"
        ),
        md("## 3 — Token utilization\n"
           "See how efficiently the token budget was used."),
        code(
            "print(f\"Token utilization : {diag.get('token_utilization', 0):.1%}\")\n"
            "\n"
            "usage = diag.get('token_usage_by_source', {})\n"
            "for source, tokens in usage.items():\n"
            "    print(f\"  {source}: {tokens} tokens\")"
        ),
        md("## 4 — Window-level metrics\n"
           "`result.window` exposes raw numbers directly."),
        code(
            "window = result.window\n"
            "print(f\"Max tokens       : {window.max_tokens}\")\n"
            "print(f\"Used tokens      : {window.used_tokens}\")\n"
            "print(f\"Remaining tokens : {window.remaining_tokens}\")\n"
            "print(f\"Utilization      : {window.utilization:.1%}\")\n"
            "print(f\"Item count       : {len(window.items)}\")"
        ),
        md("## 5 — Per-item inspection\n"
           "Walk through included items to understand what made it in."),
        code(
            "for item in result.window.items:\n"
            "    print(f\"  {item.id:8s}  source={item.source}  \"\n"
            "          f\"score={item.score}  tokens={item.token_count}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `result.diagnostics` provides a build-level summary (steps, counts, utilization).\n"
            "- `result.window` gives item-level detail (tokens, scores, priorities).\n"
            "- Use diagnostics during development to tune retrieval and filtering parameters.\n"
            "\n"
            "**Next:** [Context Window](06_context_window.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Context Window
# ---------------------------------------------------------------------------
def nb_06():
    return make_nb([
        md(
            "# Recipe 06 — Context Window\n"
            "> Work directly with the `ContextWindow` for fine-grained token management.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.models` |\n"
            "| **Key classes** | `ContextWindow`, `ContextItem`, `SourceType` |\n"
            "| **Difficulty** | Intermediate |"
        ),
        code(
            "from anchor.models import ContextWindow, ContextItem, SourceType, QueryBundle"
        ),
        md("## 1 — Create a window\n"
           "`ContextWindow` tracks a hard token ceiling. Items are added explicitly."),
        code(
            "window = ContextWindow(max_tokens=4096)\n"
            "print(f\"Capacity       : {window.max_tokens} tokens\")\n"
            "print(f\"Used           : {window.used_tokens}\")\n"
            "print(f\"Remaining      : {window.remaining_tokens}\")"
        ),
        md("## 2 — Add a single item\n"
           "Each `ContextItem` carries its own `token_count`."),
        code(
            "item = ContextItem(\n"
            "    id=\"doc-1\",\n"
            "    content=\"Context engineering is the discipline of curating the right \"\n"
            "            \"information for an LLM's context window.\",\n"
            "    source=SourceType.RETRIEVAL,\n"
            "    score=0.95,\n"
            "    priority=5,\n"
            "    token_count=20,\n"
            ")\n"
            "window.add_item(item)\n"
            "\n"
            "print(f\"Items          : {len(window.items)}\")\n"
            "print(f\"Used tokens    : {window.used_tokens}\")\n"
            "print(f\"Utilization    : {window.utilization:.1%}\")"
        ),
        md("## 3 — Add multiple items\n"
           "Add a batch of items at once."),
        code(
            "batch = [\n"
            "    ContextItem(id=\"doc-2\", content=\"RAG combines retrieval with generation.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.85, priority=3, token_count=12),\n"
            "    ContextItem(id=\"doc-3\", content=\"Token budgets prevent context overflow.\",\n"
            "               source=SourceType.RETRIEVAL, score=0.70, priority=1, token_count=10),\n"
            "    ContextItem(id=\"mem-1\", content=\"User asked about context engineering.\",\n"
            "               source=SourceType.MEMORY, score=None, priority=8, token_count=10),\n"
            "]\n"
            "for b in batch:\n"
            "    window.add_item(b)\n"
            "\n"
            "print(f\"Total items    : {len(window.items)}\")\n"
            "print(f\"Used tokens    : {window.used_tokens}\")"
        ),
        md("## 4 — Priority-based insertion\n"
           "`add_items_by_priority` sorts items by priority and fills until the budget\n"
           "is exhausted — higher-priority items go first."),
        code(
            "window2 = ContextWindow(max_tokens=50)  # tight budget\n"
            "\n"
            "candidates = [\n"
            "    ContextItem(id=\"lo\", content=\"Low priority filler text\",\n"
            "               source=SourceType.RETRIEVAL, score=0.5, priority=1, token_count=20),\n"
            "    ContextItem(id=\"hi\", content=\"Critical info\",\n"
            "               source=SourceType.RETRIEVAL, score=0.9, priority=10, token_count=20),\n"
            "    ContextItem(id=\"mid\", content=\"Moderately useful\",\n"
            "               source=SourceType.RETRIEVAL, score=0.7, priority=5, token_count=20),\n"
            "]\n"
            "\n"
            "window2.add_items_by_priority(candidates)\n"
            "\n"
            "print(f\"Included {len(window2.items)} of {len(candidates)} candidates\")\n"
            "for item in window2.items:\n"
            "    print(f\"  {item.id:5s}  priority={item.priority}  tokens={item.token_count}\")"
        ),
        md("## 5 — Remaining capacity check\n"
           "Always check `remaining_tokens` before manual insertion to avoid overflow."),
        code(
            "print(f\"Window 1 remaining : {window.remaining_tokens}\")\n"
            "print(f\"Window 2 remaining : {window2.remaining_tokens}\")\n"
            "\n"
            "# Safe insertion pattern\n"
            "new_item = ContextItem(\n"
            "    id=\"safe\", content=\"Extra info\",\n"
            "    source=SourceType.RETRIEVAL, score=0.6, priority=2, token_count=15,\n"
            ")\n"
            "if new_item.token_count <= window.remaining_tokens:\n"
            "    window.add_item(new_item)\n"
            "    print(f\"Added '{new_item.id}' — now {window.used_tokens} tokens used\")\n"
            "else:\n"
            "    print(f\"Not enough room for '{new_item.id}'\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `ContextWindow(max_tokens=N)` enforces a hard token ceiling.\n"
            "- `add_items_by_priority()` auto-fills highest-priority items first.\n"
            "- Always check `remaining_tokens` before manual adds.\n"
            "\n"
            "**Next:** [Enrichers](07_enrichers.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 07 - Enrichers
# ---------------------------------------------------------------------------
def nb_07():
    return make_nb([
        md(
            "# Recipe 07 — Enrichers\n"
            "> Augment context and queries with additional information during pipeline execution.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.pipeline.enrichment` |\n"
            "| **Key classes** | `MemoryContextEnricher`, `ContextQueryEnricher` |\n"
            "| **Difficulty** | Intermediate |"
        ),
        code(
            "from anchor.pipeline import ContextPipeline\n"
            "from anchor.pipeline.enrichment import MemoryContextEnricher, ContextQueryEnricher\n"
            "from anchor.models import QueryBundle, ContextItem, SourceType\n"
            "from anchor.formatters import GenericTextFormatter\n"
            "from anchor.memory import SlidingWindowMemory"
        ),
        md("## 1 — What are enrichers?\n"
           "Enrichers inject additional signals into the pipeline:\n"
           "\n"
           "- **MemoryContextEnricher** — injects conversation memory as context items.\n"
           "- **ContextQueryEnricher** — rewrites or augments the query using existing context."),
        md("## 2 — MemoryContextEnricher\n"
           "Automatically converts memory turns into `ContextItem` objects so they\n"
           "compete for space alongside retrieved documents."),
        code(
            "memory = SlidingWindowMemory(max_tokens=1024)\n"
            "memory.add_turn(\"user\", \"What is RAG?\")\n"
            "memory.add_turn(\"assistant\", \"RAG stands for Retrieval-Augmented Generation. \"\n"
            "                \"It combines a retriever with a language model.\")\n"
            "memory.add_turn(\"user\", \"How does the retriever work?\")\n"
            "\n"
            "enricher = MemoryContextEnricher()\n"
            "print(f\"Enricher type: {type(enricher).__name__}\")\n"
            "print(f\"Memory turns : {len(memory.get_turns())}\")"
        ),
        md("## 3 — Use the enricher in a pipeline\n"
           "Attach memory and the enricher. The enricher runs automatically during `build()`."),
        code(
            "pipeline = ContextPipeline(max_tokens=4096)\n"
            "pipeline.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline.with_formatter(GenericTextFormatter())\n"
            "pipeline.with_memory(memory)\n"
            "\n"
            "query = QueryBundle(query_str=\"Can you explain the retriever component?\")\n"
            "result = pipeline.build(query)\n"
            "\n"
            "print(f\"Items in window : {len(result.window.items)}\")\n"
            "print(f\"Token usage     : {result.window.used_tokens}\")"
        ),
        md("## 4 — ContextQueryEnricher\n"
           "Enriches the query by incorporating existing context — useful for\n"
           "multi-hop retrieval where the refined query depends on what was already found."),
        code(
            "enricher = ContextQueryEnricher()\n"
            "print(f\"Enricher type: {type(enricher).__name__}\")"
        ),
        md("## 5 — Combining enrichers\n"
           "Enrichers can be layered: memory enrichment feeds context to the query\n"
           "enricher for richer downstream retrieval."),
        code(
            "pipeline2 = ContextPipeline(max_tokens=4096)\n"
            "pipeline2.add_system_prompt(\"You are a helpful assistant.\")\n"
            "pipeline2.with_formatter(GenericTextFormatter())\n"
            "pipeline2.with_memory(memory)\n"
            "\n"
            "query = QueryBundle(query_str=\"What about reranking?\")\n"
            "result = pipeline2.build(query)\n"
            "\n"
            "print(f\"Items     : {len(result.window.items)}\")\n"
            "print(f\"Tokens    : {result.window.used_tokens}\")\n"
            "print(f\"Utilization: {result.window.utilization:.1%}\")"
        ),
        md("## 6 — Inspecting enriched items\n"
           "Check which items came from memory versus retrieval by looking at `source`."),
        code(
            "for item in result.window.items:\n"
            "    print(f\"  {item.id:12s}  source={str(item.source):20s}  \"\n"
            "          f\"tokens={item.token_count}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MemoryContextEnricher` converts memory turns into ranked context items.\n"
            "- `ContextQueryEnricher` refines queries using existing context.\n"
            "- Enrichers run automatically during `pipeline.build()` and compose freely.\n"
            "\n"
            "**Next:** Return to the [cookbook index](../README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    notebooks = [
        ("01_basic_pipeline.ipynb", nb_01),
        ("02_builtin_steps.ipynb", nb_02),
        ("03_custom_steps.ipynb", nb_03),
        ("04_async_pipelines.ipynb", nb_04),
        ("05_pipeline_diagnostics.ipynb", nb_05),
        ("06_context_window.ipynb", nb_06),
        ("07_enrichers.ipynb", nb_07),
    ]

    print(f"Generating {len(notebooks)} Pipeline notebooks in {OUTPUT_DIR}/")
    for filename, factory in notebooks:
        nb = factory()
        write_nb(nb, filename)

    print(f"\nDone — {len(notebooks)} notebooks created.")


if __name__ == "__main__":
    main()
