"""Generate all 4 Jupyter notebooks for the Tokens module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "10-tokens"))
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
# 01 - Token Budgets
# ---------------------------------------------------------------------------
def nb_token_budgets():
    write_notebook("01_token_budgets.ipynb", [
        md(
            "# Token Budgets\n"
            "> Allocate and manage token limits across different context sources.\n"
            "\n"
            "`TokenBudget` lets you define a total token ceiling and carve it into\n"
            "named allocations (system prompt, memory, retrieval, etc.) with individual\n"
            "limits, priorities, and overflow strategies.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.models import TokenBudget, BudgetAllocation, SourceType"
        ),
        md(
            "## Create a Token Budget\n"
            "Define a budget with three allocations and a reserve for the model\u2019s output."
        ),
        code(
            "budget = TokenBudget(\n"
            "    total_tokens=8192,\n"
            "    allocations=[\n"
            "        BudgetAllocation(\n"
            "            source=SourceType.SYSTEM,\n"
            "            max_tokens=500,\n"
            "            priority=10,\n"
            "            overflow_strategy=\"truncate\",\n"
            "        ),\n"
            "        BudgetAllocation(\n"
            "            source=SourceType.MEMORY,\n"
            "            max_tokens=2000,\n"
            "            priority=7,\n"
            "            overflow_strategy=\"truncate\",\n"
            "        ),\n"
            "        BudgetAllocation(\n"
            "            source=SourceType.RETRIEVAL,\n"
            "            max_tokens=4000,\n"
            "            priority=5,\n"
            "            overflow_strategy=\"drop\",\n"
            "        ),\n"
            "    ],\n"
            "    reserve_tokens=200,\n"
            ")\n"
            "\n"
            "print(f\"Total budget: {budget.total_tokens} tokens\")\n"
            "print(f\"Reserve:      {budget.reserve_tokens} tokens\")\n"
            "print(f\"Allocations:  {len(budget.allocations)}\")"
        ),
        md(
            "## Query Individual Allocations\n"
            "Use `.get_allocation()` to inspect limits for a specific source."
        ),
        code(
            "alloc = budget.get_allocation(SourceType.MEMORY)\n"
            "\n"
            "print(f\"Source:   {alloc.source.value}\")\n"
            "print(f\"Max:      {alloc.max_tokens} tokens\")\n"
            "print(f\"Priority: {alloc.priority}\")\n"
            "print(f\"Overflow: {alloc.overflow_strategy}\")"
        ),
        md(
            "## Overflow Strategies\n"
            "Each allocation declares what happens when its content exceeds the limit:\n"
            "- `\"truncate\"` \u2013 cut content to fit\n"
            "- `\"drop\"` \u2013 discard the entire block"
        ),
        code(
            "strategy = budget.get_overflow_strategy(SourceType.RETRIEVAL)\n"
            "print(f\"Retrieval overflow strategy: {strategy}\")\n"
            "\n"
            "strategy_sys = budget.get_overflow_strategy(SourceType.SYSTEM)\n"
            "print(f\"System overflow strategy:    {strategy_sys}\")"
        ),
        md(
            "## Shared Pool\n"
            "Tokens not assigned to any allocation form a shared pool available to\n"
            "lower-priority sources."
        ),
        code(
            "shared = budget.shared_pool\n"
            "\n"
            "allocated = sum(a.max_tokens for a in budget.allocations)\n"
            "print(f\"Allocated tokens: {allocated}\")\n"
            "print(f\"Reserve tokens:   {budget.reserve_tokens}\")\n"
            "print(f\"Shared pool:      {shared} tokens\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `TokenBudget` enforces a hard ceiling on total context size\n"
            "- `BudgetAllocation` assigns per-source limits with priority ordering\n"
            "- `overflow_strategy` controls truncation vs. dropping behavior\n"
            "- The shared pool absorbs leftover capacity for flexible use\n"
            "\n"
            "**Next:** [Budget Presets](02_budget_presets.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Budget Presets
# ---------------------------------------------------------------------------
def nb_budget_presets():
    write_notebook("02_budget_presets.ipynb", [
        md(
            "# Budget Presets\n"
            "> Ready-made token budgets for common application patterns.\n"
            "\n"
            "Anchor ships three factory functions that create sensible default budgets\n"
            "for chat, RAG, and agent workloads.  Use them as starting points and\n"
            "customize allocations as needed.\n"
            "\n"
            "**Time:** ~3 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.models import (\n"
            "    default_chat_budget,\n"
            "    default_rag_budget,\n"
            "    default_agent_budget,\n"
            ")"
        ),
        md(
            "## Chat Budget (8K tokens)\n"
            "Optimized for conversational assistants with moderate context."
        ),
        code(
            "chat = default_chat_budget()\n"
            "\n"
            "print(f\"Chat budget: {chat.total_tokens} tokens\")\n"
            "print(f\"Reserve:     {chat.reserve_tokens} tokens\")\n"
            "print(f\"Allocations:\")\n"
            "for alloc in chat.allocations:\n"
            "    print(f\"  {alloc.source.value:12s}  max={alloc.max_tokens:>5}  \"\n"
            "          f\"priority={alloc.priority}  overflow={alloc.overflow_strategy}\")"
        ),
        md(
            "## RAG Budget (12K tokens)\n"
            "Larger retrieval allocation for knowledge-grounded answers."
        ),
        code(
            "rag = default_rag_budget()\n"
            "\n"
            "print(f\"RAG budget: {rag.total_tokens} tokens\")\n"
            "print(f\"Reserve:    {rag.reserve_tokens} tokens\")\n"
            "print(f\"Allocations:\")\n"
            "for alloc in rag.allocations:\n"
            "    print(f\"  {alloc.source.value:12s}  max={alloc.max_tokens:>5}  \"\n"
            "          f\"priority={alloc.priority}  overflow={alloc.overflow_strategy}\")"
        ),
        md(
            "## Agent Budget (16K tokens)\n"
            "Extra room for tool results, planning traces, and multi-step reasoning."
        ),
        code(
            "agent = default_agent_budget()\n"
            "\n"
            "print(f\"Agent budget: {agent.total_tokens} tokens\")\n"
            "print(f\"Reserve:      {agent.reserve_tokens} tokens\")\n"
            "print(f\"Allocations:\")\n"
            "for alloc in agent.allocations:\n"
            "    print(f\"  {alloc.source.value:12s}  max={alloc.max_tokens:>5}  \"\n"
            "          f\"priority={alloc.priority}  overflow={alloc.overflow_strategy}\")"
        ),
        md(
            "## Compare All Three\n"
            "Quick side-by-side summary."
        ),
        code(
            "for name, b in [(\"chat\", chat), (\"rag\", rag), (\"agent\", agent)]:\n"
            "    sources = \", \".join(a.source.value for a in b.allocations)\n"
            "    print(f\"{name:6s}  total={b.total_tokens:>6}  \"\n"
            "          f\"reserve={b.reserve_tokens:>4}  sources=[{sources}]\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `default_chat_budget()` \u2192 8K tokens, balanced for conversation\n"
            "- `default_rag_budget()` \u2192 12K tokens, heavy retrieval allocation\n"
            "- `default_agent_budget()` \u2192 16K tokens, room for tool results\n"
            "- All presets return a regular `TokenBudget` you can modify after creation\n"
            "\n"
            "**Next:** [Tiktoken Counter](03_tiktoken_counter.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Tiktoken Counter
# ---------------------------------------------------------------------------
def nb_tiktoken_counter():
    write_notebook("03_tiktoken_counter.ipynb", [
        md(
            "# Tiktoken Counter\n"
            "> Accurate token counting and truncation using OpenAI\u2019s tiktoken.\n"
            "\n"
            "`TiktokenCounter` wraps the tiktoken library to provide fast, accurate\n"
            "token counting compatible with GPT-3.5/4 models.  Anchor also exposes a\n"
            "convenience function `get_default_counter()` for the most common encoding.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.tokens import TiktokenCounter, get_default_counter"
        ),
        md(
            "## Create a Counter\n"
            "Specify the tiktoken encoding name (e.g., `cl100k_base` for GPT-4)."
        ),
        code(
            "counter = TiktokenCounter(encoding_name=\"cl100k_base\")\n"
            "\n"
            "print(f\"Encoding: cl100k_base\")\n"
            "print(f\"Type:     {type(counter).__name__}\")"
        ),
        md(
            "## Count Tokens\n"
            "Pass any string to `.count_tokens()` to get the exact token count."
        ),
        code(
            "text = \"Hello, world!\"\n"
            "count = counter.count_tokens(text)\n"
            "\n"
            "print(f\"Text:   '{text}'\")\n"
            "print(f\"Tokens: {count}\")\n"
            "\n"
            "# Try a longer string\n"
            "long_text = \"The quick brown fox jumps over the lazy dog. \" * 5\n"
            "print(f\"\\nLong text tokens: {counter.count_tokens(long_text)}\")"
        ),
        md(
            "## Truncate to Token Limit\n"
            "`.truncate_to_tokens()` cuts text so it fits within a maximum token count."
        ),
        code(
            "original = \"This is a longer sentence that we want to truncate to a small number of tokens.\"\n"
            "\n"
            "truncated = counter.truncate_to_tokens(original, max_tokens=5)\n"
            "\n"
            "print(f\"Original ({counter.count_tokens(original)} tokens):\")\n"
            "print(f\"  {original}\")\n"
            "print(f\"\\nTruncated ({counter.count_tokens(truncated)} tokens):\")\n"
            "print(f\"  {truncated}\")"
        ),
        md(
            "## Default Counter\n"
            "`get_default_counter()` returns a pre-configured counter suitable for most\n"
            "use cases."
        ),
        code(
            "default = get_default_counter()\n"
            "\n"
            "print(f\"Default counter type: {type(default).__name__}\")\n"
            "print(f\"Token count test:     {default.count_tokens('Anchor is great!')}\")"
        ),
        md(
            "## Counting Different Content Types\n"
            "Token counts vary by content \u2014 code, prose, and special characters all\n"
            "tokenize differently."
        ),
        code(
            "samples = [\n"
            "    \"Hello, world!\",\n"
            "    \"def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\",\n"
            "    \"\u3053\u3093\u306b\u3061\u306f\u4e16\u754c\",\n"
            "    \"https://docs.anchor.dev/api/tokens\",\n"
            "]\n"
            "\n"
            "for s in samples:\n"
            "    tokens = counter.count_tokens(s)\n"
            "    print(f\"  {tokens:>3} tokens \u2502 {s[:50]}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `TiktokenCounter` gives exact token counts matching OpenAI models\n"
            "- `.truncate_to_tokens()` safely trims content to a budget\n"
            "- `get_default_counter()` is the quickest way to get started\n"
            "- Token counts vary by language, formatting, and content type\n"
            "\n"
            "**Next:** [Custom Tokenizer](04_custom_tokenizer.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Custom Tokenizer
# ---------------------------------------------------------------------------
def nb_custom_tokenizer():
    write_notebook("04_custom_tokenizer.ipynb", [
        md(
            "# Custom Tokenizer\n"
            "> Implement the `Tokenizer` protocol with your own counting logic.\n"
            "\n"
            "Anchor defines a `Tokenizer` protocol with two methods: `count_tokens()`\n"
            "and `truncate_to_tokens()`.  Any class that implements these methods can be\n"
            "used anywhere Anchor expects a tokenizer \u2014 no inheritance required.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.protocols import Tokenizer"
        ),
        md(
            "## The Tokenizer Protocol\n"
            "The protocol requires exactly two methods:\n"
            "- `count_tokens(text: str) -> int`\n"
            "- `truncate_to_tokens(text: str, max_tokens: int) -> str`"
        ),
        code(
            "# Inspect the protocol\n"
            "print(\"Tokenizer protocol methods:\")\n"
            "for name in [\"count_tokens\", \"truncate_to_tokens\"]:\n"
            "    print(f\"  - {name}\")"
        ),
        md(
            "## Build a Simple Word-Based Tokenizer\n"
            "A lightweight tokenizer that counts whitespace-separated words."
        ),
        code(
            "class SimpleTokenizer:\n"
            "    \"\"\"Counts tokens as whitespace-separated words.\"\"\"\n"
            "\n"
            "    def count_tokens(self, text: str) -> int:\n"
            "        return len(text.split())\n"
            "\n"
            "    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:\n"
            "        words = text.split()\n"
            "        return \" \".join(words[:max_tokens])\n"
            "\n"
            "\n"
            "tokenizer = SimpleTokenizer()\n"
            "print(f\"Type: {type(tokenizer).__name__}\")"
        ),
        md(
            "## Verify Protocol Conformance\n"
            "`isinstance()` checks work with `runtime_checkable` protocols."
        ),
        code(
            "is_valid = isinstance(tokenizer, Tokenizer)\n"
            "print(f\"Satisfies Tokenizer protocol: {is_valid}\")\n"
            "\n"
            "assert is_valid, \"SimpleTokenizer must satisfy the Tokenizer protocol\""
        ),
        md(
            "## Use the Custom Tokenizer\n"
            "Exercise both methods with sample text."
        ),
        code(
            "text = \"The quick brown fox jumps over the lazy dog\"\n"
            "\n"
            "count = tokenizer.count_tokens(text)\n"
            "print(f\"Text:       '{text}'\")\n"
            "print(f\"Word count: {count}\")\n"
            "\n"
            "truncated = tokenizer.truncate_to_tokens(text, max_tokens=5)\n"
            "print(f\"\\nTruncated to 5 tokens: '{truncated}'\")\n"
            "print(f\"Truncated count:       {tokenizer.count_tokens(truncated)}\")"
        ),
        md(
            "## Build a Character-Based Tokenizer\n"
            "Another custom implementation \u2014 this time counting individual characters."
        ),
        code(
            "class CharTokenizer:\n"
            "    \"\"\"Counts tokens as individual characters.\"\"\"\n"
            "\n"
            "    def count_tokens(self, text: str) -> int:\n"
            "        return len(text)\n"
            "\n"
            "    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:\n"
            "        return text[:max_tokens]\n"
            "\n"
            "\n"
            "char_tok = CharTokenizer()\n"
            "\n"
            "print(f\"Satisfies Tokenizer: {isinstance(char_tok, Tokenizer)}\")\n"
            "print(f\"Char count of 'Hello': {char_tok.count_tokens('Hello')}\")\n"
            "print(f\"Truncated to 3: '{char_tok.truncate_to_tokens('Hello', 3)}'\")"
        ),
        md(
            "## Key Takeaways\n"
            "- The `Tokenizer` protocol is structural \u2014 implement two methods, no base class\n"
            "- `isinstance()` verification works at runtime\n"
            "- Custom tokenizers plug into budgets, counters, and pipelines seamlessly\n"
            "- Word-based, character-based, or any counting scheme can be used\n"
            "\n"
            "**Previous:** [Tiktoken Counter](03_tiktoken_counter.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Tokens notebooks in {OUTPUT_DIR} ...")
    nb_token_budgets()
    nb_budget_presets()
    nb_tiktoken_counter()
    nb_custom_tokenizer()
    print("Done \u2714")


if __name__ == "__main__":
    main()
