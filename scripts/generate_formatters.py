"""Generate all 4 Jupyter notebooks for the Formatters module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "08-formatters"))
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
# Shared setup snippet used across notebooks 01-03
# ---------------------------------------------------------------------------
SHARED_SETUP = (
    "from anchor.models import ContextWindow, ContextItem, SourceType\n"
    "\n"
    "# Build a sample context window with mixed sources\n"
    "window = ContextWindow(max_tokens=4096)\n"
    "window.add_item(ContextItem(\n"
    "    id=\"sys-1\",\n"
    "    content=\"You are helpful.\",\n"
    "    source=SourceType.SYSTEM,\n"
    "    score=1.0,\n"
    "    priority=10,\n"
    "    token_count=5,\n"
    "))\n"
    "window.add_item(ContextItem(\n"
    "    id=\"doc-1\",\n"
    "    content=\"Python is a language.\",\n"
    "    source=SourceType.RETRIEVAL,\n"
    "    score=0.9,\n"
    "    priority=5,\n"
    "    token_count=6,\n"
    "))\n"
    "\n"
    "print(f\"Context window: {len(window.items)} items, \"\n"
    "      f\"budget {window.max_tokens} tokens\")\n"
    "for item in window.items:\n"
    "    print(f\"  [{item.source.value}] {item.id}: {item.content!r}\")"
)


# ---------------------------------------------------------------------------
# 01 - Anthropic Formatter
# ---------------------------------------------------------------------------
def nb_anthropic_formatter():
    write_notebook("01_anthropic_formatter.ipynb", [
        md(
            "# Anthropic Formatter\n"
            "> Format a ContextWindow into the Anthropic Messages API structure.\n"
            "\n"
            "`AnthropicFormatter` converts your assembled context into the dict\n"
            "format expected by the Anthropic Python SDK (`system` + `messages`).\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.formatters import AnthropicFormatter\n"
            "from anchor.models import ContextWindow, ContextItem, SourceType"
        ),
        md(
            "## Build a Sample Context Window\n"
            "Create a window with a system item and a retrieval item so we can\n"
            "see how each source type maps to the Anthropic format."
        ),
        code(SHARED_SETUP),
        md(
            "## Format for the Anthropic API\n"
            "The formatter exposes a `format_type` property and a `format()` method\n"
            "that returns the provider-specific payload."
        ),
        code(
            "formatter = AnthropicFormatter()\n"
            "print(f\"Format type: {formatter.format_type}\")"
        ),
        code(
            "output = formatter.format(window)\n"
            "print(type(output))\n"
            "print(output)"
        ),
        md(
            "## Inspect the Output Structure\n"
            "The Anthropic format separates system-level content from user/assistant\n"
            "messages, matching the Messages API contract."
        ),
        code(
            "if isinstance(output, dict):\n"
            "    for key, value in output.items():\n"
            "        print(f\"{key}: {value}\")\n"
            "else:\n"
            "    # String output — print directly\n"
            "    print(output)"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `AnthropicFormatter` produces output ready for the Anthropic Messages API.\n"
            "- System-source items are routed to the `system` field.\n"
            "- All other items appear in the `messages` list.\n"
            "- Use `format_type` to identify the formatter at runtime."
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - OpenAI Formatter
# ---------------------------------------------------------------------------
def nb_openai_formatter():
    write_notebook("02_openai_formatter.ipynb", [
        md(
            "# OpenAI Formatter\n"
            "> Format a ContextWindow into the OpenAI Chat Completions structure.\n"
            "\n"
            "`OpenAIFormatter` converts your context into the `messages` list\n"
            "expected by the OpenAI Python SDK.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.formatters import OpenAIFormatter\n"
            "from anchor.models import ContextWindow, ContextItem, SourceType"
        ),
        md("## Build a Sample Context Window"),
        code(SHARED_SETUP),
        md(
            "## Format for the OpenAI API\n"
            "The formatter returns a dict containing a `messages` list with\n"
            "role/content pairs."
        ),
        code(
            "formatter = OpenAIFormatter()\n"
            "print(f\"Format type: {formatter.format_type}\")"
        ),
        code(
            "output = formatter.format(window)\n"
            "print(type(output))\n"
            "print(output)"
        ),
        md(
            "## Walk Through the Messages\n"
            "Each context item is mapped to an appropriate chat role."
        ),
        code(
            "if isinstance(output, dict) and \"messages\" in output:\n"
            "    for i, msg in enumerate(output[\"messages\"]):\n"
            "        print(f\"Message {i}: role={msg.get('role')}, \"\n"
            "              f\"content={msg.get('content')!r}\")\n"
            "else:\n"
            "    print(output)"
        ),
        md(
            "## Compare with Anthropic\n"
            "Anchor's formatters share the same `Formatter` protocol, so\n"
            "switching providers is a one-line change."
        ),
        code(
            "from anchor.formatters import AnthropicFormatter\n"
            "\n"
            "anthropic_out = AnthropicFormatter().format(window)\n"
            "openai_out = OpenAIFormatter().format(window)\n"
            "\n"
            "print(\"Anthropic output type:\", type(anthropic_out).__name__)\n"
            "print(\"OpenAI output type:   \", type(openai_out).__name__)\n"
            "print()\n"
            "print(\"Same Formatter protocol, different providers.\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `OpenAIFormatter` produces a `messages` list compatible with OpenAI Chat Completions.\n"
            "- System items map to the `system` role; retrieval items map to `user`.\n"
            "- Swapping formatters requires no changes to your context pipeline."
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Generic Text Formatter
# ---------------------------------------------------------------------------
def nb_generic_text():
    write_notebook("03_generic_text.ipynb", [
        md(
            "# Generic Text Formatter\n"
            "> Format a ContextWindow as a plain-text string.\n"
            "\n"
            "`GenericTextFormatter` renders context as a human-readable text\n"
            "block. Useful for logging, debugging, or providers that accept\n"
            "raw text prompts.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.formatters import GenericTextFormatter\n"
            "from anchor.models import ContextWindow, ContextItem, SourceType"
        ),
        md("## Build a Sample Context Window"),
        code(SHARED_SETUP),
        md(
            "## Format as Plain Text\n"
            "The output is a single string — no dicts, no nesting."
        ),
        code(
            "formatter = GenericTextFormatter()\n"
            "print(f\"Format type: {formatter.format_type}\")"
        ),
        code(
            "output = formatter.format(window)\n"
            "print(type(output))\n"
            "print()\n"
            "print(output)"
        ),
        md(
            "## Use Cases\n"
            "Plain text is handy for:\n"
            "- **Debugging** — quickly inspect what the model will see.\n"
            "- **Logging** — write context snapshots to log files.\n"
            "- **Simple providers** — any API that takes a single prompt string."
        ),
        code(
            "# Example: write context to a log file\n"
            "import tempfile, os\n"
            "\n"
            "log_path = os.path.join(tempfile.gettempdir(), \"context_debug.txt\")\n"
            "with open(log_path, \"w\") as f:\n"
            "    f.write(output)\n"
            "\n"
            "print(f\"Wrote {len(output)} chars to {log_path}\")\n"
            "print(f\"First 120 chars: {output[:120]!r}...\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `GenericTextFormatter` returns a plain `str`.\n"
            "- Ideal for debugging, logging, or text-only LLM providers.\n"
            "- Same `Formatter` protocol as the provider-specific formatters."
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Custom Formatter
# ---------------------------------------------------------------------------
def nb_custom_formatter():
    write_notebook("04_custom_formatter.ipynb", [
        md(
            "# Custom Formatter\n"
            "> Build your own formatter using the `Formatter` protocol.\n"
            "\n"
            "Anchor's `Formatter` is a Python `Protocol`: any class with\n"
            "`format_type` and `format()` is a valid formatter — no\n"
            "inheritance required.\n"
            "\n"
            "**Time:** ~10 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.formatters import Formatter\n"
            "from anchor.models import ContextWindow, ContextItem, SourceType"
        ),
        md("## Build a Sample Context Window"),
        code(SHARED_SETUP),
        md(
            "## Define an XML Formatter\n"
            "Implement the two required members:\n"
            "- `format_type` (read-only property) — a string identifier.\n"
            "- `format(window)` — accepts a `ContextWindow`, returns any type."
        ),
        code(
            "class XMLFormatter:\n"
            "    \"\"\"Custom formatter that outputs XML.\"\"\"\n"
            "\n"
            "    @property\n"
            "    def format_type(self) -> str:\n"
            "        return \"xml\"\n"
            "\n"
            "    def format(self, window: ContextWindow) -> str:\n"
            "        lines = [\"<context>\"]\n"
            "        for item in window.items:\n"
            "            lines.append(\n"
            "                f'  <item source=\"{item.source.value}\" '\n"
            "                f'priority=\"{item.priority}\">'\n"
            "            )\n"
            "            lines.append(f\"    {item.content}\")\n"
            "            lines.append(\"  </item>\")\n"
            "        lines.append(\"</context>\")\n"
            "        return \"\\n\".join(lines)\n"
            "\n"
            "print(\"XMLFormatter defined.\")"
        ),
        md(
            "## Verify Protocol Compliance\n"
            "`isinstance()` checks work because `Formatter` is a\n"
            "`runtime_checkable` Protocol."
        ),
        code(
            "xml_fmt = XMLFormatter()\n"
            "print(f\"Is a Formatter? {isinstance(xml_fmt, Formatter)}\")\n"
            "print(f\"Format type:   {xml_fmt.format_type}\")"
        ),
        md("## Generate XML Output"),
        code(
            "xml_output = xml_fmt.format(window)\n"
            "print(xml_output)"
        ),
        md(
            "## Another Example: CSV Formatter\n"
            "The protocol is minimal, so you can target any serialisation format."
        ),
        code(
            "class CSVFormatter:\n"
            "    \"\"\"Formatter that outputs comma-separated values.\"\"\"\n"
            "\n"
            "    @property\n"
            "    def format_type(self) -> str:\n"
            "        return \"csv\"\n"
            "\n"
            "    def format(self, window: ContextWindow) -> str:\n"
            "        rows = [\"id,source,priority,content\"]\n"
            "        for item in window.items:\n"
            "            rows.append(\n"
            "                f\"{item.id},{item.source.value},\"\n"
            "                f\"{item.priority},{item.content}\"\n"
            "            )\n"
            "        return \"\\n\".join(rows)\n"
            "\n"
            "csv_fmt = CSVFormatter()\n"
            "print(f\"Is a Formatter? {isinstance(csv_fmt, Formatter)}\")\n"
            "print()\n"
            "print(csv_fmt.format(window))"
        ),
        md(
            "## Swap Formatters at Runtime\n"
            "Because every formatter shares the same protocol, you can select\n"
            "one dynamically."
        ),
        code(
            "from anchor.formatters import AnthropicFormatter, GenericTextFormatter\n"
            "\n"
            "formatters = {\n"
            "    \"anthropic\": AnthropicFormatter(),\n"
            "    \"text\": GenericTextFormatter(),\n"
            "    \"xml\": XMLFormatter(),\n"
            "    \"csv\": CSVFormatter(),\n"
            "}\n"
            "\n"
            "for name, fmt in formatters.items():\n"
            "    result = fmt.format(window)\n"
            "    preview = str(result)[:60].replace(\"\\n\", \" \")\n"
            "    print(f\"{name:10s} -> {type(result).__name__:5s} | {preview}...\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `Formatter` is a `runtime_checkable` Protocol — no base class needed.\n"
            "- Implement `format_type` (property) and `format(window)` to comply.\n"
            "- Custom formatters plug into the same pipeline as built-in ones.\n"
            "- Use `isinstance(obj, Formatter)` to verify compliance at runtime."
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Formatters notebooks in {OUTPUT_DIR}/\n")

    nb_anthropic_formatter()
    nb_openai_formatter()
    nb_generic_text()
    nb_custom_formatter()

    print("\nDone — 4 notebooks generated.")


if __name__ == "__main__":
    main()
