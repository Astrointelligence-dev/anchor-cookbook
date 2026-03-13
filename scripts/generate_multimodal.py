"""Generate the 3 Multimodal module notebooks for the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "09-multimodal"))
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
# 01 - Multimodal Converter
# ---------------------------------------------------------------------------
def nb_01():
    return make_nb([
        md(
            "# Recipe 01 — Multimodal Converter\n"
            "> Convert between MultiModalItem and ContextItem using pluggable encoders.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.multimodal` |\n"
            "| **Key classes** | `MultiModalConverter`, `MultiModalItem`, `MultiModalContent`, `ModalityType` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.multimodal import (\n"
            "    MultiModalConverter,\n"
            "    MultiModalItem,\n"
            "    MultiModalContent,\n"
            "    ModalityType,\n"
            ")\n"
            "from anchor.multimodal.encoders import TextEncoder, CompositeEncoder\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md("## Walkthrough"),
        code(
            "# 1 — Create a multimodal item\n"
            "# MultiModalContent wraps raw content with its modality type\n"
            "content = MultiModalContent(\n"
            "    modality=ModalityType.TEXT,\n"
            "    content=\"A document about AI safety and alignment research\",\n"
            ")\n"
            "print(f\"Modality : {content.modality}\")\n"
            "print(f\"Content  : {content.content}\")"
        ),
        code(
            "# MultiModalItem bundles one or more contents with metadata\n"
            "item = MultiModalItem(\n"
            "    id=\"mm-1\",\n"
            "    contents=[content],\n"
            "    source=\"wiki\",\n"
            "    score=0.9,\n"
            "    priority=5,\n"
            "    metadata={\"author\": \"research-team\"},\n"
            ")\n"
            "\n"
            "print(f\"Item ID   : {item.id}\")\n"
            "print(f\"Source    : {item.source}\")\n"
            "print(f\"Score     : {item.score}\")\n"
            "print(f\"Priority  : {item.priority}\")\n"
            "print(f\"Contents  : {len(item.contents)}\")\n"
            "print(f\"Metadata  : {item.metadata}\")"
        ),
        code(
            "# 2 — Set up encoders and convert to ContextItem\n"
            "# TextEncoder handles ModalityType.TEXT\n"
            "text_encoder = TextEncoder()\n"
            "print(f\"Encoder: {type(text_encoder).__name__}\")\n"
            "\n"
            "# CompositeEncoder routes each modality to the right encoder\n"
            "composite = CompositeEncoder(\n"
            "    encoders={ModalityType.TEXT: text_encoder},\n"
            ")\n"
            "print(f\"Composite handles: {list(composite.encoders.keys())}\")"
        ),
        code(
            "# Convert a single MultiModalItem -> ContextItem\n"
            "context_item = MultiModalConverter.to_context_item(item, encoder=composite)\n"
            "\n"
            "print(f\"Type       : {type(context_item).__name__}\")\n"
            "print(f\"Content    : {context_item.content[:60]}...\")\n"
            "print(f\"Source type: {context_item.source_type}\")"
        ),
        code(
            "# 3 — Batch conversion\n"
            "# Create several multimodal items\n"
            "items = [\n"
            "    MultiModalItem(\n"
            "        id=f\"mm-{i}\",\n"
            "        contents=[MultiModalContent(modality=ModalityType.TEXT, content=f\"Document {i}\")],\n"
            "        source=\"wiki\",\n"
            "        score=0.8 + i * 0.05,\n"
            "        priority=i,\n"
            "        metadata={},\n"
            "    )\n"
            "    for i in range(1, 4)\n"
            "]\n"
            "\n"
            "# Batch convert\n"
            "context_items = MultiModalConverter.to_context_items(items, encoder=composite)\n"
            "\n"
            "print(f\"Converted {len(context_items)} items:\")\n"
            "for ci in context_items:\n"
            "    print(f\"  {ci.content}\")"
        ),
        code(
            "# Inspect ContextItem attributes\n"
            "for ci in context_items:\n"
            "    print(f\"  source_type={ci.source_type}  content_len={len(ci.content)}\")"
        ),
        code(
            "# 4 — Round-trip: convert back to MultiModalItem\n"
            "# Convert a ContextItem back to a MultiModalItem\n"
            "mm_item = MultiModalConverter.from_context_item(\n"
            "    context_item,\n"
            "    modality=ModalityType.TEXT,\n"
            ")\n"
            "\n"
            "print(f\"Recovered ID      : {mm_item.id}\")\n"
            "print(f\"Recovered modality: {mm_item.contents[0].modality}\")\n"
            "print(f\"Recovered content : {mm_item.contents[0].content[:60]}...\")"
        ),
        code(
            "# Verify round-trip fidelity\n"
            "original_text = item.contents[0].content\n"
            "recovered_text = mm_item.contents[0].content\n"
            "print(f\"Original  : {original_text}\")\n"
            "print(f\"Recovered : {recovered_text}\")\n"
            "print(f\"Match     : {original_text == recovered_text}\")"
        ),
        code(
            "# Enumerate all supported modality types\n"
            "for modality in ModalityType:\n"
            "    print(f\"  {modality.name}: {modality.value}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MultiModalContent` pairs raw data with a `ModalityType` tag.\n"
            "- `MultiModalItem` groups contents with scoring and metadata.\n"
            "- `CompositeEncoder` dispatches encoding to modality-specific encoders.\n"
            "- `MultiModalConverter.to_context_item()` and `from_context_item()` enable round-trip conversion.\n"
            "- Batch conversion via `to_context_items()` handles multiple items efficiently.\n"
            "\n"
            "**Next:** [Image Encoding](02_image_encoding.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Image Encoding
# ---------------------------------------------------------------------------
def nb_02():
    return make_nb([
        md(
            "# Recipe 02 — Image Encoding\n"
            "> Encode image data into text descriptions using pluggable describe functions.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.multimodal.encoders` |\n"
            "| **Key classes** | `ImageDescriptionEncoder`, `CompositeEncoder`, `TextEncoder` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.multimodal.encoders import (\n"
            "    ImageDescriptionEncoder,\n"
            "    CompositeEncoder,\n"
            "    TextEncoder,\n"
            ")\n"
            "from anchor.multimodal import MultiModalContent, ModalityType"
        ),
        md("## Walkthrough"),
        code(
            "# 1 — Define a mock image description function\n"
            "# In production, this would call a vision model (e.g., Claude Vision)\n"
            "def mock_describe(image_bytes: bytes) -> str:\n"
            "    \"\"\"Simulate an image description service.\"\"\"\n"
            "    size = len(image_bytes)\n"
            "    return f\"A photograph showing a sunset over mountains ({size} bytes analysed)\"\n"
            "\n"
            "print(\"mock_describe ready\")\n"
            "print(f\"Test: {mock_describe(b'test')}\")"
        ),
        code(
            "# 2 — Create the ImageDescriptionEncoder\n"
            "# ImageDescriptionEncoder takes a callable that maps bytes -> str\n"
            "img_encoder = ImageDescriptionEncoder(describe_fn=mock_describe)\n"
            "print(f\"Encoder: {type(img_encoder).__name__}\")"
        ),
        code(
            "# Encode image content\n"
            "img_content = MultiModalContent(\n"
            "    modality=ModalityType.IMAGE,\n"
            "    content=b\"fake-image-bytes-png-data\",\n"
            ")\n"
            "\n"
            "encoded = img_encoder.encode(img_content)\n"
            "print(f\"Input modality : {img_content.modality}\")\n"
            "print(f\"Input size     : {len(img_content.content)} bytes\")\n"
            "print(f\"Encoded output : {encoded}\")"
        ),
        code(
            "# 3 — Build a CompositeEncoder with text and image support\n"
            "# CompositeEncoder routes each modality to the right encoder\n"
            "composite = CompositeEncoder(\n"
            "    encoders={\n"
            "        ModalityType.TEXT: TextEncoder(),\n"
            "        ModalityType.IMAGE: img_encoder,\n"
            "    },\n"
            ")\n"
            "\n"
            "print(f\"Supported modalities: {list(composite.encoders.keys())}\")"
        ),
        code(
            "# Encode text through the composite\n"
            "text_content = MultiModalContent(\n"
            "    modality=ModalityType.TEXT,\n"
            "    content=\"A technical diagram of a neural network\",\n"
            ")\n"
            "text_encoded = composite.encode(text_content)\n"
            "print(f\"Text encoded: {text_encoded}\")"
        ),
        code(
            "# Encode image through the composite\n"
            "image_encoded = composite.encode(img_content)\n"
            "print(f\"Image encoded: {image_encoded}\")"
        ),
        code(
            "# Show that composite correctly routes by modality\n"
            "print(f\"Text via composite  : {text_encoded[:50]}\")\n"
            "print(f\"Image via composite : {image_encoded[:50]}\")\n"
            "print(f\"Same encoder?       : {text_encoded == image_encoded}\")"
        ),
        code(
            "# 4 — Multiple images with different content\n"
            "# Simulate encoding several images\n"
            "image_payloads = [\n"
            "    b\"chart-revenue-q1-data\",\n"
            "    b\"photo-team-offsite-jpg\",\n"
            "    b\"screenshot-dashboard-png\",\n"
            "]\n"
            "\n"
            "for i, payload in enumerate(image_payloads, 1):\n"
            "    content = MultiModalContent(modality=ModalityType.IMAGE, content=payload)\n"
            "    result = composite.encode(content)\n"
            "    print(f\"Image {i}: {result}\")"
        ),
        code(
            "# Verify encoder isolation — text encoder ignores image data\n"
            "text_result = composite.encode(text_content)\n"
            "image_result = composite.encode(img_content)\n"
            "\n"
            "print(f\"Text result type  : {type(text_result).__name__}\")\n"
            "print(f\"Image result type : {type(image_result).__name__}\")\n"
            "print(f\"Results differ    : {text_result != image_result}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `ImageDescriptionEncoder` converts image bytes to text via a `describe_fn` callable.\n"
            "- Swap `mock_describe` for a real vision API (e.g., Claude Vision) in production.\n"
            "- `CompositeEncoder` dispatches to the correct encoder based on `ModalityType`.\n"
            "- Text and image modalities can be mixed freely within the same pipeline.\n"
            "\n"
            "**Next:** [Table Extraction](03_table_extraction.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Table Extraction
# ---------------------------------------------------------------------------
def nb_03():
    return make_nb([
        md(
            "# Recipe 03 — Table Extraction\n"
            "> Parse and encode tabular data from Markdown and HTML sources.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.multimodal.tables` / `anchor.multimodal.encoders` |\n"
            "| **Key classes** | `MarkdownTableParser`, `HTMLTableParser`, `TableEncoder` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.multimodal.tables import MarkdownTableParser, HTMLTableParser\n"
            "from anchor.multimodal.encoders import TableEncoder\n"
            "from anchor.multimodal import MultiModalContent, ModalityType"
        ),
        md("## Walkthrough"),
        code(
            "# 1 — Parse a Markdown table\n"
            "md_parser = MarkdownTableParser()\n"
            "\n"
            "md_table = (\n"
            "    \"| Name  | Age | Role      |\\n\"\n"
            "    \"|-------|-----|-----------|\\n\"\n"
            "    \"| Alice | 30  | Engineer  |\\n\"\n"
            "    \"| Bob   | 25  | Designer  |\\n\"\n"
            "    \"| Carol | 35  | Manager   |\"\n"
            ")\n"
            "\n"
            "parsed = md_parser.parse(md_table)\n"
            "\n"
            "print(f\"Headers : {parsed.headers}\")\n"
            "print(f\"Rows    : {len(parsed.rows)}\")\n"
            "for row in parsed.rows:\n"
            "    print(f\"  {row}\")"
        ),
        code(
            "# 2 — Parse an HTML table\n"
            "html_parser = HTMLTableParser()\n"
            "\n"
            "html_table = (\n"
            "    \"<table>\"\n"
            "    \"<tr><th>Name</th><th>Age</th><th>Role</th></tr>\"\n"
            "    \"<tr><td>Alice</td><td>30</td><td>Engineer</td></tr>\"\n"
            "    \"<tr><td>Bob</td><td>25</td><td>Designer</td></tr>\"\n"
            "    \"</table>\"\n"
            ")\n"
            "\n"
            "parsed_html = html_parser.parse(html_table)\n"
            "\n"
            "print(f\"Headers : {parsed_html.headers}\")\n"
            "print(f\"Rows    : {len(parsed_html.rows)}\")\n"
            "for row in parsed_html.rows:\n"
            "    print(f\"  {row}\")"
        ),
        code(
            "# Compare the two parsers\n"
            "print(f\"Markdown headers: {parsed.headers}\")\n"
            "print(f\"HTML headers    : {parsed_html.headers}\")\n"
            "print(f\"Markdown rows   : {len(parsed.rows)}\")\n"
            "print(f\"HTML rows       : {len(parsed_html.rows)}\")"
        ),
        code(
            "# 3 — Encode tables with TableEncoder\n"
            "# TableEncoder converts table content into a text representation\n"
            "table_encoder = TableEncoder(format=\"markdown\")\n"
            "print(f\"Encoder : {type(table_encoder).__name__}\")\n"
            "print(f\"Format  : markdown\")"
        ),
        code(
            "# Wrap the markdown table in MultiModalContent\n"
            "table_content = MultiModalContent(\n"
            "    modality=ModalityType.TABLE,\n"
            "    content=md_table,\n"
            ")\n"
            "\n"
            "encoded = table_encoder.encode(table_content)\n"
            "print(f\"Encoded table:\\n{encoded}\")"
        ),
        code(
            "# Verify encoding produces a non-empty string\n"
            "print(f\"Encoded type   : {type(encoded).__name__}\")\n"
            "print(f\"Encoded length : {len(encoded)} chars\")\n"
            "print(f\"Non-empty      : {len(encoded) > 0}\")"
        ),
        code(
            "# 4 — Larger table example\n"
            "# A more complex table with numeric data\n"
            "sales_table = (\n"
            "    \"| Quarter | Revenue  | Growth |\\n\"\n"
            "    \"|---------|----------|--------|\\n\"\n"
            "    \"| Q1 2025 | $1.2M    | 15%    |\\n\"\n"
            "    \"| Q2 2025 | $1.5M    | 25%    |\\n\"\n"
            "    \"| Q3 2025 | $1.8M    | 20%    |\\n\"\n"
            "    \"| Q4 2025 | $2.1M    | 17%    |\"\n"
            ")\n"
            "\n"
            "sales_parsed = md_parser.parse(sales_table)\n"
            "print(f\"Headers: {sales_parsed.headers}\")\n"
            "print(f\"Rows   : {len(sales_parsed.rows)}\")\n"
            "for row in sales_parsed.rows:\n"
            "    print(f\"  {row}\")"
        ),
        code(
            "# Encode the sales table\n"
            "sales_content = MultiModalContent(\n"
            "    modality=ModalityType.TABLE,\n"
            "    content=sales_table,\n"
            ")\n"
            "sales_encoded = table_encoder.encode(sales_content)\n"
            "print(f\"Encoded sales table:\\n{sales_encoded}\")"
        ),
        code(
            "# Verify both parsers handle edge cases consistently\n"
            "simple_md = \"| A | B |\\n|---|---|\\n| 1 | 2 |\"\n"
            "simple_html = \"<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>\"\n"
            "\n"
            "md_result = md_parser.parse(simple_md)\n"
            "html_result = html_parser.parse(simple_html)\n"
            "\n"
            "print(f\"Markdown parse: headers={md_result.headers}, rows={md_result.rows}\")\n"
            "print(f\"HTML parse    : headers={html_result.headers}, rows={html_result.rows}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `MarkdownTableParser` and `HTMLTableParser` extract structured headers and rows.\n"
            "- `TableEncoder` converts table content into encoded text for downstream use.\n"
            "- Both parsers produce a consistent parsed representation with `.headers` and `.rows`.\n"
            "- Tables are treated as a first-class `ModalityType.TABLE` in the multimodal system.\n"
            "\n"
            "**Back to:** [Multimodal Overview](../README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    notebooks = [
        ("01_multimodal_converter.ipynb", nb_01),
        ("02_image_encoding.ipynb", nb_02),
        ("03_table_extraction.ipynb", nb_03),
    ]

    print(f"Generating {len(notebooks)} Multimodal notebooks in {OUTPUT_DIR}/")
    for filename, factory in notebooks:
        nb = factory()
        write_nb(nb, filename)

    print(f"\nDone — {len(notebooks)} notebooks created.")


if __name__ == "__main__":
    main()
