"""Generate the setup.ipynb notebook for the Anchor cookbook."""

import nbformat
import os

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

nb.cells = [
    nbformat.v4.new_markdown_cell(
        "# Setup & Environment Verification\n"
        "> Verify your Anchor installation and run a quick smoke test.\n"
        "\n"
        "**Package:** `astro-anchor`\n"
        "**Time:** ~1 minute"
    ),
    nbformat.v4.new_code_cell(
        "# Uncomment to install:\n"
        "# !pip install astro-anchor[all] jupyter -q"
    ),
    nbformat.v4.new_code_cell(
        'import anchor\n'
        'print(f"Anchor version: {anchor.__version__}")'
    ),
    nbformat.v4.new_markdown_cell(
        "## Quick Smoke Test\n"
        "Build a minimal context pipeline to verify everything works."
    ),
    nbformat.v4.new_code_cell(
        "from anchor.pipeline import ContextPipeline\n"
        "from anchor.models import ContextItem, SourceType, QueryBundle\n"
        "from anchor.formatters import GenericTextFormatter\n"
        "\n"
        "# Create a minimal pipeline\n"
        "pipeline = ContextPipeline(max_tokens=4096)\n"
        'pipeline.add_system_prompt("You are a helpful assistant.")\n'
        "pipeline.with_formatter(GenericTextFormatter())\n"
        "\n"
        "# Build context\n"
        'query = QueryBundle(query_str="Hello, world!")\n'
        "result = pipeline.build(query)\n"
        "\n"
        'print(f"Items in window: {len(result.window.items)}")\n'
        'print(f"Token utilization: {result.window.utilization:.1%}")\n'
        'print(f"\\nFormatted output preview:")\n'
        "print(result.formatted_output[:200] if isinstance(result.formatted_output, str) else str(result.formatted_output)[:200])\n"
        'print("\\n--- Setup complete! Your environment is ready. ---")'
    ),
    nbformat.v4.new_markdown_cell(
        "## Key Takeaways\n"
        "- `astro-anchor[all]` installs all optional dependencies\n"
        "- `ContextPipeline` is the main entry point for building context\n"
        "- Use `GenericTextFormatter` for plain text output\n"
        "\n"
        "**Next:** [Basic Pipeline](00-pipeline/01_basic_pipeline.ipynb)"
    ),
]

output_path = os.path.join(os.path.dirname(__file__), "..", "setup.ipynb")
output_path = os.path.normpath(output_path)
nbformat.write(nb, output_path)
print(f"Created {output_path}")
