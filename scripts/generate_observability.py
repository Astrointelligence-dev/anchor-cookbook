"""Generate the 4 Observability module notebooks for the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "07-observability"))
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
# 01 - Tracing
# ---------------------------------------------------------------------------
def nb_01():
    return make_nb([
        md(
            "# Recipe 01 — Tracing\n"
            "> Instrument your pipeline with traces and spans for end-to-end visibility.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.observability` |\n"
            "| **Key classes** | `Tracer`, `Span`, `TraceRecord`, `SpanKind` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.observability import Tracer, Span, TraceRecord, SpanKind\n"
            "import time"
        ),
        code(
            "# Create a tracer — entry point for all tracing\n"
            "tracer = Tracer()\n"
            "print(f\"Tracer ready: {type(tracer).__name__}\")"
        ),
        md("## 1 — Start a trace and add spans"),
        code(
            "# A trace groups related spans under a single root identifier\n"
            "trace = tracer.start_trace(\"my-pipeline\", attributes={\"pipeline\": \"rag\"})\n"
            "print(f\"Trace ID   : {trace.trace_id}\")\n"
            "print(f\"Trace name : my-pipeline\")\n"
            "print(f\"Attributes : {trace.attributes}\")"
        ),
        code(
            "# Retrieval span — use SpanKind to label each operation\n"
            "span1 = tracer.start_span(\n"
            "    trace.trace_id,\n"
            "    \"retrieval\",\n"
            "    kind=SpanKind.RETRIEVAL,\n"
            "    attributes={\"top_k\": 10},\n"
            ")\n"
            "time.sleep(0.01)  # Simulate retrieval work\n"
            "tracer.end_span(span1)\n"
            "print(f\"Span 1: {span1.name} (kind={span1.kind})\")"
        ),
        code(
            "# Reranking span — child of retrieval via parent_span_id\n"
            "span2 = tracer.start_span(\n"
            "    trace.trace_id,\n"
            "    \"reranking\",\n"
            "    kind=SpanKind.RERANKING,\n"
            "    parent_span_id=span1.span_id,\n"
            ")\n"
            "time.sleep(0.01)  # Simulate reranking\n"
            "tracer.end_span(span2)\n"
            "print(f\"Span 2: {span2.name} (parent={span1.span_id[:8]}...)\")"
        ),
        md("## 2 — End the trace and inspect results"),
        code(
            "trace = tracer.end_trace(trace)\n"
            "\n"
            "print(f\"Trace   : {trace.trace_id}\")\n"
            "print(f\"Spans   : {len(trace.spans)}\")\n"
            "print(f\"Duration: {trace.total_duration_ms:.1f}ms\")"
        ),
        code(
            "# Walk the span tree — useful for flame graphs or debugging\n"
            "for span in trace.spans:\n"
            "    parent = span.parent_span_id[:8] if span.parent_span_id else \"root\"\n"
            "    print(f\"  {span.name:12s}  kind={str(span.kind):12s}  parent={parent}\")"
        ),
        md("## 3 — Multiple concurrent traces"),
        code(
            "# A single Tracer can manage many traces at once\n"
            "t1 = tracer.start_trace(\"pipeline-a\", attributes={\"model\": \"haiku\"})\n"
            "t2 = tracer.start_trace(\"pipeline-b\", attributes={\"model\": \"sonnet\"})\n"
            "\n"
            "s1 = tracer.start_span(t1.trace_id, \"embed\", kind=SpanKind.RETRIEVAL)\n"
            "time.sleep(0.005)\n"
            "tracer.end_span(s1)\n"
            "t1 = tracer.end_trace(t1)\n"
            "\n"
            "s2 = tracer.start_span(t2.trace_id, \"generate\", kind=SpanKind.RERANKING)\n"
            "time.sleep(0.005)\n"
            "tracer.end_span(s2)\n"
            "t2 = tracer.end_trace(t2)\n"
            "\n"
            "print(f\"Trace A: {t1.total_duration_ms:.1f}ms  spans={len(t1.spans)}\")\n"
            "print(f\"Trace B: {t2.total_duration_ms:.1f}ms  spans={len(t2.spans)}\")"
        ),
        code(
            "# Inspect attributes on each trace\n"
            "for t in [t1, t2]:\n"
            "    print(f\"  {t.trace_id[:8]}...  attrs={t.attributes}  \"\n"
            "          f\"spans={len(t.spans)}  dur={t.total_duration_ms:.1f}ms\")"
        ),
        code(
            "# Verify span kinds are preserved\n"
            "for t in [t1, t2]:\n"
            "    for sp in t.spans:\n"
            "        print(f\"  [{t.trace_id[:8]}] {sp.name}  kind={sp.kind}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `Tracer` manages the lifecycle of traces and spans.\n"
            "- `SpanKind` labels each operation (RETRIEVAL, RERANKING, etc.).\n"
            "- Spans can be nested via `parent_span_id` to form a tree.\n"
            "- `trace.total_duration_ms` gives end-to-end latency.\n"
            "\n"
            "**Next:** [Exporters](02_exporters.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Exporters
# ---------------------------------------------------------------------------
def nb_02():
    return make_nb([
        md(
            "# Recipe 02 — Exporters\n"
            "> Ship traces and spans to consoles, files, or OpenTelemetry collectors.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.observability.exporters` |\n"
            "| **Key classes** | `ConsoleSpanExporter`, `InMemorySpanExporter`, `FileSpanExporter`, `OTLPSpanExporter` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.observability import Tracer, SpanKind\n"
            "from anchor.observability.exporters import (\n"
            "    ConsoleSpanExporter,\n"
            "    InMemorySpanExporter,\n"
            "    FileSpanExporter,\n"
            "    OTLPSpanExporter,\n"
            ")\n"
            "import time"
        ),
        md("## 1 — Generate sample trace data"),
        code(
            "tracer = Tracer()\n"
            "trace = tracer.start_trace(\"export-demo\", attributes={\"pipeline\": \"rag\"})\n"
            "\n"
            "span1 = tracer.start_span(trace.trace_id, \"retrieval\", kind=SpanKind.RETRIEVAL, attributes={\"top_k\": 10})\n"
            "time.sleep(0.01)\n"
            "tracer.end_span(span1)\n"
            "\n"
            "span2 = tracer.start_span(trace.trace_id, \"reranking\", kind=SpanKind.RERANKING, parent_span_id=span1.span_id)\n"
            "time.sleep(0.01)\n"
            "tracer.end_span(span2)\n"
            "\n"
            "trace = tracer.end_trace(trace)\n"
            "print(f\"Trace ready: {len(trace.spans)} spans\")"
        ),
        md("## 2 — Exporter walkthrough"),
        code(
            "# ConsoleSpanExporter — prints spans to stdout\n"
            "console = ConsoleSpanExporter()\n"
            "console.export(trace.spans)\n"
            "print(\"\\n(Spans printed above by ConsoleSpanExporter)\")"
        ),
        code(
            "# InMemorySpanExporter — stores for programmatic inspection\n"
            "in_memory = InMemorySpanExporter()\n"
            "in_memory.export(trace.spans)\n"
            "\n"
            "stored = in_memory.get_spans()\n"
            "print(f\"Stored spans: {len(stored)}\")\n"
            "for s in stored:\n"
            "    print(f\"  {s.name}: kind={s.kind}\")"
        ),
        code(
            "# FileSpanExporter — writes JSON to disk\n"
            "import tempfile, os, json\n"
            "\n"
            "tmp_path = os.path.join(tempfile.gettempdir(), \"anchor_traces.json\")\n"
            "\n"
            "file_exp = FileSpanExporter(file_path=tmp_path)\n"
            "file_exp.export(trace.spans)\n"
            "\n"
            "size = os.path.getsize(tmp_path)\n"
            "print(f\"Wrote {size} bytes to {tmp_path}\")"
        ),
        code(
            "# Peek at the file contents\n"
            "with open(tmp_path) as f:\n"
            "    data = json.load(f)\n"
            "print(f\"Span records in file: {len(data)}\")\n"
            "for record in data:\n"
            "    print(f\"  {record.get('name', 'unknown')}\")"
        ),
        code(
            "# OTLPSpanExporter — sends spans over gRPC/HTTP to an OTLP endpoint\n"
            "# Uncomment when you have a collector running:\n"
            "#\n"
            "# otlp = OTLPSpanExporter(endpoint=\"http://localhost:4317\")\n"
            "# otlp.export(trace.spans)\n"
            "\n"
            "print(\"OTLPSpanExporter pattern:\")\n"
            "print('  otlp = OTLPSpanExporter(endpoint=\"http://localhost:4317\")')\n"
            "print('  otlp.export(trace.spans)')"
        ),
        md("## 3 — Combining multiple exporters"),
        code(
            "exporters = [\n"
            "    ConsoleSpanExporter(),\n"
            "    InMemorySpanExporter(),\n"
            "]\n"
            "\n"
            "for exporter in exporters:\n"
            "    exporter.export(trace.spans)\n"
            "    print(f\"Exported to {type(exporter).__name__}\")"
        ),
        code(
            "# Verify in-memory exporter accumulated spans\n"
            "mem_exporter = exporters[1]\n"
            "print(f\"InMemory exporter holds {len(mem_exporter.get_spans())} spans\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `ConsoleSpanExporter` is the fastest way to see spans during development.\n"
            "- `InMemorySpanExporter` is perfect for tests and assertions.\n"
            "- `FileSpanExporter` writes JSON for offline analysis.\n"
            "- `OTLPSpanExporter` integrates with Jaeger, Grafana Tempo, and other OTLP backends.\n"
            "\n"
            "**Next:** [Cost Tracking](03_cost_tracking.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Cost Tracking
# ---------------------------------------------------------------------------
def nb_03():
    return make_nb([
        md(
            "# Recipe 03 — Cost Tracking\n"
            "> Monitor LLM token usage and estimated costs across models.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.observability` |\n"
            "| **Key classes** | `CostTracker`, `CostEntry`, `CostSummary` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.observability import CostTracker, CostEntry, CostSummary"
        ),
        md("## 1 — Create a cost tracker and record usage"),
        code(
            "# Define per-model pricing as (input_cost_per_1k, output_cost_per_1k)\n"
            "tracker = CostTracker(model_costs={\n"
            "    \"claude-haiku-4-5-20251001\": (0.00025, 0.00125),\n"
            "    \"claude-sonnet-4-20250514\": (0.003, 0.015),\n"
            "})\n"
            "print(f\"Tracker configured for {len(tracker.model_costs)} models\")\n"
            "for model, (inp, out) in tracker.model_costs.items():\n"
            "    print(f\"  {model}: ${inp}/1k input, ${out}/1k output\")"
        ),
        code(
            "# Record two LLM calls\n"
            "tracker.record(model=\"claude-haiku-4-5-20251001\", input_tokens=500, output_tokens=200)\n"
            "tracker.record(model=\"claude-sonnet-4-20250514\", input_tokens=1000, output_tokens=500)\n"
            "\n"
            "print(\"Recorded 2 LLM calls\")"
        ),
        md("## 2 — Inspect the cost summary"),
        code(
            "summary = tracker.get_summary()\n"
            "\n"
            "print(f\"Total cost     : ${summary.total_cost:.6f}\")\n"
            "print(f\"Total input tk : {summary.total_input_tokens}\")\n"
            "print(f\"Total output tk: {summary.total_output_tokens}\")"
        ),
        code(
            "# Per-model breakdown\n"
            "for model, cost in summary.by_model.items():\n"
            "    print(f\"  {model}: ${cost:.6f}\")"
        ),
        code(
            "# Inspect individual CostEntry records\n"
            "for entry in summary.entries:\n"
            "    print(f\"  model={entry.model}  in={entry.input_tokens}  \"\n"
            "          f\"out={entry.output_tokens}  cost=${entry.cost:.6f}\")"
        ),
        md("## 3 — Track a multi-step pipeline"),
        code(
            "# Simulate a multi-step RAG pipeline\n"
            "pipeline_tracker = CostTracker(model_costs={\n"
            "    \"claude-haiku-4-5-20251001\": (0.00025, 0.00125),\n"
            "    \"claude-sonnet-4-20250514\": (0.003, 0.015),\n"
            "})\n"
            "\n"
            "# Step 1: Query expansion (cheap model)\n"
            "pipeline_tracker.record(model=\"claude-haiku-4-5-20251001\", input_tokens=100, output_tokens=50)\n"
            "\n"
            "# Step 2: Answer generation (powerful model)\n"
            "pipeline_tracker.record(model=\"claude-sonnet-4-20250514\", input_tokens=2000, output_tokens=800)\n"
            "\n"
            "# Step 3: Summary (cheap model)\n"
            "pipeline_tracker.record(model=\"claude-haiku-4-5-20251001\", input_tokens=300, output_tokens=100)\n"
            "\n"
            "result = pipeline_tracker.get_summary()\n"
            "print(f\"Pipeline total cost: ${result.total_cost:.6f}\")\n"
            "print(f\"Pipeline total tokens: {result.total_input_tokens + result.total_output_tokens}\")"
        ),
        code(
            "# Per-model breakdown for the pipeline\n"
            "print(\"Per-model breakdown:\")\n"
            "for model, cost in result.by_model.items():\n"
            "    print(f\"  {model}: ${cost:.6f}\")\n"
            "\n"
            "# Show each call in order\n"
            "print(\"\\nAll entries:\")\n"
            "for i, entry in enumerate(result.entries):\n"
            "    print(f\"  Call {i+1}: {entry.model}  \"\n"
            "          f\"in={entry.input_tokens} out={entry.output_tokens}  ${entry.cost:.6f}\")"
        ),
        code(
            "# Compare cost ratio between models\n"
            "haiku_cost = result.by_model.get(\"claude-haiku-4-5-20251001\", 0)\n"
            "sonnet_cost = result.by_model.get(\"claude-sonnet-4-20250514\", 0)\n"
            "total = result.total_cost\n"
            "\n"
            "print(f\"Haiku  share: {haiku_cost/total:.1%}\")\n"
            "print(f\"Sonnet share: {sonnet_cost/total:.1%}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `CostTracker` maps model names to per-1k-token pricing.\n"
            "- `record()` logs each LLM call; `get_summary()` aggregates everything.\n"
            "- `CostSummary` provides `total_cost`, `by_model`, and raw `entries`.\n"
            "- Track costs per pipeline step to identify optimisation opportunities.\n"
            "\n"
            "**Next:** [Metrics](04_metrics.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Metrics
# ---------------------------------------------------------------------------
def nb_04():
    return make_nb([
        md(
            "# Recipe 04 — Metrics\n"
            "> Collect and inspect operational metrics from your pipeline.\n"
            "\n"
            "| | |\n"
            "|---|---|\n"
            "| **Module** | `anchor.observability.metrics` |\n"
            "| **Key classes** | `InMemoryMetricsCollector`, `LoggingMetricsCollector`, `MetricPoint` |\n"
            "| **Difficulty** | Beginner |"
        ),
        code(
            "# Setup\n"
            "from anchor.observability.metrics import (\n"
            "    InMemoryMetricsCollector,\n"
            "    LoggingMetricsCollector,\n"
            "    MetricPoint,\n"
            ")\n"
            "from anchor.observability.exporters import OTLPMetricsExporter"
        ),
        md("## 1 — InMemoryMetricsCollector"),
        code(
            "collector = InMemoryMetricsCollector()\n"
            "print(f\"Collector: {type(collector).__name__}\")"
        ),
        code(
            "# Record several metrics during pipeline execution\n"
            "collector.record_metric(name=\"retrieval.latency_ms\", value=45.2, unit=\"ms\")\n"
            "collector.record_metric(name=\"retrieval.items_found\", value=10, unit=\"count\")\n"
            "collector.record_metric(name=\"reranking.latency_ms\", value=12.8, unit=\"ms\")\n"
            "collector.record_metric(name=\"generation.tokens_used\", value=1500, unit=\"tokens\")\n"
            "\n"
            "print(\"Recorded 4 metric data points\")"
        ),
        code(
            "# Retrieve and inspect — each MetricPoint has name, value, unit\n"
            "metrics = collector.get_metrics()\n"
            "print(f\"Total metrics: {len(metrics)}\\n\")\n"
            "\n"
            "for m in metrics:\n"
            "    print(f\"  {m.name:30s}  {m.value:>8}  {m.unit}\")"
        ),
        md("## 2 — LoggingMetricsCollector and OTLPMetricsExporter"),
        code(
            "import logging\n"
            "logging.basicConfig(level=logging.INFO, format=\"%(levelname)s: %(message)s\")\n"
            "\n"
            "logging_collector = LoggingMetricsCollector()\n"
            "logging_collector.record_metric(name=\"pipeline.tokens_used\", value=3500, unit=\"tokens\")\n"
            "logging_collector.record_metric(name=\"pipeline.latency_ms\", value=230.5, unit=\"ms\")\n"
            "\n"
            "print(\"\\n(Metrics logged above via LoggingMetricsCollector)\")"
        ),
        code(
            "# OTLPMetricsExporter — push to Prometheus, Grafana, etc.\n"
            "# Uncomment when you have an OTLP collector running:\n"
            "#\n"
            "# exporter = OTLPMetricsExporter(endpoint=\"http://localhost:4317\")\n"
            "# exporter.export(collector.get_metrics())\n"
            "\n"
            "print(\"OTLPMetricsExporter pattern:\")\n"
            "print('  exporter = OTLPMetricsExporter(endpoint=\"http://localhost:4317\")')\n"
            "print('  exporter.export(collector.get_metrics())')"
        ),
        md("## 3 — Full pipeline metrics workflow"),
        code(
            "# Track an entire pipeline run with one collector\n"
            "pipeline_metrics = InMemoryMetricsCollector()\n"
            "\n"
            "# Retrieval phase\n"
            "pipeline_metrics.record_metric(\"retrieval.latency_ms\", 55.0, \"ms\")\n"
            "pipeline_metrics.record_metric(\"retrieval.docs_returned\", 15, \"count\")\n"
            "\n"
            "# Filtering phase\n"
            "pipeline_metrics.record_metric(\"filter.docs_passed\", 8, \"count\")\n"
            "pipeline_metrics.record_metric(\"filter.docs_removed\", 7, \"count\")\n"
            "\n"
            "# Generation phase\n"
            "pipeline_metrics.record_metric(\"generation.input_tokens\", 2000, \"tokens\")\n"
            "pipeline_metrics.record_metric(\"generation.output_tokens\", 500, \"tokens\")\n"
            "pipeline_metrics.record_metric(\"generation.latency_ms\", 180.3, \"ms\")\n"
            "\n"
            "print(f\"Recorded {len(pipeline_metrics.get_metrics())} pipeline metrics\")"
        ),
        code(
            "# Display the full pipeline summary\n"
            "all_metrics = pipeline_metrics.get_metrics()\n"
            "print(f\"Pipeline metrics collected: {len(all_metrics)}\\n\")\n"
            "for m in all_metrics:\n"
            "    print(f\"  {m.name:35s}  {m.value:>8}  {m.unit}\")"
        ),
        code(
            "# Filter metrics by unit type\n"
            "latency_metrics = [m for m in all_metrics if m.unit == \"ms\"]\n"
            "token_metrics = [m for m in all_metrics if m.unit == \"tokens\"]\n"
            "count_metrics = [m for m in all_metrics if m.unit == \"count\"]\n"
            "\n"
            "print(f\"Latency metrics : {len(latency_metrics)}\")\n"
            "print(f\"Token metrics   : {len(token_metrics)}\")\n"
            "print(f\"Count metrics   : {len(count_metrics)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `InMemoryMetricsCollector` stores data points for programmatic access.\n"
            "- `LoggingMetricsCollector` integrates with Python's logging framework.\n"
            "- `OTLPMetricsExporter` pushes metrics to Prometheus, Grafana, and similar backends.\n"
            "- Record metrics at each pipeline phase for full operational visibility.\n"
            "\n"
            "**Back to:** [Observability Overview](../README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    notebooks = [
        ("01_tracing.ipynb", nb_01),
        ("02_exporters.ipynb", nb_02),
        ("03_cost_tracking.ipynb", nb_03),
        ("04_metrics.ipynb", nb_04),
    ]

    print(f"Generating {len(notebooks)} Observability notebooks in {OUTPUT_DIR}/")
    for filename, factory in notebooks:
        nb = factory()
        write_nb(nb, filename)

    print(f"\nDone — {len(notebooks)} notebooks created.")


if __name__ == "__main__":
    main()
