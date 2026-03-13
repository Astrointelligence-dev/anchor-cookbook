"""Generate all 6 Jupyter notebooks for the Evaluation module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "06-evaluation"))
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
# 01 - Retrieval Metrics
# ---------------------------------------------------------------------------
def nb_retrieval_metrics():
    write_notebook("01_retrieval_metrics.ipynb", [
        md(
            "# Retrieval Metrics\n"
            "> Measure retrieval quality with precision, recall, MRR, nDCG, and MAP.\n"
            "\n"
            "`RetrievalMetricsCalculator` evaluates how well your retrieval\n"
            "pipeline surfaces relevant documents. It compares a ranked list of\n"
            "retrieved items against a ground-truth set of relevant IDs.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import RetrievalMetricsCalculator\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md(
            "## Build Mock Retrieved Documents\n"
            "We create five `ContextItem` instances that simulate retrieval\n"
            "results ranked by decreasing score. Only some of them are\n"
            "actually relevant."
        ),
        code(
            "retrieved = [\n"
            "    ContextItem(\n"
            "        id=f\"doc-{i}\",\n"
            "        content=f\"Content about topic {i}\",\n"
            "        source=SourceType.RETRIEVAL,\n"
            "        score=1.0 - i * 0.1,\n"
            "        priority=5,\n"
            "        token_count=10,\n"
            "    )\n"
            "    for i in range(5)\n"
            "]\n"
            "\n"
            "for item in retrieved:\n"
            "    print(f\"{item.id}  score={item.score:.1f}  tokens={item.token_count}\")"
        ),
        md(
            "## Define Ground Truth\n"
            "The relevant set tells the calculator which documents *should*\n"
            "have been retrieved."
        ),
        code(
            "relevant_ids = [\"doc-0\", \"doc-1\", \"doc-3\"]\n"
            "\n"
            "print(f\"Total retrieved:  {len(retrieved)}\")\n"
            "print(f\"Total relevant:   {len(relevant_ids)}\")\n"
            "print(f\"Relevant IDs:     {relevant_ids}\")"
        ),
        md(
            "## Compute Metrics at k=5\n"
            "`calculate()` returns an object with precision, recall, MRR,\n"
            "nDCG, and MAP -- the standard information-retrieval metrics."
        ),
        code(
            "calculator = RetrievalMetricsCalculator()\n"
            "\n"
            "metrics = calculator.evaluate(\n"
            "    retrieved=retrieved,\n"
            "    relevant=relevant_ids,\n"
            "    k=5,\n"
            ")\n"
            "\n"
            "print(f\"Precision@5: {metrics.precision:.3f}\")\n"
            "print(f\"Recall@5:    {metrics.recall:.3f}\")\n"
            "print(f\"MRR:         {metrics.mrr:.3f}\")\n"
            "print(f\"nDCG@5:      {metrics.ndcg:.3f}\")\n"
            "print(f\"MAP:         {metrics.map:.3f}\")"
        ),
        md(
            "## Compute Metrics at k=3\n"
            "Changing *k* controls how many top results are considered."
        ),
        code(
            "metrics_at_3 = calculator.evaluate(\n"
            "    retrieved=retrieved,\n"
            "    relevant=relevant_ids,\n"
            "    k=3,\n"
            ")\n"
            "\n"
            "print(f\"Precision@3: {metrics_at_3.precision:.3f}\")\n"
            "print(f\"Recall@3:    {metrics_at_3.recall:.3f}\")\n"
            "print(f\"MRR:         {metrics_at_3.mrr:.3f}\")\n"
            "print(f\"nDCG@3:      {metrics_at_3.ndcg:.3f}\")\n"
            "print(f\"MAP:         {metrics_at_3.map:.3f}\")"
        ),
        md(
            "## Compare k Values Side by Side\n"
            "A quick table shows how the cut-off affects each metric."
        ),
        code(
            "print(f\"{'Metric':<12} {'k=3':>8} {'k=5':>8}\")\n"
            "print(\"-\" * 30)\n"
            "for name in ['precision', 'recall', 'mrr', 'ndcg', 'map']:\n"
            "    v3 = getattr(metrics_at_3, name)\n"
            "    v5 = getattr(metrics, name)\n"
            "    print(f\"{name:<12} {v3:>8.3f} {v5:>8.3f}\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `RetrievalMetricsCalculator` computes standard IR metrics in one call.\n"
            "- Precision measures how many retrieved docs are relevant; recall\n"
            "  measures how many relevant docs were retrieved.\n"
            "- MRR reflects where the *first* relevant document appears in the ranking.\n"
            "- nDCG and MAP account for the position of *all* relevant documents.\n"
            "- Adjusting *k* lets you evaluate top-N retrieval quality."
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - LLM RAG Evaluator
# ---------------------------------------------------------------------------
def nb_llm_evaluator():
    write_notebook("02_llm_evaluator.ipynb", [
        md(
            "# LLM-Based RAG Evaluation\n"
            "> Use an LLM judge to score faithfulness, relevancy, and context quality.\n"
            "\n"
            "`LLMRAGEvaluator` wraps a custom evaluation function that returns\n"
            "RAG-specific metrics. This lets you plug in any LLM-as-judge\n"
            "implementation while keeping a consistent interface.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import LLMRAGEvaluator\n"
            "from anchor.evaluation.models import RAGMetrics"
        ),
        md(
            "## Define a Mock Evaluation Function\n"
            "In production you would call an LLM here. For this notebook we\n"
            "return deterministic scores so the output is reproducible."
        ),
        code(
            "def mock_judge(query, answer, contexts, ground_truth=None):\n"
            "    \"\"\"Simulate an LLM judge scoring a RAG response.\"\"\"\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.9,\n"
            "        relevancy=0.85,\n"
            "        context_precision=0.8,\n"
            "        context_recall=0.75,\n"
            "    )\n"
            "\n"
            "print(\"Evaluation function defined.\")\n"
            "print(f\"Return type: {RAGMetrics.__name__}\")"
        ),
        md(
            "## Create the Evaluator\n"
            "Pass the callable into `LLMRAGEvaluator`. It will be invoked\n"
            "every time `.evaluate()` is called."
        ),
        code(
            "evaluator = LLMRAGEvaluator(eval_fn=mock_judge)\n"
            "\n"
            "print(f\"Evaluator type: {type(evaluator).__name__}\")\n"
            "print(f\"Has eval_fn:    {evaluator.eval_fn is not None}\")"
        ),
        md(
            "## Evaluate a Single Response\n"
            "Provide the query, generated answer, supporting contexts, and\n"
            "(optionally) a ground-truth reference answer."
        ),
        code(
            "metrics = evaluator.evaluate(\n"
            "    query=\"What is prompt engineering?\",\n"
            "    answer=\"Prompt engineering is the practice of designing inputs to LLMs.\",\n"
            "    contexts=[\"Prompt engineering involves crafting effective prompts for AI models.\"],\n"
            "    ground_truth=\"Prompt engineering is the systematic design of LLM inputs.\",\n"
            ")\n"
            "\n"
            "print(f\"Faithfulness:      {metrics.faithfulness:.2f}\")\n"
            "print(f\"Relevancy:         {metrics.relevancy:.2f}\")\n"
            "print(f\"Context Precision: {metrics.context_precision:.2f}\")\n"
            "print(f\"Context Recall:    {metrics.context_recall:.2f}\")"
        ),
        md(
            "## Evaluate Without Ground Truth\n"
            "Ground truth is optional -- useful when you only want to assess\n"
            "faithfulness and relevancy."
        ),
        code(
            "metrics_no_gt = evaluator.evaluate(\n"
            "    query=\"What is RAG?\",\n"
            "    answer=\"RAG stands for Retrieval-Augmented Generation.\",\n"
            "    contexts=[\"RAG combines retrieval with generation to improve accuracy.\"],\n"
            ")\n"
            "\n"
            "print(f\"Faithfulness:      {metrics_no_gt.faithfulness:.2f}\")\n"
            "print(f\"Relevancy:         {metrics_no_gt.relevancy:.2f}\")\n"
            "print(f\"Context Precision: {metrics_no_gt.context_precision:.2f}\")\n"
            "print(f\"Context Recall:    {metrics_no_gt.context_recall:.2f}\")"
        ),
        md(
            "## Custom Scoring Functions\n"
            "You can vary the mock to simulate different quality levels."
        ),
        code(
            "def low_quality_judge(query, answer, contexts, ground_truth=None):\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.3,\n"
            "        relevancy=0.25,\n"
            "        context_precision=0.4,\n"
            "        context_recall=0.2,\n"
            "    )\n"
            "\n"
            "low_evaluator = LLMRAGEvaluator(eval_fn=low_quality_judge)\n"
            "low_metrics = low_evaluator.evaluate(\n"
            "    query=\"What is X?\",\n"
            "    answer=\"X is something.\",\n"
            "    contexts=[\"Unrelated context.\"],\n"
            ")\n"
            "\n"
            "print(\"Low-quality response scores:\")\n"
            "print(f\"  Faithfulness:      {low_metrics.faithfulness:.2f}\")\n"
            "print(f\"  Relevancy:         {low_metrics.relevancy:.2f}\")\n"
            "print(f\"  Context Precision: {low_metrics.context_precision:.2f}\")\n"
            "print(f\"  Context Recall:    {low_metrics.context_recall:.2f}\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `LLMRAGEvaluator` accepts any callable that returns `RAGMetrics`.\n"
            "- Four dimensions are captured: faithfulness, relevancy,\n"
            "  context precision, and context recall.\n"
            "- Ground truth is optional; omit it for pure faithfulness checks.\n"
            "- Swap the scoring function to upgrade from a mock to a real LLM judge."
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Pipeline Evaluator
# ---------------------------------------------------------------------------
def nb_pipeline_evaluator():
    write_notebook("03_pipeline_evaluator.ipynb", [
        md(
            "# Pipeline Evaluator\n"
            "> Combine retrieval and RAG evaluation into one unified interface.\n"
            "\n"
            "`PipelineEvaluator` orchestrates both retrieval-metric calculation\n"
            "and LLM-based RAG evaluation, giving you a single object that can\n"
            "score your entire pipeline end to end.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import (\n"
            "    PipelineEvaluator,\n"
            "    RetrievalMetricsCalculator,\n"
            "    LLMRAGEvaluator,\n"
            ")\n"
            "from anchor.evaluation.models import RAGMetrics\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md(
            "## Prepare Components\n"
            "Build the retrieval calculator and a mock LLM evaluator first,\n"
            "then wire them into the `PipelineEvaluator`."
        ),
        code(
            "calculator = RetrievalMetricsCalculator()\n"
            "\n"
            "def mock_judge(query, answer, contexts, ground_truth=None):\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.9,\n"
            "        relevancy=0.85,\n"
            "        context_precision=0.8,\n"
            "        context_recall=0.75,\n"
            "    )\n"
            "\n"
            "rag_evaluator = LLMRAGEvaluator(eval_fn=mock_judge)\n"
            "\n"
            "print(\"Components ready:\")\n"
            "print(f\"  RetrievalMetricsCalculator: {type(calculator).__name__}\")\n"
            "print(f\"  LLMRAGEvaluator:            {type(rag_evaluator).__name__}\")"
        ),
        md("## Create the Pipeline Evaluator"),
        code(
            "pipeline_eval = PipelineEvaluator(\n"
            "    retrieval_calculator=calculator,\n"
            "    rag_evaluator=rag_evaluator,\n"
            ")\n"
            "\n"
            "print(f\"PipelineEvaluator created: {type(pipeline_eval).__name__}\")"
        ),
        md(
            "## Build Mock Data\n"
            "We need retrieved items and relevance labels for the retrieval\n"
            "side, plus a query/answer pair for the RAG side."
        ),
        code(
            "retrieved = [\n"
            "    ContextItem(\n"
            "        id=f\"doc-{i}\",\n"
            "        content=f\"Content about topic {i}\",\n"
            "        source=SourceType.RETRIEVAL,\n"
            "        score=1.0 - i * 0.1,\n"
            "        priority=5,\n"
            "        token_count=10,\n"
            "    )\n"
            "    for i in range(5)\n"
            "]\n"
            "relevant_ids = [\"doc-0\", \"doc-1\", \"doc-3\"]\n"
            "\n"
            "query = \"What is context engineering?\"\n"
            "answer = \"Context engineering is the systematic design of LLM context.\"\n"
            "contexts = [item.content for item in retrieved[:3]]\n"
            "ground_truth = \"Context engineering designs the information surrounding LLM calls.\"\n"
            "\n"
            "print(f\"Retrieved items: {len(retrieved)}\")\n"
            "print(f\"Relevant IDs:   {relevant_ids}\")\n"
            "print(f\"Contexts sent:  {len(contexts)}\")"
        ),
        md(
            "## Evaluate Retrieval Only\n"
            "`evaluate_retrieval()` delegates to the `RetrievalMetricsCalculator`."
        ),
        code(
            "ret_metrics = pipeline_eval.evaluate_retrieval(\n"
            "    retrieved=retrieved,\n"
            "    relevant=relevant_ids,\n"
            "    k=5,\n"
            ")\n"
            "\n"
            "print(f\"Precision@5: {ret_metrics.precision:.3f}\")\n"
            "print(f\"Recall@5:    {ret_metrics.recall:.3f}\")\n"
            "print(f\"MRR:         {ret_metrics.mrr:.3f}\")"
        ),
        md(
            "## Evaluate RAG Only\n"
            "`evaluate_rag()` delegates to the `LLMRAGEvaluator`."
        ),
        code(
            "rag_metrics = pipeline_eval.evaluate_rag(\n"
            "    query=query,\n"
            "    answer=answer,\n"
            "    contexts=contexts,\n"
            "    ground_truth=ground_truth,\n"
            ")\n"
            "\n"
            "print(f\"Faithfulness:      {rag_metrics.faithfulness:.2f}\")\n"
            "print(f\"Relevancy:         {rag_metrics.relevancy:.2f}\")\n"
            "print(f\"Context Precision: {rag_metrics.context_precision:.2f}\")\n"
            "print(f\"Context Recall:    {rag_metrics.context_recall:.2f}\")"
        ),
        md(
            "## Full Pipeline Evaluation\n"
            "`evaluate()` runs both retrieval and RAG evaluation in one call."
        ),
        code(
            "full_result = pipeline_eval.evaluate(\n"
            "    retrieved=retrieved,\n"
            "    relevant=relevant_ids,\n"
            "    query=query,\n"
            "    answer=answer,\n"
            "    contexts=contexts,\n"
            "    ground_truth=ground_truth,\n"
            "    k=5,\n"
            ")\n"
            "\n"
            "print(\"Full pipeline evaluation result:\")\n"
            "print(f\"  Type: {type(full_result).__name__}\")\n"
            "print(f\"  Contains retrieval + RAG metrics\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `PipelineEvaluator` unifies retrieval and RAG evaluation.\n"
            "- Use `evaluate_retrieval()` or `evaluate_rag()` independently,\n"
            "  or `evaluate()` for end-to-end scoring.\n"
            "- Swapping components (e.g., a real LLM judge) requires no\n"
            "  changes to the evaluation workflow."
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Batch Evaluator
# ---------------------------------------------------------------------------
def nb_batch_evaluator():
    write_notebook("04_batch_evaluator.ipynb", [
        md(
            "# Batch Evaluation\n"
            "> Evaluate multiple samples at once and aggregate the results.\n"
            "\n"
            "`BatchEvaluator` iterates over an `EvaluationDataset` and\n"
            "produces aggregate metrics across all samples, making it easy\n"
            "to benchmark your pipeline on a test set.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import (\n"
            "    BatchEvaluator,\n"
            "    PipelineEvaluator,\n"
            "    RetrievalMetricsCalculator,\n"
            "    LLMRAGEvaluator,\n"
            ")\n"
            "from anchor.evaluation.models import (\n"
            "    EvaluationSample,\n"
            "    EvaluationDataset,\n"
            "    RAGMetrics,\n"
            ")\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md(
            "## Build the Pipeline Evaluator\n"
            "Reuse the same components from the previous notebook."
        ),
        code(
            "calculator = RetrievalMetricsCalculator()\n"
            "\n"
            "def mock_judge(query, answer, contexts, ground_truth=None):\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.9,\n"
            "        relevancy=0.85,\n"
            "        context_precision=0.8,\n"
            "        context_recall=0.75,\n"
            "    )\n"
            "\n"
            "rag_evaluator = LLMRAGEvaluator(eval_fn=mock_judge)\n"
            "pipeline_eval = PipelineEvaluator(\n"
            "    retrieval_calculator=calculator,\n"
            "    rag_evaluator=rag_evaluator,\n"
            ")\n"
            "\n"
            "print(f\"Pipeline evaluator ready: {type(pipeline_eval).__name__}\")"
        ),
        md(
            "## Create Evaluation Samples\n"
            "Each `EvaluationSample` bundles a query, answer, contexts,\n"
            "retrieved items, relevant IDs, and ground truth."
        ),
        code(
            "def make_retrieved(n=5):\n"
            "    return [\n"
            "        ContextItem(\n"
            "            id=f\"doc-{i}\",\n"
            "            content=f\"Content {i}\",\n"
            "            source=SourceType.RETRIEVAL,\n"
            "            score=1.0 - i * 0.1,\n"
            "            priority=5,\n"
            "            token_count=10,\n"
            "        )\n"
            "        for i in range(n)\n"
            "    ]\n"
            "\n"
            "samples = [\n"
            "    EvaluationSample(\n"
            "        query=\"What is context engineering?\",\n"
            "        answer=\"Context engineering designs LLM inputs.\",\n"
            "        contexts=[\"Context engineering is about input design.\"],\n"
            "        retrieved=make_retrieved(),\n"
            "        relevant=[\"doc-0\", \"doc-1\"],\n"
            "        ground_truth=\"Context engineering is the systematic design of LLM inputs.\",\n"
            "    ),\n"
            "    EvaluationSample(\n"
            "        query=\"What is RAG?\",\n"
            "        answer=\"RAG combines retrieval with generation.\",\n"
            "        contexts=[\"RAG stands for Retrieval-Augmented Generation.\"],\n"
            "        retrieved=make_retrieved(),\n"
            "        relevant=[\"doc-0\", \"doc-2\", \"doc-3\"],\n"
            "        ground_truth=\"RAG augments LLM generation with retrieved context.\",\n"
            "    ),\n"
            "    EvaluationSample(\n"
            "        query=\"What are embeddings?\",\n"
            "        answer=\"Embeddings are vector representations of text.\",\n"
            "        contexts=[\"Embeddings map text to dense vectors.\"],\n"
            "        retrieved=make_retrieved(),\n"
            "        relevant=[\"doc-0\"],\n"
            "        ground_truth=\"Embeddings encode text as numerical vectors.\",\n"
            "    ),\n"
            "]\n"
            "\n"
            "print(f\"Created {len(samples)} evaluation samples.\")\n"
            "for i, s in enumerate(samples):\n"
            "    print(f\"  Sample {i}: query={s.query!r:.40}\")"
        ),
        md(
            "## Build an Evaluation Dataset\n"
            "`EvaluationDataset` is a lightweight container with an\n"
            "`.add_sample()` method."
        ),
        code(
            "dataset = EvaluationDataset()\n"
            "for s in samples:\n"
            "    dataset.add_sample(s)\n"
            "\n"
            "print(f\"Dataset size: {len(dataset.samples)} samples\")"
        ),
        md(
            "## Run Batch Evaluation\n"
            "`BatchEvaluator.evaluate_batch()` processes every sample and\n"
            "returns aggregated metrics."
        ),
        code(
            "batch = BatchEvaluator(evaluator=pipeline_eval)\n"
            "aggregated = batch.evaluate_batch(samples=dataset.samples)\n"
            "\n"
            "print(\"Aggregated retrieval metrics:\")\n"
            "print(f\"  Avg Precision: {aggregated.avg_precision:.3f}\")\n"
            "print(f\"  Avg Recall:    {aggregated.avg_recall:.3f}\")\n"
            "print(f\"  Avg MRR:       {aggregated.avg_mrr:.3f}\")\n"
            "print(f\"  Avg nDCG:      {aggregated.avg_ndcg:.3f}\")"
        ),
        md(
            "## Inspect Individual Results\n"
            "The aggregated object may expose per-sample results for\n"
            "deeper analysis."
        ),
        code(
            "print(f\"Batch evaluator type: {type(batch).__name__}\")\n"
            "print(f\"Aggregated type:      {type(aggregated).__name__}\")\n"
            "print(f\"Samples processed:    {len(dataset.samples)}\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `EvaluationSample` bundles all inputs needed for one evaluation.\n"
            "- `EvaluationDataset` collects samples into a reusable test set.\n"
            "- `BatchEvaluator` runs the pipeline evaluator over every sample\n"
            "  and returns aggregate precision, recall, MRR, and nDCG.\n"
            "- Use batch evaluation to benchmark pipeline changes on a fixed\n"
            "  dataset."
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - A/B Testing
# ---------------------------------------------------------------------------
def nb_ab_testing():
    write_notebook("05_ab_testing.ipynb", [
        md(
            "# A/B Testing\n"
            "> Compare two pipeline configurations with statistical rigour.\n"
            "\n"
            "`ABTestRunner` evaluates two systems on the same samples and\n"
            "determines a winner using aggregate metrics and a p-value\n"
            "for statistical significance.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import (\n"
            "    ABTestRunner,\n"
            "    PipelineEvaluator,\n"
            "    RetrievalMetricsCalculator,\n"
            "    LLMRAGEvaluator,\n"
            ")\n"
            "from anchor.evaluation.models import (\n"
            "    ABTestResult,\n"
            "    EvaluationSample,\n"
            "    RAGMetrics,\n"
            ")\n"
            "from anchor.models import ContextItem, SourceType"
        ),
        md(
            "## Define Two Evaluation Functions\n"
            "System A returns higher scores than System B, simulating\n"
            "a better-performing pipeline."
        ),
        code(
            "def system_a_judge(query, answer, contexts, ground_truth=None):\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.92,\n"
            "        relevancy=0.88,\n"
            "        context_precision=0.85,\n"
            "        context_recall=0.80,\n"
            "    )\n"
            "\n"
            "def system_b_judge(query, answer, contexts, ground_truth=None):\n"
            "    return RAGMetrics(\n"
            "        faithfulness=0.70,\n"
            "        relevancy=0.65,\n"
            "        context_precision=0.60,\n"
            "        context_recall=0.55,\n"
            "    )\n"
            "\n"
            "print(\"System A: high-quality mock scores\")\n"
            "print(\"System B: lower-quality mock scores\")"
        ),
        md("## Build Two Pipeline Evaluators"),
        code(
            "calculator = RetrievalMetricsCalculator()\n"
            "\n"
            "pipeline_a = PipelineEvaluator(\n"
            "    retrieval_calculator=calculator,\n"
            "    rag_evaluator=LLMRAGEvaluator(eval_fn=system_a_judge),\n"
            ")\n"
            "pipeline_b = PipelineEvaluator(\n"
            "    retrieval_calculator=calculator,\n"
            "    rag_evaluator=LLMRAGEvaluator(eval_fn=system_b_judge),\n"
            ")\n"
            "\n"
            "print(f\"Pipeline A ready: {type(pipeline_a).__name__}\")\n"
            "print(f\"Pipeline B ready: {type(pipeline_b).__name__}\")"
        ),
        md(
            "## Prepare Test Samples\n"
            "Both systems will be evaluated on the same set of samples."
        ),
        code(
            "def make_retrieved(n=5):\n"
            "    return [\n"
            "        ContextItem(\n"
            "            id=f\"doc-{i}\",\n"
            "            content=f\"Content {i}\",\n"
            "            source=SourceType.RETRIEVAL,\n"
            "            score=1.0 - i * 0.1,\n"
            "            priority=5,\n"
            "            token_count=10,\n"
            "        )\n"
            "        for i in range(n)\n"
            "    ]\n"
            "\n"
            "samples = [\n"
            "    EvaluationSample(\n"
            "        query=f\"Question {i}?\",\n"
            "        answer=f\"Answer {i}.\",\n"
            "        contexts=[f\"Context for question {i}.\"],\n"
            "        retrieved=make_retrieved(),\n"
            "        relevant=[\"doc-0\", \"doc-1\"],\n"
            "        ground_truth=f\"Ground truth {i}.\",\n"
            "    )\n"
            "    for i in range(10)\n"
            "]\n"
            "\n"
            "print(f\"Test samples: {len(samples)}\")"
        ),
        md(
            "## Create and Run the A/B Test\n"
            "`ABTestRunner` takes a base evaluator and the number of samples."
        ),
        code(
            "runner = ABTestRunner(\n"
            "    evaluator=pipeline_a,\n"
            "    num_samples=len(samples),\n"
            ")\n"
            "\n"
            "print(f\"ABTestRunner created\")\n"
            "print(f\"  Evaluator type: {type(runner.evaluator).__name__}\")\n"
            "print(f\"  Num samples:    {runner.num_samples}\")"
        ),
        md(
            "## Inspect ABTestResult Schema\n"
            "The result exposes per-system metrics, the winner, and a\n"
            "p-value for significance."
        ),
        code(
            "# In a full run you would call:\n"
            "#   result = runner.run(\n"
            "#       system_a=pipeline_a,\n"
            "#       system_b=pipeline_b,\n"
            "#       samples=samples,\n"
            "#   )\n"
            "#\n"
            "# For demonstration, we show the expected result structure:\n"
            "\n"
            "print(\"ABTestResult fields:\")\n"
            "print(\"  .system_a_metrics  -- aggregated metrics for system A\")\n"
            "print(\"  .system_b_metrics  -- aggregated metrics for system B\")\n"
            "print(\"  .winner            -- 'A', 'B', or 'tie'\")\n"
            "print(\"  .p_value           -- statistical significance\")"
        ),
        code(
            "# Simulated result for display purposes\n"
            "print(\"\\nSimulated A/B test result:\")\n"
            "print(f\"  System A faithfulness: 0.92\")\n"
            "print(f\"  System B faithfulness: 0.70\")\n"
            "print(f\"  Winner:  A\")\n"
            "print(f\"  p-value: 0.003\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `ABTestRunner` compares two pipeline configurations on identical data.\n"
            "- The result includes per-system aggregate metrics, a declared\n"
            "  winner, and a p-value for confidence.\n"
            "- Run A/B tests before deploying pipeline changes to production.\n"
            "- Increase `num_samples` for tighter statistical confidence."
        ),
    ])


# ---------------------------------------------------------------------------
# 06 - Human Evaluation
# ---------------------------------------------------------------------------
def nb_human_evaluation():
    write_notebook("06_human_evaluation.ipynb", [
        md(
            "# Human Evaluation\n"
            "> Collect and manage human judgments alongside automated metrics.\n"
            "\n"
            "`HumanEvaluationCollector` stores qualitative feedback from human\n"
            "reviewers, letting you combine expert judgment with algorithmic\n"
            "scores for a complete evaluation picture.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.evaluation import HumanEvaluationCollector\n"
            "from anchor.evaluation.models import HumanJudgment"
        ),
        md(
            "## Create a Collector\n"
            "The collector acts as an in-memory store for human judgments."
        ),
        code(
            "collector = HumanEvaluationCollector()\n"
            "\n"
            "print(f\"Collector type: {type(collector).__name__}\")\n"
            "print(f\"Judgments so far: {len(collector.get_judgments())}\")"
        ),
        md(
            "## Add a Single Judgment\n"
            "`HumanJudgment` captures who evaluated, what they saw, their\n"
            "numeric rating, and free-text feedback."
        ),
        code(
            "judgment = HumanJudgment(\n"
            "    evaluator_id=\"reviewer-1\",\n"
            "    query=\"What is prompt engineering?\",\n"
            "    answer=\"Prompt engineering is the practice of designing inputs to LLMs.\",\n"
            "    rating=4,\n"
            "    feedback=\"Good answer, covers the core concept clearly.\",\n"
            ")\n"
            "\n"
            "collector.add_judgment(judgment)\n"
            "\n"
            "print(f\"Added judgment from: {judgment.evaluator_id}\")\n"
            "print(f\"Rating:   {judgment.rating}\")\n"
            "print(f\"Feedback: {judgment.feedback}\")"
        ),
        md(
            "## Add Multiple Judgments\n"
            "Simulate several reviewers scoring different query-answer pairs."
        ),
        code(
            "more_judgments = [\n"
            "    HumanJudgment(\n"
            "        evaluator_id=\"reviewer-1\",\n"
            "        query=\"What is RAG?\",\n"
            "        answer=\"RAG combines retrieval with generation.\",\n"
            "        rating=5,\n"
            "        feedback=\"Excellent -- concise and accurate.\",\n"
            "    ),\n"
            "    HumanJudgment(\n"
            "        evaluator_id=\"reviewer-2\",\n"
            "        query=\"What is prompt engineering?\",\n"
            "        answer=\"Prompt engineering is the practice of designing inputs to LLMs.\",\n"
            "        rating=3,\n"
            "        feedback=\"Correct but could be more detailed.\",\n"
            "    ),\n"
            "    HumanJudgment(\n"
            "        evaluator_id=\"reviewer-2\",\n"
            "        query=\"What are embeddings?\",\n"
            "        answer=\"Embeddings map words to numbers.\",\n"
            "        rating=2,\n"
            "        feedback=\"Too vague, missing key details.\",\n"
            "    ),\n"
            "    HumanJudgment(\n"
            "        evaluator_id=\"reviewer-3\",\n"
            "        query=\"What is RAG?\",\n"
            "        answer=\"RAG combines retrieval with generation.\",\n"
            "        rating=4,\n"
            "        feedback=\"Solid answer, could mention accuracy benefits.\",\n"
            "    ),\n"
            "]\n"
            "\n"
            "for j in more_judgments:\n"
            "    collector.add_judgment(j)\n"
            "\n"
            "print(f\"Total judgments: {len(collector.get_judgments())}\")"
        ),
        md(
            "## Retrieve All Judgments\n"
            "`get_judgments()` returns the full list."
        ),
        code(
            "judgments = collector.get_judgments()\n"
            "\n"
            "print(f\"{'Reviewer':<12} {'Query':<30} {'Rating':>6}\")\n"
            "print(\"-\" * 50)\n"
            "for j in judgments:\n"
            "    print(f\"{j.evaluator_id:<12} {j.query:<30} {j.rating:>6}\")"
        ),
        md(
            "## Compute Summary Statistics\n"
            "Basic aggregation across all collected judgments."
        ),
        code(
            "ratings = [j.rating for j in judgments]\n"
            "avg_rating = sum(ratings) / len(ratings)\n"
            "\n"
            "reviewers = set(j.evaluator_id for j in judgments)\n"
            "queries = set(j.query for j in judgments)\n"
            "\n"
            "print(f\"Total judgments:  {len(judgments)}\")\n"
            "print(f\"Unique reviewers: {len(reviewers)}\")\n"
            "print(f\"Unique queries:   {len(queries)}\")\n"
            "print(f\"Average rating:   {avg_rating:.2f}\")\n"
            "print(f\"Min rating:       {min(ratings)}\")\n"
            "print(f\"Max rating:       {max(ratings)}\")"
        ),
        md(
            "## Per-Reviewer Breakdown\n"
            "See how each reviewer scores on average."
        ),
        code(
            "from collections import defaultdict\n"
            "\n"
            "by_reviewer = defaultdict(list)\n"
            "for j in judgments:\n"
            "    by_reviewer[j.evaluator_id].append(j.rating)\n"
            "\n"
            "print(f\"{'Reviewer':<12} {'Count':>6} {'Avg':>6}\")\n"
            "print(\"-\" * 26)\n"
            "for reviewer, scores in sorted(by_reviewer.items()):\n"
            "    avg = sum(scores) / len(scores)\n"
            "    print(f\"{reviewer:<12} {len(scores):>6} {avg:>6.2f}\")"
        ),
        md(
            "## Key Takeaways\n"
            "\n"
            "- `HumanEvaluationCollector` provides a simple API for gathering\n"
            "  qualitative ratings alongside automated metrics.\n"
            "- `HumanJudgment` captures evaluator ID, query, answer, numeric\n"
            "  rating, and free-text feedback.\n"
            "- Combine human evaluation with retrieval and RAG metrics for a\n"
            "  complete quality picture.\n"
            "- Analyse inter-reviewer agreement to gauge evaluation reliability."
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    nb_retrieval_metrics()
    nb_llm_evaluator()
    nb_pipeline_evaluator()
    nb_batch_evaluator()
    nb_ab_testing()
    nb_human_evaluation()

    print("\nDone -- 6 evaluation notebooks generated.")


if __name__ == "__main__":
    main()
