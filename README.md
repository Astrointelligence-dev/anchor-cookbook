# Anchor Cookbook

Hands-on Jupyter Notebook recipes covering the entire [Anchor](https://github.com/arthurgranja/astro-context) context engineering toolkit API. Clone this repo, install dependencies, and start exploring.

## Quick Start

```bash
git clone https://github.com/arthurgranja/anchor-cookbook.git
cd anchor-cookbook
uv venv && uv pip install -e .
jupyter notebook
```

Start with [`setup.ipynb`](setup.ipynb) to verify your environment.

## Recipes

| Module | Recipe | Description |
|--------|--------|-------------|
| **Pipeline** | [Basic Pipeline](00-pipeline/01_basic_pipeline.ipynb) | `ContextPipeline` with system prompt, memory, and formatter |
| **Pipeline** | [Built-in Steps](00-pipeline/02_builtin_steps.ipynb) | `retriever_step`, `filter_step`, `reranker_step`, and more |
| **Pipeline** | [Custom Steps](00-pipeline/03_custom_steps.ipynb) | `@pipeline.step` decorator for custom logic |
| **Pipeline** | [Async Pipelines](00-pipeline/04_async_pipelines.ipynb) | `abuild()` and async step execution |
| **Pipeline** | [Pipeline Diagnostics](00-pipeline/05_pipeline_diagnostics.ipynb) | `ContextResult` timing, tokens, step diagnostics |
| **Pipeline** | [Context Window](00-pipeline/06_context_window.ipynb) | `ContextWindow`, `ContextItem`, priority ranking |
| **Pipeline** | [Enrichers](00-pipeline/07_enrichers.ipynb) | `MemoryContextEnricher`, `ContextQueryEnricher` |
| **Memory** | [Sliding Window](01-memory/01_sliding_window.ipynb) | `SlidingWindowMemory` with token cap |
| **Memory** | [Summary Buffer](01-memory/02_summary_buffer.ipynb) | `SummaryBufferMemory` progressive summarization |
| **Memory** | [Graph Memory](01-memory/03_graph_memory.ipynb) | `SimpleGraphMemory` entity-relationship tracking |
| **Memory** | [Memory Manager](01-memory/04_memory_manager.ipynb) | `MemoryManager` facade for conversation + persistent memory |
| **Memory** | [Eviction Policies](01-memory/05_eviction_policies.ipynb) | FIFO, importance-based, paired eviction |
| **Memory** | [Decay Strategies](01-memory/06_decay_strategies.ipynb) | Ebbinghaus, linear decay, recency scoring |
| **Memory** | [Consolidation](01-memory/07_consolidation.ipynb) | Content-hash dedup, similarity merging |
| **Memory** | [Garbage Collection](01-memory/08_garbage_collection.ipynb) | Two-phase GC: expired + decayed pruning |
| **Retrieval** | [Dense Retriever](02-retrieval/01_dense_retriever.ipynb) | Embedding-based semantic search |
| **Retrieval** | [Sparse Retriever](02-retrieval/02_sparse_retriever.ipynb) | BM25 keyword matching |
| **Retrieval** | [Hybrid Retriever](02-retrieval/03_hybrid_retriever.ipynb) | Dense + sparse with Reciprocal Rank Fusion |
| **Retrieval** | [Scored Memory Retriever](02-retrieval/04_scored_memory_retriever.ipynb) | Multi-signal memory retrieval |
| **Retrieval** | [Async Retrievers](02-retrieval/05_async_retrievers.ipynb) | `AsyncDenseRetriever`, `AsyncHybridRetriever` |
| **Retrieval** | [Late Interaction](02-retrieval/06_late_interaction.ipynb) | ColBERT-style token-level matching |
| **Retrieval** | [Rerankers](02-retrieval/07_rerankers.ipynb) | Cross-encoder, FlashRank, `RerankerPipeline` |
| **Retrieval** | [Routers](02-retrieval/08_routers.ipynb) | Callback, keyword, metadata query routing |
| **Retrieval** | [Custom Retriever](02-retrieval/09_custom_retriever.ipynb) | Implementing the `Retriever` protocol |
| **Agents** | [Basic Agent](03-agents/01_basic_agent.ipynb) | `Agent` with system prompt, memory, `chat()` |
| **Agents** | [Tool Decorator](03-agents/02_tool_decorator.ipynb) | `@tool` decorator, `AgentTool` schemas |
| **Agents** | [Skills System](03-agents/03_skills_system.ipynb) | `Skill`, `SkillRegistry`, SKILL.md loading |
| **Agents** | [Agent with Retrieval](03-agents/04_agent_with_retrieval.ipynb) | Agent + ContextPipeline integration |
| **Agents** | [Streaming Agent](03-agents/05_streaming_agent.ipynb) | Streaming responses, `StreamDelta` |
| **Ingestion** | [Document Ingester](04-ingestion/01_document_ingester.ipynb) | `DocumentIngester` orchestration |
| **Ingestion** | [Chunking Strategies](04-ingestion/02_chunking_strategies.ipynb) | Fixed, recursive, semantic, code, table-aware |
| **Ingestion** | [Parsers](04-ingestion/03_parsers.ipynb) | Text, Markdown, HTML, PDF parsing |
| **Ingestion** | [Metadata Enrichment](04-ingestion/04_metadata_enrichment.ipynb) | Auto IDs, metadata extraction |
| **Ingestion** | [Parent-Child Chunks](04-ingestion/05_parent_child_chunks.ipynb) | Hierarchical chunking |
| **Query** | [HyDE Transformer](05-query/01_hyde_transformer.ipynb) | Hypothetical document generation |
| **Query** | [Multi-Query](05-query/02_multi_query.ipynb) | Query expansion + ensemble |
| **Query** | [Decomposition](05-query/03_decomposition.ipynb) | Sub-query breakdown |
| **Query** | [Step-Back](05-query/04_step_back.ipynb) | Abstraction-level reformulation |
| **Query** | [Classifiers](05-query/05_classifiers.ipynb) | Keyword, callback, embedding classifiers |
| **Query** | [Query Pipeline](05-query/06_query_pipeline.ipynb) | `QueryTransformPipeline`, chaining |
| **Evaluation** | [Retrieval Metrics](06-evaluation/01_retrieval_metrics.ipynb) | NDCG, MAP, MRR, P@K, R@K |
| **Evaluation** | [LLM Evaluator](06-evaluation/02_llm_evaluator.ipynb) | LLM-as-judge for RAG quality |
| **Evaluation** | [Pipeline Evaluator](06-evaluation/03_pipeline_evaluator.ipynb) | End-to-end pipeline assessment |
| **Evaluation** | [Batch Evaluator](06-evaluation/04_batch_evaluator.ipynb) | Dataset-level runs |
| **Evaluation** | [A/B Testing](06-evaluation/05_ab_testing.ipynb) | `ABTestRunner` comparative harness |
| **Evaluation** | [Human Evaluation](06-evaluation/06_human_evaluation.ipynb) | Human judgment collection |
| **Observability** | [Tracing](07-observability/01_tracing.ipynb) | `Tracer`, spans, structured traces |
| **Observability** | [Exporters](07-observability/02_exporters.ipynb) | Console, file, in-memory, OTLP exporters |
| **Observability** | [Cost Tracking](07-observability/03_cost_tracking.ipynb) | Token cost calculation |
| **Observability** | [Metrics](07-observability/04_metrics.ipynb) | OpenTelemetry metrics, collectors |
| **Formatters** | [Anthropic Formatter](08-formatters/01_anthropic_formatter.ipynb) | Claude API output format |
| **Formatters** | [OpenAI Formatter](08-formatters/02_openai_formatter.ipynb) | OpenAI API output format |
| **Formatters** | [Generic Text](08-formatters/03_generic_text.ipynb) | Plain text output |
| **Formatters** | [Custom Formatter](08-formatters/04_custom_formatter.ipynb) | Implementing custom formatters |
| **Multimodal** | [Multimodal Converter](09-multimodal/01_multimodal_converter.ipynb) | Document-to-multimodal conversion |
| **Multimodal** | [Image Encoding](09-multimodal/02_image_encoding.ipynb) | Image description encoder |
| **Multimodal** | [Table Extraction](09-multimodal/03_table_extraction.ipynb) | Markdown/HTML table parsing |
| **Tokens** | [Token Budgets](10-tokens/01_token_budgets.ipynb) | `TokenBudget`, allocations, overflow |
| **Tokens** | [Budget Presets](10-tokens/02_budget_presets.ipynb) | Chat, RAG, agent budget defaults |
| **Tokens** | [Tiktoken Counter](10-tokens/03_tiktoken_counter.ipynb) | `TiktokenCounter` tokenization |
| **Tokens** | [Custom Tokenizer](10-tokens/04_custom_tokenizer.ipynb) | Implementing `Tokenizer` protocol |
| **Caching** | [Cache Backend](11-caching/01_cache_backend.ipynb) | `CacheBackend` protocol |
| **Caching** | [In-Memory Cache](11-caching/02_in_memory_cache.ipynb) | `InMemoryCacheBackend` usage |
| **Storage** | [Vector Store](12-storage/01_vector_store.ipynb) | `InMemoryVectorStore` |
| **Storage** | [Document Store](12-storage/02_document_store.ipynb) | `InMemoryDocumentStore` |
| **Storage** | [Context Store](12-storage/03_context_store.ipynb) | `InMemoryContextStore` |
| **Storage** | [Entry Store](12-storage/04_entry_store.ipynb) | `InMemoryEntryStore` |
| **Storage** | [JSON File Store](12-storage/05_json_file_store.ipynb) | `JsonFileMemoryStore` persistence |
| **Storage** | [Custom Store](12-storage/06_custom_store.ipynb) | Implementing storage protocols |

**Total: 69 recipes** across 13 modules.

## Requirements

- Python 3.11+
- [astro-anchor](https://pypi.org/project/astro-anchor/) (installed via requirements.txt)

## License

MIT
