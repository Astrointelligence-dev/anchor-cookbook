"""Generate all 5 Jupyter notebooks for the Agents module of the Anchor cookbook."""

import nbformat
import os

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "03-agents"))
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
# 01 - Basic Agent
# ---------------------------------------------------------------------------
def nb_basic_agent():
    write_notebook("01_basic_agent.ipynb", [
        md(
            "# Basic Agent Configuration\n"
            "> Set up an Anchor Agent with a model, system prompt, and memory.\n"
            "\n"
            "The `Agent` class is the high-level entry point for agentic AI\n"
            "applications. It wraps the context pipeline, Anthropic SDK, and\n"
            "tool-use loop into a single fluent interface.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.agent import Agent\n"
            "from anchor.memory import SlidingWindowMemory"
        ),
        md(
            "## Create an Agent\n"
            "The `Agent` constructor accepts a model identifier and optional\n"
            "parameters for token limits, retry behaviour, and round caps."
        ),
        code(
            "agent = Agent(\n"
            "    model=\"claude-haiku-4-5-20251001\",\n"
            "    max_response_tokens=2048,\n"
            "    max_rounds=10,\n"
            ")\n"
            "\n"
            "print(f\"Agent created\")\n"
            "print(f\"  model:              claude-haiku-4-5-20251001\")\n"
            "print(f\"  max_response_tokens: 2048\")\n"
            "print(f\"  max_rounds:          10\")"
        ),
        md(
            "## Set a System Prompt\n"
            "`.with_system_prompt()` returns `self` for fluent chaining."
        ),
        code(
            "agent.with_system_prompt(\"You are a helpful assistant.\")\n"
            "\n"
            "print(\"System prompt set:\", bool(agent._system_prompt))\n"
            "print(\"Prompt text:\", repr(agent._system_prompt))"
        ),
        md(
            "## Attach Memory\n"
            "`SlidingWindowMemory` keeps recent conversation turns within a\n"
            "token budget. The agent uses it to maintain conversation context."
        ),
        code(
            "memory = SlidingWindowMemory(max_tokens=4096)\n"
            "\n"
            "print(f\"Memory type:   {type(memory).__name__}\")\n"
            "print(f\"Token budget:  {memory.max_tokens}\")\n"
            "print(f\"Current turns: {len(memory.turns)}\")"
        ),
        md(
            "## Wire Memory into the Agent\n"
            "`.with_memory()` connects the memory to the underlying pipeline\n"
            "so conversation history is included in every API call."
        ),
        code(
            "agent.with_memory(memory)\n"
            "\n"
            "print(\"Memory attached:\", agent.memory is not None)\n"
            "print(\"Pipeline has memory:\", agent.pipeline is not None)"
        ),
        md(
            "## Fluent Chaining Pattern\n"
            "In practice, you chain the configuration calls together."
        ),
        code(
            "# Typical one-liner pattern (shown for reference -- not executed\n"
            "# since Agent() requires an Anthropic client for chat calls):\n"
            "#\n"
            "# agent = (\n"
            "#     Agent(model=\"claude-haiku-4-5-20251001\")\n"
            "#     .with_system_prompt(\"You are a helpful assistant.\")\n"
            "#     .with_memory(SlidingWindowMemory(max_tokens=4096))\n"
            "# )\n"
            "#\n"
            "# for chunk in agent.chat(\"Hello!\"):\n"
            "#     print(chunk, end=\"\", flush=True)\n"
            "\n"
            "print(\"Fluent pattern shown above.\")\n"
            "print(\"agent.chat() requires an ANTHROPIC_API_KEY at runtime.\")"
        ),
        md(
            "## Inspect the Pipeline\n"
            "The agent exposes its underlying `ContextPipeline` for\n"
            "debugging and introspection."
        ),
        code(
            "pipeline = agent.pipeline\n"
            "\n"
            "print(f\"Pipeline type: {type(pipeline).__name__}\")\n"
            "print(f\"Has formatter:  True\")  # AnthropicFormatter is set by default\n"
            "print(f\"Last result:   {agent.last_result}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `Agent` combines pipeline + API client + tool loop in one class\n"
            "- `.with_system_prompt()`, `.with_memory()`, `.with_tools()` return `self` for chaining\n"
            "- `SlidingWindowMemory` is the simplest memory backend for agents\n"
            "- `agent.chat(msg)` is an iterator that yields streaming text chunks\n"
            "- Access the pipeline via `agent.pipeline` and last build via `agent.last_result`\n"
            "\n"
            "**Next:** [Tool Decorator](02_tool_decorator.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 02 - Tool Decorator
# ---------------------------------------------------------------------------
def nb_tool_decorator():
    write_notebook("02_tool_decorator.ipynb", [
        md(
            "# Tool Decorator\n"
            "> Turn plain Python functions into agent-callable tools with a single decorator.\n"
            "\n"
            "The `@tool` decorator auto-generates an `AgentTool` from the function's\n"
            "type hints and docstring. Tools can be exported to Anthropic, OpenAI, or\n"
            "generic schema formats.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.agent import tool, AgentTool"
        ),
        md(
            "## Basic `@tool` Decorator\n"
            "Apply `@tool` to auto-generate the tool name, description, and\n"
            "JSON Schema from the function signature."
        ),
        code(
            "@tool\n"
            "def calculate_sum(a: int, b: int) -> int:\n"
            "    \"\"\"Add two numbers together.\"\"\"\n"
            "    return a + b\n"
            "\n"
            "print(f\"Type: {type(calculate_sum).__name__}\")\n"
            "print(f\"Name: {calculate_sum.name}\")\n"
            "print(f\"Description: {calculate_sum.description}\")"
        ),
        md(
            "## `@tool` with Explicit Name and Description\n"
            "Override the auto-generated name and description when needed."
        ),
        code(
            "@tool(name=\"greet\", description=\"Generate a greeting\")\n"
            "def greet_user(name: str) -> str:\n"
            "    \"\"\"Greet a user by name.\"\"\"\n"
            "    return f\"Hello, {name}!\"\n"
            "\n"
            "print(f\"Name: {greet_user.name}\")        # 'greet' (overridden)\n"
            "print(f\"Description: {greet_user.description}\")  # 'Generate a greeting'"
        ),
        md(
            "## Inspect the Generated Schema\n"
            "The `input_schema` property contains the JSON Schema derived from\n"
            "type hints."
        ),
        code(
            "import json\n"
            "\n"
            "print(\"=== calculate_sum input_schema ===\")\n"
            "print(json.dumps(calculate_sum.input_schema, indent=2))"
        ),
        md(
            "## Export to Provider Formats\n"
            "`AgentTool` can export to Anthropic, OpenAI, or a generic format."
        ),
        code(
            "print(\"=== Anthropic Schema ===\")\n"
            "print(json.dumps(calculate_sum.to_anthropic_schema(), indent=2))\n"
            "\n"
            "print(\"\\n=== OpenAI Schema ===\")\n"
            "print(json.dumps(calculate_sum.to_openai_schema(), indent=2))"
        ),
        md(
            "## Direct Invocation\n"
            "The original function is accessible via the `.fn` attribute."
        ),
        code(
            "# Call the underlying function directly\n"
            "result = calculate_sum.fn(a=3, b=5)\n"
            "print(f\"calculate_sum(3, 5) = {result}\")\n"
            "\n"
            "greeting = greet_user.fn(name=\"Alice\")\n"
            "print(f\"greet('Alice') = {greeting}\")"
        ),
        md(
            "## Input Validation\n"
            "`AgentTool.validate_input()` checks tool input against the schema."
        ),
        code(
            "# Valid input\n"
            "ok, err = calculate_sum.validate_input({\"a\": 3, \"b\": 5})\n"
            "print(f\"Valid input:   ok={ok}, err={err!r}\")\n"
            "\n"
            "# Missing required field\n"
            "ok, err = calculate_sum.validate_input({\"a\": 3})\n"
            "print(f\"Missing field: ok={ok}, err={err!r}\")\n"
            "\n"
            "# Wrong type\n"
            "ok, err = calculate_sum.validate_input({\"a\": \"three\", \"b\": 5})\n"
            "print(f\"Wrong type:    ok={ok}, err={err!r}\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `@tool` auto-generates `AgentTool` from function signature + docstring\n"
            "- Override `name` and `description` with `@tool(name=..., description=...)`\n"
            "- `.input_schema` holds the JSON Schema for the tool's parameters\n"
            "- `.to_anthropic_schema()` / `.to_openai_schema()` export to provider formats\n"
            "- `.fn` gives access to the raw callable; `.validate_input()` checks schema compliance\n"
            "\n"
            "**Next:** [Skills System](03_skills_system.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 03 - Skills System
# ---------------------------------------------------------------------------
def nb_skills_system():
    write_notebook("03_skills_system.ipynb", [
        md(
            "# Skills System\n"
            "> Organise tools into named, discoverable skill groups with on-demand activation.\n"
            "\n"
            "Skills let you bundle related tools together and control when they\n"
            "become available to the agent. *Always* skills are loaded immediately;\n"
            "*on_demand* skills are advertised but only activated when the agent\n"
            "explicitly requests them.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.agent import Skill, SkillRegistry, tool, AgentTool"
        ),
        md(
            "## Define Tools for the Skill\n"
            "Create a pair of tools that will be grouped into a single skill."
        ),
        code(
            "@tool\n"
            "def search_docs(query: str) -> str:\n"
            "    \"\"\"Search documentation.\"\"\"\n"
            "    return f\"Results for: {query}\"\n"
            "\n"
            "@tool\n"
            "def save_note(content: str) -> str:\n"
            "    \"\"\"Save a note.\"\"\"\n"
            "    return f\"Saved: {content}\"\n"
            "\n"
            "print(f\"Tools created: {search_docs.name}, {save_note.name}\")"
        ),
        md(
            "## Create a Skill\n"
            "A `Skill` bundles tools with a name, description, instructions,\n"
            "activation mode, and optional tags."
        ),
        code(
            "research_skill = Skill(\n"
            "    name=\"research\",\n"
            "    description=\"Research and note-taking tools\",\n"
            "    instructions=\"Use these tools to research topics and save notes.\",\n"
            "    tools=(search_docs, save_note),\n"
            "    activation=\"on_demand\",\n"
            "    tags=(\"research\", \"notes\"),\n"
            ")\n"
            "\n"
            "print(f\"Skill name:        {research_skill.name}\")\n"
            "print(f\"Description:       {research_skill.description}\")\n"
            "print(f\"Activation:        {research_skill.activation}\")\n"
            "print(f\"Tags:              {research_skill.tags}\")\n"
            "print(f\"Tools in skill:    {len(research_skill.tools)}\")\n"
            "for t in research_skill.tools:\n"
            "    print(f\"  - {t.name}: {t.description}\")"
        ),
        md(
            "## Skill Registry\n"
            "`SkillRegistry` manages registration, activation, and tool discovery."
        ),
        code(
            "registry = SkillRegistry()\n"
            "registry.register(research_skill)\n"
            "\n"
            "print(f\"Registered skills: {list(registry._skills.keys())}\")\n"
            "print(f\"Is 'research' active? {registry.is_active('research')}\")\n"
            "print(f\"On-demand skills: {[s.name for s in registry.on_demand_skills()]}\")"
        ),
        md(
            "## Activate an On-Demand Skill\n"
            "On-demand skills become active only after explicit activation."
        ),
        code(
            "activated = registry.activate(\"research\")\n"
            "\n"
            "print(f\"Activated: {activated.name}\")\n"
            "print(f\"Is 'research' active now? {registry.is_active('research')}\")"
        ),
        md(
            "## Retrieve Active Tools\n"
            "`active_tools()` returns all tools from currently-active skills."
        ),
        code(
            "active_tools = registry.active_tools()\n"
            "\n"
            "print(f\"Active tools: {len(active_tools)}\")\n"
            "for t in active_tools:\n"
            "    print(f\"  - {t.name}: {t.description}\")"
        ),
        md(
            "## Skill Discovery Prompt\n"
            "The registry generates a prompt listing available on-demand skills\n"
            "for the agent's system message."
        ),
        code(
            "# Deactivate to see the discovery prompt\n"
            "registry.deactivate(\"research\")\n"
            "discovery = registry.skill_discovery_prompt()\n"
            "\n"
            "print(\"Discovery prompt sent to the agent:\")\n"
            "print(discovery)"
        ),
        md(
            "## Always-On Skills\n"
            "Skills with `activation=\"always\"` are active from the moment\n"
            "they are registered."
        ),
        code(
            "@tool\n"
            "def get_time() -> str:\n"
            "    \"\"\"Get the current time.\"\"\"\n"
            "    return \"2026-03-13T10:30:00Z\"\n"
            "\n"
            "utility_skill = Skill(\n"
            "    name=\"utility\",\n"
            "    description=\"General utility tools\",\n"
            "    tools=(get_time,),\n"
            "    activation=\"always\",\n"
            ")\n"
            "\n"
            "registry2 = SkillRegistry()\n"
            "registry2.register(utility_skill)\n"
            "\n"
            "print(f\"Is 'utility' active? {registry2.is_active('utility')}\")\n"
            "print(f\"Active tools: {[t.name for t in registry2.active_tools()]}\")"
        ),
        md(
            "## Built-in Skills\n"
            "Anchor ships pre-built memory and RAG skills."
        ),
        code(
            "from anchor.agent import memory_skill, rag_skill, memory_tools, rag_tools\n"
            "\n"
            "print(\"Built-in skills available:\")\n"
            "print(f\"  memory_skill: {type(memory_skill).__name__}\")\n"
            "print(f\"  rag_skill:    {type(rag_skill).__name__}\")\n"
            "print(f\"  memory_tools: callable that returns list[AgentTool]\")\n"
            "print(f\"  rag_tools:    callable that returns list[AgentTool]\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `Skill` bundles related tools with metadata and activation mode\n"
            "- `SkillRegistry` manages registration, activation, and discovery\n"
            "- `activation=\"on_demand\"` skills are advertised but loaded lazily\n"
            "- `activation=\"always\"` skills are active from registration\n"
            "- `active_tools()` returns tools from all currently-active skills\n"
            "- Anchor provides built-in `memory_skill` and `rag_skill`\n"
            "\n"
            "**Next:** [Agent with Retrieval](04_agent_with_retrieval.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 04 - Agent with Retrieval
# ---------------------------------------------------------------------------
def nb_agent_with_retrieval():
    write_notebook("04_agent_with_retrieval.ipynb", [
        md(
            "# Agent with Retrieval\n"
            "> Combine Agent + ContextPipeline + retriever for agentic RAG.\n"
            "\n"
            "This notebook shows how to wire up an Agent with a retriever so\n"
            "relevant documents are automatically injected into the context\n"
            "window before every API call.\n"
            "\n"
            "**Time:** ~7 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.agent import Agent, tool, Skill, SkillRegistry\n"
            "from anchor.memory import SlidingWindowMemory\n"
            "from anchor.models import ContextItem, SourceType\n"
            "from anchor.pipeline import ContextPipeline"
        ),
        md(
            "## Create a Mock Retriever\n"
            "In production, this would be a vector store or search API.\n"
            "Here we simulate retrieval with a static document set."
        ),
        code(
            "# Simulated document corpus\n"
            "DOCS = {\n"
            "    \"anchor-overview\": \"Anchor is a context engineering toolkit for building \"\n"
            "        \"AI applications. It provides pipelines, memory, retrieval, and agent \"\n"
            "        \"orchestration.\",\n"
            "    \"anchor-agents\": \"The Agent class wraps ContextPipeline + Anthropic SDK + \"\n"
            "        \"tool loop. Use .with_system_prompt(), .with_memory(), and .with_tools() \"\n"
            "        \"for configuration.\",\n"
            "    \"anchor-memory\": \"Anchor memory supports sliding window, summary buffer, \"\n"
            "        \"and graph memory strategies. MemoryManager is the unified facade.\",\n"
            "    \"anchor-retrieval\": \"Retrievers fetch relevant documents at query time. \"\n"
            "        \"Results are injected as ContextItems into the pipeline.\",\n"
            "    \"anchor-skills\": \"Skills bundle tools into discoverable groups. Use \"\n"
            "        \"always-on or on-demand activation to control tool availability.\",\n"
            "}\n"
            "\n"
            "print(f\"Document corpus: {len(DOCS)} documents\")\n"
            "for doc_id in DOCS:\n"
            "    print(f\"  - {doc_id}\")"
        ),
        md(
            "## Define a Search Tool\n"
            "The search tool performs keyword matching against the corpus\n"
            "and returns matching documents."
        ),
        code(
            "@tool\n"
            "def search_knowledge_base(query: str) -> str:\n"
            "    \"\"\"Search the knowledge base for relevant documents.\"\"\"\n"
            "    query_lower = query.lower()\n"
            "    results = []\n"
            "    for doc_id, content in DOCS.items():\n"
            "        if any(word in content.lower() for word in query_lower.split()):\n"
            "            results.append(f\"[{doc_id}] {content}\")\n"
            "    if not results:\n"
            "        return \"No documents found.\"\n"
            "    return \"\\n\\n\".join(results[:3])\n"
            "\n"
            "# Test the search tool\n"
            "result = search_knowledge_base.fn(query=\"memory\")\n"
            "print(\"Search results for 'memory':\")\n"
            "print(result)"
        ),
        md(
            "## Build Context Items from Retrieved Documents\n"
            "Show how retrieved content becomes `ContextItem` objects\n"
            "that feed into the pipeline."
        ),
        code(
            "# Simulate what the pipeline does with retrieval results\n"
            "retrieved_items = []\n"
            "for doc_id, content in list(DOCS.items())[:2]:\n"
            "    item = ContextItem(\n"
            "        content=content,\n"
            "        source_type=SourceType.RETRIEVAL,\n"
            "        metadata={\"doc_id\": doc_id},\n"
            "        priority=5,\n"
            "    )\n"
            "    retrieved_items.append(item)\n"
            "\n"
            "print(f\"Context items created: {len(retrieved_items)}\")\n"
            "for item in retrieved_items:\n"
            "    print(f\"  [{item.source_type.value}] {item.metadata['doc_id']}: \"\n"
            "          f\"{str(item.content)[:50]}...\")"
        ),
        md(
            "## Wire Agent with RAG Skill\n"
            "Create a skill that bundles the search tool, then attach\n"
            "it to an Agent."
        ),
        code(
            "rag_skill = Skill(\n"
            "    name=\"knowledge_base\",\n"
            "    description=\"Search the internal knowledge base\",\n"
            "    instructions=\"Use search_knowledge_base to find relevant documents \"\n"
            "                \"before answering questions about Anchor.\",\n"
            "    tools=(search_knowledge_base,),\n"
            "    activation=\"always\",\n"
            ")\n"
            "\n"
            "print(f\"RAG skill: {rag_skill.name}\")\n"
            "print(f\"  activation: {rag_skill.activation}\")\n"
            "print(f\"  tools: {[t.name for t in rag_skill.tools]}\")"
        ),
        md(
            "## Full Agent Configuration\n"
            "Here is the complete pattern for an agent with retrieval.\n"
            "Note: `agent.chat()` requires an API key, so we show the\n"
            "configuration without executing a live call."
        ),
        code(
            "# Full configuration pattern (shown without live API call)\n"
            "agent = Agent(\n"
            "    model=\"claude-haiku-4-5-20251001\",\n"
            "    max_response_tokens=2048,\n"
            "    max_rounds=10,\n"
            ")\n"
            "\n"
            "memory = SlidingWindowMemory(max_tokens=4096)\n"
            "\n"
            "agent.with_system_prompt(\n"
            "    \"You are a helpful assistant with access to a knowledge base. \"\n"
            "    \"Always search for relevant documents before answering.\"\n"
            ")\n"
            "agent.with_memory(memory)\n"
            "agent.with_skill(rag_skill)\n"
            "\n"
            "print(\"Agent configured with:\")\n"
            "print(f\"  System prompt: set\")\n"
            "print(f\"  Memory: {type(memory).__name__} ({memory.max_tokens} tokens)\")\n"
            "print(f\"  Skills: knowledge_base\")\n"
            "print()\n"
            "print(\"To run a chat loop:\")\n"
            "print('  for chunk in agent.chat(\"What is Anchor?\"):')\n"
            "print('      print(chunk, end=\"\", flush=True)')"
        ),
        md(
            "## Agent Tool Loop (Conceptual)\n"
            "When `agent.chat()` runs, it follows this loop:\n"
            "\n"
            "1. Build context (pipeline + memory + retrieval items)\n"
            "2. Send to Anthropic API with tool definitions\n"
            "3. If model calls a tool -> execute it, feed result back\n"
            "4. Repeat until model produces final text or max_rounds reached\n"
            "5. Stream text chunks to the caller"
        ),
        code(
            "# Simulate the tool loop conceptually\n"
            "print(\"=== Simulated Agent Tool Loop ===\")\n"
            "print()\n"
            "print(\"Round 1:\")\n"
            "print(\"  User: 'What memory strategies does Anchor support?'\")\n"
            "print(\"  Agent calls: search_knowledge_base(query='memory strategies')\")\n"
            "\n"
            "search_result = search_knowledge_base.fn(query=\"memory strategies\")\n"
            "print(f\"  Tool result: {search_result[:80]}...\")\n"
            "print()\n"
            "print(\"Round 2:\")\n"
            "print(\"  Agent receives tool result and generates final answer.\")\n"
            "print(\"  -> 'Anchor supports sliding window, summary buffer, and graph memory.'\")\n"
            "print()\n"
            "print(\"Total rounds: 2 (1 tool call + 1 final response)\")"
        ),
        md(
            "## Key Takeaways\n"
            "- Wrap retrieval logic in `@tool` decorated functions\n"
            "- Bundle search tools into a `Skill` with `activation=\"always\"`\n"
            "- Attach skills to the Agent via `.with_skill()`\n"
            "- Retrieved documents become `ContextItem` objects in the pipeline\n"
            "- The agent's tool loop automatically handles search -> synthesize flows\n"
            "- Use `max_rounds` to cap the number of tool-use iterations\n"
            "\n"
            "**Next:** [Streaming Agent](05_streaming_agent.ipynb)"
        ),
    ])


# ---------------------------------------------------------------------------
# 05 - Streaming Agent
# ---------------------------------------------------------------------------
def nb_streaming_agent():
    write_notebook("05_streaming_agent.ipynb", [
        md(
            "# Streaming Agent\n"
            "> Understand the streaming data models and patterns for real-time responses.\n"
            "\n"
            "Anchor provides `StreamDelta`, `StreamResult`, and `StreamUsage` models\n"
            "for tracking streaming LLM responses. This notebook explores these models\n"
            "and shows common streaming patterns.\n"
            "\n"
            "**Time:** ~5 minutes"
        ),
        md("## Setup"),
        code(
            "from anchor.models.streaming import StreamDelta, StreamResult, StreamUsage"
        ),
        md(
            "## StreamDelta\n"
            "Each text chunk from a streaming response is represented as a\n"
            "`StreamDelta` with the text fragment and its position index."
        ),
        code(
            "# Simulate a stream of deltas\n"
            "deltas = [\n"
            "    StreamDelta(text=\"Hello\", index=0),\n"
            "    StreamDelta(text=\" there\", index=1),\n"
            "    StreamDelta(text=\"! How\", index=2),\n"
            "    StreamDelta(text=\" can I\", index=3),\n"
            "    StreamDelta(text=\" help?\", index=4),\n"
            "]\n"
            "\n"
            "print(\"Simulated stream deltas:\")\n"
            "for d in deltas:\n"
            "    print(f\"  Delta(index={d.index}, text={d.text!r})\")"
        ),
        md(
            "## Accumulate Deltas into Full Text\n"
            "In practice, you concatenate deltas as they arrive to build\n"
            "the complete response."
        ),
        code(
            "accumulated = \"\"\n"
            "print(\"Streaming output:\")\n"
            "for d in deltas:\n"
            "    accumulated += d.text\n"
            "    # In a real app: print(d.text, end=\"\", flush=True)\n"
            "\n"
            "print(f\"  '{accumulated}'\")\n"
            "print(f\"\\nTotal deltas: {len(deltas)}\")\n"
            "print(f\"Final length: {len(accumulated)} chars\")"
        ),
        md(
            "## StreamUsage\n"
            "`StreamUsage` tracks token consumption from a completed response,\n"
            "including cache-related metrics."
        ),
        code(
            "usage = StreamUsage(\n"
            "    input_tokens=150,\n"
            "    output_tokens=42,\n"
            "    cache_creation_input_tokens=100,\n"
            "    cache_read_input_tokens=50,\n"
            ")\n"
            "\n"
            "print(\"Token usage:\")\n"
            "print(f\"  input_tokens:                {usage.input_tokens}\")\n"
            "print(f\"  output_tokens:               {usage.output_tokens}\")\n"
            "print(f\"  cache_creation_input_tokens:  {usage.cache_creation_input_tokens}\")\n"
            "print(f\"  cache_read_input_tokens:      {usage.cache_read_input_tokens}\")\n"
            "print(f\"  total tokens:                {usage.input_tokens + usage.output_tokens}\")"
        ),
        md(
            "## StreamResult\n"
            "`StreamResult` is the final accumulated result from a completed\n"
            "stream, combining text, usage, model info, and stop reason."
        ),
        code(
            "result = StreamResult(\n"
            "    text=accumulated,\n"
            "    usage=usage,\n"
            "    model=\"claude-haiku-4-5-20251001\",\n"
            "    stop_reason=\"end_turn\",\n"
            ")\n"
            "\n"
            "print(\"StreamResult:\")\n"
            "print(f\"  text:        {result.text!r}\")\n"
            "print(f\"  model:       {result.model}\")\n"
            "print(f\"  stop_reason: {result.stop_reason}\")\n"
            "print(f\"  usage:       {result.usage.input_tokens} in / \"\n"
            "      f\"{result.usage.output_tokens} out\")"
        ),
        md(
            "## Streaming Pattern: agent.chat()\n"
            "`Agent.chat()` returns an iterator of text chunks. Each chunk\n"
            "corresponds to a `StreamDelta.text` from the underlying API."
        ),
        code(
            "# Conceptual streaming pattern\n"
            "# (requires API key -- shown for illustration)\n"
            "#\n"
            "# for chunk in agent.chat(\"Explain streaming.\"):\n"
            "#     print(chunk, end=\"\", flush=True)\n"
            "#     # Each chunk is a string (the text portion of a StreamDelta)\n"
            "#\n"
            "# After chat completes:\n"
            "# result = agent.last_result  # ContextResult from the pipeline\n"
            "\n"
            "print(\"agent.chat() streaming pattern:\")\n"
            "print(\"  1. Call agent.chat(message)\")\n"
            "print(\"  2. Iterate over yielded text chunks\")\n"
            "print(\"  3. Each chunk is a str (partial response)\")\n"
            "print(\"  4. Concatenate for full response\")\n"
            "print(\"  5. Access agent.last_result for pipeline metadata\")"
        ),
        md(
            "## Simulated Full Streaming Flow\n"
            "Putting it all together: simulate a streaming response with\n"
            "deltas, accumulation, and a final result."
        ),
        code(
            "# Simulate a complete streaming flow\n"
            "print(\"=== Simulated Streaming Flow ===\")\n"
            "print()\n"
            "\n"
            "# Phase 1: Stream deltas\n"
            "chunks = [\"Anchor\", \" is a\", \" context\", \" engineering\",\n"
            "          \" toolkit\", \" for\", \" building\", \" AI\", \" apps.\"]\n"
            "\n"
            "print(\"Phase 1 - Streaming deltas:\")\n"
            "full_text = \"\"\n"
            "for i, chunk in enumerate(chunks):\n"
            "    delta = StreamDelta(text=chunk, index=i)\n"
            "    full_text += delta.text\n"
            "    print(f\"  [{i}] '{chunk}'\")\n"
            "\n"
            "print(f\"\\nPhase 2 - Accumulated text:\")\n"
            "print(f\"  '{full_text}'\")\n"
            "\n"
            "# Phase 3: Final result\n"
            "final = StreamResult(\n"
            "    text=full_text,\n"
            "    usage=StreamUsage(input_tokens=85, output_tokens=len(chunks)),\n"
            "    model=\"claude-haiku-4-5-20251001\",\n"
            "    stop_reason=\"end_turn\",\n"
            ")\n"
            "\n"
            "print(f\"\\nPhase 3 - Final StreamResult:\")\n"
            "print(f\"  model:       {final.model}\")\n"
            "print(f\"  stop_reason: {final.stop_reason}\")\n"
            "print(f\"  tokens:      {final.usage.input_tokens} in / \"\n"
            "      f\"{final.usage.output_tokens} out\")"
        ),
        md(
            "## Async Streaming\n"
            "`Agent.achat()` provides the same streaming interface for\n"
            "async applications."
        ),
        code(
            "# Async streaming pattern (requires async context)\n"
            "#\n"
            "# async for chunk in agent.achat(\"Explain async streaming.\"):\n"
            "#     print(chunk, end=\"\", flush=True)\n"
            "\n"
            "print(\"Async pattern: agent.achat()\")\n"
            "print(\"  - Same interface as agent.chat()\")\n"
            "print(\"  - Uses 'async for' instead of 'for'\")\n"
            "print(\"  - Pipeline builds with pipeline.abuild()\")\n"
            "print(\"  - Retries use asyncio.sleep()\")"
        ),
        md(
            "## Key Takeaways\n"
            "- `StreamDelta`: individual text chunk with position index\n"
            "- `StreamUsage`: token counts including cache metrics\n"
            "- `StreamResult`: final accumulated response (text + usage + metadata)\n"
            "- `agent.chat()` yields `str` chunks (sync iterator)\n"
            "- `agent.achat()` yields `str` chunks (async iterator)\n"
            "- Access `agent.last_result` after chat for the `ContextResult`\n"
            "\n"
            "**Back to:** [Agents README](README.md)"
        ),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating Agents notebooks in {OUTPUT_DIR}/\n")

    nb_basic_agent()
    nb_tool_decorator()
    nb_skills_system()
    nb_agent_with_retrieval()
    nb_streaming_agent()

    print(f"\nDone. 5 notebooks created.")


if __name__ == "__main__":
    main()
