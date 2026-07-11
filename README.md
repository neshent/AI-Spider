# Reasoning Agent Harness

A runnable reference implementation of the architecture described in
`ai-agent-architecture.md`: Reactive, Planning, Tool-Using, RAG,
Multi-Agent, and Autonomous Workflow patterns, wired together behind one
`Orchestrator`.

It works fully offline out of the box (no API keys, no network calls) via
a deterministic `MockLLMBackend`, so you can run and test the whole
pipeline immediately. Swap in `AnthropicLLMBackend` (or write your own
`LLMBackend`) for real reasoning quality.

## Structure

```
agent_harness/
├── __init__.py              # package entry point (Orchestrator)
├── orchestrator.py           # wires the full "Complete Agent Pipeline"
├── input_processing.py       # Input Processing stage
├── reasoning_engine.py       # Reasoning Engine (routing/synthesis)
├── llm.py                    # LLMBackend interface + Mock/Anthropic impls
├── agents/
│   ├── reactive.py           # Reactive Agent (rule matching)
│   ├── planning.py           # Planning Agent (goal -> task list)
│   ├── tool_using.py         # Tool-Using Agent (tool routing)
│   ├── rag.py                # RAG Agent (embed -> vector store -> retrieve)
│   ├── multi_agent.py        # Multi-Agent Coordinator (research/coding/testing/review)
│   └── workflow.py           # Autonomous Workflow Manager (retry/evaluate/approve)
├── tools/__init__.py         # Web Search / Python / SQL / API / GitHub / Email / Calendar / FS stubs
└── memory/store.py           # Short-term + long-term memory (JSON-backed)
cli.py                         # command-line entry point
tests/test_harness.py          # pytest suite covering every component
```

This mirrors the doc's "Complete Agent Pipeline":

```
User Request → Input Processing → Reasoning Engine → Planning Agent
→ Reactive Agent (if simple) → RAG Retrieval → Tool Execution
→ Multi-Agent Collaboration → Autonomous Workflow → Memory Update
→ Final Response
```

## Quick start

No dependencies required:

```bash
python cli.py "Build a weather application."
python cli.py "What's today's weather?"
python cli.py --interactive
```

Run the test suite (requires pytest):

```bash
pip install pytest
pytest -q
```

## Using a real LLM instead of the mock

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python cli.py "Build a blog website."
```

`Orchestrator()` auto-detects `ANTHROPIC_API_KEY` and switches from
`MockLLMBackend` to `AnthropicLLMBackend` (model: `claude-sonnet-4-6`)
automatically. You can also pass any backend explicitly:

```python
from agent_harness import Orchestrator
from agent_harness.llm import AnthropicLLMBackend

orchestrator = Orchestrator(llm=AnthropicLLMBackend(model="claude-sonnet-4-6"))
trace = orchestrator.handle("Research the best free vector databases.")
print(trace.final_response)
```

## Feeding the RAG agent real documents

```python
orchestrator.add_knowledge("company_handbook", "Our on-call rotation is weekly...")
trace = orchestrator.handle("What's our on-call rotation?")
```

The default `RAGAgent` uses a zero-dependency bag-of-words embedder and an
in-memory vector store (`agent_harness/agents/rag.py: SimpleEmbedder`,
`InMemoryVectorStore`) — swap these for a real embedding model and
FAISS/Chroma/Pinecone/Weaviate/Milvus for production use.

## Wiring real tools

`agent_harness/tools/__init__.py` ships safe stubs for `web_search`,
`python`, `sql`, `api`, `file_system`, `email`, `calendar`, and `github`.
Each stub just returns a string describing what it *would* do — replace
the `run()` method of any tool with a real integration (a search API
client, a sandboxed code executor, a DB driver, etc.).

## Inspecting the pipeline

`Orchestrator.handle()` returns a `PipelineTrace` with everything the
pipeline did at every stage (intent, plan, retrieved context, tool used,
sub-agent results, workflow retries, and the final response) — useful for
debugging or building your own UI on top.
