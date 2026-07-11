"""
Reasoning Agent Harness
=======================

A runnable reference implementation of the architecture described in
ai-agent-architecture.md. It wires together:

- Input Processing
- Reasoning Engine (LLM + Decision, pluggable backend)
- Reactive Agent
- Planning Agent
- Tool-Using Agent (Web Search / Python / SQL / API stubs)
- RAG Agent (embedding + vector store, pluggable)
- Multi-Agent Coordinator (Research / Coding / Testing / Review)
- Autonomous Workflow Manager (retry, checkpoints, logging)
- Memory & Knowledge Base (short-term + long-term)

No API keys are required out of the box: the default "LLM backend" is a
deterministic mock so the whole pipeline can be exercised offline. Swap in
a real backend by implementing `LLMBackend` in `agent_harness/llm.py`.
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
__version__ = "0.1.0"
