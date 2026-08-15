"""
Tests for the Lei Reasoning Agent pipeline.
All tests run offline using MockLLMBackend — no API keys or network needed.

Run with:
    pytest -q
"""

import os
import sys

# Ensure the project root and src/ are on the path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from lei import Orchestrator
from lei.agents.planning import PlanningAgent
from lei.agents.rag import InMemoryVectorStore, SimpleEmbedder
from lei.agents.reactive import build_default_reactive_agent
from lei.agents.workflow import AutonomousWorkflowManager
from lei.llm import MockLLMBackend
from lei.tools import get_tool_registry


def test_reactive_agent_matches_greeting():
    agent = build_default_reactive_agent()
    result = agent.handle("hello there")
    assert result is not None
    assert "Hello" in result


def test_reactive_agent_no_match_returns_none():
    agent = build_default_reactive_agent()
    result = agent.handle("please generate a quarterly report")
    assert result is None


def test_planning_agent_produces_ordered_steps():
    planner = PlanningAgent(MockLLMBackend())
    steps = planner.plan("Create an e-commerce website")
    assert len(steps) >= 3
    assert steps[0].index == 1
    assert all(step.description for step in steps)


def test_rag_retrieval_finds_relevant_document():
    embedder = SimpleEmbedder()
    store = InMemoryVectorStore(embedder)
    store.add("doc1", "FastAPI is a modern Python web framework.")
    store.add("doc2", "Bananas are a good source of potassium.")
    results = store.search("What python web framework should I use?", top_k=1)
    assert results
    assert results[0][0].doc_id == "doc1"


def test_tool_registry_has_expected_tools():
    tools = get_tool_registry()
    for name in ("web_search", "python", "sql", "api", "file_system", "email", "calendar", "github"):
        assert name in tools
        assert tools[name].run("test request")


def test_workflow_manager_retries_and_succeeds():
    manager = AutonomousWorkflowManager(max_retries=2)
    attempts = {"count": 0}

    def flaky_execute():
        attempts["count"] += 1
        return "" if attempts["count"] < 2 else "done"

    result = manager.run("test goal", execute_fn=flaky_execute)
    assert result.success
    assert result.output == "done"
    assert result.attempts == 2


def test_workflow_manager_exhausts_retries_and_fails():
    manager = AutonomousWorkflowManager(max_retries=1)
    result = manager.run("test goal", execute_fn=lambda: "")
    assert not result.success
    assert result.output is None


def test_orchestrator_end_to_end_reactive():
    orch = Orchestrator(long_term_memory_path="test_long_term_memory.json")
    trace = orch.handle("hello")
    assert trace.handled_reactively
    assert trace.final_response
    _cleanup("test_long_term_memory.json")


def test_orchestrator_end_to_end_planned_build():
    orch = Orchestrator(long_term_memory_path="test_long_term_memory2.json")
    trace = orch.handle("Build a weather application.")
    assert trace.intent == "Create Software Project"
    assert not trace.handled_reactively
    assert trace.plan
    assert trace.multi_agent_report is not None
    assert trace.workflow_result is not None
    assert trace.final_response
    _cleanup("test_long_term_memory2.json")


def _cleanup(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
