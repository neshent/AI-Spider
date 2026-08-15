"""
Pipeline runner — builds an Orchestrator from a model_id and runs a request.
Shared by both the sync /api/chat route and the streaming /api/stream route.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.lei import Orchestrator

# Knowledge docs added at runtime via /api/knowledge
_extra_knowledge: List[Tuple[str, str]] = []

_DEFAULT_KNOWLEDGE = (
    "stack_notes",
    "FastAPI is a modern, fast Python web framework for building APIs. "
    "SQLite and PostgreSQL are common relational databases for small "
    "and large deployments respectively.",
)


def build_orchestrator(llm) -> Orchestrator:
    orch = Orchestrator(llm=llm)
    orch.add_knowledge(*_DEFAULT_KNOWLEDGE)
    for doc_id, text in _extra_knowledge:
        orch.add_knowledge(doc_id, text)
    return orch


def run_pipeline(
    user_message: str,
    model_id: str,
    hf_token: str = "",
    lmstudio_url: str = "",
    lmstudio_model: str = "",
) -> dict:
    from .backends import make_backend

    llm = make_backend(
        model_id,
        hf_token=hf_token,
        lmstudio_url=lmstudio_url,
        lmstudio_model=lmstudio_model,
    )
    orch = build_orchestrator(llm)
    trace = orch.handle(user_message)

    return {
        "final_response": trace.final_response,
        "intent": trace.intent,
        "handled_reactively": trace.handled_reactively,
        "plan": [
            {"index": s.index, "description": s.description}
            for s in (trace.plan or [])
        ],
        "retrieved_context": trace.retrieved_context or [],
        "tool_used": trace.tool_used,
        "tool_output": trace.tool_output,
        "multi_agent_report": (
            trace.multi_agent_report.summary() if trace.multi_agent_report else None
        ),
        "workflow": (
            {
                "attempts": trace.workflow_result.attempts,
                "success": trace.workflow_result.success,
            }
            if trace.workflow_result
            else None
        ),
    }


def trace_to_dict(trace) -> dict:
    """Convert a PipelineTrace to a JSON-serialisable dict."""
    return {
        "intent": trace.intent,
        "handled_reactively": trace.handled_reactively,
        "plan": [
            {"index": s.index, "description": s.description}
            for s in (trace.plan or [])
        ],
        "retrieved_context": trace.retrieved_context or [],
        "tool_used": trace.tool_used,
        "tool_output": trace.tool_output,
        "multi_agent_report": (
            trace.multi_agent_report.summary() if trace.multi_agent_report else None
        ),
        "workflow": (
            {
                "attempts": trace.workflow_result.attempts,
                "success": trace.workflow_result.success,
            }
            if trace.workflow_result
            else None
        ),
    }
