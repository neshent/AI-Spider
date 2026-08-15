"""Tool-Using Agent: decide whether a tool is needed and dispatch to it."""

from __future__ import annotations

from typing import Any, Dict

from ..llm import LLMBackend
from ..tools import get_tool_registry


class ToolUsingAgent:
    def __init__(self, llm: LLMBackend):
        self._llm = llm
        self._tools = get_tool_registry()

    def maybe_run_tool(self, request: str) -> Dict[str, Any]:
        decision = self._llm.complete(
            f"Does answering this require a tool, and which one? Request: {request}"
        ).strip()

        if not decision.upper().startswith("TOOL:"):
            return {"tool_used": None, "output": None}

        tool_name = decision.split(":", 1)[1].strip()
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"tool_used": None, "output": None}

        return {"tool_used": tool_name, "output": tool.run(request)}
