"""Reasoning Engine: understand goals, decide next actions, select agent type."""

from __future__ import annotations

from .llm import LLMBackend


class ReasoningEngine:
    def __init__(self, llm: LLMBackend):
        self._llm = llm

    def should_react(self, request: str) -> bool:
        prompt = f"Should this be handled reactively (simple, no planning needed)? Request: {request}"
        return self._llm.complete(prompt).strip().upper() == "REACTIVE"

    def needs_tool(self, request: str) -> str:
        """Returns 'NONE' or 'TOOL:<name>'."""
        prompt = f"Does answering this require a tool, and which one? Request: {request}"
        return self._llm.complete(prompt).strip()

    def synthesize(self, request: str, working_context: str) -> str:
        prompt = (
            "Synthesize a final answer for the user using everything gathered so far.\n"
            f"Original request: {request}\n"
            f"Working context:\n{working_context}"
        )
        return self._llm.complete(prompt).strip()
