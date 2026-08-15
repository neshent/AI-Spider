"""
Pluggable LLM backends.

Defines LLMBackend (ABC), MockLLMBackend (offline), and
AnthropicLLMBackend (real). Select via Orchestrator(llm=...) or
get_default_backend() which auto-detects the best available option.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Optional


class LLMBackend(ABC):
    """Interface every reasoning backend must implement."""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    """
    Deterministic rule-based stand-in. Covers intent extraction, routing,
    planning, tool decisions, synthesis, and sub-agent role-play. Runs
    fully offline with zero dependencies.
    """

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        p = prompt.lower()

        if "extract the intent" in p:
            if any(k in p for k in ("website", "app", "application", "build")):
                return "Create Software Project"
            if any(k in p for k in ("weather", "today", "forecast")):
                return "Answer Factual Question"
            if any(k in p for k in ("research", "find out", "compare")):
                return "Research Task"
            return "General Query"

        if "should this be handled reactively" in p:
            simple = ("weather", "hello", "hi", "what is", "define")
            if any(m in p for m in simple) and "build" not in p:
                return "REACTIVE"
            return "PLAN"

        if "produce a numbered plan" in p:
            goal_match = re.search(r"goal:\s*(.+)", prompt, re.IGNORECASE)
            goal = goal_match.group(1).strip() if goal_match else "the task"
            return (
                "1. Research requirements and best-fit stack for: {g}\n"
                "2. Design architecture and data model\n"
                "3. Implement backend\n"
                "4. Implement frontend\n"
                "5. Test end to end\n"
                "6. Review and deliver"
            ).format(g=goal)

        if "does answering this require a tool" in p:
            if any(k in p for k in ("weather", "search", "latest", "current", "price")):
                return "TOOL:web_search"
            if any(k in p for k in ("calculate", "sum", "average", "compute", "plot")):
                return "TOOL:python"
            if any(k in p for k in ("select", "query", "database", "sql")):
                return "TOOL:sql"
            return "NONE"

        if "synthesize a final answer" in p:
            return (
                "Here is the result assembled from the reasoning pipeline "
                "(plan, tool output, and retrieved context above)."
            )

        if "act as the research agent" in p:
            return "Recommended stack: FastAPI + PostgreSQL + static frontend, deployed with Docker."
        if "act as the coding agent" in p:
            return "Generated a minimal FastAPI backend and a static HTML/JS frontend scaffold."
        if "act as the testing agent" in p:
            return "Ran smoke tests: all endpoints return 200, basic UI renders. 0 failures."
        if "act as the review agent" in p:
            return "Reviewed output: meets requirements. Approved for delivery."

        return "OK"


class AnthropicLLMBackend(LLMBackend):
    """
    Real backend using the Anthropic SDK.
    Requires: pip install anthropic  and  ANTHROPIC_API_KEY in environment.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required. Install: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )


def get_default_backend() -> LLMBackend:
    """Auto-selects AnthropicLLMBackend when key is set, else MockLLMBackend."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMBackend()
        except RuntimeError:
            pass
    return MockLLMBackend()
