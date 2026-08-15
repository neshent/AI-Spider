"""
Backend factory — returns an LLMBackend for a given model_id.
No models are registered yet. Add them here one by one as they are implemented.
"""

from __future__ import annotations

from src.lei.llm import LLMBackend


def make_backend(model_id: str, **kwargs) -> LLMBackend:
    raise NotImplementedError(
        f"No backend registered for model '{model_id}'. "
        "Add an implementation to app/backends/ and register it here."
    )
