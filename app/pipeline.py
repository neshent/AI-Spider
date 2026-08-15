"""Pipeline runner — builds an Orchestrator and runs a request."""

from __future__ import annotations

from typing import List, Tuple

from src.lei import Orchestrator
from src.lei.llm import LLMBackend

# Knowledge docs added at runtime via /api/knowledge
_extra_knowledge: List[Tuple[str, str]] = []


def build_orchestrator(llm: LLMBackend) -> Orchestrator:
    orch = Orchestrator(llm=llm)
    for doc_id, text in _extra_knowledge:
        orch.add_knowledge(doc_id, text)
    return orch
