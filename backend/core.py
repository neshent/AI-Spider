"""
backend/core.py — shared backend used by both web_app.py and cli.py.

Neither the web layer nor the CLI layer touches src/lei directly.
All agent logic, model selection, and knowledge management lives here.

                    ┌─────────────┐
                    │  backend/   │
         ┌──────────│   core.py   │──────────┐
         │          └─────────────┘          │
         v                                   v
    web_app.py                           cli.py
    (Flask routes)                   (argparse CLI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

from src.lei import Orchestrator
from src.lei.llm import LLMBackend, MockLLMBackend


# ---------------------------------------------------------------------------
# Request / Response contracts
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """Everything a caller needs to pass in for a single interaction."""
    message:  str
    model_id: str = ""
    knowledge: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Everything the backend returns to any caller."""
    text:    str
    success: bool = True
    error:   Optional[str] = None

    # Optional structured fields — callers may ignore these
    intent:             Optional[str] = None
    handled_reactively: bool = False
    plan:               List[dict] = field(default_factory=list)
    tool_used:          Optional[str] = None
    tool_output:        Optional[str] = None


# ---------------------------------------------------------------------------
# AgentBackend
# ---------------------------------------------------------------------------

class AgentBackend:
    """
    Owns the Orchestrator and exposes two clean methods:
        handle(request)         -> AgentResponse   (blocking)
        stream(request)         -> Generator[str]  (token chunks)

    web_app.py and cli.py never import from src/lei directly.
    """

    def __init__(self) -> None:
        # No model registered yet — swap MockLLMBackend for a real one
        # by calling set_model() once a backend is registered.
        self._llm: LLMBackend = MockLLMBackend()
        self._extra_knowledge: List[Tuple[str, str]] = []
        self._orch: Optional[Orchestrator] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_model(self, llm: LLMBackend) -> None:
        """Replace the active LLM backend. Clears the cached orchestrator."""
        self._llm  = llm
        self._orch = None

    def add_knowledge(self, doc_id: str, text: str) -> None:
        """Add a document to RAG knowledge. Persists across requests."""
        self._extra_knowledge.append((doc_id, text))
        self._orch = None   # rebuild next call so RAG picks it up

    def list_knowledge(self) -> List[Tuple[str, str]]:
        return list(self._extra_knowledge)

    # ------------------------------------------------------------------
    # Internal orchestrator builder
    # ------------------------------------------------------------------

    def _get_orchestrator(self) -> Orchestrator:
        if self._orch is None:
            self._orch = Orchestrator(llm=self._llm)
            for doc_id, text in self._extra_knowledge:
                self._orch.add_knowledge(doc_id, text)
        return self._orch

    # ------------------------------------------------------------------
    # Public interface used by web_app.py and cli.py
    # ------------------------------------------------------------------

    def handle(self, request: AgentRequest) -> AgentResponse:
        """Blocking request -> response. Used by CLI and JSON API route."""
        if not request.message.strip():
            return AgentResponse(text="", success=False, error="Empty message.")

        # Seed any per-request knowledge
        for doc_id, text in request.knowledge:
            self.add_knowledge(doc_id, text)

        try:
            orch  = self._get_orchestrator()
            trace = orch.handle(request.message)
        except Exception as exc:
            return AgentResponse(text="", success=False, error=str(exc))

        return AgentResponse(
            text               = trace.final_response or "",
            success            = True,
            intent             = trace.intent,
            handled_reactively = trace.handled_reactively,
            plan               = [
                {"index": s.index, "description": s.description}
                for s in (trace.plan or [])
            ],
            tool_used   = trace.tool_used,
            tool_output = trace.tool_output,
        )

    def stream(self, request: AgentRequest) -> Generator[str, None, None]:
        """
        Streaming version. Yields plain text chunks.
        Currently runs handle() in one shot and yields the full response —
        replace with a real streaming LLM call once a streaming backend is
        registered.
        """
        response = self.handle(request)
        if not response.success:
            yield f"[error] {response.error}"
            return
        yield response.text


# ---------------------------------------------------------------------------
# Singleton — both web_app.py and cli.py import this one instance
# ---------------------------------------------------------------------------

backend = AgentBackend()
