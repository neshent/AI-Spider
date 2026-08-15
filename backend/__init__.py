"""
Backend layer — the single source of truth for all Lei logic.

Both web_app.py and cli.py talk exclusively to this layer.
Neither talks directly to src/lei or app/.
"""

from .core import AgentBackend, AgentRequest, AgentResponse, backend

__all__ = ["AgentBackend", "AgentRequest", "AgentResponse", "backend"]
