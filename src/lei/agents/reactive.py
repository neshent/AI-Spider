"""Reactive Agent: rule matching -> immediate response. Fast, cheap, minimal reasoning."""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional


class ReactiveAgent:
    """
    Simple rule-matching agent. Rules are (pattern -> handler) pairs checked
    in order; first match wins. Falls back to default_handler if set.
    """

    def __init__(self):
        self._compiled: list = []
        self.default_handler: Optional[Callable[[str], str]] = None

    def add_rule(self, pattern: str, handler: Callable[[str], str]) -> None:
        self._compiled.append((re.compile(pattern, re.IGNORECASE), handler))

    def handle(self, request: str) -> Optional[str]:
        for regex, handler in self._compiled:
            if regex.search(request):
                return handler(request)
        if self.default_handler:
            return self.default_handler(request)
        return None


def build_default_reactive_agent() -> ReactiveAgent:
    agent = ReactiveAgent()
    agent.add_rule(r"\bhello\b|\bhi\b", lambda r: "Hello! How can I help you today?")
    agent.add_rule(r"\bthanks?\b|\bthank you\b", lambda r: "You're welcome!")
    agent.add_rule(
        r"\bwho are you\b",
        lambda r: "I'm a reasoning agent built on the Lei architecture.",
    )
    return agent
