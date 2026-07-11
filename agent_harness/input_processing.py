"""Input Processing: validate input, extract intent, maintain context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .llm import LLMBackend


@dataclass
class ConversationContext:
    history: List[str] = field(default_factory=list)

    def add(self, role: str, text: str) -> None:
        self.history.append(f"{role}: {text}")

    def as_text(self) -> str:
        return "\n".join(self.history)


class InputProcessor:
    def __init__(self, llm: LLMBackend):
        self._llm = llm

    def validate(self, user_request: str) -> str:
        request = (user_request or "").strip()
        if not request:
            raise ValueError("Empty user request.")
        return request

    def extract_intent(self, user_request: str) -> str:
        prompt = f"Extract the intent of this request in a few words.\nRequest: {user_request}"
        return self._llm.complete(prompt).strip()

    def process(self, user_request: str, context: ConversationContext) -> dict:
        request = self.validate(user_request)
        intent = self.extract_intent(request)
        context.add("user", request)
        return {"request": request, "intent": intent}
