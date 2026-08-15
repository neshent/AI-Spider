"""
Memory store implementations.

ShortTermMemory  — current conversation / recent tasks (in-process only).
LongTermMemory   — persisted preferences, projects, and learned facts (JSON).
MemoryManager    — facade combining both.

Swap these for a real database (SQLite/Postgres) or vector store for
production-grade persistence across restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ShortTermMemory:
    """Holds the current conversation / recent task state. Cleared per session."""

    turns: List[str] = field(default_factory=list)
    recent_tasks: List[str] = field(default_factory=list)

    def add_turn(self, text: str) -> None:
        self.turns.append(text)

    def add_task(self, task: str) -> None:
        self.recent_tasks.append(task)

    def as_text(self) -> str:
        return "\n".join(self.turns)


class LongTermMemory:
    """
    Persists preferences, projects, and learned facts to a JSON file so
    data survives across sessions.
    """

    def __init__(self, path: str = "long_term_memory.json"):
        self._path = path
        self._data: Dict[str, Any] = {
            "preferences": {},
            "projects": [],
            "learned": [],
        }
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def set_preference(self, key: str, value: Any) -> None:
        self._data["preferences"][key] = value
        self._save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._data["preferences"].get(key, default)

    def record_project(self, description: str) -> None:
        self._data["projects"].append(description)
        self._save()

    def learn(self, fact: str) -> None:
        self._data["learned"].append(fact)
        self._save()

    def dump(self) -> Dict[str, Any]:
        return self._data


class MemoryManager:
    """Facade combining short-term and long-term memory."""

    def __init__(self, long_term_path: str = "long_term_memory.json"):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(path=long_term_path)
