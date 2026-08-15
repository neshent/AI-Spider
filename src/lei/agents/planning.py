"""Planning Agent: break a complex goal into an ordered task list before execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..llm import LLMBackend


@dataclass
class PlanStep:
    index: int
    description: str
    done: bool = False


class PlanningAgent:
    def __init__(self, llm: LLMBackend):
        self._llm = llm

    def plan(self, goal: str) -> List[PlanStep]:
        prompt = f"Produce a numbered plan to accomplish this goal.\nGoal: {goal}"
        raw = self._llm.complete(prompt)
        steps: List[PlanStep] = []
        for line in raw.splitlines():
            line = line.strip()
            match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
            if match:
                steps.append(
                    PlanStep(index=len(steps) + 1, description=match.group(1))
                )
        if not steps and raw.strip():
            steps.append(PlanStep(index=1, description=raw.strip()))
        return steps
