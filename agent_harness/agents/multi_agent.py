"""Multi-Agent System: Manager dispatches to specialized sub-agents in sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from ..llm import LLMBackend


@dataclass
class SubAgentResult:
    agent_name: str
    output: str


@dataclass
class MultiAgentReport:
    results: List[SubAgentResult] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(f"[{r.agent_name}] {r.output}" for r in self.results)


class MultiAgentCoordinator:
    """
    Manager Agent -> Research Agent -> Coding Agent -> Testing Agent -> Review Agent.
    Each sub-agent is just an LLM call role-playing that agent; swap any of
    them for a real specialized pipeline (e.g. a real code-execution sandbox
    for the Coding/Testing agents).
    """

    DEFAULT_PIPELINE = ["research", "coding", "testing", "review"]

    def __init__(self, llm: LLMBackend, pipeline: List[str] = None):
        self._llm = llm
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._role_prompts: Dict[str, Callable[[str], str]] = {
            "research": lambda task: f"Act as the Research Agent. Task: {task}",
            "coding": lambda task: f"Act as the Coding Agent. Task: {task}",
            "testing": lambda task: f"Act as the Testing Agent. Task: {task}",
            "review": lambda task: f"Act as the Review Agent. Task: {task}",
        }

    def run(self, task: str) -> MultiAgentReport:
        report = MultiAgentReport()
        for agent_name in self._pipeline:
            prompt_fn = self._role_prompts.get(agent_name)
            if prompt_fn is None:
                continue
            output = self._llm.complete(prompt_fn(task)).strip()
            report.results.append(SubAgentResult(agent_name=agent_name, output=output))
        return report
