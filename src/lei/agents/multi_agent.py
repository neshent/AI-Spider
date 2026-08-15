"""Multi-Agent System: Manager dispatches to specialized sub-agents in sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
    Manager -> Research -> Coding -> Testing -> Review pipeline.
    Each sub-agent is an LLM call role-playing that specialist. Swap any
    for a real implementation (e.g. code-execution sandbox for Coding/Testing).
    """

    DEFAULT_PIPELINE = ["research", "coding", "testing", "review"]

    def __init__(self, llm: LLMBackend, pipeline: Optional[List[str]] = None):
        self._llm = llm
        self._pipeline = pipeline or self.DEFAULT_PIPELINE
        self._role_prompts: Dict[str, Callable[[str], str]] = {
            "research": lambda t: f"Act as the Research Agent. Task: {t}",
            "coding":   lambda t: f"Act as the Coding Agent. Task: {t}",
            "testing":  lambda t: f"Act as the Testing Agent. Task: {t}",
            "review":   lambda t: f"Act as the Review Agent. Task: {t}",
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
