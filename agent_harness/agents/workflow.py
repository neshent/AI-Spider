"""
Autonomous Workflow Manager: receive goal -> plan -> assign tasks -> execute
-> evaluate -> retry if needed -> deliver. Includes retry logic, logging,
and optional human-approval checkpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("agent_harness.workflow")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class WorkflowResult:
    success: bool
    output: Optional[str]
    attempts: int
    log: List[str] = field(default_factory=list)


class AutonomousWorkflowManager:
    def __init__(self, max_retries: int = 2, require_human_approval: bool = False):
        self._max_retries = max_retries
        self._require_human_approval = require_human_approval

    def run(
        self,
        goal: str,
        execute_fn: Callable[[], str],
        evaluate_fn: Callable[[str], bool] = lambda output: bool(output),
        approve_fn: Optional[Callable[[str], bool]] = None,
    ) -> WorkflowResult:
        log: List[str] = []
        attempts = 0

        for attempt in range(1, self._max_retries + 2):
            attempts = attempt
            logger.info("Attempt %s for goal: %s", attempt, goal)
            log.append(f"Attempt {attempt}: executing")

            try:
                output = execute_fn()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Execution error on attempt %s: %s", attempt, exc)
                log.append(f"Attempt {attempt}: error - {exc}")
                time.sleep(0)  # placeholder for real backoff
                continue

            if not evaluate_fn(output):
                logger.info("Attempt %s failed evaluation, retrying if possible", attempt)
                log.append(f"Attempt {attempt}: failed evaluation")
                continue

            if self._require_human_approval:
                approved = approve_fn(output) if approve_fn else True
                log.append(f"Attempt {attempt}: human approval = {approved}")
                if not approved:
                    continue

            log.append(f"Attempt {attempt}: success")
            return WorkflowResult(success=True, output=output, attempts=attempts, log=log)

        log.append("All attempts exhausted, delivering best-effort failure")
        return WorkflowResult(success=False, output=None, attempts=attempts, log=log)
