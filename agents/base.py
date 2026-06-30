"""
Abstract base class for agent systems.

Every concrete system (CrewAI, open_deep_research, FinRobot, ...) implements
this interface so run_batch.py and app.py can drive any of them identically:

    system = CrewAIAgent()
    result = system.run(task="compare iPhone vs Pixel")

`run()` is intentionally dumb about telemetry -- it just executes the task.
Tracing is handled by whatever Langfuse instrumentation is active in the
process (see telemetry/ -- owned by Person B). This class only standardizes:
  - how a task string comes in
  - how failure-injection flags are read from the environment
  - what comes back out (so run_batch.py can log it uniformly)
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureInjectionConfig:
    """Reads the Appendix B env flags once per process."""

    fail_retrieval_prob: float = field(
        default_factory=lambda: float(os.getenv("FAIL_RETRIEVAL_PROB", "0.0"))
    )
    force_loop: bool = field(
        default_factory=lambda: os.getenv("FORCE_LOOP", "false").lower() == "true"
    )
    force_timeout: bool = field(
        default_factory=lambda: os.getenv("FORCE_TIMEOUT", "false").lower() == "true"
    )
    hallucination_rate: float = field(
        default_factory=lambda: float(os.getenv("HALLUCINATION_RATE", "0.0"))
    )
    context_overflow: bool = field(
        default_factory=lambda: os.getenv("CONTEXT_OVERFLOW", "false").lower() == "true"
    )

    @property
    def synthetic_error_type(self) -> str | None:
        """Best-effort label for which motif is *configured* to fire.

        This is a hint for export_traces.py / build_dataset.py, not ground
        truth that an error actually occurred during this specific run --
        actual occurrence is probabilistic for some flags.
        """
        if self.force_loop:
            return "loop"
        if self.force_timeout:
            return "timeout"
        if self.context_overflow:
            return "context_overflow"
        if self.fail_retrieval_prob > 0:
            return "retrieval_fail"
        if self.hallucination_rate > 0:
            return "hallucination"
        return None

    def should_fail_retrieval(self) -> bool:
        return random.random() < self.fail_retrieval_prob

    def should_hallucinate(self) -> bool:
        return random.random() < self.hallucination_rate


@dataclass
class RunResult:
    """Uniform return shape from any AgentSystem.run() call."""

    task: str
    output: str | None
    success: bool
    error: str | None = None
    synthetic_error_type: str | None = None
    duration_s: float = 0.0
    agent_system: str = "unknown"
    structured_output: dict | None = None
    tokens_used: int | None = None
    retries: int = 0


class AgentSystem(ABC):
    """Subclass this once per OSS repo (CrewAI, open_deep_research, FinRobot)."""

    #: must match the `agent_system` enum in the shared trace schema
    name: str = "unknown"

    def __init__(self) -> None:
        self.failure_config = FailureInjectionConfig()

    @abstractmethod
    def _run_task(self, task: str) -> str:
        """Execute one task and return the final text output.

        Subclasses implement the actual OSS-repo-specific call here. Let
        exceptions propagate -- run() below catches them and records
        error_flag / synthetic_error_type on the RunResult.
        """
        raise NotImplementedError

    def _enrich_result(self, result: RunResult) -> RunResult:
        """Optional hook: subclasses can attach structured_output, tokens_used,
        or retries onto a successful RunResult. Default: no-op passthrough.
        """
        return result

    def run(self, task: str) -> RunResult:
        start = time.monotonic()
        try:
            output = self._run_task(task)
            result = RunResult(
                task=task,
                output=output,
                success=True,
                synthetic_error_type=self.failure_config.synthetic_error_type,
                duration_s=time.monotonic() - start,
                agent_system=self.name,
            )
            return self._enrich_result(result)
        except Exception as exc:  # noqa: BLE001 -- batch runner must not crash on one bad task
            return RunResult(
                task=task,
                output=None,
                success=False,
                error=str(exc),
                synthetic_error_type=self.failure_config.synthetic_error_type,
                duration_s=time.monotonic() - start,
                agent_system=self.name,
            )
