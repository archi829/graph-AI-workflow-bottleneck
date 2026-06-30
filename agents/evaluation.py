"""
Evaluation hooks, split into the two places evaluation actually happens:

1. IN-LOOP (guardrails): run *during* crew execution, before a task is
   considered done. CrewAI re-prompts the agent automatically on failure,
   up to `guardrail_max_retries` (verified field on Task). This is real
   retry behavior, not a cosmetic wrapper -- a guardrail that returns
   (False, reason) causes CrewAI to feed `reason` back to the agent and
   re-run the task.

2. POST-HOC (run_labels): computed once a run is complete, mirroring the
   `run_labels.{success,slow,expensive}` fields in the shared trace schema
   (Section 3) that Person C owns downstream. These are *not* a replacement
   for C's labeling pipeline -- they exist so Person A can sanity-check
   batch output locally before traces even reach Langfuse/export, and so
   run_batch.py can print useful per-batch stats.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import FinalAnswer, ResearchFindings

# -- 1. In-loop guardrails ------------------------------------------------


def research_guardrail(output):
    """Validates the Researcher task's structured output.

    Returns (True, parsed_dict) on success or (False, reason_string) on
    failure -- CrewAI re-prompts the agent with `reason_string` when False.
    """
    parsed: ResearchFindings | None = getattr(output, "pydantic", None)
    if parsed is None:
        return False, (
            "Output did not match the ResearchFindings schema "
            "(key_facts, open_questions, confidence). Re-emit valid JSON."
        )
    if len(parsed.key_facts) == 0:
        return False, "key_facts is empty -- list at least one concrete fact."
    if not (0.0 <= parsed.confidence <= 1.0):
        return False, "confidence must be between 0.0 and 1.0."
    return True, parsed.model_dump()


def final_answer_guardrail(output):
    """Validates the Writer task's structured output."""
    parsed: FinalAnswer | None = getattr(output, "pydantic", None)
    if parsed is None:
        return False, (
            "Output did not match the FinalAnswer schema "
            "(summary, details, recommendation, sources_used). Re-emit valid JSON."
        )
    if len(parsed.summary.strip()) < 10:
        return False, "summary is too short to be useful -- expand it to 2-4 real sentences."
    if len(parsed.details.strip()) < 30:
        return False, "details is too short -- the body needs real substance."
    return True, parsed.model_dump()


# -- 2. Post-hoc run-level labels ------------------------------------------


@dataclass
class RunLabels:
    """Mirrors `run_labels` in the shared trace schema (Section 3)."""

    success: bool
    slow: bool
    expensive: bool


# Thresholds are deliberately simple/explicit so they're easy to tune once
# real latency/token distributions exist (Person C owns the final
# percentile-based versions; these are local stand-ins for dev sanity checks).
SLOW_THRESHOLD_S = 45.0
EXPENSIVE_TOKEN_THRESHOLD = 6000


def compute_run_labels(
    *,
    success: bool,
    duration_s: float,
    tokens_used: int | None = None,
) -> RunLabels:
    slow = duration_s > SLOW_THRESHOLD_S
    expensive = (tokens_used or 0) > EXPENSIVE_TOKEN_THRESHOLD
    return RunLabels(success=success, slow=slow, expensive=expensive)


def evaluate_batch(records: list[dict]) -> dict:
    """Aggregate stats over a list of run_batch.py-style result dicts.

    Used by run_batch.py at the end of a batch to print a quick scorecard,
    and reusable by anyone testing the CrewAI wrapper standalone.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    successes = sum(1 for r in records if r.get("success"))
    slow = sum(
        1
        for r in records
        if compute_run_labels(
            success=r.get("success", False), duration_s=r.get("duration_s", 0.0)
        ).slow
    )
    error_types: dict[str, int] = {}
    for r in records:
        et = r.get("synthetic_error_type")
        if et:
            error_types[et] = error_types.get(et, 0) + 1

    return {
        "n": n,
        "success_rate": round(successes / n, 3),
        "slow_rate": round(slow / n, 3),
        "avg_duration_s": round(sum(r.get("duration_s", 0.0) for r in records) / n, 2),
        "synthetic_error_type_counts": error_types,
    }
