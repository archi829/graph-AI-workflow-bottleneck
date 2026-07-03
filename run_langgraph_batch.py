#!/usr/bin/env python3
"""
Standalone Batch Runner for the Custom LangGraph Agent.
Does not touch any CrewAI code or shared telemetry.
"""
from __future__ import annotations
import argparse
import json
import time
import uuid
from pathlib import Path
import os
os.environ["OTEL_SERVICE_NAME"] = "open-deep-research"
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
from agents.open_deep_research_agent import OpenDeepResearchAgent
from agents.base import FailureInjectionConfig
FAILURE_FLAGS = {
    "loop": "loop",
    "timeout": "timeout",
    "retrieval_fail": "retrieval_fail",
    "hallucination": "hallucination",
    "context_overflow": "context_overflow",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a batch of LangGraph tasks.")
    p.add_argument("--n", type=int, default=10, help="Number of tasks to run")
    p.add_argument("--faulty", action="store_true", help="Enable failure injection")
    p.add_argument("--error-type", choices=list(FAILURE_FLAGS.keys()), default="loop")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds between runs")
    p.add_argument(
        "--prob",
        type=float,
        default=None,
        help=(
            "Override probability/rate for the current --error-type. "
            "Applies to 'retrieval_fail' (fail_retrieval_prob) and "
            "'hallucination' (hallucination_rate) only; ignored otherwise. "
            "Must be between 0.0 and 1.0."
        ),
    )
    p.add_argument(
        "--system",
        choices=["open_deep_research"],
        default="open_deep_research",
        help=(
            "Kept for command-line compatibility with the CrewAI/FinRobot "
            "runners, which take the same --system flag. This script only "
            "ever runs open_deep_research, so any other value is rejected "
            "by argparse itself."
        ),
    )
    p.add_argument(
        "--skip-langfuse-check",
        action="store_true",
        help="Skip the Langfuse auth check at startup (still runs, just without the pre-flight warning).",
    )
    args = p.parse_args()

    if args.prob is not None and not (0.0 <= args.prob <= 1.0):
        p.error("--prob must be between 0.0 and 1.0")

    return args


def build_failure_config(args: argparse.Namespace) -> FailureInjectionConfig:
    if not args.faulty:
        return FailureInjectionConfig()

    default_retrieval_prob = 0.3
    default_hallucination_rate = 0.2

    return FailureInjectionConfig(
        force_loop=(args.error_type == "loop"),
        force_timeout=(args.error_type == "timeout"),
        fail_retrieval_prob=(
            (args.prob if args.prob is not None else default_retrieval_prob)
            if args.error_type == "retrieval_fail"
            else 0.0
        ),
        hallucination_rate=(
            (args.prob if args.prob is not None else default_hallucination_rate)
            if args.error_type == "hallucination"
            else 0.0
        ),
        context_overflow=(args.error_type == "context_overflow"),
    )


def check_langfuse_connection() -> bool:
    """Fail fast (but not fatally) if Langfuse isn't reachable, so you find
    out before burning a 50-run faulty batch, not after."""
    try:
        from langfuse import get_client
        client = get_client()
        ok = client.auth_check()
        if ok:
            print("Langfuse: connected \u2705")
        else:
            print(
                "Langfuse: auth check FAILED \u274c "
                "(check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST in .env). "
                "Runs will still execute, but traces will not appear in the dashboard."
            )
        return ok
    except ImportError:
        print("Langfuse: package not installed (`pip install langfuse`). Traces will not be recorded.")
        return False
    except Exception as exc:
        print(f"Langfuse: could not connect ({exc}). Traces will not be recorded.")
        return False


def main() -> None:
    args = parse_args()

    if not args.skip_langfuse_check:
        check_langfuse_connection()

    failure_cfg = build_failure_config(args)
    system = OpenDeepResearchAgent(failure_config=failure_cfg)

    # Output to our own separate folder
    out_dir = Path("data/raw/agent_system=open_deep_research")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch_{int(time.time())}.jsonl"

    pool = system.TASK_POOL
    successes, failures = 0, 0
    records = []

    print(f"Starting LangGraph Batch: n={args.n} | faulty={args.faulty} | type={args.error_type}")
    if args.faulty and args.prob is not None:
        print(f"  probability override (--prob): {args.prob}")
    print(f"Outputting to: {out_path}\n")

    with out_path.open("w") as f:
        for i in range(args.n):
            task = pool[i % len(pool)]
            result = system.run(task)

            successes += int(result.success)
            failures += int(not result.success)
            record = {
                "run_id": str(uuid.uuid4()),
                "agent_system": result.agent_system,
                "task": result.task,
                "success": result.success,
                "error": result.error,
                "synthetic_error_type": result.synthetic_error_type,
                "duration_s": round(result.duration_s, 3),
                "tokens_used": result.tokens_used,
                "retries": result.retries,
                "faulty_batch": args.faulty,
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(
                f"[{i + 1}/{args.n}] success={result.success} "
                f"({result.duration_s:.1f}s, retries={result.retries})"
                + (f" | ERROR: {result.synthetic_error_type}" if not result.success else "")
            )
            time.sleep(args.sleep)

    print(f"\nDone. {successes} succeeded, {failures} failed.")
    print(f"Run log: {out_path}")
    print("Check the Langfuse dashboard for trace spans.")


if __name__ == "__main__":
    main()