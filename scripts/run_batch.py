#!/usr/bin/env python3
"""
Batch runner -- Person A's deliverable (Section 4).

    python -m scripts.run_batch --system crewai --n 30
    python -m scripts.run_batch --system crewai --n 30 --faulty --error-type loop

Each run is one task pulled (round-robin, with repeats once exhausted) from
the system's TASK_POOL. Results are appended as JSON lines to
data/raw/agent_system=<system>/batch_<timestamp>.jsonl so Person B's
export_traces.py has something concrete to read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import REGISTRY
from agents.evaluation import evaluate_batch
from telemetry.instrument import instrument_crewai

FAILURE_FLAGS = {
    "loop": "FORCE_LOOP",
    "timeout": "FORCE_TIMEOUT",
    "retrieval_fail": "FAIL_RETRIEVAL_PROB",
    "hallucination": "HALLUCINATION_RATE",
    "context_overflow": "CONTEXT_OVERFLOW",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a batch of agent tasks.")
    p.add_argument("--system", required=True, choices=sorted(REGISTRY.keys()))
    p.add_argument("--n", type=int, default=10, help="number of tasks to run")
    p.add_argument("--faulty", action="store_true", help="enable failure injection for this batch")
    p.add_argument(
        "--error-type",
        choices=list(FAILURE_FLAGS.keys()),
        default="loop",
        help="which synthetic_error_type to inject when --faulty is set",
    )
    p.add_argument(
        "--prob",
        type=float,
        default=0.3,
        help="probability value for prob-based flags (retrieval_fail, hallucination)",
    )
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between runs (rate-limit hygiene, Section 13)")
    p.add_argument("--out", default=None, help="override output path")
    return p.parse_args()


def apply_failure_flags(args: argparse.Namespace) -> None:
    """Sets the Appendix B env flags for this process before any run."""
    for env_var in FAILURE_FLAGS.values():
        os.environ.pop(env_var, None)

    if not args.faulty:
        return

    env_var = FAILURE_FLAGS[args.error_type]
    if args.error_type in ("retrieval_fail", "hallucination"):
        os.environ[env_var] = str(args.prob)
    else:
        os.environ[env_var] = "true"


def main() -> None:
    args = parse_args()
    apply_failure_flags(args)

    if args.system == "crewai":
        instrument_crewai()

    system_cls = REGISTRY[args.system]
    system = system_cls()

    out_dir = Path(args.out or f"data/raw/agent_system={args.system}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch_{int(time.time())}.jsonl"

    pool = getattr(system, "TASK_POOL", None) or [f"Generic smoke-test task #{i}" for i in range(5)]

    successes, failures = 0, 0
    records: list[dict] = []
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
                "structured_output": result.structured_output,
                "trace_id": result.trace_id,
                "faulty_batch": args.faulty,
            }
            records.append(record)
            f.write(json.dumps(record) + "\n")
            f.flush()

            print(
                f"[{i + 1}/{args.n}] {result.agent_system} "
                f"success={result.success} ({result.duration_s:.1f}s, retries={result.retries})"
                + (f" error_type={result.synthetic_error_type}" if not result.success else "")
            )

            if (i + 1) % 10 == 0:
                try:
                    from langfuse import get_client

                    get_client().flush()
                    print(f"flushed {i + 1} traces to Langfuse")
                except Exception:
                    pass

            time.sleep(args.sleep)

    print(f"\nDone. {successes} succeeded, {failures} failed. Run log: {out_path}")
    print("Scorecard:", evaluate_batch(records))

    if args.system == "crewai":
        try:
            from langfuse import get_client

            get_client().flush()
            print("Flushed pending traces to Langfuse.")
        except ImportError:
            pass

    print(f"Next: python -m scripts.export_traces --input {out_path}")


if __name__ == "__main__":
    main()
