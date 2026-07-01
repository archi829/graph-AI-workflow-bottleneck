#!/usr/bin/env python3
"""
export_traces.py  --  Person B's deliverable (Section 4).

Reads a run-log .jsonl written by run_batch.py, fetches each trace from the
Langfuse API by its trace_id, maps spans to the Section 3 schema, and writes
one JSON file per trace to data/raw/agent_system=<system>/<trace_id>.json.

Usage:
    python export_traces.py --input data/raw/agent_system=crewai/batch_*.jsonl
    python export_traces.py --input data/raw/agent_system=crewai/batch_123.jsonl --out data/raw

Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Langfuse API client setup
# Verified against langfuse 4.12.0: LangfuseAPI is in langfuse.api.client,
# auth uses username=public_key + password=secret_key (HTTP Basic).
# ---------------------------------------------------------------------------
from langfuse.api.client import LangfuseAPI


def _make_api_client() -> LangfuseAPI:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        sys.exit(1)
    return LangfuseAPI(
        base_url=host,
        x_langfuse_public_key=public_key,
        username=public_key,   # Langfuse uses public key as Basic-auth username
        password=secret_key,   # and secret key as password
    )


# ---------------------------------------------------------------------------
# Span role mapping
# Observation.type is a free string in the Langfuse SDK ("GENERATION", "SPAN",
# etc.). CrewAIInstrumentor emits: GENERATION for LLM calls, SPAN for agent
# steps and tool calls. We refine SPAN->tool vs agent by name prefix.
# ---------------------------------------------------------------------------
_TOOL_NAME_PREFIXES = ("tool:", "tool_call", "local_knowledge", "web_search",
                       "calculator", "retrieval")


def _map_role(obs_type: str, name: str | None) -> str:
    t = (obs_type or "").upper()
    n = (name or "").lower()
    if t == "GENERATION":
        return "llm"
    if any(n.startswith(p) for p in _TOOL_NAME_PREFIXES):
        return "tool"
    return "agent"


def _extract_tool_name(role: str, name: str | None) -> str | None:
    if role != "tool":
        return None
    if name and ":" in name:
        return name.split(":", 1)[1].strip()
    return name


def _latency_ms(obs) -> float:
    """ObservationsView has a latency field in seconds (float). Fall back to
    computing from start_time/end_time if latency is None."""
    latency_s = getattr(obs, "latency", None)
    if latency_s is not None:
        return round(latency_s * 1000, 2)
    start = getattr(obs, "start_time", None)
    end = getattr(obs, "end_time", None)
    if start and end:
        delta = (end - start).total_seconds()
        return round(delta * 1000, 2)
    return 0.0


def _tokens(obs) -> tuple[int, int]:
    """Returns (tokens_in, tokens_out). Prefers usage_details (current SDK)
    over the deprecated usage field."""
    ud = getattr(obs, "usage_details", None) or {}
    if ud:
        tin = ud.get("input", 0) or 0
        tout = ud.get("output", 0) or 0
        return int(tin), int(tout)
    usage = getattr(obs, "usage", None)
    if usage:
        return int(getattr(usage, "input", 0) or 0), int(getattr(usage, "output", 0) or 0)
    return 0, 0


def _cost_usd(obs) -> float:
    cd = getattr(obs, "cost_details", None) or {}
    if cd:
        return float(cd.get("total", 0.0) or 0.0)
    # ObservationsView also has calculated_total_cost
    ctc = getattr(obs, "calculated_total_cost", None)
    if ctc is not None:
        return float(ctc)
    return 0.0


def _error_flag(obs) -> bool:
    level = getattr(obs, "level", None)
    if level is None:
        return False
    level_str = str(level).upper()
    return "ERROR" in level_str


# ---------------------------------------------------------------------------
# Core mapping: Langfuse trace -> Section 3 schema dict
# ---------------------------------------------------------------------------
def _map_trace(
    langfuse_trace,
    run_record: dict,
) -> dict:
    """Maps a TraceWithFullDetails object + our run-log record to the shared
    schema (Section 3). run_record supplies synthetic_error_type, run_id,
    and local success/duration_s since those come from our code, not Langfuse.
    """
    synthetic_error_type = run_record.get("synthetic_error_type")

    spans = []
    for obs in (langfuse_trace.observations or []):
        role = _map_role(obs.type, obs.name)
        tin, tout = _tokens(obs)
        span = {
            "span_id": obs.id,
            "parent_id": obs.parent_observation_id,
            "role": role,
            "name": obs.name or "",
            "latency_ms": _latency_ms(obs),
            "tokens_in": tin,
            "tokens_out": tout,
            "cost_usd": _cost_usd(obs),
            "model": obs.model or "",
            "tool": _extract_tool_name(role, obs.name),
            "error_flag": _error_flag(obs),
            # synthetic_error_type is run-level (all spans in a faulty run share
            # the same motif), not per-span -- matches Section 3 spec.
            "synthetic_error_type": synthetic_error_type,
        }
        spans.append(span)

    # run_labels are computed here using local values (success, duration_s from
    # the run log) + total tokens from spans, since Langfuse doesn't give us a
    # reliable "total_tokens" at the trace level for local Ollama models.
    total_tokens = sum(s["tokens_in"] + s["tokens_out"] for s in spans)
    total_latency_ms = (langfuse_trace.latency or 0) * 1000

    return {
        "trace_id": langfuse_trace.id,
        "agent_system": run_record.get("agent_system", "crewai"),
        "task": run_record.get("task", ""),
        "run_id": run_record.get("run_id", ""),
        "spans": spans,
        "run_labels": {
            # success/slow/expensive are STUBS here -- build_dataset.py
            # recomputes them percentile-correctly across the full dataset.
            # These local values let you validate a single file is structurally
            # correct before build_dataset.py runs.
            "success": run_record.get("success", False),
            "slow": False,      # overwritten by build_dataset.py
            "expensive": False, # overwritten by build_dataset.py
        },
        "meta": {
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency_ms,
            "faulty_batch": run_record.get("faulty_batch", False),
            "retries": run_record.get("retries", 0),
            "llm_model": next(
                (s["model"] for s in spans if s["role"] == "llm" and s["model"]),
                "",
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Langfuse traces to schema-conformant JSON.")
    p.add_argument(
        "--input", required=True,
        help="Path to a batch_*.jsonl file written by run_batch.py",
    )
    p.add_argument(
        "--out", default="data/raw",
        help="Root output directory (default: data/raw). Files land in <out>/agent_system=<system>/",
    )
    p.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to wait between Langfuse API calls (rate-limit hygiene). Default 0.3.",
    )
    p.add_argument(
        "--skip-missing", action="store_true",
        help="Skip records with no trace_id instead of aborting.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = _make_api_client()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    records = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} run records from {input_path}")

    exported, skipped, failed = 0, 0, 0

    for i, record in enumerate(records):
        trace_id = record.get("trace_id")
        agent_system = record.get("agent_system", "crewai")

        if not trace_id:
            if args.skip_missing:
                print(f"  [{i+1}/{len(records)}] SKIP (no trace_id) -- run in non-Langfuse mode?")
                skipped += 1
                continue
            else:
                print(f"  [{i+1}/{len(records)}] ERROR: record has no trace_id. "
                      f"Re-run the batch with Langfuse keys set, or use --skip-missing.")
                failed += 1
                continue

        try:
            # Wait a beat before each API call to avoid hitting rate limits.
            if i > 0:
                time.sleep(args.sleep)

            trace = api.trace.get(trace_id)
            mapped = _map_trace(trace, record)

            out_dir = Path(args.out) / f"agent_system={agent_system}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{trace_id}.json"
            out_file.write_text(json.dumps(mapped, indent=2, default=str))

            print(f"  [{i+1}/{len(records)}] OK  {trace_id}  "
                  f"spans={len(mapped['spans'])}  "
                  f"tokens={mapped['meta']['total_tokens']}  "
                  f"-> {out_file}")
            exported += 1

        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{len(records)}] FAIL {trace_id}: {exc}")
            failed += 1

    print(f"\nDone. exported={exported}  skipped={skipped}  failed={failed}")
    print(f"Files in: {Path(args.out) / f'agent_system={agent_system}'}")
    if exported > 0:
        print(f"Next: python build_dataset.py --raw-dir {args.out}")


if __name__ == "__main__":
    main()
