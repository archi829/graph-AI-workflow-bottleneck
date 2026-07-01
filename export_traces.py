#!/usr/bin/env python3
"""
export_traces.py  --  Person B's deliverable (Section 4).

Reads a run-log .jsonl written by run_batch.py, fetches each trace from
Langfuse by trace_id (preferred) or by time-range + name lookup (fallback),
maps spans to the Section 3 schema, and writes one JSON file per trace to:
    data/raw/agent_system=<system>/<trace_id>.json

Usage:
    python export_traces.py --input data/raw/agent_system=crewai/batch_*.jsonl
    python export_traces.py --input data/raw/agent_system=crewai/batch_123.jsonl

Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langfuse.api.client import LangfuseAPI


def _make_api_client() -> LangfuseAPI:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key  = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host        = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not public_key or not secret_key:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        sys.exit(1)
    return LangfuseAPI(
        base_url=host,
        x_langfuse_public_key=public_key,
        username=public_key,
        password=secret_key,
    )


# ---------------------------------------------------------------------------
# Span role + field mapping  (verified against Observation/ObservationsView)
# ---------------------------------------------------------------------------
_TOOL_NAME_PREFIXES = (
    "tool:", "tool_call", "local_knowledge", "web_search",
    "calculator", "retrieval",
)

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
    v = getattr(obs, "latency", None)
    if v is not None:
        return round(v * 1000, 2)
    start = getattr(obs, "start_time", None)
    end   = getattr(obs, "end_time",   None)
    if start and end:
        return round((end - start).total_seconds() * 1000, 2)
    return 0.0

def _tokens(obs) -> tuple[int, int]:
    ud = getattr(obs, "usage_details", None) or {}
    if ud:
        return int(ud.get("input", 0) or 0), int(ud.get("output", 0) or 0)
    usage = getattr(obs, "usage", None)
    if usage:
        return int(getattr(usage, "input", 0) or 0), int(getattr(usage, "output", 0) or 0)
    return 0, 0

def _cost_usd(obs) -> float:
    cd = getattr(obs, "cost_details", None) or {}
    if cd:
        return float(cd.get("total", 0.0) or 0.0)
    ctc = getattr(obs, "calculated_total_cost", None)
    return float(ctc) if ctc is not None else 0.0

def _error_flag(obs) -> bool:
    level = str(getattr(obs, "level", "") or "").upper()
    return "ERROR" in level


def _map_trace(langfuse_trace, run_record: dict) -> dict:
    synthetic_error_type = run_record.get("synthetic_error_type")
    spans = []
    for obs in (langfuse_trace.observations or []):
        role = _map_role(obs.type, obs.name)
        tin, tout = _tokens(obs)
        spans.append({
            "span_id":              obs.id,
            "parent_id":            obs.parent_observation_id,
            "role":                 role,
            "name":                 obs.name or "",
            "latency_ms":           _latency_ms(obs),
            "tokens_in":            tin,
            "tokens_out":           tout,
            "cost_usd":             _cost_usd(obs),
            "model":                obs.model or "",
            "tool":                 _extract_tool_name(role, obs.name),
            "error_flag":           _error_flag(obs),
            "synthetic_error_type": synthetic_error_type,
        })

    total_tokens   = sum(s["tokens_in"] + s["tokens_out"] for s in spans)
    total_lat_ms   = (langfuse_trace.latency or 0) * 1000

    return {
        "trace_id":     langfuse_trace.id,
        "agent_system": run_record.get("agent_system", "crewai"),
        "task":         run_record.get("task", ""),
        "run_id":       run_record.get("run_id", ""),
        "spans":        spans,
        "run_labels": {
            "success":   run_record.get("success", False),
            "slow":      False,   # overwritten by build_dataset.py
            "expensive": False,   # overwritten by build_dataset.py
        },
        "meta": {
            "total_tokens":    total_tokens,
            "total_latency_ms": total_lat_ms,
            "faulty_batch":    run_record.get("faulty_batch", False),
            "retries":         run_record.get("retries", 0),
            "llm_model":       next(
                (s["model"] for s in spans if s["role"] == "llm" and s["model"]), ""
            ),
        },
    }


# ---------------------------------------------------------------------------
# Trace lookup: by trace_id (fast) or by time-range (fallback)
# ---------------------------------------------------------------------------
def _fetch_by_id(api: LangfuseAPI, trace_id: str):
    return api.trace.get(trace_id)


def _fetch_by_time(api: LangfuseAPI, run_start_iso: str, window_s: float = 600):
    """
    Fallback when trace_id is unavailable. Queries traces in a time window
    around the run's start time and returns ALL of them so the caller can
    match by position (first-in, first-out within the window).
    window_s: how many seconds after run_start to search (default 10 min).
    """
    from_ts = datetime.datetime.fromisoformat(run_start_iso.replace("Z", "+00:00"))
    to_ts   = from_ts + datetime.timedelta(seconds=window_s)
    result  = api.trace.list(
        from_timestamp=from_ts,
        to_timestamp=to_ts,
        order_by="timestamp.asc",
        limit=50,
    )
    return result.data if result and result.data else []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Langfuse traces to schema-conformant JSON.")
    p.add_argument("--input", required=True,
                   help="Path to a batch_*.jsonl file written by run_batch.py")
    p.add_argument("--out", default="data/raw",
                   help="Root output directory (default: data/raw)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between Langfuse API calls (default 0.5)")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip records with no trace_id instead of trying time-range fallback")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    api    = _make_api_client()
    inpath = Path(args.input)

    if not inpath.exists():
        print(f"ERROR: input file not found: {inpath}")
        sys.exit(1)

    records = [json.loads(l) for l in inpath.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(records)} run records from {inpath}")
    print()

    # For the time-range fallback we need the batch file's mtime as an
    # approximate batch-start anchor (records don't store their own timestamp).
    batch_mtime = datetime.datetime.fromtimestamp(
        inpath.stat().st_mtime, tz=datetime.timezone.utc
    )
    # Assume batch ran for at most 2h before the file was written.
    batch_start = (batch_mtime - datetime.timedelta(hours=2)).isoformat()

    exported, skipped, failed = 0, 0, 0
    agent_system = records[0].get("agent_system", "crewai") if records else "crewai"

    for i, record in enumerate(records):
        trace_id = record.get("trace_id")
        time.sleep(args.sleep if i > 0 else 0)

        try:
            if trace_id:
                trace = _fetch_by_id(api, trace_id)
                print(f"  [{i+1}/{len(records)}] fetched by trace_id: {trace_id}")
            elif args.skip_missing:
                print(f"  [{i+1}/{len(records)}] SKIP (no trace_id, --skip-missing set)")
                skipped += 1
                continue
            else:
                # Time-range fallback: get all traces in the batch window
                # and pick the i-th one (relies on sequential ordering).
                candidates = _fetch_by_time(api, batch_start)
                if not candidates:
                    print(f"  [{i+1}/{len(records)}] FAIL: no traces found in time window. "
                          f"Check LANGFUSE_HOST and that the batch ran with keys set.")
                    failed += 1
                    continue
                # Pick candidate by position; fetch full detail
                candidate = candidates[min(i, len(candidates) - 1)]
                trace = _fetch_by_id(api, candidate.id)
                print(f"  [{i+1}/{len(records)}] fetched by time-range fallback: {trace.id}")

            mapped = _map_trace(trace, record)
            out_dir = Path(args.out) / f"agent_system={agent_system}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{trace.id}.json"
            out_file.write_text(json.dumps(mapped, indent=2, default=str))
            print(f"             spans={len(mapped['spans'])}  "
                  f"tokens={mapped['meta']['total_tokens']}  -> {out_file}")
            exported += 1

        except Exception as exc:  # noqa: BLE001
            print(f"  [{i+1}/{len(records)}] FAIL: {exc}")
            failed += 1

    print(f"\nDone. exported={exported}  skipped={skipped}  failed={failed}")
    if exported > 0:
        print(f"Next: python build_dataset.py --raw-dir {args.out}")
    else:
        print("\nNo traces exported. Most likely cause: the batch ran BEFORE the")
        print("instrumentation fix -- re-run run_batch.py --n 1 and try again.")


if __name__ == "__main__":
    main()