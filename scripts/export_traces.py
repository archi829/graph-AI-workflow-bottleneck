#!/usr/bin/env python3
"""
export_traces.py

Reads batch_*.jsonl from scripts.run_batch, matches each record to its exact
Langfuse trace using time-window reconstruction, maps spans to Section 3
schema, writes one JSON per trace to:
    data/raw/agent_system=<system>/<trace_id>.json

Usage:
    python -m scripts.export_traces --input data/raw/agent_system=crewai/batch_xyz.jsonl
    python -m scripts.export_traces --input data/raw/agent_system=crewai/batch_xyz.jsonl --buffer 60
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langfuse.api.client import LangfuseAPI

load_dotenv()


def _make_api_client() -> LangfuseAPI:
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        print("ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")
        sys.exit(1)
    creds = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return LangfuseAPI(
        base_url=host,
        x_langfuse_public_key=pk,
        username=pk,
        password=sk,
        headers={"Authorization": f"Basic {creds}"},
    )


_TOOL_PREFIXES = ("tool:", "local_knowledge", "web_search", "calculator", "retrieval")


def _map_role(obs_type: str, name: str | None) -> str:
    t = (obs_type or "").upper()
    n = (name or "").lower()
    if t == "GENERATION":
        return "llm"
    if any(n.startswith(p) for p in _TOOL_PREFIXES):
        return "tool"
    return "agent"


def _tool_name(role: str, name: str | None) -> str | None:
    if role != "tool":
        return None
    return name.split(":", 1)[1].strip() if name and ":" in name else name


def _latency_ms(obs) -> float:
    v = getattr(obs, "latency", None)
    if v is not None:
        return round(v * 1000, 2)
    s, e = getattr(obs, "start_time", None), getattr(obs, "end_time", None)
    return round((e - s).total_seconds() * 1000, 2) if s and e else 0.0


def _tokens(obs) -> tuple[int, int]:
    ud = getattr(obs, "usage_details", None) or {}
    if ud:
        return int(ud.get("input", 0) or 0), int(ud.get("output", 0) or 0)
    u = getattr(obs, "usage", None)
    return (int(getattr(u, "input", 0) or 0), int(getattr(u, "output", 0) or 0)) if u else (0, 0)


def _cost(obs) -> float:
    cd = getattr(obs, "cost_details", None) or {}
    if cd:
        return float(cd.get("total", 0.0) or 0.0)
    return float(getattr(obs, "calculated_total_cost", None) or 0.0)


def _is_error(obs) -> bool:
    return "ERROR" in str(getattr(obs, "level", "") or "").upper()


def _map_trace(trace, record: dict) -> dict:
    err_type = record.get("synthetic_error_type")
    spans = []
    for obs in (trace.observations or []):
        role = _map_role(obs.type, obs.name)
        tin, tout = _tokens(obs)
        spans.append(
            {
                "span_id": obs.id,
                "parent_id": obs.parent_observation_id,
                "role": role,
                "name": obs.name or "",
                "latency_ms": _latency_ms(obs),
                "tokens_in": tin,
                "tokens_out": tout,
                "cost_usd": _cost(obs),
                "model": obs.model or "",
                "tool": _tool_name(role, obs.name),
                "error_flag": _is_error(obs),
                "synthetic_error_type": err_type,
            }
        )
    total_tok = sum(s["tokens_in"] + s["tokens_out"] for s in spans)
    return {
        "trace_id": trace.id,
        "agent_system": record.get("agent_system", "crewai"),
        "task": record.get("task", ""),
        "run_id": record.get("run_id", ""),
        "spans": spans,
        "run_labels": {
            "success": record.get("success", False),
            "slow": False,
            "expensive": False,
        },
        "meta": {
            "total_tokens": total_tok,
            "total_latency_ms": (trace.latency or 0) * 1000,
            "faulty_batch": record.get("faulty_batch", False),
            "retries": record.get("retries", 0),
            "synthetic_error_type": err_type,
            "llm_model": next((s["model"] for s in spans if s["role"] == "llm" and s["model"]), ""),
        },
    }


def _extract_task_text(trace) -> str:
    for obs in (trace.observations or []):
        inp = getattr(obs, "input", None)
        if not inp or not isinstance(inp, dict):
            continue
        agent = inp.get("agent", {})
        goal = agent.get("goal", "") if isinstance(agent, dict) else ""
        if goal:
            prefix = "Gather the key facts needed to address: "
            return goal[len(prefix):].strip() if goal.startswith(prefix) else goal.strip()
    return ""


def _compute_time_windows(
    records: list[dict],
    batch_end: datetime.datetime,
    buffer_s: float = 60,
) -> list[tuple[datetime.datetime, datetime.datetime]]:
    windows = []
    cumulative = 0.0
    for record in reversed(records):
        dur = record.get("duration_s", 90.0)
        task_end = batch_end - datetime.timedelta(seconds=cumulative)
        task_start = task_end - datetime.timedelta(seconds=dur)
        windows.insert(
            0,
            (
                task_start - datetime.timedelta(seconds=buffer_s),
                task_end + datetime.timedelta(seconds=buffer_s),
            ),
        )
        cumulative += dur
    return windows


def _fetch_in_window(
    api: LangfuseAPI,
    from_ts: datetime.datetime,
    to_ts: datetime.datetime,
) -> list:
    try:
        result = api.trace.list(
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            limit=20,
        )
        return result.data if result and result.data else []
    except Exception as e:
        print(f"    warn: window query failed: {e}")
        return []


def _best_match(
    api: LangfuseAPI,
    candidates: list,
    task: str,
    already_used: set,
) -> object | None:
    task_lower = task.lower()
    first_unused = None

    for c in candidates:
        if c.id in already_used:
            continue
        try:
            full = api.trace.get(c.id)
            extracted = _extract_task_text(full).lower()
            if first_unused is None:
                first_unused = full
            if task_lower[:30] in extracted:
                return full
            time.sleep(0.05)
        except Exception:
            continue

    return first_unused


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to batch_*.jsonl")
    p.add_argument("--out", default="data/raw")
    p.add_argument(
        "--buffer",
        type=float,
        default=60,
        help="Seconds of padding around each task's time window (default 60)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    api = _make_api_client()
    inpath = Path(args.input)

    if not inpath.exists():
        print(f"ERROR: {inpath} not found")
        sys.exit(1)

    records = [json.loads(l) for l in inpath.read_text().splitlines() if l.strip()]
    if not records:
        print("ERROR: empty input file")
        sys.exit(1)

    agent_system = records[0].get("agent_system", "crewai")
    print(f"Loaded {len(records)} records from {inpath}")

    batch_end = datetime.datetime.fromtimestamp(
        inpath.stat().st_mtime, tz=datetime.timezone.utc
    )
    print(f"Batch end time (file mtime): {batch_end.isoformat()}")

    windows = _compute_time_windows(records, batch_end, buffer_s=args.buffer)
    print(
        f"Time windows reconstructed. First: {windows[0][0].strftime('%H:%M:%S')} -> "
        f"{windows[0][1].strftime('%H:%M:%S')} UTC"
    )
    print(
        f"Last:  {windows[-1][0].strftime('%H:%M:%S')} -> "
        f"{windows[-1][1].strftime('%H:%M:%S')} UTC\n"
    )

    out_dir = Path(args.out) / f"agent_system={agent_system}"
    out_dir.mkdir(parents=True, exist_ok=True)

    exported, failed = 0, 0
    already_used: set[str] = set()

    for i, (record, (from_ts, to_ts)) in enumerate(zip(records, windows)):
        task = record.get("task", "")
        trace_id = record.get("trace_id")

        try:
            if trace_id:
                trace = api.trace.get(trace_id)
            else:
                candidates = _fetch_in_window(api, from_ts, to_ts)
                if not candidates:
                    print(
                        f"  [{i+1}/{len(records)}] FAIL (no traces in window "
                        f"{from_ts.strftime('%H:%M:%S')}->{to_ts.strftime('%H:%M:%S')}): "
                        f"{task[:40]!r}"
                    )
                    failed += 1
                    continue
                trace = _best_match(api, candidates, task, already_used)
                if trace is None:
                    print(f"  [{i+1}/{len(records)}] FAIL (all candidates already used): {task[:40]!r}")
                    failed += 1
                    continue

            already_used.add(trace.id)
            mapped = _map_trace(trace, record)
            out_file = out_dir / f"{trace.id}.json"
            out_file.write_text(json.dumps(mapped, indent=2, default=str))
            print(
                f"  [{i+1}/{len(records)}] OK  spans={len(mapped['spans'])}  "
                f"tokens={mapped['meta']['total_tokens']}  "
                f"{trace.id[:20]}..."
            )
            exported += 1
            time.sleep(0.1)

        except Exception as exc:
            print(f"  [{i+1}/{len(records)}] FAIL: {exc}")
            failed += 1

    print(f"\nDone. exported={exported}  failed={failed}")
    if exported > 0:
        print(f"Files: {out_dir}")
        print(f"Next: python -m scripts.build_dataset --raw-dir {args.out}")
    else:
        print("\nZero exported. Check that LANGFUSE_HOST is correct and traces exist.")


if __name__ == "__main__":
    main()
