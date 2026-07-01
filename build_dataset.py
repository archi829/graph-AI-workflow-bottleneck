#!/usr/bin/env python3
"""
build_dataset.py  --  Person C's deliverable (Section 4).

Reads all schema-conformant JSON files from data/raw/ (written by
export_traces.py), computes percentile-correct run_labels across the
full collection, and writes data/index.jsonl -- the final artifact
Sprint 1 needs before GNN modeling can begin.

Usage:
    python build_dataset.py
    python build_dataset.py --raw-dir data/raw --out data/index.jsonl
    python build_dataset.py --raw-dir data/raw --stats   # print stats only, no write

The run_labels logic:
  success   -- from run_record.success (set by run_batch.py)
  slow      -- total_latency_ms > 75th percentile across the batch
  expensive -- total_tokens > 75th percentile across the batch

These percentile thresholds match the project doc (Section 3 note: "latency
above percentile threshold"). They're computed over the merged dataset so
TRAIL traces, CrewAI traces, and open_deep_research traces are all compared
against a single shared distribution, not per-system buckets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema validation -- catch malformed files from export_traces.py early
# ---------------------------------------------------------------------------
REQUIRED_TRACE_FIELDS = {"trace_id", "agent_system", "spans", "run_labels", "meta"}
REQUIRED_SPAN_FIELDS = {
    "span_id", "parent_id", "role", "name", "latency_ms",
    "tokens_in", "tokens_out", "cost_usd", "model", "tool",
    "error_flag", "synthetic_error_type",
}
VALID_ROLES = {"agent", "tool", "llm"}
VALID_AGENT_SYSTEMS = {"crewai", "finrobot", "open_deep_research", "trail"}
VALID_SYNTHETIC_ERROR_TYPES = {
    "loop", "timeout", "retrieval_fail", "hallucination", "context_overflow", None,
}


def validate_trace(trace: dict, path: Path) -> list[str]:
    """Returns a list of validation errors (empty = valid)."""
    errors = []
    missing = REQUIRED_TRACE_FIELDS - set(trace.keys())
    if missing:
        errors.append(f"missing top-level fields: {missing}")
        return errors  # can't validate further without these

    if trace["agent_system"] not in VALID_AGENT_SYSTEMS:
        errors.append(f"invalid agent_system: {trace['agent_system']!r}")

    for i, span in enumerate(trace.get("spans", [])):
        missing_s = REQUIRED_SPAN_FIELDS - set(span.keys())
        if missing_s:
            errors.append(f"span[{i}] missing fields: {missing_s}")
        if span.get("role") not in VALID_ROLES:
            errors.append(f"span[{i}] invalid role: {span.get('role')!r}")
        if span.get("synthetic_error_type") not in VALID_SYNTHETIC_ERROR_TYPES:
            errors.append(
                f"span[{i}] invalid synthetic_error_type: {span.get('synthetic_error_type')!r}"
            )

    return errors


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency)
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (k - lo) * (sorted_v[hi] - sorted_v[lo])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge exported traces into data/index.jsonl.")
    p.add_argument("--raw-dir", default="data/raw",
                   help="Root directory of per-trace JSON files (default: data/raw)")
    p.add_argument("--out", default="data/index.jsonl",
                   help="Output path for the merged manifest (default: data/index.jsonl)")
    p.add_argument("--slow-pct", type=float, default=75.0,
                   help="Percentile threshold for slow label (default: 75)")
    p.add_argument("--expensive-pct", type=float, default=75.0,
                   help="Percentile threshold for expensive label (default: 75)")
    p.add_argument("--stats", action="store_true",
                   help="Print dataset stats and schema validation only; don't write index.jsonl")
    p.add_argument("--strict", action="store_true",
                   help="Abort on any schema validation error instead of skipping the file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    if not raw_dir.exists():
        print(f"ERROR: raw directory not found: {raw_dir}")
        print("Run export_traces.py first.")
        sys.exit(1)

    # --- Load all trace JSON files ---
    all_files = sorted(raw_dir.rglob("*.json"))
    if not all_files:
        print(f"No .json files found under {raw_dir}. Run export_traces.py first.")
        sys.exit(1)

    print(f"Found {len(all_files)} trace files under {raw_dir}")

    traces: list[dict] = []
    validation_errors: list[tuple[Path, list[str]]] = []

    for fp in all_files:
        try:
            trace = json.loads(fp.read_text())
        except json.JSONDecodeError as e:
            print(f"  SKIP (JSON decode error): {fp}: {e}")
            continue

        errs = validate_trace(trace, fp)
        if errs:
            validation_errors.append((fp, errs))
            if args.strict:
                print(f"  VALIDATION ERROR in {fp}:")
                for e in errs:
                    print(f"    - {e}")
                sys.exit(1)
            else:
                print(f"  WARN: schema issues in {fp} (skipping): {errs}")
                continue

        traces.append(trace)

    print(f"Loaded {len(traces)} valid traces  ({len(validation_errors)} skipped due to errors)")

    if not traces:
        print("No valid traces to process. Check export_traces.py output.")
        sys.exit(1)

    # --- Compute percentile thresholds across full merged dataset ---
    latencies = [t["meta"].get("total_latency_ms", 0.0) for t in traces]
    tokens = [t["meta"].get("total_tokens", 0) for t in traces]

    slow_threshold = _percentile(latencies, args.slow_pct)
    expensive_threshold = _percentile(tokens, args.expensive_pct)

    print(f"\nRun-label thresholds ({args.slow_pct}th percentile):")
    print(f"  slow      : total_latency_ms > {slow_threshold:.0f} ms")
    print(f"  expensive : total_tokens     > {expensive_threshold:.0f} tokens")

    # --- Apply labels and build index records ---
    index_records: list[dict] = []

    system_counts: dict[str, int] = {}
    error_type_counts: dict[str | None, int] = {}
    role_counts: dict[str, int] = {}

    for trace in traces:
        latency_ms = trace["meta"].get("total_latency_ms", 0.0)
        total_tokens = trace["meta"].get("total_tokens", 0)

        # Overwrite the stub labels written by export_traces.py with the
        # correctly calibrated percentile-based values.
        trace["run_labels"]["slow"] = latency_ms > slow_threshold
        trace["run_labels"]["expensive"] = total_tokens > expensive_threshold

        # Write updated trace JSON back to disk so downstream code always
        # reads the corrected labels from the file, not just the index.
        # (We only do this if not in --stats mode to avoid side effects.)

        # Stats accumulation
        system_counts[trace["agent_system"]] = system_counts.get(trace["agent_system"], 0) + 1
        for span in trace.get("spans", []):
            et = span.get("synthetic_error_type")
            error_type_counts[et] = error_type_counts.get(et, 0) + 1
            r = span.get("role", "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1

        # The index record: lightweight manifest entry, not the full trace.
        # GNN training code reads the full JSON from path; the index is just
        # the lookup table for filtering/splitting by label/system.
        index_record = {
            "trace_id": trace["trace_id"],
            "agent_system": trace["agent_system"],
            "task": trace.get("task", ""),
            "run_id": trace.get("run_id", ""),
            "path": str(
                Path(args.raw_dir)
                / f"agent_system={trace['agent_system']}"
                / f"{trace['trace_id']}.json"
            ),
            "n_spans": len(trace.get("spans", [])),
            "total_latency_ms": latency_ms,
            "total_tokens": total_tokens,
            "llm_model": trace["meta"].get("llm_model", ""),
            "faulty_batch": trace["meta"].get("faulty_batch", False),
            "run_labels": trace["run_labels"],
            "has_error_span": any(s.get("error_flag") for s in trace.get("spans", [])),
            "synthetic_error_types": list({
                s.get("synthetic_error_type")
                for s in trace.get("spans", [])
                if s.get("synthetic_error_type")
            }),
        }
        index_records.append(index_record)

    # --- Stats printout (always) ---
    print(f"\n=== Dataset stats ({len(traces)} traces) ===")
    print(f"By agent_system:  {dict(sorted(system_counts.items()))}")
    print(f"By span role:     {dict(sorted(role_counts.items()))}")

    clean = sum(1 for r in index_records if not r["faulty_batch"])
    faulty = sum(1 for r in index_records if r["faulty_batch"])
    successes = sum(1 for r in index_records if r["run_labels"]["success"])
    slows = sum(1 for r in index_records if r["run_labels"]["slow"])
    expensive = sum(1 for r in index_records if r["run_labels"]["expensive"])

    print(f"Clean batches:    {clean}   Faulty batches: {faulty}")
    print(f"run_labels.success   = {successes}/{len(traces)} "
          f"({100*successes//len(traces)}%)")
    print(f"run_labels.slow      = {slows}/{len(traces)} "
          f"({100*slows//len(traces)}%)")
    print(f"run_labels.expensive = {expensive}/{len(traces)} "
          f"({100*expensive//len(traces)}%)")

    et_clean = {str(k): v for k, v in error_type_counts.items()}
    print(f"synthetic_error_type span distribution: {et_clean}")

    total_spans = sum(r["n_spans"] for r in index_records)
    print(f"Total spans across dataset: {total_spans}")

    # --- 300-trace target check ---
    target = 300
    if len(traces) < target:
        print(f"\n[!] {len(traces)}/{target} traces -- need {target - len(traces)} more "
              f"before dataset_v1 is ready for GNN training.")
    else:
        print(f"\n[✓] {len(traces)}/{target} target met -- dataset_v1 ready.")

    if args.stats:
        print("\n--stats mode: no files written.")
        return

    # --- Write updated trace JSONs with corrected labels ---
    for trace in traces:
        out_path = (
            Path(args.raw_dir)
            / f"agent_system={trace['agent_system']}"
            / f"{trace['trace_id']}.json"
        )
        if out_path.exists():
            out_path.write_text(json.dumps(trace, indent=2, default=str))

    # --- Write index.jsonl ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for record in index_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(index_records)} records to {out_path}")
    print("Next: load data/index.jsonl in your GNN training code.")


if __name__ == "__main__":
    main()
