#!/usr/bin/env python3
"""
backfill_tokens.py  --  One-off backfill for already-exported traces.

Problem: the 139 traces under data/raw/ were exported before export_traces.py
estimated tokens, so every span has tokens_in=tokens_out=0 and
meta.total_tokens=0. The original Langfuse export also never captured
span-level input/output text (spans only carry name/role/model/tool), so we
cannot reconstruct true prompt/completion token counts from the JSON alone.

This script estimates a *proxy* token count per span from the text that IS
present in each span (its `name`, `tool`, `model`) plus the trace-level
`task` for agent/llm spans (which is the closest available proxy for the
prompt). It then:
  * fills tokens_in / tokens_out (estimated via tiktoken cl100k_base)
  * back-fills cost_usd from estimated tokens at a documented flat rate
  * recomputes meta.total_tokens
  * rewrites the JSON in place

This is strictly better than all-zeros: it gives the GNN's tokens_total node
feature and the `expensive` run-label a real (if approximate) signal. For
ACCURATE per-span token counts, re-run export_traces.py after the tiktoken
fix (which reads the real input/output text from Langfuse).

Usage:
    python backfill_tokens.py
    python backfill_tokens.py --raw-dir data/raw --rate 0.00001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken


# Flat USD cost per token used as a placeholder for the back-filled cost_usd.
# Real cost requires the model + provider; this is a transparent approximation
# (~$0.01 / 1k tokens) so cost_usd is non-zero and ordered sensibly. Tune via
# --rate if you have a specific model's pricing in mind.
DEFAULT_COST_PER_TOKEN = 1e-5

_ENC = None

def _encoder():
    global _ENC
    if _ENC is None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False
    return _ENC

def estimate_tokens(text: str) -> int:
    enc = _encoder()
    if not enc:
        return max(0, len(str(text)) // 4)
    return len(enc.encode(str(text)))


def span_proxy_text(span: dict, task: str) -> tuple[str, str]:
    """Build (input_proxy, output_proxy) text for a span from available fields.

    - input proxy:  trace task + span name/tool/model (closest to the prompt)
    - output proxy: span name + tool (a completion proxy; spans don't store
                    generated text, so this is intentionally small)
    """
    name = span.get("name", "") or ""
    tool = span.get("tool") or ""
    model = span.get("model", "") or ""
    role = span.get("role", "") or ""

    in_parts = [task, name, tool, model]
    in_proxy = " ".join(p for p in in_parts if p)

    # Output proxy: only meaningful text we have is the span identity.
    out_proxy = " ".join(p for p in [name, tool] if p)
    # Agent/llm spans "produce" more than tool spans; pad slightly so they
    # aren't all zero, but keep it conservative.
    if role in ("agent", "llm") and not out_proxy:
        out_proxy = name or task
    return in_proxy, out_proxy


def backfill_trace(trace: dict, cost_per_token: float) -> tuple[int, int]:
    """Mutates trace spans in place. Returns (spans_changed, total_tokens)."""
    task = trace.get("task", "") or ""
    changed = 0
    for span in trace.get("spans", []):
        tin = int(span.get("tokens_in") or 0)
        tout = int(span.get("tokens_out") or 0)
        if tin == 0 or tout == 0:
            in_proxy, out_proxy = span_proxy_text(span, task)
            if tin == 0:
                tin = estimate_tokens(in_proxy)
            if tout == 0:
                tout = estimate_tokens(out_proxy)
            span["tokens_in"] = tin
            span["tokens_out"] = tout
            # Back-fill cost from estimated tokens (documented approximation).
            span["cost_usd"] = round((tin + tout) * cost_per_token, 6)
            changed += 1
        else:
            # Keep any existing real cost; if it's zero but tokens are known,
            # still estimate cost so cost_usd is non-zero.
            if float(span.get("cost_usd") or 0.0) == 0.0:
                span["cost_usd"] = round((tin + tout) * cost_per_token, 6)
    total = sum(int(s.get("tokens_in") or 0) + int(s.get("tokens_out") or 0)
                for s in trace.get("spans", []))
    trace.setdefault("meta", {})["total_tokens"] = total
    return changed, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Back-fill token/cost estimates in exported traces.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--rate", type=float, default=DEFAULT_COST_PER_TOKEN,
                    help="USD per token for back-filled cost_usd (default 1e-5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change without writing files")
    args = ap.parse_args()

    files = sorted(Path(args.raw_dir).rglob("*.json"))
    if not files:
        print(f"No .json files found under {args.raw_dir}")
        sys.exit(1)

    total_files = 0
    total_spans_changed = 0
    total_tokens = 0
    for fp in files:
        try:
            trace = json.loads(fp.read_text())
        except Exception as exc:
            print(f"  SKIP (read error): {fp}: {exc}")
            continue
        if "spans" not in trace:
            continue
        changed, ttl = backfill_trace(trace, args.rate)
        total_files += 1
        total_spans_changed += changed
        total_tokens += ttl
        if not args.dry_run:
            fp.write_text(json.dumps(trace, indent=2, default=str))
        print(f"  {fp.name}: spans_changed={changed}  total_tokens={ttl}")

    print(f"\nDone. files={total_files}  spans_backfilled={total_spans_changed}  "
          f"sum_total_tokens={total_tokens}" + ("  [DRY RUN]" if args.dry_run else ""))
    if not args.dry_run and total_files:
        print("Next: python export_gnn_graphs.py --overwrite")


if __name__ == "__main__":
    main()