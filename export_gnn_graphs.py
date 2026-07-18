#!/usr/bin/env python3
"""
export_gnn_graphs.py  —  Convert exported trace JSONs to PyTorch Geometric Data objects.

Reads the schema-conformant trace JSON files written by export_traces.py
(data/raw/agent_system=open_deep_research/*.json), constructs graph-structured
representations suitable for GNN training, and writes one .pt file per trace to:

    data/graphs/agent_system=open_deep_research/<trace_id>.pt

Each .pt file contains a torch_geometric.data.Data object with:

Node features (x — shape [N, F]):
  [0:3]   role one-hot         [agent, llm, tool]
  [3]     latency_ms            (log1p-normalized)
  [4]     tokens_total          (log1p-normalized)
  [5]     cost_usd              (log1p-normalized)
  [6]     error_flag            (0/1)
  [7]     loop_index            (0 if none, 1-based loop iteration normalized)
  [8:11]  model one-hot         [llama-3.3-70b, llama-3.1-8b, other]
  [11:14] node_type one-hot     [researcher_node, tool_router_node, writer_node, ChatGroq, web_search, timeout_injection, other]
  Total: F = 3 + 1 + 1 + 1 + 1 + 1 + 3 + NODE_TYPE_DIM

Edge attributes (edge_attr — shape [E, 2]):
  [0]     edge_type_onehot[0]   control_flow
  [1]     edge_type_onehot[1]   tool_call
  -- one-hot over EDGE_TYPES

Graph-level targets (y — shape [1]):
  Label: 0=normal, 1=loop, 2=timeout, 3=retrieval_fail,
         4=context_overflow, 5=hallucination, 6=faulty_other

Usage:
    python export_gnn_graphs.py --input-dir data/raw/agent_system=open_deep_research
    python export_gnn_graphs.py --input-dir data/raw/agent_system=open_deep_research --out data/graphs
    python export_gnn_graphs.py --input-dir data/raw  # scans all agent_system=*/ subdirs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROLE_NAMES = ["agent", "llm", "tool"]
MODEL_NAMES = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "other"]
NODE_TYPE_NAMES = [
    "planner_node",
    "researcher_0",
    "researcher_1",
    "researcher_2",
    "researcher_3",
    "researcher_4",
    "researcher_node",           # fallback for legacy traces
    "tool_router_node",          # fallback for legacy traces
    "merger_node",
    "writer_node",
    "ChatGroq",
    "tool_call:web_search",
    "tool_call:web_search_0",
    "tool_call:web_search_1",
    "tool_call:web_search_2",
    "tool_call:web_search_3",
    "tool_call:web_search_4",
    "tool_call:local_knowledge",
    "timeout_injection",
    "other",
]
EDGE_TYPES = ["control_flow", "tool_call"]

# Label encoding
LABEL_MAP = {
    "normal": 0,
    "loop": 1,
    "timeout": 2,
    "retrieval_fail": 3,
    "context_overflow": 4,
    "hallucination": 5,
}

NODE_FEATURE_DIM = len(ROLE_NAMES) + 5 + len(MODEL_NAMES) + len(NODE_TYPE_NAMES)
# 3 (role) + 5 (latency, tokens, cost, error, loop) + 3 (model) + len(NODE_TYPE_NAMES) (node_type)


def _one_hot(value: str, categories: list[str], default: str | None = None) -> list[float]:
    """Return a one-hot vector for `value` in `categories`."""
    idx = categories.index(value) if value in categories else (categories.index(default) if default else 0)
    return [1.0 if i == idx else 0.0 for i in range(len(categories))]


def _log1p_norm(v: float) -> float:
    """Log1p normalization: log(1 + v), clipped to reasonable range."""
    return math.log1p(max(0.0, v))


def _count_loop_iterations(spans: list[dict]) -> int:
    """Count how many unique loop iterations appear in the trace."""
    loop_counts = set()
    for s in spans:
        # Check if loop_count appears in span metadata (for agent spans)
        # or we can infer from repeating researcher_node names
        if "researcher_node" in s.get("name", ""):
            # Count occurrences to track loop depth
            pass
    return len(loop_counts)


def _infer_loop_count(spans: list[dict]) -> int:
    """Infer loop iterations by counting consecutive researcher_node spans."""
    count = 0
    for s in spans:
        if "researcher_node" in s.get("name", ""):
            count += 1
    return count


def _infer_edge_type(parent_name: str, child_name: str) -> str:
    """Infer edge type from parent and child span names."""
    # Tool call edges
    if "tool_call" in child_name:
        return "tool_call"
    # Default: control flow
    return "control_flow"


# ---------------------------------------------------------------------------
# Trace → PyG Data converter
# ---------------------------------------------------------------------------
def trace_to_pyg_data(trace: dict) -> Data:
    """Convert a schema-conformant trace dict into a torch_geometric.data.Data.

    Args:
        trace: A dict matching the schema from export_traces.py:
            {
                "trace_id": str,
                "agent_system": str,
                "task": str,
                "run_id": str,
                "spans": [{
                    "span_id": str,
                    "parent_id": str | None,
                    "role": str,        # "agent" | "llm" | "tool"
                    "name": str,
                    "latency_ms": float,
                    "tokens_in": int,
                    "tokens_out": int,
                    "cost_usd": float,
                    "model": str,
                    "tool": str | None,
                    "error_flag": bool,
                    "synthetic_error_type": str | None,
                }, ...],
                "run_labels": {
                    "success": bool,
                    "slow": bool,
                    "expensive": bool,
                },
                "meta": {
                    "total_tokens": int,
                    "total_latency_ms": float,
                    "faulty_batch": bool,
                    "retries": int,
                    "llm_model": str,
                },
            }

    Returns:
        Data object with:
            x:          [N, F]  node feature matrix
            edge_index: [2, E]  adjacency in COO format
            edge_attr:  [E, 2]  edge type one-hot
            y:          [1]     graph-level label (int)
            trace_id:   str     (stored in data.__dict__ for reference)
    """
    spans = trace.get("spans", [])
    if not spans:
        # Empty trace — return a dummy graph
        return Data(
            x=torch.zeros((0, NODE_FEATURE_DIM)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, len(EDGE_TYPES))),
            y=torch.tensor([0], dtype=torch.long),
            trace_id=trace.get("trace_id", "empty"),
        )

    # Build span_id → index mapping
    sid_to_idx: dict[str, int] = {}
    for i, s in enumerate(spans):
        sid_to_idx[s.get("span_id", f"_missing_{i}")] = i

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------
    node_features = []
    for s in spans:
        role = s.get("role", "agent")
        name = s.get("name", "")
        model = s.get("model", "")
        tool = s.get("tool") or ""
        err = s.get("error_flag", False)

        latency = _log1p_norm(s.get("latency_ms", 0.0))
        tokens = _log1p_norm((s.get("tokens_in", 0) or 0) + (s.get("tokens_out", 0) or 0))
        cost = _log1p_norm(s.get("cost_usd", 0.0))
        error_f = 1.0 if err else 0.0

        # Loop index: derive from position among spans with same name
        # Simple heuristic: count occurrences of this name up to this point
        # Better: look for researcher_node repetition
        full_name = name
        if "tool_call" in name and tool:
            full_name = f"tool_call:{tool}"

        # Build the feature vector
        feats: list[float] = []
        feats.extend(_one_hot(role, ROLE_NAMES, default="agent"))
        feats.append(latency)
        feats.append(tokens)
        feats.append(cost)
        feats.append(error_f)
        # Loop index placeholder — computed per-graph below
        feats.append(0.0)  # placeholder, will be filled after aggregation
        feats.extend(_one_hot(model if model in MODEL_NAMES else "other", MODEL_NAMES, default="other"))

        # Node type: match against known names
        node_type = "other"
        for nt in NODE_TYPE_NAMES:
            if nt in full_name or (nt.replace("tool_call:", "") == tool):
                node_type = nt
                break
        feats.extend(_one_hot(node_type, NODE_TYPE_NAMES, default="other"))

        node_features.append(feats)

    # Fill loop_index per span: count researcher_* occurrences (dynamic Send API)
    researcher_count = 0
    for i, s in enumerate(spans):
        name = s.get("name", "")
        if name.startswith("researcher_"):
            researcher_count += 1
            node_features[i][len(ROLE_NAMES) + 4] = _log1p_norm(float(researcher_count))

    x = torch.tensor(node_features, dtype=torch.float)

    # ------------------------------------------------------------------
    # Edge index (directed: parent → child)
    # ------------------------------------------------------------------
    edge_indices: list[list[int]] = [[], []]  # [src, dst]
    edge_attrs: list[list[float]] = []

    for i, s in enumerate(spans):
        parent_id = s.get("parent_id")
        if parent_id is None:
            continue
        parent_idx = sid_to_idx.get(parent_id)
        if parent_idx is None:
            continue

        parent_name = spans[parent_idx].get("name", "")
        child_name = s.get("name", "")
        etype = _infer_edge_type(parent_name, child_name)

        edge_indices[0].append(parent_idx)
        edge_indices[1].append(i)
        edge_attrs.append(_one_hot(etype, EDGE_TYPES, default="control_flow"))

    edge_index = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0, len(EDGE_TYPES)))

    # ------------------------------------------------------------------
    # Graph-level label
    # ------------------------------------------------------------------
    synthetic_error = next(
        (s.get("synthetic_error_type") for s in trace.get("spans", []) if s.get("synthetic_error_type")),
        None,
    )
    faulty_batch = trace.get("meta", {}).get("faulty_batch", False)
    run_labels = trace.get("run_labels", {})

    if synthetic_error and synthetic_error in LABEL_MAP:
        label = LABEL_MAP[synthetic_error]
    elif faulty_batch:
        label = 6  # faulty_other (injected but no specific error captured)
    elif run_labels.get("slow", False):
        label = 0  # normal — slowness is a symptom, not a class
    else:
        label = 0  # normal

    y = torch.tensor([label], dtype=torch.long)

    # ------------------------------------------------------------------
    # Construct Data object
    # ------------------------------------------------------------------
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=x.size(0),
    )

    # Store metadata as attributes for downstream use
    data.trace_id = trace.get("trace_id", "")
    data.task = trace.get("task", "")[:200]
    data.agent_system = trace.get("agent_system", "unknown")
    data.run_labels_success = run_labels.get("success", False)
    data.run_labels_slow = run_labels.get("slow", False)
    data.run_labels_expensive = run_labels.get("expensive", False)
    data.total_latency_ms = trace.get("meta", {}).get("total_latency_ms", 0.0)
    data.total_tokens = trace.get("meta", {}).get("total_tokens", 0)
    data.faulty_batch = faulty_batch
    data.synthetic_error_type = synthetic_error or "none"
    data.num_spans = len(spans)

    return data


# ---------------------------------------------------------------------------
# Main: scan input dir, convert all trace JSONs, write .pt files
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert exported trace JSONs to PyG Data objects.")
    p.add_argument(
        "--input-dir",
        default="data/raw",
        help="Path to the export_traces.py output directory. "
             "Scans all agent_system=*/ subdirs. Default: data/raw",
    )
    p.add_argument(
        "--out",
        default="data/graphs",
        help="Root output directory. Writes to <out>/agent_system=<sys>/<trace_id>.pt. Default: data/graphs",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files. Default: skip if exists.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_root = Path(args.input_dir)
    out_root = Path(args.out)

    if not in_root.exists():
        print(f"ERROR: input directory not found: {in_root}")
        print("Run export_traces.py first to produce trace JSON files.")
        sys.exit(1)

    # Collect all trace JSON files
    trace_files: list[Path] = []
    agent_systems: set[str] = set()

    if in_root.is_dir():
        # Scan all subdirs for trace JSON files. This includes agent_system=*/
        # subdirs as well as other sources such as langgraph_traces.
        for subdir in sorted(in_root.iterdir()):
            if subdir.is_dir():
                for f in sorted(subdir.glob("*.json")):
                    trace_files.append(f)
    elif in_root.is_file() and in_root.suffix == ".json":
        trace_files.append(in_root)

    if not trace_files:
        print(f"No trace JSON files found in {in_root}. Run export_traces.py first.")
        sys.exit(0)

    print(f"Found {len(trace_files)} trace file(s) across {len(agent_systems) or 'unknown'} agent system(s).")
    if agent_systems:
        print(f"  Agent systems: {', '.join(sorted(agent_systems))}")

    converted, skipped, failed = 0, 0, 0
    label_counts: dict[int, int] = {}

    for trace_file in trace_files:
        try:
            trace = json.loads(trace_file.read_text())
        except Exception as exc:
            print(f"  FAIL reading {trace_file}: {exc}")
            failed += 1
            continue

        agent_sys = trace.get("agent_system", "unknown")
        trace_id = trace.get("trace_id", trace_file.stem)

        out_dir = out_root / f"agent_system={agent_sys}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{trace_id}.pt"

        if out_file.exists() and not args.overwrite:
            if args.verbose:
                print(f"  SKIP (exists): {out_file}")
            skipped += 1
            continue

        try:
            data = trace_to_pyg_data(trace)
            torch.save(data, out_file)
            converted += 1
            label_counts[data.y.item()] = label_counts.get(data.y.item(), 0) + 1

            if args.verbose:
                print(
                    f"  [{converted}] {trace_id}: "
                    f"N={data.num_nodes} E={data.edge_index.size(1)} "
                    f"label={data.y.item()} -> {out_file}"
                )
        except Exception as exc:
            print(f"  FAIL converting {trace_file.name}: {exc}")
            failed += 1

    print(f"\nDone. converted={converted}  skipped={skipped}  failed={failed}")
    if label_counts:
        label_names = {v: k for k, v in LABEL_MAP.items()}
        label_names[6] = "faulty_other"
        print("  Label distribution:")
        for label_id in sorted(label_counts):
            name = label_names.get(label_id, f"class_{label_id}")
            print(f"    {name} ({label_id}): {label_counts[label_id]}")
    print(f"\nOutput directory: {out_root}")
    print("\nNext step: train a GNN model:")
    print("  from torch_geometric.loader import DataLoader")
    print("  dataset = []")
    print(f"  for pt_file in Path('{out_root}').rglob('*.pt'):")
    print("      dataset.append(torch.load(pt_file))")
    print("  loader = DataLoader(dataset, batch_size=32, shuffle=True)")


if __name__ == "__main__":
    main()