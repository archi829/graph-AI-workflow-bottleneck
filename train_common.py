#!/usr/bin/env python3
"""
train_common.py  --  Shared data loading for GNN / XGBoost bottleneck-detection training.

Loads the .pt graph files produced by export_gnn_graphs.py and exposes:
  * load_graphs()        -> (list[Data], list[int raw labels])
  * remap_labels()       -> contiguous 0..K-1 labels + mapping
  * graph_to_features()  -> fixed-size flat vector per graph (for XGBoost)

The label space in this dataset is {0:normal, 1:loop, 2:timeout,
3:retrieval_fail, 5:hallucination}. 4 (context_overflow) and 6 (faulty_other)
are absent, so remap_labels() compacts the present labels to 0..4.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

# Raw label id -> human name (mirrors export_gnn_graphs.LABEL_MAP)
LABEL_NAMES = {
    0: "normal",
    1: "loop",
    2: "timeout",
    3: "retrieval_fail",
    4: "context_overflow",
    5: "hallucination",
    6: "faulty_other",
}

NUM_NODE_FEATURES = 31  # must match NODE_FEATURE_DIM in export_gnn_graphs.py


def load_graphs(graphs_dir: str = "data/graphs"):
    """Load every .pt Data object under graphs_dir.

    Returns (graphs, labels) where labels are the raw integer graph labels
    stored in each Data.y. Graphs with zero nodes are dropped (they carry no
    signal and break global pooling).
    """
    files = sorted(Path(graphs_dir).rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {graphs_dir!r}")

    graphs, labels = [], []
    for f in files:
        # PyG Data objects require weights_only=False (or a safe-globals allowlist)
        d = torch.load(f, weights_only=False)
        if d.num_nodes == 0:
            continue
        graphs.append(d)
        labels.append(int(d.y.item()))
    return graphs, labels


def remap_labels(labels):
    """Map raw labels to contiguous 0..K-1.

    Returns (remapped_list, mapping) where mapping: raw_label -> new_index.
    """
    uniq = sorted(set(labels))
    mapping = {r: i for i, r in enumerate(uniq)}
    return [mapping[l] for l in labels], mapping


def class_weights(labels, num_classes):
    """Inverse-frequency weights: w_c = N / (num_classes * count_c).

    Returns a torch tensor of shape [num_classes] suitable for
    nn.CrossEntropyLoss(weight=...). Classes with zero count get weight 0.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.where(counts > 0, len(labels) / (num_classes * counts), 0.0)
    return torch.tensor(weights, dtype=torch.float)


def graph_to_features(data) -> np.ndarray:
    """Flatten a graph into a fixed-size feature vector for tabular models.

    Concatenates per-node mean and max of the 31 node features with a few
    graph-level aggregates (span count, log latency, log tokens).
    """
    x = data.x.numpy() if hasattr(data.x, "numpy") else np.asarray(data.x)
    if x.shape[0] == 0:
        base = np.zeros(NUM_NODE_FEATURES * 2, dtype=np.float32)
    else:
        mean = x.mean(axis=0)
        mx = x.max(axis=0)
        base = np.concatenate([mean, mx]).astype(np.float32)

    meta = np.array(
        [
            float(getattr(data, "num_spans", 0) or 0),
            float(np.log1p(getattr(data, "total_latency_ms", 0) or 0)),
            float(np.log1p(getattr(data, "total_tokens", 0) or 0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, meta])


def build_feature_matrix(graphs):
    """Stack graph_to_features() over all graphs -> [N, F] float32 array."""
    return np.stack([graph_to_features(g) for g in graphs], axis=0)