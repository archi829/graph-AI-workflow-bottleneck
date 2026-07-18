#!/usr/bin/env python3
"""
train_success_binary.py -- Binary success/fail classification from trace graphs.

Reuses the .pt files already produced by export_gnn_graphs.py. Target is
data.run_labels_success (bool), NOT the multiclass failure-type label (data.y).
No re-export needed -- this label was already stored on every graph.

Usage:
    python train_success_binary.py --model xgboost
    python train_success_binary.py --model gnn
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

from train_common import load_graphs, graph_to_features, build_feature_matrix


def get_binary_labels(graphs) -> list[int]:
    return [1 if bool(getattr(g, "run_labels_success", False)) else 0 for g in graphs]


def run_xgboost(graphs, labels):
    import xgboost as xgb

    X = build_feature_matrix(graphs)
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        fold_f1 = f1_score(y[test_idx], preds, average="macro")
        fold_f1s.append(fold_f1)
        all_true.extend(y[test_idx])
        all_pred.extend(preds)
        print(f"--- Fold {fold} ---  macro-F1 = {fold_f1:.3f}")

    print(f"\n=== 5-fold stratified CV summary (XGBoost, binary success/fail) ===")
    print(f"Per-fold macro-F1: {[round(f, 3) for f in fold_f1s]}")
    print(f"Mean macro-F1: {np.mean(fold_f1s):.3f}  Std: {np.std(fold_f1s):.3f}")
    print("\nAggregated classification report (all folds):")
    print(classification_report(all_true, all_pred, target_names=["fail", "success"]))


def run_gnn(graphs, labels):
    from torch_geometric.loader import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GNN(nn.Module):
        def __init__(self, in_dim, hidden=64, num_classes=2):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.lin = nn.Linear(hidden, num_classes)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.lin(x)

    y = np.array(labels)
    idx = np.arange(len(graphs))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    all_true, all_pred = [], []
    in_dim = graphs[0].x.shape[1]

    for fold, (train_idx, test_idx) in enumerate(skf.split(idx, y), 1):
        for g, lbl in zip(graphs, labels):
            g.y_binary = torch.tensor([lbl], dtype=torch.long)

        train_data = [graphs[i] for i in train_idx]
        test_data = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

        model = GNN(in_dim)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        model.train()
        for epoch in range(60):
            for batch in train_loader:
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(out, batch.y_binary)
                loss.backward()
                opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                preds.extend(pred.tolist())
                trues.extend(batch.y_binary.tolist())

        fold_f1 = f1_score(trues, preds, average="macro")
        fold_f1s.append(fold_f1)
        all_true.extend(trues)
        all_pred.extend(preds)
        print(f"--- Fold {fold} ---  macro-F1 = {fold_f1:.3f}")

    print(f"\n=== 5-fold stratified CV summary (GNN, binary success/fail) ===")
    print(f"Per-fold macro-F1: {[round(f, 3) for f in fold_f1s]}")
    print(f"Mean macro-F1: {np.mean(fold_f1s):.3f}  Std: {np.std(fold_f1s):.3f}")
    print("\nAggregated classification report (all folds):")
    print(classification_report(all_true, all_pred, target_names=["fail", "success"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgboost", "gnn"], required=True)
    ap.add_argument("--graphs-dir", default="data/graphs")
    args = ap.parse_args()

    graphs, _ = load_graphs(args.graphs_dir)
    labels = get_binary_labels(graphs)

    print(f"Loaded {len(graphs)} graphs. Class balance: "
          f"success={sum(labels)}  fail={len(labels)-sum(labels)}")

    if args.model == "xgboost":
        run_xgboost(graphs, labels)
    else:
        run_gnn(graphs, labels)


if __name__ == "__main__":
    main()