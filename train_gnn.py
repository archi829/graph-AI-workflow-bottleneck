#!/usr/bin/env python3
"""
train_gnn.py  --  GNN bottleneck-detection training with stratified 5-fold CV.

Why these choices (per the dataset's small, imbalanced class sizes:
84 clean vs 10-17 faulty):
  * StratifiedKFold (shuffle=True) so every fold keeps the same class ratios.
    A plain random split would leave only ~2 timeout traces in a single test
    set, which is too few to trust any accuracy number.
  * Class-weighted CrossEntropyLoss (inverse-frequency) so the 84 clean
    traces don't dominate the gradient. We do NOT discard clean examples.
  * 5 folds instead of one train/test split: with only 10 timeout traces a
    single 80/20 split is high-variance; averaging over 5 folds gives a
    far more stable estimate of per-class performance.

Run:
    python train_gnn.py
    python train_gnn.py --folds 5 --epochs 60 --hidden 64
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score

import train_common as tc


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BottleneckGNN(torch.nn.Module):
    """RGCN that uses the edge-type signal (control_flow vs tool_call)
    captured by export_gnn_graphs.py in data.edge_attr.

    Unlike a plain GCN, RGCNConv learns a separate relation-specific weight
    matrix per edge type, so the model can distinguish a slow tool-call edge
    from a slow control-flow edge -- signal that is valuable for bottleneck
    detection.
    """

    NUM_EDGE_TYPES = 2  # EDGE_TYPES = ["control_flow", "tool_call"]

    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden, num_relations=self.NUM_EDGE_TYPES)
        self.conv2 = RGCNConv(hidden, hidden, num_relations=self.NUM_EDGE_TYPES)
        self.lin = torch.nn.Linear(hidden, num_classes)
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_attr is a one-hot over EDGE_TYPES -> argmax gives the type index
        edge_type = data.edge_attr.argmax(dim=1) if data.edge_attr.numel() else None
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = self.drop(x)
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)        # [B, hidden]
        return self.lin(x)                     # [B, num_classes]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_loader(graphs, labels, batch_size, shuffle):
    data_list = []
    for g, y in zip(graphs, labels):
        g = g.clone()
        g.y = torch.tensor([y], dtype=torch.long)
        data_list.append(g)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(data.y.view(-1).cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default="data/graphs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, raw_labels = tc.load_graphs(args.graphs_dir)
    labels, mapping = tc.remap_labels(raw_labels)
    num_classes = len(mapping)
    print(f"Loaded {len(graphs)} graphs, {num_classes} classes: "
          f"{[tc.LABEL_NAMES[r] for r in sorted(mapping)]}")
    print("Class counts (remapped):",
          {tc.LABEL_NAMES[r]: labels.count(i) for i, r in sorted(mapping.items())})

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_macro_f1, fold_reports = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(graphs, labels), 1):
        tr_g, te_g = [graphs[i] for i in tr_idx], [graphs[i] for i in te_idx]
        tr_y, te_y = [labels[i] for i in tr_idx], [labels[i] for i in te_idx]

        # Class weights computed ONLY on the training fold (no leakage)
        cw = tc.class_weights(tr_y, num_classes).to(device)

        model = BottleneckGNN(tc.NUM_NODE_FEATURES, args.hidden, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        crit = torch.nn.CrossEntropyLoss(weight=cw)

        tr_loader = to_loader(tr_g, tr_y, args.batch_size, shuffle=True)
        te_loader = to_loader(te_g, te_y, args.batch_size, shuffle=False)

        for _ in range(args.epochs):
            model.train()
            for data in tr_loader:
                data = data.to(device)
                opt.zero_grad()
                out = model(data)
                loss = crit(out, data.y.view(-1))
                loss.backward()
                opt.step()

        y_true, y_pred = evaluate(model, te_loader, device)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        fold_macro_f1.append(macro)
        fold_reports.append((y_true, y_pred))
        print(f"\n--- Fold {fold} ---  macro-F1 = {macro:.3f}")

    print("\n=== 5-fold stratified CV summary (GNN) ===")
    print(f"Per-fold macro-F1: {[round(f, 3) for f in fold_macro_f1]}")
    print(f"Mean macro-F1: {np.mean(fold_macro_f1):.3f}  "
          f"Std: {np.std(fold_macro_f1):.3f}")

    # Aggregate per-class report across all folds (concatenate predictions)
    all_true = np.concatenate([r[0] for r in fold_reports])
    all_pred = np.concatenate([r[1] for r in fold_reports])
    target_names = [tc.LABEL_NAMES[r] for r in sorted(mapping)]
    print("\nAggregated classification report (all folds):")
    print(classification_report(all_true, all_pred, target_names=target_names,
                                labels=list(range(num_classes)), zero_division=0))


if __name__ == "__main__":
    main()