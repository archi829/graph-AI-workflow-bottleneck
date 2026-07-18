#!/usr/bin/env python3
"""
train_xgboost.py  --  XGBoost bottleneck-detection training with stratified 5-fold CV.

Mirrors the GNN script's evaluation protocol so the two models are comparable:
  * StratifiedKFold (shuffle=True) preserves class ratios in every fold.
  * scale_pos_weight / class weights counter the 84-vs-10/14/14/17 imbalance.
    We do NOT discard clean examples; instead we weight the loss so the
    minority faulty classes contribute proportionally.
  * 5 folds (not a single split) because a single 80/20 split leaves only
    ~2 timeout traces in test -- too few to trust.

Run:
    python train_xgboost.py
    python train_xgboost.py --folds 5 --estimators 300
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

import train_common as tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default="data/graphs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    graphs, raw_labels = tc.load_graphs(args.graphs_dir)
    X = tc.build_feature_matrix(graphs)
    labels, mapping = tc.remap_labels(raw_labels)
    num_classes = len(mapping)

    print(f"Loaded {len(graphs)} graphs, feature dim {X.shape[1]}, "
          f"{num_classes} classes: {[tc.LABEL_NAMES[r] for r in sorted(mapping)]}")
    print("Class counts (remapped):",
          {tc.LABEL_NAMES[r]: labels.count(i) for i, r in sorted(mapping.items())})

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_macro_f1, fold_reports = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, labels), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = np.array([labels[i] for i in tr_idx]), np.array([labels[i] for i in te_idx])

        # Class weights from the TRAINING fold only (no leakage)
        counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
        # weight_c = N / (num_classes * count_c)  (matches the GNN convention)
        weights = np.where(counts > 0, len(y_tr) / (num_classes * counts), 0.0)
        sample_weight = np.array([weights[c] for c in y_tr])

        model = xgb.XGBClassifier(
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_weight)

        y_pred = model.predict(X_te)
        macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_macro_f1.append(macro)
        fold_reports.append((y_te, y_pred))
        print(f"\n--- Fold {fold} ---  macro-F1 = {macro:.3f}")

    print("\n=== 5-fold stratified CV summary (XGBoost) ===")
    print(f"Per-fold macro-F1: {[round(f, 3) for f in fold_macro_f1]}")
    print(f"Mean macro-F1: {np.mean(fold_macro_f1):.3f}  "
          f"Std: {np.std(fold_macro_f1):.3f}")

    all_true = np.concatenate([r[0] for r in fold_reports])
    all_pred = np.concatenate([r[1] for r in fold_reports])
    target_names = [tc.LABEL_NAMES[r] for r in sorted(mapping)]
    print("\nAggregated classification report (all folds):")
    print(classification_report(all_true, all_pred, target_names=target_names,
                                labels=list(range(num_classes)), zero_division=0))


if __name__ == "__main__":
    main()