"""Retrain spatial probe with balanced classes from original features.

Loads the original spatial_features.npz, balances positives/negatives,
retrains LogisticRegression, saves new classifier.

Usage:
    python scripts/retrain_probe_balanced.py \
        --features /data3/ishaplan/final_version/checkpoints/spatial_grounding/spatial_features.npz \
        --output /data3/ishaplan/slake_reproduction/checkpoints/spatial_probe_corrlabels/
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to spatial_features.npz")
    parser.add_argument("--output", required=True, help="Output directory for classifier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    d = np.load(args.features)
    X = d["spatial_ratios"]
    y = d["correct_labels"]

    print(f"Loaded: {X.shape[0]} examples, {X.shape[1]} features")
    print(f"Original: {int(y.sum())}/{len(y)} positive ({100*y.mean():.1f}%)")

    # Balance: keep all positives, subsample negatives to match
    rng = np.random.RandomState(args.seed)
    pos_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]
    n_pos = len(pos_idx)

    neg_sample = rng.choice(neg_idx, size=n_pos, replace=False)
    balanced_idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(balanced_idx)

    X_bal = X[balanced_idx]
    y_bal = y[balanced_idx]
    print(f"Balanced: {len(y_bal)} examples, {int(y_bal.sum())} positive ({100*y_bal.mean():.1f}%)")

    # Train/val split (80/20)
    n_train = int(0.8 * len(y_bal))
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, len(y_bal))

    # Sweep C
    best_auroc = 0
    best_clf = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=1000, C=C)
        clf.fit(X_bal[train_idx], y_bal[train_idx])
        val_probs = clf.predict_proba(X_bal[val_idx])[:, 1]
        auroc = roc_auc_score(y_bal[val_idx], val_probs)
        print(f"  C={C}: val AUROC={auroc:.3f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_clf = clf

    print(f"Best val AUROC: {best_auroc:.3f}")

    # Check correlation with correctness on full original data
    full_probs = best_clf.predict_proba(X)[:, 1]
    corr = np.corrcoef(full_probs, y)[0, 1]
    print(f"Probe <-> correctness (full data): r={corr:.3f}")
    print(f"Mean predict_proba on positives: {full_probs[y==1].mean():.3f}")
    print(f"Mean predict_proba on negatives: {full_probs[y==0].mean():.3f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "spatial_classifier.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    print(f"Saved to {out / 'spatial_classifier.pkl'}")


if __name__ == "__main__":
    main()
