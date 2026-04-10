"""Train spatial grounding classifier from saved features.

Compares:
  - LogisticRegression (CE loss) with C sweep
  - Ridge regression (MSE loss) with alpha sweep
  - Both with and without StandardScaler normalization

Usage:
    python scripts/train_spatial_classifier_from_saved.py
    python scripts/train_spatial_classifier_from_saved.py --features checkpoints/spatial_grounding_v1_full/spatial_features.npz
"""
import argparse
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="checkpoints/spatial_grounding/spatial_features.npz")
parser.add_argument("--out_dir", default=None, help="Output dir (default: same as features)")
parser.add_argument("--val_ratio", type=float, default=0.2)
args = parser.parse_args()

data = np.load(args.features)
X = data["spatial_ratios"]
y = data["correct_labels"]

out_dir = args.out_dir or str(args.features).rsplit("/", 1)[0]

logger.info(f"Loaded {len(X)} samples, {int(y.sum())} correct ({y.mean():.1%}), "
            f"{X.shape[1]} features")

# Train/val split
n_val = int(args.val_ratio * len(X))
n_train = len(X) - n_val
idx = np.random.RandomState(42).permutation(len(X))
train_idx, val_idx = idx[:n_train], idx[n_train:]

logger.info(f"Train: {n_train}, Val: {n_val}")


def eval_model(model, X_train, y_train, X_val, y_val):
    """Fit and evaluate, return (val_auroc, val_r, train_auroc, val_preds)."""
    model.fit(X_train, y_train)

    # Get predictions — predict_proba for classifiers, predict for regressors.
    if hasattr(model, "predict_proba"):
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
    else:
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

    train_auroc = roc_auc_score(y_train, train_preds)
    val_auroc = roc_auc_score(y_val, val_preds)
    val_r = np.corrcoef(val_preds, y_val)[0, 1]
    return val_auroc, val_r, train_auroc, val_preds


X_tr, y_tr = X[train_idx], y[train_idx]
X_va, y_va = X[val_idx], y[val_idx]

best_auroc = -1.0
best_name = ""
best_model = None

# --- LogisticRegression (CE) sweep ---
logger.info("\n=== LogisticRegression (CE) ===")
logger.info(f"{'Config':<35} {'Train AUROC':>12} {'Val AUROC':>10} {'Val r':>8}")
logger.info("-" * 70)

for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    for scale in [False, True]:
        if scale:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=C)),
            ])
            name = f"LogReg C={C:<6} +scaler"
        else:
            model = LogisticRegression(max_iter=2000, C=C)
            name = f"LogReg C={C:<6}        "

        va_auc, va_r, tr_auc, _ = eval_model(model, X_tr, y_tr, X_va, y_va)
        logger.info(f"{name:<35} {tr_auc:>11.3f} {va_auc:>10.3f} {va_r:>8.3f}")

        if va_auc > best_auroc:
            best_auroc = va_auc
            best_name = name.strip()
            best_model = model

# --- Ridge regression (MSE) sweep ---
logger.info("\n=== Ridge Regression (MSE) ===")
logger.info(f"{'Config':<35} {'Train AUROC':>12} {'Val AUROC':>10} {'Val r':>8}")
logger.info("-" * 70)

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    for scale in [False, True]:
        if scale:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            name = f"Ridge α={alpha:<8} +scaler"
        else:
            model = Ridge(alpha=alpha)
            name = f"Ridge α={alpha:<8}        "

        va_auc, va_r, tr_auc, _ = eval_model(model, X_tr, y_tr, X_va, y_va)
        logger.info(f"{name:<35} {tr_auc:>11.3f} {va_auc:>10.3f} {va_r:>8.3f}")

        if va_auc > best_auroc:
            best_auroc = va_auc
            best_name = name.strip()
            best_model = model

# --- Save best ---
logger.info(f"\nBest: {best_name} | Val AUROC={best_auroc:.3f}")

out_path = f"{out_dir}/spatial_classifier.pkl"
with open(out_path, "wb") as f:
    pickle.dump(best_model, f)
logger.info(f"Saved to {out_path}")
