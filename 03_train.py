"""
Train ML to predict UMI-dedup label from non-UMI features.

4:1 stratified train/test split. Compare LR / RF / GBM vs Picard baseline.

Critical: hold out WHOLE test set, never touched during training.
         No leakage from position → multiple reads at same position can split across train/test
         so ML may memorize. We address with a SECOND split: group by start position
         (all reads at same position go to same side) to test true generalization.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import warnings

warnings.filterwarnings("ignore")

FEATURES_PATH = "/home/eagle/try3/bench/results_ml/features.parquet"
OUT_DIR = "/home/eagle/try3/bench/results_ml"

df = pd.read_parquet(FEATURES_PATH)
print(f"Loaded: {df.shape}, dup rate: {df.umi_label.mean():.3f}")

# Features for training (exclude identifiers, UMI-bearing qname, raw labels, baselines)
DROP_COLS = ["qname", "umi_label", "picard_pred"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
print(f"\nFeatures used ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].copy()
y = df["umi_label"].values
picard_pred = df["picard_pred"].values

# ============================================================
# Split 1: random 4:1 (read-level) — may leak position info
# Split 2: group-by-position 4:1 — stricter, no position overlap
# ============================================================
print("\n" + "=" * 70)
print("SPLIT 1: Random 4:1 (read-level, may have position overlap)")
print("=" * 70)

X_train, X_test, y_train, y_test, pic_train, pic_test = train_test_split(
    X, y, picard_pred, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} (dup rate {y_train.mean():.3f})")
print(f"Test:  {len(X_test)} (dup rate {y_test.mean():.3f})")

models = {
    "LogReg": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "GradBoost": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    ),
}

results_split1 = {}
print("\n--- Split 1 results ---")
print(
    f"{'model':<16} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>8}"
)
print("-" * 65)

# Picard baseline on split1 test
pic_acc = (pic_test == y_test).mean()
pic_p = ((pic_test == 1) & (y_test == 1)).sum() / max((pic_test == 1).sum(), 1)
pic_r = ((pic_test == 1) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)
pic_f1 = 2 * pic_p * pic_r / (pic_p + pic_r) if pic_p + pic_r > 0 else 0
print(
    f"{'Picard(baseline)':<16} {pic_acc:>7.3f} {pic_p:>7.3f} {pic_r:>7.3f} {pic_f1:>7.3f} {'n/a':>8} {'n/a':>8}"
)
results_split1["Picard"] = {"acc": pic_acc, "prec": pic_p, "rec": pic_r, "f1": pic_f1}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred)
    p = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(
        f"{name:<16} {acc:>7.3f} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {auc:>8.3f} {ap:>8.3f}"
    )
    results_split1[name] = {
        "acc": acc,
        "prec": p,
        "rec": r,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

# ============================================================
# Split 2: group by position (stricter — no position leakage)
# ============================================================
print("\n" + "=" * 70)
print("SPLIT 2: Group-by-position 4:1 (all reads at same start → same side)")
print("=" * 70)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=X["start"]))
X_train2, X_test2 = X.iloc[train_idx], X.iloc[test_idx]
y_train2, y_test2 = y[train_idx], y[test_idx]
pic_train2, pic_test2 = picard_pred[train_idx], picard_pred[test_idx]
print(f"Train: {len(X_train2)} (dup rate {y_train2.mean():.3f})")
print(f"Test:  {len(X_test2)} (dup rate {y_test2.mean():.3f})")
print(
    f"Position overlap train/test: "
    f"{len(set(X_train2['start']) & set(X_test2['start']))}"
)

results_split2 = {}
print("\n--- Split 2 results ---")
print(
    f"{'model':<16} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>8}"
)
print("-" * 65)

pic_acc2 = (pic_test2 == y_test2).mean()
cm = confusion_matrix(y_test2, pic_test2)
p = cm[1, 1] / max(cm[:, 1].sum(), 1)
r = cm[1, 1] / max(cm[1, :].sum(), 1)
f1 = 2 * p * r / (p + r) if p + r > 0 else 0
print(
    f"{'Picard(baseline)':<16} {pic_acc2:>7.3f} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {'n/a':>8} {'n/a':>8}"
)
results_split2["Picard"] = {"acc": pic_acc2, "prec": p, "rec": r, "f1": f1}

for name, model in models.items():
    model.fit(X_train2, y_train2)
    y_pred = model.predict(X_test2)
    y_proba = model.predict_proba(X_test2)[:, 1]

    acc = (y_pred == y_test2).mean()
    cm = confusion_matrix(y_test2, y_pred)
    p = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    auc = roc_auc_score(y_test2, y_proba)
    ap = average_precision_score(y_test2, y_proba)
    print(
        f"{name:<16} {acc:>7.3f} {p:>7.3f} {r:>7.3f} {f1:>8.3f} {auc:>8.3f} {ap:>8.3f}"
    )
    results_split2[name] = {
        "acc": acc,
        "prec": p,
        "rec": r,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

# ============================================================
# Feature importance (from GradBoost, Split 1)
# ============================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (GradientBoosting, Split 1)")
print("=" * 70)

gb = models["GradBoost"]
importances = gb.feature_importances_
fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
fi = fi.sort_values("importance", ascending=False)
print(fi.to_string(index=False))
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

# ============================================================
# Plot: ROC curves for Split 1 and Split 2
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, results, title in [
    (axes[0], results_split1, "Split 1: random 4:1"),
    (axes[1], results_split2, "Split 2: group-by-position"),
]:
    for name in ["LogReg", "RandomForest", "GradBoost"]:
        if name not in results or "y_proba" not in results[name]:
            continue
        y_proba = results[name]["y_proba"]
        y_true = y_test if "Split 1" in title else y_test2
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = results[name]["auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    # Picard as single point
    if "Picard" in results:
        p = results["Picard"]
        # Picard has no threshold; plot its TPR/FPR as a point
        cm = confusion_matrix(
            y_test if "Split 1" in title else y_test2,
            pic_test if "Split 1" in title else pic_test2,
        )
        tpr = cm[1, 1] / cm[1, :].sum()
        fpr = cm[0, 1] / cm[0, :].sum()
        ax.scatter(
            [fpr],
            [tpr],
            marker="x",
            s=100,
            color="black",
            label=f"Picard (FPR={fpr:.2f}, TPR={tpr:.2f})",
            zorder=5,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=120, bbox_inches="tight")
print(f"\nSaved: {OUT_DIR}/roc_curves.png")

# Summary table
summary = []
for split_name, results in [
    ("split1_random", results_split1),
    ("split2_groupby_pos", results_split2),
]:
    for model_name, m in results.items():
        row = {"split": split_name, "model": model_name}
        row.update({k: v for k, v in m.items() if not isinstance(v, np.ndarray)})
        summary.append(row)
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print(f"Saved: {OUT_DIR}/model_comparison.csv")
