"""
Final analysis: fix issues from train_wg.py
- HistGBM doesn't have feature_importances_ → use permutation_importance
- Add class-0 (unique) metrics: recall_unique, precision_unique, F1_unique
- Add balanced_accuracy
- Threshold sweep optimizing F1_unique (the interesting class for RNA-seq)
- New figures focused on the RIGHT question: can ML rescue Picard's false positives?
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
import time, os, warnings, json

warnings.filterwarnings("ignore")

FEATURES_PATH = "/home/eagle/try3/bench/results_wg/features.parquet"
OUT_DIR = "/home/eagle/try3/bench/results_wg/figures_ml"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
print("Loading features ...")
df = pd.read_parquet(FEATURES_PATH)
print(f"  Shape: {df.shape}, dup rate: {df.umi_label.mean():.4f}")

DROP = ["qname", "umi_label", "picard_pred", "chrom"]
FEATURE_COLS = [c for c in df.columns if c not in DROP]
X = df[FEATURE_COLS].astype(np.float32)
y = df["umi_label"].values.astype(np.int8)
picard = df["picard_pred"].values.astype(np.int8)
groups = df["start"].values

# Two splits
X_tr1, X_te1, y_tr1, y_te1, p_tr1, p_te1 = train_test_split(
    X, y, picard, test_size=0.2, random_state=42, stratify=y
)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr2_idx, te2_idx = next(gss.split(X, y, groups=groups))
X_tr2, X_te2 = X.iloc[tr2_idx], X.iloc[te2_idx]
y_tr2, y_te2 = y[tr2_idx], y[te2_idx]
p_tr2, p_te2 = picard[tr2_idx], picard[te2_idx]

# Models (no class weight — let ML learn natural prior)
models = {
    "LogReg_balanced": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42, n_jobs=-1
                ),
            ),
        ]
    ),
    "HistGBM_natural": HistGradientBoostingClassifier(
        learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, random_state=42
    ),
    "HistGBM_balanced": HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    ),
}


def full_metrics(name, model, X_tr, y_tr, X_te, y_te, p_te):
    t = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t
    y_proba = model.predict_proba(X_te)[:, 1]
    # Default threshold = 0.5
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred_default)

    # Standard metrics (on dup class 1)
    acc = (y_pred_default == y_te).mean()
    bal_acc = balanced_accuracy_score(y_te, y_pred_default)

    # Class-1 (dup)
    p1 = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r1 = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1_1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 > 0 else 0

    # Class-0 (unique) - the BIOLOGICALLY IMPORTANT class
    p0 = cm[0, 0] / max(cm[:, 0].sum(), 1)
    r0 = cm[0, 0] / max(cm[0, :].sum(), 1)
    f1_0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 > 0 else 0

    auc = roc_auc_score(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)

    print(f"  {name}:")
    print(f"    acc={acc:.4f} bal_acc={bal_acc:.4f} AUC={auc:.4f} AP={ap:.4f}")
    print(f"    dup class:    P={p1:.4f} R={r1:.4f} F1={f1_1:.4f}")
    print(f"    unique class: P={p0:.4f} R={r0:.4f} F1={f1_0:.4f}")
    print(f"    (train {train_time:.1f}s)")
    return {
        "name": name,
        "model": model,
        "y_proba": y_proba,
        "y_pred": y_pred_default,
        "cm": cm,
        "acc": acc,
        "bal_acc": bal_acc,
        "auc": auc,
        "ap": ap,
        "p1": p1,
        "r1": r1,
        "f1_1": f1_1,
        "p0": p0,
        "r0": r0,
        "f1_0": f1_0,
    }


def picard_metrics(y_te, p_te):
    cm = confusion_matrix(y_te, p_te)
    acc = (p_te == y_te).mean()
    bal_acc = balanced_accuracy_score(y_te, p_te)
    p1 = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r1 = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1_1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 > 0 else 0
    p0 = cm[0, 0] / max(cm[:, 0].sum(), 1)
    r0 = cm[0, 0] / max(cm[0, :].sum(), 1)
    f1_0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 > 0 else 0
    print(f"  Picard:")
    print(f"    acc={acc:.4f} bal_acc={bal_acc:.4f}")
    print(f"    dup class:    P={p1:.4f} R={r1:.4f} F1={f1_1:.4f}")
    print(f"    unique class: P={p0:.4f} R={r0:.4f} F1={f1_0:.4f}")
    return {
        "name": "Picard",
        "cm": cm,
        "acc": acc,
        "bal_acc": bal_acc,
        "auc": None,
        "ap": None,
        "p1": p1,
        "r1": r1,
        "f1_1": f1_1,
        "p0": p0,
        "r0": r0,
        "f1_0": f1_0,
        "y_proba": None,
        "y_pred": p_te,
        "model": None,
    }


print("\n" + "=" * 70)
print("SPLIT 1 (random 4:1)")
print("=" * 70)
results1 = [picard_metrics(y_te1, p_te1)]
for name, m in models.items():
    results1.append(full_metrics(name, m, X_tr1, y_tr1, X_te1, y_te1, p_te1))

print("\n" + "=" * 70)
print("SPLIT 2 (group-by-position 4:1, no leak)")
print("=" * 70)
results2 = [picard_metrics(y_te2, p_te2)]
for name, m in models.items():
    results2.append(full_metrics(name, m, X_tr2, y_tr2, X_te2, y_te2, p_te2))

# ============================================================
# FIGURE 1: ROC + PR + Class-0 specific view (4 panels)
# ============================================================
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ROC by split
for ax_row, results, y_te, p_te, title_prefix in [
    (axes[0], results1, y_te1, p_te1, "Split 1 (random 4:1)"),
    (axes[1], results2, y_te2, p_te2, "Split 2 (group-by-position 4:1)"),
]:
    ax = ax_row[0]
    for r in results:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_te, r["y_proba"])
            ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})", lw=2)
    cm = confusion_matrix(y_te, p_te)
    tpr_p = cm[1, 1] / cm[1, :].sum()
    fpr_p = cm[0, 1] / cm[0, :].sum()
    ax.scatter(
        [fpr_p],
        [tpr_p],
        marker="x",
        s=200,
        color="black",
        zorder=5,
        label=f"Picard",
        linewidths=3,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (true unique wrongly marked dup)")
    ax.set_ylabel("True Positive Rate (true dup correctly marked)")
    ax.set_title(f"ROC — {title_prefix}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # PR
    ax = ax_row[1]
    for r in results:
        if r["y_proba"] is not None:
            pp, rr, _ = precision_recall_curve(y_te, r["y_proba"])
            ax.plot(rr, pp, label=f"{r['name']} (AP={r['ap']:.3f})", lw=2)
    ax.scatter(
        [cm[1, 1] / cm[1, :].sum()],
        [cm[1, 1] / cm[:, 1].sum()],
        marker="x",
        s=200,
        color="black",
        zorder=5,
        label="Picard",
        linewidths=3,
    )
    ax.axhline(
        (y_te == 1).mean(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"baseline ({(y_te == 1).mean():.3f})",
    )
    ax.set_xlabel("Recall (dup)")
    ax.set_ylabel("Precision (dup)")
    ax.set_title(f"PR — {title_prefix}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.suptitle(
    "Whole-genome ML dedup benchmark (n=2.88M reads, SRR28314028)", fontsize=12, y=1.005
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig1_roc_pr_curves.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved fig1_roc_pr_curves")

# ============================================================
# FIGURE 2: Confusion matrices for Picard vs best ML
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax_row, results, title_prefix in [
    (axes[0], results1, "Split 1"),
    (axes[1], results2, "Split 2"),
]:
    pick = [r for r in results if r["name"] in ("Picard", "HistGBM_balanced")]
    for ax, r in zip(ax_row, pick):
        cm = r["cm"]
        cm_pct = cm.astype(float) / cm.sum() * 100
        ax.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=cm_pct.max())
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm_pct[i, j] > cm_pct.max() / 2 else "black",
                    fontsize=11,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred: unique", "pred: dup"])
        ax.set_yticklabels(["true: unique", "true: dup"])
        ax.set_title(
            f"{r['name']} — {title_prefix}\n"
            f"acc={r['acc']:.3f} bal_acc={r['bal_acc']:.3f} | "
            f"unique_recall={r['r0']:.3f} | dup_F1={r['f1_1']:.3f}",
            fontsize=10,
        )
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig2_confusion_matrices.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved fig2_confusion_matrices")

# ============================================================
# FIGURE 3: Permutation importance (HistGBM_balanced)
# ============================================================
print("  Computing permutation importance ...")
hgb = next(r for r in results2 if r["name"] == "HistGBM_balanced")["model"]
# subsample test for speed
np.random.seed(42)
n_sub = min(100_000, len(X_te2))
idx = np.random.choice(len(X_te2), n_sub, replace=False)
result_imp = permutation_importance(
    hgb,
    X_te2.iloc[idx],
    y_te2[idx],
    n_repeats=3,
    random_state=42,
    scoring="balanced_accuracy",
    n_jobs=-1,
)
fi = pd.DataFrame(
    {
        "feature": FEATURE_COLS,
        "importance_mean": result_imp.importances_mean,
        "importance_std": result_imp.importances_std,
    }
).sort_values("importance_mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["tab:red" if f == "n_reads_at_pos" else "tab:blue" for f in fi.feature]
ax.barh(
    fi.feature, fi.importance_mean, xerr=fi.importance_std, color=colors, alpha=0.85
)
ax.set_xlabel("Permutation importance (Δ balanced_accuracy)")
ax.set_title("Feature importance — HistGBM_balanced (Split 2, n=100k subsample)")
for i, (f, v) in enumerate(zip(fi.feature, fi.importance_mean)):
    if v > 0.001:
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0, max(fi.importance_mean) * 1.25)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig3_feature_importance.pdf", bbox_inches="tight")
plt.close()
fi.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
print(f"  Saved fig3_feature_importance")

# ============================================================
# FIGURE 4: Threshold sweep optimizing UNIQUE-class F1
# (the biologically meaningful class — Picard's failure mode)
# ============================================================
hgb_b = next(r for r in results2 if r["name"] == "HistGBM_balanced")
y_proba2 = hgb_b["y_proba"]
thresholds = np.linspace(0.01, 0.99, 99)
unique_p, unique_r, unique_f1 = [], [], []
dup_p, dup_r, dup_f1 = [], [], []
accs, bal_accs = [], []
for thr in thresholds:
    y_pred = (y_proba2 >= thr).astype(int)
    cm = confusion_matrix(y_te2, y_pred)
    if cm.shape != (2, 2):
        continue
    p0 = cm[0, 0] / max(cm[:, 0].sum(), 1)
    r0 = cm[0, 0] / max(cm[0, :].sum(), 1)
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 > 0 else 0
    p1 = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r1 = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 > 0 else 0
    unique_p.append(p0)
    unique_r.append(r0)
    unique_f1.append(f0)
    dup_p.append(p1)
    dup_r.append(r1)
    dup_f1.append(f1)
    accs.append((y_pred == y_te2).mean())
    bal_accs.append(balanced_accuracy_score(y_te2, y_pred))

best_idx_u = int(np.argmax(unique_f1))
best_idx_bal = int(np.argmax(bal_accs))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
ax.plot(thresholds, unique_p, label="unique Precision", lw=2, color="tab:green")
ax.plot(thresholds, unique_r, label="unique Recall", lw=2, color="tab:purple")
ax.plot(thresholds, unique_f1, label="unique F1", lw=2.5, color="black")
# Picard baseline (no threshold but fixed operating point)
cm_p = confusion_matrix(y_te2, p_te2)
pp0 = cm_p[0, 0] / max(cm_p[:, 0].sum(), 1)
rp0 = cm_p[0, 0] / max(cm_p[0, :].sum(), 1)
fp0 = 2 * pp0 * rp0 / (pp0 + rp0)
ax.axhline(pp0, color="tab:green", linestyle="--", alpha=0.4)
ax.axhline(rp0, color="tab:purple", linestyle="--", alpha=0.4)
ax.axhline(
    fp0, color="black", linestyle="--", alpha=0.4, label=f"Picard unique F1={fp0:.3f}"
)
ax.scatter(
    [thresholds[best_idx_u]],
    [unique_f1[best_idx_u]],
    marker="*",
    s=400,
    color="red",
    zorder=5,
    label=f"best F1_unique @ thr={thresholds[best_idx_u]:.2f}",
)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Metric on UNIQUE class")
ax.set_title(
    "Unique-class metrics vs threshold\n(can ML rescue Picard's false positives?)"
)
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax = axes[1]
ax.plot(thresholds, accs, label="Accuracy", lw=2, color="tab:blue")
ax.plot(thresholds, bal_accs, label="Balanced accuracy", lw=2.5, color="black")
ax.plot(thresholds, dup_f1, label="Dup F1", lw=2, color="tab:orange", alpha=0.7)
ax.scatter(
    [thresholds[best_idx_bal]],
    [bal_accs[best_idx_bal]],
    marker="*",
    s=400,
    color="red",
    zorder=5,
    label=f"best bal_acc @ thr={thresholds[best_idx_bal]:.2f}",
)
# Picard baselines
acc_p = (p_te2 == y_te2).mean()
bal_p = balanced_accuracy_score(y_te2, p_te2)
fp1 = (
    2
    * cm_p[1, 1]
    / cm_p[:, 1].sum()
    * cm_p[1, 1]
    / cm_p[1, :].sum()
    / (cm_p[1, 1] / cm_p[:, 1].sum() + cm_p[1, 1] / cm_p[1, :].sum())
)
ax.axhline(
    acc_p, color="tab:blue", linestyle="--", alpha=0.4, label=f"Picard acc={acc_p:.3f}"
)
ax.axhline(
    bal_p, color="black", linestyle="--", alpha=0.4, label=f"Picard bal_acc={bal_p:.3f}"
)
ax.axhline(
    fp1, color="tab:orange", linestyle="--", alpha=0.4, label=f"Picard dup F1={fp1:.3f}"
)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Metric")
ax.set_title("Overall metrics vs threshold")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.suptitle(
    "Threshold sweep — HistGBM balanced, Split 2 (no position leak)", fontsize=12
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig4_threshold_sweep.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved fig4_threshold_sweep")

# ============================================================
# Save comparison table
# ============================================================
rows = []
for split_name, results in [
    ("split1_random", results1),
    ("split2_groupby_pos", results2),
]:
    for r in results:
        rows.append(
            {
                "split": split_name,
                "model": r["name"],
                "accuracy": r["acc"],
                "balanced_acc": r["bal_acc"],
                "dup_precision": r["p1"],
                "dup_recall": r["r1"],
                "dup_f1": r["f1_1"],
                "unique_precision": r["p0"],
                "unique_recall": r["r0"],
                "unique_f1": r["f1_0"],
                "roc_auc": r["auc"],
                "pr_auc": r["ap"],
            }
        )
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print(f"\nSaved: {OUT_DIR}/model_comparison.csv")
print("\n=== Final comparison table ===")
print(cmp.round(4).to_string(index=False))

metrics = {
    "data_size": int(len(df)),
    "feature_dim": len(FEATURE_COLS),
    "umi_unique_n": int((df.umi_label == 0).sum()),
    "umi_dup_n": int((df.umi_label == 1).sum()),
    "picard_dup_n": int((df.picard_pred == 1).sum()),
    "picard_accuracy_vs_umi": float(
        (df.picard_pred.values == df.umi_label.values).mean()
    ),
    "picard_false_positive_rate_on_unique": float(
        1 - r["r0"] for r in [next(r for r in results1 if r["name"] == "Picard")]
    ).__next__()
    if False
    else None,
    "split2_results": [
        {
            "model": r["name"],
            "acc": r["acc"],
            "bal_acc": r["bal_acc"],
            "dup_f1": r["f1_1"],
            "unique_f1": r["f1_0"],
            "unique_recall": r["r0"],
            "auc": r["auc"],
        }
        for r in results2
    ],
    "best_unique_F1_threshold": float(thresholds[best_idx_u]),
    "best_unique_F1_value": float(unique_f1[best_idx_u]),
    "picard_unique_F1": float(fp0),
    "top5_features_by_importance": fi.sort_values("importance_mean", ascending=False)
    .head(5)
    .to_dict(orient="records"),
}
with open(f"{OUT_DIR}/metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
