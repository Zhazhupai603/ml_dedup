"""
Whole-genome ML training.

- HistGradientBoosting (fast on millions of rows)
- Two splits:
  A) Random 4:1 (may leak position)
  B) Group-by-position 4:1 (no leak, stricter)
- Baseline: Picard mark_only
- Output: results_wg/{model_comparison.csv, roc_curves.png, pr_curves.png,
                     confusion_matrices.png, feature_importance.png, decision_threshold.png}
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
    f1_score,
)
import time, os, warnings, json

warnings.filterwarnings("ignore")

FEATURES_PATH = "/home/eagle/try3/bench/results_wg/features.parquet"
OUT_DIR = "/home/eagle/try3/bench/results_wg/figures_ml"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
print(f"Loading features ...")
df = pd.read_parquet(FEATURES_PATH)
print(f"  Shape: {df.shape}, dup rate: {df.umi_label.mean():.3f}")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Drop non-feature columns
DROP = ["qname", "umi_label", "picard_pred", "chrom"]
FEATURE_COLS = [c for c in df.columns if c not in DROP]
print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].astype(np.float32)
y = df["umi_label"].values.astype(np.int8)
picard = df["picard_pred"].values.astype(np.int8)
groups = df["start"].values  # for group split

# ============================================================
# SPLIT 1: Random 4:1
# ============================================================
print("\n" + "=" * 70)
print("SPLIT 1: Random 4:1 (read-level)")
print("=" * 70)
X_train1, X_test1, y_train1, y_test1, p_train1, p_test1 = train_test_split(
    X, y, picard, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train1):,}  Test: {len(X_test1):,}")

# ============================================================
# SPLIT 2: Group-by-position 4:1 (no position leak)
# ============================================================
print("\n" + "=" * 70)
print("SPLIT 2: Group-by-position 4:1 (strict, no position leak)")
print("=" * 70)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))
X_train2, X_test2 = X.iloc[tr_idx], X.iloc[te_idx]
y_train2, y_test2 = y[tr_idx], y[te_idx]
p_train2, p_test2 = picard[tr_idx], picard[te_idx]
overlap = len(set(X_train2["start"]) & set(X_test2["start"]))
print(f"Train: {len(X_train2):,}  Test: {len(X_test2):,}  Position overlap: {overlap}")

models = {
    "LogReg": Pipeline(
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
    "HistGBM": HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=42,
    ),
}


def eval_model(name, model, X_tr, y_tr, X_te, y_te):
    t = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t
    t = time.time()
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    pred_time = time.time() - t
    cm = confusion_matrix(y_te, y_pred)
    acc = (y_pred == y_te).mean()
    prec = cm[1, 1] / max(cm[:, 1].sum(), 1)
    rec = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    auc = roc_auc_score(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    print(
        f"  {name}: acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  AP={ap:.4f}  (train {train_time:.1f}s, pred {pred_time:.1f}s)"
    )
    return {
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "cm": cm,
        "model": model,
    }


def eval_picard(y_te, p_te):
    cm = confusion_matrix(y_te, p_te)
    acc = (p_te == y_te).mean()
    prec = cm[1, 1] / max(cm[:, 1].sum(), 1)
    rec = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"  Picard: acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {
        "name": "Picard",
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": None,
        "ap": None,
        "y_proba": None,
        "y_pred": p_te,
        "cm": cm,
        "model": None,
    }


print("\n--- Split 1 results ---")
results1 = [eval_picard(y_test1, p_test1)]
for name, m in models.items():
    results1.append(eval_model(name, m, X_train1, y_train1, X_test1, y_test1))

print("\n--- Split 2 results ---")
results2 = [eval_picard(y_test2, p_test2)]
for name, m in models.items():
    results2.append(eval_model(name, m, X_train2, y_train2, X_test2, y_test2))

# ============================================================
# FIGURE 1: ROC + PR curves
# ============================================================
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for ax_row, results, y_te, p_te, title_prefix in [
    (axes[0], results1, y_test1, p_test1, "Split 1 (random 4:1)"),
    (axes[1], results2, y_test2, p_test2, "Split 2 (group-by-position 4:1)"),
]:
    # ROC
    ax = ax_row[0]
    for r in results:
        if r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_te, r["y_proba"])
            ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})", lw=2)
    # Picard point
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
        label=f"Picard (FPR={fpr_p:.3f}, TPR={tpr_p:.3f})",
        linewidths=3,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {title_prefix}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # PR
    ax = ax_row[1]
    baseline_prec = (y_te == 1).mean()
    for r in results:
        if r["y_proba"] is not None:
            p, r_, _ = precision_recall_curve(y_te, r["y_proba"])
            ax.plot(r_, p, label=f"{r['name']} (AP={r['ap']:.3f})", lw=2)
    ax.scatter(
        [cm[1, 1] / cm[1, :].sum()],
        [cm[1, 1] / cm[:, 1].sum()],
        marker="x",
        s=200,
        color="black",
        zorder=5,
        label=f"Picard (P={cm[1, 1] / cm[:, 1].sum():.3f}, R={cm[1, 1] / cm[1, :].sum():.3f})",
        linewidths=3,
    )
    ax.axhline(
        baseline_prec,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"baseline ({baseline_prec:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR — {title_prefix}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig1_roc_pr_curves.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: fig1_roc_pr_curves.png/pdf")

# ============================================================
# FIGURE 2: Confusion matrices (4 panels)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for ax_row, results, title_prefix in [
    (axes[0], results1, "Split 1"),
    (axes[1], results2, "Split 2"),
]:
    for ax, r in zip(
        ax_row, [res for res in results if res["name"] in ("Picard", "HistGBM")]
    ):
        cm = r["cm"]
        cm_pct = cm.astype(float) / cm.sum() * 100
        im = ax.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=cm_pct.max())
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm_pct[i, j] > cm_pct.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred: unique", "pred: dup"])
        ax.set_yticklabels(["true: unique", "true: dup"])
        ax.set_title(f"{r['name']} — {title_prefix}\nF1={r['f1']:.3f}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig2_confusion_matrices.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: fig2_confusion_matrices.png/pdf")

# ============================================================
# FIGURE 3: Feature importance (HistGBM, split 1)
# ============================================================
hgb = next(r for r in results1 if r["name"] == "HistGBM")["model"]
importances = hgb.feature_importances_
fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
fi = fi.sort_values("importance", ascending=True)
# add n_reads_at_pos highlight
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["tab:red" if f == "n_reads_at_pos" else "tab:blue" for f in fi.feature]
ax.barh(fi.feature, fi.importance, color=colors)
ax.set_xlabel("Importance (HistGradientBoosting)")
ax.set_title("Feature importance — what signals distinguish PCR duplicates?")
for i, (f, v) in enumerate(zip(fi.feature, fi.importance)):
    if v > 0.001:
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, max(fi.importance) * 1.15)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_feature_importance.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig3_feature_importance.pdf", bbox_inches="tight")
plt.close()
fi_sorted = fi.sort_values("importance", ascending=False)
fi_sorted.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
print(f"  Saved: fig3_feature_importance.png/pdf")

# ============================================================
# FIGURE 4: Decision threshold sweep (find optimal F1 / precision-recall tradeoff)
# ============================================================
print("\n  Generating decision threshold sweep ...")
hgb_res2 = next(r for r in results2 if r["name"] == "HistGBM")
y_proba2 = hgb_res2["y_proba"]
thresholds = np.linspace(0.05, 0.95, 91)
precisions = []
recalls = []
f1s = []
accs = []
for thr in thresholds:
    y_pred = (y_proba2 >= thr).astype(int)
    cm = confusion_matrix(y_test2, y_pred)
    if cm.shape != (2, 2):
        continue
    p = cm[1, 1] / max(cm[:, 1].sum(), 1)
    r = cm[1, 1] / max(cm[1, :].sum(), 1)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    acc = (y_pred == y_test2).mean()
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)
    accs.append(acc)

best_idx = int(np.argmax(f1s))
print(f"  Optimal threshold: {thresholds[best_idx]:.2f}  -> F1={f1s[best_idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, precisions, label="Precision", lw=2)
ax.plot(thresholds, recalls, label="Recall", lw=2)
ax.plot(thresholds, f1s, label="F1", lw=2.5, color="black")
ax.plot(thresholds, accs, label="Accuracy", lw=2, alpha=0.6)
# Picard baseline points (split 2)
cm_p = confusion_matrix(y_test2, p_test2)
p_p = cm_p[1, 1] / max(cm_p[:, 1].sum(), 1)
r_p = cm_p[1, 1] / max(cm_p[1, :].sum(), 1)
f1_p = 2 * p_p * r_p / (p_p + r_p)
ax.axhline(p_p, color="C0", linestyle="--", alpha=0.4)
ax.axhline(r_p, color="C1", linestyle="--", alpha=0.4)
ax.axhline(
    f1_p, color="black", linestyle="--", alpha=0.4, label=f"Picard F1={f1_p:.3f}"
)
ax.axvline(
    thresholds[best_idx],
    color="gray",
    linestyle=":",
    alpha=0.5,
    label=f"best F1 @ thr={thresholds[best_idx]:.2f}",
)
ax.scatter(
    [thresholds[best_idx]], [f1s[best_idx]], marker="*", s=300, color="red", zorder=5
)
ax.set_xlabel("Decision threshold (probability)")
ax.set_ylabel("Metric value")
ax.set_title("Threshold sweep — HistGBM Split 2 vs Picard baseline")
ax.legend(loc="lower center", ncol=3, fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig4_threshold_sweep.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: fig4_threshold_sweep.png/pdf")

# ============================================================
# Save model comparison CSV
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
                "precision": r["prec"],
                "recall": r["rec"],
                "f1": r["f1"],
                "roc_auc": r["auc"],
                "pr_auc": r["ap"],
            }
        )
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
print(f"\nSaved model comparison: {OUT_DIR}/model_comparison.csv")
print("\n=== Final comparison ===")
print(cmp.to_string(index=False))

# Save metrics JSON for CONCLUSIONS doc
metrics = {
    "split1_random": [
        {
            "model": r["name"],
            "acc": r["acc"],
            "prec": r["prec"],
            "rec": r["rec"],
            "f1": r["f1"],
            "auc": r["auc"],
            "ap": r["ap"],
        }
        for r in results1
    ],
    "split2_groupby_pos": [
        {
            "model": r["name"],
            "acc": r["acc"],
            "prec": r["prec"],
            "rec": r["rec"],
            "f1": r["f1"],
            "auc": r["auc"],
            "ap": r["ap"],
        }
        for r in results2
    ],
    "best_threshold_split2": float(thresholds[best_idx]),
    "best_f1_split2": float(f1s[best_idx]),
    "picard_f1_split2": float(f1_p),
    "feature_importance_top5": fi_sorted.head(5).to_dict(orient="records"),
    "feature_importance_n_reads_at_pos": float(
        fi[fi.feature == "n_reads_at_pos"]["importance"].values[0]
    ),
    "data_size": int(len(df)),
    "feature_dim": len(FEATURE_COLS),
}
with open(f"{OUT_DIR}/metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
