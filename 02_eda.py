"""
EDA: feature distributions split by UMI label.
Output: results_ml/eda_report.txt + results_ml/eda_*.png
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FEATURES_PATH = "/home/eagle/try3/bench/results_ml/features.parquet"
OUT_DIR = "/home/eagle/try3/bench/results_ml"

df = pd.read_parquet(FEATURES_PATH)
print(f"Loaded: {df.shape}")
print(
    f"Label balance: 0 (unique) = {(df.umi_label == 0).sum()}, 1 (dup) = {(df.umi_label == 1).sum()}"
)
print(
    f"Class imbalance ratio: {(df.umi_label == 1).sum() / (df.umi_label == 0).sum():.2f}"
)

# Features to inspect
numeric_feats = [
    "start",
    "mapq",
    "read_len",
    "n_M",
    "n_softclip",
    "gc_content",
    "mean_q",
    "std_q",
    "min_q",
    "as_score",
    "nm_mismatch",
    "density_50bp",
    "density_500bp",
    "pos_pct",
]

# Also add: number of reads at same start position (key signal)
pos_counts = df["start"].value_counts()
df["n_reads_same_pos"] = df["start"].map(pos_counts)
numeric_feats.append("n_reads_same_pos")

# Save enriched df
df.to_parquet(FEATURES_PATH, index=False)

print("\n=== EDA: feature stats by label ===")
print(
    f"{'feature':<22} {'unique_mean':>12} {'dup_mean':>12} {'effect_size':>12} {'p_value':>12}"
)
print("-" * 75)

eda_results = []
for f in numeric_feats:
    g0 = df.loc[df.umi_label == 0, f].dropna()
    g1 = df.loc[df.umi_label == 1, f].dropna()
    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((g0.std() ** 2) * (len(g0) - 1) + (g1.std() ** 2) * (len(g1) - 1))
        / (len(g0) + len(g1) - 2)
    )
    d = (g1.mean() - g0.mean()) / pooled_std if pooled_std > 0 else 0
    # Mann-Whitney U
    try:
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    except Exception:
        p = np.nan
    eda_results.append(
        {
            "feature": f,
            "unique_mean": g0.mean(),
            "dup_mean": g1.mean(),
            "cohen_d": d,
            "p_value": p,
        }
    )
    print(f"{f:<22} {g0.mean():>12.3f} {g1.mean():>12.3f} {d:>12.3f} {p:>12.2e}")

eda_df = pd.DataFrame(eda_results).sort_values("cohen_d", key=abs, ascending=False)
eda_df.to_csv(f"{OUT_DIR}/eda_feature_stats.csv", index=False)

# Plot top discriminating features
top_feats = eda_df.head(6)["feature"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flat, top_feats):
    for label, color, name in [
        (0, "tab:blue", "unique (UMI kept)"),
        (1, "tab:red", "dup (UMI removed)"),
    ]:
        vals = df.loc[df.umi_label == label, feat].dropna()
        # subsample for plot if huge
        if len(vals) > 2000:
            vals = vals.sample(2000, random_state=0)
        ax.hist(vals, bins=40, alpha=0.5, color=color, label=name, density=True)
    ax.set_title(
        f"{feat}\n(d={eda_df.loc[eda_df.feature == feat, 'cohen_d'].values[0]:.2f})",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.set_xlabel(feat, fontsize=9)
plt.suptitle("Feature distributions: UMI-unique vs UMI-duplicate", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_distributions.png", dpi=120, bbox_inches="tight")
print(f"\nSaved: {OUT_DIR}/eda_distributions.png")
print(f"Saved: {OUT_DIR}/eda_feature_stats.csv")

# Strand breakdown
print("\n=== Strand × label crosstab ===")
ct = pd.crosstab(df["strand"], df["umi_label"], normalize="index")
ct.columns = ["unique", "dup"]
ct.index = ["forward", "reverse"]
print(ct)
