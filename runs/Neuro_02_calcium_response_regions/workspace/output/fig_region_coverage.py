#!/usr/bin/env python3
"""Supporting figure for Finding 2: region x condition coverage matrix and
within-receptor-stimulated AUC by region, illustrating the region x condition
coverage confound.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE = "/data/user/conversations/conv-3b1fef04-0d44-4901-ba71-4d83eb3a2660/project"
df = pd.read_csv(BASE + "/data/cellular_calcium_response_summaries.tsv", sep="\t")
df["region"] = df["anatomical_region_id"]
OUT = Path(BASE) / "output"

regions = ["region_1", "region_2", "region_3", "region_4"]
conds = ["baseline", "neural_activity_blocked", "receptor_stimulated"]
cov = df.groupby(["region", "condition"]).size().unstack(fill_value=0)
cov = cov.reindex(index=regions, columns=conds)
cov = cov.fillna(0)

# within receptor_stimulated AUC by region
rs = df[df.condition == "receptor_stimulated"]
auc_by_region = rs.groupby("region")["mean_auc_au"].agg(["mean", "count"])
auc_by_region = auc_by_region.reindex(regions)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

# Heatmap of coverage
im = ax1.imshow(cov.values, cmap="Blues", vmin=0, vmax=float(cov.values.max()))
ax1.set_xticks(np.arange(len(conds))); ax1.set_xticklabels(conds, rotation=30, ha="right")
ax1.set_yticks(np.arange(len(regions))); ax1.set_yticklabels(regions)
for i in range(len(regions)):
    for j in range(len(conds)):
        v = int(cov.values[i, j])
        ax1.text(j, i, str(v), ha="center", va="center",
                 color="black" if v < float(cov.values.max()) * 0.6 else "white", fontsize=10)
ax1.set_xlabel("condition"); ax1.set_ylabel("region")
ax1.set_title("Cell counts by region x condition\n(zeros = no cells recorded)", fontsize=11)
fig.colorbar(im, ax=ax1, label="n cells")

# within receptor-stimulated AUC by region
labels = [r for r in regions]
vals = auc_by_region["mean"].dropna().tolist()
present = [r for r in regions if r in auc_by_region.index and not np.isnan(auc_by_region.loc[r, "mean"])]
vals = [auc_by_region.loc[r, "mean"] for r in present]
colors = ["#D98E8E" if r == "region_3" else "#8EC5E0" for r in present]
ax2.bar(present, vals, color=colors, edgecolor="black", linewidth=0.5)
for r, v in zip(present, vals):
    ax2.text(r, v + 5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("mean_auc_au")
ax2.set_title("Mean AUC under receptor_stimulated\nby region (region_3 shown in red)", fontsize=11)
ax2.tick_params(axis="x", rotation=20)
plt.suptitle("Region x condition coverage confound (Finding 2 boundary)", fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT / "fig_region_coverage.png", dpi=150, bbox_inches="tight")
print("wrote fig_region_coverage.png")
