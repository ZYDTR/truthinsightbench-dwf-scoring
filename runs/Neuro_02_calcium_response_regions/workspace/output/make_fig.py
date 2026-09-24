#!/usr/bin/env python3
"""Generate supporting figure for the calcium response report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import json

BASE = "/data/user/conversations/conv-3b1fef04-0d44-4901-ba71-4d83eb3a2660/project"
df = pd.read_csv(BASE + "/data/cellular_calcium_response_summaries.tsv", sep="\t")
df["region"] = df["anatomical_region_id"]
df["batch"] = df["recording_batch_id"]
clean = df[df.region.isin(["region_1", "region_2"])].copy()

MEAS = ["mean_amplitude_dff0", "mean_time_to_peak_s", "mean_peak_width_s", "mean_auc_au"]

def fit(m, f, sub):
    return smf.ols(f"{m} ~ {f}", data=sub).fit()

# Condition model (clean region_1/2)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
cond = {}
for ax, m in zip(axes.flat, MEAS):
    mod = fit(m, "C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1')) + C(batch)", sub=clean)
    # baseline = reference; add its predicted mean
    base = mod.params["Intercept"]
    cond[m] = {"baseline": base,
               "neural_activity_blocked": base + mod.params["C(condition, Treatment(reference='baseline'))[T.neural_activity_blocked]"],
               "receptor_stimulated": base + mod.params["C(condition, Treatment(reference='baseline'))[T.receptor_stimulated]"]}
    cats = ["baseline", "neural_activity_blocked", "receptor_stimulated"]
    vals = [cond[m][c] for c in cats]
    bars = ax.bar(cats, vals, color=["#8EC5E0", "#F2C8A0", "#D98E8E"], edgecolor="black", linewidth=0.5)
    ax.set_title(f"{m.replace('mean_','').replace('_dff0',' ').replace('_s',' ').replace('_au',' ')}", fontsize=11)
    ax.set_ylabel(m, fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Condition effect on calcium-event measures (region_1/2 subset, OLS controlled for region + batch)",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(BASE + "/output/fig_condition_effect.png", dpi=150, bbox_inches="tight")
print("saved fig_condition_effect.png")

# Second figure: batch-age confound + AUC
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
ba = df.groupby("batch")["animal_age_days"].agg(["count", "min"])
ax1.bar(ba.index, ba["count"], color="#8EC5E0", edgecolor="black", linewidth=0.5)
ax1.set_title("Cells per recording batch\n(each batch = single age)", fontsize=11)
ax1.tick_params(axis="x", rotation=45)
ax1.set_xlabel("recording_batch_id")
ax1.set_ylabel("n cells")

# batch AUC
auc_b = df.groupby("batch")["mean_auc_au"].mean().sort_index()
ax2.bar(auc_b.index, auc_b, color="#D98E8E", edgecolor="black", linewidth=0.5)
ax2.set_title("Mean AUC per recording batch", fontsize=11)
ax2.tick_params(axis="x", rotation=45)
ax2.set_xlabel("recording_batch_id")
ax2.set_ylabel("mean_auc_au")
plt.tight_layout()
plt.savefig(BASE + "/output/fig_batch_age.png", dpi=150, bbox_inches="tight")
print("saved fig_batch_age.png")
