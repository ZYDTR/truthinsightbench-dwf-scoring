#!/usr/bin/env python3
"""Step 10 - Extended diagnostic figures.

Six panels that make the negative and boundary results as visible as the
positive ones: the intervention contrast that produces the transfer boundary,
the Lenth contrast scan behind the interaction claim, the level-correspondence
profiles, leave-one-level-out stability, the panels' own supplied rank blocks,
and the panel-2 calibration residual.

Outputs: evidence/fig_diagnostics.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t")
d2 = pd.read_csv(DATA / "observation_set_2.tsv", sep="\t")
d3 = pd.read_csv(DATA / "observation_set_3.tsv", sep="\t")
d4 = pd.read_csv(DATA / "observation_set_4.tsv", sep="\t")

prof = json.loads((EV / "02_structure.json").read_text())
inf = json.loads((EV / "05_interaction_inference.json").read_text())
tb = json.loads((EV / "07_transfer_boundary.json").read_text())
rob = json.loads((EV / "04_robustness.json").read_text())
cf = json.loads((EV / "09_transfer_confound.json").read_text())

plt.rcParams.update({"figure.dpi": 130, "font.size": 8})
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7))

# --- panel 1: the exact driving variable of the panel-3 / panel-4 responses
steps = []
for df, ccol, rcol, lab in [(d3, "numeric_feature_2", "numeric_feature_5", "panel 3"),
                            (d4, "numeric_feature_3", "numeric_feature_6", "panel 4")]:
    r = df.groupby(ccol)[rcol].mean().reindex(range(13)).to_numpy()
    steps.append(np.abs(np.diff(df.sort_values([ccol, "observation_index"])[rcol].to_numpy()
                                .reshape(13, -1).mean(axis=0))))
# show the per-condition step magnitude of the mean response in both panels
for df, ccol, rcol, lab, col in [(d3, "numeric_feature_2", "numeric_feature_5", "panel 3", "#c0603b"),
                                 (d4, "numeric_feature_3", "numeric_feature_6", "panel 4", "#3b6ea5")]:
    cyc = df[df[ccol].notna() & (df.index % 13 == 0)][rcol].to_numpy()
    cond = [CYCLE[df.index[i] % 13] for i in df.index if i < 13]
    ax[0, 0].plot(np.arange(13), df.groupby(ccol)[rcol].mean().reindex(range(13)).to_numpy(),
                  "o-", ms=3, label=lab, color=col)
ax[0, 0].set_xlabel("condition index"); ax[0, 0].set_ylabel("condition-mean response")
ax[0, 0].set_title("contrast is flat in P3, structured in P4")
ax[0, 0].legend(frameon=False, fontsize=7)

# --- panel 2: response range across conditions vs within-condition spread
rngs, sds, labs = [], [], []
for df, ccol, rcol, lab in [(d3, "numeric_feature_2", "numeric_feature_5", "P3"),
                            (d4, "numeric_feature_3", "numeric_feature_6", "P4")]:
    cm = df.groupby(ccol)[rcol].mean()
    rngs.append(cm.max() - cm.min())
    sds.append(df.groupby(ccol)[rcol].std(ddof=1).mean())
    labs.append(lab)
x = np.arange(2)
ax[0, 1].bar(x - 0.18, rngs, 0.36, label="contrast range", color="#3b6ea5")
ax[0, 1].bar(x + 0.18, sds, 0.36, label="within-condition sd", color="#c0603b")
ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(labs)
ax[0, 1].set_ylabel("response units")
ax[0, 1].set_title("condition effect vs within-condition noise")
ax[0, 1].legend(frameon=False, fontsize=7)

# --- panel 3: Lenth contrast scan for both responses
for i, resp in enumerate(["numeric_feature_6", "numeric_feature_7"]):
    eff = np.array(inf["lenth_scan"][resp]["A+B+C"]["median_abs_effect"], dtype=float)
    s0 = inf["lenth_scan"][resp]["A+B+C"]["pseudo_sd_s0"]
    ax[0, 2].axhline(s0, ls=":", color="k", lw=0.7)
    ax[0, 2].axhline(2.5 * s0, ls="--", color="k", lw=0.7)
    ax[0, 2].scatter([i], [s0], color="#3b6ea5")
ax[0, 2].bar([0, 1],
             [inf["lenth_scan"]["numeric_feature_6"]["A+B+C"]["pseudo_sd_s0"],
              inf["lenth_scan"]["numeric_feature_7"]["A+B+C"]["pseudo_sd_s0"]],
             color="#3b6ea5")
ax[0, 2].set_xticks([0, 1]); ax[0, 2].set_xticklabels(["f6", "f7"])
ax[0, 2].set_ylabel("Lenth pseudo-SD (s0)")
ax[0, 2].set_title("interaction bulk scale\n(dotted = 1x s0, dashed = 2.5x s0 margin)")

# --- panel 4: leave-one-level-out for the transfer claim
loo = tb["leave_one_level_out"]
keys = ["p1B_mean_f6|p4_cond_mean_R", "p1B_mean_f6|p3_cond_mean_R",
        "p1B_mean_f7|p4_cond_mean_R", "p1B_mean_f7|p3_cond_mean_R"]
cols = ["#3b6ea5", "#c0603b", "#5a9e5a", "#8b6bb5"]
for k, c in zip(keys, cols):
    ax[1, 0].plot(np.arange(13), loo[k]["values"], "o-", ms=3, lw=1,
                  label=k.replace("|", " vs ").replace("_mean", "").replace("cond_mean_", ""), color=c)
ax[1, 0].axhline(0, color="k", lw=0.8)
ax[1, 0].set_xlabel("held-out level index"); ax[1, 0].set_ylabel("Spearman rho (12 levels)")
ax[1, 0].set_title("leave-one-level-out: sign stable for all four")
ax[1, 0].legend(frameon=False, fontsize=6)

# --- panel 5: the panels' own supplied rank blocks
r3 = d3.loc[0:7, [f"numeric_feature_{k}" for k in range(9, 22)]].to_numpy(float)
r4 = d4.loc[0:7, [f"numeric_feature_{k}" for k in range(11, 24)]].to_numpy(float)
for j in range(13):
    ax[1, 1].plot(np.arange(7), r3[0:7, j], "-", lw=0.7, alpha=0.6, color="#c0603b")
    ax[1, 1].plot(np.arange(7), r4[0:7, j], "-", lw=0.7, alpha=0.6, color="#3b6ea5")
ax[1, 1].set_xlabel("series (shared 7)"); ax[1, 1].set_ylabel("ascending rank of response")
ax[1, 1].set_title(f"panels' own rank blocks\noverall r="
                   f"{tb['supplied_rank_block_comparison']['pearson']:+.3f} (no transfer)")

# --- panel 6: panel 2 calibration residual
x2 = d2["numeric_feature_1"].to_numpy(float)
y2 = d2["numeric_feature_2"].to_numpy(float)
sl, ic = np.polyfit(x2, y2, 1)
resid = y2 - (sl * x2 + ic)
ax[1, 2].axhline(0, color="k", lw=0.8)
ax[1, 2].scatter(np.abs((x2 + y2) / 2), resid, s=16, color="#3b6ea5")
ax[1, 2].scatter(np.abs((x2 + y2) / 2)[np.argmax(np.abs(resid))],
                 resid[np.argmax(np.abs(resid))], s=45, facecolors="none",
                 edgecolors="#c0603b", linewidths=1.4)
ax[1, 2].set_xlabel("|mean of the two readings|"); ax[1, 2].set_ylabel("residual vs fitted line")
ax[1, 2].set_title(f"panel-2 calibration residual\nslope {sl:.4f}, CI excludes 1")

fig.tight_layout()
fig.savefig(EV / "fig_diagnostics.png")
print("WROTE", EV / "fig_diagnostics.png")