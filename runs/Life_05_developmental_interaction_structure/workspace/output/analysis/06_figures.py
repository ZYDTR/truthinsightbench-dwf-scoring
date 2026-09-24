#!/usr/bin/env python3
"""Step 06 - Figures summarising the stage-indexed interaction structure.

F1  stage trend under five estimators (naive vs composition-controlled)
F2  within-unit strength persistence vs the lability of the evidence tier
F3  trajectory taxonomy composition and threshold sensitivity
F4  leave-one-family-out transport: pooled gain vs the shuffled-stage control
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

units = pd.read_csv(OUT / "analysis_table.tsv", sep="\t")
Y = "interaction_strength_like_value"
trend = json.loads((OUT / "stage_trend.json").read_text())
coupling = json.loads((OUT / "evidence_coupling.json").read_text())
traj = json.loads((OUT / "trajectories_holdout.json").read_text())
decomp = json.loads((OUT / "structure_decomposition.json").read_text())
STAGES = np.arange(18)

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
labels = {
    "B0_record_weighted": ("record-weighted (naive)", "#888888", "--"),
    "B1_family_equal": ("family-equal", "#1f77b4", "-"),
    "B3_post_stratified": ("post-stratified", "#d62728", "-"),
    "B4_within_unit_centered": ("within-unit centred", "#2ca02c", "-"),
}
for key, (lab, colour, style) in labels.items():
    ax.plot(STAGES, trend["stage_curves"][key], style, color=colour, marker="o", ms=3.5, label=lab)
ax.set_xlabel("stage_index (ordered)")
ax.set_ylabel("mean interaction_strength_like_value")
ax.set_title("A. Stage trend survives composition controls")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.hist(units[Y], bins=60, color="#4c72b0", alpha=0.85)
ax.set_xlabel("interaction_strength_like_value")
ax.set_ylabel("records (deduplicated)")
ax.set_title(f"B. Strength distribution\nSD={units[Y].std(ddof=1):.2f}, range [{units[Y].min():.2f}, {units[Y].max():.2f}]")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "F1_stage_trend_and_distribution.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
pval_curve = units.groupby("stage_index")[Y].mean()
um = units.groupby("unit_id")[Y].transform("mean")
centred = units[Y] - um
axis_floor = (units["association_pvalue"] == 0.0009).groupby(units.stage_index).mean()
ax.plot(STAGES, axis_floor.values * 100, color="#c44e52", marker="o", ms=3.5)
ax.set_xlabel("stage_index (ordered)")
ax.set_ylabel("% records at p floor 0.0009")
ax.set_title("A. Evidence floor prevalence is non-monotone")
ax.grid(alpha=0.25)
ax.set_ylim(60, 100)

ax = axes[1]
lv = coupling["mean_strength_by_p_level"]
xs = [float(k) for k in lv]
raw = [lv[k]["raw_mean"] for k in lv]
within = [lv[k]["within_unit_mean"] for k in lv]
ax.plot(xs, raw, "o-", color="#4c72b0", label="raw mean")
ax.plot(xs, within, "s-", color="#2ca02c", label="within-unit centred")
ax.set_xscale("log")
ax.set_xlabel("association_pvalue (log scale, 6 discrete tiers)")
ax.set_ylabel("mean strength")
ax.set_title("B. Non-floor tiers sit below the floor tier")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "F2_evidence_tier.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 3
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
cc = traj["partA_taxonomy"]["class_counts"]
order = ["persistent_level", "transient_sign_reversing", "unresolved_flat", "emerging", "declining"]
vals = [cc.get(k, 0) for k in order]
ax.barh(range(len(order)), vals, color=["#2ca02c", "#c44e52", "#999999", "#1f77b4", "#ff7f0e"])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([o.replace("_", " ") for o in order], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("units (of 579 with >=3 stages)")
ax.set_title("A. Trajectory taxonomy")
for i, v in enumerate(vals):
    ax.text(v + 3, i, str(v), va="center", fontsize=8)
ax.grid(alpha=0.25, axis="x")

ax = axes[1]
grid = pd.DataFrame(traj["partA_threshold_sensitivity"])
for cls, colour in zip(order, ["#2ca02c", "#c44e52", "#999999", "#1f77b4", "#ff7f0e"]):
    ax.plot(
        grid.slope_threshold.astype(str),
        grid[cls],
        "o-",
        color=colour,
        label=cls.replace("_", " "),
        alpha=0.85,
        ms=4,
    )
ax.set_xlabel("slope threshold (flat-SD threshold varied over 0.30/0.50/0.75)")
ax.set_ylabel("units assigned")
ax.set_title("B. Threshold sensitivity")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "F3_trajectory_taxonomy.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 4
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
pooled = decomp["partB_pooled"]
ctrl = decomp["partB_stage_shuffle_control"]
bars = [pooled["mae_null_unit_mean"], pooled["mae_trend"], pooled["mae_null_unit_mean"] + ctrl["mean_improvement_under_shuffled_stage"]]
ax.bar(["unit mean\n(null)", "unit mean\n+ stage trend", "unit mean +\nshuffled stage"], bars,
       color=["#888888", "#2ca02c", "#c44e52"])
ax.set_ylabel("held-out MAE")
ax.set_title(f"A. Family-holdout MAE\nimprovement {pooled['improvement_pct']:.1f}%, permutation p < {1/ctrl['n_perm']:.4f}")
for i, v in enumerate(bars):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
ax.grid(alpha=0.25, axis="y")

ax = axes[1]
# Per-family gains come from the repaired leave-one-family-out run in
# holdout_replication.json, which is the authoritative implementation.  The
# earlier family_holdout_annotated.tsv used mean-of-record-gains per family
# (gain 103/111) rather than (MAE_null - MAE_trend) per family (102/112).
hrep = json.loads((OUT / "holdout_replication.json").read_text())
pf = pd.DataFrame(hrep["leave_one_family_out_per_family"]).rename(
    columns={"group": "family", "mae_improvement": "gain"}
)
ax.scatter(pf.n_records, pf.gain, s=18, alpha=0.7, color="#4c72b0", edgecolors="none")
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("held-out family size (records, log scale)")
ax.set_ylabel("per-family MAE gain")
ax.set_title(f"B. Transport is heterogeneous\n{(pf.gain > 0).sum()} of {len(pf)} improve, {(pf.gain < 0).sum()} worsen")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "F4_holdout_transport.png", dpi=170)
plt.close(fig)

print("wrote:")
for p in sorted(FIG.glob("*.png")):
    print(" ", p.relative_to(ROOT), p.stat().st_size, "bytes")