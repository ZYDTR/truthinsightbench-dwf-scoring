"""WS6 figure: leave-one-molecule-out influence + alternative-loss comparison.

Run:  python3 output/analysis/ws6_figure.py
Reads output/results/ws6_robustness.json (written by ws6_robustness.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
sys.path.insert(0, str(ROOT / "output" / "analysis"))
import common  # noqa: E402

R = json.loads((ROOT / "output" / "results" / "ws6_robustness.json").read_text())
CH = ["quantum", "learned", "reference"]
COL = {"quantum": "#1f77b4", "learned": "#2ca02c", "reference": "#d62728"}

df = common.build_analysis_table(verbose=False)
ae = {c: df[f"{c}_ae"].to_numpy(float) for c in CH}
mol = df.parent_structure_smiles.to_numpy()
mols, counts = np.unique(mol, return_counts=True)
N = len(df)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ---- (a) leave-one-molecule-out influence --------------------------------
ax = axes[0, 0]
# influence = dMAE when molecule i is removed. Full 419 x 3 from the artifact
# summary stats plus a direct recomputation for the plotted values.
for c in CH:
    s = R["check1_leave_one_molecule_out"]["channels"][c]
    ax.barh(c, s["max_abs_delta_mae"], color=COL[c], alpha=0.85,
            label=f"{c} (max |dMAE|={s['max_abs_delta_mae']:.3f})")
ax.set_xlabel("max |change in MAE| when one molecule is removed (kcal/mol)")
ax.set_title("(a) Leave-one-molecule-out influence\n419 molecules, 543 observations",
             fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3, axis="x")
for c in CH:
    s = R["check1_leave_one_molecule_out"]["channels"][c]
    ax.text(s["max_abs_delta_mae"], c, f"  {s['max_abs_delta_mae']:.3f}", va="center",
            fontsize=9)

# ---- (b) rank stability: MAE under every leave-one-molecule-out ----------
ax = axes[0, 1]
sums = {c: float(ae[c].sum()) for c in CH}
grp = {c: np.array([ae[c][mol == m].sum() for m in mols]) for c in CH}
for c in CH:
    loo = (sums[c] - grp[c]) / (N - counts)
    ax.hist(loo, bins=40, color=COL[c], alpha=0.55, label=f"{c} ({loo.min():.3f}-{loo.max():.3f})")
ax.set_xlabel("MAE with one molecule removed (kcal/mol)")
ax.set_ylabel("number of leave-out subsets")
ax.set_title("(b) MAE across all 419 leave-one-molecule-out fits\n"
             "ranking quantum<learned<reference changes in 0/419 subsets", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- (c) alternative loss definitions ------------------------------------
ax = axes[1, 0]
metrics = ["mae", "rmse", "median_ae", "p90_ae", "frac_within_2", "frac_within_5"]
labels = ["MAE", "RMSE", "median AE", "90th pct AE", "frac <=2", "frac <=5"]
md = R["check2_alternative_error_definitions"]["metrics"]
x = np.arange(len(metrics))
w = 0.26
for i, c in enumerate(CH):
    vals = [md[m]["per_channel"][c] for m in metrics]
    ax.bar(x + (i - 1) * w, vals, w, color=COL[c], label=c)
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("value (log scale)")
ax.set_title("(c) Channel comparison under six error definitions\n"
             "the ordering quantum<learned<reference holds in all six", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

# ---- (d) headline paired difference with row vs cluster bootstrap --------
ax = axes[1, 1]
hp = R["headline_paired_tests"]
cb = R["check1_leave_one_molecule_out"]["cluster_bootstrap_over_molecules"]
names, means, los, his, clos, chis = [], [], [], [], [], []
for k in ["quantum_vs_learned", "quantum_vs_reference", "learned_vs_reference"]:
    names.append(k.replace("_vs_", "\nvs "))
    means.append(hp[k]["mean_abs_err_diff"])
    los.append(hp[k]["boot95_ci"][0])
    his.append(hp[k]["boot95_ci"][1])
    clos.append(cb[k]["cluster_boot95_ci"][0])
    chis.append(cb[k]["cluster_boot95_ci"][1])
y = np.arange(len(names))
for i in range(len(names)):
    ax.plot([clos[i], chis[i]], [y[i] + 0.13] * 2, color="black", lw=3, alpha=0.45,
            label="cluster bootstrap (molecules)" if i == 0 else None)
    ax.plot([los[i], his[i]], [y[i] - 0.13] * 2, color="#ff7f0e", lw=3, alpha=0.9,
            label="row bootstrap (observations)" if i == 0 else None)
    ax.plot(means[i], y[i], "o", color="black", ms=6)
    ax.text(means[i], y[i] + 0.3, f"{means[i]:+.3f}", ha="center", fontsize=9)
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.set_ylim(-0.7, len(names) - 0.3)
ax.set_xlabel("mean paired |error| difference (kcal/mol)")
ax.set_title("(d) Headline paired differences, 95% CIs\n"
             "row bootstrap (orange) vs molecule-cluster bootstrap (grey)", fontsize=11)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3, axis="x")

fig.suptitle("WS6 robustness suite: leave-one-molecule-out influence and alternative-loss comparison",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT / "output" / "figures" / "ws6_robustness.png"
fig.savefig(out, dpi=150)
print(f"[ws6] wrote {out}")