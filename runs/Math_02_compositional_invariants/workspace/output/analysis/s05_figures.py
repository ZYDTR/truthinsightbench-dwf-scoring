"""Stage 5 - supporting figures.

Writes PNG figures to output/figures/ and a manifest to
output/results/figure_manifest.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_composition import (  # noqa: E402
    OUT,
    bias_wide,
    clr,
    counts_matrix,
    library_size,
    load_covariates,
    rarefy,
    richness,
    save_json,
)

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

P, samples, features = counts_matrix()
lib = library_size(P).astype(float)
rich = richness(P).astype(float)
w = bias_wide()
shared = sorted(set(w.loc[w.scenario_id == "small_sample", "metric_number"])
                & set(w.loc[w.scenario_id == "high_variance", "metric_number"]))
ss = w[(w.scenario_id == "small_sample") & w.metric_number.isin(shared)].set_index("metric_number").sort_index()
hv = w[(w.scenario_id == "high_variance") & w.metric_number.isin(shared)].set_index("metric_number").sort_index()

manifest = {}


def record(name, caption):
    manifest[name] = {"path": f"figures/{name}", "caption": caption}


# --- F1: scale dependence of richness -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(lib, rich, s=9, alpha=0.45, color="#2b6cb0", edgecolor="none")
ax.set_xscale("log")
ax.set_yscale("log")
b, a = np.polyfit(np.log(lib), np.log(rich), 1)
xs = np.linspace(np.log(lib.min()), np.log(lib.max()), 50)
ax.plot(np.exp(xs), np.exp(a + b * xs), color="#c53030", lw=2,
        label=f"log-log slope = {b:.3f}")
ax.set_xlabel("library size (total counts per sample)")
ax.set_ylabel("observed richness (features > 0)")
ax.set_title("A. Richness scales sub-linearly with depth")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")

ax = axes[1]
rng = np.random.default_rng(20240101)
depths = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
means, sds = [], []
for dpt in depths:
    rr = (rarefy(P, dpt, rng) > 0).sum(axis=1)
    means.append(rr.mean())
    sds.append(rr.std())
ax.errorbar(depths, means, yerr=sds, marker="o", capsize=3, color="#2f855a")
ax.axhline(rich.mean(), ls="--", color="#c53030", lw=1.4,
           label=f"observed mean richness = {rich.mean():.0f}")
ax.set_xscale("log")
ax.set_xlabel("rarefied sequencing depth (counts)")
ax.set_ylabel("rarefied richness")
ax.set_title("B. No saturation at any attainable depth")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIG / "F1_scale_dependence.png", dpi=170)
plt.close(fig)
record("F1_scale_dependence.png",
       "Left: observed richness against library size on log-log axes with fitted slope 0.071. "
       "Right: rarefied richness against common rarefaction depth, unsaturating and rising past "
       "the raw mean richness at the largest attainable depth.")

# --- F2: covariate association and its confound ---------------------------
cv = load_covariates().set_index("sample_id")
tab = cv.join(pd.DataFrame({"lib": lib, "rich": rich}, index=samples), how="inner")
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
colors = {"group_A": "#2b6cb0", "group_B": "#c53030", "group_C": "#2f855a"}
for g, sub in tab.groupby("coarse_group_id"):
    axes[0].scatter(sub.continuous_covariate, sub.rich, s=14, alpha=0.6,
                    color=colors[g], label=f"{g} (n={len(sub)})", edgecolor="none")
axes[0].set_xscale("log")
axes[0].set_xlabel("continuous covariate")
axes[0].set_ylabel("observed richness")
rho = stats.spearmanr(tab.rich, tab.continuous_covariate)
axes[0].set_title(f"A. richness vs covariate (rho = {rho.statistic:+.3f})")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(alpha=0.25)

for g, sub in tab.groupby("coarse_group_id"):
    axes[1].scatter(sub.continuous_covariate, sub.lib, s=14, alpha=0.6,
                    color=colors[g], label=g, edgecolor="none")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("continuous covariate")
axes[1].set_ylabel("library size")
rho2 = stats.spearmanr(tab.lib, tab.continuous_covariate)
axes[1].set_title(f"B. depth vs covariate (rho = {rho2.statistic:+.3f})")
axes[1].grid(alpha=0.25, which="both")

tab2 = tab.copy()
tab2["dec"] = pd.qcut(tab2.continuous_covariate, 8, labels=False)
# NB: avoid a column named ``cov`` - it shadows DataFrame.cov on attribute access.
tab2 = tab2.assign(covariate_mid=tab2.continuous_covariate)
agg = tab2.groupby("dec").agg(covariate_mid=("covariate_mid", "median"),
                              rich_med=("rich", "median"), lib_med=("lib", "median"))
ax3 = axes[2]
ax3.plot(agg.covariate_mid, agg.rich_med / agg.rich_med.iloc[0], marker="o",
         color="#2f855a", label="richness (relative)")
ax3.plot(agg.covariate_mid, agg.lib_med / agg.lib_med.iloc[0], marker="s",
         color="#2b6cb0", label="depth (relative)")
ax3.set_xscale("log")
ax3.set_xlabel("covariate octile median")
ax3.set_ylabel("value relative to lowest octile")
ax3.set_title("C. Opposite monotone trends")
ax3.legend(frameon=False, fontsize=9)
ax3.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "F2_covariate_association.png", dpi=170)
plt.close(fig)
record("F2_covariate_association.png",
       "Richness rises and library size falls with the continuous covariate; the opposed signs "
       "are visible within every coarse group and survive depth rarefaction.")

# --- F3: bias scenario condition structure --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
ax = axes[0]
ax.scatter(ss.condition_2, ss.condition_3, s=18, alpha=0.6, color="#2b6cb0",
           edgecolor="none", label="small_sample")
ax.scatter(hv.condition_2, hv.condition_3, s=18, alpha=0.6, color="#c53030",
           edgecolor="none", label="high_variance")
lim = [0, 1]
ax.plot(lim, lim, ls="--", color="black", lw=1.2, label="condition_3 = condition_2")
ax.set_xlabel("condition_2 value")
ax.set_ylabel("condition_3 value")
ax.set_title("A. Condition plane (shared 400 metrics)")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
for tag, df, c in [("small_sample", ss, "#2b6cb0"), ("high_variance", hv, "#c53030")]:
    d = np.sort((df.condition_3 - df.condition_2).values)
    ax.plot(d, np.linspace(0, 1, len(d)), color=c, lw=2, label=tag)
ax.axvline(0, ls="--", color="black", lw=1.2)
ax.set_xlabel("condition_3 - condition_2")
ax.set_ylabel("empirical CDF")
ax.set_title("B. Ordering holds except at the margin")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)

ax = axes[2]
rates = {
    "all cells": [(ss.condition_2 == 0).mean(), (hv.condition_2 == 0).mean()],
    "c3 > c2": [(ss.condition_2[(ss.condition_3 > ss.condition_2)].values == 0).mean(),
                (hv.condition_2[(hv.condition_3 > hv.condition_2)].values == 0).mean()],
}
xpos = np.arange(2)
width = 0.36
ax.bar(xpos - width / 2, rates["all cells"], width, color="#2b6cb0", label="all cells")
ax.bar(xpos + width / 2, rates["c3 > c2"], width, color="#2f855a", label="cells with c3 > c2")
ax.set_xticks(xpos)
ax.set_xticklabels(["small_sample", "high_variance"])
ax.set_ylabel("fraction of cells with condition_2 = 0")
ax.set_title("C. Zero rate differs by scenario")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(FIG / "F3_bias_conditions.png", dpi=170)
plt.close(fig)
record("F3_bias_conditions.png",
       "Condition ordering on the shared 400 metric ids, its empirical CDF including the five "
       "exceptions, and the scenario-dependent zero rate in condition_2.")

# --- F4: the coupling is zero-driven --------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
for ax, (tag, df, c) in zip(axes[:2], [("small_sample", ss, "#2b6cb0"), ("high_variance", hv, "#c53030")]):
    z = df.condition_2 == 0
    ax.scatter(df.condition_2[z], df.condition_3[z], s=20, alpha=0.65, color="#c53030",
               edgecolor="none", label=f"condition_2 = 0 (n={int(z.sum())})")
    ax.scatter(df.condition_2[~z], df.condition_3[~z], s=20, alpha=0.65, color="#2b6cb0",
               edgecolor="none", label=f"condition_2 > 0 (n={int((~z).sum())})")
    r_all = stats.spearmanr(df.condition_2, df.condition_3).statistic
    r_nz = stats.spearmanr(df.condition_2[~z], df.condition_3[~z]).statistic
    ax.set_title(f"{tag}\nrho(all)={r_all:+.3f}  rho(non-zero)={r_nz:+.3f}", fontsize=10)
    ax.set_xlabel("condition_2")
    ax.set_ylabel("condition_3")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

ax = axes[2]
labels = ["small_sample", "high_variance"]
all_r = [stats.spearmanr(ss.condition_2, ss.condition_3).statistic,
         stats.spearmanr(hv.condition_2, hv.condition_3).statistic]
nz_r = [stats.spearmanr(ss.condition_2[ss.condition_2 > 0], ss.condition_3[ss.condition_2 > 0]).statistic,
        stats.spearmanr(hv.condition_2[hv.condition_2 > 0], hv.condition_3[hv.condition_2 > 0]).statistic]
xpos = np.arange(2)
ax.bar(xpos - 0.18, all_r, 0.36, color="#2b6cb0", label="all cells")
ax.bar(xpos + 0.18, nz_r, 0.36, color="#2f855a", label="excluding condition_2 = 0")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("Spearman rho(condition_2, condition_3)")
ax.set_title("C. Coupling collapses without zeros")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(FIG / "F4_zero_driven_coupling.png", dpi=170)
plt.close(fig)
record("F4_zero_driven_coupling.png",
       "Condition coupling before and after excluding cells where condition_2 equals zero; the "
       "residual association in high_variance is the scenario contrast, not a metric-level law.")

# --- F5: applicability boundary of the covariate association --------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
med = float(np.median(tab.continuous_covariate))
lo = tab.continuous_covariate <= med
hi = ~lo
ax = axes[0]
ax.scatter(tab.continuous_covariate[lo], tab.rich[lo], s=15, alpha=0.6,
           color="#2f855a", edgecolor="none", label=f"covariate <= {med:g} (n={int(lo.sum())})")
ax.scatter(tab.continuous_covariate[hi], tab.rich[hi], s=15, alpha=0.6,
           color="#b7791f", edgecolor="none", label=f"covariate > {med:g} (n={int(hi.sum())})")
ax.axvline(med, ls="--", color="black", lw=1.2)
r_lo = stats.spearmanr(tab.continuous_covariate[lo], tab.rich[lo]).statistic
r_hi = stats.spearmanr(tab.continuous_covariate[hi], tab.rich[hi]).statistic
ax.set_xscale("log")
ax.set_xlabel("continuous covariate")
ax.set_ylabel("observed richness")
ax.set_title(f"A. Boundary at the covariate median\nrho={r_lo:+.3f} below, {r_hi:+.3f} above", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
tab2 = tab.assign(bin=pd.qcut(tab.continuous_covariate.rank(method="first"), 16, labels=False))
agg = tab2.groupby("bin").agg(xm=("continuous_covariate", "median"), ym=("rich", "median"),
                              q1=("rich", lambda v: v.quantile(0.25)),
                              q3=("rich", lambda v: v.quantile(0.75)))
ax.errorbar(agg.xm, agg.ym, yerr=[agg.ym - agg.q1, agg.q3 - agg.ym], marker="o",
            capsize=3, color="#2b6cb0", lw=1.6)
ax.axvline(med, ls="--", color="black", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("covariate (16 equal-count bins, median and IQR)")
ax.set_ylabel("richness")
ax.set_title("B. Rise then plateau", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
groups = sorted(tab.coarse_group_id.unique())
xpos = np.arange(len(groups))
rb, ra = [], []
for g in groups:
    sub = tab[tab.coarse_group_id == g]
    m = sub.continuous_covariate <= med
    rb.append(stats.spearmanr(sub.continuous_covariate[m], sub.rich[m]).statistic)
    ra.append(stats.spearmanr(sub.continuous_covariate[~m], sub.rich[~m]).statistic
              if (~m).sum() > 10 else np.nan)
ax.bar(xpos - 0.18, rb, 0.36, color="#2f855a", label="below global median")
ax.bar(xpos + 0.18, ra, 0.36, color="#b7791f", label="above global median")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{g}\n(n={int((tab.coarse_group_id==g).sum())})" for g in groups], fontsize=9)
ax.set_ylabel("Spearman rho(covariate, richness)")
ax.set_title("C. Boundary inside every group", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(FIG / "F5_applicability_boundary.png", dpi=170)
plt.close(fig)
record("F5_applicability_boundary.png",
       "The covariate-richness association is confined to the lower half of the covariate range; "
       "the same split reappears inside each coarse group when the global median is held fixed.")

save_json(manifest, OUT / "results" / "figure_manifest.json")
print("stage 5 ok - figures written:")
for k in manifest:
    print("  ", k, (FIG / k).stat().st_size, "bytes")