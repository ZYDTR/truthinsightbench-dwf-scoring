"""Figures 3 and 4 (size spectrum and counting quantisation) + plot-data CSVs.

Everything is recomputed from the two frozen TSVs and from the count-tally
reconstruction stored in output/results/07_resolution_spectrum.json.  No number
is hard-coded.

Figure 3
  A  mean particle-size percentage vs bin midpoint (log-log) for all 22 bins,
     with 95% bootstrap CIs over the 30 samples (4000 draws).  Three marker
     styles separate bins that are nonzero in all / some / none of the samples.
  B  log10 total concentration vs the per-sample log-log spectral slope fitted
     over size bins 1-9 (low-edge ascending), points coloured by the number of
     nonzero size bins in the sample; Spearman rho and p printed.

Figure 4
  A  log10 total concentration vs the smallest reported nonzero percentage in
     the sample (log y).  A horizontal line at 100 marks "one particle in the
     whole tally"; the curve 100/N is drawn at the samples whose tally N was
     reconstructed in 07.  The 3 most extreme samples are labelled.
  B  number of nonzero size bins vs log10 total concentration, Spearman rho/p.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
RESULTS = PROJECT / "output" / "results"
FIGURES = PROJECT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


comp = read_tsv(DATA / "particle_composition_profiles.tsv")
env = read_tsv(DATA / "sample_environment_profiles.tsv")
n = len(comp)

SIZE_BINS = [c[len("particle_size_"):-len("_percent")] for c in comp[0]
             if c.startswith("particle_size_") and c.endswith("_percent")]
# Bin edges are encoded in the labels: "11um" is the open lower bin (midpoint
# taken as 11 um, exactly as in the frozen pipeline), "gt_500um" is the open
# upper bin (midpoint convention 600 um, as in output/analysis/common.py), and
# every other label lo_hi gives the arithmetic midpoint.
def bin_midpoint(label):
    if label == "gt_500um":
        return 600.0
    if "_" not in label:
        return float(label.replace("um", ""))
    lo, hi = label.replace("um", "").split("_")
    return 0.5 * (float(lo) + float(hi))


SIZE_MID = np.array([bin_midpoint(b) for b in SIZE_BINS])

sp = np.array([[float(r[f"particle_size_{b}_percent"]) for b in SIZE_BINS]
               for r in comp])                       # 30 x 22 size percentages
polymer_cols = [c for c in comp[0] if c.startswith("polymer_")
                and c.endswith("_percent")]
pp = np.array([[float(r[c]) for c in polymer_cols] for r in comp])
tot = np.array([float(r["particle_total_per_L"]) for r in comp])
ltot = np.log10(tot)
groups = [r["collection_group_id"] for r in comp]
sample_ids = [r["sample_id"] for r in comp]

n_nonzero_bin = (sp > 0).sum(axis=0)          # per bin
n_nonzero_sample = (sp > 0).sum(axis=1)       # per sample
smallest_nonzero = np.array(
    [min(np.concatenate([sp[i][sp[i] > 0], pp[i][pp[i] > 0]])) for i in range(n)])

# reconstructed tallies (adversarially re-checked; used only for the 100/N curve)
tal = json.loads((RESULTS / "07_resolution_spectrum.json").read_text())["tally_bounds"]
tally_by_id = {d["sample_id"]: d["N_common_min"] for d in tal}
N_arr = np.array([tally_by_id.get(sid, None) if tally_by_id.get(sid) is not None
                  else np.nan for sid in sample_ids], float)

# --------------------------------------------------------------------------
# derived statistics
# --------------------------------------------------------------------------
# per-sample log-log slope over size bins 1-9 (nonzero bins only)
lmid = np.log10(SIZE_MID)
slope9 = np.array([
    np.polyfit(lmid[:9][sp[j, :9] > 0], np.log10(sp[j, :9][sp[j, :9] > 0]), 1)[0]
    for j in range(n)])
rho_b, p_b = stats.spearmanr(ltot, slope9)
rho_4b, p_4b = stats.spearmanr(ltot, n_nonzero_sample)

# 95% bootstrap CI for the mean percentage of each bin, resampling samples
BOOT = 4000
rng = np.random.default_rng(20260919)
boot_means = np.empty((BOOT, len(SIZE_BINS)))
for b in range(BOOT):
    idx = rng.integers(0, n, n)
    boot_means[b] = sp[idx].mean(axis=0)
ci_lo = np.quantile(boot_means, 0.025, axis=0)
ci_hi = np.quantile(boot_means, 0.975, axis=0)
mean_pct = sp.mean(axis=0)

# --------------------------------------------------------------------------
# plot-data CSV: figure 3
# --------------------------------------------------------------------------
with open(RESULTS / "fig3_size_spectrum.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["bin_label", "bin_midpoint_um", "mean_percent", "ci_lo", "ci_hi",
                "n_nonzero"])
    for i, b in enumerate(SIZE_BINS):
        w.writerow([b, f"{SIZE_MID[i]:g}", f"{mean_pct[i]:.6f}",
                    f"{ci_lo[i]:.6f}", f"{ci_hi[i]:.6f}", int(n_nonzero_bin[i])])

# --------------------------------------------------------------------------
# figure 3
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=200)

ax = axes[0]
FLOOR = 2e-4
plot_y = np.where(mean_pct > 0, mean_pct, FLOOR)
all_bins = n_nonzero_bin == n
some_bins = (n_nonzero_bin > 0) & (n_nonzero_bin < n)
none_bins = n_nonzero_bin == 0

lo_plot = np.where(ci_lo > 0, ci_lo, FLOOR)
ax.errorbar(SIZE_MID[some_bins], plot_y[some_bins],
            yerr=[plot_y[some_bins] - lo_plot[some_bins],
                  np.maximum(ci_hi[some_bins] - plot_y[some_bins], 0)],
            fmt="o", ms=6, color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0,
            capsize=3, label=f"nonzero in some samples (<{n})", zorder=3)
ax.errorbar(SIZE_MID[all_bins], plot_y[all_bins],
            yerr=[plot_y[all_bins] - lo_plot[all_bins],
                  np.maximum(ci_hi[all_bins] - plot_y[all_bins], 0)],
            fmt="s", ms=8, mfc="#d62728", mec="black", color="#d62728",
            ecolor="#d62728", elinewidth=1.2, capsize=3,
            label=f"nonzero in all {n} samples", zorder=4)
ax.errorbar(SIZE_MID[none_bins], plot_y[none_bins], fmt="v", ms=6,
            mfc="none", mec="#7f7f7f", color="#7f7f7f", linestyle="none",
            label="never nonzero (plotted at display floor)", zorder=2)
ax.axhline(FLOOR, color="#7f7f7f", lw=0.8, ls=":", zorder=1)
ax.text(SIZE_MID.min() * 0.92, FLOOR * 1.25, "display floor: bin never nonzero "
        "(no value exists)", fontsize=7.5, color="#555555")
for i, b in enumerate(SIZE_BINS):
    if n_nonzero_bin[i] == 0:
        continue  # seven coincident floor markers get one shared label below
    dy = 8 if i % 2 == 0 else -14
    ax.annotate(f"{n_nonzero_bin[i]}/{n}", (SIZE_MID[i], plot_y[i]),
                textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=6.2, color="#333333")
n_never = int(none_bins.sum())
ax.annotate(f"{n_never} bins never nonzero (0/{n}); all on the floor line",
            (SIZE_MID[none_bins][0], FLOOR), textcoords="offset points",
            xytext=(0, -22), ha="left", fontsize=7.2, color="#666666")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-4, 300)
ax.set_xlim(6, 900)
ax.set_xlabel("bin midpoint (um, log scale)")
ax.set_ylabel("mean size percentage per sample (log scale)")
ax.set_title("A  Particle size spectrum (22 bins, mean +/- 95% bootstrap CI,\n"
             f"{BOOT} draws over {n} depth intervals)", fontsize=10)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

ax = axes[1]
sc = ax.scatter(ltot, slope9, c=n_nonzero_sample, cmap="viridis", s=55,
                edgecolor="black", linewidth=0.5, zorder=3)
# OLS trend for display only
b1, b0 = np.polyfit(ltot, slope9, 1)
xs = np.linspace(ltot.min(), ltot.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#d62728", lw=1.3, ls="--", zorder=2,
        label=f"OLS: slope = {b1:.2f} per decade (display only)")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("nonzero size bins in the sample (of 22)", fontsize=9)
ax.text(0.03, 0.05, f"Spearman rho = {rho_b:.3f}\np = {p_b:.3g}\nn = {n}",
        transform=ax.transAxes, fontsize=10, va="bottom",
        bbox=dict(fc="white", ec="#888888", alpha=0.9))
ax.set_xlabel("log10 total particle concentration (per L)")
ax.set_ylabel("fitted log-log slope, size bins 1-9")
ax.set_title("B  Spectral slope vs abundance, coloured by size resolution",
             fontsize=10)
ax.grid(True, ls=":", lw=0.4, alpha=0.6)
ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Figure 3 - Depth-resolved ice particle size spectrum "
             "(30 intervals, 5 collection groups)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES / "fig3_size_spectrum.png", dpi=200)
plt.close(fig)

# --------------------------------------------------------------------------
# plot-data CSV: figure 4
# --------------------------------------------------------------------------
with open(RESULTS / "fig4_quantisation.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sample_id", "collection_group_id", "total_per_L", "log10_total",
                "N_common_min", "quantum_pp", "n_nonzero_size_bins",
                "smallest_nonzero_percent"])
    for i in range(n):
        N = N_arr[i]
        w.writerow([sample_ids[i], groups[i], f"{tot[i]:g}", f"{ltot[i]:.6f}",
                    "" if not np.isfinite(N) else f"{N:g}",
                    "" if not np.isfinite(N) else f"{100.0 / N:.6f}",
                    int(n_nonzero_sample[i]), f"{smallest_nonzero[i]:.6f}"])

# --------------------------------------------------------------------------
# figure 4
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=200)

ax = axes[0]
ax.scatter(ltot, smallest_nonzero, s=60, color="#1f77b4", edgecolor="black",
           linewidth=0.5, zorder=4)
ax.axhline(100.0, color="#d62728", lw=1.3, ls="--", zorder=2,
           label="100 = one particle in the whole tally")
have_N = np.isfinite(N_arr)
order = np.argsort(ltot[have_N])
xs = ltot[have_N][order]
q = 100.0 / N_arr[have_N][order]
ax.plot(xs, q, "-", color="#2ca02c", lw=1.2, zorder=3,
        label="100/N, N = reconstructed tally (17 samples)")
ax.plot(xs, q, "o", ms=7, mfc="none", mec="#2ca02c", mew=1.4, zorder=5)
# label the three most extreme samples: lowest abundance, highest abundance,
# and the second-lowest abundance
extreme = [int(np.argmin(tot)), int(np.argmax(tot)),
           int(np.argsort(tot)[1])]
for j in extreme:
    ax.annotate(sample_ids[j], (ltot[j], smallest_nonzero[j]),
                textcoords="offset points", xytext=(8, 8), fontsize=8.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#444444"))
ax.set_yscale("log")
ax.set_ylim(1e-3, 200)
ax.set_xlabel("log10 total particle concentration (per L)")
ax.set_ylabel("smallest reported nonzero percentage (log scale)")
ax.set_title("A  Detection floor vs abundance\n(smallest nonzero percentage, "
             "both classification families)", fontsize=10)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.scatter(ltot, n_nonzero_sample, s=60, color="#ff7f0e", edgecolor="black",
           linewidth=0.5, zorder=3)
b1, b0 = np.polyfit(ltot, n_nonzero_sample, 1)
xs = np.linspace(ltot.min(), ltot.max(), 50)
ax.plot(xs, b0 + b1 * xs, color="#d62728", lw=1.3, ls="--", zorder=2,
        label=f"OLS: {b1:.2f} bins per decade (display only)")
ax.text(0.03, 0.05, f"Spearman rho = {rho_4b:.3f}\np = {p_4b:.3g}\nn = {n}",
        transform=ax.transAxes, fontsize=10, va="bottom",
        bbox=dict(fc="white", ec="#888888", alpha=0.9))
ax.set_xlabel("log10 total particle concentration (per L)")
ax.set_ylabel("number of nonzero size bins (of 22)")
ax.set_title("B  Size-bin resolution vs abundance", fontsize=10)
ax.grid(True, ls=":", lw=0.4, alpha=0.6)
ax.legend(loc="upper left", fontsize=8)

fig.suptitle("Figure 4 - Counting quantisation and the detection floor in the "
             "ice particle tallies", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES / "fig4_quantisation.png", dpi=200)
plt.close(fig)

# --------------------------------------------------------------------------
# console report
# --------------------------------------------------------------------------
print("FIG3 panel A (mean % [95% CI], n nonzero):")
for i, b in enumerate(SIZE_BINS):
    print(f"  {b:12s} mid={SIZE_MID[i]:7.1f} mean={mean_pct[i]:10.6f} "
          f"[{ci_lo[i]:10.6f},{ci_hi[i]:10.6f}] n={int(n_nonzero_bin[i])}")
print(f"FIG3 panel B: Spearman rho={rho_b:.6f} p={p_b:.6g} over n={n}")
print("FIG4 panel A: smallest nonzero percent range "
      f"{smallest_nonzero.min():.4f}..{smallest_nonzero.max():.4f}; "
      f"N known for {int(have_N.sum())}/{n} samples; "
      f"100/N range {np.nanmin(100.0/N_arr[have_N]):.4f}.."
      f"{np.nanmax(100.0/N_arr[have_N]):.4f}")
print(f"FIG4 panel B: Spearman rho={rho_4b:.6f} p={p_4b:.6g} over n={n}")
for f in ["fig3_size_spectrum.png", "fig4_quantisation.png"]:
    p = FIGURES / f
    print(f"{p}: exists={p.exists()} bytes={p.stat().st_size if p.exists() else 0}")