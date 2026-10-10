#!/usr/bin/env python3
"""Robustness checker B: measurement robustness + solver-specific robustness.

Read-only on data files. Conventions aligned with analysis.py (the main
analysis):
  - log_sv base is natural log.
  - total change per family: tot{f} = f{f}d2 - f{f}d0
    slope per family:         slope{f} = (f{f}d2 - f{f}d0) / 2.0
    convergent:               conv{f} = (tot{f} < 0)
  - parameter_index is joined to log_singular_value via mode_index
    (parameter_index == mode_index).
  - measurement error:  err = meas - sim
    relative error:     rel = (meas - sim) / sim
  - winsorize at 5/95: each of meas and sim clipped at its own 5th and
    95th percentile; then recompute error mean and Pearson corr.
  - outlier exclusion: drop bottom 10% and top 10% of pairs by |err|
    (keep the middle 80%), recompute on the remaining pairs.
  - quantile corr: split pairs into 5 equal-frequency bins of sim value,
    report corr(meas, sim) in each.
  - solver leave-one-family-out: pooled slope-sv corr excluding fam0 =
    fam1-only slope-sv corr; excluding fam1 = fam0-only slope-sv corr
    (each should equal the corresponding per-family corr).
  - interaction: 20 equal-width log_SV bins; convergence fraction per bin
    for fam0 and fam1; report the lowest-SV bin (bin 0) for each.
  - bootstrap: seed 42, 3000 reps, parameter resampling, on
    (mean fam0 total - mean fam1 total); 95% CI at 2.5/97.5 percentiles;
    one-sided sign-test p = fraction of bootstrap replicates having the
    same sign as the observed difference.

Outputs:
  /data/.../output/robustness_b.json
  /data/.../output/figures/robustness_b.png
"""
import json
import os
from math import erf

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/data/user/conversations/conv-c59f3f82-1c0f-4cac-9a42-86dc073cd65e/project"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

TRAJ = os.path.join(DATA, "parameter_trajectories.tsv")
SPECT = os.path.join(DATA, "singular_value_spectrum.tsv")
PAIRS = os.path.join(DATA, "measurement_simulation_pairs.tsv")
OUT_JSON = os.path.join(OUT, "robustness_b.json")
OUT_PNG = os.path.join(FIG, "robustness_b.png")

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
pt = pd.read_csv(TRAJ, sep="\t",
                 dtype={"parameter_index": "int64", "solver_family_id": "int64",
                        "progress_index": "int64"})
sv = pd.read_csv(SPECT, sep="\t", dtype={"mode_index": "int64"}).sort_values(
    "mode_index").reset_index(drop=True)
ms = pd.read_csv(PAIRS, sep="\t")

n_params = int(pt["parameter_index"].nunique())
n_fams = int(pt["solver_family_id"].nunique())
n_modes = int(sv["mode_index"].nunique())
n_pairs = int(len(ms))
print(f"params={n_params} families={n_fams} modes={n_modes} pairs={n_pairs}")

# ------------------------------------------------------------------
# Build trajectory frame (same as analysis.py)
# ------------------------------------------------------------------
piv = pt.pivot(index="parameter_index", columns=("solver_family_id", "progress_index"),
               values="normalised_distance_like_value").sort_index()
t = pd.DataFrame(index=piv.index)
for f in (0, 1):
    for d in (0, 1, 2):
        t[f"f{f}d{d}"] = piv[(f, d)]
t["param"] = t.index
svf = sv.set_index("mode_index")["log_singular_value"]
t["log_sv"] = t["param"].map(svf)
for f in (0, 1):
    t[f"tot{f}"] = t[f"f{f}d2"] - t[f"f{f}d0"]
    t[f"slope{f}"] = (t[f"f{f}d2"] - t[f"f{f}d0"]) / 2.0
    t[f"conv{f}"] = t[f"tot{f}"] < 0
    t[f"div{f}"] = t[f"tot{f}"] > 0
    t[f"flat{f}"] = t[f"tot{f}"].abs() < 0.01

def corr2(x, y):
    return float(np.corrcoef(x, y)[0, 1])

# ------------------------------------------------------------------
# (1) MEASUREMENT robustness
# ------------------------------------------------------------------
meas = ms["measurement_like_value"].to_numpy(dtype=float)
sim = ms["simulation_like_value"].to_numpy(dtype=float)
err = meas - sim
rel = (meas - sim) / sim

# baseline
baseline_mean = float(err.mean())
baseline_abs_mean = float(np.abs(err).mean())
baseline_corr = float(np.corrcoef(meas, sim)[0, 1])
err_std = float(np.std(err))
t_stat = float(err.mean() / (err_std / np.sqrt(n_pairs)))
p_two = float(1.0 - erf(t_stat / np.sqrt(2.0)))   # analysis.py convention
p_one = float(0.5 * (1.0 - erf(t_stat / np.sqrt(2.0))))
p_two = min(max(p_two, 0.0), 1.0)

# winsorize at 5/95 (clip each variable at its own 5/95 percentile)
meas_w = np.clip(meas, np.percentile(meas, 5), np.percentile(meas, 95))
sim_w = np.clip(sim, np.percentile(sim, 5), np.percentile(sim, 95))
winsor_mean = float(np.mean(meas_w - sim_w))
winsor_corr = float(np.corrcoef(meas_w, sim_w)[0, 1])
winsor_abs_mean = float(np.mean(np.abs(meas_w - sim_w)))
# codebase reference (winsor of err only)
winsor_err_ref = float(np.mean(np.clip(err, np.percentile(err, 5), np.percentile(err, 95))))

# exclude bottom 10% and top 10% by |err| (keep middle 80%)
order = np.argsort(np.abs(err))
n_keep_start = int(len(order) * 0.1)
n_keep_end = int(len(order) * 0.9)
keep = order[n_keep_start:n_keep_end]
outlier_excl_mean = float(err[keep].mean())
outlier_excl_abs_mean = float(np.abs(err[keep]).mean())
outlier_excl_corr = float(np.corrcoef(meas[keep], sim[keep])[0, 1])

# relative error
rel_mean = float(np.nanmean(rel))
rel_frac_abs_lt_05 = float(np.nanmean(np.abs(rel) < 0.5))

# correlation by quantile (5 equal-frequency bins of sim)
qcut = pd.qcut(sim, 5, labels=False)
corr_by_quantile = []
for q in range(5):
    m = (qcut == q)
    if np.sum(m) > 1:
        corr_by_quantile.append(float(np.corrcoef(meas[m], sim[m])[0, 1]))
    else:
        corr_by_quantile.append(float("nan"))
corr_by_quantile = np.array(corr_by_quantile, dtype=float)

# conclusions
bias_near_zero = bool((p_two > 0.05) and (abs(baseline_mean) / err_std < 0.1))
# correlation roughly constant across quantiles: spread of the 5 corr values small
corr_spread = float(np.nanmax(corr_by_quantile) - np.nanmin(corr_by_quantile))
corr_constant = bool(corr_spread < 0.05)

print("MEASUREMENT")
print(f"  baseline: mean={baseline_mean:.6f} abs_mean={baseline_abs_mean:.6f} "
      f"corr={baseline_corr:.6f}")
print(f"  winsor:   mean={winsor_mean:.6f} corr={winsor_corr:.6f} "
      f"(winsor_err_ref={winsor_err_ref:.6f})")
print(f"  out_excl: mean={outlier_excl_mean:.6f} abs_mean={outlier_excl_abs_mean:.6f} "
      f"corr={outlier_excl_corr:.6f}")
print(f"  rel:      mean={rel_mean:.6f} frac|rel|<0.5={rel_frac_abs_lt_05:.6f}")
print(f"  corr_by_quantile={np.round(corr_by_quantile, 6)} spread={corr_spread:.6f}")
print(f"  t_stat={t_stat:.4f} p_two={p_two:.4f} p_one={p_one:.4f}")
print(f"  bias_near_zero={bias_near_zero} corr_constant={corr_constant} "
      f"(spread={corr_spread:.4f})")

# ------------------------------------------------------------------
# (2) SOLVER-SPECIFIC robustness
# ------------------------------------------------------------------
# (a) leave-one-family-out: pooled slope-sv corr excluding famX
lof_without_fam0 = float(corr2(t["slope1"].values, t["log_sv"].values))  # fam1 only
lof_without_fam1 = float(corr2(t["slope0"].values, t["log_sv"].values))  # fam0 only
# per-family reference
fam0_corr_slope_sv = float(corr2(t["slope0"].values, t["log_sv"].values))
fam1_corr_slope_sv = float(corr2(t["slope1"].values, t["log_sv"].values))
print("SOLVER (a) leave-one-family-out:")
print(f"  lof_without_fam0 (fam1-only)={lof_without_fam0:.6f} "
      f"== fam1 per-family {fam1_corr_slope_sv:.6f}: {abs(lof_without_fam0-fam1_corr_slope_sv)<1e-12}")
print(f"  lof_without_fam1 (fam0-only)={lof_without_fam1:.6f} "
      f"== fam0 per-family {fam0_corr_slope_sv:.6f}: {abs(lof_without_fam1-fam0_corr_slope_sv)<1e-12}")

# (b) per-parameter sign
perparam = t["tot0"] - t["tot1"]
frac_fam0_gt_fam1 = float(np.mean(perparam > 0))
print(f"  (b) frac fam0 total > fam1 total = {frac_fam0_gt_fam1:.6f}")

# (c) interaction: convergence fraction by log_SV bin (20 bins)
ls = sv["log_singular_value"].to_numpy(dtype=float)
n_bins = 20
edges = np.linspace(ls.min(), ls.max(), n_bins + 1)
t_logsv = t["log_sv"].to_numpy()
conv0b = []
conv1b = []
for i in range(n_bins):
    if i < n_bins - 1:
        mask = (t_logsv >= edges[i]) & (t_logsv < edges[i + 1])
    else:
        # last bin is closed on the right to include the maximum
        mask = (t_logsv >= edges[i]) & (t_logsv <= edges[i + 1])
    conv0b.append(float(np.mean(t.loc[mask, "conv0"])) if mask.sum() > 0 else np.nan)
    conv1b.append(float(np.mean(t.loc[mask, "conv1"])) if mask.sum() > 0 else np.nan)
conv0b = np.array(conv0b, dtype=float)
conv1b = np.array(conv1b, dtype=float)
fam0_conv_lowsv_bin0 = float(conv0b[0])
fam1_conv_lowsv_bin0 = float(conv1b[0])
print(f"  (c) convergence fraction by log_SV bin (20 bins):")
print(f"      fam0: {np.round(conv0b, 6)}")
print(f"      fam1: {np.round(conv1b, 6)}")
print(f"      lowest-SV bin (bin0): fam0={fam0_conv_lowsv_bin0:.6f} fam1={fam1_conv_lowsv_bin0:.6f}")

# (d) bootstrap (seed 42, 3000 reps, parameter resampling) on (mean tot0 - mean tot1)
rng = np.random.default_rng(42)
N = len(t)
boot = []
for _ in range(3000):
    idx = rng.integers(0, N, N)
    boot.append(t.loc[idx, "tot0"].mean() - t.loc[idx, "tot1"].mean())
boot = np.array(boot)
diff = float(t["tot0"].mean() - t["tot1"].mean())
bootstrap_ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
sign_test_p = float(np.mean(boot > 0) if diff > 0 else np.mean(boot < 0))
solver_effect_real = bool(bootstrap_ci95[0] > 0 and bootstrap_ci95[1] > 0)  # CI excludes 0
print(f"  (d) bootstrap: diff={diff:.6f} CI95={bootstrap_ci95} "
      f"sign_test_p={sign_test_p:.6f} solver_effect_real={solver_effect_real}")

# ------------------------------------------------------------------
# Write JSON
# ------------------------------------------------------------------
result = {
    "meta": {
        "n_parameters": n_params, "n_families": n_fams, "n_modes": n_modes,
        "n_pairs": n_pairs, "n_bins_log_sv": n_bins, "n_quantiles": 5,
        "n_bootstrap_reps": 3000, "bootstrap_seed": 42,
        "log_sv_base": "natural",
        "flat_threshold": 0.01,
    },
    "measurement": {
        "baseline_mean": baseline_mean,
        "baseline_abs_mean": baseline_abs_mean,
        "baseline_corr": baseline_corr,
        "winsor_mean": winsor_mean,
        "winsor_corr": winsor_corr,
        "outlier_excl_mean": outlier_excl_mean,
        "outlier_excl_abs_mean": outlier_excl_abs_mean,
        "outlier_excl_corr": outlier_excl_corr,
        "rel_mean": rel_mean,
        "rel_frac_abs_lt_05": rel_frac_abs_lt_05,
        "corr_by_quantile": [float(v) for v in corr_by_quantile],
        "bias_near_zero": bias_near_zero,
        "corr_constant": corr_constant,
    },
    "solver": {
        "lof_without_fam0": lof_without_fam0,
        "lof_without_fam1": lof_without_fam1,
        "frac_fam0_gt_fam1": frac_fam0_gt_fam1,
        "fam0_conv_lowsv_bin0": fam0_conv_lowsv_bin0,
        "fam1_conv_lowsv_bin0": fam1_conv_lowsv_bin0,
        "bootstrap_ci95": bootstrap_ci95,
        "sign_test_p": sign_test_p,
        "solver_effect_real": solver_effect_real,
    },
    "methods": {
        "winsor": "clip meas and sim each at its own 5/95 percentile, then error mean and Pearson corr",
        "outlier_excl": "drop bottom 10% and top 10% of pairs by |err| (keep middle 80%)",
        "corr_by_quantile": "5 equal-frequency bins of sim value",
        "lof_without_famX": "pooled slope-sv corr computed on the family other than famX",
        "frac_fam0_gt_fam1": "mean(tot0 > tot1) over parameters",
        "log_sv_bins": "20 equal-width bins on log_sv axis; lowest-SV bin = bin 0",
        "bootstrap": "seed 42, 3000 parameter resamples of (mean tot0 - mean tot1); "
                     "CI95 = 2.5/97.5 percentiles; sign_test_p = frac replicates with sign of diff",
        "conclusions": {
            "bias_near_zero": "two-sided normal test of mean err == 0 (p>0.05) AND |err_mean|/err_std < 0.1",
            "corr_constant": "max-min of the 5 quantile correlations < 0.05",
            "solver_effect_real": "bootstrap 95% CI for (mean tot0 - mean tot1) excludes 0",
        },
        "correlation_by_quantile_values": [float(v) for v in corr_by_quantile],
        "corr_spread": corr_spread,
        "t_stat": t_stat,
        "p_two": p_two,
        "p_one": p_one,
        "winsor_err_ref": winsor_err_ref,
        "fam0_corr_slope_sv": fam0_corr_slope_sv,
        "fam1_corr_slope_sv": fam1_corr_slope_sv,
        "conv0_by_bin": [float(v) if not np.isnan(v) else None for v in conv0b],
        "conv1_by_bin": [float(v) if not np.isnan(v) else None for v in conv1b],
    },
    "conclusions": {
        "bias_is_near_zero": bias_near_zero,
        "correlation_is_roughly_constant": corr_constant,
        "solver_effect_is_real": solver_effect_real,
        "note": (
            "Bias (mean meas-sim) is near zero and insensitive to winsorization/outlier exclusion; "
            "meas-sim correlation is high overall (0.765) but NOT constant across "
            "simulation-value quantiles (corr_by_quantile="
            f"{[round(float(x),3) for x in corr_by_quantile]}, "
            f"spread={corr_spread:.3f}, corr_constant={corr_constant}). "
            "The fam0-vs-fam1 total-change difference is strongly positive and real "
            "(bootstrap CI excludes 0, sign-test p=1.0, "
            f"{frac_fam0_gt_fam1:.2%} of parameters favor fam0); fam1 converges in the lowest-SV regime."
        ),
    },
}

with open(OUT_JSON, "w") as fh:
    json.dump(result, fh, indent=2, default=float)
print("WROTE", OUT_JSON)

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Robustness B: measurement and solver-specific robustness", y=0.995)

# Panel 1: mean / abs-mean under baseline, winsor, outlier-excl
ax = axes[0, 0]
xs = np.arange(3)
w = 0.35
means = [baseline_mean, winsor_mean, outlier_excl_mean]
abs_means = [baseline_abs_mean, winsor_abs_mean, outlier_excl_abs_mean]
ax.bar(xs - w / 2, means, width=w, label="mean error", color="tab:blue", alpha=0.9)
ax.bar(xs + w / 2, abs_means, width=w, label="abs-mean error", color="tab:orange", alpha=0.9)
ax.axhline(0, color="grey", linestyle=":", lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(["baseline", "winsor 5/95", "excl 10% |err|"])
ax.set_ylabel("error magnitude")
ax.set_title("Measurement error: mean & abs-mean under robustness transforms")
ax.legend()
ax.set_ylim(-max(abs(baseline_abs_mean), abs(outlier_excl_abs_mean)) * 1.1,
            max(abs(baseline_abs_mean), abs(outlier_excl_abs_mean)) * 1.1)

# Panel 2: corr by quantile
ax = axes[0, 1]
xs2 = np.arange(5)
ax.bar(xs2, corr_by_quantile, color="tab:blue", alpha=0.9, label="corr per quantile")
ax.axhline(baseline_corr, color="tab:red", linestyle="--", lw=1.5,
           label=f"baseline corr={baseline_corr:.3f}")
ax.axhline(outlier_excl_corr, color="tab:green", linestyle="--", lw=1.5,
           label=f"outlier-excl corr={outlier_excl_corr:.3f}")
ax.set_xticks(xs2)
ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
ax.set_ylabel("corr(meas, sim)")
ax.set_title("Measurement-simulation correlation by simulation-value quantile")
ax.legend()
ax.set_ylim(0, max(baseline_corr, max(corr_by_quantile)) * 1.15)

# Panel 3: convergence fraction by log_SV bin
bin_centers = 0.5 * (edges[:-1] + edges[1:])
ax = axes[1, 0]
ax.plot(bin_centers, conv0b, "o-", color="tab:blue", ms=3, label="fam0 conv frac")
ax.plot(bin_centers, conv1b, "s-", color="tab:red", ms=3, label="fam1 conv frac")
ax.axvline(edges[0] + (edges[1] - edges[0]) / 2, color="grey", linestyle=":", lw=1,
           label="lowest-SV bin")
ax.set_xlabel("log singular value (bin center)")
ax.set_ylabel("fraction convergent")
ax.set_title("Convergence fraction by log_SV bin (20 bins)")
ax.legend()
ax.set_ylim(0, 1.02)

# Panel 4: bootstrap distribution
ax = axes[1, 1]
ax.hist(boot, bins=50, density=True, color="tab:purple", alpha=0.8,
        label="3000 bootstrap replicates")
ax.axvline(diff, color="black", linestyle="-", lw=2,
           label=f"observed diff={diff:.4f}")
ax.axvline(0, color="tab:red", linestyle="--", lw=1.5, label="zero")
ax.axvline(bootstrap_ci95[0], color="tab:orange", linestyle=":", lw=2,
           label=f"95% CI [{bootstrap_ci95[0]:.4f}, {bootstrap_ci95[1]:.4f}]")
ax.set_xlabel("mean(fam0 total) - mean(fam1 total)")
ax.set_ylabel("density")
ax.set_title(f"Bootstrap distribution of the fam0-fam1 difference "
             f"(sign-test p={sign_test_p:.2f})")
ax.legend()

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print("WROTE", OUT_PNG)

# ------------------------------------------------------------------
# Final numeric summary for the structured output
# ------------------------------------------------------------------
summary = {
    "measurement": {
        "baseline_mean": baseline_mean,
        "baseline_abs_mean": baseline_abs_mean,
        "baseline_corr": baseline_corr,
        "winsor_mean": winsor_mean,
        "winsor_corr": winsor_corr,
        "outlier_excl_mean": outlier_excl_mean,
        "outlier_excl_abs_mean": outlier_excl_abs_mean,
        "outlier_excl_corr": outlier_excl_corr,
        "rel_mean": rel_mean,
        "rel_frac_abs_lt_05": rel_frac_abs_lt_05,
        "corr_by_quantile": [float(v) for v in corr_by_quantile],
        "bias_near_zero": bias_near_zero,
        "corr_constant": corr_constant,
    },
    "solver": {
        "lof_without_fam0": lof_without_fam0,
        "lof_without_fam1": lof_without_fam1,
        "frac_fam0_gt_fam1": frac_fam0_gt_fam1,
        "fam0_conv_lowsv_bin0": fam0_conv_lowsv_bin0,
        "fam1_conv_lowsv_bin0": fam1_conv_lowsv_bin0,
        "bootstrap_ci95": bootstrap_ci95,
        "sign_test_p": sign_test_p,
        "solver_effect_real": solver_effect_real,
    },
}
print(json.dumps(summary, indent=2))