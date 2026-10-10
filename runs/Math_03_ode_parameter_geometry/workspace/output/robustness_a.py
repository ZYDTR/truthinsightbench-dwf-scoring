#!/usr/bin/env python3
"""Robustness checker A: alternative progress-scale definitions, alternative
spectrum models, and parameter batch leave-out. Read-only on data files.

Conventions aligned with analysis.py (the main analysis):
  - log_sv base is natural log.
  - spectrum fits: linear in mode index, power-law (log-log) using log1p(mode_index),
    exponential (log vs mode index).
  - early/tail slopes: first int(n*0.1) modes and last n-int(n*0.8) modes, slope of
    log_sv vs mode_index (same style as slope_early5 / slope_late20 in analysis.py).
"""
import json
import os

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
OUT_JSON = os.path.join(OUT, "robustness_a.json")
OUT_PNG = os.path.join(FIG, "robustness_a.png")

# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
pt = pd.read_csv(TRAJ, sep="\t",
                 dtype={"parameter_index": "int64", "solver_family_id": "int64",
                        "progress_index": "int64"})
sv = pd.read_csv(SPECT, sep="\t", dtype={"mode_index": "int64"}).sort_values(
    "mode_index").reset_index(drop=True)

n_params = int(pt["parameter_index"].nunique())
n_fams = int(pt["solver_family_id"].nunique())
n_modes = int(sv["mode_index"].nunique())
print(f"params={n_params} families={n_fams} modes={n_modes}")

# ---------------------------------------------------------------
# (1) Alternative progress-scale definitions of convergence
# ---------------------------------------------------------------
piv = pt.pivot(index="parameter_index", columns=("solver_family_id", "progress_index"),
               values="normalised_distance_like_value").sort_index()
t = pd.DataFrame(index=piv.index)
for f in (0, 1):
    for d in (0, 1, 2):
        t[f"f{f}d{d}"] = piv[(f, d)]

X = np.array([0.0, 1.0, 2.0])  # progress_index

def quad_fitted_values(vals):
    """vals: (n,3) array [d0,d1,d2]. Return fitted values at p=0 and p=2
    from a quadratic fit of value vs progress_index per row (np.polyfit, deg 2).
    With 3 points the quadratic is exact interpolation, so fitted(0)=d0, fitted(2)=d2."""
    n = vals.shape[0]
    f0 = np.empty(n)
    f2 = np.empty(n)
    for i in range(n):
        coef = np.polyfit(X, vals[i], 2)  # [a, b, c] for a*p^2 + b*p + c
        f0[i] = coef[0]*0.0**2 + coef[1]*0.0 + coef[2]
        f2[i] = coef[0]*2.0**2 + coef[1]*2.0 + coef[2]
    return f0, f2

def metrics(f):
    d0 = t[f"f{f}d0"].to_numpy(dtype=float)
    d1 = t[f"f{f}d1"].to_numpy(dtype=float)
    d2 = t[f"f{f}d2"].to_numpy(dtype=float)
    out = {}
    # (a) total, (b) first-step, (c) last-step
    for name, chg in [("total", d2 - d0), ("first_step", d1 - d0), ("last_step", d2 - d1)]:
        out[name] = {
            "mean_change": float(np.mean(chg)),
            "frac_conv": float(np.mean(chg < 0)),
            "frac_flat": float(np.mean(np.abs(chg) < 0.01)),
        }
    # (d) quadratic fit of value vs progress_index (0,1,2)
    vals = np.column_stack([d0, d1, d2])
    q0, q2 = quad_fitted_values(vals)
    chg_q = q2 - q0
    out["quadratic"] = {
        "mean_change": float(np.mean(chg_q)),
        "frac_conv": float(np.mean(chg_q < 0)),
        "frac_flat": float(np.mean(np.abs(chg_q) < 0.01)),
        "fitted_at_0_exact_match_d0": bool(np.allclose(q0, d0)),
        "fitted_at_2_exact_match_d2": bool(np.allclose(q2, d2)),
    }
    return out

m0 = metrics(0)
m1 = metrics(1)

# Stability: all convergence fractions (4 definitions x 2 families) > 0.95
conv_fracs = [m0[k]["frac_conv"] for k in ("total", "first_step", "last_step", "quadratic")] + \
             [m1[k]["frac_conv"] for k in ("total", "first_step", "last_step", "quadratic")]
stable_above_095 = bool(np.all(np.array(conv_fracs) > 0.95))
print("conv fracs (fam0,fam1 by definition):")
for k in ("total", "first_step", "last_step", "quadratic"):
    print(f"  {k}: fam0={m0[k]['frac_conv']:.6f} fam1={m1[k]['frac_conv']:.6f}")
print("stable_above_095:", stable_above_095)

# ---------------------------------------------------------------
# (2) Alternative spectrum models
# ---------------------------------------------------------------
m_idx = sv["mode_index"].to_numpy(dtype=float)
ls = sv["log_singular_value"].to_numpy(dtype=float)
svv = np.exp(ls)
n = len(m_idx)

def ols_slope_R2(x, y):
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(c[1]), float(r2), c

# (a) linear in mode index: sv = a + b*i   (fit on raw scale)
lin_slope, lin_R2, lin_c = ols_slope_R2(m_idx, svv)

# (b) power-law (log-log): log(sv) = a + b*log1p(i)
pl_slope, pl_R2, pl_c = ols_slope_R2(np.log1p(m_idx), ls)

# (c) exponential (log vs index): log(sv) = a + b*i
exp_slope, exp_R2, exp_c = ols_slope_R2(m_idx, ls)

models = {
    "linear": {"slope": lin_slope, "R2": lin_R2},
    "powerlaw": {"slope": pl_slope, "R2": pl_R2},
    "exponential": {"slope": exp_slope, "R2": exp_R2},
}
best_model = max(models, key=lambda k: models[k]["R2"])
print("spectrum models:")
for k, v in models.items():
    print(f"  {k}: slope={v['slope']:.6f} R2={v['R2']:.6f}")
print("best_model:", best_model)

# early / tail slopes: slope of log_sv vs mode_index (same style as analysis.py)
early_n = int(n * 0.1)          # 423
tail_start = int(n * 0.8)       # 3385
early_slope, _, _ = ols_slope_R2(m_idx[:early_n], ls[:early_n])
tail_slope, _, _ = ols_slope_R2(m_idx[tail_start:], ls[tail_start:])
nonlinear_decay = bool(abs(early_slope - tail_slope) > 0.01)
print(f"early_n={early_n} tail_n={n - tail_start} early_slope={early_slope:.6f} "
      f"tail_slope={tail_slope:.6f} nonlinear_decay={nonlinear_decay}")

# ---------------------------------------------------------------
# (3) Parameter batch leave-out
# ---------------------------------------------------------------
n_batches = 4
order = np.arange(n_params)  # parameters are 0..n_params-1
batches_idx = [order[i * (n_params // n_batches):(i + 1) * (n_params // n_batches)]
               for i in range(n_batches)]
b_size = n_params // n_batches
print(f"batch size={b_size} batches={[[int(b.min()), int(b.max())] for b in batches_idx]}")

batch_results = {"n_batches": n_batches, "batch_size": b_size,
                 "batch_ranges": [], "batch_means_fam0": [], "batch_means_fam1": [],
                 "batch_conv_frac_fam0": [], "batch_conv_frac_fam1": [],
                 "batches": []}
all_conv = []
for i, bidx in enumerate(batches_idx):
    br = {"batch": i, "index_range": [int(bidx.min()), int(bidx.max())], "n": int(len(bidx))}
    for f in (0, 1):
        d0 = t.loc[bidx, f"f{f}d0"].to_numpy(dtype=float)
        d2 = t.loc[bidx, f"f{f}d2"].to_numpy(dtype=float)
        tot = d2 - d0
        br[f"family{f}"] = {
            "mean_total": float(np.mean(tot)),
            "frac_conv": float(np.mean(tot < 0)),
        }
        batch_results[f"batch_means_fam{f}"].append(float(np.mean(tot)))
        batch_results[f"batch_conv_frac_fam{f}"].append(float(np.mean(tot < 0)))
        all_conv.append(float(np.mean(tot < 0)))
    batch_results["batch_ranges"].append([int(bidx.min()), int(bidx.max())])
    batch_results["batches"].append(br)
    print(f"batch {i} [{br['index_range'][0]},{br['index_range'][1]}]: "
          f"fam0 mean_tot={br['family0']['mean_total']:.6f} conv={br['family0']['frac_conv']:.6f} "
          f"fam1 mean_tot={br['family1']['mean_total']:.6f} conv={br['family1']['frac_conv']:.6f}")

min_conv = float(np.min(all_conv))
max_conv = float(np.max(all_conv))
batch_stable = bool(np.all(np.array(all_conv) > 0.95))
print(f"min_conv_frac={min_conv:.6f} max_conv_frac={max_conv:.6f} batch_stable={batch_stable}")

# ---------------------------------------------------------------
# Build JSON
# ---------------------------------------------------------------
def pack(mf):
    return {
        "total": {
            "mean_change": mf["total"]["mean_change"],
            "frac_conv": mf["total"]["frac_conv"],
            "frac_flat": mf["total"]["frac_flat"],
        },
        "first_step": {
            "mean_change": mf["first_step"]["mean_change"],
            "frac_conv": mf["first_step"]["frac_conv"],
            "frac_flat": mf["first_step"]["frac_flat"],
        },
        "last_step": {
            "mean_change": mf["last_step"]["mean_change"],
            "frac_conv": mf["last_step"]["frac_conv"],
            "frac_flat": mf["last_step"]["frac_flat"],
        },
        "quadratic": {
            "mean_change": mf["quadratic"]["mean_change"],
            "frac_conv": mf["quadratic"]["frac_conv"],
            "frac_flat": mf["quadratic"]["frac_flat"],
            "fitted_at_0_exact_match_d0": mf["quadratic"]["fitted_at_0_exact_match_d0"],
            "fitted_at_2_exact_match_d2": mf["quadratic"]["fitted_at_2_exact_match_d2"],
        },
    }

result = {
    "meta": {
        "n_parameters": n_params, "n_families": n_fams, "n_progress": 3,
        "n_modes": n_modes, "n_batches": n_batches, "batch_size": b_size,
        "flat_threshold": 0.01,
    },
    "progress_scale": {
        "family0": pack(m0),
        "family1": pack(m1),
        "stable_above_095": stable_above_095,
        "all_conv_fracs": conv_fracs,
    },
    "spectrum_models": {
        "linear_slope": lin_slope, "linear_R2": lin_R2,
        "powerlaw_slope": pl_slope, "powerlaw_R2": pl_R2,
        "exponential_slope": exp_slope, "exponential_R2": exp_R2,
        "best_model": best_model,
        "early_n": early_n, "tail_n": n - tail_start,
        "early_slope": early_slope, "tail_slope": tail_slope,
        "nonlinear_decay": nonlinear_decay,
    },
    "batch_leaveout": {
        "n_batches": n_batches,
        "batch_size": b_size,
        "batch_means_fam0": batch_results["batch_means_fam0"],
        "batch_means_fam1": batch_results["batch_means_fam1"],
        "batch_conv_frac_fam0": batch_results["batch_conv_frac_fam0"],
        "batch_conv_frac_fam1": batch_results["batch_conv_frac_fam1"],
        "batch_ranges": batch_results["batch_ranges"],
        "min_conv_frac": min_conv,
        "max_conv_frac": max_conv,
        "stable": batch_stable,
        "batches": batch_results["batches"],
    },
}

with open(OUT_JSON, "w") as fh:
    json.dump(result, fh, indent=2, default=float)
print("WROTE", OUT_JSON)

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel 1: convergence fraction by definition x family
defs = ["total", "first_step", "last_step", "quadratic"]
xpos = np.arange(len(defs))
w = 0.35
ax = axes[0, 0]
ax.bar(xpos - w / 2, [m0[k]["frac_conv"] for k in defs], width=w,
       label="family0", color="tab:blue", alpha=0.85)
ax.bar(xpos + w / 2, [m1[k]["frac_conv"] for k in defs], width=w,
       label="family1", color="tab:red", alpha=0.85)
ax.axhline(0.95, color="grey", linestyle="--", lw=1, label="0.95 threshold")
ax.set_xticks(xpos)
ax.set_xticklabels(defs)
ax.set_ylabel("convergence fraction")
ax.set_title("Convergence fraction by progress-scale definition")
ax.legend()
ax.set_ylim(0, 1.02)

# Panel 2: spectrum model fits (log-log)
ax = axes[0, 1]
ax.loglog(m_idx + 1, svv, "k.", ms=1, label="sv")
ax.loglog(m_idx + 1, np.exp(pl_c[0] + pl_c[1] * np.log1p(m_idx)), "-",
          color="tab:orange", label=f"power-law (R2={pl_R2:.2f})")
ax.loglog(m_idx + 1, np.exp(exp_c[0] + exp_c[1] * m_idx), "-",
          color="tab:green", label=f"exponential (R2={exp_R2:.2f})")
# raw linear fit is ~0 for large index; show on a clipped scale
lin_pred = lin_c[0] + lin_c[1] * m_idx
lin_plot = lin_pred.clip(min=1e-30)
ax.loglog(m_idx + 1, lin_plot, "--", color="tab:grey",
          label=f"linear (R2={lin_R2:.2f})")
ax.set_xlabel("mode index")
ax.set_ylabel("singular value")
ax.set_title("Singular-value spectrum and model fits (best: " + best_model + ")")
ax.legend(fontsize=8)

# Panel 3: batch mean total change
ax = axes[1, 0]
bw = len(batches_idx)
xpos = np.arange(bw)
w = 0.35
ax.bar(xpos - w / 2, batch_results["batch_means_fam0"], width=w,
       label="family0", color="tab:blue", alpha=0.85)
ax.bar(xpos + w / 2, batch_results["batch_means_fam1"], width=w,
       label="family1", color="tab:red", alpha=0.85)
ax.axhline(0, color="grey", linestyle=":", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"b{i}" for i in range(bw)])
ax.set_ylabel("mean total change (d2-d0)")
ax.set_title("Mean total change by parameter batch")
ax.legend()

# Panel 4: batch convergence fraction
ax = axes[1, 1]
ax.bar(xpos - w / 2, batch_results["batch_conv_frac_fam0"], width=w,
       label="family0", color="tab:blue", alpha=0.85)
ax.bar(xpos + w / 2, batch_results["batch_conv_frac_fam1"], width=w,
       label="family1", color="tab:red", alpha=0.85)
ax.axhline(0.95, color="grey", linestyle="--", lw=1, label="0.95 threshold")
ax.set_xticks(xpos)
ax.set_xticklabels([f"b{i}" for i in range(bw)])
ax.set_ylabel("convergence fraction")
ax.set_title("Convergence fraction by parameter batch")
ax.legend()
ax.set_ylim(0, 1.02)

fig.suptitle("Robustness A: progress-scale, spectrum-model, and batch leave-out checks", y=0.99)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=120)
plt.close(fig)
print("WROTE", OUT_PNG)

# Print the exact numbers for the structured output
print(json.dumps({
    "progress_scale": {
        "total": [m0["total"]["mean_change"], m1["total"]["mean_change"],
                  m0["total"]["frac_conv"], m1["total"]["frac_conv"]],
        "first_step": [m0["first_step"]["mean_change"], m1["first_step"]["mean_change"],
                       m0["first_step"]["frac_conv"], m1["first_step"]["frac_conv"]],
        "last_step": [m0["last_step"]["mean_change"], m1["last_step"]["mean_change"],
                      m0["last_step"]["frac_conv"], m1["last_step"]["frac_conv"]],
        "quadratic": [m0["quadratic"]["mean_change"], m1["quadratic"]["mean_change"],
                      m0["quadratic"]["frac_conv"], m1["quadratic"]["frac_conv"]],
        "stable_above_095": stable_above_095,
    },
    "spectrum": {
        "linear_R2": lin_R2, "linear_slope": lin_slope,
        "powerlaw_R2": pl_R2, "powerlaw_slope": pl_slope,
        "exponential_R2": exp_R2, "exponential_slope": exp_slope,
        "best_model": best_model,
        "early_slope": early_slope, "tail_slope": tail_slope,
        "nonlinear_decay": nonlinear_decay,
    },
    "batch": {
        "n_batches": n_batches,
        "batch_means_fam0": batch_results["batch_means_fam0"],
        "batch_means_fam1": batch_results["batch_means_fam1"],
        "min_conv_frac": min_conv, "max_conv_frac": max_conv,
        "stable": batch_stable,
    },
}, indent=2))
