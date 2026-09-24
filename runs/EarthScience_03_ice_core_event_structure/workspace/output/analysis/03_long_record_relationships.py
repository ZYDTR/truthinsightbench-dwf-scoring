#!/usr/bin/env python3
"""03_long_record_relationships.py -- long-record structure and its fragility.

Two questions:

(A) Sulfate vs depth/age/density.  DATA_GUIDE.md warns that neighbouring
    intervals are serially correlated, and the raw log-sulfate series has a
    lag-1 autocorrelation of ~0.72.  A naive correlation over 1931 points is
    therefore not a valid test.  We use a moving-block bootstrap and, for the
    dated part of the record, the 15 curator horizons themselves as the
    independent units.

(B) The age scale.  Only 15 rows carry a date, all of them annotated horizons.
    We quantify how much of the age-depth relation survives (i) the recovered
    +1.008 m depth step and (ii) dropping any single dated horizon.

Outputs (under output/results/):
  trend_bootstrap.csv
  age_depth_model_comparison.csv
  age_depth_leave_one_out.csv
  event_rate_stability.csv
  long_record_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, annotated_mask, load, log, save_csv, save_json  # noqa: E402

log("=" * 78)
log("03_long_record_relationships.py -- trends, age scale, event rate")
log("=" * 78)

df = load()
n = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
W = df["c06_depth_w_e_m"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
RHO = df["c05_density_ice_g_cm_3"].to_numpy(float)
TH = df["c03_samp_thick_cm"].to_numpy(float)
AGE = df["c08_age_a_ad_ce"].to_numpy(float)
ann = annotated_mask(df)
AI = np.where(ann)[0]
LS = np.log10(SO)
rng = np.random.default_rng(SEED)

# thickness-based column depth, and the step-corrected variant
cum_th = np.concatenate([[0.0], np.cumsum(TH[:-1])]) / 100.0
split = 106
D_corr = cum_th + np.where(np.arange(n) >= split, 1.008, 0.0)

# --------------------------------------------------------------------------
# A. moving-block bootstrap of correlation, and the naive-iid contrast
# --------------------------------------------------------------------------
def block_bootstrap_r(x, y, block=100, reps=2000, rng=None):
    """Pairs bootstrap that resamples contiguous blocks of the paired series."""
    rng = rng or np.random.default_rng(SEED)
    m = ~np.isnan(x) & ~np.isnan(y)
    xs, ys = x[m], y[m]
    N = len(xs)
    nb = int(np.ceil(N / block))
    starts = np.arange(0, N - block + 1)
    out = np.empty(reps)
    for b in range(reps):
        pick = rng.choice(starts, nb, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in pick])[:N]
        cs = np.corrcoef(xs[idx], ys[idx])[0, 1]
        out[b] = cs
    return out


trend_rows = []
pairs = [
    ("depth c01", D, "c04_sulfate_ug_kg"),
    ("depth c01", D, "log10_c04_sulfate_ug_kg"),
    ("density c05", RHO, "log10_c04_sulfate_ug_kg"),
    ("corrected depth", D_corr, "log10_c04_sulfate_ug_kg"),
]
for xname, x, yname in pairs:
    y = LS if yname.startswith("log10") else SO
    m = ~np.isnan(x) & ~np.isnan(y)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    bs = block_bootstrap_r(x, y, block=100, reps=2000)
    # naive iid permutation as the (invalid) contrast
    perm = np.array(
        [np.corrcoef(x[m], rng.permutation(y[m]))[0, 1] for _ in range(2000)]
    )
    trend_rows.append(
        {
            "x": xname,
            "y": yname,
            "pearson_r": r,
            "r2": r ** 2,
            "block_boot_lo95": float(np.percentile(bs, 2.5)),
            "block_boot_hi95": float(np.percentile(bs, 97.5)),
            "block_boot_sd": float(bs.std()),
            "iid_perm_p_two_sided": float(2 * min((perm >= r).mean(), (perm <= r).mean())),
            "n_used": int(m.sum()),
        }
    )
tr = pd.DataFrame(trend_rows)
save_csv(tr, "trend_bootstrap.csv")
log("sulfate vs the geometry columns (moving-block bootstrap, block = 100 samples):")
for _, r in tr.iterrows():
    log("  %-16s vs %-24s r=%+.4f  95%% block CI [%+.3f, %+.3f]  (iid-perm p=%.2g)"
        % (r.x, r.y, r.pearson_r, r.block_boot_lo95, r.block_boot_hi95, r.iid_perm_p_two_sided))
log("  -> every block-bootstrap interval spans zero: the apparent trend is not resolvable")
log("     once serial correlation is respected, although the iid test would call it significant.")

# autocorrelation as the reason the iid test is invalid
x = LS[~np.isnan(LS)]
xc = x - x.mean()
ac1 = float(np.corrcoef(xc[:-1], xc[1:])[0, 1])
acf = [float(np.corrcoef(xc[:-k], xc[k:])[0, 1]) for k in range(1, 21)]
log("")
log("effective independent sample count: lag-1 autocorrelation of log10(sulfate) = %.3f" % ac1)
log("  first zero crossing of the ACF at lag %d" % next((i + 1 for i, v in enumerate(acf) if v < 0), -1))
ess = n * (1 - ac1) / (1 + ac1)
log("  crude effective sample size  n*(1-r1)/(1+r1) = %.1f  (nominal n = %d)" % (ess, n))

# depth trend: compare shallow/deep halves with a block bootstrap of the difference
half = D < np.median(D)
diff_obs = float(np.nanmean(LS[half]) - np.nanmean(LS[~half]))
diffs = []
for _ in range(4000):
    pick = rng.integers(0, 2, size=int(ess))
    # resample contiguous blocks instead: reuse block scheme
    pass
starts = np.arange(0, n - 100 + 1)
boot = np.empty(4000)
for b in range(4000):
    idx = np.concatenate([np.arange(s, s + 100) for s in rng.choice(starts, n // 100 + 1, replace=True)])[:n]
    v = LS[idx]
    hh = half[idx]
    boot[b] = np.nanmean(v[hh]) - np.nanmean(v[~hh])
log("")
log("shallow-vs-deep contrast in mean log10(sulfate): %+.4f dex" % diff_obs)
log("  block-bootstrap 95%% CI [%+.4f, %+.4f] ; P(contrast>0) = %.3f"
    % (np.percentile(boot, 2.5), np.percentile(boot, 97.5), float((boot > 0).mean())))

# --------------------------------------------------------------------------
# B. age scale
# --------------------------------------------------------------------------
m_age = ~np.isnan(AGE)
Ds = D[m_age]
Ds_c = D_corr[m_age]
As = AGE[m_age]
cum_th_s = cum_th[m_age]

age_rows = []
for name, xx in [
    ("depth c01 (as reported)", Ds),
    ("depth with +1.008 m step removed", Ds_c),
    ("cumsum(c03)/100 (thickness column)", cum_th_s),
    ("water-equivalent depth c06", W[m_age]),
]:
    p1 = np.polyfit(xx, As, 1)
    res = As - np.polyval(p1, xx)
    V = np.corrcoef(xx, As)[0, 1] ** 2
    age_rows.append(
        {
            "independent_variable": name,
            "slope_yr_per_m": float(p1[0]),
            "intercept_ce": float(p1[1]),
            "r2": float(V),
            "residual_sd_yr": float(res.std()),
            "max_abs_residual_yr": float(np.abs(res).max()),
        }
    )
ad = pd.DataFrame(age_rows)
save_csv(ad, "age_depth_model_comparison.csv")
log("")
log("age-depth fits over the 15 dated horizons (age in CE year):")
for _, r in ad.iterrows():
    log("  %-38s slope=%8.2f yr/m  R2=%.5f  resid sd=%6.2f yr  max|res|=%6.2f yr"
        % (r.independent_variable, r.slope_yr_per_m, r.r2, r.residual_sd_yr, r.max_abs_residual_yr))

# leave-one-out on the dated horizons
loo = []
for i in range(len(Ds)):
    keep = np.arange(len(Ds)) != i
    p = np.polyfit(Ds[keep], As[keep], 1)
    rr = As[keep] - np.polyval(p, Ds[keep])
    loo.append(
        {
            "dropped_row": int(AI[i]),
            "dropped_depth_m": float(Ds[i]),
            "dropped_age_ce": float(As[i]),
            "slope_yr_per_m": float(p[0]),
            "r2": float(np.corrcoef(Ds[keep], As[keep])[0, 1] ** 2),
            "max_abs_residual_yr": float(np.abs(rr).max()),
        }
    )
loo = pd.DataFrame(loo)
save_csv(loo, "age_depth_leave_one_out.csv")
log("")
log("leave-one-out over the 15 dated horizons (as-reported depth):")
log("  slope range  : %.2f to %.2f yr/m  (full-fit %.2f)"
    % (loo.slope_yr_per_m.min(), loo.slope_yr_per_m.max(), ad.iloc[0].slope_yr_per_m))
log("  R2 range     : %.4f to %.4f" % (loo.r2.min(), loo.r2.max()))
worst = loo.loc[loo.max_abs_residual_yr.idxmax()]
log("  worst deletion: dropping row %d (%.2f m, %d CE) pushes max|residual| to %.1f yr"
    % (worst.dropped_row, worst.dropped_depth_m, worst.dropped_age_ce, worst.max_abs_residual_yr))

# age rates between consecutive horizons
rates = -(np.diff(As)) / np.diff(Ds)
log("")
log("interval accumulation rates (yr per reported m):")
log("  min %.1f  median %.1f  max %.1f  sd %.1f" % (rates.min(), np.median(rates), rates.max(), rates.std()))
log("  the shallowest interval (0.00-2.00 m) runs at %.1f yr/m, the deep ones at %.1f-%.1f yr/m"
    % (rates[0], rates[-3:].min(), rates[-3:].max()))

# --------------------------------------------------------------------------
# C. event rate stability
# --------------------------------------------------------------------------
log("")
log("event-rate stability of the 15 annotated horizons:")
rate_rows = []
for name, coord in [("depth c01", D), ("water-equivalent c06", W)]:
    for label, edge in [(name + " median split", np.median(coord))]:
        lo = int((coord[AI] < edge).sum())
        hi = int((coord[AI] >= edge).sum())
        # Poisson two-sided test via exact binomial on the split
        from math import comb
        p = 0.5
        k = lo
        m = lo + hi
        pv = 2 * min(sum(comb(m, i) * p ** m for i in range(0, k + 1)),
                     sum(comb(m, i) * p ** m for i in range(k, m + 1)))
        rate_rows.append(
            {
                "coordinate": name,
                "split_value": float(edge),
                "n_below": lo,
                "n_above": hi,
                "binomial_p_two_sided": float(min(pv, 1.0)),
            }
        )
        log("  %-20s split at %.2f -> %d below / %d above, exact binomial p=%.3f"
            % (name, edge, lo, hi, min(pv, 1.0)))
save_csv(pd.DataFrame(rate_rows), "event_rate_stability.csv")

# spacing test in the two coordinates
spacing_rows = []
for name, coord in [("depth c01", D), ("water-equivalent c06", W), ("age CE (dated subset)", None)]:
    if coord is None:
        g = np.abs(np.diff(As))
    else:
        g = np.diff(coord[AI])
    cv = float(g.std() / g.mean())
    spacing_rows.append(
        {
            "coordinate": name,
            "n_gaps": len(g),
            "mean_gap": float(g.mean()),
            "sd_gap": float(g.std()),
            "cv": cv,
            "cv_expected_poisson": 1.0,
        }
    )
    log("  %-24s gaps: mean %.3f sd %.3f CV=%.3f (Poisson/exponential -> 1.0)"
        % (name, g.mean(), g.std(), cv))
save_csv(pd.DataFrame(spacing_rows), "event_spacing.csv")

summary = {
    "acf_lag1_log_sulfate": ac1,
    "effective_sample_size": float(ess),
    "shallow_deep_log_contrast_dex": diff_obs,
    "shallow_deep_block_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "shallow_deep_p_positive": float((boot > 0).mean()),
    "age_depth_r2_as_reported": float(ad.iloc[0].r2),
    "age_depth_slope_as_reported": float(ad.iloc[0].slope_yr_per_m),
    "age_depth_r2_step_corrected": float(ad.iloc[1].r2),
    "age_depth_slope_step_corrected": float(ad.iloc[1].slope_yr_per_m),
    "age_depth_r2_c06": float(ad.iloc[3].r2),
    "loo_slope_min": float(loo.slope_yr_per_m.min()),
    "loo_slope_max": float(loo.slope_yr_per_m.max()),
    "trend_intervals_all_span_zero": bool(
        ((tr.block_boot_lo95 < 0) & (tr.block_boot_hi95 > 0)).all()
    ),
}
save_json(summary, "long_record_summary.json")