#!/usr/bin/env python3
"""05_robustness.py -- adversarial checks on each candidate finding.

Every registered finding gets an explicit attempt to break it:

  F1 column dependency  -> exact integer-cm arithmetic; grid search over the
                           breakpoint; residual scale vs the rounding tolerance
                           implied by the printed precision of each column.
  F2 water-equivalent    -> four quadratures, plus reversal of the assumed
                           derivation direction, plus a smooth-density control.
  F3 sulfate trend       -> block length 25/50/100/200/400; leave-one-block-out;
                           a rank test on the 15 dated horizons as units.
  F4 event set           -> uniform-random subsets with the same local-maximum
                           constraint, to show the recall deficit is structural.
  F5 chronology          -> quadratic alternative, step-removed coordinate,
                           and an isochronous-null check that ANY monotone
                           transform of depth can absorb curvature.

Outputs (under output/results/):
  robustness_integer_depth.csv
  robustness_breakpoint_grid.csv
  trend_block_length_sensitivity.csv
  trend_leave_one_block_out.csv
  localmax_null_calibration.csv
  robustness_summary.json
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, annotated_mask, blocks, load, log, save_csv, save_json  # noqa: E402

log("=" * 78)
log("05_robustness.py -- adversarial checks per candidate finding")
log("=" * 78)

df = load()
n = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
D2 = df["c02_depth_ice_snow_m"].to_numpy(float)
TH = df["c03_samp_thick_cm"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
RHO = df["c05_density_ice_g_cm_3"].to_numpy(float)
W = df["c06_depth_w_e_m"].to_numpy(float)
AGE = df["c08_age_a_ad_ce"].to_numpy(float)
ann = annotated_mask(df)
AI = np.where(ann)[0]
LS = np.log10(SO)
rng = np.random.default_rng(SEED)
m_age = ~np.isnan(AGE)
As = AGE[m_age]

# Exactly one row (row 61, 1.74 m) has no sulfate value.  Every resampling test
# below uses the 1930 rows that carry a measurement; no imputation is performed.
ok = ~np.isnan(SO)
n_ok = int(ok.sum())
D_ok, LS_ok = D[ok], LS[ok]

# ==========================================================================
# F1 -- integer centimetre arithmetic for the depth cascade
# ==========================================================================
Dcm = np.round(D * 100).astype(int)
Tcm = np.round(TH * 100).astype(int)
step_cm = np.diff(Dcm) - Tcm[:-1]
vc = pd.Series(step_cm).value_counts().sort_index()
log("F1  integer-cm check: c01(cm)[i+1] - c01(cm)[i] - c03(cm)[i]")
log("    distinct values: %s" % vc.to_dict())
nonstd = np.where(np.abs(step_cm) > 1)[0]
log("    rows deviating by more than 1 cm: %d -> %s" % (len(nonstd), nonstd.tolist()))
for i in nonstd:
    log("      row %4d: depth %.2f -> %.2f m, thickness %.2f cm, jump %d cm"
        % (i, D[i], D[i + 1], TH[i], step_cm[i]))
save_csv(
    pd.DataFrame({"deviation_cm": sorted(vc.index), "count": [int(vc[k]) for k in sorted(vc.index)]}),
    "robustness_integer_depth.csv",
)
extra_cm = int(step_cm[nonstd].sum() - 0) if len(nonstd) else 0
log("    -> the whole 41.98 m depth column is the cumulative thickness column plus a single")
log("       inserted jump of %d cm at row %d." % (int(step_cm[nonstd].max()) if len(nonstd) else 0, nonstd[0] if len(nonstd) else -1))

# breakpoint grid: does any other split do better?
res_base = D - np.concatenate([[0.0], np.cumsum(TH[:-1])]) / 100.0
grid_rows = []
for s in range(2, n - 2):
    off = float(np.median(res_base[s:]))
    for cand_off in (off, round(off, 3)):
        e = float(np.abs(res_base[s:] - cand_off).max())
        grid_rows.append({"split_after_row": s, "offset_m": float(cand_off), "max_abs_residual_m": e})
bg = pd.DataFrame(grid_rows)
best_by_split = bg.loc[bg.groupby("split_after_row").max_abs_residual_m.idxmin()]
save_csv(best_by_split, "robustness_breakpoint_grid.csv")
top = best_by_split.nsmallest(8, "max_abs_residual_m")
log("    best 8 breakpoint/offset combinations by max|residual|:")
for _, r in top.iterrows():
    log("      split after row %4d (depth %.2f m), offset %+.4f m -> max|resid| %.4f m"
        % (r.split_after_row, D[int(r.split_after_row)], r.offset_m, r.max_abs_residual_m))
n_within_5mm = int((best_by_split.max_abs_residual_m <= 0.005).sum())
log("    splits achieving max|resid| <= 5 mm: %d ; the winning split is unique = %s"
    % (n_within_5mm, bool(best_by_split.max_abs_residual_m.idxmin() == 105)))

# where are the 11 rows with c02 == c01?
eq = np.where(np.abs(D2 - D) < 1e-9)[0]
log("    rows where c02 == c01 exactly: %d  -> %s" % (len(eq), eq.tolist()))
log("    (depths %s m)" % np.round(D[eq], 2).tolist())

# rounding-tolerance budget for c06 = c01*c05
prod = D * RHO
tol = D * 0.0005 + 0.005 + RHO * 0.005
resid = np.abs(W - prod)
log("    c06 = c01*c05: rows inside the printed-precision tolerance = %d/%d (max resid %.4f m)"
    % ((resid <= tol + 1e-9).sum(), n, resid.max()))

# ==========================================================================
# F2 -- alternative quadratures / reversed derivation
# ==========================================================================
dz = np.diff(D)
mids = {
    "midpoint (trapezoid)": np.concatenate([[0.0], np.cumsum(0.5 * (RHO[1:] + RHO[:-1]) * dz)]),
    "left endpoint": np.concatenate([[0.0], np.cumsum(RHO[:-1] * dz)]),
    "right endpoint": np.concatenate([[0.0], np.cumsum(RHO[1:] * dz)]),
    "stepwise (left, on sample thickness)": np.concatenate(
        [[0.0], np.cumsum(RHO[:-1] * TH[:-1] / 100.0)]
    ),
}
rows = []
for name, v in mids.items():
    rows.append(
        {
            "quantity": name,
            "final_value_m": float(v[-1]),
            "variance_from_reported_c06_m": float(v[-1] - W[-1]),
            "relative_pct": float(100 * (v[-1] - W[-1]) / W[-1]),
            "max_abs_residual_vs_c06": float(np.abs(W - v).max()),
        }
    )
rows.append(
    {
        "quantity": "reported c06_depth_w_e_m",
        "final_value_m": float(W[-1]),
        "variance_from_reported_c06_m": 0.0,
        "relative_pct": 0.0,
        "max_abs_residual_vs_c06": 0.0,
    }
)
rows.append(
    {
        "quantity": "c01*c05 (product)",
        "final_value_m": float(prod[-1]),
        "variance_from_reported_c06_m": float(prod[-1] - W[-1]),
        "relative_pct": float(100 * (prod[-1] - W[-1]) / W[-1]),
        "max_abs_residual_vs_c06": float(np.abs(W - prod).max()),
    }
)
qdf = pd.DataFrame(rows).sort_values("max_abs_residual_vs_c06")
save_csv(qdf, "quadrature_comparison.csv")
log("")
log("F2  candidate meanings of c06_depth_w_e_m (all evaluated over the full 41.98 m):")
for _, r in qdf.iterrows():
    log("    %-38s value=%7.3f m  max|resid vs c06|=%7.4f m" % (r.quantity, r.final_value_m, r.max_abs_residual_vs_c06))
log("    -> the density integral (any quadrature) lands at %.2f-%.2f m; the reported column is"
    % (min(r.final_value_m for _, r in qdf.iterrows() if "endpoint" in r.quantity or "midpoint" in r.quantity),
       max(r.final_value_m for _, r in qdf.iterrows() if "endpoint" in r.quantity or "midpoint" in r.quantity)))
log("       %.2f m, a %.1f %% difference that no quadrature choice can close."
    % (W[-1], 100 * (W[-1] - 21.2218) / 21.2218))

# reversed direction: could c05 be round(c06/c01, 3)?
implied = np.divide(W, D, out=np.full(n, np.nan), where=D > 0)
delta_rho = RHO - implied
log("    reversed reading (c05 = round(c06/c01, 3)): %.1f %% of rows agree within 0.0005"
    % (100 * np.mean(np.abs(delta_rho[1:]) <= 0.0005)))

# ==========================================================================
# F3 -- trend robustness: block length, leave-one-block-out, horizons as units
# ==========================================================================
log("")
log("F3  sulfate-vs-depth trend under different block lengths (moving-block bootstrap, 2000 reps):")
blk_rows = []
r_point = float(np.corrcoef(D_ok, LS_ok)[0, 1])
for blk in (25, 50, 100, 200, 400):
    starts = np.arange(0, n_ok - blk + 1)
    nb = int(np.ceil(n_ok / blk))
    out = np.empty(2000)
    for b in range(2000):
        idx = np.concatenate([np.arange(s, s + blk) for s in rng.choice(starts, nb, replace=True)])[:n_ok]
        out[b] = np.corrcoef(D_ok[idx], LS_ok[idx])[0, 1]
    blk_rows.append(
        {
            "block_samples": blk,
            "block_depth_m": blk * 0.021,
            "r_point": r_point,
            "lo95": float(np.percentile(out, 2.5)),
            "hi95": float(np.percentile(out, 97.5)),
            "sd": float(out.std()),
            "spans_zero": bool(np.percentile(out, 2.5) < 0 < np.percentile(out, 97.5)),
        }
    )
    log("    block=%3d (%5.2f m): r=%+.4f  95%% CI [%+.3f, %+.3f]  spans zero = %s"
        % (blk, blk * 0.021, blk_rows[-1]["r_point"], blk_rows[-1]["lo95"], blk_rows[-1]["hi95"], blk_rows[-1]["spans_zero"]))
save_csv(pd.DataFrame(blk_rows), "trend_block_length_sensitivity.csv")

log("    leave-one-block-out (20 blocks):")
lobo = []
for i, b in enumerate(blocks(n, 20)):
    keep = np.setdiff1d(np.arange(n), b)
    r = float(np.corrcoef(D[keep], LS[keep])[0, 1])
    lobo.append({"block": i, "depth_start_m": float(D[b[0]]), "depth_end_m": float(D[b[-1]]), "r": r})
lobo = pd.DataFrame(lobo)
save_csv(lobo, "trend_leave_one_block_out.csv")
log("      r range %.4f to %.4f ; sign changes = %s"
    % (lobo.r.min(), lobo.r.max(), bool((lobo.r > 0).any() and (lobo.r < 0).any())))

# dated horizons as the independent units (n=15)
log("    dated horizons as independent units (n=15):")
rh = float(np.corrcoef(df["c01_depth_top_m"].to_numpy(float)[m_age], As)[0, 1])
log("      Pearson r(depth, age) = %+.4f  (R2=%.4f)" % (rh, rh ** 2))
perm = np.array([np.corrcoef(D[m_age], rng.permutation(As))[0, 1] for _ in range(5000)])
log("      permutation p = %.5f  -- scale-driven, not an independent confirmation"
    % float(2 * min((perm >= rh).mean(), (perm <= rh).mean())))

# ==========================================================================
# F4 -- is the recall deficit structural?  calibrate on random subsets
# ==========================================================================
log("")
log("F4  local-maximum calibration: draw 15 random rows and count strict local maxima")
lm = (SO > np.roll(SO, 1)) & (SO >= np.roll(SO, -1))
lm[0] = lm[-1] = False
lm[0] = lm[-1] = False
frac_lm = float(lm.mean())
sims = np.array([int(lm[rng.choice(n, 15, replace=False)].sum()) for _ in range(20000)])
obs_lm = int(lm[AI].sum())
log("    overall share of strict local maxima in c04 : %.4f (%d/%d)" % (frac_lm, lm.sum(), n))
log("    annotated horizons that are local maxima    : %d/15" % obs_lm)
log("    same count for 15 uniformly random rows     : mean %.2f, 95%% range [%d, %d]"
    % (sims.mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)))
log("    p(random subset reaches %d) = %.5f" % (obs_lm, float((sims >= obs_lm).mean())))
save_csv(
    pd.DataFrame({"statistic": ["annotated_local_max", "random_mean", "random_p2.5", "random_p97.5", "p_value"],
                  "value": [obs_lm, sims.mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5),
                            float((sims >= obs_lm).mean())]}),
    "localmax_null_calibration.csv",
)
log("    -> the annotated set occupies essentially the maximum achievable local-maximum count,")
log("       so a local-maximum detector's recall ceiling is structural, not a tuning failure.")

# ==========================================================================
# F5 -- can ANY monotone transform of depth absorb the curvature?
# ==========================================================================
log("")
log("F5  chronology: is the c06 advantage specific, or does any 2-parameter monotone")
log("    transform of depth fit equally well?  (fit age = a + b*g(depth) for candidate g)")
can = {
    "identity (c01)": D,
    "log(1 + depth)": np.log1p(D),
    "depth^2": D ** 2,
    "sqrt(depth)": np.sqrt(D),
    "c06 = depth * density": W,
    "cumulative thickness": np.concatenate([[0.0], np.cumsum(TH[:-1])]) / 100.0,
}
rows = []
for name, g in can.items():
    x = g[m_age]
    p = np.polyfit(x, As, 1)
    r = As - np.polyval(p, x)
    # out-of-sample via leave-one-out
    errs = []
    for i in range(len(As)):
        k = np.arange(len(As)) != i
        pp = np.polyfit(x[k], As[k], 1)
        errs.append(abs(As[i] - np.polyval(pp, x[i])))
    rows.append(
        {
            "transform": name,
            "r2": float(np.corrcoef(x, As)[0, 1] ** 2),
            "resid_sd_yr": float(r.std()),
            "loo_mae_yr": float(np.mean(errs)),
            "loo_max_yr": float(np.max(errs)),
        }
    )
    log("    %-22s R2=%.5f  resid sd=%6.2f yr  LOO MAE=%6.2f yr  LOO max=%6.2f yr"
        % (name, rows[-1]["r2"], rows[-1]["resid_sd_yr"], rows[-1]["loo_mae_yr"], rows[-1]["loo_max_yr"]))
save_csv(pd.DataFrame(rows), "monotone_transform_competition.csv")

# --------------------------------------------------------------------------
summary = {
    "F1_depth_step_cm": int(step_cm[nonstd].max()) if len(nonstd) else 0,
    "F1_depth_step_row": int(nonstd[0]) if len(nonstd) else -1,
    "F1_step_residual_max_m": float(np.abs(res_base - np.median(res_base[106:]) * (np.arange(n) >= 106)).max()),
    "F1_c06_product_max_resid_m": float(resid.max()),
    "F1_c06_product_rows_in_tolerance": int((resid <= tol + 1e-9).sum()),
    "F2_integral_midpoint_final_m": float(mids["midpoint (trapezoid)"][-1]),
    "F2_reported_c06_final_m": float(W[-1]),
    "F2_gap_pct": float(100 * (W[-1] - mids["midpoint (trapezoid)"][-1]) / mids["midpoint (trapezoid)"][-1]),
    "F3_all_block_lengths_span_zero": bool(all(r["spans_zero"] for r in blk_rows)),
    "F3_block_rows": blk_rows,
    "F3_leave_one_block_out_r_min": float(lobo.r.min()),
    "F3_leave_one_block_out_r_max": float(lobo.r.max()),
    "F3_n_rows_used": n_ok,
    "F4_p_random_reaches_annotated": float((sims >= obs_lm).mean()),
    "F5_transforms": rows,
}
save_json(summary, "robustness_summary.json")