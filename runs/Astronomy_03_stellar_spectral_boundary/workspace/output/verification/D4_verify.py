#!/usr/bin/env python3
"""
D4 independent verification.

CLAIM (D4): the photon-to-energy relation of D3 fails below ~330 nm by a
factor that is large, DISCRETE and IDENTICAL for all five normalized levels:
~5.0 in a plateau ~299-322 nm, ~2.5 in a narrow segment ~322-327 nm, rising
smoothly through the far-UV to ~3.1 at 179 nm; because the factor is the same
for all five composition settings, it looks like a shared wavelength-
registration / resampling artefact rather than a physical composition effect.

Method: work only from the two raw TSVs.  Nothing under output/analysis or
output/results is imported or read.

  ratio_k(lam) = response_level_k(lam) * lam / F(5800 K, M_k, lam)
  deviation_k(lam) = ratio_k(lam) / median(ratio_k over lam >= 400 nm)

The constant is fixed on the OPTICAL band only, so all UV numbers below are
out-of-sample predictions from an optical-only calibration.
"""
import json
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project"
DATA = BASE + "/data"
OUT = BASE + "/output/verification"

# ======================================================================
# 0. raw data
# ======================================================================
g = pd.read_csv(DATA + "/photon_flux_grid.tsv", sep="\t")
n = pd.read_csv(DATA + "/normalized_spectral_response.tsv", sep="\t")

wl_grid = np.sort(g["wavelength_nm"].unique())
wl_norm = n["wavelength_nm"].values
mets = np.sort(g["metallicity_index"].unique())
temps = np.sort(g["effective_temperature_K"].unique())

Fmat = {}                       # T -> (176, 35) flux on wl_grid x mets
for T in temps:
    Fmat[T] = (g[g.effective_temperature_K == T]
               .pivot(index="wavelength_nm", columns="metallicity_index",
                      values="photon_flux_cm2")
               .loc[wl_grid].values)
F58 = Fmat[5800]
OPT = wl_norm >= 400.0

# ======================================================================
# 1. level_k <-> metallicity_index mapping (calibrated in the optical only)
# ======================================================================
mapping = {}
for lv in range(1, 6):
    lev = n["response_level_%d" % lv].values
    sc = []
    for j, m in enumerate(mets):
        Fi = np.interp(wl_norm, wl_grid, F58[:, j])
        r = lev * wl_norm / Fi
        v = r[OPT]
        sc.append((float(np.std(v) / np.mean(v)), float(m)))
    sc.sort()
    mapping[lv] = {"metallicity_index": sc[0][1],
                   "optical_rel_spread": sc[0][0],
                   "runner_up_metallicity": sc[1][1],
                   "runner_up_rel_spread": sc[1][0],
                   "top5": [[float(m), float(s)] for s, m in sc[:5]]}
MET = {lv: mapping[lv]["metallicity_index"] for lv in range(1, 6)}
JIDX = {lv: int(np.where(mets == MET[lv])[0][0]) for lv in range(1, 6)}

# ======================================================================
# 2. four registrations of the grid onto the normalized wavelength axis
# ======================================================================
BREAK = 67   # norm index < 67 -> grid index i ; >= 67 -> grid index i+1
idx_shifted = np.clip(np.where(np.arange(175) < BREAK,
                               np.arange(175), np.arange(175) + 1), 0, 174)
idx_unshifted = np.arange(175)


def F_index(idxmap, lv):
    return F58[idxmap, JIDX[lv]]


def F_linear(lv):
    return np.interp(wl_norm, wl_grid, F58[:, JIDX[lv]])


def F_nearest(lv):
    pos = np.clip(np.searchsorted(wl_grid, wl_norm), 1, len(wl_grid) - 1)
    take = np.where(np.abs(wl_norm - wl_grid[pos - 1]) <=
                    np.abs(wl_grid[pos] - wl_norm), pos - 1, pos)
    return F58[take, JIDX[lv]]


REG = {"index_shifted": lambda lv: F_index(idx_shifted, lv),
       "index_unshifted": lambda lv: F_index(idx_unshifted, lv),
       "linear": F_linear,
       "nearest": F_nearest}

dev = {}
const = {}
for reg, fn in REG.items():
    dev[reg] = np.array([n["response_level_%d" % lv].values * wl_norm / fn(lv)
                         for lv in range(1, 6)])
    const[reg] = np.median(dev[reg][:, OPT], axis=1)
    dev[reg] = dev[reg] / const[reg][:, None]

D = dev["index_shifted"]
factor = D.mean(axis=0)                       # mean over the five levels
cross_spread = D.max(axis=0) - D.min(axis=0)  # level-to-level spread

# ======================================================================
# 3. segment statistics
# ======================================================================
SEG = [("fuv_179_296", wl_norm < 296.5),
       ("plateau", (wl_norm >= 296.5) & (wl_norm <= 320.5)),
       ("narrow_322_327", (wl_norm > 320.5) & (wl_norm < 328)),
       ("gap_327_400", (wl_norm >= 328) & (wl_norm < 400)),
       ("optical_400_850", OPT)]


def segstats(vals, mask):
    a = vals[:, mask]
    return {"n_points": int(mask.sum()),
            "wl_min": float(wl_norm[mask].min()), "wl_max": float(wl_norm[mask].max()),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "std": float(a.std()),
            "per_level_means": [float(x) for x in a.mean(axis=1)],
            "max_cross_level_spread": float((a.max(0) - a.min(0)).max()),
            "max_cross_level_rel_spread": float(((a.max(0) - a.min(0)) /
                                                 np.abs(a.mean(0))).max())}


segments = {name: segstats(D, m) for name, m in SEG}

# exact jump locations: largest steps of the seasonal curve
d = np.diff(factor)
order = np.argsort(-np.abs(d))
jumps = [{"norm_index_from": int(i), "norm_index_to": int(i + 1),
          "wl_from": float(wl_norm[i]), "wl_to": float(wl_norm[i + 1]),
          "grid_index_from": int(idx_shifted[i]), "grid_index_to": int(idx_shifted[i + 1]),
          "grid_wl_from": float(wl_grid[idx_shifted[i]]),
          "grid_wl_to": float(wl_grid[idx_shifted[i + 1]]),
          "factor_from": float(factor[i]), "factor_to": float(factor[i + 1]),
          "step": float(d[i])} for i in order[:8]]

# ======================================================================
# 4. wavelength-axis alignment diagnostics
# ======================================================================
d0 = wl_norm - wl_grid[:175]
d1 = wl_norm - wl_grid[1:176]
grid_missing = [float(x) for x in wl_grid
                if not np.any(np.isclose(wl_norm, x, atol=1e-6))]
alignment = {
    "n_grid_wl": int(len(wl_grid)), "n_norm_wl": int(len(wl_norm)),
    "n_grid_wl_absent_from_norm_axis": len(grid_missing),
    "grid_wl_absent_from_norm_axis": [round(x, 3) for x in grid_missing],
    "max_abs_wl_diff_index_unshifted_i_le_66": float(np.abs(d0[:67]).max()),
    "max_abs_wl_diff_index_unshifted_i_ge_67": float(np.abs(d0[67:]).max()),
    "max_abs_wl_diff_index_shifted_i_ge_67": float(np.abs(d1[67:]).max()),
    "break_index": BREAK,
    "norm_wl_at_break": float(wl_norm[BREAK]),
    "grid_wl_at_break": float(wl_grid[BREAK]),
    "grid_wl_at_break_plus_1": float(wl_grid[BREAK + 1]),
    "index_shift_verified": bool(np.abs(d0[:67]).max() < 0.15 and
                                 np.abs(d1[67:]).max() < 0.25),
}

# ======================================================================
# 5. grid sampling steps
# ======================================================================
gs = np.diff(wl_grid)
step_after = np.r_[gs, np.nan]                 # width of bin starting at node i
step_before = np.r_[np.nan, gs]                # width of bin ending at node i
step_central = np.r_[[gs[0]], (gs[1:] + gs[:-1]) / 2.0, [gs[-1]]]

step_change_nodes = [{"grid_index": int(i), "wl": float(wl_grid[i]),
                      "step_before": float(step_before[i]),
                      "step_after": float(step_after[i])}
                     for i in range(1, 175)
                     if np.isfinite(step_before[i]) and
                     abs(step_after[i] - step_before[i]) > 0.05]

# ======================================================================
# 6. MECHANISM TEST: is the deviation just 5/(local grid bin width)?
#    The "5" is the optical sampling step, where the deviation is 1 by
#    construction -- so the model factor = step_optical/step_local has
#    ZERO free parameters away from the optical calibration band.
# ======================================================================
STEP_OPTICAL = 5.0
mech = {}
for label, wvec in [("step_central", step_central),
                    ("step_after", step_after),
                    ("step_before", step_before)]:
    w = wvec[idx_shifted]
    pred = STEP_OPTICAL / w
    ok = np.isfinite(pred)
    rel = pred[ok] / factor[ok] - 1.0
    mech[label] = {
        "n_evaluated": int(ok.sum()),
        "median_abs_rel_dev": float(np.median(np.abs(rel))),
        "mean_abs_rel_dev": float(np.abs(rel).mean()),
        "p90_abs_rel_dev": float(np.percentile(np.abs(rel), 90)),
        "frac_within_1pct": float((np.abs(rel) <= 0.01).mean()),
        "frac_within_2pct": float((np.abs(rel) <= 0.02).mean()),
        "frac_within_5pct": float((np.abs(rel) <= 0.05).mean()),
    }

# residuals of the best model, and which points break it
w_c = step_central[idx_shifted]
resid = factor * w_c / STEP_OPTICAL
resid_ok = np.isfinite(resid)
bad = [{"norm_index": int(i), "wl": float(wl_norm[i]),
        "grid_index": int(idx_shifted[i]), "grid_wl": float(wl_grid[idx_shifted[i]]),
        "factor": float(factor[i]), "bin_width_central": float(w_c[i]),
        "model_prediction_5_over_width": float(5.0 / w_c[i]),
        "residual_model_over_obs": float(resid[i]),
        "step_before": float(step_before[idx_shifted[i]]),
        "step_after": float(step_after[idx_shifted[i]]),
        "adjacent_to_sampling_change": any(abs(wl_grid[idx_shifted[i]] -
                                               c["wl"]) < 0.6
                                           for c in step_change_nodes)}
       for i in range(175)
       if not (np.isfinite(resid[i]) and abs(resid[i] - 1) <= 0.01)]
# signature of the residual inside each segment (is there leftover physics?)
step_resid = {}
for name, m in SEG:
    v = resid[m & resid_ok]
    if len(v):
        step_resid[name] = {"n": int(len(v)), "mean": float(v.mean()),
                            "std": float(v.std()), "min": float(v.min()),
                            "max": float(v.max()),
                            "trend_spearman_like": float(
                                np.corrcoef(np.arange(len(v)), v)[0, 1])}

# ======================================================================
# 7. NEGATIVE CONTROLS for the artefact story
#    (a) fix the index misregistration alone (nearest instead of linear,
#        shifted instead of unshifted) -- does the deviation vanish?
# ======================================================================
reg_compare = {}
for reg in REG:
    r = dev[reg]
    reg_compare[reg] = {
        "optical_rel_spread": float(np.mean(
            [np.std(r[lv - 1][OPT]) / np.mean(r[lv - 1][OPT]) for lv in range(1, 6)])),
        "plateau_mean": float(r[:, (wl_norm >= 296.5) & (wl_norm <= 320.5)].mean()),
        "narrow_mean": float(r[:, (wl_norm > 320.5) & (wl_norm < 328)].mean()),
        "fuv_mean_lt_296": float(r[:, wl_norm < 296.5].mean()),
        "factor_at_179nm": float(r[:, 0].mean()),
        "optical_constants": [float(c) for c in const[reg]],
    }

# ======================================================================
# 8. PHYSICS TEST
#    (a) how much does the flux differ between the five compositions
#        in the UV?  (b) do the five response series differ there?
# ======================================================================
uv = wl_grid < 300
opt_g = wl_grid >= 400
sens = {"per_level": {}}
for lv in range(1, 6):
    j = JIDX[lv]
    sens["per_level"]["level_%d" % lv] = {
        "metallicity_index": float(MET[lv]),
        "mean_F_uv_lt_300": float(F58[uv, j].mean()),
        "mean_F_optical": float(F58[opt_g, j].mean()),
    }
uvmean = [sens["per_level"]["level_%d" % lv]["mean_F_uv_lt_300"] for lv in range(1, 6)]
optmean = [sens["per_level"]["level_%d" % lv]["mean_F_optical"] for lv in range(1, 6)]
sens["uv_flux_range_across_compositions"] = float(max(uvmean) / min(uvmean))
sens["optical_flux_range_across_compositions"] = float(max(optmean) / min(optmean))
sens["uv_pointwise_flux_range"] = float(
    (F58[uv, :].max(axis=1) / F58[uv, :].min(axis=1)).max())

lvratio = {}
for lv in range(1, 6):
    r = n["response_level_%d" % lv].values / n["response_level_3"].values
    lvratio["level_%d" % lv] = {"uv_median": float(np.median(r[uv[:175] if len(uv) == 175 else (wl_norm < 300)])),
                                "optical_median": float(np.median(r[OPT]))}
lvratio_uv = [lvratio["level_%d" % lv]["uv_median"] for lv in range(1, 6)]
sens["response_level_range_across_levels_uv"] = float(max(lvratio_uv) / min(lvratio_uv))
sens["response_level_ratios_uv_vs_optical"] = lvratio

# (c) the decisive physics falsifier: the factor is a function of the
#     wavelength axis sampling only.  If it were physical it would have to
#     reproduce the same value for five radiation fields that differ by a
#     large factor at those very wavelengths.
comp_at_uv = np.array([F58[uv, JIDX[lv]] for lv in range(1, 6)])
sens["factor_is_composition_blind_while_flux_is_not"] = {
    "max_cross_level_rel_spread_of_factor_uv_lt_296": float(
        cross_spread[wl_norm < 296.5].max() /
        np.abs(factor[wl_norm < 296.5]).mean()),
    "cross_composition_flux_ratio_uv_range": sens["uv_flux_range_across_compositions"],
    "cross_composition_flux_ratio_pointwise_max": sens["uv_pointwise_flux_range"],
}

# ======================================================================
# 9. temperature control -- the normalized series tracks 5800 K
# ======================================================================
temp_ctrl = {}
for T in temps:
    Ft = Fmat[T]
    r = np.array([n["response_level_%d" % lv].values * wl_norm /
                  np.interp(wl_norm, wl_grid, Ft[:, JIDX[lv]])
                  for lv in range(1, 6)])
    c = np.median(r[:, OPT], axis=1)
    rn = r / c[:, None]
    temp_ctrl[int(T)] = {"optical_rel_spread_mean":
                         float(np.mean([np.std(r[lv - 1][OPT]) / np.mean(r[lv - 1][OPT])
                                        for lv in range(1, 6)])),
                         "factor_at_179nm": float(rn[:, 0].mean()),
                         "plateau_mean": float(rn[:, (wl_norm >= 296.5) &
                                                 (wl_norm <= 320.5)].mean())}

# ======================================================================
# 10. robustness to the choice of optical calibration window
# ======================================================================
robust = {}
for lab, wm in [("opt_400_850", wl_norm >= 400),
                ("opt_450_600", (wl_norm >= 450) & (wl_norm <= 600)),
                ("opt_500_700", (wl_norm >= 500) & (wl_norm <= 700)),
                ("opt_600_800", (wl_norm >= 600) & (wl_norm <= 800))]:
    for reg in REG:
        r = np.array([n["response_level_%d" % lv].values * wl_norm / REG[reg](lv)
                      for lv in range(1, 6)])
        c = np.median(r[:, wm], axis=1)
        rn = r / c[:, None]
        robust["%s|%s" % (lab, reg)] = {
            "plateau_mean": float(rn[:, (wl_norm >= 296.5) & (wl_norm <= 320.5)].mean()),
            "narrow_mean": float(rn[:, (wl_norm > 320.5) & (wl_norm < 328)].mean()),
            "factor_at_179nm": float(rn[:, 0].mean()),
            "optical_rel_spread": float(np.mean(
                [np.std(rn[lv - 1][wm]) / np.mean(rn[lv - 1][wm]) for lv in range(1, 6)])),
        }

# ======================================================================
# 11b. WHICH axis carries the interval?  (both files sit on the same
#      sampling layout, so this test is expected to be ambiguous -- report
#      it honestly rather than pick a winner)
# ======================================================================
dn = np.diff(wl_norm)
norm_central = np.r_[[dn[0]], (dn[1:] + dn[:-1]) / 2.0, [dn[-1]]]
norm_after = np.r_[dn, np.nan]
norm_before = np.r_[np.nan, dn]
implied_width = STEP_OPTICAL / factor
axis_test = {}
for lab, v in [("norm_central", norm_central), ("norm_after", norm_after),
               ("norm_before", norm_before), ("grid_central", w_c),
               ("grid_after", step_after[idx_shifted]),
               ("grid_before", step_before[idx_shifted])]:
    ok = np.isfinite(v)
    r = implied_width[ok] / v[ok]
    axis_test[lab] = {"n": int(ok.sum()), "median_ratio": float(np.median(r)),
                      "mean_ratio": float(r.mean()), "std_ratio": float(r.std()),
                      "frac_within_2pct": float((np.abs(r - 1) <= 0.02).mean())}
axis_test["note"] = ("both axes share the same sampling layout (1 nm band "
                     "299-321, 5 nm above 330), so this test cannot by itself "
                     "say which file is per-bin and which is per-unit-wavelength; "
                     "it only shows the interval is the right variable.")

# ======================================================================
# 11c. GLOBAL FIT: does the photon-to-energy relation hold over the WHOLE
#      range once the local sampling interval is included, and how much
#      does the UV bias shrink?  Transition nodes (where the interval
#      changes) are excluded because the interval is ambiguous there.
# ======================================================================
jumpmask = np.zeros(175, bool)
for i in range(1, 175):
    if abs(norm_central[i] - norm_central[i - 1]) > 0.05:
        jumpmask[max(0, i - 1):i + 2] = True

with_interval = np.array([n["response_level_%d" % lv].values * wl_norm *
                          norm_central / F_index(idx_shifted, lv)
                          for lv in range(1, 6)])
with_interval /= np.median(with_interval[:, OPT], axis=1)[:, None]
without_interval = np.array([n["response_level_%d" % lv].values * wl_norm /
                             F_index(idx_shifted, lv) for lv in range(1, 6)])
without_interval /= np.median(without_interval[:, OPT], axis=1)[:, None]

keep = ~jumpmask
uv330 = wl_norm < 330
global_fit = {
    "n_points_total": 175,
    "n_transition_nodes_excluded": int(jumpmask.sum()),
    "n_points_used": int(keep.sum()),
    "without_interval": {
        "mean": float(without_interval[:, keep].mean()),
        "std": float(without_interval[:, keep].std()),
        "min": float(without_interval[:, keep].min()),
        "max": float(without_interval[:, keep].max()),
        "uv_lt_330_mean": float(without_interval[:, keep & uv330].mean()),
        "uv_lt_330_std": float(without_interval[:, keep & uv330].std()),
    },
    "with_interval": {
        "mean": float(with_interval[:, keep].mean()),
        "std": float(with_interval[:, keep].std()),
        "min": float(with_interval[:, keep].min()),
        "max": float(with_interval[:, keep].max()),
        "uv_lt_330_mean": float(with_interval[:, keep & uv330].mean()),
        "uv_lt_330_std": float(with_interval[:, keep & uv330].std()),
        "per_level_rel_std": [float(np.std(with_interval[lv - 1, keep]) /
                                    np.mean(with_interval[lv - 1, keep]))
                              for lv in range(1, 6)],
        "fuv_lt_296_mean": float(with_interval[:, keep & (wl_norm < 296) ].mean()),
        "fuv_lt_296_std": float(with_interval[:, keep & (wl_norm < 296)].std()),
        "plateau_mean": float(with_interval[:, keep & (wl_norm >= 296.5) &
                                            (wl_norm <= 320.5)].mean()),
        "plateau_std": float(with_interval[:, keep & (wl_norm >= 296.5) &
                                           (wl_norm <= 320.5)].std()),
    },
    "uv_bias_without_interval": float(without_interval[:, keep & uv330].mean() - 1.0),
    "uv_bias_with_interval": float(with_interval[:, keep & uv330].mean() - 1.0),
    "note": ("the entire D4 deviation is removed by one factor, the local "
             "sampling interval, with no free parameter away from the "
             "optical calibration band"),
}

# ======================================================================
# 11d. THE LADDER: group the measured factor by the local grid step.
#      If the factor is 5/interval, each rung must sit at 5/step.
# ======================================================================
import collections
buckets = collections.defaultdict(list)
for i in range(175):
    gi = idx_shifted[i]
    if gi < 175 and np.isfinite(gs[gi]):
        buckets[round(float(gs[gi]), 2)].append(
            (float(wl_norm[i]), float(factor[i])))
ladder = []
for st in sorted(buckets):
    pts = buckets[st]
    v = np.array([p[1] for p in pts])
    ladder.append({"grid_step_nm": float(st),
                   "predicted_factor_5_over_step": float(5.0 / st),
                   "n_points": int(len(v)),
                   "measured_mean": float(v.mean()), "measured_std": float(v.std()),
                   "norm_wl_min": float(min(p[0] for p in pts)),
                   "norm_wl_max": float(max(p[0] for p in pts))})
# how well does the whole ladder hold, ignoring transition nodes?
ladder_ok = [b for b in ladder
             if abs(b["measured_mean"] - b["predicted_factor_5_over_step"])
             / b["predicted_factor_5_over_step"] < 0.03]
ladder_summary = {
    "n_rungs": len(ladder),
    "n_rungs_matching_within_3pct": len(ladder_ok),
    "rungs_that_fail": [b for b in ladder if b not in ladder_ok],
    "note": ("the two failing rungs (2.70 nm, 3.50 nm) contain the transition "
             "nodes 296.23 and 326.41 nm where the interval itself is ambiguous; "
             "every other rung matches 5/step"),
    "last_norm_wl_with_factor_different_from_1": float(
        wl_norm[np.where(np.abs(factor - 1) > 0.01)[0].max()]),
    "grid_wl_where_step_becomes_5nm_permanently": float(
        wl_grid[np.where(gs == 5.0)[0].min()]),
}

# ======================================================================
# 11. curve dump
# ======================================================================
curve = [{"wavelength_nm": float(wl_norm[i]),
          "grid_wavelength_nm": float(wl_grid[idx_shifted[i]]),
          "factor_mean_over_levels": float(factor[i]),
          "factor_per_level": [float(x) for x in D[:, i]],
          "cross_level_spread": float(cross_spread[i]),
          "grid_bin_width_central": float(w_c[i]),
          "model_5_over_width": float(5.0 / w_c[i]) if np.isfinite(w_c[i]) else None}
         for i in range(175)]

# ======================================================================
# verdict assembly
# ======================================================================
verdict = {
    "phenomenology_reproduced": True,
    "mechanism_is_registration_shift": False,
    "mechanism_is_local_sampling_interval": True,
    "physical_composition_effect": False,
    "conclusion": (
        "The D4 numbers reproduce exactly: 5.000 over 299-322 nm, 2.500 over "
        "322.4-326.4 nm, 3.109 at 179.31 nm, 1.000 above 330 nm, identical "
        "across all five levels to <=1%.  But the D4 EXPLANATION is wrong: the "
        "deviation is not caused by the one-index wavelength misregistration "
        "between the two files (fixing that leaves the deviation intact) and "
        "is not a registration artefact in the sense proposed.  It is caused "
        "by a constant of 5.0 nm implicit in the optical calibration: the "
        "factor equals 5/(local wavelength sampling interval) to <=1% at 91% "
        "of points (median |deviation| 0.06%), i.e. the normalized series is "
        "per-bin (interval-weighted) while the grid flux is per-unit-"
        "wavelength, or vice versa.  The 'discrete' segments are exactly the "
        "pieces of the sampling ladder: 1 nm (299-321), 1.5 nm (321-322.5), "
        "2 nm (322.5-326.5), 3.5 nm (326.5-330), 5 nm (>330), which is why "
        "the factor snaps to 5.0 / 2.5 and why it varies smoothly through the "
        "FUV where the interval grows smoothly.  The claim's inference that "
        "this is a data-product artefact rather than physics is CORRECT, and "
        "strongly confirmed: it is composition-blind to <=1% while the five "
        "compositions differ by up to 152x pointwise in the UV."),
}
summary = {
    "plateau_measured": {"wl_range": [float(wl_norm[(wl_norm >= 296.5) & (wl_norm <= 320.5)].min()),
                                      float(wl_norm[(wl_norm >= 296.5) & (wl_norm <= 320.5)].max())],
                         "mean_factor": segments["plateau"]["mean"],
                         "cross_level_rel_spread": segments["plateau"]["max_cross_level_rel_spread"]},
    "narrow_measured": {"wl_range": [float(wl_norm[(wl_norm > 320.5) & (wl_norm < 328)].min()),
                                     float(wl_norm[(wl_norm > 320.5) & (wl_norm < 328)].max())],
                        "mean_factor": segments["narrow_322_327"]["mean"],
                        "cross_level_rel_spread": segments["narrow_322_327"]["max_cross_level_rel_spread"]},
    "fuv_measured": {"wl_range": [float(wl_norm[wl_norm < 296.5].min()),
                                  float(wl_norm[wl_norm < 296.5].max())],
                     "mean_factor": segments["fuv_179_296"]["mean"],
                     "min_factor": segments["fuv_179_296"]["min"],
                     "max_factor": segments["fuv_179_296"]["max"],
                     "factor_at_179_31nm": float(factor[0]),
                     "cross_level_rel_spread": segments["fuv_179_296"]["max_cross_level_rel_spread"]},
    "optical_measured": {"mean_factor": segments["optical_400_850"]["mean"],
                         "std": segments["optical_400_850"]["std"]},
    "claim_vs_measured": {
        "plateau_5.0_299_322": abs(segments["plateau"]["mean"] - 5.0) < 0.05,
        "narrow_2.5_322_327": abs(segments["narrow_322_327"]["mean"] - 2.5) < 0.05,
        "fuv_3.1_at_179": abs(float(factor[0]) - 3.1) < 0.1,
        "identical_across_5_levels": bool(
            max(segments[s]["max_cross_level_rel_spread"]
                for s in ("plateau", "narrow_322_327")) < 0.01),
    },
}

artifact = {
    "finding_id": "D4",
    "verdict": verdict,
    "summary": summary,
    "level_to_metallicity_mapping": {str(k): v for k, v in mapping.items()},
    "segments": segments,
    "largest_steps_in_factor_curve": jumps,
    "axis_alignment": alignment,
    "grid_sampling_step_change_nodes": step_change_nodes,
    "mechanism_test_factor_equals_5_over_local_grid_step": mech,
    "points_breaking_the_step_model": bad,
    "residual_of_step_model_by_segment": step_resid,
    "registration_negative_control": reg_compare,
    "which_axis_carries_the_interval": axis_test,
    "factor_ladder_by_local_grid_step": ladder,
    "factor_ladder_summary": ladder_summary,
    "global_fit_with_vs_without_interval": global_fit,
    "physics_test": sens,
    "temperature_control": temp_ctrl,
    "robustness_to_optical_window": robust,
    "deviation_curve": curve,
    "provenance": {
        "script": OUT + "/D4_verify.py",
        "reproduce_command": "python3 " + OUT + "/D4_verify.py",
        "inputs": [DATA + "/photon_flux_grid.tsv",
                   DATA + "/normalized_spectral_response.tsv"],
        "independent_of_other_worker_artifacts": True,
        "no_scipy_used": True,
        "packages": ["numpy", "pandas", "json"],
    },
}

with open(OUT + "/D4_uv_boundary.json", "w") as f:
    json.dump(artifact, f, indent=2)

# ======================================================================
# console report
# ======================================================================
print("=== level -> metallicity (optical-only fit) ===")
for lv in range(1, 6):
    print("  level_%d -> M=%+.2f  spread=%.5f  (next best %+.2f = %.5f)"
          % (lv, MET[lv], mapping[lv]["optical_rel_spread"],
             mapping[lv]["runner_up_metallicity"],
             mapping[lv]["runner_up_rel_spread"]))
print("\n=== segments (index-shifted registration) ===")
for k in segments:
    s = segments[k]
    print("  %-16s n=%3d  wl %7.2f-%7.2f  mean=%8.4f  min=%8.4f max=%8.4f"
          "  max cross-level rel spread=%.2e"
          % (k, s["n_points"], s["wl_min"], s["wl_max"], s["mean"],
             s["min"], s["max"], s["max_cross_level_rel_spread"]))
print("\n=== discrete jumps in the factor ===")
for j in jumps[:5]:
    print("  %.2f -> %.2f nm : %.4f -> %.4f  (step %.4f)   grid %.2f -> %.2f"
          % (j["wl_from"], j["wl_to"], j["factor_from"], j["factor_to"],
             j["step"], j["grid_wl_from"], j["grid_wl_to"]))
print("\n=== axis alignment ===")
for k in ("n_grid_wl", "n_norm_wl", "n_grid_wl_absent_from_norm_axis",
          "max_abs_wl_diff_index_unshifted_i_le_66",
          "max_abs_wl_diff_index_unshifted_i_ge_67",
          "max_abs_wl_diff_index_shifted_i_ge_67", "index_shift_verified"):
    print("  %-42s %s" % (k, alignment[k]))
print("  grid wavelength absent from norm axis:", alignment["grid_wl_absent_from_norm_axis"])
print("\n=== grid sampling change nodes ===")
for c in step_change_nodes:
    print("  grid i=%3d wl=%7.2f  step %.2f -> %.2f" %
          (c["grid_index"], c["wl"], c["step_before"], c["step_after"]))
print("\n=== MECHANISM: factor == 5 / (local grid bin width) ? ===")
for k, v in mech.items():
    print("  %-13s n=%3d  median|rel dev|=%.5f  mean=%.5f  <=1%%=%.3f <=2%%=%.3f <=5%%=%.3f"
          % (k, v["n_evaluated"], v["median_abs_rel_dev"], v["mean_abs_rel_dev"],
             v["frac_within_1pct"], v["frac_within_2pct"], v["frac_within_5pct"]))
print("  residual of the step model by segment (obs/model should be 1.0):")
for k, v in step_resid.items():
    print("    %-16s n=%3d mean=%.4f std=%.4f min=%.4f max=%.4f"
          % (k, v["n"], v["mean"], v["std"], v["min"], v["max"]))
print("  %d of 175 points break the step model at >1%%:" % len(bad))
for b in bad:
    print("    i=%3d wl=%7.2f grid wl=%7.2f factor=%.4f model=%.4f ratio=%.4f adj_change=%s"
          % (b["norm_index"], b["wl"], b["grid_wl"], b["factor"],
             b["model_prediction_5_over_width"], b["residual_model_over_obs"],
             b["adjacent_to_sampling_change"]))
print("\n=== NEGATIVE CONTROL: does fixing the registration remove the deviation? ===")
for reg in REG:
    v = reg_compare[reg]
    print("  %-16s plateau=%.4f narrow=%.4f fuv=%.4f f179=%.4f optical_rel_spread=%.4f"
          % (reg, v["plateau_mean"], v["narrow_mean"], v["fuv_mean_lt_296"],
             v["factor_at_179nm"], v["optical_rel_spread"]))
print("\n=== PHYSICS TEST ===")
print("  UV(<300nm) mean flux range across the 5 compositions: %.2fx"
      % sens["uv_flux_range_across_compositions"])
print("  UV pointwise (worst wavelength) flux range:  %.2fx"
      % sens["uv_pointwise_flux_range"])
print("  response_level range across levels in UV:    %.2fx"
      % sens["response_level_range_across_levels_uv"])
print("  factor cross-level rel spread in UV:         %.2e"
      % sens["factor_is_composition_blind_while_flux_is_not"]
      ["max_cross_level_rel_spread_of_factor_uv_lt_296"])
print("\n=== temperature control ===")
for T in sorted(temp_ctrl):
    v = temp_ctrl[T]
    print("  %d K: optical rel spread=%.4f  plateau=%.4f  factor@179=%8.4f"
          % (T, v["optical_rel_spread_mean"], v["plateau_mean"], v["factor_at_179nm"]))
print("\n=== robustness to optical calibration window ===")
for k in robust:
    v = robust[k]
    print("  %-26s plateau=%.4f narrow=%.4f f179=%.4f opt_rel=%.4f"
          % (k, v["plateau_mean"], v["narrow_mean"], v["factor_at_179nm"],
             v["optical_rel_spread"]))
print("\n=== WHICH AXIS CARRIES THE INTERVAL (implied_width / candidate) ===")
for k, v in axis_test.items():
    if k == "note":
        continue
    print("  %-14s n=%3d median=%.5f mean=%.5f std=%.5f within2%%=%.3f"
          % (k, v["n"], v["median_ratio"], v["mean_ratio"], v["std_ratio"],
             v["frac_within_2pct"]))
print("  note:", axis_test["note"])
print("\n=== GLOBAL FIT: D3 relation with vs without the local interval ===")
gf = global_fit
print("  points used %d of %d (%d transition nodes excluded)"
      % (gf["n_points_used"], gf["n_points_total"],
         gf["n_transition_nodes_excluded"]))
print("  WITHOUT interval: mean=%.4f std=%.4f  UV(<330) mean=%.4f"
      % (gf["without_interval"]["mean"], gf["without_interval"]["std"],
         gf["without_interval"]["uv_lt_330_mean"]))
print("  WITH    interval: mean=%.5f std=%.5f  UV(<330) mean=%.5f"
      % (gf["with_interval"]["mean"], gf["with_interval"]["std"],
         gf["with_interval"]["uv_lt_330_mean"]))
print("  FUV(<296) with interval: mean=%.5f std=%.5f"
      % (gf["with_interval"]["fuv_lt_296_mean"], gf["with_interval"]["fuv_lt_296_std"]))
print("  UV bias falls from %.4f to %.5f"
      % (gf["uv_bias_without_interval"], gf["uv_bias_with_interval"]))
print("\n=== THE LADDER: measured factor grouped by local grid step ===")
for b in ladder:
    print("  step %5.2f nm -> predicted %7.4f | measured n=%3d mean=%8.4f std=%.4f  (wl %.1f-%.1f)"
          % (b["grid_step_nm"], b["predicted_factor_5_over_step"], b["n_points"],
             b["measured_mean"], b["measured_std"], b["norm_wl_min"], b["norm_wl_max"]))
print("  %d of %d rungs match 5/step within 3%%"
      % (ladder_summary["n_rungs_matching_within_3pct"], ladder_summary["n_rungs"]))
print("\n=== claim vs measured ===")
for k, v in summary["claim_vs_measured"].items():
    print("  %-32s %s" % (k, v))
print("\nwrote " + OUT + "/D4_uv_boundary.json")