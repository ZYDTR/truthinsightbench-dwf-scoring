#!/usr/bin/env python3
"""Independent verification of FINDING D1.

Claim: The response of photon flux to the ordered composition (metallicity)
index reverses sign at a wavelength boundary near 420-470 nm, and that
boundary is essentially the SAME in all three effective-temperature slices.
Blueward of the boundary, raising the index LOWERS the flux (strongly:
several orders of magnitude at 180 nm).  Redward of the boundary, raising the
index RAISES the flux (weakly: a few percent near 500-850 nm).

Works ONLY from the raw TSVs.  No scipy (not installed): OLS and Spearman are
computed with numpy directly.

Output: D1_metallicity_crossover.json next to this script.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/data/photon_flux_grid.tsv"
OUT = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/output/verification/D1_metallicity_crossover.json"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def ols_slope(x, y):
    """OLS slope of y on x, plus R^2.  No scipy needed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    slope = sxy / sxx
    intercept = ym - slope * xm
    yhat = intercept + slope * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def spearman_rho(x, y):
    """Spearman rank correlation (Pearson correlation of ranks).
    Average ranks for ties (numpy argsort-based, tie-safe via pandas rank)."""
    xr = pd.Series(x).rank().to_numpy(dtype=float)
    yr = pd.Series(y).rank().to_numpy(dtype=float)
    xm, ym = xr.mean(), yr.mean()
    num = ((xr - xm) * (yr - ym)).sum()
    den = np.sqrt(((xr - xm) ** 2).sum() * ((yr - ym) ** 2).sum())
    return num / den if den > 0 else float("nan")


def slopes_for_slice(sub, mmask, use_rank=False, log=True):
    """Return (wavelengths, slopes) for one temperature slice.

    sub: DataFrame for one temperature (long form).
    mmask: boolean mask over the metallicity column selecting the window.
    """
    wl = np.sort(sub.wavelength_nm.unique())
    out = []
    for w in wl:
        g = sub[sub.wavelength_nm == w]
        g = g[mmask(g.metallicity_index.to_numpy())]
        m = g.metallicity_index.to_numpy(dtype=float)
        f = g.photon_flux_cm2.to_numpy(dtype=float)
        y = np.log(f) if log else f
        if use_rank:
            # rank-transform both variables -> Spearman rho is the OLS slope of
            # ranks (not exactly, but the sign/monotone trend is what we need);
            # we use rho directly for the sign test.
            out.append(spearman_rho(m, f))
        else:
            out.append(ols_slope(m, y)[0])
    return np.asarray(wl), np.asarray(out)


def first_sign_change(wl, s):
    """First wavelength where slope changes sign, by linear interpolation
    between the two bracketing wavelengths.  Returns dict with the bracket."""
    for i in range(1, len(s)):
        if s[i - 1] == 0:
            return {"w_cross": float(wl[i - 1]), "bracket": [float(wl[i - 1]), float(wl[i - 1])],
                    "slope_before": float(s[i - 1]), "slope_after": float(s[i]),
                    "index_before": i - 1, "index_after": i - 1, "exact_node": True}
        if np.sign(s[i]) != np.sign(s[i - 1]) and s[i] != 0:
            w0, w1 = wl[i - 1], wl[i]
            s0, s1 = s[i - 1], s[i]
            wc = w0 + (w1 - w0) * (0.0 - s0) / (s1 - s0)
            return {"w_cross": float(wc), "bracket": [float(w0), float(w1)],
                    "slope_before": float(s0), "slope_after": float(s1),
                    "index_before": i - 1, "index_after": i,
                    "exact_node": False}
    return None


# ----------------------------------------------------------------------------
# load raw data
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA, sep="\t")
T_LEVELS = sorted(df.effective_temperature_K.unique())
WINDOWS = {"M<=0.0": 0.0, "M<=0.25": 0.25, "M<=0.5": 0.5}

result = {
    "finding_id": "D1",
    "datafile": DATA,
    "method": ("OLS slope of natural-log photon_flux_cm2 on metallicity_index per "
               "(T, wavelength); crossover = first sign change, linearly interpolated "
               "between bracketing wavelengths"),
    "n_wavelengths": int(df.wavelength_nm.nunique()),
    "n_rows": int(len(df)),
}

# ----------------------------------------------------------------------------
# 1. primary: M <= +0.50, OLS on log flux
# ----------------------------------------------------------------------------
primary = {}
wl_ref = None
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl, s = slopes_for_slice(sub, lambda m: m <= 0.50)
    wl_ref = wl
    cross = first_sign_change(wl, s)
    i180 = int(np.argmin(np.abs(wl - 179.4)))
    i850 = int(np.argmin(np.abs(wl - 850.0)))
    i500 = int(np.where(wl == 500)[0][0])
    mask_r = wl >= 500
    primary[f"T{T}"] = {
        "w_cross_nm": cross["w_cross"] if cross else None,
        "bracket_nm": cross["bracket"] if cross else None,
        "bracket_slopes": [cross["slope_before"], cross["slope_after"]] if cross else None,
        "slope_at_179.4nm": float(s[i180]),
        "slope_at_850nm": float(s[i850]),
        "slope_500nm": float(s[i500]),
        "median_slope_wl_ge_500nm": float(np.median(s[mask_r])),
        "min_slope": float(s.min()), "min_slope_wl": float(wl[int(np.argmin(s))]),
        "max_slope": float(s.max()), "max_slope_wl": float(wl[int(np.argmax(s))]),
        "all_slopes_below_bracket_negative": bool(np.all(s[:cross["index_before"] + 1] < 0)) if cross else None,
        "all_slopes_above_bracket_positive": bool(np.all(s[cross["index_after"]:] > 0)) if cross else None,
        "n_negative_slopes": int((s < 0).sum()),
        "n_positive_slopes": int((s > 0).sum()),
    }
    primary[f"T{T}"]["_wl"] = wl.tolist()
    primary[f"T{T}"]["_slopes"] = s.tolist()

# perilog-magnitude statements -------------------------------------------------
mag = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    for w in (179.4, 850.0):
        g = sub[(sub.wavelength_nm == w) & (sub.metallicity_index <= 0.50)]
        g = g.sort_values("metallicity_index")
        lo = g[g.metallicity_index == -1.0].photon_flux_cm2.iloc[0]
        hi = g[g.metallicity_index == 0.50].photon_flux_cm2.iloc[0]
        mag[f"T{T}_{w}nm"] = {
            "flux_M_minus1": float(lo), "flux_M_plus0.5": float(hi),
            "ratio_M0.5_over_Mm1": float(hi / lo),
            "log10_ratio": float(np.log10(hi / lo)),
            "pct_change": float(100.0 * (hi / lo - 1.0)),
        }
# endpoint slopes over the FULL ladder (including M>0.5) for a sanity check
full_ladder = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl, s_full = slopes_for_slice(sub, lambda m: np.ones_like(m, dtype=bool))
    full_ladder[f"T{T}"] = {"w_cross_nm": (first_sign_change(wl, s_full) or {}).get("w_cross")}

result["primary_M_le_0.50"] = {
    k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in primary.items()
}
result["endpoint_fluxes_and_ratios"] = mag
result["full_ladder_M_le_0.90_crossing_nm"] = full_ladder

# ----------------------------------------------------------------------------
# 2. robustness (a): Spearman-style rank trend
# ----------------------------------------------------------------------------
rob_rank = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl, rho = slopes_for_slice(sub, lambda m: m <= 0.50, use_rank=True)
    cross = first_sign_change(wl, rho)
    rob_rank[f"T{T}"] = {
        "w_cross_nm": cross["w_cross"] if cross else None,
        "bracket_nm": cross["bracket"] if cross else None,
        "rho_at_179.4nm": float(rho[int(np.argmin(np.abs(wl - 179.4)))]),
        "rho_at_850nm": float(rho[int(np.argmin(np.abs(wl - 850.0)))]),
        "median_rho_wl_ge_500nm": float(np.median(rho[wl >= 500])),
    }
result["robustness_a_spearman_rank"] = rob_rank

# ----------------------------------------------------------------------------
# 3. robustness (b): leave each temperature slice out (for the "same boundary"
#    part of the claim: does the remaining evidence still put the boundary in
#    420-470 nm and consistent across temperatures?)
# ----------------------------------------------------------------------------
rob_loo = {}
for T in T_LEVELS:
    wl = np.sort(df[df.effective_temperature_K != T].wavelength_nm.unique())
    sub = df[(df.effective_temperature_K != T)]
    # pool all remaining temperatures (this is the "leave one slice out" test)
    med = []
    for w in wl:
        g = sub[(sub.wavelength_nm == w) & (sub.metallicity_index <= 0.50)]
        med.append(ols_slope(g.metallicity_index.to_numpy(float),
                             np.log(g.photon_flux_cm2.to_numpy(float)))[0])
    med = np.asarray(med)
    cross = first_sign_change(wl, med)
    rob_loo[f"drop_T{T}"] = {"w_cross_nm": cross["w_cross"] if cross else None,
                             "bracket_nm": cross["bracket"] if cross else None}
result["robustness_b_leave_one_temperature_out_pooled"] = rob_loo

# spread of the three per-slice crossing wavelengths
cross_vals = [primary[f"T{T}"]["w_cross_nm"] for T in T_LEVELS]
result["crossing_spread_nm"] = {
    "per_slice": {f"T{T}": v for T, v in zip(T_LEVELS, cross_vals)},
    "min": float(np.min(cross_vals)), "max": float(np.max(cross_vals)),
    "range": float(np.max(cross_vals) - np.min(cross_vals)),
}

# ----------------------------------------------------------------------------
# 4. robustness (c): vary the metallicity window
# ----------------------------------------------------------------------------
rob_win = {}
for wname, mmax in WINDOWS.items():
    rob_win[wname] = {}
    for T in T_LEVELS:
        sub = df[df.effective_temperature_K == T]
        wl, s = slopes_for_slice(sub, lambda m, mm=mmax: m <= mm)
        cross = first_sign_change(wl, s)
        rob_win[wname][f"T{T}"] = {
            "w_cross_nm": cross["w_cross"] if cross else None,
            "bracket_nm": cross["bracket"] if cross else None,
            "median_slope_wl_ge_500nm": float(np.median(s[wl >= 500])),
        }
result["robustness_c_metallicity_window"] = rob_win

# ----------------------------------------------------------------------------
# 5. sensitivity to the bracketing wavelengths: for each slice, which
#    wavelength pairs bracket the crossing, and what happens if either
#    endpoint is nudged (interpolation vs midpoint vs either endpoint)?
# ----------------------------------------------------------------------------
sens = {}
for T in T_LEVELS:
    wl = np.asarray(primary[f"T{T}"]["_wl"])
    s = np.asarray(primary[f"T{T}"]["_slopes"])
    cross = first_sign_change(wl, s)
    i = cross["index_after"]
    # what if we simply reported the last negative or first positive node?
    sens[f"T{T}"] = {
        "last_negative_node_nm": float(wl[i - 1]),
        "first_positive_node_nm": float(wl[i]),
        "interpolated_nm": float(cross["w_cross"]),
        "midpoint_nm": float(0.5 * (wl[i - 1] + wl[i])),
        "slope_last_negative": float(s[i - 1]),
        "slope_first_positive": float(s[i]),
        "node_spacing_nm": float(wl[i] - wl[i - 1]),
        # how sensitive is the interpolated value to a +/-20% error in the two
        # bracket slopes?  (crude but explicit)
        "interp_if_slope_before_x1.2": float(wl[i - 1] + (wl[i] - wl[i - 1]) * (0.0 - 1.2 * s[i - 1]) / (s[i] - 1.2 * s[i - 1])),
        "interp_if_slope_after_x1.2": float(wl[i - 1] + (wl[i] - wl[i - 1]) * (0.0 - s[i - 1]) / (1.2 * s[i] - s[i - 1])),
        "slopes_3_before": [float(x) for x in s[max(0, i - 3):i]],
        "slopes_3_after": [float(x) for x in s[i:i + 3]],
    }
result["robustness_d_bracket_sensitivity"] = sens

# ----------------------------------------------------------------------------
# 6. does the boundary lie in 420-470 nm?  and is the sign pattern clean?
# ----------------------------------------------------------------------------
in_range = {f"T{T}": bool(420.0 <= primary[f"T{T}"]["w_cross_nm"] <= 470.0) for T in T_LEVELS}
result["boundary_in_claimed_420_470_window"] = in_range
result["boundary_consistent_across_T"] = bool(
    (np.max(cross_vals) - np.min(cross_vals)) <= 20.0
)

# ----------------------------------------------------------------------------
# 7. pre-boundary / post-boundary sign sweep counts (all wavelengths)
# ----------------------------------------------------------------------------
pattern = {}
for T in T_LEVELS:
    wl = np.asarray(primary[f"T{T}"]["_wl"])
    s = np.asarray(primary[f"T{T}"]["_slopes"])
    cross = first_sign_change(wl, s)
    i = cross["index_after"]
    pattern[f"T{T}"] = {
        "n_neg_blueward_of_cross": int((s[:i] < 0).sum()),
        "n_wl_blueward": int(i),
        "n_pos_redward_of_cross": int((s[i:] > 0).sum()),
        "n_wl_redward": int(len(s) - i),
        "blueward_all_negative": bool(np.all(s[:i] < 0)),
        "redward_all_positive": bool(np.all(s[i:] > 0)),
    }
result["sign_pattern_counts"] = pattern

# ----------------------------------------------------------------------------
# 8. the "several orders of magnitude at 180 nm" sub-claim
# ----------------------------------------------------------------------------
result["orders_of_magnitude_179.4nm"] = {
    f"T{T}": float(np.log10(mag[f"T{T}_179.4nm"]["ratio_M0.5_over_Mm1"])) for T in T_LEVELS
}
result["pct_change_850nm"] = {
    f"T{T}": float(mag[f"T{T}_850.0nm"]["pct_change"]) for T in T_LEVELS
}

# ----------------------------------------------------------------------------
# 9. crossing-definition variants (the first-sign-change rule is brittle when
#    a slice has a local wobble, e.g. T=6300K near 410-430 nm)
# ----------------------------------------------------------------------------
def last_sign_change(wl, s):
    """The LAST index where the sign changes; i.e. the start of the final
    contiguous run of same-sign slopes (the 'persistent' boundary)."""
    for i in range(len(s) - 1, 0, -1):
        if s[i] != 0 and s[i - 1] != 0 and np.sign(s[i]) != np.sign(s[i - 1]):
            w0, w1, s0, s1 = wl[i - 1], wl[i], s[i - 1], s[i]
            wc = w0 + (w1 - w0) * (0.0 - s0) / (s1 - s0)
            return {"w_cross": float(wc), "bracket": [float(w0), float(w1)],
                    "slope_before": float(s0), "slope_after": float(s1),
                    "index_before": i - 1, "index_after": i}
    return None


def median_smooth(s, k=3):
    h = k // 2
    out = s.copy()
    for i in range(len(s)):
        lo, hi = max(0, i - h), min(len(s), i + h + 1)
        out[i] = np.median(s[lo:hi])
    return out


def quartile_trend_sign(wl, sub):
    """Coarse monotone test: sign of log-flux difference between the top
    quartile and bottom quartile of the metallicity ladder (M<=0.5)."""
    out = []
    for w in wl:
        g = sub[(sub.wavelength_nm == w) & (sub.metallicity_index <= 0.50)]
        m = g.metallicity_index.to_numpy(float)
        y = np.log(g.photon_flux_cm2.to_numpy(float))
        q1, q3 = np.quantile(m, [0.25, 0.75])
        out.append(y[m >= q3].mean() - y[m <= q1].mean())
    return np.asarray(out)


variants = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl = np.asarray(primary[f"T{T}"]["_wl"])
    s = np.asarray(primary[f"T{T}"]["_slopes"])
    s_sm = median_smooth(s, 3)
    qd = quartile_trend_sign(wl, sub)
    variants[f"T{T}"] = {
        "first_sign_change_nm": (first_sign_change(wl, s) or {}).get("w_cross"),
        "last_sign_change_persistent_nm": (last_sign_change(wl, s) or {}).get("w_cross"),
        "first_sign_change_after_3pt_median_smooth_nm": (first_sign_change(wl, s_sm) or {}).get("w_cross"),
        "first_sign_change_quartile_trend_nm": (first_sign_change(wl, qd) or {}).get("w_cross"),
        "n_sign_changes": int(sum(1 for i in range(1, len(s))
                                  if s[i] != 0 and s[i - 1] != 0 and np.sign(s[i]) != np.sign(s[i - 1]))),
        "sign_changes_nm": [[float(wl[i - 1]), float(wl[i])] for i in range(1, len(s))
                            if s[i] != 0 and s[i - 1] != 0 and np.sign(s[i]) != np.sign(s[i - 1])],
    }
result["crossing_definition_variants"] = variants

# moving-average smoothed profile: the most defensible "bulk trend" crossing.
ma_cross = {}
for T in T_LEVELS:
    wl = np.asarray(primary[f"T{T}"]["_wl"])
    s = np.asarray(primary[f"T{T}"]["_slopes"])
    row = {}
    for k in (5, 9, 15):  # odd windows, in nodes
        cs = np.convolve(s, np.ones(k) / k, mode="valid")
        cwl = wl[k // 2: len(wl) - k // 2]
        fx = first_sign_change(cwl, cs)
        row[f"MA{k}_nm"] = fx["w_cross"] if fx else None
    row["MA9_profile_min"] = float(np.min(np.convolve(s, np.ones(9) / 9, mode="valid")))
    ma_cross[f"T{T}"] = row
result["moving_average_crossing"] = ma_cross
ma9 = {f"T{T}": ma_cross[f"T{T}"]["MA9_nm"] for T in T_LEVELS}
result["moving_average_crossing_summary"] = {
    "MA5": {f"T{T}": ma_cross[f"T{T}"]["MA5_nm"] for T in T_LEVELS},
    "MA9": ma9,
    "MA15": {f"T{T}": ma_cross[f"T{T}"]["MA15_nm"] for T in T_LEVELS},
    "MA9_min": float(np.min(list(ma9.values()))), "MA9_max": float(np.max(list(ma9.values()))),
    "MA9_range_nm": float(np.max(list(ma9.values())) - np.min(list(ma9.values()))),
    "MA9_all_in_420_470": bool(all(420.0 <= x <= 470.0 for x in ma9.values() if x is not None)),
}

persistent = {f"T{T}": variants[f"T{T}"]["last_sign_change_persistent_nm"] for T in T_LEVELS}
result["persistent_crossing"] = {
    "per_slice": persistent,
    "min": float(np.min(list(persistent.values()))),
    "max": float(np.max(list(persistent.values()))),
    "range": float(np.max(list(persistent.values())) - np.min(list(persistent.values()))),
    "all_in_420_470": bool(all(420.0 <= pv <= 470.0 for pv in persistent.values())),
}

# ----------------------------------------------------------------------------
# 10. redward-side statistics: how weak is the positive trend?
# ----------------------------------------------------------------------------
red = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl = np.asarray(primary[f"T{T}"]["_wl"])
    s = np.asarray(primary[f"T{T}"]["_slopes"])
    msk = wl >= 500
    red[f"T{T}"] = {
        "n_nodes_wl_ge_500": int(msk.sum()),
        "n_positive_slopes_wl_ge_500": int((s[msk] > 0).sum()),
        "frac_positive_wl_ge_500": float((s[msk] > 0).mean()),
        "n_negative_slopes_wl_ge_500": int((s[msk] < 0).sum()),
        "median_slope_wl_ge_500": float(np.median(s[msk])),
        "min_slope_wl_ge_500": float(s[msk].min()),
        "max_slope_wl_ge_500": float(s[msk].max()),
        "median_pct_flux_change_over_Mspan_1.5_Mm1_to_Mp0.5":
            float(100 * (np.exp(np.median(s[msk]) * 1.5) - 1)),
        "median_pct_flux_change_over_Mspan_1.9_full_ladder":
            float(100 * (np.exp(np.median(s[msk]) * 1.9) - 1)),
    }
    # empirical pct change at specific wavelengths, per M-window
    red[f"T{T}"]["empirical_pct_change"] = {}
    for w in (500.0, 600.0, 700.0, 850.0):
        g = sub[(sub.wavelength_nm == w)]
        g_low = g[g.metallicity_index == -1.0].photon_flux_cm2.iloc[0]
        g_m05 = g[g.metallicity_index == 0.50].photon_flux_cm2.iloc[0]
        g_m09 = g[g.metallicity_index == 0.90].photon_flux_cm2.iloc[0]
        red[f"T{T}"]["empirical_pct_change"][f"{w:.0f}nm"] = {
            "Mm1_to_Mp0.5_pct": float(100 * (g_m05 / g_low - 1)),
            "Mm1_to_Mp0.9_pct": float(100 * (g_m09 / g_low - 1)),
        }
result["redward_statistics"] = red

# ----------------------------------------------------------------------------
# 11. blueward amplitude: how many dex at 180 nm, both windows
# ----------------------------------------------------------------------------
blue = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    g = sub[sub.wavelength_nm == 179.4]
    for wname, mmax in (("full_ladder_Mm1_to_Mp0.9", 0.90), ("Mm1_to_Mp0.5", 0.50)):
        gl = g[g.metallicity_index == -1.0].photon_flux_cm2.iloc[0]
        gh = g[g.metallicity_index == mmax].photon_flux_cm2.iloc[0]
        blue[f"T{T}_{wname}"] = {"log10_ratio": float(np.log10(gh / gl)),
                                 "factor": float(gh / gl)}
    # also the widest possible: min over all M to max over all M
    f = g.photon_flux_cm2.to_numpy(float)
    blue[f"T{T}_max_over_min_any_M"] = {"log10_ratio": float(np.log10(f.max() / f.min())),
                                        "factor": float(f.max() / f.min())}
result["blueward_amplitude_179.4nm"] = blue

# ----------------------------------------------------------------------------
# 12. joint cross-temperature consistency + fit quality
# ----------------------------------------------------------------------------
slope_mat = np.vstack([np.asarray(primary[f"T{T}"]["_slopes"]) for T in T_LEVELS])
wl_arr = np.asarray(primary[f"T{T_LEVELS[0]}"]["_wl"])
all_neg = np.all(slope_mat < 0, axis=0)
all_pos = np.all(slope_mat > 0, axis=0)
result["joint_cross_temperature_consistency"] = {
    "last_wavelength_all_three_slices_negative": float(wl_arr[np.where(all_neg)[0].max()]),
    "first_wavelength_all_three_slices_positive": float(wl_arr[np.where(all_pos)[0].min()]),
    "n_wavelengths_all_three_negative": int(all_neg.sum()),
    "n_wavelengths_all_three_positive": int(all_pos.sum()),
    "joint_bracket_nm": [float(wl_arr[np.where(all_neg)[0].max()]),
                         float(wl_arr[np.where(all_pos)[0].min()])],
    "n_wavelengths_with_mixed_sign_across_slices": int((~all_neg & ~all_pos).sum()),
}
# position of the "mixed" band(s)
mixed = ~all_neg & ~all_pos
runs, start = [], None
for i, mv in enumerate(mixed):
    if mv and start is None:
        start = i
    elif not mv and start is not None:
        runs.append([float(wl_arr[start]), float(wl_arr[i - 1])])
        start = None
if start is not None:
    runs.append([float(wl_arr[start]), float(wl_arr[-1])])
result["joint_cross_temperature_consistency"]["mixed_sign_bands_nm"] = runs

fitq = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    row = {}
    for w in (179.4, 850.0):
        g = sub[(sub.wavelength_nm == w) & (sub.metallicity_index <= 0.50)]
        sl, r2 = ols_slope(g.metallicity_index.to_numpy(float),
                           np.log(g.photon_flux_cm2.to_numpy(float)))
        row[f"{w}nm"] = {"slope": float(sl), "r2": float(r2),
                         "spearman_rho": float(spearman_rho(g.metallicity_index.to_numpy(float),
                                                            g.photon_flux_cm2.to_numpy(float)))}
        # is the 179.4 nm response perfectly monotone across the 21 levels?
        f = g.sort_values("metallicity_index").photon_flux_cm2.to_numpy(float)
        row[f"{w}nm"]["strictly_monotone_decreasing"] = bool(np.all(np.diff(f) < 0))
        row[f"{w}nm"]["strictly_monotone_increasing"] = bool(np.all(np.diff(f) > 0))
    fitq[f"T{T}"] = row
result["fit_quality_endpoints"] = fitq

# ----------------------------------------------------------------------------
# 12b. MODEL-FREE test: no regression at all.  R(w) = flux(M=+win)/flux(M=-1).
#      The reversal is the wavelength where R crosses 1.
# ----------------------------------------------------------------------------
def ratio_crossings(wl, R):
    xs = []
    for i in range(1, len(R)):
        if R[i] != R[i - 1] and (R[i - 1] - 1.0) * (R[i] - 1.0) < 0:
            xs.append(round(float(wl[i - 1] + (wl[i] - wl[i - 1]) * (1.0 - R[i - 1]) / (R[i] - R[i - 1])), 4))
    return xs


model_free = {}
for T in T_LEVELS:
    sub = df[df.effective_temperature_K == T]
    wl = np.sort(sub.wavelength_nm.unique())
    R = []
    for w in wl:
        g = sub[sub.wavelength_nm == w]
        lo = g[g.metallicity_index == -1.0].photon_flux_cm2.iloc[0]
        hi = g[g.metallicity_index == 0.50].photon_flux_cm2.iloc[0]
        R.append(hi / lo)
    R = np.asarray(R)
    allx = ratio_crossings(wl, R)
    band = [x for x in allx if 380.0 <= x <= 600.0]
    model_free[f"T{T}"] = {
        "all_R1_crossings_nm": allx,
        "crossings_380_600nm": band,
        "R_at_400nm": float(R[np.where(wl == 400)[0][0]]),
        "R_at_450nm": float(R[np.where(wl == 450)[0][0]]),
        "R_at_500nm": float(R[np.where(wl == 500)[0][0]]),
        "R_at_850nm": float(R[np.where(wl == 850)[0][0]]),
    }
result["model_free_flux_ratio_crossing"] = model_free

# ----------------------------------------------------------------------------
# 13. verdict / discrepancies
# ----------------------------------------------------------------------------
v = {}
v["claimed_boundary_window_nm"] = [420, 470]
v["my_per_slice_boundary_naive_first_sign_change_nm"] = {
    f"T{T}": primary[f"T{T}"]["w_cross_nm"] for T in T_LEVELS}
v["my_per_slice_boundary_robust_persistent_nm"] = persistent
v["my_per_slice_boundary_model_free_ratio_nm"] = {
    f"T{T}": (model_free[f"T{T}"]["crossings_380_600nm"][0]
              if model_free[f"T{T}"]["crossings_380_600nm"] else None) for T in T_LEVELS}
v["n_slices_inside_window_naive"] = int(sum(1 for T in T_LEVELS
                                            if 420 <= primary[f"T{T}"]["w_cross_nm"] <= 470))
# Most defensible robust extraction = MA9 (bulk-trend) or the model-free ratio
# crossing (no regression at all).
robust_ok = {f"T{T}": bool(420.0 <= ma9[f"T{T}"] <= 470.0
                           and 420.0 <= model_free[f"T{T}"]["crossings_380_600nm"][0] <= 470.0)
             for T in T_LEVELS}
v["robust_definitions_per_slice_nm"] = {
    f"T{T}": {"MA9_rolling_mean": round(ma9[f"T{T}"], 3),
              "model_free_ratio": model_free[f"T{T}"]["crossings_380_600nm"][0],
              "both_in_420_470": robust_ok[f"T{T}"]} for T in T_LEVELS}
v["n_slices_inside_window_robust"] = int(sum(robust_ok.values()))
v["spread_naive_nm"] = result["crossing_spread_nm"]["range"]
v["spread_robust_nm"] = float(max(ma9.values()) - min(ma9.values()))
v["spread_model_free_nm"] = float(
    max(model_free[f"T{T}"]["crossings_380_600nm"][0] for T in T_LEVELS)
    - min(model_free[f"T{T}"]["crossings_380_600nm"][0] for T in T_LEVELS))
v["blueward_orders_of_magnitude_179.4nm_full_ladder"] = {
    f"T{T}": float(blue[f"T{T}_full_ladder_Mm1_to_Mp0.9"]["log10_ratio"]) for T in T_LEVELS}
v["redward_pct_change_850nm_full_ladder"] = {
    f"T{T}": red[f"T{T}"]["empirical_pct_change"]["850nm"]["Mm1_to_Mp0.9_pct"] for T in T_LEVELS}
v["discrepancies"] = [
    ("DIRECTION + BOUNDARY: confirmed. Raising the metallicity index lowers flux blueward "
     "of ~435-449 nm and raises it redward; the crossover is 448.7 (5300K), 445.8 (5800K) "
     "under the literal first-sign-change rule and the same for the persistent/model-free "
     "definitions; the model-free flux-ratio test (flux(M=+0.5)/flux(M=-1.0)=1) gives "
     "448.69 / 445.99 / 434.43 nm."),
    ("T=6300K EXCEPTION (definitional fragility): the literal 'first sign change' rule gives "
     "409.87 nm for the 6300K slice, just BELOW the claimed 420 nm window edge. That value is "
     "not robust: the 6300K slope profile has 3 sign changes (405-410, 415-420, 430-435) and "
     "the fluxes at 410-420 nm vary by only ~1-2% over the whole ladder (slopes +0.0011, "
     "+0.0012, -0.0024), i.e. a null band, not a real reversal. The persistent-last-crossing "
     "and model-free definitions both give 434.4 nm for this slice, inside the window."),
    ("MAGNITUDE, blueward: 'several orders of magnitude at 180 nm' is an overstatement. Over "
     "the full ladder M=-1.00 -> +0.90 the 179.4 nm flux falls by a factor of 85-152, i.e. "
     "1.93-2.18 dex = about TWO orders of magnitude (1.39-1.83 dex over the evenly sampled "
     "M<=+0.50 sub-ladder)."),
    ("MAGNITUDE, redward: 'a few percent near 500-850 nm' understates it. Over the full "
     "ladder (M=-1 -> +0.9) the empirical change is +7.2/+9.7/+10.5% at 850 nm, +9.3/+13.3/"
     "+15.7% at 700 nm and +14.1/+16.9/+19.0% at 600 nm for T=5300/5800/6300 K; the median "
     "log-slope for wl>=500 nm is 0.044/0.060/0.072 per unit M (+6.8/+9.5/+11.4% over the "
     "M<=0.5 ladder). So the right description is 'weak but ~5-15%', and 'weak' is accurate "
     "at 850 nm where it is ~7-10%."),
    ("REDWARD NOT UNIFORMLY POSITIVE: at T=5300K the slopes at 500-525 nm are locally "
     "negative (min -0.053 at 515 nm; empirical M=-1 -> +0.9 change at 500 nm is -4.6%), so "
     "93.0% of nodes with wl>=500 nm are positive there (100% for the other two slices)."),
    ("CROSS-TEMPERATURE 'SAME': supported. The two most defensible definitions both put all "
     "three slices inside 420-470 nm and agree to within ~27 nm (model-free flux-ratio: "
     "448.69 / 445.99 / 434.43 nm, range 14.3 nm; 9-node rolling-mean slope: 457.0 / 445.9 / "
     "429.6 nm, range 27.5 nm = 6.2% of the mean). Both show a systematic trend to bluer "
     "crossover at higher T, so 'essentially the same' is fair but not exact. The literal "
     "first-sign-change rule gives a 38.8 nm range because of the 6300K null band above."),
]
v["crossing_definition_sensitivity_nm"] = {
    f"T{T}": {"naive_first": primary[f"T{T}"]["w_cross_nm"],
              "persistent": persistent[f"T{T}"],
              "rolling_mean_MA9": round(ma9[f"T{T}"], 3),
              "model_free": (model_free[f"T{T}"]["crossings_380_600nm"][0]
                             if model_free[f"T{T}"]["crossings_380_600nm"] else None),
              "bracket": primary[f"T{T}"]["bracket_nm"]} for T in T_LEVELS}
result["verdict"] = v

# ----------------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------------
result["commands"] = {
    "run": "python3 /data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/output/verification/D1_verify.py",
    "python": sys.version,
    "script": __file__ if "__file__" in globals() else "D1_verify.py",
}
result["reproduce_from_raw_data"] = True

with open(OUT, "w") as fh:
    json.dump(result, fh, indent=2, default=str)

# ---- console summary --------------------------------------------------------
print("=== D1 primary (M <= +0.50, OLS slope of ln flux on M) ===")
for T in T_LEVELS:
    p = primary[f"T{T}"]
    print(f"T={T}K  w_cross={p['w_cross_nm']:.2f} nm  bracket={p['bracket_nm']}  "
          f"slope(179.4)={p['slope_at_179.4nm']:.4f}  slope(850)={p['slope_at_850nm']:.5f}  "
          f"median(>=500)={p['median_slope_wl_ge_500nm']:.5f}")
print("\n=== crossing spread ===", result["crossing_spread_nm"])
print("\n=== Spearman ===")
for T in T_LEVELS:
    r = rob_rank[f"T{T}"]
    print(f"T={T}K w_cross={r['w_cross_nm']} rho(179.4)={r['rho_at_179.4nm']:.3f} "
          f"rho(850)={r['rho_at_850nm']:.3f} med_rho(>=500)={r['median_rho_wl_ge_500nm']:.3f}")
print("\n=== window variation ===")
for wn in WINDOWS:
    print(wn, {f"T{T}": rob_win[wn][f"T{T}"]["w_cross_nm"] for T in T_LEVELS})
print("\n=== leave-one-T-out (pooled) ===", rob_loo)
print("\n=== bracket sensitivity ===")
for T in T_LEVELS:
    print(f"T={T}K", sens[f"T{T}"])
print("\n=== in 420-470 window ===", in_range)
print("=== sign pattern ===")
for T in T_LEVELS:
    print(f"T={T}K", pattern[f"T{T}"])
print("\n=== orders of mag at 179.4 nm ===", result["orders_of_magnitude_179.4nm"])
print("=== pct change at 850 nm ===", result["pct_change_850nm"])
print("\n=== crossing definition variants ===")
for T in T_LEVELS:
    vv = variants[f"T{T}"]
    print(f"T={T}K n_sign_changes={vv['n_sign_changes']} at {vv['sign_changes_nm']}")
    print(f"    first={vv['first_sign_change_nm']} last/persistent={vv['last_sign_change_persistent_nm']} "
          f"smoothed={vv['first_sign_change_after_3pt_median_smooth_nm']} quartile={vv['first_sign_change_quartile_trend_nm']}")
print("\n=== persistent crossing ===", result["persistent_crossing"])
print("\n=== moving-average crossing ===")
for T in T_LEVELS:
    print(f"T={T}K", ma_cross[f"T{T}"])
print("MA summary:", result["moving_average_crossing_summary"])
print("\n=== redward strength ===")
for T in T_LEVELS:
    r = red[f"T{T}"]
    print(f"T={T}K frac_pos(>=500)={r['frac_positive_wl_ge_500']:.3f} median_slope={r['median_slope_wl_ge_500']:.5f} "
          f"min={r['min_slope_wl_ge_500']:.5f} -> median pct over full ladder={r['median_pct_flux_change_over_Mspan_1.9_full_ladder']:.2f}%")
    print(f"    empirical: {r['empirical_pct_change']}")
print("\n=== blueward 179.4 nm amplitude ===", blue)
print("\n=== joint consistency ===", result["joint_cross_temperature_consistency"])
print("\n=== fit quality endpoints ===", fitq)
print("\n=== VERDICT ===")
for k, val in v.items():
    if k != "discrepancies":
        print(" ", k, "=", val)
print("  discrepancies:")
for d in v["discrepancies"]:
    print("   -", d)
print("\nwrote", OUT)