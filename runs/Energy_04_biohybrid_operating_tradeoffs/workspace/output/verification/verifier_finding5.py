#!/usr/bin/env python3
"""Independent falsification check of Finding 5 (dose scan + perturbation trajectories).

Written from scratch against the raw TSVs only. Does NOT import or run analysis.py.
Independent methods used where possible:
  * OLS via normal equations (explicit sums) AND via an independent grid search.
  * Pearson r via covariance formula by hand AND via rank (Spearman) transform.
  * Bootstrap with a DIFFERENT seed and 30000 draws; plus a paired-draws variant
    (same draws for both conditions) and a common-scaling variant.
"""
import json
import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
DATA = f"{ROOT}/data"
OUT = {}

# ---------------------------------------------------------------- load raw
dose = pd.read_csv(f"{DATA}/dose_response.tsv", sep="\t")
pert = pd.read_csv(f"{DATA}/perturbation_time_response.tsv", sep="\t")

x = dose["dose"].to_numpy(float)
y = dose["response_mean"].to_numpy(float)
sd_col = dose["response_sd"].to_numpy(float)
n_sd_missing = int(np.isnan(sd_col).sum())
OUT["dose_raw"] = {
    "n_rows": int(len(dose)),
    "dose": x.tolist(),
    "response_mean": y.tolist(),
    "response_sd_all_nan": bool(np.all(np.isnan(sd_col))),
    "n_sd_missing": n_sd_missing,
}

# ---------------------------------------------------------------- helpers
def ols_normal_eq(xx, yy):
    """Slope/intercept via explicit normal equations (sums)."""
    n = len(xx)
    sx, sy = xx.sum(), yy.sum()
    sxx = (xx * xx).sum()
    sxy = (xx * yy).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept

def r2_of(xx, yy, slope, intercept):
    pred = intercept + slope * xx
    ss_res = ((yy - pred) ** 2).sum()
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot

def slope_grid(xx, yy, lo=-1.0, hi=1.0, steps=200001):
    """Independent check of OLS slope by grid search on SSE (only for well-scaled data)."""
    grid = np.linspace(lo, hi, steps)
    # intercept optimal for a given slope is mean(y - slope*x)
    best_s, best_sse = None, np.inf
    for s in grid:
        b = (yy - s * xx).mean()
        sse = ((yy - (b + s * xx)) ** 2).sum()
        if sse < best_sse:
            best_sse, best_s = sse, s
    return float(best_s)

def pearson_cov(xx, yy):
    """Pearson r via the covariance formula, computed by hand."""
    n = len(xx)
    mx, my = xx.mean(), yy.mean()
    cov = ((xx - mx) * (yy - my)).sum() / (n - 1)
    sx = xx.std(ddof=1)
    sy = yy.std(ddof=1)
    return cov / (sx * sy)

def spearman(xx, yy):
    rx = pd.Series(xx).rank().to_numpy()
    ry = pd.Series(yy).rank().to_numpy()
    return pearson_cov(rx, ry)

def loglinear_fit(t, v):
    """OLS of log(v) on t. Returns slope, intercept, R2 (normal eq + grid check)."""
    s, b = ols_normal_eq(t, np.log(v))
    s_grid = slope_grid(t, np.log(v), lo=s - 1, hi=s + 1, steps=200001)
    return s, b, r2_of(t, np.log(v), s, b), s_grid

def linear_fit(t, v):
    s, b = ols_normal_eq(t, v)
    return s, b, r2_of(t, v, s, b)

# ================================================================ 5a DOSE
slope, intercept = ols_normal_eq(x, y)
lin_r2 = r2_of(x, y, slope, intercept)
slope_grid_val = slope_grid(x, y, lo=-1.0, hi=1.0, steps=200001)

# log-log: fit log(response) ~ log(dose); slope = Hill-like exponent
lx, ly = np.log(x), np.log(y)
ll_slope, ll_int = ols_normal_eq(lx, ly)
ll_r2 = r2_of(lx, ly, ll_slope, ll_int)
ll_slope_grid = slope_grid(lx, ly, lo=ll_slope - 1, hi=ll_slope + 1, steps=200001)

# second difference (curvature)
second_diff = (y[2] - y[1]) - (y[1] - y[0])

fold = y.max() / y.min()
apex_idx = int(np.argmax(y))
monotone_inc = bool(np.all(np.diff(y) > 0))
monotone_dec = bool(np.all(np.diff(y) < 0))

# leave-one-setting-out linear fits (each fits 2 points exactly -> R2==1)
loo = []
for k in range(3):
    m = np.ones(3, bool); m[k] = False
    s, b = ols_normal_eq(x[m], y[m])
    pred = b + s * x[k]
    loo.append({
        "dropped": k + 1, "slope": float(s), "intercept": float(b),
        "predicted_at_dropped": float(pred), "observed": float(y[k]),
        "residual": float(y[k] - pred),
        "r2_on_two_points": float(r2_of(x[m], y[m], s, b)),
    })

# quadratic in dose = exact interpolation of 3 points (Vandermonde solve)
V = np.vstack([np.ones(3), x, x ** 2]).T
coef = np.linalg.solve(V, y)
quad_pred = V @ coef
quad_resid = float(((quad_pred - y) ** 2).sum())
quad_vertex = -coef[1] / (2 * coef[2])
# independent check: solve the normal equations of the quadratic least squares
A = np.vstack([np.ones(3), x, x ** 2]).T
normal_coef = np.linalg.solve(A.T @ A, A.T @ y)

# resolution of the dose axis vs vertex location
dose_resolution = min(np.diff(np.sort(x)))
# what response can 3 points with no SD actually rule out? magnitude of the dip
dip = y[2] - y[1]  # fall from apex to third setting

OUT["dose_5a"] = {
    "linear_slope": float(slope), "linear_intercept": float(intercept),
    "linear_r2": float(lin_r2), "linear_slope_grid_check": slope_grid_val,
    "loglog_slope": float(ll_slope), "loglog_intercept": float(ll_int),
    "loglog_r2": float(ll_r2), "loglog_slope_grid_check": ll_slope_grid,
    "second_difference": float(second_diff),
    "fold_change_max_over_min": float(fold),
    "interior_apex_index_1based": apex_idx + 1, "interior_apex_dose": float(x[apex_idx]),
    "non_monotone": bool((not monotone_inc) and (not monotone_dec)),
    "leave_one_setting_out": loo,
    "quadratic_coef_const_dose_dose2": coef.tolist(),
    "quadratic_residual_sse": quad_resid,
    "quadratic_fitted_equals_observed": bool(np.allclose(quad_pred, y)),
    "quadratic_vertex_dose": float(quad_vertex),
    "quadratic_vertex_inside_scan": bool(x.min() < quad_vertex < x.max()),
    "normal_eq_coef_agreement": float(np.max(np.abs(normal_coef - coef))),
    "dose_resolution_min_gap": float(dose_resolution),
    "apex_to_third_setting_drop": float(dip),
    "no_sd_means_no_ci": True,
}

# ======================================================= 5b PERTURBATION
t = pert["time"].to_numpy(float)
means = {c: pert[f"{c}_mean"].to_numpy(float) for c in ("control", "condition_a", "condition_b")}
sds = {c: pert[f"{c}_sd"].to_numpy(float) for c in ("control", "condition_a", "condition_b")}

pert_out = {}
for c in ("control", "condition_a", "condition_b"):
    v, s = means[c], sds[c]
    t0, t6 = float(v[0]), float(v[-1])
    change = t6 - t0
    pct = 100.0 * change / t0
    lin_s, lin_b, lin_r2 = linear_fit(t, v)
    lg_s, lg_b, lg_r2, lg_grid = loglinear_fit(t, v)
    hl = float(np.log(2) / (-lg_s)) if lg_s < 0 else None
    pooled_sd = float(np.sqrt(np.mean(s ** 2)))
    pert_out[c] = {
        "t0": t0, "t6": t6, "change": float(change), "pct_change": float(pct),
        "strictly_decreasing": bool(np.all(np.diff(v) < 0)),
        "linear_slope": float(lin_s), "linear_r2": float(lin_r2),
        "loglinear_slope": float(lg_s), "loglinear_r2": float(lg_r2),
        "loglinear_slope_grid_check": lg_grid,
        "half_life": hl,
        "pooled_sd": pooled_sd,
        "effect_over_pooled_sd": float(abs(change) / pooled_sd),
        "ratio_to_control_each_t": (v / means["control"]).tolist(),
    }

out_a, out_b, out_c = pert_out["condition_a"], pert_out["condition_b"], pert_out["control"]
pot_change = abs(out_b["change"]) / abs(out_a["change"])
pot_rate = abs(out_b["loglinear_slope"]) / abs(out_a["loglinear_slope"])

# independent alternative summary: nonparametric MG-ratio-ish endpoint ratio and
# a rank-based ordering check (Spearman of condition_b vs condition_a levels)
order_r = pearson_cov(means["condition_b"], means["condition_a"])
order_rho = spearman(means["condition_b"], means["condition_a"])

OUT["perturbation_5b"] = {
    "time": t.tolist(),
    "trajectories": pert_out,
    "potency_ratio_change": float(pot_change),
    "potency_ratio_loglinear_rate": float(pot_rate),
    "pearson_b_vs_a_levels": float(order_r),
    "spearman_b_vs_a_levels": float(order_rho),
    "n_sd_missing_in_perturbation": int(np.isnan(pert[[c + "_sd" for c in means]].to_numpy(float)).sum()),
}

# ================================================================ BOOTSTRAP
# Report: 20000 draws, seed 20230919, perturbing ONLY t=0 and t=6 by their supplied SDs.
def bootstrap_endpoints(seed, ndraws, paired=False, common_scale=False):
    rng = np.random.default_rng(seed)
    res = {}
    for c in ("condition_a", "condition_b"):
        v, s = means[c], sds[c]
        base = v[-1] - v[0]
        if paired:
            z0 = rng.standard_normal(ndraws)
            z6 = rng.standard_normal(ndraws)
        else:
            z0 = rng.standard_normal(ndraws)
            z6 = rng.standard_normal(ndraws)
        if common_scale:
            # multiplicative lognormal noise with SD/mean scale (relative error)
            d0 = v[0] * (1 + (s[0] / v[0]) * z0)
            d6 = v[-1] * (1 + (s[-1] / v[-1]) * z6)
        else:
            d0 = v[0] + s[0] * z0
            d6 = v[-1] + s[-1] * z6
        d = d6 - d0
        res[c] = {"z0": z0, "z6": z6, "delta": d,
                  "mean": float(d.mean()),
                  "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_ge_0": float((d >= 0).mean())}
    diff = res["condition_b"]["delta"] - res["condition_a"]["delta"]
    res["diff_b_minus_a"] = {
        "mean": float(diff.mean()),
        "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "p_le_0": float((diff <= 0).mean()),
        "n_le_0": int((diff <= 0).sum()), "ndraws": int(ndraws),
        "delta": diff,
    }
    return res

boot = bootstrap_endpoints(12345, 30000)
boot_paired = bootstrap_endpoints(999, 30000, paired=True)
boot_rel = bootstrap_endpoints(424242, 30000, common_scale=True)

# also reproduce the report's own seed to check exact reproducibility of its numbers
boot_report_seed = bootstrap_endpoints(20230919, 20000)

OUT["bootstrap"] = {
    "mine_seed_12345_30000": {k: {kk: vv for kk, vv in v.items() if kk not in ("z0", "z6", "delta")}
                              for k, v in {kk: vv for kk, vv in boot.items()}.items()},
    "mine_seed_999_30000_paired_precision_scaling": {
        k: {kk: vv for kk, vv in v.items() if kk not in ("z0", "z6", "delta")}
        for k, v in boot_paired.items()},
    "mine_seed_424242_30000_relative_noise": {
        k: {kk: vv for kk, vv in v.items() if kk not in ("z0", "z6", "delta")}
        for k, v in boot_rel.items()},
    "report_seed_20230919_20000_replication": {
        k: {kk: vv for kk, vv in v.items() if kk not in ("z0", "z6", "delta")}
        for k, v in boot_report_seed.items()},
}

# Self-consistency check of the report's "optimistic" caveat:
# does dropping / fixing intermediate timepoints actually change the answer?
# compare endpoint-only CI (above) with a CI that ALSO resamples t=2 and t=4.
def bootstrap_all_timepoints(seed, ndraws, paired=True):
    rng = np.random.default_rng(seed)
    res = {}
    for c in ("condition_a", "condition_b"):
        v, s = means[c], sds[c]
        draws = np.empty((ndraws, 4))
        for j in range(4):
            draws[:, j] = v[j] + s[j] * rng.standard_normal(ndraws)
        d = draws[:, -1] - draws[:, 0]
        res[c] = {"mean": float(d.mean()),
                  "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                  "p_ge_0": float((d >= 0).mean()), "delta": d}
    diff = res["condition_b"]["delta"] - res["condition_a"]["delta"]
    res["diff_b_minus_a"] = {"mean": float(diff.mean()),
                             "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                             "p_le_0": float((diff <= 0).mean())}
    return res

boot_all = bootstrap_all_timepoints(777, 30000)
OUT["bootstrap_all_timepoints_seed_777"] = {
    k: {kk: vv for kk, vv in v.items() if kk != "delta"} for k, v in boot_all.items()}

# --- RNG-stream-layout test: does generating 4 columns (but using only the
# --- endpoints) shift the same-seed interval?  Explains my same-seed mismatch.
rng = np.random.default_rng(20230919)
nboot = 20000
stream = {}
for c in ("condition_a", "condition_b"):
    v, s = means[c], sds[c]
    draws = rng.normal(v, s, size=(nboot, len(v)))   # 4 cols consumed
    d = draws[:, -1] - draws[:, 0]
    stream[c] = {"ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                 "mean": float(d.mean())}
da = rng.normal(means["condition_a"], sds["condition_a"], size=(nboot, 4))
db = rng.normal(means["condition_b"], sds["condition_b"], size=(nboot, 4))
diff = (db[:, -1] - db[:, 0]) - (da[:, -1] - da[:, 0])
stream["diff_b_minus_a"] = {"ci": [float(np.percentile(diff, 2.5)),
                                   float(np.percentile(diff, 97.5))],
                            "mean": float(diff.mean()),
                            "p_le_0": float((diff <= 0).mean())}
OUT["rng_stream_layout_same_seed_20230919"] = stream

# --- independent separation check: per-timepoint z of (b - a) from supplied SDs,
# --- plus a sign test across the 3 post-baseline timepoints.
z_by_t = []
for j in range(4):
    se = np.sqrt(sds["condition_b"][j] ** 2 + sds["condition_a"][j] ** 2)
    z_by_t.append(float((means["condition_b"][j] - means["condition_a"][j]) / se))
OUT["separation_b_minus_a"] = {
    "z_by_timepoint": z_by_t,
    "n_negative_after_t0": int(sum(1 for j in range(1, 4) if z_by_t[j] < 0)),
    "combined_stouffer_z": float(np.sum(z_by_t[1:]) / np.sqrt(3)),
    "b_lt_a_at_all_postbaseline": bool(all(means["condition_b"][j] < means["condition_a"][j]
                                             for j in range(1, 4))),
    "control_abs_drift": float(abs(means["control"][-1] - means["control"][0])),
    "control_max_abs_dev_from_t0": float(np.max(np.abs(means["control"] - means["control"][0]))),
    "control_range_over_pooled_sd": float((means["control"].max() - means["control"].min())
                                          / pert_out["control"]["pooled_sd"]),
}

# --- dose-scan robustness: log-log slope when the interior setting is dropped
x13, y13 = x[[0, 2]], y[[0, 2]]
ll_slope_13 = float(np.log(y13[1] / y13[0]) / np.log(x13[1] / x13[0]))
OUT["dose_extra"] = {
    "loglog_slope_dropping_setting2": ll_slope_13,
    "loglog_slope_all_three": float(ll_slope),
    "slope_change_when_middle_dropped": ll_slope_13 - float(ll_slope),
    "quadratic_vertex_vs_observed_apex_dose": [float(quad_vertex), float(x[apex_idx])],
}

# --- SD inventory of the perturbation file (report says "six supplied SDs")
all_sd = pert[[f"{c}_sd" for c in ("control", "condition_a", "condition_b")]].to_numpy(float)
OUT["sd_inventory"] = {
    "n_sd_values_in_file": int(all_sd.size),
    "n_missing": int(np.isnan(all_sd).sum()),
    "min": float(np.nanmin(all_sd)), "max": float(np.nanmax(all_sd)),
    "n_endpoint_sd_control_a_b": int(2 * 3),
}

# --- Falsification attempt on 5a: under a Poisson counting model (the only
# --- plausible noise model available for a "response_mean"), is the apex
# --- significant?  If the apex IS significant under this model, the report's
# --- "unresolved" verdict would be understating a real signal.
se_12 = np.sqrt(y[1] + y[0])
se_23 = np.sqrt(y[2] + y[1])
se_13 = np.sqrt(y[2] + y[0])
OUT["dose_poisson_sensitivity"] = {
    "note": "assumes response_sd = sqrt(mean), a Poisson-like model NOT given by the data",
    "z_rise_1_to_2": float((y[1] - y[0]) / se_12),
    "z_fall_2_to_3": float((y[2] - y[1]) / se_23),
    "z_endpoints_1_to_3": float((y[2] - y[0]) / se_13),
    "apex_significant_poisson": bool(abs((y[1] - y[0]) / se_12) > 1.96
                                     and abs((y[2] - y[1]) / se_23) > 1.96),
}

# --- Falsification attempt on 5b: exact rank separation of the post-baseline
# --- timepoints (a vs b), plus a truly shared-error (perfectly correlated
# --- endpoint errors) bootstrap, the most conservative pairing of a and b.
post = slice(1, 4)
OUT["rank_separation_postbaseline"] = {
    "a_values_t2_t4_t6": means["condition_a"][post].tolist(),
    "b_values_t2_t4_t6": means["condition_b"][post].tolist(),
    "all_b_below_all_a": bool(means["condition_b"][post].max() < means["condition_a"][post].min()),
    "exact_mannwhitney_two_sided_p_3v3": 2.0 / 20.0,
}

rng = np.random.default_rng(31337)
nboot = 30000
z0 = rng.standard_normal(nboot)
z6 = rng.standard_normal(nboot)
va, sa = means["condition_a"], sds["condition_a"]
vb, sb = means["condition_b"], sds["condition_b"]
# shared systematic component in each condition's endpoint error (correlation 1)
da_c = (va[-1] + sa[-1] * z6) - (va[0] + sa[0] * z0)
db_c = (vb[-1] + sb[-1] * z6) - (vb[0] + sb[0] * z0)
diff_c = db_c - da_c
OUT["bootstrap_shared_endpoint_error"] = {
    "diff_mean": float(diff_c.mean()),
    "diff_ci95": [float(np.percentile(diff_c, 2.5)), float(np.percentile(diff_c, 97.5))],
    "p_le_0": float((diff_c <= 0).mean()),
}

# --- Most conservative "n=4 timepoints" uncertainty: unpaired t-style test of
# --- b vs a on the 4 per-timepoint means using their supplied SDs.
pooled = np.sqrt((sa ** 2).mean() + (sb ** 2).mean())
diff_means = (vb - va)
se_diff4 = np.sqrt((sa ** 2 + sb ** 2).mean())
OUT["conservative_n4_test"] = {
    "mean_b_minus_a_per_timepoint": float(diff_means.mean()),
    "mean_se_per_timepoint": float(se_diff4),
    "t_like": float(diff_means.mean() / (se_diff4 / np.sqrt(4))),
}

# --- The report's list of exact claims in prose, checked verbatim
prose = {
    "control_t0_stated": (0.32243, float(means["control"][0])),
    "control_t6_stated": (0.32397, float(means["control"][-1])),
    "a_change_stated": (-0.04320, float(out_a["change"])),
    "b_change_stated": (-0.09437, float(out_b["change"])),
    "a_half_life_stated": (29.4788, float(out_a["half_life"])),
    "b_half_life_stated": (12.5433, float(out_b["half_life"])),
    "a_ratio_stated": (2.1845, float(out_b["change"] / out_a["change"])),
    "b_rate_ratio_stated": (2.3502, float(out_b["loglinear_slope"] / out_a["loglinear_slope"])),
    "bootstrap_a_ci_low_stated": (-0.054636, float(np.percentile(boot_report_seed["condition_a"]["delta"], 2.5))),
    "bootstrap_a_ci_high_stated": (-0.031516, float(np.percentile(boot_report_seed["condition_a"]["delta"], 97.5))),
    "bootstrap_b_ci_low_stated": (-0.105804, float(np.percentile(boot_report_seed["condition_b"]["delta"], 2.5))),
    "bootstrap_b_ci_high_stated": (-0.083168, float(np.percentile(boot_report_seed["condition_b"]["delta"], 97.5))),
    "bootstrap_diff_ci_low_stated": (-0.067253, float(np.percentile(boot_report_seed["diff_b_minus_a"]["delta"], 2.5))),
    "bootstrap_diff_ci_high_stated": (-0.034826, float(np.percentile(boot_report_seed["diff_b_minus_a"]["delta"], 97.5))),
}
OUT["prose_claims_check"] = {k: {"stated": v[0], "recomputed": v[1],
                                 "abs_diff": abs(v[0] - v[1])} for k, v in prose.items()}

# --- third independent check on R2: squared Pearson correlation (by hand)
r_xy = pearson_cov(x, y)
r_log = pearson_cov(np.log(x), np.log(y))
OUT["r2_via_squared_pearson"] = {
    "linear_r2": float(r_xy ** 2),
    "loglog_r2": float(r_log ** 2),
    "matches_reported_linear_r2": bool(abs(r_xy ** 2 - lin_r2) < 1e-12),
    "matches_reported_loglog_r2": bool(abs(r_log ** 2 - ll_r2) < 1e-12),
}

# --- exact count of non-positive diff draws under the report's stream layout
rng = np.random.default_rng(20230919)
nboot = 20000
for c in ("condition_a", "condition_b"):
    v, s = means[c], sds[c]
    _ = rng.normal(v, s, size=(nboot, 4))
da = rng.normal(means["condition_a"], sds["condition_a"], size=(nboot, 4))
db = rng.normal(means["condition_b"], sds["condition_b"], size=(nboot, 4))
diff = (db[:, -1] - db[:, 0]) - (da[:, -1] - da[:, 0])
OUT["report_layout_exact_counts"] = {
    "n_diff_le_0": int((diff <= 0).sum()), "ndraws": nboot,
    "p_le_0": float((diff <= 0).mean()),
    "diff_mean": float(diff.mean()),
    "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
}

# ================================================================ OUTPUT
print(json.dumps(OUT, indent=1))

with open(f"{ROOT}/output/verification/verifier_finding5_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)