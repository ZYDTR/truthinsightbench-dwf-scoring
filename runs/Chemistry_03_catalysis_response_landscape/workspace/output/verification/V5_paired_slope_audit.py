#!/usr/bin/env python3
"""
V5 -- Fresh-eyes falsification audit of CLAIM D5 about observation_set_2.tsv.

Claim D5 under test:
  "observation_set_2.tsv contains two paired readings (numeric_feature_1 and
   numeric_feature_2) of the same 41 quantities. They agree almost perfectly
   (Pearson ~0.998, Spearman ~0.986) but with a reproducibly COMPRESSED slope:
   OLS of numeric_feature_2 on numeric_feature_1 gives slope ~0.9752 with 95% CI
   [0.9577, 0.9926], which excludes 1, while the mean difference is NOT
   significantly different from zero (mean ~0.483, p ~ 0.278). The largest single
   disagreement is 6.75 units and 22 of 528 pairs invert order."

Independently re-derive every number from the raw TSV; then attack the
"compressed slope" interpretation with:
  (a) lattice-rows-only vs irregular-rows-only vs all-rows slope
  (b) errors-in-variables / Deming / total-least-squares attenuation correction
  (c) bootstrap + leave-one-out slope distributions
Plus: what numeric_feature_3 actually contains, and whether row 21 (index) is
driving the slope.

NOTE ON ROW LABELLING: observation_index is written zero-padded (00..40), so the
file's own `observation_index` and the 1-based file line number differ by one.
We report BOTH for the largest-disagreement row.

Writes V5_paired_slope_audit.json next to this script.
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/observation_set_2.tsv"
OUT = os.path.join(HERE, "V5_paired_slope_audit.json")

RNG = np.random.default_rng(20260919)
NBOOT = 20000

results = {}


def rec(key, val):
    results[key] = val
    return val


def ols(x, y):
    """OLS of y on x with intercept. Returns slope, intercept, se_slope, ci, r2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    ci = m.conf_int(alpha=0.05)[1]
    return dict(
        n=int(len(x)),
        slope=float(m.params[1]),
        intercept=float(m.params[0]),
        se_slope=float(m.bse[1]),
        slope_ci95=[float(ci[0]), float(ci[1])],
        t=float(m.tvalues[1]),
        p=float(m.pvalues[1]),
        r2=float(m.rsquared),
        resid_sd=float(np.sqrt(m.mse_resid)),
    )


# ---------------------------------------------------------------- load
raw = open(DATA, "rb").read()
lines = raw.decode("utf-8").split("\n")
rec("raw_audit", {
    "file_bytes": len(raw),
    "n_lines_including_trailing": len(lines),
    "header": lines[0].rstrip("\r").split("\t"),
    "last_data_line": lines[-2].rstrip("\r") if lines[-1] == "" else lines[-1].rstrip("\r"),
    "index_column_is_zero_padded": bool(lines[1].startswith("00\t")),
    "delimiter": "tab",
})

df = pd.read_csv(DATA, sep="\t")
rec("shape", list(df.shape))
rec("shape_claim_baseline", "1430x8 for set1; set2 baseline said 41 rows")
rec("columns", list(df.columns))
rec("dtypes", {c: str(t) for c, t in df.dtypes.items()})
rec("observation_index_values", df["observation_index"].tolist())

x = df["numeric_feature_1"].to_numpy(float)
y = df["numeric_feature_2"].to_numpy(float)
idx = df["observation_index"].to_numpy(int)
n = len(df)
rec("n_rows", int(n))

# ---------------------------------------------------- 1. headline numbers
pear_r, pear_p = st.pearsonr(x, y)
spear_r, spear_p = st.spearmanr(x, y)
kend_r, kend_p = st.kendalltau(x, y)

d = y - x  # "reading 2 minus reading 1"
tt = st.ttest_rel(y, x)
tt1 = st.ttest_1samp(d, 0.0)

rec("headline", {
    "pearson_r": float(pear_r),
    "pearson_p": float(pear_p),
    "pearson_r2": float(pear_r ** 2),
    "spearman_r": float(spear_r),
    "spearman_p": float(spear_p),
    "kendall_tau": float(kend_r),
    "kendall_p": float(kend_p),
    "claim_pearson_0.998": {"value": float(pear_r), "match": bool(round(pear_r, 3) == 0.998)},
    "claim_spearman_0.986": {"value": float(spear_r), "match": bool(round(spear_r, 3) == 0.986)},
})

fit_all = ols(x, y)
fit_all["slope_rounded"] = round(fit_all["slope"], 4)
fit_all["ci_rounded"] = [round(v, 4) for v in fit_all["slope_ci95"]]
fit_all["ci_excludes_1"] = bool(fit_all["slope_ci95"][1] < 1.0)
fit_all["claim_slope_0.9752"] = {"value": fit_all["slope"], "match": bool(round(fit_all["slope"], 4) == 0.9752)}
fit_all["claim_ci_0.9577_0.9926"] = {
    "value": fit_all["slope_ci95"],
    "match": bool(round(fit_all["slope_ci95"][0], 4) == 0.9577 and round(fit_all["slope_ci95"][1], 4) == 0.9926),
}
rec("ols_feature2_on_feature1_all", fit_all)

rec("mean_difference", {
    "definition": "mean(numeric_feature_2 - numeric_feature_1)",
    "mean_diff": float(d.mean()),
    "sd_diff": float(d.std(ddof=1)),
    "se_diff": float(st.sem(d)),
    "t_paired": float(tt.statistic),
    "p_paired": float(tt.pvalue),
    "t_1samp": float(tt1.statistic),
    "p_1samp": float(tt1.pvalue),
    "ci95_mean_diff": [float(v) for v in st.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=st.sem(d))],
    "claim_mean_0.483": {"value": float(d.mean()), "match": bool(round(d.mean(), 3) == 0.483)},
    "claim_p_0.278": {"value": float(tt1.pvalue), "match": bool(round(tt1.pvalue, 3) == 0.278)},
    "mean_diff_significantly_nonzero_at_0.05": bool(tt1.pvalue < 0.05),
})

# ---------------------------------------------- 3. what is numeric_feature_3?
f3 = df["numeric_feature_3"]
rec("numeric_feature_3", {
    "non_missing_count": int(f3.notna().sum()),
    "n_missing": int(f3.isna().sum()),
    "missing_fraction": float(f3.isna().mean()),
    "unique_values_non_missing": sorted(set(f3.dropna().tolist())),
    "unique_values_as_list": [float(v) for v in f3.dropna().unique()],
    "rows_where_present": [int(i) for i in idx[f3.notna().to_numpy()]],
    "variance_of_non_missing": float(f3.dropna().var(ddof=1)) if f3.notna().sum() > 1 else None,
    "essentially_empty": bool(f3.notna().sum() <= 1),
})

# ------------------------------------- 4. largest absolute disagreement + row
absd = np.abs(d)
j = int(np.argmax(absd))
order = np.argsort(-absd)[:5]
rec("largest_disagreement", {
    "max_abs_diff": float(absd[j]),
    "claim_6.75": {"value": float(absd[j]), "match": bool(round(absd[j], 2) == 6.75)},
    "observation_index": int(idx[j]),
    "file_line_number_1based": int(idx[j] + 2),  # +1 header, +1 for 1-based
    "numeric_feature_1": float(x[j]),
    "numeric_feature_2": float(y[j]),
    "signed_diff_f2_minus_f1": float(d[j]),
    "top5_by_abs_diff": [
        {
            "observation_index": int(idx[k]),
            "file_line_number_1based": int(idx[k] + 2),
            "f1": float(x[k]),
            "f2": float(y[k]),
            "abs_diff": float(absd[k]),
            "signed_diff": float(d[k]),
        }
        for k in order
    ],
})

# leverage / influence of that row
infl = sm.OLS(y, sm.add_constant(x)).fit().get_influence()
hat = infl.hat_matrix_diag
cooks = infl.cooks_distance[0]
dfbeta_slope = infl.dfbetas[:, 1]
rec("influence", {
    "max_leverage_row_index": int(idx[int(np.argmax(hat))]),
    "max_leverage": float(hat.max()),
    "mean_leverage": float(hat.mean()),
    "largest_row_leverage": float(hat[j]),
    "largest_row_cooks_D": float(cooks[j]),
    "largest_row_dfbeta_slope": float(dfbeta_slope[j]),
    "max_cooks_row_index": int(idx[int(np.argmax(cooks))]),
    "max_abs_dfbeta_row_index": int(idx[int(np.argmax(np.abs(dfbeta_slope)))]),
})

fit_no_j = ols(np.delete(x, j), np.delete(y, j))
fit_no_j["ci_excludes_1"] = bool(fit_no_j["slope_ci95"][1] < 1.0)
fit_no_j["slope_rounded"] = round(fit_no_j["slope"], 4)
rec("ols_without_largest_disagreement_row", fit_no_j)

# --------------------------------------------------- 2(a) lattice vs irregular
# Detect (do not assume) the evenly spaced block: search for the largest set of
# rows that lie on a GAP-FREE equally spaced grid (a contiguous arithmetic
# progression), generalising the file's first-8-rows lattice. Also record the
# direct, layout-based check that file rows 0..7 are an exact AP.
best = None
for i in range(n):
    for j in range(i + 1, n):
        gap = abs(x[j] - x[i])
        if gap < 1e-9:
            continue
        for m in range(1, 9):
            s = gap / m
            a = x[i]
            k = np.round((x - a) / s)
            on = np.abs((x - a) - k * s) < 1e-9
            if on.sum() < 4:
                continue
            ks = np.sort(k[on])
            if np.any(np.diff(ks) != 1):          # must be a contiguous grid, no holes
                continue
            if len(set(np.round(x[on], 9))) != int(on.sum()):   # no duplicate members
                continue
            score = int(on.sum())
            if best is None or score > best[0] or (score == best[0] and s > best[1]):
                best = (score, s, a, on.copy())

score, spacing, anchor, lattice_mask = best
first8_is_exact_ap = bool(
    np.allclose(np.diff(x[:8]), np.diff(x[:8])[0], atol=1e-9) and len(set(np.diff(x[:8]))) == 1
)
lat_idx = idx[lattice_mask]
irr_idx = idx[~lattice_mask]
rec("row_classification", {
    "method": "largest gap-free equally spaced grid found by exhaustive anchor/step search (no hard-coding)",
    "file_rows_0_to_7_are_exact_arithmetic_progression": first8_is_exact_ap,
    "file_rows_0_to_7_x": [float(v) for v in x[:8]],
    "file_rows_0_to_7_step": float(np.diff(x[:8])[0]),
    "grid_matches_file_rows_0_to_7": bool(set(lat_idx.tolist()) == set(range(8))),
    "lattice_rows_observation_index": [int(v) for v in lat_idx],
    "lattice_x_values_sorted": [float(v) for v in np.sort(x[lattice_mask])],
    "lattice_spacing": float(spacing),
    "irregular_rows_observation_index": [int(v) for v in irr_idx],
    "n_lattice": int(lattice_mask.sum()),
    "n_irregular": int((~lattice_mask).sum()),
})

fit_lat = ols(x[lattice_mask], y[lattice_mask])
fit_lat["ci_excludes_1"] = bool(fit_lat["slope_ci95"][1] < 1.0)
fit_lat["slope_rounded"] = round(fit_lat["slope"], 4)
rec("ols_lattice_rows_only", fit_lat)

fit_irr = ols(x[~lattice_mask], y[~lattice_mask])
fit_irr["ci_excludes_1"] = bool(fit_irr["slope_ci95"][1] < 1.0)
fit_irr["slope_rounded"] = round(fit_irr["slope"], 4)
rec("ols_irregular_rows_only", fit_irr)

# lattice / irregular mean-difference for completeness
rec("mean_diff_by_block", {
    "lattice": float(d[lattice_mask].mean()),
    "irregular": float(d[~lattice_mask].mean()),
    "lattice_ttest_p": float(st.ttest_1samp(d[lattice_mask], 0.0).pvalue),
    "irregular_ttest_p": float(st.ttest_1samp(d[~lattice_mask], 0.0).pvalue),
})

rec("mixture_artefact_test_2a", {
    "slope_lattice": fit_lat["slope"],
    "slope_irregular": fit_irr["slope"],
    "slope_all": fit_all["slope"],
    "all_three_below_1": bool(fit_lat["slope"] < 1 and fit_irr["slope"] < 1 and fit_all["slope"] < 1),
    "all_three_ci_exclude_1": bool(
        fit_lat["ci_excludes_1"] and fit_irr["ci_excludes_1"] and fit_all["ci_excludes_1"]
    ),
    "verdict": "mixture artefact REJECTED" if (fit_lat["slope"] < 1 and fit_irr["slope"] < 1) else "mixture artefact plausible",
})

# --------------------------------------- 2(b) errors-in-variables / EIV correction
# Reliability from the most defensible datum in the file: the 8 lattice rows
# where reading 1 is an exact design setting (sd 46.19) but reading 2 wobbles by
# 1.10 units about the reference line y = x.
lat_resid = (y[lattice_mask] - x[lattice_mask])
rec("lattice_residuals_f1_is_exact", {
    "x_sd_lattice": float(x[lattice_mask].std(ddof=1)),
    "x_exact_design_setting": True,
    "resid_sd_ddof1": float(lat_resid.std(ddof=1)),
    "resid_rms": float(np.sqrt(np.mean(lat_resid ** 2))),
    "resid_values": [float(v) for v in lat_resid],
    "implied_reading2_noise_sd": float(lat_resid.std(ddof=1)),
})


def deming(xa, ya, delta):
    """Deming regression, delta = var(err_x)/var(err_y). Returns slope, intercept."""
    xa = np.asarray(xa, float); ya = np.asarray(ya, float)
    mx, my = xa.mean(), ya.mean()
    sxx = np.var(xa, ddof=1); syy = np.var(ya, ddof=1); sxy = np.cov(xa, ya, ddof=1)[0, 1]
    num = (syy - delta * sxx) + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)
    den = 2 * sxy
    b = num / den
    return float(b), float(my - b * mx)


def tls_slope(xa, ya):
    """Total least squares (= Deming with delta=1): slope of first PCA component."""
    A = np.column_stack([np.asarray(xa, float), np.asarray(ya, float)])
    A = A - A.mean(0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[0]
    return float(v[1] / v[0])


sd_r2 = float(lat_resid.std(ddof=1))
sd_r1 = float(fit_all["resid_sd"])  # observed scatter of f2 about OLS(f2~f1), ~ mixing both noises

eiv = {
    "observed_ols_slope": fit_all["slope"],
    "residual_sd_of_f2_about_ols_line": sd_r1,
    "reading2_noise_sd_implied_by_lattice": sd_r2,
    "implied_total_residual_variance": sd_r1 ** 2,
    "implied_reading2_noise_variance": sd_r2 ** 2,
}
# Bootstrap CIs for Deming under each delta (5000 reps)
def boot_deming(delta, nboot=5000):
    bs = np.empty(nboot)
    boots_idx = RNG.integers(0, n, (nboot, n))
    for k in range(nboot):
        ii = boots_idx[k]
        bs[k] = deming(x[ii], y[ii], delta)[0]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(bs.mean()), float(bs.std(ddof=1))


for delta in [0.0, 0.05, 0.10, 0.111, 0.15, 0.25, 0.5, 1.0]:
    lo, hi, mu, sd = boot_deming(delta)
    eiv[f"deming_delta_{delta}"] = {
        "slope": deming(x, y, delta)[0],
        "ci95_bootstrap": [lo, hi],
        "bootstrap_mean": mu,
        "bootstrap_sd": sd,
        "ci_excludes_1": bool(hi < 1.0 or lo > 1.0),
        "consistent_with_1": bool(lo <= 1.0 <= hi),
    }

# delta implied by the data itself
implied_delta = sd_r2 ** 2 / max(sd_r1 ** 2 - sd_r2 ** 2, 1e-12)
eiv["implied_delta_from_lattice"] = float(implied_delta)
b_impl, a_impl = deming(x, y, implied_delta)
lo_i, hi_i, mu_i, sd_i = boot_deming(implied_delta)
eiv["deming_at_implied_delta"] = {
    "delta": float(implied_delta),
    "slope": b_impl,
    "intercept": a_impl,
    "ci95_bootstrap": [lo_i, hi_i],
    "consistent_with_1": bool(lo_i <= 1.0 <= hi_i),
    "ci_excludes_1": bool(hi_i < 1.0 or lo_i > 1.0),
}
eiv["tls_slope_delta1"] = tls_slope(x, y)
eiv["tls_consistent_with_1_note"] = (
    "TLS/Deming with delta=1 assumes equal noise on both readings; here the lattice "
    "rows show reading 1 is an exact setting, so delta is far below 1."
)
# geometric-mean / reduced major axis for reference
eiv["rma_slope"] = float(y.std(ddof=1) / x.std(ddof=1))
rec("errors_in_variables_2b", eiv)

# ----------------------------------- 2(c) bootstrap + leave-one-out on the slope
# Nonparametric case-resampling bootstrap of the OLS slope
bs = np.empty(NBOOT)
boots_idx = RNG.integers(0, n, (NBOOT, n))
xc = x - x.mean(); yc = y - y.mean()
for k in range(NBOOT):
    ii = boots_idx[k]
    xk = x[ii]; yk = y[ii]
    bs[k] = ((xk - xk.mean()) * (yk - yk.mean())).sum() / ((xk - xk.mean()) ** 2).sum()
boot_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
rec("bootstrap_ols_slope", {
    "method": "nonparametric case resampling, n=41, B=20000, seed 20260919",
    "n_boot": NBOOT,
    "mean": float(bs.mean()),
    "sd": float(bs.std(ddof=1)),
    "percentile_ci95": boot_ci,
    "ci_excludes_1": bool(boot_ci[1] < 1.0),
    "fraction_boot_slopes_below_1": float(np.mean(bs < 1.0)),
    "fraction_boot_slopes_above_1": float(np.mean(bs > 1.0)),
    "analytic_ci95": fit_all["slope_ci95"],
})

loo = np.empty(n)
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    loo[i] = ols(x[m], y[m])["slope"]
rec("leave_one_out_slope", {
    "n": int(n),
    "slopes": [float(v) for v in loo],
    "n_below_1": int((loo < 1).sum()),
    "n_above_1": int((loo > 1).sum()),
    "fraction_below_1": float((loo < 1).mean()),
    "min": float(loo.min()),
    "min_at_observation_index": int(idx[int(np.argmin(loo))]),
    "max": float(loo.max()),
    "max_at_observation_index": int(idx[int(np.argmax(loo))]),
    "mean": float(loo.mean()),
    "sd": float(loo.std(ddof=1)),
})

# ----------------------------------- additional independent structural checks
# Is f3 essentially a constant-zero / dropped column? Is there any duplicate pair?
rec("set2_structure_extra", {
    "any_duplicate_f1_values": bool(df["numeric_feature_1"].duplicated().any()),
    "f1_min": float(x.min()), "f1_max": float(x.max()),
    "f2_min": float(y.min()), "f2_max": float(y.max()),
    "n_f1_negative": int((x < 0).sum()),
    "n_exact_zeros_f1": int((x == 0).sum()),
    "monotone_increase_f2_vs_f1_violations": int((np.diff(y) < 0).sum()),
    "kendall_tau_b": float(st.kendalltau(x, y, variant="b").statistic),
    "difference_sd_over_x_sd": float(d.std(ddof=1) / x.std(ddof=1)),
    "slope_from_var_ratio_1_minus_ratio": float(1 - np.var(d, ddof=1) / np.var(x, ddof=1)),
})

# ============================ KEY EIV RESULT: the correction is BOUNDED ========
# Deming slope -> OLS slope as delta -> 0 AND as delta -> infinity, so the
# attenuation correction has an interior maximum. Scan delta on a log grid to
# find the LARGEST slope ANY noise ratio can produce, with its bootstrap CI.
deltas = np.concatenate([[0.0], np.logspace(-4, 6, 220)])
scan = []
for dl in deltas:
    b, _ = deming(x, y, dl)
    scan.append((float(dl), float(b)))
scan_arr = np.array(scan)
argmax_k = int(np.argmax(scan_arr[:, 1]))
d_star = float(scan_arr[argmax_k, 0])
b_star = float(scan_arr[argmax_k, 1])
lo_s, hi_s, mu_s, sd_s = boot_deming(d_star, nboot=5000)
rec("eiv_bounded_correction", {
    "reasoning": "Deming slope -> OLS slope both as delta->0 and delta->inf, so max over delta is interior; "
                 "no noise ratio can inflate the slope beyond b_max.",
    "delta_at_max": d_star,
    "max_attainable_deming_slope_b_max": b_star,
    "ols_slope": fit_all["slope"],
    "b_max_ci95_bootstrap": [lo_s, hi_s],
    "b_max_ci_excludes_1": bool(hi_s < 1.0 or lo_s > 1.0),
    "b_max_below_1": bool(b_star < 1.0),
    "scan_delta_slope": [[float(a), float(b)] for a, b in scan_arr[::10]],
})

# Robust noise estimate (MAD) -- the sd-based implied delta may be outlier-driven
resid_ols = y - (fit_all["intercept"] + fit_all["slope"] * x)
mad = float(np.median(np.abs(resid_ols - np.median(resid_ols))))
robust_sd = 1.4826 * mad
rec("robust_noise_check", {
    "resid_sd": float(np.std(resid_ols, ddof=1)),
    "resid_mad": mad,
    "resid_robust_sd": robust_sd,
    "resid_min": float(resid_ols.min()), "resid_max": float(resid_ols.max()),
    "n_abs_resid_gt_3": int((np.abs(resid_ols) > 3).sum()),
    "n_abs_resid_gt_5": int((np.abs(resid_ols) > 5).sum()),
    "interpretation": "heavy-tailed residuals: sd is set by a few large deviations, MAD-based sd is far smaller",
    "implied_delta_from_lattice_over_robust": float(sd_r2 ** 2 / max(robust_sd ** 2 - sd_r2 ** 2, 1e-12))
    if robust_sd ** 2 > sd_r2 ** 2 else None,
})

# ------------------------------- robust regressions (outlier-insensitive slope)
ts_slope, ts_int, ts_lo, ts_hi = st.theilslopes(y, x, 0.95)
rec("theil_sen", {
    "slope": float(ts_slope), "intercept": float(ts_int),
    "ci95": [float(ts_lo), float(ts_hi)],
    "ci_excludes_1": bool(ts_hi < 1.0 or ts_lo > 1.0),
    "below_1": bool(ts_slope < 1.0),
})
try:
    from statsmodels.robust.norms import HuberT
    huber = sm.RLM(y, sm.add_constant(x), M=HuberT()).fit()
    ci_h = [float(v) for v in huber.conf_int()[1]]
    rec("huber_rlm", {"slope": float(huber.params[1]), "intercept": float(huber.params[0]),
                      "ci95": ci_h,
                      "ci_excludes_1": bool(ci_h[1] < 1.0 or ci_h[0] > 1.0)})
except Exception as e:  # pragma: no cover
    rec("huber_rlm", {"error": str(e)})

# through-origin fit (no intercept) -- does compression survive dropping intercept?
b0 = float((x * y).sum() / (x * x).sum())
resid0 = y - b0 * x
se0 = float(np.sqrt((resid0 ** 2).sum() / (n - 1) / (x ** 2).sum()))
rec("through_origin_fit", {
    "slope": b0, "se": se0,
    "ci95": [b0 - 1.96 * se0, b0 + 1.96 * se0],
    "below_1": bool(b0 < 1.0),
})

# ---------------------------------------- concordant / discordant pair counts
conc = disc = 0
for i in range(n):
    for k in range(i + 1, n):
        sx = np.sign(x[i] - x[k]); sy = np.sign(y[i] - y[k])
        if sx * sy > 0:
            conc += 1
        elif sx * sy < 0:
            disc += 1
rec("pair_concordance", {
    "n_pairs_total": n * (n - 1) // 2,
    "concordant": conc, "discordant": disc, "tied": n * (n - 1) // 2 - conc - disc,
    "claim_22_of_528_invert": {"discordant": disc, "total": n * (n - 1) // 2,
                               "match": bool(disc == 22 and n * (n - 1) // 2 == 528)},
    "spearman_from_concordance": float((conc - disc) / (conc + disc)),
    "spearman_direct": float(spear_r),
    "spearman_formula_agrees": bool(abs((conc - disc) / (conc + disc) - spear_r) < 1e-9),
})

# ------------------------------------------------ curvature / local-slope checks
Xq = sm.add_constant(np.column_stack([x, x ** 2]))
mq = sm.OLS(y, Xq).fit()
rec("quadratic_curvature", {
    "coef_x": float(mq.params[1]), "coef_x2": float(mq.params[2]),
    "p_x2": float(mq.pvalues[2]),
    "curvature_significant": bool(mq.pvalues[2] < 0.05),
    "slope_at_x_minus_60": float(mq.params[1] + 2 * mq.params[2] * (-60)),
    "slope_at_x_plus_80": float(mq.params[1] + 2 * mq.params[2] * 80),
})

for lo_c, hi_c, label in [(-35, 35, "core_abs_x_le_35"), (0, 100, "positive_x_only"), (-100, 0, "negative_x_only")]:
    m = (x >= lo_c) & (x <= hi_c)
    if m.sum() >= 5:
        f = ols(x[m], y[m])
        rec(f"local_slope_{label}", {k: f[k] for k in ["n", "slope", "slope_ci95", "p", "r2"]} |
            {"ci_excludes_1": bool(f["slope_ci95"][1] < 1.0), "below_1": bool(f["slope"] < 1.0)})

# ratio check -- multiplicative shrinkage vs additive offset
ratio_m = np.abs(x) > 10
ratios = y[ratio_m] / x[ratio_m]
rec("ratio_check", {
    "n_used": int(ratio_m.sum()),
    "ratio_mean": float(ratios.mean()), "ratio_median": float(np.median(ratios)),
    "ratio_sd": float(ratios.std(ddof=1)),
    "ttest_ratio_vs_1_p": float(st.ttest_1samp(ratios, 1.0).pvalue),
    "n_ratio_below_1": int((ratios < 1).sum()),
})

# ---------------------------------------------------------------- write JSON
with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2, default=str)

# ---------------------------------------------------------------- console
def p(label, obj):
    print(f"{label}: {json.dumps(obj, default=str)}")

print("=" * 78)
print("V5 FALSIFICATION AUDIT -- observation_set_2.tsv")
print("=" * 78)
p("shape", results["shape"])
p("pearson_r", round(pear_r, 6))
p("spearman_r", round(spear_r, 6))
p("OLS all", {k: fit_all[k] for k in ["slope", "intercept", "se_slope", "slope_ci95", "p", "r2", "n"]})
p("mean diff / p", {"mean": d.mean(), "p_paired": tt.pvalue, "p_1samp": tt1.pvalue})
p("f3", results["numeric_feature_3"])
p("largest disagreement", results["largest_disagreement"])
p("lattice slope", {k: fit_lat[k] for k in ["n", "slope", "slope_ci95", "p", "r2"]})
p("irregular slope", {k: fit_irr[k] for k in ["n", "slope", "slope_ci95", "p", "r2"]})
p("mixture test", results["mixture_artefact_test_2a"])
p("implied delta", implied_delta)
p("deming @ implied delta", eiv["deming_at_implied_delta"])
p("deming delta=1 (TLS)", {"slope": eiv["deming_delta_1.0"]["slope"], "ci": eiv["deming_delta_1.0"]["ci95_bootstrap"]})
p("bootstrap", results["bootstrap_ols_slope"])
p("LOO", {k: results["leave_one_out_slope"][k] for k in ["n_below_1", "n_above_1", "min", "max"]})
p("without largest row", {k: fit_no_j[k] for k in ["slope", "slope_ci95", "p"]})
print("wrote", OUT)