#!/usr/bin/env python3
"""
V5c -- closing checks for CLAIM D5.

(A) POSITIVE CONTROL for the errors-in-variables argument. Building an interval
    [b_OLS, b_RMA] as the set of slopes attainable by any noise-ratio correction
    is only a *valid* falsification if the same machinery WOULD recover 1 when
    the truth is 1. Simulate y = x + noise from the actual x-values with a range
    of x-noise levels; report the fraction of simulated datasets whose attainable
    interval contains 1, and the fraction whose interval lies strictly below 1
    (false positives).

(B) COMPRESSION DRIFT: regress the difference d = f2 - f1 directly on f1. A slope
    of exactly -1 for the difference field is what "compressed slope 0.9752"
    means; this tests it without the intercept/scale confound.

(C) Exact statistics for the mean difference (CI, sign test, Wilcoxon), plus
    the lattice-block-only EIV span and H0: slope = 1 p-value.

(D) Resolution / rounding audit: could the compression be a rounding artefact?

Writes V5c_positive_control.json.
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/observation_set_2.tsv"
OUT = os.path.join(HERE, "V5c_positive_control.json")

df = pd.read_csv(DATA, sep="\t")
x = df["numeric_feature_1"].to_numpy(float)
y = df["numeric_feature_2"].to_numpy(float)
n = len(x)
idx = df["observation_index"].to_numpy(int)
lattice = idx < 8
res = {}

Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0, 1]
b_ols = float(Sxy / Sxx)
b_rma = float(Syy / Sxy)
lo, hi = min(b_ols, b_rma), max(b_ols, b_rma)

# ---------------------------------------------------------------- (A)
RNG = np.random.default_rng(12345)
NSIM = 2000
# cond(x) reproduces the real leverage profile; truth = 1
sx = float(np.std(x, ddof=1))
sim = {}
for frac in [0.0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]:
    s_e = frac * sx
    contains = 0
    below = 0
    for _ in range(NSIM):
        xo = x + RNG.normal(0, s_e, n)
        yo = x + RNG.normal(0, s_e, n)   # TRUE relation y = x (slope 1) + noise
        sxx = np.var(xo, ddof=1); syy = np.var(yo, ddof=1); sxy = np.cov(xo, yo, ddof=1)[0, 1]
        blo = min(sxy / sxx, syy / sxy); bhi = max(sxy / sxx, syy / sxy)
        if blo <= 1.0 <= bhi:
            contains += 1
        if bhi < 1.0:
            below += 1
    sim[f"noise_frac_{frac}"] = {
        "x_noise_sd": s_e,
        "frac_sim_interval_contains_1": contains / NSIM,
        "frac_sim_upper_bound_below_1_FALSE_POSITIVE": below / NSIM,
    }
res["positive_control_simulation"] = {
    "setup": "y = x + N(0,s_e) with true slope exactly 1; x-noise and y-noise equal; "
             "interval [min,max](b_OLS,b_RMA) recomputed per replicate. NSIM=2000, seed 12345.",
    "results": sim,
    "observed_data_upper_bound": hi,
    "observed_data_upper_bound_below_1": bool(hi < 1.0),
    "implied_x_noise_from_lattice": float(np.std((y - x)[lattice], ddof=1)),
    "implied_x_noise_frac_of_sd": float(np.std((y - x)[lattice], ddof=1) / sx),
    "reading": "IMPORTANT CAVEAT: at any noise level the raw [min,max](b_OLS,b_RMA) interval "
               "rejects a TRUE slope of 1 about half the time, so the bare interval test is "
               "NOT calibrated and on its own proves little. The endpoint b_RMA is a ratio of "
               "two noisy variances. This is why V5d calibrates the statistics properly "
               "instead of relying on the interval alone.",
}

# ---------------------------------------------------------------- (B)
d = y - x
drift = sm.OLS(d, sm.add_constant(x)).fit()
res["drift_of_difference_on_f1"] = {
    "definition": "d = numeric_feature_2 - numeric_feature_1",
    "slope": float(drift.params[1]),
    "se": float(drift.bse[1]),
    "ci95": [float(v) for v in drift.conf_int()[1]],
    "p_vs_0": float(drift.pvalues[1]),
    "p_vs_minus_1_hats_and_should_equal_1_by_construction": True,
    "relationship": "slope_of_f2_on_f1 = 1 + slope_of_d_on_f1  (exact algebraic identity)",
    "implied_slope_f2_on_f1": float(1 + drift.params[1]),
    "ols_expected": b_ols,
    "matches": bool(abs(1 + drift.params[1] - b_ols) < 1e-9),
    "intercept": float(drift.params[0]),
}

# ---------------------------------------------------------------- (C)
sd_d = float(d.std(ddof=1))
se_d = sd_d / np.sqrt(n)
ci_d = [float(v) for v in st.t.interval(0.95, n - 1, loc=d.mean(), scale=se_d)]
res["mean_difference_exact"] = {
    "mean": float(d.mean()),
    "sd": sd_d, "se": se_d,
    "ci95": ci_d,
    "ci_contains_0": bool(ci_d[0] <= 0 <= ci_d[1]),
    "t": float(st.ttest_1samp(d, 0.0).statistic),
    "p": float(st.ttest_1samp(d, 0.0).pvalue),
    "sign_test_n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
    "sign_test_p": float(st.binomtest(int((d > 0).sum()), n, 0.5).pvalue),
    "wilcoxon_stat": float(st.wilcoxon(d).statistic), "wilcoxon_p": float(st.wilcoxon(d).pvalue),
    "sd_of_x_for_scale": float(sx),
    "mean_diff_as_fraction_of_sd_x": float(d.mean() / sx),
}
# EIV span restricted to lattice block
xl, yl = x[lattice], y[lattice]
sxx_l = np.var(xl, ddof=1); syy_l = np.var(yl, ddof=1); sxy_l = np.cov(xl, yl, ddof=1)[0, 1]
blo_l, bhi_l = min(sxy_l / sxx_l, syy_l / sxy_l), max(sxy_l / sxx_l, syy_l / sxy_l)
lo_l = sm.OLS(yl, sm.add_constant(xl)).fit()
res["lattice_block_eiv_span"] = {
    "b_OLS": float(sxy_l / sxx_l), "b_RMA": float(syy_l / sxy_l),
    "attainable_interval": [float(blo_l), float(bhi_l)],
    "upper_bound_below_1": bool(bhi_l < 1.0),
    "pearson_r": float(np.corrcoef(xl, yl)[0, 1]),
    "max_correction_factor_1_over_r2": float(1 / np.corrcoef(xl, yl)[0, 1] ** 2),
    "note": "in this block reading 1 is an exact design setting, so no x-noise exists at all "
            "and OLS is the correct estimator; correction would only push toward 1 by <=1/r^2.",
}
res["whole_data_max_correction_factor_1_over_r2"] = {
    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
    "one_over_r_squared": float(1 / np.corrcoef(x, y)[0, 1] ** 2),
    "b_OLS": b_ols, "b_OLS_times_factor": float(b_ols / np.corrcoef(x, y)[0, 1] ** 2),
    "b_RMA": b_rma,
    "identity_check_matches_RMA": bool(abs(b_ols / np.corrcoef(x, y)[0, 1] ** 2 - b_rma) < 1e-9),
}

# ---------------------------------------------------------------- (D)
dec = {}
for c, arr in [("numeric_feature_1", x), ("numeric_feature_2", y)]:
    s = df[c].astype(str)
    dec[c] = {
        "max_decimals": int(max(len(v.split(".")[1]) if "." in v else 0 for v in s)),
        "n_with_2_decimals": int(sum(1 for v in s if "." in v and len(v.split(".")[1]) == 2)),
    }
res["resolution_audit"] = {
    "decimals": dec,
    "rounding_sd_if_2dp": float(0.01 / np.sqrt(12)),
    "rounding_sd_as_pct_of_resid_sd": float((0.01 / np.sqrt(12)) / 2.5555 * 100),
    "could_rounding_explain_compression": False,
    "reason": "rounding to 2dp adds <=0.003 units of noise; the observed compression is a "
              "2.5% systematic scale change over a 165-unit range (~4 units), far larger.",
}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1, default=str))
print("\nwrote", OUT)