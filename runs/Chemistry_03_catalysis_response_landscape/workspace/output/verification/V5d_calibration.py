#!/usr/bin/env python3
"""
V5d -- CALIBRATION of the key test statistics for CLAIM D5. This is the decisive
check on whether "the compressed slope" survives its most serious alternative
(errors-in-variables).

V5b established the algebraic fact that for any assumed noise ratio lambda the
Deming point estimate lies in [b_OLS, b_RMA] = [0.97516, 0.97814]; the whole
EIV correction budget (0.00299) is ~7x smaller than the shortfall from 1
(0.02186). V5c's positive control showed the ENDPOINTS of that interval are
noisy statistics, so the bound's *statistical* strength must be calibrated.
Here we do exactly that, under two worlds in which the TRUE slope is 1:

  World A ("OLS-valid"): x is an exact design setting (no x-noise).
      y = x + e, e ~ N(0, s_e). Correct estimator = OLS. Tests whether
      "OLS slope 0.9752, CI excludes 1" is a calibrated rejection.

  World B ("EIV", lambda=1): equal noise on both readings, true slope 1.
      x_obs = xi + u, y_obs = xi + v, var(u)=var(v). Correct estimator =
      Deming(lambda=1). Tests whether the observed Deming(1)=0.9766 and the
      observed b_RMA=0.9781 are unusual under beta=1.

  World C: same as A but the LATTICE block only (n=8), where x is trivially
      exact; y = x + e with e ~ N(0, 1.10) taken from the lattice residuals
      about the reference line y = x. Tests the lattice-only slope 0.9616.

Reports, per world: the distribution of the estimator, the fraction of
replicates at or below the observed statistic (= calibrated one-sided p), the
95% CI coverage of beta=1 (should be ~0.95 if the CI is honest), and the
fraction of CIs that exclude 1 with the slope on the low side (type-I error).
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/observation_set_2.tsv"
OUT = os.path.join(HERE, "V5d_calibration.json")

df = pd.read_csv(DATA, sep="\t")
x = df["numeric_feature_1"].to_numpy(float)
y = df["numeric_feature_2"].to_numpy(float)
idx = df["observation_index"].to_numpy(int)
n = len(x)
lattice = idx < 8
RNG = np.random.default_rng(999)
NSIM = 4000
res = {}

# observed statistics
m_obs = sm.OLS(y, sm.add_constant(x)).fit()
b_obs = float(m_obs.params[1]); se_obs = float(m_obs.bse[1])
resid_obs = float(np.sqrt(m_obs.mse_resid))
Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0, 1]
b_rma_obs = float(Syy / Sxy)

xl = x[lattice]; yl = y[lattice]
nl = len(xl)
m_lat = sm.OLS(yl, sm.add_constant(xl)).fit()
b_lat_obs = float(m_lat.params[1])
y_noise_lat = float(np.std(yl - xl, ddof=1))


def ols_stat(xx, yy):
    b = float(np.polyfit(xx, yy, 1)[0])
    r = yy - (np.polyfit(xx, yy, 1)[0] * xx + np.polyfit(xx, yy, 1)[1])
    se = float(np.sqrt((r ** 2).sum() / (len(xx) - 2) / ((xx - xx.mean()) ** 2).sum()))
    return b, se


def deming_stat(xx, yy, lam):
    sxx = np.var(xx, ddof=1); syy = np.var(yy, ddof=1); sxy = np.cov(xx, yy, ddof=1)[0, 1]
    b = ((syy - lam * sxx) + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return float(b)


# ------------------------------------------------------------ World A
bs, ses = np.empty(NSIM), np.empty(NSIM)
for k in range(NSIM):
    yy = x + RNG.normal(0, resid_obs, n)
    bs[k], ses[k] = ols_stat(x, yy)
ci_lo = bs - st.t.ppf(0.975, n - 2) * ses
ci_hi = bs + st.t.ppf(0.975, n - 2) * ses
res["worldA_x_exact_design_true_slope_1"] = {
    "n": n, "nsim": NSIM, "y_noise_sd": resid_obs,
    "statistic": "OLS slope of y on x",
    "observed_slope": b_obs,
    "sim_mean": float(bs.mean()), "sim_sd": float(bs.std(ddof=1)),
    "sim_pct2.5": float(np.percentile(bs, 2.5)), "sim_pct97.5": float(np.percentile(bs, 97.5)),
    "calibrated_one_sided_p_for_observed": float(np.mean(bs <= b_obs)),
    "simulated_ci95_coverage_of_1": float(np.mean((ci_lo <= 1) & (1 <= ci_hi))),
    "sim_frac_ci_excludes_1_low_side_TYPE_I": float(np.mean(ci_hi < 1)),
    "sim_frac_ci_excludes_1_high_side": float(np.mean(ci_lo > 1)),
    "observed_ci_lo": float(m_obs.conf_int()[1][0]), "observed_ci_hi": float(m_obs.conf_int()[1][1]),
    "interpretation": "if x carries no noise this is the honest test; observed p is the calibrated p",
}

# ------------------------------------------------------------ World B
# lambda = 1: equal noise. Calibrate total noise so the y-residual scatter about
# the fitted line matches the observed 2.5555:  var(v) + beta^2*var(u) = 2*s^2
s_e = resid_obs / np.sqrt(2.0)
bd = np.empty(NSIM); brma = np.empty(NSIM); bols = np.empty(NSIM)
for k in range(NSIM):
    u = RNG.normal(0, s_e, n)
    v = RNG.normal(0, s_e, n)
    xo = x + u
    yo = x + v
    bd[k] = deming_stat(xo, yo, 1.0)
    brma[k] = np.var(yo, ddof=1) / np.cov(xo, yo, ddof=1)[0, 1]
    bols[k] = np.cov(xo, yo, ddof=1)[0, 1] / np.var(xo, ddof=1)
# honest CI for Deming(1) via the delta/bootstrap-free analytic approximation:
# var(b_deming) ~ (var of OLS-like scatter)/(n*Sxx) rescaled; use the observed
# bootstrap sd from V5b as the width and check calibration of the *point* stat only
res["worldB_eiv_equal_noise_true_slope_1"] = {
    "n": n, "nsim": NSIM, "lambda": 1.0,
    "y_noise_sd": s_e, "x_noise_sd": s_e,
    "note": "x simulated as the observed x plus independent noise; both readings equally noisy; TRUE slope 1",
    "deming1_observed": deming_stat(x, y, 1.0),
    "sim_mean_deming1": float(bd.mean()), "sim_sd_deming1": float(bd.std(ddof=1)),
    "sim_pct2.5_deming1": float(np.percentile(bd, 2.5)),
    "calibrated_one_sided_p_deming1": float(np.mean(bd <= deming_stat(x, y, 1.0))),
    "b_rma_observed": b_rma_obs,
    "sim_mean_b_rma": float(brma.mean()), "sim_sd_b_rma": float(brma.std(ddof=1)),
    "calibrated_one_sided_p_b_rma": float(np.mean(brma <= b_rma_obs)),
    "sim_mean_b_ols": float(bols.mean()), "sim_sd_b_ols": float(bols.std(ddof=1)),
    "observed_b_ols": b_obs,
    "calibrated_one_sided_p_b_ols": float(np.mean(bols <= b_obs)),
    "interpretation": "how unusual the observed compressed slope is if the two readings "
                      "really are equally-noisy measurements of the same quantity with true slope 1",
}

# ------------------------------------------------------------ World C (lattice)
bl = np.empty(NSIM); sl = np.empty(NSIM)
for k in range(NSIM):
    yy = xl + RNG.normal(0, y_noise_lat, nl)
    bl[k], sl[k] = ols_stat(xl, yy)
ci_l_lo = bl - st.t.ppf(0.975, nl - 2) * sl
ci_l_hi = bl + st.t.ppf(0.975, nl - 2) * sl
res["worldC_lattice_block_x_exact_true_slope_1"] = {
    "n": nl, "nsim": NSIM, "y_noise_sd": y_noise_lat,
    "statistic": "OLS slope of y on x (lattice rows only; x is an exact design setting)",
    "observed_slope": b_lat_obs,
    "sim_mean": float(bl.mean()), "sim_sd": float(bl.std(ddof=1)),
    "calibrated_one_sided_p_for_observed": float(np.mean(bl <= b_lat_obs)),
    "simulated_ci95_coverage_of_1": float(np.mean((ci_l_lo <= 1) & (1 <= ci_l_hi))),
    "sim_frac_ci_excludes_1_low_side_TYPE_I": float(np.mean(ci_l_hi < 1)),
    "observed_ci_lo": float(m_lat.conf_int()[1][0]), "observed_ci_hi": float(m_lat.conf_int()[1][1]),
    "interpretation": "this block is EIV-free by construction (x = -60..80 exact); "
                      "its p-value needs no attenuation correction",
}
# also: lattice under a noisier y assumption calibrated to the irregular block
alt_sd = 2.9
bl2 = np.array([ols_stat(xl, xl + RNG.normal(0, alt_sd, nl))[0] for _ in range(NSIM)])
res["worldC_sensitivity_to_y_noise"] = {
    "note": "same lattice test but with y-noise set to the larger irregular-block value",
    "y_noise_sd": alt_sd, "n": nl, "nsim": NSIM,
    "calibrated_one_sided_p_for_observed": float(np.mean(bl2 <= b_lat_obs)),
    "ci95_coverage_of_1": float(np.mean(
        (bl2 - st.t.ppf(0.975, nl - 2) * bl2.std(ddof=1) <= 1))),
}

# ------------------------------------------- union of EIV-corrected CIs
lam_grid = np.logspace(-8, 8, 300)
pts = np.array([deming_stat(x, y, l) for l in lam_grid])
res["union_of_corrected_point_estimates"] = {
    "min_over_lambda": float(pts.min()), "max_over_lambda": float(pts.max()),
    "covers_the_interval": [float(min(b_obs, b_rma_obs)), float(max(b_obs, b_rma_obs))],
    "observed_ols": b_obs,
    "shortfall_from_1_of_best_case": float(1 - pts.max()),
    "total_eiv_correction_budget": float(pts.max() - pts.min()),
    "ratio_shortfall_to_budget": float((1 - pts.max()) / (pts.max() - pts.min())),
    "best_case_95pct_upper_limit_v5b": 0.998424930294016,
    "best_case_upper_limit_below_1": bool(0.998424930294016 < 1.0),
}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, default=str)
print(json.dumps(res, indent=1, default=str))
print("\nwrote", OUT)