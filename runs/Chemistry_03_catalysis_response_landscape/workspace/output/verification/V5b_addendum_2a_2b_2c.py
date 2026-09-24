#!/usr/bin/env python3
"""
V5b -- addendum to V5. Fixes and sharpens three things:

1. PARAMETERISATION FIX. In V5's deming() the tuning parameter obeys the
   standard identity
        beta(lambda) = [ (Syy - lambda*Sxx) + sqrt((Syy - lambda*Sxx)^2
                        + 4*lambda*Sxy^2) ] / (2*Sxy)
   with lambda = var(error_y) / var(error_x)   <-- NOT var(x)/var(y) as V5's
   docstring claimed. Consequences, both re-verified numerically here:
        lambda -> 0   gives beta -> Syy/Sxy  (inverse regression / RMA-like)
        lambda -> inf gives beta -> Sxy/Sxx  (ordinary least squares)
   So the *set of every slope attainable by ANY errors-in-variables correction*
   is exactly the closed interval [b_OLS, b_RMA]. This is the correct and much
   sharper form of test (b): the correction is BOUNDED, so we can bound how far
   attenuation alone can move the slope.

2. Proper p-values for H0: slope = 1 (V5 only reported H0: slope = 0).
   These decide whether "CI excludes 1" survives on the subsets.

3. Discordance counted on the whole file (all 41 rows) AND on the 33 irregular
   rows, to see if the claim's "22 of 528 pairs" is a subset count (33*32/2 =
   528) or a whole-file count (41*40/2 = 820).

Extra scale-free checks: log-space slope (multiplicative-error EIV analogue),
ratio test, reverse regression. Writes V5b_addendum.json.
"""

import json
import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/observation_set_2.tsv"
OUT = os.path.join(HERE, "V5b_addendum.json")

df = pd.read_csv(DATA, sep="\t")
x = df["numeric_feature_1"].to_numpy(float)
y = df["numeric_feature_2"].to_numpy(float)
n = len(x)
idx = df["observation_index"].to_numpy(int)
lattice = idx < 8
irregular = ~lattice
res = {}


def fit(xx, yy):
    m = sm.OLS(yy, sm.add_constant(xx)).fit()
    ci = m.conf_int(alpha=0.05)[1]
    se = float(m.bse[1])
    b = float(m.params[1])
    t1 = (b - 1.0) / se
    dfree = len(xx) - 2
    return {
        "n": int(len(xx)), "slope": b, "intercept": float(m.params[0]), "se_slope": se,
        "ci95": [float(ci[0]), float(ci[1])],
        "ci_excludes_1": bool(ci[1] < 1.0 or ci[0] > 1.0),
        "t_vs_0": float(m.tvalues[1]), "p_vs_0": float(m.pvalues[1]),
        "t_vs_1": float(t1), "df": int(dfree),
        "p_vs_1": float(2 * st.t.sf(abs(t1), dfree)),
        "p_vs_1_lt_0.05": bool(2 * st.t.sf(abs(t1), dfree) < 0.05),
        "r2": float(m.rsquared),
    }


# ---------------------------------------------------- 1. Deming orientation
def deming(xx, yy, lam):
    xx = np.asarray(xx, float); yy = np.asarray(yy, float)
    Sxx = np.var(xx, ddof=1); Syy = np.var(yy, ddof=1)
    Sxy = np.cov(xx, yy, ddof=1)[0, 1]
    b = ((Syy - lam * Sxx) + np.sqrt((Syy - lam * Sxx) ** 2 + 4 * lam * Sxy ** 2)) / (2 * Sxy)
    return float(b), float(yy.mean() - b * xx.mean())


Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0, 1]
b_ols = Sxy / Sxx
b_rma = Syy / Sxy
res["deming_orientation_check"] = {
    "note": "lambda = var(err_y)/var(err_x); lam->0 gives Syy/Sxy, lam->inf gives Sxy/Sxx",
    "Sxx": float(Sxx), "Syy": float(Syy), "Sxy": float(Sxy),
    "Sxy_over_Sxx_OLS": float(b_ols),
    "Syy_over_Sxy_RMA": float(b_rma),
    "deming_lam_1e-12": deming(x, y, 1e-12)[0],
    "deming_lam_1e12": deming(x, y, 1e12)[0],
    "deming_lam_1": deming(x, y, 1.0)[0],
    "limits_match_OLS_and_RMA": bool(
        abs(deming(x, y, 1e12)[0] - b_ols) < 1e-6 and abs(deming(x, y, 1e-12)[0] - b_rma) < 1e-6
    ),
}

# monotonicity of beta(lambda) => attainable set is exactly the interval [b_ols, b_rma]
lams = np.logspace(-8, 8, 400)
betas = np.array([deming(x, y, l)[0] for l in lams])
res["attainable_slope_interval_over_ALL_noise_ratios"] = {
    "min": float(betas.min()), "max": float(betas.max()),
    "interval": [float(b_ols), float(b_rma)],
    "monotone_decreasing_in_lambda": bool(np.all(np.diff(betas) <= 1e-12)),
    "b_OLS": float(b_ols), "b_RMA": float(b_rma),
    "gap_between_OLS_and_RMA": float(b_rma - b_ols),
    "max_attainable_slope": float(max(b_ols, b_rma)),
    "max_attainable_below_1_by": float(1.0 - max(b_ols, b_rma)),
    "headline": (
        "No choice whatsoever of noise ratio can push the corrected slope above "
        f"{max(b_ols, b_rma):.5f}; the EIV correction can move the slope by at most "
        f"{abs(b_rma - b_ols):.5f}."
    ),
}

# bootstrap the two endpoints = the bounds of the attainable interval
RNG = np.random.default_rng(7)
B = 20000
bi = RNG.integers(0, n, (B, n))
b_o = np.empty(B); b_r = np.empty(B); b_tls = np.empty(B)
for k in range(B):
    ii = bi[k]; xk = x[ii]; yk = y[ii]
    sxx = np.var(xk, ddof=1); syy = np.var(yk, ddof=1); sxy = np.cov(xk, yk, ddof=1)[0, 1]
    b_o[k] = sxy / sxx
    b_r[k] = syy / sxy
    b_tls[k] = deming(xk, yk, 1.0)[0]
mx = np.maximum(b_o, b_r)
res["bootstrap_bounds_of_attainable_interval"] = {
    "B": B, "seed": 7, "method": "nonparametric case resampling, max(b_OLS,b_RMA) per replicate",
    "b_OLS_ci95": [float(np.percentile(b_o, 2.5)), float(np.percentile(b_o, 97.5))],
    "b_RMA_ci95": [float(np.percentile(b_r, 2.5)), float(np.percentile(b_r, 97.5))],
    "upper_bound_stat_ci95": [float(np.percentile(mx, 2.5)), float(np.percentile(mx, 97.5))],
    "upper_bound_ci_excludes_1": bool(np.percentile(mx, 97.5) < 1.0),
    "upper_bound_ci_upper_endpoint": float(np.percentile(mx, 97.5)),
    "frac_replicates_upper_bound_at_or_above_1": float(np.mean(mx >= 1.0)),
    "tls_lam1_ci95": [float(np.percentile(b_tls, 2.5)), float(np.percentile(b_tls, 97.5))],
    "tls_lam1_ci_excludes_1": bool(np.percentile(b_tls, 97.5) < 1.0),
    "tls_lam1_frac_below_1": float(np.mean(b_tls < 1.0)),
}

# ---------------------------------------------------- 2. slope-vs-1 p-values
res["fits_with_H0_slope_eq_1"] = {
    "all_41_rows": fit(x, y),
    "lattice_8_rows": fit(x[lattice], y[lattice]),
    "irregular_33_rows": fit(x[irregular], y[irregular]),
}
# excluding the single largest-disagreement row (observation_index 19)
m = idx != 19
res["fits_with_H0_slope_eq_1"]["all_40_excluding_idx19"] = fit(x[m], y[m])
# excluding the two 80+ outliers (idx 19, 20)
m2 = ~np.isin(idx, [19, 20])
res["fits_with_H0_slope_eq_1"]["all_39_excluding_idx19_20"] = fit(x[m2], y[m2])

# ---------------------------------------------------- 3. discordance counts
def disc(xx, yy):
    c = d = t = 0
    for i in range(len(xx)):
        for k in range(i + 1, len(xx)):
            sx = np.sign(xx[i] - xx[k]); sy = np.sign(yy[i] - yy[k])
            if sx * sy > 0:
                c += 1
            elif sx * sy < 0:
                d += 1
            else:
                t += 1
    return c, d, t


c_all, d_all, t_all = disc(x, y)
c_irr, d_irr, t_irr = disc(x[irregular], y[irregular])
c_lat, d_lat, t_lat = disc(x[lattice], y[lattice])
res["discordance_counts"] = {
    "all_41_rows": {"pairs": c_all + d_all + t_all, "concordant": c_all, "discordant": d_all, "tied": t_all},
    "irregular_33_rows": {"pairs": c_irr + d_irr + t_irr, "concordant": c_irr, "discordant": d_irr, "tied": t_irr},
    "lattice_8_rows": {"pairs": c_lat + d_lat + t_lat, "concordant": c_lat, "discordant": d_lat, "tied": t_lat},
    "claim_22_of_528": {
        "whole_file_would_be_820_pairs": bool(c_all + d_all == 820),
        "irregular_subset_is_528_pairs": bool(c_irr + d_irr + t_irr == 528),
        "irregular_subset_discordant_equals_22": bool(d_irr == 22),
        "whole_file_discordant_equals_22": bool(d_all == 22),
        "verdict": (
            "reconciles as the 33 irregular rows only"
            if (d_irr == 22 and c_irr + d_irr + t_irr == 528)
            else ("reconciles as whole file" if d_all == 22 else "DOES NOT reconcile either way")
        ),
    },
    "spearman_from_irregular_concordance": float((c_irr - d_irr) / (c_irr + d_irr + t_irr)),
}

# ---------------------------------------------------- 4. scale-free checks
# multiplicative-error analogue: work in log space on same-sign, non-tiny pairs
mask = (np.sign(x) == np.sign(y)) & (np.abs(x) > 1) & (np.abs(y) > 1)
lx = np.log(np.abs(x[mask])); ly = np.log(np.abs(y[mask]))
logfit = sm.OLS(ly, sm.add_constant(lx)).fit()
res["log_space_multiplicative_check"] = {
    "n_pairs_used": int(mask.sum()),
    "slope_OLS_in_logs": float(logfit.params[1]),
    "se": float(logfit.bse[1]),
    "ci95": [float(v) for v in logfit.conf_int()[1]],
    "ci_excludes_1": bool(logfit.conf_int()[1][1] < 1.0 or logfit.conf_int()[1][0] > 1.0),
    "p_vs_1": float(2 * st.t.sf(abs((logfit.params[1] - 1) / logfit.bse[1]), int(mask.sum()) - 2)),
    "interpretation": "scale-free (multiplicative) compression: a power law y ~ x^0.98 fits better than slope 1",
}
# heteroscedasticity of |resid| vs |x|
r = y - (np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1])
h = sm.OLS(np.abs(r), sm.add_constant(np.abs(x))).fit()
res["heteroscedasticity_absresid_on_absx"] = {
    "slope": float(h.params[1]), "p": float(h.pvalues[1]),
    "resid_sd_at_x0": float(np.std(r[np.abs(x) < 30], ddof=1)),
    "resid_sd_at_x80": float(np.std(r[np.abs(x) > 60], ddof=1)),
}
# reverse regression bound
rr = sm.OLS(x, sm.add_constant(y)).fit()
res["reverse_regression"] = {
    "slope_x_on_y": float(rr.params[1]),
    "implied_y_on_x": float(1 / rr.params[1]),
    "note": "1/(slope of x on y) is the lambda->0 Deming endpoint; both bracket OLS",
}
res["ratio_check_same_sign_only"] = {
    "n": int(mask.sum()),
    "mean_ratio": float((y[mask] / x[mask]).mean()),
    "p_ratio_eq_1": float(st.ttest_1samp(y[mask] / x[mask], 1.0).pvalue),
    "median_ratio": float(np.median(y[mask] / x[mask])),
    "frac_below_1": float(np.mean(y[mask] / x[mask] < 1)),
}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2, default=str)

print(json.dumps(res, indent=1, default=str))
print("\nwrote", OUT)