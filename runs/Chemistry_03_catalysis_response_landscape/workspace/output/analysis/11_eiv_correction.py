#!/usr/bin/env python3
"""Step 11 - Correction of the errors-in-variables argument in Finding 5.

The first version of the Finding-5 robustness argument claimed that sweeping the
Deming noise ratio lambda from 0 to infinity bounded the corrected slope to
[0.97516, 0.97814], and treated the upper endpoint as evidence against
attenuation.  A positive control run by an independent verification worker
(output/verification/V5c_positive_control.json) showed that this interval
contains the true slope of 1 in only ~15% of replicates when the truth IS 1, i.e.
the interval is far too narrow to serve as a bound.

This script (a) reproduces that positive control, (b) applies the correct
de-attenuation test b_OLS / r^2, and (c) runs calibrated null simulations for
the three noise models.  Its results are what Result.md now reports.

Outputs: evidence/11_eiv_correction.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
rng = np.random.default_rng(20260919)
OUT: dict = {}

d2 = pd.read_csv(DATA / "observation_set_2.tsv", sep="\t")
x = d2["numeric_feature_1"].to_numpy(float)
y = d2["numeric_feature_2"].to_numpy(float)
r = float(np.corrcoef(x, y)[0, 1])
b_ols = float(np.polyfit(x, y, 1)[0])
OUT["observed"] = {
    "n": int(len(x)),
    "pearson_r": r,
    "r_squared": r ** 2,
    "b_OLS": b_ols,
    "b_RMA": float(((y - y.mean()) ** 2).sum() / ((x - x.mean()) * (y - y.mean())).sum()),
    "b_OLS_over_r2_deattenuated": b_ols / r ** 2,
    "max_upward_correction_pct": 100 * (1 / r ** 2 - 1),
}


def b_ols_rma(xv, yv):
    sxx = ((xv - xv.mean()) ** 2).sum()
    syy = ((yv - yv.mean()) ** 2).sum()
    sxy = ((xv - xv.mean()) * (yv - yv.mean())).sum()
    return sxy / sxx, syy / sxy


# --------------------- (a) positive control that breaks the withdrawn argument
OUT["positive_control_withdrawn_interval"] = {}
for se in [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]:
    hits = 0
    nsim = 3000
    for _ in range(nsim):
        xt = rng.normal(15, 46, 41)
        xo = xt + rng.normal(0, se, 41)
        yo = xt + rng.normal(0, se, 41)          # true slope is exactly 1
        b1, b2 = b_ols_rma(xo, yo)
        if min(b1, b2) <= 1.0 <= max(b1, b2):
            hits += 1
    OUT["positive_control_withdrawn_interval"][f"noise_sd_{se}"] = {
        "frac_interval_contains_1": hits / nsim, "n_sim": nsim,
        "verdict": "interval too narrow" if hits / nsim < 0.4 else "ok",
    }

# --------------------- (b) positive control for the CORRECT test
OUT["positive_control_deattenuation"] = {}
for sx in [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]:
    frac = 0
    nsim = 3000
    for _ in range(nsim):
        xt = rng.normal(15, 46, 41)
        xo = xt + rng.normal(0, sx, 41)
        yo = xt.copy()                            # true slope exactly 1, noise on x only
        rr = float(np.corrcoef(xo, yo)[0, 1])
        bb = float(np.polyfit(xo, yo, 1)[0])
        if bb / rr ** 2 >= 1.0:
            frac += 1
    OUT["positive_control_deattenuation"][f"x_noise_sd_{sx}"] = {
        "frac_corrected_slope_reaches_1": frac / nsim, "n_sim": nsim}


# --------------------- (c) calibrated null simulations, three noise models
def null_ols_p(obs, y_noise_sd, n, nsim=10000):
    sims = np.empty(nsim)
    for i in range(nsim):
        xt = rng.normal(15, 46, n)
        xo = xt + rng.normal(0, y_noise_sd, n)
        yo = xt + rng.normal(0, y_noise_sd, n)
        sims[i] = np.polyfit(xo, yo, 1)[0]
    return {"observed": obs, "null_mean": float(sims.mean()),
            "null_sd": float(sims.std(ddof=1)),
            "p_one_sided_ge_observed": float((sims <= obs).mean()),
            "n_sim": nsim}


resid_sd_all = float((y - np.polyval(np.polyfit(x, y, 1), x)).std(ddof=1))
OUT["null_equal_noise_all41"] = null_ols_p(b_ols, resid_sd_all, 41)

lat = d2.iloc[0:8]
xl = lat["numeric_feature_1"].to_numpy(float)
yl = lat["numeric_feature_2"].to_numpy(float)
b_lat = float(np.polyfit(xl, yl, 1)[0])
# Reading 1 is an EXACT design setting in this block, so the noise of reading 2 is
# measured about the design value itself, NOT about the fitted line (which would
# absorb part of the slope deficit and understate the noise).
resid_lat = float((yl - xl).std(ddof=1))
OUT["lattice_block"] = {
    "n": 8, "design_values": [float(v) for v in xl],
    "b_OLS": b_lat,
    "residual_sd_of_reading2_about_design_value": resid_lat,
    "residual_sd_about_fitted_line_incorrect": float(
        (yl - np.polyval(np.polyfit(xl, yl, 1), xl)).std(ddof=1)),
    "mean_offset_reading2_minus_design": float((yl - xl).mean()),
    "note": "reading 1 takes exact design settings here, so no de-attenuation is "
            "needed and OLS is the correct estimator; the noise of reading 2 is "
            "measured about the design value",
}
sims = np.empty(10000)
for i in range(10000):
    yo = xl + rng.normal(0, resid_lat, 8)         # true slope exactly 1
    sims[i] = np.polyfit(xl, yo, 1)[0]
OUT["null_lattice_true_slope_1"] = {
    "observed": b_lat, "null_mean": float(sims.mean()),
    "null_sd": float(sims.std(ddof=1)),
    "p_one_sided_ge_observed": float((sims <= b_lat).mean()), "n_sim": 10000}
# sensitivity: with the larger irregular-block noise
for sd in [2.9, 3.5]:
    s2 = np.empty(6000)
    for i in range(6000):
        yo = xl + rng.normal(0, sd, 8)
        s2[i] = np.polyfit(xl, yo, 1)[0]
    OUT["null_lattice_true_slope_1"][f"sensitivity_noise_sd_{sd}"] = {
        "p_one_sided_ge_observed": float((s2 <= b_lat).mean()), "n_sim": 6000}

# --------------------- (d) bootstrap the corrected statistic
bs = np.empty(20000)
for i in range(20000):
    idx = rng.integers(0, 41, 41)
    rr = float(np.corrcoef(x[idx], y[idx])[0, 1])
    bb = float(np.polyfit(x[idx], y[idx], 1)[0])
    bs[i] = bb / rr ** 2
OUT["bootstrap_deattenuated"] = {
    "point": b_ols / r ** 2,
    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "frac_reaching_1": float((bs >= 1).mean()), "n_boot": 20000,
}
OUT["verdict"] = {
    "withdrawn_argument": "the lambda-sweep interval [0.97516, 0.97814] is ~10x too "
                          "narrow to bracket the truth and is not used as evidence",
    "current_support": "de-attenuation moves the slope by at most 0.31% (b/r^2 = "
                       "0.978144); the lattice block, where reading 1 is an exact "
                       "design setting, gives slope 0.9616 at calibrated p = 0.019",
    "boundary": "the log-scale (multiplicative) test does NOT reject slope 1 "
                "(0.9328, CI [0.8426, 1.0231]); the claim is scoped to the arithmetic scale",
}

with (EV / "11_eiv_correction.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))