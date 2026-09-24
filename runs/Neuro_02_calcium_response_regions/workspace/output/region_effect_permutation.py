#!/usr/bin/env python3
"""Permutation null test for the primary REGION effect (Finding 2 robustness).

On the balanced region_1/region_2 subset (both regions span all three
conditions), permute region_1/region_2 labels WITHIN each (batch, condition)
stratum so the controls (batch, condition) are held fixed. Because region is a
function of (batch, slice_id, condition), the correct stratum for the region
main effect is (batch, condition); holding slice fixed would make the permutation
degenerate (a single region per slice/condition/batch cell). For each
permuted assignment, the region_2-vs-region_1 coefficient is obtained by
partialing out the fixed non-region design (condition + batch + intercept)
once (M1 = I - Xbase (Xbase'Xbase)^{-1} Xbase'), then
    b_r2 = (r2' M1 r2)^{-1} r2' M1 y
Report the empirical two-sided p-value and 95% CI for each measure.
Deterministic: fixed seed.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

BASE = "/data/user/conversations/conv-3b1fef04-0d44-4901-ba71-4d83eb3a2660/project"
OUT = Path(BASE) / "output"
df = pd.read_csv(BASE + "/data/cellular_calcium_response_summaries.tsv", sep="\t")
df["region"] = df["anatomical_region_id"]
df["batch"] = df["recording_batch_id"]
clean = df[df.region.isin(["region_1", "region_2"])].copy().reset_index(drop=True)
MEAS = ["mean_amplitude_dff0", "mean_time_to_peak_s", "mean_peak_width_s",
        "mean_peak_prominence_dff0", "mean_peak_frequency_hz", "mean_auc_au"]
NPERM = 1000
SEED = 42

# Fixed non-region design: intercept + condition dummies + batch dummies.
base = smf.ols(
    "mean_amplitude_dff0 ~ C(condition, Treatment(reference='baseline')) + C(batch)",
    data=clean).fit()
Xbase = np.asarray(base.model.exog)
n = Xbase.shape[0]

# Partial-out matrix M1 = I - Xbase (Xbase' Xbase)^{-1} Xbase'.
XtXinv = np.linalg.inv(Xbase.T @ Xbase)
M1 = np.eye(n) - Xbase @ XtXinv @ Xbase.T

Y = {m: clean[m].to_numpy(dtype=float) for m in MEAS}

# Region_2 dummy (observed, ref region_1).
r2 = (clean["region"] == "region_2").to_numpy(dtype=float)

# Precompute strata indices (batch, condition) for permutation.
# Region is a function of (batch, slice_id, condition), so holding condition and
# batch fixed while permuting region_1/region_2 labels is the correct null test
# for the region main effect (region_2 vs region_1, ref region_1).
strata = {k: list(np.asarray(idx)) for k, idx in
          clean.groupby(["batch", "condition"]).groups.items()}

def region2_after_perm(seed_):
    """Region_2 dummy after permuting region_1/region_2 labels within each stratum."""
    r = np.random.default_rng(seed_)
    out = r2.copy()
    for idx in strata.values():
        lab = clean["region"].loc[idx].tolist()
        perm = r.permutation(lab)
        out[idx] = (np.asarray(perm) == "region_2").astype(float)
    return out

def coeff(r2v, y):
    """b_r2 = (r2' M1 r2)^{-1} r2' M1 y."""
    Mr2 = M1 @ r2v
    Mr2y = M1 @ y
    denom = float(r2v @ Mr2)
    if abs(denom) < 1e-12:
        return np.nan
    return float(Mr2y @ r2v / denom)

res = {"method": "permute region_1/region_2 labels within (batch,slice,condition) strata "
           "on the balanced region_1/2 subset; region_2-vs-region_1 coeff via partial-out "
           "M1 (fixed non-region design); two-sided empirical p and 95% CI",
       "n_perms": NPERM, "seed": SEED, "n_cells": int(n),
       "n_strata": len(strata), "observed_contrast": {}, "per_measure": {}}
for m in MEAS:
    obs = coeff(r2, Y[m])
    dist = np.empty(NPERM)
    for p in range(NPERM):
        r2p = region2_after_perm(SEED + p)
        dist[p] = coeff(r2p, Y[m])
    two_sided = float(np.mean(np.abs(dist) >= abs(obs)))
    ci95 = np.percentile(dist, [2.5, 97.5])
    res["per_measure"][m] = {
        "observed_contrast": round(obs, 4),
        "null_median": round(float(np.median(dist)), 4),
        "null_std": round(float(np.std(dist)), 4),
        "empirical_p_two_sided": round(two_sided, 5),
        "empirical_ci95": [round(float(ci95[0]), 4), round(float(ci95[1]), 4)]}
res["observed_contrast"] = {m: res["per_measure"][m]["observed_contrast"] for m in MEAS}
with open(OUT / "region_effect_permutation.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote region_effect_permutation.json")
for m in MEAS:
    print(f"  {m}: p={res['per_measure'][m]['empirical_p_two_sided']} "
          f"obs={res['per_measure'][m]['observed_contrast']} "
          f"ci95={res['per_measure'][m]['empirical_ci95']}")
