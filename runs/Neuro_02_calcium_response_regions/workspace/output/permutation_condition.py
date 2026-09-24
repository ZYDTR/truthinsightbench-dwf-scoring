#!/usr/bin/env python3
"""Permutation null test for the CONDITION effect (Finding 1 robustness).

On the region_1/2 subset, permute condition labels WITHIN each (region, batch,
slice) stratum so all controls (region, batch, slice) are held fixed. For each
permuted assignment, refit the full OLS model
  measure ~ region + condition + batch
and extract the receptor-vs-baseline coefficient. The empirical two-sided p-value
and 95% CI are computed from the null distribution of this coefficient.
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
clean = df[df.region.isin(["region_1", "region_2"])].copy()
MEAS = ["mean_amplitude_dff0", "mean_time_to_peak_s", "mean_peak_width_s",
        "mean_peak_prominence_dff0", "mean_peak_frequency_hz", "mean_auc_au"]
NPERM = 1000
SEED = 42

FORMULA = ("C(region, Treatment(reference='region_1')) + "
           "C(condition, Treatment(reference='baseline')) + C(batch)")
REFKEY = "C(condition, Treatment(reference='baseline'))[T.receptor_stimulated]"

def permute_condition(d, seed):
    r = np.random.default_rng(seed)
    out = d.copy()
    for keys, idx in out.groupby(["region", "batch", "slice_id"]).groups.items():
        idx = list(idx)
        lab = out.loc[idx, "condition"].tolist()
        perm = r.permutation(lab)
        out.loc[idx, "condition"] = perm
    return out

def receptor_coef(dd):
    mod = smf.ols(REFKEY.split("")[0] + " ~ " + FORMULA, data=dd)  # placeholder; see below
    return None

def fit_coef(dd, m):
    mod = smf.ols(f"{m} ~ {FORMULA}", data=dd).fit()
    return float(mod.params[REFKEY])

res = {"method": "permute condition labels within (region,batch,slice) strata; refit "
           "measure ~ region+condition+batch each time; receptor-vs-baseline coeff; "
           "two-sided empirical p and 95% CI",
       "n_perms": NPERM, "seed": SEED, "n_cells": int(len(clean)),
       "observed_contrast": {}, "per_measure": {}}
for m in MEAS:
    obs = fit_coef(clean, m)
    dist = np.empty(NPERM)
    for p in range(NPERM):
        d = permute_condition(clean, SEED + p)
        dist[p] = fit_coef(d, m)
    two_sided = float(np.mean(np.abs(dist) >= abs(obs)))
    ci95 = np.percentile(dist, [2.5, 97.5])
    res["per_measure"][m] = {
        "observed_contrast": round(obs, 4),
        "null_median": round(float(np.median(dist)), 4),
        "null_std": round(float(np.std(dist)), 4),
        "empirical_p_two_sided": round(two_sided, 5),
        "empirical_ci95": [round(float(ci95[0]), 4), round(float(ci95[1]), 4)]}
res["observed_contrast"] = {m: res["per_measure"][m]["observed_contrast"] for m in MEAS}
with open(OUT / "permutation_condition.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote permutation_condition.json")
for m in MEAS:
    print(f"  {m}: p={res['per_measure'][m]['empirical_p_two_sided']} "
          f"obs={res['per_measure'][m]['observed_contrast']} "
          f"ci95={res['per_measure'][m]['empirical_ci95']}")
