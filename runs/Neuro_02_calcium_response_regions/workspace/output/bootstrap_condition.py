#!/usr/bin/env python3
"""Stratified nonparametric bootstrap for the receptor-vs-baseline condition
effect (Finding 1 robustness).

Resamples cells WITHIN each (region, batch, slice) stratum WITH REPLACEMENT
(preserving the nested structure and all variables: condition, region, batch,
and y). For each replicate refit  measure ~ region + condition + batch  and
record the receptor-vs-baseline coefficient. The bootstrap distribution is
centered on the observed effect; we report the 95% percentile CI and a
bootstrap p-value P(sign flips). Deterministic: fixed seed.
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
NBOOT = 500
SEED = 42
rng = np.random.default_rng(SEED)
REFKEY = "C(condition, Treatment(reference='baseline'))[T.receptor_stimulated]"
FORMULA = ("C(region, Treatment(reference='region_1')) + "
           "C(condition, Treatment(reference='baseline')) + C(batch)")

# Precompute strata (cell index lists) for resampling.
strata = {k: np.asarray(idx) for k, idx in
          clean.groupby(["region", "batch", "slice_id"]).groups.items()}
strata = list(strata.values())

def fit_coeff(dd, m):
    try:
        mod = smf.ols(f"{m} ~ {FORMULA}", data=dd).fit()
        return float(mod.params[REFKEY])
    except Exception:
        return np.nan

res = {"method": "stratified nonparametric bootstrap (resample cells with replacement "
           "within region,batch,slice strata; refit measure~region+condition+batch); "
           "receptor-vs-baseline coeff; 95% percentile CI + sign-flip p",
       "n_boot": NBOOT, "seed": SEED, "n_cells": int(len(clean)),
       "n_strata": len(strata), "observed_contrast": {}, "per_measure": {}}
for m in MEAS:
    obs = fit_coeff(clean, m)
    coefs = np.empty(NBOOT)
    for b in range(NBOOT):
        rows = []
        for idx in strata:
            n = len(idx)
            pick = rng.integers(0, n, size=n)
            rows.append(idx[pick])
        d = clean.loc[np.concatenate(rows)].reset_index(drop=True)
        coefs[b] = fit_coeff(d, m)
    valid = coefs[~np.isnan(coefs)]
    ci95 = np.percentile(valid, [2.5, 97.5])
    sign_flip_p = float(np.mean(np.sign(valid) != np.sign(obs)))
    dist_p = float(np.mean(np.abs(valid - obs) >= np.abs(obs)))  # not standard; keep sign-flip
    res["per_measure"][m] = {
        "observed_contrast": round(float(obs), 4),
        "bootstrap_mean": round(float(np.mean(valid)), 4),
        "bootstrap_std": round(float(np.std(valid)), 4),
        "ci95_percentile": [round(float(ci95[0]), 4), round(float(ci95[1]), 4)],
        "sign_flip_p": round(sign_flip_p, 5)}
res["observed_contrast"] = {m: res["per_measure"][m]["observed_contrast"] for m in MEAS}
with open(OUT / "bootstrap_condition.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote bootstrap_condition.json (n_strata=%d)" % len(strata))
for m in MEAS:
    pm = res["per_measure"][m]
    print(f"  {m}: obs={pm['observed_contrast']} boot_mean={pm['bootstrap_mean']} "
          f"ci95={pm['ci95_percentile']} sign_flip_p={pm['sign_flip_p']}")
