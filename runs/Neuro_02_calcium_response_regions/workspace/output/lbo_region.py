#!/usr/bin/env python3
"""Leave-one-batch-out (grouped holdout) robustness for the primary REGION
effect (Finding 2).

For each of the 9 batches, remove it, refit on the balanced region_1/region_2
subset  measure ~ region + condition + batch, and record the region_2-vs-region_1
coefficient (ref region_1). This tests whether the region effect persists
independent of any single batch/age. Deterministic: fixed seed.
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
REFKEY = "C(region, Treatment(reference='region_1'))[T.region_2]"
FORMULA = ("C(region, Treatment(reference='region_1')) + "
           "C(condition, Treatment(reference='baseline')) + C(batch)")
SEED = 42

res = {"method": "leave-one-batch-out on balanced region_1/2 subset; refit "
           "measure ~ region+condition+batch; region_2-vs-region_1 coeff; "
           "report per-batch coeff, sign stability, and p-value",
       "seed": SEED, "n_batches": int(clean.batch.nunique()),
       "observed_full": {}, "per_measure": {}}

# full-sample coefficient for reference
for m in MEAS:
    mod = smf.ols(f"{m} ~ {FORMULA}", data=clean).fit()
    res["observed_full"][m] = round(float(mod.params[REFKEY]), 4)

for m in MEAS:
    per_batch = []
    for b in sorted(clean.batch.unique()):
        d = clean[clean.batch != b].copy()
        # ensure both regions remain
        if not set(d.region.unique()) >= {"region_1", "region_2"}:
            per_batch.append(None)
            continue
        mod = smf.ols(f"{m} ~ {FORMULA}", data=d).fit()
        v = float(mod.params[REFKEY])
        p = float(mod.pvalues[REFKEY])
        per_batch.append({"batch": b, "n_cells": int(d.shape[0]),
                          "coef": round(v, 4), "p": round(p, 6)})
    coefs = [x["coef"] for x in per_batch if x["coef"] is not None]
    res["per_measure"][m] = {
        "n_batches_tested": len(coefs),
        "all_same_sign_as_full": all(np.sign(v) == np.sign(res["observed_full"][m]) for v in coefs),
        "n_significant": sum(1 for x in per_batch if x and x["p"] < 0.05),
        "batch_coefs": per_batch,
    }

with open(OUT / "lbo_region.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote lbo_region.json")
for m in MEAS:
    pm = res["per_measure"][m]
    print(f"  {m}: same_sign={pm['all_same_sign_as_full']} "
          f"sig={pm['n_significant']}/{pm['n_batches_tested']}")
