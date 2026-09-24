#!/usr/bin/env python3
"""Quantify the confounding structures that bound the region and age findings.

(a) Region x condition coverage confound (Finding 2 boundary):
    - Full-model OLS region AUC main effect (region_3 vs region_1).
    - Direct within-condition comparison: region_3 (only receptor_stimulated) vs
      region_1 restricted to receptor_stimulated. If these are ~equal, the
      region_3 AUC "effect" is entirely the receptor effect, not a region effect.

(b) Batch = age perfect confound (Finding 3/4 boundary):
    - Confirm each batch has a single age; report the number of batches.
    - Show that adding both age and batch to a model is singular (rank-deficient).

(c) Multicollinearity via Variance Inflation Factor (VIF) for the full
    condition + region + batch model.

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
MEAS_AUC = "mean_auc_au"

res = {}

# --- (a) region x condition coverage confound ---
mod_full = smf.ols(
    f"{MEAS_AUC} ~ C(region, Treatment(reference='region_1')) + "
    "C(condition, Treatment(reference='baseline')) + C(batch)", data=df).fit()
region3_coef = float(mod_full.params["C(region, Treatment(reference='region_1'))[T.region_3]"])
region3_p = float(mod_full.pvalues["C(region, Treatment(reference='region_1'))[T.region_3]"])

# Direct within-condition comparison: region_3 (all receptor_stimulated) vs
# region_1 restricted to receptor_stimulated.
rs = df[df.condition == "receptor_stimulated"]
auc_r3 = float(rs.loc[rs.region == "region_3", MEAS_AUC].mean())
auc_r1_rs = float(rs.loc[rs.region == "region_1", MEAS_AUC].mean())
auc_r2_rs = float(rs.loc[rs.region == "region_2", MEAS_AUC].mean())
auc_r4 = float(df.loc[(df.region == "region_4") & (df.condition != "receptor_stimulated"), MEAS_AUC].mean())
auc_r4_rs = None  # region_4 has no receptor_stimulated

res["region_condition_confound"] = {
    "full_model_region3_vs_region1_AUC_coef": round(region3_coef, 4),
    "full_model_region3_vs_region1_AUC_p": round(region3_p, 5),
    "within_receptor_stimulated_AUC": {
        "region_1": round(auc_r1_rs, 4), "region_2": round(auc_r2_rs, 4),
        "region_3": round(auc_r3, 4),
        "region_3_vs_region_1_receptor": round(auc_r3 - auc_r1_rs, 4),
        "region_2_vs_region_1_receptor": round(auc_r2_rs - auc_r1_rs, 4),
        "region_1_n": int(rs.loc[rs.region == "region_1"].shape[0]),
        "region_2_n": int(rs.loc[rs.region == "region_2"].shape[0]),
        "region_3_n": int(rs.loc[rs.region == "region_3"].shape[0]),
    },
    "region_4_AUC_non_receptor": round(auc_r4, 4),
    "region_4_n_non_receptor": int(df.loc[(df.region == "region_4") & (df.condition != "receptor_stimulated")].shape[0]),
    "interpretation": "The full-model region_3 AUC contrast is driven by region_3 "
                     "being sampled only under receptor_stimulated. Within receptor_stimulated "
                     "the region_3 vs region_1 AUC gap is much smaller.",
}

# --- (b) batch = age perfect confound ---
ba = df.groupby("batch")["animal_age_days"].agg(["count", "nunique", "min", "max"])
res["batch_age_confound"] = {
    "n_batches": int(len(ba)),
    "all_batches_single_age": int((ba["nunique"] == 1).sum()),
    "mapping": {str(k): int(v) for k, v in ba["min"].items()},
    "age_unique_values": sorted(int(x) for x in df.animal_age_days.unique()),
    "note": "Each batch has exactly one age, so batch and age are perfectly "
            "confounded and cannot be separated in this data.",
}

# Demonstrate singularity: a model with both age and batch is rank-deficient.
ols_sing = smf.ols(
    f"{MEAS_AUC} ~ animal_age_days + C(batch) + C(condition, Treatment(reference='baseline')) + "
    "C(region, Treatment(reference='region_1'))", data=df)
mod_sing = ols_sing.fit()
rank_def = int(np.linalg.matrix_rank(mod_sing.model.exog))
n_params = len(mod_sing.params)
res["batch_age_confound"]["rank_deficient_model"] = {
    "model": "AUC ~ age + batch + condition + region",
    "exog_rank": rank_def, "n_params": n_params,
    "is_singular": rank_def < n_params,
}

# --- (c) VIF for the full model ---
ols_full = smf.ols(f"{MEAS_AUC} ~ C(region, Treatment(reference='region_1')) + "
            "C(condition, Treatment(reference='baseline')) + C(batch)", data=df)
X = np.asarray(mod_full.model.exog)
# Remove intercept for VIF computation (VIF of intercept is meaningless).
Xv = X[:, 1:]
vif = {}
for j in range(Xv.shape[1]):
    yj = Xv[:, j]
    Xj = np.delete(Xv, j, axis=1)
    if np.ptp(Xj) < 1e-12:
        continue
    yy = np.ones(Xv.shape[0])
    Xjy = np.column_stack([yy, Xj])
    coef, *_ = np.linalg.lstsq(Xjy, yj, rcond=None)
    r2 = 1 - np.sum((yj - Xjy @ coef)**2) / np.sum((yj - yj.mean())**2)
    vif[X.shape[1] + j - 1] = round(1.0 / (1.0 - r2), 4)
# Map column index back to term name
colnames = list(mod_full.model.exog_names)
vif_named = {}
for j in range(Xv.shape[1]):
    name = colnames[j + 1]
    vif_named[name] = vif[X.shape[1] + j - 1]
res["vif_full_model"] = {"max_vif": max(vif_named.values()), "vif": vif_named}

with open(OUT / "confounding_quantification.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote confounding_quantification.json")
print("  region_3 vs region_1 AUC (full model):", res["region_condition_confound"]["full_model_region3_vs_region1_AUC_coef"])
print("  within receptor_stimulated:", res["region_condition_confound"]["within_receptor_stimulated_AUC"])
print("  batch_age all single:", res["batch_age_confound"]["all_batches_single_age"], "of", res["batch_age_confound"]["n_batches"])
print("  singular model:", res["batch_age_confound"]["rank_deficient_model"])
print("  max VIF:", res["vif_full_model"]["max_vif"])
for k, v in res["vif_full_model"]["vif"].items():
    print("    VIF", k, "=", v)
