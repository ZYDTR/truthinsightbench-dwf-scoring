#!/usr/bin/env python3
"""Region-, condition-, and batch-dependent calcium response structure.

Analysis unit: one cell (cell_observation) nested within slice, batch, biological source.
Measures (per-cell means): amplitude, time-to-peak, peak width, prominence, frequency, AUC.
Factors: condition, anatomical_region_id, animal_age_days, recording_batch_id.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

DATA = Path(__file__).parent.resolve() / "data" / "cellular_calcium_response_summaries.tsv"
OUT = Path(__file__).parent.resolve() / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, sep="\t")
MEAS = ["mean_amplitude_dff0","mean_time_to_peak_s","mean_peak_width_s",
        "mean_peak_prominence_dff0","mean_peak_frequency_hz","mean_auc_au"]
df["region"] = df["anatomical_region_id"]
df["batch"] = df["recording_batch_id"]

def fit(m, f, sub=None):
    d = sub if sub is not None else df
    return smf.ols(f"{m} ~ {f}", data=d).fit()

def coef_p(mod, *needles):
    """Return (value, pvalue) for a param whose name contains any needle."""
    for name, v in mod.params.items():
        if any(n in name for n in needles):
            return round(float(v), 6), round(float(mod.pvalues[name]), 6)
    return None, None

def pick(mod, needles):
    out = {}
    for name, v in mod.params.items():
        if any(n in name for n in needles):
            out[name] = {"coef": round(float(v),6), "p": round(float(mod.pvalues[name]),6)}
    return out

print("=== 1. CONDITION (ctrl region + batch) ===")
cond = {}
for m in MEAS:
    mod = fit(m, "C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1')) + C(batch)")
    cond[m] = {"formula": "condition + region + batch", "r2": round(float(mod.rsquared),4),
               "n": int(mod.nobs), "terms": pick(mod, ["condition"])}

print("=== 2. REGION (ctrl condition + batch) ===")
reg = {}
for m in MEAS:
    mod = fit(m, "C(region, Treatment(reference='region_1')) + C(condition, Treatment(reference='baseline')) + C(batch)")
    reg[m] = {"formula": "region + condition + batch", "r2": round(float(mod.rsquared),4),
              "n": int(mod.nobs), "terms": pick(mod, ["region"])}

clean = df[df.region.isin(["region_1","region_2"])].copy()

print("=== 3. CLEAN region_1/2 CONDITION (ctrl region + batch) ===")
ccond = {}
for m in MEAS:
    mod = fit(m, "C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1')) + C(batch)", sub=clean)
    ccond[m] = {"n": int(mod.nobs), "r2": round(float(mod.rsquared),4), "terms": pick(mod, ["condition"])}

print("=== 4. BATCH (ctrl condition + region) ===")
batch = {}
for m in MEAS:
    mod = fit(m, "C(batch, Treatment(reference='recording_batch_02')) + C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1'))")
    batch[m] = {"r2": round(float(mod.rsquared),4), "n": int(mod.nobs), "terms": pick(mod, ["batch"])}

print("=== 5. AGE linear (ctrl condition + region) ===")
age = {}
for m in MEAS:
    mod = fit(m, "animal_age_days + C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1'))")
    age[m] = {"r2": round(float(mod.rsquared),4), "slope": round(float(mod.params["animal_age_days"]),6),
              "p": round(float(mod.pvalues["animal_age_days"]),6)}

ba = df.groupby("batch")["animal_age_days"].agg(["count","nunique","min","max"])
ba["one_age"] = (ba["nunique"]==1).astype(int)
cov = df.groupby(["region","condition"]).size().unstack(fill_value=0)
raw = df.groupby("condition")[MEAS].mean().round(3)
bmeans = df.groupby("batch")[MEAS].mean().round(3)

print("=== 6. LBO (leave one batch out) condition effect ===")
lbo = {}
for m in ["mean_amplitude_dff0","mean_auc_au","mean_peak_width_s"]:
    vals = []
    for b in sorted(df.batch.unique()):
        d = df[df.batch != b]
        mod = fit(m, "C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1')) + C(batch)", sub=d)
        v, p = coef_p(mod, "T.receptor_stimulated", "receptor_stimulated")
        vals.append(p)
    vals = [x for x in vals if x is not None]
    lbo[m] = {"n_batches_tested": len(vals), "all_significant": all(x < 0.05 for x in vals),
              "max_p": max(vals), "min_p": min(vals), "vals": vals}

med = {}
for m in MEAS:
    med[m] = {c: round(float(v),4) for c, v in clean.groupby("condition")[m].median().items()}

print("=== 7. Outlier treatment (trim 2%) condition effect region_1/2 ===")
ot = {}
for m in ["mean_auc_au","mean_peak_width_s"]:
    d = clean.copy()
    lo, hi = np.percentile(d[m], [2,98])
    d[m] = d[m].clip(lo, hi)
    mod = fit(m, "C(condition, Treatment(reference='baseline')) + C(region, Treatment(reference='region_1')) + C(batch)", sub=d)
    v, p = coef_p(mod, "receptor_stimulated")
    ot[m] = {"coef_receptor": v, "p": p}

print("=== 8. Region effect in clean subset (region_1/2) ===")
creg = {}
for m in MEAS:
    mod = fit(m, "C(region, Treatment(reference='region_1')) + C(condition, Treatment(reference='baseline')) + C(batch)", sub=clean)
    v, p = coef_p(mod, "region_2")
    creg[m] = {"coef_region2": v, "p_region2": p}

print("=== 9. Interaction region x condition (clean, ctrl batch) ===")
intr = {}
for m in ["mean_peak_width_s","mean_auc_au","mean_amplitude_dff0"]:
    mod = fit(m, "C(region, Treatment(reference='region_1')) * C(condition, Treatment(reference='baseline')) + C(batch)", sub=clean)
    inter = {}
    for name, v in mod.pvalues.items():
        if "T.region_2" in name and ":" in name:
            inter[name] = round(float(v),6)
    intr[m] = {"formula": "region*condition + batch (clean)", "interaction_p": inter}

print("=== 10. Slice effect (ctrl condition + batch) ===")
sle = {}
for m in ["mean_auc_au","mean_peak_width_s"]:
    mod = fit(m, "C(slice_id) + C(condition, Treatment(reference='baseline')) + C(batch)")
    n_slices = int(df.slice_id.nunique())
    sle[m] = {"r2": round(float(mod.rsquared),4), "slice_F_p": round(float(mod.f_pvalue),6),
              "n_slices": n_slices}

def save(name, o):
    with open(OUT/name, "w") as f:
        json.dump(o, f, indent=2, default=float)

save("cond_region_batch.json", cond)
save("region_condition_batch.json", reg)
save("clean_condition.json", ccond)
save("batch_ctrl.json", batch)
save("age_trend.json", age)
save("batch_age.json", {"mapping": ba.to_dict(orient="index"), "all_one_to_one": int(ba["one_age"].sum()), "n_batches": int(len(ba))})
save("coverage.json", cov.to_dict(orient="index"))
save("raw_condition_means.json", raw.to_dict(orient="index"))
save("batch_means.json", bmeans.to_dict(orient="index"))
save("lbo_condition.json", lbo)
save("median_clean.json", med)
save("outlier_clean.json", ot)
save("region_clean.json", creg)
save("interaction_clean.json", intr)
save("slice_effect.json", sle)

print("\nbatch<->age (one_age all? %d/%d):" % (int(ba["one_age"].sum()), len(ba)))
print(ba["one_age"].to_string())
print("\ncoverage region x condition:\n", cov.to_string())
print("\nlbo:", json.dumps(lbo, indent=2))
print("\ncond region_1/2 (receptor vs baseline):\n",
      json.dumps({m: ccond[m]["terms"] for m in MEAS}, indent=2))
print("\nregion (region_2 vs region_1):\n",
      json.dumps({m: reg[m]["terms"] for m in MEAS}, indent=2))
print("\nage:\n", json.dumps(age, indent=2))
print("\nbatch terms (AUC):\n", json.dumps(batch["mean_auc_au"]["terms"], indent=2))
print("\nDONE")
