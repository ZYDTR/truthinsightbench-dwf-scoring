#!/usr/bin/env python3
"""Robustness checks for the five findings. Reads results.json (optional) and
recomputes robustness metrics from the raw data. Writes robustness.json.
"""
import json, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
# analysis/ lives at <project>/output/analysis, so the project root is two
# levels up; the bundled data directory is <project>/data.
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data")
DATA = os.environ.get("MEMBRANE_DATA_DIR", DEFAULT_DATA)
OUT  = os.environ.get("MEMBRANE_OUT_DIR", BASE)

def load(name):
    return pd.read_csv(os.path.join(DATA, name), sep="\t")

cond = load("conductivity_replicates.tsv")
lc   = load("local_current_distributions.tsv")
bc   = load("bias_current_curves.tsv")
cr   = load("condition_response.tsv")
dt   = load("device_time_response.tsv")

rv = cond.loc[cond["sample_class"] == "reference_membrane", "proton_conductivity_mS_cm2"].to_numpy(dtype=float)
mv = cond.loc[cond["sample_class"] == "modified_membrane", "proton_conductivity_mS_cm2"].to_numpy(dtype=float)

R = {}

# ---------- F1 robustness: alternative definitions + leave-one-out ----------
f1 = {
    "ratio_median": float(np.median(mv) / np.median(rv)),
    "ratio_mean": float(mv.mean() / rv.mean()),
    "ratio_median_of_means": float(mv.mean() / rv.mean()),
    "min_pairwise_ratio": float(mv.min() / rv.max()),
    "max_pairwise_ratio": float(mv.max() / rv.min()),
    "no_overlap": bool(mv.min() > rv.max()),
    "lopo_ratio_median": [],
    "lopo_ratio_mean": [],
    "lopo_min_pairwise_ratio": [],
}
for i in range(len(mv)):
    m = np.delete(mv, i)
    f1["lopo_ratio_median"].append(round(float(np.median(m) / np.median(rv)), 4))
    f1["lopo_ratio_mean"].append(round(float(m.mean() / rv.mean()), 4))
    f1["lopo_min_pairwise_ratio"].append(round(float(m.min() / rv.max()), 4))
# leave-one-out on reference
f1["lopo_ref_removed_ratio_mean"] = [
    round(float(mv.mean() / np.delete(rv, i).mean()), 4) for i in range(len(rv))
]
R["F1_conductivity_robustness"] = f1

# ---------- F2 robustness: threshold sensitivity + overlap ----------
f2 = {}
for c, g in lc.groupby("sample_class"):
    v = g["current_pA"].to_numpy(dtype=float)
    f2[c] = {
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "n": int(len(v)),
    }
f2["median_ratio"] = float(f2["modified_membrane"]["median"] / f2["reference_membrane"]["median"])
f2["mean_ratio"] = float(f2["modified_membrane"]["mean"] / f2["reference_membrane"]["mean"])
f2["reference_max"] = float(lc.loc[lc["sample_class"] == "reference_membrane", "current_pA"].max())
f2["modified_p10"] = float(lc[lc["sample_class"] == "modified_membrane"].current_pA.quantile(0.1))
f2["overlap_low_tail"] = "reference_max > modified_p10"  # overlap exists in low tail
f2["threshold_frac_gt"] = {}
for thr in (1.0, 5.0, 10.0, 50.0, 100.0, 200.0):
    f2["threshold_frac_gt"][f"{thr:g}"] = {
        c: float((g["current_pA"].to_numpy(dtype=float) > thr).mean())
        for c, g in lc.groupby("sample_class")
    }
# alternative: log1p scale comparison
f2["log1p_ratio_modified_vs_ref"] = float(
    np.mean(np.log1p(lc[lc["sample_class"] == "modified_membrane"].current_pA.to_numpy(dtype=float))) /
    np.mean(np.log1p(lc[lc["sample_class"] == "reference_membrane"].current_pA.to_numpy(dtype=float))))
R["F2_local_current_robustness"] = f2

# ---------- F3 robustness: alternative conductance definitions + linearity ----------
f3 = {}
for c, g in bc.groupby("sample_class"):
    x = g["bias_V"].to_numpy(dtype=float)
    y = g["current_A"].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum((y - (a*x+b))**2))
    f3[c] = {
        "n": int(len(g)),
        "slope_S": float(a),
        "R2": float(1 - sse/sst) if sst > 0 else float('nan'),
        "I_over_V_mean_S": float(np.nanmean(y[x != 0] / x[x != 0])),
        "I_over_V_median_S": float(np.nanmedian(y[x != 0] / x[x != 0])),
    }
f3["slope_ratio"] = float(f3["modified_membrane"]["slope_S"] / f3["reference_membrane"]["slope_S"])
f3["IoverV_ratio"] = float(f3["modified_membrane"]["I_over_V_mean_S"] / f3["reference_membrane"]["I_over_V_mean_S"])
f3["IoverV_median_ratio"] = float(f3["modified_membrane"]["I_over_V_median_S"] / f3["reference_membrane"]["I_over_V_median_S"])
# reference is symmetric (odd function); check I(-V) = -I(V)
def mirror_asymmetry(g):
    g = g.sort_values("bias_V")
    pos = g[g.bias_V > 0].copy()
    neg = g[g.bias_V < 0].copy()
    if pos.empty or neg.empty:
        return float("nan")
    asym = []
    for v in np.asarray(pos.bias_V):
        nv = -v
        idx = int(np.argmin(np.abs(np.asarray(neg.bias_V) - nv)))
        i = int(np.argmin(np.abs(np.asarray(pos.bias_V) - v)))
        asym.append(abs(float(pos.current_A.iloc[i]) + float(neg.current_A.iloc[idx])))
    return float(np.mean(asym)) if asym else float("nan")
f3["reference_mirror_asymmetry_A"] = mirror_asymmetry(bc[bc["sample_class"] == "reference_membrane"])
f3["modified_mirror_asymmetry_A"] = mirror_asymmetry(bc[bc["sample_class"] == "modified_membrane"])
f3["both_linear"] = all(f3[c]["R2"] > 0.99 for c in ("modified_membrane", "reference_membrane"))
R["F3_bias_current_robustness"] = f3

# ---------- F4 robustness: null condition response ----------
f4 = {}
for g, grp in cr.groupby("test_condition"):
    v = grp["dimensionless_delta"].to_numpy(dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    lo = float(v.mean() - 1.96*sd/np.sqrt(len(v)))
    hi = float(v.mean() + 1.96*sd/np.sqrt(len(v)))
    f4[g] = {"mean": float(v.mean()), "ci95": [lo, hi], "excludes_zero": bool(not (lo <= 0 <= hi)),
             "median": float(np.median(v))}
allv = cr["dimensionless_delta"].to_numpy(dtype=float)
f4["overall_mean"] = float(allv.mean())
f4["overall_ci95"] = [float(allv.mean()-1.96*allv.std(ddof=1)/np.sqrt(len(allv))),
                       float(allv.mean()+1.96*allv.std(ddof=1)/np.sqrt(len(allv)))]
gases = [g for g in cr["test_condition"].unique()]
f4["any_gas_significant"] = any(f4[g]["excludes_zero"] for g in gases)
# leave-one-out by gas: does the global mean stay near zero?
f4["lopo_gas_mean"] = {}
for g in cr["test_condition"].unique():
    rest = cr[cr["test_condition"] != g]["dimensionless_delta"].to_numpy(dtype=float)
    f4["lopo_gas_mean"][g] = round(float(rest.mean()), 4)
# per-device means (already near zero)
f4["per_device_mean"] = {dev: round(float(grp["dimensionless_delta"].mean()), 4)
                         for dev, grp in cr.groupby("device_id")}
R["F4_condition_response_robustness"] = f4

# ---------- F5 robustness: null time response ----------
f5 = {}
slopes = {}
for dev, grp in dt.groupby("device_id"):
    grp = grp.sort_values("time_day")
    x = grp["time_day"].to_numpy(dtype=float)
    y = grp["dimensionless_delta"].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum((y - (a*x+b))**2))
    r2 = float(1 - sse/sst) if sst > 0 else float('nan')
    try:
        a2, b2, c2 = np.polyfit(x, y, 2)
        sse2 = float(np.sum((y - (a2*x**2 + b2*x + c2))**2))
        r2q = float(1 - sse2/sst) if sst > 0 else float('nan')
    except Exception:
        r2q = float('nan')
    slopes[dev] = {"slope_per_day": float(a), "R2_linear": r2, "R2_quadratic": r2q}
f5["per_device"] = slopes
sl = np.array([slopes[d]["slope_per_day"] for d in slopes])
f5["mean_slope"] = float(sl.mean())
f5["std_slope"] = float(sl.std(ddof=1))
f5["lopo_mean_slope"] = [round(float(np.mean([sl[i] for i in range(len(sl)) if i != k])), 5)
                         for k in range(len(sl))]
f5["lopo_mean_slope_range"] = [round(float(np.min(np.delete(sl,k))),5) for k in range(len(sl))]
f5["frac_positive"] = float((sl > 0).mean())
f5["frac_negative"] = float((sl < 0).mean())
# quadratic vs linear: does quadratic add explanatory power?
q2 = [slopes[d].get("R2_quadratic", float("nan")) for d in slopes]
q2 = [v for v in q2 if not (isinstance(v, float) and np.isnan(v))]
f5["quadratic_R2_mean"] = float(np.mean(q2)) if q2 else float("nan")
R["F5_time_response_robustness"] = f5

with open(os.path.join(OUT, "robustness.json"), "w") as fh:
    json.dump(R, fh, indent=2)
print("Wrote robustness.json to", OUT)
print("F1 lopo_ratio_median:", f1["lopo_ratio_median"])
print("F2 threshold_frac_gt_100:", f2["threshold_frac_gt"]["100"])
print("F3 slope_ratio:", f3["slope_ratio"], "IoverV_ratio:", f3["IoverV_ratio"])
print("F4 any_gas_significant:", f4["any_gas_significant"])
print("F5 mean_slope:", f5["mean_slope"], "lopo_mean_slope:", f5["lopo_mean_slope"])
