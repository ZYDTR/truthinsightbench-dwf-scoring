#!/usr/bin/env python3
"""Core quantitative analysis for Material_03_membrane_selective_permeation.
Reads data/ and writes results.json (structured) + results.tsv (flat).
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

bc   = load("bias_current_curves.tsv")
cr   = load("condition_response.tsv")
cond = load("conductivity_replicates.tsv")
dt   = load("device_time_response.tsv")
lc   = load("local_current_distributions.tsv")
sm   = load("spatial_current_map.tsv")

def ci(mean, sd, n, z=1.96):
    if n < 2 or sd <= 0:
        return [float(mean), float(mean)]
    se = sd / np.sqrt(n)
    return [float(mean - z*se), float(mean + z*se)]

R = {}

# ---------- Finding 1: proton conductivity ----------
f1 = {}
for c, g in cond.groupby("sample_class"):
    v = g["proton_conductivity_mS_cm2"].to_numpy(dtype=float)
    f1[c] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
             "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
             "min": float(v.min()), "max": float(v.max())}
rv = cond.loc[cond["sample_class"] == "reference_membrane", "proton_conductivity_mS_cm2"].to_numpy(dtype=float)
mv = cond.loc[cond["sample_class"] == "modified_membrane", "proton_conductivity_mS_cm2"].to_numpy(dtype=float)
f1["ratio_mean"] = float(mv.mean() / rv.mean())
f1["ratio_median"] = float(np.median(mv) / np.median(rv))
f1["min_pairwise_ratio"] = float(mv.min() / rv.max())
f1["max_pairwise_ratio"] = float(mv.max() / rv.min())
f1["all_modified_gt_all_reference"] = bool(mv.min() > rv.max())
R["finding_1_conductivity"] = f1

# ---------- Finding 2: local current distributions ----------
f2 = {}
for c, g in lc.groupby("sample_class"):
    v = g["current_pA"].to_numpy(dtype=float)
    f2[c] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
             "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
             "max": float(v.max()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
             "frac_gt_100": float((v > 100).mean()), "frac_lt_1": float((v < 1).mean())}
f2["median_ratio_modified_vs_ref"] = float(f2["modified_membrane"]["median"] / f2["reference_membrane"]["median"])
f2["reference_max_pA"] = float(lc.loc[lc["sample_class"] == "reference_membrane", "current_pA"].max())
f2["modified_min_pA"] = float(lc.loc[lc["sample_class"] == "modified_membrane", "current_pA"].min())
for thr in (50.0, 10.0):
    f2[f"frac_gt_{int(thr)}_pA"] = {
        c: float((g["current_pA"].to_numpy(dtype=float) > thr).mean())
        for c, g in lc.groupby("sample_class")
    }
R["finding_2_local_current"] = f2

# ---------- Finding 3: bias-current linearity & conductance ----------
f3 = {}
for c, g in bc.groupby("sample_class"):
    x = g["bias_V"].to_numpy(dtype=float)
    y = g["current_A"].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum((y - yhat)**2))
    r2 = 1.0 - sse / sst if sst > 0 else float('nan')
    f3[c] = {"n": int(len(g)), "G_slope_S": float(a), "intercept_A": float(b),
             "R2_with_intercept": float(r2),
             "I_over_V_mean_S": float(np.nanmean(y[x != 0] / x[x != 0]))}
f3["conductance_ratio_modified_vs_ref"] = float(f3["modified_membrane"]["G_slope_S"] / f3["reference_membrane"]["G_slope_S"])
R["finding_3_bias_current"] = f3

# ---------- Finding 4: condition response (null test) ----------
f4 = {"per_gas": {}, "global": {}}
for g, grp in cr.groupby("test_condition"):
    v = grp["dimensionless_delta"].to_numpy(dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    f4["per_gas"][g] = {"n": int(len(v)), "mean": float(v.mean()), "std": sd,
                        "ci95_mean": ci(v.mean(), sd, len(v)), "ci_excludes_0": bool(abs(v.mean()) > 1.96*sd/np.sqrt(len(v)))}
allv = cr["dimensionless_delta"].to_numpy(dtype=float)
f4["global"]["overall_mean"] = float(allv.mean())
f4["global"]["overall_std"] = float(allv.std(ddof=1))
f4["global"]["n_obs"] = int(len(allv))
per_dev = {}
for dev, grp in cr.groupby("device_id"):
    v = grp["dimensionless_delta"].to_numpy(dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    per_dev[dev] = {"n": int(len(v)), "mean": float(v.mean()), "ci95_mean": ci(v.mean(), sd, len(v))}
f4["global"]["per_device_means"] = per_dev
f4["global"]["any_gas_mean_significant"] = any(f4["per_gas"][g]["ci_excludes_0"] for g in f4["per_gas"])
R["finding_4_condition_response"] = f4

# ---------- Finding 5: long-term device time response (null test) ----------
f5 = {"per_device": {}, "overall": {}}
for dev, grp in dt.groupby("device_id"):
    grp = grp.sort_values("time_day")
    x = grp["time_day"].to_numpy(dtype=float)
    y = grp["dimensionless_delta"].to_numpy(dtype=float)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    sst = float(np.sum((y - y.mean())**2))
    sse = float(np.sum((y - yhat)**2))
    r2 = 1.0 - sse / sst if sst > 0 else float('nan')
    try:
        a2, b2, c2 = np.polyfit(x, y, 2)
        yhat2 = a2 * x**2 + b2 * x + c2
        sse2 = float(np.sum((y - yhat2)**2))
        r2q = 1.0 - sse2 / sst if sst > 0 else float('nan')
    except Exception:
        r2q = float('nan')
    f5["per_device"][dev] = {"n": int(len(y)), "days": grp["time_day"].tolist(),
                            "slope_per_day": float(a), "R2_linear": float(r2),
                            "R2_quadratic": float(r2q)}
slopes = np.array([f5["per_device"][dev]["slope_per_day"] for dev in f5["per_device"]])
f5["overall"]["mean_slope"] = float(slopes.mean())
f5["overall"]["std_slope"] = float(slopes.std(ddof=1))
f5["overall"]["frac_positive"] = float((slopes > 0).mean())
f5["overall"]["frac_negative"] = float((slopes < 0).mean())
n = len(slopes)
lopo = [float(np.mean([slopes[i] for i in range(n) if i != k])) for k in range(n)]
f5["overall"]["leave_one_out_mean_slope"] = [round(x, 5) for x in lopo]
R["finding_5_time_response"] = f5

# ---------- supporting spatial map ----------
f6 = {}
ys = sorted(sm["y_um"].unique()); xs = sorted(sm["x_um"].unique())
ix = {v: i for i, v in enumerate(xs)}
iy = {v: i for i, v in enumerate(ys)}
arr = np.full((len(ys), len(xs)), np.nan)
for _, r in sm.iterrows():
    arr[iy[r["y_um"]], ix[r["x_um"]]] = r["current_pA"]
f6["pixels"] = int(len(sm))
f6["mean_pA"] = float(arr.mean())
f6["std_pA"] = float(arr.std())
f6["neg_fraction"] = float((arr < 0).mean())
f6["max_pA"] = float(arr.max())
f6["min_pA"] = float(arr.min())
nb = 4
H, W = arr.shape
bmeans = []
for ny in range(H // nb):
    for nx in range(W // nb):
        b = arr[ny*nb:(ny+1)*nb, nx*nb:(nx+1)*nb]
        bmeans.append({"block": f"{ny},{nx}", "mean": float(b.mean()), "min": float(b.min()),
                       "max": float(b.max()), "neg": int((b < 0).sum())})
f6["block_means_4x4"] = bmeans
yy, xx = np.meshgrid(np.arange(H), np.arange(W))
cy, cx = (H - 1) / 2, (W - 1) / 2
dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
f6["current_vs_radial_distance_corr"] = float(np.corrcoef(arr.ravel(), dist.ravel())[0, 1])
R["supporting_spatial_map"] = f6

# ---------- write outputs ----------
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=2)

def flatten(d, prefix):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten(v, f"{prefix}.{k}")
        else:
            try:
                yield [prefix, k, float(v)]
            except (TypeError, ValueError):
                yield [prefix, k, str(v)]

rows = []
for key, d in R.items():
    rows.extend(flatten(d, key))

with open(os.path.join(OUT, "results.tsv"), "w") as fh:
    fh.write("finding\tkey\tvalue\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")

print("Wrote results.json and results.tsv to", OUT)
print("F1 ratio_mean:", R["finding_1_conductivity"]["ratio_mean"], "ratio_median:", R["finding_1_conductivity"]["ratio_median"])
print("F2 median_ratio:", R["finding_2_local_current"]["median_ratio_modified_vs_ref"])
print("F3 conductance_ratio:", R["finding_3_bias_current"]["conductance_ratio_modified_vs_ref"])
print("F4 gas means:", {g: R["finding_4_condition_response"]["per_gas"][g]["mean"] for g in R["finding_4_condition_response"]["per_gas"]})
print("F5 mean_slope:", R["finding_5_time_response"]["overall"]["mean_slope"])
