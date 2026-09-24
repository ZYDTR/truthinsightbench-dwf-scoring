#!/usr/bin/env python3
"""Panel A analysis: core V(B) characterization for fig2_critical_buffer."""
import os, json
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project/data/source_data/figures/fig2_critical_buffer"
OUT = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project/output/results"
os.makedirs(OUT, exist_ok=True)

def rr2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

def max_rel_err(data, pred, rel_to):
    """max |data-pred| / |data| when rel_to='data', /|pred| when rel_to='pred'."""
    diff = np.abs(data - pred)
    if rel_to == "data":
        mask = np.abs(data) > 1e-12
        val = diff[mask] / np.abs(data[mask])
    else:
        mask = np.abs(pred) > 1e-12
        val = diff[mask] / np.abs(pred[mask])
    return float(np.max(val)) if len(val) else float("nan")

def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat)**2)))

def fit_power(B, V):
    gamma, la = np.polyfit(np.log(B), np.log(V), 1)
    a = float(np.exp(la)); yhat = a * B**gamma
    return dict(a=a, gamma=float(gamma), r2=rr2(V, yhat))

def fit_loglin(B, V):
    c2, c1 = np.polyfit(np.log(B), V, 1)
    yhat = c1 + c2*np.log(B)
    return dict(c1=float(c1), c2=float(c2), r2=rr2(V, yhat))

def fit_exp(B, V):
    b, lb = np.polyfit(B, np.log(V), 1)
    a = float(np.exp(lb)); yhat = a * np.exp(b*B)
    return dict(a=a, b=float(b), r2=rr2(V, yhat))

def fit_lin(B, V):
    b, a = np.polyfit(B, V, 1)
    yhat = a + b*B
    return dict(a=float(a), b=float(b), r2=rr2(V, yhat))

def fit_recip(B, V):
    x = 1.0/B
    a = float(np.sum(x*V) / np.sum(x*x))
    yhat = a * x
    return dict(a=float(a), r2=rr2(V, yhat))

rows = []
def add(curve, **k):
    if "curve" in k:
        del k["curve"]
    rows.append({**k, "panel": "panel_a", "curve": str(curve)})

B  = below = pd.read_csv(BASE + "/data/below_bc.csv")
V  = below["Plot 2"].to_numpy(dtype=float)
T1 = below["theory1"].to_numpy(dtype=float)
T2 = below["theory2"].to_numpy(dtype=float)
B  = below["Plot 1"].to_numpy(dtype=float)
above = pd.read_csv(BASE + "/data/above_bc.csv")
B2 = above["Plot 1"].to_numpy(dtype=float)
V2 = above["Plot 2"].to_numpy(dtype=float)
mean = pd.read_csv(BASE + "/data/meanfield_data.csv")
theo = pd.read_csv(BASE + "/data/theory_data.csv")

# ---- 1. below fits ----
add("below", model="power", **fit_power(B, V))
add("below", model="loglin", **fit_loglin(B, V))
add("below", model="exp", **fit_exp(B, V))

# ---- 2. above fits ----
add("above", model="linear", **fit_lin(B2, V2))
add("above", model="exp", **fit_exp(B2, V2))
add("above", model="recip", **fit_recip(B2, V2))

# ---- 3. comparisons ----
def cmp(name, data, pred, overlap=None, n=None):
    r = pearson(data, pred)
    add("cmp", theory=name,
        pearson_r=r, rmse=rmse(data, pred),
        max_rel_err_vs_data=max_rel_err(data, pred, "data"),
        max_rel_err_vs_pred=max_rel_err(data, pred, "pred"),
        n=int(n) if n is not None else int(len(data)))

cmp("theory1", V, T1)
cmp("theory2", V, T2)

def cmp_grid(name, grid_df, lo=1.0, hi=9.0):
    gB = grid_df["B"].to_numpy(dtype=float); gV = grid_df["V"].to_numpy(dtype=float)
    m = (B >= lo) & (B <= hi)
    dB, dV = B[m], V[m]
    pred = np.interp(dB, gB, gV)
    add("cmp", theory=name, overlap=f"{lo}-{hi}",
        pearson_r=pearson(dV, pred), rmse=rmse(dV, pred),
        max_rel_err_vs_data=max_rel_err(dV, pred, "data"),
        max_rel_err_vs_pred=max_rel_err(dV, pred, "pred"), n=int(m.sum()))
cmp_grid("theory_data", theo)
cmp_grid("meanfield_data", mean)

# ---- 4. Bc estimate ----
above_mean = float(V2.mean()); above_med = float(np.median(V2)); above_std = float(V2.std())
def below_median_bins(B, V, lo, hi, nbins):
    edges = np.linspace(lo, hi, nbins+1); res = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (B >= e0) & (B < e1)
        if m.sum() > 0: res.append((0.5*(e0+e1), float(np.median(V[m])), int(m.sum())))
    return np.array(res)
bins = below_median_bins(B, V, 1.0, 3.0, 12)
dep = np.searchsorted(bins[:,1], 1.25*above_mean)
bc_b = float(bins[dep,0]) if dep < len(bins) else float(bins[-1,0])
i0 = int(np.argmin(B)); below_start = (float(B[i0]), float(V[i0]))
ov = (B >= 1.0) & (B <= 2.0)
overlap = (float(B[ov].mean()), float(np.median(V[ov])))
add("Bc", above_flat_mean=above_mean, above_flat_med=above_med, above_flat_std=above_std,
    above_flat_cv=above_std/above_mean,
    Bc_departure_B=bc_b, Bc_departure_V=above_mean,
    below_start_B=below_start[0], below_start_V=below_start[1],
    overlap_B=overlap[0], overlap_V=overlap[1])

# ---- 5. robustness ----
for lo, hi in [(1.0,10.0),(1.0,50.0),(1.0,200.0)]:
    m = (B >= lo) & (B <= hi); f = fit_power(B[m], V[m])
    add("robust_window", lo=lo, hi=hi, a=f["a"], gamma=f["gamma"], r2=f["r2"], n=int(m.sum()))
q90 = float(np.percentile(B, 90)); m = B <= q90; f = fit_power(B[m], V[m])
add("robust_leaveout_byB", cutoff_B=q90, a=f["a"], gamma=f["gamma"], r2=f["r2"], n=int(m.sum()))
q90v = float(np.percentile(V, 90)); m2 = V <= q90v; f2 = fit_power(B[m2], V[m2])
add("robust_leaveout_byV", cutoff_V=q90v, a=f2["a"], gamma=f2["gamma"], r2=f2["r2"], n=int(m2.sum()))
above_s = above.sort_values("Plot 1").reset_index(drop=True)
for name, df in [("above_with_first5", above_s), ("above_without_first5", above_s.iloc[5:])]:
    Bx, Vx = df["Plot 1"].to_numpy(dtype=float), df["Plot 2"].to_numpy(dtype=float)
    fl, fe, fr = fit_lin(Bx, Vx), fit_exp(Bx, Vx), fit_recip(Bx, Vx)
    add("robust_above", subset=name, n=int(len(df)),
        lin_a=fl["a"], lin_b=fl["b"], lin_r2=fl["r2"],
        exp_a=fe["a"], exp_b=fe["b"], exp_r2=fe["r2"],
        recip_a=fr["a"], recip_r2=fr["r2"])

# ---- 6. explanations ----
with5 = next(r for r in rows if r.get("curve")=="robust_above" and r["subset"]=="above_with_first5")
without5 = next(r for r in rows if r.get("curve")=="robust_above" and r["subset"]=="above_without_first5")
def pick(r): return {k: float(v) for k, v in r.items() if k.startswith(("lin_","exp_","recip_"))}
pw = {f"{r['lo']:.0f}-{r['hi']:.0f}": r for r in rows if r.get("curve")=="robust_window"}

meta = {
    "panel": "panel_a",
    "description": "Core V(B) characterization for fig2_critical_buffer",
    "data": {
        "below_n": int(len(B)), "below_B_range": [float(B.min()), float(B.max())],
        "below_V_range": [float(V.min()), float(V.max())],
        "above_n": int(len(B2)), "above_B_range": [float(B2.min()), float(B2.max())],
        "above_V_range": [float(V2.min()), float(V2.max())],
        "above_flat_mean": above_mean, "above_flat_std": above_std, "above_flat_cv": above_std/above_mean,
        "meanfield_n": int(len(mean)), "theory_n": int(len(theo)),
    },
    "Bc_estimate": {
        "above_flat_mean_V": above_mean, "above_flat_std": above_std,
        "Bc_departure_B": bc_b, "Bc_departure_V": above_mean,
        "below_start_B": below_start[0], "below_start_V": below_start[1],
        "overlap_B_in_1_to_2": overlap[0], "overlap_V": overlap[1],
        "note": "Bc is the low-B boundary where the rising 'below' curve departs the flat 'above' "
                "plateau (V~1.065). Departure = below-median crossing 1.25*above_mean. The transition "
                "is at the low-B boundary (B~1.0-1.2), not a sharp mid-range kink; the above regime is a flat plateau.",
    },
    "robustness": {
        "power_windows": {w: {"gamma": float(v["gamma"]), "a": float(v["a"]), "r2": float(v["r2"]), "n": int(v["n"])} for w, v in pw.items()},
        "leaveout_top10pct_byB": next(r for r in rows if r.get("curve")=="robust_leaveout_byB"),
        "leaveout_top10pct_byV": next(r for r in rows if r.get("curve")=="robust_leaveout_byV"),
        "above_with_first5": pick(with5),
        "above_without_first5": pick(without5),
    },
    "explanations": {
        "smooth_no_transition": "rejected",
        "segmented_piecewise": "supported",
        "nonlinear_powerlaw": "supported",
        "no_identifiable_boundary": "rejected",
        "finite_size_dependent": "inconclusive",
    },
    "verdict_rationale": {
        "smooth_no_transition": "Above curve is flat (all trend models R2~0, CV~5%); below curve rises to V~366. No single smooth model fits both.",
        "segmented_piecewise": "Below regime: power law gamma~1.0, R2~0.90 (near-linear). Above regime: flat plateau V~1.065. Two clearly distinct regimes.",
        "nonlinear_powerlaw": "Below curve well fit by V=a*B^gamma with gamma~1.00, R2~0.90 full range; gamma stable across windows [1,10]/[1,50]/[1,200].",
        "no_identifiable_boundary": "Regimes are clearly distinct (below V~366 vs above V~1.06); a low-B boundary is identifiable.",
        "finite_size_dependent": "No system-size parameter present in these curves; not testable from this data.",
    },
}
df = pd.DataFrame(rows)
df.to_csv(OUT + "/panel_a_results.csv", index=False)
with open(OUT + "/panel_a_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print("Wrote", OUT + "/panel_a_results.csv", len(df), "rows; and", OUT + "/panel_a_meta.json")
print(df.to_string(index=False))
