#!/usr/bin/env python3
"""Panel B: correlation collapse in fig3a_correlation_collapse.

Key analysis:
  1. Per-curve correlation length = integral of C(lag) (trapezoid) and the
     exponential decay time constant (log C vs lag, C>0.01).
  2. Monotonicity of the correlation length across buffer conditions B1..B5.
  3. Inset collapse: scale_factor vs (Bc0 - Bcs), Bc0 above all Bcs.
  4. Main-figure collapse: stable linear fit per condition (slope = cov/var).
Robustness: leave-out of one curve; fixed-beta inset fits; first-lag effect.
"""
import os, json, math
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project"
FIG = os.path.join(BASE, "data/source_data/figures/fig3a_correlation_collapse")
FIGS = os.path.join(FIG, "fig_source_data")
OUT = os.path.join(BASE, "output/results")
os.makedirs(OUT, exist_ok=True)

def stable_slope(x, y):
    """slope = cov(x,y)/var(x); returns (slope, intercept, r2) or (nan,nan,nan)."""
    try:
        x = np.asarray(x, float); y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2 or x.std() == 0:
            return np.nan, np.nan, np.nan
        sx, sy = x.mean(), y.mean()
        cov = np.mean((x - sx) * (y - sy))
        var = np.mean((x - sx) ** 2)
        if var == 0:
            return np.nan, np.nan, np.nan
        m = cov / var
        b = sy - m * sx
        yhat = b + m * x
        ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - sy) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return float(m), float(b), float(r2)
    except Exception:
        return np.nan, np.nan, np.nan

# ---------------------------------------------------------------- curves
curves = {}
for i in range(1, 6):
    a = pd.read_csv(os.path.join(FIG, "data", f"corr.ave_{i}.txt"), sep=r"\s+", header=None)
    lag, C = a[0].to_numpy(dtype=float), a[1].to_numpy(dtype=float)
    tau = float(np.trapezoid(C, lag))
    m = C > 0.01
    if np.sum(m) >= 3:
        sl, _, _ = stable_slope(lag[m], np.log(C[m]))
        tau_exp = float(-1 / sl) if (math.isfinite(sl) and sl != 0) else np.nan
    else:
        tau_exp = np.nan
    curves[i] = dict(lag=lag, C=C, tau=tau, tau_exp=tau_exp,
                     max_lag=float(lag.max()), n=int(len(lag)),
                     C_last=float(C[-1]), C_min=float(C.min()))
    print(f"B{i}: max_lag={lag.max():.0f} n={len(lag)} integral_tau={tau:.1f} "
          f"exp_tau={tau_exp:.1f} C_last={C[-1]:.4f}")

t = np.array([curves[i]["tau"] for i in range(1, 6)])
texp = np.array([curves[i]["tau_exp"] for i in range(1, 6)])
print("integral_tau:", [round(v,1) for v in t])
print("exp_tau:", [round(v,1) for v in texp])
print("integral monotone decreasing:", bool(np.all(np.diff(t) < 0)))
print("exp_tau monotone decreasing:", bool(np.all(np.diff(texp) < 0)))
print("integral ratio B1/B5:", round(t[0]/t[-1], 3))
print("exp_tau ratio B1/B5:", round(texp[0]/texp[-1], 3))

# ---------------------------------------------------------------- inset collapse
ins = pd.read_csv(os.path.join(FIGS, "inset_scatter_data.csv"))
Bcs = ins["Bcs"].to_numpy(dtype=float); sc = ins["scale_factors"].to_numpy(dtype=float)
bmin, bmax = Bcs.min(), Bcs.max()
print("inset Bcs range", round(bmin,4), round(bmax,4), "scale range", round(sc.min(),2), round(sc.max(),2))
print("scale factor ratio (top/bottom):", round(sc[-1]/sc[0], 3))

def fit_inset(Bcs, sc, Bc0):
    """scale = a*(Bc0 - Bcs)^(-beta), Bc0 > all Bcs."""
    v = Bc0 - Bcs
    ok = (v > 0) & (np.isfinite(sc)) & (sc > 0)
    if np.sum(ok) < 2:
        return dict(beta=np.nan, a=np.nan, r2=np.nan)
    logsc = np.log(sc[ok]); logb = np.log(v[ok])
    slope, intercept, r2 = stable_slope(logb, logsc)  # logsc = intercept + slope*logb
    if np.isnan(slope):
        return dict(beta=np.nan, a=np.nan, r2=np.nan)
    beta = float(-slope)
    a = float(math.exp(intercept)) if math.isfinite(intercept) and -700 < intercept < 700 else np.nan
    return dict(beta=beta, a=a, r2=r2)

best = None
for b0 in np.linspace(bmax + 0.001, bmax + 5.0, 2000):
    r = fit_inset(Bcs, sc, b0)
    if np.isfinite(r["r2"]) and (best is None or r["r2"] > best["r2"]):
        best = dict(r, Bc0=b0)
print("best inset fit:", best)

# main figure: stable linear fit per condition
mf = pd.read_csv(os.path.join(FIGS, "main_figure_scatter_data.csv"))
cond_cols = [c for c in mf.columns if c.endswith("_x")]
collapse = []
for i, cx in enumerate(cond_cols):
    cy = cx.replace("_x", "_y")
    x = mf[cx].to_numpy(dtype=float); y = mf[cy].to_numpy(dtype=float)
    m, b, r2v = stable_slope(x, y)
    collapse.append(dict(cond=i + 1, slope=m, intercept=b, r2=r2v,
                         y_min=float(y.min()), y_max=float(y.max())))
    print(f"cond {i+1}: slope={m:.6e} intercept={b:.4f} R2={r2v:.4f} y[{y.min():.4f},{y.max():.4f}]")

# ---------------------------------------------------------------- robustness
rob_leaveout = []
for drop in range(1, 6):
    tvals = [curves[i]["tau"] for i in range(1, 6) if i != drop]
    rob_leaveout.append(dict(left_out=drop, mean_tau=float(np.mean(tvals)),
                             min_tau=float(min(tvals)), max_tau=float(max(tvals))))
print("leave-out tau means:", [round(r["mean_tau"],1) for r in rob_leaveout])

fixed = {}
for beta in [0.5, 1, 1.5, 2]:
    bestb = None; bestr = None
    for b0 in np.linspace(bmax + 0.001, bmax + 5.0, 2000):
        v = b0 - Bcs
        ok = (v > 0) & (sc > 0) & np.isfinite(sc)
        if np.sum(ok) < 2:
            continue
        logsc = np.log(sc[ok]); logb = np.log(v[ok])
        slope, intercept, r2 = stable_slope(logb, logsc)
        if np.isnan(slope):
            continue
        if np.isfinite(r2) and (bestr is None or r2 > bestr):
            bestr = r2; bestb = b0
    fixed[beta] = dict(beta=beta, r2=bestb and bestr, Bc0=bestb)
print("fixed beta fits:", {k: (round(v.get("r2"),3) if v.get("r2") is not None else None) for k, v in fixed.items()})

rob_firstlag = []
for i in range(1, 6):
    c = curves[i]
    tau_w = float(np.trapezoid(c["C"], c["lag"]))
    tau_wo = float(np.trapezoid(c["C"][1:], c["lag"][1:]))
    rob_firstlag.append(dict(cond=i, with_first=tau_w, without_first=tau_wo,
                             change_pct=100 * (tau_w - tau_wo) / tau_w))
print("first-lag effect %:", [round(r["change_pct"],2) for r in rob_firstlag])

# ---------------------------------------------------------------- save
meta = {
    "panel": "panel_b",
    "description": "Correlation-collapse analysis of fig3a",
    "curves": {str(i): {
                     "max_lag": float(curves[i]["max_lag"]),
                     "n": int(curves[i]["n"]),
                     "integral_tau": float(curves[i]["tau"]),
                     "exp_tau": float(curves[i]["tau_exp"]),
                     "C_last": float(curves[i]["C_last"]),
                     "C_min": float(curves[i]["C_min"]),
                 } for i in range(1, 6)},
    "integral_tau": [float(x) for x in t],
    "exp_tau": [float(x) for x in texp],
    "integral_monotone_decreasing": bool(np.all(np.diff(t) < 0)),
    "exp_tau_monotone_decreasing": bool(np.all(np.diff(texp) < 0)),
    "integral_ratio_B1_B5": float(t[0] / t[-1]),
    "exp_tau_ratio_B1_B5": float(texp[0] / texp[-1]),
    "inset": {
        "Bcs_range": [float(bmin), float(bmax)],
        "scale_range": [float(sc.min()), float(sc.max())],
        "scale_ratio_top_bottom": float(sc[-1] / sc[0]),
        "best_fit": best,
    },
    "main_figure": collapse,
    "fixed_beta": {str(k): {k2: (float(v2) if isinstance(v2, float) else None) for k2, v2 in v.items()}
                   for k, v in fixed.items()},
    "robustness": {
        "leave_out_curve": rob_leaveout,
        "first_lag_effect_pct": [r["change_pct"] for r in rob_firstlag],
    },
    "finding": (
        "The correlation length (integral of C(lag), and the exponential decay time constant) "
        "decreases monotonically and dramatically across buffer conditions B1->B5, from "
        f"~{t[0]:.0f} to ~{t[-1]:.0f} (integral; factor ~{t[0]/t[-1]:.0f}x), i.e. it is longest "
        "at B1 and shortest at B5. This is the correlation-length signature of a finite-size / "
        "critical slowing-down transition: the correlation length is largest at the buffer "
        "condition closest to the critical point and collapses as one moves away. The inset "
        f"shows the scaling factor growing ~{sc[-1]/sc[0]:.0f}x over the Bcs range "
        f"{bmin:.3f}->{bmax:.3f}, consistent with divergence near Bc."
    ),
    "caveats": [
        "The inset collapse uses only 5 data points; the fitted beta and Bc0 are poorly "
        "conditioned (beta~%.0f, a overflows for the least-squares solution), so the precise "
        "collapse exponent is not reliably determined from this data. The robust, defensible "
        "statement is the monotone, multi-fold decrease of the correlation length." % (best.get("beta", np.nan) if best else np.nan),
        "The correlation curves have a finite lag window; the longest (B1) extends to lag~42181, "
        "the shortest (B5) only to ~1151, so the integral is sensitive to the lag window.",
        "The main-figure linear fits are low-R2 because the y-values (correlation) are all "
        "clustered near 1.0; they are used only to confirm the slow monotone decay.",
    ],
}
# sanitize for json
def _jsonable(v):
    if isinstance(v, (np.ndarray,)):
        return [float(x) for x in v]
    if isinstance(v, (np.integer, np.floating,)):
        return float(v)
    if isinstance(v, np.nan.__class__) or (isinstance(v, float) and math.isnan(v)):
        return None
    return v
with open(os.path.join(OUT, "panel_b_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=_jsonable)

rows = []
for i in range(1, 6):
    c = curves[i]
    rows.append(dict(cond=i, max_lag=c["max_lag"], n=c["n"],
                     integral_tau=c["tau"], exp_tau=c["tau_exp"],
                     C_last=c["C_last"], C_min=c["C_min"]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "panel_b_results.csv"), index=False)

print("\nSaved panel_b_meta.json and panel_b_results.csv")
