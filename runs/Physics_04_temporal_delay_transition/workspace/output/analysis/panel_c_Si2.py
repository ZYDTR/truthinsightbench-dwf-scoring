#!/usr/bin/env python3
"""Panel C: finite-size scaling of Bc in SI.2.

Fits tempnet_Bc and meanfield_Bc to a + b/N (and a + b/N^delta with free delta),
assesses convergence, difference between the two, and robustness to the lower-N cutoff.
"""
import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project/data/source_data/figures/SI.2_finite_size/data"
OUT_DIR = "/data/user/conversations/conv-b21b6bd1-7fcf-4b5d-b5bb-9ede8d1a609d/project/output/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load and align data (by index, via .npy)
# ----------------------------------------------------------------------
N = np.load(os.path.join(DATA_DIR, "N_values.npy")).astype(float)
mf = np.load(os.path.join(DATA_DIR, "meanfield_Bc.npy")).astype(float)
tn = np.load(os.path.join(DATA_DIR, "tempnet_Bc.npy")).astype(float)

n_count = len(N)
print(f"n = {n_count}; N range = [{N.min():.0f}, {N.max():.0f}]")
print("N series:", np.unique(N))

# sanity: csv vs npy alignment
N_csv = np.array(pd.read_csv(os.path.join(DATA_DIR, "N_values.csv"), header=None).values).ravel()
print("csv/npy N identical:", np.array_equal(N_csv, N))

# ----------------------------------------------------------------------
# Fitting helpers
# ----------------------------------------------------------------------
def fit_linear_in_x(y, x):
    """Least squares y = a + b*x. Returns a, b, R2, SSE."""
    A = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    resid = y - (a + b * x)
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return a, b, r2, sse

def fit_free_delta(y, N, delta_grid=np.arange(0.1, 4.1, 0.0005)):
    """Fit y = a + b*N^(-delta). For fixed delta, linear in (a,b): exact LS.
    Returns a, b, delta*, R2, SSE, and the full grid profile (delta, sse)."""
    best = None
    profile = []
    for d in delta_grid:
        x = N ** (-d)
        a, b, r2, sse = fit_linear_in_x(y, x)
        if not np.isfinite(sse):
            continue
        profile.append((d, sse))
        if best is None or sse < best[3]:
            best = (a, b, d, sse)
    # refine around the best delta with a finer grid
    d0 = best[2]
    d0 = float(np.clip(d0, 1e-3, 1e6))
    fine = np.linspace(max(1e-3, d0 - 0.05), d0 + 0.05, 2001)
    for d in fine:
        x = N ** (-d)
        a, b, r2, sse = fit_linear_in_x(y, x)
        if not np.isfinite(sse):
            continue
        profile.append((d, sse))
        if sse < best[3]:
            best = (a, b, d, sse)
    return best[0], best[1], best[2], best[3], profile

def fit_block(y, N, min_N):
    """Fit the 1/N model on the subset N >= min_N."""
    mask = N >= min_N
    if mask.sum() < 2:
        return None
    ym = y[mask]
    a, b, r2, sse = fit_linear_in_x(ym, 1.0 / N[mask])
    a2, b2, d, sse2, _ = fit_free_delta(ym, N[mask], delta_grid=np.arange(0.1, 4.1, 0.005))
    if mask.sum() < 3:
        a2, b2, d, sse2 = np.nan, np.nan, np.nan, np.nan
    sst = float(np.sum((ym - ym.mean())**2))
    return {
        "n_pts": int(mask.sum()),
        "Nmin": float(N[mask].min()),
        "Nmax": float(N[mask].max()),
        "a_1overN": a,
        "b_1overN": b,
        "r2_1overN": r2,
        "a_free_delta": a2,
        "b_free_delta": b2,
        "delta": d,
        "r2_free_delta": float(1.0 - sse2 / sst) if sst > 0 else float("nan"),
    }

# ----------------------------------------------------------------------
# 2 & 3. Full-data fits
# ----------------------------------------------------------------------
print("\n--- tempnet_Bc fits ---")
a_t, b_t, r2_t, sse_t = fit_linear_in_x(tn, 1.0 / N)
a2_t, b2_t, d_t, sse2_t, _ = fit_free_delta(tn, N)
sst_t = float(np.sum((tn - tn.mean())**2))
print(f"a (thermo Bc) 1/N fit      : {a_t:.10f}  b = {b_t:.10f}  R2 = {r2_t:.6f}")
print(f"a (thermo Bc) free-delta   : {a2_t:.10f}  b = {b2_t:.10f}  delta = {d_t:.6f}  R2 = {1 - sse2_t/sst_t:.6f}")

print("\n--- meanfield_Bc fits ---")
a_m, b_m, r2_m, sse_m = fit_linear_in_x(mf, 1.0 / N)
a2_m, b2_m, d_m, sse2_m, _ = fit_free_delta(mf, N)
sst_m = float(np.sum((mf - mf.mean())**2))
print(f"a (thermo Bc) 1/N fit      : {a_m:.10f}  b = {b_m:.10f}  R2 = {r2_m:.6f}")
print(f"a (thermo Bc) free-delta   : {a2_m:.10f}  b = {b2_m:.10f}  delta = {d_m:.6f}  R2 = {1 - sse2_m/sst_m:.6f}")

# ----------------------------------------------------------------------
# 4. tempnet - meanfield vs N
# ----------------------------------------------------------------------
diff = tn - mf
print("\n--- tempnet_Bc - meanfield_Bc ---")
print(f"min = {diff.min():.6f} at N={N[np.argmin(diff)]}; max = {diff.max():.6f} at N={N[np.argmax(diff)]}")
print(f"mean = {diff.mean():.6f}; |diff| at N=20: {abs(diff[0]):.6f}; at N=50000: {abs(diff[-1]):.6f}")
# fit diff ~ c/N and report trend
c_d, b_d, r2_d, sse_d = fit_linear_in_x(diff, 1.0 / N)
sst_diff = float(np.sum((diff - diff.mean())**2))
r2_diff = 1.0 - sse_d / sst_diff if sst_diff > 0 else float("nan")
print(f"diff ~ c + b*(1/N): c = {c_d:.6f}, b = {b_d:.6f}, R2 = {r2_diff:.6f}")
# monotonic-ish check: compare mean |diff| in low vs high N
half = n_count // 2
lo = float(np.mean(np.abs(diff[:half])))
hi = float(np.mean(np.abs(diff[half:])))
print(f"mean|diff| low half (N<=100): {lo:.6f}; high half (N>=200): {hi:.6f}; ratio high/low = {hi/lo:.4f}")

# ----------------------------------------------------------------------
# 5. Robustness: fit over N>=40, N>=80, N>=120
# ----------------------------------------------------------------------
print("\n--- robustness (lower-N cutoffs) ---")
rows = []
for name, min_N in [("full", 0), ("N>=40", 40), ("N>=80", 80), ("N>=120", 120)]:
    for label, yy, yname in [("tempnet", tn, "tempnet_Bc"), ("meanfield", mf, "meanfield_Bc")]:
        blk = fit_block(yy, N, min_N)
        rows.append({
            "cutoff": name,
            "series": yname,
            "n_pts": blk["n_pts"],
            "Nmin": blk["Nmin"],
            "Nmax": blk["Nmax"],
            "a_1overN": blk["a_1overN"],
            "b_1overN": blk["b_1overN"],
            "r2_1overN": blk["r2_1overN"],
            "a_free_delta": blk["a_free_delta"],
            "b_free_delta": blk["b_free_delta"],
            "delta": blk["delta"],
            "r2_free_delta": blk["r2_free_delta"],
        })
        print(f"{name:>7} | {yname:<11} | n={blk['n_pts']:<3} [{int(blk['Nmin'])}..{int(blk['Nmax'])}] | "
              f"a(1/N)={blk['a_1overN']:.6f} R2={blk['r2_1overN']:.4f} | "
              f"a(free-d)={blk['a_free_delta']:.6f} delta={blk['delta']:.4f} R2={blk['r2_free_delta']:.4f}")

# ----------------------------------------------------------------------
# Save results CSV
# ----------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "panel_c_results.csv"), index=False)

# ----------------------------------------------------------------------
# Meta JSON
# ----------------------------------------------------------------------
meta = {
    "panel": "panel_c",
    "title": "Finite-size scaling of Bc (SI.2)",
    "n_points": int(n_count),
    "N_range": [float(N.min()), float(N.max())],
    "N_values_unique": [float(x) for x in np.unique(N)],
    "alignment": "aligned by index; csv and npy N arrays identical",
    "tempnet_fit_1_over_N": {
        "a": float(a_t), "b": float(b_t), "R2": float(r2_t), "SSE": float(sse_t),
        "thermodynamic_Bc": float(a_t),
    },
    "tempnet_fit_free_delta": {
        "a": float(a2_t), "b": float(b2_t), "delta": float(d_t),
        "R2": float(1.0 - sse2_t / sst_t), "SSE": float(sse2_t),
        "thermodynamic_Bc": float(a2_t),
    },
    "meanfield_fit_1_over_N": {
        "a": float(a_m), "b": float(b_m), "R2": float(r2_m), "SSE": float(sse_m),
        "thermodynamic_Bc": float(a_m),
    },
    "meanfield_fit_free_delta": {
        "a": float(a2_m), "b": float(b2_m), "delta": float(d_m),
        "R2": float(1.0 - sse2_m / sst_m), "SSE": float(sse2_m),
        "thermodynamic_Bc": float(a2_m),
    },
    "difference": {
        "tempnet_minus_meanfield_min": float(diff.min()),
        "tempnet_minus_meanfield_max": float(diff.max()),
        "diff_at_N20": float(diff[0]),
        "diff_at_N50000": float(diff[-1]),
        "fit_c": float(c_d), "fit_b": float(b_d), "R2": float(r2_diff),
        "mean_abs_diff_low_half": float(lo),
        "mean_abs_diff_high_half": float(hi),
        "ratio_high_over_low": float(hi / lo),
    },
    "robustness": {
        "tempnet": {
            cut: {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in next(r for r in rows if r["cutoff"] == cut and r["series"] == "tempnet_Bc").items()}
            for cut in ["full", "N>=40", "N>=80", "N>=120"]
        },
        "meanfield": {
            cut: {k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in next(r for r in rows if r["cutoff"] == cut and r["series"] == "meanfield_Bc").items()}
            for cut in ["full", "N>=40", "N>=80", "N>=120"]
        },
    },
}

with open(os.path.join(OUT_DIR, "panel_c_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nResults saved to", os.path.join(OUT_DIR, "panel_c_results.csv"))
print("Meta saved to", os.path.join(OUT_DIR, "panel_c_meta.json"))
