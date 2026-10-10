#!/usr/bin/env python3
"""
Robustness checks for the governing-equation discovery analysis.

Additional checks on the synthetic and experimental fields:
  1. Extra PDE candidates (with 3rd derivative term) on the synthetic field.
  2. Different derivative stencils for the wave equation U_tt = c2*U_xx.
  3. Contiguous holdout splits for the wave equation.
  4. Experimental per-channel: mean growth, diffusion (2nd moment), peak trajectory.
  5. Nonlinearity checks on the synthetic field.

Results -> output/robustness_results.json (JSON, no NaN).
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
OUT = OUTPUT_DIR / "robustness_results.json"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d = np.load(DATA_DIR / "spatiotemporal_fields.npz")
t_s = d["synthetic_time"].astype(float)
x_s = d["synthetic_space"].astype(float)
U = d["synthetic_field"]                 # (T_s, X_s)
T_s, X_s = len(t_s), len(x_s)
dt_s = float(t_s[1] - t_s[0])
dx_s = float(x_s[1] - x_s[0])

t_e = d["experimental_time"].astype(float)
x_e = d["experimental_space"].astype(float)
U_e = d["experimental_field"]            # (T_e, X_e, 3)
T_e, X_e, N_ch = U_e.shape

print(f"Synthetic: T={T_s} X={X_s} dt={dt_s} dx={dx_s:.6f} t[{t_s[0]},{t_s[-1]}] x[{x_s[0]},{x_s[-1]}]")
print(f"Experimental: T={T_e} X={X_e} ch={N_ch} dt={dt_s*0} dt_e={float(t_e[1]-t_e[0])} dx_e={float(x_e[1]-x_e[0])}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm(a):
    return float(np.linalg.norm(a))

def rel_resid(resid, ref):
    n = norm(ref)
    if not np.isfinite(n) or n <= 0:
        return float("nan")
    r = np.linalg.norm(resid)
    return float(r / n) if np.isfinite(r) else float("nan")

def fit_linear(y, X):
    """Least-squares fit y = X @ w. Returns (w, resid)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ w
    return w, resid

def to_jsonable(v):
    """Convert numpy scalars/arrays, nan/inf -> None."""
    if isinstance(v, np.ndarray):
        a = np.asarray(v, dtype=float)
        return [None if (not np.isfinite(x)) else float(x) for x in a.ravel()]
    v = float(v)
    return None if not np.isfinite(v) else v

# ---------------------------------------------------------------------------
# Derivatives (synthetic)
# ---------------------------------------------------------------------------
def U_t_central(U, dt):
    out = np.zeros_like(U)
    out[1:-1, :] = (U[2:, :] - 2 * U[1:-1, :] + U[:-2, :]) / (2 * dt)
    return out

def U_tt_central(U, dt):
    out = np.zeros_like(U)
    out[1:-1, :] = (U[2:, :] - 2 * U[1:-1, :] + U[:-2, :]) / (dt * dt)
    return out

def U_tt_5pt(U, dt):
    """Correct 5-point central 2nd-derivative (O(dx^4)):
       (-U[i+2] + 16U[i+1] - 30U[i] + 16U[i-1] - U[i-2])/(12 dt^2)."""
    out = np.zeros_like(U)
    out[2:-2, :] = (-U[4:, :] + 16 * U[3:-1, :] - 30 * U[2:-2, :] + 16 * U[1:-3, :] - U[:-4, :]) / (12 * dt * dt)
    return out

def U_x_central(U, dx):
    out = np.zeros_like(U)
    out[:, 1:-1] = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / (2 * dx)
    return out

def U_x_fwd(U, dx):
    out = np.zeros_like(U)
    out[:, :-1] = (U[:, 1:] - U[:, :-1]) / dx
    return out

def U_x_bwd(U, dx):
    out = np.zeros_like(U)
    out[:, 1:] = (U[:, 1:] - U[:, :-1]) / dx
    return out

def U_xx_central(U, dx):
    out = np.zeros_like(U)
    out[:, 1:-1] = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / (dx * dx)
    return out

def U_xx_5pt(U, dx):
    """Correct 5-point central 2nd-derivative (O(dx^4)):
       (-U[j+2] + 16U[j+1] - 30U[j] + 16U[j-1] - U[j-2])/(12 dx^2)."""
    out = np.zeros_like(U)
    out[:, 2:-2] = (-U[:, 4:] + 16 * U[:, 3:-1] - 30 * U[:, 2:-2] + 16 * U[:, 1:-3] - U[:, :-4]) / (12 * dx * dx)
    return out

def U_xxx_3pt(U, dx):
    """3rd-order central 3rd-derivative: (U[j+2]-3U[j+1]+3U[j-1]-U[j-2])/(2 dx^3), j=2..X-3."""
    out = np.zeros_like(U)
    out[:, 2:-2] = (U[:, 4:] - 3 * U[:, 3:-1] + 3 * U[:, 1:-3] - U[:, :-4]) / (2 * dx * dx * dx)
    return out

def U_xx_spectral(U, dx):
    """Spectral (FFT in space) 2nd derivative, applied along x (axis=1)."""
    nx = U.shape[1]
    F = np.fft.rfft(U, axis=1)
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    Uxx = np.real(np.fft.irfft(F * (-(k * k))[None, :], n=nx, axis=1))
    return Uxx

Ut = U_t_central(U, dt_s)
Utt = U_tt_central(U, dt_s)
Utt5 = U_tt_5pt(U, dt_s)
Ux = U_x_central(U, dx_s)
Uxx = U_xx_central(U, dx_s)
Uxx5 = U_xx_5pt(U, dx_s)
Uxxx = U_xxx_3pt(U, dx_s)
Uxxsp = U_xx_spectral(U, dx_s)

print(f"|U| max={np.abs(U).max():.6f}, U[0,x] all zero={bool(np.all(U[0]==0))}")
print(f"U[:,0] max={np.abs(U[:,0]).max():.3e}, U[:,{-1}] max={np.abs(U[:,-1]).max():.3e} (periodic-FFT boundary ok)")

# ---------------------------------------------------------------------------
# 1. Additional PDE candidates
# ---------------------------------------------------------------------------
print("\n=== 1) Additional PDE candidates (synthetic) ===")
# interior where U_t (rows 1..T-2) and U_xxx (cols 2..X-3) are both defined
i1 = slice(1, T_s - 1)   # U_t, U_tt valid
j1 = slice(2, X_s - 2)   # U_xxx valid (and U_xx, U_x valid)

# --- 1a: U_t = a*U_xx + b*U_xxx + c*U_x + d*U*U_x + e*U + f
Ut_v = Ut[i1, j1].ravel()
Uxx_v = Uxx[i1, j1].ravel()
Uxxx_v = Uxxx[i1, j1].ravel()
Ux_v = Ux[i1, j1].ravel()
UUx_v = (U[i1, j1] * Ux[i1, j1]).ravel()
U_v = U[i1, j1].ravel()
ones_v = np.ones_like(Ut_v)
A = np.column_stack([Uxx_v, Uxxx_v, Ux_v, UUx_v, U_v, ones_v])
w, r = fit_linear(Ut_v, A)
a, b, c, dd, e, f = (float(x) for x in w)
p1 = {
    "n_points": int(len(Ut_v)),
    "coeffs": {"a_Uxx": a, "b_Uxxx": b, "c_Ux": c, "d_UUx": dd, "e_U": e, "f_const": f},
    "rel_resid": rel_resid(r, Ut_v),
}
print(f"  U_t = a Uxx + b Uxxx + c Ux + d U Ux + e U + f")
print(f"    a={a:.6g} b={b:.6g} c={c:.6g} d={dd:.6g} e={e:.6g} f={f:.6g}  rel_resid={p1['rel_resid']:.6f}")

# --- 1b: U_tt = a*U_xx + b*U_xxx + d*U
Utt_v = Utt[i1, j1].ravel()
Uxx_vb = Uxx[i1, j1].ravel()
Uxxx_vb = Uxxx[i1, j1].ravel()
U_vb = U[i1, j1].ravel()
A = np.column_stack([Uxx_vb, Uxxx_vb, U_vb])
w, r = fit_linear(Utt_v, A)
a2, b2, d2 = (float(x) for x in w)
p2 = {
    "n_points": int(len(Utt_v)),
    "coeffs": {"a_Uxx": a2, "b_Uxxx": b2, "d_U": d2},
    "rel_resid": rel_resid(r, Utt_v),
}
print(f"  U_tt = a Uxx + b Uxxx + d U")
print(f"    a={a2:.6g} b={b2:.6g} d={d2:.6g}  rel_resid={p2['rel_resid']:.6f}")

pde_candidates = {"U_t_6term": p1, "U_tt_3term": p2}

# ---------------------------------------------------------------------------
# 2. Wave equation with different stencils
# ---------------------------------------------------------------------------
print("\n=== 2) Wave U_tt = c2*U_xx, different stencils ===")

def fit_c2(Utt_arr, Uxx_arr, i, j, label):
    Utv = Utt_arr[i, j].ravel()
    Uxv = Uxx_arr[i, j].ravel()
    denom = float(np.dot(Uxv, Uxv))
    if denom <= 0 or not np.isfinite(denom):
        return {"c2": float("nan"), "rel_resid": float("nan"), "n_points": int(len(Utv))}
    c2 = float(np.dot(Utv, Uxv) / denom)
    resid = Utv - c2 * Uxv
    return {"c2": c2, "rel_resid": rel_resid(resid, Utv), "n_points": int(len(Utv))}

# common interior columns for 3-point spatial: cols 1..X-2
j3 = slice(1, X_s - 1)
i3 = slice(1, T_s - 1)          # 3-point time valid
i5 = slice(2, T_s - 2)          # 5-point time valid
j5 = slice(2, X_s - 2)          # 5-point space valid

res3 = fit_c2(Utt, Uxx, i3, j3, "3pt")
res5 = fit_c2(Utt5, Uxx5, i5, j5, "5pt")
# Forward 2nd-order: (U[i+2]-2U[i+1]+U[i])/dt^2 ; valid rows 0..T-3
Utt_fwd = np.zeros_like(U)
Utt_fwd[:-2, :] = (U[2:, :] - 2 * U[1:-1, :] + U[:-2, :]) / (dt_s * dt_s)
Uxx_fwd = np.zeros_like(U)
Uxx_fwd[:, :-2] = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / (dx_s * dx_s)
resFwd = fit_c2(Utt_fwd, Uxx_fwd, slice(0, T_s - 2), slice(0, X_s - 2), "fwd")
# Backward: (U[i]-2U[i-1]+U[i-2])/dt^2 ; valid rows 2..T-1
Utt_bwd = np.zeros_like(U)
Utt_bwd[2:, :] = (U[2:, :] - 2 * U[1:-1, :] + U[:-2, :]) / (dt_s * dt_s)
Uxx_bwd = np.zeros_like(U)
Uxx_bwd[:, 2:] = (U[:, 2:] - 2 * U[:, 1:-1] + U[:, :-2]) / (dx_s * dx_s)
resBwd = fit_c2(Utt_bwd, Uxx_bwd, slice(2, T_s), slice(2, X_s), "bwd")
# Spectral: central U_tt (rows 1..T-2), spectral U_xx (all cols)
resSpec = fit_c2(Utt, Uxxsp, i3, slice(0, X_s), "spectral")

wave_stencils = {
    "3pt_central": res3,
    "5pt_central": res5,
    "2nd_order_forward": resFwd,
    "2nd_order_backward": resBwd,
    "spectral_fft_x": resSpec,
}
for k, v in wave_stencils.items():
    print(f"  {k:18s}: c2={v['c2']:.6g}, rel_resid={v['rel_resid']:.6f}, n={v['n_points']}")

# ---------------------------------------------------------------------------
# 3. Contiguous holdout splits (wave)
# ---------------------------------------------------------------------------
print("\n=== 3) Wave holdout splits ===")
# full-grid central second derivatives
Utt_full = Utt
Uxx_full = Uxx

def collect(rows):
    rows = np.asarray(rows, dtype=int)
    rows = rows[(rows >= 1) & (rows <= T_s - 2)]  # U_tt valid rows
    tt = Utt_full[rows, 1:-1].ravel()
    xx = Uxx_full[rows, 1:-1].ravel()
    return tt, xx, rows

def fit_holdout(train_times, test_times, label):
    tt_tr, xx_tr, _ = collect(train_times)
    tt_te, xx_te, _ = collect(test_times)
    denom = float(np.dot(xx_tr, xx_tr))
    c2_tr = float(np.dot(tt_tr, xx_tr) / denom) if denom > 0 else float("nan")
    if denom <= 0 or not np.isfinite(c2_tr):
        return {"c2_train": float("nan"), "rel_resid_test": float("nan"),
                "n_train": int(len(tt_tr)), "n_test": int(len(tt_te))}
    resid = tt_te - c2_tr * xx_te
    return {"c2_train": c2_tr, "rel_resid_test": rel_resid(resid, tt_te),
            "n_train": int(len(tt_tr)), "n_test": int(len(tt_te))}

def idx_range(tmin, tmax, inclusive=True):
    # index range for t in [tmin, tmax] (inclusive both ends)
    idx = np.where((t_s >= tmin) & (t_s <= tmax))[0]
    return list(idx)

splits = [
    ("train[0,3]_test(3,10]", list(range(0, 31)), list(range(31, 101))),
    ("train[0,5]_test(5,10]", list(range(0, 51)), list(range(51, 101))),
    ("train[0,7]_test(7,10]", list(range(0, 71)), list(range(71, 101))),
    ("train[3,10]_test[0,3)", list(range(30, 101)), list(range(0, 30))),
]
holdout = {}
for label, tr, te in splits:
    r = fit_holdout(tr, te, label)
    holdout[label] = r
    print(f"  {label:26s}: c2_train={r['c2_train']:.6g}, rel_resid_test={r['rel_resid_test']:.6f} "
          f"(n_tr={r['n_train']}, n_te={r['n_test']})")

# ---------------------------------------------------------------------------
# 4. Experimental per-channel
# ---------------------------------------------------------------------------
print("\n=== 4) Experimental per-channel ===")
x_mean = float(x_e.mean())
exp_ch = []
for ci in range(N_ch):
    Uch = U_e[:, :, ci]  # (T_e, X_e)
    # --- mean exponential growth: log(mean(t)) = a + b t
    means = np.array([Uch[ti, :].mean() for ti in range(T_e)])
    if np.all(means > 0) and np.all(np.isfinite(means)):
        logm = np.log(means)
        wt = np.column_stack([t_e, np.ones_like(t_e)])
        w, _ = fit_linear(logm, wt)
        a_g, b_g = float(w[1]), float(w[0])
        resid_g = logm - (a_g + b_g * t_e)
        growth = {"b_per_h": b_g, "a_intercept": a_g,
                  "rel_resid": rel_resid(resid_g, logm), "mean_values": means.tolist()}
    else:
        growth = {"b_per_h": float("nan"), "mean_values": means.tolist()}
    print(f"  ch{ci}: mean growth b={growth.get('b_per_h', float('nan')):.6g}/h, means={np.round(means,7)}")

    # --- diffusion: M2(t) = mean((x - mean(x))^2 * profile)/mean(profile)
    dxv = x_e - x_mean
    M2s = np.array([
        np.mean(dxv * dxv * Uch[ti, :]) / Uch[ti, :].mean()
        if Uch[ti, :].mean() > 0 else float("nan")
        for ti in range(T_e)
    ])
    good = np.isfinite(M2s)
    if good.sum() >= 3:
        wt = np.column_stack([t_e[good], np.ones_like(t_e[good])])
        w, _ = fit_linear(M2s[good], wt)
        a_m, b_m = float(w[1]), float(w[0])
        resid_m = M2s[good] - (a_m + b_m * t_e[good])
        diff = {"b_units2_per_h": b_m, "a_intercept": a_m,
                "rel_resid": rel_resid(resid_m, M2s[good]), "M2_values": M2s.tolist()}
    else:
        diff = {"b_units2_per_h": float("nan"), "M2_values": M2s.tolist()}
    print(f"  ch{ci}: diffusion b={diff.get('b_units2_per_h', float('nan')):.6g} units^2/h")

    # --- peak trajectory
    peaks_x = np.array([x_e[np.argmax(Uch[ti, :])] for ti in range(T_e)])
    peaks_v = np.array([Uch[ti, :].max() for ti in range(T_e)])
    goodp = np.isfinite(peaks_x)
    if goodp.sum() >= 2:
        wt = np.column_stack([t_e[goodp], np.ones_like(t_e[goodp])])
        w, _ = fit_linear(peaks_x[goodp], wt)
        a_p, b_p = float(w[1]), float(w[0])
        resid_p = peaks_x[goodp] - (a_p + b_p * t_e[goodp])
        peak = {"b_units_per_h": b_p, "a_intercept": a_p,
                "rel_resid": rel_resid(resid_p, peaks_x[goodp]),
                "peak_x": peaks_x.tolist(), "peak_values": peaks_v.tolist()}
    else:
        peak = {"b_units_per_h": float("nan"), "peak_x": peaks_x.tolist()}
    print(f"  ch{ci}: peak b={peak.get('b_units_per_h', float('nan')):.6g} units/h, peak_x={np.round(peaks_x,0).astype(int).tolist()}")

    exp_ch.append({
        "channel": ci,
        "mean_growth": growth,
        "diffusion": diff,
        "peak_trajectory": peak,
    })

# ---------------------------------------------------------------------------
# 5. Nonlinearity checks (synthetic)
# ---------------------------------------------------------------------------
print("\n=== 5) Nonlinearity checks (synthetic) ===")

def xonly_std(Rgrid, mask):
    """Std of (R - mean_x(R))/mean_x(R), mean_x taken per time row over valid x."""
    devs = []
    for ti in range(T_s):
        v = mask[ti]
        if int(v.sum()) < 5:
            continue
        Rv = Rgrid[ti, v]
        Rv = Rv[np.isfinite(Rv)]
        if Rv.size < 5:
            continue
        mv = float(np.mean(Rv))
        if not np.isfinite(mv) or abs(mv) < 1e-12:
            continue
        devs.append((Rv - mv) / mv)
    if not devs:
        return float("nan"), 0
    dev = np.concatenate(devs)
    return float(np.std(dev)), int(dev.size)

# 5a: U_t/U as function of x only
with np.errstate(divide="ignore", invalid="ignore"):
    R_U = Ut / U
maskU = np.isfinite(R_U) & (np.abs(U) > 1e-6)
stdU, nU = xonly_std(R_U, maskU)
print(f"  U_t/U (|U|>1e-6): std((R-mean_x)/mean_x)={stdU:.6g}, n={nU}")

# 5b: U_t/U_x as function of x only
with np.errstate(divide="ignore", invalid="ignore"):
    R_Ux = Ut / Ux
maskUx = np.isfinite(R_Ux) & (np.abs(Ux) > 1e-6)
stdUx, nUx = xonly_std(R_Ux, maskUx)
print(f"  U_t/U_x (|Ux|>1e-6): std((R-mean_x)/mean_x)={stdUx:.6g}, n={nUx}")

# 5c: U_t/U_xx as function of x only
with np.errstate(divide="ignore", invalid="ignore"):
    R_Uxx = Ut / Uxx
maskUxx = np.isfinite(R_Uxx) & (np.abs(Uxx) > 1e-6)
stdUxx, nUxx = xonly_std(R_Uxx, maskUxx)
print(f"  U_t/U_xx (|Uxx|>1e-6): std((R-mean_x)/mean_x)={stdUxx:.6g}, n={nUxx}")

nonlin = {
    "U_t_over_U": {"std": stdU, "n_points": nU},
    "U_t_over_Ux": {"std": stdUx, "n_points": nUx},
    "U_t_over_Uxx": {"std": stdUxx, "n_points": nUxx},
}

# ---------------------------------------------------------------------------
# Assemble results
# ---------------------------------------------------------------------------
results = {
    "meta": {
        "synthetic": {"T": int(T_s), "X": int(X_s), "dt": float(dt_s), "dx": float(dx_s),
                      "t_range": [float(t_s[0]), float(t_s[-1])],
                      "x_range": [float(x_s[0]), float(x_s[-1])],
                      "U_max": float(np.abs(U).max()),
                      "U0_zero": bool(np.all(U[0] == 0))},
        "experimental": {"T": int(T_e), "X": int(X_e), "channels": int(N_ch),
                         "dt_h": float(t_e[1] - t_e[0]), "dx": float(x_e[1] - x_e[0]),
                         "x_range": [float(x_e[0]), float(x_e[-1])],
                         "t_h": [float(x) for x in t_e]},
        "conventions": {
            "rel_resid": "||resid||_2 / ||reference||_2",
            "U_xxx_3pt": "(U[j+2]-3U[j+1]+3U[j-1]-U[j-2])/(2 dx^3)",
            "U_xx_5pt": "(-U[j+2]+16U[j+1]-30U[j]+16U[j-1]-U[j-2])/(12 dx^2)",
            "U_xxx_3pt_valid": "j in [2, X-3]",
            "wave_stencil_interior": "3pt: rows1..T-2 cols1..X-2; 5pt: rows2..T-3 cols2..X-3",
            "holdout": "train/test disjoint; boundary index assigned to the interval whose label closes on it; U_tt valid rows 1..T-2",
            "M2": "mean((x-mean(x))^2*profile)/mean(profile), x in source units",
            "nonlin_std": "std over all valid points of (R - mean_x(R))/mean_x(R); mean_x per time row over valid x",
        },
    },
    "pde_candidates_synthetic": {
        "U_t_6term": {
            "n_points": p1["n_points"],
            "coeffs": {k: to_jsonable(v) for k, v in p1["coeffs"].items()},
            "rel_resid": to_jsonable(p1["rel_resid"]),
        },
        "U_tt_3term": {
            "n_points": p2["n_points"],
            "coeffs": {k: to_jsonable(v) for k, v in p2["coeffs"].items()},
            "rel_resid": to_jsonable(p2["rel_resid"]),
        },
    },
    "wave_stencils_synthetic": {
        k: {"c2": to_jsonable(v["c2"]), "rel_resid": to_jsonable(v["rel_resid"]),
            "n_points": int(v["n_points"])}
        for k, v in wave_stencils.items()
    },
    "wave_holdout_splits": {
        k: {"c2_train": to_jsonable(v["c2_train"]), "rel_resid_test": to_jsonable(v["rel_resid_test"]),
            "n_train": int(v["n_train"]), "n_test": int(v["n_test"])}
        for k, v in holdout.items()
    },
    "experimental_per_channel": {
        f"ch{ci}": {
            "mean_growth_b_per_h": to_jsonable(exp_ch[ci]["mean_growth"].get("b_per_h")),
            "diffusion_b_units2_per_h": to_jsonable(exp_ch[ci]["diffusion"].get("b_units2_per_h")),
            "peak_b_units_per_h": to_jsonable(exp_ch[ci]["peak_trajectory"].get("b_units_per_h")),
            "mean_values": [to_jsonable(x) for x in exp_ch[ci]["mean_growth"]["mean_values"]],
            "M2_values": [to_jsonable(x) for x in exp_ch[ci]["diffusion"]["M2_values"]],
            "peak_x": [to_jsonable(x) for x in exp_ch[ci]["peak_trajectory"]["peak_x"]],
        }
        for ci in range(N_ch)
    },
    "nonlinearity_synthetic": {
        "U_t_over_U": {"std": to_jsonable(nonlin["U_t_over_U"]["std"]), "n_points": nonlin["U_t_over_U"]["n_points"]},
        "U_t_over_Ux": {"std": to_jsonable(nonlin["U_t_over_Ux"]["std"]), "n_points": nonlin["U_t_over_Ux"]["n_points"]},
        "U_t_over_Uxx": {"std": to_jsonable(nonlin["U_t_over_Uxx"]["std"]), "n_points": nonlin["U_t_over_Uxx"]["n_points"]},
    },
}

# Write JSON (no NaN)
with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)

print(f"\nSaved robustness results to {OUT}")
print("DONE")
