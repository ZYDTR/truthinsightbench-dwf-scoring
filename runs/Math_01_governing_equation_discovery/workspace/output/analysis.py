#!/usr/bin/env python3
"""
Governing-Equation Structure and Failure Boundaries in Sparse Spatiotemporal Fields

Analysis of the dense synthetic field and sparse experimental field.
Tests multiple PDE candidates, uses alternative derivative estimators,
identifies failure boundaries, and applies robustness checks.
"""

import numpy as np
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

# Load data
d = np.load(DATA_DIR / "spatiotemporal_fields.npz")
coord = json.loads((DATA_DIR / "coordinate_definitions.json").read_text())

t_s = d["synthetic_time"]
x_s = d["synthetic_space"]
U = d["synthetic_field"]  # shape (T_s, X_s)
T_s, X_s = len(t_s), len(x_s)
dt_s, dx_s = t_s[1]-t_s[0], x_s[1]-x_s[0]

t_e = d["experimental_time"]
x_e = d["experimental_space"]
U_e = d["experimental_field"]  # shape (T_e, X_e, 3)
T_e, X_e, N_ch = len(t_e), len(x_e), 3
dt_e, dx_e = t_e[1]-t_e[0], x_e[1]-x_e[0]

print("=" * 70)
print("SYNTHETIC FIELD ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------------------------
# Derivative computation (multiple methods)
# ---------------------------------------------------------------------------

def compute_derivatives(U, t, x):
    """Compute first and second derivatives using central, forward, backward schemes."""
    T, X = U.shape
    dt, dx = t[1]-t[0], x[1]-x[0]

    # Central differences (interior)
    U_t_cent = np.zeros_like(U)
    U_tt_cent = np.zeros_like(U)
    U_x_cent = np.zeros_like(U)
    U_xx_cent = np.zeros_like(U)
    U_xxt_cent = np.zeros_like(U)

    for i in range(1, T-1):
        U_t_cent[i, :] = (U[i+1, :] - U[i-1, :]) / (2*dt)
        U_tt_cent[i, :] = (U[i+1, :] - 2*U[i, :] + U[i-1, :]) / (dt**2)
    for j in range(1, X-1):
        U_x_cent[:, j] = (U[:, j+1] - U[:, j-1]) / (2*dx)
        U_xx_cent[:, j] = (U[:, j+1] - 2*U[:, j] + U[:, j-1]) / (dx**2)
    for i in range(1, T-1):
        for j in range(1, X-1):
            U_xxt_cent[i, j] = (U_t_cent[i+1, j] - 2*U_t_cent[i, j] + U_t_cent[i-1, j]) / (dt**2)

    # Forward differences
    U_t_fwd = np.zeros_like(U)
    U_tt_fwd = np.zeros_like(U)
    U_x_fwd = np.zeros_like(U)
    U_xx_fwd = np.zeros_like(U)
    for i in range(T-1):
        U_t_fwd[i, :] = (U[i+1, :] - U[i, :]) / dt
        U_tt_fwd[i, :] = (U[i+2, :] - 2*U[i+1, :] + U[i, :]) / (dt**2) if i+2 < T else 0
    for j in range(X-1):
        U_x_fwd[:, j] = (U[:, j+1] - U[:, j]) / dx
        U_xx_fwd[:, j] = (U[:, j+2] - 2*U[:, j+1] + U[:, j]) / (dx**2) if j+2 < X else 0

    # Backward differences
    U_t_bwd = np.zeros_like(U)
    U_tt_bwd = np.zeros_like(U)
    U_x_bwd = np.zeros_like(U)
    U_xx_bwd = np.zeros_like(U)
    for i in range(1, T):
        U_t_bwd[i, :] = (U[i, :] - U[i-1, :]) / dt
        U_tt_bwd[i, :] = (U[i, :] - 2*U[i-1, :] + U[i-2, :]) / (dt**2) if i-2 >= 0 else 0
    for j in range(1, X):
        U_x_bwd[:, j] = (U[:, j] - U[:, j-1]) / dx
        U_xx_bwd[:, j] = (U[:, j] - 2*U[:, j-1] + U[:, j-2]) / (dx**2) if j-2 >= 0 else 0

    # 5-point stencil for second derivatives (more accurate, O(dx^4))
    # (-U[i+2] + 16U[i+1] - 30U[i] + 16U[i-1] - U[i-2]) / (12*h^2)
    U_xx_5pt = np.zeros_like(U)
    U_tt_5pt = np.zeros_like(U)
    for i in range(2, T-2):
        U_tt_5pt[i, :] = (-U[i+2, :] + 16*U[i+1, :] - 30*U[i, :] + 16*U[i-1, :] - U[i-2, :]) / (12*dt**2)
    for j in range(2, X-2):
        U_xx_5pt[:, j] = (-U[:, j+2] + 16*U[:, j+1] - 30*U[:, j] + 16*U[:, j-1] - U[:, j-2]) / (12*dx**2)

    return {
        "U_t_cent": U_t_cent, "U_tt_cent": U_tt_cent, "U_x_cent": U_x_cent, "U_xx_cent": U_xx_cent,
        "U_t_fwd": U_t_fwd, "U_tt_fwd": U_tt_fwd, "U_x_fwd": U_x_fwd, "U_xx_fwd": U_xx_fwd,
        "U_t_bwd": U_t_bwd, "U_tt_bwd": U_tt_bwd, "U_x_bwd": U_x_bwd, "U_xx_bwd": U_xx_bwd,
        "U_xx_5pt": U_xx_5pt, "U_tt_5pt": U_tt_5pt,
    }

derivs = compute_derivatives(U, t_s, x_s)
U_t, U_tt, U_x, U_xx = derivs["U_t_cent"], derivs["U_tt_cent"], derivs["U_x_cent"], derivs["U_xx_cent"]

# ---------------------------------------------------------------------------
# PDE candidate library
# ---------------------------------------------------------------------------

def fit_pde(U_t, U_tt, U_x, U_xx, U, method="lstsq"):
    """Fit various PDE candidates and return coefficients + residual."""
    # Use interior region where all derivatives are defined
    i_range = slice(1, len(t_s)-2)
    j_range = slice(2, len(x_s)-3)

    U_t_i = U_t[i_range, j_range].ravel()
    U_tt_i = U_tt[i_range, j_range].ravel()
    U_x_i = U_x[i_range, j_range].ravel()
    U_xx_i = U_xx[i_range, j_range].ravel()
    U_i = U[i_range, j_range].ravel()
    Uxx_i = U_xx[i_range, j_range].ravel()

    results = {}

    # 1. Pure wave: U_tt = c2 * U_xx
    A = Uxx_i.reshape(-1, 1)
    sol = np.linalg.lstsq(A, U_tt_i, rcond=None)[0]
    c2 = sol[0]
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_tt_i.reshape(-1, 1)).ravel()
    results["wave"] = {"c2": c2, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_tt_i), "n_points": len(U_tt_i)}

    # 2. Damped wave: U_tt + gamma*U_t + c2*U_xx = 0
    A = np.column_stack([U_tt_i, U_t_i, Uxx_i])
    # U_tt = -gamma*U_t - c2*U_xx
    sol = np.linalg.lstsq(np.column_stack([U_t_i, Uxx_i]), -U_tt_i, rcond=None)[0]
    gamma, c2 = sol
    resid = np.linalg.norm(np.column_stack([U_t_i, Uxx_i]) @ np.expand_dims(sol, 1) + U_tt_i.reshape(-1,1)).ravel()
    results["damped_wave"] = {"gamma": gamma, "c2": c2, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_tt_i)}

    # 3. Heat: U_t = D * U_xx
    A = Uxx_i.reshape(-1, 1)
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    D = sol[0]
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["heat"] = {"D": D, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    # 4. Advection-diffusion: U_t = c*U_x + D*U_xx
    A = np.column_stack([U_x_i, Uxx_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    c, D = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["adv_diff"] = {"c": c, "D": D, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    # 5. Reaction-diffusion: U_t = D*U_xx + k*U
    A = np.column_stack([Uxx_i, U_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    D, k = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["rxn_diff"] = {"D": D, "k": k, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    # 6. KdV: U_t + U*U_x + U_xx = 0  =>  U_t = -U*U_x - U_xx
    A = np.column_stack([U_i * U_x_i, Uxx_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    a, b = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["kdv"] = {"a": a, "b": b, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    # 7. Burgers: U_t + U*U_x + D*U_xx = 0
    A = np.column_stack([U_i * U_x_i, Uxx_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    a, D = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["burgers"] = {"a": a, "D": D, "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    # 8. General: U_tt = a*U_xx + b*U_t + c*U + d*U_x + e
    A = np.column_stack([Uxx_i, U_t_i, U_i, U_x_i, np.ones_like(U_tt_i)])
    sol = np.linalg.lstsq(A, U_tt_i, rcond=None)[0]
    a, b, c, d, e = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_tt_i.reshape(-1,1)).ravel()
    results["general_wave"] = {"a": a, "b": b, "c": c, "d": d, "e": e,
                                "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_tt_i)}

    # 9. General: U_t = a*U_xx + b*U_x + c*U + d*U*U_x + e
    A = np.column_stack([Uxx_i, U_x_i, U_i, U_i * U_x_i, np.ones_like(U_t_i)])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    a, b, c, d, e = sol
    resid = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    results["general_heat"] = {"a": a, "b": b, "c": c, "d": d, "e": e,
                                "rel_resid": np.linalg.norm(resid)/np.linalg.norm(U_t_i)}

    return results

pde_results = fit_pde(U_t, U_tt, U_x, U_xx, U)
print("\nPDE candidate fits (synthetic field):")
for name, r in pde_results.items():
    print(f"  {name:15s}: {r}")

# ---------------------------------------------------------------------------
# Alternative derivative estimators
# ---------------------------------------------------------------------------

def fit_pde_alt_derivs(derivs, U, U_t, U_tt, U_x, U_xx):
    """Fit wave equation using alternative derivative estimators."""
    results = {}

    for label, (U_t_d, U_tt_d, U_xx_d) in [
        ("central", (U_t, U_tt, U_xx)),
        ("forward", (derivs["U_t_fwd"], derivs["U_tt_fwd"], derivs["U_xx_fwd"])),
        ("backward", (derivs["U_t_bwd"], derivs["U_tt_bwd"], derivs["U_xx_bwd"])),
    ]:
        i_range = slice(1, len(t_s)-2)
        j_range = slice(2, len(x_s)-3)
        U_tt_i = U_tt_d[i_range, j_range].ravel()
        U_xx_i = U_xx_d[i_range, j_range].ravel()
        if np.all(U_xx_i == 0): continue
        c2 = np.dot(U_xx_i, U_tt_i) / np.dot(U_xx_i, U_xx_i)
        resid = np.linalg.norm(U_tt_i - c2*U_xx_i) / np.linalg.norm(U_tt_i)
        results[label] = {"c2": c2, "rel_resid": resid}

    # 5-point stencil (fit on the interior where both 5-pt derivatives are
    # defined: rows 2..T-3, cols 2..X-3; matches output/robustness.py)
    U_tt_5pt_i = derivs["U_tt_5pt"][2:-2, 2:-2].ravel()
    U_xx_5pt_i = derivs["U_xx_5pt"][2:-2, 2:-2].ravel()
    if np.any(U_xx_5pt_i != 0):
        c2 = np.dot(U_xx_5pt_i, U_tt_5pt_i) / np.dot(U_xx_5pt_i, U_xx_5pt_i)
        resid = np.linalg.norm(U_tt_5pt_i - c2*U_xx_5pt_i) / np.linalg.norm(U_tt_5pt_i)
        results["5pt_stencil"] = {"c2": c2, "rel_resid": resid}

    return results

alt_deriv_results = fit_pde_alt_derivs(derivs, U, U_t, U_tt, U_x, U_xx)
print("\nAlternative derivative estimators (wave equation U_tt = c2*U_xx):")
for label, r in alt_deriv_results.items():
    print(f"  {label:15s}: c2={r['c2']:.6f}, rel_resid={r['rel_resid']:.6f}")

# ---------------------------------------------------------------------------
# Failure boundary analysis
# ---------------------------------------------------------------------------

def compute_residual_field(U, dt, dx):
    """Compute the residual of U_tt - U_xx at each point."""
    U_tt = np.zeros_like(U)
    U_xx = np.zeros_like(U)
    for i in range(1, len(U)-1):
        U_tt[i, :] = (U[i+1, :] - 2*U[i, :] + U[i-1, :]) / dt**2
    for j in range(1, len(U[0])-1):
        U_xx[:, j] = (U[:, j+1] - 2*U[:, j] + U[:, j-1]) / dx**2
    return U_tt - U_xx

resid_field = compute_residual_field(U, dt_s, dx_s)
print("\nFailure boundary analysis (residual = U_tt - U_xx):")
print(f"  Overall: min={resid_field.min():.6f}, max={resid_field.max():.6f}, std={resid_field.std():.6f}")
print(f"  |resid| > 0.1: {np.sum(np.abs(resid_field) > 0.1)}/{len(resid_field.flat)} points")
print(f"  |resid| > 0.5: {np.sum(np.abs(resid_field) > 0.5)}/{len(resid_field.flat)} points")
print(f"  |resid| > 1.0: {np.sum(np.abs(resid_field) > 1.0)}/{len(resid_field.flat)} points")

# Where is the residual large?
for ti in [10, 25, 50, 75]:
    r_row = resid_field[ti, :]
    print(f"  t={t_s[ti]}: |resid| max={np.abs(r_row).max():.6f}, mean={np.abs(r_row).mean():.6f}, "
          f"fraction>0.1={np.mean(np.abs(r_row)>0.1):.3f}")

# Check if residual is concentrated at boundaries
print("\n  Boundary concentration:")
for ti in [10, 25, 50, 75]:
    r_row = resid_field[ti, :]
    boundary = np.abs(r_row[0] + r_row[-1]) / 2
    interior = np.mean(np.abs(r_row[2:-2]))
    print(f"    t={t_s[ti]}: boundary={boundary:.6f}, interior={interior:.6f}, ratio={boundary/interior:.3f}")

# ---------------------------------------------------------------------------
# Initial condition analysis
# ---------------------------------------------------------------------------

print("\nInitial condition analysis:")
print(f"  U(0,x) = all zeros: {np.all(U[0, :] == 0)}")
print(f"  U_t(0,x) (forward): max={np.abs((U[1,:]-U[0,:])/dt_s).max():.6f}")
print(f"  U_t(0,x) antisymmetric: {np.allclose((U[1,:]-U[0,:])/dt_s, -((U[1,:]-U[0,:])/dt_s)[::-1])}")
print(f"  U_t(0,x) is x * const: {np.abs(np.dot((U[1,:]-U[0,:])/dt_s, x_s)/np.dot(x_s, x_s) - 0.05).max() < 0.01}")

# Check if U_t(0,x) is a known function
U_t0 = (U[1, :] - U[0, :]) / dt_s
print(f"\n  U_t(0,x) values (first 10): {np.round(U_t0[:10], 4)}")
print(f"  U_t(0,x) * 20 (scaled): {np.round(U_t0[:10] * 20, 4)}")

# Check if U_t(0,x) = x * a for some a (linear)
sol = np.linalg.lstsq(x_s[:, None], U_t0, rcond=None)[0]
print(f"  U_t(0,x) ~ a*x: a={sol[0]:.6f}, rel_resid={np.linalg.norm(U_t0 - sol[0]*x_s)/np.linalg.norm(U_t0):.6f}")

# Check if U_t(0,x) = sin(k*x) for some k
for k_test in np.linspace(0.1, 5.0, 50):
    resid = np.linalg.norm(U_t0 - np.sin(k_test * x_s)) / np.linalg.norm(U_t0)
    if resid < 0.3:
        print(f"  U_t(0,x) ~ sin(k*x): k={k_test:.4f}, rel_resid={resid:.6f}")

# ---------------------------------------------------------------------------
# Contiguous holdout validation
# ---------------------------------------------------------------------------

def validate_holdout(U, dt, dx, train_times, test_times):
    """Fit PDE on train_times, validate on test_times (contiguous-time holdout)."""
    results = {}
    T = len(U)

    # Spatial second derivative at every (t, x) grid point (interior in x).
    U_xx_full = np.zeros_like(U)
    U_xx_full[:, 1:-1] = (U[:, 2:] - 2*U[:, 1:-1] + U[:, :-2]) / dx**2

    # Temporal second derivative at interior time rows (interior in t).
    U_tt_full = np.zeros_like(U)
    U_tt_full[1:-1, :] = (U[2:, :] - 2*U[1:-1, :] + U[:-2, :]) / dt**2

    def collect(times):
        rows = np.asarray(times)
        # Temporal second derivative at each requested time row.
        tt = U_tt_full[rows, :].ravel()
        # Spatial second derivative at each requested time row.
        xx = U_xx_full[rows, :].ravel()
        return tt, xx

    U_tt_t, U_xx_t = collect(train_times)
    c2_train = (np.dot(U_xx_t, U_tt_t) / np.dot(U_xx_t, U_xx_t)) if np.dot(U_xx_t, U_xx_t) > 0 else 0.0

    U_tt_test, U_xx_test = collect(test_times)
    if np.dot(U_xx_test, U_xx_test) > 0:
        resid_test = np.linalg.norm(U_tt_test - c2_train * U_xx_test) / np.linalg.norm(U_tt_test)
    else:
        resid_test = float('inf')

    results[f"train={np.array(train_times).tolist()}, test={np.array(test_times).tolist()}"] = {
        "c2_train": c2_train, "rel_resid_test": resid_test
    }
    return results

# Train on first 60%, test on last 40%
n_train = int(0.6 * len(t_s))
holdout_results = validate_holdout(U, dt_s, dx_s, range(n_train), range(n_train, len(t_s)))
print("\nContiguous holdout validation (wave equation):")
for key, r in holdout_results.items():
    print(f"  {key}: c2_train={r['c2_train']:.6f}, rel_resid_test={r['rel_resid_test']:.6f}")

# Train on last 60%, test on first 40%
n_test = int(0.4 * len(t_s))
holdout_results2 = validate_holdout(U, dt_s, dx_s, range(n_test, len(t_s)), range(0, n_test))
print("\nReverse holdout (train last 60%, test first 40%):")
for key, r in holdout_results2.items():
    print(f"  {key}: c2_train={r['c2_train']:.6f}, rel_resid_test={r['rel_resid_test']:.6f}")

# ---------------------------------------------------------------------------
# Fourier mode analysis
# ---------------------------------------------------------------------------

print("\n=== Fourier mode analysis ===")
n_x = len(x_s)
F_r = np.fft.rfft(U, axis=1)
K_r = np.fft.rfftfreq(n_x, dx_s)

# For each mode, track amplitude over time
mode_results = []
for k_idx in range(1, len(K_r)-1):
    k = K_r[k_idx]
    sig = F_r[:, k_idx]
    amp = np.abs(sig)
    if amp.max() < 0.1:
        continue

    # Fit amplitude envelope: log(amp) = log(A0) - gamma*t
    mask = amp > 1e-10
    if mask.sum() > 10:
        log_amp = np.log(amp[mask])
        sol = np.linalg.lstsq(np.column_stack([t_s[mask], np.ones_like(t_s[mask])]), log_amp, rcond=None)[0]
        gamma = -sol[0]
        A0 = np.exp(sol[1])
    else:
        gamma = 0
        A0 = amp.max()

    # Fit phase: unwrap and linear fit
    phase = np.unwrap(np.angle(sig))
    sol = np.linalg.lstsq(t_s[:, None], phase, rcond=None)[0]
    w = sol[0]

    # Check phase linearity
    phase_pred = w * t_s
    phase_resid = np.linalg.norm(phase - phase_pred) / (np.max(phase) - np.min(phase))

    mode_results.append({"k": k, "w": w, "gamma": gamma, "A0": A0,
                          "phase_resid": phase_resid, "amp_max": amp.max()})

print(f"{'k':>8} {'w':>10} {'gamma':>10} {'A0':>10} {'phase_resid':>12} {'amp_max':>10}")
for mr in mode_results:
    print(f"{mr['k']:>8.4f} {mr['w']:>10.4f} {mr['gamma']:>10.4f} {mr['A0']:>10.4f} {mr['phase_resid']:>12.4f} {mr['amp_max']:>10.4f}")

# Check dispersion relation
ks = [mr["k"] for mr in mode_results]
ws = [mr["w"] for mr in mode_results]
gammas = [mr["gamma"] for mr in mode_results]

# Linear: w = c*k
sol = np.polyfit(ks, ws, 1)
c_lin, d_lin = sol
resid_lin = np.linalg.norm(ws - (c_lin*np.array(ks) + d_lin)) / np.linalg.norm(ws)
print(f"\nLinear dispersion w = c*k + d: c={c_lin:.6f}, d={d_lin:.6f}, rel_resid={resid_lin:.6f}")

# Damping: gamma = c*k
sol_g = np.polyfit(ks, gammas, 1)
c_g, d_g = sol_g
print(f"Linear damping gamma = c*k + d: c={c_g:.6f}, d={d_g:.6f}")

# ---------------------------------------------------------------------------
# Check if U satisfies a specific PDE with known coefficients
# ---------------------------------------------------------------------------

print("\n=== Specific PDE tests ===")

# Test: U_tt = U_xx (c=1)
resid_c1 = U_tt - U_xx
print(f"U_tt = U_xx: rel_resid = {np.linalg.norm(resid_c1)/np.linalg.norm(U_tt):.6f}")

# Test: U_tt = 0.07 * U_xx
c2_test = 0.07
resid_c2 = U_tt - c2_test * U_xx
print(f"U_tt = {c2_test}*U_xx: rel_resid = {np.linalg.norm(resid_c2)/np.linalg.norm(U_tt):.6f}")

# Test: U_t = D * U_xx (heat)
D_test = 0.05
resid_heat = U_t - D_test * U_xx
print(f"U_t = {D_test}*U_xx: rel_resid = {np.linalg.norm(resid_heat)/np.linalg.norm(U_t):.6f}")

# Test: U_t + U*U_x + U_xx = 0 (KdV)
resid_kdv = U_t + U * U_x + U_xx
print(f"U_t + U*U_x + U_xx = 0: rel_resid = {np.linalg.norm(resid_kdv)/np.linalg.norm(U_t):.6f}")

# ---------------------------------------------------------------------------
# Check for forcing term
# ---------------------------------------------------------------------------

print("\n=== Forcing term analysis ===")
# If U_tt = c2*U_xx + f(x,t), what is f?
c2_best = alt_deriv_results["central"]["c2"]
f = U_tt - c2_best * U_xx
print(f"Using c2={c2_best:.6f}:")
print(f"  f range: [{f.min():.4f}, {f.max():.4f}], std={f.std():.4f}")
print(f"  f at t=1 (first 10): {np.round(f[1, :10], 4)}")
print(f"  f at t=5 (first 10): {np.round(f[5, :10], 4)}")
print(f"  f at t=10 (first 10): {np.round(f[10, :10], 4)}")

# Is f separable? f(x,t) = g(x)*h(t)
# Project f onto spatial modes
F_f = np.fft.rfft(f, axis=1)
print("\n  Spatial FFT of f at selected times:")
for ti in [1, 5, 10, 15]:
    ff = F_f[ti, :]
    top = np.argsort(np.abs(ff))[::-1][:3]
    print(f"    t={t_s[ti]}: top modes {[f'{ff[i]:.4f}' for i in top]}")

# Check if f is proportional to U (reaction term)
print(f"\n  f/U ratio (where U≠0):")
mask = np.abs(U) > 1e-6
ratio = f[mask] / U[mask]
print(f"    mean={ratio.mean():.6f}, std={ratio.std():.6f}, range=[{ratio.min():.6f}, {ratio.max():.6f}]")

# Check if f is proportional to U_xx (dispersion)
print(f"  f/U_xx ratio (where U_xx≠0):")
mask2 = np.abs(U_xx) > 1e-6
ratio2 = f[mask2] / U_xx[mask2]
print(f"    mean={ratio2.mean():.6f}, std={ratio2.std():.6f}, range=[{ratio2.min():.6f}, {ratio2.max():.6f}]")

# Save synthetic results
synthetic_results = {
    "pde_fits": pde_results,
    "alt_derivatives": alt_deriv_results,
    "initial_condition": {
        "U0_zero": bool(np.all(U[0, :] == 0)),
        "Ut0_max": float(np.abs((U[1,:]-U[0,:])/dt_s).max()),
        "Ut0_antisymmetric": bool(np.allclose((U[1,:]-U[0,:])/dt_s, -((U[1,:]-U[0,:])/dt_s)[::-1])),
    },
    "mode_analysis": mode_results,
    "residual_field_stats": {
        "min": float(resid_field.min()),
        "max": float(resid_field.max()),
        "std": float(resid_field.std()),
        "frac_gt_0.1": float(np.mean(np.abs(resid_field) > 0.1)),
        "frac_gt_0.5": float(np.mean(np.abs(resid_field) > 0.5)),
    },
}

# ---------------------------------------------------------------------------
# EXPERIMENTAL FIELD ANALYSIS
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENTAL FIELD ANALYSIS")
print("=" * 70)

# For each channel, compute derivatives and fit PDEs
for ci in range(N_ch):
    print(f"\n--- Channel {ci} ---")
    U_ch = U_e[:, :, ci]  # shape (T_e, X_e)
    T_e, X_e = U_ch.shape

    # Compute derivatives
    U_t_e = np.zeros_like(U_ch)
    U_tt_e = np.zeros_like(U_ch)
    U_x_e = np.zeros_like(U_ch)
    U_xx_e = np.zeros_like(U_ch)

    for i in range(1, T_e-1):
        U_t_e[i, :] = (U_ch[i+1, :] - U_ch[i-1, :]) / (2*dt_e)
        U_tt_e[i, :] = (U_ch[i+1, :] - 2*U_ch[i, :] + U_ch[i-1, :]) / (dt_e**2)
    for j in range(1, X_e-1):
        U_x_e[:, j] = (U_ch[:, j+1] - U_ch[:, j-1]) / (2*dx_e)
        U_xx_e[:, j] = (U_ch[:, j+1] - 2*U_ch[:, j] + U_ch[:, j-1]) / (dx_e**2)

    # Fit PDEs
    i_range = slice(1, T_e-2)
    j_range = slice(1, X_e-2)

    U_t_i = U_t_e[i_range, j_range].ravel()
    U_tt_i = U_tt_e[i_range, j_range].ravel()
    U_x_i = U_x_e[i_range, j_range].ravel()
    U_xx_i = U_xx_e[i_range, j_range].ravel()
    U_i = U_ch[i_range, j_range].ravel()

    # Heat equation: U_t = D * U_xx
    sol = np.linalg.lstsq(U_xx_i.reshape(-1,1), U_t_i, rcond=None)[0]
    D = sol[0]
    resid_heat = np.linalg.norm(U_t_i - D*U_xx_i) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0
    print(f"  Heat: D={D:.6e}, rel_resid={resid_heat:.4f}")

    # Advection-diffusion: U_t = c*U_x + D*U_xx
    A = np.column_stack([U_x_i, U_xx_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    c, D = sol
    resid_adv = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    resid_adv = np.linalg.norm(resid_adv) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0
    print(f"  Adv-diff: c={c:.6e}, D={D:.6e}, rel_resid={resid_adv:.4f}")

    # Reaction-diffusion: U_t = D*U_xx + k*U
    A = np.column_stack([U_xx_i, U_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    D, k = sol
    resid_rxn = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    resid_rxn = np.linalg.norm(resid_rxn) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0
    print(f"  Rxn-diff: D={D:.6e}, k={k:.6f}, rel_resid={resid_rxn:.4f}")

    # Wave equation: U_tt = c2 * U_xx
    sol = np.linalg.lstsq(U_xx_i.reshape(-1,1), U_tt_i, rcond=None)[0]
    c2 = sol[0]
    resid_wave = np.linalg.norm(U_tt_i - c2*U_xx_i) / np.linalg.norm(U_tt_i) if np.linalg.norm(U_tt_i) > 0 else 0
    print(f"  Wave: c2={c2:.6e}, rel_resid={resid_wave:.4f}")

    # Check if the field is growing exponentially
    # U(t) = U(0) * exp(k*t)
    # log(U(t)) = log(U(0)) + k*t
    # This doesn't work directly because U has spatial structure

    # Instead, check the temporal growth rate of the mean
    means = [U_ch[ti, :].mean() for ti in range(T_e)]
    if all(m > 0 for m in means):
        log_means = np.log(np.array(means))
        tarr = np.array([t_e[ti] for ti in range(T_e)], dtype=float)
        # log-linear growth: log(mean) = log(mean0) + k*t  (intercept required)
        sol = np.linalg.lstsq(np.column_stack([tarr, np.ones_like(tarr)]), log_means, rcond=None)[0]
        k_growth = sol[0]
        print(f"  Temporal growth rate (mean): k={k_growth:.6f}/h")

    # Check peak position over time
    peak_positions = [np.argmax(U_ch[ti, :]) for ti in range(T_e)]
    peak_positions_x = [x_e[peak_positions[ti]] for ti in range(T_e)]
    print(f"  Peak positions (x): {peak_positions_x}")
    print(f"  Peak values: {[round(U_ch[ti, :].max(), 6) for ti in range(T_e)]}")

# Check channel similarity
print("\n=== Channel similarity ===")
for ci in range(N_ch):
    for cj in range(ci+1, N_ch):
        diff = U_e[:, :, ci] - U_e[:, :, cj]
        rel_diff = np.abs(diff).mean() / np.abs(U_e[:, :, ci]).mean()
        print(f"  ch{ci} vs ch{cj}: rel_diff={rel_diff:.4f}, max_diff={np.abs(diff).max():.6f}")

# Check if channels are shifted versions
print("\n=== Channel shift analysis ===")
for ci in range(N_ch):
    for cj in range(ci+1, N_ch):
        best_shift = 0
        best_diff = np.inf
        for shift in range(-10, 11):
            if shift == 0: continue
            shifted = np.roll(U_e[:, :, cj], shift, axis=1)
            # Only compare overlapping region
            if shift > 0:
                diff = np.abs(U_e[:, :, ci][:, shift:] - shifted[:, :X_e-shift])
            elif shift < 0:
                diff = np.abs(U_e[:, :, ci][:, :X_e+shift] - shifted[:, -shift:])
            rel_d = diff.mean() / np.abs(U_e[:, :, ci]).mean()
            if rel_d < best_diff:
                best_diff = rel_d
                best_shift = shift
        print(f"  ch{ci} ~ ch{cj} with shift={best_shift}, rel_diff={best_diff:.4f}")

# Save experimental results
experimental_results = {
    "channels": [],
    "channel_similarity": []
}

for ci in range(N_ch):
    U_ch = U_e[:, :, ci]
    U_t_e = np.zeros_like(U_ch)
    U_tt_e = np.zeros_like(U_ch)
    U_x_e = np.zeros_like(U_ch)
    U_xx_e = np.zeros_like(U_ch)
    for i in range(1, T_e-1):
        U_t_e[i, :] = (U_ch[i+1, :] - U_ch[i-1, :]) / (2*dt_e)
        U_tt_e[i, :] = (U_ch[i+1, :] - 2*U_ch[i, :] + U_ch[i-1, :]) / (dt_e**2)
    for j in range(1, X_e-1):
        U_x_e[:, j] = (U_ch[:, j+1] - U_ch[:, j-1]) / (2*dx_e)
        U_xx_e[:, j] = (U_ch[:, j+1] - 2*U_ch[:, j] + U_ch[:, j-1]) / (dx_e**2)

    i_range = slice(1, T_e-2)
    j_range = slice(1, X_e-2)
    U_t_i = U_t_e[i_range, j_range].ravel()
    U_tt_i = U_tt_e[i_range, j_range].ravel()
    U_x_i = U_x_e[i_range, j_range].ravel()
    U_xx_i = U_xx_e[i_range, j_range].ravel()
    U_i = U_ch[i_range, j_range].ravel()

    # Heat
    sol = np.linalg.lstsq(U_xx_i.reshape(-1,1), U_t_i, rcond=None)[0]
    D = sol[0]
    resid_heat = np.linalg.norm(U_t_i - D*U_xx_i) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0

    # Adv-diff
    A = np.column_stack([U_x_i, U_xx_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    c, D2 = sol
    resid_adv = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    resid_adv = np.linalg.norm(resid_adv) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0

    # Rxn-diff
    A = np.column_stack([U_xx_i, U_i])
    sol = np.linalg.lstsq(A, U_t_i, rcond=None)[0]
    D3, k = sol
    resid_rxn = np.linalg.norm(A @ np.expand_dims(sol, 1) - U_t_i.reshape(-1,1)).ravel()
    resid_rxn = np.linalg.norm(resid_rxn) / np.linalg.norm(U_t_i) if np.linalg.norm(U_t_i) > 0 else 0

    # Wave
    sol = np.linalg.lstsq(U_xx_i.reshape(-1,1), U_tt_i, rcond=None)[0]
    c2 = sol[0]
    resid_wave = np.linalg.norm(U_tt_i - c2*U_xx_i) / np.linalg.norm(U_tt_i) if np.linalg.norm(U_tt_i) > 0 else 0

    peak_positions = [int(np.argmax(U_ch[ti, :])) for ti in range(T_e)]
    means = [float(U_ch[ti, :].mean()) for ti in range(T_e)]
    tarr = np.array([t_e[ti] for ti in range(T_e)], dtype=float)
    k_growth = float(np.linalg.lstsq(np.column_stack([tarr, np.ones_like(tarr)]), np.log(np.array(means)), rcond=None)[0][0]) if all(m > 0 for m in means) else 0.0

    experimental_results["channels"].append({
        "channel": ci,
        "heat": {"D": float(D), "rel_resid": float(resid_heat)},
        "adv_diff": {"c": float(c), "D": float(D2), "rel_resid": float(resid_adv)},
        "rxn_diff": {"D": float(D3), "k": float(k), "rel_resid": float(resid_rxn)},
        "wave": {"c2": float(c2), "rel_resid": float(resid_wave)},
        "peak_positions_x": [int(x_e[p]) for p in peak_positions],
        "peak_values": [float(U_ch[ti, :].max()) for ti in range(T_e)],
        "temporal_growth_rate": k_growth,
        "mean_values": means,
    })

# Save all results
results = {
    "synthetic": synthetic_results,
    "experimental": experimental_results,
}

output_path = OUTPUT_DIR / "results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
