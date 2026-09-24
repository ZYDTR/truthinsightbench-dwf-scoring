"""
verify_C1.py -- independent adversarial re-derivation of candidate claim C1:

C1: "d18o_pdb contains a dominant periodic component with best-fit period ~108.8 ka
     (trend + 2 harmonics R^2 ~ 0.43); the fundamental amplitude is ~0.38 per mille;
     the phase of that fundamental is stable across 150-ka sliding windows
     (circular resultant length R > 0.95) while its amplitude varies by only ~1.5x."

Independent strategy (deliberately different from the producing step):
  1. Period by a log-spaced coarse grid + bounded golden/ternary refinement
     + parabolic interpolation (NOT a uniform 0.065-ka linear scan);
     independent Lomb-Scargle with fine local oversampling.
  2. Harmonics justified by AICc + F-test, not assumed.
  3. Phase stability re-run at DIFFERENT window widths (100/120/150/180/210 ka),
     non-overlapping greedy blocks (minimal shared data), and window-overlap audit.
  4. Nulls: uniform-phase R null at the true window count; AR(1) red-noise
     surrogates whose windows are refit (vectorised); AR(1) null for R^2 (<=300 sims).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats, signal

ROOT = "/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project"
RES = os.path.join(ROOT, "output", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(20260919)

# ----------------------------------------------------------------------------
# 0. load
# ----------------------------------------------------------------------------
df = pd.read_csv(os.path.join(ROOT, "data", "observations.tsv"), sep="\t")
df.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
n_all = len(df)
df = df.dropna(subset=["d18o_pdb"]).reset_index(drop=True)
x = df["age_ka_bp_1"].to_numpy(float)
y = df["d18o_pdb"].to_numpy(float)
n = len(y)
T = float(x.max() - x.min())
yc = y - y.mean()
sst = float(yc @ yc)

# autocorrelation diagnostics (row step, after NaN deletion vs original spacing)
rho_rows = float(np.corrcoef(y[:-1], y[1:])[0, 1])
raw = pd.read_csv(os.path.join(ROOT, "data", "observations.tsv"), sep="\t")
raw.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
v = raw["d18o_pdb"].to_numpy(float)
pair = ~(np.isnan(v[:-1]) | np.isnan(v[1:]))
rho_pairs = float(np.corrcoef(v[:-1][pair], v[1:][pair])[0, 1])
rho_fit = rho_rows
out = {"step": "verify_C1", "n_rows_all": n_all, "n_rows_used": int(n),
       "span_ka": T, "rho_lag1_compressed_rows": rho_rows,
       "rho_lag1_original_adjacent_pairs": rho_pairs,
       "rho_used_for_ar1_null": rho_fit}


# ----------------------------------------------------------------------------
# 1. design matrix / R^2
# ----------------------------------------------------------------------------
def dmat(xv, period, nharm, trend=True, x0=None):
    x0 = xv.mean() if x0 is None else x0
    cols = [np.ones_like(xv)]
    if trend:
        cols.append(xv - x0)
    for k in range(1, nharm + 1):
        cols.append(np.cos(2 * np.pi * k * xv / period))
        cols.append(np.sin(2 * np.pi * k * xv / period))
    return np.column_stack(cols)


def r2_of(period, nharm=2, trend=True, xv=None, yv=None):
    xv = x if xv is None else xv
    yv = y if yv is None else yv
    A = dmat(xv, period, nharm, trend)
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r = yv - A @ beta
    return float(1.0 - (r @ r) / ((yv - yv.mean()) ** 2).sum()), float(r @ r), beta


def refiner(nharm, trend=True, xv=None, yv=None):
    return lambda p: -r2_of(p, nharm, trend, xv, yv)[0]


def refine(lo, hi, nharm=2, trend=True, xv=None, yv=None):
    res = optimize.minimize_scalar(refiner(nharm, trend, xv, yv),
                                   bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-8, "maxiter": 500})
    return float(res.x)


# ----------------------------------------------------------------------------
# 2. period search: log grid + bounded golden refinement + parabolic interp
# ----------------------------------------------------------------------------
grid = np.exp(np.linspace(np.log(35.0), np.log(400.0), 1200))
r2g = np.array([r2_of(p, 2)[0] for p in grid])
locmax = [(grid[i], r2g[i]) for i in range(1, len(grid) - 1)
          if r2g[i] > r2g[i - 1] and r2g[i] >= r2g[i + 1]]
refined = sorted([(refine(grid[max(np.searchsorted(grid, p) - 1, 0)],
                          grid[min(np.searchsorted(grid, p) + 1, len(grid) - 1)], 2), p)
                  for p, _ in locmax], key=lambda t: -r2_of(t[0], 2)[0])
best_p_gs = refined[0][0]

i0 = int(np.argmax(r2g))
y1, y2, y3 = r2g[i0 - 1], r2g[i0], r2g[i0 + 1]
dpar = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
p_parab = grid[i0] + dpar * (grid[i0 + 1] - grid[i0 - 1]) / 2.0

r2_2h, sse_2h, beta_2h = r2_of(best_p_gs, 2)
r2_1h, sse_1h, beta_1h = r2_of(best_p_gs, 1)
amp_1h = float(np.hypot(beta_1h[2], beta_1h[3]))

r2g1 = np.array([r2_of(p, 1)[0] for p in grid])
i1 = int(np.argmax(r2g1))
best_p_1h = refine(grid[max(i1 - 1, 0)], grid[min(i1 + 1, len(grid) - 1)], 1)
r2_1h_own, sse_1h_own, beta_1h_own = r2_of(best_p_1h, 1)
amp_1h_own = float(np.hypot(beta_1h_own[2], beta_1h_own[3]))
phase_1h_own = float(np.arctan2(beta_1h_own[3], beta_1h_own[2]))

out["period_search"] = {
    "method": "1200-pt log grid 35-400 ka + bounded golden refine + parabolic interp",
    "n_local_maxima_found": len(locmax),
    "best_period_ka": best_p_gs,
    "best_period_parabolic_interp_ka": float(p_parab),
    "best_period_from_coarse_grid_ka": float(grid[i0]),
    "r2_trend_plus_2harmonics": float(r2_2h),
    "r2_fundamental_only_at_that_period": float(r2_1h),
    "fundamental_amplitude_per_mille_at_that_period": amp_1h,
    "best_period_fundamental_only_ka": float(best_p_1h),
    "r2_fundamental_only_at_own_period": float(r2_1h_own),
    "fundamental_amplitude_per_mille_own_period": amp_1h_own,
    "fundamental_phase_rad_own_period": phase_1h_own,
    "n_cycles_in_record": float(T / best_p_gs),
    "top_refined_local_maxima": [
        {"period_ka": float(p), "r2_2harm": float(r2_of(p, 2)[0])} for p, _ in refined[:6]],
}


# ----------------------------------------------------------------------------
# 3. harmonic-count justification
# ----------------------------------------------------------------------------
def aicc(sse, k, nn):
    return nn * np.log(max(sse, 1e-12) / nn) + 2 * k + 2 * k * (k + 1) / (nn - k - 1)


At = np.column_stack([np.ones_like(x), x - x.mean()])
bt, *_ = np.linalg.lstsq(At, y, rcond=None)
rt = y - At @ bt
sse_trend_only = float(rt @ rt)
sse_3h = r2_of(best_p_gs, 3)[1]
models = {"trend_only": (sse_trend_only, 2), "trend+1harm": (sse_1h_own, 4),
          "trend+2harm": (sse_2h, 6), "trend+3harm": (sse_3h, 8)}
F2 = ((sse_1h_own - sse_2h) / 2) / (sse_2h / (n - 6))
F3 = ((sse_2h - sse_3h) / 2) / (sse_3h / (n - 8))
out["model_selection"] = {
    k: {"sse": float(v[0]), "k_params": v[1], "r2": float(1 - v[0] / sst),
        "aicc": float(aicc(v[0], v[1], n))} for k, v in models.items()}
out["model_selection"]["F_add_harmonic2"] = float(F2)
out["model_selection"]["p_add_harmonic2_naive_n593"] = float(1 - stats.f.cdf(F2, 2, n - 6))
out["model_selection"]["p_add_harmonic2_ar1_neff91"] = float(1 - stats.f.cdf(F2, 2, 91 - 6))
out["model_selection"]["F_add_harmonic3"] = float(F3)
out["model_selection"]["p_add_harmonic3_naive_n593"] = float(1 - stats.f.cdf(F3, 2, n - 8))
out["model_selection"]["best_aicc_given_neff"] = {
    "n_eff_593": min(models, key=lambda k: models[k][0] / 1 + 0),
}
# note: the AICc values above use n=593 (row count); recompute with n_eff = 91.7
neff = 91.73
out["model_selection"]["aicc_neff91"] = {
    k: float(aicc(v[0] * (neff / n), v[1], int(neff))) for k, v in models.items()}

harm_amps = {}
beta_f = np.linalg.lstsq(dmat(x, best_p_gs, 3), y, rcond=None)[0]
for k in range(1, 4):
    harm_amps[k] = float(np.hypot(beta_f[2 + 2 * (k - 1)], beta_f[3 + 2 * (k - 1)]))
out["harmonic_amplitudes_at_best_period_per_mille"] = harm_amps
out["r2_at_2harm_at_108_8_exact"] = float(r2_of(108.8, 2)[0])


# ----------------------------------------------------------------------------
# 4. independent Lomb-Scargle with fine local oversampling
# ----------------------------------------------------------------------------
def ls_power(xv, yv, f):
    yv = yv - yv.mean()
    t = 2 * np.pi * f * xv
    c, s = np.cos(t), np.sin(t)
    cc, ss, cs = c @ c, s @ s, c @ s
    tau = 0.5 * np.arctan2(2 * cs, cc - ss)
    ct, st = np.cos(t - tau), np.sin(t - tau)
    return float(0.5 * ((ct @ yv) ** 2 / (ct @ ct) + (st @ yv) ** 2 / (st @ st)))


fgrid = np.linspace(1 / 400.0, 1 / 35.0, 6000)
pw = np.array([ls_power(x, y, f) for f in fgrid])
i_ = int(np.argmax(pw))
dfine = np.linspace((fgrid[i_ - 1] + fgrid[i_]) / 2, (fgrid[i_] + fgrid[i_ + 1]) / 2, 2001)
pf = np.array([ls_power(x, y, f) for f in dfine])
j = int(np.argmax(pf))
y1, y2, y3 = pf[j - 1], pf[j], pf[j + 1]
ls_f = dfine[j] + 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3) * (dfine[1] - dfine[0])
out["lomb_scargle_independent"] = {
    "n_freqs_coarse": len(fgrid), "best_period_ka": float(1 / dfine[j]),
    "best_period_parabolic_ka": float(1 / ls_f), "power": float(pf[j]),
}

# ----------------------------------------------------------------------------
# 5. sliding windows at DIFFERENT widths
# ----------------------------------------------------------------------------
def window_scan(W, dW, band=0.015):
    starts = np.arange(x.min(), x.max() - W + 1e-9, dW)
    rows = []
    for s in starts:
        m = (x >= s) & (x <= s + W)
        if m.sum() < 40:
            continue
        xv, yv = x[m], y[m]
        pb = refine(best_p_gs * (1 - band), best_p_gs * (1 + band), 1, True, xv, yv)
        A = dmat(xv, pb, 1, True)
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        r = yv - A @ b
        Am = dmat(xv, best_p_gs, 1, True)
        bm, *_ = np.linalg.lstsq(Am, yv, rcond=None)
        rm = yv - Am @ bm
        rows.append({"start": float(s), "end": float(s + W), "n": int(m.sum()),
                     "period_ka": float(pb), "amp": float(np.hypot(b[2], b[3])),
                     "phase": float(np.arctan2(b[3], b[2])),
                     "r2_1harm_freeP": float(1 - (r @ r) / ((yv - yv.mean()) ** 2).sum()),
                     "amp_frozenP": float(np.hypot(bm[2], bm[3])),
                     "phase_frozenP": float(np.arctan2(bm[3], bm[2])),
                     "r2_1harm_frozenP": float(1 - (rm @ rm) / ((yv - yv.mean()) ** 2).sum())})
    return pd.DataFrame(rows)


audit, frames = {}, {}
for W, dW in [(100, 10), (120, 10), (120, 20), (150, 10), (150, 20), (180, 10),
              (180, 20), (210, 10), (250, 10)]:
    d = window_scan(W, dW)
    frames[f"W{int(W)}_step{int(dW)}"] = d
    ph = d["phase"].to_numpy()
    z = np.exp(1j * ph).mean()
    audit[f"W{int(W)}_step{int(dW)}"] = {
        "n_windows": int(len(d)), "window_width_ka": W, "step_ka": dW,
        "adjacent_window_data_overlap_fraction": float(max(0.0, (W - dW) / W)),
        "window_R_mean_ka": float(d["period_ka"].mean()),
        "window_R_sd_ka": float(d["period_ka"].std(ddof=1)),
        "amp_min": float(d["amp"].min()), "amp_max": float(d["amp"].max()),
        "amp_ratio_max_over_min": float(d["amp"].max() / d["amp"].min()),
        "phase_resultant_R_freeP": float(np.abs(z)),
        "phase_circmean_deg": float(np.degrees(np.angle(z))),
        "phase_sd_rad_from_R": float(np.sqrt(-2 * np.log(max(np.abs(z), 1e-12)))),
        "mean_window_r2_1harm_freeP": float(d["r2_1harm_freeP"].mean()),
        "min_window_r2_1harm_freeP": float(d["r2_1harm_freeP"].min()),
        "mean_window_r2_1harm_frozenP": float(d["r2_1harm_frozenP"].mean()),
        "n_windows_with_r2_below_0.2": int((d["r2_1harm_freeP"] < 0.2).sum()),
    }
    d.round(6).to_csv(os.path.join(RES, f"verify_C1_windows_W{int(W)}_step{int(dW)}.csv"), index=False)
out["sliding_window_scan"] = audit

# ----------------------------------------------------------------------------
# 6a. uniform-phase null for R at the REAL window counts
# ----------------------------------------------------------------------------
uni = {}
for key, v in audit.items():
    nw = v["n_windows"]
    th = rng.uniform(0, 2 * np.pi, size=(200000, nw))
    Rn = np.abs(np.exp(1j * th).sum(axis=1)) / nw
    uni[key] = {"n_windows": nw, "observed_R": v["phase_resultant_R_freeP"],
                "p_R_ge_obs": float((Rn >= v["phase_resultant_R_freeP"]).mean()),
                "null_R_median": float(np.median(Rn)),
                "null_R_95": float(np.quantile(Rn, 0.95))}
    for nw2 in [2, 3, 4]:
        if nw2 < nw:
            th2 = rng.uniform(0, 2 * np.pi, size=(200000, nw2))
            R2 = np.abs(np.exp(1j * th2).sum(axis=1)) / nw2
            uni[key][f"p_R_ge_obs_if_only_{nw2}_independent_windows"] = float(
                (R2 >= v["phase_resultant_R_freeP"]).mean())
out["R_null_uniform_phase"] = uni


# ----------------------------------------------------------------------------
# 6b. AR(1) surrogate windows (vectorised; period frozen at global best)
# ----------------------------------------------------------------------------
sd = float(np.std(y, ddof=1))
def ar1_matrix(nsim, nn, rho_, sd_):
    u = rng.normal(0, sd_ * np.sqrt(1 - rho_ ** 2), size=(nsim, nn))
    e = signal.lfilter([1.0], [1.0, -rho_], u, axis=1)
    return e * (sd_ / np.std(e, axis=1, ddof=1))[:, None]


def surrogate_window_scan(W, dW, nsim=250, freeze=True):
    starts = np.arange(x.min(), x.max() - W + 1e-9, dW)
    keep = [(x >= s) & (x <= s + W) for s in starts]
    keep = [m for m in keep if m.sum() >= 40]
    E = ar1_matrix(nsim, n, rho_fit, sd)
    nw = len(keep)
    PH = np.empty((nsim, nw))
    AM = np.empty((nsim, nw))
    R2s = np.empty((nsim, nw))
    for w, m in enumerate(keep):
        xv, yv_obs = x[m], y[m]
        p = best_p_gs if freeze else refine(best_p_gs * 0.95, best_p_gs * 1.05, 1, True, xv, yv_obs)
        A = dmat(xv, p, 1, True)
        G = np.linalg.pinv(A)
        B = G @ E[:, m].T                      # (4, nsim)
        fit = A @ B
        R2 = 1 - ((E[:, m].T - fit) ** 2).sum(axis=0) / ((E[:, m].T - E[:, m].mean(axis=1)) ** 2).sum(axis=0)
        PH[:, w] = np.arctan2(B[3], B[2])
        AM[:, w] = np.hypot(B[2], B[3])
        R2s[:, w] = R2
    Rs = np.abs(np.exp(1j * PH).mean(axis=1))
    ARs = AM.max(axis=1) / np.maximum(AM.min(axis=1), 1e-12)
    R2mean = R2s.mean(axis=1)
    return Rs, ARs, R2mean


sur = {}
for W in [120, 150, 180]:
    key = f"W{int(W)}_step20"
    Rs, ARs, R2m = surrogate_window_scan(W, 20.0, nsim=250)
    obsR = audit[key]["phase_resultant_R_freeP"]
    obsA = audit[key]["amp_ratio_max_over_min"]
    obsR2 = audit[key]["mean_window_r2_1harm_frozenP"]
    sur[key] = {
        "nsim": 250, "period_frozen_for_surrogates_at_ka": best_p_gs,
        "obs_R": obsR, "obs_amp_ratio": obsA, "obs_mean_window_r2": obsR2,
        "ar1_null_R_mean": float(Rs.mean()), "ar1_null_R_median": float(np.median(Rs)),
        "ar1_null_R_95": float(np.quantile(Rs, 0.95)), "ar1_null_R_max": float(Rs.max()),
        "p_R_ge_obs": float((Rs >= obsR).mean()),
        "ar1_null_amp_ratio_median": float(np.median(ARs)),
        "ar1_null_amp_ratio_95": float(np.quantile(ARs, 0.95)),
        "p_amp_ratio_le_obs": float((ARs <= obsA).mean()),
        "ar1_null_mean_window_r2_median": float(np.median(R2m)),
        "p_r2_ge_obs": float((R2m >= obsR2).mean()),
    }
out["R_null_ar1_surrogate_windows"] = sur


# ----------------------------------------------------------------------------
# 7. forced fixed-phase decomposition (strongest version of the claim)
# ----------------------------------------------------------------------------
forced = {}
for W in [120, 150, 180]:
    key = f"W{int(W)}_step20"
    d = frames[key]
    g = amp_1h_own
    u = beta_1h_own[2] * np.cos(2 * np.pi * x / best_p_1h) + beta_1h_own[3] * np.sin(2 * np.pi * x / best_p_1h)
    u = u / g                                # unit-amplitude global fundamental shape
    amps, r2s = [], []
    for s in d["start"].to_numpy():
        m = (x >= s) & (x <= s + W)
        xv, yv = x[m], y[m]
        A = np.column_stack([np.ones_like(xv), xv - x.mean(), u[m]])
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        r = yv - A @ b
        amps.append(float(b[2]))
        r2s.append(float(1 - (r @ r) / ((yv - yv.mean()) ** 2).sum()))
    amps = np.array(amps)
    forced[key] = {"global_fundamental_phase_rad": phase_1h_own,
                   "fixed_shape_amp_min": float(amps.min()),
                   "fixed_shape_amp_max": float(amps.max()),
                   "fixed_shape_amp_ratio": float(amps.max() / max(amps.min(), 1e-9)),
                   "mean_r2_fixed_phase": float(np.mean(r2s)),
                   "min_r2_fixed_phase": float(np.min(r2s))}
out["fixed_phase_forced_decomposition"] = forced


# ----------------------------------------------------------------------------
# 8. AR(1) red-noise null for R^2 (<=300 sims, period fixed = conservative)
# ----------------------------------------------------------------------------
NSIM = 300
E = ar1_matrix(NSIM, n, rho_fit, sd)
def r2_batch(period, nharm):
    A = dmat(x, period, nharm, True)
    G = np.linalg.pinv(A)
    B = G @ E.T
    fit = A @ B
    ss = ((E.T - fit) ** 2).sum(axis=0)
    sst_e = ((E.T - E.mean(axis=1)) ** 2).sum(axis=0)
    return 1 - ss / sst_e

null2 = r2_batch(best_p_gs, 2)
null1 = r2_batch(best_p_1h, 1)
out["ar1_red_noise_null"] = {
    "nsim": NSIM, "rho": rho_fit, "sd": sd,
    "observed_r2_2harm": float(r2_2h), "observed_r2_1harm": float(r2_1h_own),
    "null_r2_2harm_mean": float(null2.mean()), "null_r2_2harm_sd": float(null2.std(ddof=1)),
    "null_r2_2harm_p95": float(np.quantile(null2, 0.95)), "null_r2_2harm_max": float(null2.max()),
    "p_value_2harm": float((null2 >= r2_2h).mean()),
    "null_r2_1harm_mean": float(null1.mean()), "null_r2_1harm_p95": float(np.quantile(null1, 0.95)),
    "p_value_1harm": float((null1 >= r2_1h_own).mean()),
    "note": "period frozen at the observed optimum for surrogates (conservative w.r.t. the producer's re-scan-per-surrogate protocol)",
}
# same but with a naive rho=0.79
E2 = ar1_matrix(NSIM, n, float(np.sqrt(0.79) / np.sqrt(0.916) * rho_fit) if False else 0.79, sd)
def r2_batch_E(Emat, period, nharm):
    A = dmat(x, period, nharm, True)
    G = np.linalg.pinv(A)
    B = G @ Emat.T
    fit = A @ B
    return 1 - ((Emat.T - fit) ** 2).sum(axis=0) / ((Emat.T - Emat.mean(axis=1)) ** 2).sum(axis=0)
n2b = r2_batch_E(E2, best_p_gs, 2)
out["ar1_red_noise_null_rho079"] = {
    "null_mean": float(n2b.mean()), "null_p95": float(np.quantile(n2b, 0.95)),
    "p_value_2harm": float((n2b >= r2_2h).mean())}

out["trend_vs_periodic"] = {
    "r2_linear_trend_only": float(1 - sse_trend_only / sst),
    "r2_trend_plus_1harm": float(r2_1h_own),
    "r2_trend_plus_2harm": float(r2_2h),
    "slope_per_mille_per_ka": float(beta_1h_own[1]),
    "delta_r2_harmonics_over_trend": float(r2_1h_own - (1 - sse_trend_only / sst)),
}

# ----------------------------------------------------------------------------
# 9. greedy NON-overlapping windows (minimal shared data)
# ----------------------------------------------------------------------------
greedy = {}
for W in [120, 150, 180]:
    starts, s = [], x.min()
    while s + W <= x.max():
        starts.append(s)
        s += W
    ph, am, rr = [], [], []
    for s in starts:
        m = (x >= s) & (x <= s + W)
        xv, yv = x[m], y[m]
        pb = refine(best_p_gs * 0.95, best_p_gs * 1.05, 1, True, xv, yv)
        A = dmat(xv, pb, 1, True)
        b, *_ = np.linalg.lstsq(A, yv, rcond=None)
        r = yv - A @ b
        ph.append(np.arctan2(b[3], b[2]))
        am.append(float(np.hypot(b[2], b[3])))
        rr.append(float(1 - (r @ r) / ((yv - yv.mean()) ** 2).sum()))
    ph = np.array(ph); am = np.array(am)
    z = np.exp(1j * ph).mean()
    nw = len(starts)
    th = rng.uniform(0, 2 * np.pi, size=(200000, nw))
    Rn = np.abs(np.exp(1j * th).sum(axis=1)) / nw
    greedy[f"W{int(W)}_nonoverlap"] = {
        "n_windows": nw, "R": float(np.abs(z)),
        "phases_rad": [float(p) for p in ph], "phase_circmean_rad": float(np.angle(z)),
        "amps": [float(a) for a in am], "per_window_r2_1harm": rr,
        "amp_ratio": float(am.max() / max(am.min(), 1e-9)),
        "p_R_ge_obs_uniform_phase": float((Rn >= np.abs(z)).mean()),
        "per_window_amp_ci_halfwidth_approx": [float(2 * np.sqrt(2 * (1 - v) / 60)) for v in rr],
    }
out["greedy_nonoverlapping_windows"] = greedy

# ----------------------------------------------------------------------------
# 9b. ADVERSARIAL: is "R > 0.95" forced by pinning the period?
#     (a) sweep the free-period band inside each window
#     (b) equal non-overlapping segments with fixed vs free period
#     (c) two independent halves
# ----------------------------------------------------------------------------
def fit4(xv, yv, p):
    A = np.column_stack([np.ones_like(xv), xv - xv.mean(),
                         np.cos(2 * np.pi * xv / p), np.sin(2 * np.pi * xv / p)])
    b, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r = yv - A @ b
    return (float(np.hypot(b[2], b[3])), float(np.arctan2(b[3], b[2])),
            float(1 - (r @ r) / ((yv - yv.mean()) ** 2).sum()))


def rscan(W, dW, band):
    starts = np.arange(x.min(), x.max() - W + 1e-9, dW)
    ph, am, pp = [], [], []
    for s in starts:
        m = (x >= s) & (x <= s + W)
        if m.sum() < 40:
            continue
        xv, yv = x[m], y[m]
        pb = optimize.minimize_scalar(
            lambda q: -fit4(xv, yv, q)[2], bounds=(best_p_1h / band, best_p_1h * band),
            method="bounded").x
        a, th, r = fit4(xv, yv, pb)
        ph.append(th); am.append(a); pp.append(pb)
    ph = np.array(ph); am = np.array(am); pp = np.array(pp)
    z = np.exp(1j * ph).mean()
    return {"band_pct": (band - 1) * 100, "R": float(np.abs(z)),
            "amp_ratio": float(am.max() / max(am.min(), 1e-9)),
            "window_period_min_ka": float(pp.min()), "window_period_max_ka": float(pp.max()),
            "n_windows": len(ph)}


out["period_band_sensitivity_W150_step10"] = [rscan(150, 10, b)
                                             for b in [1.015, 1.05, 1.15, 1.25, 1.5]]

seg = {}
for k in [2, 3, 4]:
    W = T / k
    phF, amF, phR, amR = [], [], [], []
    for i in range(k):
        lo, hi = x.min() + i * W, x.min() + (i + 1) * W
        m = (x >= lo) & (x <= hi)
        xv, yv = x[m], y[m]
        a, th, r = fit4(xv, yv, best_p_1h)             # FIXED period
        phF.append(th); amF.append(a)
        pr = optimize.minimize_scalar(lambda q: -fit4(xv, yv, q)[2],
                                      bounds=(best_p_1h * 0.8, best_p_1h * 1.25),
                                      method="bounded").x     # FREE period
        a, th, r = fit4(xv, yv, pr)
        phR.append(th); amR.append(a)
    phF = np.array(phF); phR = np.array(phR); amF = np.array(amF)
    def punif(R, nw):
        th = rng.uniform(0, 2 * np.pi, size=(200000, nw))
        return float((np.abs(np.exp(1j * th).sum(axis=1)) / nw >= R).mean())
    RF = float(np.abs(np.exp(1j * phF).mean()))
    RR = float(np.abs(np.exp(1j * phR).mean()))
    seg[f"k{k}_W{W:.1f}ka"] = {
        "cycles_per_segment": float(W / best_p_1h),
        "R_fixed_period": RF, "p_uniform_fixed": punif(RF, k),
        "phases_fixed_rad": [float(v) for v in phF],
        "amp_fixed": [float(v) for v in amF],
        "amp_ratio_fixed": float(amF.max() / amF.min()),
        "R_free_period": RR, "p_uniform_free": punif(RR, k),
        "phases_free_rad": [float(v) for v in phR],
    }
out["nonoverlapping_segments"] = seg

hl = {}
for lbl, m in [("first_half", (x <= x.min() + T / 2)), ("second_half", (x > x.min() + T / 2))]:
    xv, yv = x[m], y[m]
    a, th, r = fit4(xv, yv, best_p_1h)
    pr = optimize.minimize_scalar(lambda q: -fit4(xv, yv, q)[2],
                                  bounds=(best_p_1h * 0.9, best_p_1h * 1.1), method="bounded").x
    a2, th2, r2 = fit4(xv, yv, pr)
    hl[lbl] = {"fixed": {"amp": a, "phase": th, "r2": r},
               "free": {"period_ka": float(pr), "amp": a2, "phase": th2, "r2": r2}}
dF = hl["second_half"]["fixed"]["phase"] - hl["first_half"]["fixed"]["phase"]
dF = (dF + np.pi) % (2 * np.pi) - np.pi
dR = hl["second_half"]["free"]["phase"] - hl["first_half"]["free"]["phase"]
dR = (dR + np.pi) % (2 * np.pi) - np.pi
hl["delta_phase_fixed_deg"] = float(np.degrees(dF))
hl["delta_phase_free_deg"] = float(np.degrees(dR))
hl["R_two_halves_fixed"] = float(abs(np.cos(dF / 2)))
hl["R_two_halves_free"] = float(abs(np.cos(dR / 2)))
out["independent_halves"] = hl

# ----------------------------------------------------------------------------
# 10. verdict flags
# ----------------------------------------------------------------------------
out["claim_C1"] = {"period_ka": 108.8, "r2_trend_2harm": 0.43,
                   "fundamental_amp_per_mille": 0.38, "window_width_ka": 150.0,
                   "R_gt": 0.95, "amp_ratio_about": 1.5}
Rs10 = {k: v["phase_resultant_R_freeP"] for k, v in audit.items() if k.endswith("step10")}
out["verdict_flags"] = {
    "period_reproduced_within_3ka": bool(abs(best_p_gs - 108.8) <= 3.0),
    "r2_reproduced_within_0.03": bool(abs(r2_2h - 0.43) <= 0.03),
    "amp_reproduced_within_0.05": bool(abs(amp_1h_own - 0.38) <= 0.05),
    "R_by_width_step10": Rs10,
    "R150_reproduces_gt_0.95": bool(audit["W150_step10"]["phase_resultant_R_freeP"] > 0.95),
    "amp_ratio_W150_reproduces_about_1.5": bool(
        abs(audit["W150_step20"]["amp_ratio_max_over_min"] - 1.5) <= 0.3),
    "R_is_uninformative_vs_uniform_phase_null_at_true_window_count": bool(
        uni["W150_step20"]["p_R_ge_obs"] > 0.10),
    "R_depends_on_window_width": bool(np.ptp(list(Rs10.values())) > 0.03),
    "r2_vs_ar1_null_p": float(out["ar1_red_noise_null"]["p_value_2harm"]),
}
out["verdict_flags"]["n_independent_windows_150ka"] = int(np.floor(T / 150.0))

with open(os.path.join(RES, "verify_C1.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)

print(json.dumps(out, indent=1, default=float))