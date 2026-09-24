#!/usr/bin/env python3
"""
Adversarial verification of CANDIDATE CLAIM C5:

  "A single mean-level change point near 336.6 ka in d18o_pdb reduces the
   within-segment sum of squares from ~92.9 to ~83.2, which is strong enough
   to be preferred by BIC when the sample size is taken as the raw row count
   593, but the inference must be re-tested against an effective sample size
   derived from the serial correlation (n_eff of order 90-100)."

Independent re-derivation of:
  (1) the change-point scan / SSE values,
  (2) the BIC comparison at n=593 and at a self-derived n_eff,
  (3) serial-correlation-preserving null tests (AR(1), phase-randomised,
      smooth-trend + moving-block residual bootstrap),
  (4) competing smooth-trend / single-oscillation models.

Everything is computed from the raw TSV. Vectorised; <=300 surrogates per null.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/user/conversations/conv-715fe573-e9f2-40bc-ac27-f8ef02db9d25/project")
DATA = ROOT / "data" / "observations.tsv"
OUTJ = ROOT / "output" / "results" / "verify_C5.json"

RNG_SEED = 20260919
N_SURR = 300
MIN_SEG = 40

# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
raw = pd.read_csv(DATA, sep="\t")
raw.columns = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
n_rows_total = len(raw)
cc = raw.dropna(subset=["d18o_pdb"]).reset_index(drop=True)
age = cc["age_ka_bp_1"].to_numpy(float)
y = cc["d18o_pdb"].to_numpy(float)
n = len(y)
dstep = np.diff(age)
step_med = float(np.median(dstep))
span = float(age[-1] - age[0])


def seg_sse(yv, k):
    """SSE of two-segment mean model split after k rows; k=0 or n -> no split."""
    if k <= 0 or k >= len(yv):
        return float(((yv - yv.mean()) ** 2).sum())
    a, b = yv[:k], yv[k:]
    return float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())


def scan(yv, min_seg=MIN_SEG):
    """Vectorised single change-point scan. Returns k grid and SSE(k)."""
    nn = len(yv)
    cs = np.concatenate([[0.0], np.cumsum(yv)])
    cs2 = np.concatenate([[0.0], np.cumsum(yv * yv)])
    S, S2 = cs[-1], cs2[-1]
    k = np.arange(min_seg, nn - min_seg + 1)
    s1, s2 = cs[k], cs2[k]
    left = s2 - s1 * s1 / k
    right = (S2 - s2) - (S - s1) ** 2 / (nn - k)
    return k, left + right


def scan_batch(Y, min_seg=MIN_SEG):
    """Batched scan over rows of Y (B,n). Returns (B, nk) SSE array + k grid."""
    B, nn = Y.shape
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(Y, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((B, 1)), np.cumsum(Y * Y, axis=1)], axis=1)
    S, S2 = cs[:, -1:], cs2[:, -1:]
    k = np.arange(min_seg, nn - min_seg + 1)
    s1, s2 = cs[:, k], cs2[:, k]
    left = s2 - s1 * s1 / k
    right = (S2 - s2) - (S - s1) ** 2 / (nn - k)
    return k, left + right


def bic(n_used, sse, p):
    """Gaussian BIC, p = # estimated mean-side params + 1 (variance)."""
    return n_used * np.log(sse / n_used) + p * np.log(n_used)


k_grid, sse_k = scan(y)
sse0 = float(((y - y.mean()) ** 2).sum())
kbest = int(k_grid[np.argmin(sse_k)])
sse_best = float(sse_k.min())
red_frac_obs = 1.0 - sse_best / sse0
age_boundary = float(0.5 * (age[kbest - 1] + age[kbest]))
age_left = float(age[kbest - 1])

# the location named by the claim
k_claim = int(np.searchsorted(age, 336.6))
sse_at_claim = seg_sse(y, k_claim)
age_at_claim = float(0.5 * (age[k_claim - 1] + age[k_claim]))

# ----------------------------------------------------------------------------
# effective sample size estimates (several methods)
# ----------------------------------------------------------------------------
yc = y - y.mean()
den = float((yc * yc).sum())


def acf(x, kmax):
    xc = x - x.mean()
    d = float((xc * xc).sum())
    return np.array([float((xc[: len(xc) - j] * xc[j:]).sum() / d) for j in range(kmax + 1)])


rho = acf(y, 200)
rho1 = float(rho[1])

# (a) Sokal-window integrated autocorrelation time: tau_int = 0.5 + sum_{j>=1} rho_j
tau_int = None
for M in range(1, 200):
    t = 0.5 + float(rho[1 : M + 1].sum())
    if M >= 5.0 * t:
        tau_int = t
        break
n_eff_sokal = n / (2.0 * tau_int)

# (b) AR(1)-consistent plug-in
n_eff_ar1_plugin = n * (1.0 - rho1) / (1.0 + rho1)

# (c) AR(1) MLE fit of the residual series
def ar1_negll(phi, x):
    xn = x[1:] - phi * x[:-1]
    v = np.var(xn)
    return 0.5 * len(xn) * np.log(v) + 0.5 * float((xn * xn).sum()) / v


try:
    from scipy.optimize import minimize_scalar

    fit = minimize_scalar(lambda p: ar1_negll(p, y), bounds=(-0.99, 0.99), method="bounded")
    rho_hat = float(fit.x)
except Exception:
    rho_hat = rho1
n_eff_ar1_mle = n * (1.0 - rho_hat) / (1.0 + rho_hat)

# (d) blocking estimator from contiguous block means
block_est = {}
for L in (10, 20, 40, 80):
    nb = n // L
    bm = y[: nb * L].reshape(nb, L).mean(axis=1)
    v_bm = float(bm.var(ddof=1))
    block_est[L] = float(n * float(y.var(ddof=1)) / (L * v_bm))

neff_vals = [n_eff_sokal, n_eff_ar1_plugin, n_eff_ar1_mle] + list(block_est.values())
n_eff_lo, n_eff_hi = float(np.min(neff_vals)), float(np.max(neff_vals))
n_eff_med = float(np.median(neff_vals))

# ----------------------------------------------------------------------------
# BIC comparisons
# ----------------------------------------------------------------------------
bic_raw = {
    "no_split": bic(n, sse0, 2),
    "one_split": bic(n, sse_best, 3),
}
bic_raw["delta"] = bic_raw["one_split"] - bic_raw["no_split"]
bic_raw["prefers_split"] = bool(bic_raw["delta"] < 0)

bic_at_neff = {}
for key, ne in [("sokal", n_eff_sokal), ("ar1_plugin", n_eff_ar1_plugin),
                ("ar1_mle", n_eff_ar1_mle), ("block10", block_est[10]),
                ("block20", block_est[20]), ("block40", block_est[40]),
                ("block80", block_est[80]), ("median", n_eff_med),
                ("claim_91", 91.0)]:
    d = bic(ne, sse_best, 3) - bic(ne, sse0, 2)
    bic_at_neff[key] = {"n_eff": ne, "delta": float(d), "prefers_split": bool(d < 0)}
bic_at_neff["n_eff_for_break_even"] = float(
    np.log(n_eff_med) / (-np.log(sse_best / sse0)) if sse_best < sse0 else np.nan
)
# more precisely: delta = ne*ln(ratio) + ln(ne) = 0 -> solve numerically
ratio = sse_best / sse0
from scipy.optimize import brentq

f = lambda x: x * np.log(ratio) + np.log(x)
try:
    breakeven = float(brentq(f, 1.0, 1e4))
except Exception:
    breakeven = float("nan")
bic_at_neff["n_eff_break_even_exact"] = breakeven

# third segment (two change points) via full 2-D scan
def two_cp_scan(yv, min_seg=MIN_SEG):
    nn = len(yv)
    cs = np.concatenate([[0.0], np.cumsum(yv)])
    cs2 = np.concatenate([[0.0], np.cumsum(yv * yv)])
    # SSE of segment i..j (i<j): use cumsums
    i = np.arange(nn + 1)
    # segment cost matrix via broadcasting on (start, end)
    A = cs[None, :] - cs[:, None]      # sum over [start, end)
    A2 = cs2[None, :] - cs2[:, None]
    L = (np.arange(nn + 1)[None, :] - np.arange(nn + 1)[:, None]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = A2 - A * A / L
    cost[L <= 0] = np.inf
    best = (np.inf, -1, -1)
    ks = np.arange(min_seg, nn - 2 * min_seg + 1)
    ks2 = np.arange(min_seg, nn - min_seg + 1)
    c1 = cost[0, ks]                       # 0..k1
    c3 = cost[ks2, nn]                     # k2..n
    mid = cost[np.ix_(ks, ks2)]            # k1..k2  (rows=start, cols=end)
    valid = ks2[None, :] - ks[:, None] >= min_seg
    tot = c1[:, None] + mid + c3[None, :]
    tot = np.where(valid, tot, np.inf)
    idx = np.unravel_index(np.argmin(tot), tot.shape)
    return ks[idx[0]], ks2[idx[1]], float(tot[idx])


k1, k2, sse2 = two_cp_scan(y)

# ----------------------------------------------------------------------------
# null tests (serial-correlation preserving), all vectorised, N_SURR each
# ----------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)


def max_reduction_stats(Y, min_seg=MIN_SEG):
    """For each row of Y: max SSE-reduction fraction over scan, location k, age."""
    nn = Y.shape[1]
    s0 = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    kk, sse = scan_batch(Y, min_seg)
    j = np.argmin(sse, axis=1)
    red = 1.0 - sse[np.arange(len(Y)), j] / s0
    kb = kk[j]
    return red, kb


# --- null A: AR(1) surrogates (same rho, same n, Gaussian, row-ordered)
phi = rho_hat
eps = rng.standard_normal((N_SURR, n))
Yar = np.empty_like(eps)
Yar[:, 0] = eps[:, 0] / np.sqrt(1 - phi * phi)
for j in range(1, n):
    Yar[:, j] = phi * Yar[:, j - 1] + eps[:, j]
Yar = Yar * y.std(ddof=1) + y.mean()
red_ar, kb_ar = max_reduction_stats(Yar)
p_ar_max = float((np.sum(red_ar >= red_frac_obs) + 1) / (N_SURR + 1))
kbest_age_ar = 0.5 * (age[np.clip(kb_ar - 1, 0, n - 1)] + age[np.clip(kb_ar, 0, n - 1)])
frac_near_336 = float(np.mean(np.abs(kbest_age_ar - age_boundary) <= 20.0))
# BIC-neff rule on the nulls (is the BIC@n_eff criterion a valid test?)
ne_ref = n_eff_med
null_prefers = (ne_ref * np.log(np.clip(1 - red_ar, 1e-12, None)) + np.log(ne_ref)) < 0
frac_null_bic_prefers = float(np.mean(null_prefers))

# --- null B: phase-randomised surrogates on a uniform age grid
t_uni = np.arange(age[0], age[-1], step_med)
y_uni = np.interp(t_uni, age, y)
p_uni = len(y_uni)
F = np.fft.rfft(y_uni - y_uni.mean())
amp = np.abs(F)
ph = rng.uniform(0, 2 * np.pi, size=(N_SURR, len(F)))
ph[:, 0] = 0.0
if p_uni % 2 == 0:
    ph[:, -1] = 0.0
SUR = np.fft.irfft(amp[None, :] * np.exp(1j * ph), n=p_uni, axis=1) + y_uni.mean()
red_pr, kb_pr = max_reduction_stats(SUR)
p_pr_max = float((np.sum(red_pr >= red_frac_obs) + 1) / (N_SURR + 1))

# --- null C: smooth trend + moving-block bootstrap of residuals
from statsmodels.nonparametric.smoothers_lowess import lowess

frac_sm = 0.15
sm = lowess(y, age, frac=frac_sm, return_sorted=False)
res = y - sm
res_c = res - res.mean()


def mb_bootstrap(resid, n_out, L, rng, nrep):
    nb = int(np.ceil(n_out / L))
    starts = rng.integers(0, len(resid) - L + 1, size=(nrep, nb))
    idx = starts[:, :, None] + np.arange(L)[None, None, :]
    return resid[idx.reshape(nrep, -1)[:, :n_out]]


nullC = {}
for L in (20, 40):
    Rb = mb_bootstrap(res_c, n, L, rng, N_SURR)
    Yc = sm[None, :] + Rb
    red_c, kb_c = max_reduction_stats(Yc)
    p_c = float((np.sum(red_c >= red_frac_obs) + 1) / (N_SURR + 1))
    age_best_c = 0.5 * (age[np.clip(kb_c - 1, 0, n - 1)] + age[np.clip(kb_c, 0, n - 1)])
    nullC[f"blockL{L}"] = {
        "p_value_max_reduction": p_c,
        "null_red_mean": float(red_c.mean()),
        "null_red_p95": float(np.quantile(red_c, 0.95)),
        "frac_null_best_cp_within_20ka_of_336.6": float(
            np.mean(np.abs(age_best_c - age_boundary) <= 20.0)
        ),
    }

# --- null D: circular block shuffle of the raw series (preserves local shape,
#     destroys the long-range arrangement / the step)
nullD = {}
for L in (40, 80):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(N_SURR, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    Yd = y[idx.reshape(N_SURR, -1)[:, :n]]
    red_d, kb_d = max_reduction_stats(Yd)
    p_d = float((np.sum(red_d >= red_frac_obs) + 1) / (N_SURR + 1))
    age_best_d = 0.5 * (age[np.clip(kb_d - 1, 0, n - 1)] + age[np.clip(kb_d, 0, n - 1)])
    nullD[f"blockL{L}"] = {
        "p_value_max_reduction": p_d,
        "null_red_mean": float(red_d.mean()),
        "null_red_p95": float(np.quantile(red_d, 0.95)),
        "frac_null_best_cp_within_10ka_of_336.6": float(
            np.mean(np.abs(age_best_d - age_boundary) <= 10.0)
        ),
    }

# ----------------------------------------------------------------------------
# smooth alternatives: linear trend, single sinusoid (+trend), lowess
# ----------------------------------------------------------------------------
def sse_of(yv, X):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ yv)
    r = yv - X @ beta
    return float((r * r).sum()), beta


Xlin = np.column_stack([np.ones(n), age - age.mean()])
sse_lin, _ = sse_of(y, Xlin)

periods = np.arange(40.0, 320.0, 0.25)
best_sin = (np.inf, None, None)
for P in periods:
    w = 2 * np.pi * age / P
    X = np.column_stack([np.ones(n), np.sin(w), np.cos(w)])
    s, b = sse_of(y, X)
    if s < best_sin[0]:
        best_sin = (s, float(P), b)
sse_sin, P_best, beta_sin = best_sin
# sinusoid + linear trend
best_sint = (np.inf, None, None)
for P in periods:
    w = 2 * np.pi * age / P
    X = np.column_stack([np.ones(n), age - age.mean(), np.sin(w), np.cos(w)])
    s, b = sse_of(y, X)
    if s < best_sint[0]:
        best_sint = (s, float(P), b)
sse_sint, P_best_t, _ = best_sint

sse_smooth, _ = sse_of(y, np.column_stack([np.ones(n), age - age.mean(), (age - age.mean()) ** 2]))
# lowess "model" with rough df ~ span/frac
sse_lowess = float((res * res).sum())

r2 = lambda s: 1.0 - s / sse0
model_table = {
    "none": {"sse": sse0, "p_params": 2, "r2": 0.0},
    "linear_trend": {"sse": sse_lin, "p_params": 3, "r2": r2(sse_lin)},
    "quadratic": {"sse": sse_smooth, "p_params": 4, "r2": r2(sse_smooth)},
    "one_change_point": {"sse": sse_best, "p_params": 3, "r2": r2(sse_best),
                         "cp_age_ka": age_boundary},
    "two_change_points": {"sse": sse2, "p_params": 4, "r2": r2(sse2),
                          "cp_age_ka": [0.5 * (age[k1 - 1] + age[k1]),
                                        0.5 * (age[k2 - 1] + age[k2])]},
    "sinusoid": {"sse": sse_sin, "p_params": 4, "r2": r2(sse_sin), "period_ka": P_best},
    "sinusoid_plus_trend": {"sse": sse_sint, "p_params": 5, "r2": r2(sse_sint),
                            "period_ka": P_best_t},
    "lowess_frac0.15": {"sse": sse_lowess, "p_params": 4, "r2": r2(sse_lowess)},
}
for kk, v in model_table.items():
    v["bic_raw_n593"] = float(bic(n, v["sse"], v["p_params"]))
    v["bic_neff_median"] = float(bic(n_eff_med, v["sse"], v["p_params"]))
    v["bic_neff_sokal"] = float(bic(n_eff_sokal, v["sse"], v["p_params"]))

# does the change point survive in the *detrended* / de-oscillated series?
for name, Xd in [("linear", Xlin),
                 ("sinusoid", np.column_stack([np.ones(n),
                                               np.sin(2 * np.pi * age / P_best),
                                               np.cos(2 * np.pi * age / P_best)])),
                 ("sinusoid_trend", np.column_stack([np.ones(n), age - age.mean(),
                                                     np.sin(2 * np.pi * age / P_best_t),
                                                     np.cos(2 * np.pi * age / P_best_t)]))]:
    _, bd = sse_of(y, Xd)
    resid_d = y - Xd @ bd
    kk_d, sse_d = scan(resid_d)
    s0_d = float(((resid_d - resid_d.mean()) ** 2).sum())
    j = int(np.argmin(sse_d))
    model_table[f"cp_in_residual_of_{name}"] = {
        "best_cp_age_ka": float(0.5 * (age[kk_d[j] - 1] + age[kk_d[j]])),
        "reduction_fraction_of_residual_sse": float(1 - sse_d[j] / s0_d),
        "p_params": 3,
    }

# ----------------------------------------------------------------------------
# robustness: trimming the record / minimum segment size / weighted scan
# ----------------------------------------------------------------------------
robust = {}
for frac in (0.0, 0.05, 0.10, 0.20):
    cut = int(round(frac * n))
    yv, av = y[: n - cut], age[: n - cut]
    kk2, ss2 = scan(yv)
    j = int(np.argmin(ss2))
    s00 = float(((yv - yv.mean()) ** 2).sum())
    robust[f"trim_last_{int(frac*100)}pct"] = {
        "best_cp_age_ka": float(0.5 * (av[kk2[j] - 1] + av[kk2[j]])),
        "reduction_fraction": float(1 - ss2[j] / s00),
        "mean_shift": float(yv[kk2[j]:].mean() - yv[: kk2[j]].mean()),
    }
for ms in (20, 40, 60, 80):
    kk3, ss3 = scan(y, ms)
    j = int(np.argmin(ss3))
    robust[f"min_seg_{ms}"] = {
        "best_cp_age_ka": float(0.5 * (age[kk3[j] - 1] + age[kk3[j]])),
        "reduction_fraction": float(1 - ss3[j] / sse0),
    }

# age-weighted scan (down-weights the irregular coarse tail)
w = np.gradient(age)
w = w / w.mean()
wy = y * np.sqrt(w)
ck, sse_w = scan(wy)
jw = int(np.argmin(sse_w))
s0w = float(((wy - wy.mean()) ** 2).sum())
robust["age_weighted"] = {
    "best_cp_age_ka": float(0.5 * (age[ck[jw] - 1] + age[ck[jw]])),
    "reduction_fraction": float(1 - sse_w[jw] / s0w),
}

# how much of the right-segment anomaly is carried by the last few rows?
tail = {}
for ntail in (0, 3, 6, 10):
    yv = y[: n - ntail] if ntail else y
    av = age[: n - ntail] if ntail else age
    kk4, ss4 = scan(yv)
    j = int(np.argmin(ss4))
    s00 = float(((yv - yv.mean()) ** 2).sum())
    tail[f"drop_last_{ntail}_rows"] = {
        "best_cp_age_ka": float(0.5 * (av[kk4[j] - 1] + av[kk4[j]])),
        "reduction_fraction": float(1 - ss4[j] / s00),
        "mean_shift": float(yv[kk4[j]:].mean() - yv[: kk4[j]].mean()),
    }

# ----------------------------------------------------------------------------
# summary verdict quantities
# ----------------------------------------------------------------------------
result = {
    "claim_id": "C5",
    "sha256": sha,
    "n_rows_total": n_rows_total,
    "n_used": n,
    "age_span_ka": span,
    "median_age_step_ka": step_med,
    "scan": {
        "best_k_rows_left": kbest,
        "best_cp_age_ka_midpoint": age_boundary,
        "best_cp_age_ka_left_row": age_left,
        "sse_no_split": sse0,
        "sse_best_split": sse_best,
        "sse_at_claim_336.6ka": sse_at_claim,
        "claim_split_age_ka": age_at_claim,
        "sse_reduction_fraction_best": red_frac_obs,
        "sse_reduction_absolute": sse0 - sse_best,
        "mean_left": float(y[:kbest].mean()),
        "mean_right": float(y[kbest:].mean()),
        "mean_shift": float(y[kbest:].mean() - y[:kbest].mean()),
        "rows_right_segment": int(n - kbest),
    },
    "rho_lag1": rho1,
    "ar1_phi_mle": rho_hat,
    "n_eff": {
        "sokal_tau_int_samples": float(tau_int),
        "sokal_tau_int_ka": float(tau_int * step_med),
        "n_eff_sokal": float(n_eff_sokal),
        "n_eff_ar1_plugin": float(n_eff_ar1_plugin),
        "n_eff_ar1_mle": float(n_eff_ar1_mle),
        "n_eff_block": block_est,
        "n_eff_range": [n_eff_lo, n_eff_hi],
        "n_eff_median": n_eff_med,
        "claim_n_eff_range": [90, 100],
        "ratio_n_to_neff_median": float(n / n_eff_med),
    },
    "bic": {
        "raw_n593": bic_raw,
        "at_self_derived_neff": bic_at_neff,
        "n_eff_break_even_exact": breakeven,
    },
    "two_change_point_model": {
        "k1": k1, "k2": k2,
        "cp_ages_ka": [float(0.5 * (age[k1 - 1] + age[k1])),
                       float(0.5 * (age[k2 - 1] + age[k2]))],
        "sse": sse2,
        "delta_bic_neff_median_vs_one_cp": float(
            bic(n_eff_med, sse2, 4) - bic(n_eff_med, sse_best, 3)
        ),
        "delta_bic_raw_n593_vs_one_cp": float(bic(n, sse2, 4) - bic(n, sse_best, 3)),
    },
    "nulls": {
        "observed_max_reduction_fraction": red_frac_obs,
        "ar1": {
            "n_surrogates": N_SURR,
            "p_value": p_ar_max,
            "null_red_mean": float(red_ar.mean()),
            "null_red_p95": float(np.quantile(red_ar, 0.95)),
            "frac_null_best_cp_within_20ka_of_observed": frac_near_336,
            "frac_null_surviving_bic_neff_rule": frac_null_bic_prefers,
        },
        "phase_randomised": {
            "n_surrogates": N_SURR,
            "p_value": p_pr_max,
            "null_red_mean": float(red_pr.mean()),
            "null_red_p95": float(np.quantile(red_pr, 0.95)),
        },
        "smooth_residual_mbb": nullC,
        "circular_block_shuffle": nullD,
    },
    "models": model_table,
    "robustness": robust,
    "tail_sensitivity": tail,
}

OUTJ.parent.mkdir(parents=True, exist_ok=True)
OUTJ.write_text(json.dumps(result, indent=1, default=float))

# console summary
print(json.dumps({k: result[k] for k in
                  ["scan", "rho_lag1", "ar1_phi_mle", "n_eff", "bic",
                   "two_change_point_model"]}, indent=1, default=float))
print("NULLS:", json.dumps(result["nulls"], indent=1, default=float))
print("MODELS:", json.dumps(model_table, indent=1, default=float))
print("ROBUST:", json.dumps(robust, indent=1, default=float))
print("TAIL:", json.dumps(tail, indent=1, default=float))