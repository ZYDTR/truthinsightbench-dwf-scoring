#!/usr/bin/env python3
"""INDEPENDENT verification of Draft Finding 3 (sulfate long-record trend).

Written from scratch for the verification stage.  Deliberately uses DIFFERENT
uncertainty machinery from the original analysis (which used a moving-block
bootstrap):

  (1) exact Gaussian AR(1) profile-MLE + GLS (whitened-regression) slope test
  (2) effective-sample-size (Bartlett / Geyer) interval for the correlation
  (3) stationary bootstrap with geometric block lengths (Politis-Romano)
  (4) block permutation test
  (5) Newey-West HAC standard errors
  (6) power / minimum-detectable-slope analysis at 80% power, validated by
      simulation of AR(1) noise with the observed autocorrelation

Nothing here re-uses code from output/analysis/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
DATA = Path("/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv")
SEED = 20260919
rng = np.random.default_rng(SEED)

out: dict = {}
lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- data ------
df = pd.read_csv(DATA, sep="\t")
n_all = len(df)
depth_all = df["c01_depth_top_m"].to_numpy(float)
we = df["c06_depth_w_e_m"].to_numpy(float)
dens = df["c05_density_ice_g_cm_3"].to_numpy(float)
so4 = df["c04_sulfate_ug_kg"].to_numpy(float)
age = df["c08_age_a_ad_ce"].to_numpy(float)
y_raw = np.log10(so4)

ok = ~np.isnan(y_raw)
y = y_raw[ok]
x = depth_all[ok]
w = we[ok]
rho_col = dens[ok]
n = len(y)
say(f"rows total {n_all}, rows with sulfate {n}  (dropped {n_all - n} NaN at depth "
    f"{depth_all[np.where(~ok)[0]]} m)")
say(f"depth c01 range {x.min():.2f}-{x.max():.2f} m ; water-equiv c06 range "
    f"{w.min():.2f}-{w.max():.2f} m")
say(f"log10 sulfate: mean {y.mean():.4f} dex, sd {y.std(ddof=1):.4f} dex")


# ------------------------------------------------- helpers ------------------
def acf(v: np.ndarray, kmax: int) -> np.ndarray:
    c = v - v.mean()
    denom = float(np.dot(c, c))
    # FFT cross-correlation for speed and exactness
    nfft = 1 << int(np.ceil(np.log2(2 * len(c))))
    F = np.fft.rfft(c, nfft)
    acov = np.fft.irfft(F * np.conj(F), nfft)[:kmax + 1]
    return acov[1:] / denom


def ar1_whiten(v: np.ndarray, rho: float) -> np.ndarray:
    """Exact whitening for a stationary AR(1): v*_1 = sqrt(1-rho^2) v_1,
    v*_t = v_t - rho v_{t-1}; then v' R^-1 v = ||v*||^2."""
    o = np.empty_like(v)
    o[0] = np.sqrt(1.0 - rho * rho) * v[0]
    o[1:] = v[1:] - rho * v[:-1]
    return o


def ar1_profile(rho: float, X: np.ndarray, yv: np.ndarray):
    Xw = ar1_whiten(X, rho)
    yw = ar1_whiten(yv, rho)
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ yw)
    rw = yw - Xw @ beta
    rss = float(rw @ rw)
    sigma2 = rss / len(yv)
    ll = -0.5 * len(yv) * np.log(sigma2) - 0.5 * np.log(1.0 - rho * rho)
    return ll, beta, sigma2, XtX_inv, rss


def ar1_mle(X: np.ndarray, yv: np.ndarray, lo: float = 0.0, hi: float = 0.97,
            ngrid: int = 194):
    grid = np.linspace(lo, hi, ngrid)
    lls = np.array([ar1_profile(r, X, yv)[0] for r in grid])
    i = int(np.argmax(lls))
    if 0 < i < len(grid) - 1:
        a, b, c = lls[i - 1], lls[i], lls[i + 1]
        den = a - 2 * b + c
        step = float(np.clip(0.5 * (a - c) / den, -1, 1)) if den != 0 else 0.0
        rho_hat = grid[i] + step * (grid[1] - grid[0])
    else:
        rho_hat = grid[i]
    ll, beta, sigma2, XtX_inv, rss = ar1_profile(rho_hat, X, yv)
    return rho_hat, ll, beta, sigma2, XtX_inv, rss


def gls_slope(xx: np.ndarray, yv: np.ndarray):
    X = np.column_stack([np.ones_like(xx), xx])
    rho_hat, ll, beta, sigma2, XtX_inv, rss = ar1_mle(X, yv)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    b = float(beta[1])
    t = b / se
    dfree = len(xx) - 2
    p = float(2 * stats.t.sf(abs(t), dfree))
    tc = float(stats.t.ppf(0.975, dfree))
    return {"rho_mle": float(rho_hat), "slope": b, "se": se,
            "ci_lo": b - tc * se, "ci_hi": b + tc * se, "p": p,
            "sigma_innov": float(np.sqrt(sigma2))}


def gls_slope_fixed_rho(xx, yv, rho):
    X = np.column_stack([np.ones_like(xx), xx])
    Xw = ar1_whiten(X, rho)
    yw = ar1_whiten(yv, rho)
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ yw)
    rw = yw - Xw @ beta
    s2 = float(rw @ rw) / len(xx)
    return float(beta[1]), float(np.sqrt(s2 * XtX_inv[1, 1]))


def sb_indices(nn: int, mean_block: float, reps: int, rng: np.random.Generator,
               chunk: int = 250) -> np.ndarray:
    """Vectorised Politis-Romano stationary-bootstrap index matrix (reps x nn)."""
    p = 1.0 / mean_block
    tcol = np.arange(nn)
    out = np.empty((reps, nn), dtype=np.int32)
    for s in range(0, reps, chunk):
        e = min(s + chunk, reps)
        m = e - s
        restart = rng.random((m, nn)) < p
        restart[:, 0] = True
        rr = np.where(restart, tcol[None, :], -1)
        rs = np.maximum.accumulate(rr, axis=1)
        sv = rng.integers(0, nn, size=(m, nn))
        start = np.take_along_axis(sv, rs, axis=1)
        out[s:e] = (start + (tcol[None, :] - rs)) % nn
    return out


def sb_corr(yv, xv, mean_block, reps, rng):
    idx = sb_indices(len(yv), mean_block, reps, rng)
    yb = yv[idx]
    xb = xv[idx]
    yb = yb - yb.mean(1, keepdims=True)
    xb = xb - xb.mean(1, keepdims=True)
    num = (yb * xb).sum(1)
    den = np.sqrt((yb * yb).sum(1) * (xb * xb).sum(1))
    return num / den


def corr(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def newyewest_se(xx, resid, lag):
    X = np.column_stack([np.ones_like(xx), xx])
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        wv = 1.0 - k / (lag + 1.0)
        G = u[k:].T @ u[:-k]
        S += wv * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(np.sqrt(V[1, 1]))


def mdes_from_se(se, alpha=0.05, power=0.80, dfree=None):
    """Minimum detectable |slope| at the requested power (two-sided alpha)."""
    if dfree is None:
        return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se
    tcrit = stats.t.ppf(1 - alpha / 2, dfree)

    def pw(b):
        return stats.nct.sf(tcrit, dfree, b / se) + stats.nct.cdf(-tcrit, dfree, b / se)
    lo, hi = 0.0, 60.0 * se
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ar1_noise(nn, rho, sig, rng):
    e = rng.standard_normal(nn) * sig
    z = lfilter([1.0], [1.0, -rho], e)
    z[0] = e[0] / np.sqrt(1 - rho * rho)
    return z


# ==========================================================================
say("")
say("=" * 78)
say("PART 1 -- point correlation and shallow/deep contrast (recomputation)")
say("=" * 78)
r_obs = corr(y, x)
slope_naive, icept, rval, pval_naive, se_naive = stats.linregress(x, y)
say(f"(a) Pearson r(log10 c04, c01 depth) = {r_obs:+.6f}   R2 = {r_obs**2:.6f}")
say(f"    OLS slope = {slope_naive:+.6f} dex/m ; naive SE = {se_naive:.6f} ; "
    f"naive p = {pval_naive:.4g}")
sp = stats.spearmanr(x, y)
say(f"    Spearman rho = {sp.statistic:+.4f} (p={sp.pvalue:.4g})")
say(f"    r(log10 c04, c05 density)  = {corr(y, rho_col):+.6f}")
say(f"    r(log10 c04, c06 w-equiv)  = {corr(y, w):+.6f}")

perm = np.array([corr(y, rng.permutation(x)) for _ in range(4000)])
p_iid_perm = float(2 * min((perm >= r_obs).mean(), (perm <= r_obs).mean()))
say(f"(b) own i.i.d. permutation test (4000 draws): p = {p_iid_perm:.4f} "
    f"(claim quotes 0.003)")

half_full = depth_all < np.median(depth_all)   # original convention: median over all 1931
half = half_full[ok]
d_obs = float(y[half].mean() - y[~half].mean())
say(f"(c) shallow/deep contrast (split at median of ALL {n_all} depths = row "
    f"{int(np.ceil(n_all/2))}):")
say(f"    n_shallow = {half.sum()}, n_deep = {(~half).sum()}")
say(f"    observed contrast = {d_obs:+.6f} dex          (claim: -0.0128 dex)")
half2 = x < np.median(x)
d_obs2 = float(y[half2].mean() - y[~half2].mean())
say(f"    convention check, median over the {n} valid rows: n_shallow={half2.sum()}, "
    f"contrast = {d_obs2:+.6f} dex")
half3 = x <= x.mean()
say(f"    convention check, mean split: n_shallow={half3.sum()}, "
    f"contrast = {y[half3].mean() - y[~half3].mean():+.6f} dex")
out["r_point"] = r_obs
out["r2_point"] = r_obs ** 2
out["slope_ols_naive"] = float(slope_naive)
out["p_ols_naive"] = float(pval_naive)
out["p_iid_perm_own"] = float(p_iid_perm)
out["contrast_obs"] = d_obs
out["contrast_obs_altmedian"] = d_obs2

# ==========================================================================
say("")
say("=" * 78)
say("PART 2 -- serial-correlation-aware uncertainty (own methods)")
say("=" * 78)
ac1_raw = float(np.corrcoef(y[:-1], y[1:])[0, 1])
acfs = acf(y, 60)
first_neg = next((i + 1 for i, v in enumerate(acfs) if v < 0), None)
say(f"lag-1 autocorrelation of log10(sulfate) = {ac1_raw:.4f}  "
    f"(claim mentions rho_used ~0.718)")
say(f"ACF lags 1-8  : {np.round(acfs[:8], 3)}")
say(f"ACF lags 9-16 : {np.round(acfs[8:16], 3)}")
say(f"first negative ACF lag: {first_neg}")

n_eff_bart = n * (1 - ac1_raw) / (1 + ac1_raw)
gps = 0.0
for k in range(0, 60, 2):
    pr = acfs[k] + acfs[k + 1]
    if pr < 0:
        break
    gps += pr
tau_geyer = 2 * (1 + gps)
n_eff_geyer = n / tau_geyer
say("(A) effective sample size:")
say(f"    Bartlett  n(1-r1)/(1+r1)      = {n_eff_bart:.1f}  (nominal n = {n})")
say(f"    Geyer initial-positive-seq    = {n_eff_geyer:.1f}  (tau={tau_geyer:.2f})")
for label, ne in (("Bartlett", n_eff_bart), ("Geyer", n_eff_geyer)):
    se_r = (1 - r_obs ** 2) / np.sqrt(ne)
    z = stats.norm.ppf(0.975)
    t_ess = r_obs / se_r
    p_ess = float(2 * stats.norm.sf(abs(t_ess)))
    spans = (r_obs - z * se_r) < 0 < (r_obs + z * se_r)
    say(f"    {label:8s}: SE(r) = {se_r:.5f} ; 95% CI for r = "
        f"[{r_obs - z*se_r:+.4f}, {r_obs + z*se_r:+.4f}] ; p = {p_ess:.3f} "
        f"({'spans' if spans else 'EXCLUDES'} zero)")
    out[f"ess_{label}_neff"] = float(ne)
    out[f"ess_{label}_ci"] = [float(r_obs - z * se_r), float(r_obs + z * se_r)]
    out[f"ess_{label}_p"] = p_ess

gls = gls_slope(x, y)
say("(B) Gaussian AR(1) errors, rho by exact profile MLE, GLS/whitened regression:")
say(f"    rho_MLE = {gls['rho_mle']:.4f}")
say(f"    GLS slope = {gls['slope']:+.6f} dex/m ; AR(1)-GLS SE = {gls['se']:.6f} "
    f"(naive OLS SE {se_naive:.6f}; inflation x{gls['se']/se_naive:.1f})")
say(f"    95% CI = [{gls['ci_lo']:+.6f}, {gls['ci_hi']:+.6f}] ; p = {gls['p']:.3f}  "
    f"({'SPANS' if gls['ci_lo'] < 0 < gls['ci_hi'] else 'EXCLUDES'} zero)")
out["gls"] = gls

say("(C) stationary bootstrap (Politis-Romano geometric blocks), 2000 reps:")
sb_rows = []
for mb in (25, 50, 100, 200, 400):
    dist = sb_corr(y, x, mb, 2000, rng)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    spans = bool(lo < 0 < hi)
    say(f"    mean block {mb:4d}: r 95% CI [{lo:+.4f}, {hi:+.4f}]  sd={dist.std():.4f}"
        f"  {'spans zero' if spans else 'EXCLUDES zero'}")
    sb_rows.append({"mean_block": mb, "lo95_r": lo, "hi95_r": hi, "sd_r": dist.std(),
                    "spans_zero": spans})

say("    same for the shallow/deep contrast:")
sb_rows_c = []
for mb in (25, 50, 100, 200, 400):
    idx = sb_indices(n, mb, 2000, rng)
    yb = y[idx]
    hb = half[idx]
    ns = hb.sum(1).astype(float)
    nd = n - ns
    with np.errstate(divide="ignore", invalid="ignore"):
        cd = (yb * hb).sum(1) / ns - (yb * ~hb).sum(1) / nd
    n_bad = int(np.isnan(cd).sum())          # degenerate resamples (one side empty)
    cdv = cd[~np.isnan(cd)]
    lo, hi = np.percentile(cdv, [2.5, 97.5])
    spans = bool(lo < 0 < hi)
    say(f"    contrast, mean block {mb:4d}: 95% CI [{lo:+.4f}, {hi:+.4f}] dex  "
        f"P(>0)={float((cdv > 0).mean()):.3f}  (degenerate reps {n_bad}/2000)  "
        f"{'spans zero' if spans else 'EXCLUDES zero'}")
    sb_rows_c.append({"mean_block": mb, "lo95_dex": lo, "hi95_dex": hi,
                      "p_positive": float((cdv > 0).mean()), "spans_zero": spans,
                      "degenerate_reps": n_bad})

Xc = np.column_stack([np.ones(n), half.astype(float)])
rho_c, ll_c, beta_c, s2_c, XtXinv_c, rss_c = ar1_mle(Xc, y)
se_c = float(np.sqrt(s2_c * XtXinv_c[1, 1]))
tc = float(stats.t.ppf(0.975, n - 2))
p_c = float(2 * stats.t.sf(abs(beta_c[1] / se_c), n - 2))
say(f"    contrast, AR(1)-GLS (rho_MLE={rho_c:.4f}): diff = {beta_c[1]:+.6f} dex, "
    f"SE = {se_c:.5f}")
say(f"      95% CI [{beta_c[1]-tc*se_c:+.6f}, {beta_c[1]+tc*se_c:+.6f}] ; p = {p_c:.3f}"
    f"  ({'SPANS' if beta_c[1]-tc*se_c < 0 < beta_c[1]+tc*se_c else 'EXCLUDES'} zero)")
out["contrast_ar1_gls"] = {"rho": float(rho_c), "diff": float(beta_c[1]), "se": se_c,
                           "lo": float(beta_c[1] - tc * se_c),
                           "hi": float(beta_c[1] + tc * se_c), "p": p_c}

say("(D) block permutation test (consecutive blocks of y permuted, 2000 perms):")
bp_rows = []
for L in (50, 100, 200):
    nb = n // L
    blocks = y[:nb * L].reshape(nb, L)
    tail = y[nb * L:]
    perm_r = np.empty(2000)
    for i in range(2000):
        yp = np.concatenate([blocks[rng.permutation(nb)].ravel(), tail])
        perm_r[i] = corr(yp, x)
    p_bp = float((np.abs(perm_r) >= abs(r_obs)).mean())
    say(f"    block length {L:4d}: p = {p_bp:.4f}  (null |r| sd = {perm_r.std():.4f})")
    bp_rows.append({"block": L, "p": p_bp, "null_sd_r": float(perm_r.std())})

say("(E) Newey-West HAC SE of the OLS slope:")
ols_resid = y - (icept + slope_naive * x)
for lag in (7, 27, 50):
    se_h = newyewest_se(x, ols_resid, lag)
    t_h = slope_naive / se_h
    p_h = float(2 * stats.t.sf(abs(t_h), n - 2))
    lo_h, hi_h = slope_naive - 1.96 * se_h, slope_naive + 1.96 * se_h
    say(f"    lag {lag:3d}: SE = {se_h:.6f}, 95% CI = [{lo_h:+.6f}, {hi_h:+.6f}], "
        f"p = {p_h:.3f}  ({'spans' if lo_h < 0 < hi_h else 'EXCLUDES'} zero)")
    out[f"nw_lag{lag}_se"] = float(se_h)

# ==========================================================================
say("")
say("=" * 78)
say("PART 3 -- CRITICAL ATTACK: power and minimum detectable slope")
say("=" * 78)
sig_innov = gls["sigma_innov"]
rho_hat = gls["rho_mle"]
dfree = n - 2
mdes_gls = mdes_from_se(gls["se"], dfree=dfree)
mdes_naive = mdes_from_se(se_naive, dfree=dfree)

xd = x - x.mean()
R = rho_hat ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
# Cov(AR(1) errors) = gamma0 * R with gamma0 = sigma_innov^2 / (1 - rho^2).
# (Using sigma_innov^2 directly here would understate the variance by
#  ~(1-rho^2) = 0.49 -- caught by the simulation check below.)
gamma0 = sig_innov ** 2 / (1.0 - rho_hat ** 2)
Vb_ols_ar1 = gamma0 * (xd @ R @ xd) / (xd @ xd) ** 2
se_ols_ar1 = float(np.sqrt(Vb_ols_ar1))
mdes_ols_ar1 = mdes_from_se(se_ols_ar1, dfree=dfree)

say(f"observed |slope| = {abs(slope_naive):.6f} dex/m ; rho = {rho_hat:.3f} ; "
    f"sigma_innov = {sig_innov:.4f} dex")
say("minimum detectable |slope| at 80% power, alpha=0.05 (two-sided):")
say(f"    naive OLS  (ignores autocorrelation) : {mdes_naive:.6f} dex/m  <- INVALID "
    f"(type-I error inflated, see size check)")
say(f"    OLS with analytic AR(1) variance     : {mdes_ols_ar1:.6f} dex/m "
    f"(SE {se_ols_ar1:.6f})")
say(f"    AR(1) GLS (efficient, rho known)     : {mdes_gls:.6f} dex/m   <- best case")
say(f"observed/MDES(AR1-GLS) = {abs(slope_naive)/mdes_gls:.2f}")
full_range = float(x.max() - x.min())
for lab, m in (("naive OLS", mdes_naive), ("AR(1) aware", mdes_gls)):
    say(f"    a trend of {m:.5f} dex/m would give {m*full_range:.4f} dex "
        f"({10**(m*full_range):.3f}x sulfate) across the {full_range:.2f} m core")
say(f"whereas the OBSERVED point estimate is {slope_naive*full_range:+.4f} dex "
    f"({10**(slope_naive*full_range):.3f}x)")
out["mdes_naive"] = float(mdes_naive)
out["mdes_ols_ar1"] = float(mdes_ols_ar1)
out["mdes_gls"] = float(mdes_gls)
out["full_depth_range_m"] = full_range
out["se_ols_ar1_analytic"] = se_ols_ar1

sim_rng = np.random.default_rng(SEED + 1)
NSIM = 800
null_gls_rej = null_ols_rej = 0
sim_se = np.empty(NSIM)
sim_slope_ols = np.empty(NSIM)
for i in range(NSIM):
    ysim = y.mean() + ar1_noise(n, rho_hat, sig_innov, sim_rng)
    b_gls, s_gls = gls_slope_fixed_rho(x, ysim, rho_hat)
    sim_se[i] = s_gls
    if abs(b_gls / s_gls) > stats.t.ppf(0.975, dfree):
        null_gls_rej += 1
    sl, ic, rr, pp, ss = stats.linregress(x, ysim)
    sim_slope_ols[i] = sl
    if abs(sl / ss) > stats.t.ppf(0.975, dfree):
        null_ols_rej += 1
say(f"validation, {NSIM} AR(1) simulations with NO trend "
    f"(rho={rho_hat:.3f}, sigma_innov={sig_innov:.4f}):")
say(f"    simulated SD(OLS slope) = {sim_slope_ols.std(ddof=1):.6f} vs analytic "
    f"OLS-AR(1) SE {se_ols_ar1:.6f} (ratio "
    f"{sim_slope_ols.std(ddof=1)/se_ols_ar1:.3f}) -> analytic formula validated")
say(f"    simulated SE(GLS slope) = {sim_se.mean():.6f} vs analytic GLS SE "
    f"{gls['se']:.6f} (ratio {sim_se.mean()/gls['se']:.3f}) -> efficient, ratio "
    f"OLS/GLS = {se_ols_ar1/gls['se']:.2f}")
say(f"    naive OLS t-test rejects at 5% in {100*null_ols_rej/NSIM:.1f}% of runs "
    f"-> its p-values are fiction")
say(f"    AR(1)-GLS t-test rejects at 5% in {100*null_gls_rej/NSIM:.1f}% of runs "
    f"-> calibrated")

grid_b = np.array([0.0005, 0.001, 0.0015, 0.002, 0.0022, 0.0025, 0.003, 0.004,
                   0.006, 0.008])
pow_gls, pow_ols = [], []
NSIM2 = 400
tcrit = stats.t.ppf(0.975, dfree)
for b in grid_b:
    rj_gls = rj_ols = 0
    for i in range(NSIM2):
        ysim = y.mean() + b * x + ar1_noise(n, rho_hat, sig_innov, sim_rng)
        bb, ss = gls_slope_fixed_rho(x, ysim, rho_hat)
        rj_gls += abs(bb / ss) > tcrit
        sl, ic, rr, pp, sse = stats.linregress(x, ysim)
        rj_ols += abs(sl / sse) > tcrit
    pow_gls.append(rj_gls / NSIM2)
    pow_ols.append(rj_ols / NSIM2)
    say(f"    slope {b:.5f} dex/m -> power GLS {pow_gls[-1]:.3f} | naive OLS "
        f"{pow_ols[-1]:.3f}")
pow_gls = np.array(pow_gls)
pow_ols = np.array(pow_ols)
mdes_sim = float(np.interp(0.8, pow_gls, grid_b)) if pow_gls.min() < 0.8 <= pow_gls.max() else float("nan")
say(f"    simulated MDES(80% power, AR(1)-GLS) by interpolation = {mdes_sim:.6f} dex/m")
say(f"    analytic  MDES(80% power, AR(1)-GLS)                   = {mdes_gls:.6f} dex/m")
out["mdes_gls_simulated"] = mdes_sim
out["sim_null_rej_ols"] = null_ols_rej / NSIM
out["sim_null_rej_gls"] = null_gls_rej / NSIM

say("VERDICT OF THE POWER ATTACK:")
say(f"    the record can exclude only |trend| > ~{mdes_gls:.4f} dex/m "
    f"({10**(mdes_gls*full_range):.2f}x over {full_range:.1f} m, ~0.9 kyr).")
say(f"    the 95% CI upper bound on the slope is {gls['ci_hi']:+.6f} dex/m = "
    f"{10**(gls['ci_hi']*full_range):.2f}x, so a rise of up to "
    f"{100*(10**(gls['ci_hi']*full_range)-1):.0f}% across the core is NOT excluded.")
say(f"    a 2x (100%) increase would need {np.log10(2)/full_range:.5f} dex/m -> "
    f"power ~1.0; strongly excluded.")

# ==========================================================================
say("")
say("=" * 78)
say("PART 4 -- sensitivity: c06 abscissa, and the 15 dated horizons only")
say("=" * 78)
say("(a) water-equivalent depth c06 instead of c01")
r_w = corr(y, w)
gls_w = gls_slope(w, y)
sl_w, ic_w, rr_w, pp_w, se_w = stats.linregress(w, y)
say(f"    r = {r_w:+.6f} (R2={r_w**2:.6f}) ; OLS slope = {sl_w:+.6f} dex/m")
say(f"    rho_MLE = {gls_w['rho_mle']:.4f} ; AR(1)-GLS slope = {gls_w['slope']:+.6f} "
    f"dex/m, SE = {gls_w['se']:.6f}")
say(f"    95% CI [{gls_w['ci_lo']:+.6f}, {gls_w['ci_hi']:+.6f}], p = {gls_w['p']:.3f} "
    f"({'spans' if gls_w['ci_lo'] < 0 < gls_w['ci_hi'] else 'EXCLUDES'} zero)")
wr = float(w.max() - w.min())
mdes_w = mdes_from_se(gls_w["se"], dfree=dfree)
say(f"    MDES(80%, AR1-GLS) = {mdes_w:.6f} dex per m water-equivalent; range "
    f"{wr:.2f} m w.e. -> {10**(mdes_w*wr):.2f}x")
out["c06"] = {"r": r_w, "gls": gls_w, "mdes": float(mdes_w)}
say(f"    NOTE: c06 is cm-rounded (19 pairs of exactly-equal consecutive values); "
    f"c01 is finer-grained.")

say("(b) 15 dated horizons only")
ann_all = ~np.isnan(age)
say(f"    dated rows in the full table: {int(ann_all.sum())} (all retained, none is "
    f"the sulfate-NaN row) -> values below are on exactly those rows")
ya = y_raw[ann_all]
xa = depth_all[ann_all]
wa = we[ann_all]
say(f"    n = {int(ann_all.sum())} ; depth span {xa.min():.2f}-{xa.max():.2f} m ; "
    f"age span {age[ann_all].max():.0f}-{age[ann_all].min():.0f} CE")
say(f"    mean log10 sulfate on dated horizons = {ya.mean():.4f} dex vs "
    f"{y.mean():.4f} dex for the full record (+{ya.mean()-y.mean():.3f} dex "
    f"SELECTION SHIFT -- dated horizons are sulfate-enriched)")
ra = corr(ya, xa)
la = stats.linregress(xa, ya)
ra_w = corr(ya, wa)
spa = stats.spearmanr(xa, ya)
say(f"    r(log10 c04, c01) = {ra:+.4f} ; OLS slope {la.slope:+.6f} dex/m, naive SE "
    f"{la.stderr:.6f}, p = {la.pvalue:.3f}")
say(f"    r(log10 c04, c06) = {ra_w:+.4f}")
say(f"    Spearman rho = {spa.statistic:+.3f} (p={spa.pvalue:.3f})")
ac1a = float(np.corrcoef(ya[:-1], ya[1:])[0, 1])
say(f"    lag-1 autocorrelation within the 15 dated values = {ac1a:+.3f}")
mdes_a = mdes_from_se(la.stderr, dfree=13)
say(f"    MDES(80% power) with n=15 naive = {mdes_a:.6f} dex/m "
    f"({10**(mdes_a*(xa.max()-xa.min())):.2f}x over the dated span)")
say(f"    -> the dated subsample is ~{mdes_a/mdes_gls:.1f}x less sensitive than the "
    f"full record; it cannot resolve anything below {mdes_a:.4f} dex/m")
out["dated15"] = {"n": int(ann_all.sum()), "r_c01": float(ra), "r_c06": float(ra_w),
                  "slope": float(la.slope), "se": float(la.stderr),
                  "p": float(la.pvalue), "mdes": float(mdes_a),
                  "mean_log_shift": float(ya.mean() - y.mean())}

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
ax = axs[0, 0]
ax.plot(x, y, ".", ms=2, alpha=0.3, color="steelblue", label="samples")
nb = 20
edges = np.linspace(x.min(), x.max(), nb + 1)
bi = np.digitize(x, edges) - 1
bm, bc = [], []
for b in range(nb):
    m = bi == b
    if m.sum() > 3:
        bm.append(y[m].mean())
        bc.append(x[m].mean())
ax.plot(bc, bm, "o-", color="darkorange", ms=5, lw=1.5, label="20-bin means")
ax.plot(x, icept + slope_naive * x, "r-", lw=1.5,
        label=f"OLS slope {slope_naive:+.4f} dex/m")
ax.set_xlabel("c01 depth (m)")
ax.set_ylabel("log10 sulfate (ug/kg)")
ax.set_title(f"r = {r_obs:+.4f}, R2 = {r_obs**2:.4f}")
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.bar(np.arange(1, 61), acfs, color="grey", width=0.8)
ax.axhline(1.96 / np.sqrt(n), ls="--", color="crimson", lw=1)
ax.axhline(-1.96 / np.sqrt(n), ls="--", color="crimson", lw=1)
ax.plot(np.arange(1, 61), rho_hat ** np.arange(1, 61), "b-", lw=1.5,
        label=f"AR(1) rho={rho_hat:.3f}")
ax.set_xlabel("lag (samples)")
ax.set_ylabel("autocorrelation")
ax.set_title(f"ACF of log sulfate (lag-1 = {ac1_raw:.3f})")
ax.legend(fontsize=8)

ax = axs[1, 0]
dist = sb_corr(y, x, 100, 4000, np.random.default_rng(SEED + 7))
ax.hist(dist, bins=50, color="steelblue", alpha=0.8)
lo, hi = np.percentile(dist, [2.5, 97.5])
ax.axvline(r_obs, color="crimson", lw=2, label=f"observed {r_obs:+.4f}")
ax.axvline(0, color="k", lw=1)
ax.axvline(lo, color="darkorange", ls="--", lw=1.5)
ax.axvline(hi, color="darkorange", ls="--", lw=1.5,
           label=f"95% SB CI [{lo:+.3f}, {hi:+.3f}]")
ax.set_xlabel("r under stationary bootstrap (mean block 100)")
ax.set_ylabel("count")
ax.set_title("stationary bootstrap: interval spans zero")
ax.legend(fontsize=8)

ax = axs[1, 1]
ax.plot(grid_b, pow_gls, "o-", label="AR(1)-GLS test (valid)")
ax.plot(grid_b, pow_ols, "s--", label="naive OLS test (invalid)")
ax.axhline(0.8, color="grey", ls=":", lw=1)
ax.axvline(mdes_gls, color="crimson", lw=1.5, label=f"MDES GLS {mdes_gls:.5f} dex/m")
ax.axvline(abs(slope_naive), color="k", lw=1.5, ls="-.",
           label=f"observed |slope| {abs(slope_naive):.5f}")
ax.set_xlabel("true slope (dex/m)")
ax.set_ylabel("power (alpha=0.05)")
ax.set_title(f"power: MDES(80%) = {mdes_gls:.4f} dex/m "
             f"({10**(mdes_gls*full_range):.2f}x over {full_range:.0f} m)")
ax.legend(fontsize=8)
fig.tight_layout()
fig_path = HERE / "verify_draft3_sulfate_trend.png"
fig.savefig(fig_path, dpi=130)
say("")
say(f"figure written: {fig_path}")

# ------------------------------------------------------------------ outputs
with open(HERE / "verify_draft3_sulfate_trend.json", "w") as f:
    json.dump(out, f, indent=2)
with open(HERE / "verify_draft3_sulfate_trend_log.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
pd.DataFrame(sb_rows).to_csv(HERE / "sb_corr_intervals.csv", index=False)
pd.DataFrame(sb_rows_c).to_csv(HERE / "sb_contrast_intervals.csv", index=False)
pd.DataFrame(bp_rows).to_csv(HERE / "block_perm_pvalues.csv", index=False)
pd.DataFrame({"slope_dex_per_m": grid_b, "power_gls": pow_gls,
              "power_naive_ols": pow_ols}).to_csv(HERE / "power_curve.csv", index=False)
print("done")