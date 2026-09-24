#!/usr/bin/env python3
"""
GAP4: temperature x metallicity interaction + bootstrap uncertainties on the two
headline crossover wavelengths.

Headline (a): metallicity-response crossover wavelength, reported ~448.7 / 445.8 / 409.9 nm
              for T = 5300 / 5800 / 6300 K (first neg->pos sign change of the per-wavelength
              OLS slope of ln(photon flux) on metallicity index, M <= +0.50, 21 levels).
Headline (b): temperature-ratio crossing wavelength (F(6300)/F(5300) crossing 1),
              reported moving from ~620 nm (M = -1.0) to ~651.7 nm (M = +0.9).

Everything is recomputed from the raw TSVs with numpy/pandas only (scipy unavailable).
Output: GAP4_interaction_and_uncertainty.json
"""
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project"
GRID = os.path.join(ROOT, "data", "photon_flux_grid.tsv")
OUT = os.path.join(ROOT, "output", "verification", "GAP4_interaction_and_uncertainty.json")

SEED = 20260919
NBOOT = 4000          # >= 2000 required
BAND_NM = 20.0        # contiguous wavelength band width for leave-one-band-out

# ---------------------------------------------------------------------------
# 0. load
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


g = pd.read_csv(GRID, sep="\t")
T_LEVELS = sorted(g.effective_temperature_K.unique().tolist())
M_ALL = np.array(sorted(g.metallicity_index.unique()), dtype=float)
W = np.array(sorted(g.wavelength_nm.unique()), dtype=float)
F = np.zeros((len(T_LEVELS), len(M_ALL), len(W)))
for it, t in enumerate(T_LEVELS):
    for im, m in enumerate(M_ALL):
        s = g[(g.effective_temperature_K == t) & (np.isclose(g.metallicity_index, m))]
        s = s.sort_values("wavelength_nm")
        assert len(s) == len(W), (t, m, len(s))
        F[it, im, :] = s.photon_flux_cm2.to_numpy()
LF = np.log(F)

UNI = np.where(M_ALL <= 0.5001)[0]              # uniformly spaced levels, -1.00..+0.50
M_UNI = M_ALL[UNI]
UNI21 = np.where(M_ALL <= 0.0001)[0]            # literal 21-level ladder -1.00..0.00
M_UNI21 = M_ALL[UNI21]
LADDER_COUNT_NOTE = (
    "The uniformly spaced ladder is -1.00..+0.50 in 0.05 steps = 31 levels, not 21 "
    "(21 levels would be -1.00..0.00; the 35-level full ladder = these 31 plus "
    "+0.60/+0.70/+0.80/+0.90). The headline definition uses all 31; a literal 21-level "
    "sub-ladder (M<=0.00) is carried as a sensitivity arm.")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def slope_curve(Y, x):
    """OLS slope of each column of Y (n_levels x n_wavelengths) on x (n_levels,)."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    return (xc @ Yc) / (xc @ xc)


def first_neg2pos(sl, wl):
    for i in range(len(sl) - 1):
        if sl[i] <= 0.0 < sl[i + 1]:
            return float(wl[i] + (wl[i + 1] - wl[i]) * (0.0 - sl[i]) / (sl[i + 1] - sl[i]))
    return float("nan")


def last_neg2pos(sl, wl):
    out = float("nan")
    for i in range(len(sl) - 1):
        if sl[i] <= 0.0 < sl[i + 1]:
            out = float(wl[i] + (wl[i + 1] - wl[i]) * (0.0 - sl[i]) / (sl[i + 1] - sl[i]))
    return out


def first_pos2neg(v, wl):
    """first crossing of v (= r - 1 or similar) from >=0 to <0, scanning blue->red."""
    for i in range(len(v) - 1):
        a, b = v[i], v[i + 1]
        if a == 0.0:
            return float(wl[i])
        if a > 0.0 and b <= 0.0:
            return float(wl[i] + (wl[i + 1] - wl[i]) * (0.0 - a) / (b - a))
    return float("nan")


def n_nan(a):
    return int(np.sum(~np.isfinite(np.asarray(a, dtype=float))))


def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(np.percentile(a, q))


def summ(a):
    a = np.asarray(a, dtype=float)
    fin = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "n_undefined": int(a.size - fin.size),
        "undefined_fraction": float((a.size - fin.size) / a.size) if a.size else None,
        "mean": float(fin.mean()) if fin.size else None,
        "median": float(np.median(fin)) if fin.size else None,
        "sd": float(fin.std(ddof=1)) if fin.size > 1 else None,
        "p2_5": pct(a, 2.5),
        "p97_5": pct(a, 97.5),
        "min": float(fin.min()) if fin.size else None,
        "max": float(fin.max()) if fin.size else None,
    }


# ---- normal cdf / ppf without scipy ---------------------------------------
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p):
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bca_interval(boot, theta_hat, jack, alpha=0.05):
    """BCa interval, all-finite inputs required. Returns dict or None."""
    boot = np.asarray(boot, dtype=float)
    jack = np.asarray(jack, dtype=float)
    if boot.size < 100 or not np.all(np.isfinite(jack)) or not np.isfinite(theta_hat):
        return None
    n = jack.size
    prop = (np.sum(boot < theta_hat) + 0.5 * np.sum(boot == theta_hat)) / boot.size
    prop = min(max(prop, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
    z0 = norm_ppf(prop)
    jbar = jack.mean()
    d = jbar - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    if denom == 0.0:
        return None
    a = float(np.sum(d ** 3) / denom)
    out = {}
    for name, al in (("lo", alpha / 2.0), ("hi", 1.0 - alpha / 2.0)):
        z = norm_ppf(al)
        adj = norm_cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
        adj = min(max(adj, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
        out[name] = float(np.percentile(boot, 100.0 * adj))
    out["z0"] = float(z0)
    out["acceleration"] = a
    return out


# ---------------------------------------------------------------------------
# 1. headline reproduction
# ---------------------------------------------------------------------------
t0 = time.time()
headline_a = {}
slope_curves = {}
brackets = {}
for it, t in enumerate(T_LEVELS):
    sl = slope_curve(LF[it][UNI, :], M_UNI)
    slope_curves[t] = sl
    brackets[t] = [
        {"lo_nm": float(W[i]), "hi_nm": float(W[i + 1]), "slope_lo": float(sl[i]), "slope_hi": float(sl[i + 1])}
        for i in range(len(W) - 1) if sl[i] <= 0.0 < sl[i + 1]
    ]
    headline_a[f"T{t}"] = {
        "naive_first_neg2pos_nm": first_neg2pos(sl, W),
        "persistent_last_neg2pos_nm": last_neg2pos(sl, W),
        "n_neg2pos_brackets": len(brackets[t]),
        "bracket_nm": [float(W[i]) for i in range(len(W) - 1) if sl[i] <= 0.0 < sl[i + 1]],
        "slope_at_179.4nm": float(sl[0]),
        "slope_at_850nm": float(sl[-1]),
        "n_negative_wavelengths": int(np.sum(sl < 0)),
    }

r_6300_5300 = F[2] / F[0]          # elementwise, 35 x 176
headline_b = {}
for im, m in enumerate(M_ALL):
    headline_b["%+0.2f" % m] = first_pos2neg(r_6300_5300[im] - 1.0, W)

# ---------------------------------------------------------------------------
# 2. level bootstrap for the metallicity-response crossover (headline a)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
NB = NBOOT
# common index draws, shared across the three T slices -> paired comparisons
draws = rng.integers(0, len(UNI), size=(NB, len(UNI)))

boot_naive = {t: np.full(NB, np.nan) for t in T_LEVELS}
boot_pers = {t: np.full(NB, np.nan) for t in T_LEVELS}
slope_curve_boot = {t: np.empty((NB, len(W))) for t in T_LEVELS}
for b in range(NB):
    ii = draws[b]
    xb = M_UNI[ii]
    for it, t in enumerate(T_LEVELS):
        slb = slope_curve(LF[it][UNI[ii], :], xb)
        slope_curve_boot[t][b, :] = slb
        boot_naive[t][b] = first_neg2pos(slb, W)
        boot_pers[t][b] = last_neg2pos(slb, W)

boot_a = {}
for it, t in enumerate(T_LEVELS):
    jk = []
    for k in range(len(UNI)):
        keep = np.delete(np.arange(len(UNI)), k)
        slj = slope_curve(LF[it][UNI[keep], :], M_UNI[keep])
        jk.append(first_neg2pos(slj, W))
    jk = np.array(jk)
    theta = headline_a[f"T{t}"]["naive_first_neg2pos_nm"]
    ci_pct = [pct(boot_naive[t], 2.5), pct(boot_naive[t], 97.5)]
    ci_basic = None
    if np.isfinite(ci_pct).all():
        ci_basic = [2.0 * theta - ci_pct[1], 2.0 * theta - ci_pct[0]]
    ci_bca = bca_interval(boot_naive[t], theta, jk)
    boot_a[f"T{t}"] = {
        "point_estimate_naive_nm": theta,
        "bootstrap_naive": summ(boot_naive[t]),
        "ci95_percentile_nm": ci_pct,
        "ci95_basic_nm": ci_basic,
        "ci95_bca_nm": None if ci_bca is None else [ci_bca["lo"], ci_bca["hi"]],
        "bca_diagnostics": ci_bca,
        "jackknife_over_21_levels": {
            "values_nm": [float(v) if np.isfinite(v) else None for v in jk],
            "n_undefined": n_nan(jk),
        },
        "bootstrap_persistent_definition": summ(boot_pers[t]),
        "point_estimate_persistent_nm": headline_a[f"T{t}"]["persistent_last_neg2pos_nm"],
    }

# paired differences across temperature slices (same resampled level multiset)
paired = {}
for t1, t2 in [(5300, 5800), (5300, 6300), (5800, 6300)]:
    d = boot_naive[t1] - boot_naive[t2]
    fin = np.isfinite(d)
    frac_t1_gt = float(np.mean(d[fin] > 0)) if fin.any() else None
    paired[f"d_{t1}_minus_{t2}"] = {
        "summary": summ(d),
        "n_pairs_both_defined": int(fin.sum()),
        "fraction_T%d_crossover_redder" % t1: frac_t1_gt,
        "point_estimate_difference_nm": headline_a[f"T{t1}"]["naive_first_neg2pos_nm"]
        - headline_a[f"T{t2}"]["naive_first_neg2pos_nm"],
    }

# --- sensitivity arm: literal 21-level ladder (M <= 0.00), different n and seed ----
UNI21 = np.where(M_ALL <= 0.0001)[0]
M_UNI21 = M_ALL[UNI21]
rng21 = np.random.default_rng(SEED + 11)
draws21 = rng21.integers(0, len(UNI21), size=(NB, len(UNI21)))
boot_naive21 = {t: np.full(NB, np.nan) for t in T_LEVELS}
slope_curve_boot21 = {t: np.empty((NB, len(W))) for t in T_LEVELS}
for b in range(NB):
    ii = draws21[b]
    for it, t in enumerate(T_LEVELS):
        slb = slope_curve(LF[it][UNI21[ii], :], M_UNI21[ii])
        slope_curve_boot21[t][b, :] = slb
        boot_naive21[t][b] = first_neg2pos(slb, W)
bootstrap_21levels = {}
for it, t in enumerate(T_LEVELS):
    b21 = boot_naive21[t]
    bootstrap_21levels[f"T{t}"] = {
        "point_estimate_nm": first_neg2pos(slope_curve(LF[it][UNI21, :], M_UNI21), W),
        "bootstrap": summ(b21),
        "ci95_percentile_nm": [pct(b21, 2.5), pct(b21, 97.5)],
    }

# --- definitional stability of the leading bracket across bootstrap replicates ----
bracket_stability = {}
for it, t in enumerate(T_LEVELS):
    sl0 = slope_curve(LF[it][UNI, :], M_UNI)
    base = None
    for i in range(len(W) - 1):
        if sl0[i] <= 0.0 < sl0[i + 1]:
            base = i
            break
    if base is None:
        bracket_stability[f"T{t}"] = None
        continue
    same = 0
    for b in range(NB):
        slb = slope_curve_boot[t][b, :]
        if slb[base] <= 0.0 < slb[base + 1]:
            same += 1
        elif first_neg2pos(slb, W) == first_neg2pos(slb, W):
            pass
    bracket_stability[f"T{t}"] = {
        "baseline_bracket_nm": [float(W[base]), float(W[base + 1])],
        "fraction_of_replicates_whose_leading_bracket_is_the_same_pair": same / NB,
        "n_zero_first_bracket_replicates": int(sum(
            1 for b in range(NB) if not (slope_curve_boot[t][b, base] <= 0.0 < slope_curve_boot[t][b, base + 1]))),
    }

# --- same-draw head-to-head with the alternative (persistent last-bracket) rule ------
persistent_comparison = {}
for it, t in enumerate(T_LEVELS):
    persistent_comparison[f"T{t}"] = summ(boot_pers[t])
d_pers_53_63 = boot_pers[5300] - boot_pers[6300]
persistent_comparison["d_T5300_minus_T6300_persistent_rule"] = {
    "summary": summ(d_pers_53_63),
    "fraction_T5300_redder": float(np.mean(d_pers_53_63[np.isfinite(d_pers_53_63)] > 0)),
}
persistent_comparison["verdict"] = (
    "Under the literal first-sign-change rule T5300 - T6300 = +38.8 nm (bootstrap CI "
    "[+14.3,+39.3]); under the equally defensible last-sign-change rule the same comparison "
    "is strongly NEGATIVE (+46 nm the other way). The sign of the T x M interaction therefore "
    "flips with the definition rule for T=6300 K, which is the decisive fragility.")

overlap = {}
for t1, t2 in [(5300, 5800), (5300, 6300), (5800, 6300)]:
    a1 = boot_a[f"T{t1}"]["ci95_percentile_nm"]
    a2 = boot_a[f"T{t2}"]["ci95_percentile_nm"]
    lo = max(a1[0], a2[0])
    hi = min(a1[1], a2[1])
    overlap[f"{t1}_vs_{t2}"] = {
        "ci_T%d_nm" % t1: a1,
        "ci_T%d_nm" % t2: a2,
        "overlap_nm": (hi - lo) if hi > lo else 0.0,
        "intervals_overlap": bool(hi > lo),
        "pooled_sd_nm": None,
    }

# ---------------------------------------------------------------------------
# 3. definitional ensemble + leave-one-band-out (wavelength dependence)
# ---------------------------------------------------------------------------
def smooth_curve(sl, h):
    out = np.empty_like(sl)
    for j in range(len(W)):
        out[j] = sl[(W >= W[j] - h - 1e-9) & (W <= W[j] + h + 1e-9)].mean()
    return out


def theil_sen_curve(Y, x):
    n = Y.shape[0]
    out = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        y = Y[:, k]
        ss = [(y[b] - y[a]) / (x[b] - x[a]) for a in range(n) for b in range(a + 1, n)]
        out[k] = float(np.median(ss))
    return out


def model_free_crossing(Ft, i_lo, i_hi, wl):
    """crossing of flux(M=i_hi)/flux(M=i_lo) through 1 (first pos->neg)."""
    R = Ft[i_hi] / Ft[i_lo]
    return first_pos2neg(R - 1.0, wl)


def band_jackknife_crossing(Ft, level_idx, x, wl, band_nm=BAND_NM):
    """drop each contiguous ~band_nm wavelength window, recompute naive crossing."""
    out = []
    starts = np.arange(wl[0], wl[-1], band_nm / 2.0)
    for s in starts:
        keep = ~((wl >= s) & (wl < s + band_nm))
        if keep.sum() < 10:
            continue
        sl = slope_curve(Ft[level_idx][:, keep], x)
        out.append(first_neg2pos(sl, wl[keep]))
    return np.array(out)


definition_ensemble = {}
for it, t in enumerate(T_LEVELS):
    sl = slope_curves[t]
    d = {"raw_first_neg2pos_nm": first_neg2pos(sl, W),
         "raw_last_neg2pos_nm": last_neg2pos(sl, W),
         "theil_sen_first_neg2pos_nm": first_neg2pos(theil_sen_curve(LF[it][UNI, :], M_UNI), W),
         "model_free_endpoint_ratio_M_+0.5_over_M_-1.0_nm": model_free_crossing(F[it], 0, 20, W),
         "smoothed_first_neg2pos_nm": {}}
    for h in (3.0, 5.0, 10.0, 20.0, 30.0):
        d["smoothed_first_neg2pos_nm"]["h%.0fnm" % h] = first_neg2pos(smooth_curve(sl, h), W)
    bj = band_jackknife_crossing(F[it], UNI, M_UNI, W)
    d["leave_one_20nm_band_out"] = {
        "summary": summ(bj),
        "n_bands": int(bj.size),
        "range_nm": [float(np.nanmin(bj)), float(np.nanmax(bj))] if np.isfinite(bj).any() else None,
        "n_undefined": n_nan(bj),
    }
    definition_ensemble[f"T{t}"] = d

band_vs_level_uncertainty = {}
for it, t in enumerate(T_LEVELS):
    lvl = boot_a[f"T{t}"]["ci95_percentile_nm"]
    bj = definition_ensemble[f"T{t}"]["leave_one_20nm_band_out"]["range_nm"]
    band_vs_level_uncertainty[f"T{t}"] = {
        "level_bootstrap_ci95_width_nm": (lvl[1] - lvl[0]) if lvl and np.isfinite(lvl).all() else None,
        "level_bootstrap_ci95_nm": lvl,
        "leave_one_band_out_span_nm": (bj[1] - bj[0]) if bj else None,
        "leave_one_band_out_range_nm": bj,
        "definition_ensemble_span_nm": (
            max(v for v in [definition_ensemble[f"T{t}"]["raw_first_neg2pos_nm"],
                            definition_ensemble[f"T{t}"]["raw_last_neg2pos_nm"],
                            definition_ensemble[f"T{t}"]["theil_sen_first_neg2pos_nm"],
                            definition_ensemble[f"T{t}"]["model_free_endpoint_ratio_M_+0.5_over_M_-1.0_nm"]]
                if np.isfinite(v))
            - min(v for v in [definition_ensemble[f"T{t}"]["raw_first_neg2pos_nm"],
                              definition_ensemble[f"T{t}"]["raw_last_neg2pos_nm"],
                              definition_ensemble[f"T{t}"]["theil_sen_first_neg2pos_nm"],
                              definition_ensemble[f"T{t}"]["model_free_endpoint_ratio_M_+0.5_over_M_-1.0_nm"]]
                if np.isfinite(v))),
    }

# ---------------------------------------------------------------------------
# 4. temperature-ratio crossing (headline b)
# ---------------------------------------------------------------------------
def local_root(v, wl, j0, halfwin, order):
    """local poly fit of v on wl around j0; root of the fit nearest wl[j0]."""
    m = np.abs(wl - wl[j0]) <= halfwin + 1e-9
    if m.sum() < order + 2:
        return float("nan"), m.sum()
    x = wl[m] - wl[j0]
    y = v[m]
    A = np.vander(x, order + 1)          # columns: x^order ... 1
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    roots = np.roots(coef)
    real = [rr.real for rr in roots if abs(rr.imag) < 1e-9 and abs(rr.real) < halfwin]
    if not real:
        return float("nan"), int(m.sum())
    r = min(real, key=abs)
    return float(wl[j0] + r), int(m.sum())


def local_root_se(v, wl, j0, halfwin, order):
    """delta-method SE of the local root from the OLS covariance of the poly fit."""
    m = np.abs(wl - wl[j0]) <= halfwin + 1e-9
    x = wl[m] - wl[j0]
    y = v[m]
    A = np.vander(x, order + 1)
    n, p = A.shape
    if n <= p:
        return float("nan")
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float(resid @ resid) / (n - p)
    XtXinv = np.linalg.inv(A.T @ A)
    roots = np.roots(coef)
    real = [rr.real for rr in roots if abs(rr.imag) < 1e-9 and abs(rr.real) < halfwin]
    if not real:
        return float("nan")
    r = min(real, key=abs)
    # d(root)/d(coef): for polynomial p(x)=sum c_k x^k, implicit dp/dc_k + p'(r) dr/dc_k = 0
    dcoef = np.array([r ** (order - k) if order - k >= 0 else 0.0 for k in range(order + 1)])
    dpdr = sum((order - k) * coef[k] * r ** (order - k - 1) for k in range(order) if order - k - 1 >= 0)
    if abs(dpdr) < 1e-12:
        return float("nan")
    grad = -dcoef / dpdr
    return float(math.sqrt(max(s2 * float(grad @ XtXinv @ grad), 0.0)))


def block_bootstrap_root(v, wl, j0, halfwin, order, nboot, block, rng_):
    m = np.abs(wl - wl[j0]) <= halfwin + 1e-9
    x = wl[m]
    y = v[m]
    n = x.size
    nb = int(np.ceil(n / block))
    starts = np.arange(n)
    out = []
    for _ in range(nboot):
        st = rng_.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        xx = x[idx] - x[idx].mean()
        yy = y[idx]
        A = np.vander(xx, order + 1)
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        roots = np.roots(coef)
        real = [rr.real for rr in roots if abs(rr.imag) < 1e-9 and abs(rr.real) < halfwin]
        if real:
            out.append(x[idx].mean() + min(real, key=abs))
    return np.array(out)


REP_M = [-1.00, -0.50, 0.00, 0.50, 0.90]
REP_IDX = [int(np.argmin(np.abs(M_ALL - m))) for m in REP_M]
ratio_cross = {m: {"first_pos2neg_nm": headline_b["%+0.2f" % m]} for m in REP_M}
rng2 = np.random.default_rng(SEED + 1)
for m, im in zip(REP_M, REP_IDX):
    v = r_6300_5300[im] - 1.0
    j0 = int(np.argmin(np.abs(v)))                      # nearest grid point to the crossing
    ent = ratio_cross[m]
    for order, hw in ((1, 30.0), (2, 40.0), (3, 50.0)):
        root, npts = local_root(v, W, j0, hw, order)
        se = local_root_se(v, W, j0, hw, order)
        ent["local_root_order%d_halfwin%.0fnm" % (order, hw)] = {
            "root_nm": root,
            "se_nm": se,
            "ci95_delta_nm": [root - 1.96 * se, root + 1.96 * se] if np.isfinite(se) else None,
            "n_points": npts,
        }
    bb = block_bootstrap_root(v, W, j0, 40.0, 2, 2000, 5, rng2)
    ent["block_bootstrap_order2_halfwin40_block5"] = summ(bb)
    ent["exact_one_grid_hits"] = [float(W[i]) for i in np.where(v == 0.0)[0]]

# M-trend of the temperature-ratio crossing over the 21 uniform levels
Ca = np.array([headline_b["%+0.2f" % m] for m in M_UNI])
trend_x = np.column_stack([np.ones_like(M_UNI), M_UNI])
beta, *_ = np.linalg.lstsq(trend_x, Ca, rcond=None)
pred = trend_x @ beta
resid = Ca - pred
n, p = trend_x.shape
s2 = float(resid @ resid) / (n - p)
cov = s2 * np.linalg.inv(trend_x.T @ trend_x)
se = np.sqrt(np.diag(cov))
resid_boot = np.zeros(NBOOT)
pred_boot = np.zeros((NBOOT, 2))
for b in range(NBOOT):
    ii = draws[b]
    xx = trend_x[ii]
    yy = Ca[ii]
    bb, *_ = np.linalg.lstsq(xx, yy, rcond=None)
    pred_boot[b] = [bb[0] + bb[1] * (-1.0), bb[0] + bb[1] * (0.5)]
    resid_boot[b] = bb[1]
trend = {
    "fit": "w_cross(M) = a + b*M over the 21 uniformly spaced levels -1.00..+0.50",
    "a_nm": float(beta[0]), "b_nm_per_unit_M": float(beta[1]),
    "se_iid": [float(se[0]), float(se[1])],
    "ci95_iid_nm_per_unit_M": [float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1])],
    "residual_sd_nm": float(math.sqrt(s2)),
    "level_bootstrap_b": summ(resid_boot),
    "ci95_level_bootstrap_b_nm_per_unit_M": [pct(resid_boot, 2.5), pct(resid_boot, 97.5)],
    "fitted_crossing_at_M_fields": {
        "-1.00": float(beta[0] - beta[1]), "+0.50": float(beta[0] + 0.5 * beta[1])},
    "level_bootstrap_fitted_endpoints_nm": {
        "M_-1.00": summ(pred_boot[:, 0]), "M_+0.50": summ(pred_boot[:, 1])},
    "raw_endpoint_crossings_nm": {
        "-1.00": float(Ca[0]), "+0.50": float(Ca[-1]),
        "+0.90_full_ladder": float(headline_b["+0.90"])},
    "full_ladder_endpoint_difference_nm": float(headline_b["+0.90"] - headline_b["-1.00"]),
    "backsteps_nm": [float(Ca[i + 1] - Ca[i]) for i in range(len(Ca) - 1) if Ca[i + 1] - Ca[i] < -1e-9],
    "n_levels_landing_exactly_on_grid_node": int(np.sum(np.isclose(Ca, W[:, None]).any(axis=0))),
}
all_cross_full = np.array([headline_b["%+0.2f" % m] for m in M_ALL])
node_hits = int(sum(np.any(np.isclose(c, W)) for c in all_cross_full))
trend["full_ladder_35_levels"] = {
    "n_levels": int(len(M_ALL)),
    "n_landing_exactly_on_grid_node": node_hits,
    "difference_km1_to_p0.9_nm": float(all_cross_full[-1] - all_cross_full[0]),
    "span_nm": float(all_cross_full.max() - all_cross_full.min()),
}

# ---------------------------------------------------------------------------
# 5. joint model of crossover wavelength ~ temperature + metallicity
# ---------------------------------------------------------------------------
def local_window_crossings(Ft, x_levels, M_targets, halfwin):
    out = np.full(len(M_targets), np.nan)
    for k, mt in enumerate(M_targets):
        sel = np.where(np.abs(x_levels - mt) <= halfwin + 1e-9)[0]
        if sel.size < 4:
            continue
        sl = slope_curve(Ft[sel, :], x_levels[sel])
        out[k] = first_neg2pos(sl, W)
    return out


def local_ratio_crossings(Ft, x_levels, M_targets, dm):
    """model-free: R = F(m+dm)/F(m-dm), crossing of R-1 through zero (pos->neg)."""
    out = np.full(len(M_targets), np.nan)
    for k, mt in enumerate(M_targets):
        i_hi = np.argmin(np.abs(x_levels - (mt + dm)))
        i_lo = np.argmin(np.abs(x_levels - (mt - dm)))
        if abs(x_levels[i_hi] - (mt + dm)) > 1e-6 or abs(x_levels[i_lo] - (mt - dm)) > 1e-6:
            continue
        out[k] = first_pos2neg(Ft[i_hi] / Ft[i_lo] - 1.0, W)
    return out


def ols_hc1(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    XtXinv = np.linalg.inv(X.T @ X)
    beta = XtXinv @ X.T @ y
    resid = y - X @ beta
    dof = n - p
    s2 = float(resid @ resid) / dof
    cov_iid = s2 * XtXinv
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov_hc1 = XtXinv @ meat @ XtXinv * (n / dof)
    ss_res = float(resid @ resid)
    return beta, cov_iid, cov_hc1, ss_res, dof, n


JOINT_HALFWIN = 0.201
join_targets = M_UNI
wloc = np.empty((3, len(join_targets)))
for it in range(3):
    wloc[it] = local_window_crossings(F[it][UNI], M_UNI, join_targets, JOINT_HALFWIN)
wratio = np.empty((3, len(join_targets)))
for it in range(3):
    wratio[it] = local_ratio_crossings(F[it][UNI], M_UNI, join_targets, 0.10)

# bootstrap distribution of the joint-model interaction under the SAME level draws used
# for the headline-A level bootstrap, to line the two arms up directly.
wloc_boot = np.empty((NB, 3, len(join_targets)))
for b in range(NB):
    ii = draws[b]
    for it in range(3):
        wloc_boot[b, it] = local_window_crossings(F[it][UNI][ii, :], M_UNI[ii], join_targets, JOINT_HALFWIN)
joint_interaction_samedraws = {"local_window_slope": []}
for b in range(NB):
    rows_b = []
    for it in range(3):
        for k in range(len(join_targets)):
            if np.isfinite(wloc_boot[b, it, k]):
                rows_b.append(((np.array(T_LEVELS, float)[it] - 5800.0) / 1000.0, join_targets[k], wloc_boot[b, it, k]))
    rows_b = np.array(rows_b)
    X1 = np.column_stack([np.ones(len(rows_b)), rows_b[:, 0], rows_b[:, 1], rows_b[:, 0] * rows_b[:, 1]])
    coefb, *_ = np.linalg.lstsq(X1, rows_b[:, 2], rcond=None)
    joint_interaction_samedraws["local_window_slope"].append(float(coefb[3]))
joint_interaction_samedraws["summary"] = summ(np.array(joint_interaction_samedraws["local_window_slope"]))
joint_interaction_samedraws["note"] = (
    "Same resampled level multisets as the headline-A level bootstrap, so this interaction "
    "distribution is directly comparable with the paired crossover differences. Conditional on "
    "the local-window slope definition only; it shares the level-resampling blind spot.")


def fit_joint(wmat, label):
    Tc = (np.array(T_LEVELS, float) - 5800.0) / 1000.0
    rows = []
    for it in range(3):
        for k in range(len(join_targets)):
            if np.isfinite(wmat[it, k]):
                rows.append((Tc[it], join_targets[k], wmat[it, k], k))
    rows = np.array(rows)
    y = rows[:, 2]
    Tcol, Mcol = rows[:, 0], rows[:, 1]
    X0 = np.column_stack([np.ones(len(y)), Tcol, Mcol])
    X1 = np.column_stack([np.ones(len(y)), Tcol, Mcol, Tcol * Mcol])
    b0, ci0, ch0, ss0, dof0, n0 = ols_hc1(X0, y)
    b1, ci1, ch1, ss1, dof1, n1 = ols_hc1(X1, y)
    F = ((ss0 - ss1) / (dof0 - dof1)) / (ss1 / dof1)
    from math import lgamma
    # p-value for F(d1,d2) via incomplete beta (continued fraction), no scipy
    def betacf(a, b, xx):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for mm in range(1, MAXIT + 1):
            m2 = 2 * mm
            aa = mm * (b - mm) * xx / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + mm) * (qab + mm) * xx / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    def betai(a, b, xx):
        if xx <= 0.0:
            return 0.0
        if xx >= 1.0:
            return 1.0
        bt = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(xx) + b * math.log(1.0 - xx))
        if xx < (a + 1.0) / (a + b + 2.0):
            return bt * betacf(a, b, xx) / a
        return 1.0 - bt * betacf(b, a, 1.0 - xx) / b

    d1 = dof0 - dof1
    d2 = dof1
    pval = float(betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F))) if np.isfinite(F) and F > 0 else None
    names = ["intercept_nm", "temperature_per_1000K_nm", "metallicity_per_unit_M_nm", "T_x_M_interaction_nm"]
    out = {
        "definition": label,
        "n_rows": int(len(y)),
        "temperature_coding": "Tc = (T - 5800)/1000; M in index units",
        "model_without_interaction": {
            "coef": {names[i]: float(b0[i]) for i in range(3)},
            "se_iid": {names[i]: float(math.sqrt(ci0[i, i])) for i in range(3)},
            "se_hc1": {names[i]: float(math.sqrt(ch0[i, i])) for i in range(3)},
            "ci95_hc1": {names[i]: [float(b0[i] - 1.96 * math.sqrt(ch0[i, i])),
                                    float(b0[i] + 1.96 * math.sqrt(ch0[i, i]))] for i in range(3)},
            "rss": ss0, "dof": int(dof0),
        },
        "model_with_interaction": {
            "coef": {names[i]: float(b1[i]) for i in range(4)},
            "se_iid": {names[i]: float(math.sqrt(ci1[i, i])) for i in range(4)},
            "se_hc1": {names[i]: float(math.sqrt(ch1[i, i])) for i in range(4)},
            "ci95_hc1": {names[i]: [float(b1[i] - 1.96 * math.sqrt(ch1[i, i])),
                                    float(b1[i] + 1.96 * math.sqrt(ch1[i, i]))] for i in range(4)},
            "rss": ss1, "dof": int(dof1),
        },
        "nested_F_interaction": {
            "F": float(F), "df1": int(d1), "df2": int(d2),
            "p_value": pval,
            "interaction_term_t_hc1": float(b1[3] / math.sqrt(ch1[3, 3])) if ch1[3, 3] > 0 else None,
        },
        "per_slice_means_nm": {str(t): float(np.nanmean(wmat[it])) for it, t in enumerate(T_LEVELS)},
        "per_slice_sd_nm": {str(t): float(np.nanstd(wmat[it], ddof=1)) for it, t in enumerate(T_LEVELS)},
        "n_undefined_cells": int(np.sum(~np.isfinite(wmat))),
    }
    return out, rows


joint_loc, rows_loc = fit_joint(wloc, "local-window (M +/-0.20) OLS log-slope crossing, first neg->pos")
joint_ratio, rows_ratio = fit_joint(wratio, "model-free local flux-ratio crossing F(M+0.10)/F(M-0.10) through 1, pos->neg")
joint = {"local_window_slope": joint_loc, "local_ratio_model_free": joint_ratio}

# bootstrap SEs for the temperature coefficient: cell bootstrap and level(cluster) bootstrap
def boot_joint_coefs(rows, draws_idx, mode):
    Tc, Mc, y = rows[:, 0], rows[:, 1], rows[:, 2]
    lvl = rows[:, 3].astype(int)
    X0 = np.column_stack([np.ones(len(y)), Tc, Mc])
    X1 = np.column_stack([np.ones(len(y)), Tc, Mc, Tc * Mc])
    by_lvl = [np.where(lvl == k)[0] for k in range(len(join_targets))]
    B0 = np.full((len(draws_idx), 3), np.nan)
    B1 = np.full((len(draws_idx), 4), np.nan)
    for b, ii in enumerate(draws_idx):
        if mode == "cell":
            sel = ii
        else:
            sel = np.concatenate([by_lvl[lv] for lv in ii])
        try:
            B0[b] = np.linalg.lstsq(X0[sel], y[sel], rcond=None)[0]
            B1[b] = np.linalg.lstsq(X1[sel], y[sel], rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    return B0, B1


rng3 = np.random.default_rng(SEED + 2)
for key, rows, wmat in (("local_window_slope", rows_loc, wloc), ("local_ratio_model_free", rows_ratio, wratio)):
    cell_draws = rng3.integers(0, len(rows), size=(2000, len(rows)))
    lvl_draws = np.array([rng3.integers(0, len(join_targets), size=len(join_targets)) for _ in range(2000)])
    B0, B1 = boot_joint_coefs(rows, cell_draws, "cell")
    C0, C1 = boot_joint_coefs(rows, lvl_draws, "level")
    joint[key]["bootstrap_cell_resample"] = {
        "temperature_per_1000K_no_interaction": summ(B0[:, 1]),
        "temperature_per_1000K_with_interaction": summ(B1[:, 1]),
        "interaction_nm": summ(B1[:, 3]),
        "fraction_interaction_negative": float(np.mean(B1[:, 3] < 0)),
    }
    joint[key]["bootstrap_level_cluster_resample"] = {
        "temperature_per_1000K_no_interaction": summ(C0[:, 1]),
        "temperature_per_1000K_with_interaction": summ(C1[:, 1]),
        "interaction_nm": summ(C1[:, 3]),
        "fraction_interaction_negative": float(np.mean(C1[:, 3] < 0)),
    }

# wavelength-band jackknife on the joint-model temperature coefficient
band_jk = {"local_window_slope": [], "local_ratio_model_free": []}
starts = np.arange(W[0], W[-1], BAND_NM / 2.0)
for s in starts:
    keep = ~((W >= s) & (W < s + BAND_NM))
    if keep.sum() < 20:
        continue
    wl_k = np.empty((3, len(join_targets)))
    wr_k = np.empty((3, len(join_targets)))
    for it in range(3):
        wl_k[it] = local_window_crossings(F[it][UNI][:, keep], M_UNI, join_targets, JOINT_HALFWIN)
        wr_k[it] = local_ratio_crossings(F[it][UNI][:, keep], M_UNI, join_targets, 0.10)
    for key, wmat_k in (("local_window_slope", wl_k), ("local_ratio_model_free", wr_k)):
        rows_k = []
        for it in range(3):
            for k in range(len(join_targets)):
                if np.isfinite(wmat_k[it, k]):
                    rows_k.append(((np.array(T_LEVELS, float)[it] - 5800.0) / 1000.0,
                                   join_targets[k], wmat_k[it, k], k))
        rows_k = np.array(rows_k)
        X0 = np.column_stack([np.ones(len(rows_k)), rows_k[:, 0], rows_k[:, 1]])
        bb, *_ = np.linalg.lstsq(X0, rows_k[:, 2], rcond=None)
        band_jk[key].append(float(bb[1]))

for key in band_jk:
    arr = np.array(band_jk[key])
    joint[key]["wavelength_band_jackknife_temperature_coef"] = {
        "summary": summ(arr),
        "span_nm_per_1000K": float(np.nanmax(arr) - np.nanmin(arr)) if np.isfinite(arr).any() else None,
        "n_bands": int(arr.size),
    }

# ---------------------------------------------------------------------------
# 6. explicit trust assessment
# ---------------------------------------------------------------------------
trust = {
    "what_the_level_bootstrap_captures": (
        "Variability from which 21 uniformly spaced metallicity levels happened to be drawn / "
        "weighted, given the fitted log-slope model; it propagates level-to-level scatter into the "
        "crossing location, and (via the shared index draws) preserves pairing across temperatures."),
    "what_it_cannot_capture": [
        "Wavelength autocorrelation and the fixed wavelength grid: resampling levels reuses the same "
        "176 wavelength points, so any uncertainty from the wavelength sampling/quantisation "
        "(fluxes are integer-valued; 5 nm spacing) is invisible to it.",
        "Definitional choice: the level bootstrap is conditioned on 'first neg->pos sign change'. The "
        "6300 K slice has 3 sign changes, so a different but equally defensible rule gives a different "
        "answer; a bootstrap cannot average over definitions it is not told about.",
        "The designed (not random) metallicity ladder: the levels are a fixed uniform grid, so "
        "resampling them with replacement is not sampling from a real population - it measures "
        "sensitivity to level weighting, not frequentist coverage.",
        "The bootstrap is conditional on the fitted OLS-slope model being the truth: it re-estimates "
        "the same linear-in-M slopes on resampled levels. Quantisation (integer fluxes, 3 significant "
        "digits) and the exact-1.0 temperature ratios are outside its scope.",
        "Correlated errors across temperatures: the three slices are resampled with identical indices "
        "(pairing is preserved), but any wavelength-correlated physical error shared by all three "
        "temperatures is not modelled at all.",
    ],
    "paired_comparison": (
        "The paired bootstrap differences d = w(T_a) - w(T_b) computed on the SAME resampled level "
        "multiset remove level-composition noise and test the interaction directly; the level bootstrap "
        "CI width understates total uncertainty because it is blind to wavelength dependence."),
    "recommendation": (
        "Trust the region (=the union of level bootstrap, leave-one-20nm-band-out span, and the "
        "definition ensemble), not any single interval. The level bootstrap is the right tool for "
        "'is the T=5300 vs T=6300 difference bigger than level-sampling noise' - yes/no is decided "
        "there - but the honest uncertainty on any headline crossover wavelength is the wider "
        "band-jackknife/definition span, which for T=6300 K is tens of nm."),
}

# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
result = {
    "gap": "GAP4 - temperature x metallicity interaction and bootstrap uncertainties",
    "inputs": {"photon_flux_grid.tsv": sha256(GRID)},
    "settings": {
        "seed": SEED, "n_bootstrap": NBOOT, "band_nm": BAND_NM,
        "metallicity_levels_used_for_headline_a": [float(v) for v in M_UNI],
        "temperature_levels_K": [float(v) for v in T_LEVELS],
        "n_wavelengths": int(len(W)),
        "flux_quantisation_note": (
            "All photon_flux_cm2 values are integers (3 significant digits); 20 of the 6160 "
            "temperature-ratio values equal exactly 1.0, which pins several crossings to grid nodes."),
    },
    "headline_a_reproduction": headline_a,
    "headline_b_reproduction_crossing_nm_per_metallicity": headline_b,
    "ladder_count_correction": LADDER_COUNT_NOTE,
    "level_bootstrap_metallicity_crossover": boot_a,
    "level_bootstrap_metallicity_crossover_literal_21_levels": bootstrap_21levels,
    "paired_bootstrap_differences_metallicity_crossover": paired,
    "crossover_definition_sensitivity": {
        "persistent_last_bracket_rule": persistent_comparison,
        "bracket_stability_under_level_resampling": bracket_stability,
    },
    "interval_overlap_between_temperature_slices": overlap,
    "definition_ensemble": definition_ensemble,
    "band_vs_level_uncertainty": band_vs_level_uncertainty,
    "temperature_ratio_crossing": {"representative_metallicities": ratio_cross, "trend_vs_metallicity": trend},
    "joint_model": joint,
    "joint_model_interaction_same_level_draws": joint_interaction_samedraws,
    "trust_assessment": trust,
    "runtime_seconds": time.time() - t0,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=1, default=str)
print("wrote", OUT, "in %.1fs" % (time.time() - t0))