#!/usr/bin/env python3
"""
INDEPENDENT shape-identifiability analysis of the cofactor loading scan.

Only direct evidence used: data/cofactor_loading_response.tsv
    loading %   : 0.2, 1.0, 3.0, 8.0
    rate umol/h : 1081, 1217, 470, 268          (n = 4, no replicates)

Question: what can these four points identify about the response shape?
  1. Fit candidate shapes, each <= 3 fitted parameters (plus a k=4 and a
     non-parametric monotone bound as complexity references). Report parameters,
     R2, adjusted R2, RMSE, AIC, AICc, BIC, LOOCV PRESS / LOO-AIC, peak location.
  2. Report the AICc/AIC/LOOCV sets of statistically indistinguishable models
     (Delta <= 2). Flag that AICc is undefined for k=3 at n=4, so the AICc-only
     ranking is structurally blind to the flexible shapes.
  3. Peak-location identifiability: profile each flexible family over a peak
     anchored at m; compare m = 0.2 vs m = 1.0, plus the exact bound argument.
  4. Axis sensitivity: best monotone fit on the L axis == best monotone fit on
     ANY strictly increasing (e.g. log) axis, because monotone reparametrisation
     preserves the ordering of the four y-values. Quantify.

Every fit: deterministic multi-start derivative-free search (coarse grid then
Nelder-Mead / bounded scalar refinement) with the linear coefficients solved by
exact least squares at each parameter value. No other file under output/ is read.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

BASE = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project"
DATA = os.path.join(BASE, "data", "cofactor_loading_response.tsv")
OUTDIR = os.path.join(BASE, "output", "verification")
OUT = os.path.join(OUTDIR, "cofactor_shape_identifiability_agentB.json")

df = pd.read_csv(DATA, sep="\t")
L = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)
n = len(L)
SST = float(np.sum((y - y.mean()) ** 2))
TINY = 1e-12 * SST


# ---------------------------------------------------------------- fitting core
def lsq_active(G, yv, nonneg=()):
    """LS with coef_j >= 0 for j in nonneg (small active-set loop)."""
    m = G.shape[1]
    fixed = {}
    coef = np.zeros(m)
    for _ in range(m + 2):
        free = [j for j in range(m) if j not in fixed]
        resid = yv.astype(float).copy()
        for j, v in fixed.items():
            resid = resid - G[:, j] * v
        coef = np.zeros(m)
        if free:
            cf, *_ = np.linalg.lstsq(G[:, free], resid, rcond=None)
            for jj, j in enumerate(free):
                coef[j] = cf[jj]
        for j, v in fixed.items():
            coef[j] = v
        viol = [j for j in nonneg if j not in fixed and coef[j] < 0.0]
        if not viol:
            break
        for j in viol:
            fixed[j] = 0.0
    rss = float(np.sum((yv - G @ coef) ** 2))
    return rss, coef


def solve_theta(obj, bounds, logdims, nstart=6):
    """Global-ish deterministic minimisation of obj(theta) (1-D or 2-D)."""
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    ld = np.array(logdims, bool)
    d = len(lo)

    def wrap(th):
        th = np.clip(th, lo, hi)
        return obj(np.where(ld, np.exp(th), th))

    if d == 1:
        grid = np.linspace(lo[0], hi[0], 200)
        v = [wrap(np.array([g])) for g in grid]
        i = int(np.argmin(v))
        a = grid[max(0, i - 1)]
        b = grid[min(len(grid) - 1, i + 1)]
        if b > a:
            r = minimize_scalar(wrap, bounds=(a, b), method="bounded",
                                options=dict(xatol=1e-10))
            best_s, best_v = r.x, r.fun
        else:
            best_s, best_v = grid[i], v[i]
        return float(best_v), np.where(ld, np.exp(best_s), best_s)

    # 2-D: coarse grid starts -> Nelder-Mead
    g0 = np.linspace(lo[0], hi[0], nstart)
    g1 = np.linspace(lo[1], hi[1], nstart)
    starts = [(a, b) for a in g0 for b in g1]
    # add the corners/edges of a finer grid
    g0f = np.linspace(lo[0], hi[0], 2 * nstart + 1)
    g1f = np.linspace(lo[1], hi[1], 2 * nstart + 1)
    starts += [(g0f[i], g1f[j]) for i in range(len(g0f)) for j in range(len(g1f))
               if i % 2 == 1 and j % 2 == 1]
    best_s, best_v = None, np.inf
    for s in starts:
        r = minimize(wrap, np.array(s, float), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-10, fatol=1e-10))
        if r.fun < best_v:
            best_v, best_s = float(r.fun), r.x
    return best_v, np.where(ld, np.exp(best_s), best_s)


def make_fit(build_G, rss_y, nonneg, bounds, logdims, **kw):
    """Return (rss, theta, coef, pred) for one dataset (rss_y, length n)."""
    if len(bounds) == 0:
        th = np.array([])
        G = build_G(th, None)
        rss, coef = lsq_active(G, rss_y, nonneg)
        return rss, th, coef, G @ coef

    def obj(th):
        rss, _ = lsq_active(build_G(th, None), rss_y, nonneg)
        return rss

    rss, th = solve_theta(obj, bounds, logdims, **kw)
    G = build_G(th, None)
    rss, coef = lsq_active(G, rss_y, nonneg)
    return rss, th, coef, G @ coef


# ------------------------------------------------------------------ model zoo
# every entry: name, k, bounds, logdims, nonneg, build_G(theta)->basis columns,
#              pred_full(theta, coef, Lx) -> curve, pnames, shape description
def models():
    M = []

    def add(name, k, bounds, logdims, nonneg, build_G, pred_full, pnames, shape,
            mechanism=None):
        M.append(dict(name=name, k=k, bounds=bounds, logdims=logdims, nonneg=nonneg,
                      build_G=build_G, pred_full=pred_full, pnames=pnames,
                      shape=shape, mechanism=mechanism))

    # --- monotone non-increasing families -----------------------------------
    add("M1_monotone_power_decay", 2, [(1e-6, 8.0)], [False], {0},
        lambda th, _: (L ** (-th[0]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** (-th[0]), ["b"],
        "monotone non-increasing", "power law r=a*L^-b")

    add("M2_monotone_exponential_decay", 2, [(1e-6, 20.0)], [False], {0},
        lambda th, _: np.exp(-th[0] * L)[:, None],
        lambda th, c, Lx: c[0] * np.exp(-th[0] * Lx), ["b"],
        "monotone non-increasing", "exponential decay r=a*exp(-bL)")

    add("M3_monotone_exponential_to_offset", 3, [(1e-5, 20.0)], [False], {1},
        lambda th, _: np.column_stack([np.ones_like(L), np.exp(-th[0] * L)]),
        lambda th, c, Lx: c[0] + c[1] * np.exp(-th[0] * Lx), ["b"],
        "monotone non-increasing", "r=c+A*exp(-bL)")

    add("M4_monotone_decreasing_in_log_axis", 2, [], [], {1},
        lambda th, _: np.column_stack([np.ones_like(L), -np.log(L)]),
        lambda th, c, Lx: c[0] - c[1] * np.log(Lx), [],
        "monotone non-increasing on the log axis", "r=c0+s*lnL, s<=0")

    # --- monotone non-decreasing (saturating) --------------------------------
    add("M5_saturating_rise_michaelis_menten", 2,
        [(np.log(1e-3), np.log(1e3))], [True], {0},
        lambda th, _: (L / (th[0] + L))[:, None],
        lambda th, c, Lx: c[0] * Lx / (th[0] + Lx), ["K"],
        "monotone non-decreasing", "r=V*L/(K+L)")

    add("M6_saturating_rise_hill", 3,
        [(np.log(1e-3), np.log(1e3)), (np.log(0.05), np.log(10.0))], [True, True], {0},
        lambda th, _: (L ** th[1] / (th[0] ** th[1] + L ** th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** th[1] / (th[0] ** th[1] + Lx ** th[1]),
        ["K", "h"], "monotone non-decreasing", "Hill r=V*L^h/(K^h+L^h)")

    # --- rise then fall ------------------------------------------------------
    add("M7_substrate_inhibition", 3,
        [(np.log(1e-4), np.log(1e4)), (np.log(1e-5), np.log(1e5))], [True, True], {0},
        lambda th, _: (L / (th[0] + L + L ** 2 / th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx / (th[0] + Lx + Lx ** 2 / th[1]),
        ["K", "Ki"], "rise then fall, interior peak at sqrt(K*Ki)",
        "r=V*L/(K+L+L^2/Ki)")

    add("M8_gamma_rise_times_decay", 3,
        [(np.log(0.02), np.log(20.0)), (np.log(0.02), np.log(200.0))], [True, True], {0},
        lambda th, _: (L ** th[0] * np.exp(-L / th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** th[0] * np.exp(-Lx / th[1]),
        ["p", "tau"], "rise then fall, interior peak at p*tau",
        "r=A*L^p*exp(-L/tau)")

    add("M9_quadratic_in_L", 3, [], [], set(),
        lambda th, _: np.column_stack([np.ones_like(L), L, L ** 2]),
        lambda th, c, Lx: c[0] + c[1] * Lx + c[2] * Lx ** 2, [],
        "free quadratic (peak only if c2<0)")

    add("M10_quadratic_in_logL", 3, [], [], set(),
        lambda th, _: np.column_stack([np.ones_like(L), np.log(L), np.log(L) ** 2]),
        lambda th, c, Lx: c[0] + c[1] * np.log(Lx) + c[2] * np.log(Lx) ** 2, [],
        "free quadratic in log L (peak only if c2<0)")

    add("M11_linear_in_L", 2, [], [], set(),
        lambda th, _: np.column_stack([np.ones_like(L), L]),
        lambda th, c, Lx: c[0] + c[1] * Lx, [], "straight line")

    # --- two-regime / kink ---------------------------------------------------
    add("M12_plateau_then_linear_decline_kink", 3,
        [(np.log(1e-3), np.log(1e3))], [True], {1},
        lambda th, _: np.column_stack([np.ones_like(L),
                                       -np.maximum(0.0, L - th[0])]),
        lambda th, c, Lx: c[0] - c[1] * np.maximum(0.0, Lx - th[0]), ["Lk"],
        "flat up to a break Lk, then linear decline",
        "r=c0-c1*max(0,L-Lk), c1>=0")

    add("M13_plateau_then_exponential_decline", 3,
        [(np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e3))], [True, True], {0},
        lambda th, _: (np.where(L <= th[0], 1.0, np.exp(-(L - th[0]) / th[1])))[:, None],
        lambda th, c, Lx: c[0] * np.where(Lx <= th[0], 1.0,
                                          np.exp(-(Lx - th[0]) / th[1])),
        ["Lk", "tau"], "plateau then exponential decline")

    add("M14_broken_stick_peak_4par", 4,
        [(np.log(0.05), np.log(20.0)), (1e-3, 1e4), (1e-3, 1e4)], [True, False, False],
        set(),
        lambda th, _: np.column_stack([
            np.ones_like(L),
            np.where(L <= th[0], L, th[0]),
            np.where(L <= th[0], 0.0, L - th[0])]),
        lambda th, c, Lx: (c[0]
                           + c[1] * np.where(Lx <= th[0], Lx, th[0])
                           + c[2] * np.where(Lx <= th[0], 0.0, Lx - th[0])),
        ["Lp", "b", "d"], "continuous broken stick, b>=0, d<=0 -> peak at Lp")
    return M


MODELS = models()


# ------------------------------------------------------- information criteria
def aic(rss, k):
    return n * np.log(max(rss, TINY) / n) + 2 * k


def aicc(rss, k):
    return None if (n - k - 1) <= 0 else aic(rss, k) + 2 * k * (k + 1) / (n - k - 1)


def bic(rss, k):
    return n * np.log(max(rss, TINY) / n) + k * np.log(n)


def curve_peak(md, th, coef):
    Lg = np.linspace(L.min() * 0.2, L.max() * 1.2, 4001)
    yg = np.asarray(md["pred_full"](th, coef, Lg), float)
    i = int(np.argmax(yg))
    return float(Lg[i]), float(yg[i]), bool(Lg[0] < Lg[i] < Lg[-1])


def fit_full(md, Lv=None, yv=None, nstart=6):
    """Fit md on (Lv, yv) -- defaults to the full 4-point dataset."""
    if Lv is None:
        Lv, yv = L, y
    old = globals()["L"], globals()["y"], globals()["n"]
    globals()["L"], globals()["y"], globals()["n"] = Lv, yv, len(Lv)

    def obj(th):
        rss, _ = lsq_active(md["build_G"](th, None), yv, md["nonneg"])
        return rss

    if len(md["bounds"]) == 0:
        th = np.array([])
    else:
        _, th = solve_theta(obj, md["bounds"], md["logdims"], nstart=nstart)
    G = md["build_G"](th, None)
    rss, coef = lsq_active(G, yv, md["nonneg"])
    pred = G @ coef
    globals()["L"], globals()["y"], globals()["n"] = old
    return rss, th, coef, pred


def loocv(md, nstart=5):
    pr, err = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        rss, th, coef, _ = fit_full(md, L[tr], y[tr], nstart=nstart)
        pi = float(np.asarray(md["pred_full"](th, coef, np.array([L[i]]))).ravel()[0])
        pr.append(pi)
        err.append(pi - y[i])
    err = np.array(err)
    press = float(np.sum(err ** 2))
    return dict(pred=[float(p) for p in pr], press=press,
                rmse=float(np.sqrt(press / n)), mae=float(np.mean(np.abs(err))),
                max_abs=float(np.max(np.abs(err))))


# ------------------------------------------------------------------- run fits
rows = []
for md in MODELS:
    rss, th, coef, pred = fit_full(md)
    r2 = 1.0 - rss / SST
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - md["k"] - 1) if n - md["k"] - 1 > 0 else None
    lv = loocv(md)
    pk, pkv, interior = curve_peak(md, th, coef)
    rows.append(dict(
        name=md["name"], mechanism=md["mechanism"], shape=md["shape"], k=md["k"],
        param_names=list(md["pnames"]),
        params={p: float(v) for p, v in zip(md["pnames"], th)},
        linear_coefs=[float(v) for v in coef],
        rss=float(rss), r2=float(r2), adj_r2=adj, rmse=float(np.sqrt(rss / n)),
        aic=float(aic(rss, md["k"])), aicc=aicc(rss, md["k"]), bic=float(bic(rss, md["k"])),
        df_resid=n - md["k"], pred=[float(v) for v in pred],
        loo=lv, loo_aic=float(n * np.log(max(lv["press"], TINY) / n) + 2 * md["k"]),
        peak_L=pk, peak_value=pkv, peak_interior=interior,
        is_monotone=("monotone" in md["shape"]),
        saturates_parameters=bool(n - md["k"] <= 0),
    ))


# --------------------- non-parametric monotone bounds (isotonic regression) --
def pava(yv, increasing):
    blocks = [[float(v)] for v in yv]
    means = [float(v) for v in yv]
    while True:
        merged = False
        for i in range(len(means) - 1):
            bad = means[i] > means[i + 1] if increasing else means[i] < means[i + 1]
            if bad:
                blocks[i:i + 2] = [blocks[i] + blocks[i + 1]]
                means[i:i + 2] = [sum(blocks[i]) / len(blocks[i])]
                merged = True
                break
        if not merged:
            break
    fit = []
    for b, m in zip(blocks, means):
        fit.extend([m] * len(b))
    return np.array(fit), len(blocks)


iso_inc, k_inc = pava(y, True)
iso_dec, k_dec = pava(y, False)
rss_iso_inc = float(np.sum((y - iso_inc) ** 2))
rss_iso_dec = float(np.sum((y - iso_dec) ** 2))


def iso_loo(increasing):
    """LOO for isotonic: refit on 3 points, predict held-out with the level of the
    fitted block whose L-range is nearest (step functions are flat elsewhere)."""
    pr = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        f, _ = pava(y[tr], increasing)
        cand = [np.argmin(np.abs(np.array(tr) - i))]  # nearest block in index space
        pi = float(f[cand[0]])
        pr.append(pi)
    err = np.array(pr) - y
    return float(np.sqrt(np.mean(err ** 2))), float(np.sum(err ** 2)), [float(p) for p in pr]


iso_dec_loo_rmse, iso_dec_press, iso_dec_loo_pred = iso_loo(False)
iso_rows = []
for nm, fit, rss_iso, kb in [("I1_isotonic_decreasing_best_monotone", iso_dec, rss_iso_dec, k_dec),
                             ("I2_isotonic_increasing_best_monotone", iso_inc, rss_iso_inc, k_inc)]:
    iso_rows.append(dict(name=nm, k=kb, rss=float(rss_iso), blocks=int(kb),
                         fit=[float(v) for v in fit],
                         r2=1.0 - rss_iso / SST, rmse=float(np.sqrt(rss_iso / n)),
                         aic=float(aic(rss_iso, kb)), aicc=aicc(rss_iso, kb),
                         bic=float(bic(rss_iso, kb)),
                         note=("best possible fit of ANY monotone non-decreasing/"
                               "non-increasing function of L (adaptive complexity; "
                               "k = number of level blocks)")))

# --------------------------------------- indistinguishable sets, all criteria
def within2(vals, key, ascending=True):
    v = [(r["name"], key(r)) for r in vals if key(r) is not None]
    v.sort(key=lambda t: t[1], reverse=not ascending)
    best = v[0][1]
    return [dict(name=nm, value=float(val), delta=float(val - best)) for nm, val in v
            if abs(val - best) <= 2.0]


crit = dict(
    aic_within_2=within2(rows, lambda r: r["aic"]),
    aicc_within_2=within2(rows, lambda r: r["aicc"]),
    bic_within_2=within2(rows, lambda r: r["bic"]),
    loo_aic_within_2=within2(rows, lambda r: r["loo_aic"]),
)
crit["aic_note"] = ("AICc = AIC + 2k(k+1)/(n-k-1) has denominator 0 when n=4, k=3, so it is "
                    "UNDEFINED for every 3-parameter model. The finite-AICc ranking is "
                    "structurally restricted to the 2-parameter models and can never rank "
                    "the flexible peaked shapes. AICc-based conclusions at n=4 with k=3 are "
                    "artefacts and are not evidence.")
crit["best_by_each_criterion"] = {
    "aic": min(rows, key=lambda r: r["aic"])["name"],
    "aicc": min([r for r in rows if r["aicc"] is not None],
                key=lambda r: r["aicc"])["name"],
    "bic": min(rows, key=lambda r: r["bic"])["name"],
    "loo_aic": min(rows, key=lambda r: r["loo_aic"])["name"],
}

# --------------------------------------------- peak-location identifiability
def repo_L(Lv):
    old = globals()["L"]
    globals()["L"] = Lv
    return old


def prof_sub_inhibition(m):
    """1 free shape parameter: r = V*L/(K+L+L^2/Ki) with mode sqrt(K*Ki) = m."""
    def obj(th):
        K = np.exp(th[0])
        G = (L / (K + L + L ** 2 * K / m ** 2))[:, None]
        rss, _ = lsq_active(G, y, {0})
        return rss
    v, th = solve_theta(obj, [(np.log(1e-5), np.log(1e5))], [True])
    return float(v), float(np.exp(th[0]))


def prof_gamma(m):
    """r = A*L^p*exp(-L/tau) with peak p*tau = m; 1 free parameter tau (p=m/tau)."""
    def obj(th):
        tau = np.exp(th[0])
        p = m / tau
        G = (L ** p * np.exp(-L / tau))[:, None]
        rss, _ = lsq_active(G, y, {0})
        return rss
    v, th = solve_theta(obj, [(np.log(1e-3), np.log(1e3))], [True])
    return float(v), float(np.exp(th[0]))


def prof_quad_vertex(m):
    """r = c0 - a*(L-m)^2 with a>0 (peak exactly at m); 2 linear params."""
    G = np.column_stack([np.ones_like(L), (L - m) ** 2])
    rss, coef = lsq_active(G, y, set())
    if coef[1] >= 0:      # forces a peak; a<0 means no interior peak at m
        return float("inf"), None
    return float(rss), float(coef[1])


PROFILE = {"substrate_inhibition": prof_sub_inhibition,
           "gamma": prof_gamma,
           "quadratic_vertex": prof_quad_vertex}

m_grid = np.exp(np.linspace(np.log(0.02), np.log(24.0), 96))
profiles = {}
for pname, fn in PROFILE.items():
    rsss = [fn(float(m))[0] for m in m_grid]
    rsss = np.array([v if np.isfinite(v) else np.nan for v in rsss])
    i = int(np.nanargmin(rsss))
    thr2 = rsss[i] * np.exp(2.0 / n)          # same k -> Delta AIC <= 2
    ok = np.isfinite(rsss) & (rsss <= thr2)
    # dedicated anchors at the two candidate locations
    r02 = fn(0.2)[0]
    r10 = fn(1.0)[0]
    kk = 2                                     # mode fixed -> K (or tau) + scale
    a02 = aicc(r02, kk) if np.isfinite(r02) else None
    a10 = aicc(r10, kk) if np.isfinite(r10) else None
    profiles[pname] = dict(
        mode_at_profile_min=float(m_grid[i]), rss_min=float(rsss[i]),
        rss_at_mode_0p2=float(r02) if np.isfinite(r02) else None,
        rss_at_mode_1p0=float(r10) if np.isfinite(r10) else None,
        aic_at_mode_0p2=aic(r02, kk) if np.isfinite(r02) else None,
        aic_at_mode_1p0=aic(r10, kk) if np.isfinite(r10) else None,
        aicc_at_mode_0p2=a02, aicc_at_mode_1p0=a10,
        delta_aic_mode_0p2_vs_1p0=float(n * np.log(r02 / r10))
        if (np.isfinite(r02) and np.isfinite(r10)) else None,
        delta_aicc_mode_0p2_vs_1p0=(None if (a02 is None or a10 is None)
                                    else float(a02 - a10)),
        mode_set_within2_aic=[float(v) for v in m_grid[ok]],
        mode_interval_within2_aic=[float(m_grid[ok].min()), float(m_grid[ok].max())]
        if ok.any() else None,
        rss_profile=[None if not np.isfinite(v) else float(v) for v in rsss],
    )

peakloc = dict(
    profiles=profiles,
    m_grid=[float(v) for v in m_grid],
    hard_bound_from_single_peakedness=dict(
        interval_percent=[0.2, 3.0],
        argument=("For ANY continuous curve with a single interior maximum, "
                  "rate(1.0)=1217 > rate(0.2)=1081 forces mode > 0.2 and "
                  "rate(1.0)=1217 > rate(3.0)=470 forces mode < 3.0. The peak "
                  "location is therefore bounded to (0.2, 3.0) for every smooth "
                  "single-peaked description; 0.2 itself is excluded.")),
    pairwise_rise_p_value=dict(
        contrast="rate(1.0) > rate(0.2)",
        two_sided_p=0.5,
        note=("This rise is a single pair of unreplicated measurements; under "
              "exchangeability its two-sided p is 0.5. It is the only evidence "
              "that pushes the mode above 0.2.")),
    pitchers=dict(
        cross_family_mode_estimates={k: profiles[k]["mode_at_profile_min"]
                                     for k in profiles}),
    verdict=("NOT IDENTIFIABLE between 0.2 and 1.0: within the substrate-inhibition "
             "and concave-quadratic families the two anchors differ by < 2 AIC units "
             "(indistinguishable), while the gamma family weakly prefers 0.2 "
             "(Delta AIC = -3.3). No family supports 1.0 over 0.2 by more than 2 units, "
             "and the form-free bound is only (0.2, 3.0). Family-internal optima "
             "(0.48 vs 0.58 vs up to 2.5) disagree, i.e. the 'location' is an artefact "
             "of the assumed functional form.")
)

# --------------------------------------------------------- axis sensitivity
axis = dict(
    log_axis_loading=np.log(L).tolist(),
    best_monotone_rss_same_for_any_increasing_axis=float(rss_iso_dec),
    best_monotone_fit=iso_dec.tolist(),
    best_peaked_param_rss=float(min(r["rss"] for r in rows)),
    delta_aic_best_peaked_vs_best_monotone=float(
        aic(min(r["rss"] for r in rows), 3) - aic(rss_iso_dec, k_dec)),
    monotone_model_rss_on_L_axis={
        "exponential_decay_M2": float(next(r["rss"] for r in rows
                                           if r["name"].startswith("M2"))),
        "power_decay_M1": float(next(r["rss"] for r in rows
                                     if r["name"].startswith("M1"))),
        "linear_in_logL_M4": float(next(r["rss"] for r in rows
                                        if r["name"].startswith("M4"))),
        "plateau_then_decline_kink_M12": float(next(r["rss"] for r in rows
                                                    if r["name"].startswith("M12"))),
        "plateau_then_exp_decline_M13": float(next(r["rss"] for r in rows
                                                   if r["name"].startswith("M13"))),
        "isotonic_bound": float(rss_iso_dec)},
    conclusion=("A strictly increasing reparametrisation of the loading axis (log or any "
                "other) cannot change the order of the four response values, so it cannot "
                "remove the interior maximum: the isotonic (PAVA) RSS = 9248 is the best "
                "achievable by ANY monotone description on ANY increasing axis, and a "
                "monotone description costs only ~1.5 AIC units versus the best peaked "
                "fit. Conversely, whether the recorded settings were internally linear or "
                "log-scaled is NOT identifiable from the four numbers: both are monotone "
                "relabellings of the same design, and all fits that depend only on the "
                "ordering are identical. Only the numeric position of a break/peak in "
                "true loading units would change."),
)

# ------------------------------------------------------------ permutations
def spearman(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


perms = list(itertools.permutations(range(n)))
rho = spearman(L, y)
tau = float(stats.kendalltau(L, y).statistic)
p_rho = float(np.mean([abs(spearman(L, y[list(p)])) >= abs(rho) - 1e-12 for p in perms]))
p_tau = float(np.mean([abs(stats.kendalltau(L, y[list(p)]).statistic) >= abs(tau) - 1e-12
                       for p in perms]))
p_int = float(np.mean([int(np.argmax(y[list(p)])) in (1, 2) for p in perms]))
p_v = float(np.mean([(y[list(p)][0] < y[list(p)][1]) and (y[list(p)][1] > y[list(p)][2])
                     and (y[list(p)][2] > y[list(p)][3]) for p in perms]))
perm = dict(n_perm=len(perms), spearman_rho=rho, two_sided_p_spearman=p_rho,
            kendall_tau=tau, two_sided_p_kendall=p_tau,
            p_interior_maximum_at_all=p_int, p_exact_rise_then_fall_pattern=p_v,
            note=("Exact enumeration over all 24 orderings. The smallest attainable "
                  "two-sided p for a monotone association at n=4 is 1/3, so monotone "
                  "decline can never reach p<0.05 here. The 'rise then fall' pattern is "
                  "a post-hoc selected pattern (p=0.125); the unselected event "
                  "'maximum is interior' has p=0.5."))

# ------------------------------------------------------------------ output
out = dict(
    data=dict(file="data/cofactor_loading_response.tsv",
              loading_percent=[float(v) for v in L],
              rate_umol_per_h=[float(v) for v in y], n_points=n,
              analysis_unit=("One complete loading scan (4 settings, no replicates). "
                             "Points within the scan are NOT independent replicates."),
              sst=float(SST), y_mean=float(y.mean())),
    model_table=sorted(rows, key=lambda r: r["aic"]),
    isotonic_bounds=iso_rows,
    indistinguishable_sets=crit,
    peak_location=peakloc,
    axis_sensitivity=axis,
    permutation_tests=perm,
    limitations=[
        "n = 4 unreplicated points: residual df = 1 for any 3-parameter model, and no "
        "independent error estimate exists. RMSE/R2 below are descriptive only.",
        "AICc is undefined for k=3 at n=4 (n-k-1=0). The AICc ranking contains only the "
        "2-parameter models and therefore cannot compare the peaked shapes at all.",
        "AIC with n=4 is severely over-optimistic; its 'best' model (substrate "
        "inhibition, RSS=6336) beats the form-free best monotone description by only "
        "~1.5 AIC units, far below any separating threshold.",
        "With 4 points one can always find a 3-parameter family passing near them; the "
        "comparison is a parsimony bookkeeping exercise, not an identification.",
        "All conclusions are conditional on these four settings; nothing here shows "
        "what happens between or beyond them.",
    ],
    summary_verdict=("UNRESOLVED: the four points cannot separate the candidate shapes. "
                     "No model wins by >2 AIC units over every rival under any criterion, "
                     "and the criteria disagree with one another (AIC picks a peaked "
                     "substrate-inhibition curve; LOOCV-AIC and the finite-AICc ranking "
                     "pick a monotone exponential decline). The peak location is bounded "
                     "only to the interval (0.2, 3.0), and even the presence of a peak "
                     "cannot be established at the 2-AIC-unit level."),
)

os.makedirs(OUTDIR, exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2)

# ------------------------------------------------------------------ console
print(f"n={n}  y={y.tolist()}  L={L.tolist()}")
print(f"\n{'model':40s} {'k':>2s} {'RSS':>11s} {'R2':>7s} {'RMSE':>7s} {'AIC':>8s} "
      f"{'AICc':>8s} {'BIC':>8s} {'LOO-AIC':>8s} {'peak_L':>7s}")
for r in sorted(rows, key=lambda r: r["aic"]):
    ac = "undef" if r["aicc"] is None else f"{r['aicc']:.2f}"
    print(f"{r['name']:40s} {r['k']:2d} {r['rss']:11.1f} {r['r2']:7.4f} {r['rmse']:7.1f} "
          f"{r['aic']:8.2f} {ac:>8s} {r['bic']:8.2f} {r['loo_aic']:8.2f} "
          f"{r['peak_L']:7.3f}")
print(f"{'I1_isotonic_decreasing (bound)':40s} {k_dec:2d} {rss_iso_dec:11.1f} "
      f"{1-rss_iso_dec/SST:7.4f} {np.sqrt(rss_iso_dec/n):7.1f} "
      f"{aic(rss_iso_dec,k_dec):8.2f} {'undef':>8s} {bic(rss_iso_dec,k_dec):8.2f} "
      f"{n*np.log(iso_dec_press/n)+2*k_dec:8.2f}")
print(f"{'I2_isotonic_increasing (bound)':40s} {k_inc:2d} {rss_iso_inc:11.1f} "
      f"{1-rss_iso_inc/SST:7.4f} {np.sqrt(rss_iso_inc/n):7.1f} "
      f"{aic(rss_iso_inc,k_inc):8.2f} {'undef':>8s} {bic(rss_iso_inc,k_inc):8.2f}")
print("\nisotonic decreasing fit:", iso_dec.tolist(), " increasing:", iso_inc.tolist())
print("\n-- indistinguishable sets (Delta <= 2)")
for kk in ["aic_within_2", "aicc_within_2", "bic_within_2", "loo_aic_within_2"]:
    print(f"  {kk}: {[(d['name'], round(d['delta'],2)) for d in crit[kk]]}")
print("  best by criterion:", crit["best_by_each_criterion"])
print("\n-- peak location")
for kk, v in profiles.items():
    print(f"  {kk}: mode*={v['mode_at_profile_min']:.3f} "
          f"RSS(0.2)={None if v['rss_at_mode_0p2'] is None else round(v['rss_at_mode_0p2'],1)} "
          f"RSS(1.0)={None if v['rss_at_mode_1p0'] is None else round(v['rss_at_mode_1p0'],1)} "
          f"dAIC={None if v['delta_aic_mode_0p2_vs_1p0'] is None else round(v['delta_aic_mode_0p2_vs_1p0'],2)} "
          f"dAICc={None if v['delta_aicc_mode_0p2_vs_1p0'] is None else round(v['delta_aicc_mode_0p2_vs_1p0'],2)} "
          f"2-unit mode interval={v['mode_interval_within2_aic']}")
print("\n-- axis: best monotone RSS =", rss_iso_dec, " best peaked RSS =",
      min(r["rss"] for r in rows), " dAIC =",
      round(aic(min(r["rss"] for r in rows), 3) - aic(rss_iso_dec, k_dec), 2))
print("\n-- permutations:", {k: v for k, v in perm.items() if k != "note"})
print("\nwrote", OUT)