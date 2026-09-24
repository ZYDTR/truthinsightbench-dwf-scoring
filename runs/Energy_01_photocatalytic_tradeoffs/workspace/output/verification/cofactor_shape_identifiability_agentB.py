#!/usr/bin/env python3
"""
Independent shape-identifiability analysis of the cofactor loading scan.

Data (the ONLY direct evidence): data/cofactor_loading_response.tsv
  cofactor_loading_percent : 0.2, 1.0, 3.0, 8.0   (n = 4, no replicates)
  product_evolution_rate_umol_h : 1081, 1217, 470, 268

What this script does, from scratch:
  1. Fits a family of candidate response shapes, each with <= 3 fitted parameters.
     For each: parameters, R2, adjusted R2, RMSE, AIC, AICc (with the n/k
     degeneracy flagged), BIC, leave-one-out CV RMSE.
  2. Reports the set of models statistically indistinguishable from the best
     (Delta AICc <= 2 where AICc is defined; Delta AIC and LOOCV as fallbacks,
     because AICc is undefined for k = 3 when n = 4).
  3. Tests peak-location identifiability by profiling three model families whose
     maximum is anchored at a prescribed load m: substrate-inhibition
     reparametrised by its mode, a gamma (rise-times-decay) family, and a
     quadratic with the vertex anchored at m. Compares m = 0.2 vs m = 1.0.
  4. Axis-sensitivity: (a) best monotone fit to y vs L (isotonic / PAVA), which
     is also the best fit to y vs any strictly increasing transform of L, and
     (b) explicit fits of monotone models on the log axis. Because a strictly
     increasing reparametrisation cannot change the ordering of the four
     y-values, an interior maximum survives any log rescaling of the axis.

No external data, no other file under output/ is read.  Every fit here is done
with grid search + active-set linear least squares, i.e. deterministic.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project"
DATA = os.path.join(BASE, "data", "cofactor_loading_response.tsv")
OUTDIR = os.path.join(BASE, "output", "verification")
OUT = os.path.join(OUTDIR, "cofactor_shape_identifiability_agentB.json")

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA, sep="\t")
L = df.iloc[:, 0].to_numpy(float)
y = df.iloc[:, 1].to_numpy(float)
n = len(L)
SS_TOT = float(np.sum((y - y.mean()) ** 2))

# ----------------------------------------------------------------------------
# fitting machinery
# ----------------------------------------------------------------------------
def lsq_active(G, yv, nonneg=()):
    """Least squares with inequality coef_j >= 0 for j in nonneg (active set)."""
    m = G.shape[1]
    fixed = {}
    coef = np.zeros(m)
    for _ in range(m + 1):
        free = [j for j in range(m) if j not in fixed]
        resid = yv.astype(float).copy()
        for j, v in fixed.items():
            resid = resid - G[:, j] * v
        if free:
            cf, *_ = np.linalg.lstsq(G[:, free], resid, rcond=None)
            coef = np.zeros(m)
            for jj, j in enumerate(free):
                coef[j] = cf[jj]
            for j, v in fixed.items():
                coef[j] = v
        else:
            coef = np.zeros(m)
            for j, v in fixed.items():
                coef[j] = v
        viol = [j for j in nonneg if j not in fixed and coef[j] < 0.0]
        if not viol:
            break
        for j in viol:
            fixed[j] = 0.0
    pred = G @ coef
    rss = float(np.sum((yv - pred) ** 2))
    return rss, coef


def zoom_search(obj, lo, hi, npts=33, rounds=6, shrink=0.35, logdims=None):
    """Deterministic coarse-grid + zoom-in search for a 1-D or 2-D objective."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    dim = len(lo)
    logdims = np.asarray(logdims if logdims is not None else [False] * dim, bool)

    def decode(s):
        return np.exp(s) if logdims.any() else s

    best_s = None
    best_v = np.inf

    def ev(s):
        nonlocal best_s, best_v
        th = np.where(logdims, np.exp(s), s)
        v = obj(th)
        if v < best_v:
            best_v, best_s = v, np.array(s, float)
        return v

    if dim == 1:
        for s0 in np.linspace(lo[0], hi[0], npts):
            ev(np.array([s0]))
    else:
        g0 = np.linspace(lo[0], hi[0], npts)
        g1 = np.linspace(lo[1], hi[1], npts)
        for a in g0:
            for b in g1:
                ev(np.array([a, b]))

    c = best_s.copy()
    half = (hi - lo) * 0.5
    for _ in range(rounds):
        half = half * shrink
        if dim == 1:
            ss = np.linspace(max(lo[0], c[0] - half[0]), min(hi[0], c[0] + half[0]), npts)
            for s0 in ss:
                ev(np.array([s0]))
        else:
            k = max(9, npts // 2)
            ss0 = np.linspace(max(lo[0], c[0] - half[0]), min(hi[0], c[0] + half[0]), k)
            ss1 = np.linspace(max(lo[1], c[1] - half[1]), min(hi[1], c[1] + half[1]), k)
            for a in ss0:
                for b in ss1:
                    ev(np.array([a, b]))
        c = best_s.copy()
    th = np.where(logdims, np.exp(best_s), best_s)
    return best_v, th


def fit_model(md, Lv, yv, npts=33, rounds=6):
    build_G = md["build_G"]
    if len(md["bounds"]) == 0:
        th = np.array([])
        G = build_G(th, Lv)
        rss, coef = lsq_active(G, yv, md["nonneg"])
    else:
        def obj(th):
            G = build_G(th, Lv)
            rss, _ = lsq_active(G, yv, md["nonneg"])
            return rss
        lo = [b[0] for b in md["bounds"]]
        hi = [b[1] for b in md["bounds"]]
        _, th = zoom_search(obj, lo, hi, npts=npts, rounds=rounds, logdims=md["logdims"])
        G = build_G(th, Lv)
        rss, coef = lsq_active(G, yv, md["nonneg"])
    pred = G @ coef
    return dict(rss=float(rss), theta=th, coef=coef, pred=pred)


def build_models():
    M = []

    def add(name, k, bounds, logdims, nonneg, build_G, pred_full, pnames, shape,
            alias=None):
        M.append(dict(name=name, k=k, bounds=bounds, logdims=logdims, nonneg=nonneg,
                      build_G=build_G, pred_full=pred_full, pnames=pnames,
                      shape=shape, alias=alias))

    # 1. monotone power-law decay  r = a L^-b
    add("monotone_power_decay", 2, [(1e-6, 8.0)], [False], {0},
        lambda th, Lv: (Lv ** (-th[0]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** (-th[0]),
        ["b"], "monotone non-increasing")

    # 2. monotone exponential decay  r = a exp(-b L)
    add("monotone_exponential_decay", 2, [(1e-6, 20.0)], [False], {0},
        lambda th, Lv: np.exp(-th[0] * Lv)[:, None],
        lambda th, c, Lx: c[0] * np.exp(-th[0] * Lx),
        ["b"], "monotone non-increasing")

    # 3. exponential decay to an offset  r = c + A exp(-b L), A >= 0
    add("monotone_exponential_offset", 3, [(1e-5, 20.0)], [False], {1},
        lambda th, Lv: np.column_stack([np.ones_like(Lv), np.exp(-th[0] * Lv)]),
        lambda th, c, Lx: c[0] + c[1] * np.exp(-th[0] * Lx),
        ["b"], "monotone non-increasing (3 par)")

    # 4. saturating rise without inhibition (Michaelis-Menten)  r = V L/(K+L)
    add("saturating_rise_michaelis_menten", 2, [(np.log(0.01), np.log(100.0))], [True], {0},
        lambda th, Lv: (Lv / (th[0] + Lv))[:, None],
        lambda th, c, Lx: c[0] * Lx / (th[0] + Lx),
        ["K"], "monotone non-decreasing")

    # 5. saturating rise, Hill 3 par  r = V L^h/(K^h + L^h)
    add("saturating_rise_hill", 3,
        [(np.log(0.01), np.log(100.0)), (np.log(0.05), np.log(10.0))], [True, True], {0},
        lambda th, Lv: (Lv ** th[1] / (th[0] ** th[1] + Lv ** th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** th[1] / (th[0] ** th[1] + Lx ** th[1]),
        ["K", "h"], "monotone non-decreasing (3 par)")

    # 6. substrate inhibition  r = V L / (K + L + L^2/Ki)   (peak at sqrt(K Ki))
    add("substrate_inhibition", 3,
        [(np.log(0.005), np.log(200.0)), (np.log(0.005), np.log(500.0))], [True, True], {0},
        lambda th, Lv: (Lv / (th[0] + Lv + Lv ** 2 / th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx / (th[0] + Lx + Lx ** 2 / th[1]),
        ["K", "Ki"], "rise then fall, interior peak")

    # 7. plateau-then-linear-decline kink (= flat up to a break then decline)
    add("plateau_then_linear_decline_kink", 3,
        [(np.log(0.02), np.log(30.0))], [True], {1},
        lambda th, Lv: np.column_stack([np.ones_like(Lv), -np.maximum(0.0, Lv - th[0])]),
        lambda th, c, Lx: c[0] - c[1] * np.maximum(0.0, Lx - th[0]),
        ["Lk"], "flat up to Lk then linear decline",
        alias="flat_up_to_break_then_decline")

    # 8. quadratic in L (free, may peak)
    add("quadratic_in_L", 3, [], [], set(),
        lambda th, Lv: np.column_stack([np.ones_like(Lv), Lv, Lv ** 2]),
        lambda th, c, Lx: c[0] + c[1] * Lx + c[2] * Lx ** 2,
        [], "free quadratic, may peak or not")

    # 9. quadratic in log L (free, may peak)
    add("quadratic_in_logL", 3, [], [], set(),
        lambda th, Lv: np.column_stack([np.ones_like(Lv), np.log(Lv), np.log(Lv) ** 2]),
        lambda th, c, Lx: c[0] + c[1] * np.log(Lx) + c[2] * np.log(Lx) ** 2,
        [], "free quadratic in log L")

    # 10. straight line in L (free slope; k=2)
    add("linear_in_L", 2, [], [], set(),
        lambda th, Lv: np.column_stack([np.ones_like(Lv), Lv]),
        lambda th, c, Lx: c[0] + c[1] * Lx,
        [], "straight line, free slope")

    # 11. axis-sensitivity model: monotone non-increasing in log L
    #     r = c0 + s * ln L, s <= 0  ->  basis [1, -lnL], coef >= 0
    add("monotone_decreasing_in_log_axis", 2, [], [], {1},
        lambda th, Lv: np.column_stack([np.ones_like(Lv), -np.log(Lv)]),
        lambda th, c, Lx: c[0] - c[1] * np.log(Lx),
        [], "monotone non-increasing on the log axis")

    # 12. gamma / rise-times-decay  r = A L^p exp(-L/tau)  (peak at p*tau)
    add("gamma_rise_times_decay", 3,
        [(np.log(0.03), np.log(10.0)), (np.log(0.03), np.log(100.0))], [True, True], {0},
        lambda th, Lv: (Lv ** th[0] * np.exp(-Lv / th[1]))[:, None],
        lambda th, c, Lx: c[0] * Lx ** th[0] * np.exp(-Lx / th[1]),
        ["p", "tau"], "rise then fall, interior peak at p*tau")

    return M


MODELS = build_models()

# ----------------------------------------------------------------------------
# information criteria and fit statistics
# ----------------------------------------------------------------------------
TINY = 1e-12 * float(np.sum(y ** 2))


def aic(rss, k):
    return n * np.log(max(rss, TINY) / n) + 2 * k


def aicc(rss, k):
    denom = n - k - 1
    if denom <= 0:
        return None
    return aic(rss, k) + 2 * k * (k + 1) / denom


def bic(rss, k):
    return n * np.log(max(rss, TINY) / n) + k * np.log(n)


def fit_stats(md):
    f = fit_model(md, L, y)
    pred = f["pred"]
    rss = f["rss"]
    k = md["k"]
    r2 = 1.0 - rss / SS_TOT
    if n - k - 1 > 0:
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    else:
        adj = None
    rmse = float(np.sqrt(rss / n))
    # peak location of the fitted curve, on a dense grid
    Lg = np.linspace(L.min() * 0.25, L.max() * 1.25, 20001)
    yg = md["pred_full"](f["theta"], f["coef"], Lg)
    ipk = int(np.argmax(yg))
    peak_L = float(Lg[ipk])
    interior = bool(Lg[0] < peak_L < Lg[-1])
    return dict(name=md["name"], alias=md.get("alias"), shape=md["shape"], k=k,
                param_names=list(md["pnames"]),
                params={p: float(v) for p, v in zip(md["pnames"], f["theta"])},
                linear_coefs=[float(v) for v in f["coef"]],
                rss=float(rss), r2=float(r2), adj_r2=adj, rmse=rmse,
                aic=float(aic(rss, k)), aicc=aicc(rss, k), bic=float(bic(rss, k)),
                df_resid=n - k, interpolates_exactly=bool(rss < TINY),
                pred=[float(v) for v in pred],
                fitted_peak_L=peak_L, fitted_peak_interior=interior,
                fitted_curve_max=float(yg[ipk]))


# ----------------------------------------------------------------------------
# leave-one-out cross validation (refit on 3 points, predict the held-out point)
# ----------------------------------------------------------------------------
def loocv(md):
    errs = []
    preds = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        try:
            f = fit_model(md, L[tr], y[tr], npts=25, rounds=5)
            Lg = np.array([L[i]])
            pi = float(md["pred_full"](f["theta"], f["coef"], Lg)[0])
        except Exception:
            pi = float("nan")
        preds.append(pi)
        errs.append(pi - y[i])
    errs = np.array(errs, float)
    ok = np.isfinite(errs)
    return dict(pred=[float(p) for p in preds],
                rmse=float(np.sqrt(np.mean(errs[ok] ** 2))) if ok.any() else None,
                mae=float(np.mean(np.abs(errs[ok]))) if ok.any() else None,
                max_abs=float(np.max(np.abs(errs[ok]))) if ok.any() else None,
                n_ok=int(ok.sum()))


# ----------------------------------------------------------------------------
# peak-location identifiability: profile fits with the mode anchored at m
# ----------------------------------------------------------------------------
def profile_sub_inhibition(m_grid):
    out, valid = [], []
    for m in m_grid:
        def obj(th):
            K = np.exp(th[0])
            G = (L / (K + L + L ** 2 * K / m ** 2))[:, None]
            rss, _ = lsq_active(G, y, {0})
            return rss
        v, _ = zoom_search(obj, [np.log(0.005)], [np.log(500.0)], npts=41, rounds=4,
                           logdims=[True])
        out.append(float(v))
        valid.append(True)
    return np.array(out), np.array(valid, bool)


def profile_gamma(m_grid):
    out, valid = [], []
    for m in m_grid:
        def obj(th):
            tau = np.exp(th[0])
            p = m / tau
            G = (L ** p * np.exp(-L / tau))[:, None]
            rss, _ = lsq_active(G, y, {0})
            return rss
        v, _ = zoom_search(obj, [np.log(0.02)], [np.log(200.0)], npts=41, rounds=4,
                           logdims=[True])
        out.append(float(v))
        valid.append(True)
    return np.array(out), np.array(valid, bool)


def profile_quadratic_vertex(m_grid):
    out, valid = [], []
    for m in m_grid:
        G = np.column_stack([np.ones_like(L), (L - m) ** 2])
        rss, coef = lsq_active(G, y, set())
        is_peak = coef[1] < 0.0
        out.append(float(rss) if is_peak else float("inf"))
        valid.append(bool(is_peak))
    return np.array(out), np.array(valid, bool)


def summarise_profile(m_grid, rss, valid):
    rss_m = np.where(valid, rss, np.inf)
    i = int(np.argmin(rss_m))
    rmin = float(rss_m[i])
    thr2 = rmin * np.exp(2.0 / n)  # same k -> Delta AIC = n ln(rss/rmin) <= 2
    in2 = (rss_m <= thr2) & valid
    def at(mval):
        j = int(np.argmin(np.abs(m_grid - mval)))
        return float(rss_m[j]) if valid[j] else None
    def aicc2(rssv):
        return aicc(rssv, 2) if rssv is not None and np.isfinite(rssv) else None
    a02, a10 = aicc2(at(0.2)), aicc2(at(1.0))
    return dict(mode_at_min=float(m_grid[i]), rss_min=rmin,
                rss_at_m_0p2=at(0.2), rss_at_m_1p0=at(1.0),
                aicc_at_m_0p2=a02, aicc_at_m_1p0=a10,
                delta_aicc_m0p2_vs_m1p0=(None if (a02 is None or a10 is None) else float(a02 - a10)),
                delta_aic_m0p2_vs_m1p0=(None if (at(0.2) is None or at(1.0) is None)
                                        else float(n * np.log(max(at(0.2), TINY) / max(at(1.0), TINY)))),
                set_within_2_aic_units=[float(v) for v in m_grid[in2]],
                m_lo=float(m_grid[in2].min()) if in2.any() else None,
                m_hi=float(m_grid[in2].max()) if in2.any() else None)


# ----------------------------------------------------------------------------
# monotone (isotonic) baselines: the best possible fit of ANY monotone response
# ----------------------------------------------------------------------------
def pava(yv, increasing):
    vals = list(map(float, yv))
    blocks = [[v] for v in vals]
    means = [v for v in vals]
    while True:
        merged = False
        for i in range(len(means) - 1):
            bad = means[i] > means[i + 1] if increasing else means[i] < means[i + 1]
            if bad:
                new = blocks[i] + blocks[i + 1]
                blocks[i:i + 2] = [new]
                means[i:i + 2] = [sum(new) / len(new)]
                merged = True
                break
        if not merged:
            break
    fit = []
    for b, m in zip(blocks, means):
        fit.extend([m] * len(b))
    return np.array(fit)


iso_inc = pava(y, True)
iso_dec = pava(y, False)
rss_iso_inc = float(np.sum((y - iso_inc) ** 2))
rss_iso_dec = float(np.sum((y - iso_dec) ** 2))

# ----------------------------------------------------------------------------
# permutation tests (n = 4, exact, all 24 orderings)
# ----------------------------------------------------------------------------
rx = stats.rankdata(L)
ry = stats.rankdata(y)


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


perms = list(itertools.permutations(range(n)))
rho_obs = spearman(rx, ry)
p_rho = float(np.mean([abs(spearman(rx, ry[list(p)])) >= abs(rho_obs) - 1e-12 for p in perms]))
tau_obs = float(stats.kendalltau(L, y).statistic)
p_tau = float(np.mean([abs(stats.kendalltau(L, y[list(p)]).statistic) >= abs(tau_obs) - 1e-12
                       for p in perms]))
# permutation of the crude "interior maximum" event (max at the 2nd or 3rd point)
p_interior_max = float(np.mean([int(np.argmax(y[list(p)])) in (1, 2) for p in perms]))
# the specific rise-then-fall pattern: y1<y2, y2>y3, y3>y4
p_rise_then_fall = float(np.mean([(y[list(p)][0] < y[list(p)][1]) and
                                  (y[list(p)][1] > y[list(p)][2]) and
                                  (y[list(p)][2] > y[list(p)][3]) for p in perms]))

# ----------------------------------------------------------------------------
# run everything
# ----------------------------------------------------------------------------
results = []
for md in MODELS:
    st = fit_stats(md)
    st["loocv"] = loocv(md)
    results.append(st)

# indistinguishable sets
best_aicc = min([r["aicc"] for r in results if r["aicc"] is not None])
set_aicc = [r["name"] for r in results
            if r["aicc"] is not None and r["aicc"] - best_aicc <= 2.0]
best_aic = min(r["aic"] for r in results)
set_aic = [dict(name=r["name"], delta=float(r["aic"] - best_aic))
           for r in sorted(results, key=lambda r: r["aic"]) if r["aic"] - best_aic <= 2.0]
best_loo = min(r["loocv"]["rmse"] for r in results if r["loocv"]["rmse"] is not None)
set_loo = [dict(name=r["name"], loocv_rmse=r["loocv"]["rmse"])
           for r in sorted(results, key=lambda r: r["loocv"]["rmse"])
           if r["loocv"]["rmse"] - best_loo <= 2.0]

# rank ordering for the report
order_aic = [(r["name"], r["rss"], r["k"], r["aic"]) for r in sorted(results, key=lambda r: r["aic"])]

# peak profiles
m_grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 220))
p_sub = summarise_profile(m_grid, *profile_sub_inhibition(m_grid))
p_gam = summarise_profile(m_grid, *profile_gamma(m_grid))
p_qua = summarise_profile(m_grid, *profile_quadratic_vertex(m_grid))
profiles = dict(m_grid=[float(v) for v in m_grid],
                substrate_inhibition=dict(profile=p_sub,
                                          rss=[float(v) for v in profile_sub_inhibition(m_grid)[0]]),
                gamma=dict(profile=p_gam, rss=[float(v) for v in profile_gamma(m_grid)[0]]),
                quadratic_vertex=dict(profile=p_qua,
                                      rss=[float(v) for v in profile_quadratic_vertex(m_grid)[0]]))

# axis sensitivity: monotone models on the log axis vs the best peaked models
axis_test = dict(
    isotonic_increasing_rss=rss_iso_inc,
    isotonic_decreasing_rss=rss_iso_dec,
    isotonic_decreasing_fit=[float(v) for v in iso_dec],
    isotonic_increasing_fit=[float(v) for v in iso_inc],
    log_axis_values=[float(v) for v in np.log(L)],
    best_any_monotone_rss=min(rss_iso_inc, rss_iso_dec),
    note=("The isotonic (PAVA) fits are the minimum-RSS fits over ALL monotone "
          "functions of L. Under any strictly increasing reparametrisation h(L) "
          "(log included) the ordering of the four points is unchanged, so the "
          "isotonic RSS is also the best monotone fit on the transformed axis: "
          "no monotone axis rescaling can remove the interior peak."),
)

out = dict(
    data=dict(file=os.path.relpath(DATA, BASE),
              loading_percent=[float(v) for v in L],
              rate_umol_per_h=[float(v) for v in y],
              n_points=n,
              note="No replicates: one complete loading scan is the analysis unit. "
                   "The four points are NOT independent replicates of anything."),
    model_table=results,
    indistinguishable_sets=dict(
        by_aicc_within_2=set_aicc,
        by_aicc_note=("AICc = AIC + 2k(k+1)/(n-k-1) is undefined for k=3 when n=4 "
                      "(denominator zero); the finite-AICc ranking therefore contains "
                      "only k<=2 models, all of which are monotone."),
        by_aic_within_2=set_aic,
        by_loocv_within_2=set_loo,
        aic_ranking=order_aic),
    peak_location=dict(profiles=profiles,
                       shape_only_bound="(0.2, 3.0)",
                       bound_derivation=("For a continuous single-peaked curve: rate(1.0) > "
                                         "rate(0.2) forces mode > 0.2; rate(1.0) > rate(3.0) "
                                         "forces mode < 3.0. Any mode in (0.2, 3.0) is "
                                         "consistent with the four values.")),
    axis_sensitivity=axis_test,
    permutation_tests=dict(n_perm=len(perms), spearman_rho=rho_obs, p_two_sided_spearman=p_rho,
                           kendall_tau=tau_obs, p_two_sided_kendall=p_tau,
                           p_interior_max_at_all=p_interior_max,
                           p_rise_then_fall_pattern=p_rise_then_fall,
                           note="n=4 -> exact permutation p-values; the smallest attainable "
                                "two-sided p for a monotone association is 1/3."),
    limitations=[
        "n = 4 points, no replicates, no independent error estimate, residual df = 1 for "
        "any 3-parameter model: the noise level is not estimable from these data.",
        "AICc is undefined for k=3 at n=4 (n-k-1 = 0). AICc cannot rank the flexible, "
        "peaked descriptions; any AICc-based conclusion here is an artefact of the "
        "denominator, not evidence.",
        "With 3 parameters and 4 points, several families can interpolate the data "
        "exactly (RSS ~ 0), so AIC -> -infinity and the comparison is degenerate.",
        "The four points carry 4 numbers; they cannot distinguish among all shapes "
        "that pass near those four numbers.",
    ],
)

os.makedirs(OUTDIR, exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2)

# ----------------------------------------------------------------------------
# console summary
# ----------------------------------------------------------------------------
print("n =", n)
print("AICc finite best:", best_aicc, "set:", set_aicc)
print()
print(f"{'model':38s} {'k':>2s} {'RSS':>12s} {'R2':>8s} {'RMSE':>9s} {'AIC':>10s} "
      f"{'AICc':>10s} {'LOO':>10s}")
for r in sorted(results, key=lambda r: r["aic"]):
    ac = "undef" if r["aicc"] is None else f"{r['aicc']:.2f}"
    lo = "nan" if r["loocv"]["rmse"] is None else f"{r['loocv']['rmse']:.1f}"
    print(f"{r['name']:38s} {r['k']:2d} {r['rss']:12.2f} {r['r2']:8.4f} {r['rmse']:9.1f} "
          f"{r['aic']:10.2f} {ac:>10s} {lo:>10s}")
print()
print("isotonic RSS increasing:", round(rss_iso_inc, 1), " decreasing:", round(rss_iso_dec, 1))
print("best peaked-family RSS at data:", min(r["rss"] for r in results))
print()
for nm, p in [("sub_inhibition", p_sub), ("gamma", p_gam), ("quad_vertex", p_qua)]:
    print(nm, "mode*=", round(p["mode_at_min"], 3),
          "RSS(0.2)=", None if p["rss_at_m_0p2"] is None else round(p["rss_at_m_0p2"], 1),
          "RSS(1.0)=", None if p["rss_at_m_1p0"] is None else round(p["rss_at_m_1p0"], 1),
          "dAICc(0.2-1.0)=", None if p["delta_aicc_m0p2_vs_m1p0"] is None else round(p["delta_aicc_m0p2_vs_m1p0"], 2),
          "2-unit set=[", None if p["m_lo"] is None else round(p["m_lo"], 3), ",",
          None if p["m_hi"] is None else round(p["m_hi"], 3), "]")
print()
print("permutation: rho=", round(rho_obs, 3), "p=", p_rho, " tau=", round(tau_obs, 3),
      "p=", p_tau, " p(interior max)=", p_interior_max, " p(rise-then-fall)=", p_rise_then_fall)
print("wrote", OUT)