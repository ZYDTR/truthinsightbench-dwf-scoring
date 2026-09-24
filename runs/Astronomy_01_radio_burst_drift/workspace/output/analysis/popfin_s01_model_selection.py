#!/usr/bin/env python3
"""
popfin_s01_model_selection.py  -- tag "popfin"

DIMENSION: population structure of the drift-rate distributions in
TruthInsightBench Astronomy_01_radio_burst_drift.

DATA: data/normalized_views/drift_population_series.tsv
  948 rows = 6 population series x {fast, slow} regimes, INTEGER event counts per
  drift-rate bin.  ANALYSIS UNIT: one (population series, drift regime) = one
  workbook block (fast) or the union of two workbook blocks (slow).  There are 6
  such units and, for event-level questions, only 5 independent burst events.
  Bins are nested repeats of the same unit, never independent samples.

MODELS (fit to the ORIGINAL bin grid of each series; Poisson likelihood on the
binned counts):

  M1 LOGN   lam_i = A * w_i * LN(s_i | mu, sigma)                k = 2
  M2 MIX2   lam_i = A * w_i * [p*LN(s_i|m1,s1) + (1-p)*LN(s_i|m2,s2)]  k = 5
  M3 POW    lam_i = w_i * C * s_i^(-alpha)                        k = 2
  M4 EXP    lam_i = w_i * C * exp(-lambda s_i)                    k = 2

  A (M1/M2) and C (M3/M4) are scale parameters.  The Poisson MLE of a free
  overall scale is the observed total for ANY shape, and both are profiled out in
  closed form here, so no scale parameter is counted in k.  Counting it would add
  +1 to EVERY model and would not change any dAIC/dBIC or any ranking.  The
  absolute likelihood values therefore differ from a "full" Poisson likelihood by
  the constant sum_i log(n_i!), which is identical for all four models.

  w_i = bin width taken as the difference of contiguous midpoint edges.  All
  grids are contiguous, so this is midpoint-rule quadrature of the underlying
  continuous density (a few slow grids have a partially-halved final bin).

OPTIMISATION: (a) dense brute-force grid over the shape parameters; (b) L-BFGS-B
polish with ANALYTIC gradients of the best grid cells; (c) seeded random starts in
the smooth re-parameterisation, also polished.  Fixed seed.  Plain multi-start
L-BFGS-B with numerical gradients on a barrier parameterisation was verified to
under-converge here (it returned a mixture NLL higher than the single-component
NLL, which is impossible since M1 is nested in M2).

DEGENERACY GUARD for M2: flagged when a weight < 0.02, a sigma < 0.03 dex, a
median leaves the grid support, or a sigma sits at its bound.

BOOTSTRAP (falsification step): counts are simulated from the best-fit SINGLE
log-normal of each series, M1 and M2 are refitted with the same optimiser, and
the rate at which BIC prefers the two-component model is recorded.

OUTPUTS: output/results/popfin_s01_*.csv + popfin_s01_fit_table.json,
         output/results/figures/popfin_s01_*.png, output/logs/popfin_s01.log
"""
import csv
import gc
import json
import math
import os
import time
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
IN_TSV = os.path.join(ROOT, "data/normalized_views/drift_population_series.tsv")
OUT_RES = os.path.join(ROOT, "output/results")
OUT_FIG = os.path.join(OUT_RES, "figures")
LOG = os.path.join(ROOT, "output/logs/popfin_s01.log")

SEED = 20221013
LN10 = math.log(10.0)
C0 = -0.5 * math.log(2.0 * math.pi) - LN10  # -log(sigma*ln10) - 0.5log(2pi), sigma in dex
LOG_LINES = []


def log(msg):
    s = str(msg)
    LOG_LINES.append(s)
    print(s, flush=True)


# ---------------------------------------------------------------------------
# problem bundle + Poisson objective
# ---------------------------------------------------------------------------
class Prob:
    """Pre-computed per-series quantities so the objective costs only a few
    numpy ops on small arrays (arrays are <= 139 bins, so overhead dominates)."""
    __slots__ = ("s", "l10", "logs", "w", "logw", "n", "A", "N")

    def __init__(self, c, n, w):
        self.s = c
        self.l10 = np.log10(c)
        self.logs = np.log(c)
        self.w = w
        self.logw = np.log(w)
        self.n = n
        self.A = float(np.sum(n))
        self.N = float(np.sum(n))


def poisson_nll(loglam, n):
    """-log L up to the constant sum log(n!); lam = exp(loglam) clipped at -700 so
    a bin whose model intensity underflows to 0 contributes nothing when empty."""
    ll = np.clip(loglam, -700.0, 700.0)
    return float(np.sum(np.exp(ll)) - np.dot(n, ll))


CONST_LOG = None  # placeholder; the scale constant is applied inside each nll


def _m1_parts(theta, P):
    mu, ls = theta
    s1 = 10.0 ** ls
    u = P.l10 - mu
    lf = C0 - ls * LN10 - u * u / (2.0 * s1 * s1) - P.logs
    return mu, ls, s1, u, lf


def nll_M1(theta, P):
    _, _, _, _, lf = _m1_parts(theta, P)
    return poisson_nll(math.log(P.A) + P.logw + lf, P.n)


def nll_M1_g(theta, P):
    """value AND analytic gradient, both on the SAME scale as nll_M1
    (including the log(A) and log(w) terms, which are constants in theta but must
    not be dropped or the reported nll would be offset by a theta-independent
    amount and dBIC between models would be corrupted)."""
    mu, ls, s1, u, lf = _m1_parts(theta, P)
    loglam = math.log(P.A) + P.logw + lf
    lam = np.exp(np.clip(loglam, -700, 700))
    f = float(np.sum(lam) - np.dot(P.n, loglam))
    d = lam - P.n
    g_mu = float(np.dot(d, u) / (s1 * s1))
    g_ls = float(np.dot(d, LN10 * (u * u / (s1 * s1) - 1.0)))
    return f, np.array([g_mu, g_ls])


def _m2_parts(theta, P):
    m, ld, ls1, ls2, lp = theta
    d = 10.0 ** ld
    s1 = 10.0 ** ls1
    s2 = 10.0 ** ls2
    u1 = P.l10 - (m - d)
    u2 = P.l10 - (m + d)
    lf1 = C0 - ls1 * LN10 - u1 * u1 / (2.0 * s1 * s1) - P.logs
    lf2 = C0 - ls2 * LN10 - u2 * u2 / (2.0 * s2 * s2) - P.logs
    p = 1.0 / (1.0 + math.exp(-lp))
    e1 = np.exp(np.clip(lf1, -700, 0))
    e2 = np.exp(np.clip(lf2, -700, 0))
    f_ = np.maximum(p * e1 + (1.0 - p) * e2, 1e-300)
    return m, ld, ls1, ls2, lp, d, s1, s2, u1, u2, p, e1, e2, f_


def nll_M2(theta, P):
    """Smooth parameterisation: medians = 10**(m -+ d), d = 10**ld, so the
    ordering mu1 <= mu2 is automatic and the surface has no barrier."""
    *_, f_ = _m2_parts(theta, P)
    return poisson_nll(math.log(P.A) + P.logw + np.log(f_), P.n)


def nll_M2_g(theta, P):
    (m, ld, ls1, ls2, lp, d, s1, s2, u1, u2, p, e1, e2, f_) = _m2_parts(theta, P)
    logf = np.log(f_)
    loglam = math.log(P.A) + P.logw + logf
    lam = np.exp(np.clip(loglam, -700, 700))
    nll = float(np.sum(lam) - np.dot(P.n, loglam))
    a1 = p * e1
    a2 = (1.0 - p) * e2
    dd = 10.0 ** ld * LN10
    g1 = u1 / (s1 * s1)
    g2 = u2 / (s2 * s2)
    df = np.empty((5, len(P.s)))
    df[0] = a1 * g1 + a2 * g2                        # d/dm
    df[1] = -a1 * g1 * dd + a2 * g2 * dd             # d/dld
    df[2] = a1 * LN10 * (u1 * u1 / (s1 * s1) - 1.0)  # d/dl10s1
    df[3] = a2 * LN10 * (u2 * u2 / (s2 * s2) - 1.0)  # d/dl10s2
    df[4] = p * (1.0 - p) * (e1 - e2)                # d/dlogit_p
    grad = df @ ((lam - P.n) / f_)
    return nll, grad




# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
B1 = [(-3.0, 0.35), (-2.2, 0.35)]  # bounds (mu, log10 sigma) for M1
B2 = [(-3.0, 1.0), (-2.7, 0.6), (-2.3, 0.35), (-2.3, 0.35), (-9.0, 9.0)]  # M2 smooth


def _polish(fn, P, starts, bounds, n_top, maxiter, tol):
    """fn(theta, P) -> (nll, grad); polish the n_top best grid cells."""
    starts = sorted(starts, key=lambda t: t[1])[:n_top]
    sols = []
    for x0, _ in starts:
        r = minimize(fn, np.asarray(x0, float), args=(P,), jac=True,
                     method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=tol, gtol=tol * 1e-2))
        if np.all(np.isfinite(r.x)):
            sols.append((float(r.fun), np.asarray(r.x, float)))
    if not sols:
        return None, 1e300, 0, 0
    sols.sort(key=lambda t: t[0])
    best = sols[0][0]
    nbest = sum(1 for f, _ in sols if f <= best + max(1e-6, 1e-7 * abs(best)))
    return sols[0][1], best, len(sols), nbest


def fit_M1(P, level="full"):
    cfg = {"full": dict(nmu=80, nls=60, ntop=8, maxiter=800, tol=1e-12, nrand=0),
           "boot": dict(nmu=30, nls=18, ntop=3, maxiter=200, tol=1e-9, nrand=4)}[level]
    mus = np.linspace(math.log10(P.s[0]) - 1.0, math.log10(P.s[-1]) + 0.2, cfg["nmu"])
    lss = np.linspace(math.log10(0.01), math.log10(2.0), cfg["nls"])
    starts = []
    base = math.log(P.A) + P.logw + C0 - P.logs
    for ls in lss:
        s1 = 10.0 ** ls
        pre = base - ls * LN10
        for mu in mus:
            u = P.l10 - mu
            starts.append(((mu, ls),
                           poisson_nll(pre - u * u / (2.0 * s1 * s1), P.n)))
    if cfg["nrand"]:
        rng = np.random.default_rng(SEED + 7 + int(P.N) % 89)
        for _ in range(cfg["nrand"]):
            starts.append(((rng.uniform(mus[0], mus[-1]), rng.uniform(lss[0], lss[-1])), 1e299))
    x, f, nopt, nbest = _polish(nll_M1_g, P, starts, B1, cfg["ntop"],
                                cfg["maxiter"], cfg["tol"])
    return dict(x=x, nll=f, n_polished=nopt, n_at_best=nbest, n_grid=len(starts))


def fit_M2(P, level="full"):
    cfg = {"full": dict(nmed=18, sigs=(0.05, 0.12, 0.25, 0.45, 0.75), npis=13,
                        nrand=80, ntop=16, maxiter=900, tol=1e-12),
           "boot": dict(nmed=14, sigs=(0.05, 0.12, 0.25, 0.45, 0.75), npis=9,
                        nrand=25, ntop=8, maxiter=400, tol=1e-10)}[level]
    sigs = cfg["sigs"]
    pis = np.linspace(0.05, 0.95, cfg["npis"])
    meds = np.unique(np.concatenate([
        np.geomspace(max(P.s[0] / 40.0, 1e-3), P.s[-1] * 1.5, cfg["nmed"]),
        P.s[[0, len(P.s) // 2, -1]]]))
    mus = np.log10(meds)
    base = math.log(P.A) + P.logw + C0 - P.logs

    starts = []
    for i1 in range(len(mus)):
        for i2 in range(i1, len(mus)):
            for sg1 in sigs:
                u1 = P.l10 - mus[i1]
                e1 = np.exp(np.clip(-math.log(sg1) - u1 * u1 / (2 * sg1 ** 2), -700, 0))
                for sg2 in sigs:
                    u2 = P.l10 - mus[i2]
                    e2 = np.exp(np.clip(-math.log(sg2) - u2 * u2 / (2 * sg2 ** 2), -700, 0))
                    for p in pis:
                        mix = np.maximum(p * e1 + (1 - p) * e2, 1e-300)
                        starts.append(((0.5 * (mus[i1] + mus[i2]),
                                        math.log10(max(0.5 * abs(mus[i2] - mus[i1]), 1e-3)),
                                        math.log10(sg1), math.log10(sg2),
                                        math.log(p / (1 - p))),
                                       poisson_nll(base + np.log(mix), P.n)))
                    if i1 == i2:
                        break
    n_grid = len(starts)
    rng = np.random.default_rng(SEED + 101 + int(P.N) % 97)
    for _ in range(cfg["nrand"]):
        starts.append(((rng.uniform(mus[0] - 0.4, mus[-1]),
                        rng.uniform(-2.0, -0.15),
                        rng.uniform(math.log10(0.04), math.log10(0.9)),
                        rng.uniform(math.log10(0.04), math.log10(0.9)),
                        rng.uniform(-3, 3)), 1e299))
    x, f, nopt, nbest = _polish(nll_M2_g, P, starts, B2, cfg["ntop"],
                                cfg["maxiter"], cfg["tol"])
    return dict(x=x, nll=f, n_polished=nopt, n_at_best=nbest, n_grid=n_grid,
                n_starts=len(starts))


def nll_M3(alpha, P):
    g = float(np.sum(P.w * np.exp(-alpha * P.logs)))
    if not np.isfinite(g) or g <= 0:
        return 1e300
    return float(P.N - np.dot(P.n, P.logw + math.log(P.N / g) - alpha * P.logs))


def nll_M4(b, P):
    g = float(np.sum(P.w * np.exp(-b * P.s)))
    if not np.isfinite(g) or g <= 0:
        return 1e300
    return float(P.N - np.dot(P.n, P.logw + math.log(P.N / g) - b * P.s))


def fit_M3M4(P, which, level="full"):
    fn = nll_M3 if which == "M3" else nll_M4
    step = 0.004 if level == "full" else 0.03
    grid = np.arange(-6.0, 6.0 + step, step)
    vals = np.array([fn(a, P) for a in grid])
    i = int(np.argmin(vals))
    lo = grid[max(0, i - 2)]
    hi = grid[min(len(grid) - 1, i + 2)]
    r = minimize_scalar(fn, args=(P,), bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-10))
    x = float(r.x) if np.isfinite(r.fun) and r.fun <= vals[i] else float(grid[i])
    f = float(fn(x, P))
    nloc = int(np.sum(np.diff(np.sign(np.diff(vals))) > 0))
    return dict(x=np.array([x]), nll=f, n_polished=1, n_at_best=1, n_grid=len(grid),
                n_local_minima_on_grid=nloc)


MODELS = ("M1_lognormal", "M2_lognormal_mixture", "M3_powerlaw", "M4_exponential")
KPAR = {"M1_lognormal": 2, "M2_lognormal_mixture": 5,
        "M3_powerlaw": 2, "M4_exponential": 2}


def bin_widths(c):
    d = np.diff(c)
    e = np.concatenate([[c[0] - d[0] / 2.0], (c[:-1] + c[1:]) / 2.0,
                        [c[-1] + d[-1] / 2.0]])
    return np.diff(e)


def phys_M2(theta):
    m, ld, ls1, ls2, lp = theta
    d = 10.0 ** ld
    p = 1.0 / (1.0 + math.exp(-lp))
    return p, 10.0 ** (m - d), 10.0 ** ls1, 10.0 ** (m + d), 10.0 ** ls2


def degeneracy_flags(p, m1, s1, m2, s2, c):
    lo, hi = float(c[0]), float(c[-1])
    fl = []
    if p < 0.02 or p > 0.98:
        fl.append("weight<0.02")
    elif p < 0.05 or p > 0.95:
        fl.append("weight<0.05")
    if s1 < 0.03 or s2 < 0.03:
        fl.append("sigma<0.03dex")
    if m1 < lo or m2 < lo:
        fl.append("median_below_grid_edge" + ("_far" if min(m1, m2) < lo / 2 else ""))
    if m2 > hi:
        fl.append("median_above_grid_edge")
    if s1 > 1.9 or s2 > 1.9:
        fl.append("sigma_at_bound")
    return ";".join(fl)


def gic(nll, k, N):
    return -2.0 * nll + 2.0 * k, -2.0 * nll + k * math.log(max(N, 2.0))


def concentration_stats(n):
    tot = float(np.sum(n))
    N = len(n)
    srt = np.sort(n)[::-1]
    out = dict(frac_top1_bin=float(srt[:1].sum() / tot),
               frac_top3_bins=float(srt[:3].sum() / tot),
               frac_top5_bins=float(srt[:5].sum() / tot),
               HHI_allgrid=float(np.sum((n / tot) ** 2)),
               n_bins=int(N), n_occupied=int((n > 0).sum()),
               frac_occupied=float((n > 0).sum() / N), total_events=tot)
    out["gini_allgrid"] = float(2.0 * np.sum(np.arange(1, N + 1) * np.sort(n)) / (N * tot)
                                - (N + 1.0) / N)
    oc = np.sort(n[n > 0])
    No = len(oc)
    out["gini_occupied_only"] = float(2.0 * np.sum(np.arange(1, No + 1) * oc)
                                      / (No * oc.sum()) - (No + 1.0) / No)
    p = n[n > 0] / tot
    out["shannon_H_nats"] = float(-np.sum(p * np.log(p)))
    out["shannon_live_bins"] = float(math.exp(out["shannon_H_nats"]))
    out["shannon_live_bins_occ"] = float(math.exp(
        -np.sum((oc / oc.sum()) * np.log(oc / oc.sum()))))
    out["bic_resolvable_bins"] = float(tot / math.log(max(tot, 2.0)))
    return out


def bootstrap_false_pref(P, nboot, level="boot"):
    mu_sim = np.maximum(P.A * P.w * np.exp(np.clip(
        C0 - math.log(10.0 ** 0.0) - (P.l10 - 0.0) ** 2, -700, 0)), 0.0)
    return mu_sim  # replaced in main(); kept for clarity


# ---------------------------------------------------------------------------
def load_series():
    by = defaultdict(list)
    with open(IN_TSV, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by[(r["population_series_id"], r["drift_regime"])].append(
                (float(r["distribution_bin_center"]), int(float(r["event_count"]))))
    return {k: (np.array([x for x, _ in sorted(v)], float),
                np.array([y for _, y in sorted(v)], float)) for k, v in by.items()}


def run_bootstrap(c, n, w, x1_hat, nboot, level="boot"):
    P = Prob(c, n, w)
    mu_hat, ls_hat = x1_hat
    s1 = 10.0 ** ls_hat
    u = P.l10 - mu_hat
    lf = C0 - ls_hat * LN10 - u * u / (2 * s1 * s1) - P.logs
    mu_sim = np.maximum(P.A * P.w * np.exp(np.clip(math.log(P.A) * 0 + lf, -700, 0)), 0.0)
    mu_sim = np.maximum(np.exp(np.clip(math.log(P.A) + P.logw + lf, -700, 0)), 1e-12)
    rng = np.random.default_rng(SEED + 777)
    dbic, daic = [], []
    ndeg = 0
    for _ in range(nboot):
        nb = rng.poisson(mu_sim).astype(float)
        if nb.sum() <= 0:
            continue
        Pb = Prob(c, nb, w)
        r1 = fit_M1(Pb, level)
        r2 = fit_M2(Pb, level)
        if r1["x"] is None or r2["x"] is None:
            continue
        N = float(nb.sum())
        dbic.append((-2 * r2["nll"] + 5 * math.log(N)) - (-2 * r1["nll"] + 2 * math.log(N)))
        daic.append((-2 * r2["nll"] + 10) - (-2 * r1["nll"] + 4))
        p_, m1p, s1p, m2p, s2p = phys_M2(r2["x"])
        if degeneracy_flags(p_, m1p, s1p, m2p, s2p, c):
            ndeg += 1
    dbic = np.array(dbic)
    daic = np.array(daic)
    return dict(n_ok=len(dbic), n_boot=nboot,
                rate_bic=float(np.mean(dbic < 0)) if len(dbic) else float("nan"),
                rate_aic=float(np.mean(daic < 0)) if len(daic) else float("nan"),
                median_dBIC=float(np.median(dbic)) if len(dbic) else float("nan"),
                mean_dBIC=float(np.mean(dbic)) if len(dbic) else float("nan"),
                sd_dBIC=float(np.std(dbic)) if len(dbic) else float("nan"),
                p05_dBIC=float(np.percentile(dbic, 5)) if len(dbic) else float("nan"),
                p95_dBIC=float(np.percentile(dbic, 95)) if len(dbic) else float("nan"),
                n_degenerate=ndeg, samples=[float(v) for v in dbic])


def main():
    t_start = time.time()
    log("=" * 82)
    log("popfin_s01_model_selection   seed=%d   %s" % (SEED, time.strftime("%Y-%m-%d %H:%M")))
    log("models: M1 log-normal k=2 | M2 two-component log-normal mixture k=5 |")
    log("        M3 power law k=2 | M4 exponential k=2")
    log("likelihood: Poisson on binned counts, original per-series grid; the overall")
    log("scale is profiled in closed form (not counted in k; identical convention for")
    log("all models, so all dAIC/dBIC are unaffected)")
    log("analysis unit: (population series, drift regime); 6 series x 2 regimes; the")
    log("event-level sample is 5 bursts; bins are nested repeats of one unit")
    log("=" * 82)
    series = load_series()

    fit_rows, param_rows, conc_rows, boot_rows, cal_rows = [], [], [], [], []
    store = {}

    for key in sorted(series):
        sid, regime = key
        c, n = series[key]
        w = bin_widths(c)
        P = Prob(c, n, w)
        tot = P.N
        t0 = time.time()
        r1 = fit_M1(P, "full")
        r2 = fit_M2(P, "full")
        r3 = fit_M3M4(P, "M3", "full")
        r4 = fit_M3M4(P, "M4", "full")
        fits = dict(M1_lognormal=r1, M2_lognormal_mixture=r2,
                    M3_powerlaw=r3, M4_exponential=r4)
        t_fit = time.time() - t0
        assert r1["x"] is not None and r2["x"] is not None
        assert r2["nll"] <= r1["nll"] + 1e-7, (
            "mixture must nest the single log-normal: %.8f > %.8f" % (r2["nll"], r1["nll"]))

        aic = {m: gic(fits[m]["nll"], KPAR[m], tot)[0] for m in MODELS}
        bic = {m: gic(fits[m]["nll"], KPAR[m], tot)[1] for m in MODELS}
        rank = sorted(MODELS, key=lambda m: bic[m])

        mu1, ls1 = r1["x"]
        sig1 = 10.0 ** ls1
        lam = {}
        lam["M1_lognormal"] = P.A * w * np.exp(np.clip(
            C0 - ls1 * LN10 - (P.l10 - mu1) ** 2 / (2 * sig1 ** 2) - P.logs, -700, 0))
        p1, m1p, s1p, m2p, s2p = phys_M2(r2["x"])
        lf1 = C0 - math.log(s1p) - (P.l10 - math.log10(m1p)) ** 2 / (2 * s1p ** 2) - P.logs
        lf2 = C0 - math.log(s2p) - (P.l10 - math.log10(m2p)) ** 2 / (2 * s2p ** 2) - P.logs
        mx = p1 * np.exp(np.clip(lf1, -700, 0)) + (1 - p1) * np.exp(np.clip(lf2, -700, 0))
        lam["M2_lognormal_mixture"] = P.A * w * mx
        a3 = float(r3["x"][0])
        g3 = float(np.sum(w * np.power(c, -a3)))
        lam["M3_powerlaw"] = w * (tot / g3) * np.power(c, -a3)
        b4 = float(r4["x"][0])
        g4 = float(np.sum(w * np.exp(-b4 * c)))
        lam["M4_exponential"] = w * (tot / g4) * np.exp(-b4 * c)

        log("-" * 80)
        log("%s / %s : %d bins (%d occupied), %d events, median bin width %.5f [%.1fs]"
            % (sid, regime, len(c), int((n > 0).sum()), tot, np.median(w), t_fit))
        for m in MODELS:
            if m == "M1_lognormal":
                pr = dict(median_MHz_s=10.0 ** mu1, sigma_dex=sig1)
            elif m == "M2_lognormal_mixture":
                pr = dict(comp1_weight=p1, comp1_median_MHz_s=m1p, comp1_sigma_dex=s1p,
                          comp2_weight=1 - p1, comp2_median_MHz_s=m2p, comp2_sigma_dex=s2p)
            elif m == "M3_powerlaw":
                pr = dict(alpha=a3)
            else:
                pr = dict(lambda_per_MHz_s=b4)
            deg = degeneracy_flags(p1, m1p, s1p, m2p, s2p, c) if m == "M2_lognormal_mixture" else ""
            chi2 = float(np.sum((n - lam[m]) ** 2 / lam[m]))
            dof = max(len(c) - KPAR[m], 1)
            log("  %-21s k=%d nll=%13.4f AIC=%13.4f BIC=%13.4f chi2/df=%7.2f %s"
                % (m, KPAR[m], fits[m]["nll"], aic[m], bic[m], chi2 / dof,
                   ("DEGENERATE[" + deg + "]") if deg else ""))
            log("      " + "  ".join("%s=%.6g" % (a, b) for a, b in pr.items()))
            fit_rows.append(dict(series=sid, regime=regime, model=m, k=KPAR[m],
                                 nll=fits[m]["nll"], AIC=aic[m], BIC=bic[m],
                                 dAIC_vs_best=aic[m] - aic[rank[0]],
                                 dBIC_vs_best=bic[m] - bic[rank[0]],
                                 rank_by_BIC=rank.index(m) + 1,
                                 pearson_chi2=chi2, pearson_chi2_over_df=chi2 / dof,
                                 degeneracy=deg, n_bins=len(c),
                                 n_occupied=int((n > 0).sum()), total_events=tot,
                                 n_grid_evals=fits[m]["n_grid"],
                                 n_polished=fits[m]["n_polished"],
                                 n_at_best=fits[m]["n_at_best"]))
            for a, b in pr.items():
                param_rows.append(dict(series=sid, regime=regime, model=m, param=a, value=b))
        log("  BIC ranking (best first): " + " < ".join(rank))
        dB = bic["M2_lognormal_mixture"] - bic["M1_lognormal"]
        dA = aic["M2_lognormal_mixture"] - aic["M1_lognormal"]
        log("  M2 vs M1: dAIC=%+.3f dBIC=%+.3f -> %s; best=%-21s (dBIC to runner-up %+.2f)"
            % (dA, dB, "M2" if dB < 0 else "M1", rank[0], bic[rank[1]] - bic[rank[0]]))

        cc = concentration_stats(n)
        cc.update(dict(series=sid, regime=regime, best_model_BIC=rank[0],
                       dBIC_best_vs_runnerup=float(bic[rank[1]] - bic[rank[0]]),
                       dBIC_M2_vs_M1=float(dB), dAIC_M2_vs_M1=float(dA),
                       M1_median_MHz_s=float(10.0 ** mu1), M1_sigma_dex=float(sig1),
                       M2_degenerate_flag=degeneracy_flags(p1, m1p, s1p, m2p, s2p, c)))
        conc_rows.append(cc)
        log("  concentration: top1=%.3f top3=%.3f top5=%.3f | HHI=%.4f Gini=%.4f | "
            "H=%.3f nats e^H=%.2f live bins of %d | N/lnN=%.1f"
            % (cc["frac_top1_bin"], cc["frac_top3_bins"], cc["frac_top5_bins"],
               cc["HHI_allgrid"], cc["gini_allgrid"], cc["shannon_H_nats"],
               cc["shannon_live_bins"], cc["n_bins"], cc["bic_resolvable_bins"]))
        store[key] = (c, n, w, fits, lam, dict(p1=p1, m1=m1p, s1=s1p, m2=m2p, s2=s2p),
                      cc["shannon_H_nats"])
        del P
        gc.collect()

    # ---- optimiser calibration on the REAL data (reduced bootstrap grid) ----
    log("")
    log("optimiser calibration: reduced 'boot' grid on the REAL data vs the full grid")
    for key in sorted(series):
        sid, regime = key
        c, n, w, fits, lam, mp, _ = store[key]
        Pb = Prob(c, n, w)
        f1 = fit_M1(Pb, "boot")
        f2 = fit_M2(Pb, "boot")
        d1 = f1["nll"] - fits["M1_lognormal"]["nll"]
        d2 = f2["nll"] - fits["M2_lognormal_mixture"]["nll"]
        cal_rows.append(dict(series=sid, regime=regime,
                             dNLL_M1_boot_minus_full=float(d1),
                             dNLL_M2_boot_minus_full=float(d2)))
        log("  %s/%s  dNLL(M1)=%.5f dNLL(M2)=%.5f" % (sid, regime, d1, d2))
    bad = [r for r in cal_rows if r["dNLL_M1_boot_minus_full"] > 0.5
           or r["dNLL_M2_boot_minus_full"] > 0.5]
    log("  reduced grid matched the full-grid optimum (within 0.5 nll units) for "
        "%d/%d series-regimes" % (len(cal_rows) - len(bad), len(cal_rows)))

    # ---- bootstrap ----
    log("")
    log("=" * 82)
    log("PARAMETRIC BOOTSTRAP (falsification step): counts simulated from the best-fit")
    log("SINGLE log-normal; M1 and M2 refitted with the same optimiser; record the rate")
    log("at which BIC prefers the 2-component model when the truth is one component.")
    log("=" * 82)
    NBOOT = {"fast": 200, "slow": 150}
    for key in sorted(series):
        sid, regime = key
        c, n, w, fits, lam, mp, _ = store[key]
        nb = NBOOT[regime]
        t0 = time.time()
        res = run_bootstrap(c, n, w, fits["M1_lognormal"]["x"], nb, "boot")
        tot = float(np.sum(n))
        obs_dB = gic(fits["M2_lognormal_mixture"]["nll"], 5, tot)[1] - \
                 gic(fits["M1_lognormal"]["nll"], 2, tot)[1]
        obs_dA = gic(fits["M2_lognormal_mixture"]["nll"], 5, tot)[0] - \
                 gic(fits["M1_lognormal"]["nll"], 2, tot)[0]
        pv = float(np.mean(np.array(res["samples"]) <= obs_dB)) if res["n_ok"] else float("nan")
        boot_rows.append(dict(
            series=sid, regime=regime, n_boot=nb, n_ok=res["n_ok"],
            obs_dBIC_M2_minus_M1=obs_dB, obs_dAIC_M2_minus_M1=obs_dA,
            boot_false_pref_rate_BIC=res["rate_bic"], boot_false_pref_rate_AIC=res["rate_aic"],
            boot_median_dBIC=res["median_dBIC"], boot_mean_dBIC=res["mean_dBIC"],
            boot_sd_dBIC=res["sd_dBIC"], boot_p05_dBIC=res["p05_dBIC"],
            boot_p95_dBIC=res["p95_dBIC"], boot_pvalue_dBIC_le_obs=pv,
            n_degenerate_boot_M2_fits=res["n_degenerate"]))
        log("%s/%s n=%d ok=%d | obs dBIC=%+.2f dAIC=%+.2f | bootstrap P(BIC picks M2 | "
            "1 component)=%.3f P(AIC)=%.3f | null dBIC mean %+.2f sd %.2f [%+.2f,%+.2f] | "
            "p(dBIC<=obs)=%.4f | degenerate M2 refits %d [%.0fs]"
            % (sid, regime, nb, res["n_ok"], obs_dB, obs_dA, res["rate_bic"],
               res["rate_aic"], res["mean_dBIC"], res["sd_dBIC"], res["p05_dBIC"],
               res["p95_dBIC"], pv, res["n_degenerate"], time.time() - t0))
        store[key] = (c, n, w, fits, lam, mp, res["samples"])
        gc.collect()

    # ---- write ----
    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    def wr(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            wt = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            wt.writeheader()
            wt.writerows(rows)

    wr(os.path.join(OUT_RES, "popfin_s01_fit_table.csv"), fit_rows)
    wr(os.path.join(OUT_RES, "popfin_s01_model_params.csv"), param_rows)
    wr(os.path.join(OUT_RES, "popfin_s01_concentration.csv"), conc_rows)
    wr(os.path.join(OUT_RES, "popfin_s01_bootstrap_false_pref.csv"), boot_rows)
    wr(os.path.join(OUT_RES, "popfin_s01_optimiser_calibration.csv"), cal_rows)
    with open(os.path.join(OUT_RES, "popfin_s01_fit_table.json"), "w") as f:
        json.dump(dict(
            dimension="population structure: drift-rate distribution model comparison",
            seed=SEED, generated=time.strftime("%Y-%m-%d %H:%M"),
            analysis_unit=("one (population series, drift regime) = one workbook block "
                           "(fast) or the union of two blocks (slow). 6 series, 12 units, "
                           "5 independent burst events at the event level. Bins are "
                           "nested repeats, not independent samples."),
            likelihood=("Poisson on the binned counts, original per-series grid; overall "
                        "scale profiled in closed form (identical convention for all 4 "
                        "models, so dAIC/dBIC are unaffected). Absolute nll omits the "
                        "constant sum log(n_i!)."),
            models=dict(M1="log-normal k=2", M2="2-component log-normal mixture k=5",
                        M3="power law in drift k=2", M4="exponential in drift k=2"),
            fits=fit_rows, params=param_rows, concentration=conc_rows,
            bootstrap=boot_rows, optimiser_calibration=cal_rows), f, indent=1)

    make_figures(store, fit_rows, boot_rows)
    log("")
    log("wrote results + figures; total %.1fs" % (time.time() - t_start))
    with open(LOG, "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


def make_figures(store, fit_rows, boot_rows):
    cols = {"M1_lognormal": "#1f77b4", "M2_lognormal_mixture": "#d62728",
            "M3_powerlaw": "#2ca02c", "M4_exponential": "#ff7f0e"}
    degmap = {(r["series"], r["regime"], r["model"]): r["degeneracy"] for r in fit_rows}
    for key in sorted(store):
        sid, regime = key
        c, n, w, fits, lam, mp, samples = store[key]
        lbl = "%s | %s" % (sid.replace("population_series_", "ps"), regime)
        fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
        ax.bar(c, n, width=w * 0.95, color="0.82", edgecolor="0.5", lw=0.4,
               label="observed counts")
        for m in MODELS:
            d = degmap.get((sid, regime, m), "")
            ax.plot(c, lam[m], lw=1.8, color=cols[m],
                    label="%s%s" % (m.split("_", 1)[0], " [FLAGGED]" if d else ""))
        ax.set_xlabel("drift rate (MHz/s)")
        ax.set_ylabel("events per bin")
        ax.set_title("Poisson fits, original grid - %s (N=%d, %d bins)"
                     % (lbl, int(n.sum()), len(c)))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_FIG, "popfin_s01_fitgrid_%s_%s.png" % (sid, regime)))
        plt.close(fig)
        if regime == "fast":
            fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
            ax.bar(c, n, width=w * 0.4, color="0.82", edgecolor="0.5", lw=0.4)
            for m in MODELS:
                ax.plot(c, lam[m], lw=1.2, color=cols[m], label=m)
            ax.set_xscale("log")
            ax.set_xlabel("drift rate (MHz/s, log)")
            ax.set_ylabel("events per bin")
            ax.set_title("log-x view - %s" % lbl)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_FIG, "popfin_s01_fitlogx_%s_%s.png" % (sid, regime)))
            plt.close(fig)

    labels = ["%s\n%s" % (r["series"].replace("population_series_", "ps"), r["regime"])
              for r in boot_rows]
    vals = [r["boot_false_pref_rate_BIC"] for r in boot_rows]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    x = np.arange(len(vals))
    ax.bar(x, vals, color=["#d62728" if v >= 0.5 else ("#ff7f0e" if v >= 0.2 else "#2ca02c")
                           for v in vals])
    ax.axhline(0.05, ls="--", c="k", lw=1, label="5% nominal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("P(BIC prefers 2 components | truth is 1 log-normal)")
    ax.set_title("Parametric-bootstrap false-preference rate of the two-component mixture\n"
                 "counts simulated from each series' best-fit single log-normal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "popfin_s01_bootstrap_false_pref.png"))
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(13, 11), dpi=125)
    for ax, key in zip(axes.ravel(), sorted(store)):
        s = store[key][6]
        row = next(r for r in boot_rows if r["series"] == key[0] and r["regime"] == key[1])
        ax.hist(s, bins=30, color="0.75", edgecolor="0.5", lw=0.3)
        ax.axvline(0, c="k", lw=1)
        ax.axvline(row["obs_dBIC_M2_minus_M1"], c="#d62728", lw=2)
        ax.set_title("%s|%s obs dBIC=%+.1f rate=%.2f"
                     % (key[0].replace("population_series_", "ps"), key[1],
                        row["obs_dBIC_M2_minus_M1"], row["boot_false_pref_rate_BIC"]),
                     fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Null dBIC(M2-M1) from single-log-normal simulations; red = observed\n"
                 "(negative dBIC = 2-component preferred)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_FIG, "popfin_s01_null_dBIC_distributions.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()