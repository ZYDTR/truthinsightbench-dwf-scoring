#!/usr/bin/env python3
"""driftpop_s02 - Competing model fits to each drift-population series.

Binned-count maximum likelihood on the NATIVE binning of each series
(no re-binning), for:
  (a) single Gaussian in log10(drift)  == single log-normal in drift
  (b) two-component Gaussian mixture in log10(drift)
  (c) exponential in log10(drift) ( = power law in drift with index 2 )
  (d) power law in drift (free index)
  (e) uniform in drift (flat null)
  (f) uniform in log10(drift)
  (g) three-component mixture (fast-regime grids only)

Every model is a density in drift x, expressed in log10 space as
p(x) dx = f(l) dl with l = log10 x, and integrated exactly over the native
bin edges of each block:
  P_i = int_{l_i}^{l_i+1} f(l) dl ,  mu_i = A * P_i ,  y_i ~ Poisson(mu_i).
The expected count A is a free parameter (profiled by the optimizer); the
multinomial likelihood with p_i = P_i / sum(P) is reported alongside.

A shared-density variant ("combined") fits ONE density to both the fast and
the slow native grids of a series at once, with each block's bin probabilities
normalised over their own edges and then stacked. This tests whether a single
continuous population can describe the whole drift range.

Analysis unit: one source-group x polarization series (population_series_id).
Within a series the counts are bursts (nested repeats), not independent events;
AIC/BIC are therefore reported per series, and the number of independent units
is 6 (series) or 5 (complete dynamic spectra), never the bin or pixel count.

A parametric bootstrap LR test (single vs two-component) is run to avoid the
asymptotic chi-square shortcut, which is invalid on the mixture boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)

POP = DATA / "normalized_views" / "drift_population_series.tsv"
STRUCT = OUT / "driftpop_s01_binning_structure.csv"

RNG = np.random.default_rng(20221013)
EPS = 1e-300


# --------------------------------------------------------------- bin geometry
def grid_from_centers(centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover exact uniform grid edges from a monotone block of centers."""
    c = np.sort(np.asarray(centers, float))
    idx = np.arange(len(c), dtype=float)
    slope = float(np.linalg.lstsq(np.vstack([idx, np.ones_like(idx)]).T, c, rcond=None)[0][0])
    edges = np.concatenate([[c[0] - slope / 2], c + slope / 2])
    return edges, slope


# ----------------------------------------------------------------- density P_i
def p_normal(edges, mu, sigma):
    return np.diff(stats.norm.cdf(np.log10(edges), loc=mu, scale=sigma))


def p_mix2(edges, w, mu1, s1, mu2, s2):
    return w * p_normal(edges, mu1, s1) + (1.0 - w) * p_normal(edges, mu2, s2)


def p_mix3(edges, p, mus, sigs):
    return sum(p[k] * p_normal(edges, mus[k], sigs[k]) for k in range(3))


def p_expo_log(edges, theta):
    """f(l) proportional to exp(-theta*l) on the edges' log range."""
    l = np.log10(edges)
    lo, hi = l[0], l[-1]
    if abs(theta) < 1e-9:
        return np.diff(l) / (hi - lo)
    Z = (np.exp(-theta * lo) - np.exp(-theta * hi)) / theta
    return (np.exp(-theta * l[:-1]) - np.exp(-theta * l[1:])) / theta / Z


def p_powerlaw_x(edges, alpha):
    """p(x) proportional to x**-alpha on [x_lo, x_hi]."""
    lo, hi = edges[0], edges[-1]
    if abs(alpha - 1.0) < 1e-9:
        return np.diff(np.log(edges)) / np.log(hi / lo)
    Z = (hi ** (1 - alpha) - lo ** (1 - alpha)) / (1 - alpha)
    return (edges[1:] ** (1 - alpha) - edges[:-1] ** (1 - alpha)) / (1 - alpha) / Z


def p_uniform_x(edges):
    return np.diff(edges) / (edges[-1] - edges[0])


def p_uniform_log(edges):
    l = np.log10(edges)
    return np.diff(l) / (l[-1] - l[0])


# -------------------------------------------------------- model specification
def _spec_from_summary(name, lo, hi, lc, y, prob_fn):
    """(k_shape, prob_fn, starts, bounds) given the log10 support and bin mids."""
    if name in ("uniform_x", "uniform_log"):
        return 0, prob_fn, [np.array([])], []
    if name == "expo_log":
        return 1, prob_fn, [np.array([v]) for v in (-2.0, -0.5, 0.5, 1.5, 3.0)], [(-25.0, 25.0)]
    if name == "powerlaw_x":
        return 1, prob_fn, [np.array([v]) for v in (-1.0, 0.0, 1.0, 2.0, 3.0)], [(-12.0, 12.0)]
    if name == "lognorm":
        mean = float(np.average(lc, weights=y)) if y.sum() > 0 else 0.5 * (lo + hi)
        var = float(np.average((lc - mean) ** 2, weights=y)) if y.sum() > 0 else 1.0
        return (2, prob_fn,
                [np.array([mean, max(np.sqrt(var), 0.1)]),
                 np.array([mean, 0.3]), np.array([mean, 1.0]),
                 np.array([0.5 * (lo + hi), 0.5])],
                [(lo - 5, hi + 5), (1e-3, 10.0)])
    counts = np.maximum(y.astype(int), 0)
    if name == "mix2":
        q = np.quantile(np.repeat(lc, counts), [0.1, 0.25, 0.5, 0.75, 0.9])
        starts = []
        for mu1, mu2 in [(q[0], q[3]), (q[1], q[4]), (lo, q[2]), (lo, hi)]:
            for w in (0.3, 0.5, 0.7):
                starts.append(np.array([w, mu1, 0.15, mu2, 0.15]))
                starts.append(np.array([w, mu1, 0.35, mu2, 0.55]))
        return (5, prob_fn, starts,
                [(1e-4, 1 - 1e-4), (lo - 5, hi + 5), (1e-3, 10.0),
                 (lo - 5, hi + 5), (1e-3, 10.0)])
    if name == "mix3":
        q = np.quantile(np.repeat(lc, counts), [0.1, 0.3, 0.5, 0.7, 0.9])
        starts = []
        for (m1, m2, m3) in [(q[0], q[2], q[4]), (q[0], q[2], q[3]), (q[1], q[2], q[4])]:
            for (w1, w2) in [(0.3, 0.3), (0.2, 0.35), (0.4, 0.2)]:
                starts.append(np.array([w1, w2, m1, 0.15, m2, 0.15, m3, 0.15]))
                starts.append(np.array([w1, w2, m1, 0.35, m2, 0.45, m3, 0.35]))
        return (8, prob_fn, starts,
                [(1e-4, 1 - 1e-4), (1e-4, 1 - 1e-4),
                 (lo - 5, hi + 5), (1e-3, 10.0),
                 (lo - 5, hi + 5), (1e-3, 10.0),
                 (lo - 5, hi + 5), (1e-3, 10.0)])
    raise ValueError(name)


def single_prob_fn(name, edges):
    l = np.log10(edges)
    if name == "uniform_x":
        return lambda p: p_uniform_x(edges)
    if name == "uniform_log":
        return lambda p: p_uniform_log(edges)
    if name == "expo_log":
        return lambda p: p_expo_log(edges, p[0])
    if name == "powerlaw_x":
        return lambda p: p_powerlaw_x(edges, p[0])
    if name == "lognorm":
        return lambda p: p_normal(edges, p[0], abs(p[1]))
    if name == "mix2":
        return lambda p: p_mix2(edges, p[0], p[1], abs(p[2]), p[3], abs(p[4]))
    if name == "mix3":
        def f(p):
            w = np.abs(p[:2])
            w = np.concatenate([w, [max(1.0 - w.sum(), 1e-6)]])
            w = w / w.sum()
            return p_mix3(edges, w, np.array([p[2], p[4], p[6]]),
                          np.abs(np.array([p[3], p[5], p[7]])))
        return f
    raise ValueError(name)


def multi_block_prob_fn(name, edge_list):
    """Density integrated separately on each block's own edges, then stacked."""
    def prob_of(p):
        parts = [single_prob_fn(name, e)(p) for e in edge_list]
        P = np.concatenate(parts)
        S = P.sum()
        if not np.isfinite(S) or S <= 0:
            return None
        return P / S
    return prob_of


def _support(edge_list):
    l_lo = np.log10(np.concatenate([e[:-1] for e in edge_list]))
    l_hi = np.log10(np.concatenate([e[1:] for e in edge_list]))
    return float(l_lo.min()), float(l_hi.max()), 0.5 * (l_lo + l_hi)


def _fit(objective, starts, bounds, k_shape, N):
    full = list(bounds) + [(np.log(max(N, 1)) - 12, np.log(max(N, 1)) + 12)]
    lo_b = np.array([b[0] + 1e-9 for b in full])
    hi_b = np.array([b[1] - 1e-9 for b in full])
    best = None
    for s in starts:
        theta0 = np.concatenate([s, [np.log(N)]]) if k_shape else np.array([np.log(N)])
        x0 = np.clip(theta0, lo_b, hi_b)
        with np.errstate(all="ignore"):
            r = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=full,
                                  options=dict(maxiter=4000, maxfun=8000,
                                               ftol=1e-11, gtol=1e-9))
        if best is None or r.fun < best.fun:
            best = optimize.OptimizeResult(x=np.array(r.x), fun=float(r.fun))
    with np.errstate(all="ignore"):
        r2 = optimize.minimize(objective, best.x, method="Nelder-Mead",
                               options=dict(maxiter=20000, maxfev=40000,
                                            xatol=1e-9, fatol=1e-8))
    if r2.fun <= best.fun:
        best = optimize.OptimizeResult(x=np.array(r2.x), fun=float(r2.fun))
    return best


def poisson_negll(y, P, A):
    P = np.clip(P, EPS, None)
    P = P / P.sum()
    mu = np.clip(A * P, EPS, None)
    return float(-(np.sum(y * np.log(mu)) - np.sum(mu)))


def summarise(name, k_shape, p_hat, A_hat, y, P, n_bins):
    Pn = P / P.sum()
    ll_pois = -poisson_negll(y, Pn, A_hat)
    mu = A_hat * Pn
    chi2 = float(np.sum((y - mu) ** 2 / np.clip(mu, 1e-12, None)))
    k_pois = k_shape + 1
    return dict(
        model=name, k_shape=k_shape,
        loglik_poisson=ll_pois,
        loglik_multinomial=float(np.sum(y * np.log(np.clip(Pn, EPS, None)))),
        AIC_poisson=2 * k_pois - 2 * ll_pois,
        BIC_poisson=k_pois * np.log(n_bins) - 2 * ll_pois,
        AIC_multinomial=2 * k_shape - 2 * float(np.sum(y * np.log(np.clip(Pn, EPS, None)))),
        BIC_multinomial=k_shape * np.log(n_bins) - 2 * float(np.sum(y * np.log(np.clip(Pn, EPS, None)))),
        A_hat=A_hat, n_bins=n_bins, n_events=float(y.sum()),
        pearson_chi2=chi2, df_res=n_bins - k_pois,
        pearson_dispersion=chi2 / max(n_bins - k_pois, 1),
        params=json.dumps({"shape": np.asarray(p_hat, float).tolist(), "A": A_hat}),
    )


def fit_model(y, edge_list, name, n_starts=None):
    """Poisson MLE of `name` fitted to one or more blocks of bins."""
    N = float(y.sum())
    n_bins = len(y)
    k_shape, _, starts, bounds = _spec_from_summary(
        name, *_support(edge_list), y, single_prob_fn(name, edge_list[0]) if len(edge_list) == 1 else None)
    if len(edge_list) > 1:
        k_shape, _, starts, bounds = _spec_from_summary(
            name, *_support(edge_list), y, multi_block_prob_fn(name, edge_list))
        prob_of = multi_block_prob_fn(name, edge_list)
    else:
        prob_of = single_prob_fn(name, edge_list[0])

    if n_starts is not None:
        starts = starts[:max(1, n_starts)]

    def objective(theta):
        p = theta[:k_shape]
        A = float(np.exp(theta[-1]))
        P = prob_of(p) if k_shape else np.asarray(prob_of(np.array([])), float)
        if P is None:
            return 1e12
        return poisson_negll(y, P, A)

    best = _fit(objective, starts, bounds, k_shape, N)
    p_hat = best.x[:k_shape]
    A_hat = float(np.exp(best.x[-1]))
    P = prob_of(p_hat) if k_shape else np.asarray(prob_of(np.array([])), float)
    k_shape_eff = 0 if name in ("uniform_x", "uniform_log") else k_shape
    return summarise(name, k_shape_eff, p_hat, A_hat, y, P, n_bins)


def bootstrap_lr(y, edges, fit1, fit2, B=200):
    """Parametric bootstrap of the 2-vs-1 component LR on FIXED bins."""
    p1 = np.array(json.loads(fit1["params"])["shape"], float)
    P1 = p_normal(edges, p1[0], abs(p1[1]))
    P1 = P1 / P1.sum()
    mu = fit1["A_hat"] * P1
    lr_obs = 2 * (fit2["loglik_poisson"] - fit1["loglik_poisson"])
    null = np.full(B, np.nan)
    for b in range(B):
        ysim = RNG.poisson(mu).astype(float)
        if ysim.sum() == 0:
            continue
        f1 = fit_model(ysim, [edges], "lognorm", n_starts=2)
        f2 = fit_model(ysim, [edges], "mix2", n_starts=6)
        null[b] = 2 * (f2["loglik_poisson"] - f1["loglik_poisson"])
    null = null[np.isfinite(null)]
    return dict(lr_obs=float(lr_obs), p_bootstrap=float((np.sum(null >= lr_obs) + 1) / (len(null) + 1)),
                B=int(len(null)), null_q50=float(np.quantile(null, 0.5)),
                null_q95=float(np.quantile(null, 0.95)),
                null_q99=float(np.quantile(null, 0.99)), null_max=float(null.max()))


def main() -> None:
    pop = pd.read_csv(POP, sep="\t")
    _ = pd.read_csv(STRUCT)  # structure already verified in driftpop_s01

    FAST = ["uniform_x", "uniform_log", "expo_log", "powerlaw_x", "lognorm", "mix2", "mix3"]
    SLOW = ["uniform_x", "uniform_log", "expo_log", "powerlaw_x", "lognorm", "mix2"]

    rows, boot_rows = [], []
    for pid in sorted(pop.population_series_id.unique()):
        sub = pop[pop.population_series_id == pid]
        edges_by_regime = {}
        for regime, models in (("fast", FAST), ("slow", SLOW)):
            g = sub[sub.drift_regime == regime].sort_values("distribution_bin_center")
            y = g.event_count.to_numpy(float)
            edges, w = grid_from_centers(g.distribution_bin_center.to_numpy(float))
            edges_by_regime[regime] = (edges, y)
            got = {}
            for m in models:
                fit = fit_model(y, [edges], m, n_starts=8 if m in ("mix2", "mix3") else None)
                fit.update(population_series_id=pid, drift_regime=regime, bin_width=w,
                           edge_min=float(edges[0]), edge_max=float(edges[-1]))
                rows.append(fit)
                got[m] = fit
            if y.sum() > 0:
                br = bootstrap_lr(y, edges, got["lognorm"], got["mix2"], B=200)
                br.update(population_series_id=pid, drift_regime=regime)
                boot_rows.append(br)
        # single shared density on both native grids
        e_f, y_f = edges_by_regime["fast"]
        e_s, y_s = edges_by_regime["slow"]
        y = np.concatenate([y_f, y_s])
        for m in ["uniform_x", "uniform_log", "expo_log", "powerlaw_x", "lognorm", "mix2"]:
            fit = fit_model(y, [e_f, e_s], m, n_starts=8 if m == "mix2" else None)
            fit.update(population_series_id=pid, drift_regime="combined_native",
                       bin_width=np.nan, edge_min=float(e_f[0]),
                       edge_max=float(e_s[-1]))
            rows.append(fit)

    fitdf = pd.DataFrame(rows)
    for col in ("delta_AIC_poisson", "delta_BIC_poisson", "rank_AIC", "rank_BIC"):
        fitdf[col] = np.nan
    for (pid, reg), idx in fitdf.groupby(["population_series_id", "drift_regime"]).groups.items():
        d = fitdf.loc[idx]
        fitdf.loc[idx, "delta_AIC_poisson"] = d.AIC_poisson - d.AIC_poisson.min()
        fitdf.loc[idx, "delta_BIC_poisson"] = d.BIC_poisson - d.BIC_poisson.min()
        fitdf.loc[idx, "rank_AIC"] = d.AIC_poisson.rank(method="min")
        fitdf.loc[idx, "rank_BIC"] = d.BIC_poisson.rank(method="min")
    fitdf.to_csv(OUT / "driftpop_s02_model_fits.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "driftpop_s02_lr_bootstrap.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["population_series_id", "model", "k_shape", "loglik_poisson", "AIC_poisson",
            "BIC_poisson", "delta_AIC_poisson", "delta_BIC_poisson",
            "pearson_dispersion", "rank_AIC", "rank_BIC"]
    for reg in ("fast", "slow", "combined_native"):
        d = fitdf[fitdf.drift_regime == reg]
        print(f"\n=== {reg}: Poisson fits on the native bins ===")
        print(d[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"\n--- {reg}: winners ---")
        for pid, idx in d.groupby("population_series_id").groups.items():
            dd = d.loc[idx].sort_values("AIC_poisson")
            wa, wb = dd.iloc[0], d.loc[idx].sort_values("BIC_poisson").iloc[0]
            print(f"  {pid}: AIC -> {wa.model:12s} (dAIC to next = "
                  f"{dd.AIC_poisson.iloc[1] - wa.AIC_poisson:7.2f}); "
                  f"BIC -> {wb.model:12s}")
    print("\n=== parametric-bootstrap LR: 1 vs 2 components (same bins) ===")
    print(pd.DataFrame(boot_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== overdispersion of the single log-normal (Pearson chi2/df) ===")
    d = fitdf[(fitdf.model == "lognorm")]
    print(d[["population_series_id", "drift_regime", "n_bins", "n_events",
             "pearson_chi2", "df_res", "pearson_dispersion"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    json.dump(fitdf.to_dict(orient="records"),
              open(OUT / "driftpop_s02_model_fits.json", "w"), indent=2, default=float)
    json.dump(boot_rows, open(OUT / "driftpop_s02_lr_bootstrap.json", "w"), indent=2, default=float)
    print("\nwrote", OUT / "driftpop_s02_model_fits.csv", "+ .json")


if __name__ == "__main__":
    main()