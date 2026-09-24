#!/usr/bin/env python3
"""popfin_s00_probe.py -- throwaway calibration probe: find the M1 and M2 optima
for a couple of series by brute force, with an INDEPENDENT likelihood
implementation (scipy.stats.poisson.logpmf) to validate popfin_s01's objective."""
import csv, math, sys
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"


def load(key):
    by = {}
    with open(ROOT + "/data/normalized_views/drift_population_series.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["population_series_id"], r["drift_regime"])
            by.setdefault(k, []).append((float(r["distribution_bin_center"]),
                                         int(float(r["event_count"]))))
    v = sorted(by[key])
    return np.array([x for x, _ in v]), np.array([y for _, y in v], float)


def widths(c):
    d = np.diff(c)
    e = np.concatenate([[c[0] - d[0] / 2], (c[:-1] + c[1:]) / 2, [c[-1] + d[-1] / 2]])
    return np.diff(e)


def ln_pdf(s, mu, sig):
    if not (1e-6 < sig < 1e3):
        return np.full_like(np.asarray(s, float), -1e6)
    return (-0.5 * math.log(2 * math.pi) - math.log(sig * math.log(10))
            - (np.log10(s) - mu) ** 2 / (2 * sig ** 2) - np.log(s))


def nll_M1_indep(theta, c, n, w):
    mu, ls = theta
    sig = 10.0 ** ls
    lam = n.sum() * w * np.exp(np.clip(ln_pdf(c, mu, sig), -700, 0))
    return -float(np.sum(poisson.logpmf(n, np.maximum(lam, 1e-300))))


def nll_M2_indep(theta, c, n, w):
    """smooth parameterisation: medians = 10**(m +- 10**ld), no barrier."""
    m, ld, ls1, ls2, logitp = theta
    d = 10.0 ** ld
    mu1, mu2 = m - d, m + d
    p = 1.0 / (1.0 + math.exp(-logitp))
    lf1, lf2 = ln_pdf(c, mu1, 10.0 ** ls1), ln_pdf(c, mu2, 10.0 ** ls2)
    mix = p * np.exp(np.clip(lf1, -700, 0)) + (1 - p) * np.exp(np.clip(lf2, -700, 0))
    lam = n.sum() * w * np.maximum(mix, 1e-300)
    return -float(np.sum(poisson.logpmf(n, np.maximum(lam, 1e-300))))


def brute_M1(c, n, w, step=0.01):
    mus = np.arange(math.log10(c[0]) - 0.8, math.log10(c[-1]) + 0.1, step)
    lss = np.arange(math.log10(0.01), math.log10(1.5), step)
    best = (None, 1e300)
    base = math.log(n.sum()) + np.log(w)
    for ls in lss:
        sig = 10.0 ** ls
        lf = -0.5 * math.log(2 * math.pi) - math.log(sig * math.log(10)) - np.log(c)
        for mu in mus:
            v = -0.0
            ll = base + lf - (np.log10(c) - mu) ** 2 / (2 * sig ** 2)
            lam = np.exp(np.clip(ll, -700, 0))
            v = float(np.sum(poisson.logpmf(n, np.maximum(lam, 1e-300))))
            if -v < best[1]:
                best = ((mu, ls), -v)
    return best


def brute_M2(c, n, w, step=0.05):
    """independent coarse brute force over the 5-d smooth space."""
    l10 = np.log10(c)
    base = math.log(n.sum()) + np.log(w)
    mus = np.arange(math.log10(c[0]) - 0.7, math.log10(c[-1]) + 0.1, step)
    lds = np.arange(-2.0, 0.4, 0.25)
    sigs = [0.05, 0.10, 0.18, 0.30, 0.50, 0.80]
    ps = [0.05, 0.12, 0.25, 0.5, 0.75, 0.88, 0.95]
    best = (None, 1e300)
    for m in mus:
        for ld in lds:
            d = 10.0 ** ld
            for s1 in sigs:
                for s2 in sigs:
                    lf1 = -0.5 * math.log(2 * math.pi) - math.log(s1 * math.log(10)) - np.log(c) - (l10 - (m - d)) ** 2 / (2 * s1 ** 2)
                    lf2 = -0.5 * math.log(2 * math.pi) - math.log(s2 * math.log(10)) - np.log(c) - (l10 - (m + d)) ** 2 / (2 * s2 ** 2)
                    e1 = np.exp(np.clip(lf1, -700, 0))
                    e2 = np.exp(np.clip(lf2, -700, 0))
                    for p in ps:
                        mix = p * e1 + (1 - p) * e2
                        lam = np.maximum(np.exp(np.clip(base + np.log(np.maximum(mix, 1e-300)), -700, 0)), 1e-300)
                        v = -float(np.sum(poisson.logpmf(n, lam)))
                        if v < best[1]:
                            best = ((m, ld, math.log10(s1), math.log10(s2), math.log(p / (1 - p))), v)
    return best


if __name__ == "__main__":
    for key in [("population_series_1", "fast"), ("population_series_4", "fast"),
                ("population_series_3", "slow")]:
        c, n = load(key)
        w = widths(c)
        b1 = brute_M1(c, n, w)
        print(key, "M1 brute", b1, "med=%.5f sig=%.5f" % (10 ** b1[0][0], 10 ** b1[0][1]))
        r = minimize(nll_M1_indep, np.array(b1[0]), args=(c, n, w), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-10, fatol=1e-10))
        r2 = minimize(nll_M1_indep, np.array(b1[0]), args=(c, n, w), method="L-BFGS-B",
                      options=dict(maxiter=4000, ftol=1e-14, gtol=1e-12))
        print("   M1 NM", r.fun, r.x, "| LBFGS ", r2.fun, r2.x)
        b2 = brute_M2(c, n, w)
        print("   M2 brute", b2)
        r3 = minimize(nll_M2_indep, np.array(b2[0]), args=(c, n, w), method="L-BFGS-B",
                      options=dict(maxiter=6000, ftol=1e-14, gtol=1e-12))
        print("   M2 LBFGS", r3.fun, r3.x, r3.message)
        r4 = minimize(nll_M2_indep, np.array(b2[0]), args=(c, n, w), method="Nelder-Mead",
                      options=dict(maxiter=20000, maxfev=20000, xatol=1e-9, fatol=1e-9))
        print("   M2 NM   ", r4.fun, r4.x)