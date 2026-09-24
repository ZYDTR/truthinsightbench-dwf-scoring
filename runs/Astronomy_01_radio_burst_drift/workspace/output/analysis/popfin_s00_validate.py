#!/usr/bin/env python3
"""popfin_s00_validate.py -- validate popfin_s01's Poisson objectives against an
INDEPENDENT implementation (scipy.stats.poisson.logpmf, no clipping) on the same
data, and locate the M1/M2 optima by fine brute force + Powell/Nelder-Mead.

Smooth M2 parameterisation (no ordering barrier, no grid-edge barrier):
    med1 = 10**(m - 10**ld),  med2 = 10**(m + 10**ld),  ld <= ~0.5
    weights: p, 1-p via logit
"""
import csv
import math
import sys

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
sys.path.insert(0, ROOT + "/output/analysis")
import popfin_s01_model_selection as P  # noqa: E402

LOGP = math.log(2 * math.pi)


def load(key):
    by = {}
    with open(ROOT + "/data/normalized_views/drift_population_series.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["population_series_id"], r["drift_regime"])
            by.setdefault(k, []).append((float(r["distribution_bin_center"]),
                                         int(float(r["event_count"]))))
    v = sorted(by[key])
    return np.array([x for x, _ in v]), np.array([y for _, y in v], float)


def lnpdf(s, mu, sig):
    return (-0.5 * LOGP - math.log(sig * math.log(10))
            - (np.log10(s) - mu) ** 2 / (2 * sig ** 2) - np.log(s))


def indep_nll_M1(theta, c, n, w):
    """scipy.stats based; lam floor 1e-300 only (no saturation)."""
    mu, ls = theta
    if not (-6 < ls < 1.0):
        return 1e18
    lam = np.maximum(n.sum() * w * np.exp(lnpdf(c, mu, 10.0 ** ls)), 1e-300)
    return -float(np.sum(poisson.logpmf(n, lam)))


def indep_nll_M2(theta, c, n, w):
    m, ld, ls1, ls2, logitp = theta
    if not (-4 < ld < 1.0) or not (-6 < ls1 < 1.0) or not (-6 < ls2 < 1.0):
        return 1e18
    d = 10.0 ** ld
    mu1, mu2 = m - d, m + d
    p = 1.0 / (1.0 + math.exp(-logitp))
    e1 = np.exp(lnpdf(c, mu1, 10.0 ** ls1))
    e2 = np.exp(lnpdf(c, mu2, 10.0 ** ls2))
    lam = np.maximum(n.sum() * w * (p * e1 + (1 - p) * e2), 1e-300)
    return -float(np.sum(poisson.logpmf(n, lam)))


def pop_nll_M1(theta, c, n, w):
    return P.nll_M1(np.asarray(theta, float), w, c, n)


def pop_nll_M2(theta_smooth, c, n, w):
    """map the smooth parameterisation into popfin's (mu1,l10s1,mu2,l10s2,logit)."""
    m, ld, ls1, ls2, logitp = theta_smooth
    d = 10.0 ** ld
    return P.nll_M2(np.array([m - d, ls1, m + d, ls2, logitp]), w, c, n)


if __name__ == "__main__":
    keys = [("population_series_1", "fast"), ("population_series_4", "fast"),
            ("population_series_3", "slow"), ("population_series_4", "slow"),
            ("population_series_1", "slow")]
    for key in keys:
        c, n = load(key)
        w = P.bin_widths(c)
        print("=" * 70)
        print(key, "bins=%d total=%d" % (len(c), n.sum()))

        # ---- M1: fine brute force, independent objective ----
        mus = np.arange(math.log10(c[0]) - 0.7, math.log10(c[-1]) + 0.05, 0.004)
        lss = np.arange(math.log10(0.02), math.log10(1.2), 0.004)
        M2d = np.add.outer(-0.5 * lss, 0.0)
        best = (None, 1e300)
        for ls in lss:
            sig = 10.0 ** ls
            base = (-0.5 * LOGP - math.log(sig * math.log(10)) - np.log(c)[:, None]
                    - (np.log10(c)[:, None] - mus[None, :]) ** 2 / (2 * sig ** 2))
            lam = np.maximum(np.exp(np.clip(base + math.log(n.sum()) + np.log(w)[:, None], -690, 690)), 1e-300)
            vals = -poisson.logpmf(n[:, None], lam).sum(axis=0)
            i = int(np.argmin(vals))
            if vals[i] < best[1]:
                best = ((mus[i], ls), float(vals[i]))
        print("  M1 brute (indep): nll=%.6f at mu=%.5f ls=%.5f (med=%.5f sig=%.5f)"
              % (best[1], best[0][0], best[0][1], 10 ** best[0][0], 10 ** best[0][1]))
        r = minimize(indep_nll_M1, np.array(best[0]), args=(c, n, w), method="Powell",
                     options=dict(maxiter=20000, maxfev=20000, xtol=1e-10, ftol=1e-12))
        print("  M1 Powell(indep): nll=%.6f x=%s" % (r.fun, r.x))
        # popfin objective at the same point + its own fit
        pn_at = pop_nll_M1(r.x, c, n, w)
        conv = pn_at - r.fun
        print("  offset popfin - indep = %.4f (should be constant = sum gammaln(n+1))"
              % conv)
        f1 = P.fit_M1(c, n, w, "full")
        print("  popfin fit_M1: nll=%.6f x=%s (med=%.5f sig=%.5f)"
              % (f1["nll"], f1["x"], 10 ** f1["x"][0], 10 ** f1["x"][1]))
        print("  popfin nll at Powell point = %.6f ; at its own point = %.6f ; diff=%.6f"
              % (pn_at, f1["nll"], f1["nll"] - pn_at))

        # ---- M2: independent multi-start ----
        rng = np.random.default_rng(7)
        st = []
        for _ in range(400):
            m = rng.uniform(math.log10(c[0]) - 0.4, math.log10(c[-1]))
            ld = rng.uniform(-2.0, -0.15)
            ls1 = rng.uniform(math.log10(0.04), math.log10(1.0))
            ls2 = rng.uniform(math.log10(0.04), math.log10(1.0))
            lp = rng.uniform(-3, 3)
            st.append((m, ld, ls1, ls2, lp))
        vals = [(indep_nll_M2(np.array(s), c, n, w), s) for s in st]
        vals.sort(key=lambda t: t[0])
        print("  M2 indep best-of-%d random: nll=%.6f x=%s" % (len(st), vals[0][0], vals[0][1]))
        for _, s in vals[:5]:
            r2 = minimize(indep_nll_M2, np.array(s), args=(c, n, w), method="Powell",
                          options=dict(maxiter=20000, maxfev=40000, xtol=1e-10, ftol=1e-12))
            if r2.fun < vals[0][0]:
                vals[0] = (float(r2.fun), tuple(r2.x))
        print("  M2 indep after Powell polish: nll=%.6f x=%s" % (vals[0][0], vals[0][1]))
        pn2_at = pop_nll_M2(vals[0][1], c, n, w)
        print("  popfin nll at indep M2 optimum = %.6f  (diff %.6f)" % (pn2_at, pn2_at - vals[0][0]))
        f2 = P.fit_M2(c, n, w, "full")
        print("  popfin fit_M2: nll=%.6f x=%s" % (f2["nll"], f2["x"]))
        print("  popfin M2 vs indep M2: dNLL=%.6f" % (f2["nll"] - pn2_at))