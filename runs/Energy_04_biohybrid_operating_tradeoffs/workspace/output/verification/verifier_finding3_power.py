#!/usr/bin/env python3
"""Fast exact power for Pearson r via noncentral t (vectorised erf), n=259."""
import numpy as np
from math import erf, sqrt, lgamma, erfc

_v = np.vectorize(lambda z: 0.5 * (1.0 + erf(float(z) / sqrt(2.0))))


def nct_cdf(tval, df, delta, vmax=300.0, ngrid=60001):
    v = np.linspace(1e-8, vmax, ngrid)
    logw = (df / 2 - 1) * np.log(v) - v / 2 - (df / 2) * np.log(2) - lgamma(df / 2)
    w = np.exp(logw - logw.max())
    cdf = _v(tval * np.sqrt(v / df) - delta)
    return float(np.trapezoid(cdf * w, v) / np.trapezoid(w, v))


def tcrit(df, alpha=0.05):
    lo, hi = 0.0, 200.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if nct_cdf(mid, df, 0.0) < 1 - alpha / 2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    n = 259
    df = n - 2
    tc = tcrit(df)
    print(f"critical |t| df={df} alpha=0.05 -> {tc:.5f} (textbook 1.9692)")
    # sanity: nct_cdf(tc,df,0) should be 0.975
    print(f"check: CDF(tc) = {nct_cdf(tc, df, 0.0):.5f}")

    def power(rho, alpha=0.05):
        d = rho * sqrt(n) / sqrt(1 - rho ** 2)
        return 1 - nct_cdf(tc, df, d) + nct_cdf(-tc, df, d)

    for rho in (0.10, 0.15, 0.17, 0.174, 0.18, 0.20):
        print(f"  power(rho={rho:.3f}) = {power(rho):.4f}")

    lo, hi = 0.01, 0.6
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if power(mid) < 0.80:
            lo = mid
        else:
            hi = mid
    print(f"EXACT detectable rho at 80% power, n=259: {0.5*(lo+hi):.5f}")
    print(f"Fisher-z closed form: {np.tanh((1.959963984540054+0.8416212335729143)/np.sqrt(n-3)):.5f}")
    print(f"report claim: ~0.174")
    # tail residual significance
    import pandas as pd
    dfd = pd.read_csv("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv", sep="\t")
    x = dfd.condition_a.to_numpy(float)[294:]; y = dfd.condition_b.to_numpy(float)[294:]
    dx, dy = np.diff(x), np.diff(y)

    def pr(x_, y_):
        return float(np.corrcoef(x_, y_)[0, 1])

    r1 = pr(dx, dy); t1 = r1 * sqrt((len(dx) - 2) / (1 - r1 ** 2))
    print(f"first-diff r={r1:+.6f} t={t1:.3f} p={erfc(abs(t1)/sqrt(2)):.3e}")
    rmx = x - pd.Series(x).rolling(15, center=True, min_periods=1).median().to_numpy()
    rmy = y - pd.Series(y).rolling(15, center=True, min_periods=1).median().to_numpy()
    r2 = pr(rmx, rmy); t2 = r2 * sqrt((len(rmx) - 2) / (1 - r2 ** 2))
    print(f"detrended  r={r2:+.6f} t={t2:.3f} p={erfc(abs(t2)/sqrt(2)):.3e}")
    print(f"residual lag-1 autocorr a={pr(rmx[:-1], rmx[1:]):.4f} b={pr(rmy[:-1], rmy[1:]):.4f}")
    # effective n via Bartlett-style adjustment
    print(f"levels tail lag-1 autocorr a={pr(x[:-1], x[1:]):.6f} b={pr(y[:-1], y[1:]):.6f}")