#!/usr/bin/env python3
"""Monte-Carlo, method-free check of the detectable |r| at 80% power, n=259 (chunked)."""
import numpy as np

n = 259
rng = np.random.default_rng(31337)


def r_chunk(X, Y):
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Xc * Yc).sum(axis=1) / np.sqrt((Xc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1))


def tstat(r):
    return r * np.sqrt((n - 2) / (1 - r ** 2))


def sample_r(rho, rep, chunk=4000):
    out = []
    X0 = rng.standard_normal((chunk, n))
    # regenerate per chunk
    X0 = None
    done = 0
    while done < rep:
        c = min(chunk, rep - done)
        X = rng.standard_normal((c, n))
        Y = rho * X + np.sqrt(1 - rho ** 2) * rng.standard_normal((c, n))
        out.append(r_chunk(X, Y))
        done += c
    return np.concatenate(out)


# empirical null threshold
tnull = np.abs(tstat(sample_r(0.0, 60000)))
tcrit = np.quantile(tnull, 0.95)
print(f"empirical |t| null 95th pct (60k draws) = {tcrit:.4f}  (textbook t_257,.975 = 1.9692)")


def power(rho, rep=20000):
    r = sample_r(rho, rep)
    return float(np.mean(np.abs(tstat(r)) > tcrit))


for rho in (0.16, 0.17, 0.172, 0.174, 0.176, 0.18):
    print(f"  MC power(rho={rho:.3f}) = {power(rho):.4f}")

lo, hi = 0.01, 0.5
for _ in range(10):
    mid = 0.5 * (lo + hi)
    if power(mid, rep=12000) < 0.80:
        lo = mid
    else:
        hi = mid
print(f"MC detectable rho at 80% power (bisection ~+-0.005) = {0.5*(lo+hi):.4f}")
print("report claim: ~0.174 ; exact noncentral-t: 0.172 ; Fisher-z: 0.1733")
