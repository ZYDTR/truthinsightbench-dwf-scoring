"""Dependency-free statistical helpers for the interface-catalysis scan task.

The runtime has numpy/pandas but no scipy/statsmodels (verified: `import scipy`
raises ModuleNotFoundError).  All inferential quantities below are therefore
implemented from closed-form definitions and a deterministic exact/permutation
enumeration, which is preferable for this task's small-n complete-unit design
(2 blocks x 4 conditions = 8 curves, 10 ordered scan settings).

Everything is deterministic: no RNG is used, so every number in the result
artifacts is bit-reproducible.
"""

from __future__ import annotations

from itertools import combinations
from math import comb, exp, lgamma, log, sqrt

import numpy as np


# --------------------------------------------------------------------------
# Special functions and distributions
# --------------------------------------------------------------------------
def log_gamma(x: float) -> float:
    return lgamma(x)


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def beta_continued_fraction(a: float, b: float, x: float, *, max_iter: int = 300,
                            eps: float = 3e-16, tiny: float = 1e-300) -> float:
    """Lentz-algorithm evaluation of the incomplete beta function (NR 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(a * log(x) + b * log(1.0 - x) - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """P(T >= t) for a Student-t variate with `df` degrees of freedom."""
    if not np.isfinite(t):
        return 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return tail if t >= 0 else 1.0 - tail


def student_t_ppf(p: float, df: float) -> float:
    """Two-sided-compatible inverse Student-t CDF via bisection (df can be float)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - student_t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z confidence interval for a Pearson correlation (n pairs)."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / sqrt(n - 3.0)
    crit = student_t_ppf(1.0 - alpha / 2.0, n - 3)
    lo, hi = z - crit * se, z + crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def fisher_z_pvalue(r: float, n: int) -> float:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan")
    z = np.arctanh(r) * sqrt(n - 3.0)
    return float(2.0 * student_t_sf(abs(z), n - 3))


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    """Ordinary least squares with classical (homoscedastic) inference.

    `X` must already contain any intercept column.  Returns coefficients,
    standard errors, t statistics, two-sided p values, R^2, adjusted R^2,
    residual standard deviation, and the hat values.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    rss = float(resid @ resid)
    df_resid = n - k
    dof = max(df_resid, 1)
    sigma2 = rss / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    pval = np.array([2.0 * student_t_sf(abs(t), dof) for t in tstat])
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if tss > 0 else float("nan")
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    return {
        "n": n,
        "k": k,
        "df_resid": df_resid,
        "beta": beta,
        "se": se,
        "t": tstat,
        "p": pval,
        "rss": rss,
        "r2": r2,
        "adj_r2": adj_r2,
        "resid_sd": sqrt(sigma2),
        "resid": resid,
        "fitted": X @ beta,
        "hat": hat,
    }


def aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian-error AIC/BIC on the residual-sum-of-squares scale (k incl. sigma)."""
    rss = max(rss, 1e-300)
    aic = n * log(rss / n) + 2.0 * k
    bic = n * log(rss / n) + k * log(n)
    return float(aic), float(bic)


# --------------------------------------------------------------------------
# Rank statistics and exact permutation inference
# --------------------------------------------------------------------------
def average_ranks(values) -> np.ndarray:
    """Average ranks with ties handled by the mid-rank convention."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> tuple[float, float]:
    """Spearman rank correlation (Pearson on mid-ranks) and its n>=4 p value."""
    rx, ry = average_ranks(x), average_ranks(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, fisher_z_pvalue(r, len(rx))


def kendall_tau_b(x, y) -> float:
    """Kendall tau-b with tie correction."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    conc = disc = tie_x = tie_y = 0
    for i, j in combinations(range(n), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            tie_x += 1
            tie_y += 1
        elif dx == 0:
            tie_x += 1
        elif dy == 0:
            tie_y += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    n0 = comb(n, 2)
    denom = sqrt((n0 - tie_x) * (n0 - tie_y))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def permutation_p(direction: int, T_obs: float, values, *, target=None,
                  larger_is_left: bool = True, max_exact: int = 500_000,
                  exact: bool = True) -> dict:
    """Exact/complete one-sided permutation p value for the sign test of `values`.

    Statistic: sum of values on the left group.  With a two-level grouping of
    n1/n2 units the rejection set is every assignment whose statistic is at
    least as extreme as the observed one in the direction used.  This is the
    fully enumerated version of the exact sign test (no RNG, no asymptotics).

    `direction` = -1 tests "left group is smaller"; +1 tests "left group larger".
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    k = int(round(T_obs)) if float(T_obs).is_integer() else T_obs
    # T_obs must be identifiable: recover the left group size from the caller.
    n1 = int(round(target)) if target is not None else None
    if n1 is None:
        raise ValueError("left_group_size must be supplied")
    if comb(n, n1) <= max_exact and exact:
        idx = range(n)
        stats = np.array([values[list(c)].sum() for c in combinations(idx, n1)])
        if direction < 0:
            count = int((stats <= T_obs + 1e-12).sum())
        else:
            count = int((stats >= T_obs - 1e-12).sum())
        total = len(stats)
        return {
            "p": count / total,
            "method": "exact_complete_enumeration",
            "enumerated_assignments": total,
            "rejecting_assignments": count,
            "left_group_size": n1,
            "direction": int(direction),
        }
    return {"p": float("nan"), "method": "not_enumerated",
            "left_group_size": n1, "direction": int(direction)}


def within_curve_monotone(values) -> bool:
    """True iff non-decreasing (ties allowed)."""
    v = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(v) >= 0))


def segments_with_T(values, T_grid):
    """First T at which the value reaches/exceeds each integer step 1..max."""
    v = np.asarray(values, dtype=float)
    T = np.asarray(T_grid, dtype=float)
    out = {}
    for step in range(1, int(np.floor(v.max())) + 1):
        hit = np.nonzero(v >= step)[0]
        if hit.size:
            out[step] = float(T[hit[0]])
    return out