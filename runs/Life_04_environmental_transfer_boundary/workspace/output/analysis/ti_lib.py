"""Shared statistical helpers for the Life_04 environmental-transfer analysis.

The runtime has numpy + pandas only (no scipy/statsmodels/sklearn), so every
routine here is implemented from first principles with exact or permutation
based inference.  All randomness is seeded for reproducibility.

Conventions
-----------
* "assay" is `quantitative_assay_value`; lower values are treated as stronger
  signal only where explicitly stated (rank statistics are direction-aware).
* Unit-level blocking: rows sharing a `unit_id` are repeated observations and
  never enter a test as independent subjects unless a function is explicitly
  labelled `naive_site_level` (used only to demonstrate the design effect).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

SEED = 20200101  # T0-stamped seed; keeps every stochastic step reproducible


def sub_rng(tag: str, salt: int = 0) -> np.random.Generator:
    """Independent, order-invariant RNG stream for one analysis section.

    Sections draw from named streams rather than one shared generator, so a
    section keeps the same random numbers whether it is executed alone or as
    part of the full script, and reordering sections cannot silently change
    any reported p-value.
    """
    digest = hashlib.sha256(f"{SEED}:{tag}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


# --------------------------------------------------------------------------
# descriptive / rank statistics
# --------------------------------------------------------------------------
def ranks(x) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(ranks(x), ranks(y))


def auc_lower(pos, neg) -> float:
    """P(score of a positive unit < score of a negative unit) + 0.5*ties.

    Used when *lower* values are expected to accompany positivity.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    lt = eq = 0.0
    for p in pos:
        lt += float(np.sum(p < neg))
        eq += float(np.sum(p == neg))
    return (lt + 0.5 * eq) / (len(pos) * len(neg))


def mann_whitney_u(pos, neg) -> float:
    """U for the 'positive group has lower values' direction (ties = 0.5)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    u = 0.0
    for p in pos:
        u += float(np.sum(p < neg)) + 0.5 * float(np.sum(p == neg))
    return u


# --------------------------------------------------------------------------
# exact tests
# --------------------------------------------------------------------------
def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> dict:
    """Exact Fisher test for [[a, b], [c, d]] with fixed margins.

    Returns two-sided p (sum of tables no more likely than observed) and the
    one-sided p for enrichment in cell (a).
    """
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)

    def pmf(k: int) -> float:
        return math.comb(row1, k) * math.comb(n - row1, col1 - k) / math.comb(n, col1)

    p_obs = pmf(a)
    tol = 1e-9
    p_two = sum(pmf(k) for k in range(lo, hi + 1) if pmf(k) <= p_obs * (1 + tol))
    p_greater = sum(pmf(k) for k in range(a, hi + 1))
    return {
        "table": [[a, b], [c, d]],
        "p_two_sided": float(min(1.0, p_two)),
        "p_greater": float(min(1.0, p_greater)),
        "odds_ratio": float("inf") if b * c == 0 else (a * d) / (b * c),
    }


def risk_ratio_ci(a: int, b: int, c: int, d: int, z: float = 1.959963985) -> dict:
    """Risk ratio with Katz log-normal CI (no continuity correction)."""
    n1, n2 = a + b, c + d
    if n1 == 0 or n2 == 0 or a == 0 or c == 0:
        return {"risk_ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rr = (a / n1) / (c / n2)
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return {
        "risk_ratio": float(rr),
        "ci_low": float(math.exp(math.log(rr) - z * se)),
        "ci_high": float(math.exp(math.log(rr) + z * se)),
    }


def perm_pvalue(observed: float, null: np.ndarray, greater: bool = True) -> float:
    """Monte-Carlo permutation p with the +1 correction (never returns 0)."""
    null = np.asarray(null, dtype=float)
    if greater:
        hits = int(np.sum(null >= observed))
    else:
        hits = int(np.sum(null <= observed))
    return float((hits + 1) / (len(null) + 1))


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (site-level Bernoulli checks)."""
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(pr for pr in probs if pr <= obs * (1 + 1e-9))))


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------
def logistic_fit(X, y, penalty: float = 0.0, firth: bool = False, max_iter: int = 200, tol: float = 1e-11) -> dict:
    """Logistic regression by IRLS.

    ``penalty`` adds a ridge term (used as a separation fallback); ``firth``
    uses the Firth bias-reducing penalised score, which stays finite under
    quasi-complete separation.  Returns coefficients on the log-odds scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    converged = False
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-10)
        if firth:
            W = np.sqrt(w)
            XW = X * W[:, None]
            H = XW @ np.linalg.pinv(XW.T @ XW) @ XW.T
            h = np.clip(np.diag(H), 0.0, 1.0)
            U = X.T @ (y - mu + h * (0.5 - mu))
            I = X.T @ (X * w[:, None])
        else:
            U = X.T @ (y - mu) - penalty * beta
            I = X.T @ (X * w[:, None]) + penalty * np.eye(p)
        step = np.linalg.solve(I + 1e-10 * np.eye(p), U)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(mu * (1 - mu), 1e-10)
    cov_model = np.linalg.pinv(X.T @ (X * w[:, None]))
    ll = float(np.sum(y * np.log(np.maximum(mu, 1e-12)) + (1 - y) * np.log(np.maximum(1 - mu, 1e-12))))
    return {
        "beta": beta.tolist(),
        "se_model": np.sqrt(np.diag(cov_model)).tolist(),
        "cov_model": cov_model.tolist(),
        "fitted": mu,
        "loglik": ll,
        "converged": bool(converged),
        "firth": bool(firth),
        "penalty": float(penalty),
    }


def cluster_robust_cov(X, y, beta, groups) -> np.ndarray:
    """Cluster sandwich covariance (CR0) with the unit as the cluster."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    mu = 1.0 / (1.0 + np.exp(-np.clip(X @ np.asarray(beta, dtype=float), -30, 30)))
    w = np.maximum(mu * (1 - mu), 1e-10)
    bread = np.linalg.pinv(X.T @ (X * w[:, None]))
    meat = np.zeros((X.shape[1], X.shape[1]))
    resid = y - mu
    for g in np.unique(groups):
        m = groups == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    return bread @ meat @ bread


def wald_p(beta_j: float, se_j: float) -> dict:
    """Two-sided Wald p from a normal approximation (no scipy)."""
    if not np.isfinite(se_j) or se_j <= 0:
        return {"z": float("nan"), "p": float("nan")}
    z = beta_j / se_j
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"z": float(z), "p": float(p)}


def sufficiency_key(X, y) -> tuple:
    """Sufficient statistics of a logistic fit, ``X.T @ y``.

    Two outcome vectors sharing this key give identical fits **for a fixed
    design matrix**, because the logistic log-likelihood depends on y only
    through ``X.T @ y``.  This does *not* hold when the design changes between
    fits (e.g. leave-one-out), so it must not be used as a cache key there.
    """
    return tuple(np.asarray(X, dtype=float).T @ np.asarray(y, dtype=float))


def loo_loglik(X, y, fitter) -> float:
    """Leave-one-out cross-validated log-likelihood for a logistic fitter.

    Deliberately uncached: each held-out unit changes the design matrix, so
    fits cannot be reused even when two permuted outcomes share a statistic.
    Cost is 30 IRLS solves per call.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    ll = 0.0
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        fit = fitter(X[mask], y[mask])
        eta = float(np.clip(X[i] @ np.asarray(fit["beta"]), -30, 30))
        p = 1.0 / (1.0 + math.exp(-eta))
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll += float(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
    return ll


# --------------------------------------------------------------------------
# clustering / design effect
# --------------------------------------------------------------------------
def icc_binary(cluster_ids, values) -> dict:
    """One-way random-effects ICC for a 0/1 outcome (ANOVA estimator).

    Also returns the design effect Deff = 1 + (k0 - 1) * ICC with k0 the
    adjusted mean cluster size, i.e. how many times the variance of a
    site-level mean is inflated relative to simple random sampling.
    """
    cluster_ids = np.asarray(cluster_ids)
    values = np.asarray(values, dtype=float)
    groups = [values[cluster_ids == g] for g in np.unique(cluster_ids)]
    k = np.array([len(g) for g in groups], dtype=float)
    n = k.sum()
    a = len(groups)
    means = np.array([g.mean() for g in groups])
    grand = values.mean()
    ssb = float(np.sum(k * (means - grand) ** 2))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df_b, df_w = a - 1, n - a
    msb, msw = ssb / df_b, ssw / df_w
    k0 = (n - float(np.sum(k**2)) / n) / (a - 1)
    denom = msb + (k0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    icc = max(0.0, min(1.0, icc))
    deff = 1.0 + (k0 - 1.0) * icc
    return {
        "icc": float(icc),
        "ms_between": float(msb),
        "ms_within": float(msw),
        "mean_cluster_size": float(k.mean()),
        "adjusted_k0": float(k0),
        "design_effect": float(deff),
        "effective_n_sites": float(n / deff) if deff > 0 else float("nan"),
        "n_sites": int(n),
        "n_clusters": int(a),
    }


def overdispersion_test(counts, sizes, p_hat: float, rng, n_rep: int = 20000) -> dict:
    """Test whether unit-level positive counts are overdispersed vs Binomial."""
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    obs_var = float(counts.var(ddof=1))
    binom_var = float(np.mean(sizes * p_hat * (1 - p_hat)))
    sims = rng.binomial(sizes.astype(int), p_hat, size=(n_rep, len(sizes))).var(axis=1, ddof=1)
    return {
        "observed_variance_of_counts": obs_var,
        "binomial_variance_of_counts": binom_var,
        "variance_ratio": float(obs_var / binom_var) if binom_var > 0 else float("nan"),
        "p_overdispersed": perm_pvalue(obs_var, sims, greater=True),
        "n_rep": int(n_rep),
    }


# --------------------------------------------------------------------------
# vectorised permutation machinery (block-aware, constant-preserving)
# --------------------------------------------------------------------------
def perm_index_within_groups(codes, n_rep: int, rng) -> np.ndarray:
    """Index matrix ``src`` of shape (n_rep, n) that permutes values *within*
    each group, holding group membership fixed.

    ``v[src]`` therefore yields, for every replicate, a rearrangement of ``v``
    that preserves the per-group multiset - the correct null for unit-blocked
    data where the unit total (not the site label) is the natural sufficient
    statistic.
    """
    codes = np.asarray(codes)
    n = len(codes)
    src = np.empty((n_rep, n), dtype=int)
    for g in np.unique(codes):
        m = np.flatnonzero(codes == g)
        k = len(m)
        order = np.argsort(rng.random((n_rep, k)), axis=1, kind="stable")
        src[:, m] = m[order]
    return src


def perm_index_global(n: int, n_rep: int, rng) -> np.ndarray:
    """Plain global permutation index matrix (for unit-level tests)."""
    return np.argsort(rng.random((n_rep, n)), axis=1, kind="stable")


def spearman_perm_null(x, y, n_rep: int, rng, within_groups=None) -> tuple:
    """Vectorised permutation null for Spearman rho between a 0/1 (or numeric)
    outcome ``y`` and predictor ``x``.

    Returns (observed_rho, null_array).  Because ranks of a permuted vector are
    a permutation of the original ranks, the whole null is computed with one
    matrix product.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = ranks(x)
    ry = ranks(y)
    def _corr(a_ranks):
        za = (a_ranks - a_ranks.mean()) / a_ranks.std()
        return za
    zx = _corr(rx)
    zy = _corr(ry)
    obs = float(np.mean(zx * zy))
    src = perm_index_within_groups(within_groups, n_rep, rng) if within_groups is not None \
        else perm_index_global(len(y), n_rep, rng)
    zy_perm = (ry[src] - ry.mean()) / ry.std()
    null = zy_perm @ zx / len(y)
    return obs, null


def group_indicator(codes, groups=None) -> tuple:
    """(indicator matrix, group labels) with indicator[g, i] = 1 if row i in g."""
    codes = np.asarray(codes)
    labels = np.unique(codes) if groups is None else np.asarray(groups)
    M = np.zeros((len(labels), len(codes)), dtype=float)
    for j, g in enumerate(labels):
        M[j, codes == g] = 1.0
    return M, labels


def all_or_nothing(cluster_ids, values, positive=1) -> dict:
    """Share of units whose sampled sites are unanimously positive/negative."""
    cluster_ids = np.asarray(cluster_ids)
    values = np.asarray(values, dtype=float)
    n_units = n_all_pos = n_all_neg = 0
    for g in np.unique(cluster_ids):
        v = values[cluster_ids == g]
        n_units += 1
        if v.min() == v.max() == positive:
            n_all_pos += 1
        elif v.min() == v.max() == 0:
            n_all_neg += 1
    return {
        "n_units": int(n_units),
        "n_unanimous_positive": int(n_all_pos),
        "n_unanimous_negative": int(n_all_neg),
        "share_unanimous": float((n_all_pos + n_all_neg) / n_units),
    }

def perm_group_spread_null(indicator, codes, values, n_rep: int, rng, chunk: int = 5000) -> np.ndarray:
    """Null distribution of the max-minus-min of per-group means, under
    within-cluster permutation of ``values``.

    Equivalent to computing ``indicator @ values[perm]`` for every replicate in
    one shot, but the replicate dimension is processed in bounded chunks
    (``chunk`` rows at a time) so peak memory stays near ``chunk x n_rows``
    instead of ``n_rep x n_rows``.  For a 200,000-replicate study over 212 rows
    that is the difference between ~17 MB and ~340 MB of working set, which on a
    memory-limited runtime is the difference between seconds and thrashing.

    ``indicator`` must be a (n_groups, n_rows) 0/1 matrix, ``codes`` the cluster
    id per row, ``values`` the 0/1 outcome per row.  Returns an array of length
    ``n_rep``.
    """
    indicator = np.asarray(indicator, dtype=float)
    values = np.asarray(values, dtype=float)
    counts = indicator.sum(axis=1)[:, None]
    spread = np.empty(n_rep, dtype=float)
    done = 0
    while done < n_rep:
        m = min(chunk, n_rep - done)
        src = perm_index_within_groups(codes, m, rng)
        block = values[src].astype(float)
        means = (block @ np.ascontiguousarray(indicator.T)).T / counts
        spread[done:done + m] = means.max(axis=0) - means.min(axis=0)
        done += m
    return spread
