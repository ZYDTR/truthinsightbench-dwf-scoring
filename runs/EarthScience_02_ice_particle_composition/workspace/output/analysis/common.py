"""Shared loading / statistics helpers for the ice-particle task.

All statistics are implemented locally (numpy only) so that the pipeline runs
even when optional third-party statistical packages are unavailable.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = PROJECT / "output"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"

POLYMERS = [
    "PE", "varnish", "PA", "EVA", "CE_alkylated", "PES", "PP", "PE_Cl",
    "PS", "PC", "PVC", "nitrile_rubber", "PCL", "rubber", "AB", "PLA", "PI",
]

# 22 particle-size bins, ordered as in the source table (low edge ascending).
SIZE_BINS = [
    "11um", "11_25um", "25_50um", "50_75um", "75_100um", "100_125um",
    "125_150um", "150_175um", "175_200um", "200_225um", "225_250um",
    "250_275um", "275_300um", "300_325um", "325_350um", "350_375um",
    "375_400um", "400_425um", "425_450um", "450_475um", "475_500um",
    "gt_500um",
]
SIZE_MID = np.array([
    11, 18, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5, 237.5,
    262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5, 462.5, 487.5,
    600.0,
])


TEXT_COLS = {"sample_id", "collection_group_id", "ice_regime"}


def _num(value: str):
    value = value.strip()
    if value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def load():
    """Return (comp, env) as dict-of-lists plus group/regime maps."""
    comp_rows = list(csv.DictReader(open(DATA / "particle_composition_profiles.tsv"),
                                    delimiter="\t"))
    env_rows = list(csv.DictReader(open(DATA / "sample_environment_profiles.tsv"),
                                   delimiter="\t"))
    return comp_rows, env_rows


def matrices(comp_rows):
    """Build numeric matrices keyed by column name (NaN where missing)."""
    cols = list(comp_rows[0].keys())
    M = {c: np.array([_num(r[c]) for r in comp_rows], dtype=float)
         for c in cols if c not in TEXT_COLS}
    M["sample_id"] = [r["sample_id"] for r in comp_rows]
    M["collection_group_id"] = [r["collection_group_id"] for r in comp_rows]
    return M


def group_labels(M):
    groups = M["collection_group_id"]
    return sorted(set(groups))


def spearman(x, y):
    """Spearman rho with a two-sided p-value from the exact-free t approximation."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    rx = _rank(x)
    ry = _rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    p = _t_pvalue(rho, n - 2)
    return rho, p, n


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    return r, _t_pvalue(r, n - 2), n


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def _t_pvalue(r, df):
    """Two-sided p for a correlation via the t distribution (stdlib math)."""
    if not np.isfinite(r) or df <= 0:
        return np.nan
    if abs(r) >= 1.0:
        return 0.0
    t = r * math.sqrt(df / (1 - r * r))
    return _t_sf_two_sided(t, df)


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_two_sided(t, df):
    """P(|T| > |t|) for Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def t_pvalue_two_sided(t, df):
    return _t_sf_two_sided(t, df)


def f_pvalue(F, df1, df2):
    if not np.isfinite(F) or F <= 0:
        return np.nan
    x = df2 / (df2 + df1 * F)
    return _betai(df2 / 2.0, df1 / 2.0, x)


def ols(y, X, names=None):
    """Plain OLS with classical (homoskedastic) standard errors.

    X must already include an intercept column if wanted.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    df = n - k
    sigma2 = float(resid @ resid) / df if df > 0 else np.nan
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pvals = np.array([_t_sf_two_sided(t, df) for t in tstat])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / df if df > 0 else np.nan
    return {
        "n": n, "k": k, "df": df, "beta": beta, "se": se, "t": tstat,
        "p": pvals, "r2": r2, "adj_r2": adj, "resid": resid,
        "names": names or [f"x{i}" for i in range(k)],
        "sigma2": sigma2,
    }


def ols_cluster(y, X, groups, names=None):
    """OLS with cluster-robust (CR1) standard errors.

    The analysis unit is one depth interval; intervals within one collection
    group are repeated observations, so groups are the clustering dimension.
    """
    fit = ols(y, X, names)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    resid = fit["resid"]
    groups = np.asarray(groups)
    XtX_inv = np.linalg.pinv(X.T @ X)
    unique = list(dict.fromkeys(groups.tolist()))
    G = len(unique)
    n, k = X.shape
    meat = np.zeros((k, k))
    for g in unique:
        m = groups == g
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    if G > 1 and n > k:
        scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    else:
        scale = 1.0
    cov = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tstat = fit["beta"] / se
    # cluster-robust inference uses G-1 degrees of freedom
    df = max(G - 1, 1)
    pvals = np.array([_t_sf_two_sided(t, df) for t in tstat])
    out = dict(fit)
    out.update({"se": se, "t": tstat, "p": pvals, "df_cluster": df,
                "n_clusters": G, "cov": cov, "se_type": "cluster_CR1"})
    return out


def perm_test(fn, n_perm=20000, seed=12345, observed=None):
    """Permutation test helper: fn(idx_perm) returns a statistic."""
    rng = np.random.default_rng(seed)
    obs = observed
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = fn(rng)
    null = null[np.isfinite(null)]
    if obs is None:
        return null
    extreme = int(np.sum(np.abs(null) >= abs(obs)))
    p = (extreme + 1) / (len(null) + 1)
    return {"observed": float(obs), "p_perm": float(p),
            "n_perm": int(len(null)),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975))}