"""Independent verification of Finding 1 claims (rows/regimes/time-axis validity).

Written from scratch; reads only data/*.tsv and output/Result.md (for exact claims).
Does NOT import or execute output/analysis/analysis.py.

Methods deliberately differ from the obvious ones:
  * OLS via normal equations AND via a brute-force grid search / closed form.
  * Pearson r via np.corrcoef AND via the raw covariance formula by hand AND via
    a rank-based Spearman cross-check.
  * Categorical R2 via group means AND via a dummy-variable OLS solve.
"""

import json
import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
df = pd.read_csv(f"{ROOT}/data/time_response_curves.tsv", sep="\t")
n = len(df)
row = np.arange(n)
CAND = ["promoter_like_signal", "condition_a", "condition_b"]

out = {}


def r_pearson(a, b):
    """Pearson r via raw covariance formula (not np.corrcoef)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    am, bm = a.mean(), b.mean()
    cov = ((a - am) * (b - bm)).sum() / len(a)
    sa = np.sqrt(((a - am) ** 2).sum() / len(a))
    sb = np.sqrt(((b - bm) ** 2).sum() / len(b))
    return cov / (sa * sb)


def r_pearson_np(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def ols_normaleq(x, y):
    """OLS slope/intercept/R2 via normal equations on design [1, x]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.solve(A.T @ A, A.T @ y)
    pred = A @ beta
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float(beta[1]), float(beta[0]), float(1 - sse / sst)


def ols_grid(x, y, lo=-100.0, hi=120.0, steps=200001):
    """Independent check of the slope by a brute-force grid search on R2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    # R2 as a function of slope only, with intercept profiled out:
    #   sse(b) = syy - 2 b sxy + b^2 sxx   (about means)
    syy = ((y - ym) ** 2).sum()
    best = None
    for b in np.linspace(lo, hi, steps):
        sse = syy - 2 * b * sxy + b * b * sxx
        if best is None or sse < best[1]:
            best = (b, sse)
    b_grid, sse_grid = best
    r2_grid = 1 - sse_grid / syy
    return float(b_grid), float(r2_grid)


def spearman(a, b):
    """Rank correlation implemented via ranks + Pearson (no scipy)."""
    a = pd.Series(np.asarray(a, float)).rank().to_numpy()
    b = pd.Series(np.asarray(b, float)).rank().to_numpy()
    return r_pearson(a, b)


def r2_categorical_groups(y, g):
    """Between-group SS / total SS."""
    y = np.asarray(y, float)
    g = np.asarray(g)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_between = 0.0
    for k in np.unique(g):
        m = g == k
        ss_between += m.sum() * (y[m].mean() - y.mean()) ** 2
    return float(ss_between / ss_tot)


def r2_categorical_dummies(y, g):
    """Independent check: OLS on a full dummy-variable design, R2 = 1 - SSE/SST."""
    y = np.asarray(y, float)
    g = np.asarray(g)
    levels = np.unique(g)
    D = np.column_stack([(g == k).astype(float) for k in levels])  # one-hot, no intercept
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    pred = D @ beta
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - sse / sst)


# ---------------------------------------------------------------- 1. regimes
t = df["time"].to_numpy()
m_ge2 = t >= 2
idx_ge2 = np.where(m_ge2)[0]

# contiguous runs of rows with time >= 2
runs = []
start = None
for i in range(n):
    if m_ge2[i]:
        if start is None:
            start = i
    else:
        if start is not None:
            runs.append((int(start), int(i - 1)))
            start = None
if start is not None:
    runs.append((int(start), int(n - 1)))

out["n_rows"] = n
out["n_distinct_time"] = int(df["time"].nunique())
out["time_support"] = {int(k): int(v) for k, v in df["time"].value_counts().items()}
out["n_rows_time_ge_2"] = int(m_ge2.sum())
out["contiguous_runs_ge2"] = runs
out["n_rows_in_contiguous_runs"] = int(sum(e - s + 1 for s, e in runs))

# regime boundaries per report
A = slice(0, 259)
B = slice(259, 294)
C = slice(294, 1367)
out["regime_n"] = {"A": 259, "B": 35, "C": 1073}
out["regime_time_values"] = {
    "A": sorted(int(x) for x in df["time"].iloc[A].unique()),
    "B": sorted(int(x) for x in df["time"].iloc[B].unique()),
    "C": sorted(int(x) for x in df["time"].iloc[C].unique()),
}
out["regime_multipoint_times_B"] = sorted(int(x) for x in df["time"].iloc[B].unique() if x >= 2)
out["B_has_multi_point_count"] = len(out["regime_multipoint_times_B"])

# rows outside B with time >= 2
outside = [int(i) for i in idx_ge2 if not (259 <= i <= 293)]
out["rows_outside_B_time_ge_2"] = outside
out["n_rows_outside_B_time_ge_2"] = len(outside)

# fitted columns exactly zero in A / first nonzero row
for c in ["condition_c", "fit_a", "fit_b"]:
    v = df[c].to_numpy()
    nz = np.where(v != 0)[0]
    out[f"{c}_all_zero_in_A"] = bool((v[0:259] == 0).all())
    out[f"{c}_first_nonzero_row"] = int(nz[0]) if len(nz) else None
    out[f"{c}_n_zero_total"] = int((v == 0).sum())

# ------------------------------------------------- 2. categorical R2 on time
prom = df["promoter_like_signal"].to_numpy()
out["r2_cat_time_pooled_groups"] = r2_categorical_groups(prom, t)
out["r2_cat_time_pooled_dummies"] = r2_categorical_dummies(prom, t)
out["r2_cat_time_C_groups"] = r2_categorical_groups(prom[294:], t[294:])
out["r2_cat_time_C_dummies"] = r2_categorical_dummies(prom[294:], t[294:])
# cross-check with total SS decomposition done a third way: sum of within-group
within = 0.0
for k in np.unique(t[294:]):
    m = t[294:] == k
    within += ((prom[294:][m] - prom[294:][m].mean()) ** 2).sum()
tot = ((prom[294:] - prom[294:].mean()) ** 2).sum()
out["r2_cat_time_C_within_decomp"] = float(1 - within / tot)
# pooled, via within decomposition too
within_all = 0.0
for k in np.unique(t):
    m = t == k
    within_all += ((prom[m] - prom[m].mean()) ** 2).sum()
tot_all = ((prom - prom.mean()) ** 2).sum()
out["r2_cat_time_pooled_within_decomp"] = float(1 - within_all / tot_all)

# ------------------------------------------------- 3. value-range separation
sep = {}
for c in CAND:
    v = df[c].to_numpy()
    a, cc = v[0:259], v[294:]
    sep[c] = {
        "A_min": int(a.min()), "A_max": int(a.max()),
        "C_min": int(cc.min()), "C_max": int(cc.max()),
        "C_rows_le_A_max": int((cc <= a.max()).sum()),
        "C_n": int(len(cc)),
        "C_frac_le_A_max": float((cc <= a.max()).mean()),
        "overlap": bool((cc <= a.max()).any()),
    }
out["range_separation"] = sep

# ------------------------------------------------- 4. leave-one-regime-out trend
lot = {}
for name, sl in [("drop_A", None), ("drop_B", None), ("drop_C", None)]:
    keep = np.ones(n, bool)
    if name == "drop_A":
        keep[0:259] = False
    elif name == "drop_B":
        keep[259:294] = False
    else:
        keep[294:] = False
    x = t[keep].astype(float)
    y = prom[keep]
    b, a0, r2 = ols_normaleq(x, y)
    bg, r2g = ols_grid(x, y)
    lot[name] = {"n_kept": int(keep.sum()), "slope": b, "intercept": a0, "r2": r2,
                 "slope_grid": bg, "r2_grid": r2g}
b, a0, r2 = ols_normaleq(t.astype(float), prom)
lot["pooled"] = {"n_kept": n, "slope": b, "intercept": a0, "r2": r2}
bg, r2g = ols_grid(t.astype(float), prom)
lot["pooled"]["slope_grid"] = bg
lot["pooled"]["r2_grid"] = r2g
out["leave_one_regime_out"] = lot

# ------------------------------------------------- 5. boundary sensitivity 7x7
a_end0, b_end0 = 258, 293
sens = []
for da in [-40, -20, -10, 0, 10, 20, 40]:
    for db in [-40, -20, -10, 0, 10, 20, 40]:
        a1, b1 = a_end0 + da, b_end0 + db
        if not (60 <= a1 < b1 - 10) or b1 >= n - 30:
            continue
        base = df.iloc[: a1 + 1]
        tail = df.iloc[b1 + 1:]
        sens.append({
            "a_end": a1, "b_end": b1,
            "baseline_n": len(base), "tail_n": len(tail),
            "baseline_ab_r": r_pearson(base["condition_a"], base["condition_b"]),
            "tail_ab_r": r_pearson(tail["condition_a"], tail["condition_b"]),
        })
out["n_admissible_variants"] = len(sens)
out["tail_ab_r_min"] = min(s["tail_ab_r"] for s in sens)
out["tail_ab_r_max"] = max(s["tail_ab_r"] for s in sens)
out["baseline_ab_r_min"] = min(s["baseline_ab_r"] for s in sens)
out["baseline_ab_r_max"] = max(s["baseline_ab_r"] for s in sens)
out["boundary_variants"] = sens

# ------------------------------------------------- 6. report's own counterevidence
b_A = df["condition_b"].iloc[:259]
out["condition_b_zero_in_A"] = int((b_A == 0).sum())
out["condition_b_nonzero_in_A"] = int((b_A != 0).sum())
out["condition_b_A_n"] = int(len(b_A))

# exact duplicate rows (all 7 columns)
dupmask = df.duplicated(keep=False)
out["n_rows_in_duplicate_groups"] = int(dupmask.sum())
out["n_excess_duplicate_rows"] = int(df.duplicated().sum())
out["n_unique_rows"] = int(len(df.drop_duplicates()))
out["n_distinct_row_patterns_count"] = int(df.drop_duplicates().shape[0])

# ------------------------------------------------- 7. regime-level correlations (context)
def regime_corr(sl):
    d = df.iloc[sl]
    return {f"{c1}~{c2}": r_pearson(d[c1], d[c2]) for c1, c2 in
            [("promoter_like_signal", "condition_a"),
             ("promoter_like_signal", "condition_b"),
             ("condition_a", "condition_b")]}
out["regime_corr"] = {"A": regime_corr(A), "B": regime_corr(B), "C": regime_corr(C),
                      "pooled": regime_corr(slice(0, n))}
out["spearman_ab_pooled"] = spearman(df["condition_a"], df["condition_b"])
out["pearson_ab_pooled"] = r_pearson(df["condition_a"], df["condition_b"])
out["pearson_ab_pooled_np"] = r_pearson_np(df["condition_a"], df["condition_b"])

# single-valuedness: max within-time range / global range for promoter
g = df.groupby("time")["promoter_like_signal"].agg(["count", "min", "max"])
multi = g[g["count"] > 1]
grange = prom.max() - prom.min()
out["prom_max_within_time_range"] = float((multi["max"] - multi["min"]).max())
out["prom_max_within_time_range_over_global"] = float((multi["max"] - multi["min"]).max() / grange)

# 2.6% share claims
out["pct_B_rows"] = 35 / 1367 * 100
out["pct_fitted_nonzero_context"] = float((df[["condition_c", "fit_a", "fit_b"]]
                                           .iloc[259:].to_numpy() != 0).any(axis=1).mean() * 100)

print(json.dumps(out, indent=1, default=str))
with open(f"{ROOT}/output/verification/verifier_finding1_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)