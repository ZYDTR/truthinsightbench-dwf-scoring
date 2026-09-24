"""Supplementary falsification checks for Finding 1.

Focus:
  (a) "pooled fit gives slope 0.900 / R2 0.2497" -- which subset actually yields that?
  (b) per-regime categorical R2 to test "pooled value carried entirely by regime B"
  (c) duplicate-row count interpretation
  (d) rows >= 259 with promoter_like_signal <= 7
  (e) wider/looser boundary grids (does the [0.679, 0.973] range survive?)
  (f) fitted-column nonzero counts and coverage
  (g) hashes vs data_manifest.json (background integrity)
"""

import hashlib
import json
import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
df = pd.read_csv(f"{ROOT}/data/time_response_curves.tsv", sep="\t")
n = len(df)
t = df["time"].to_numpy()
prom = df["promoter_like_signal"].to_numpy()
res = {}


def r_hand(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    am, bm = a.mean(), b.mean()
    num = ((a - am) * (b - bm)).sum()
    return float(num / np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum()))


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.solve(A.T @ A, A.T @ y)
    pred = A @ beta
    sse = ((y - pred) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    return float(beta[1]), float(beta[0]), float(1 - sse / sst)


def r2cat(y, g):
    y = np.asarray(y, float); g = np.asarray(g)
    sst = ((y - y.mean()) ** 2).sum()
    ssb = sum(m.sum() * (y[m].mean() - y.mean()) ** 2 for m in (g == k for k in np.unique(g)))
    return float(ssb / sst)


# (a) which subsets give slope ~0.900 / R2 ~0.2497?
subsets = {
    "pooled_all_1367": np.ones(n, bool),
    "drop_A_keep_BC": np.arange(n) >= 259,
    "drop_B_keep_AC": ~((np.arange(n) >= 259) & (np.arange(n) <= 293)),
    "drop_C_keep_AB": np.arange(n) <= 293,
    "A_only": np.arange(n) <= 258,
    "B_only": (np.arange(n) >= 259) & (np.arange(n) <= 293),
    "C_only": np.arange(n) >= 294,
}
res["subset_fits"] = {}
for k, m in subsets.items():
    s, i, r2 = ols(t[m].astype(float), prom[m])
    res["subset_fits"][k] = {"n": int(m.sum()), "slope": s, "intercept": i, "r2": r2}

# (b) categorical R2 by time within each regime
res["r2cat_by_regime"] = {
    "A": r2cat(prom[:259], t[:259]),
    "B": r2cat(prom[259:294], t[259:294]),
    "C": r2cat(prom[294:], t[294:]),
    "pooled": r2cat(prom, t),
}
# contribution decomposition: SS_between total vs each regime's between-SS
gm = prom.mean()
sst = ((prom - gm) ** 2).sum()
contrib = {}
for name, sl in [("A", slice(0, 259)), ("B", slice(259, 294)), ("C", slice(294, n))]:
    y = prom[sl]; g = t[sl]
    ssb = sum((g == k).sum() * (y[g == k].mean() - gm) ** 2 for k in np.unique(g))
    contrib[name] = {"between_ss": float(ssb), "share_of_sst": float(ssb / sst)}
res["between_ss_shares"] = contrib

# (c) duplicates
res["duplicates"] = {
    "rows_in_duplicate_groups_keepFalse": int(df.duplicated(keep=False).sum()),
    "excess_rows_duplicated_default": int(df.duplicated().sum()),
    "unique_rows": int(df.drop_duplicates().shape[0]),
    "dup_by_promoter_only": int(df.duplicated(subset=["promoter_like_signal"]).sum()),
}
# where are the duplicate rows?
dup_excess = df[df.duplicated()]
res["duplicates"]["excess_rows_in_regime_A"] = int((dup_excess.index <= 258).sum())
res["duplicates"]["excess_rows_in_regime_B"] = int(((dup_excess.index >= 259) & (dup_excess.index <= 293)).sum())
res["duplicates"]["excess_rows_in_regime_C"] = int((dup_excess.index >= 294).sum())

# (d) rows >= 259 with promoter <= 7
m = np.arange(n) >= 259
res["rows_ge259_prom_le7"] = [int(i) for i in np.where(m & (prom <= 7))[0]]
res["n_rows_ge259_prom_le7"] = int((m & (prom <= 7)).sum())
m2 = np.arange(n) >= 294
res["n_rows_in_C_prom_le7"] = int((m2 & (prom <= 7)).sum())
res["n_rows_in_B_prom_le7"] = int(((np.arange(n) >= 259) & (np.arange(n) <= 293) & (prom <= 7)).sum())

# (e) boundary grids -- the report's exact 7x7 (37 admissible) plus looser grids
def grid(da_list, db_list, min_a=60, gap=10, max_b=n - 31):
    rows = []
    for da in da_list:
        for db in db_list:
            a1, b1 = 258 + da, 293 + db
            if not (min_a <= a1 < b1 - gap) or b1 > max_b:
                continue
            base = df.iloc[: a1 + 1]; tail = df.iloc[b1 + 1:]
            rows.append((a1, b1, len(base), len(tail),
                         r_hand(base["condition_a"], base["condition_b"]),
                         r_hand(tail["condition_a"], tail["condition_b"])))
    return rows


rep_grid = grid([-40, -20, -10, 0, 10, 20, 40], [-40, -20, -10, 0, 10, 20, 40])
res["report_grid"] = {
    "n_variants": len(rep_grid),
    "tail_min": min(r[5] for r in rep_grid),
    "tail_max": max(r[5] for r in rep_grid),
    "base_min": min(r[4] for r in rep_grid),
    "base_max": max(r[4] for r in rep_grid),
}
wide = grid(list(range(-40, 41, 10)), list(range(-40, 41, 10)))
res["wide_grid_10step_81_candidates"] = {
    "n_variants": len(wide),
    "tail_min": min(r[5] for r in wide),
    "tail_max": max(r[5] for r in wide),
}
vwide = grid(list(range(-60, 61, 10)), list(range(-60, 61, 10)), min_a=40, gap=5, max_b=n - 21)
res["wider_grid_60"] = {
    "n_variants": len(vwide),
    "tail_min": min(r[5] for r in vwide),
    "tail_max": max(r[5] for r in vwide),
    "base_min": min(r[4] for r in vwide),
    "base_max": max(r[4] for r in vwide),
}
# worst-case tail variants
sv = sorted(rep_grid, key=lambda r: r[5])
res["report_grid_worst_tail"] = [{"a_end": r[0], "b_end": r[1], "tail_n": r[3], "tail_ab_r": r[5]} for r in sv[:3]]
res["report_grid_best_tail"] = [{"a_end": r[0], "b_end": r[1], "tail_n": r[3], "tail_ab_r": r[5]} for r in sv[-3:]]

# (f) fitted columns: first nonzero, nonzero counts per regime
for c in ["condition_c", "fit_a", "fit_b"]:
    v = df[c].to_numpy()
    res[f"{c}"] = {
        "first_nonzero": int(np.where(v != 0)[0][0]),
        "n_zero": int((v == 0).sum()),
        "n_zero_in_A": int((v[:259] == 0).sum()),
        "n_zero_in_B": int((v[259:294] == 0).sum()),
        "n_zero_in_C": int((v[294:] == 0).sum()),
    }

# (g) integrity hashes vs manifest
man = json.load(open(f"{ROOT}/data_manifest.json"))
res["manifest_hashes"] = {}
for f in ["time_response_curves.tsv", "perturbation_time_response.tsv", "dose_response.tsv"]:
    h = hashlib.sha256(open(f"{ROOT}/data/{f}", "rb").read()).hexdigest()
    res["manifest_hashes"][f] = {"observed": h}

print(json.dumps(res, indent=1))
with open(f"{ROOT}/output/verification/verifier_finding1_extra_results.json", "w") as fh:
    json.dump(res, fh, indent=1)