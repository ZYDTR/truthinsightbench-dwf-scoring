#!/usr/bin/env python3
"""Step 07 - The transfer boundary between the panels.

Central question of the objective: does a condition-level structure recovered in
one panel transfer to another?  Three executable tests:

  1. Level-correspondence test.  Panel 1 carries a 13-level factor (B) and panels
     3 and 4 each carry 13 conditions.  Test whether the natural identity mapping
     B=0..12 <-> condition 0..12 reproduces a shared profile, against the full
     13! permutation null (estimated by Monte Carlo).
  2. Leave-one-level-out stability of that correspondence.
  3. Magnitude-vs-shape decomposition of the panel 3 <-> panel 4 cell transfer.

Outputs: evidence/07_transfer_boundary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
RNG = np.random.default_rng(4242)
CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t")
d3 = pd.read_csv(DATA / "observation_set_3.tsv", sep="\t")
d4 = pd.read_csv(DATA / "observation_set_4.tsv", sep="\t")
OUT: dict = {}

profiles = {
    "p1B_mean_f6": d1.groupby("numeric_feature_3")["numeric_feature_6"].mean().to_numpy(),
    "p1B_mean_f7": d1.groupby("numeric_feature_3")["numeric_feature_7"].mean().to_numpy(),
    "p3_cond_mean_R": d3.groupby("numeric_feature_2")["numeric_feature_5"].mean().reindex(range(13)).to_numpy(),
    "p4_cond_mean_R": d4.groupby("numeric_feature_3")["numeric_feature_6"].mean().reindex(range(13)).to_numpy(),
    "p3_cond_mean_B": d3.groupby("numeric_feature_2")["numeric_feature_4"].mean().reindex(range(13)).to_numpy(),
    "p4_cond_mean_B": d4.groupby("numeric_feature_3")["numeric_feature_5"].mean().reindex(range(13)).to_numpy(),
    "p3_cond_mean_A": d3.groupby("numeric_feature_2")["numeric_feature_3"].mean().reindex(range(13)).to_numpy(),
    "p4_cond_mean_A": d4.groupby("numeric_feature_3")["numeric_feature_4"].mean().reindex(range(13)).to_numpy(),
}
OUT["profiles"] = {k: [float(v) for v in vv] for k, vv in profiles.items()}

# ------------------------------------------------- 1. level-correspondence test
NPERM = 50000
# vectorised permutation null: Spearman on 13 levels == Pearson on ranks, so the
# whole null can be computed with one matrix product per pair.
P = np.array([RNG.permutation(13) for _ in range(NPERM)])          # NPERM x 13
corr = {}
for a in profiles:
    for b in profiles:
        if a >= b:
            continue
        x, y = profiles[a], profiles[b]
        rho = float(stats.spearmanr(x, y).statistic)
        r = float(np.corrcoef(x, y)[0, 1])
        xr = (stats.rankdata(x) - 7.0)
        yr = (stats.rankdata(y) - 7.0)
        yp = yr[P]                                                  # NPERM x 13
        denom = np.sqrt((xr ** 2).sum() * (yp ** 2).sum(axis=1))
        perm = (yp @ xr) / denom
        corr[f"{a}|{b}"] = {
            "spearman_identity": rho, "pearson_identity": r,
            "perm_p_two_sided": float((np.abs(perm) >= abs(rho)).mean()),
            "perm_p_one_sided_positive": float((perm >= rho).mean()),
            "perm_null_sd": float(perm.std(ddof=1)),
            "perm_null_abs_p95": float(np.percentile(np.abs(perm), 95)),
            "n_perm": NPERM,
        }
OUT["level_correspondence"] = corr

# which single permutation maximises agreement with p1B_mean_f6? (best possible)
best = {}
target = profiles["p1B_mean_f6"]
order = np.array([np.argsort(np.argsort(target - target.mean()))])
for other in ["p4_cond_mean_R", "p3_cond_mean_R", "p4_cond_mean_B", "p3_cond_mean_B"]:
    y = profiles[other]
    # rank-match every candidate alignment by reassigning ranks
    ry = stats.rankdata(y)
    best_achievable = float(stats.spearmanr(stats.rankdata(target), ry).statistic)
    best[other] = {"identity_spearman": float(stats.spearmanr(target, y).statistic),
                   "rank_order_is_identical": bool(np.array_equal(
                       np.argsort(np.argsort(target)), np.argsort(np.argsort(y)))),
                   "n_inversions": int(sum(
                       1 for i, j in __import__("itertools").combinations(range(13), 2)
                       if (target[i] - target[j]) * (y[i] - y[j]) < 0)),
                   "n_pairs": 78}
OUT["ordering_comparison"] = best

# ------------------------------------------- 2. leave-one-level-out stability
loo = {}
for pair in ["p1B_mean_f6|p4_cond_mean_R", "p1B_mean_f6|p3_cond_mean_R",
             "p1B_mean_f7|p4_cond_mean_R", "p1B_mean_f7|p3_cond_mean_R"]:
    a, b = pair.split("|")
    x, y = profiles[a], profiles[b]
    vals = []
    for k in range(13):
        m = np.ones(13, bool)
        m[k] = False
        vals.append(float(stats.spearmanr(x[m], y[m]).statistic))
    loo[pair] = {"values": vals, "min": float(min(vals)), "max": float(max(vals)),
                 "n_folds": 13, "all_same_sign_as_full": bool(
                     all(np.sign(v) == np.sign(stats.spearmanr(x, y).statistic) for v in vals))}
OUT["leave_one_level_out"] = loo

# ------------------------------- 3. magnitude vs shape in the panel3<->panel4 map
def frame(df, A, B, R):
    z = pd.DataFrame({"pos": df.index % 13, "A": df[A], "B": df[B], "R": df[R]})
    z["cond"] = z["pos"].map(lambda p: CYCLE[p])
    z["series"] = df.index // 13
    return z


f3 = frame(d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5")
f4 = frame(d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")
m = f3.merge(f4, on=["series", "cond"], suffixes=("_3", "_4"))
OUT["n_matched_cells"] = int(len(m))

mag = {}
for col, lab in [("A", "A"), ("B", "B")]:
    x, y = m[f"{col}_3"].to_numpy(float), m[f"{col}_4"].to_numpy(float)
    lx, ly = np.log(x), np.log(y)
    sl, ic = np.polyfit(lx, ly, 1)
    mag[lab] = {
        "pearson_raw": float(np.corrcoef(x, y)[0, 1]),
        "spearman_raw": float(stats.spearmanr(x, y).statistic),
        "pearson_log": float(np.corrcoef(lx, ly)[0, 1]),
        "log_slope": float(sl), "log_intercept": float(ic),
        "r2_log": float(1 - ((ly - (sl * lx + ic)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()),
        "within_condition_pearson_mean": float(np.mean([
            np.corrcoef(g[f"{col}_3"], g[f"{col}_4"])[0, 1] for _, g in m.groupby("cond")])),
        "between_condition_pearson": float(np.corrcoef(
            m.groupby("cond")[f"{col}_3"].mean(), m.groupby("cond")[f"{col}_4"].mean())[0, 1]),
    }
OUT["magnitude_transfer"] = mag

# shape: remove the condition and series means, then correlate the residual response
shape = {}
for src, lab in [("R", "R")]:
    r3, r4 = m["R_3"].to_numpy(float), m["R_4"].to_numpy(float)
    shape["raw"] = {"pearson": float(np.corrcoef(r3, r4)[0, 1]),
                    "spearman": float(stats.spearmanr(r3, r4).statistic)}
    res3 = r3 - m.groupby("cond")["R_3"].transform("mean").to_numpy()
    res4 = r4 - m.groupby("cond")["R_4"].transform("mean").to_numpy()
    shape["condition_centred"] = {"pearson": float(np.corrcoef(res3, res4)[0, 1]),
                                  "spearman": float(stats.spearmanr(res3, res4).statistic)}
    z3 = (r3 - r3.mean()) / r3.std(ddof=1)
    z4 = (r4 - r4.mean()) / r4.std(ddof=1)
    shape["z_scored_rmse"] = float(np.sqrt(((z3 - z4) ** 2).mean()))
    shape["z_scored_mae"] = float(np.abs(z3 - z4).mean())
    # how well does a single affine map between the panels do?
    sl, ic = np.polyfit(r3, r4, 1)
    shape["global_affine_map"] = {"slope": float(sl), "intercept": float(ic),
                                  "r2": float(np.corrcoef(r3, r4)[0, 1] ** 2),
                                  "max_abs_resid": float(np.abs(r4 - (sl * r3 + ic)).max())}
OUT["response_shape_transfer"] = shape

# the same statistic computed for the rank-block that the panels themselves supply
ranks3 = d3.loc[0:7, [f"numeric_feature_{k}" for k in range(9, 22)]].to_numpy(float)
ranks4 = d4.loc[0:7, [f"numeric_feature_{k}" for k in range(11, 24)]].to_numpy(float)
# align on the 8 shared series and the 13 conditions (both blocks use CYCLE order)
common = np.array([0, 1, 2, 3, 4, 5, 6, 7])
a3 = ranks3[common, :]                 # 8 series x 13 conditions, ascending ranks
b4 = ranks4[0:7, :]                    # 7 series x 13 conditions
OUT["supplied_rank_block_comparison"] = {
    "note": "both panels supply their own across-series rank block; compare them "
            "directly on the 7 shared series",
    "pearson": float(np.corrcoef(a3[0:7, :].ravel(), b4.ravel())[0, 1]),
    "spearman": float(stats.spearmanr(a3[0:7, :].ravel(), b4.ravel()).statistic),
    "per_condition_pearson": [float(np.corrcoef(a3[0:7, j], b4[:, j])[0, 1]) for j in range(13)],
    "mean_per_condition_pearson": float(np.mean(
        [np.corrcoef(a3[0:7, j], b4[:, j])[0, 1] for j in range(13)])),
}

with (EV / "07_transfer_boundary.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)

print("LEVEL CORRESPONDENCE (identity mapping, 13 levels):")
for k, v in sorted(OUT["level_correspondence"].items(), key=lambda kv: -abs(kv[1]["spearman_identity"])):
    print(f"  {k:38s} rho={v['spearman_identity']:+.4f} r={v['pearson_identity']:+.4f} p_perm={v['perm_p_two_sided']:.5f}")
print("\nLEAVE-ONE-LEVEL-OUT:")
for k, v in OUT["leave_one_level_out"].items():
    print(f"  {k}: [{v['min']:+.3f}, {v['max']:+.3f}]  sign-stable={v['all_same_sign_as_full']}")
print("\nMAGNITUDE:", json.dumps(OUT["magnitude_transfer"], indent=1))
print("\nSHAPE:", json.dumps(OUT["response_shape_transfer"], indent=1))
print("\nRANK BLOCK:", json.dumps(OUT["supplied_rank_block_comparison"], indent=1))