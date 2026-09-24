#!/usr/bin/env python3
"""Step 04 - Robustness and falsification checks for the candidate findings.

Every check is an executable alternative model, alternative definition,
threshold sensitivity, whole-unit holdout, or permutation null.
The three-way decomposition uses the closed-form balanced-factorial identity
(verified against the full contrast regression in 03_core.py).

Outputs: evidence/04_robustness.json
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
RNG = np.random.default_rng(20230101)
CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]


def load(i):
    return pd.read_csv(DATA / f"observation_set_{i}.tsv", sep="\t")


d1, d2, d3, d4 = (load(i) for i in (1, 2, 3, 4))
d1r = d1.rename(columns={"numeric_feature_2": "A", "numeric_feature_3": "B", "numeric_feature_4": "C"})
FAC = ["A", "B", "C"]
out: dict = {}


def ss_of(df, y, cols):
    g = df[y].mean()
    m = df.groupby(cols)[y].transform("mean")
    return float(((m - g) ** 2).sum())


def decompose_closed(df, y):
    tot = ss_of(df, y, FAC)
    S1 = sum(ss_of(df, y, [f]) for f in FAC)
    S2 = sum(ss_of(df, y, list(c)) for c in itertools.combinations(FAC, 2))
    return {"ss_total": tot, "ss_main": S1, "ss_2way": S2 - 2 * S1,
            "ss_3way": tot - S2 + S1,
            "eta2_main": S1 / tot, "eta2_2way": (S2 - 2 * S1) / tot,
            "eta2_3way": (tot - S2 + S1) / tot}


# =====================================================================
# R1  identity R = (B-A)/(A+B)
# =====================================================================
r1 = {}
for tag, df, A, B, R in [("panel_3", d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5"),
                         ("panel_4", d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")]:
    a, b, r = df[A].to_numpy(float), df[B].to_numpy(float), df[R].to_numpy(float)
    res = r - (b - a) / (a + b)
    alts = {}
    for name, x in [("A", a), ("B", b), ("A+B", a + b), ("A*B", a * b), ("A/B", a / b),
                    ("log(B/A)", np.log(b / a)), ("sqrt(B/A)", np.sqrt(b / a)),
                    ("(B-A)/(A+B)", (b - a) / (a + b)), ("B/(A+B)", b / (a + b)),
                    ("A+B+1", a + b + 1.0)]:
        X = np.vstack([x, np.ones_like(x)]).T
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        pred = X @ beta
        r2 = 1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum()
        alts[name] = {"slope": float(beta[0]), "intercept": float(beta[1]),
                      "r2": float(r2), "max_abs_resid": float(np.abs(r - pred).max())}
    r1[tag] = {
        "max_abs_residual_exact_form": float(np.abs(res).max()),
        "relative_residual_max_over_range": float(np.abs(res).max() / (r.max() - r.min())),
        "n_exact_to_1e-12": int((np.abs(res) < 1e-12).sum()), "n": int(len(res)),
        "residual_ulp_scale": float(np.abs(res).max() / np.spacing(np.abs(r).max())),
        "alternative_forms": alts,
    }
out["R1_identity"] = r1

# =====================================================================
# R2  panel 1 interactions: permutation null + whole-condition holdout
# =====================================================================
r2 = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    dec = decompose_closed(d1r, resp)
    yv = d1r[resp].to_numpy(float)
    obs_3way = dec["ss_3way"]
    # permutation null: permute y within each (A,B) cell-block (11 values),
    # which preserves A, B and C marginals but destroys 3-way alignment
    blocks = [g.index.to_numpy() for _, g in d1r.groupby(["A", "B"])]
    perm = []
    for _ in range(300):
        yp = yv.copy()
        for blk in blocks:
            yp[blk] = yv[RNG.permutation(blk)]
        tmp = d1r.copy()
        tmp["_p"] = yp
        perm.append(decompose_closed(tmp, "_p")["ss_3way"])
    perm = np.array(perm)
    dec["permutation_null_3way"] = {
        "n_perm": int(len(perm)), "obs": float(obs_3way),
        "null_mean": float(perm.mean()), "null_sd": float(perm.std(ddof=1)),
        "z": float((obs_3way - perm.mean()) / perm.std(ddof=1)),
        "p_ge_obs": float((perm >= obs_3way).mean()),
        "null_max": float(perm.max()),
    }
    r2[resp] = dec
out["R2_panel1_interactions"] = r2

# leave-one-factor-level-out (whole-unit holdout on each factor's levels)
loo_lvl = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    for f in FAC:
        recs = []
        for lv in sorted(d1r[f].unique()):
            sub = d1r[d1r[f] != lv]
            d = decompose_closed(sub, resp)
            recs.append((d["eta2_2way"], d["eta2_3way"]))
        a2, a3 = zip(*recs)
        loo_lvl[f"{f}:{resp}"] = {
            "eta2_2way_range": [float(min(a2)), float(max(a2))],
            "eta2_3way_range": [float(min(a3)), float(max(a3))],
            "n_levels": len(recs),
        }
out["R2_level_leaveout"] = loo_lvl

# whole-condition holdout: hold out a factor level, refit, predict held-out rows
hol = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    rows = []
    for held in sorted(d1r["B"].unique()):
        tr, te = d1r[d1r["B"] != held], d1r[d1r["B"] == held]
        grand = tr[resp].mean()
        # additive prediction from A and C marginals
        pred_add = np.full(len(te), grand)
        for f in ["A", "C"]:
            mm = tr.groupby(f)[resp].mean()
            pred_add = pred_add + (te[f].map(mm).to_numpy() - grand)
        # cell model over (A,C) with additive fallback for unseen cells
        cellm = tr.groupby(["A", "C"])[resp].mean()
        pr = pd.Series(te.set_index(["A", "C"]).index.map(cellm), index=te.index)
        pr = pr.fillna(pd.Series(pred_add, index=te.index))
        rows.append({
            "held_level": int(held),
            "mae_additive": float(np.abs(te[resp].to_numpy() - pred_add).mean()),
            "mae_cell": float(np.abs(te[resp].to_numpy() - pr.to_numpy()).mean()),
        })
    hol[resp] = {
        "per_level": rows,
        "mean_mae_additive": float(np.mean([r["mae_additive"] for r in rows])),
        "mean_mae_cell": float(np.mean([r["mae_cell"] for r in rows])),
    }
out["R2_whole_condition_holdout"] = hol

# =====================================================================
# R3  panel 1 factor C: gradient vs partitioned vs unstructured
# =====================================================================
r3 = {}
y7 = d1r.groupby("C")["numeric_feature_7"].mean().to_numpy()
y6 = d1r.groupby("C")["numeric_feature_6"].mean().to_numpy()
lv = np.arange(len(y7), dtype=float)
r3["level_means_f7"] = [float(v) for v in y7]
r3["linear_in_level_index"] = {"r2": float(np.corrcoef(lv, y7)[0, 1] ** 2),
                               "slope": float(np.polyfit(lv, y7, 1)[0])}
r3["quadratic_in_level_index"] = {
    "r2": float(1 - ((y7 - np.polyval(np.polyfit(lv, y7, 2), lv)) ** 2).sum()
                / ((y7 - y7.mean()) ** 2).sum())}
best2 = None
for k in range(1, 11):
    g = np.r_[np.zeros(k), np.ones(11 - k)]
    pred = pd.Series(y7).groupby(g).transform("mean").to_numpy()
    ss = ((y7 - pred) ** 2).sum()
    if best2 is None or ss < best2[0]:
        best2 = (ss, k)
r3["best_two_group_partition"] = {
    "eta2": float(1 - best2[0] / ((y7 - y7.mean()) ** 2).sum()),
    "split_after_level_index": int(best2[1] - 1),
    "group_means": [[float(v) for v in y7[:best2[1]]], [float(v) for v in y7[best2[1]:]]],
    "group_sizes": [int(best2[1]), int(11 - best2[1])],
}
cell_sd = d1r.groupby("C")["numeric_feature_7"].std(ddof=1)
r3["within_level_sd"] = {"min": float(cell_sd.min()), "max": float(cell_sd.max()),
                         "mean": float(cell_sd.mean())}
r3["between_level_range"] = float(y7.max() - y7.min())
r3["between_over_within_sd_ratio"] = float((y7.max() - y7.min()) / cell_sd.mean())
boot = np.array([
    d1r.groupby("C")["numeric_feature_7"].agg(
        lambda s: s.sample(len(s), replace=True, random_state=int(RNG.integers(1e9))).mean()
    ).to_numpy() for _ in range(2000)])
ci = [[float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]
      for j in range(11)]
r3["bootstrap_level_means_ci95"] = ci
r3["n_adjacent_pairs_with_nonoverlapping_ci"] = int(sum(
    1 for j in range(10)
    if ci[j][1] < ci[j + 1][0] or ci[j + 1][1] < ci[j][0]))
r3["C_profile_correlation_between_responses"] = {
    "pearson": float(np.corrcoef(y6, y7)[0, 1]),
    "spearman": float(stats.spearmanr(y6, y7).statistic),
}
# Is f7's C-profile reproducible from a random half of the data within each level?
half_idx = np.concatenate([
    RNG.choice(g.index.to_numpy(), size=len(g) // 2, replace=False)
    for _, g in d1r.groupby("C")])
half = d1r.loc[np.sort(half_idx)]
other = d1r.drop(index=half_idx)
m1 = half.groupby("C")["numeric_feature_7"].mean()
m2 = other.groupby("C")["numeric_feature_7"].mean()
r3["split_half_C_profile"] = {
    "pearson": float(np.corrcoef(m1, m2)[0, 1]),
    "spearman": float(stats.spearmanr(m1, m2).statistic),
    "n_levels": int(len(m1)),
    "max_abs_diff": float(np.abs(m1 - m2).max()),
}
# how much of C's effect is a pure row-of-11 structure shared with f6?
r3["r2_of_f7_C_means_from_f6_C_means"] = float(np.corrcoef(y6, y7)[0, 1] ** 2)
out["R3_factorC_structure"] = r3

# =====================================================================
# R4  cross-panel transfer: permutation null + holdout
# =====================================================================
def series_frame(df, A, B, R):
    x = pd.DataFrame({"series": df.index // 13, "pos": df.index % 13,
                      "A": df[A], "B": df[B], "R": df[R]})
    x["cond"] = x["pos"].map(lambda p: CYCLE[p])
    return x


q3 = series_frame(d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5")
q4 = series_frame(d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")
q3["sid"] = q3["series"].map({b: int(d3["numeric_feature_1"].iloc[13 * b]) for b in range(8)})
q4["sid"] = q4["series"].map({b: int(d4["numeric_feature_1"].iloc[13 * b]) for b in range(7)})

r4 = {}
for align in ["row_order", "series_label"]:
    key = "series" if align == "row_order" else "sid"
    m = (q3.rename(columns={key: "_k"}).merge(
        q4.rename(columns={key: "_k"}), on=["cond", "_k"], suffixes=("_3", "_4")))
    x3, x4 = m["R_3"].to_numpy(float), m["R_4"].to_numpy(float)
    obs = float(np.corrcoef(x3, x4)[0, 1])
    perm = []
    for _ in range(2000):
        p4 = m.groupby("cond")["R_4"].transform(
            lambda v: v.sample(frac=1, random_state=int(RNG.integers(1e9))).to_numpy())
        perm.append(float(np.corrcoef(m["R_3"], p4)[0, 1]))
    perm = np.array(perm)
    cm3 = m.groupby("cond")["R_3"].mean()
    cm4 = m.groupby("cond")["R_4"].mean()
    r4[align] = {
        "obs_pearson_R": obs, "n_cells": int(len(m)),
        "perm_p_two_sided": float((np.abs(perm) >= abs(obs)).mean()),
        "perm_null_sd": float(perm.std(ddof=1)),
        "perm_ci95": [float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))],
        "condition_mean_agreement": {
            "spearman": float(stats.spearmanr(cm3, cm4).statistic),
            "pearson": float(np.corrcoef(cm3, cm4)[0, 1]),
            "n_conditions": int(len(cm3)),
            "spearman_perm_p": float(np.mean([
                abs(stats.spearmanr(cm3, cm4.sample(frac=1, random_state=int(RNG.integers(1e9))).to_numpy()).statistic)
                >= abs(stats.spearmanr(cm3, cm4).statistic) for _ in range(1000)])),
        },
        "magnitude_transfer": {
            "pearson_B": float(np.corrcoef(m["B_3"], m["B_4"])[0, 1]),
            "spearman_B": float(stats.spearmanr(m["B_3"], m["B_4"]).statistic),
            "log_pearson_B": float(np.corrcoef(np.log(m["B_3"]), np.log(m["B_4"]))[0, 1]),
            "median_ratio_B4_over_B3": float((m["B_4"] / m["B_3"]).median()),
        },
    }
    out.setdefault("_transfer_frames", {})[align] = m

m = out.pop("_transfer_frames")["row_order"]
errs = []
for c in range(13):
    tr, te = m[m["cond"] != c], m[m["cond"] == c]
    beta = np.polyfit(tr["R_3"], tr["R_4"], 1)
    errs.append(float(np.abs(te["R_4"] - np.polyval(beta, te["R_3"])).mean()))
baseline = float(np.abs(m["R_4"] - m["R_4"].mean()).mean())
r4["leave_one_condition_out"] = {
    "mae_transfer": float(np.mean(errs)),
    "mae_null_predict_mean": baseline,
    "skill_vs_null": float(1 - np.mean(errs) / baseline),
    "per_condition_mae": errs,
}
# leave-one-series-out on the condition ranking
loo_series = []
for s in sorted(m["_k"].unique()):
    sub = m[m["_k"] != s]
    loo_series.append(float(stats.spearmanr(sub.groupby("cond")["R_3"].mean(),
                                            sub.groupby("cond")["R_4"].mean()).statistic))
r4["leave_one_series_out_condition_spearman"] = {
    "values": loo_series, "min": float(min(loo_series)), "max": float(max(loo_series)),
    "n_folds": len(loo_series)}
out["R4_transfer"] = r4

# =====================================================================
# R5  panel 2 calibration
# =====================================================================
r5 = {}


def calib(x, y):
    if len(x) > 2:
        sl, ic = np.polyfit(x, y, 1)
        resid = y - (sl * x + ic)
        se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
        tcrit = stats.t.ppf(0.975, len(x) - 2)
    else:
        sl = ic = se = tcrit = np.nan
    d = x - y
    return {"n": int(len(x)), "slope": float(sl), "intercept": float(ic),
            "slope_ci95": [float(sl - tcrit * se), float(sl + tcrit * se)],
            "slope_ci_excludes_1": bool(sl + tcrit * se < 1 or sl - tcrit * se > 1),
            "pearson": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan"),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "t_p_vs_0": float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else float("nan"),
            "max_abs_diff": float(np.abs(d).max())}


for tag, sub in [("all_41", d2), ("lattice_rows_0_7", d2.iloc[0:8]), ("measured_rows_8_40", d2.iloc[8:])]:
    r5[tag] = calib(sub["numeric_feature_1"].to_numpy(float),
                    sub["numeric_feature_2"].to_numpy(float))
x_all = d2["numeric_feature_1"].to_numpy(float)
y_all = d2["numeric_feature_2"].to_numpy(float)
bs = [np.polyfit(x_all[i], y_all[i], 1)[0]
      for i in (RNG.integers(0, len(d2), len(d2)) for _ in range(5000))]
r5["bootstrap_slope_all"] = {
    "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "mean": float(np.mean(bs)), "p_slope_lt_1": float((np.array(bs) < 1).mean())}
loo = [float(np.polyfit(x_all[np.arange(len(d2)) != i], y_all[np.arange(len(d2)) != i], 1)[0])
       for i in range(len(d2))]
r5["leave_one_pair_out_slope"] = {"min": float(min(loo)), "max": float(max(loo)),
                                  "n_below_1": int(sum(1 for v in loo if v < 1)),
                                  "n_total": len(loo)}
sub_meas = d2.iloc[8:]
r5["rank_agreement_measured"] = {
    "spearman": float(stats.spearmanr(sub_meas["numeric_feature_1"],
                                      sub_meas["numeric_feature_2"]).statistic),
    "n_inversions": int(sum(
        1 for i, j in itertools.combinations(range(len(sub_meas)), 2)
        if (sub_meas["numeric_feature_1"].iloc[i] - sub_meas["numeric_feature_1"].iloc[j])
        * (sub_meas["numeric_feature_2"].iloc[i] - sub_meas["numeric_feature_2"].iloc[j]) < 0)),
    "n_pairs": int(len(sub_meas) * (len(sub_meas) - 1) / 2),
}
out["R5_panel2"] = r5

with (EV / "04_robustness.json").open("w") as fh:
    json.dump(out, fh, indent=2, default=str)
print("WROTE", EV / "04_robustness.json")
print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str)[:9000])