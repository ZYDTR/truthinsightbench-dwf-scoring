#!/usr/bin/env python3
"""Step 03 - Core scientific analysis.

Implements, with executed code only:
  A. Panel 1 : REST-ANOVA style decomposition of the 10x13x11 unreplicated
     factorial into grand mean / main effects / 2-way / 3-way, using explicit
     sum-to-zero contrast coding (so the SS terms really are orthogonal).
     Compares linear vs non-linear readings of the dominant factor.
  B. Panels 3/4 : the exact response identity R = (B-A)/(A+B).
  C. Cross-panel transfer between panels 3 and 4 under two independent
     series-alignment conventions, with leave-one-series-out and rank checks.
  D. Panel 2 : calibration/agreement analysis of the two paired readings.

Outputs: evidence/03_core.json
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
EV.mkdir(parents=True, exist_ok=True)

CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
OUT: dict = {}


def load(i):
    return pd.read_csv(DATA / f"observation_set_{i}.tsv", sep="\t")


d1, d2, d3, d4 = (load(i) for i in (1, 2, 3, 4))

# =========================================================== A. panel 1 ANOVA
p1 = d1.rename(columns={"numeric_feature_2": "A", "numeric_feature_3": "B",
                        "numeric_feature_4": "C"})
FAC = ["A", "B", "C"]


def contrast_design(df: pd.DataFrame, terms: list[tuple[str, ...]]) -> np.ndarray:
    """Sum-to-zero (effects) coding for main effects and interactions."""
    cols = []
    for term in terms:
        blocks = []
        for f in term:
            lv = sorted(df[f].unique())
            k = len(lv)
            block = np.zeros((len(df), k - 1))
            for j in range(k - 1):
                block[:, j] = np.where(df[f].to_numpy() == lv[j], 1.0,
                                       np.where(df[f].to_numpy() == lv[-1], -1.0, 0.0))
            blocks.append(block)
        # tensor product of the term blocks
        prod = blocks[0]
        for b in blocks[1:]:
            prod = (prod[:, :, None] * b[:, None, :]).reshape(len(df), -1)
        cols.append(prod)
    X = np.hstack(cols)
    return X


def fitted_ss(y: np.ndarray, X: np.ndarray) -> float:
    """Sum of squares explained by a sum-to-zero contrast design (no intercept
    column: for a balanced factorial every column already sums to zero, so the
    modelled deviation from the grand mean is X @ beta itself)."""
    if X.shape[1] == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((X @ beta) ** 2).sum())


def decompose(y: np.ndarray, df: pd.DataFrame) -> dict:
    terms = {
        "main": [("A",), ("B",), ("C",)],
        "twoway": [("A", "B"), ("A", "C"), ("B", "C")],
        "threeway": [("A", "B", "C")],
    }
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    res = {"grand_mean": float(grand), "ss_total": ss_tot}
    ss_cum = 0.0
    X_cum = None
    for name in ["main", "twoway", "threeway"]:
        X = contrast_design(df, terms[name])
        X_cum = X if X_cum is None else np.hstack([X_cum, X])
        ss_cum = fitted_ss(y, X_cum)
        res[f"ss_{name}_cumulative"] = ss_cum
        res[f"eta2_{name}_cumulative"] = ss_cum / ss_tot
        res[f"df_{name}"] = int(X.shape[1])
    # incremental SS (orthogonal by construction -> cumulative differences)
    res["ss_main"] = res["ss_main_cumulative"]
    res["ss_twoway_increment"] = res["ss_twoway_cumulative"] - res["ss_main_cumulative"]
    res["ss_threeway_increment"] = res["ss_threeway_cumulative"] - res["ss_twoway_cumulative"]
    res["eta2_main"] = res["ss_main"] / ss_tot
    res["eta2_twoway_increment"] = res["ss_twoway_increment"] / ss_tot
    res["eta2_threeway_increment"] = res["ss_threeway_increment"] / ss_tot
    res["residual_after_3way"] = ss_tot - res["ss_threeway_cumulative"]
    return res


def single_factor_eta2(y, g):
    s = pd.DataFrame({"y": y, "g": g}).dropna()
    tot = ((s["y"] - s["y"].mean()) ** 2).sum()
    m = s.groupby("g")["y"].transform("mean")
    return float(1 - ((s["y"] - m) ** 2).sum() / tot)


p1res = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    y = p1[resp].to_numpy(float)
    dec = decompose(y, p1[FAC])
    dec["single_factor_eta2"] = {f: single_factor_eta2(p1[resp], p1[f]) for f in FAC}
    dec["level_means"] = {f: {int(k): float(v) for k, v in p1.groupby(f)[resp].mean().items()}
                          for f in FAC}
    p1res[resp] = dec

# per-pair 2-way interaction SS (orthogonal increment over main effects)
pair_ss = {}
Xmain = contrast_design(p1, [("A",), ("B",), ("C",)])
for a, b in itertools.combinations(FAC, 2):
    X = np.hstack([Xmain, contrast_design(p1, [(a, b)])])
    for resp in ["numeric_feature_6", "numeric_feature_7"]:
        y = p1[resp].to_numpy(float)
        ss = fitted_ss(y, X)
        pair_ss[f"{a}x{b}:{resp}"] = ss - p1res[resp]["ss_main"]
p1res["pair_interaction_ss"] = pair_ss
# unreplicated-factorial inference: the 3-way contrast is used as the
# residual error term, and the 2-way interactions are tested against it.
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    d = p1res[resp]
    df_res = d["df_threeway"]
    ms_res = d["ss_threeway_increment"] / df_res
    d["residual_ms_using_3way"] = ms_res
    d["df_residual_using_3way"] = df_res
    for k, v in pair_ss.items():
        if not k.endswith(resp):
            continue
        df_term = {"A": 9, "B": 12, "C": 10}
        a, b = k.split(":")[0].split("x")
        dfi = (df_term[a]) * (df_term[b])
        F = (v / dfi) / ms_res
        d.setdefault("pair_F_vs_3way", {})[k] = {
            "ss": v, "df": dfi, "F": float(F),
            "p": float(1 - stats.f.cdf(F, dfi, df_res)),
            "partial_eta2_approx": float(v / (v + d["ss_threeway_increment"])),
        }
    F3 = (d["ss_threeway_increment"] / df_res) / ms_res
    d["threeway_F_vs_itself"] = float(F3)
    # best linear model search: how far does an additive (main-effect only)
    # model get on each response, in eta^2 terms
    d["additive_eta2"] = d["ss_main"] / d["ss_total"]
OUT["panel_1_anova"] = p1res

# linear vs monotone vs free reading of the dominant factor C on f7
cmeans = p1.groupby("C")["numeric_feature_7"].mean().to_numpy()
lv = np.arange(len(cmeans))
lin_r2 = float(np.corrcoef(lv, cmeans)[0, 1] ** 2)
OUT["panel_1_factorC_linearity"] = {
    "level_means_f7": [float(v) for v in cmeans],
    "r2_linear_in_level_index": lin_r2,
    "n_sign_changes": int(np.sum(np.diff(np.sign(np.diff(cmeans))) != 0)),
    "spearman_level_vs_mean": float(stats.spearmanr(lv, cmeans).statistic),
    "range_level_means": float(cmeans.max() - cmeans.min()),
    "max_adjacent_jump": float(np.abs(np.diff(cmeans)).max()),
}

# ============================================== B. exact identity in panels 3/4
ident = {}
for tag, df, A, B, R in [("panel_3", d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5"),
                         ("panel_4", d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")]:
    a, b, r = df[A].to_numpy(float), df[B].to_numpy(float), df[R].to_numpy(float)
    pred = (b - a) / (a + b)
    ident[tag] = {
        "n": int(len(r)),
        "max_abs_residual_of_(B-A)/(A+B)": float(np.abs(r - pred).max()),
        "max_abs_residual_of_2B/(A+B)-1": float(np.abs(r - (2 * b / (a + b) - 1)).max()),
        "exact_within_1e-12": bool(np.abs(r - pred).max() < 1e-12),
        "range_A": [float(a.min()), float(a.max())],
        "range_B": [float(b.min()), float(b.max())],
        "range_R": [float(r.min()), float(r.max())],
        "pearson_R_vs_AB_ratio": float(np.corrcoef(r, a / b)[0, 1]),
        "pearson_R_vs_magnitude_A": float(np.corrcoef(r, a)[0, 1]),
        "pearson_R_vs_magnitude_B": float(np.corrcoef(r, b)[0, 1]),
    }
OUT["response_identity"] = ident

# ======================================================= C. cross-panel transfer
def build_panel(df, A, B, R, series_col):
    x = pd.DataFrame({
        "series": df.index // 13,
        "cond": (df.index % 13),
        "series_label": df[series_col],
        "A": df[A], "B": df[B], "R": df[R],
    })
    x["cond"] = x["cond"].map(lambda p: CYCLE[p])
    return x


q3 = build_panel(d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5", "numeric_feature_1")
q4 = build_panel(d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6", "numeric_feature_1")

# series label per block (forward-fill the block-start identifier)
q3["series_id"] = pd.factorize(q3["series"])[0]
q4_blk_label = {b: int(d4["numeric_feature_1"].iloc[13 * b]) for b in range(7)}
q4["series_id"] = q4["series"].map(q4_blk_label)
q3_blk_label = {b: int(d3["numeric_feature_1"].iloc[13 * b]) for b in range(8)}
q3["series_id"] = q3["series"].map(q3_blk_label)

transfer = {}
for align, col3, col4 in [("row_order", "series", "series"), ("series_label", "series_id", "series_id")]:
    m = q3.rename(columns={col3: "_key"}).merge(
        q4.rename(columns={col4: "_key"}), on=["cond", "_key"], suffixes=("_3", "_4"))
    rec = {"n_cells": int(len(m)), "alignment": align}
    for col in ["A", "B", "R"]:
        x, y = m[f"{col}_3"].to_numpy(float), m[f"{col}_4"].to_numpy(float)
        rec[f"pearson_{col}"] = float(np.corrcoef(x, y)[0, 1])
        rec[f"spearman_{col}"] = float(stats.spearmanr(x, y).statistic)
    # per-condition agreement between the two panels' condition mean response
    cm = m.groupby("cond")[["R_3", "R_4"]].mean()
    rec["pearson_condition_mean_R"] = float(np.corrcoef(cm["R_3"], cm["R_4"])[0, 1])
    rec["spearman_condition_mean_R"] = float(stats.spearmanr(cm["R_3"], cm["R_4"]).statistic)
    # leave-one-series-out
    loo = []
    for s in sorted(m["_key"].unique()):
        sub = m[m["_key"] != s]
        if len(sub) > 3:
            loo.append(float(np.corrcoef(sub["R_3"], sub["R_4"])[0, 1]))
    rec["loo_pearson_R"] = {"min": float(min(loo)), "max": float(max(loo)),
                            "n_folds": len(loo),
                            "values": [float(v) for v in loo]}
    transfer[align] = rec
OUT["cross_panel_transfer"] = transfer

# condition ordering agreement (which conditions are "best")
ranks3 = q3.groupby("cond")["R"].mean().rank(ascending=False)
ranks4 = q4.groupby("cond")["R"].mean().rank(ascending=False)
OUT["condition_ranking_agreement"] = {
    "rank_panel3": {int(k): float(v) for k, v in ranks3.items()},
    "rank_panel4": {int(k): float(v) for k, v in ranks4.items()},
    "spearman_between_panels": float(stats.spearmanr(ranks3, ranks4).statistic),
    "top3_panel3": [int(k) for k in ranks3.nsmallest(3).index],
    "top3_panel4": [int(k) for k in ranks4.nsmallest(3).index],
    "bottom3_panel3": [int(k) for k in ranks3.nlargest(3).index],
    "bottom3_panel4": [int(k) for k in ranks4.nlargest(3).index],
}

# how much of R is explained by condition and by series, in each panel
def variance_split(q, label):
    r = {}
    for fac in ["cond", "series"]:
        r[f"eta2_{fac}"] = single_factor_eta2(q["R"], q[fac])
    r["eta2_both"] = float(
        1 - ((q["R"] - q.groupby(["cond", "series"])["R"].transform("mean")) ** 2).sum()
        / ((q["R"] - q["R"].mean()) ** 2).sum())
    return r


OUT["variance_split"] = {"panel_3": variance_split(q3, "p3"), "panel_4": variance_split(q4, "p4")}

# ======================================================== D. panel 2 calibration
x, y = d2["numeric_feature_1"].to_numpy(float), d2["numeric_feature_2"].to_numpy(float)
diff = x - y
slope, intercept = np.polyfit(x, y, 1)
OUT["panel_2_calibration"] = {
    "n": int(len(x)),
    "pearson": float(np.corrcoef(x, y)[0, 1]),
    "spearman": float(stats.spearmanr(x, y).statistic),
    "ols_slope": float(slope), "ols_intercept": float(intercept),
    "mean_difference": float(diff.mean()),
    "sd_difference": float(diff.std(ddof=1)),
    "bias_ci95": [float(v) for v in stats.t.interval(0.95, len(diff) - 1,
                                                     loc=diff.mean(), scale=stats.sem(diff))],
    "one_sample_t": {k: float(v) for k, v in zip(["t", "p"],
                                                 stats.ttest_1samp(diff, 0.0))},
    "max_abs_difference": float(np.abs(diff).max()),
    "shapiro_on_difference": {k: float(v) for k, v in zip(
        ["W", "p"], stats.shapiro(diff))},
    "slope_ci95": [float(v) for v in stats.t.interval(
        0.95, len(x) - 2, loc=slope,
        scale=np.sqrt(((y - (slope * x + intercept)) ** 2).sum() / (len(x) - 2)
                      / ((x - x.mean()) ** 2).sum()))],
    "spearman_footrule_total_abs_rank_shift": float(
        np.abs(stats.rankdata(x) - stats.rankdata(y)).sum()),
    "n_rank_inversions": int(sum(
        1 for i, j in itertools.combinations(range(len(x)), 2)
        if (x[i] - x[j]) * (y[i] - y[j]) < 0)),
    "abs_deviation_by_magnitude": {
        "corr_absdiff_vs_absmean": float(np.corrcoef(
            np.abs(diff), np.abs((x + y) / 2))[0, 1]),
    },
}

with (EV / "03_core.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)

# ------------------------------------------------------------------- printout
print("=" * 76)
print("PANEL 1 ANOVA (unreplicated 10x13x11 factorial)")
for resp, d in p1res.items():
    if resp == "pair_interaction_ss":
        continue
    print(f"  {resp}: grand={d['grand_mean']:.4f} SS_tot={d['ss_total']:.3f}")
    print(f"     eta2 main={d['eta2_main']:.4f}  2way_incr={d['eta2_twoway_increment']:.4f}"
          f"  3way_incr={d['eta2_threeway_increment']:.4f}  resid={d['residual_after_3way']:.3e}")
    print(f"     single-factor eta2: " +
          " ".join(f"{k}={v:.4f}" for k, v in d["single_factor_eta2"].items()))
print("  pair 2-way interaction SS:", json.dumps(pair_ss, indent=4))
print("  factor C linearity:", json.dumps(OUT["panel_1_factorC_linearity"], indent=4))
print("\nIDENTITY:", json.dumps(ident, indent=2))
print("\nTRANSFER:", json.dumps(transfer, indent=2))
print("\nCOND RANK:", json.dumps(OUT["condition_ranking_agreement"], indent=2))
print("\nVARIANCE SPLIT:", json.dumps(OUT["variance_split"], indent=2))
print("\nPANEL2:", json.dumps(OUT["panel_2_calibration"], indent=2))