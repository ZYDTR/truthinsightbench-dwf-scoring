#!/usr/bin/env python3
"""Step 02 - Stage trend in interaction strength, with composition baselines.

Question: does `interaction_strength_like_value` shift across ordered stages?

The naive baseline is the record-weighted per-stage mean.  It is confounded
because (a) which interaction families are present differs by stage, and
(b) families differ hugely in mean strength.  We therefore compare the naive
estimator against:

  B1  family-equal weighting  (mean over families present, then across families)
  B2  unit-equal weighting    (mean over units present, then across units)
  B3  post-stratification     (reweight records so family shares match the
                               whole-table shares)
  B4  within-unit centering   (removes all time-invariant unit/family level)

and report the family-clustered bootstrap CI of the within-unit slope.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
units = pd.read_csv(OUT / "analysis_table.tsv", sep="\t")

STAGES = np.arange(int(units.stage_index.max()) + 1)
Y = "interaction_strength_like_value"
res: dict[str, object] = {}


# ---------------------------------------------------------------- estimators
def record_weighted(sub: pd.DataFrame) -> np.ndarray:
    return np.array([sub.loc[sub.stage_index == s, Y].mean() for s in STAGES])


def family_equal(sub: pd.DataFrame) -> np.ndarray:
    """Mean over interaction families present at each stage (families equally weighted)."""
    fam = sub.groupby(["stage_index", "interaction_id"])[Y].mean()
    per_stage = fam.groupby("stage_index").mean()
    return per_stage.reindex(STAGES).to_numpy(dtype=float)


def unit_equal(sub: pd.DataFrame) -> np.ndarray:
    """Mean over analysis units present at each stage (units equally weighted)."""
    un = sub.groupby(["stage_index", "unit_id"])[Y].mean()
    per_stage = un.groupby("stage_index").mean()
    return per_stage.reindex(STAGES).to_numpy(dtype=float)


def post_stratified(sub: pd.DataFrame) -> np.ndarray:
    share = sub.groupby("interaction_id").size() / len(sub)
    stage_n = sub.groupby("stage_index").size()
    fam_stage_n = sub.groupby(["stage_index", "interaction_id"]).size()
    w = np.array(
        [
            share[iid] / (fam_stage_n.loc[(st, iid)] / stage_n.loc[st])
            for st, iid in zip(sub.stage_index, sub.interaction_id)
        ]
    )
    tmp = sub.assign(_w=w)
    return np.array(
        [
            np.average(tmp.loc[tmp.stage_index == s, Y], weights=tmp.loc[tmp.stage_index == s, "_w"])
            for s in STAGES
        ]
    )


def within_centered(sub: pd.DataFrame) -> np.ndarray:
    um = sub.groupby("unit_id")[Y].transform("mean")
    centered = sub[Y] - um
    return np.array([centered[sub.stage_index == s].mean() for s in STAGES]) + sub[Y].mean()


ESTIMATORS = {
    "B0_record_weighted": record_weighted,
    "B1_family_equal": family_equal,
    "B2_unit_equal": unit_equal,
    "B3_post_stratified": post_stratified,
    "B4_within_unit_centered": within_centered,
}

curves = {name: fn(units) for name, fn in ESTIMATORS.items()}
res["stage_curves"] = {k: [round(float(v), 6) for v in c] for k, c in curves.items()}

res["curve_vs_stage"] = {}
for name, c in curves.items():
    slope = float(np.polyfit(STAGES, c, 1)[0])
    res["curve_vs_stage"][name] = {
        "pearson_r": round(float(np.corrcoef(STAGES, c)[0, 1]), 4),
        "spearman_rho": round(float(stats.spearmanr(STAGES, c).statistic), 4),
        "ols_slope_per_stage": round(slope, 5),
        "early_mean_stages_0_8": round(float(np.mean(c[:9])), 4),
        "late_mean_stages_9_17": round(float(np.mean(c[9:])), 4),
        "late_minus_early": round(float(np.mean(c[9:]) - np.mean(c[:9])), 4),
    }

# ---------------------------------------------- unit-level (analysis unit)
pair = units.groupby("unit_id")[Y].agg(["size", "min", "max", "mean", "std"])
multi = pair[pair["size"] >= 2]
res["unit_level"] = {
    "n_units_total": int(len(pair)),
    "n_units_multi_stage": int(len(multi)),
    "n_sign_persistent": int(((multi["min"] > 0) | (multi["max"] < 0)).sum()),
    "n_sign_reversing": int(((multi["min"] < 0) & (multi["max"] > 0)).sum()),
}

# per-unit slope (>=3 stages), tested against zero
slopes = []
for _, g in units.groupby("unit_id"):
    if g.stage_index.nunique() >= 3:
        slopes.append(np.polyfit(g.stage_index, g[Y], 1)[0])
slopes = np.array(slopes)
res["per_unit_slope"] = {
    "n_units": int(len(slopes)),
    "mean_slope": round(float(slopes.mean()), 5),
    "median_slope": round(float(np.median(slopes)), 5),
    "sd_slope": round(float(slopes.std(ddof=1)), 5),
    "fraction_positive": round(float((slopes > 0).mean()), 4),
    "ttest_vs_zero": {
        "statistic": round(float(stats.ttest_1samp(slopes, 0).statistic), 4),
        "pvalue": float(stats.ttest_1samp(slopes, 0).pvalue),
    },
    "wilcoxon_vs_zero": {
        "statistic": round(float(stats.wilcoxon(slopes).statistic), 2),
        "pvalue": float(stats.wilcoxon(slopes).pvalue),
    },
    "sign_test": {
        "n_positive": int((slopes > 0).sum()),
        "n_negative": int((slopes < 0).sum()),
        "pvalue_two_sided": float(
            stats.binomtest(int((slopes > 0).sum()), len(slopes), 0.5).pvalue
        ),
    },
}

# -------------------------------------------------- family-clustered bootstrap
rng = np.random.default_rng(20220920)
fams = units.interaction_id.unique()
by_fam = {f: g for f, g in units.groupby("interaction_id")}


def within_unit_slope(sub: pd.DataFrame) -> float:
    um = sub.groupby("unit_id")[Y].transform("mean")
    return float(np.polyfit(sub.stage_index, sub[Y] - um, 1)[0])


point = within_unit_slope(units)
boot = np.array(
    [
        within_unit_slope(
            pd.concat([by_fam[f] for f in rng.choice(fams, size=len(fams), replace=True)])
        )
        for _ in range(1000)
    ]
)
res["within_unit_slope_family_clustered_bootstrap"] = {
    "point": round(point, 5),
    "ci95_low": round(float(np.percentile(boot, 2.5)), 5),
    "ci95_high": round(float(np.percentile(boot, 97.5)), 5),
    "n_boot": int(len(boot)),
}

# ------------------------------------------------------- shape: linear vs ?
x = STAGES.astype(float)
ydata = curves["B4_within_unit_centered"]
lin = np.polyfit(x, ydata, 1)
quad = np.polyfit(x, ydata, 2)
ss = ((ydata - ydata.mean()) ** 2).sum()
res["trend_shape_b4"] = {
    "r2_linear": round(float(1 - ((ydata - np.polyval(lin, x)) ** 2).sum() / ss), 4),
    "r2_quadratic": round(float(1 - ((ydata - np.polyval(quad, x)) ** 2).sum() / ss), 4),
    "quadratic_curvature_coef": round(float(quad[0]), 6),
    "stage0_rank_of_18": int(stats.rankdata(ydata)[0]),
    "slope_excluding_stage0": round(
        float(np.polyfit(x[1:], ydata[1:], 1)[0]), 5
    ),
    "pearson_excluding_stage0": round(float(np.corrcoef(x[1:], ydata[1:])[0, 1]), 4),
}

# --------------------------- does stage count drive the naive stage mean?
res["naive_confound_check"] = {
    "corr_stage_rows_vs_stage_mean": round(
        float(
            np.corrcoef(
                units.groupby("stage_index").size().to_numpy(),
                curves["B0_record_weighted"],
            )[0, 1]
        ),
        4,
    ),
    "corr_family_mean_strength_vs_n_stages_present": round(
        float(
            np.corrcoef(
                units.groupby("interaction_id")[Y].mean(),
                units.groupby("interaction_id").stage_index.nunique(),
            )[0, 1]
        ),
        4,
    ),
}

(OUT / "stage_trend.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res, indent=2))