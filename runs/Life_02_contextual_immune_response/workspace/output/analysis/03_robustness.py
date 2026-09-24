#!/usr/bin/env python3
"""Targeted robustness and alternative-explanation checks.

Each block exists to attack a specific candidate claim:

  R1  multi-condition class difference under correct repeated-measures handling
  R2  tie/threshold sensitivity of the compartment ordering failure
  R3  does the compartment ordering result survive a scale-free (rank-only) test?
  R4  is the assay effect on magnitude an artefact of replicate precision?
  R5  internal consistency of the two endpoint types (shared denominator?)
  R6  shared time trend in the time-resolved file, leave-one-replicate-out
  R7  outlier treatment: recompute headline effects with a trimmed/dropped cell

Run:  python3 output/analysis/03_robustness.py
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    LOGS,
    ensure_dirs,
    load,
    mean_ci,
    pformat,
    spearman,
    write_json,
    write_table,
)

ensure_dirs()
SECTION_LOG: list[str] = []


def say(message: str = "") -> None:
    print(message)
    SECTION_LOG.append(message)


CELL_KEYS = ["assay_id", "endpoint_type", "compartment_id"]
PRIMARY_FC = "fc_geo_mean_ratio"
PRIMARY_LOG2 = "log2fc_geo_mean_ratio"

# rebuild the same cell table used by script 01
comp = load("compartment")
rep = comp.pivot_table(
    index=CELL_KEYS + ["replicate_id"], columns="condition_role", values="response_value"
).reset_index()
rep.columns.name = None
rep["log2_fold_change"] = np.log2(rep["perturbation"] / rep["baseline"])
cell = (
    comp.groupby(CELL_KEYS + ["condition_role"])["response_value"]
    .mean()
    .reset_index()
    .pivot_table(index=CELL_KEYS, columns="condition_role", values="response_value")
    .reset_index()
)
cell.columns.name = None
cell = cell.merge(
    rep.groupby(CELL_KEYS)["log2_fold_change"].mean().rename(PRIMARY_LOG2), on=CELL_KEYS
)
cell[PRIMARY_FC] = 2.0 ** cell[PRIMARY_LOG2]
cell = cell.sort_values(CELL_KEYS).reset_index(drop=True)

# --------------------------------------------------------------------------
# R1  multi-condition: replicate-level slopes, correct repeated-measures unit
# --------------------------------------------------------------------------
say("=" * 78)
say("R1  Multi-condition class contrast under repeated-measures handling")
say("=" * 78)

multi = load("multi_condition").assign(
    setting_index=lambda d: d["compound_setting_id"].str.extract(r"(\d+)").astype(float)
)
slope_rows = []
for (klass, replicate), block in multi.groupby(
    ["response_class_id", "replicate_id"], sort=True
):
    fit = stats.linregress(block["setting_index"], block["response_value"])
    slope_rows.append(
        {
            "response_class_id": klass,
            "replicate_id": replicate,
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r_squared": float(fit.rvalue) ** 2,
        }
    )
slopes = pd.DataFrame(slope_rows)
a_slopes = slopes.loc[slopes.response_class_id == "response_class_A", "slope"].to_numpy()
b_slopes = slopes.loc[slopes.response_class_id == "response_class_B", "slope"].to_numpy()
slope_test = stats.ttest_ind(a_slopes, b_slopes, equal_var=False)

say("   per-replicate slope across the 4 compound settings (3 replicates per class):")
for _, row in slopes.iterrows():
    say(f"      {row['response_class_id']} {row['replicate_id']}: "
        f"slope = {row['slope']:+.3f} (R2 = {row['r_squared']:.3f})")
say(f"   class A slopes {np.round(a_slopes, 3).tolist()} vs class B slopes "
    f"{np.round(b_slopes, 3).tolist()}")
say(f"   Welch t on 3 vs 3 replicate slopes: t = {float(slope_test.statistic):+.3f}, "
    f"p = {pformat(float(slope_test.pvalue))}")
say(f"   the largest class B replicate slope ({b_slopes.max():.3f}) is still below the "
    f"smallest class A replicate slope ({a_slopes.min():.3f}); complete separation = "
    f"{bool(b_slopes.max() < a_slopes.min())}")

# per-replicate contrast is also testable as a rank/permutation statement
n_perm = 0
exceed = 0
pooled = np.concatenate([a_slopes, b_slopes])
for combo in itertools.combinations(range(6), 3):
    n_perm += 1
    group_a = pooled[list(combo)]
    group_b = np.delete(pooled, list(combo))
    if abs(group_a.mean() - group_b.mean()) >= abs(a_slopes.mean() - b_slopes.mean()):
        exceed += 1
say(f"   exact permutation over all C(6,3)=20 label assignments: "
    f"p = {pformat(exceed / n_perm)}")

# mixed model with replicate nested in class
from statsmodels.formula.api import mixedlm

converged = None
try:
    mm = mixedlm(
        "response_value ~ setting_index * C(response_class_id)",
        multi,
        groups=multi["replicate_id"],
    ).fit(reml=True)
    term = "setting_index:C(response_class_id)[T.response_class_B]"
    converged = {
        "formula": "response_value ~ setting_index * C(response_class_id), random intercept for replicate_id",
        "interaction_coef": float(mm.params.get(term, np.nan)),
        "interaction_p": float(mm.pvalues.get(term, np.nan)),
        "params": {str(k): float(v) for k, v in mm.params.items()},
        "pvalues": {str(k): float(v) for k, v in mm.pvalues.items()},
        "converged": bool(mm.converged),
    }
    say(f"   mixed model (replicate random intercept), setting x class interaction: "
        f"{converged['interaction_coef']:+.3f} per setting "
        f"(p = {pformat(converged['interaction_p'])}, converged = {converged['converged']})")
except Exception as error:
    say(f"   [mixed model FAILED] {type(error).__name__}: {error}")
    converged = {"error": f"{type(error).__name__}: {error}"}

write_table(slopes, "robustness_multicondition_replicate_slopes.csv")

# --------------------------------------------------------------------------
# R2  tie/threshold sensitivity of the compartment ordering
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R2  Tie sensitivity of the compartment fold-change ordering")
say("=" * 78)

order_detail = []
for (assay, endpoint), block in cell.groupby(["assay_id", "endpoint_type"], sort=True):
    ordered = block.sort_values(PRIMARY_FC, ascending=False).reset_index(drop=True)
    gaps = [
        float(ordered[PRIMARY_FC].iloc[0] - ordered[PRIMARY_FC].iloc[1]),
        float(ordered[PRIMARY_FC].iloc[1] - ordered[PRIMARY_FC].iloc[2]),
    ]
    relative_gap = gaps[1] / ordered[PRIMARY_FC].iloc[1]
    order_detail.append(
        {
            "assay_id": assay,
            "endpoint_type": endpoint,
            "order": " > ".join(ordered["compartment_id"]),
            "fc_1st": float(ordered[PRIMARY_FC].iloc[0]),
            "fc_2nd": float(ordered[PRIMARY_FC].iloc[1]),
            "fc_3rd": float(ordered[PRIMARY_FC].iloc[2]),
            "gap_1_2": gaps[0],
            "gap_2_3": gaps[1],
            "relative_gap_2_3": relative_gap,
        }
    )
order_detail = pd.DataFrame(order_detail)
say("   top-2 vs bottom-1 separation per assay x endpoint:")
for _, row in order_detail.iterrows():
    say(f"      {row['assay_id']} {row['endpoint_type']:18s} order "
        f"{row['order']:48s} FC {row['fc_1st']:.3f}/{row['fc_2nd']:.3f}/{row['fc_3rd']:.3f} "
        f"gap(2-3) = {row['gap_2_3']:+.4f} ({row['relative_gap_2_3'] * 100:+.2f}% of 2nd)")

# does the top compartment reproduce? count how often the same compartment is highest
for endpoint in ["endpoint_count", "endpoint_fraction"]:
    block = cell[cell["endpoint_type"] == endpoint]
    tops = block.loc[block.groupby("assay_id")[PRIMARY_FC].idxmax(), "compartment_id"].tolist()
    bottoms = block.loc[block.groupby("assay_id")[PRIMARY_FC].idxmin(), "compartment_id"].tolist()
    say(f"   {endpoint:18s} highest-FC compartment per assay: {tops}; "
        f"lowest: {bottoms}")

say("   robustness: recompute the ordering excluding the pair with the smallest gap")
ordering_variants = []
for (assay, endpoint), block in cell.groupby(["assay_id", "endpoint_type"], sort=True):
    ordered = block.sort_values(PRIMARY_FC, ascending=False).reset_index(drop=True)
    gap = float(ordered[PRIMARY_FC].iloc[1] - ordered[PRIMARY_FC].iloc[2])
    ambiguous = abs(gap) < 0.05 * ordered[PRIMARY_FC].iloc[1]
    ordering_variants.append(
        {
            "assay_id": assay,
            "endpoint_type": endpoint,
            "bottom_two_gap": gap,
            "bottom_two_ambiguous_at_5pct": bool(ambiguous),
            "unambiguous_top_compartment": str(ordered["compartment_id"].iloc[0]),
        }
    )
ordering_variants = pd.DataFrame(ordering_variants)
n_ambig = int(ordering_variants["bottom_two_ambiguous_at_5pct"].sum())
say(f"   assay x endpoint strata where the bottom two compartments are within 5% of each "
    f"other: {n_ambig}/{len(ordering_variants)}")

write_table(order_detail, "robustness_ordering_gaps.csv")
write_table(ordering_variants, "robustness_ordering_ambiguity.csv")

# --------------------------------------------------------------------------
# R3  scale-free rank test for the compartment gradient
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R3  Is the compartment gradient reproducible as a rank-only statement?")
say("=" * 78)

rank_rows = []
for endpoint in ["endpoint_count", "endpoint_fraction"]:
    block = cell[cell["endpoint_type"] == endpoint]
    pivot = block.pivot(index="assay_id", columns="compartment_id", values=PRIMARY_FC)
    ranks = pivot.rank(axis=1)
    say(f"   {endpoint}: rank matrix")
    for assay, row in ranks.iterrows():
        say(f"      {assay}: " + "  ".join(f"{c}={int(v)}" for c, v in row.items()))
    # Friedman test: blocks = assays (rank within assay), treatments = compartments.
    # scipy.friedmanchisquare takes one array per treatment, each of length n_blocks,
    # so the per-compartment columns must be passed -- passing the assay rows would
    # transpose the design and block by compartment instead.
    friedman = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    manual_ranks = ranks.to_numpy()
    n_blocks, k_treat = manual_ranks.shape
    rank_sums = manual_ranks.sum(axis=0)
    manual_chi2 = (
        12.0
        / (n_blocks * k_treat * (k_treat + 1))
        * ((rank_sums - n_blocks * (k_treat + 1) / 2.0) ** 2).sum()
    )
    say(f"      Friedman p = {pformat(float(friedman.pvalue))} "
        f"(chi2 = {manual_chi2:.4f}, df = {k_treat - 1}, "
        f"check via rank sums {rank_sums.tolist()})")
    say(f"      compartments whose rank is identical in all 3 assays: "
        f"{sum(1 for j in range(ranks.shape[1]) if len(set(ranks.to_numpy()[:, j])) == 1)}/3")
    rank_rows.append(
        {
            "endpoint_type": endpoint,
            "friedman_statistic": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "manual_chi2": float(manual_chi2),
            "rank_sums": [float(v) for v in rank_sums],
            "identical_rank_columns": int(
                sum(1 for j in range(ranks.shape[1]) if len(set(ranks.to_numpy()[:, j])) == 1)
            ),
        }
    )
rank_frame = pd.DataFrame(rank_rows)
write_table(rank_frame, "robustness_rank_concordance.csv")

# --------------------------------------------------------------------------
# R4  is the assay effect driven by replicate precision?
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R4  Is the assay effect on magnitude explained by replicate precision?")
say("=" * 78)

precision = (
    rep.groupby(["assay_id"])["log2_fold_change"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "log2fc_mean", "std": "log2fc_sd", "count": "n"})
    .reset_index()
)
say("   within-assay replicate scatter of log2FC:")
for _, row in precision.iterrows():
    say(f"      {row['assay_id']}: mean log2FC = {row['log2fc_mean']:.3f}, "
        f"SD = {row['log2fc_sd']:.3f}, n = {int(row['n'])}")
# Levene / Bartlett across assays
groups = [
    g["log2_fold_change"].to_numpy()
    for _, g in rep.groupby("assay_id", sort=True)
]
levene = stats.levene(*groups, center="median")
say(f"   Levene (median-centred) homogeneity of replicate variance across assays: "
    f"W = {float(levene.statistic):.3f}, p = {pformat(float(levene.pvalue))}")
# weight each replicate by the inverse of its assay's variance -> does the assay
# ordering of log2FC survive?
rep_weighted = rep.copy()
rep_weighted["weight"] = rep_weighted["assay_id"].map(
    1.0 / precision.set_index("assay_id")["log2fc_sd"] ** 2
)
weighted_cell = (
    rep_weighted.groupby(CELL_KEYS)
    .apply(
        lambda g: pd.Series(
            {
                "weighted_log2fc": np.average(
                    g["log2_fold_change"], weights=g["weight"]
                )
            }
        ),
        include_groups=False,
    )
    .reset_index()
)
merged = cell.merge(weighted_cell, on=CELL_KEYS)
say(f"   correlation between unweighted and precision-weighted cell log2FC: "
    f"Pearson r = {float(np.corrcoef(merged[PRIMARY_LOG2], merged['weighted_log2fc'])[0, 1]):.4f}")
assay_compare = (
    merged.groupby("assay_id")[[PRIMARY_LOG2, "weighted_log2fc"]].mean().round(4)
)
say("   assay means before vs after precision weighting:")
for assay, row in assay_compare.iterrows():
    say(f"      {assay}: unweighted = {row[PRIMARY_LOG2]:.4f}, "
        f"weighted = {row['weighted_log2fc']:.4f}")
say(f"   assay ordering preserved after weighting: "
    f"{list(assay_compare[PRIMARY_LOG2].sort_values(ascending=False).index) == list(assay_compare['weighted_log2fc'].sort_values(ascending=False).index)}")

write_table(precision, "robustness_assay_precision.csv")
write_table(merged[CELL_KEYS + [PRIMARY_LOG2, "weighted_log2fc"]], "robustness_weighted_cells.csv")

# --------------------------------------------------------------------------
# R5  endpoint internal consistency: shared denominator?
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R5  Endpoint internal consistency (are count and fraction linked?)")
say("=" * 78)

wide_ep = cell.pivot_table(
    index=["assay_id", "compartment_id", "condition_role"],
    columns="endpoint_type",
    values=["response_value"] if "response_value" in cell.columns else None,
) if False else None

ep = comp.pivot_table(
    index=["assay_id", "compartment_id", "condition_role", "replicate_id"],
    columns="endpoint_type",
    values="response_value",
).reset_index()
ep.columns.name = None
ep["implied_total"] = ep["endpoint_count"] / ep["endpoint_fraction"] * 100.0
say("   implied denominator (count / fraction) if fraction were a percentage of count:")
for (assay, role), block in ep.groupby(["assay_id", "condition_role"], sort=True):
    ratio_cv = float(block["implied_total"].std(ddof=1) / block["implied_total"].mean())
    say(f"      {assay} {role:13s} implied total mean = "
        f"{block['implied_total'].mean():14.2f}, CV across compartments = {ratio_cv:.3f}")
overall_cv = float(ep["implied_total"].std(ddof=1) / ep["implied_total"].mean())
say(f"   overall CV of the implied denominator = {overall_cv:.3f}: "
    f"{'constant' if overall_cv < 0.25 else 'NOT constant -> the two endpoints are independent readouts, not a shared-denominator pair'}")
corr = stats.pearsonr(np.log10(ep["endpoint_count"]), ep["endpoint_fraction"])
say(f"   Pearson r between log10(count) and fraction across all replicate rows: "
    f"r = {float(corr.statistic):+.3f}, p = {pformat(float(corr.pvalue))}, n = {len(ep)}")
within = []
for (assay, role), block in ep.groupby(["assay_id", "condition_role"], sort=True):
    if block["endpoint_fraction"].nunique() > 1:
        rho = spearman(block["endpoint_count"], block["endpoint_fraction"])
        within.append(rho)
        say(f"      within {assay} {role:13s}: Spearman(count, fraction) = {rho:+.3f}")
say(f"   mean within-stratum Spearman = {np.mean(within):+.3f}")

# --------------------------------------------------------------------------
# R6  shared time trend robustness
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R6  Time-resolved shared trend robustness")
say("=" * 78)

time_df = load("time_resolved").rename(
    columns={"normalized_response_percent": "response"}
)
time_trend_rows = []
for role, block in time_df.groupby("condition_role", sort=True):
    fit = stats.linregress(block["time_units"], block["response"])
    say(f"   {role:13s} OLS over all 28 observations: slope = {fit.slope:+.4f} "
        f"percent/unit (SE {fit.stderr:.4f}, p = {pformat(float(fit.pvalue))}, "
        f"R2 = {float(fit.rvalue) ** 2:.3f})")
    time_trend_rows.append(
        {
            "condition_role": role,
            "slope": float(fit.slope),
            "stderr": float(fit.stderr),
            "p": float(fit.pvalue),
            "r_squared": float(fit.rvalue) ** 2,
        }
    )
time_trends = pd.DataFrame(time_trend_rows)

# leave-one-replicate-out for the shared trend, per role
trend_loo = []
for role, block in time_df.groupby("condition_role", sort=True):
    for replicate in sorted(block["replicate_id"].unique()):
        sub = block[block["replicate_id"] != replicate]
        fit = stats.linregress(sub["time_units"], sub["response"])
        trend_loo.append(
            {
                "condition_role": role,
                "left_out_replicate": replicate,
                "slope": float(fit.slope),
                "p": float(fit.pvalue),
            }
        )
trend_loo = pd.DataFrame(trend_loo)
say("   leave-one-replicate-out shared trend slopes:")
for role, block in trend_loo.groupby("condition_role", sort=True):
    say(f"      {role:13s} slope range {block['slope'].min():+.4f} to "
        f"{block['slope'].max():+.4f}; all p < 0.05 = "
        f"{bool((block['p'] < 0.05).all())}")
# pooled trend ignoring role
pooled_fit = stats.linregress(time_df["time_units"], time_df["response"])
say(f"   pooled (roles combined, n=56): slope = {pooled_fit.slope:+.4f} "
    f"percent/unit, p = {pformat(float(pooled_fit.pvalue))}")
# change from first to last time point
first = time_df[time_df["time_units"] == time_df["time_units"].min()]["response"].mean()
last = time_df[time_df["time_units"] == time_df["time_units"].max()]["response"].mean()
say(f"   mean response at t={time_df['time_units'].min():.0f} = {first:.3f}; "
    f"at t={time_df['time_units'].max():.0f} = {last:.3f}; change = {last - first:+.3f} pp")
write_table(time_trends, "robustness_time_trends.csv")
write_table(trend_loo, "robustness_time_trend_loo.csv")

# --------------------------------------------------------------------------
# R7  outlier treatment on the headline compartment effect
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("R7  Outlier treatment on the headline compartment effect")
say("=" * 78)

full_ci = mean_ci(cell[PRIMARY_LOG2])
say(f"   all 18 cells: mean log2FC = {full_ci['mean']:.3f} "
    f"[{full_ci['ci_low']:.3f}, {full_ci['ci_high']:.3f}]")
# drop the single highest and single lowest cell
trimmed = cell[PRIMARY_LOG2].sort_values().iloc[1:-1]
trim_ci = mean_ci(trimmed)
say(f"   16 cells after dropping highest and lowest cell: mean = {trim_ci['mean']:.3f} "
    f"[{trim_ci['ci_low']:.3f}, {trim_ci['ci_high']:.3f}]")
# per-assay leave-out on the cell mean
assay_loo = []
for assay in sorted(cell["assay_id"].unique()):
    sub = cell[cell["assay_id"] != assay]
    ci = mean_ci(sub[PRIMARY_LOG2])
    assay_loo.append(
        {
            "left_out_assay": assay,
            "n_cells": ci["n"],
            "mean_log2fc": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
        }
    )
assay_loo = pd.DataFrame(assay_loo)
say("   leave-one-assay-out on the 18-cell mean log2FC:")
for _, row in assay_loo.iterrows():
    say(f"      omit {row['left_out_assay']}: mean = {row['mean_log2fc']:.3f} "
        f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] (all > 0 = "
        f"{bool(row['ci_low'] > 0)})")
# also on the raw replicate scale
rep_ci = mean_ci(rep["log2_fold_change"])
say(f"   replicate scale (n={rep_ci['n']}): mean log2FC = {rep_ci['mean']:.3f} "
    f"[{rep_ci['ci_low']:.3f}, {rep_ci['ci_high']:.3f}], min = "
    f"{rep['log2_fold_change'].min():.3f}")
n_below = int((rep["log2_fold_change"] <= 0).sum())
say(f"   replicate pairs with log2FC <= 0: {n_below}/{len(rep)} "
    f"(all in {sorted(rep.loc[rep['log2_fold_change'] <= 0, 'assay_id'].unique())})")
for _, row in rep[rep["log2_fold_change"] <= 0].iterrows():
    say(f"      {row['assay_id']} {row['endpoint_type']} {row['compartment_id']} "
        f"{row['replicate_id']}: baseline={row['baseline']:.6g}, "
        f"perturbation={row['perturbation']:.6g}, log2FC={row['log2_fold_change']:+.4f}")

write_table(assay_loo, "robustness_assay_leave_out.csv")

# --------------------------------------------------------------------------
summary = {
    "R1_multicondition": {
        "replicate_slopes": slopes.to_dict(orient="records"),
        "welch_t": float(slope_test.statistic),
        "welch_p": float(slope_test.pvalue),
        "complete_separation": bool(b_slopes.max() < a_slopes.min()),
        "exact_permutation_p": exceed / n_perm,
        "mixed_model": converged,
    },
    "R2_ordering": {
        "order_detail": order_detail.to_dict(orient="records"),
        "ambiguous_bottom_pairs": n_ambig,
    },
    "R3_rank_concordance": rank_frame.to_dict(orient="records"),
    "R4_precision": {
        "per_assay": precision.to_dict(orient="records"),
        "levene_p": float(levene.pvalue),
        "unweighted_vs_weighted_r": float(
            np.corrcoef(merged[PRIMARY_LOG2], merged["weighted_log2fc"])[0, 1]
        ),
        "assay_means": assay_compare.to_dict(),
    },
    "R5_endpoints": {
        "implied_denominator_cv": overall_cv,
        "pearson_log_count_fraction": {
            "r": float(corr.statistic),
            "p": float(corr.pvalue),
            "n": int(len(ep)),
        },
        "mean_within_stratum_spearman": float(np.mean(within)),
    },
    "R6_time_trends": {
        "per_role": time_trends.to_dict(orient="records"),
        "pooled_slope": float(pooled_fit.slope),
        "pooled_p": float(pooled_fit.pvalue),
        "first_to_last_change_pp": float(last - first),
    },
    "R7_outlier": {
        "full": full_ci,
        "trimmed": trim_ci,
        "assay_leave_out": assay_loo.to_dict(orient="records"),
        "replicate_scale": rep_ci,
        "replicate_pairs_nonpositive": n_below,
    },
}
write_json("robustness_summary.json", summary)

(LOGS / "03_robustness.log").write_text("\n".join(SECTION_LOG) + "\n", encoding="utf-8")
print("\n[done] 03_robustness.py")