#!/usr/bin/env python3
"""Core analysis: context dependence of the perturbation response across
compartments, endpoint types and mechanistic interventions.

Analysis units follow DATA_GUIDE.md: one biological replicate nested within an
assay/compartment/condition cell; repeated endpoints of a shared assay are NOT
independent experiments, so every inferential test is run either on the
condition-role mean per cell or on the assay as the held-out unit.

Covers
  S1  compartment cells: perturbation magnitude and its context dependence
  S2  cross-assay reproducibility of the compartment ordering (Kendall's W)
  S3  count vs fraction endpoint conflict
  S4  complete-assay leave-one-out cross-validation of magnitude prediction
  S5  intervention setting ordering concordance across assays
  S6  multi-condition dose response by response class

Run:  python3 output/analysis/01_compartment_intervention.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import mixedlm, ols

from common import (
    LOGS,
    ensure_dirs,
    kendall_w,
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
STRATUM_KEYS = ["assay_id", "endpoint_type"]

# --------------------------------------------------------------------------
# S1  Compartment cells
# --------------------------------------------------------------------------
comp = load("compartment")
assert comp["response_value"].notna().all(), "unexpected missingness in compartment file"
assert set(comp["condition_role"]) == {"baseline", "perturbation"}
assert set(comp["endpoint_type"]) == {"endpoint_count", "endpoint_fraction"}

# replicate-level table: baseline and perturbation read side by side
rep = comp.pivot_table(
    index=CELL_KEYS + ["replicate_id"],
    columns="condition_role",
    values="response_value",
).reset_index()
rep.columns.name = None
rep["fold_change"] = rep["perturbation"] / rep["baseline"]
rep["log2_fold_change"] = np.log2(rep["fold_change"])

# cell-level summary: three alternative fold-change definitions
frac_up = (
    rep.assign(up=rep["log2_fold_change"] > 0)
    .groupby(CELL_KEYS)["up"]
    .mean()
    .rename("replicate_fraction_up")
)
cell = (
    comp.groupby(CELL_KEYS + ["condition_role"])["response_value"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .pivot_table(index=CELL_KEYS, columns="condition_role", values="mean")
    .reset_index()
)
cell.columns.name = None
cell = cell.merge(frac_up, on=CELL_KEYS)

cell["fc_ratio_of_means"] = cell["perturbation"] / cell["baseline"]
cell["log2fc_ratio_of_means"] = np.log2(cell["fc_ratio_of_means"])
geo = rep.groupby(CELL_KEYS)["log2_fold_change"].mean().rename("log2fc_geo_mean_ratio")
arith = rep.groupby(CELL_KEYS)["fold_change"].mean().rename("fc_arith_mean_ratio")
cell = cell.merge(geo, on=CELL_KEYS).merge(arith, on=CELL_KEYS)
cell["fc_geo_mean_ratio"] = 2.0 ** cell["log2fc_geo_mean_ratio"]
n_rep = rep.groupby(CELL_KEYS)["fold_change"].size().rename("n_replicates")
cell = cell.merge(n_rep, on=CELL_KEYS).sort_values(CELL_KEYS).reset_index(drop=True)

# primary definition = geometric mean of per-replicate ratios (log-scale symmetric,
# does not let the highest-variance replicate drive the cell value)
PRIMARY_FC = "fc_geo_mean_ratio"
PRIMARY_LOG2 = "log2fc_geo_mean_ratio"

say("=" * 78)
say("S1  Compartment response cells: design and perturbation magnitude")
say("=" * 78)
say(f"rows={len(comp)}  cells={len(cell)} (expected 3 assay x 2 endpoint x 3 compartment = 18)"
    f"  replicates per cell={sorted(cell['n_replicates'].unique())}")
say("no missing response_value; no imputation applied")

for (assay, endpoint), block in cell.groupby(STRATUM_KEYS, sort=True):
    say(f"\n{assay} / {endpoint}")
    for _, row in block.iterrows():
        say(
            f"   {row['compartment_id']}: baseline={row['baseline']:.6g} "
            f"perturbation={row['perturbation']:.6g} "
            f"FC_geo={row[PRIMARY_FC]:.4f} FC_rom={row['fc_ratio_of_means']:.4f} "
            f"frac_replicates_up={row['replicate_fraction_up']:.2f}"
        )

all_up_cells = bool((cell[PRIMARY_FC] > 1).all())
n_up = int((rep["log2_fold_change"] > 0).sum())
n_pairs = int(rep["log2_fold_change"].notna().sum())
sign_p = float(stats.binomtest(n_up, n_pairs, 0.5, alternative="greater").pvalue)
ci_cell = mean_ci(cell[PRIMARY_LOG2])
t_cell = stats.ttest_1samp(cell[PRIMARY_LOG2], 0.0)

say("\n-- perturbation effect --")
say(f"   cells with FC > 1: {int((cell[PRIMARY_FC] > 1).sum())}/{len(cell)} "
    f"(all cells up = {all_up_cells})")
say(f"   replicate pairs up: {n_up}/{n_pairs}; one-sided binomial p = {pformat(sign_p)}")
say(f"   cell mean log2FC = {ci_cell['mean']:.3f} "
    f"[95% CI {ci_cell['ci_low']:.3f}, {ci_cell['ci_high']:.3f}], n = {ci_cell['n']} cells")
say(f"   one-sample t vs 0: t = {float(t_cell.statistic):.3f}, p = {pformat(float(t_cell.pvalue))}")
say(f"   median FC = {cell[PRIMARY_FC].median():.3f} "
    f"(range {cell[PRIMARY_FC].min():.3f}-{cell[PRIMARY_FC].max():.3f})")
lin_ci = mean_ci(cell["fc_arith_mean_ratio"])

# variance decomposition of log2FC on the 18 cell means
grand = cell[PRIMARY_LOG2].mean()
ss_total = float(((cell[PRIMARY_LOG2] - grand) ** 2).sum())
ss_assay = float(
    cell.groupby("assay_id")[PRIMARY_LOG2].apply(
        lambda s: len(s) * (s.mean() - grand) ** 2
    ).sum()
)
ss_endpoint = float(
    cell.groupby("endpoint_type")[PRIMARY_LOG2].apply(
        lambda s: len(s) * (s.mean() - grand) ** 2
    ).sum()
)
ss_compartment = float(
    cell.groupby("compartment_id")[PRIMARY_LOG2].apply(
        lambda s: len(s) * (s.mean() - grand) ** 2
    ).sum()
)
ss_within = ss_total - ss_assay - ss_endpoint - ss_compartment
assay_levels = {
    a: float(g[PRIMARY_LOG2].mean()) for a, g in cell.groupby("assay_id")
}
endpoint_levels = {
    e: float(g[PRIMARY_LOG2].mean()) for e, g in cell.groupby("endpoint_type")
}
compartment_levels = {
    c: float(g[PRIMARY_LOG2].mean()) for c, g in cell.groupby("compartment_id")
}

say(f"\n   variance of cell log2FC explained by each design factor "
    f"(SS fraction, one-way):")
say(f"      assay_id       {ss_assay / ss_total:.3f}   levels(log2FC): "
    f"{ {k: round(v, 3) for k, v in assay_levels.items()} }")
say(f"      endpoint_type  {ss_endpoint / ss_total:.3f}   levels(log2FC): "
    f"{ {k: round(v, 3) for k, v in endpoint_levels.items()} }")
say(f"      compartment_id {ss_compartment / ss_total:.3f}   levels(log2FC): "
    f"{ {k: round(v, 3) for k, v in compartment_levels.items()} }")

# permutation test for the assay effect on the 18 cell means
observed_assay_ss = ss_assay
rng = np.random.default_rng(20200101)
perm_assay = []
labels = cell["assay_id"].to_numpy()
values = cell[PRIMARY_LOG2].to_numpy()
for _ in range(20000):
    shuffled = rng.permutation(labels)
    group_means = pd.Series(values).groupby(shuffled).mean()
    counts = pd.Series(values).groupby(shuffled).size()
    perm_assay.append(float((counts * (group_means - grand) ** 2).sum()))
perm_assay = np.array(perm_assay)
assay_perm_p = float((perm_assay >= observed_assay_ss).mean())

mixed_fit = None
try:
    model = mixedlm(
        "log2_fold_change ~ C(endpoint_type)",
        rep,
        groups=rep["assay_id"],
    ).fit(reml=True)
    mixed_fit = {
        "formula": "replicate log2_fold_change ~ C(endpoint_type), random intercept for assay_id",
        "n_replicate_rows": int(len(rep)),
        "params": {str(k): float(v) for k, v in model.params.items()},
        "bse": {str(k): float(v) for k, v in model.bse.items()},
        "pvalues": {str(k): float(v) for k, v in model.pvalues.items()},
        "endpoint_coef": float(
            model.params.get("C(endpoint_type)[T.endpoint_fraction]", np.nan)
        ),
        "endpoint_p": float(
            model.pvalues.get("C(endpoint_type)[T.endpoint_fraction]", np.nan)
        ),
        "group_variance": float(model.cov_re.iloc[0, 0]),
        "residual_variance": float(model.scale),
    }
    say(f"   mixed model (replicate-level, random intercept for assay): "
        f"endpoint coef = {mixed_fit['endpoint_coef']:+.3f} log2FC "
        f"(p = {pformat(mixed_fit['endpoint_p'])}); "
        f"assay variance = {mixed_fit['group_variance']:.4f}, "
        f"residual = {mixed_fit['residual_variance']:.4f}")
except Exception as error:  # reported, never hidden
    say(f"   [mixed model FAILED] {type(error).__name__}: {error}")
    mixed_fit = {"error": f"{type(error).__name__}: {error}"}

say(f"   permutation test for assay identity on 18 cell means: "
    f"observed SS = {observed_assay_ss:.4f}, p = {pformat(assay_perm_p)} "
    f"(20000 label permutations, seed 20200101)")
say(f"   arithmetic-mean-of-ratios definition (alternative): mean FC = "
    f"{lin_ci['mean']:.3f} [{lin_ci['ci_low']:.3f}, {lin_ci['ci_high']:.3f}]")

write_table(cell, "compartment_cell_effects.csv")
write_table(rep, "compartment_replicate_effects.csv")

# --------------------------------------------------------------------------
# S2  Cross-assay reproducibility of the compartment ordering
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("S2  Can the compartment ordering be reproduced from the other assays?")
say("=" * 78)

order_rows = []
for endpoint, measure, column in [
    ("endpoint_count", "baseline_level", "baseline"),
    ("endpoint_count", "fold_change", PRIMARY_FC),
    ("endpoint_fraction", "baseline_level", "baseline"),
    ("endpoint_fraction", "fold_change", PRIMARY_FC),
]:
    block = cell[cell["endpoint_type"] == endpoint]
    mat = block.pivot(
        index="assay_id", columns="compartment_id", values=column
    )
    ranks = mat.rank(axis=1)
    w_value = kendall_w(ranks.to_numpy())
    k, n = ranks.shape
    chi2 = k * (n - 1) * w_value
    p_value = float(stats.chi2.sf(chi2, df=n - 1))
    order_rows.append(
        {
            "endpoint_type": endpoint,
            "measure": measure,
            "kendall_w": w_value,
            "chi2": chi2,
            "df": n - 1,
            "p_value": p_value,
            "compartment_order_per_assay": "; ".join(
                f"{assay}:{' > '.join(ranks.loc[assay].sort_values(ascending=False).index)}"
                for assay in ranks.index
            ),
        }
    )
    say(f"\n   {endpoint:18s} {measure:20s} Kendall W = {w_value:.4f} "
        f"(chi2 = {chi2:.3f}, df = {n - 1}, p = {pformat(p_value)})")
    for assay in ranks.index:
        ordered = ranks.loc[assay].sort_values(ascending=False)
        say(f"      {assay}: " + " > ".join(ordered.index))

order_frame = pd.DataFrame(order_rows)
say("\n   within-assay: does the baseline-level ordering match the fold-change ordering?")
for endpoint in ["endpoint_count", "endpoint_fraction"]:
    block = cell[cell["endpoint_type"] == endpoint]
    for assay, sub in block.groupby("assay_id"):
        by_baseline = sub.sort_values("baseline", ascending=False)["compartment_id"].to_numpy()
        by_fc = sub.sort_values(PRIMARY_FC, ascending=False)["compartment_id"].to_numpy()
        same = list(by_baseline) == list(by_fc)
        say(f"      {endpoint:18s} {assay}: baseline order "
            f"{' > '.join(by_baseline)} | FC order {' > '.join(by_fc)}  identical={same}")

write_table(order_frame, "compartment_ordering_reproducibility.csv")

# --------------------------------------------------------------------------
# S3  count vs fraction endpoint conflict
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("S3  Endpoint-type conflict: count vs fraction")
say("=" * 78)

endpoint_rows = []
for endpoint, block in cell.groupby("endpoint_type", sort=True):
    cs = mean_ci(block[PRIMARY_LOG2])
    endpoint_rows.append(
        {
            "endpoint_type": endpoint,
            "n_cells": int(len(block)),
            "mean_fc": float(block[PRIMARY_FC].mean()),
            "sd_fc": float(block[PRIMARY_FC].std(ddof=1)),
            "cv_fc": float(block[PRIMARY_FC].std(ddof=1) / block[PRIMARY_FC].mean()),
            "mean_log2fc": cs["mean"],
            "log2fc_ci_low": cs["ci_low"],
            "log2fc_ci_high": cs["ci_high"],
        }
    )
endpoint_frame = pd.DataFrame(endpoint_rows)

# paired comparison of the two endpoints, stratum = assay x compartment
pair_rows = []
for (assay, compartment), block in cell.groupby(["assay_id", "compartment_id"], sort=True):
    rec = dict(zip(block["endpoint_type"], block[PRIMARY_FC]))
    if {"endpoint_count", "endpoint_fraction"} <= set(rec):
        pair_rows.append(
            {
                "assay_id": assay,
                "compartment_id": compartment,
                "fc_count": rec["endpoint_count"],
                "fc_fraction": rec["endpoint_fraction"],
                "delta_log2fc_count_minus_fraction": np.log2(rec["endpoint_count"])
                - np.log2(rec["endpoint_fraction"]),
            }
        )
paired = pd.DataFrame(pair_rows)
paired_t = stats.ttest_rel(np.log2(paired["fc_count"]), np.log2(paired["fc_fraction"]))
paired_ci = mean_ci(paired["delta_log2fc_count_minus_fraction"])
n_frac_lower = int((paired["fc_fraction"] < paired["fc_count"]).sum())
sign_p_ep = float(
    stats.binomtest(n_frac_lower, len(paired), 0.5, alternative="two-sided").pvalue
)

# do the two endpoints agree on the compartment ordering?
ordering_rows = []
for assay, block in cell.groupby("assay_id", sort=True):
    wide = block.pivot(index="compartment_id", columns="endpoint_type", values="baseline")
    rho_baseline = spearman(wide["endpoint_count"], wide["endpoint_fraction"])
    wide_fc = block.pivot(index="compartment_id", columns="endpoint_type", values=PRIMARY_FC)
    rho_fc = spearman(wide_fc["endpoint_count"], wide_fc["endpoint_fraction"])
    ordering_rows.append(
        {
            "assay_id": assay,
            "spearman_baseline_count_vs_fraction": rho_baseline,
            "spearman_fc_count_vs_fraction": rho_fc,
        }
    )
ordering_frame = pd.DataFrame(ordering_rows)

say("   per-endpoint perturbation magnitude:")
for _, row in endpoint_frame.iterrows():
    say(f"      {row['endpoint_type']:18s} n={int(row['n_cells'])} "
        f"meanFC={row['mean_fc']:.3f} SD={row['sd_fc']:.3f} CV={row['cv_fc']:.3f} "
        f"log2FC={row['mean_log2fc']:.3f} [{row['log2fc_ci_low']:.3f}, {row['log2fc_ci_high']:.3f}]")
say(f"   paired within assay x compartment (n={len(paired)}): fraction FC < count FC in "
    f"{n_frac_lower}/{len(paired)} strata (two-sided sign test p = {pformat(sign_p_ep)})")
say(f"   mean delta log2FC (count - fraction) = {paired_ci['mean']:+.3f} "
    f"[{paired_ci['ci_low']:+.3f}, {paired_ci['ci_high']:+.3f}]; "
    f"paired t = {float(paired_t.statistic):.3f}, p = {pformat(float(paired_t.pvalue))}")
say("   compartment-ordering agreement between endpoints (Spearman over 3 compartments):")
for _, row in ordering_frame.iterrows():
    say(f"      {row['assay_id']}: baseline rho = "
        f"{row['spearman_baseline_count_vs_fraction']:+.2f}, "
        f"FC rho = {row['spearman_fc_count_vs_fraction']:+.2f}")
n_opposite = int((ordering_frame["spearman_fc_count_vs_fraction"] < 0).sum())
say(f"   assays where count and fraction give OPPOSITE compartment orderings (FC): "
    f"{n_opposite}/{len(ordering_frame)}")

write_table(endpoint_frame, "endpoint_type_comparison.csv")
write_table(paired, "endpoint_paired_deltas.csv")
write_table(ordering_frame, "endpoint_ordering_agreement.csv")

# --------------------------------------------------------------------------
# S4  Complete-assay leave-one-out cross-validation (assay is the held-out unit)
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("S4  Complete-assay leave-one-out: can baseline context predict magnitude?")
say("=" * 78)

loo_rows = []
for held_out in sorted(cell["assay_id"].unique()):
    train_cells = cell[cell["assay_id"] != held_out]
    test_cells = cell[cell["assay_id"] == held_out]
    for endpoint in sorted(cell["endpoint_type"].unique()):
        tr = train_cells[train_cells["endpoint_type"] == endpoint]
        te = test_cells[test_cells["endpoint_type"] == endpoint]

        # M1 assay-invariant: carry the training geometric-mean FC
        m1 = float(2.0 ** tr[PRIMARY_LOG2].mean())
        # M2 level-transfer: linear log2FC ~ log10(baseline) fitted on training cells
        slope, intercept = np.polyfit(np.log10(tr["baseline"]), tr[PRIMARY_LOG2], 1)
        # M3 rank-transfer: linear log2FC ~ within-stratum baseline rank
        slope_r, intercept_r = np.polyfit(
            tr["baseline"].rank(), tr[PRIMARY_LOG2], 1
        )
        for _, row in te.iterrows():
            rank_here = float(te["baseline"].rank()[row.name])
            loo_rows.append(
                {
                    "held_out_assay": held_out,
                    "endpoint_type": endpoint,
                    "compartment_id": row["compartment_id"],
                    "observed_fc": row[PRIMARY_FC],
                    "baseline": row["baseline"],
                    "pred_assay_invariant": m1,
                    "pred_level_transfer": float(
                        2.0 ** np.polyval([slope, intercept], np.log10(row["baseline"]))
                    ),
                    "pred_rank_transfer": float(
                        2.0 ** np.polyval([slope_r, intercept_r], rank_here)
                    ),
                    "train_log10_baseline_range": float(
                        np.log10(tr["baseline"]).max() - np.log10(tr["baseline"]).min()
                    ),
                }
            )
loo = pd.DataFrame(loo_rows)

loocv_metrics = {}
for column in ["pred_assay_invariant", "pred_level_transfer", "pred_rank_transfer"]:
    error = loo["observed_fc"] - loo[column]
    loocv_metrics[column] = {
        "mae_fc": float(error.abs().mean()),
        "rmse_fc": float(np.sqrt((error ** 2).mean())),
        "mean_bias_fc": float(error.mean()),
        "spearman_pred_vs_observed": float(spearman(loo[column], loo["observed_fc"])),
    }

rank_rows = []
for held_out in sorted(loo["held_out_assay"].unique()):
    for endpoint in sorted(loo["endpoint_type"].unique()):
        block = loo[(loo["held_out_assay"] == held_out) & (loo["endpoint_type"] == endpoint)]
        if len(block) < 2:
            continue
        rank_rows.append(
            {
                "held_out_assay": held_out,
                "endpoint_type": endpoint,
                "n_compartments": int(len(block)),
                "spearman_rank_transfer": spearman(
                    block["observed_fc"], block["pred_rank_transfer"]
                ),
                "spearman_level_transfer": spearman(
                    block["observed_fc"], block["pred_level_transfer"]
                ),
            }
        )
rank_frame = pd.DataFrame(rank_rows)

say(f"   held-out unit = complete assay ({sorted(cell['assay_id'].unique())}); "
    f"each fold predicts all 6 cells of one assay from the other two assays")
for name, value in loocv_metrics.items():
    say(f"   {name:22s} MAE_FC={value['mae_fc']:.4f} RMSE_FC={value['rmse_fc']:.4f} "
        f"bias={value['mean_bias_fc']:+.4f} pooled rho={value['spearman_pred_vs_observed']:.3f}")
say(f"   within-assay compartment ordering (Spearman over 3 compartments, 6 folds):")
say(f"      rank-transfer  mean rho = {rank_frame['spearman_rank_transfer'].mean():+.3f} "
    f"per-fold {[round(v, 2) for v in rank_frame['spearman_rank_transfer']]}")
say(f"      level-transfer mean rho = {rank_frame['spearman_level_transfer'].mean():+.3f} "
    f"per-fold {[round(v, 2) for v in rank_frame['spearman_level_transfer']]}")
say(f"   NOTE: training-set baseline spans "
    f"{np.log10(loo['train_log10_baseline_range'].max()):.2f} decades between assays, "
    f"so a pooled log10(baseline) model is dominated by assay identity rather than "
    f"by the within-assay compartment contrast.")

write_table(loo, "loocv_assay_predictions.csv")
write_table(rank_frame, "loocv_rank_accuracy.csv")

# --------------------------------------------------------------------------
# S5  Intervention settings across assays
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("S5  Mechanistic intervention settings: ordering across assays")
say("=" * 78)

inter = load("intervention")
assert inter["response_value"].notna().all()
inter_mean = (
    inter.groupby(["assay_id", "intervention_setting_id"])["response_value"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(
        columns={"mean": "response_mean", "std": "response_sd", "count": "n_replicates"}
    )
)
mat = inter_mean.pivot(
    index="assay_id", columns="intervention_setting_id", values="response_mean"
)
n_assays, n_settings = mat.shape
ranks = mat.rank(axis=1)
kendall = kendall_w(ranks.to_numpy())
chi2_kw = n_assays * (n_settings - 1) * kendall
kendall_p = float(stats.chi2.sf(chi2_kw, df=n_settings - 1))
friedman = stats.friedmanchisquare(*[mat.loc[a].to_numpy() for a in mat.index])

say(f"   design: {n_assays} assays x {n_settings} settings x 3 replicates = {len(inter)} rows")
for assay, row in mat.iterrows():
    say(f"   {assay:8s} " + "  ".join(f"{c[-1]}={v:9.4g}" for c, v in row.items()))
say("   rank matrix (1 = lowest response):")
for assay, row in ranks.iterrows():
    say(f"   {assay:8s} " + "  ".join(f"{c[-1]}={int(v)}" for c, v in row.items()))
say(f"   Kendall's W = {kendall:.4f} (chi2 = {chi2_kw:.4f}, df = {n_settings - 1}, "
    f"p = {pformat(kendall_p)}); Friedman p = {pformat(float(friedman.pvalue))}")
say(f"   lowest-response setting per assay: "
    f"{ {a: str(mat.loc[a].idxmin()) for a in mat.index} }")
for a in mat.index:
    for b in mat.index:
        if a < b:
            say(f"   pairwise Spearman({a}, {b}) = {spearman(mat.loc[a], mat.loc[b]):+.3f}")

loo_inter_rows = []
for held_out in mat.index:
    pooled = mat.drop(index=held_out).mean(axis=0)
    loo_inter_rows.append(
        {
            "held_out_assay": held_out,
            "spearman_pooled_training_vs_held_out": spearman(pooled, mat.loc[held_out]),
            "rank_agreement_positions": int(
                (pooled.rank() == mat.loc[held_out].rank()).sum()
            ),
        }
    )
loo_inter = pd.DataFrame(loo_inter_rows)
say("   leave-one-assay-out (train on 2 assays, rank the third):")
for _, row in loo_inter.iterrows():
    say(f"      hold out {row['held_out_assay']}: rho = "
        f"{row['spearman_pooled_training_vs_held_out']:+.3f}, "
        f"exact rank positions matched = {int(row['rank_agreement_positions'])}/{n_settings}")
say(f"   mean rho = {loo_inter['spearman_pooled_training_vs_held_out'].mean():+.3f}")

swap_rows = []
for setting in mat.columns:
    ranks_here = ranks[setting]
    swap_rows.append(
        {
            "intervention_setting_id": setting,
            "rank_assay_1": int(ranks_here["assay_1"]),
            "rank_assay_2": int(ranks_here["assay_2"]),
            "rank_assay_3": int(ranks_here["assay_3"]),
            "rank_spread": int(ranks_here.max() - ranks_here.min()),
        }
    )
swap_frame = pd.DataFrame(swap_rows)
say("   per-setting rank spread across assays (0 = perfectly reproducible):")
for _, row in swap_frame.iterrows():
    say(f"      {row['intervention_setting_id']}: ranks "
        f"({row['rank_assay_1']},{row['rank_assay_2']},{row['rank_assay_3']}) "
        f"spread = {row['rank_spread']}")
say("   counterexample: settings 2 and 5 swap between assay_2 and assays 1/3 "
    "(assay_2 ranks setting_5 above setting_2; assay_1 and assay_3 do the reverse)")

write_table(inter_mean, "intervention_setting_means.csv")
write_table(mat.reset_index().rename_axis(None, axis=1), "intervention_assay_by_setting_matrix.csv")
write_table(loo_inter, "intervention_leave_one_assay_out.csv")
write_table(swap_frame, "intervention_setting_rank_spread.csv")

# --------------------------------------------------------------------------
# S6  Multi-condition dose response by response class
# --------------------------------------------------------------------------
say("\n" + "=" * 78)
say("S6  Multi-condition response by class")
say("=" * 78)

multi = load("multi_condition")
assert multi["response_value"].notna().all()
multi = multi.assign(
    setting_index=multi["compound_setting_id"].str.extract(r"(\d+)").astype(float)
)
class_rows = []
for klass, block in multi.groupby("response_class_id", sort=True):
    fit = stats.linregress(block["setting_index"], block["response_value"])
    per_setting = block.groupby("compound_setting_id")["response_value"].mean()
    deltas = np.diff(per_setting.to_numpy())
    class_rows.append(
        {
            "response_class_id": klass,
            "n_observations": int(len(block)),
            "slope_per_setting": float(fit.slope),
            "slope_stderr": float(fit.stderr),
            "slope_p": float(fit.pvalue),
            "intercept": float(fit.intercept),
            "r_squared": float(fit.rvalue ** 2),
            "setting_means": ",".join(f"{v:.3f}" for v in per_setting.to_numpy()),
            "min_step_delta": float(deltas.min()),
            "max_step_delta": float(deltas.max()),
            "monotone_increasing": bool((deltas > 0).all()),
        }
    )
class_frame = pd.DataFrame(class_rows)

anova_model = ols(
    "response_value ~ C(response_class_id) * C(compound_setting_id)", data=multi
).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
interaction_term = "C(response_class_id):C(compound_setting_id)"
interaction_p = float(anova_table.loc[interaction_term, "PR(>F)"])
interaction_F = float(anova_table.loc[interaction_term, "F"])

say(f"   design: 2 classes x 4 settings x 3 replicates = {len(multi)} rows")
for _, row in class_frame.iterrows():
    say(f"   {row['response_class_id']}: slope = {row['slope_per_setting']:.3f} "
        f"+/- {row['slope_stderr']:.3f} per setting (p = {pformat(row['slope_p'])}), "
        f"R2 = {row['r_squared']:.4f}, monotone = {row['monotone_increasing']}")
    say(f"      setting means = [{row['setting_means']}]")
say(f"   class x setting interaction: F = {interaction_F:.3f}, p = {pformat(interaction_p)}")
write_table(class_frame, "multi_condition_class_fits.csv")
write_table(
    multi.groupby(["response_class_id", "compound_setting_id"])["response_value"]
    .agg(["mean", "std", "count"])
    .reset_index(),
    "multi_condition_setting_means.csv",
)
write_table(anova_table.reset_index().rename(columns={"index": "term"}), "multi_condition_anova.csv")

# --------------------------------------------------------------------------
# Persist machine-readable summary
# --------------------------------------------------------------------------
summary = {
    "task_id": "Life_02_contextual_immune_response",
    "primary_fold_change_definition": "geometric mean of per-replicate perturbation/baseline ratios",
    "analysis_units": {
        "compartment": "biological replicate nested in assay x compartment x endpoint x condition",
        "intervention": "biological replicate nested in assay x intervention setting",
        "multi_condition": "biological replicate nested in response class x compound setting",
        "time_resolved": "biological replicate nested in condition role x time point",
    },
    "S1_perturbation": {
        "cells_total": int(len(cell)),
        "cells_with_fc_gt_1": int((cell[PRIMARY_FC] > 1).sum()),
        "replicate_pairs_up": n_up,
        "replicate_pairs_total": n_pairs,
        "sign_test_p": sign_p,
        "cell_mean_log2fc": ci_cell,
        "cell_ttest_p": float(t_cell.pvalue),
        "median_fc": float(cell[PRIMARY_FC].median()),
        "min_fc": float(cell[PRIMARY_FC].min()),
        "max_fc": float(cell[PRIMARY_FC].max()),
        "ss_fraction_assay": ss_assay / ss_total,
        "ss_fraction_endpoint": ss_endpoint / ss_total,
        "ss_fraction_compartment": ss_compartment / ss_total,
        "assay_levels_log2fc": assay_levels,
        "endpoint_levels_log2fc": endpoint_levels,
        "compartment_levels_log2fc": compartment_levels,
        "assay_permutation_p": assay_perm_p,
        "mixed_model": mixed_fit,
    },
    "S2_ordering_reproducibility": order_frame.to_dict(orient="records"),
    "S3_endpoint_conflict": {
        "endpoint_summary": endpoint_frame.to_dict(orient="records"),
        "paired": {
            "n_strata": int(len(paired)),
            "fraction_lower_count": n_frac_lower,
            "sign_test_p": sign_p_ep,
            "mean_delta_log2fc": paired_ci,
            "paired_t": float(paired_t.statistic),
            "paired_t_p": float(paired_t.pvalue),
        },
        "ordering_agreement": ordering_frame.to_dict(orient="records"),
        "assays_with_opposite_ordering": n_opposite,
    },
    "S4_loocv": {
        "metrics": loocv_metrics,
        "rank_transfer_mean_spearman": float(rank_frame["spearman_rank_transfer"].mean()),
        "level_transfer_mean_spearman": float(rank_frame["spearman_level_transfer"].mean()),
        "per_fold": rank_frame.to_dict(orient="records"),
    },
    "S5_intervention": {
        "kendall_w": kendall,
        "kendall_chi2": float(chi2_kw),
        "kendall_p": kendall_p,
        "friedman_p": float(friedman.pvalue),
        "lowest_setting_per_assay": {str(a): str(mat.loc[a].idxmin()) for a in mat.index},
        "leave_one_assay_out": loo_inter.to_dict(orient="records"),
        "setting_rank_spread": swap_frame.to_dict(orient="records"),
    },
    "S6_multi_condition": {
        "class_fits": class_frame.to_dict(orient="records"),
        "interaction_F": interaction_F,
        "interaction_p": interaction_p,
        "anova": anova_table.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
    },
}
write_json("core_analysis_summary.json", summary)

(LOGS / "01_compartment_intervention.log").write_text(
    "\n".join(SECTION_LOG) + "\n", encoding="utf-8"
)
print("\n[done] 01_compartment_intervention.py")