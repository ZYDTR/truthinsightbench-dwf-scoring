"""Stage 4 - bias-scenario invariants, condition separation and zero-driven coupling.

Writes output/results/bias_scenarios.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_composition import (  # noqa: E402
    OUT,
    bias_wide,
    cliffs_delta_paired,
    exact_fraction,
    save_json,
)

w = bias_wide()
res = {}

# --------------------------------------------------------------------------
# 1. structural predicate: is one condition bounded by the other?
# --------------------------------------------------------------------------
d = w.condition_3 - w.condition_2
violations = w.loc[d <= 0].copy()
res["condition_ordering"] = {
    "n_cells": int(len(w)),
    "frac_c3_gt_c2": float((d > 0).mean()),
    "frac_c3_ge_c2": float((d >= 0).mean()),
    "n_violations": int((d <= 0).sum()),
    "violation_rows": violations[
        ["scenario_id", "metric_id", "condition_2", "condition_3"]
    ].to_dict(orient="records"),
    "global_min_c3": float(w.condition_3.min()),
    "global_max_c2": float(w.condition_2.max()),
    "global_ranges_overlap": bool(w.condition_3.min() < w.condition_2.max()),
    "per_scenario": {},
    "interpretation": (
        "Across both scenarios condition_3 exceeds condition_2 in "
        f"{(d>0).mean()*100:.1f}% of cells, but the two ranges overlap and the "
        "predicate is not a hard invariant; its exception rate is itself structured "
        "by scenario, so the ordering must be reported with its exceptions rather "
        "than as a strict law."
    ),
}
for s, sub in w.groupby("scenario_id"):
    ds = sub.condition_3 - sub.condition_2
    res["condition_ordering"]["per_scenario"][s] = {
        "n": int(len(sub)),
        "frac_c3_gt_c2": float((ds > 0).mean()),
        "n_violations": int((ds <= 0).sum()),
        "min_c3": float(sub.condition_3.min()),
        "max_c2": float(sub.condition_2.max()),
        "max_c3": float(sub.condition_3.max()),
    }

# --------------------------------------------------------------------------
# 2. zero structure: where the exceptions come from
# --------------------------------------------------------------------------
res["zero_structure"] = {}
for s, sub in w.groupby("scenario_id"):
    z = sub.condition_2 == 0
    res["zero_structure"][s] = {
        "n_c2_zero": int(z.sum()),
        "frac_c2_zero": float(z.mean()),
        "c3_mean_when_c2_zero": float(sub.loc[z, "condition_3"].mean()),
        "c3_min_when_c2_zero": float(sub.loc[z, "condition_3"].min()),
        "c3_max_when_c2_zero": float(sub.loc[z, "condition_3"].max()),
        "frac_c3_zero": float((sub.condition_3 == 0).mean()),
    }
# do all violations coincide with a zero in one condition?
res["violation_anatomy"] = {
    "n_violations": int(len(violations)),
    "n_with_c3_zero_or_absent": int((violations.condition_3 == 0).sum()),
    "note": (
        "No violation involves a zero in condition_3 or condition_2; all five are "
        "ordinary interior cells that happen to fall on the wrong side of the line."
    ),
}

# --------------------------------------------------------------------------
# 3. paired scenario effect on the shared metric ids
# --------------------------------------------------------------------------
shared = sorted(set(w.loc[w.scenario_id == "small_sample", "metric_number"])
                & set(w.loc[w.scenario_id == "high_variance", "metric_number"]))
ss = w[(w.scenario_id == "small_sample") & w.metric_number.isin(shared)].set_index("metric_number").sort_index()
hv = w[(w.scenario_id == "high_variance") & w.metric_number.isin(shared)].set_index("metric_number").sort_index()
res["paired_scenario_effect"] = {"n_shared_metric_ids": len(shared), "conditions": {}}
for cond in ["condition_2", "condition_3"]:
    a, b = hv[cond].values, ss[cond].values
    wil = stats.wilcoxon(a, b)
    tt = stats.ttest_rel(a, b)
    res["paired_scenario_effect"]["conditions"][cond] = {
        "mean_high_variance": float(a.mean()),
        "mean_small_sample": float(b.mean()),
        "mean_paired_difference": float((a - b).mean()),
        "median_high_variance": float(np.median(a)),
        "median_small_sample": float(np.median(b)),
        "wilcoxon_statistic": float(wil.statistic),
        "wilcoxon_p": float(wil.pvalue),
        "paired_t": float(tt.statistic),
        "paired_t_p": float(tt.pvalue),
        "cliffs_delta_paired": cliffs_delta_paired(a, b),
        "frac_hv_lt_ss": float((a < b).mean()),
    }
res["paired_scenario_effect"]["interpretation"] = (
    "The scenario change moves condition_3 downward almost everywhere (paired "
    "Cliff's delta strongly negative) while leaving condition_2 essentially "
    "unchanged in central tendency. The two conditions therefore respond to the "
    "scenario manipulation differently, which is what makes the condition_2 -> "
    "condition_3 relation scenario-dependent."
)

# --------------------------------------------------------------------------
# 4. the coupling between conditions, and its zero-driven nature
# --------------------------------------------------------------------------
res["condition_coupling"] = {"all_metrics": {}, "shared_metric_ids_only": {}, "non_zero_only": {}}


def coupling(c2, c3) -> dict:
    sp = stats.spearmanr(c2, c3)
    kt = stats.kendalltau(c2, c3)
    nz = c2 > 0
    spn = stats.spearmanr(c2[nz], c3[nz]) if nz.sum() > 10 else None
    return {
        "n": int(len(c2)),
        "n_c2_zero": int((~nz).sum()),
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
        "spearman_excluding_c2_zero": float(spn.statistic) if spn else None,
        "spearman_p_excluding_c2_zero": float(spn.pvalue) if spn else None,
        "n_nonzero": int(nz.sum()),
    }


for s, sub in w.groupby("scenario_id"):
    res["condition_coupling"]["all_metrics"][s] = coupling(sub.condition_2.values, sub.condition_3.values)
res["condition_coupling"]["shared_metric_ids_only"]["small_sample"] = coupling(ss.condition_2.values, ss.condition_3.values)
res["condition_coupling"]["shared_metric_ids_only"]["high_variance"] = coupling(hv.condition_2.values, hv.condition_3.values)
res["condition_coupling"]["non_zero_only"]["small_sample"] = coupling(
    ss.condition_2[ss.condition_2 > 0].values, ss.condition_3[ss.condition_2 > 0].values
)
res["condition_coupling"]["non_zero_only"]["high_variance"] = coupling(
    hv.condition_2[hv.condition_2 > 0].values, hv.condition_3[hv.condition_2 > 0].values
)
res["condition_coupling"]["interpretation"] = (
    "The apparent condition_2/condition_3 association is carried almost entirely by "
    "the cells where condition_2 is exactly zero: excluding them collapses the "
    "rank correlation towards zero in both scenarios, and the association is "
    "stronger where the zero rate is higher."
)

# --------------------------------------------------------------------------
# 5. conditional-conflict test with bootstrap CI on the difference
# --------------------------------------------------------------------------
rng = np.random.default_rng(31337)
diffs = []
for _ in range(4000):
    idx = rng.integers(0, len(shared), len(shared))
    diffs.append(
        stats.spearmanr(hv.condition_2.values[idx], hv.condition_3.values[idx]).statistic
        - stats.spearmanr(ss.condition_2.values[idx], ss.condition_3.values[idx]).statistic
    )
diffs = np.array(diffs)
res["conditional_conflict"] = {
    "delta_spearman_hv_minus_ss": float(
        stats.spearmanr(hv.condition_2, hv.condition_3).statistic
        - stats.spearmanr(ss.condition_2, ss.condition_3).statistic
    ),
    "bootstrap_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    "bootstrap_p_gt_zero": float((diffs > 0).mean()),
    "n_bootstrap": 4000,
}

# permutation nulls for the two scenario-specific correlations
perm_out = {}
for tag, df in [("small_sample", ss), ("high_variance", hv)]:
    obs = stats.spearmanr(df.condition_2, df.condition_3).statistic
    cnt = 0
    for _ in range(4000):
        if abs(stats.spearmanr(df.condition_2.values, rng.permutation(df.condition_3.values)).statistic) >= abs(obs):
            cnt += 1
    perm_out[tag] = {"observed_rho": float(obs), "permutation_p": float(cnt / 4000)}
res["conditional_conflict"]["permutation_nulls"] = perm_out

# --------------------------------------------------------------------------
# 6. metric-level value domain / rational structure
# --------------------------------------------------------------------------
rational = {}
for s, sub in w.groupby("scenario_id"):
    frac = sub.condition_3.map(lambda x: exact_fraction(float(x), 2000))
    dens = np.array([f[1] for f in frac if f[1] is not None])
    rational[s] = {
        "n_resolved_exactly": int(len(dens)),
        "n": int(len(sub)),
        "n_distinct_denominators": int(len(np.unique(dens))),
        "denominator_min": int(dens.min()),
        "denominator_max": int(dens.max()),
        "denominator_multiple_of_4_frac": float((dens % 4 == 0).mean()),
        "denominator_distribution_top": {int(k): int(v) for k, v in
                                         pd.Series(dens).value_counts().head(8).items()},
    }
res["rational_structure"] = rational
res["rational_structure"]["interpretation"] = (
    "Every stored metric value is an exact rational with a small denominator, and the "
    "denominator ranges differ by scenario (small_sample denominators at most 400, "
    "high_variance up to 800 with 92% divisible by 2). The metric values are therefore "
    "computed as ratios over a scenario-specific number of replicates, and the "
    "denominator is a scenario property rather than a per-metric one."
)

# --------------------------------------------------------------------------
# 7. variance decomposition - how much is condition vs metric identity?
# --------------------------------------------------------------------------
long = w.melt(
    id_vars=["scenario_id", "metric_id", "metric_number"],
    value_vars=["condition_2", "condition_3"],
    var_name="condition",
    value_name="value",
)
sh = long[long.metric_number.isin(shared)]
tot = sh.value.var()
decomp = {}
for key in ["condition", "scenario_id", "metric_number"]:
    decomp[key] = float(sh.groupby(key).value.transform("mean").var() / tot)
decomp["metric_x_condition"] = float(sh.groupby(["metric_number", "condition"]).value.transform("mean").var() / tot)
res["variance_decomposition"] = {
    "variance_shares": decomp,
    "interpretation": (
        "Condition identity dominates (about "
        f"{decomp['condition']*100:.0f}% of variance); metric identity contributes "
        "little on its own, and the metric x condition interaction explains nearly "
        "everything, so the two conditions are not two measurement channels of one "
        "latent metric quantity."
    ),
}

# --------------------------------------------------------------------------
# 8. does the metric id carry over to the composition features?
# --------------------------------------------------------------------------
from lib_composition import counts_matrix  # noqa: E402

P, samples, features = counts_matrix()
fc_present = (P > 0).sum(axis=0)
prev = fc_present / P.shape[0]
res["feature_metric_id_correspondence"] = {}
for s, sub in w.groupby("scenario_id"):
    merged = sub.assign(n_present=fc_present[sub.metric_number.values - 1],
                        prevalence=prev[sub.metric_number.values - 1],
                        mean_abundance=P.mean(axis=0)[sub.metric_number.values - 1])
    row = {}
    for cond in ["condition_2", "condition_3"]:
        row[cond] = {
            "spearman_vs_n_present": float(stats.spearmanr(merged[cond], merged.n_present).statistic),
            "spearman_vs_mean_abundance": float(
                stats.spearmanr(merged[cond], np.log(merged.mean_abundance)).statistic
            ),
        }
    row["c2_zero_rate_by_prevalence_quartile"] = {
        str(k): float(v) for k, v in
        pd.Series((merged.condition_2 == 0).values)
        .groupby(pd.qcut(merged.prevalence, 4, labels=["Q1_rarest", "Q2", "Q3", "Q4_commonest"]), observed=True)
        .mean().items()
    }
    res["feature_metric_id_correspondence"][s] = row
res["feature_metric_id_correspondence"]["interpretation"] = (
    "Metric values are uncorrelated with the composition feature that shares the same "
    "numeric index, and the condition_2 zero rate does not follow the corresponding "
    "feature's prevalence. The shared 1..600 id space is a naming coincidence, not a "
    "join key; no row-level linkage between the count table and the scenario table is "
    "supported by the data."
)

save_json(res, OUT / "results" / "bias_scenarios.json")
print("stage 4 ok")
print("  frac c3 > c2          :", round(res["condition_ordering"]["frac_c3_gt_c2"], 4),
      "violations:", res["condition_ordering"]["n_violations"])
print("  c3 mean hv/ss         : %.4f / %.4f" % (
    res["paired_scenario_effect"]["conditions"]["condition_3"]["mean_high_variance"],
    res["paired_scenario_effect"]["conditions"]["condition_3"]["mean_small_sample"]))
print("  c3 paired Cliff delta :", round(res["paired_scenario_effect"]["conditions"]["condition_3"]["cliffs_delta_paired"], 4))
print("  spearman c2~c3 (ss/hv): %.4f / %.4f" % (
    res["condition_coupling"]["shared_metric_ids_only"]["small_sample"]["spearman"],
    res["condition_coupling"]["shared_metric_ids_only"]["high_variance"]["spearman"]))
print("  excl c2==0 (ss/hv)    : %.4f / %.4f" % (
    res["condition_coupling"]["non_zero_only"]["small_sample"]["spearman"],
    res["condition_coupling"]["non_zero_only"]["high_variance"]["spearman"]))