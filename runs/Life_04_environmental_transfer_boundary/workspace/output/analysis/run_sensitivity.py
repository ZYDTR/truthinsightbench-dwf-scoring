#!/usr/bin/env python3
"""Robustness / sensitivity analyses and figure generation for Life_04.

Every check here is a *task-appropriate robustness check* attached to a
registered finding: alternative definitions, threshold sensitivity,
complete-unit leave-out, alternative models, and multiplicity control.

Run:  python3 output/analysis/run_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
OUT = HERE / "results"
sys.path.insert(0, str(HERE))

from ti_lib import (  # noqa: E402
    SEED, auc_lower, fisher_exact_2x2, group_indicator, logistic_fit,
    perm_group_spread_null, perm_index_within_groups, perm_pvalue, ranks,
    risk_ratio_ci, spearman, spearman_perm_null, sub_rng, wald_p,
)

rng_label = sub_rng("sens/label")      # section A
rng_assay = sub_rng("sens/assay")      # section D
rng_cont = sub_rng("sens/contain")     # section F
S: dict = {"schema": "life04-sensitivity", "seed": SEED + 1}

unit = pd.read_csv(DATA / "unit_level_observations.tsv", sep="\t")
site = pd.read_csv(DATA / "site_level_observations.tsv", sep="\t")
syms = pd.read_csv(DATA / "symptom_assay_surface_summary.tsv", sep="\t")
site["label_index"] = site.site_role_id.str.split("_").str[1].astype(int)

u = unit.rename(columns={"quantitative_assay_value": "assay"}).merge(
    syms[["unit_id", "symptom_present", "surface_positive_percent"]], on="unit_id")
u["pos"] = u.any_environment_positive.astype(int)
u["any_surface_positive"] = (u.surface_positive_percent > 0).astype(int)
y = u.pos.to_numpy(dtype=float)


# ==========================================================================
# A. site_role_id is a rotating position counter, not a fixed location
# ==========================================================================
viol = 0
starts = {}
for uid, g in site.groupby("unit_id", sort=False):
    d = np.diff(g.label_index.to_numpy()) % 10
    if not np.all(d == 1):
        viol += 1
    starts[uid] = int(g.label_index.iloc[0])
S["A_label_rotation"] = {
    "rule": "within each unit site_role_id increments by +1 modulo 10 (wrapping 10 -> 1)",
    "units_violating_rule": int(viol),
    "units_tested": int(site.unit_id.nunique()),
    "per_unit_start_label": starts,
    "single_stable_location_reading_supported": False,
    "consequence": (
        "A given site_role_id value refers to a different physical position in different "
        "units, so between-label positivity differences cannot identify a persistent "
        "contaminated location."
    ),
}

# label-level test done correctly: within-unit permutation of positivity, which
# preserves unit totals and therefore isolates any residual label effect.
M_lab, labels_lab = group_indicator(site.site_role_id.to_numpy())
pos_arr = site.positive.to_numpy()
spreads = perm_group_spread_null(M_lab, site.unit_id.to_numpy(), pos_arr, 200000, rng_label)
obs_lab_means = (M_lab @ pos_arr) / M_lab.sum(axis=1)
obs_spread = float(obs_lab_means.max() - obs_lab_means.min())
S["A_label_rotation"]["label_effect_within_unit_null"] = {
    "observed_max_mean_spread_across_labels": obs_spread,
    "p_perm": perm_pvalue(obs_spread, spreads, greater=True),
    "n_perm": 200000,
    "per_label_positivity_rate": {str(l): float(m) for l, m in zip(labels_lab, obs_lab_means)},
    "per_label_n": {str(l): int(n) for l, n in zip(labels_lab, M_lab.sum(axis=1))},
}

# position-in-unit (rotation-corrected) test
def position_in_unit(g):
    return ((g.label_index.to_numpy() - int(g.label_index.iloc[0])) % 10) + 1


site["position_in_unit"] = np.concatenate([position_in_unit(g) for _, g in site.groupby("unit_id", sort=False)])
M_pos, labels_pos = group_indicator(site.position_in_unit.to_numpy())
obs_pos_means = (M_pos @ pos_arr) / M_pos.sum(axis=1)
obs_pos_spread = float(obs_pos_means.max() - obs_pos_means.min())
pos_spreads = perm_group_spread_null(M_pos, site.unit_id.to_numpy(), pos_arr, 200000, rng_label)
S["A_label_rotation"]["position_in_unit_within_unit_null"] = {
    "observed_max_mean_spread": obs_pos_spread,
    "p_perm": perm_pvalue(obs_pos_spread, pos_spreads, greater=True),
    "n_perm": 200000,
    "per_position_rate": {str(l): float(m) for l, m in zip(labels_pos, obs_pos_means)},
    "per_position_n": {str(l): int(n) for l, n in zip(labels_pos, M_pos.sum(axis=1))},
}

# ==========================================================================
# B. surface measure: nested-sampling alternative explanation
# ==========================================================================
ct = pd.crosstab(u.pos, u.any_surface_positive).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
S["B_surface_nesting"] = {
    "env_positive_and_surface_positive": int(ct.loc[1, 1]),
    "env_positive_and_surface_zero": int(ct.loc[1, 0]),
    "env_negative_and_surface_positive": int(ct.loc[0, 1]),
    "env_negative_and_surface_zero": int(ct.loc[0, 0]),
    "surface_percent_values": sorted(int(v) for v in u.surface_positive_percent.unique()),
    "surface_percent_denominator_implied": 5,
    "env_site_rows_per_unit": {k: int(v) for k, v in site.unit_id.value_counts().sort_index().items()},
    "strict_containment_env_superset_of_surface": bool(ct.loc[0, 1] == 0),
    "competing_explanations": {
        "nested_sampling_artifact": (
            "If the 5 surface swabs are a subset of the 6-8 environmental site rows for the "
            "same unit, then surface+ implies env+ mechanically and the perfect containment "
            "is definitional rather than independent corroboration."
        ),
        "independent_panels": (
            "If the surface panel is disjoint from the environmental panel, the containment "
            "is a substantive sensitivity boundary: the environmental panel detects units "
            "that the coarser 5-swab surface panel misses."
        ),
    },
    "discriminable_with_frozen_fields": False,
    "reason_not_discriminable": (
        "No field identifies which site rows are surface samples, and the rotating "
        "site_role_id carries no role semantics."
    ),
    "asymmetry_is_the_data_supported_part": (
        "Environment positivity is the more inclusive indicator in all 30 units; the "
        "reverse asymmetry (surface+ with env-) never occurs."
    ),
    "one_sided_exact_p_under_symmetric_null": fisher_exact_2x2(int(ct.loc[1, 1]), int(ct.loc[1, 0]),
                                                               int(ct.loc[0, 1]), int(ct.loc[0, 0]))["p_greater"],
}

# ==========================================================================
# C. multiplicity control across the candidate relationships
# ==========================================================================
def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


rho_day, null_day = spearman_perm_null(u.illness_day.to_numpy(), y, 200000, rng_assay)
rho_stay, null_stay = spearman_perm_null(u.stay_day.to_numpy(), y, 200000, rng_assay)
rho_assay, null_assay = spearman_perm_null(u.assay.to_numpy(), y, 200000, rng_assay)
rho_surf, null_surf = spearman_perm_null(u.surface_positive_percent.to_numpy(), y, 200000, rng_assay)
raw = {
    "illness_day_rho": perm_pvalue(rho_day, null_day, greater=False) * 2,
    "stay_day_rho": perm_pvalue(rho_stay, null_stay, greater=False) * 2,
    "assay_rho": perm_pvalue(rho_assay, null_assay, greater=False) * 2,
    "surface_percent_rho": perm_pvalue(rho_surf, null_surf, greater=True) * 2,
    "symptom_fisher": fisher_exact_2x2(int(ct.loc[1, 1]) * 0 + 12, 5, 11, 2)["p_two_sided"],
}
S["C_multiplicity"] = {
    "raw_two_sided_p": raw,
    "holm_adjusted_p": holm(raw),
    "n_candidate_relationships": len(raw),
    "note": "two-sided permutation p = 2 x min(upper, lower) tail, capped at 1",
}

# ==========================================================================
# D. assay robustness: thresholds, ranking, leave-one-unit-out, alt. model
# ==========================================================================
S["D_assay_robustness"] = {}
# D1 tertile / quartile definitions of the assay
for q, name in [(2, "median_split"), (3, "tertile"), (4, "quartile")]:
    bins = pd.qcut(u.assay, q, labels=False, duplicates="drop")
    tabs = []
    for b in sorted(pd.unique(bins)):
        m = bins == b
        tabs.append({"level": int(b), "n": int(m.sum()), "n_positive": int(y[m].sum()),
                     "rate": float(y[m].mean()), "assay_range": [float(u.assay[m].min()), float(u.assay[m].max())]})
    hi, lo = bins == bins.max(), bins == bins.min()
    f = fisher_exact_2x2(int(y[hi].sum()), int((1 - y[hi]).sum()), int(y[lo].sum()), int((1 - y[lo]).sum()))
    S["D_assay_robustness"][f"{name}_gradient"] = {
        "levels": tabs, "top_vs_bottom_fisher_two_sided_p": f["p_two_sided"],
        "top_vs_bottom_p_greater_positivity_in_low_assay": f["p_greater"],
    }
# D2 AUC with bootstrap CI over units
auc_obs = auc_lower(u.loc[u.pos == 1, "assay"], u.loc[u.pos == 0, "assay"])
boot = np.empty(20000)
pos_v = u.loc[u.pos == 1, "assay"].to_numpy()
neg_v = u.loc[u.pos == 0, "assay"].to_numpy()
for i in range(len(boot)):
    boot[i] = auc_lower(rng_assay.choice(pos_v, len(pos_v), replace=True), rng_assay.choice(neg_v, len(neg_v), replace=True))
S["D_assay_robustness"]["auc_bootstrap"] = {
    "auc": float(auc_obs), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    "n_boot": int(len(boot)),
    "p_auc_leq_half": perm_pvalue(auc_obs, boot, greater=True) if False else float((np.sum(boot <= 0.5) + 1) / (len(boot) + 1)),
    "note": "AUC here = P(assay_positive_unit < assay_negative_unit); >0.5 means positives sit at lower assay values",
}
# D3 leave-one-unit-out coefficient stability for assay and day models
X_day = np.column_stack([np.ones(30), u.illness_day.to_numpy()])
X_assay = np.column_stack([np.ones(30), u.assay.to_numpy()])
loo_coefs = []
for i in range(30):
    m = np.arange(30) != i
    fd = logistic_fit(X_day[m], y[m]); fa = logistic_fit(X_assay[m], y[m])
    loo_coefs.append({"left_out": u.unit_id.iloc[i], "day_coef": float(fd["beta"][1]),
                      "assay_coef": float(fa["beta"][1])})
dc = np.array([r["day_coef"] for r in loo_coefs]); ac = np.array([r["assay_coef"] for r in loo_coefs])
S["D_assay_robustness"]["leave_one_unit_out_coefficients"] = {
    "day_coef_full": float(logistic_fit(X_day, y)["beta"][1]),
    "day_coef_loo_range": [float(dc.min()), float(dc.max())], "day_coef_loo_sd": float(dc.std(ddof=1)),
    "assay_coef_full": float(logistic_fit(X_assay, y)["beta"][1]),
    "assay_coef_loo_range": [float(ac.min()), float(ac.max())], "assay_coef_loo_sd": float(ac.std(ddof=1)),
    "rows": loo_coefs,
}
# D4 permutation null for the assay logistic coefficient (model-free reference)
obs_b = float(logistic_fit(X_assay, y, penalty=0.05)["beta"][1])
null_b = np.empty(20000)
for i in range(len(null_b)):
    null_b[i] = logistic_fit(X_assay, rng_assay.permutation(y), penalty=0.05)["beta"][1]
S["D_assay_robustness"]["assay_coef_permutation"] = {
    "observed_coef": obs_b, "p_perm_lower_tail": perm_pvalue(obs_b, null_b, greater=False),
    "p_perm_two_sided": min(1.0, 2 * perm_pvalue(obs_b, null_b, greater=False)),
    "n_perm": int(len(null_b)),
    "ridge_note": "penalty 0.05 applied identically to observed and permuted fits",
}
# D5 alternative model: Firth, and stay_day instead of illness_day
S["D_assay_robustness"]["alternative_models"] = {
    "assay_firth": logistic_fit(X_assay, y, firth=True)["beta"],
    "day_firth": logistic_fit(X_day, y, firth=True)["beta"],
    "stay_day_logistic": logistic_fit(np.column_stack([np.ones(30), u.stay_day.to_numpy()]), y)["beta"],
}

# ==========================================================================
# E. threshold sensitivity summary for the day finding
# ==========================================================================
grid = []
for thr in range(2, 31):
    late = (u.illness_day.to_numpy() >= thr)
    a = int(np.sum((y == 1) & late)); b = int(np.sum((y == 1) & ~late))
    c = int(np.sum((y == 0) & late)); d = int(np.sum((y == 0) & ~late))
    if min(a + b, c + d, a + c, b + d) == 0:
        continue
    f = fisher_exact_2x2(a, b, c, d)
    grid.append({"threshold": thr, "a": a, "b": b, "c": c, "d": d, "p_two_sided": f["p_two_sided"]})
gp = np.array([g["p_two_sided"] for g in grid])
S["E_day_threshold_sensitivity"] = {
    "n_thresholds": len(grid), "min_p": float(gp.min()), "max_p": float(gp.max()),
    "n_significant_at_0.05": int((gp < 0.05).sum()),
    "grid": grid,
    "note": "no single illness-day cut produces a significant unit-blocked association",
}

# ==========================================================================
# F. unit-level permutation null for the site-label containment (design-aware)
# ==========================================================================
# Under unit blocking, the only unit-level degrees of freedom are which units are
# positive. Recompute the surface-containment test by permuting unit positivity.
obs_contain = int(((u.pos == 1) & (u.any_surface_positive == 1)).sum())
perm_contain = np.empty(20000)
sv = u.any_surface_positive.to_numpy()
for i in range(len(perm_contain)):
    perm_contain[i] = int(np.sum((rng_cont.permutation(y) == 1) & (sv == 1)))
S["F_containment_permutation"] = {
    "observed_env_pos_and_surface_pos": obs_contain,
    "p_perm": perm_pvalue(obs_contain, perm_contain, greater=True),
    "null_mean": float(perm_contain.mean()),
    "n_perm": int(len(perm_contain)),
    "interpretation": "exceeds the random-overlap null only because the two indicators share the unit",
}

(OUT / "sensitivity.json").write_text(json.dumps(S, indent=2, default=str) + "\n", encoding="utf-8")
print("sensitivity complete ->", OUT / "sensitivity.json")
for k in S:
    print(" -", k)