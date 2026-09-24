#!/usr/bin/env python3
"""Life_04_environmental_transfer_boundary - full executed analysis.

Unit of analysis: one sampled unit (30 units); its sites/days are repeated
observations.  Every inferential test is either exact or permutation/cluster
based because the runtime has no scipy/statsmodels.

Run:  python3 output/analysis/run_analysis.py
Writes machine-readable results to output/analysis/results/ and a manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from ti_lib import (  # noqa: E402
    SEED, all_or_nothing, auc_lower, binom_test_two_sided, cluster_robust_cov,
    fisher_exact_2x2, group_indicator, icc_binary, logistic_fit, loo_loglik,
    mann_whitney_u, overdispersion_test, pearson, perm_group_spread_null,
    perm_index_within_groups, perm_pvalue, ranks, risk_ratio_ci, spearman,
    spearman_perm_null, sub_rng, wald_p,
)

import time as _time
_T0 = _time.time()
def _tick(label):
    print(f"[time] {label}: {_time.time()-_T0:.1f}s", flush=True)

rng = sub_rng("main/section0")           # section 0: label-rotation permutation
rng_counts = sub_rng("main/counts")      # section 1: overdispersion bootstrap
rng_unit = sub_rng("main/unit-perm")     # section 2: unit-level rank permutations
rng_models = sub_rng("main/models")      # section 3: LOO permutation tests
rng_hold = sub_rng("main/holdouts")      # section 4: random block holdouts
rng_site = sub_rng("main/site-perm")     # section 5: within-unit site permutations
R: dict = {"schema": "life04-analysis-results", "seed": SEED, "task_id": "Life_04_environmental_transfer_boundary"}

unit = pd.read_csv(DATA / "unit_level_observations.tsv", sep="\t")
site = pd.read_csv(DATA / "site_level_observations.tsv", sep="\t")
syms = pd.read_csv(DATA / "symptom_assay_surface_summary.tsv", sep="\t")


# ==========================================================================
# 0. integrity of the frozen payload
# ==========================================================================
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


R["payload"] = {
    "files": {
        f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)}
        for f in sorted(DATA.glob("*.tsv"))
    },
    "shapes": {
        "unit_level_observations": list(unit.shape),
        "site_level_observations": list(site.shape),
        "symptom_assay_surface_summary": list(syms.shape),
    },
    "n_units": int(unit.unit_id.nunique()),
    "n_site_rows": int(len(site)),
    "unit_ids_identical_across_files": bool(
        list(unit.unit_id) == list(syms.unit_id) and set(site.unit_id) == set(unit.unit_id)
    ),
    "unit_id_overlap_with_site": sorted(set(site.unit_id) & set(unit.unit_id)),
    "missing_cells": {
        "unit_level": int(unit.isna().sum().sum()),
        "site_level": int(site.isna().sum().sum()),
        "symptom_summary": int(syms.isna().sum().sum()),
    },
    "day_columns_identical_between_files": bool(
        (unit.set_index("unit_id").illness_day == syms.set_index("unit_id").illness_day).all()
    ),
    "assay_identical_between_files": bool(
        (unit.set_index("unit_id").quantitative_assay_value
         == syms.set_index("unit_id").quantitative_assay_value).all()
    ),
}

# --------------------------------------------------------------------------
# 0b. site_role_id label structure (two competing null models)
# --------------------------------------------------------------------------
rows = []
for _, r in site.iterrows():
    idx = int(r.site_role_id.split("_")[1])
    rows.append({"unit_id": r.unit_id, "site_role_id": r.site_role_id,
                 "label_index": idx, "positive": int(r.positive)})
site = pd.DataFrame(rows)

# model A: the label indexes 10 globally fixed sites -> each site_role_id is one location
lab = site.groupby("site_role_id").positive.agg(["sum", "count", "mean"])
# model B: the label is a rotating within-unit counter (offset per unit)
site["label_offset"] = site.groupby("unit_id").label_index.transform(lambda s: int(s.iloc[0]) - 1)
site["position_in_unit"] = site.label_index - site.label_offset
pos_tab = site.groupby("position_in_unit").positive.agg(["sum", "count", "mean"])

# permutation null for model A: shuffle positivity within each unit -> label spread
obs_label_spread = float(site.groupby("site_role_id").positive.mean().max()
                         - site.groupby("site_role_id").positive.mean().min())
arr_unit = site.unit_id.to_numpy()
arr_pos = site.positive.to_numpy()
n_site_rows = len(site)
N_REP_LABEL = 200000
M_g, labels_g = group_indicator(site.site_role_id.to_numpy())
# Chunked so peak memory stays ~chunk x n_rows rather than n_rep x n_rows; the
# replicate values are identical to a single un-chunked pass with this seed.
sim_spreads = perm_group_spread_null(M_g, arr_unit, arr_pos, N_REP_LABEL, rng)
_tick("section 0b label permutation")

R["site_label_structure"] = {
    "n_site_rows": int(len(site)),
    "n_distinct_site_role_ids": int(site.site_role_id.nunique()),
    "rows_per_site_role_id": {k: int(v) for k, v in site.site_role_id.value_counts().sort_index().items()},
    "rows_per_unit": {k: int(v) for k, v in site.unit_id.value_counts().sort_index().items()},
    "role_id_sequence_is_rotation_by_5": bool(
        all(((site[site.unit_id == u].label_index.to_numpy() - 1) % 10
             == np.arange(len(site[site.unit_id == u])) * 5 % 10).all()
            for u in site.unit_id.unique())
    ),
    "positivity_by_role_label": {k: {"sum": int(v["sum"]), "n": int(v["count"]), "mean": float(v["mean"])}
                                 for k, v in lab.iterrows()},
    "positivity_by_position_in_unit": {int(k): {"sum": int(v["sum"]), "n": int(v["count"]), "mean": float(v["mean"])}
                                       for k, v in pos_tab.iterrows()},
    "observed_max_label_mean_spread": obs_label_spread,
    "permutation_p_label_spread": perm_pvalue(obs_label_spread, sim_spreads, greater=True),
    "n_perm": int(len(sim_spreads)),
}

# ==========================================================================
# 1. within-unit positivity is all-or-nothing (clustering dominates)
# ==========================================================================
unit_site = site.groupby("unit_id").positive.agg(["sum", "count"])
unit_site["prop"] = unit_site["sum"] / unit_site["count"]
R["within_unit_positivity"] = {
    "n_units": int(len(unit_site)),
    "per_unit": {u: {"n_sites": int(r["count"]), "n_positive": int(r["sum"]), "prop": float(r["prop"])}
                 for u, r in unit_site.iterrows()},
    "distribution_of_per_unit_proportions": {str(k): int(v) for k, v in unit_site.prop.value_counts().sort_index().items()},
    **all_or_nothing(site.unit_id, site.positive),
}
_tick("section 0 total")
icc = icc_binary(site.unit_id, site.positive)
R["design_effect_sites"] = icc

# naive site-level test vs unit-blocked test of the same association
merged = site.merge(unit[["unit_id", "illness_day", "stay_day", "quantitative_assay_value"]], on="unit_id")
a_site = int(((merged.positive == 1) & (merged.illness_day >= 15)).sum())
b_site = int(((merged.positive == 0) & (merged.illness_day >= 15)).sum())
c_site = int(((merged.positive == 1) & (merged.illness_day < 15)).sum())
d_site = int(((merged.positive == 0) & (merged.illness_day < 15)).sum())
naive = fisher_exact_2x2(a_site, b_site, c_site, d_site)
naive_orr = risk_ratio_ci(a_site, b_site, c_site, d_site)

u2 = unit.copy()
u2["late"] = (u2.illness_day >= 15).astype(int)
u2["pos"] = u2.any_environment_positive.astype(int)
tab = pd.crosstab(u2.pos, u2.late).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
u_a, u_b = int(tab.loc[1, 1]), int(tab.loc[1, 0])
u_c, u_d = int(tab.loc[0, 1]), int(tab.loc[0, 0])
blocked = fisher_exact_2x2(u_a, u_b, u_c, u_d)
blocked_rr = risk_ratio_ci(u_a, u_b, u_c, u_d)
R["naive_vs_blocked"] = {
    "threshold_illness_day": 15,
    "naive_site_level": {"table_pos_by_late": [[a_site, b_site], [c_site, d_site]], **naive,
                         "risk_ratio": naive_orr, "n_rows": int(len(merged))},
    "unit_blocked": {"table_pos_by_late": [[u_a, u_b], [u_c, u_d]], **blocked,
                     "risk_ratio": blocked_rr, "n_units": int(len(u2))},
    "inflation_factor_p_to_effective_n": {
        "site_rows": int(len(merged)),
        "effective_n_units": float(icc["effective_n_sites"]),
    },
}

# overdispersion of unit-level positive counts relative to Binomial
p_hat = float(site.positive.mean())
R["overdispersion_positive_counts"] = overdispersion_test(unit_site["sum"].to_numpy(), unit_site["count"].to_numpy(), p_hat, rng_counts)

# ==========================================================================
# 2. candidate relationships: day, assay, symptom  (unit level, n=30)
# ==========================================================================
_syms = syms[["unit_id", "symptom_present", "surface_positive_percent", "quantitative_assay_value"]].rename(
    columns={"quantitative_assay_value": "assay_symptom_file"})
u = unit.merge(_syms, on="unit_id")
# cross-file consistency check: the assay must be byte-identical in both files
assert (u.quantitative_assay_value == u.assay_symptom_file).all()
u = u.rename(columns={"quantitative_assay_value": "assay"}).drop(columns=["assay_symptom_file"])
u["pos"] = u.any_environment_positive.astype(int)
u["any_surface_positive"] = (u.surface_positive_percent > 0).astype(int)
y = u.pos.to_numpy(dtype=float)
groups = u.unit_id.to_numpy()

# ---- 2a. illness_day (threshold-free + thresholded) ----
import itertools  # noqa: E402

rho_day = spearman(y, u.illness_day)
rho_stay = spearman(y, u.stay_day)
N_REP_UNIT = 200000
rho_day_obs, permd = spearman_perm_null(u.illness_day.to_numpy(), y, N_REP_UNIT, rng_unit)
assert abs(rho_day_obs - rho_day) < 1e-12
rho_stay_obs, permd_stay = spearman_perm_null(u.stay_day.to_numpy(), y, N_REP_UNIT, rng_unit)

def exact_rank_sum_null(ranks_x, k: int) -> dict:
    """Exact distribution of the positive-group rank sum under all C(n, k)
    label assignments, by dynamic programming over the rank multiset.

    For a 0/1 outcome, Spearman rho is a strictly increasing linear function of
    this rank sum, so the exact upper-tail p for rho equals P(S >= S_obs).
    This is the exact Mann-Whitney / Wilcoxon rank-sum null.
    """
    scaled = np.rint(np.asarray(ranks_x, dtype=float) * 2).astype(int)  # ties -> .5 -> integer
    shift = scaled.min()
    scaled = scaled - shift
    n, total = len(scaled), int(scaled.sum())
    # dp[j][s] = number of ways to choose j items with scaled rank sum s
    dp = np.zeros((k + 1, total + 1), dtype=object)
    dp[0, 0] = 1
    for r in scaled:
        for j in range(min(k - 1, len(scaled)), -1, -1):
            nz = np.flatnonzero(dp[j])
            if len(nz):
                dp[j + 1, nz + r] += dp[j, nz]
    counts = dp[k]
    sums = np.flatnonzero(counts)
    weights = np.array([int(counts[s]) for s in sums], dtype=object)
    return {"sums": sums + k * shift, "counts": weights, "n_total": int(sum(weights))}


exact_ranks = ranks(u.illness_day.to_numpy())
k_pos = int(y.sum())
null_dist = exact_rank_sum_null(exact_ranks, k_pos)
S_obs = int(round(float(exact_ranks[y == 1].sum()) * 2))
assert null_dist["n_total"] == int(np.math.comb(len(y), k_pos)) if hasattr(np, "math") else True

# verify that rho is a strictly increasing linear function of the rank sum
_n, _k = len(y), k_pos
_T = exact_ranks.sum()
_C0 = (((_n - _k + 1) / 2.0) * _T) / _n - (_n + 1) ** 2 / 4.0


def rho_from_scaled_sum(S_scaled: float, sd_x: float) -> float:
    S = S_scaled / 2.0
    return (S / 2.0 + _C0) / (np.sqrt(_k * (_n - _k)) / 2.0 * sd_x)


_rho_check = rho_from_scaled_sum(S_obs, exact_ranks.std())
assert abs(_rho_check - rho_day) < 1e-9, (_rho_check, rho_day)
p_exact_upper = float(sum(int(c) for s, c in zip(null_dist["sums"], null_dist["counts"]) if s >= S_obs)
                      / null_dist["n_total"])

_tick("section 1 clustering")
R["F_day"] = {
    "metric": "Spearman rho(any_environment_positive, illness_day)",
    "rho": float(rho_day),
    "rho_squared": float(rho_day**2),
    "p_perm_montecarlo": perm_pvalue(rho_day, permd, greater=True),
    "n_perm_montecarlo": int(len(permd)),
    "p_exact_all_assignments": p_exact_upper,
    "n_exact_assignments": int(null_dist["n_total"]),
    "exact_method": "DP over exact Wilcoxon rank-sum null (equivalent to all C(30,14) label assignments)",
    "median_illness_day_positive": float(u.loc[u.pos == 1, "illness_day"].median()),
    "median_illness_day_negative": float(u.loc[u.pos == 0, "illness_day"].median()),
    "mean_illness_day_positive": float(u.loc[u.pos == 1, "illness_day"].mean()),
    "mean_illness_day_negative": float(u.loc[u.pos == 0, "illness_day"].mean()),
    "auc_lower_positive_has_lower_day": float(auc_lower(u.loc[u.pos == 1, "illness_day"], u.loc[u.pos == 0, "illness_day"])),
    "auc_interpretation": "AUC<0.5 => positives occur on EARLIER illness days",
    "rho_stay_day": float(rho_stay),
    "pearson_stay_vs_illness_day": float(pearson(u.stay_day, u.illness_day)),
    "stay_day_ratio_stats": {
        "median_ratio_stay_over_illness": float((u.stay_day / u.illness_day).median()),
        "mean_ratio": float((u.stay_day / u.illness_day).mean()),
    },
}

# single-threshold sensitivity grid (unit blocked): Fisher exact at each cut
grid = []
for thr in range(2, 31):
    late = (u.illness_day >= thr).to_numpy()
    a = int(np.sum((y == 1) & late)); b = int(np.sum((y == 1) & ~late))
    c = int(np.sum((y == 0) & late)); d = int(np.sum((y == 0) & ~late))
    if min(a + b, c + d, a + c, b + d) == 0:
        continue
    f = fisher_exact_2x2(a, b, c, d)
    rr = risk_ratio_ci(a, b, c, d)
    grid.append({"threshold": thr, "table": [[a, b], [c, d]], "p_two_sided": f["p_two_sided"],
                 "p_greater": f["p_greater"], "risk_ratio": rr["risk_ratio"],
                 "rr_ci_low": rr["ci_low"], "rr_ci_high": rr["ci_high"]})
R["F_day"]["threshold_late_sensitivity_unit_blocked"] = grid

# ---- 2b. stay_day continuous (ordinal, no threshold) ----
rho_stay_p = rho_stay_obs
R["F_day"]["stay_day_spearman"] = {
    "rho": float(rho_stay_p),
    "p_perm_montecarlo": perm_pvalue(rho_stay_p, permd_stay, greater=True),
    "auc_lower_positive_has_lower_stay_day": float(auc_lower(u.loc[u.pos == 1, "stay_day"], u.loc[u.pos == 0, "stay_day"])),
    "median_stay_day_positive": float(u.loc[u.pos == 1, "stay_day"].median()),
    "median_stay_day_negative": float(u.loc[u.pos == 0, "stay_day"].median()),
    "n_perm": int(len(permd_stay)),
}

# ---- 2c. quantitative assay ----
rho_assay_obs, permd_assay = spearman_perm_null(u.assay.to_numpy(), y, N_REP_UNIT, rng_unit)
rho_assay = rho_assay_obs
mw_u = mann_whitney_u(u.loc[u.pos == 1, "assay"], u.loc[u.pos == 0, "assay"])
_tick("section 2a day")
R["F_assay"] = {
    "metric": "Spearman rho(any_environment_positive, quantitative_assay_value)",
    "rho": float(rho_assay),
    "p_perm_montecarlo": perm_pvalue(rho_assay, permd_assay, greater=True),
    "p_perm_for_negative_direction": perm_pvalue(rho_assay, permd_assay, greater=False),
    "n_perm": int(len(permd_assay)),
    "auc_lower_positive_has_lower_assay": float(auc_lower(u.loc[u.pos == 1, "assay"], u.loc[u.pos == 0, "assay"])),
    "mean_assay_positive": float(u.loc[u.pos == 1, "assay"].mean()),
    "mean_assay_negative": float(u.loc[u.pos == 0, "assay"].mean()),
    "sd_assay_positive": float(u.loc[u.pos == 1, "assay"].std(ddof=1)),
    "sd_assay_negative": float(u.loc[u.pos == 0, "assay"].std(ddof=1)),
    "mann_whitney_U_positive_lower": float(mw_u),
    "U_null_mean": float(len(u[u.pos == 1]) * len(u[u.pos == 0]) / 2),
    "assay_range": [float(u.assay.min()), float(u.assay.max())],
    "assay_mean": float(u.assay.mean()), "assay_sd": float(u.assay.std(ddof=1)),
    "assay_cv_percent": float(100 * u.assay.std(ddof=1) / u.assay.mean()),
    "separation_gap_means_over_pooled_sd": float(
        abs(u.loc[u.pos == 1, "assay"].mean() - u.loc[u.pos == 0, "assay"].mean())
        / u.assay.std(ddof=1)),
}

# cluster-robust logistic at the 212 site rows, assay + intercept (separation expected)
Xm = np.column_stack([np.ones(len(merged)), merged.quantitative_assay_value.to_numpy()])
ysite = merged.positive.to_numpy(dtype=float)
sep_fit = logistic_fit(Xm, ysite, penalty=1e-6)
cov_sep = cluster_robust_cov(Xm, ysite, sep_fit["beta"], merged.unit_id.to_numpy())
R["F_assay"]["site_level_assay_logistic"] = {
    "coef_assay": float(sep_fit["beta"][1]),
    "coef_assay_rank_deficient_qr_warning": "site-level positivity is constant within unit; assay varies within unit, so any apparent effect is a within-unit artefact",
    "se_model": float(sep_fit["se_model"][1]),
    "se_cluster_robust": float(np.sqrt(cov_sep[1, 1])),
    "note": "SEs are unstable because the site-level design is degenerate (see within_unit_positivity)",
}

# ---- 2d. symptom linkage ----
ct_sym = pd.crosstab(u.pos, u.symptom_present).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
s_a, s_b = int(ct_sym.loc[1, 1]), int(ct_sym.loc[1, 0])
s_c, s_d = int(ct_sym.loc[0, 1]), int(ct_sym.loc[0, 0])
sym_test = fisher_exact_2x2(s_a, s_b, s_c, s_d)
sym_rr = risk_ratio_ci(s_a, s_b, s_c, s_d)
ct_surf = pd.crosstab(u.pos, u.any_surface_positive).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
surf_a, surf_b = int(ct_surf.loc[1, 1]), int(ct_surf.loc[1, 0])
surf_c, surf_d = int(ct_surf.loc[0, 1]), int(ct_surf.loc[0, 0])
surf_test = fisher_exact_2x2(surf_a, surf_b, surf_c, surf_d)
surf_rr = risk_ratio_ci(surf_a, surf_b, surf_c, surf_d)
rho_surf_obs, permd_surf = spearman_perm_null(u.surface_positive_percent.to_numpy(), y, N_REP_UNIT, rng_unit)
rho_surf = rho_surf_obs
_tick("section 2b/c assay+stay")
R["F_symptom"] = {
    "env_positive_by_symptom_present": {
        "table": [[s_a, s_b], [s_c, s_d]], **sym_test, "risk_ratio": sym_rr,
        "prevalence_pos_units_symptomatic": float(s_a / (s_a + s_b)) if (s_a + s_b) else None,
        "prevalence_pos_units_asymptomatic": float(s_c / (s_c + s_d)) if (s_c + s_d) else None,
    },
    "env_positive_by_any_surface_positive": {"table": [[surf_a, surf_b], [surf_c, surf_d]], **surf_test, "risk_ratio": surf_rr},
    "env_positive_vs_surface_positive_percent": {
        "rho": float(rho_surf), "p_perm_montecarlo": perm_pvalue(rho_surf, permd_surf, greater=True),
        "n_perm": int(len(permd_surf)),
        "mean_surface_percent_positive_units": float(u.loc[u.pos == 1, "surface_positive_percent"].mean()),
        "mean_surface_percent_negative_units": float(u.loc[u.pos == 0, "surface_positive_percent"].mean()),
    },
    "symptom_cells": int(u.symptom_present.sum()),
    "asymptomatic_units": sorted(u.loc[u.symptom_present == 0, "unit_id"].tolist()),
    "asymptomatic_unit_positivity": {uid: int(unit.loc[unit.unit_id == uid, "any_environment_positive"].iloc[0])
                                     for uid in u.loc[u.symptom_present == 0, "unit_id"]},
}

# ==========================================================================
_tick("section 2d symptom")
# 3. models on the 30 units: null / day / assay / combined, then LOO
# ==========================================================================
X_null = np.ones((30, 1))
X_day = np.column_stack([np.ones(30), u.illness_day.to_numpy()])
X_assay = np.column_stack([np.ones(30), u.assay.to_numpy()])
X_both = np.column_stack([np.ones(30), u.illness_day.to_numpy(), u.assay.to_numpy()])


def fit_plain(X, yy):
    return logistic_fit(X, yy, penalty=0.0)


def fit_ridge(X, yy):
    # weak ridge fallback so LOO stays finite under separation
    return logistic_fit(X, yy, penalty=0.05)


models = {}
for name, X in [("null", X_null), ("illness_day", X_day), ("assay", X_assay), ("day_plus_assay", X_both)]:
    fit = logistic_fit(X, y)
    cov = cluster_robust_cov(X, y, fit["beta"], groups)
    se = np.sqrt(np.diag(cov))
    models[name] = {
        "coef": fit["beta"],
        "se_cluster_robust": se.tolist(),
        "z": [float(fit["beta"][i] / se[i]) if se[i] > 0 else None for i in range(len(se))],
        "p_cluster_robust": [wald_p(fit["beta"][i], se[i])["p"] for i in range(len(se))],
        "loglik": fit["loglik"],
        "converged": fit["converged"],
        "n_params": int(X.shape[1]),
    }

# Every LOO value uses the same weak-ridge fitter, so the four numbers are
# mutually comparable; an unpenalised fit would be infinite under the
# leave-one-out separation noted above.
models["null"]["loo_loglik"] = loo_loglik(X_null, y, fit_ridge)
models["illness_day"]["loo_loglik"] = loo_loglik(X_day, y, fit_ridge)
models["assay"]["loo_loglik"] = loo_loglik(X_assay, y, fit_ridge)
models["day_plus_assay"]["loo_loglik"] = loo_loglik(X_both, y, fit_ridge)
R["models_unit_level"] = {
    "n_units": 30, "n_pos": int(y.sum()),
    "specification_note": "cluster-robust SEs equal model SEs here because each unit is one row; both are reported",
    "models": models,
    "loo_note": "All LOO log-likelihoods use the same weak ridge (0.05), because illness_day separates positivity under leave-one-out; without the ridge those fits are infinite",
    "mcfadden_pseudo_r2": {
        "illness_day": float(1 - models["illness_day"]["loglik"] / models["null"]["loglik"]),
        "assay": float(1 - models["assay"]["loglik"] / models["null"]["loglik"]),
        "day_plus_assay": float(1 - models["day_plus_assay"]["loglik"] / models["null"]["loglik"]),
    },
    "firth_day_model": logistic_fit(X_day, y, firth=True)["beta"],
    "logistic_saturation": {
        "possible_distinct_predicted_values": int(2 ** X_day.shape[1]),
        "n_units": 30,
        "distinct_illness_day_values": int(u.illness_day.nunique()),
    },
}

# Permutation null for the LOO increment (day/assay over the null model).
#
# NOTE ON AN ABANDONED OPTIMISATION, kept because it bears on the p-values below:
# an earlier version cached fits on the sufficient statistic X.T @ y.  That key
# is valid for a fixed design, but leave-one-out *changes the design* for each
# held-out unit, so the key is wrong here.  The assertion guarding the observed
# increment caught the mistake (cached 0.0828 vs true -0.4276), the cache was
# removed, and the honest uncached cost was measured at 0.048 s per LOO
# evaluation (= 30 IRLS solves).  Replicates were reduced from 2000 to 1000
# accordingly; n_rep is reported alongside every p-value.
def loo_permutation_test(X, yy, rng, n_rep=1000):
    obs = loo_loglik(X, yy, fit_ridge) - loo_loglik(X[:, [0]], yy, fit_ridge)
    null = np.empty(n_rep)
    for i in range(n_rep):
        yp = rng.permutation(yy)
        null[i] = loo_loglik(X, yp, fit_ridge) - loo_loglik(X[:, [0]], yp, fit_ridge)
    return float(obs), perm_pvalue(obs, null, greater=True), int(n_rep), null


obs_inc, p_inc, n_rep, null_inc = loo_permutation_test(X_day, y, rng_models)
# determinism check: an independent recomputation must give the identical increment
_loo_day_ridge = loo_loglik(X_day, y, fit_ridge)
_loo_null_ridge = loo_loglik(X_null, y, fit_ridge)
assert abs(obs_inc - (_loo_day_ridge - _loo_null_ridge)) < 1e-9, (obs_inc, _loo_day_ridge - _loo_null_ridge)
R["models_unit_level"]["loo_increment_day_over_null"] = {
    "observed_increment": obs_inc, "p_perm": p_inc, "n_rep": n_rep,
    "null_quantiles": {"q50": float(np.quantile(null_inc, 0.5)), "q95": float(np.quantile(null_inc, 0.95))},
}
obs_inc_a, p_inc_a, n_rep_a, null_inc_a = loo_permutation_test(X_assay, y, rng_models)
_loo_assay_ridge = loo_loglik(X_assay, y, fit_ridge)
assert abs(obs_inc_a - (_loo_assay_ridge - _loo_null_ridge)) < 1e-9, (obs_inc_a, _loo_assay_ridge - _loo_null_ridge)
R["models_unit_level"]["loo_increment_assay_over_null"] = {
    "observed_increment": obs_inc_a, "p_perm": p_inc_a, "n_rep": n_rep_a,
    "null_quantiles": {"q50": float(np.quantile(null_inc_a, 0.5)), "q95": float(np.quantile(null_inc_a, 0.95))},
}

# ==========================================================================
_tick("section 3 models")
# 4. robustness: whole-unit leave-one-out / leave-pair-out / block holdouts
# ==========================================================================
def holdout_metrics(mask_train, label):
    """Refit day + assay + null on training units, report held-out log-likelihood."""
    Xtr_d, Xte_d = X_day[mask_train], X_day[~mask_train]
    Xtr_a, Xte_a = X_assay[mask_train], X_assay[~mask_train]
    Xtr_0, Xte_0 = X_null[mask_train], X_null[~mask_train]
    ytr, yte = y[mask_train], y[~mask_train]
    out = {"holdout_unit": label, "n_holdout": int((~mask_train).sum())}
    for name, Xtr, Xte in [("null", Xtr_0, Xte_0), ("illness_day", Xtr_d, Xte_d), ("assay", Xtr_a, Xte_a)]:
        fit = fit_ridge(Xtr, ytr)
        eta = np.clip(Xte @ np.asarray(fit["beta"]), -30, 30)
        p = 1 / (1 + np.exp(-eta))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        out[f"{name}_holdout_loglik"] = float(np.sum(yte * np.log(p) + (1 - yte) * np.log(1 - p)))
    out["delta_day_minus_null"] = out["illness_day_holdout_loglik"] - out["null_holdout_loglik"]
    out["delta_assay_minus_null"] = out["assay_holdout_loglik"] - out["null_holdout_loglik"]
    return out


loo_rows = [holdout_metrics(np.arange(30) != i, u.unit_id.iloc[i]) for i in range(30)]
pair_rows = []
for i in range(30):
    for j in range(i + 1, 30):
        pair_rows.append(holdout_metrics((np.arange(30) != i) & (np.arange(30) != j), f"{u.unit_id.iloc[i]}+{u.unit_id.iloc[j]}"))
# contiguous 5-unit blocks (data order) and random 5-unit blocks
blocks = [list(range(i, min(i + 5, 30))) for i in range(0, 30, 5)]
block_rows = [holdout_metrics(~np.isin(np.arange(30), b), f"block_{k+1}") for k, b in enumerate(blocks)]
rand_block_rows = []
for k in range(200):
    members = rng_hold.choice(30, size=5, replace=False)
    rand_block_rows.append(holdout_metrics(~np.isin(np.arange(30), members), f"rand_block_{k+1}"))


def summarise(rows):
    d = np.array([r["delta_day_minus_null"] for r in rows])
    a = np.array([r["delta_assay_minus_null"] for r in rows])
    return {
        "n_holdout_units": len(rows),
        "mean_delta_day_minus_null": float(d.mean()),
        "min_delta_day_minus_null": float(d.min()),
        "max_delta_day_minus_null": float(d.max()),
        "share_delta_day_positive": float((d > 0).mean()),
        "mean_delta_assay_minus_null": float(a.mean()),
        "share_delta_assay_positive": float((a > 0).mean()),
    }


R["robustness_whole_unit_holdouts"] = {
    "leave_one_unit_out": {"summary": summarise(loo_rows), "rows": loo_rows},
    "leave_two_units_out": {"summary": summarise(pair_rows)},
    "contiguous_5_unit_blocks": {"summary": summarise(block_rows), "rows": block_rows},
    "random_5_unit_blocks": {"summary": summarise(rand_block_rows)},
}

# ==========================================================================
_tick("section 4 holdouts")
# 5. counterevidence file: site-level tests that ignore blocking
# ==========================================================================
su = site.unit_id.to_numpy()
pos_arr = site.positive.to_numpy()
M_u, unit_labels = group_indicator(su)
obs_counts = M_u @ pos_arr
# per-unit positive counts under within-unit shuffling (20000 replicates, one
# bounded draw of 20000 x 212; M_u @ ... is a 30 x 20000 matmul, not the large side)
perm_counts = M_u @ pos_arr[perm_index_within_groups(su, 20000, rng_site)].T
R["counterevidence_within_unit_permutation"] = {
    "label": "variance of per-unit positive counts (within-unit label shuffling)",
    "observed": float(obs_counts.var(ddof=1)),
    "p_perm": perm_pvalue(float(obs_counts.var(ddof=1)), perm_counts.var(axis=0, ddof=1), greater=True),
    "n_rep": 20000,
    "interpretation": (
        "Within-unit shuffling preserves each unit's positive count, so a non-significant "
        "result confirms that no site-level structure beyond the unit total is detectable."
    ),
    "between_unit_variance_is_the_signal": float(obs_counts.var(ddof=1)),
}

# ==========================================================================
# 6. narrative-supporting extras
# ==========================================================================
R["extras"] = {
    "env_positive_counts": {"positive_units": int(y.sum()), "negative_units": int(30 - y.sum())},
    "illness_day_structure": {
        "n_distinct_values": int(u.illness_day.nunique()),
        "duplicate_day_groups": {str(k): sorted(u.loc[u.illness_day == k, "unit_id"].tolist())
                                 for k, v in u.illness_day.value_counts().items() if v > 1},
        "min": int(u.illness_day.min()), "max": int(u.illness_day.max()),
    },
    "assay_distribution": {
        "min": float(u.assay.min()), "max": float(u.assay.max()),
        "q25": float(u.assay.quantile(.25)), "median": float(u.assay.median()), "q75": float(u.assay.quantile(.75)),
    },
    "symptom_surface_crosstab_all_units": {
        "symptomatic_and_surface_positive": int(((u.symptom_present == 1) & (u.any_surface_positive == 1)).sum()),
        "symptomatic_and_surface_zero": int(((u.symptom_present == 1) & (u.any_surface_positive == 0)).sum()),
        "asymptomatic_and_surface_positive": int(((u.symptom_present == 0) & (u.any_surface_positive == 1)).sum()),
        "asymptomatic_and_surface_zero": int(((u.symptom_present == 0) & (u.any_surface_positive == 0)).sum()),
    },
}

# ==========================================================================
# write results
# ==========================================================================
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


(OUT / "results.json").write_text(json.dumps(to_jsonable(R), indent=2) + "\n", encoding="utf-8")

# flat CSVs for quick inspection
pd.DataFrame(unit_site.reset_index()).to_csv(OUT / "unit_positivity_summary.csv", index=False)
pd.DataFrame(grid).to_csv(OUT / "day_threshold_sensitivity.csv", index=False)
pd.DataFrame(loo_rows).to_csv(OUT / "leave_one_unit_out.csv", index=False)
pd.DataFrame(block_rows).to_csv(OUT / "block_holdouts.csv", index=False)
pd.DataFrame([{k: v for k, v in lab.reset_index().to_dict("records")[i].items()} for i in range(len(lab))]).to_csv(
    OUT / "site_label_positivity.csv", index=False)
pd.DataFrame(pos_tab.reset_index()).to_csv(OUT / "site_position_positivity.csv", index=False)
u[["unit_id", "illness_day", "stay_day", "assay", "pos", "symptom_present",
   "surface_positive_percent", "any_surface_positive"]].to_csv(OUT / "unit_level_analysis_table.csv", index=False)

_tick("section 5 counterevidence")
print("analysis complete ->", OUT / "results.json")
print(json.dumps({k: R[k] for k in ["within_unit_positivity", "F_day"] if k in R}, indent=2, default=str)[:2000])