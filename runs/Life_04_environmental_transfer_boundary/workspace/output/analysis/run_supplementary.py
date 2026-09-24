#!/usr/bin/env python3
"""Functional-form and confounding checks for the Life_04 assay relationship.

Motivated by the sensitivity run: the assay gradient is flat except for a
high-positivity low-assay region.  This script asks whether the data support a
threshold shape rather than a monotone dose-response, and whether the assay
association survives adjustment for the co-recorded day and symptom fields.

Run:  python3 output/analysis/run_supplementary.py
"""

from __future__ import annotations

import json
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
    SEED, fisher_exact_2x2, logistic_fit, perm_pvalue, spearman, sub_rng,
)

rng_cp = sub_rng("supp/changepoint")   # section 1
rng_adj = sub_rng("supp/adjusted")     # section 3
P: dict = {"schema": "life04-supplementary", "seed": SEED + 2}

unit = pd.read_csv(DATA / "unit_level_observations.tsv", sep="\t")
syms = pd.read_csv(DATA / "symptom_assay_surface_summary.tsv", sep="\t")
u = unit.rename(columns={"quantitative_assay_value": "assay"}).merge(
    syms[["unit_id", "symptom_present", "surface_positive_percent"]], on="unit_id")
y = u.any_environment_positive.to_numpy(dtype=float)
assay = u.assay.to_numpy()
n = len(y)

# --------------------------------------------------------------------------
# 1. changepoint grid: single-threshold models vs the no-threshold null
# --------------------------------------------------------------------------
def neg_ll(p, yy):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -float(np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


ll_null = neg_ll(np.full(n, y.mean()), y)
ll_linear = -logistic_fit(np.column_stack([np.ones(n), assay]), y)["loglik"]

changepoints = []
for c in np.unique(np.round(np.sort(assay), 2)):
    if c <= assay.min() or c >= assay.max():
        continue
    below = (assay < c).astype(float)
    fit = logistic_fit(np.column_stack([np.ones(n), below]), y)
    changepoints.append({"cut": float(c), "coef_below": float(fit["beta"][1]),
                         "loglik": fit["loglik"], "neg_ll": -fit["loglik"],
                         "rate_below": float(y[assay < c].mean()), "n_below": int((assay < c).sum()),
                         "rate_above": float(y[assay >= c].mean()), "n_above": int((assay >= c).sum())})
cp = pd.DataFrame(changepoints)
best = cp.loc[cp.neg_ll.idxmin()]
P["changepoint_grid"] = {
    "n_candidate_cuts": int(len(cp)),
    "null_neg_ll": ll_null,
    "linear_neg_ll": ll_linear,
    "best_cut": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in best.items()},
    "best_cut_neg_ll": float(best.neg_ll),
    "delta_negll_best_vs_null": float(ll_null - best.neg_ll),
    "delta_negll_best_vs_linear": float(ll_linear - best.neg_ll),
    "aic_null": 2 * 1 + 2 * ll_null,
    "aic_linear": 2 * 2 + 2 * ll_linear,
    "aic_best_cut": 2 * 2 + 2 * best.neg_ll,
    "all_cuts": changepoints,
}

# Permutation null for the best changepoint's improvement over the null model.
# KEY in the permuted world the predictor (below-the-cut) never changes, only the
# outcome does - so the fitted rate for a permuted outcome is exactly its group
# mean. This makes the whole search O(n_rep x n_cuts) with pure arithmetic.
cuts = np.array([c for c in np.unique(np.round(np.sort(assay), 2))
                 if assay.min() < c < assay.max()])
below_mat = np.column_stack([(assay < c).astype(float) for c in cuts])   # (n, n_cuts)
n_below = below_mat.sum(axis=0)                                          # (n_cuts,)
n_above = n - n_below

obs_imp = float(ll_null - best.neg_ll)
N_PERM_CP = 20000
perm_y = y[rng_cp.permutation(np.tile(np.arange(n), (N_PERM_CP, 1)), axis=1)]  # (n_rep, n)
sum_below = perm_y @ below_mat                                           # (n_rep, n_cuts)
sum_above = perm_y.sum(axis=1, keepdims=True) - sum_below
p_below = sum_below / n_below
p_above = sum_above / n_above
n_succ_b = sum_below
n_succ_a = sum_above
ll_b = (n_succ_b * np.log(np.clip(p_below, 1e-12, 1)) + (n_below - n_succ_b) * np.log(np.clip(1 - p_below, 1e-12, 1)))
ll_a = (n_succ_a * np.log(np.clip(p_above, 1e-12, 1)) + (n_above - n_succ_a) * np.log(np.clip(1 - p_above, 1e-12, 1)))
llb = np.max(ll_b + ll_a, axis=1)                                        # (n_rep,)
l0 = perm_y.sum(axis=1) * np.log(np.clip(perm_y.mean(axis=1), 1e-12, 1)) + \
     (n - perm_y.sum(axis=1)) * np.log(np.clip(1 - perm_y.mean(axis=1), 1e-12, 1))
null_imp = l0 - llb
P["changepoint_grid"]["best_cut_improvement_permutation"] = {
    "observed": obs_imp, "p_perm": perm_pvalue(obs_imp, null_imp, greater=True), "n_perm": int(len(null_imp)),
    "null_q95": float(np.quantile(null_imp, 0.95)),
    "note": "the null search re-optimises the cut on every permutation, so it accounts for the selection of the best cut"
}

# --------------------------------------------------------------------------
# 2. below/above-best-cut 2x2 in full
# --------------------------------------------------------------------------
cut = float(best.cut)
below = assay < cut
a = int(np.sum((y == 1) & below)); b = int(np.sum((y == 1) & ~below))
c_ = int(np.sum((y == 0) & below)); d = int(np.sum((y == 0) & ~below))
P["best_cut_2x2"] = {"cut": cut, "table_pos_by_below": [[a, b], [c_, d]], **fisher_exact_2x2(a, b, c_, d),
                     "rate_below": a / (a + c_) if (a + c_) else None,
                     "rate_above": b / (b + d) if (b + d) else None}

# sanity: the vectorised permutation arithmetic must reproduce the fitted best-cut log-likelihood
_j = int(np.argmin(np.abs(cuts - cut)))
_obs_sb = float(y @ below_mat[:, _j])
_nb = float(n_below[_j])
_pb, _pa = _obs_sb / _nb, (y.sum() - _obs_sb) / (n - _nb)
# same clipping as the vectorised path: a saturated cell contributes 0 x log(eps)
_pb_c, _pa_c = min(max(_pb, 1e-12), 1 - 1e-12), min(max(_pa, 1e-12), 1 - 1e-12)
_ll_vec = (_obs_sb * np.log(_pb_c) + (_nb - _obs_sb) * np.log(1 - _pb_c)
           + (y.sum() - _obs_sb) * np.log(_pa_c) + (n - _nb - (y.sum() - _obs_sb)) * np.log(1 - _pa_c))
assert abs(_ll_vec - best.loglik) < 1e-6, (_ll_vec, best.loglik)
P["changepoint_grid"]["saturated_cell_warning"] = {
    "rate_below_best_cut": float(_pb), "n_below_best_cut": int(_nb),
    "note": ("the best cut produces a saturated low-assay cell (all units positive), so the "
             "likelihood improvement is driven by perfect separation, not by a graded effect"),
}

# --------------------------------------------------------------------------
# 3. confounding structure among the recorded unit-level covariates
# --------------------------------------------------------------------------
cov = pd.DataFrame({
    "illness_day": u.illness_day, "stay_day": u.stay_day, "assay": u.assay,
    "symptom_present": u.symptom_present, "surface_percent": u.surface_positive_percent,
    "env_positive": u.any_environment_positive,
})
P["covariate_correlation_matrix_spearman"] = {
    a_: {b_: (None if a_ == b_ else float(spearman(cov[a_], cov[b_]))) for b_ in cov.columns}
    for a_ in cov.columns
}

# adjusted assay effect: assay + symptom + illness_day
X_adj = np.column_stack([np.ones(n), u.assay.to_numpy(), u.symptom_present.to_numpy(), u.illness_day.to_numpy()])
fit_adj = logistic_fit(X_adj, y)
P["adjusted_models"] = {
    "assay_only": {"coef": logistic_fit(np.column_stack([np.ones(n), assay]), y)["beta"]},
    "assay_plus_symptom_plus_day": {"coef": fit_adj["beta"], "loglik": fit_adj["loglik"]},
    "assay_plus_symptom": {"coef": logistic_fit(np.column_stack([np.ones(n), assay, u.symptom_present.to_numpy()]), y)["beta"]},
    "assay_plus_day": {"coef": logistic_fit(np.column_stack([np.ones(n), assay, u.illness_day.to_numpy()]), y)["beta"]},
}

# permutation null for the adjusted assay coefficient (rank of |coef|)
obs_adj = float(fit_adj["beta"][1])
null_adj = np.empty(20000)
for i in range(len(null_adj)):
    null_adj[i] = logistic_fit(X_adj, rng_adj.permutation(y), penalty=0.05)["beta"][1]
P["adjusted_models"]["assay_coef_after_adjustment_permutation"] = {
    "observed": obs_adj,
    "p_perm_lower_tail": perm_pvalue(obs_adj, null_adj, greater=False),
    "p_perm_two_sided": min(1.0, 2 * perm_pvalue(obs_adj, null_adj, greater=False)),
    "n_perm": int(len(null_adj)),
}

# is the low-assay cluster just the early-illness-day cluster?
P["low_assay_cluster_composition"] = {
    "cut": cut,
    "n_below": int(below.sum()),
    "n_below_positive": int(np.sum((y == 1) & below)),
    "median_illness_day_below": float(u.illness_day[below].median()),
    "median_illness_day_above": float(u.illness_day[~below].median()),
    "median_stay_day_below": float(u.stay_day[below].median()),
    "median_stay_day_above": float(u.stay_day[~below].median()),
    "symptom_rate_below": float(u.symptom_present[below].mean()),
    "symptom_rate_above": float(u.symptom_present[~below].mean()),
    "assay_at_or_below_22_92_units": sorted(u.loc[assay <= 22.92, "unit_id"].tolist()),
    "assay_at_or_below_22_92_positivity": float(y[assay <= 22.92].mean()),
    "assay_above_22_92_positivity": float(y[assay > 22.92].mean()),
}

(OUT / "supplementary.json").write_text(json.dumps(P, indent=2, default=str) + "\n", encoding="utf-8")
print("supplementary complete ->", OUT / "supplementary.json")
print(json.dumps({k: P[k] for k in ["best_cut_2x2", "low_assay_cluster_composition"]}, indent=1, default=str))
print("changepoint best:", json.dumps(P["changepoint_grid"]["best_cut"]))
print("AIC null/linear/cut:", P["changepoint_grid"]["aic_null"], P["changepoint_grid"]["aic_linear"], P["changepoint_grid"]["aic_best_cut"])