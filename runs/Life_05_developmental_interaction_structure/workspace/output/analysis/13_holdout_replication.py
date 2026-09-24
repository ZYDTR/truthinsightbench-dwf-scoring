#!/usr/bin/env python3
"""
Independent replication of the leave-one-family-out holdout test (and the
analogous leave-one-cell-pair-out test).

Claim under test: "the stage trend transports out of sample to interaction
families that were entirely excluded when estimating it."

Pipeline (implemented here from scratch; a single function recomputes the whole
pipeline from a given stage labelling so the permutation control uses exactly
the same code path as the observed fit):

    keep only records whose unit_id (interaction x cell_pair) spans >= 2 stages
    loo        = leave-one-out mean of the unit's OTHER records  (null prediction)
    resid      = strength - loo
    delta[s]   = mean of resid over observations at stage s whose group
                 (family or cell pair) is NOT the record's own group
    trend pred = loo + delta[s]

  * pooled MAE / RMSE for null vs trend and % improvement
  * permutation control: stage_index permuted WITHIN each unit (unit membership
    and the per-unit multiset of stages preserved), full pipeline recomputed,
    >= 100 permutations -> null distribution of the improvement + p-value
  * per-family improved vs worse counts, two-sided sign test, heterogeneity
    diagnostics (family-weighted vs record-weighted improvement, concentration)
  * identical leave-one-CELL-PAIR-out test
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr

BASE = "/data/user/conversations/conv-9877e7dd-aaaa-49c9-a430-e071a69303ee/project"
IN_TSV = os.path.join(BASE, "output", "results", "analysis_table.tsv")
OUT_JSON = os.path.join(BASE, "output", "results", "holdout_replication.json")

N_PERM = 500
SEED = 20260920
N_STAGES = 18

# ---------------------------------------------------------------- load + restrict
df = pd.read_csv(IN_TSV, sep="\t")
for col in ("stage_index", "interaction_id", "cell_pair_id", "unit_id"):
    assert col in df.columns, col

stages_per_unit = df.groupby("unit_id")["stage_index"].nunique()
keep_units = stages_per_unit.index[stages_per_unit >= 2]
d = (
    df[df["unit_id"].isin(set(keep_units))]
    .sort_values(["unit_id", "stage_index"], kind="stable")
    .reset_index(drop=True)
)
assert not d.duplicated(["unit_id", "stage_index"]).any(), "multiple rows per unit-stage"

y = d["interaction_strength_like_value"].to_numpy(dtype=float)
stage = d["stage_index"].to_numpy(dtype=np.int64)
unit_uniq, unit_codes = np.unique(d["unit_id"].to_numpy(), return_inverse=True)
fam_uniq, fam_codes = np.unique(d["interaction_id"].to_numpy(), return_inverse=True)
cp_uniq, cp_codes = np.unique(d["cell_pair_id"].to_numpy(), return_inverse=True)
unit_codes = unit_codes.astype(np.int64)
fam_codes = fam_codes.astype(np.int64)
cp_codes = cp_codes.astype(np.int64)

n_units = len(unit_uniq)
n_fams = len(fam_uniq)
n_cps = len(cp_uniq)
N = len(y)

u_sum = np.bincount(unit_codes, weights=y, minlength=n_units)
u_n = np.bincount(unit_codes, minlength=n_units).astype(float)
assert u_n.min() >= 2.0

# unit order used for the within-unit permutation (d is already grouped by unit)
unit_order = np.argsort(unit_codes, kind="stable")
codes_sorted = unit_codes[unit_order]
stage_sorted = stage[unit_order]


# ---------------------------------------------------------------- pipeline
def unit_loo_mean():
    """Null prediction: mean of the unit's OTHER records (stage-invariant)."""
    return (u_sum[unit_codes] - y) / (u_n[unit_codes] - 1.0)


def holdout_delta(stage_lab, group_codes, n_groups, resid, n_stages=N_STAGES):
    """delta[s, g] = mean resid at stage s over records whose group != g."""
    grid = stage_lab * n_groups + group_codes
    s_sum = np.bincount(stage_lab, weights=resid, minlength=n_stages)
    s_cnt = np.bincount(stage_lab, minlength=n_stages).astype(float)
    g_sum = np.bincount(grid, weights=resid, minlength=n_stages * n_groups).reshape(
        n_stages, n_groups
    )
    g_cnt = np.bincount(grid, minlength=n_stages * n_groups).reshape(
        n_stages, n_groups
    ).astype(float)
    num = s_sum[:, None] - g_sum
    den = s_cnt[:, None] - g_cnt
    safe = den > 0
    delta = np.zeros_like(num, dtype=float)
    delta[safe] = num[safe] / den[safe]
    return delta, int(safe.sum())


def run_pipeline(stage_lab, group_codes, n_groups):
    """Recompute the complete pipeline for a given stage labelling."""
    loo = unit_loo_mean()
    resid = y - loo
    delta, n_cells = holdout_delta(stage_lab, group_codes, n_groups, resid)
    pred = loo + delta[stage_lab, group_codes]
    return {"loo": loo, "resid": resid, "delta": delta, "pred": pred,
            "n_delta_cells": n_cells}


def scores(pred, y_true=None):
    yy = y if y_true is None else y_true
    e = yy - pred
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def permute_stage_within_unit(rng):
    """Shuffle stage labels inside each unit block (random-key lexsort)."""
    keys = rng.random(N)
    ordk = np.lexsort((keys, codes_sorted))
    sl = np.empty_like(stage)
    sl[unit_order] = stage_sorted[ordk]
    return sl


def check_multiset(sl):
    """Invariant: the shuffled labels preserve each unit's stage multiset."""
    for u in (0, 1, 2, n_units // 2, n_units - 1):
        m = unit_codes == u
        assert sorted(sl[m].tolist()) == sorted(stage[m].tolist()), u
    return True


def run_holdout(group_codes, group_uniq, label_attr, group_label):
    """Observed fit + permutation control + per-group diagnostics."""
    obs = run_pipeline(stage, group_codes, len(group_uniq))
    mae_n, rmse_n = scores(obs["loo"])
    mae_t, rmse_t = scores(obs["pred"])
    imp_abs = mae_n - mae_t
    imp_pct = 100.0 * imp_abs / mae_n if mae_n > 0 else 0.0

    obs_err_abs = np.abs(y - obs["pred"])
    null_err_abs = np.abs(y - obs["loo"])
    share_records_improved = float(np.mean(obs_err_abs < null_err_abs))
    share_records_worse = float(np.mean(obs_err_abs > null_err_abs))

    # permutation control: shuffle stage within unit, recompute full pipeline
    rng = np.random.default_rng(SEED)
    perm_imp, perm_mae_t, perm_rmse_t, perm_checked = [], [], [], 0
    for i in range(N_PERM):
        sl = permute_stage_within_unit(rng)
        if i < 5:
            check_multiset(sl)
            perm_checked += 1
        p = run_pipeline(sl, group_codes, len(group_uniq))
        mt, rt = scores(p["pred"])
        perm_mae_t.append(mt)
        perm_rmse_t.append(rt)
        perm_imp.append(mae_n - mt)
    perm_imp = np.asarray(perm_imp)
    perm_mae_t = np.asarray(perm_mae_t)
    perm_rmse_t = np.asarray(perm_rmse_t)

    n_ge = int(np.sum(perm_imp >= imp_abs))
    n_le = int(np.sum(perm_imp <= imp_abs))
    p_one = (1 + n_ge) / (1 + N_PERM)
    p_two = min(1.0, 2.0 * min(p_one, (1 + n_le) / (1 + N_PERM)))
    sd = float(perm_imp.std(ddof=1))
    z = float((imp_abs - perm_imp.mean()) / sd) if sd > 0 else None

    # per-group table
    rows = []
    for g in np.unique(group_codes):
        m = group_codes == g
        mn, rn = scores(obs["loo"][m], y[m])
        mt, rt = scores(obs["pred"][m], y[m])
        a, b = null_err_abs[m], obs_err_abs[m]
        pval = None
        if len(a) >= 6 and not np.allclose(a, b):
            try:
                pval = float(wilcoxon(a, b).pvalue)
            except Exception:
                pval = None
        rows.append(
            {
                "group": str(group_uniq[int(g)]),
                "n_records": int(m.sum()),
                "n_units": int(np.unique(unit_codes[m]).size),
                "mae_null": mn,
                "mae_trend": mt,
                "rmse_null": rn,
                "rmse_trend": rt,
                "mae_improvement": mn - mt,
                "improved": bool(mt < mn),
                "tied": bool(mt == mn),
                "share_records_improved": float(np.mean(b < a)),
                "wilcoxon_p": pval,
            }
        )
    tab = pd.DataFrame(rows)
    n_better = int((tab["mae_improvement"] > 0).sum())
    n_worse = int((tab["mae_improvement"] < 0).sum())
    n_tied = int((tab["mae_improvement"] == 0).sum())
    n_groups = int(len(tab))

    # two-sided sign test over non-tied groups only (ties carry no sign)
    sign_p2 = float(binomtest(n_better, n_better + n_worse, 0.5,
                              alternative="two-sided").pvalue)
    sign_p1 = float(binomtest(n_better, n_better + n_worse, 0.5,
                              alternative="greater").pvalue)

    # one-sample tests on the per-group improvement distribution (vs 0)
    fam_imp = tab["mae_improvement"].to_numpy()
    try:
        w_p = float(wilcoxon(fam_imp).pvalue)
    except Exception:
        w_p = None
    sign_records = {
        "n_records": int(tab["n_records"].sum()),
        "n_units": int(n_units),
        "n_groups": n_groups,
        "mae_null": mae_n,
        "rmse_null": rmse_n,
        "mae_trend": mae_t,
        "rmse_trend": rmse_t,
        "mae_improvement_abs": imp_abs,
        "mae_improvement_pct": imp_pct,
        "rmse_improvement_abs": rmse_n - rmse_t,
        "rmse_improvement_pct": 100.0 * (rmse_n - rmse_t) / rmse_n,
        "share_records_improved": share_records_improved,
        "share_records_worse": share_records_worse,
        "n_delta_cells_defined": obs["n_delta_cells"],
        "permutation_control": {
            "n_permutations": N_PERM,
            "seed": SEED,
            "within_unit_multiset_verified": bool(perm_checked == 5),
            "perm_mae_trend_mean": float(perm_mae_t.mean()),
            "perm_mae_trend_sd": sd,
            "perm_improvement_mean": float(perm_imp.mean()),
            "perm_improvement_sd": sd,
            "perm_improvement_min": float(perm_imp.min()),
            "perm_improvement_max": float(perm_imp.max()),
            "perm_improvement_q025": float(np.percentile(perm_imp, 2.5)),
            "perm_improvement_q50": float(np.percentile(perm_imp, 50)),
            "perm_improvement_q975": float(np.percentile(perm_imp, 97.5)),
            "observed_improvement": imp_abs,
            "n_perm_ge_observed": n_ge,
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "z_of_observed": z,
        },
        "per_group_summary": {
            "n_groups": n_groups,
            "n_improved": n_better,
            "n_worse": n_worse,
            "n_tied": n_tied,
            "share_improved": float(n_better / n_groups),
            "sign_test_p_two_sided": sign_p2,
            "sign_test_p_one_sided_improve": sign_p1,
            "sign_test_note": "two-sided binomial sign test on non-tied groups only",
            "improvement_unweighted_mean": float(fam_imp.mean()),
            "improvement_unweighted_median": float(np.median(fam_imp)),
            "improvement_q25": float(np.percentile(fam_imp, 25)),
            "improvement_q75": float(np.percentile(fam_imp, 75)),
            "improvement_min": float(fam_imp.min()),
            "improvement_max": float(fam_imp.max()),
            "improvement_sd": float(fam_imp.std(ddof=1)),
            "one_sample_wilcoxon_p_vs_zero": w_p,
            "records_weighted_mean_improvement": imp_abs,
            "n_groups_wilcoxon_p05_improved": int(
                np.sum(
                    (tab["wilcoxon_p"].notna())
                    & (tab["wilcoxon_p"] < 0.05)
                    & (tab["mae_improvement"] > 0)
                )
            ),
            "n_groups_wilcoxon_p05_worse": int(
                np.sum(
                    (tab["wilcoxon_p"].notna())
                    & (tab["wilcoxon_p"] < 0.05)
                    & (tab["mae_improvement"] < 0)
                )
            ),
            "n_groups_wilcoxon_testable": int(tab["wilcoxon_p"].notna().sum()),
        },
    }

    # concentration of the pooled improvement across groups
    contrib = tab["n_records"].to_numpy() * fam_imp  # sums to N * imp_abs
    order = np.argsort(-contrib)
    top5 = float(contrib[order[:5]].sum() / contrib.sum()) if contrib.sum() != 0 else None
    rho, rho_p = spearmanr(tab["n_records"], fam_imp)
    sign_records["per_group_summary"]["top5_groups_share_of_total_improvement"] = top5
    sign_records["per_group_summary"]["spearman_n_vs_improvement"] = {
        "rho": float(rho),
        "p": float(rho_p),
    }
    sign_records["per_group_summary"]["improvement_unweighted_mean_pct_of_null_mae"] = float(
        100.0 * fam_imp.mean() / mae_n
    )

    # diagnostics binned by group size (how much of the pooled gain is large groups)
    bins = [(2, 4), (5, 9), (10, 29), (30, 59), (60, 10**9)]
    size_bins = []
    for lo, hi in bins:
        m = (tab["n_records"] >= lo) & (tab["n_records"] <= hi)
        if not m.any():
            continue
        size_bins.append(
            {
                "n_records_range": [int(lo), int(hi if hi < 10**9 else tab["n_records"].max())],
                "n_groups": int(m.sum()),
                "n_groups_improved": int(tab.loc[m, "improved"].sum()),
                "total_records": int(tab.loc[m, "n_records"].sum()),
                "sum_weighted_improvement": float(
                    (tab.loc[m, "n_records"] * tab.loc[m, "mae_improvement"]).sum()
                ),
                "mean_unweighted_improvement": float(tab.loc[m, "mae_improvement"].mean()),
            }
        )
    sign_records["per_group_summary"]["size_bin_attribution"] = size_bins

    # robustness: drop all groups with < 10 records (only medium/large groups kept)
    for cut, tag in [(10, "drop_groups_under10"), (30, "drop_groups_under30")]:
        keep = tab["n_records"].to_numpy() >= cut
        if keep.sum() >= 2:
            names = set(tab.loc[keep, "group"])
            mask = np.isin(
                np.array([str(group_uniq[int(c)]) for c in group_codes], dtype=object),
                list(names),
            )
            rn, _ = scores(obs["loo"][mask], y[mask])
            rt, _ = scores(obs["pred"][mask], y[mask])
            sign_records["per_group_summary"][tag] = {
                "n_groups_kept": int(keep.sum()),
                "n_records_kept": int(mask.sum()),
                "mae_null": rn,
                "mae_trend": rt,
                "mae_improvement_abs": rn - rt,
                "mae_improvement_pct": 100.0 * (rn - rt) / rn if rn > 0 else None,
            }

    # per-stage breakdown (observed fit)
    stage_rows = []
    for s in range(N_STAGES):
        m = stage == s
        if not m.any():
            continue
        sn, _ = scores(obs["loo"][m], y[m])
        st_, _ = scores(obs["pred"][m], y[m])
        stage_rows.append(
            {
                "stage_index": s,
                "n_records": int(m.sum()),
                "mae_null": sn,
                "mae_trend": st_,
                "improvement": sn - st_,
                "share_records_improved": float(
                    np.mean(obs_err_abs[m] < null_err_abs[m])
                ),
                "mean_applied_offset": float(
                    obs["delta"][stage[m], group_codes[m]].mean()
                ),
            }
        )

    # robustness: pooled improvement after dropping top contributors
    keep_mask = np.ones(N, dtype=bool)
    tab_by_name = tab.set_index("group")
    top5_names = tab_by_name.index[order[:5]]
    keep_mask &= ~np.isin(
        np.array([str(group_uniq[int(c)]) for c in group_codes], dtype=object),
        top5_names,
    )
    if keep_mask.sum() > 0:
        rob_n, _ = scores(obs["loo"][keep_mask], y[keep_mask])
        rob_t, _ = scores(obs["pred"][keep_mask], y[keep_mask])
        rob = {
            "n_records_excluding_top5_groups": int(keep_mask.sum()),
            "mae_null": rob_n,
            "mae_trend": rob_t,
            "mae_improvement_abs": rob_n - rob_t,
            "mae_improvement_pct": 100.0 * (rob_n - rob_t) / rob_n,
            "groups_excluded": [str(x) for x in top5_names],
        }
    else:
        rob = None

    return {
        "summary": sign_records,
        "per_group": tab,
        "per_stage": stage_rows,
        "robustness_drop_top5": rob,
        "obs": obs,
    }


# ---------------------------------------------------------------- run both holdouts
fam_res = run_holdout(fam_codes, fam_uniq, "interaction_id", "family")
cp_res = run_holdout(cp_codes, cp_uniq, "cell_pair_id", "cell_pair")

# context: global within-unit stage profile (no holdout)
delta_all = np.bincount(stage, weights=fam_res["obs"]["resid"], minlength=N_STAGES) / np.bincount(
    stage, minlength=N_STAGES
)

# context: in-sample (own-group-included) reference fit - the optimistic bound
# that the leave-one-group-out test is meant to be compared against
delta_ins = np.bincount(stage, weights=fam_res["obs"]["resid"], minlength=N_STAGES) / np.bincount(
    stage, minlength=N_STAGES
)
pred_insample = fam_res["obs"]["loo"] + delta_ins[stage]
mae_ins, rmse_ins = scores(pred_insample)
insample_ref = {
    "note": ("same functional form but delta[s] NOT held out: uses the record's "
             "own family too, so it is the in-sample optimistic reference"),
    "mae": mae_ins,
    "rmse": rmse_ins,
    "mae_improvement_over_null_abs": fam_res["summary"]["mae_null"] - mae_ins,
    "mae_improvement_over_null_pct":
        100.0 * (fam_res["summary"]["mae_null"] - mae_ins) / fam_res["summary"]["mae_null"],
    "holdout_retained_share_of_insample_improvement": (
        fam_res["summary"]["mae_improvement_abs"]
        / (fam_res["summary"]["mae_null"] - mae_ins)
        if (fam_res["summary"]["mae_null"] - mae_ins) != 0 else None
    ),
}


# ---------------------------------------------------------------- serialise
def tab_to_records(tab):
    out = []
    for _, r in tab.iterrows():
        out.append(
            {
                k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                    (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else
                     (str(v) if k == "group" else
                      (int(v) if isinstance(v, (int, np.integer)) else
                       (float(v) if isinstance(v, (float, np.floating)) else v)))))
                for k, v in r.items()
            }
        )
    return out


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


result = {
    "meta": {
        "input_file": IN_TSV,
        "n_rows_in_analysis_table": int(len(df)),
        "rows_used": int(N),
        "n_units_used": int(n_units),
        "n_families_used": int(n_fams),
        "n_cell_pairs_used": int(n_cps),
        "min_stages_required": 2,
        "n_stages": N_STAGES,
        "n_permutations": N_PERM,
        "seed": SEED,
        "unit_leave_one_out_denominator_min": float((u_n - 1).min()),
        "pipeline": (
            "null = unit leave-one-out mean; trend = null + delta[stage] where "
            "delta is the mean (strength - unit LOO mean) at that stage over "
            "records whose group (family / cell pair) differs from the record's "
            "own group; permutation control shuffles stage_index within each "
            "unit and recomputes the entire pipeline"
        ),
    },
    "leave_one_family_out": fam_res["summary"],
    "leave_one_family_out_per_stage": fam_res["per_stage"],
    "leave_one_family_out_per_family": tab_to_records(fam_res["per_group"]),
    "leave_one_family_out_robustness_drop_top5": fam_res["robustness_drop_top5"],
    "in_sample_reference_fit": insample_ref,
    "leave_one_cell_pair_out": cp_res["summary"],
    "leave_one_cell_pair_out_per_stage": cp_res["per_stage"],
    "leave_one_cell_pair_out_per_cell_pair": tab_to_records(cp_res["per_group"]),
    "context_delta_all_records_per_stage": {
        str(s): float(delta_all[s]) for s in range(N_STAGES)
    },
}
result = json_safe(result)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(result, fh, indent=2, allow_nan=False)


# ---------------------------------------------------------------- print report
def report(title, res):
    s = res["summary"]
    pc = s["permutation_control"]
    pg = s["per_group_summary"]
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"records={s['n_records']}  units={s['n_units']}  groups={pg['n_groups']}")
    print(f"NULL  (unit LOO mean):        MAE={s['mae_null']:.5f}  RMSE={s['rmse_null']:.5f}")
    print(f"TREND (+ held-out delta[s]):  MAE={s['mae_trend']:.5f}  RMSE={s['rmse_trend']:.5f}")
    print(f"improvement: MAE {s['mae_improvement_abs']:.5f} ({s['mae_improvement_pct']:.3f}%)  "
          f"RMSE {s['rmse_improvement_abs']:.5f} ({s['rmse_improvement_pct']:.3f}%)")
    print(f"records improved={s['share_records_improved']:.4f} worse={s['share_records_worse']:.4f}")
    print(f"permutation ({pc['n_permutations']} within-unit shuffles, multiset verified="
          f"{pc['within_unit_multiset_verified']}): mean={pc['perm_improvement_mean']:.5f} "
          f"sd={pc['perm_improvement_sd']:.5f} range=[{pc['perm_improvement_min']:.5f},"
          f"{pc['perm_improvement_max']:.5f}]  p1={pc['p_one_sided']:.4f} "
          f"p2={pc['p_two_sided']:.4f}  (#perm>=obs {pc['n_perm_ge_observed']}/{pc['n_permutations']})")
    print(f"per-group: improved={pg['n_improved']} worse={pg['n_worse']} tied={pg['n_tied']} "
          f"share={pg['share_improved']:.4f} sign-test p2={pg['sign_test_p_two_sided']:.4f} "
          f"p1={pg['sign_test_p_one_sided_improve']:.4f}")
    print(f"  improvement across groups: unweighted mean={pg['improvement_unweighted_mean']:.5f} "
          f"median={pg['improvement_unweighted_median']:.5f} "
          f"IQR=[{pg['improvement_q25']:.5f},{pg['improvement_q75']:.5f}] "
          f"min={pg['improvement_min']:.5f} max={pg['improvement_max']:.5f} "
          f"sd={pg['improvement_sd']:.5f}")
    print(f"  one-sample Wilcoxon vs 0: p={pg['one_sample_wilcoxon_p_vs_zero']}   "
          f"top5 share of total improvement={pg['top5_groups_share_of_total_improvement']}")
    print(f"  per-group Wilcoxon p<.05 (n testable={pg['n_groups_wilcoxon_testable']}): "
          f"{pg['n_groups_wilcoxon_p05_improved']} improved, {pg['n_groups_wilcoxon_p05_worse']} worse")
    rob = res["robustness_drop_top5"]
    if rob:
        print(f"  drop top-5 contributing groups: n={rob['n_records_excluding_top5_groups']} "
              f"MAE {rob['mae_null']:.5f} -> {rob['mae_trend']:.5f} "
              f"({rob['mae_improvement_pct']:.3f}%)")


report("LEAVE-ONE-FAMILY-OUT HOLDOUT", fam_res)
report("LEAVE-ONE-CELL-PAIR-OUT HOLDOUT", cp_res)
print("=" * 78)
print(f"wrote {OUT_JSON}")