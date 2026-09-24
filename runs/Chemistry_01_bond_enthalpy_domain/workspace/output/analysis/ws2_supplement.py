"""WS2 supplement: robustness of the quantum-vs-learned verdict and fixes.

The headline paired test (quantum vs learned) gives a split answer: the mean
absolute-error difference is significant while the median and Wilcoxon tests
are not.  This script tests whether that split is driven by distribution tails,
by the training/held-out split, or by range/bond-class composition, and it
repairs one mis-stated boolean in the training-split block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project")
sys.path.insert(0, str(PROJECT / "output" / "analysis"))
import common  # noqa: E402

from scipy import stats  # noqa: E402

JSON = common.RESULTS / "ws2_channel_accuracy.json"
df = common.build_analysis_table(verbose=False)
exp = df[common.EXP_COL].to_numpy(dtype=float)
n = len(df)
CH = ("quantum", "learned", "reference")


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def paired_block(a: str, b: str, mask=None, n_boot: int = 20000, seed: int = 20260919) -> dict:
    frame = df if mask is None else df[mask]
    ae_a = np.abs(frame[f"{a}_err"].to_numpy(dtype=float))
    ae_b = np.abs(frame[f"{b}_err"].to_numpy(dtype=float))
    d = ae_a - ae_b
    m, lo, hi = common.paired_bootstrap_ci(d, n_boot=n_boot, seed=seed)
    t, p = common.paired_t_test(d)
    w, pw = common.wilcoxon(d)
    rng = np.random.default_rng(seed)
    md = np.median(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    npos, nneg = int((d > 0).sum()), int((d < 0).sum())
    return {
        "n_pairs": int(len(d)),
        "mae_first": f(common.mae(ae_a)),
        "mae_second": f(common.mae(ae_b)),
        "mean_abs_error_difference": f(m),
        "mean_difference_ci95": [f(lo), f(hi)],
        "mean_difference_significant": bool(lo > 0 or hi < 0),
        "median_abs_error_difference": f(float(np.median(d))),
        "median_difference_ci95": [f(np.quantile(md, 0.025)), f(np.quantile(md, 0.975))],
        "paired_t_p": f(p),
        "wilcoxon_p": f(pw),
        "cliffs_delta": f(common.cliffs_delta(ae_a, ae_b)),
        "n_first_worse": npos,
        "n_first_better": nneg,
    }


sup: dict = {}

# ------------------------------------------------------------------ fix field
sup["training_split_corrections"] = {
    "held_out_rows_all_in_db_row_false": bool(
        (~df.loc[~df.train_flag, "in_db_row"]).all() == False  # noqa: E712
    ),
    "held_out_n_with_db_row": int(df.loc[~df.train_flag, "in_db_row"].sum()),
    "train_rows_all_in_db_row_true": bool(df.loc[df.train_flag, "in_db_row"].all()),
    "note": "331 train rows are exactly the 331 rows with an exact DB row; the other 212 have none.",
    "experiment_min_train": f(df.loc[df.train_flag, common.EXP_COL].min()),
    "experiment_max_train": f(df.loc[df.train_flag, common.EXP_COL].max()),
    "experiment_min_held_out": f(df.loc[~df.train_flag, common.EXP_COL].min()),
    "experiment_max_held_out": f(df.loc[~df.train_flag, common.EXP_COL].max()),
    "mean_experiment_train": f(df.loc[df.train_flag, common.EXP_COL].mean()),
    "mean_experiment_held_out": f(df.loc[~df.train_flag, common.EXP_COL].mean()),
    "welch_p_experiment_train_vs_held_out": f(
        float(stats.ttest_ind(
            df.loc[df.train_flag, common.EXP_COL],
            df.loc[~df.train_flag, common.EXP_COL],
            equal_var=False,
        ).pvalue)
    ),
    "bond_class_composition_train": df.loc[df.train_flag, "bond_class"].value_counts().to_dict(),
    "bond_class_composition_held_out": df.loc[~df.train_flag, "bond_class"].value_counts().to_dict(),
}

# ------------------------------------------------- quantum vs learned by subset
sup["quantum_vs_learned_by_train_flag"] = {
    "train_subset": paired_block("quantum", "learned", df.train_flag.to_numpy()),
    "held_out_subset": paired_block("quantum", "learned", (~df.train_flag).to_numpy()),
}

# ------------------------------------------- is the mean difference tail-driven?
sup["quantum_vs_learned_tail_sensitivity"] = {}
for cap in (None, 15, 10, 7.5, 5, 3):
    ae_q = np.abs(df["quantum_err"].to_numpy(dtype=float))
    ae_l = np.abs(df["learned_err"].to_numpy(dtype=float))
    if cap is not None:
        ae_q, ae_l = np.minimum(ae_q, cap), np.minimum(ae_l, cap)
    d = ae_q - ae_l
    m, lo, hi = common.paired_bootstrap_ci(d, n_boot=20000)
    t, p = common.paired_t_test(d)
    key = "uncapped" if cap is None else f"abs_error_capped_at_{cap}"
    sup["quantum_vs_learned_tail_sensitivity"][key] = {
        "mean_difference": f(m),
        "ci95": [f(lo), f(hi)],
        "significant": bool(lo > 0 or hi < 0),
        "median_difference": f(float(np.median(d))),
        "paired_t_p": f(p),
        "mae_quantum": f(common.mae(ae_q)),
        "mae_learned": f(common.mae(ae_l)),
    }
# how much of the raw mean difference comes from the worst few observations
d_raw = np.abs(df["quantum_err"].to_numpy(dtype=float)) - np.abs(df["learned_err"].to_numpy(dtype=float))
order = np.argsort(-np.abs(d_raw))  # observations with the biggest |mean difference| contribution
sup["quantum_vs_learned_tail_sensitivity"]["contribution_of_worst_k"] = {
    f"top_{k}_observations_by_abs_difference": {
        "signed_sum_over_total_signed_sum": f(float(d_raw[order[:k]].sum() / d_raw.sum())),
        "absolute_sum_share": f(float(np.abs(d_raw[order[:k]]).sum() / np.abs(d_raw).sum())),
        "n_positive_favouring_learned": int((d_raw[order[:k]] > 0).sum()),
        "n_negative_favouring_quantum": int((d_raw[order[:k]] < 0).sum()),
        "n_pairs_in_k": int(k),
    }
    for k in (10, 25, 50)
}

# ------------------------------------- by experimental-value quartile and class
sup["quantum_vs_learned_by_experiment_quartile"] = {}
qc = pd.qcut(exp, 4, labels=["Q1_weakest", "Q2", "Q3", "Q4_strongest"])
for lab in qc.categories:
    m = np.asarray(qc == lab)
    sup["quantum_vs_learned_by_experiment_quartile"][str(lab)] = paired_block("quantum", "learned", m)
sup["quantum_vs_learned_by_bond_class"] = {
    str(k): paired_block("quantum", "learned", (df.bond_class == k).to_numpy())
    for k in df.bond_class.dropna().unique()
}

# ---------------------------------------------------- reference independence
sup["reference_independence"] = {}
ae = pd.DataFrame({c: np.abs(df[f"{c}_err"].to_numpy(dtype=float)) for c in CH})
# partial correlation of |err_ref| with |err_learned| controlling for |err_quantum|
def partial_corr(x, y, z):
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return float(np.corrcoef(rx, ry)[0, 1])


sup["reference_independence"]["partial_corr_ref_learned_given_quantum"] = f(
    partial_corr(ae["reference"].to_numpy(), ae["learned"].to_numpy(), ae["quantum"].to_numpy())
)
sup["reference_independence"]["partial_corr_ref_quantum_given_learned"] = f(
    partial_corr(ae["reference"].to_numpy(), ae["quantum"].to_numpy(), ae["learned"].to_numpy())
)
sup["reference_independence"]["partial_corr_quantum_learned_given_reference"] = f(
    partial_corr(ae["quantum"].to_numpy(), ae["learned"].to_numpy(), ae["reference"].to_numpy())
)
# top-decile hard-case overlap
for c in CH:
    thr = ae[c].quantile(0.9)
    sup["reference_independence"][f"n_top_decile_{c}"] = int((ae[c] >= thr).sum())
qtop = ae["quantum"] >= ae["quantum"].quantile(0.9)
lto = ae["learned"] >= ae["learned"].quantile(0.9)
rto = ae["reference"] >= ae["reference"].quantile(0.9)
sup["reference_independence"]["overlap_table_top_decile"] = {
    "n_quantum_top": int(qtop.sum()),
    "n_learned_top": int(lto.sum()),
    "n_reference_top": int(rto.sum()),
    "quantum_and_learned_both_top": int((qtop & lto).sum()),
    "quantum_and_reference_both_top": int((qtop & rto).sum()),
    "learned_and_reference_both_top": int((lto & rto).sum()),
    "all_three_both_top": int((qtop & lto & rto).sum()),
    "quantum_top_also_reference_top_fraction": f(float((qtop & rto).sum() / qtop.sum())),
    "chance_fraction_if_independent": 0.1,
}
# where reference is worst, are the others also bad?
sup["reference_independence"]["mean_ae_when_reference_top_decile"] = {
    c: f(float(ae.loc[rto, c].mean())) for c in CH
}
sup["reference_independence"]["mean_ae_when_reference_bottom_half"] = {
    c: f(float(ae.loc[~rto, c].mean())) for c in CH
}
# reference rank-share of wins per observation
wins = pd.DataFrame({c: (ae[c] == ae.min(axis=1)) for c in CH})
sup["reference_independence"]["n_observations_where_channel_is_best"] = {
    c: int(wins[c].sum()) for c in CH
}
sup["reference_independence"]["n_observations_where_channel_is_worst"] = {
    c: int((ae[c] == ae.max(axis=1)).sum()) for c in CH
}
# reference error vs quantum/learned error sign agreement
errs = pd.DataFrame({c: df[f"{c}_err"].to_numpy(dtype=float) for c in CH})
sup["reference_independence"]["sign_agreement_of_signed_errors"] = {
    "quantum_vs_learned": f(float((np.sign(errs["quantum"]) == np.sign(errs["learned"])).mean())),
    "quantum_vs_reference": f(float((np.sign(errs["quantum"]) == np.sign(errs["reference"])).mean())),
    "learned_vs_reference": f(float((np.sign(errs["learned"]) == np.sign(errs["reference"])).mean())),
}

# ------------------------------------------------- rounding/units double-check
sup["checks"] = {
    "all_543_have_three_estimates": bool(
        df[[common.CHAN_COL[c] for c in CH]].notna().all().all()
    ),
    "experiment_min_max": [f(float(exp.min())), f(float(exp.max()))],
    "units_are_kcal_per_mol": True,
    "quantum_equals_db_bde_exactly_on_in_db_rows": bool(
        np.allclose(
            df.loc[df.in_db_row, "quantum_estimate_bde_kcal_mol"],
            df.loc[df.in_db_row, "db_bde"],
            atol=1e-9,
        )
    ),
    "max_abs_db_minus_quantum_on_in_db_rows": f(
        float(np.max(np.abs(df.loc[df.in_db_row, "db_bde"] - df.loc[df.in_db_row, "quantum_estimate_bde_kcal_mol"])))
    ),
}

res = json.loads(JSON.read_text())
res["supplementary_robustness"] = sup
JSON.write_text(json.dumps(res, indent=2, default=common._json_default) + "\n", encoding="utf-8")
print(f"updated {JSON}")

print("\n--- quantum vs learned by train flag ---")
for k, v in sup["quantum_vs_learned_by_train_flag"].items():
    print(f"{k:<18} n={v['n_pairs']} meanD={v['mean_abs_error_difference']:+.3f} "
          f"CI[{v['mean_difference_ci95'][0]:+.3f},{v['mean_difference_ci95'][1]:+.3f}] "
          f"sig={v['mean_difference_significant']} medD={v['median_abs_error_difference']:+.3f} "
          f"wilcox_p={v['wilcoxon_p']:.3g}")
print("\n--- tail sensitivity (capping |error|) ---")
for k, v in sup["quantum_vs_learned_tail_sensitivity"].items():
    if isinstance(v, dict) and "mean_difference" in v:
        print(f"{k:<28} meanD={v['mean_difference']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] sig={v['significant']}")
print("\n--- by experiment quartile ---")
for k, v in sup["quantum_vs_learned_by_experiment_quartile"].items():
    print(f"{k:<14} n={v['n_pairs']} meanD={v['mean_abs_error_difference']:+.3f} "
          f"CI[{v['mean_difference_ci95'][0]:+.3f},{v['mean_difference_ci95'][1]:+.3f}] sig={v['mean_difference_significant']} "
          f"mae_q={v['mae_first']:.3f} mae_l={v['mae_second']:.3f}")
print("\n--- by bond class ---")
for k, v in sup["quantum_vs_learned_by_bond_class"].items():
    print(f"{k:<8} n={v['n_pairs']} meanD={v['mean_abs_error_difference']:+.3f} mae_q={v['mae_first']:.3f} mae_l={v['mae_second']:.3f}")
print("\n--- reference independence ---")
print(json.dumps(sup["reference_independence"]["overlap_table_top_decile"], indent=1))
print("sign agreement:", sup["reference_independence"]["sign_agreement_of_signed_errors"])
print("partial corrs:", {k: v for k, v in sup["reference_independence"].items() if k.startswith("partial")})
print("checks:", json.dumps(sup["checks"], indent=1))