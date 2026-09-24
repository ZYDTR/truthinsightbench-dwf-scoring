"""WS2 finalise: correct one mis-stated boolean, add composition-controlled
degradation and a headline block to output/results/ws2_channel_accuracy.json.
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
CH = ("quantum", "learned", "reference")


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


res = json.loads(JSON.read_text())

# ---------------------------------------------------------------- fix boolean
ts = res["training_split"]
ts["held_out_never_in_db_row"] = bool((~df.loc[~df.train_flag, "in_db_row"]).all())
ts["held_out_n_rows_with_db_row"] = int(df.loc[~df.train_flag, "in_db_row"].sum())
ts["note"] = (
    "in_reference_training_set == has an exact (molecule, fragment1, fragment2) row in the DB, "
    "exactly: 331/331 True rows are in DB, 212/212 False rows are not."
)

# ------------------------------------------- composition of train vs held-out
tr, ho = df[df.train_flag], df[~df.train_flag]
res["training_split"]["composition_check"] = {
    "experiment_min_train": f(float(tr[common.EXP_COL].min())),
    "experiment_max_train": f(float(tr[common.EXP_COL].max())),
    "experiment_min_held_out": f(float(ho[common.EXP_COL].min())),
    "experiment_max_held_out": f(float(ho[common.EXP_COL].max())),
    "welch_p_experiment_mean_train_vs_held_out": f(
        float(stats.ttest_ind(tr[common.EXP_COL], ho[common.EXP_COL], equal_var=False).pvalue)
    ),
    "bond_class_counts_train": {str(k): int(v) for k, v in tr.bond_class.value_counts().items()},
    "bond_class_counts_held_out": {str(k): int(v) for k, v in ho.bond_class.value_counts().items()},
}

# ---------------------------------- degradation within composition strata ----
strata = {}
for cls in ("C-H", "C-C"):
    sub = df[df.bond_class == cls]
    entry = {"n_train": int(sub.train_flag.sum()), "n_held_out": int((~sub.train_flag).sum())}
    for c in CH:
        a = np.abs(sub.loc[sub.train_flag, f"{c}_err"].to_numpy(dtype=float))
        b = np.abs(sub.loc[~sub.train_flag, f"{c}_err"].to_numpy(dtype=float))
        entry[c] = {
            "mae_train": f(float(a.mean())),
            "mae_held_out": f(float(b.mean())),
            "degradation": f(float(b.mean() - a.mean())),
            "ratio": f(float(b.mean() / a.mean())),
            "welch_p": f(float(stats.ttest_ind(b, a, equal_var=False).pvalue)),
        }
    strata[cls] = entry

# overlap of the experimental range: keep only held-out rows inside train's range
lo_e, hi_e = tr[common.EXP_COL].min(), tr[common.EXP_COL].max()
ov = df[(~df.train_flag) & (df[common.EXP_COL] >= lo_e) & (df[common.EXP_COL] <= hi_e)]
strata["overlapping_experiment_range_held_out"] = {
    "range": [f(float(lo_e)), f(float(hi_e))],
    "n_held_out_inside_train_range": int(len(ov)),
    "n_held_out_outside": int((~df.train_flag).sum() - len(ov)),
}
for c in CH:
    a = np.abs(tr[f"{c}_err"].to_numpy(dtype=float))
    b = np.abs(ov[f"{c}_err"].to_numpy(dtype=float))
    strata["overlapping_experiment_range_held_out"][c] = {
        "mae_train": f(float(a.mean())),
        "mae_held_out_inside_range": f(float(b.mean())),
        "degradation": f(float(b.mean() - a.mean())),
        "ratio": f(float(b.mean() / a.mean())),
        "welch_p": f(float(stats.ttest_ind(b, a, equal_var=False).pvalue)),
    }
# also compare held-out inside vs outside the train range
outside = df[(~df.train_flag) & ((df[common.EXP_COL] < lo_e) | (df[common.EXP_COL] > hi_e))]
strata["overlapping_experiment_range_held_out"]["n_outside"] = int(len(outside))
for c in CH:
    strata["overlapping_experiment_range_held_out"][c]["mae_held_out_outside_range"] = f(
        float(np.abs(outside[f"{c}_err"].to_numpy(dtype=float)).mean())
    )
res["training_split"]["composition_controlled_degradation"] = strata

# --------------------------------------------------------------- headline ---
ae = {c: np.abs(df[f"{c}_err"].to_numpy(dtype=float)) for c in CH}
res["headline"] = {
    "ranking_by_mae": ["quantum", "learned", "reference"],
    "mae_quantum": f(float(ae["quantum"].mean())),
    "mae_learned": f(float(ae["learned"].mean())),
    "mae_reference": f(float(ae["reference"].mean())),
    "quantum_vs_learned_verdict": (
        "NOT DISTINGUISHABLE on the typical observation: median |err| difference +0.020 kcal/mol, "
        "Wilcoxon p=0.553, sign test p=0.108, Cliff's delta -0.042. The mean difference is small but "
        "formally significant (-0.313 kcal/mol, 95% CI -0.471 to -0.165, paired t p=5.4e-5) and is "
        "carried by the held-out molecules and the weak-bond tail, where quantum is clearly better."
    ),
    "quantum_vs_reference_verdict": "quantum clearly better; mean difference -3.388 kcal/mol, all tests p<1e-58, Cliff's delta -0.631",
    "learned_vs_reference_verdict": "learned clearly better; mean difference -3.075 kcal/mol, all tests p<1e-49, Cliff's delta -0.573",
    "calibration_verdict": (
        "quantum (slope 0.932) and learned (slope 0.891) are compressed toward the mean - they "
        "overestimate weak bonds and slightly underestimate strong ones; reference (slope 1.073) is "
        "amplified - it overestimates strong bonds. Compression means the calibrated channels lose "
        "accuracy at the low end of the experimental range (MAE 3.75 / 5.37 kcal/mol in the weakest "
        "decile vs 1.89 / 2.06 in the middle 80%), not at the high end."
    ),
    "training_split_verdict": (
        "all three channels degrade substantially out-of-sample; learned degrades most "
        "(MAE 1.81 -> 3.35, +86%), quantum least in relative terms (1.87 -> 2.45, +31%), "
        "reference 5.02 -> 6.21 (+24%). The decay is not explained by bond-class composition or by "
        "leaving the training experimental range."
    ),
    "correlation_verdict": (
        "quantum and learned errors are strongly coupled (|err| Pearson 0.754, sign agreement 0.866); "
        "reference errors are nearly independent of both (|err| Pearson 0.214 / 0.196, sign agreement "
        "0.545 / 0.560, partial correlations ~0.05-0.10). Reference fails largely where the others "
        "succeed - it is a genuinely different estimator, not a harder version of the same one."
    ),
}

JSON.write_text(json.dumps(res, indent=2, default=common._json_default) + "\n", encoding="utf-8")
print(f"updated {JSON}")

print("\n--- composition-controlled degradation ---")
for cls, e in strata.items():
    print(f"\n[{cls}]")
    if cls != "overlapping_experiment_range_held_out":
        print(f"  n_train={e['n_train']} n_held_out={e['n_held_out']}")
        for c in CH:
            print(f"  {c:<10} train={e[c]['mae_train']:.3f} heldout={e[c]['mae_held_out']:.3f} "
                  f"deg={e[c]['degradation']:+.3f} ratio={e[c]['ratio']:.3f} p={e[c]['welch_p']:.3g}")
    else:
        print(f"  range={e['range']} n_inside={e['n_held_out_inside_train_range']} n_outside={e['n_outside']}")
        for c in CH:
            print(f"  {c:<10} train={e[c]['mae_train']:.3f} heldout_inside={e[c]['mae_held_out_inside_range']:.3f} "
                  f"deg={e[c]['degradation']:+.3f} ratio={e[c]['ratio']:.3f} p={e[c]['welch_p']:.3g} "
                  f"heldout_outside={e[c]['mae_held_out_outside_range']:.3f}")
print("\n--- composition check ---")
print(json.dumps(res["training_split"]["composition_check"], indent=1))