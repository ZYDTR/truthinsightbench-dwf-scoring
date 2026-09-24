"""WS2 finalise part 2: unit-sensitivity of the paired verdicts and an honest
replacement for the training-split headline claim.

The part-1 headline asserted the out-of-sample decay is "not explained by
bond-class composition or by leaving the training range".  The stratified
numbers in the same JSON contradict that for quantum and reference, so the
claim is replaced with what the strata actually show.
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

JSON = common.RESULTS / "ws2_channel_accuracy.json"
df = common.build_analysis_table(verbose=False)
CH = ("quantum", "learned", "reference")
EXP = common.EXP_COL


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


res = json.loads(JSON.read_text())

# ------------------------------------------------------------- unit sensitivity
units = {
    "bond_observation": df,
    "unique_molecule": df.groupby("parent_structure_smiles", as_index=False).agg(
        {EXP: "mean", **{common.CHAN_COL[c]: "mean" for c in CH}}
    ),
    "unique_fragment_pair": df.groupby(
        ["fragment_a_smiles", "fragment_b_smiles"], as_index=False
    ).agg({EXP: "mean", **{common.CHAN_COL[c]: "mean" for c in CH}}),
}
unit_out = {}
for uname, frame in units.items():
    ae = {c: np.abs(frame[common.CHAN_COL[c]].to_numpy(dtype=float) - frame[EXP].to_numpy(dtype=float))
          for c in CH}
    block = {"n_units": int(len(frame))}
    for c in CH:
        block[c] = {
            "mae": f(float(ae[c].mean())),
            "rmse": f(float(np.sqrt((ae[c] ** 2).mean()))),
            "median_abs_error": f(float(np.median(ae[c]))),
            "mean_signed_bias": f(
                float(frame[common.CHAN_COL[c]].mean() - frame[EXP].mean())
            ),
        }
    for a, b in (("quantum", "learned"), ("quantum", "reference"), ("learned", "reference")):
        d = ae[a] - ae[b]
        m, lo, hi = common.paired_bootstrap_ci(d, n_boot=20000)
        t, p = common.paired_t_test(d)
        w, pw = common.wilcoxon(d)
        npos, nneg = int((d > 0).sum()), int((d < 0).sum())
        block[f"{a}_vs_{b}"] = {
            "mean_abs_error_difference": f(m),
            "ci95": [f(lo), f(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "median_difference": f(float(np.median(d))),
            "median_difference_ci95_incl_zero": bool(
                np.quantile(
                    np.median(
                        d[np.random.default_rng(20260919).integers(0, len(d), size=(20000, len(d)))],
                        axis=1,
                    ),
                    0.025,
                )
                <= 0
                <= np.quantile(
                    np.median(
                        d[np.random.default_rng(20260919).integers(0, len(d), size=(20000, len(d)))],
                        axis=1,
                    ),
                    0.975,
                )
            ),
            "paired_t_p": f(p),
            "wilcoxon_p": f(pw),
            "cliffs_delta": f(common.cliffs_delta(ae[a], ae[b])),
            "n_first_worse": npos,
            "n_first_better": nneg,
            "n_tied": int((d == 0).sum()),
            "sign_test_p": f(float(__import__("scipy.stats", fromlist=["stats"]).binomtest(npos, npos + nneg, 0.5).pvalue)),
        }
    unit_out[uname] = block
res["paired_comparisons_by_analysis_unit"] = unit_out

# also spell out how many observations change hands in the quantum/learned pair
aeq = np.abs(df["quantum_err"].to_numpy(dtype=float))
ael = np.abs(df["learned_err"].to_numpy(dtype=float))
d = aeq - ael
res["paired_comparisons_by_analysis_unit"]["quantum_vs_learned_detail"] = {
    "n_ties_identical_abs_error": int((d == 0).sum()),
    "n_quantum_better": int((d < 0).sum()),
    "n_learned_better": int((d > 0).sum()),
    "median_abs_diff_among_discordant": f(float(np.median(np.abs(d[d != 0])))),
    "mean_abs_error_difference_excluding_ties": f(float(d[d != 0].mean())),
    "n_molecule_units_where_quantum_better": int(
        (
            np.abs(units["unique_molecule"]["quantum_estimate_bde_kcal_mol"] - units["unique_molecule"][EXP])
            < np.abs(units["unique_molecule"]["learned_estimate_bde_kcal_mol"] - units["unique_molecule"][EXP])
        ).sum()
    ),
    "n_molecule_units": int(len(units["unique_molecule"])),
}

# ------------------------------------------------- corrected split narrative --
tr, ho = df[df.train_flag], df[~df.train_flag]
lo_e, hi_e = tr[EXP].min(), tr[EXP].max()
inside = ho[(ho[EXP] >= lo_e) & (ho[EXP] <= hi_e)]
outside = ho[(ho[EXP] < lo_e) | (ho[EXP] > hi_e)]
corr = {}
for c in CH:
    a = float(np.abs(tr[f"{c}_err"]).mean())
    b = float(np.abs(ho[f"{c}_err"]).mean())
    bi = float(np.abs(inside[f"{c}_err"]).mean())
    bo = float(np.abs(outside[f"{c}_err"]).mean())
    corr[c] = {
        "aggregate_degradation": f(b - a),
        "degradation_in_range_excluding_4_extrapolation_rows": f(bi - a),
        "share_of_aggregate_degradation_from_the_4_out_of_range_rows": f((b - bi) / (b - a)) if b != a else None,
        "mae_out_of_range_rows": f(bo),
        "n_out_of_range_rows": int(len(outside)),
    }
res["training_split"]["degradation_decomposition"] = corr

res["headline"]["training_split_verdict"] = (
    "All three channels are less accurate on the 212 held-out molecules, but the decay is "
    "NOT uniform and is partly a composition/extrapolation artefact. Aggregate MAE: quantum "
    "1.87 -> 2.45 (+0.58, Welch p=0.005), learned 1.81 -> 3.35 (+1.55, p=7e-9), reference "
    "5.02 -> 6.21 (+1.18, p=0.001). Stratified by bond class the picture splits: on C-H, "
    "quantum shows no degradation at all (1.95 -> 2.03, p=0.70) and reference none (5.71 -> 5.72, "
    "p=0.98), while learned still degrades sharply (1.95 -> 3.08, p=4e-5); on C-C all three "
    "degrade (quantum 1.81 -> 3.73, learned 1.70 -> 4.15, reference 4.50 -> 7.78). "
    "Only 4 of the 212 held-out rows lie outside the training experimental range (61.4-160.2 "
    "kcal/mol) and they are wildly wrong for all three channels (MAE 11.3-15.1 kcal/mol), but "
    "they account for only ~29% of quantum's, ~15% of learned's and ~9% of reference's aggregate "
    "degradation. So: learned's out-of-sample decay is robust to every control; quantum's and "
    "reference's aggregate decay is largely, though not entirely, a bond-class composition effect "
    "(the held-out set is 75% C-H while the training set is 56% C-C)."
)

JSON.write_text(json.dumps(res, indent=2, default=common._json_default) + "\n", encoding="utf-8")
print(f"updated {JSON}")

print("\n--- per-unit summary ---")
for uname, b in unit_out.items():
    print(f"\n[{uname}] n={b['n_units']}")
    for c in CH:
        print(f"  {c:<10} MAE={b[c]['mae']:.3f} medAE={b[c]['median_abs_error']:.3f} bias={b[c]['mean_signed_bias']:+.3f}")
    for k in ("quantum_vs_learned", "quantum_vs_reference", "learned_vs_reference"):
        v = b[k]
        print(f"  {k:<22} meanD={v['mean_abs_error_difference']:+.3f} "
              f"CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] excl0={v['ci_excludes_zero']} "
              f"medD={v['median_difference']:+.3f} t_p={v['paired_t_p']:.3g} wilcox_p={v['wilcoxon_p']:.3g} "
              f"cliff={v['cliffs_delta']:+.3f} n_better/worse={v['n_first_better']}/{v['n_first_worse']}")
print("\n--- quantum vs learned detail ---")
print(json.dumps(unit_out["quantum_vs_learned_detail"], indent=1))
print("\n--- degradation decomposition ---")
print(json.dumps(corr, indent=1))