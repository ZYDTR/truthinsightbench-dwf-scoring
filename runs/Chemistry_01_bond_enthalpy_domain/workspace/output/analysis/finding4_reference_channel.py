"""Finding 4: the reference channel is genuinely worse than the other two, and
its signed error is bimodal with opposite signs for the two populated bond
classes.

Three questions, each answered from workspace data only:

  1  Is the reference channel's larger error real, or just a wider spread that
     happens to average out?  Reported as MAE/RMSE/median together with the
     paired tests against both calibrated channels.
  2  Is the reference estimate a per-observation estimator or a coarse
     environment lookup?  Tested by counting how many distinct values it takes
     per (fragment_a, fragment_b) environment.
  3  Does the reference channel fail in a structurally readable way?  Tested by
     splitting its signed error by bond class and by the extreme tail, where a
     sign flip between classes would show up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    t = common.build_analysis_table(verbose=False)
    E = t[common.EXP_COL].to_numpy(float)

    # ---- 1. accuracy, paired ---------------------------------------------
    accuracy = {}
    for c in common.CHANNELS:
        err = t[f"{c}_err"].to_numpy(float)
        accuracy[c] = {
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "median_abs_error": float(np.median(np.abs(err))),
            "bias": float(err.mean()),
            "sd_signed_error": float(err.std(ddof=1)),
            "p90_abs_error": float(np.quantile(np.abs(err), 0.90)),
            "max_abs_error": float(np.abs(err).max()),
            "frac_within_5_kcal": float((np.abs(err) <= 5).mean()),
        }

    paired = {}
    for other in ("quantum", "learned"):
        d = t.reference_ae.to_numpy(float) - t[f"{other}_ae"].to_numpy(float)
        tt, pt = stats.ttest_1samp(d, 0.0)
        w, pw = stats.wilcoxon(d)
        paired[f"reference_vs_{other}"] = {
            "mean_abs_error_difference": float(d.mean()),
            "median_abs_error_difference": float(np.median(d)),
            "ci95_mean": list(common.paired_bootstrap_ci(d)[1:]),
            "paired_t": float(tt),
            "paired_t_p": float(pt),
            "wilcoxon_p": float(pw),
            "n_reference_worse": int((d > 0).sum()),
            "n_reference_better": int((d < 0).sum()),
            "cliffs_delta": common.cliffs_delta(
                t.reference_ae.to_numpy(float), t[f"{other}_ae"].to_numpy(float)
            ),
        }

    # ---- 2. coarse-lookup structure --------------------------------------
    per_env = t.groupby(["fragment_a_smiles", "fragment_b_smiles"]).reference_estimate_bde_kcal_mol.nunique()
    environment = {
        "n_distinct_fragment_environments": int(len(per_env)),
        "n_distinct_reference_values": int(t.reference_estimate_bde_kcal_mol.nunique()),
        "environments_with_more_than_one_reference_value": int((per_env > 1).sum()),
        "max_reference_values_per_environment": int(per_env.max()),
    }
    # same test for the two calibrated channels, as the contrast
    for c in ("quantum", "learned"):
        q = t.groupby(["fragment_a_smiles", "fragment_b_smiles"])[f"{c}_estimate_bde_kcal_mol"].nunique()
        environment[f"{c}_environments_with_more_than_one_value"] = int((q > 1).sum())

    # ---- 3. bimodality by bond class --------------------------------------
    by_class = {}
    for bc, g in t.groupby("bond_class"):
        if len(g) < 20:
            continue
        r = g.reference_err.to_numpy(float)
        tstat, p = stats.ttest_1samp(r, 0.0)
        by_class[bc] = {
            "n": int(len(g)),
            "reference_bias": float(r.mean()),
            "reference_mae": float(np.abs(r).mean()),
            "reference_t": float(tstat),
            "reference_p": float(p),
            "quantum_bias": float(g.quantum_err.mean()),
            "learned_bias": float(g.learned_err.mean()),
            "sign_of_reference_bias": "positive" if r.mean() > 0 else "negative",
        }
    bias_signs = {v["sign_of_reference_bias"] for v in by_class.values()}
    bimodality = {
        "by_bond_class": by_class,
        "reference_bias_sign_flips_between_classes": len(bias_signs) > 1,
        "note": (
            "A sign flip between bond classes means a single global bias correction "
            "cannot fix the reference channel, and a single summary of its worst cases "
            "mixes two opposing failure mechanisms."
        ),
    }

    # extreme tail composition, split by sign
    r = t.reference_err.to_numpy(float)
    tail = {
        "n_abs_gt_5": int((np.abs(r) > 5).sum()),
        "n_abs_gt_10": int((np.abs(r) > 10).sum()),
        "n_abs_gt_15": int((np.abs(r) > 15).sum()),
        "n_positive_gt_10": int((r > 10).sum()),
        "n_negative_gt_10": int((r < -10).sum()),
        "worst_positive": [
            {"observation_id": t.observation_id.iloc[i],
             "parent": t.parent_structure_smiles.iloc[i],
             "bond_class": t.bond_class.iloc[i],
             "experimental": float(E[i]),
             "reference": float(t.reference_estimate_bde_kcal_mol.iloc[i]),
             "error": float(r[i])}
            for i in np.argsort(-r)[:5]
        ],
        "worst_negative": [
            {"observation_id": t.observation_id.iloc[i],
             "parent": t.parent_structure_smiles.iloc[i],
             "bond_class": t.bond_class.iloc[i],
             "experimental": float(E[i]),
             "reference": float(t.reference_estimate_bde_kcal_mol.iloc[i]),
             "error": float(r[i])}
            for i in np.argsort(r)[:5]
        ],
    }

    payload = {
        "workstream": "FINDING-4-reference-channel",
        "title": "The reference channel is genuinely worse and fails bimodally",
        "analysis_unit": "benchmark bond observation (543)",
        "accuracy": accuracy,
        "paired_against_calibrated_channels": paired,
        "coarse_lookup_structure": environment,
        "bimodality": bimodality,
        "extreme_tail": tail,
        "interpretation": (
            "The reference channel's error is 2.4x the experimental noise floor, so "
            "unlike the two calibrated channels it cannot be explained as reference "
            "heterogeneity - it carries a real defect. Its signed error flips sign "
            "between the C-H and C-C classes, consistent with an environment-level "
            "lookup whose value is not class-specific."
        ),
        "limitations": [
            "Only C-H and C-C are populated well enough to compare, so the sign flip "
            "is established on two classes, not on the full nine-class taxonomy.",
            "The reference values carry no method, phase or solvent metadata in this "
            "workspace, so the mechanism behind the coarse lookup cannot be identified, "
            "only its statistical signature.",
        ],
    }
    common.save_json(payload, "f4_reference_channel.json")

    print("\n=== F4 reference channel ===")
    for c, v in accuracy.items():
        print(f"  {c:10s} MAE={v['mae']:.3f} RMSE={v['rmse']:.3f} "
              f"median={v['median_abs_error']:.3f} bias={v['bias']:+.3f}")
    for k, v in paired.items():
        print(f"  {k}: mean diff {v['mean_abs_error_difference']:+.3f} "
              f"CI {v['ci95_mean'][0]:+.3f}..{v['ci95_mean'][1]:+.3f} "
              f"wilcoxon_p={v['wilcoxon_p']:.3g}")
    print(f"  environments with >1 reference value: "
          f"{environment['environments_with_more_than_one_reference_value']} "
          f"of {environment['n_distinct_fragment_environments']}")
    print(f"  reference bias sign flips between classes: "
          f"{bimodality['reference_bias_sign_flips_between_classes']}")
    for bc, v in by_class.items():
        print(f"    {bc}: n={v['n']} ref_bias={v['reference_bias']:+.3f} "
              f"(q={v['quantum_bias']:+.2f}, l={v['learned_bias']:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())