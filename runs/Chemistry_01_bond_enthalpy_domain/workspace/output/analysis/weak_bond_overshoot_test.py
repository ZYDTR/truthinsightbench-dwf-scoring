"""Does the weak-bond overshoot survive the regression-to-the-mean baseline?

Both calibrated channels appear to overshoot weak bonds: selecting rows with
low experimental BDE and finding the estimates sit above them is exactly what
regression to the mean produces when the reference values carry noise, even if
the estimators are unbiased in truth.

This script tests the two explanations against each other using only workspace
data:

  O1  genuine bias      the channels really are too high for weak bonds.
  O2  selection effect  E = truth + noise, so conditioning on low E drags
                        selected rows below truth and manufactures overshoot.

Discriminating test: the selection effect predicts that the overshoot at a
given experimental value is proportional to how far the experimental value sits
below the local truth, and it must also appear in the QUANTUM channel whenever
that channel is an accurate proxy for truth.  If quantum is essentially the
database's own computed value on the matched subset, then the overshoot of
quantum against experiment is bounded by the reference noise at every
magnitude -- it cannot be a property of the estimator alone.

A second, sharper test: the reference channel is a coarse lookup and is much
noisier.  The selection explanation predicts the overshoot shrinks when we use
the CHANNEL as the conditioning variable instead of the experimental value
(conditioning on an estimator does not select on reference noise).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def bin_overshoot(E, pred, edges):
    """Signed bias of pred-E within bins of the EXPERIMENTAL value."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E >= lo) & (E < hi)
        if m.sum() == 0:
            continue
        d = pred[m] - E[m]
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        p = None
        if n_pos + n_neg > 0:
            from scipy import stats

            p = float(stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue)
        rows.append(
            {
                "bin": f"[{lo},{hi})",
                "n": int(m.sum()),
                "mean_experimental": float(E[m].mean()),
                "bias_pred_minus_exp": float(d.mean()),
                "n_overestimate": n_pos,
                "n_underestimate": n_neg,
                "sign_test_p": p,
            }
        )
    return rows


def conditional_bias_test(E, X, pred, label):
    """Overshoot conditioned on the ESTIMATE rather than on the experiment.

    If the channels are unbiased in truth, then within a narrow band of
    predicted values the experimental values should scatter symmetrically
    around the prediction.  A genuine bias shows up as a non-zero mean
    difference conditioned on the prediction itself.
    """
    d = E - pred  # experimental minus prediction
    rows = []
    for lo, hi in zip([-1e9, 40, 60, 70, 80, 90, 100, 110], [40, 60, 70, 80, 90, 100, 110, 1e9]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() < 5:
            continue
        dd = d[m]
        from scipy import stats

        t, p = stats.ttest_1samp(dd, 0.0)
        rows.append(
            {
                "predicted_bin": f"[{lo},{hi})",
                "n": int(m.sum()),
                "mean_exp_minus_pred": float(dd.mean()),
                "t": float(t),
                "p": float(p),
                "reject_unbiased": bool(p < 0.05),
            }
        )
    return {"conditioning_variable": label, "bins": rows}


def main() -> int:
    t = common.build_analysis_table(verbose=False)
    E = t[common.EXP_COL].to_numpy(float)
    Q = t[common.CHAN_COL["quantum"]].to_numpy(float)
    L = t[common.CHAN_COL["learned"]].to_numpy(float)
    R = t[common.CHAN_COL["reference"]].to_numpy(float)
    matched = t.in_db_row.to_numpy(bool)

    edges = [0, 60, 70, 80, 90, 100, 110, 200]

    # ---- Test 1: overshoot as a function of the experimental value --------
    by_exp = {c: bin_overshoot(E, p, edges) for c, p in
              (("quantum", Q), ("learned", L), ("reference", R))}

    # ---- Test 2: condition on the estimate instead -----------------------
    by_pred = {
        c: conditional_bias_test(E, p, p, c) for c, p in
        (("quantum", Q), ("learned", L), ("reference", R))
    }

    # ---- Test 3: does the quantum channel overshoot on the DB-matched
    # subset, where the quantum value is (near-)frozen to the database?
    q_matched_bias = {}
    for lo, hi in ((0, 70), (70, 80), (80, 1e9)):
        m = matched & (E >= lo) & (E < hi)
        if m.sum() < 3:
            continue
        q_matched_bias[f"[{lo},{hi})"] = {
            "n": int(m.sum()),
            "bias_quantum_minus_exp": float((Q[m] - E[m]).mean()),
            "mean_db_bde": float(t.db_bde.to_numpy()[m].mean()),
            "bias_db_minus_exp": float((t.db_bde.to_numpy()[m] - E[m]).mean()),
        }

    # ---- Test 4: the selection-effect prediction -------------------------
    # Under O2, conditioning on E < c selects rows whose noise is negative.
    # The apparent overshoot should then be of the same order as the
    # reference noise sigma, and it should NOT grow much beyond it.
    floor = common.RESULTS / "headline_noise_floor.json"
    sigma_ref = None
    if floor.exists():
        import json

        sigma_ref = json.loads(floor.read_text())["noise_floor"]["sigma_ref"]

    lowest_bin = by_exp["learned"][0] if by_exp["learned"] else None
    sel_effect_prediction = (
        "Under the selection explanation the overshoot in the lowest bin should be "
        "bounded by roughly the reference sigma; observed learned overshoot is "
        f"{lowest_bin['bias_pred_minus_exp']:.2f} kcal/mol against sigma_ref={sigma_ref:.2f}."
        if lowest_bin and sigma_ref
        else "insufficient data"
    )

    payload = {
        "workstream": "ADVERSARIAL-weak-bond-overshoot",
        "competing_explanations": {
            "O1_genuine_bias": "channels are genuinely too high for weak bonds",
            "O2_selection_regression_to_mean": (
                "E = truth + noise; conditioning on low E manufactures apparent overshoot "
                "even for an unbiased estimator"
            ),
        },
        "test1_overshoot_by_experimental_bin": by_exp,
        "test2_bias_conditioned_on_estimate": by_pred,
        "test3_quantum_overshoot_on_db_matched": q_matched_bias,
        "test4_selection_prediction": sel_effect_prediction,
        "sigma_ref_used": sigma_ref,
    }
    common.save_json(payload, "weak_bond_overshoot_test.json")

    print("\n=== overshoot binned by EXPERIMENTAL value (bias = pred - exp) ===")
    for c in ("quantum", "learned", "reference"):
        print(f"-- {c}")
        for r in by_exp[c]:
            print(f"   {r['bin']:>10s} n={r['n']:3d} bias={r['bias_pred_minus_exp']:+7.3f} "
                  f"sign+ {r['n_overestimate']:3d}/{r['n_underestimate']:3d} p={r['sign_test_p']}")
    print("\n=== bias conditioned on the ESTIMATE (exp - pred; 0 = unbiased) ===")
    for c in ("quantum", "learned", "reference"):
        print(f"-- {c}")
        for r in by_pred[c]["bins"]:
            print(f"   {r['predicted_bin']:>12s} n={r['n']:3d} mean(exp-pred)={r['mean_exp_minus_pred']:+7.3f} "
                  f"p={r['p']:.3g} reject_unbiased={r['reject_unbiased']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())