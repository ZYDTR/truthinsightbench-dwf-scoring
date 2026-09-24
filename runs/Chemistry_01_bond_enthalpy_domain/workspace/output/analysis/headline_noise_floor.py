"""Headline analysis: how much of each channel's error is the experimental
reference's own heterogeneity?

The benchmark has no repeated measurements from one laboratory.  Its
experimental values are literature values, so the "error" of an estimator
against them contains both the estimator's own error and the reference's
uncertainty.  This script estimates the reference-uncertainty floor with two
independent methods that do not use any external data:

  Method A  matched near-duplicate pairs.  Find pairs of observations whose
            QUANTUM estimates are nearly identical (within tol).  Physically
            these are the same bond type in a nearly identical environment, so
            the quantum channel assigns them the same value.  Any remaining
            experimental difference is (mostly) reference heterogeneity.
            This method does NOT assume the channels are independent of the
            experimental error, which is why it is the primary estimate.

  Method B  moment decomposition.  With three pairwise differences among
            E, Q, L built from two error terms each, solving the 3x3 system
            gives sigma_ref^2 = [V(E-Q) + V(E-L) - V(Q-L)] / 2 under the
            assumption that the reference error is independent of both
            channels' errors.

Both methods are reported; agreement between them is the robustness check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# sqrt(2/pi): E|X| for X ~ N(0, sigma).  Used to convert a sigma into the MAE
# an oracle estimator would show against noisy reference values.
MAE_FROM_SIGMA = float(np.sqrt(2.0 / np.pi))


def _pair_diffs(E, Q, L, tol):
    """Upper-triangle differences for all observation pairs with |dQ| <= tol.

    Fully vectorised: the 543x543 pair matrix is small enough to materialise.
    """
    dQ_full = Q[:, None] - Q[None, :]
    iu = np.triu_indices(len(E), k=1)
    mask = np.abs(dQ_full[iu]) <= tol
    dE = (E[:, None] - E[None, :])[iu][mask]
    dQ = dQ_full[iu][mask]
    dL = (L[:, None] - L[None, :])[iu][mask]
    return dE, dQ, dL


def sigma_from_pairs(dE):
    """sd of a difference of two independent draws is sigma*sqrt(2)."""
    return float(np.std(dE, ddof=1) / np.sqrt(2))


def matched_pair_sigma(E, Q, L, tol, min_pairs=50):
    """Sigma of the experimental reference from near-quantum-duplicate pairs."""
    dE, dQ, dL = _pair_diffs(E, Q, L, tol)
    if len(dE) < min_pairs:
        return None
    return {
        "tol": tol,
        "n_pairs": int(len(dE)),
        "sd_delta_experimental": float(np.std(dE, ddof=1)),
        "sigma_ref": sigma_from_pairs(dE),
        "quantum_selfcheck_sigma": sigma_from_pairs(dQ),
        "learned_selfcheck_sigma": sigma_from_pairs(dL),
        "oracle_mae": sigma_from_pairs(dE) * MAE_FROM_SIGMA,
    }


def bootstrap_sigma_ci(E, Q, tol, n_boot=2000, seed=20260919, alpha=0.05):
    """Bootstrap CI for the matched-pair sigma, resampling observations."""
    rng = np.random.default_rng(seed)
    n = len(E)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        dE, _, _ = _pair_diffs(E[idx], Q[idx], Q[idx], tol)
        if len(dE) < 20:
            continue
        out.append(sigma_from_pairs(dE))
    out = np.asarray(out)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(out.mean()), int(len(out))


def main() -> int:
    table = common.build_analysis_table(verbose=False)
    E = table[common.EXP_COL].to_numpy(float)
    Q = table[common.CHAN_COL["quantum"]].to_numpy(float)
    L = table[common.CHAN_COL["learned"]].to_numpy(float)
    R = table[common.CHAN_COL["reference"]].to_numpy(float)

    observed = {
        c: {
            "mae": common.mae(table[f"{c}_err"].to_numpy()),
            "rmse": common.rmse(table[f"{c}_err"].to_numpy()),
            "bias": float(table[f"{c}_err"].mean()),
        }
        for c in common.CHANNELS
    }

    # ---- Method A --------------------------------------------------------
    method_a = [r for r in (matched_pair_sigma(E, Q, L, t) for t in (0.05, 0.1, 0.2, 0.5, 1.0)) if r]
    primary = next(r for r in method_a if r["tol"] == 0.1)
    boot_lo, boot_hi, boot_mean, n_boot_ok = bootstrap_sigma_ci(E, Q, 0.1)

    # ---- Method B --------------------------------------------------------
    v_eq = float(np.var(E - Q, ddof=1))
    v_el = float(np.var(E - L, ddof=1))
    v_ql = float(np.var(Q - L, ddof=1))
    sigma_ref_sq = (v_eq + v_el - v_ql) / 2.0
    sigma_ref_b = float(np.sqrt(sigma_ref_sq))
    sigma_q_truth = float(np.sqrt(max(v_eq - sigma_ref_sq, 0.0)))
    sigma_l_truth = float(np.sqrt(max(v_el - sigma_ref_sq, 0.0)))

    floor = {
        "sigma_ref": primary["sigma_ref"],
        "sigma_ref_ci95": [boot_lo, boot_hi],
        "oracle_mae": primary["oracle_mae"],
        "oracle_mae_ci95": [boot_lo * MAE_FROM_SIGMA, boot_hi * MAE_FROM_SIGMA],
    }

    verdict = {}
    for c in common.CHANNELS:
        m = observed[c]["mae"]
        excess = m - floor["oracle_mae"]
        verdict[c] = {
            "mae": m,
            "oracle_floor_mae": floor["oracle_mae"],
            "excess_over_floor_kcal": excess,
            "ratio_mae_to_floor": m / floor["oracle_mae"],
            "mae_inside_floor_ci95": bool(floor["oracle_mae_ci95"][0] <= m <= floor["oracle_mae_ci95"][1]),
            "implied_sigma_channel_vs_truth": (
                sigma_q_truth if c == "quantum" else sigma_l_truth if c == "learned" else None
            ),
        }

    # ---- floor-definition sensitivity ------------------------------------
    # The Gaussian MAE conversion assumes a normal dE distribution; the matched
    # pairs are strongly heavy-tailed, so the floor must be reported under
    # several scale estimators rather than one.  A conclusion is only robust if
    # it holds under every definition.
    dE, _, _ = _pair_diffs(E, Q, L, 0.1)
    from scipy import stats as _stats

    sd = float(np.std(dE, ddof=1))
    mad = float(_stats.median_abs_deviation(dE, scale="normal"))
    iqr = float(_stats.iqr(dE))
    floor_variants = {
        "gaussian_from_sd": {
            "sigma": sd / np.sqrt(2),
            "floor_mae": sd / np.sqrt(2) * MAE_FROM_SIGMA,
            "definition": "sigma from sd of paired differences; MAE = sigma*sqrt(2/pi)",
        },
        "empirical_heavy_tailed": {
            "sigma": float(np.mean(np.abs(dE)) / np.sqrt(2)),
            "floor_mae": float(np.mean(np.abs(dE)) / np.sqrt(2)),
            "definition": "mean|dE|/sqrt(2); no normality assumption",
        },
        "robust_mad": {
            "sigma": mad / np.sqrt(2),
            "floor_mae": mad / np.sqrt(2) * MAE_FROM_SIGMA,
            "definition": "median absolute deviation scaled to sigma",
        },
        "robust_iqr": {
            "sigma": iqr / 1.349 / np.sqrt(2),
            "floor_mae": iqr / 1.349 / np.sqrt(2) * MAE_FROM_SIGMA,
            "definition": "IQR/1.349 as sigma; insensitive to the heavy tail",
        },
        "median_abs_diff": {
            "sigma": None,
            "floor_mae": float(np.median(np.abs(dE)) / np.sqrt(2)),
            "definition": "median|dE|/sqrt(2); the most conservative reading",
        },
    }
    dE_distribution = {
        "n_pairs": int(len(dE)),
        "sd": sd,
        "skew": float(_stats.skew(dE)),
        "excess_kurtosis": float(_stats.kurtosis(dE)),
        "shapiro_p": float(_stats.shapiro(dE).pvalue),
        "median_abs": float(np.median(np.abs(dE))),
        "mean_abs": float(np.mean(np.abs(dE))),
        "note": (
            "The distribution is strongly non-Gaussian (skew and excess kurtosis "
            "both positive, Shapiro p far below 0.05), so the Gaussian-form floor is "
            "the least conservative of the five and the robust forms are the ones a "
            "sceptical reader should prefer."
        ),
    }
    floor_ratio_table = {
        c: {k: round(observed[c]["mae"] / v["floor_mae"], 3) for k, v in floor_variants.items()}
        for c in common.CHANNELS
    }
    robust_conclusions = {
        "reference_worse_than_floor_under_all_definitions": bool(
            min(floor_ratio_table["reference"].values()) > 1.5
        ),
        "reference_ratio_range": [
            min(floor_ratio_table["reference"].values()),
            max(floor_ratio_table["reference"].values()),
        ],
        "quantum_ratio_range": [
            min(floor_ratio_table["quantum"].values()),
            max(floor_ratio_table["quantum"].values()),
        ],
        "quantum_at_or_below_floor_under_all_definitions": bool(
            max(floor_ratio_table["quantum"].values()) <= 1.0
        ),
        "statement": (
            "The reference channel's excess is robust: it stays well above every floor "
            "definition. The quantum channel's 'at or below floor' reading is NOT robust: "
            "it holds under the Gaussian and empirical floors but not under the robust "
            "MAD/IQR/median floors, because the paired-difference distribution is heavy "
            "tailed. The defensible claim is that quantum's error is of the same order as "
            "the reference heterogeneity, not that it is provably below it."
        ),
    }

    payload = {
        "workstream": "HEADLINE-noise-floor",
        "question": (
            "What part of each channel's measured error against the benchmark is the "
            "benchmark's own experimental reference heterogeneity rather than channel error?"
        ),
        "analysis_unit": "bond observation (543 rows); sigma estimated from observation pairs",
        "observed_channel_error": observed,
        "method_A_matched_near_duplicate_pairs": {
            "rationale": (
                "Pairs whose quantum estimates agree within tol are near-identical bonds. "
                "Their experimental spread estimates reference heterogeneity directly and "
                "does not assume channel/reference independence."
            ),
            "scans": method_a,
            "primary_tol": 0.1,
            "primary": primary,
            "bootstrap_sigma_ci95": [boot_lo, boot_hi],
            "bootstrap_mean_sigma": boot_mean,
            "bootstrap_replicates_used": n_boot_ok,
        },
        "method_B_moment_decomposition": {
            "rationale": (
                "sigma_ref^2 = [V(E-Q)+V(E-L)-V(Q-L)]/2 under independent errors; "
                "a closed-form cross-check on method A."
            ),
            "var_E_minus_Q": v_eq,
            "var_E_minus_L": v_el,
            "var_Q_minus_L": v_ql,
            "sigma_ref": sigma_ref_b,
            "sigma_ref_sq": sigma_ref_sq,
            "implied_sigma_quantum_vs_truth": sigma_q_truth,
            "implied_sigma_learned_vs_truth": sigma_l_truth,
        },
        "agreement_between_methods": {
            "sigma_ref_method_A": primary["sigma_ref"],
            "sigma_ref_method_B": sigma_ref_b,
            "absolute_difference": abs(primary["sigma_ref"] - sigma_ref_b),
            "agree": bool(abs(primary["sigma_ref"] - sigma_ref_b) < 0.25),
        },
        "noise_floor": floor,
        "channel_verdict_vs_floor": verdict,
        "floor_definition_sensitivity": {
            "variants": floor_variants,
            "paired_difference_distribution": dE_distribution,
            "mae_to_floor_ratios": floor_ratio_table,
            "robust_conclusions": robust_conclusions,
        },
        "interpretation": (
            "The calibrated channels' errors are of the same order as the benchmark's own "
            "reference heterogeneity, so a large part of their measured MAE is not channel "
            "error at all. The reference channel's error is roughly two to five times every "
            "floor definition and is therefore genuinely worse. Within the calibrated pair, "
            "the honest statement is bounded: quantum's MAE sits below the Gaussian-form "
            "floor but not below the robust-scale floors, so the sample supports 'quantum "
            "and learned are close to each other and to the reference noise' rather than "
            "'quantum is provably at the noise floor'."
        ),
        "limitations": [
            "Method A assumes near-identical quantum estimates imply near-identical true bonds; "
            "residual structural differences inflate the estimate, making the floor an upper bound "
            "on pure reference heterogeneity.",
            "Method B assumes the reference error is uncorrelated with both channels' errors; "
            "if the channels were fitted to these references the assumption fails and sigma_ref "
            "is overestimated.",
            "The experimental values are literature values from heterogeneous sources, not "
            "independent repeats from one laboratory; that is exactly why the floor exists.",
        ],
    }
    common.save_json(payload, "headline_noise_floor.json")

    print("\n=== HEADLINE: experimental reference noise floor ===")
    print(f"method A sigma_ref (tol=0.1, n_pairs={primary['n_pairs']}) = {primary['sigma_ref']:.4f}"
          f"  CI95 [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(f"method B sigma_ref                                      = {sigma_ref_b:.4f}")
    print(f"oracle MAE floor = {floor['oracle_mae']:.4f}  CI95 "
          f"[{floor['oracle_mae_ci95'][0]:.4f}, {floor['oracle_mae_ci95'][1]:.4f}]")
    for c in common.CHANNELS:
        v = verdict[c]
        print(f"  {c:10s} MAE={v['mae']:.4f}  ratio_to_floor={v['ratio_mae_to_floor']:.3f}  "
              f"inside_floor_CI={v['mae_inside_floor_ci95']}")
    print("\n=== floor-definition sensitivity (MAE / floor) ===")
    print(f"  paired-difference distribution: skew={dE_distribution['skew']:.2f} "
          f"exkurt={dE_distribution['excess_kurtosis']:.2f} "
          f"shapiro_p={dE_distribution['shapiro_p']:.2e}")
    hdr = "  {:<12s}".format("channel") + "".join(f"{k:>24s}" for k in floor_variants)
    print(hdr)
    for c in common.CHANNELS:
        print(f"  {c:<12s}" + "".join(f"{floor_ratio_table[c][k]:>24.3f}" for k in floor_variants))
    print(f"  robust statement: {robust_conclusions['statement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())