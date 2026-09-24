"""Stage 5 -- mechanism test for the ultraviolet breakdown of the
photon-to-energy relation found in Stage 3.

Stage 3 showed that `response_level_k * wavelength` is proportional to
`F(T=5800 K, M_k, wavelength)` over 330-850 nm, and that the relation fails
blueward of ~330 nm by a factor that is large and -- crucially -- identical
for all five composition settings.

This stage tests the functional form of that failure:

    H_step:  the deviation factor equals a numerator N divided by the LOCAL
             GRID WAVELENGTH STEP at that wavelength,
             deviation == N / (local grid step),
             with the local grid step evaluated by a left-edge convention.

This predicts a plateau at 5.0 wherever the grid step is exactly 1.0 nm, at
2.5 wherever it is exactly 2.0 nm, and 1.0 in the optical where the step is
5.0 nm -- which is what is observed.

IMPORTANT CORRECTION.  An earlier draft of this stage named a more specific
mechanism: that the series was produced by re-binning the grid onto a uniform
5 nm ladder by *summation* over the samples in each bin.  An independent
auditor tested four explicit summation conventions and REJECTED summation by
a factor of ~40 (mean absolute relative deviation 0.53 for an exact 5 nm
window sum, versus 0.013 for the `5/dw` form; see
`output/verification/AUDIT_main_session.md` section 7a).  What the data
support is a per-wavelength *coverage/interval-weighting* relation, not
summation.  The distinction matters and is carried into the report.

The `N` numerator is also NOT independently determined by these data:
`output/verification/GAP2_mechanism.json` fits it from the ultraviolet alone
and obtains N_hat with a 95% interval that contains 5.0 nm
(distinguishable_from_5nm = false).  What is established is the `1/step`
functional form; the identity of the constant is not.

Run:  python3 analysis/stage5_uv_mechanism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_flux_grid, load_normalized, save_json  # noqa: E402


def main() -> None:
    cube, ts, ms, ws, _ = load_flux_grid()
    nw, levels, _ = load_normalized()
    names = sorted(levels)
    out = {}

    # Deviation curve: constant fitted on the optical band ONLY (>= 400 nm),
    # using the mid-composition level (M = 0.0) as the reference.
    j_mid = int(np.argmin(np.abs(ms - 0.0)))
    g_mid = np.interp(nw, ws, cube[1, j_mid])
    optical = nw >= 400.0
    C = float(np.exp(np.mean(np.log(
        levels["response_level_3"][optical] * nw[optical] / g_mid[optical]))))
    dev = levels["response_level_3"] * nw / (C * g_mid)

    # Local grid wavelength step, evaluated at each normalised wavelength.
    # np.interp places the step at its lower edge, which is what a left-edge
    # binning convention implies.
    step = np.interp(nw, ws[:-1], np.diff(ws))

    pred = 5.0 / step
    rel = dev / pred - 1.0

    out["deviation_curve_sampled_nm"] = {
        str(int(w)): {
            "wavelength_nm": float(nw[np.argmin(np.abs(nw - w))]),
            "deviation": float(dev[np.argmin(np.abs(nw - w))]),
            "local_grid_step_nm": float(step[np.argmin(np.abs(nw - w))]),
            "predicted_5_over_step": float(pred[np.argmin(np.abs(nw - w))]),
        }
        for w in (180, 200, 220, 240, 260, 280, 296, 299, 300, 305, 310, 315,
                  320, 322, 324, 326, 330, 340, 400, 600, 850)}

    # Region-by-region performance of the prediction.
    regions = {"179-296 (ramp)": (179, 296), "299-322 (plateau)": (299, 322),
               "322-327 (narrow)": (322, 327), "330-850 (optical)": (330, 850),
               "all": (179, 850)}
    per_region = {}
    for lab, (a, b) in regions.items():
        m = (nw >= a) & (nw <= b)
        per_region[lab] = {
            "n_points": int(m.sum()),
            "mean_measured_deviation": float(dev[m].mean()),
            "mean_predicted": float(pred[m].mean()),
            "mean_abs_rel_error": float(np.abs(rel[m]).mean()),
            "max_abs_rel_error": float(np.abs(rel[m]).max()),
        }
    out["prediction_accuracy"] = per_region

    # The two decisive special cases: plateaus at 1 nm and 2 nm grid steps.
    plate = (nw >= 299) & (nw <= 322)
    nar = (nw >= 322) & (nw <= 327)
    out["exact_plateaus"] = {
        "1nm_region_299_322": {
            "n": int(plate.sum()),
            "measured_mean": float(dev[plate].mean()),
            "predicted": 5.0,
            "abs_error": float(abs(dev[plate].mean() - 5.0)),
            "rel_error": float(abs(dev[plate].mean() - 5.0) / 5.0),
        },
        "2nm_region_322_327": {
            "n": int(nar.sum()),
            "measured_mean": float(dev[nar].mean()),
            "predicted": 2.5,
            "abs_error": float(abs(dev[nar].mean() - 2.5)),
            "rel_error": float(abs(dev[nar].mean() - 2.5) / 2.5),
        },
        "optical_330_850": {
            "n": int((nw >= 330).sum()),
            "measured_mean": float(dev[nw >= 330].mean()),
            "predicted": 1.0,
            "abs_error": float(abs(dev[nw >= 330].mean() - 1.0)),
        },
    }

    # Alternative registration conventions, tested so the result does not
    # depend on one arbitrary interpolation choice.
    step_norm = np.gradient(nw)
    alternatives = {
        "5_over_grid_step": float(np.abs(dev / (5.0 / step) - 1).mean()),
        "5_over_normalized_file_own_step": float(np.abs(dev / (5.0 / step_norm) - 1).mean()),
        "constant_5": float(np.abs(dev / 5.0 - 1).mean()),
        "constant_1": float(np.abs(dev / 1.0 - 1).mean()),
    }
    out["alternative_models"] = alternatives
    out["best_model"] = min(alternatives, key=alternatives.get)

    # Composition independence: the same factor for all five levels, which is
    # what rules out a physical composition-driven explanation.
    devs = []
    for k, mval in zip(names, (-0.9, -0.5, 0.0, 0.5, 0.9)):
        g = np.interp(nw, ws, cube[1, int(np.argmin(np.abs(ms - mval)))])
        devs.append(levels[k] * nw / (C * g))
    devs = np.array(devs)
    out["composition_independence"] = {
        "per_level_plateau_mean_299_322": [float(devs[i][plate].mean())
                                           for i in range(devs.shape[0])],
        "rel_spread_across_levels_plateau": float(
            devs[:, plate].mean(axis=1).std() / devs[:, plate].mean()),
        "metallicity_sensitivity_at_300nm_log_slope": float(
            np.polyfit(ms, np.log(cube[1, :, int(np.argmin(np.abs(ws - 300)))]), 1)[0]),
        "interpretation": (
            "The five levels span a large range of composition sensitivity at "
            "300 nm, yet the factor is the same to <1%. A physical "
            "composition effect cannot be composition-independent."),
    }

    # Falsification test: the hypothesis predicts that the deviation changes
    # value only where the grid's wavelength step changes. Test that by
    # checking, for each deviation jump, whether a grid-step change occurs
    # within one local sampling interval. (An earlier draft's predicate was
    # vacuous -- `np.argmin(...) is not None` is always True; caught by
    # output/verification/AUDIT_main_session.md.)
    jumps = np.where(np.abs(np.diff(dev)) > 0.5)[0]
    step_jumps = np.where(np.abs(np.diff(step)) > 0.3)[0]
    pairing = []
    for x in jumps:
        if step_jumps.size:
            nearest = step_jumps[np.argmin(np.abs(step_jumps - x))]
            pairing.append({
                "deviation_jump_nm": float(nw[x]),
                "nearest_step_change_nm": float(nw[nearest]),
                "distance_nm": float(abs(nw[nearest] - nw[x])),
                "within_one_local_step": bool(
                    abs(nw[nearest] - nw[x]) <= max(step[x], 1.0)),
            })
    out["transition_alignment"] = {
        "deviation_jump_wavelengths_nm": [float(nw[x]) for x in jumps],
        "grid_step_jump_wavelengths_nm": [float(nw[x]) for x in step_jumps],
        "pairing": pairing,
        "all_deviation_jumps_accompany_a_step_change": bool(
            pairing and all(p["within_one_local_step"] for p in pairing)),
    }

    path = save_json("stage5_uv_mechanism.json", out)
    print(f"wrote {path}")
    print(f"best model among alternatives: {out['best_model']} "
          f"(mean|rel err| = {alternatives[out['best_model']]:.5f})")
    for lab, (a, b) in (("179-296 ramp", (179, 296)), ("299-322 plateau", (299, 322)),
                        ("322-327 narrow", (322, 327)), ("330-850 optical", (330, 850))):
        m = (nw >= a) & (nw <= b)
        print(f"  {lab:>16s}: measured {dev[m].mean():7.4f}  "
              f"predicted {pred[m].mean():7.4f}  mean|rel err| {np.abs(rel[m]).mean():.5f}")
    print(f"plateau value 299-322 nm: {out['exact_plateaus']['1nm_region_299_322']['measured_mean']:.4f} "
          f"(predicted 5.0, rel err {out['exact_plateaus']['1nm_region_299_322']['rel_error']:.4f})")
    print(f"narrow value 322-327 nm: "
          f"{out['exact_plateaus']['2nm_region_322_327']['measured_mean']:.4f} "
          f"(predicted 2.5, rel err {out['exact_plateaus']['2nm_region_322_327']['rel_error']:.4f})")


if __name__ == "__main__":
    main()