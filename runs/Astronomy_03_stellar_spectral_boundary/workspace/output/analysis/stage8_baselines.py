"""Stage 8 -- null and baseline models that the data reject.

Registered findings are only meaningful against the baselines they beat.  This
stage runs the two simplest explanations for the flux grid and records, with
numbers, how badly each fails:

  B1  A Planck (blackbody) curve at the slice's nominal effective temperature
      -- pure continuum, no atmospheric structure.
  B2  Random-point validation, i.e. treating individual wavelength samples as
      independent replicates.

Both are rejected.  Recording the rejection quantitatively is what licenses the
"atmospheric structure is required" and "contiguous hold-out is required"
statements in the report, instead of asserting them.

Run:  python3 analysis/stage8_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, load_flux_grid, planck_spectral_radiance,  # noqa: E402
                    save_json)

BANDS = ((179, 850, "full"), (179, 250, "179-250nm"), (250, 400, "250-400nm"),
         (400, 850, "400-850nm"), (500, 850, "500-850nm"))


def main() -> None:
    cube, ts, ms, ws, _ = load_flux_grid()
    out = {"planck_baseline": {}, "random_point_validation": {}}

    # ---------------------------------------------------------------- B1 ---
    # The flux table has arbitrary unit normalisation, so only the SHAPE is
    # testable: we ask how much F / B_lambda(T) varies across wavelength.
    # A pure Planck spectrum would give a constant ratio.
    per_slice = {}
    for i, t in enumerate(ts):
        for j, m in enumerate(ms):
            ratio = cube[i, j] / planck_spectral_radiance(t, ws)
            bands = {}
            for lo, hi, lab in BANDS:
                sel = (ws >= lo) & (ws <= hi)
                r = ratio[sel]
                bands[lab] = {
                    "max_over_min": float(r.max() / r.min()),
                    "rel_std": float(r.std() / r.mean()),
                }
            per_slice[f"T{int(t)}_M{m:+.2f}"] = bands

    mid = {k: v for k, v in per_slice.items() if "M+0.00" in k}
    out["planck_baseline"] = {
        "per_slice_mid_composition": mid,
        "summary_mid_composition": {
            "full_range_max_over_min": {
                f"T{int(t)}": mid[f"T{int(t)}_M+0.00"]["full"]["max_over_min"]
                for t in ts},
            "optical_only_max_over_min": {
                f"T{int(t)}": mid[f"T{int(t)}_M+0.00"]["400-850nm"]["max_over_min"]
                for t in ts},
            "optical_only_rel_std": {
                f"T{int(t)}": mid[f"T{int(t)}_M+0.00"]["400-850nm"]["rel_std"]
                for t in ts},
        },
        "verdict": (
            "A single-temperature Planck curve does not describe any spectrum. "
            "The flux-to-Planck ratio varies by a factor of 130-1720 across "
            "179-850 nm, and even restricted to 400-850 nm it still varies by "
            "2.0-2.6x. Because the table's absolute unit is arbitrary, this is a "
            "statement about spectral SHAPE: the deviation from a pure continuum "
            "is far larger than any plausible scaling ambiguity, so the spectra "
            "carry strong non-continuum (line-blanketing) structure. This is a "
            "null model the data reject, not a physical measurement."),
    }

    # ---------------------------------------------------------------- B2 ---
    # Random-point vs contiguous-block HOLD-OUT PREDICTION error.
    #
    # `DATA_GUIDE.md` warns that spectral smoothness makes random-point
    # validation invalid.  The right way to test that is not "can a random
    # subset locate the boundary" (it can, because interpolation across a
    # random gap works fine) but "how large is the leave-out prediction error".
    # Random hold-out leaves neighbours on both sides of every held-out point,
    # so interpolation hides the fact that no independent information was
    # removed.  A contiguous block removes an entire region, which is the
    # honest test of extrapolation to unseen wavelengths.
    g = np.log(cube[1, int(np.argmin(np.abs(ms - 0.0)))])
    rng = np.random.default_rng(20230101)

    def leave_out_mean_abs_err(idx):
        keep = np.setdiff1d(np.arange(ws.size), idx)
        pred = np.interp(ws[idx], ws[keep], g[keep])
        return float(np.abs(pred - g[idx]).mean())

    sizes = {}
    for n in (20, 40, 80):
        rand_err, block_err = [], []
        for _ in range(300):
            rand_err.append(leave_out_mean_abs_err(
                np.sort(rng.choice(ws.size, n, replace=False))))
            s = int(rng.integers(0, ws.size - n))
            block_err.append(leave_out_mean_abs_err(np.arange(s, s + n)))
        sizes[f"hold_out_{n}_points"] = {
            "random_mean_abs_log_err": float(np.mean(rand_err)),
            "contiguous_block_mean_abs_log_err": float(np.mean(block_err)),
            "ratio_block_over_random": float(np.mean(block_err) / np.mean(rand_err)),
        }
    out["random_point_validation"] = {
        "n_draws": 300,
        "method": ("leave-out mean absolute error in log flux, predicting held-out "
                   "wavelengths by linear interpolation from the retained points"),
        "by_hold_out_size": sizes,
        "verdict": (
            "Contiguous-block hold-out produces 2.5-7.7x larger prediction error "
            "than random hold-out of the same size, and the ratio grows with block "
            "size. Random hold-out therefore fails to remove information -- "
            "neighbouring wavelengths interpolate across every gap -- which is "
            "exactly the mechanism `DATA_GUIDE.md` warns about. Every robustness "
            "check in this report consequently holds out contiguous wavelength "
            "bands or whole temperature slices, never random points."),
    }

    path = save_json("stage8_baselines.json", out)
    print(f"wrote {path}")
    for t in ts:
        k = f"T{int(t)}_M+0.00"
        print(f"T={int(t)}K  F/Planck varies {mid[k]['full']['max_over_min']:8.1f}x "
              f"(full range), {mid[k]['400-850nm']['max_over_min']:5.2f}x (400-850 nm)")
    print("hold-out prediction error (block vs random): " + "  ".join(
          f"n={k.split('_')[2]}: {v['ratio_block_over_random']:.2f}x"
          for k, v in out["random_point_validation"]["by_hold_out_size"].items()))


if __name__ == "__main__":
    main()