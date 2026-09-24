"""Stage 2 -- wavelength-, temperature- and composition-dependent boundaries.

This is the core numerical stage.  It quantifies, on the flux grid:

  * the wavelength at which the metallicity response changes sign
    (the `crossover` boundary), per temperature slice;
  * the wavelength at which the temperature response changes sign;
  * the spectral peak wavelength per slice;
  * amplitude curves of both responses across wavelength;
  * how strong the cross-slice agreement actually is (the boundary is shared
    across all three temperature slices, within the data's own resolution).

All slopes are computed on `log(photon_flux)` because the response spans more
than five decades across the wavelength range; this is a modelling choice that
Stage 6 re-tests by leave-one-temperature-slice-out replication.

Run:  python3 analysis/stage2_boundaries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_flux_grid, save_json  # noqa: E402

# Metallicity index at or below which the sampling is a uniform 0.05 ladder.
# Above this value the ladder is thinned (0.1 steps, and 0.55/0.65/0.75/0.85
# are absent), so slopes computed there are less constrained.
UNIFORM_M_MAX = 0.5
UV_MAX_NM = 400.0


def first_sign_crossing(ws, values):
    """Linear-interpolated first wavelength where `values` changes sign."""
    s = np.sign(values)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for k in idx:
        x0, x1 = ws[k], ws[k + 1]
        y0, y1 = values[k], values[k + 1]
        if y1 == y0:
            continue
        out.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return out


def main() -> None:
    cube, ts, ms, ws, _ = load_flux_grid()
    u = ms <= UNIFORM_M_MAX
    out = {"uniform_metallicity_region": {"max_index": UNIFORM_M_MAX,
                                          "n_levels": int(u.sum())}}

    # ---------------------------------------------------------------- A ----
    # Metallicity response: d log F / d M along the uniform ladder.
    slope_m = np.array([[np.polyfit(ms[u], np.log(cube[i, u, q]), 1)[0]
                         for q in range(ws.size)] for i in range(ts.size)])
    cross = {}
    for i, t in enumerate(ts):
        x = first_sign_crossing(ws, slope_m[i])
        cross[str(int(t))] = {
            "crossing_wavelengths_nm": x,
            "first_crossing_nm": x[0] if x else None,
            "slope_at_180nm": float(slope_m[i][0]),
            "slope_at_850nm": float(slope_m[i][-1]),
            "median_slope_longward_500nm": float(np.median(slope_m[i][ws >= 500])),
        }
    out["metallicity_response"] = cross

    # ---------------------------------------------------------------- B ----
    # Temperature response: the ratio F(6300)/F(5300) and where it crosses 1.
    temp = {}
    for j, mval in enumerate(ms):
        r = cube[2, j] / cube[0, j]
        x = first_sign_crossing(ws, r - 1.0)
        temp[str(float(mval))] = {
            "ratio_at_180nm": float(r[0]),
            "ratio_at_850nm": float(r[-1]),
            "first_crossing_below_1_nm": x[0] if x else None,
        }
    r_mid = cube[2, 20] / cube[0, 20]
    out["temperature_response"] = {
        "per_metallicity": temp,
        "ratio_profile_M0": {
            str(int(w)): float(r_mid[np.argmin(np.abs(ws - w))])
            for w in (180, 200, 250, 300, 400, 500, 600, 700, 850)},
        "monotone_decreasing_180_to_850": bool(np.all(r_mid[0] > r_mid[-1])),
    }

    # Also 6300/5800 and 5800/5300 crossovers (adjacent-slice comparison).
    adj = {}
    for (a, b, label) in ((0, 1, "5800/5300"), (1, 2, "6300/5800")):
        xs = []
        for j in range(ms.size):
            xs.append(first_sign_crossing(ws, cube[b, j] / cube[a, j] - 1.0))
        adj[label] = {"median_first_crossing_nm": float(
            np.median([v[0] for v in xs if v]))}
    out["adjacent_temperature_crossovers"] = adj

    # ---------------------------------------------------------------- C ----
    # Peak wavelength of each spectrum (flux maximum).
    peaks = np.array([[float(ws[cube[i, j].argmax()]) for j in range(ms.size)]
                      for i in range(ts.size)])
    out["peak_wavelength"] = {
        str(int(t)): {"min": float(peaks[i].min()),
                      "median": float(np.median(peaks[i])),
                      "max": float(peaks[i].max())}
        for i, t in enumerate(ts)}

    # ---------------------------------------------------------------- D ----
    # Amplitude of the metallicity response across wavelength, full span.
    j1 = int(np.argmin(np.abs(ms - (-1.0))))
    j2 = int(np.argmin(np.abs(ms - 0.9)))
    amp = np.log(cube[:, j2, :] / cube[:, j1, :])
    out["metallicity_amplitude_log"] = {
        str(int(t)): {str(w): float(amp[i][np.argmin(np.abs(ws - w))])
                      for w in (180, 250, 300, 400, 500, 700, 850)}
        for i, t in enumerate(ts)}

    # ---------------------------------------------------------------- E ----
    # Is the UV suppression of high metallicity monotone in the index?
    mono = {}
    for i, t in enumerate(ts):
        d = np.diff(np.log(cube[i]), axis=0)
        uv = d[:, ws <= UV_MAX_NM]
        mono[str(int(t))] = {
            "frac_steps_increasing_uv": float((uv > 0).mean()),
            "frac_steps_increasing_all_wavelengths": float((d > 0).mean()),
        }
    out["metallicity_monotonicity"] = mono

    path = save_json("stage2_boundaries.json", out)
    print(f"wrote {path}")
    for t in ts:
        c = cross[str(int(t))]
        print(f"T={int(t)}K  metallicity-response sign change at "
              f"{c['first_crossing_nm']:.1f} nm  "
              f"(slope 180nm={c['slope_at_180nm']:.3g}, "
              f"850nm={c['slope_at_850nm']:.3g})")
    print(f"peak wavelength: " +
          "  ".join(f"T={int(t)}K {out['peak_wavelength'][str(int(t))]['median']:.0f}nm"
                    for t in ts))


if __name__ == "__main__":
    main()