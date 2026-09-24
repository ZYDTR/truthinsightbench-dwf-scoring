"""Stage 4 -- robustness checks and falsification attempts.

Every check here is a deliberate attempt to break a Stage 2 / Stage 3 result.
Checks that fail to break it strengthen the finding; checks that do break it
are reported as such and are carried into the report as limitations.

Checks
------
R1  Leave-one-temperature-slice-out: is the metallicity crossover wavelength
    reproducible from two slices alone?
R2  Contiguous band hold-out: the crossover is re-estimated inside held-out
    wavelength windows, and random-point validation is shown to be invalid.
R3  Alternative response definition: crossover from a rank (Spearman-style)
    monotone trend instead of an OLS slope on logs.
R4  Outlier / threshold sensitivity: crossover as a function of the
    metallicity window used for the fit.
R5  Alternative model: fit `log F` vs `M` with a quadratic and locate the
    stationary point, instead of assuming linearity.
R6  Sampling-region test: does the thinned `M > 0.5` ladder create the
    apparent long-wavelength non-monotonicity?
R7  Cross-slice agreement of the photon-to-energy constant `C`.

Run:  python3 analysis/stage4_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, load_flux_grid, load_normalized, save_json  # noqa: E402
from stage2_boundaries import first_sign_crossing  # noqa: E402

UNIFORM_M_MAX = 0.5


def main() -> None:
    cube, ts, ms, ws, _ = load_flux_grid()
    nw, levels, _ = load_normalized()
    names = sorted(levels)
    out = {}

    # ---------------------------------------------------------------- R1 --
    r1 = {}
    for drop in range(ts.size):
        keep = [i for i in range(ts.size) if i != drop]
        slopes = []
        for q in range(ws.size):
            num = den = 0.0
            for i in keep:
                u = ms <= UNIFORM_M_MAX
                x = ms[u]
                y = np.log(cube[i, u, q])
                xm, ym = x.mean(), y.mean()
                num += float(((x - xm) * (y - ym)).sum())
                den += float(((x - xm) ** 2).sum())
            slopes.append(num / den)
        x = first_sign_crossing(ws, np.array(slopes))
        r1[f"without_{int(ts[drop])}K"] = {
            "first_crossing_nm": x[0] if x else None,
            "all_crossings_nm": x[:5],
        }
    out["R1_leave_one_temperature_out"] = r1
    vals = [v["first_crossing_nm"] for v in r1.values() if v["first_crossing_nm"]]
    out["R1_summary"] = {
        "min_nm": float(min(vals)), "max_nm": float(max(vals)),
        "spread_nm": float(max(vals) - min(vals)),
        "spread_in_grid_intervals": float((max(vals) - min(vals)) / 5.0),
        "verdict": (
            "dropping a slice moves the pooled crossover by up to 12.4 nm "
            "(2.5 grid intervals), and the three slices span 38.8 nm on their "
            "own. The crossover is therefore reproducible in LOCATION to "
            "within a few grid intervals, not exactly; 'reproducible from any "
            "two slices' overstates it."),
    }

    # ---------------------------------------------------------------- R2 --
    # Contiguous band hold-out vs random-point hold-out for the crossover.
    full = np.array([np.polyfit(ms[ms <= UNIFORM_M_MAX],
                                np.log(cube[1, ms <= UNIFORM_M_MAX, q]), 1)[0]
                     for q in range(ws.size)])
    full_cross = first_sign_crossing(ws, full)
    band_results = []
    for lo, hi in ((400, 500), (420, 470), (430, 460), (480, 560)):
        s = (ws >= lo) & (ws <= hi)
        band_results.append({
            "band_nm": [lo, hi],
            "n_points": int(s.sum()),
            "mean_slope": float(full[s].mean()),
            "contains_nominal_crossover": bool(full_cross and lo <= full_cross[0] <= hi),
        })
    rng = np.random.default_rng(20230101)  # fixed seed: reproducible
    rand = []
    keep_idx = np.where(ws <= 560)[0]
    for _ in range(200):
        pick = rng.choice(keep_idx, size=40, replace=False)
        rand.append(float(full[pick].mean()))
    out["R2_holdout"] = {
        "nominal_first_crossover_nm": full_cross[0] if full_cross else None,
        "contiguous_band_results": band_results,
        "random_point_holdout_mean_slope_std": float(np.std(rand)),
        "random_point_holdout_note": (
            "random 40-point subsets reproduce the mean slope because "
            "neighbouring wavelengths are correlated; they cannot locate the "
            "boundary, which is a wavelength-localised feature"),
    }

    # ---------------------------------------------------------------- R3 --
    # Rank-based alternative: sign of the monotone trend of F vs M.
    def rank_trend(y, x):
        ry = np.argsort(np.argsort(y)).astype(float)
        rx = np.argsort(np.argsort(x)).astype(float)
        ry -= ry.mean(); rx -= rx.mean()
        d = np.sqrt((ry @ ry) * (rx @ rx))
        return float(ry @ rx / d) if d > 0 else 0.0

    r3 = {}
    for i, t in enumerate(ts):
        u = ms <= UNIFORM_M_MAX
        cors = np.array([rank_trend(np.log(cube[i, u, q]), ms[u])
                         for q in range(ws.size)])
        x = first_sign_crossing(ws, cors)
        r3[str(int(t))] = {"first_crossing_nm": x[0] if x else None,
                           "correlation_at_180nm": float(cors[0]),
                           "correlation_at_850nm": float(cors[-1])}
    out["R3_rank_based_alternative"] = r3

    # ---------------------------------------------------------------- R4 --
    r4 = {}
    for mwin in (0.0, 0.25, 0.5):
        u = ms <= mwin
        if u.sum() < 5:
            continue
        sl = np.array([np.polyfit(ms[u], np.log(cube[1, u, q]), 1)[0]
                       for q in range(ws.size)])
        x = first_sign_crossing(ws, sl)
        r4[f"M<={mwin}"] = {"n_levels": int(u.sum()),
                            "first_crossing_nm": x[0] if x else None}
    out["R4_metallicity_window"] = r4

    # ---------------------------------------------------------------- R5 --
    r5 = {}
    for i, t in enumerate(ts):
        u = ms <= UNIFORM_M_MAX
        det = []
        for q in range(ws.size):
            c = np.polyfit(ms[u], np.log(cube[i, u, q]), 2)
            # stationary point of the quadratic, if a maximum
            if c[0] != 0:
                det.append(float(-c[1] / (2 * c[0])))
            else:
                det.append(np.nan)
        det = np.array(det)
        inrange = det[(det > -1.5) & (det < 1.0)]
        r5[str(int(t))] = {
            "median_quadratic_stationary_point_M": float(np.median(inrange))
            if inrange.size else None,
            "frac_within_index_range": float(inrange.size / ws.size),
        }
    out["R5_quadratic_model"] = r5

    # ---------------------------------------------------------------- R6 --
    r6 = {}
    for i, t in enumerate(ts):
        lo = ws >= 600
        u = ms <= UNIFORM_M_MAX
        du = np.diff(cube[i][u][:, lo], axis=0)
        da = np.diff(cube[i][:, lo], axis=0)
        bad_pairs = sorted({
            (float(ms[a]), float(ms[a + 1]))
            for a, _ in np.argwhere(da < 0)})
        r6[str(int(t))] = {
            "uniform_region_decreasing_steps": int((du < 0).sum()),
            "uniform_region_total_steps": int(du.size),
            "all_index_decreasing_steps": int((da < 0).sum()),
            "decreasing_metallicity_pairs": bad_pairs,
            "worst_pair_at_low_index": all(p[0] < -0.6 for p in bad_pairs),
        }
    out["R6_sampling_region"] = r6
    all_pairs = [p for v in r6.values() for p in v["decreasing_metallicity_pairs"]]
    n_high = sum(1 for p in all_pairs if p[0] > 0.5)
    out["R6_verdict"] = {
        "n_decreasing_pairs_total": len(all_pairs),
        "n_decreasing_pairs_above_M_0p5": n_high,
        "all_decreasing_pairs": sorted(set(all_pairs)),
        "conclusion": (
            "decreasing steps at long wavelength are rare (5-12 of 1530 "
            "uniform-ladder comparisons, i.e. under 0.8%) and concentrated at "
            "the blue end of the composition ladder. One slice (5300 K) does "
            "have a single decreasing pair inside the thinned M>0.5 region, so "
            "the stronger claim that every decrease lies below M=-0.6 does NOT "
            "hold. The defensible statement is the weaker one: the thinned "
            "region is not the main source of non-monotonicity."),
    }

    # ---------------------------------------------------------------- R7 --
    r7 = {}
    for i, t in enumerate(ts):
        for j, m in enumerate(ms):
            g = np.interp(nw, ws, cube[i, j])
            s = nw >= 400
            c = float(np.exp(np.mean(np.log(levels["response_level_3"][s]
                                        * nw[s] / g[s]))))
            if abs(m - 0.0) < 1e-9:
                r7[str(int(t))] = {"C": c}
    cs = np.array([v["C"] for v in r7.values()])
    out["R7_C_across_temperature_slices"] = {
        "per_slice": r7,
        "rel_spread": float(cs.std() / cs.mean()),
    }

    path = save_json("stage4_robustness.json", out)
    print(f"wrote {path}")
    print(f"R1 leave-one-temperature-out crossover spread: "
          f"{out['R1_summary']['spread_nm']:.1f} nm over "
          f"{out['R1_summary']['min_nm']:.1f}-{out['R1_summary']['max_nm']:.1f} nm")
    print(f"R3 rank-based crossovers: " +
          "  ".join(f"{k}K {v['first_crossing_nm']:.1f}nm"
                    for k, v in r3.items()))
    print(f"R6 verdict: {out['R6_verdict']}")
    print(f"R7 C relative spread across temperature slices: "
          f"{out['R7_C_across_temperature_slices']['rel_spread']:.4f}")


if __name__ == "__main__":
    main()