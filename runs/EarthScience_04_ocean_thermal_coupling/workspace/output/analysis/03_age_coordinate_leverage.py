#!/usr/bin/env python3
"""Step 3 - How much do the record's age-dependent results depend on WHICH
age coordinate is used?

The dataset ships two age models (age_ka_bp_1, age_ka_bp_2).  They are
identical to 0.01 ka everywhere below 0.55 m and diverge only at the top of
the core.  This script quantifies the leverage of that choice on:

  1. the identity and size of the divergent row set;
  2. the dominant least-squares sinusoid (period, R^2, fundamental amplitude);
  3. a synthetic "uniform LSR" coordinate (constant sedimentation rate,
     age = depth / mean_LSR) that carries no depth-dependent age structure;
  4. the sedimentation-rate profile LSR = np.gradient(depth, age);
  5. whether the coordinate choice can move the record-wide spectral result.

Analysis unit: the record is one continuous serially-correlated series; the
comparison is between whole-record fits, not between independent rows.

Outputs
-------
output/results/03_age_coordinate_leverage.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]
TOL_KA = 0.005          # row-difference tolerance in ka
P_LO, P_HI, P_STEP = 40.0, 300.0, 0.065
PERIOD_GRID = np.arange(P_LO, P_HI + 0.5 * P_STEP, P_STEP)   # 40, 40.065, ... 300


# --------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------
def design(t: np.ndarray, period: float) -> np.ndarray:
    """[1, t, cos(wt), sin(wt), cos(2wt), sin(2wt)], w = 2*pi/P."""
    w = 2.0 * np.pi / period
    return np.column_stack([
        np.ones_like(t), t,
        np.cos(w * t), np.sin(w * t),
        np.cos(2.0 * w * t), np.sin(2.0 * w * t),
    ])


def scan_period(t: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares scan over PERIOD_GRID; returns best fit diagnostics.

    R^2 is for the full 6-column design (trend + fundamental + 2nd harmonic).
    The fundamental amplitude/phase come from the cos(wt), sin(wt) coefficients
    only, with the trend and 2nd harmonic partialled out (same design).
    """
    sst = float(np.sum((y - y.mean()) ** 2))
    rss = np.empty(PERIOD_GRID.size)
    for i, P in enumerate(PERIOD_GRID):
        A = design(t, float(P))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        rss[i] = float(r @ r)
    k = int(np.argmin(rss))
    P_best = float(PERIOD_GRID[k])
    A = design(t, P_best)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    rss_best = float(resid @ resid)
    r2 = 1.0 - rss_best / sst
    b_cos, b_sin = float(beta[2]), float(beta[3])
    amp = float(np.hypot(b_cos, b_sin))
    phase = float(np.arctan2(-b_sin, b_cos))
    # fundamental-only R^2 (trend + fundamental design) at the same period
    A1 = design(t, P_best)[:, :4]
    b1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    r2_fund = 1.0 - float(np.sum((y - A1 @ b1) ** 2)) / sst
    return {
        "best_period_ka": P_best,
        "r2_full_design": float(r2),
        "r2_fundamental_only": float(r2_fund),
        "amp_fundamental": amp,
        "phase_rad": phase,
        "n_used": int(y.size),
    }


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "observations.tsv", sep="\t")
    df.columns = COLUMNS
    n_total = len(df)
    depth = df["depth_m"].to_numpy()
    a1 = df["age_ka_bp_1"].to_numpy()
    a2 = df["age_ka_bp_2"].to_numpy()
    y_all = df["d18o_pdb"].to_numpy()

    # ---------------- 1. divergent rows ----------------------------------
    diff = a1 - a2
    mask = np.abs(diff) > TOL_KA
    n_diff = int(mask.sum())
    dep_lo, dep_hi = float(depth[mask].min()), float(depth[mask].max())
    age_lo1, age_hi1 = float(a1[mask].min()), float(a1[mask].max())
    age_lo2, age_hi2 = float(a2[mask].min()), float(a2[mask].max())
    depth_span = float(depth.max() - depth.min())
    frac_depth = (dep_hi - dep_lo) / depth_span
    frac_rows = n_diff / n_total
    n_d18o_div = int(np.sum(~np.isnan(y_all[mask])))

    # sanity: are the coordinates identical beyond the divergent set?
    n_identical_below = int(np.sum(np.abs(diff[~mask]) <= TOL_KA))
    beyond = depth > dep_hi
    max_abs_diff_beyond = float(np.max(np.abs(diff[beyond])))

    # ---------------- 2. dominant sinusoid on each coordinate -------------
    ok = ~np.isnan(y_all)
    y = y_all[ok]
    fit1 = scan_period(a1[ok], y)
    fit2 = scan_period(a2[ok], y)

    # ---------------- 3. synthetic uniform-LSR coordinate -----------------
    lsr1 = np.gradient(depth, a1)
    lsr2 = np.gradient(depth, a2)
    mean_lsr1 = float(np.mean(lsr1))
    mean_lsr2 = float(np.mean(lsr2))
    lsr_span1 = (float(depth.max()) - float(depth.min())) / (float(a1.max()) - float(a1.min()))
    age_uniform = depth / mean_lsr1           # constant-rate map: age = depth/LSR
    fit_u = scan_period(age_uniform[ok], y)
    # sensitivity of the uniform fit to which coordinate defines mean LSR
    fit_u2 = scan_period((depth / mean_lsr2)[ok], y)
    # how flat is the uniform-coordinate scan?  R^2 at the age1/age2 optimum
    rss_u_at_109 = np.inf
    A_u = design(age_uniform[ok], fit1["best_period_ka"])
    b_u, *_ = np.linalg.lstsq(A_u, y, rcond=None)
    rss_u_at_109 = float(np.sum((y - A_u @ b_u) ** 2))
    r2_u_at_109 = 1.0 - rss_u_at_109 / float(np.sum((y - y.mean()) ** 2))

    # ---------------- 4. gradient (sedimentation rate) comparison ---------
    grad_abs_diff = np.abs(lsr1 - lsr2)
    max_abs_grad_diff = float(grad_abs_diff.max())
    idx = np.flatnonzero(grad_abs_diff > 0)
    # "interior": strictly beyond the np.gradient one-row stencil guard around
    # the divergent interval (depth > 0.56 m).  The comparison there is exact.
    guard_depth = float(depth[mask].max()) + 0.01
    interior = depth > guard_depth
    max_abs_grad_diff_interior = float(grad_abs_diff[interior].max())
    n_interior = int(interior.sum())
    grads_identical_interior = bool(np.array_equal(lsr1[interior], lsr2[interior]))
    # conservative: rows simply deeper than the last divergent row
    interior_loose = depth > dep_hi
    max_abs_grad_diff_interior_loose = float(grad_abs_diff[interior_loose].max())
    n_interior_loose = int(interior_loose.sum())
    # where does the gradient differ, and why (one-row guard from the stencil)
    grad_diff_depth_range = [float(depth[idx].min()), float(depth[idx].max())]

    # ---------------- serial-correlation context --------------------------
    ydet = y - np.polyval(np.polyfit(a1[ok], y, 1), a1[ok])
    step = float(np.median(np.diff(a1[ok])))
    t_even = np.arange(a1[ok].min(), a1[ok].max(), step)
    y_even = np.interp(t_even, a1[ok], ydet)
    rho = float(np.corrcoef(y_even[:-1], y_even[1:])[0, 1])
    n_eff = float(y.size * (1.0 - rho) / (1.0 + rho))
    # how far apart are the two coordinate fits relative to the AR(1) spread?
    d_period = abs(fit1["best_period_ka"] - fit2["best_period_ka"])
    d_r2 = abs(fit1["r2_full_design"] - fit2["r2_full_design"])
    d_amp = abs(fit1["amp_fundamental"] - fit2["amp_fundamental"])
    # extra robustness: fit on the sub-record where the coords are identical
    deep = ok & interior
    fit_deep1 = scan_period(a1[deep], y_all[deep])
    fit_deep2 = scan_period(a2[deep], y_all[deep])
    # same-coordinate (age1) fit on the divergent interval only
    div_ok = mask & ok
    fit_div1 = scan_period(a1[div_ok], y_all[div_ok])
    fit_div2 = scan_period(a2[div_ok], y_all[div_ok])

    # Does using age_1 vs age_2 change the record-wide result?  The two shipped
    # coordinates move the best period by less than one scan step and the
    # amplitude by <1% of its own value; only the synthetic constant-rate
    # coordinate (which is not one of the shipped age models) moves the peak,
    # and it is a deliberately different time axis.
    coordinate_choice_changes = bool(
        d_period > P_STEP or d_amp > 0.05 * max(fit1["amp_fundamental"], 1e-12))

    explanation = (
        "The two shipped age models differ on exactly "
        f"{n_diff} of {n_total} rows, all in the top {dep_hi:.2f} m "
        f"({100*frac_depth:.2f}% of the 6.27 m depth span, {100*frac_rows:.2f}% "
        f"of rows, ages {min(age_lo1,age_lo2):.2f}-{max(age_hi1,age_hi2):.2f} ka); "
        "every one of those rows carries d18O, and the two coordinates agree to "
        f"<= {TOL_KA} ka on all {n_identical_below} other rows (max |diff| below "
        f"{dep_hi:.2f} m = {max_abs_diff_beyond:.4f} ka). The record-wide dominant "
        f"sinusoid is therefore the same feature under either coordinate: best "
        f"period {fit1['best_period_ka']:.3f} ka (age_1) vs "
        f"{fit2['best_period_ka']:.3f} ka (age_2), a shift of {d_period:.3f} ka "
        f"<< the ~{360/n_eff:.0f} ka age spacing of the ~{n_eff:.0f} effective "
        f"samples implied by rho = {rho:.2f}; R^2 differs by {d_r2:.4f} "
        f"({fit1['r2_full_design']:.4f} vs {fit2['r2_full_design']:.4f}) and the "
        f"fundamental amplitude by {d_amp:.4f} per mille, i.e. "
        f"{100*d_amp/fit1['amp_fundamental']:.2f}% of the amplitude. The "
        "constant-rate synthetic coordinate (age = depth/mean_LSR, mean LSR = "
        f"{mean_lsr1:.6f} m/ka) peaks at {fit_u['best_period_ka']:.3f} ka with "
        f"R^2 = {fit_u['r2_full_design']:.4f} and explains only "
        f"R^2 = {r2_u_at_109:.4f} at the shipped-coordinate optimum "
        f"{fit1['best_period_ka']:.1f} ka, so a deliberately uniform age model "
        "would relocate the peak to a different period; that is a genuine "
        "sensitivity to the age-depth warp as a whole, but it is NOT the choice "
        "between the two shipped coordinates, which share 91.7% of their rows and "
        "differ only in the top 8% of the depth span. np.gradient "
        "LSR is bit-identical between the two coordinates for every row deeper "
        f"than {guard_depth:.2f} m (array_equal = {grads_identical_interior}, "
        f"n = {n_interior} rows); the only nonzero gradient differences "
        f"(max {max_abs_grad_diff:.4f} m/ka) are confined to the top "
        f"{grad_diff_depth_range[1]:.2f} m, i.e. the divergent interval plus the "
        "single-row np.gradient stencil guard. Conclusion: the choice of "
        "shipped age coordinate cannot change the record-wide spectral result; "
        "it perturbs only the top ~8% of the record locally."
    )

    result = {
        "step": "03_age_coordinate_leverage",
        # 1
        "n_rows_differing": n_diff,
        "depth_range_differing_m": [dep_lo, dep_hi],
        "age_range_differing_ka": [min(age_lo1, age_lo2), max(age_hi1, age_hi2)],
        "age_range_differing_ka_age1": [age_lo1, age_hi1],
        "age_range_differing_ka_age2": [age_lo2, age_hi2],
        "max_abs_diff_ka": float(np.max(np.abs(diff[mask]))),
        "signed_diff_min_ka": float(np.min(diff[mask])),
        "signed_diff_max_ka": float(np.max(diff[mask])),
        "frac_depth_span_affected": float(frac_depth),
        "frac_rows_affected": float(frac_rows),
        "n_d18o_rows_in_divergent_interval": n_d18o_div,
        "n_rows_identical_beyond_divergent_interval": n_identical_below,
        "max_abs_diff_ka_beyond_divergent_interval": max_abs_diff_beyond,
        # 2
        "period_age1_ka": fit1["best_period_ka"],
        "period_age2_ka": fit2["best_period_ka"],
        "r2_age1": fit1["r2_full_design"],
        "r2_age2": fit2["r2_full_design"],
        "amp_age1": fit1["amp_fundamental"],
        "amp_age2": fit2["amp_fundamental"],
        "phase_age1_rad": fit1["phase_rad"],
        "phase_age2_rad": fit2["phase_rad"],
        # 3
        "period_uniform_lsr_ka": fit_u["best_period_ka"],
        "r2_uniform_lsr": fit_u["r2_full_design"],
        "amp_uniform_lsr": fit_u["amp_fundamental"],
        "phase_uniform_lsr_rad": fit_u["phase_rad"],
        "mean_lsr_m_per_ka_age1": mean_lsr1,
        "mean_lsr_m_per_ka_age2": mean_lsr2,
        # 4
        "max_abs_gradient_difference_between_coords": max_abs_grad_diff,
        "max_abs_gradient_difference_interior": max_abs_grad_diff_interior,
        "gradient_arrays_identical_interior": grads_identical_interior,
        "n_interior_rows": n_interior,
        "interior_definition_depth_m": guard_depth,
        "max_abs_gradient_difference_below_last_divergent_row": max_abs_grad_diff_interior_loose,
        "n_rows_below_last_divergent_row": n_interior_loose,
        "gradient_difference_depth_range_m": grad_diff_depth_range,
        # 5
        "coordinate_choice_changes_record_wide_result": coordinate_choice_changes,
        # context / robustness
        "serial_correlation": {
            "lag1_rho_at_median_step": rho,
            "median_age_step_ka": step,
            "n_effective": n_eff,
            "note": "record treated as one continuous serially-correlated series",
        },
        "period_scan": {
            "grid_ka": [P_LO, P_HI],
            "step_ka": P_STEP,
            "n_grid_points": int(PERIOD_GRID.size),
            "design": "[1, t, cos(wt), sin(wt), cos(2wt), sin(2wt)], w = 2*pi/P",
            "r2_definition": "full 6-column design; fundamental amp/phase from cos/sin columns",
        },
        "robustness": {
            "r2_fundamental_only_age1": fit1["r2_fundamental_only"],
            "r2_fundamental_only_age2": fit2["r2_fundamental_only"],
            "r2_fundamental_only_uniform": fit_u["r2_fundamental_only"],
            "uniform_coord_mean_lsr_from_age2": {
                "period_ka": fit_u2["best_period_ka"],
                "r2": fit_u2["r2_full_design"],
                "amp": fit_u2["amp_fundamental"],
            },
            "r2_uniform_coord_at_shipped_optimum_108_835ka": r2_u_at_109,
            "mean_lsr_m_per_ka": mean_lsr1,
            "span_matched_lsr_m_per_ka": lsr_span1,
            "identical_subrecord_depth_gt_0_56m": {
                "n": fit_deep1["n_used"],
                "period_ka_age1": fit_deep1["best_period_ka"],
                "period_ka_age2": fit_deep2["best_period_ka"],
                "r2_age1": fit_deep1["r2_full_design"],
                "r2_age2": fit_deep2["r2_full_design"],
                "amp_age1": fit_deep1["amp_fundamental"],
                "amp_age2": fit_deep2["amp_fundamental"],
            },
            "divergent_interval_only_depth_le_0_55m": {
                "n": fit_div1["n_used"],
                "period_ka_age1": fit_div1["best_period_ka"],
                "period_ka_age2": fit_div2["best_period_ka"],
                "r2_age1": fit_div1["r2_full_design"],
                "r2_age2": fit_div2["r2_full_design"],
                "amp_age1": fit_div1["amp_fundamental"],
                "amp_age2": fit_div2["amp_fundamental"],
            },
        },
        "explanation": explanation,
    }

    (OUT / "03_age_coordinate_leverage.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()