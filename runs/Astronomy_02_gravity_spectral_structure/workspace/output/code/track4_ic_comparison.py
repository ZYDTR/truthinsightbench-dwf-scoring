"""TRACK 4 - Information-criterion comparison against the supplied uniform alternative.

Data groups:
  data/uniform_alternative_comparison.tsv   (24 rows: uniform_velocity_cm_s, information_criterion)
  data/candidate_spectrum_comparison.tsv    (507 rows: candidate_family, vmax, vmin, f, w, IC)
  data/ranked_full_parameter_grid.tsv       (250 rows: rank, vmax, vmin, f, w, delta_IC)

Analysis unit for this track: one complete candidate spectral parameter combination
(the declared independent unit). Grid rows are NOT independent replicates, so no
p-value / standard error is computed over the 507 or 250 rows. Robustness is checked
by whole-slice leave-one-out over slices (velocity, frequency, vmax), which is a
deterministic sensitivity check, not an inferential test.

Outputs: output/results/ic_comparison.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code")
from common import load_candidates, load_uniform, load_ranked_grid, write_json  # noqa: E402

RES = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/results")
OUT_JSON = RES / "ic_comparison.json"

# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------
uni = load_uniform().rename(columns={"uniform_velocity_cm_s": "v",
                                     "information_criterion": "ic"})
uni = uni.sort_values("v").reset_index(drop=True)

cand = load_candidates().rename(columns={"peak_width_microhz": "w",
                                         "peak_frequency_microhz": "f",
                                         "max_velocity_cm_s": "vmax",
                                         "min_velocity_cm_s": "vmin",
                                         "information_criterion": "ic"})
rank = load_ranked_grid()  # columns: candidate_rank, vmax, vmin, f, w, ic (delta IC)

FA = cand[cand.candidate_family == "frequency_amplitude_scan"].copy()
MV = cand[cand.candidate_family == "minimum_velocity_scan"].copy()
MV_DIAG = MV[np.isclose(MV.vmax, MV.vmin)].copy()  # vmax == vmin diagonal

assert len(cand) == 507 and len(rank) == 250 and len(uni) == 24
assert len(FA) == 304 and len(MV) == 203

R = {
    "track": "TRACK 4 - information-criterion comparison vs the supplied uniform alternative",
    "analysis_unit": ("one complete candidate spectral parameter combination; the 507 candidate rows "
                      "and 250 ranked rows are NOT independent replicates, so no p-value or standard "
                      "error is computed over grid rows. Uncertainty is handled by whole-slice "
                      "leave-one-out sensitivity (velocity slice, frequency slice, vmax slice)."),
    "inputs": {
        "uniform_alternative_comparison.tsv": {"rows": int(len(uni)),
                                               "velocity_min": float(uni.v.min()),
                                               "velocity_max": float(uni.v.max())},
        "candidate_spectrum_comparison.tsv": {"rows": int(len(cand)),
                                              "n_frequency_amplitude_scan": int(len(FA)),
                                              "n_minimum_velocity_scan": int(len(MV)),
                                              "ic_min": float(cand.ic.min()),
                                              "ic_max": float(cand.ic.max())},
        "ranked_full_parameter_grid.tsv": {"rows": int(len(rank)),
                                           "delta_ic_min": float(rank.ic.min()),
                                           "delta_ic_max": float(rank.ic.max()),
                                           "vmax_values": sorted(int(x) for x in rank.vmax.unique()),
                                           "vmin_values": sorted(float(x) for x in rank.vmin.unique()),
                                           "width_values": sorted(int(x) for x in rank.w.unique())},
    },
}

# --------------------------------------------------------------------------------------
# Q1. Is the uniform table exactly the max_velocity == min_velocity diagonal of
#     the minimum_velocity_scan family?
# --------------------------------------------------------------------------------------
sub = R.setdefault("q1_uniform_is_mv_diagonal", {})

# 1a. exact float64 equality of the information-criterion values
diag_map = {float(r.vmin): float(r.ic) for _, r in MV_DIAG.iterrows()}
matched, unmatched = [], []
for _, r in uni.iterrows():
    v = float(r.v)
    if v in diag_map:
        matched.append({"v": v, "uniform_ic": float(r.ic), "diagonal_ic": diag_map[v],
                        "abs_diff": abs(float(r.ic) - diag_map[v])})
    else:
        unmatched.append({"v": v, "uniform_ic": float(r.ic)})

abs_diffs = np.array([m["abs_diff"] for m in matched]) if matched else np.array([np.nan])
sub["n_uniform_rows"] = int(len(uni))
sub["n_diagonal_rows_in_minimum_velocity_scan"] = int(len(MV_DIAG))
sub["n_matching_rows"] = int(len(matched))
sub["fraction_uniform_rows_matching"] = float(len(matched) / len(uni))
sub["max_abs_difference"] = float(np.max(abs_diffs))
sub["bit_identical"] = bool(np.all(abs_diffs == 0.0))
sub["n_unmatched_rows"] = int(len(unmatched))
sub["unmatched_rows"] = unmatched
sub["all_matches_at_peak_frequency_microhz"] = float(MV.f.min())
sub["all_matches_at_width_microhz"] = float(MV.w.min())
sub["candidate_family_min_vmin_evaluated"] = float(MV.vmin.min())
sub["candidate_family_min_vmax_evaluated"] = float(MV.vmax.min())
sub["candidate_family_diagonal_min_velocity"] = float(MV_DIAG.vmin.min())
sub["unmatched_velocities"] = [u["v"] for u in unmatched]
sub["unmatched_velocity_below_diagonal_floor"] = bool(
    all(u["v"] < MV_DIAG.vmin.min() for u in unmatched))
sub["unmatched_ics_absent_from_candidate_table"] = bool(
    all(not np.any(np.isclose(cand.ic.values, u["uniform_ic"])) for u in unmatched))
sub["unmatched_note"] = (
    "The 5 unmatched rows (v = 0.001, 0.002, 0.003, 0.005, 0.01 cm/s) are the vmax==vmin "
    "diagonal for velocities at or below the family's diagonal floor: minimum_velocity_scan "
    "evaluates the diagonal only for v >= 0.02 cm/s, so no vmax==vmin row exists at v <= 0.01. "
    "Their IC values (5551.0, 5062.9, 4467.1, 3656.9, 2797.3) appear nowhere in the 507-row "
    "candidate table, consistent with them being extrapolations of the same scan family to "
    "smaller velocities rather than a different model.")
sub["scope_of_identity"] = (
    "Identity holds exactly (bit-identical IC values) at every velocity where both tables "
    "evaluate the diagonal, i.e. 19/19 = 100% of overlapping velocities (v = 0.02 ... 500 cm/s) "
    "and 19/24 = 79.2% of the uniform table's rows.")
sub["max_abs_diff_vs_mv_diagonal_table"] = float(
    np.max(np.abs(MV_DIAG.sort_values("vmin").ic.values
                  - uni[uni.v.isin(diag_map)].sort_values("v").ic.values)))

# 1b. symmetric check: every diagonal row is present in the uniform table
sub["n_diagonal_rows_recovered_in_uniform"] = int(
    sum(1 for _, r in MV_DIAG.iterrows() if float(r.vmin) in set(uni.v.values)))

sub["interpretation"] = (
    f"{len(matched)}/{len(uni)} uniform rows ({100*len(matched)/len(uni):.1f}%) are numerically "
    "IDENTICAL (max |difference| = 0.0, bit-identical IC values) to the vmax==vmin diagonal slice "
    "of the minimum_velocity_scan family, all at peak_frequency = 1210 uHz and width = 300 uHz. "
    "The uniform table is therefore NOT an independent alternative model: it is a 1-D sub-slice of "
    "the very same evaluated family (the diagonal of the 2-D vmax x vmin scan). The remaining "
    f"{len(unmatched)} rows (velocities {[u['v'] for u in unmatched]}, IC "
    f"{[u['uniform_ic'] for u in unmatched]}) are the same diagonal continued below the family's "
    "evaluated diagonal floor (vmax==vmin exists only for v >= 0.02 cm/s; 0.001-0.005 cm/s appear "
    "nowhere in the candidate table), so they are almost certainly extrapolations of the same scan, "
    "not a distinct model. Consequence: any 'uniform vs candidate' comparison at a shared velocity "
    "is a comparison of a model with itself, and the uniform table cannot serve as an independent "
    "null for a family-level IC test.")

# --------------------------------------------------------------------------------------
# Q2. IC-vs-velocity curve of the uniform table on a log velocity axis
# --------------------------------------------------------------------------------------
sub = R.setdefault("q2_uniform_curve_log_velocity", {})
x = np.log10(uni.v.values)
y = uni.ic.values
sub["x_axis"] = "log10(uniform_velocity_cm_s) [dex]"
sub["n_points"] = int(len(uni))

imin = int(np.argmin(y))
sub["discrete_minimum"] = {"velocity_cm_s": float(uni.v.values[imin]),
                           "log10_velocity": float(x[imin]),
                           "ic": float(y[imin]),
                           "is_interior": bool(0 < imin < len(y) - 1)}
sub["neighbours_of_minimum"] = [
    {"velocity_cm_s": float(uni.v.values[j]), "ic": float(y[j])}
    for j in sorted(set([min(imin - 1, len(y) - 1), imin, max(imin + 1, 0)]))
]

# unimodality: sign of first differences (uniform log-spaced eh? no: compute on the log axis)
d = np.diff(y)
sub["n_negative_steps_before_min"] = int(np.sum(d[:imin] < 0))
sub["n_nonnegative_steps_before_min"] = int(np.sum(d[:imin] >= 0))
sub["n_positive_steps_after_min"] = int(np.sum(d[imin:] > 0))
sub["n_nonpositive_steps_after_min"] = int(np.sum(d[imin:] <= 0))
sub["sign_changes_in_first_difference"] = int(np.sum(np.diff(np.sign(d)) != 0))
sub["unimodal"] = bool(sub["sign_changes_in_first_difference"] == 1)

# smoothness: slopes between consecutive points on the log axis
# (piecewise slope = IC change per decade of velocity; a perfectly smooth convex curve
#  would have monotonically non-decreasing slopes)
slopes = d / np.diff(x)
sub["slope_ic_per_dex_min"] = float(np.min(slopes))
sub["slope_ic_per_dex_max"] = float(np.max(slopes))
sub["slopes_ic_per_dex"] = [float(s) for s in slopes]
sub["slope_sequence_monotone_nondecreasing"] = bool(np.all(np.diff(slopes) >= 0))
sub["n_slope_inversions"] = int(np.sum(np.diff(slopes) < 0))
# split by the velocity span of each piecewise slope (slope i spans v[i] -> v[i+1])
_v_mid = np.sqrt(uni.v.values[:-1] * uni.v.values[1:])  # geometric midpoint of each slope segment
sub["n_slope_inversions_segments_below_1cm_s"] = int(np.sum((np.diff(slopes) < 0) & (_v_mid[1:] <= 1.0)))
sub["n_slope_inversions_segments_above_1cm_s"] = int(np.sum((np.diff(slopes) < 0) & (_v_mid[1:] > 1.0)))
sub["max_abs_slope_inversion_ic_per_dex"] = float(np.max(np.abs(np.diff(slopes))))
inv = [{"between_velocity": [float(uni.v.values[i + 1]), float(uni.v.values[i + 2])],
        "slope_drop_ic_per_dex": float(slopes[i + 1] - slopes[i])}
       for i in range(len(slopes) - 1) if slopes[i + 1] < slopes[i]]
sub["slope_inversions"] = inv
sub["second_difference_sign_changes"] = int(np.sum(np.diff(np.sign(np.diff(y, 2))) != 0))
sub["convex_on_log_axis_strict"] = bool(np.all(np.diff(slopes) >= 0))
lo_dex = float(x[imin] - x[0])
hi_dex = float(x[-1] - x[imin])
sub["low_velocity_branch"] = {
    "range_velocity_cm_s": [float(uni.v.values[0]), float(uni.v.values[imin])],
    "n_points": int(imin + 1), "n_decades": lo_dex,
    "total_ic_change": float(y[imin] - y[0]),
    "mean_slope_ic_per_dex": float((y[imin] - y[0]) / lo_dex),
    "note": (f"steeply decreasing: IC falls from 5551.0 at 0.001 cm/s to 1522.1 at 0.1 cm/s, "
             f"i.e. {abs(float(y[imin] - y[0])):.1f} IC units lost over {lo_dex:.2f} decades "
             f"({float((y[imin] - y[0]) / lo_dex):.0f} IC/dex mean). All 8 steps are negative.")}
sub["high_velocity_branch"] = {
    "range_velocity_cm_s": [float(uni.v.values[imin]), float(uni.v.values[-1])],
    "n_points": int(len(y) - imin), "n_decades": hi_dex,
    "total_ic_change": float(y[-1] - y[imin]),
    "mean_slope_ic_per_dex": float((y[-1] - y[imin]) / hi_dex),
    "note": (f"increasing: IC rises from 1522.1 at 0.1 cm/s to 13553.2 at 500 cm/s, i.e. "
             f"{float(y[-1] - y[imin]):.1f} IC units gained over {hi_dex:.2f} decades "
             f"({float((y[-1] - y[imin]) / hi_dex):.0f} IC/dex mean). All 15 steps are positive.")}


def quad_fit(xs, ys):
    c = np.polyfit(xs, ys, 2)  # c[0] x^2 + c[1] x + c[2]
    pred = np.polyval(c, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    A, B, C = float(c[0]), float(c[1]), float(c[2])
    xv = -B / (2 * A)
    return {"quadratic_coeffs_ic_per_dex2": [A, B, C],
            "curvature_A_ic_per_dex2": A,
            "vertex_log10_velocity": float(xv),
            "vertex_velocity_cm_s": float(10 ** xv),
            "vertex_ic": float(np.polyval(c, xv)),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "n_points": int(len(xs)),
            "window_velocity_cm_s": [float(10 ** xs.min()), float(10 ** xs.max())],
            "halfwidth_dex_for_delta_ic_2": float(np.sqrt(2.0 / A)) if A > 0 else None,
            "halfwidth_dex_for_delta_ic_10": float(np.sqrt(10.0 / A)) if A > 0 else None}


# full-range quadratic (expected to be a poor global model; reported for completeness)
sub["quadratic_fit_full_range"] = quad_fit(x, y)
# local windows around the minimum
for name, lo, hi in [("local_3pt", 0.05, 0.2), ("local_5pt", 0.02, 0.5),
                     ("local_7pt", 0.01, 1.0)]:
    m = (uni.v.values >= lo) & (uni.v.values <= hi)
    sub[f"quadratic_fit_{name}"] = quad_fit(x[m], y[m])
    sub[f"quadratic_fit_{name}"]["is_inside_range"] = bool(
        lo <= sub[f"quadratic_fit_{name}"]["vertex_velocity_cm_s"] <= hi)
# curvature by finite difference on the 3 points straddling the grid minimum
i = imin
A_fd = (y[i + 1] - 2 * y[i] + y[i - 1]) / ((x[i + 1] - x[i]) ** 2) if 0 < i < len(y) - 1 else None
sub["curvature_finite_difference_ic_per_dex2"] = float(A_fd) if A_fd is not None else None
sub["curvature_finite_difference_window_velocity"] = [float(uni.v.values[i - 1]), float(uni.v.values[i + 1])]
sub["curvature_interpretation"] = (
    "Local parabola curvature near the minimum is A ~ 1.5e3-3.0e3 IC per dex^2 (A = 2*1e3 implies "
    "dIC = A*(dv/v)^2, so a factor 1.2 in velocity costs ~2-4 IC units). The vertex of the local "
    "3-point (0.05/0.1/0.2 cm/s) fit is at 0.086 cm/s with vertex IC 1508.9 (below the sampled "
    "minimum 1522.1 by 13.2, i.e. 0.9%), and the 5- and 7-point fits put the vertex at 0.079 and "
    "0.077 cm/s. The sampled minimum at 0.1 cm/s is therefore resolved to within a factor ~1.2-1.3 "
    "in velocity, and the true continuum minimum is not excluded below the sampled grid.")

sub["r2_local_windows"] = {k: sub[f"quadratic_fit_{k}"]["r2"]
                           for k in ["local_3pt", "local_5pt", "local_7pt", "full_range"]}
sub["smoothness_verdict"] = (
    "The curve is strictly UNIMODAL: the first difference is negative for all 8 steps below "
    "v = 0.1 cm/s and positive for all 15 steps above it (exactly one sign change). It is SMOOTH "
    "but not perfectly convex on the log axis: the piecewise slope (IC per decade) rises overall "
    f"from -3652 to +4602 IC/dex with {sub['n_slope_inversions']} small local inversions "
    f"(largest {sub['max_abs_slope_inversion_ic_per_dex']:.0f} IC/dex, at 0.002-0.003 cm/s), i.e. "
    "local floor noise/noise-realisation structure of order a few hundred IC units, not a second "
    "basin. A local quadratic in log10(v) fits the 3 points around the minimum with R^2 = 1.000 "
    "and 7 points with R^2 = 0.974; the global quadratic is poorer (R^2 = "
    f"{sub['quadratic_fit_full_range']['r2']:.4f}) because the curve is steeper than parabolic far "
    "from the minimum.")

# --------------------------------------------------------------------------------------
# Q3. Uniform minimum vs candidate-family minimum
# --------------------------------------------------------------------------------------
sub = R.setdefault("q3_uniform_vs_candidate_minimum", {})
cand_min = float(cand.ic.min())
cand_argmin = cand.loc[cand.ic.idxmin()]
uni_min = float(uni.ic.min())
uni_argmin = uni.loc[uni.ic.idxmin()]
rank_min = float(rank.ic.min())
rank_argmin = rank.loc[rank.ic.idxmin()]

sub["uniform_min"] = {"ic": uni_min, "velocity_cm_s": float(uni_argmin.v),
                      "family": "uniform alternative (= vmax==vmin diagonal of minimum_velocity_scan, "
                                "f = 1210 uHz, w = 300 uHz)"}
sub["candidate_family_min"] = {"ic": cand_min,
                               "candidate_family": str(cand_argmin.candidate_family),
                               "vmax_cm_s": float(cand_argmin.vmax),
                               "vmin_cm_s": float(cand_argmin.vmin),
                               "peak_frequency_microhz": float(cand_argmin.f),
                               "width_microhz": float(cand_argmin.w)}
sub["ranked_full_grid_min"] = {"delta_ic": rank_min,
                               "vmax_cm_s": float(rank_argmin.vmax),
                               "vmin_cm_s": float(rank_argmin.vmin),
                               "peak_frequency_microhz": float(rank_argmin.f),
                               "width_microhz": float(rank_argmin.w),
                               "candidate_rank": int(rank_argmin.candidate_rank)}
sub["ic_difference_uniform_minus_candidate"] = float(uni_min - cand_min)
sub["ic_ratio_uniform_over_candidate"] = float(uni_min / cand_min)
sub["candidate_rows_better_than_uniform_min"] = int(np.sum(cand.ic < uni_min))
sub["candidate_fraction_better_than_uniform_min"] = float(np.mean(cand.ic < uni_min))
sub["ranked_rows_better_than_uniform_min"] = int(np.sum(rank.ic < uni_min))
sub["note"] = (
    "A lower IC supports a RELATIVE ordering among the evaluated candidates only. It is not proof "
    "that the preferred family is physically unique, and in this dataset the comparison is partly "
    "circular: the uniform table is a literal sub-slice of the minimum_velocity_scan family "
    "(Q1), so family-versus-uniform is a nested comparison, not an independent model comparison.")

# --------------------------------------------------------------------------------------
# Q4. Is delta_information_criterion the absolute IC (offset zero)?
# --------------------------------------------------------------------------------------
sub = R.setdefault("q4_delta_ic_offset_check", {})
KEY = ["vmax", "vmin", "f", "w"]


def key_tuples(df):
    return [tuple(float(v) for v in row) for row in df[KEY].to_numpy()]


def key_tuples_rounded(df, nd=6):
    return [tuple(round(float(v), nd) for v in row) for row in df[KEY].to_numpy()]


cand_keys = key_tuples(cand)
rank_keys = key_tuples(rank)
cand_map = {k: float(v) for k, v in zip(cand_keys, cand.ic.values)}
rank_map = {k: float(v) for k, v in zip(rank_keys, rank.ic.values)}
exact_common = sorted(set(cand_map) & set(rank_map), key=lambda t: rank_map[t])
off_exact = np.array([abs(cand_map[k] - rank_map[k]) for k in exact_common])

cand_keys_r = key_tuples_rounded(cand)
rank_keys_r = key_tuples_rounded(rank)
cand_map_r = {k: float(v) for k, v in zip(cand_keys_r, cand.ic.values)}
rank_map_r = {k: float(v) for k, v in zip(rank_keys_r, rank.ic.values)}
common_r = sorted(set(cand_map_r) & set(rank_map_r), key=lambda t: rank_map_r[t])
off_r = np.array([abs(cand_map_r[k] - rank_map_r[k]) for k in common_r])

sub["key_columns"] = KEY
sub["overlap_rows_exact_float64_key"] = int(len(exact_common))
sub["max_abs_offset_exact_key"] = float(off_exact.max()) if len(off_exact) else None
sub["overlap_rows_key_rounded_1e-6"] = int(len(common_r))
sub["max_abs_offset_rounded_key"] = float(off_r.max()) if len(off_r) else None
sub["n_rows_offset_gt_1e-9"] = int(np.sum(off_r > 1e-9))
sub["extra_rows_recovered_by_tolerance"] = int(len(common_r) - len(exact_common))
sub["extra_rows_detail"] = (
    "The ranked file stores vmin = 0.03 as 0.0299999999999999 (2 ULP below the candidate file's "
    "0.03 = 0x1.eb851eb851eb8p-6), which breaks exact bitwise key matching for 3 rows "
    "(vmax=20/30/50 at f=1210, w=300). All 3 have |ic - delta_ic| < 1e-13 once keys are rounded."
)
sub["float_representation_difference_0.03_cm_s"] = float(abs(0.03 - 0.0299999999999999))
sub["relative_offset_vs_best"] = float(off_r.max() / min(cand_min, 1.0))
sub["offset_summary"] = (
    f"29 rows overlap on the parameter tuple (26 by strict float64 equality of the stored values; "
    f"+3 recovered when keys are rounded to 1e-6). Max |information_criterion - "
    f"delta_information_criterion| = {off_r.max():.2e} IC units, i.e. the offset is exactly zero "
    f"to float64 rounding noise (the paired values are identical to ~14 significant digits). "
    "delta_information_criterion IS the absolute IC with offset zero, not an IC re-referenced to "
    "the grid minimum. Consequence: the ranked table's best row (delta_ic = 0.0 at vmax=30, "
    "vmin=0.05, f=1010, w=275) is absent from candidate_spectrum_comparison.tsv and is genuinely "
    "better on the same IC scale than every row of that table (min 43.9). Any claim that the "
    "candidate table's minimum is the family optimum is therefore grid-limited, not a property of "
    "the model family.")
sub["overlap_rows"] = []
for k in common_r:
    m = rank[(rank.vmax.astype(float) == k[0]) & (np.isclose(rank.vmin.astype(float), k[1])) &
             (rank.f.astype(float) == k[2]) & (rank.w.astype(float) == k[3])]
    sub["overlap_rows"].append({"vmax": k[0], "vmin": k[1], "f": k[2], "w": k[3],
                                "candidate_ic": cand_map_r[k], "ranked_delta_ic": rank_map_r[k],
                                "abs_offset": abs(cand_map_r[k] - rank_map_r[k]),
                                "candidate_rank": int(m.candidate_rank.iloc[0]) if len(m) else None})

# --------------------------------------------------------------------------------------
# Q5. Best achievable IC per family
# --------------------------------------------------------------------------------------
sub = R.setdefault("q5_best_per_family", {})
fa_min_row = FA.loc[FA.ic.idxmin()]
mv_min_row = MV.loc[MV.ic.idxmin()]
mvdiag_min_row = MV_DIAG.loc[MV_DIAG.ic.idxmin()]
sub["frequency_amplitude_scan"] = {
    "n_rows": int(len(FA)), "best_ic": float(fa_min_row.ic),
    "vmax_cm_s": float(fa_min_row.vmax), "vmin_cm_s": float(fa_min_row.vmin),
    "peak_frequency_microhz": float(fa_min_row.f), "width_microhz": float(fa_min_row.w)}
sub["minimum_velocity_scan"] = {
    "n_rows": int(len(MV)), "best_ic": float(mv_min_row.ic),
    "vmax_cm_s": float(mv_min_row.vmax), "vmin_cm_s": float(mv_min_row.vmin),
    "peak_frequency_microhz": float(mv_min_row.f), "width_microhz": float(mv_min_row.w),
    "diagonal_subset_best_ic": float(mvdiag_min_row.ic),
    "diagonal_subset_best_velocity_cm_s": float(mvdiag_min_row.vmin)}
sub["uniform_alternative"] = {"n_rows": int(len(uni)), "best_ic": uni_min,
                              "velocity_cm_s": float(uni_argmin.v)}
sub["ranked_full_grid_context"] = {"n_rows": int(len(rank)), "best_delta_ic": rank_min,
                                   "vmax_cm_s": float(rank_argmin.vmax),
                                   "vmin_cm_s": float(rank_argmin.vmin),
                                   "peak_frequency_microhz": float(rank_argmin.f),
                                   "width_microhz": float(rank_argmin.w)}
sub["ranking"] = ["frequency_amplitude_scan", "minimum_velocity_scan", "uniform_alternative"]
sub["preferred_family"] = "frequency_amplitude_scan"
sub["margin_fa_over_mv"] = {"ic_difference": float(mv_min_row.ic - fa_min_row.ic),
                            "ratio": float(mv_min_row.ic / fa_min_row.ic),
                            "pct_of_best": float(100 * (mv_min_row.ic - fa_min_row.ic) / fa_min_row.ic)}
sub["margin_candidate_over_uniform"] = {"ic_difference": float(uni_min - cand_min),
                                        "ratio": float(uni_min / cand_min)}
sub["nested_warning"] = (
    "uniform_alternative is a literal subset of minimum_velocity_scan (Q1), so the gap "
    f"{uni_min:.1f} vs {mv_min_row.ic:.1f} is a nested-subset gap, not an independent model "
    "comparison. The only non-nested family comparison is frequency_amplitude_scan (43.9) vs "
    "minimum_velocity_scan (50.7): Delta = 6.8 IC units, ratio 1.155 - a marginal margin.")
sub["note"] = ("The ranked full-parameter grid (a wider, differently-sampled grid with "
               "width in {250,275,300,325} uHz) reaches delta_ic = 0.0 at "
               "(vmax=30, vmin=0.05, f=1010, w=275), better than every row of "
               "candidate_spectrum_comparison.tsv (best 43.9). Same IC scale (Q4), different grid.")

# --------------------------------------------------------------------------------------
# Q6. Boundary / sensitivity checks
# --------------------------------------------------------------------------------------
sub = R.setdefault("q6_boundary_sensitivity", {})
var = {}

# (a) only rows with min_velocity > 0
cand_pos = cand[cand.vmin > 0]
uni_pos = uni[uni.v > 0]
var["a_min_velocity_gt_0"] = {
    "candidate_family": {"n_rows": int(len(cand_pos)), "best_ic": float(cand_pos.ic.min()),
                         "family": str(cand_pos.loc[cand_pos.ic.idxmin()].candidate_family),
                         "vmax_cm_s": float(cand_pos.loc[cand_pos.ic.idxmin()].vmax),
                         "vmin_cm_s": float(cand_pos.loc[cand_pos.ic.idxmin()].vmin),
                         "peak_frequency_microhz": float(cand_pos.loc[cand_pos.ic.idxmin()].f),
                         "changed": bool(cand_pos.ic.min() != cand_min)},
    "uniform_alternative": {"n_rows": int(len(uni_pos)), "best_ic": float(uni_pos.ic.min()),
                            "velocity_cm_s": float(uni_pos.loc[uni_pos.ic.idxmin()].v),
                            "changed": bool(uni_pos.ic.min() != uni_min)},
    "frequency_amplitude_scan_rows_retained": int(np.sum(FA.vmin > 0)),
    "note": ("All 304 frequency_amplitude_scan rows have vmin = 0.0 exactly and are removed; the "
             "candidate minimum moves 43.9 -> 50.7 (minimum_velocity_scan). The uniform table is "
             "untouched (all 24 velocities > 0, min 1522.1 at 0.1)."),
}

# (b) only rows present in the top-250 ranked table
in_rank_mask = np.array([k in rank_map_r for k in cand_keys_r])
cand_in_rank = cand[in_rank_mask].copy()
var["b_only_rows_in_top250_ranked"] = {
    "candidate_family": {"n_rows": int(len(cand_in_rank)),
                         "best_ic": float(cand_in_rank.ic.min()),
                         "vmax_cm_s": float(cand_in_rank.loc[cand_in_rank.ic.idxmin()].vmax),
                         "vmin_cm_s": float(cand_in_rank.loc[cand_in_rank.ic.idxmin()].vmin),
                         "peak_frequency_microhz": float(cand_in_rank.loc[cand_in_rank.ic.idxmin()].f),
                         "width_microhz": float(cand_in_rank.loc[cand_in_rank.ic.idxmin()].w),
                         "changed": bool(cand_in_rank.ic.min() != cand_min)},
    "uniform_alternative": {"n_rows": 0, "best_ic": None,
                            "reason": ("no uniform (vmax==vmin) row appears in the top-250 ranked "
                                       "table: the ranked grid contains 0 rows with vmax == vmin, "
                                       "its vmax axis starts at 1 cm/s and its vmin axis stops at "
                                       "0.1 cm/s, while the uniform velocities run 0.001-500 cm/s. "
                                       "The uniform table is disjoint from the ranked table's "
                                       "parameter support, so restriction (b) leaves the uniform "
                                       "minimum unchanged at 1522.1 (v = 0.1).")},
    "ranked_top250_itself": {"best_delta_ic": rank_min, "vmax_cm_s": float(rank_argmin.vmax),
                             "vmin_cm_s": float(rank_argmin.vmin),
                             "peak_frequency_microhz": float(rank_argmin.f),
                             "width_microhz": float(rank_argmin.w)},
    "note": ("Restricting the candidate family to the 29 rows that also appear in the top-250 ranked "
             "table keeps the global optimum (43.9 at vmax=20, vmin=0, f=1010, w=300); the ranked "
             "table reaches 0.0 on a grid that additionally varies the width "
             "(w = 275 uHz, a value never evaluated in candidate_spectrum_comparison.tsv)."),
}
# family attribution of the top-250 rows via the same rounded-key convention as Q4
rank_fam = pd.DataFrame([{"key": k, "vmax": k[0], "vmin": k[1], "f": k[2], "w": k[3],
                          "ic": rank_map_r[k], "candidate_rank": rk}
                         for k, rk in zip(rank_keys_r, rank.candidate_rank.values)])
top250 = rank_fam.assign(candidate_family=[{
    k: f for k, f in zip(cand_keys_r, cand.candidate_family.values)}.get(k)
    for k in rank_keys_r])
var["b_top250_by_family"] = {str(k): int(v) for k, v in
                             top250.candidate_family.value_counts(dropna=False).items()}
var["b_top250_family_counts"] = {
    "matched_to_candidate_table": int(top250.candidate_family.notna().sum()),
    "not_in_candidate_table": int(top250.candidate_family.isna().sum())}

# (c) best row per peak-frequency value
per_f = cand.groupby("f").ic.min()
per_f_best_rows = cand.loc[cand.groupby("f").ic.idxmin()]
var["c_only_best_row_per_peak_frequency"] = {
    "candidate_family": {"n_frequencies": int(len(per_f)),
                         "best_ic": float(per_f.min()),
                         "peak_frequency_microhz": float(per_f.idxmin()),
                         "family": str(per_f_best_rows.loc[per_f_best_rows.f == per_f.idxmin()].candidate_family.iloc[0]),
                         "vmax_cm_s": float(per_f_best_rows.loc[per_f_best_rows.f == per_f.idxmin()].vmax.iloc[0]),
                         "vmin_cm_s": float(per_f_best_rows.loc[per_f_best_rows.f == per_f.idxmin()].vmin.iloc[0]),
                         "changed": bool(per_f.min() != cand_min)},
    "uniform_alternative": {"n_velocities": int(len(uni)), "best_ic": uni_min,
                            "note": ("the uniform table has no peak-frequency axis: every matched "
                                     "row sits at f = 1210 uHz, so this restriction is a no-op and "
                                     "the minimum stays 1522.1 at v = 0.1 cm/s.")},
    "per_frequency_minimum_ic": {str(float(k)): float(v) for k, v in per_f.items()},
    "ranked_top250_best_per_peak_frequency": {
        "best_delta_ic": float(rank.groupby("f").ic.min().min()),
        "peak_frequency_microhz": float(rank.groupby("f").ic.min().idxmin()),
        "n_frequencies": int(rank.f.nunique()),
        "note": ("applying the same 'best row per peak-frequency value' rule to the ranked top-250 "
                 "gives delta_ic = 0.0 at f = 1010 uHz - the same frequency as the candidate-family "
                 "optimum and as the ranked global best, confirming the preferred frequency band is "
                 "860-1210 uHz with a single best value at 1010 uHz."),
    },
}
var["c_per_frequency_minimum_ranked"] = {
    str(float(k)): float(v) for k, v in rank.groupby("f").ic.min().items()}

# leave-one-slice-out robustness (deterministic, not an inferential test)
loo_uni = {}
for i, v in enumerate(uni.v.values):
    y2 = np.delete(y, i)
    x2 = np.delete(x, i)
    loo_uni[str(float(v))] = {"argmin_velocity_cm_s": round(float(10 ** x2[int(np.argmin(y2))]), 12),
                              "min_ic": float(y2.min())}
var["c_loo_uniform_argmin"] = {"original": float(uni.v.values[imin]),
                               "values": loo_uni,
                               "argmin_changes_in_n_slices": int(sum(
                                   1 for k in loo_uni if loo_uni[k]["argmin_velocity_cm_s"] != float(uni.v.values[imin]))),
                               "argmin_set": sorted({loo_uni[k]["argmin_velocity_cm_s"] for k in loo_uni})}
fa_per_vmax = FA.groupby("vmax").ic.min()
loo_fa = {}
for v in fa_per_vmax.index:
    g = fa_per_vmax.drop(v)
    loo_fa[str(float(v))] = {"argmin_vmax_cm_s": float(g.idxmin()), "min_ic": float(g.min())}
var["c_loo_frequency_amplitude_scan_vmax"] = {
    "original_vmax_cm_s": float(fa_per_vmax.idxmin()),
    "values": loo_fa,
    "argmin_set": sorted({loo_fa[k]["argmin_vmax_cm_s"] for k in loo_fa})}
loo_mv = {}
mv_per_vmax = MV.groupby("vmax").ic.min()
for v in mv_per_vmax.index:
    loo_mv[str(float(v))] = {"argmin_vmax_cm_s": float(mv_per_vmax.drop(v).idxmin()),
                             "min_ic": float(mv_per_vmax.drop(v).min())}
var["c_loo_minimum_velocity_scan_vmax"] = {
    "original_vmax_cm_s": float(mv_per_vmax.idxmin()), "argmin_set": sorted(
        {loo_mv[k]["argmin_vmax_cm_s"] for k in loo_mv})}
loo_f = {}
for fv in per_f.index:
    if fv == per_f.idxmin():
        loo_f[str(float(fv))] = {"new_best_ic": float(per_f.drop(fv).min()),
                                 "new_best_frequency_microhz": float(per_f.drop(fv).idxmin()),
                                 "row_family": str(per_f_best_rows.loc[per_f_best_rows.f == per_f.drop(fv).idxmin()].candidate_family.iloc[0]),
                                 "row_vmax_cm_s": float(per_f_best_rows.loc[per_f_best_rows.f == per_f.drop(fv).idxmin()].vmax.iloc[0])}
var["c_loo_candidate_peak_frequency"] = loo_f

fa_loo_set = sorted({loo_fa[k]["argmin_vmax_cm_s"] for k in loo_fa})
mv_loo_set = sorted({loo_mv[k]["argmin_vmax_cm_s"] for k in loo_mv})
var["velocity_scale_stability"] = {
    "stable": True,
    "verdict": (
        "The preferred VELOCITY SCALE is stable. The frequency_amplitude_scan argmin is "
        f"vmax = {float(fa_per_vmax.idxmin()):.1f} cm/s and moves to {fa_loo_set[-1]:.1f} cm/s (a "
        f"factor 1.5) only when the winning vmax = 20 slice is itself deleted; the "
        f"minimum_velocity_scan argmin is vmax = {float(mv_per_vmax.idxmin()):.1f} cm/s moving to "
        f"{mv_loo_set[-1]:.1f} cm/s under leave-one-vmax-out; the top-29 ranked rows all use "
        "vmax in {20, 30}; the uniform slice prefers v = 0.1 cm/s moving to 0.05 cm/s (factor 2) "
        "only when 0.1 is deleted. Across every restriction (a), (b), (c) and every leave-one-out "
        "the preferred scale stays in the 0.03-0.1 cm/s band for the uniform/vmin axis and in the "
        "20-50 cm/s band for the vmax axis. The FREQUENCY choice is less stable: dropping the "
        f"winning f = 1010 uHz moves the candidate argmin to "
        f"{loo_f[str(float(per_f.idxmin()))]['new_best_frequency_microhz']:.0f} uHz "
        f"(IC 50.7, +6.8 units), which is the same marginal margin as the family gap."),
    "uniform_argmin_set_leave_one_out_cm_s": sorted({loo_uni[k]["argmin_velocity_cm_s"] for k in loo_uni}),
    "frequency_amplitude_scan_argmin_set_leave_one_out_cm_s": fa_loo_set,
    "minimum_velocity_scan_argmin_set_leave_one_out_cm_s": mv_loo_set,
}

sub["variants"] = var

# --------------------------------------------------------------------------------------
# Q7. IC scale / magnitude
# --------------------------------------------------------------------------------------
sub = R.setdefault("q7_ic_scale", {})
best = cand_min
best_any = rank_min
sub["best_ic_candidate_family"] = best
sub["best_ic_ranked_full_grid"] = best_any
sub["candidate_family_ic_distribution"] = {
    "mean": float(cand.ic.mean()), "sd": float(cand.ic.std(ddof=1)),
    "min": float(cand.ic.min()), "median": float(cand.ic.median()),
    "max": float(cand.ic.max()), "iqr": [float(cand.ic.quantile(0.25)), float(cand.ic.quantile(0.75))]}
sub["candidate_ic_iqr_width"] = float(cand.ic.quantile(0.75) - cand.ic.quantile(0.25))
sub["interpretation_of_delta_ic"] = {
    "delta_2": {"pct_of_best": float(100 * 2 / best),
                "pct_of_candidate_sd": float(100 * 2 / cand.ic.std(ddof=1)),
                "pct_of_full_range": float(100 * 2 / (cand.ic.max() - cand.ic.min())),
                "meaning": ("2 IC units is 4.6% of the best value and 0.08% of the family's "
                            "spread: the rank-1 (0.0) and rank-2 (2.0) rows of the ranked table "
                            "are effectively tied; a 2-unit gap carries no practical weight. On the "
                            "candidate table only 3 of 507 rows lie within 10 IC units of the best "
                            "value (43.9), so a 2-unit gap is well inside the family's fine "
                            "structure.")},
    "delta_10": {"pct_of_best": float(100 * 10 / best),
                 "n_ranked_rows_within": int(np.sum(rank.ic <= rank_min + 10)),
                 "meaning": ("10 IC units is 23% of the best value - 4 of the 250 ranked rows are "
                             "within 10 units of the best row, all with vmax in {20,30}, i.e. "
                             "still part of the same broad optimum.")},
    "delta_100": {"pct_of_best": float(100 * 100 / best), "multiple_of_best": float(100 / best),
                  "pct_of_candidate_iqr": float(100 * 100 / sub["candidate_ic_iqr_width"]),
                  "meaning": ("100 IC units is 2.3x the best value and 2.9% of the candidate "
                              "family's IQR width (3441.3 IC units). It is comparable to the whole "
                              "spread of the top-250 ranked table (0-137.2, so 100 units reaches "
                              "rank 162 of 250) - decisively worse than the optimum at the level "
                              "of the whole family, but smaller than the family's overall scatter "
                              "because a third of the family has IC > 4000.")},
}
sub["scale_interpretation"] = (
    "IC here is an absolute scale with values 0-13553 across the 507 candidate rows; the best "
    "value is 43.9 (candidate table) / 0.0 (ranked grid, arbitrarily zeroed at its own optimum - "
    "Q4 shows this is the same absolute scale). A difference of 2 is negligible (4.6% of the best, "
    "the top-2 ranked rows are practically tied); 10 is a mild, still-in-family difference (23% of "
    "best, 4 ranked rows within it); 100 is decisive against a candidate (2.3x the best, larger "
    "than the entire span of the top-250 ranked set). The family-level margins found here (6.8 "
    "between families; 1522.1 between the preferred candidate and the nested uniform slice) sit at "
    "the marginal and large ends respectively.")
sub["top_rank_gaps"] = {
    "rank1": rank_min, "rank2": float(rank.sort_values("candidate_rank").ic.iloc[1]),
    "rank3": float(rank.sort_values("candidate_rank").ic.iloc[2]),
    "rank10": float(rank.sort_values("candidate_rank").ic.iloc[9]),
    "rank30": float(rank.sort_values("candidate_rank").ic.iloc[29]),
    "rank250": float(rank.sort_values("candidate_rank").ic.iloc[-1])}
sub["n_ranked_rows_within_2_of_best"] = int(np.sum(rank.ic <= rank_min + 2))
sub["n_ranked_rows_within_10_of_best"] = int(np.sum(rank.ic <= rank_min + 10))
sub["n_ranked_rows_within_100_of_best"] = int(np.sum(rank.ic <= rank_min + 100))
sub["n_candidate_rows_within_10_of_best"] = int(np.sum(cand.ic <= cand_min + 10))
sub["n_candidate_rows_within_10pct_of_best"] = int(np.sum(cand.ic <= 1.10 * cand_min))
sub["n_candidate_rows_better_than_500"] = int(np.sum(cand.ic < 500))
sub["winning_margin_verdict"] = (
    f"Preferred family margin: frequency_amplitude_scan {best:.1f} vs minimum_velocity_scan "
    f"{float(MV.ic.min()):.1f} = Delta 6.8 IC units (15.5% of the best value, ratio 1.155) - a "
    f"MARGINAL margin, smaller than a single grid step in the ranked table (rank1 -> rank3 = 5.7). "
    f"Margin over the uniform slice: 1522.1 vs {best:.1f} = Delta 1478.2, ratio 34.7 - a LARGE "
    f"margin, but it is a NESTED-SUBSET comparison (Q1), not an independent model comparison. "
    f"Within the ranked top-250, only {int(np.sum(rank.ic <= rank_min + 2))} rows lie within 2 IC "
    f"units of the best and {int(np.sum(rank.ic <= rank_min + 10))} within 10, so the top few "
    f"candidates are effectively indistinguishable on IC. Scale reading on this absolute IC axis: "
    f"Delta ~2 = negligible (rank1/rank2 tied), Delta ~10 = mild/in-family (23% of best), "
    f"Delta ~100 = decisive against a candidate (2.3x best). Hence the family-level preference "
    f"between frequency_amplitude_scan and minimum_velocity_scan (Delta 6.8) is MARGINAL, while "
    f"the preference of either over the uniform slice is LARGE but circular.")

R["summary"] = {
    "uniform_is_diagonal_of_minimum_velocity_scan": True,
    "uniform_matching_rows": R["q1_uniform_is_mv_diagonal"]["n_matching_rows"],
    "uniform_max_abs_diff": R["q1_uniform_is_mv_diagonal"]["max_abs_difference"],
    "uniform_min_ic": uni_min, "uniform_argmin_velocity_cm_s": float(uni_argmin.v),
    "uniform_curve_unimodal": R["q2_uniform_curve_log_velocity"]["unimodal"],
    "uniform_local_curvature_ic_per_dex2": R["q2_uniform_curve_log_velocity"]["quadratic_fit_local_3pt"]["curvature_A_ic_per_dex2"],
    "candidate_min_ic": cand_min, "minimum_velocity_scan_min_ic": float(MV.ic.min()),
    "frequency_amplitude_scan_min_ic": float(FA.ic.min()),
    "ranked_full_grid_min_delta_ic": rank_min,
    "delta_ic_offset_max_abs": R["q4_delta_ic_offset_check"]["max_abs_offset_rounded_key"],
    "overlap_rows": R["q4_delta_ic_offset_check"]["overlap_rows_key_rounded_1e-6"],
    "preferred_family_within_candidate_set": "frequency_amplitude_scan",
    "preferred_velocity_scale_cm_s": {"frequency_amplitude_scan_vmax": float(fa_min_row.vmax),
                                      "minimum_velocity_scan_vmax": float(mv_min_row.vmax),
                                      "uniform_velocity": float(uni_argmin.v)},
    "preferred_velocity_scale_stable": True,
    "preferred_peak_frequency_microhz": 1010.0,
    "candidate_table_misses_family_optimum": (
        "The ranked full-parameter grid reaches absolute IC 0.0 at (vmax=30, vmin=0.05, f=1010, "
        "w=275), i.e. 43.9 IC units better than the best row of candidate_spectrum_comparison.tsv. "
        "This is the same absolute scale (Q4: offset ~1e-14), so the candidate table does not "
        "contain the overall IC optimum of the evaluated family."),
    "winning_margins": ("marginal between the two candidate families (Delta 6.8 IC, ratio 1.155); "
                        "large vs the nested uniform slice (Delta 1478.2, ratio 34.7) but that "
                        "comparison is a nested subset of the same family, not an independent "
                        "alternative"),
}

write_json(OUT_JSON, R)
print("wrote", OUT_JSON)
print("Q1 matching rows:", R["q1_uniform_is_mv_diagonal"]["n_matching_rows"], "/",
      R["q1_uniform_is_mv_diagonal"]["n_uniform_rows"],
      "max|diff| =", R["q1_uniform_is_mv_diagonal"]["max_abs_difference"])
print("Q2 uniform min:", uni_min, "at v =", float(uni_argmin.v), "unimodal:", R["q2_uniform_curve_log_velocity"]["unimodal"])
print("Q3 ratio uniform/candidate:", uni_min / cand_min)
print("Q4 overlap:", R["q4_delta_ic_offset_check"]["overlap_rows_key_rounded_1e-6"],
      "max offset:", R["q4_delta_ic_offset_check"]["max_abs_offset_rounded_key"])
print("Q5 bests: FA", float(FA.ic.min()), "MV", float(MV.ic.min()), "uniform", uni_min,
      "ranked", rank_min)