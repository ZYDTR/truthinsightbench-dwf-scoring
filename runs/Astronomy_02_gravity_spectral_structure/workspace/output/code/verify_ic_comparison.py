#!/usr/bin/env python3
"""Adversarial verification of TRACK 4 (ic_comparison).

Independent recomputation from the raw TSVs. Does NOT import track4_ic_comparison.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
DATA = PROJ / "data"
RES = PROJ / "output" / "results"
CODE = PROJ / "output" / "code"

unif = pd.read_csv(DATA / "uniform_alternative_comparison.tsv", sep="\t")
cand = pd.read_csv(DATA / "candidate_spectrum_comparison.tsv", sep="\t")
rank = pd.read_csv(DATA / "ranked_full_parameter_grid.tsv", sep="\t")

out: dict = {"track": "ic_comparison", "notes": []}
recomp: list[dict] = []


def add(quantity, claimed, recomputed, note=""):
    try:
        d = abs(float(claimed) - float(recomputed))
    except (TypeError, ValueError):
        d = None
    recomp.append({"quantity": quantity, "claimed": claimed,
                   "recomputed": recomputed, "abs_diff": d, "note": note})


u_v = unif["uniform_velocity_cm_s"].to_numpy(float)
u_ic = unif["information_criterion"].to_numpy(float)
out["uniform_rows"] = int(len(unif))
out["uniform_v"] = u_v.tolist()
out["uniform_ic"] = u_ic.tolist()

# ---------- Q1: uniform == diagonal of minimum_velocity_scan ----------
mv = cand[cand["candidate_family"] == "minimum_velocity_scan"].copy()
fa = cand[cand["candidate_family"] == "frequency_amplitude_scan"].copy()
out["n_mv_rows"] = int(len(mv))
out["n_fa_rows"] = int(len(fa))
out["mv_vmin_values"] = sorted(mv["min_velocity_cm_s"].unique().tolist())
out["mv_vmax_values"] = sorted(mv["max_velocity_cm_s"].unique().tolist())
out["mv_freq_values"] = sorted(mv["peak_frequency_microhz"].unique().tolist())
out["mv_width_values"] = sorted(mv["peak_width_microhz"].unique().tolist())
out["fa_vmin_values"] = sorted(fa["min_velocity_cm_s"].unique().tolist())
out["fa_vmax_values"] = sorted(fa["max_velocity_cm_s"].unique().tolist())

# diagonal rows in MV: vmax == vmin
diag = mv[np.isclose(mv["max_velocity_cm_s"], mv["min_velocity_cm_s"], atol=0, rtol=0)]
out["n_mv_exact_diagonal_rows"] = int(len(diag))
out["mv_exact_diagonal_velocities"] = sorted(diag["min_velocity_cm_s"].unique().tolist())

# match uniform velocities to diagonal rows: check ALL candidate rows (any family) too
rows_match = []
for v, ic in zip(u_v, u_ic):
    sel = cand[np.isclose(cand["min_velocity_cm_s"], v, atol=0, rtol=0)
               & np.isclose(cand["max_velocity_cm_s"], v, atol=0, rtol=0)]
    if len(sel) == 0:
        # tolerance match
        sel = cand[np.isclose(cand["min_velocity_cm_s"], v, rtol=1e-9, atol=1e-12)
                   & np.isclose(cand["max_velocity_cm_s"], v, rtol=1e-9, atol=1e-12)]
    if len(sel):
        for _, r in sel.iterrows():
            rows_match.append({
                "uniform_v": float(v), "uniform_ic": float(ic),
                "cand_family": r["candidate_family"],
                "cand_f": float(r["peak_frequency_microhz"]),
                "cand_w": float(r["peak_width_microhz"]),
                "cand_ic": float(r["information_criterion"]),
                "abs_diff": abs(float(ic) - float(r["information_criterion"])),
            })
matches_df = pd.DataFrame(rows_match)
out["q1_n_uniform_rows_with_diag_match"] = int(matches_df["uniform_v"].nunique()) if len(matches_df) else 0
out["q1_n_match_rows"] = int(len(matches_df))
out["q1_max_abs_diff"] = float(matches_df["abs_diff"].max()) if len(matches_df) else None
out["q1_all_matches_family"] = sorted(matches_df["cand_family"].unique().tolist()) if len(matches_df) else []
out["q1_all_matches_f"] = sorted(matches_df["cand_f"].unique().tolist()) if len(matches_df) else []
out["q1_all_matches_w"] = sorted(matches_df["cand_w"].unique().tolist()) if len(matches_df) else []
out["q1_match_velocities"] = sorted(matches_df["uniform_v"].unique().tolist()) if len(matches_df) else []
unmatched_v = sorted(set(u_v.tolist()) - set(out["q1_match_velocities"]))
out["q1_unmatched_velocities"] = unmatched_v
out["q1_unmatched_ics"] = [float(u_ic[np.isclose(u_v, v, atol=0, rtol=0)][0]) for v in unmatched_v]
# do unmatched IC values appear anywhere in candidate table?
cand_ic = set(np.round(cand["information_criterion"].to_numpy(float), 6).tolist())
out["q1_unmatched_ics_present_in_candidate"] = [bool(round(ic, 6) in cand_ic) for ic in out["q1_unmatched_ics"]]

add("Q1 matching uniform rows (diagonal)", 19, out["q1_n_uniform_rows_with_diag_match"])
add("Q1 max abs difference", 0.0, out["q1_max_abs_diff"])

# ---------- Q2: uniform curve on log axis ----------
order = np.argsort(u_v)
lv = np.log10(u_v[order])
ic = u_ic[order]
imin = int(np.argmin(ic))
out["q2_min_velocity"] = float(u_v[order][imin])
out["q2_min_ic"] = float(ic[imin])
out["q2_min_index"] = imin
d = np.diff(ic)
out["q2_n_negative_steps"] = int((d < 0).sum())
out["q2_n_positive_steps"] = int((d > 0).sum())
out["q2_n_zero_steps"] = int((d == 0).sum())
out["q2_neg_steps_before_min"] = int((d[:imin] < 0).sum())
out["q2_pos_steps_after_min"] = int((d[imin:] > 0).sum())
out["q2_unimodal_strict"] = bool((d[:imin] < 0).all() and (d[imin:] > 0).all())
slopes = d / np.diff(lv)
out["q2_slope_min"] = float(slopes.min())
out["q2_slope_max"] = float(slopes.max())
ds = np.diff(slopes)
out["q2_n_slope_inversions"] = int((ds < 0).sum())
out["q2_n_slope_inversions_above_1"] = int((ds[slopes[1:] > 0] < 0).sum())  # rough
# local quadratic around the minimum on log10 axis, using 3 pts (imin-1,imin,imin+1)
if 0 < imin < len(ic) - 1:
    xs = lv[imin - 1: imin + 2]
    ys = ic[imin - 1: imin + 2]
    A = np.polyfit(xs, ys, 2)
    r2_3 = float(1 - np.sum((ys - np.polyval(A, xs)) ** 2) / np.sum((ys - ys.mean()) ** 2))
    vx = -A[1] / (2 * A[0])
    out["q2_local3_A"] = float(A[0])
    out["q2_local3_vertex_log10"] = float(vx)
    out["q2_local3_vertex_v"] = float(10 ** vx)
    out["q2_local3_vertex_ic"] = float(np.polyval(A, vx))
    out["q2_local3_r2"] = r2_3
    out["q2_local3_coeffs"] = A.tolist()
    out["q2_local3_window_v"] = [float(10 ** xs[0]), float(10 ** xs[-1])]
# full range quadratic
Af = np.polyfit(lv, ic, 2)
out["q2_full_A"] = float(Af[0])
out["q2_full_vertex_v"] = float(10 ** (-Af[1] / (2 * Af[0])))
out["q2_full_r2"] = float(1 - np.sum((ic - np.polyval(Af, lv)) ** 2) / np.sum((ic - ic.mean()) ** 2))
# finite difference curvature at the min
if 0 < imin < len(ic) - 1:
    h1 = lv[imin] - lv[imin - 1]
    h2 = lv[imin + 1] - lv[imin]
    out["q2_fd_curvature"] = float(2 * ((ic[imin + 1] - ic[imin]) / h2 - (ic[imin] - ic[imin - 1]) / h1) / (h1 + h2))
out["q2_neighbours"] = {str(float(u_v[order][i])): float(ic[i]) for i in (imin - 1, imin, imin + 1)}

add("Q2 argmin velocity (uniform)", 0.1, out["q2_min_velocity"])
add("Q2 min IC (uniform)", 1522.1, out["q2_min_ic"])
add("Q2 local 3pt curvature A", 3047.372210798297, out["q2_local3_A"])
add("Q2 local 3pt vertex velocity", 0.08592725781343306, out["q2_local3_vertex_v"])
add("Q2 full-range curvature A", 830.429221034377, out["q2_full_A"])
add("Q2 fd curvature", 6094.744421596573, out["q2_fd_curvature"])

# ---------- Q3: candidate family minimum ----------
out["q3_cand_min_ic"] = float(cand["information_criterion"].min())
best = cand.loc[cand["information_criterion"].idxmin()]
out["q3_cand_argmin"] = {k: (float(best[k]) if k != "candidate_family" else best[k])
                         for k in ["candidate_family", "max_velocity_cm_s", "min_velocity_cm_s",
                                   "peak_frequency_microhz", "peak_width_microhz"]}
out["q3_ic_diff"] = float(out["q2_min_ic"] - out["q3_cand_min_ic"])
out["q3_ic_ratio"] = float(out["q2_min_ic"] / out["q3_cand_min_ic"])
out["q3_n_cand_rows_better_than_uniform_min"] = int((cand["information_criterion"] < out["q2_min_ic"]).sum())
out["q3_frac_cand_better"] = float((cand["information_criterion"] < out["q2_min_ic"]).mean())
out["q3_n_ranked_better"] = int((rank["delta_information_criterion"] < out["q2_min_ic"]).sum())
add("Q3 candidate min IC", 43.9, out["q3_cand_min_ic"])
add("Q3 IC difference uniform-minus-candidate", 1478.1999999999998, out["q3_ic_diff"])
add("Q3 IC ratio", 34.67198177676538, out["q3_ic_ratio"])
add("Q3 candidate rows better than uniform min", 232, out["q3_n_cand_rows_better_than_uniform_min"])

# ---------- Q4: delta_ic == absolute IC ----------
rk = rank.rename(columns={"frequency_width_microhz": "peak_width_microhz",
                          "delta_information_criterion": "delta_information_criterion",
                          "candidate_rank": "candidate_rank"})
out["q4_column_names_candidate"] = cand.columns.tolist()
out["q4_column_names_ranked"] = rank.columns.tolist()
merge_exact = cand.merge(rk, on=["max_velocity_cm_s", "min_velocity_cm_s",
                                 "peak_frequency_microhz", "peak_width_microhz"],
                         how="inner", suffixes=("_c", "_r"))
out["q4_overlap_exact"] = int(len(merge_exact))
if len(merge_exact):
    off = np.abs(merge_exact["information_criterion"] - merge_exact["delta_information_criterion"])
    out["q4_max_abs_offset_exact_merge"] = float(off.max())
    out["q4_n_off_gt_1e-9"] = int((off > 1e-9).sum())
# rounded-key join
cand_r = cand.round({"max_velocity_cm_s": 6, "min_velocity_cm_s": 6,
                     "peak_frequency_microhz": 6, "peak_width_microhz": 6})
rank_r = rk.round({"max_velocity_cm_s": 6, "min_velocity_cm_s": 6,
                   "peak_frequency_microhz": 6, "peak_width_microhz": 6})
merge_round = cand_r.merge(rank_r, on=["max_velocity_cm_s", "min_velocity_cm_s",
                                       "peak_frequency_microhz", "peak_width_microhz"],
                           how="inner", suffixes=("_c", "_r"))
out["q4_overlap_rounded"] = int(len(merge_round))
offr = np.abs(merge_round["information_criterion"] - merge_round["delta_information_criterion"])
out["q4_max_abs_offset_rounded"] = float(offr.max())
out["q4_n_off_gt_1e-9_rounded"] = int((offr > 1e-9).sum())
add("Q4 overlap rows (exact key)", 26, out["q4_overlap_exact"])
add("Q4 overlap rows (rounded key)", 29, out["q4_overlap_rounded"])
add("Q4 max abs offset", 1.4210854715202004e-14, out["q4_max_abs_offset_rounded"])

# ---------- Q4b: is the ranked file a COMPLETE cartesian grid? ----------
rk_vmax = sorted(rk["max_velocity_cm_s"].round(6).unique().tolist())
rk_vmin = sorted(rk["min_velocity_cm_s"].round(6).unique().tolist())
rk_f = sorted(rk["peak_frequency_microhz"].round(6).unique().tolist())
rk_w = sorted(rk["peak_width_microhz"].round(6).unique().tolist())
out["q4b_rank_axes"] = {"vmax": rk_vmax, "vmin": rk_vmin, "f": rk_f, "w": rk_w}
out["q4b_cartesian_size"] = len(rk_vmax) * len(rk_vmin) * len(rk_f) * len(rk_w)
out["q4b_actual_rows"] = int(len(rk))
out["q4b_is_complete_cartesian"] = bool(out["q4b_cartesian_size"] == len(rk))
# coverage per (f,w) etc
out["q4b_n_vmax_vmin_pairs"] = int(len(rk[["max_velocity_cm_s", "min_velocity_cm_s"]].drop_duplicates()))
out["q4b_n_fw_pairs"] = int(len(rk[["peak_frequency_microhz", "peak_width_microhz"]].drop_duplicates()))
# duplicate keys?
out["q4b_n_duplicate_param_rows"] = int(len(rk) - len(rk.drop_duplicates(
    subset=["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "peak_width_microhz"])))
# per-vmax counts
out["q4b_rows_per_vmax"] = {str(k): int(v) for k, v in rk.groupby("max_velocity_cm_s").size().items()}
out["q4b_rows_per_f"] = {str(k): int(v) for k, v in rk.groupby("peak_frequency_microhz").size().items()}
out["q4b_rows_per_w"] = {str(k): int(v) for k, v in rk.groupby("peak_width_microhz").size().items()}
out["q4b_ranked_has_diagonal"] = int(len(rk[np.isclose(rk["max_velocity_cm_s"], rk["min_velocity_cm_s"])]))

# ---------- Q5: best per family ----------
out["q5_fa_best"] = float(fa["information_criterion"].min())
out["q5_fa_n"] = int(len(fa))
fb = fa.loc[fa["information_criterion"].idxmin()]
out["q5_fa_argmin"] = {k: float(fb[k]) for k in ["max_velocity_cm_s", "min_velocity_cm_s",
                                                 "peak_frequency_microhz", "peak_width_microhz"]}
out["q5_mv_best"] = float(mv["information_criterion"].min())
out["q5_mv_n"] = int(len(mv))
mb = mv.loc[mv["information_criterion"].idxmin()]
out["q5_mv_argmin"] = {k: float(mb[k]) for k in ["max_velocity_cm_s", "min_velocity_cm_s",
                                                 "peak_frequency_microhz", "peak_width_microhz"]}
out["q5_rank_best"] = float(rk["delta_information_criterion"].min())
rb = rk.loc[rk["delta_information_criterion"].idxmin()]
out["q5_rank_argmin"] = {k: float(rb[k]) for k in ["max_velocity_cm_s", "min_velocity_cm_s",
                                                   "peak_frequency_microhz", "peak_width_microhz"]}
out["q5_margin_fa_mv_diff"] = float(out["q5_mv_best"] - out["q5_fa_best"])
out["q5_margin_fa_mv_ratio"] = float(out["q5_mv_best"] / out["q5_fa_best"])
out["q5_margin_uniform_best"] = float(out["q2_min_ic"] - out["q3_cand_min_ic"])
add("Q5 FA best IC", 43.9, out["q5_fa_best"])
add("Q5 MV best IC", 50.7, out["q5_mv_best"])
add("Q5 ranked best delta_ic", 0.0, out["q5_rank_best"])
add("Q5 FA-over-MV margin", 6.800000000000004, out["q5_margin_fa_mv_diff"])
add("Q5 FA-over-MV ratio", 1.1548974943052392, out["q5_margin_fa_mv_ratio"])

# ---------- Q6: boundary/sensitivity ----------
a = cand[cand["min_velocity_cm_s"] > 0]
out["q6a_cand_n"] = int(len(a))
out["q6a_cand_best"] = float(a["information_criterion"].min()) if len(a) else None
out["q6a_cand_best_family"] = a.loc[a["information_criterion"].idxmin(), "candidate_family"] if len(a) else None
out["q6a_uniform_best"] = float(u_ic[u_v > 0].min()) if (u_v > 0).any() else None
out["q6a_uniform_argmin_v"] = float(u_v[u_v > 0][np.argmin(u_ic[u_v > 0])]) if (u_v > 0).any() else None
add("Q6a candidate best (vmin>0)", 50.7, out["q6a_cand_best"])
add("Q6a uniform best (v>0)", 1522.1, out["q6a_uniform_best"])

# (b) rows present in top-250 ranked table (rounded-key join)
out["q6b_ranked_keys_columns"] = rank_r.columns.tolist()
sub = cand_r.merge(rank_r[["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz",
                           "peak_width_microhz"]].drop_duplicates(),
    on=["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "peak_width_microhz"],
    how="inner")
out["q6b_cand_subset_n"] = int(len(sub))
out["q6b_cand_best"] = float(sub["information_criterion"].min()) if len(sub) else None
out["q6b_uniform_n"] = int(len(unif.merge(
    rank_r.rename(columns={"max_velocity_cm_s": "uniform_velocity_cm_s"}),
    on="uniform_velocity_cm_s", how="inner")))
out["q6b_uniform_argmin_still"] = float(u_v[np.argmin(u_ic)])
out["q6b_family_counts"] = {str(k): int(v) for k, v in sub.groupby("candidate_family").size().items()}
add("Q6b candidate subset n", 29, out["q6b_cand_subset_n"])
add("Q6b candidate best in subset", 43.9, out["q6b_cand_best"])

# (c) best row per peak frequency
per_f = cand.groupby("peak_frequency_microhz")["information_criterion"].min()
out["q6c_n_freq_cand"] = int(len(per_f))
out["q6c_cand_best"] = float(per_f.min())
out["q6c_cand_best_f"] = float(per_f.idxmin())
per_f_rank = rk.groupby("peak_frequency_microhz")["delta_information_criterion"].min()
out["q6c_n_freq_rank"] = int(len(per_f_rank))
out["q6c_rank_best"] = float(per_f_rank.min())
out["q6c_rank_best_f"] = float(per_f_rank.idxmin())
out["q6c_uniform_best"] = float(u_ic.min())
add("Q6c candidate best per-f", 43.9, out["q6c_cand_best"])
add("Q6c ranked f count", 10, out["q6c_n_freq_rank"])

# leave-one-out on uniform
loo = {}
for v in u_v:
    m = ~np.isclose(u_v, v, atol=0, rtol=0)
    loo[str(float(v))] = {"argmin_v": float(u_v[m][np.argmin(u_ic[m])]), "min_ic": float(u_ic[m].min())}
out["q6_loo_uniform"] = loo
argmin_set = sorted({d["argmin_v"] for d in loo.values()} | {float(u_v[np.argmin(u_ic)])})
out["q6_loo_uniform_argmin_set"] = argmin_set
add("Q6 uniform LOO argmin set", [0.05, 0.1], argmin_set)

# leave-one-out on candidate family by peak frequency
loo_f = {}
for f0 in sorted(cand["peak_frequency_microhz"].unique()):
    m = cand["peak_frequency_microhz"] != f0
    sub2 = cand[m]
    b = sub2.loc[sub2["information_criterion"].idxmin()]
    loo_f[str(float(f0))] = {"new_best_ic": float(b["information_criterion"]),
                             "new_best_f": float(b["peak_frequency_microhz"]),
                             "family": b["candidate_family"]}
out["q6_loo_candidate_frequency"] = loo_f
add("Q6 LOO drop f=1010 -> best IC", 50.7, loo_f.get("1010.0", {}).get("new_best_ic"))
add("Q6 LOO drop f=1010 -> best f", 1210.0, loo_f.get("1010.0", {}).get("new_best_f"))

# leave-one-out by vmax within FA and MV
def loo_vmax(df):
    res = {}
    for v0 in sorted(df["max_velocity_cm_s"].unique()):
        m = df["max_velocity_cm_s"] != v0
        s = df[m]
        if len(s) == 0:
            continue
        b = s.loc[s["information_criterion"].idxmin()]
        res[str(float(v0))] = {"argmin_vmax": float(b["max_velocity_cm_s"]),
                               "min_ic": float(b["information_criterion"])}
    return res


lv_fa = loo_vmax(fa)
lv_mv = loo_vmax(mv)
out["q6_loo_fa_vmax"] = lv_fa
out["q6_loo_mv_vmax"] = lv_mv
out["q6_loo_fa_vmax_argmin_set"] = sorted({d["argmin_vmax"] for d in lv_fa.values()} | {out["q5_fa_argmin"]["max_velocity_cm_s"]})
out["q6_loo_mv_vmax_argmin_set"] = sorted({d["argmin_vmax"] for d in lv_mv.values()} | {out["q5_mv_argmin"]["max_velocity_cm_s"]})
add("Q6 FA LOO vmax argmin set", [20.0, 30.0], out["q6_loo_fa_vmax_argmin_set"])
add("Q6 MV LOO vmax argmin set", [30.0, 50.0], out["q6_loo_mv_vmax_argmin_set"])

# ---------- Q7: distribution / scale ----------
ic_all = cand["information_criterion"].to_numpy(float)
out["q7_n"] = int(len(ic_all))
out["q7_mean"] = float(ic_all.mean())
out["q7_sd"] = float(ic_all.std(ddof=1))
out["q7_median"] = float(np.median(ic_all))
out["q7_min"] = float(ic_all.min())
out["q7_max"] = float(ic_all.max())
out["q7_q1"] = float(np.percentile(ic_all, 25))
out["q7_q3"] = float(np.percentile(ic_all, 75))
out["q7_iqr"] = float(out["q7_q3"] - out["q7_q1"])
add("Q7 candidate mean", 2583.8761341222885, out["q7_mean"])
add("Q7 candidate sd", 2577.6645235428273, out["q7_sd"])
add("Q7 candidate median", 1682.3, out["q7_median"])
add("Q7 candidate IQR width", 3441.3, out["q7_iqr"])

r_ic = rk["delta_information_criterion"].to_numpy(float)
r_sorted = np.sort(r_ic)
out["q7_rank_within_2"] = int((r_ic <= 2).sum())
out["q7_rank_within_10"] = int((r_ic <= 10).sum())
out["q7_rank_within_100"] = int((r_ic <= 100).sum())
out["q7_top_gaps"] = {f"rank{i+1}": float(r_sorted[i]) for i in range(min(5, len(r_sorted)))}
out["q7_rank10"] = float(r_sorted[9])
out["q7_rank30"] = float(r_sorted[29])
out["q7_rank250"] = float(r_sorted[-1])
out["q7_n_cand_within_10_of_best"] = int((ic_all <= ic_all.min() + 10).sum())
out["q7_n_cand_better_than_500"] = int((ic_all < 500).sum())
add("Q7 ranked within 10", 4, out["q7_rank_within_10"])
add("Q7 ranked within 100", 162, out["q7_rank_within_100"])
add("Q7 ranked rank3", 5.7, out["q7_top_gaps"]["rank3"])
add("Q7 ranked rank30", 38.4, out["q7_rank30"])
add("Q7 ranked max (rank250)", 137.2, out["q7_rank250"])

# CONSISTENCY CHECKS ------------------------------------------------------
problems = []
# 1. Does the uniform table have a frequency axis? Check whether reported q3 attribution is consistent.
# 2. Does the candidate table contain the ranked optimum? (it should not, per report)
opt = (5.0, 0.0)  # sanity
# 3. Check whether distance-based similarity better than chance
out["chance_null"] = {}
# Build a null: compare uniform IC values against candidate IC values under a permuted velocity mapping
rng = np.random.default_rng(12345)
# candidate values at f=1210, w=300 only
c1210 = cand[(cand["peak_frequency_microhz"] == 1210.0) & (cand["peak_width_microhz"] == 300.0)]
out["n_cand_f1210_w300"] = int(len(c1210))
mv1210 = c1210[c1210["candidate_family"] == "minimum_velocity_scan"]
out["n_mv_f1210_w300"] = int(len(mv1210))
mv1210d = mv1210[np.isclose(mv1210["max_velocity_cm_s"], mv1210["min_velocity_cm_s"])]
out["n_mv_f1210_w300_diag"] = int(len(mv1210d))

# ---------- Claim verification: which statements in the report are wrong ----------
# (i) Report claims "candidate_family_diagonal_min_velocity = 0.02". Check.
out["claim_check_diag_min_velocity"] = out["mv_exact_diagonal_velocities"][:3] if out["mv_exact_diagonal_velocities"] else None
# (ii) Report says 'all 304 FA rows have vmin = 0.0 exactly' -> check
out["fa_vmin_unique"] = sorted(fa["min_velocity_cm_s"].unique().tolist())
# (iii) Report Q6b says ranked has 'vmax axis starts at 1 cm/s and vmin stops at 0.1'; check
out["rank_vmax_min"] = float(rk["max_velocity_cm_s"].min())
out["rank_vmin_max"] = float(rk["min_velocity_cm_s"].max())
# (iv) Does report's q6b claim "29 rows matched" agree with my (b)?
# (v) FM check: the report has duplicate mismatched metadata for q1 in self-report vs json
# (vi) key numeric: check that candidate min row f=1010, w=300 exists
best_row = cand.loc[cand["information_criterion"].idxmin()]
out["cand_best_row"] = {k: float(best_row[k]) for k in ["max_velocity_cm_s", "min_velocity_cm_s",
                                                        "peak_frequency_microhz", "peak_width_microhz",
                                                        "information_criterion"]}
# (vii) IS the uniform table literally a slice of MV *with the exact same f,w*? yes if matches all at f=1210,w=300.
# but check covariance: is every uniform IC exactly matched to a *diagonal* (vmax==vmin) row?
diag_check = []
for v, icv in zip(u_v, u_ic):
    sel = mv[(mv["min_velocity_cm_s"] == v) & (mv["max_velocity_cm_s"] == v)]
    diag_check.append(bool(len(sel) and np.isclose(sel["information_criterion"].iloc[0], icv)))
out["q1_all_matches_are_exact_diagonal"] = all(diag_check)  # False if any of 24 fails
out["q1_n_diag_exact_of_24"] = int(sum(diag_check))
out["q1_diag_exact_flags"] = diag_check

# (viii) alternative hypothesis test: could the uniform table simply be a DIFFERENT slice (e.g. vmax fixed)?
# check whether uniform IC values match any other single row of the candidate table at the same IC
val_match_rows = []
for v, icv in zip(u_v, u_ic):
    sel = cand[np.isclose(cand["information_criterion"], icv, atol=1e-9)]
    val_match_rows.append({"uniform_v": float(v), "n_cand_rows_with_same_ic": int(len(sel)),
                           "families": sorted(sel["candidate_family"].unique().tolist()),
                           "vmin": sorted(sel["min_velocity_cm_s"].unique().tolist()),
                           "vmax": sorted(sel["max_velocity_cm_s"].unique().tolist())})
out["q1_value_matches"] = val_match_rows

# Count how many candidate rows share each uniform IC value (as a chance baseline)
all_ics = np.round(cand["information_criterion"].to_numpy(float), 6)
uniq = np.unique(all_ics)
out["q1_chance_note"] = {
    "n_candidate_rows": int(len(cand)),
    "n_unique_ic_values": int(len(uniq)),
    "p_random_cand_value_equals_a_given_uniform_value": float(1.0 / len(uniq)),
    "expected_matches_if_random": float(24.0 * len(cand) / len(uniq)),
}

out["independent_recomputations"] = recomp
out["problems_found"] = problems
out["raw"] = {
    "q1_detail": matches_df.to_dict(orient="records") if len(matches_df) else [],
}

verif = {
    "track": "ic_comparison",
    "artifacts_exist": (RES / "ic_comparison.json").exists() and (CODE / "track4_ic_comparison.py").exists(),
    "independent_recomputations": recomp,
    "problems_found": problems,
}

with open(RES / "_verify_ic_working.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("raw", "q6_loo_uniform", "q6_loo_candidate_frequency",
                                                            "q6_loo_fa_vmax", "q6_loo_mv_vmax",
                                                            "q1_value_matches", "q2_neighbours",
                                                            "uniform_v", "uniform_ic")}, indent=2, default=str))