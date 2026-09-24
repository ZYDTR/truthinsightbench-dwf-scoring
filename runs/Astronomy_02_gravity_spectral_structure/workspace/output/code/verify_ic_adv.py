"""Adversarial verification of TRACK 4 (ic): independent recomputation.

Deliberately does NOT import or execute track4_ic_comparison.py for any number.
Reads only data/*.tsv (the sole direct evidence) and writes a scratch JSON of
independently recomputed quantities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
D = P / "data"

uni = pd.read_csv(D / "uniform_alternative_comparison.tsv", sep="\t").sort_values(
    "uniform_velocity_cm_s").reset_index(drop=True)
cand = pd.read_csv(D / "candidate_spectrum_comparison.tsv", sep="\t")
rank = pd.read_csv(D / "ranked_full_parameter_grid.tsv", sep="\t")

O = {}

# ---------------------------------------------------------------- A. structure
v = uni["uniform_velocity_cm_s"].to_numpy(float)
ic = uni["information_criterion"].to_numpy(float)
O["A_uniform_n"] = int(len(uni))
O["A_uniform_velocities"] = [float(x) for x in v]
O["A_uniform_ic"] = [float(x) for x in ic]

MV = cand[cand.candidate_family == "minimum_velocity_scan"]
FA = cand[cand.candidate_family == "frequency_amplitude_scan"]
O["A_n_MV"] = int(len(MV))
O["A_n_FA"] = int(len(FA))
O["A_MV_vmax_uniq"] = sorted(float(x) for x in MV.max_velocity_cm_s.unique())
O["A_MV_vmin_uniq"] = sorted(float(x) for x in MV.min_velocity_cm_s.unique())
O["A_MV_pairs"] = int(MV[["max_velocity_cm_s", "min_velocity_cm_s"]].drop_duplicates().shape[0])

# structure of the MV (vmax, vmin) support
pairs = MV[["max_velocity_cm_s", "min_velocity_cm_s"]].drop_duplicates()
O["A_MV_pairs_le"] = int(np.sum(pairs.max_velocity_cm_s.to_numpy() >= pairs.min_velocity_cm_s.to_numpy()))
O["A_MV_pairs_lt"] = int(np.sum(pairs.max_velocity_cm_s.to_numpy() > pairs.min_velocity_cm_s.to_numpy()))
full_tri = len(O["A_MV_vmax_uniq"]) * 0  # placeholder
ax_v = sorted(MV.max_velocity_cm_s.unique())
ax_m = sorted(MV.min_velocity_cm_s.unique())
tri = sum(1 for a in ax_v for b in ax_m if b <= a)
O["A_MV_full_vmin_le_vmax_count"] = int(tri)
# missing (vmax,vmin) with vmin<=vmax relative to a full triangle
have = set(zip(pairs.max_velocity_cm_s.to_numpy(), pairs.min_velocity_cm_s.to_numpy()))
missing = [(a, b) for a in ax_v for b in ax_m if b <= a and (a, b) not in have]
O["A_MV_missing_vmin_le_vmax"] = [[float(a), float(b)] for a, b in missing][:40]
O["A_MV_n_missing_vmin_le_vmax"] = len(missing)

# ------------------------------------------------------- B. Q1: diagonal match
diag = MV[np.isclose(MV.max_velocity_cm_s.to_numpy(float),
                     MV.min_velocity_cm_s.to_numpy(float), rtol=0, atol=0)]
diag_map = {float(r.min_velocity_cm_s): float(r.information_criterion) for _, r in diag.iterrows()}
matched, unmatched = [], []
for vv, ii in zip(v, ic):
    if vv in diag_map:
        matched.append((vv, ii, diag_map[vv], abs(ii - diag_map[vv])))
    else:
        unmatched.append((vv, ii))
d_m = np.array([m[3] for m in matched])
O["B_n_diag_rows_MV"] = int(len(diag))
O["B_n_matched"] = int(len(matched))
O["B_n_unmatched"] = int(len(unmatched))
O["B_frac_matched_of_uniform"] = float(len(matched) / len(v))
O["B_max_abs_diff_matched"] = float(d_m.max()) if len(d_m) else None
O["B_all_bit_identical"] = bool(np.all(d_m == 0.0))
O["B_unmatched"] = [[float(a), float(b)] for a, b in unmatched]
O["B_diag_f_uniq"] = sorted(float(x) for x in diag.peak_frequency_microhz.unique())
O["B_diag_w_uniq"] = sorted(float(x) for x in diag.peak_width_microhz.unique())
O["B_diag_min_v"] = float(diag.min_velocity_cm_s.min())
O["B_unmatched_below_diag_floor"] = bool(all(a < diag.min_velocity_cm_s.min() for a, _ in unmatched))
# do the unmatched IC values occur anywhere in cand?
cand_ic = cand.information_criterion.to_numpy(float)
O["B_unmatched_ic_present_in_cand"] = [bool(np.any(np.isclose(cand_ic, b, rtol=0, atol=0)))
                                       for _, b in unmatched]
# every diagonal row recovered in the uniform table (symmetric direction)
O["B_n_diag_recovered_in_uniform"] = int(sum(1 for k in diag_map if k in set(v)))
# candidate-family support vs uniform velocities
O["B_uniform_vel_are_MV_vmin_plus5"] = bool(
    set(float(x) for x in v) == set(float(x) for x in ax_m) | {0.001, 0.002, 0.003, 0.005})

# ------------------------------------------------- C. Q2: curve + quadratic fits
x = np.log10(v)
y = ic
imin = int(np.argmin(y))
O["C_argmin_v"] = float(v[imin])
O["C_argmin_ic"] = float(y[imin])
O["C_argmin_index"] = imin
O["C_is_interior"] = bool(0 < imin < len(y) - 1)
d1 = np.diff(y)
O["C_n_neg_steps"] = int(np.sum(d1 < 0))
O["C_n_pos_steps"] = int(np.sum(d1 > 0))
O["C_sign_changes_first_diff"] = int(np.sum(np.diff(np.sign(d1)) != 0))
O["C_unimodal"] = bool(int(np.sum(np.diff(np.sign(d1)) != 0)) == 1)
slopes = d1 / np.diff(x)
O["C_slopes"] = [float(s) for s in slopes]
O["C_slope_min"] = float(slopes.min())
O["C_slope_max"] = float(slopes.max())
O["C_n_slope_inversions"] = int(np.sum(np.diff(slopes) < 0))

def quad(xs, ys):
    c = np.polyfit(xs, ys, 2)
    pred = np.polyval(c, xs)
    r2 = float(1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2))
    xv = -c[1] / (2 * c[0])
    return {"A": float(c[0]), "B": float(c[1]), "C": float(c[2]),
            "xvertex": float(xv), "vvertex": float(10 ** xv),
            "vertex_ic": float(np.polyval(c, xv)), "r2": r2,
            "halfwidth2": float(np.sqrt(2.0 / c[0])), "halfwidth10": float(np.sqrt(10.0 / c[0])),
            "n": int(len(xs))}

O["C_quad_full"] = quad(x, y)
for nm, lo, hi in [("3pt", 0.05, 0.2), ("5pt", 0.02, 0.5), ("7pt", 0.01, 1.0)]:
    m = (v >= lo) & (v <= hi)
    O[f"C_quad_{nm}"] = quad(x[m], y[m])
O["C_curv_fd_3pt"] = float((y[imin + 1] - 2 * y[imin] + y[imin - 1]) / (x[imin + 1] - x[imin]) ** 2)
O["C_curv_fd_window"] = [float(v[imin - 1]), float(v[imin + 1])]
O["C_total_change"] = float(y[-1] - y[0])
O["C_low_branch"] = {"v": [float(v[0]), float(v[imin])], "ic": [float(y[0]), float(y[imin])],
                     "decades": float(x[imin] - x[0])}
O["C_high_branch"] = {"v": [float(v[imin]), float(v[-1])], "ic": [float(y[imin]), float(y[-1])],
                      "decades": float(x[-1] - x[imin])}
# neighbours of min
O["C_neighbours"] = [[float(v[j]), float(y[j])] for j in (imin - 1, imin, imin + 1)]

# ------------------------------------------------------- D. Q3/Q5: minima
cmin_i = int(np.argmin(cand_ic))
O["D_cand_min_ic"] = float(cand_ic[cmin_i])
O["D_cand_min_row"] = {k: (float(cand.iloc[cmin_i][k]) if cand.iloc[cmin_i][k] != "frequency_amplitude_scan"
                           else str(cand.iloc[cmin_i][k])) for k in cand.columns}
fa_ic = FA.information_criterion.to_numpy(float)
mv_ic = MV.information_criterion.to_numpy(float)
fa_i = int(np.argmin(fa_ic)); mv_i = int(np.argmin(mv_ic))
O["D_FA_min_ic"] = float(fa_ic[fa_i])
O["D_FA_min_row"] = {k: float(FA.iloc[fa_i][k]) if k != "candidate_family" else str(FA.iloc[fa_i][k])
                     for k in FA.columns}
O["D_MV_min_ic"] = float(mv_ic[mv_i])
O["D_MV_min_row"] = {k: float(MV.iloc[mv_i][k]) if k != "candidate_family" else str(MV.iloc[mv_i][k])
                     for k in MV.columns}
O["D_diag_min_ic"] = float(diag.information_criterion.min())
O["D_diag_min_v"] = float(diag.loc[diag.information_criterion.idxmin(), "min_velocity_cm_s"])
rank_ic = rank.delta_information_criterion.to_numpy(float)
O["D_rank_min_delta_ic"] = float(rank_ic.min())
O["D_rank_min_row"] = {k: float(rank.loc[rank_ic.argmin(), k]) for k in rank.columns}
O["D_delta_uniform_minus_cand"] = float(y[imin] - cand_ic[cmin_i])
O["D_ratio_uniform_over_cand"] = float(y[imin] / cand_ic[cmin_i])
O["D_n_cand_better_than_uniform_min"] = int(np.sum(cand_ic < y[imin]))
O["D_frac_cand_better"] = float(np.mean(cand_ic < y[imin]))
O["D_n_rank_better_than_uniform_min"] = int(np.sum(rank_ic < y[imin]))
O["D_margin_fa_mv"] = float(mv_ic[mv_i] - fa_ic[fa_i])
O["D_ratio_fa_mv"] = float(mv_ic[mv_i] / fa_ic[fa_i])

# ---------------------------------------------- E. Q4: delta_ic offset check
KC = ["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "peak_width_microhz"]
KR = ["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "frequency_width_microhz"]


def keys(df, cols, nd=None):
    a = df[cols].to_numpy(float)
    if nd is not None:
        a = np.round(a, nd)
    return [tuple(r) for r in a]


ck = keys(cand, KC); rk = keys(rank, KR)
ckr = keys(cand, KC, 6); rkr = keys(rank, KR, 6)
cand_ic_map = {k: float(i) for k, i in zip(ck, cand_ic)}
rank_ic_map = {k: float(i) for k, i in zip(rk, rank_ic)}
cand_ic_map_r = {k: float(i) for k, i in zip(ckr, cand_ic)}
rank_ic_map_r = {k: float(i) for k, i in zip(rkr, rank_ic)}
common_exact = set(cand_ic_map) & set(rank_ic_map)
common_round = set(cand_ic_map_r) & set(rank_ic_map_r)
off_exact = [abs(cand_ic_map[k] - rank_ic_map[k]) for k in common_exact]
off_round = [abs(cand_ic_map_r[k] - rank_ic_map_r[k]) for k in common_round]
O["E_overlap_exact"] = int(len(common_exact))
O["E_overlap_round6"] = int(len(common_round))
O["E_max_abs_off_exact"] = float(max(off_exact)) if off_exact else None
O["E_max_abs_off_round6"] = float(max(off_round)) if off_round else None
O["E_n_off_gt_1e9"] = int(np.sum(np.array(off_round) > 1e-9))
O["E_extra_recovered"] = int(len(common_round) - len(common_exact))
O["E_dup_keys_cand"] = int(len(ck) - len(set(ck)))
O["E_dup_keys_rank"] = int(len(rk) - len(set(rk)))
O["E_rank_vmin_0p03_repr"] = float(rank.min_velocity_cm_s.unique()[np.argmin(
    np.abs(rank.min_velocity_cm_s.unique() - 0.03))])
O["E_0p03_diff"] = float(abs(0.03 - O["E_rank_vmin_0p03_repr"]))
# offset sign: is cand_ic - rank_ic a constant 0?
O["E_min_signed_off"] = float(min(cand_ic_map_r[k] - rank_ic_map_r[k] for k in common_round))
O["E_max_signed_off"] = float(max(cand_ic_map_r[k] - rank_ic_map_r[k] for k in common_round))

# ------------------------------------ F. is the ranked table a full cartesian?
O["F_rank_n"] = int(len(rank))
O["F_rank_axes"] = {
    "vmax": sorted(float(x) for x in rank.max_velocity_cm_s.unique()),
    "vmin": sorted(float(x) for x in rank.min_velocity_cm_s.unique()),
    "f": sorted(float(x) for x in rank.peak_frequency_microhz.unique()),
    "w": sorted(float(x) for x in rank.frequency_width_microhz.unique())}
cart = (len(O["F_rank_axes"]["vmax"]) * len(O["F_rank_axes"]["vmin"]) *
        len(O["F_rank_axes"]["f"]) * len(O["F_rank_axes"]["w"]))
O["F_cartesian_size"] = int(cart)
O["F_fraction_of_cartesian"] = float(len(rank) / cart)
O["F_ranks_contiguous_1_to_250"] = bool(
    set(rank.candidate_rank.astype(int)) == set(range(1, 251)))
O["F_ic_sorted_by_rank"] = bool(np.all(np.diff(rank_ic[rank.candidate_rank.argsort()]) >= 0))
O["F_n_ranked_rows_with_vmax_eq_vmin"] = int(np.sum(
    rank.max_velocity_cm_s.to_numpy(float) == rank.min_velocity_cm_s.to_numpy(float)))
O["F_rank_vmin_max"] = float(rank.min_velocity_cm_s.max())
O["F_rank_vmax_min"] = float(rank.max_velocity_cm_s.min())
pv = rank.groupby(["max_velocity_cm_s", "min_velocity_cm_s"]).size().sort_index()
O["F_pairs_present"] = int(len(pv))
O["F_pairs_full"] = int(len(O["F_rank_axes"]["vmax"]) * len(O["F_rank_axes"]["vmin"]))
O["F_pair_count_span"] = [int(pv.min()), int(pv.max())]

# ------------------------------------------------------- G. summary-scale stats
O["G_cand_mean"] = float(np.mean(cand_ic))
O["G_cand_sd_ddof1"] = float(np.std(cand_ic, ddof=1))
O["G_cand_median"] = float(np.median(cand_ic))
O["G_cand_q25"] = float(np.quantile(cand_ic, 0.25))
O["G_cand_q75"] = float(np.quantile(cand_ic, 0.75))
O["G_cand_min"] = float(cand_ic.min()); O["G_cand_max"] = float(cand_ic.max())
O["G_n_within_10_of_best"] = int(np.sum(cand_ic <= cand_ic.min() + 10))
O["G_n_within_10pct_of_best"] = int(np.sum(cand_ic <= 1.1 * cand_ic.min()))
O["G_n_better_than_500"] = int(np.sum(cand_ic < 500))
O["G_n_rank_within_2"] = int(np.sum(rank_ic <= rank_ic.min() + 2))
O["G_n_rank_within_10"] = int(np.sum(rank_ic <= rank_ic.min() + 10))
O["G_n_rank_within_100"] = int(np.sum(rank_ic <= rank_ic.min() + 100))
O["G_top_rank_gaps"] = {f"rank{k}": float(sorted(rank_ic)[k - 1]) for k in (1, 2, 3, 10, 30, 250)}

# ------------------------------------------------------------- H. LOO checks
loo_uni = {}
for i in range(len(v)):
    y2 = np.delete(y, i); x2 = np.delete(x, i)
    loo_uni[float(v[i])] = (float(10 ** x2[int(np.argmin(y2))]), float(y2.min()))
O["H_loo_uni_argmins"] = {str(k): vv for k, vv in loo_uni.items()}
O["H_loo_uni_argmin_set"] = sorted({vv[0] for vv in loo_uni.values()})
fa_per_vmax = FA.groupby("max_velocity_cm_s").information_criterion.min()
O["H_fa_vmax_argmin"] = float(fa_per_vmax.idxmin())
mv_per_vmax = MV.groupby("max_velocity_cm_s").information_criterion.min()
O["H_mv_vmax_argmin"] = float(mv_per_vmax.idxmin())
pf = cand.groupby("peak_frequency_microhz").information_criterion.min()
O["H_per_f_min"] = {str(float(k)): float(xx) for k, xx in pf.items()}
O["H_per_f_argmin"] = float(pf.idxmin())
O["H_per_f_ranked"] = {str(float(k)): float(xx) for k, xx in
                       rank.groupby("peak_frequency_microhz").delta_information_criterion.min().items()}

# variant (a): vmin > 0
cp = cand[cand.min_velocity_cm_s > 0]
O["H_a_cand_pos_n"] = int(len(cp))
O["H_a_cand_pos_best"] = float(cp.information_criterion.min())
O["H_a_cand_pos_fam"] = str(cp.loc[cp.information_criterion.idxmin(), "candidate_family"])
O["H_a_fa_rows_with_vmin_gt0"] = int(np.sum(FA.min_velocity_cm_s > 0))

# variant (b): rows also in ranked top-250
inr = np.array([k in rank_ic_map_r for k in ckr])
O["H_b_cand_in_rank_n"] = int(inr.sum())
O["H_b_best_ic"] = float(cand_ic[inr].min()) if inr.any() else None

print(json.dumps(O, indent=2, default=str))
Path(P / "output/results/_verify_ic_adv_scratch.json").write_text(
    json.dumps(O, indent=2, default=str), encoding="utf-8")