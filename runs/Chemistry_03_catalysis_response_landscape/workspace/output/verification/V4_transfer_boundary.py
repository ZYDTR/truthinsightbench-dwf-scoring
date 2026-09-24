#!/usr/bin/env python3
"""
Independent falsification of CLAIM D4: "transfer boundary across panels".

All quantities re-derived from the raw TSVs; the main agent's numbers are treated
only as targets to check, never as inputs.

Writes V4_transfer_boundary.json next to this file.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20230919)
NPERM = 50000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data"

OUT = {"claim_id": "D4", "nperm_permutation_tests": NPERM}

d1 = pd.read_csv(os.path.join(DATA, "observation_set_1.tsv"), sep="\t")
d3 = pd.read_csv(os.path.join(DATA, "observation_set_3.tsv"), sep="\t")
d4 = pd.read_csv(os.path.join(DATA, "observation_set_4.tsv"), sep="\t")

# =====================================================================  TASK 1
# Rebuild (series, condition) cell tables straight from the raw row layout.
RANK3 = [f"numeric_feature_{i}" for i in range(9, 22)]    # 13 rank columns
RANK4 = [f"numeric_feature_{i}" for i in range(11, 24)]   # 13 rank columns


def build_panel(df, cond_col, resp_col, mag_cols, rank_cols, n_blocks, tag):
    recs, meta = [], []
    for b in range(n_blocks):
        blk = df.iloc[13 * b: 13 * b + 13]
        lab = blk["numeric_feature_1"].dropna()
        label = float(lab.iloc[0]) if len(lab) else np.nan
        order = [int(v) for v in blk[cond_col]]
        meta.append({"file_block": b, "series_label": label, "cond_order": order})
        for _, r in blk.iterrows():
            recs.append({"file_block": b, "series_label": label,
                         "cond": int(r[cond_col]), "response": float(r[resp_col]),
                         "magA": float(r[mag_cols[0]]), "magB": float(r[mag_cols[1]])})
    cells = pd.DataFrame(recs)
    rank_block = df.iloc[0:n_blocks, :][rank_cols].to_numpy(float)
    index_col = df.iloc[0:n_blocks, :][rank_cols[0]]  # placeholder, refined below
    return cells, pd.DataFrame(meta), rank_block


c3, m3, RB3 = build_panel(d3, "numeric_feature_2", "numeric_feature_5",
                          ["numeric_feature_3", "numeric_feature_4"], RANK3, 8, "set3")
c4, m4, RB4 = build_panel(d4, "numeric_feature_3", "numeric_feature_6",
                          ["numeric_feature_4", "numeric_feature_5"], RANK4, 7, "set4")

LAYOUT = m3["cond_order"].iloc[0]
OUT["task1_panel_layout"] = {
    "set3_rows": int(len(c3)), "set4_rows": int(len(c4)),
    "set3_n_series": int(c3.series_label.nunique()),
    "set4_n_series": int(c4.series_label.nunique()),
    "within_block_condition_order": LAYOUT,
    "order_is_NOTSORTED": LAYOUT != sorted(LAYOUT),
    "set3_blocks_all_same_order": bool(m3.cond_order.map(lambda o: o == LAYOUT).all()),
    "set4_blocks_all_same_order": bool(m4.cond_order.map(lambda o: o == LAYOUT).all()),
    "set3_cond_col": "numeric_feature_2", "set4_cond_col": "numeric_feature_3",
    "set3_resp_col": "numeric_feature_5", "set4_resp_col": "numeric_feature_6",
    "set3_block_file_order_labels": m3.series_label.tolist(),
    "set4_block_file_order_labels": m4.series_label.tolist(),
    "block_label_order_differs_between_panels": bool(
        m3.series_label.tolist()[:7] != m4.series_label.tolist()),
    "same_condition_order_across_panels": bool(
        m3.cond_order.iloc[0] == m4.cond_order.iloc[0]),
    "conditions_seen_per_block": sorted(c3.cond.unique().tolist()),
    "cells_per_series_3": int(c3.groupby("series_label").size().iloc[0]),
    "cells_per_series_4": int(c4.groupby("series_label").size().iloc[0]),
}

# response identity check
OUT["task1_response_identity"] = {
    "set3_max_abs_dev_from_(magB-magA)/(magB+magA)":
        float(np.abs((c3.magB - c3.magA) / (c3.magB + c3.magA) - c3.response).max()),
    "set4_max_abs_dev_from_(magB-magA)/(magB+magA)":
        float(np.abs((c4.magB - c4.magA) / (c4.magB + c4.magA) - c4.response).max()),
}

# --- verify the rank blocks against ranks recomputed from the raw response ----
def verify_rank_block(cells, RB, rank_cols, layout, tag):
    """Check the rank block cell-by-cell, honouring the within-block FILE order:
    rank column j corresponds to condition layout[j]."""
    tab = cells.pivot_table(index="file_block", columns="cond", values="response")
    rk = tab.rank(axis=0)  # 1-based ranks across series, per condition
    rows_ok, cols_ok, cell_ok = {}, {}, 0
    for i in range(RB.shape[0]):
        expect = np.array([rk.loc[i, int(layout[j])] for j in range(13)])
        rows_ok[i] = bool(np.allclose(RB[i, :] + 1.0, expect, atol=1e-9))
        cell_ok += int(np.sum(np.isclose(RB[i, :] + 1.0, expect, atol=1e-9)))
    for j, col in enumerate(rank_cols):
        cols_ok[col] = int(layout[j]) if np.allclose(
            RB[:, j] + 1.0, rk[int(layout[j])].to_numpy(), atol=1e-9) else None
    return {"rank_block_rows_reproduce_recomputed_ranks": rows_ok,
            "rank_block_col_j_maps_to_condition": cols_ok,
            "cells_agreeing": f"{cell_ok}/{RB.size}",
            "all_cells_agree": bool(cell_ok == RB.size),
            "rank_block_is_0_based": True,
            "rank_col_names": rank_cols,
            "rank_column_order_equals_within_block_layout": list(layout)}


OUT["task1_rank_block_verification"] = {
    "set3": verify_rank_block(c3, RB3, RANK3, LAYOUT, "set3"),
    "set4": verify_rank_block(c4, RB4, RANK4, LAYOUT, "set4"),
}
# extra index columns carried beside the rank block
OUT["task1_rank_block_index_columns"] = {
    "set3_numeric_feature_8": d3["numeric_feature_8"].dropna().tolist(),
    "set4_numeric_feature_10": d4["numeric_feature_10"].dropna().tolist(),
    "set4_numeric_feature_9_all_nan": bool(d4["numeric_feature_9"].isna().all()),
    "set4_rank_block_last_row": RB4[-1, :].tolist(),
    "note": ("set4's rank block has 8 rows but only 7 series: row 7 is a constant-3 "
             "filler whose index entry (numeric_feature_10) is 0."),
}

# =====================================================================  TASK 2
f6_mean = d1.groupby("numeric_feature_3")["numeric_feature_6"].mean().reindex(range(13))
f7_mean = d1.groupby("numeric_feature_3")["numeric_feature_7"].mean().reindex(range(13))
n1 = d1.groupby("numeric_feature_3")["numeric_feature_6"].size()
cmean4 = c4.groupby("cond")["response"].mean().reindex(range(13))
cmean3 = c3.groupby("cond")["response"].mean().reindex(range(13))
x1 = f6_mean.to_numpy()


def perm_spearman(x, y, nperm=NPERM, rng=RNG):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = float(stats.spearmanr(x, y).statistic)
    null = np.array([stats.spearmanr(x, rng.permutation(y)).statistic
                     for _ in range(nperm)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (nperm + 1))
    return rho, p


def perm_pearson(x, y, nperm=NPERM, rng=RNG):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r = float(stats.pearsonr(x, y).statistic)
    rng_ = np.random.default_rng(7)
    null = np.array([stats.pearsonr(x, rng_.permutation(y)).statistic
                     for _ in range(nperm)])
    return r, float((np.sum(np.abs(null) >= abs(r)) + 1) / (nperm + 1))


def cyclic(x, y):
    out = {}
    for k in range(13):
        yy = np.roll(np.asarray(y, float), -k)  # level i -> condition (i+k) mod 13
        out[k] = {"spearman": float(stats.spearmanr(x, yy).statistic),
                  "pearson": float(stats.pearsonr(x, yy).statistic)}
    return out


rho_a, p_a = perm_spearman(x1, cmean4.to_numpy())
sh_a = cyclic(x1, cmean4.to_numpy())
best_a = max(sh_a, key=lambda k: sh_a[k]["spearman"])
rho_b, p_b = perm_spearman(x1, cmean3.to_numpy())
sh_b = cyclic(x1, cmean3.to_numpy())
rho_a7, p_a7 = perm_spearman(f7_mean.to_numpy(), cmean4.to_numpy())

OUT["task2_claim_a_panel1_f6_levels_vs_panel4_cond_means"] = {
    "n_levels": 13, "obs_per_level": int(n1.iloc[0]),
    "spearman_identity": rho_a, "perm_p": p_a,
    "pearson_identity": float(stats.pearsonr(x1, cmean4).statistic),
    "identity_shift_is_best": bool(best_a == 0),
    "best_shift": int(best_a), "best_shift_rho": sh_a[best_a]["spearman"],
    "second_best_shift_rho": sorted(sh_a[k]["spearman"] for k in sh_a)[-2],
    "all_shift_rhos": {str(k): sh_a[k]["spearman"] for k in sh_a},
    "panel1_f6_level_means": {int(k): float(v) for k, v in f6_mean.items()},
    "panel4_cond_mean_response": {int(k): float(v) for k, v in cmean4.items()},
    "control_panel1_f7_vs_panel4": {"spearman": rho_a7, "perm_p": p_a7},
    "panel4_cond_mean_response_sd": float(cmean4.std()),
    "panel1_level_mean_sd": float(f6_mean.std()),
}
OUT["task2_claim_b_panel1_f6_levels_vs_panel3_cond_means"] = {
    "n_levels": 13,
    "spearman_identity": rho_b, "perm_p": p_b,
    "pearson_identity": float(stats.pearsonr(x1, cmean3).statistic),
    "identity_shift_is_best": bool(max(sh_b, key=lambda k: sh_b[k]["spearman"]) == 0),
    "best_shift": int(max(sh_b, key=lambda k: sh_b[k]["spearman"])),
    "best_shift_rho": max(sh_b[k]["spearman"] for k in sh_b),
    "all_shift_rhos": {str(k): sh_b[k]["spearman"] for k in sh_b},
    "panel3_cond_mean_response": {int(k): float(v) for k, v in cmean3.items()},
}

# =====================================================================  TASK 3
# (c) matched-cell transfer, both candidate series alignments.
lab3 = sorted(c3.series_label.unique())
lab4 = sorted(c4.series_label.unique())
shared_labels = sorted(set(lab3) & set(lab4))

by_label = c3.merge(c4, on=["series_label", "cond"], suffixes=("3", "4"))
by_block = c3.merge(c4, on=["file_block", "cond"], suffixes=("3", "4"))
OUT["task3_matched_cells"] = {
    "shared_series_labels": shared_labels,
    "labels_only_in_set3": sorted(set(lab3) - set(lab4)),
    "n_cells_label_alignment": int(len(by_label)),
    "n_cells_file_block_alignment": int(len(by_block)),
    "set4_block_order_labels": m4.series_label.tolist(),
}


def transfer_block(m, name, nboot=20000):
    al = np.log10(m.magA3.to_numpy()), np.log10(m.magA4.to_numpy())
    bl = np.log10(m.magB3.to_numpy()), np.log10(m.magB4.to_numpy())
    y = m.response3.to_numpy(), m.response4.to_numpy()
    rresp = stats.pearsonr(*y)
    rA = stats.pearsonr(*al)
    rB = stats.pearsonr(*bl)
    rngl = np.random.default_rng(11)
    idx = rngl.integers(0, len(m), size=(nboot, len(m)))
    boot_resp = np.array([np.corrcoef(y[0][i], y[1][i])[0, 1] for i in idx[:2000]])
    boot_A = np.array([np.corrcoef(al[0][i], al[1][i])[0, 1] for i in idx[:2000]])
    boot_B = np.array([np.corrcoef(bl[0][i], bl[1][i])[0, 1] for i in idx[:2000]])
    return {
        "alignment": name, "n": int(len(m)),
        "response_pearson": float(rresp.statistic), "response_p": float(rresp.pvalue),
        "response_spearman": float(stats.spearmanr(*y).statistic),
        "log_magA_pearson": float(rA.statistic), "log_magA_p": float(rA.pvalue),
        "log_magB_pearson": float(rB.statistic), "log_magB_p": float(rB.pvalue),
        "boot_ci95_response": [float(np.percentile(boot_resp, 2.5)),
                               float(np.percentile(boot_resp, 97.5))],
        "boot_ci95_log_magB": [float(np.percentile(boot_B, 2.5)),
                               float(np.percentile(boot_B, 97.5))],
        "boot_ci95_log_magA": [float(np.percentile(boot_A, 2.5)),
                               float(np.percentile(boot_A, 97.5))],
    }


OUT["task3_cell_transfer_by_alignment"] = [
    transfer_block(by_label, "series_label_identity (H1)"),
    transfer_block(by_block, "file_block_order (H2, what the main agent's -0.086 implies)"),
]

# alignment-free: condition means
cmA3 = np.log10(c3.groupby("cond").magA.mean().reindex(range(13)))
cmA4 = np.log10(c4.groupby("cond").magA.mean().reindex(range(13)))
cmB3 = np.log10(c3.groupby("cond").magB.mean().reindex(range(13)))
cmB4 = np.log10(c4.groupby("cond").magB.mean().reindex(range(13)))
OUT["task3_alignment_free_condition_means"] = {
    "response_cond_mean_pearson": float(stats.pearsonr(cmean3, cmean4).statistic),
    "response_cond_mean_spearman": float(stats.spearmanr(cmean3, cmean4).statistic),
    "response_cond_mean_perm_p": perm_pearson(cmean3.to_numpy(), cmean4.to_numpy())[1],
    "log_magA_cond_mean_pearson": float(stats.pearsonr(cmA3, cmA4).statistic),
    "log_magB_cond_mean_pearson": float(stats.pearsonr(cmB3, cmB4).statistic),
    "log_magA_cond_mean_spearman": float(stats.spearmanr(cmA3, cmA4).statistic),
    "log_magB_cond_mean_spearman": float(stats.spearmanr(cmB3, cmB4).statistic),
    "magA_means_set3": {int(k): float(v) for k, v in cmA3.items()},
    "magB_means_set3": {int(k): float(v) for k, v in cmB3.items()},
    "magA_means_set4": {int(k): float(v) for k, v in cmA4.items()},
    "magB_means_set4": {int(k): float(v) for k, v in cmB4.items()},
}

# ---- rank blocks: compare the data-supplied rank blocks on shared series ----
RB3_7 = np.array([RB3[i] for i in range(8) if lab3[i] in shared_labels], float)  # label order
RB4_7 = np.array([RB4[i] for i in range(7)], float)                              # block order


def rb_pearson(a, b):
    a, b = a.ravel(), b.ravel()
    return float(stats.pearsonr(a, b).statistic)


def rb_double_centered(a, b):
    A = a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + a.mean()
    B = b - b.mean(1, keepdims=True) - b.mean(0, keepdims=True) + b.mean()
    return float(stats.pearsonr(A.ravel(), B.ravel()).statistic)


# H1: set_3 row with label L <-> set_4 row whose label is L
h1_order = [lab3.index(L) for L in shared_labels]          # rows of RB3
h1_target = [m4.series_label.tolist().index(L) for L in shared_labels]  # rows of RB4
RB3_h1 = RB3[h1_order]
RB4_h1 = RB4[h1_target]
# H2: set_3 block i <-> set_4 block i
RB3_h2 = RB3[:7]
RB4_h2 = RB4[:7]

OUT["task3_rank_block_supplied_comparison"] = {
    "rank3_block": {"row_order": "file block order", "row_labels": lab3,
                    "col_j_maps_to_condition": {str(j): int(LAYOUT[j]) for j in range(13)},
                    "values_are_0_based_ranks": True},
    "rank4_block": {"row_order": "file block order (row 7 is filler)",
                    "row_labels": m4.series_label.tolist() + [None],
                    "col_j_maps_to_condition": {str(j): int(LAYOUT[j]) for j in range(13)},
                    "values_are_0_based_ranks": True},
    "H1_label_alignment_Pearson": rb_pearson(RB3_h1, RB4_h1),
    "H1_label_alignment_double_centered_Pearson": rb_double_centered(RB3_h1, RB4_h1),
    "H1_label_alignment_Spearman": float(stats.spearmanr(RB3_h1.ravel(),
                                                         RB4_h1.ravel()).statistic),
    "H1_label_alignment_exact_agreement_share": float(np.mean(RB3_h1 == RB4_h1)),
    "H2_block_order_Pearson": rb_pearson(RB3_h2, RB4_h2),
    "H2_block_order_double_centered_Pearson": rb_double_centered(RB3_h2, RB4_h2),
    "H2_block_order_Spearman": float(stats.spearmanr(RB3_h2.ravel(),
                                                     RB4_h2.ravel()).statistic),
    "H2_block_order_exact_agreement_share": float(np.mean(RB3_h2 == RB4_h2)),
    "H1_per_condition_pearson": {str(j): float(stats.pearsonr(RB3_h1[:, j],
                                                              RB4_h1[:, j]).statistic)
                                 for j in range(13)},
    "H2_per_condition_pearson": {str(j): float(stats.pearsonr(RB3_h2[:, j],
                                                              RB4_h2[:, j]).statistic)
                                 for j in range(13)},
    "H1_mean_per_condition_pearson": float(np.mean([
        stats.pearsonr(RB3_h1[:, j], RB4_h1[:, j]).statistic for j in range(13)])),
    "H2_mean_per_condition_pearson": float(np.mean([
        stats.pearsonr(RB3_h2[:, j], RB4_h2[:, j]).statistic for j in range(13)])),
}


# calibrate: best possible alignment over all 8P7 = 40320 row injections
def calibrate_alignment(RB3full, RB4full):
    perms = np.array(list(itertools.permutations(range(RB3full.shape[0]),
                                                 RB4full.shape[0])))
    A = RB3full[perms]                       # (nperm, 7, 13)
    B = RB4full[None, :, :]
    n = A.shape[0]
    Af = A.reshape(n, -1)
    Bf = B.reshape(1, -1)
    def zp(M):
        return (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    raw = (zp(Af) * zp(np.repeat(Bf, n, 0))).mean(1)
    # double-centred
    Ac = A - A.mean(2, keepdims=True) - A.mean(1, keepdims=True) + A.mean((1, 2), keepdims=True)
    Bc = B - B.mean(2, keepdims=True) - B.mean(1, keepdims=True) + B.mean((1, 2), keepdims=True)
    dbl = (zp(Ac.reshape(n, -1)) * zp(np.repeat(Bc.reshape(1, -1), n, 0))).mean(1)
    top = np.argsort(raw)[::-1][:5]
    return {
        "n_permutations": int(n),
        "raw_pearson_best": float(raw[top[0]]),
        "raw_pearson_best_perm_set3_row_order_for_set4_rows_0to6":
            [int(v) for v in perms[top[0]]],
        "raw_pearson_null_mean": float(raw.mean()),
        "raw_pearson_null_sd": float(raw.std()),
        "raw_pearson_perm_p_for_best": float((np.sum(raw >= raw[top[0]]) + 1) / (n + 1)),
        "raw_pearson_top5": [{"r": float(raw[i]),
                              "perm": [int(v) for v in perms[i]]} for i in top],
        "double_centered_best": float(dbl.max()),
        "double_centered_best_perm": [int(v) for v in perms[int(np.argmax(dbl))]],
        "double_centered_perm_p_for_best": float(
            (np.sum(dbl >= dbl.max()) + 1) / (n + 1)),
        "double_centered_null_mean": float(dbl.mean()),
        "double_centered_null_sd": float(dbl.std()),
        "rank_of_H1_label_alignment_raw": float((raw < rb_pearson(RB3_h1, RB4_h1)).mean()),
        "rank_of_H2_block_alignment_raw": float((raw < rb_pearson(RB3_h2, RB4_h2)).mean()),
        "H1_perm_tuple": [int(v) for v in h1_order],
        "H2_perm_tuple": [0, 1, 2, 3, 4, 5, 6],
    }


OUT["task3_rank_block_alignment_calibration"] = calibrate_alignment(RB3, RB4)


# Max-statistic null: is the best-over-all-alignments value (0.685) stronger than
# what two INDEPENDENT rank blocks would produce over the same 40320 injections?
def max_statistic_null(RB3full, RB4full, nsim=200, rng=0):
    rng = np.random.default_rng(rng)
    perms = np.array(list(itertools.permutations(range(RB3full.shape[0]),
                                                 RB4full.shape[0])))
    n = len(perms)

    def best_of(M3, M4):
        A = M3[perms].reshape(n, -1)
        B = M4.reshape(1, -1).repeat(n, 0)
        Az = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
        Bz = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
        return float((Az * Bz).mean(1).max())

    sims = []
    for _ in range(nsim):
        s3 = RB3full.copy()
        # random independent re-ranking of each column, preserving the 0..7 / 0..6
        # marginal rank sets and the within-row structure of set_4's filler row
        for j in range(s3.shape[1]):
            s3[:, j] = rng.permutation(s3[:, j])
        s4 = RB4full.copy()
        for j in range(s4.shape[1]):
            s4[:7, j] = rng.permutation(s4[:7, j])
        sims.append(best_of(s3, s4))
    sims = np.array(sims)
    return {"observed_best_over_all_alignments":
                calibrate_alignment(RB3full, RB4full)["raw_pearson_best"],
            "independent_blocks_best_of_40320_mean": float(sims.mean()),
            "independent_blocks_best_of_40320_sd": float(sims.std()),
            "independent_blocks_best_of_40320_upper95": float(np.percentile(sims, 95)),
            "simulated_p_observed_best": float((np.sum(sims >= 0.6854190440443328) + 1)
                                               / (nsim + 1)),
            "nsim": nsim}


OUT["task3_rankblock_max_statistic_null"] = max_statistic_null(RB3, RB4)


# calibrate the CELL-LEVEL magnitude / response transfer over the same 40320
# series injections: is the reported "+0.53" specific to a correct alignment,
# or does ANY alignment produce it (shared condition profile)?
def calibrate_values(M3, M4, tag):
    """M3: (8,13) set_3 value grid by label; M4: (7,13) set_4 grid in FILE order."""
    perms = np.array(list(itertools.permutations(range(M3.shape[0]), M4.shape[0])))
    A = M3[perms].reshape(len(perms), -1)
    B = M4.reshape(1, -1).repeat(len(perms), 0)
    Az = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    Bz = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    raw = (Az * Bz).mean(1)
    # H1 label alignment: set4 file row i has label m4.series_label[i]
    h1 = [lab3.index(L) for L in m4.series_label.tolist()]
    k1 = int(np.where((perms == np.array(h1)).all(1))[0][0])
    h2 = [0, 1, 2, 3, 4, 5, 6]
    k2 = int(np.where((perms == np.array(h2)).all(1))[0][0])
    return {
        "statistic": tag,
        "H1_label_alignment_value": float(raw[k1]),
        "H1_percentile_within_all_alignments": float((raw < raw[k1]).mean() * 100),
        "H2_block_order_value": float(raw[k2]),
        "H2_percentile_within_all_alignments": float((raw < raw[k2]).mean() * 100),
        "best_value_over_all_alignments": float(raw.max()),
        "best_alignment_perm": [int(v) for v in perms[int(np.argmax(raw))]],
        "null_mean": float(raw.mean()), "null_sd": float(raw.std()),
        "null_min": float(raw.min()),
        "frac_of_alignments_with_positive_value": float((raw > 0).mean()),
    }


gA3 = np.log10(c3.pivot_table(index="series_label", columns="cond",
                              values="magA").to_numpy(float))
gA4 = np.log10(c4.pivot_table(index="series_label", columns="cond",
                              values="magA").to_numpy(float))
gB3 = np.log10(c3.pivot_table(index="series_label", columns="cond",
                              values="magB").to_numpy(float))
gB4 = np.log10(c4.pivot_table(index="series_label", columns="cond",
                              values="magB").to_numpy(float))
gY3 = c3.pivot_table(index="series_label", columns="cond",
                     values="response").to_numpy(float)
gY4 = c4.pivot_table(index="series_label", columns="cond",
                     values="response").to_numpy(float)
# set_4 grids are indexed by SORTED label; reorder to FILE order for the H2 test
f4ord = [sorted(c4.series_label.unique()).index(L) for L in m4.series_label.tolist()]
OUT["task3_celllevel_alignment_calibration"] = [
    calibrate_values(gA3, gA4[f4ord], "log10 magA (set3 f3 vs set4 f4)"),
    calibrate_values(gB3, gB4[f4ord], "log10 magB (set3 f4 vs set4 f5)"),
    calibrate_values(gY3, gY4[f4ord], "response"),
]
# --- how strong is the rank-block signal really?  max-statistic null, and is the
# --- winning alignment a coherent (label-shift) correspondence?
def rankblock_extra(RB3full, RB4full, nsim=300, seed=5):
    rng = np.random.default_rng(seed)
    perms = np.array(list(itertools.permutations(range(RB3full.shape[0]),
                                                 RB4full.shape[0])))
    n = len(perms)

    def dbl_best(M3, M4):
        A = M3[perms]
        B = np.repeat(M4[None, :, :], n, 0)
        Ac = A - A.mean(2, keepdims=True) - A.mean(1, keepdims=True) + A.mean((1, 2), keepdims=True)
        Bc = B - B.mean(2, keepdims=True) - B.mean(1, keepdims=True) + B.mean((1, 2), keepdims=True)
        Af = Ac.reshape(n, -1)
        Bf = Bc.reshape(n, -1)
        Az = (Af - Af.mean(1, keepdims=True)) / Af.std(1, keepdims=True)
        Bz = (Bf - Bf.mean(1, keepdims=True)) / Bf.std(1, keepdims=True)
        return (Az * Bz).mean(1)

    obs = dbl_best(RB3full, RB4full)
    sims = []
    for _ in range(nsim):
        s3 = RB3full.copy()
        for j in range(s3.shape[1]):
            s3[:, j] = rng.permutation(s3[:, j])
        s4 = RB4full.copy()
        for j in range(s4.shape[1]):
            s4[:7, j] = rng.permutation(s4[:7, j])
        sims.append(float(dbl_best(s3, s4).max()))
    sims = np.array(sims)
    # cyclic label-shift family for the rank blocks (set3 label L vs set4 label L+k)
    cyc = {}
    for k in range(7):
        tgt = [lab3.index(((L + k) % 7)) for L in range(7)]
        tgt = [lab3.index(L) for L in [((L + k) % 7) for L in range(7)]]
        src = [lab3.index(L) for L in range(7)]
        # rows of RB4: label-L rows located at file-block positions
        pos4 = [m4.series_label.tolist().index(L) for L in [((L + k) % 7) for L in range(7)]]
        cyc[k] = {"pearson": rb_pearson(RB3[src], RB4[pos4]),
                  "double_centered": rb_double_centered(RB3[src], RB4[pos4])}
    return {
        "observed_best_double_centered_over_all_alignments": float(obs.max()),
        "winning_alignment_set4_file_row_i_uses_set3_row": [
            int(v) for v in perms[int(np.argmax(obs))]],
        "winning_alignment_maps_set4_label_to_set3_label": {
            str(m4.series_label.tolist()[i]): float(lab3[p])
            for i, p in enumerate(perms[int(np.argmax(obs))])},
        "independent_blocks_max_double_centered_mean": float(sims.mean()),
        "independent_blocks_max_double_centered_sd": float(sims.std()),
        "independent_blocks_max_double_centered_upper95": float(np.percentile(sims, 95)),
        "simulated_p_observed_best_double_centered": float(
            (np.sum(sims >= obs.max()) + 1) / (nsim + 1)),
        "cyclic_label_shift_rank_block_pearson": {
            str(k): cyc[k]["pearson"] for k in cyc},
        "cyclic_label_shift_rank_block_double_centered": {
            str(k): cyc[k]["double_centered"] for k in cyc},
        "best_cyclic_label_shift": int(max(cyc, key=lambda k: cyc[k]["pearson"])),
        "best_cyclic_label_shift_pearson": cyc[max(cyc, key=lambda k: cyc[k]["pearson"])]["pearson"],
        "nsim": nsim,
    }


OUT["task3_rankblock_signal_strength"] = rankblock_extra(RB3, RB4)

OUT["task3_celllevel_alignment_calibration_note"] = (
    "H1 = series labels match across panels; H2 = file block i matches file block i. "
    "If the H2 value sits near the middle/mean of the null distribution, the reported "
    "cell-level transfer is NOT evidence of matched-cell correspondence.")

# does the rank data itself pick out a series correspondence?
OUT["task3_series_level_magnitude_support"] = {}
mA3_s = np.log10(c3.groupby("series_label").magA.mean())
mA4_s = np.log10(c4.groupby("series_label").magA.mean())
mB3_s = np.log10(c3.groupby("series_label").magB.mean())
mB4_s = np.log10(c4.groupby("series_label").magB.mean())
for nm, (s3, s4) in {"log_magA": (mA3_s, mA4_s), "log_magB": (mB3_s, mB4_s)}.items():
    h1 = stats.pearsonr(s3[shared_labels], s4[shared_labels])
    s4_block = s4[[m4.series_label.tolist()[:7].index(L) for L in shared_labels]]
    h2 = stats.pearsonr(s3[shared_labels], s4_block)
    OUT["task3_series_level_magnitude_support"][nm] = {
        "H1_label_pearson": float(h1.statistic), "H1_p": float(h1.pvalue),
        "H2_block_pearson": float(h2.statistic), "H2_p": float(h2.pvalue),
        "set3_values": {str(k): float(v) for k, v in s3.items()},
        "set4_values": {str(k): float(v) for k, v in s4.items()}}

# =====================================================================  TASK 4
logcmA4 = np.log10(c4.groupby("cond").magA.mean().reindex(range(13)).to_numpy())
logcmB4 = np.log10(c4.groupby("cond").magB.mean().reindex(range(13)).to_numpy())


def partial_spearman(x, y, Z, nperm=NPERM):
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.atleast_2d(np.asarray(Z, float)).T if np.asarray(Z).ndim == 1 \
        else np.asarray(Z, float)
    def resid(v, z):
        zz = np.column_stack([np.ones(len(v))] + [pd.Series(z[:, j]).rank().to_numpy()
                                                  for j in range(z.shape[1])])
        return pd.Series(v).rank().to_numpy() - zz @ np.linalg.lstsq(zz, pd.Series(v).rank().to_numpy(), rcond=None)[0]
    ex, ey = resid(x, Z), resid(y, Z)
    r = float(np.corrcoef(ex, ey)[0, 1])
    rng_ = np.random.default_rng(3)
    null = np.array([np.corrcoef(ex, resid(rng_.permutation(y), Z))[0, 1]
                     for _ in range(nperm)])
    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (nperm + 1))
    return {"partial_spearman": r, "perm_p": p}


# ---------------- EXACT permutation p for the 13-level Spearman, by DP --------
# Spearman rho on n=13 is a deterministic function of D = sum_i d_i^2, where d_i
# are rank differences under a uniform random permutation.  Enumerate the exact
# null distribution of D by DP over (used-value set, accumulated D).  13! is far
# too large to enumerate directly, but 2^13 * 729 states is not.
def exact_spearman_p(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    obs_d = float(np.sum((rx - ry) ** 2))
    obs_rho = 1 - 6 * obs_d / (n * (n * n - 1))
    dmax = int(n * (n * n - 1) / 3)
    # dp[mask][D] = number of arrangements of the ranks in positions 0..popcount(mask)-1
    popcount = [bin(m).count("1") for m in range(1 << n)]
    dp = np.zeros((1 << n, dmax + 1), dtype=object)
    dp[0, 0] = 1
    for mask in range(1 << n):
        pos = popcount[mask]
        if pos == n:
            continue
        row = dp[mask]
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            continue
        tgt_rank = pos + 1  # x-rank fixed in position order (x sorted -> rx = pos+1)
        for v in range(n):   # y-rank value assigned to this position
            if mask >> v & 1:
                continue
            d2 = int((tgt_rank - (v + 1)) ** 2)
            sel = nz[nz + d2 <= dmax]
            dp[mask | (1 << v)][sel + d2] += row[sel]
    dist = dp[(1 << n) - 1]
    tot = int(sum(dist))
    # rho >= obs_rho  <=>  D <= obs_d ; the null distribution of D is symmetric
    # about (n^3-n)/6, so the opposite tail is D >= (n^3-n)/3 - obs_d.
    dmid = (n ** 3 - n) // 6
    same_tail = int(sum(dist[:int(obs_d) + 1]))          # P(rho >= +obs)
    opp_tail = int(sum(dist[2 * dmid - int(obs_d):]))    # P(rho <= -obs)
    return {"n": n, "observed_rho": float(obs_rho), "observed_D": obs_d,
            "exact_one_sided_p_same_direction": float(same_tail / tot),
            "exact_two_sided_p": float((same_tail + opp_tail) / tot),
            "n_permutations_enumerated": tot,
            "monte_carlo_p_50000": None,
            "check_symmetry": "2*one_sided should equal two_sided for these data"}


OUT["task2_exact_permutation_p"] = {
    "claim_a_panel1_vs_panel4": exact_spearman_p(x1, cmean4.to_numpy()),
    "claim_b_panel1_vs_panel3": exact_spearman_p(x1, cmean3.to_numpy()),
    "control_panel1_f7_vs_panel4": exact_spearman_p(f7_mean.to_numpy(),
                                                    cmean4.to_numpy()),
}
OUT["task2_exact_permutation_p"]["claim_a_panel1_vs_panel4"]["monte_carlo_p_50000"] = p_a
OUT["task2_exact_permutation_p"]["claim_b_panel1_vs_panel3"]["monte_carlo_p_50000"] = p_b

OUT["task4_confound_check_claim_a"] = {
    "marginal_spearman_f6levels_vs_panel4_response": rho_a,
    "marginal_perm_p": p_a,
    "partial_ctrl_panel4_logmagA": partial_spearman(x1, cmean4.to_numpy(), logcmA4),
    "partial_ctrl_panel4_logmagB": partial_spearman(x1, cmean4.to_numpy(), logcmB4),
    "partial_ctrl_panel4_logmagA_and_B": partial_spearman(
        x1, cmean4.to_numpy(), np.column_stack([logcmA4, logcmB4])),
    "spearman_f6levels_vs_panel4_logmagA": float(stats.spearmanr(x1, logcmA4).statistic),
    "spearman_f6levels_vs_panel4_logmagB": float(stats.spearmanr(x1, logcmB4).statistic),
    "spearman_f6levels_vs_panel4_logmagA_perm_p":
        perm_spearman(x1, logcmA4)[1],
    "spearman_f6levels_vs_panel4_logmagB_perm_p":
        perm_spearman(x1, logcmB4)[1],
    "spearman_panel4_response_vs_its_own_logmagA": float(
        stats.spearmanr(cmean4.to_numpy(), logcmA4).statistic),
    "spearman_panel4_response_vs_its_own_logmagB": float(
        stats.spearmanr(cmean4.to_numpy(), logcmB4).statistic),
    "panel1_f6_level_means": {int(k): float(v) for k, v in f6_mean.items()},
    "panel4_cond_mean_logmagA": {int(k): float(v) for k, v in
                                 zip(range(13), logcmA4)},
    "panel4_cond_mean_logmagB": {int(k): float(v) for k, v in
                                 zip(range(13), logcmB4)},
}

# same confound logic applied to the panel1 <-> panel3 comparison for symmetry
logcmA3 = np.log10(c3.groupby("cond").magA.mean().reindex(range(13)).to_numpy())
logcmB3 = np.log10(c3.groupby("cond").magB.mean().reindex(range(13)).to_numpy())
# Is the panel-4 RESPONSE itself just a monotone read-out of panel-4 magnitude?
# (if so, claim (a)'s "response correspondence" is a magnitude correspondence)
m4c = c4.groupby("cond")[["magA", "magB"]].mean().reindex(range(13))
resp_in_means = (m4c.magB - m4c.magA) / (m4c.magB + m4c.magA)
OUT["task4_panel4_response_is_a_function_of_its_magnitudes"] = {
    "cond_mean_resp_recomputed_from_cond_mean_mags_vs_actual": float(
        stats.pearsonr(resp_in_means, cmean4).statistic),
    "max_abs_diff": float(np.abs(resp_in_means - cmean4).max()),
    "note": "response = (magB-magA)/(magB+magA) exactly at cell level; here at mean level",
}

OUT["task4_cyclic_shift_magnitude_check"] = {}
for tag, yv in {"panel4_cond_mean_logmagA": logcmA4,
                "panel4_cond_mean_logmagB": logcmB4,
                "panel3_cond_mean_logmagA": logcmA3,
                "panel3_cond_mean_logmagB": logcmB3}.items():
    sh = cyclic(x1, yv)
    best = max(sh, key=lambda k: sh[k]["spearman"])
    OUT["task4_cyclic_shift_magnitude_check"][tag] = {
        "rho_identity": sh[0]["spearman"],
        "best_shift": int(best), "best_shift_rho": sh[best]["spearman"],
        "identity_is_best": bool(best == 0),
    }

OUT["task4_symmetry_panel1_vs_panel3"] = {
    "marginal_spearman": rho_b,
    "partial_ctrl_panel3_logmagA": partial_spearman(x1, cmean3.to_numpy(), logcmA3),
    "partial_ctrl_panel3_logmagB": partial_spearman(x1, cmean3.to_numpy(), logcmB3),
    "partial_ctrl_panel3_both": partial_spearman(
        x1, cmean3.to_numpy(), np.column_stack([logcmA3, logcmB3])),
}

# ============================================================ verdict summary
OUT["verdict_components"] = {
    "a_rho_and_p_confirmed":
        bool(abs(rho_a - 0.86) < 0.02 and p_a < 0.001 and best_a == 0),
    "b_rho_and_p_confirmed": bool(abs(rho_b - 0.32) < 0.03 and p_b > 0.10),
    "c_response_transfer_fails_confirmed_under_both_alignments": bool(
        all(abs(b["response_pearson"]) < 0.25
            for b in OUT["task3_cell_transfer_by_alignment"])
        and abs(OUT["task3_alignment_free_condition_means"]["response_cond_mean_pearson"]) < 0.15),
    "c_magnitude_transfer_confirmed_under_both_alignments": bool(
        all(b["log_magA_pearson"] > 0.25 and b["log_magB_pearson"] > 0.35
            for b in OUT["task3_cell_transfer_by_alignment"])),
    "c_matched_cell_number_-0.086_reproduced": bool(
        abs(OUT["task3_cell_transfer_by_alignment"][1]["response_pearson"] + 0.0865) < 0.005),
    "c_matched_cell_number_-0.086_not_reproduced_by_label_alignment": bool(
        abs(OUT["task3_cell_transfer_by_alignment"][0]["response_pearson"] + 0.1958) < 0.005),
    "reported_+0.53_logmagB_matches_block_order_alignment": bool(
        abs(OUT["task3_cell_transfer_by_alignment"][1]["log_magB_pearson"] - 0.5294) < 0.005),
    "reported_-0.086_alignment_contradicts_the_panels_own_series_labels": bool(
        m4.series_label.tolist() != sorted(m4.series_label.tolist())),
    "panel4_series_labels_IN_FILE_ORDER": m4.series_label.tolist(),
    "celllevel_response_pearson_label_alignment":
        OUT["task3_cell_transfer_by_alignment"][0]["response_pearson"],
    "celllevel_response_pearson_file_block_alignment":
        OUT["task3_cell_transfer_by_alignment"][1]["response_pearson"],
    "celllevel_logmagB_percentile_label_alignment":
        OUT["task3_celllevel_alignment_calibration"][1][
            "H1_percentile_within_all_alignments"],
    "celllevel_logmagB_percentile_file_block_alignment":
        OUT["task3_celllevel_alignment_calibration"][1][
            "H2_percentile_within_all_alignments"],
}
OUT["what_the_data_do_show"] = {
    "panel_level_magnitude_transfer_lognA_pearson":
        OUT["task3_alignment_free_condition_means"]["log_magA_cond_mean_pearson"],
    "panel_level_magnitude_transfer_lognB_pearson":
        OUT["task3_alignment_free_condition_means"]["log_magB_cond_mean_pearson"],
    "panel_level_response_transfer_pearson":
        OUT["task3_alignment_free_condition_means"]["response_cond_mean_pearson"],
    "panel_level_response_transfer_perm_p":
        OUT["task3_alignment_free_condition_means"]["response_cond_mean_perm_p"],
    "rankblock_best_possible_alignment_pearson":
        OUT["task3_rank_block_alignment_calibration"]["raw_pearson_best"],
    "rankblock_best_possible_alignment_perm_p":
        OUT["task3_rank_block_alignment_calibration"]["raw_pearson_perm_p_for_best"],
    "rankblock_observed_best_over_independent_blocks_p":
        OUT["task3_rankblock_max_statistic_null"]["simulated_p_observed_best"],
    "rankblock_double_centered_max_statistic_p":
        OUT["task3_rankblock_signal_strength"][
            "simulated_p_observed_best_double_centered"],
    "series_level_magnitude_lognA_label_alignment_p":
        OUT["task3_series_level_magnitude_support"]["log_magA"]["H1_p"],
    "series_level_magnitude_lognB_label_alignment_p":
        OUT["task3_series_level_magnitude_support"]["log_magB"]["H1_p"],
}


def default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


with open(os.path.join(HERE, "V4_transfer_boundary.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=default)

print(json.dumps(OUT["verdict_components"], indent=1))
print(json.dumps(OUT["task1_panel_layout"], indent=1))
print(json.dumps(OUT["task2_claim_a_panel1_f6_levels_vs_panel4_cond_means"], indent=1))
print(json.dumps(OUT["task2_claim_b_panel1_f6_levels_vs_panel3_cond_means"], indent=1))
print(json.dumps(OUT["task3_cell_transfer_by_alignment"], indent=1))
print(json.dumps(OUT["task3_alignment_free_condition_means"], indent=1))
print(json.dumps({k: v for k, v in OUT["task3_rank_block_supplied_comparison"].items()
                  if not isinstance(v, dict)}, indent=1))
print(json.dumps({k: v for k, v in OUT["task3_rank_block_alignment_calibration"].items()
                  if not isinstance(v, list)}, indent=1))
print(json.dumps(OUT["task4_confound_check_claim_a"], indent=1))
print("WROTE", os.path.join(HERE, "V4_transfer_boundary.json"))