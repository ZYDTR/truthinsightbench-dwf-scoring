"""Step 5 - Robustness battery for the replicated-surface claims (task t20).

Stress-tests three headline claims from steps 1-2:
  (C1) the (a1,a2) x 7-channel log surface is near rank-1 ("single latent gain axis"),
  (C2) the channel amplification chan6/chan0 slope ratio is ~29,
  (C3) the channel-regime boundary sits at channels 0-2 | 3-6.

Six probes, each with the exact definition used:
  1. alternative normalisations of the 49x7 surface (log, log10, sqrt, rank, per-channel
     z-score, per-channel min-max, raw linear);
  2. outlier treatment (drop top-1% within-cell-CV cells / winsorise 1-99 pct within
     channel / drop global min+max per channel);
  3. exhaustive 63 two-block partitions of the 7 channels with permutation p-values;
  4. bootstrap CIs (4000 resamples of complete (a1,a2) grid conditions);
  5. complete-unit leave-out (each replicate_id, each output channel);
  6. permutation null for the single-gain-axis claim (500 synthetic surfaces).

Run: python3 output/analysis/step5_robustness.py
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from common import (
    GRID,
    KEY4,
    LOGS,
    RESULTS,
    block_permutation,
    cell_means,
    flush,
    load_panels,
    rank1_profile,
    r2,
    save_json,
)

T_START = time.time()
LOG: list[str] = []
frames = load_panels(LOG)
rep = frames["rep"]

X7 = np.arange(7, dtype=float)
X7C = X7 - X7.mean()          # centred 0..6 levels
SXX = float((X7C ** 2).sum())  # = 28, denominator of every OLS slope below

CHANNELS = list(range(7))


# ------------------------------------------------------------------ helpers
def rank1_table(df: pd.DataFrame, by=("input_axis_1_index", "input_axis_2_index")) -> tuple[np.ndarray, int, tuple]:
    """Replicate-averaged log cell means pivoted to (cells x 7 channels), with any cell
    that lost all of its replicates removed from every channel so the matrix stays complete."""
    cm = df.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    T = cm.pivot_table(index=list(by), columns="output_channel_index", values="ly").sort_index()
    n_before = len(T)
    T = T.dropna(axis=0, how="any")
    return T.values, int(n_before - len(T)), T.shape


def slopes_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel OLS slopes of the log surface on each input axis.

    Definition (identical to step1/t04): the (a1,a2)x7 table of replicate-averaged log
    cell means (all 8 bins averaged) is used; slope_a1[ch] = mean over the a2 levels of
    the OLS slope of the value on the a1 level index (0..6), and symmetrically for a2.
    Levels whose cell is missing (only possible in the outlier variants) are skipped.
    """
    cm = df.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    s1 = np.full(7, np.nan)
    s2 = np.full(7, np.nan)
    xa1 = np.arange(7, dtype=float)
    for ch in range(7):
        sub = cm[cm["output_channel_index"] == ch]
        P = sub.pivot_table(index="input_axis_2_index", columns="input_axis_1_index", values="ly").sort_index()
        slopes = [np.polyfit(P.columns.values.astype(float), P.loc[a].dropna().values, 1)[0]
                  for a in P.index if P.loc[a].notna().sum() >= 2]
        s1[ch] = float(np.mean(slopes))
        Q = sub.pivot_table(index="input_axis_1_index", columns="input_axis_2_index", values="ly").sort_index()
        slopes = [np.polyfit(Q.columns.values.astype(float), Q.loc[a].dropna().values, 1)[0]
                  for a in Q.index if Q.loc[a].notna().sum() >= 2]
        s2[ch] = float(np.mean(slopes))
    return s1, s2


def ratio_from(df: pd.DataFrame) -> dict:
    """chan6/chan0 amplification ratio, plus the noise-floor-free peak/subgrid variant."""
    s1, s2 = slopes_from_df(df)
    out = {
        "slope_a1": [float(v) for v in s1],
        "slope_a2": [float(v) for v in s2],
        "ratio_a1_ch6_over_ch0": float(s1[6] / s1[0]),
        "ratio_a2_ch6_over_ch0": float(s2[6] / s2[0]),
        "slopes_are_single_signed_rising": bool(np.all(np.diff(s1) > 0)),
    }
    pk = int(np.argmax(np.abs(s1)))  # renormalise to the peak channel: no tiny-denominator inflation
    out["peak_channel"] = pk
    out["peak_slope"] = float(s1[pk])
    out["ratio_channel_to_peak"] = [float(v / s1[pk]) for v in s1]
    return out


BASE, BASE_DROPPED, BASE_SHAPE = rank1_table(rep)
assert BASE.shape == (49, 7), BASE.shape
BASE_INFO = rank1_profile(BASE)
BASE_RATIO = ratio_from(rep)
BASE_RANK1 = float(BASE_INFO["rank1_r2"])

# fast vectorised bootstrap machinery: SUB[condition, bin, chan] with condition = a1*7 + a2
_cm = rep.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
_T = _cm.pivot_table(index=GRID, columns="output_channel_index", values="ly").sort_index()
_idx = _T.index.to_frame(index=False)
COND = _idx["input_axis_1_index"].values.astype(int) * 7 + _idx["input_axis_2_index"].values.astype(int)
BIN = _idx["signal_bin_index"].values.astype(int)
SUB = np.empty((49, 8, 7))
SUB[COND, BIN, :] = _T.values
assert not np.isnan(SUB).any()
COND_A1 = np.array([c // 7 for c in range(49)], dtype=float)  # a1 level of each condition
COND_A2 = np.array([c % 7 for c in range(49)], dtype=float)   # a2 level of each condition
assert np.allclose(COND_A1[COND], _idx["input_axis_1_index"].values)
assert np.allclose(COND_A2[COND], _idx["input_axis_2_index"].values)

LOG.append(
    "baseline: 49x7 log surface rank1 R2=%.6f rank2=%.6f ; a1 slopes ch0=%.6f ch6=%.6f ratio=%.4f ; "
    "peak/subgrid ratio (ch6/ch5)=%.4f"
    % (BASE_RANK1, BASE_INFO["rank2_r2"], BASE_RATIO["slope_a1"][0], BASE_RATIO["slope_a1"][6],
       BASE_RATIO["ratio_a1_ch6_over_ch0"],
       BASE_RATIO["ratio_channel_to_peak"][6] / BASE_RATIO["ratio_channel_to_peak"][5])
)

RESULT: dict = {
    "task": "t20_robustness",
    "baseline": {
        "definition": "replicate-averaged (49 conditions x 7 channels) log-response cell means; "
                      "rank-1 R2 = 1 - sum(s[1:]^2)/sum(s^2) after centring each channel (common.rank1_profile)",
        "rank1_r2_49x7": BASE_RANK1,
        "rank2_r2_49x7": float(BASE_INFO["rank2_r2"]),
        "rank1_r2_392x7": float(rank1_profile(rank1_table(rep, by=GRID)[0])["rank1_r2"]),
        "amplification": BASE_RATIO,
    },
    "definitions": {
        "rank1_r2": "SVD of the column-centred matrix M (cells x channels); R2_1 = 1 - sum_{k>=2} s_k^2 / sum_k s_k^2",
        "amplification ratio": "slope_a1[ch6] / slope_a1[ch0]; slope_a1[ch] = mean over the 7 a2 levels of the OLS "
                               "slope of log cell-mean response on the a1 level index 0..6 (step1/t04 definition)",
        "analysis unit": "one complete (a1,a2) input-grid condition; adjacent cells and replicates are not independent",
    },
}

# =====================================================================
# 1. alternative normalisations of the same 49x7 surface
# =====================================================================
lin_cm = rep.groupby(GRID + ["output_channel_index"], as_index=False)["response_value"].mean()
MLIN = lin_cm.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                          columns="output_channel_index", values="response_value").sort_index().values
MLOG = np.exp(BASE)  # identical to BASE on the log scale; kept explicit for the report
assert np.allclose(np.log(MLOG), BASE)

TRANSFORMS: dict[str, np.ndarray] = {}
TRANSFORMS["linear"] = MLIN
TRANSFORMS["natural_log"] = BASE
TRANSFORMS["log10"] = np.log10(MLIN)
TRANSFORMS["sqrt"] = np.sqrt(MLIN)
TRANSFORMS["rank"] = np.apply_along_axis(
    lambda col: pd.Series(col).rank().values, 0, MLIN
)
TRANSFORMS["per_channel_zscore"] = (BASE - BASE.mean(axis=0, keepdims=True)) / BASE.std(axis=0, ddof=1, keepdims=True)
TRANSFORMS["per_channel_minmax"] = (BASE - BASE.min(axis=0, keepdims=True)) / (
    BASE.max(axis=0, keepdims=True) - BASE.min(axis=0, keepdims=True)
)
TRANSFORMS["rank_within_channel"] = np.apply_along_axis(  # supplementary to the requested "rank"
    lambda col: pd.Series(col).rank().values, 0, BASE
)

log_load = np.asarray(BASE_INFO["loadings_comp1"], dtype=float)
norm_block: dict[str, dict] = {}
for name, M in TRANSFORMS.items():
    prof = rank1_profile(M)
    lead = np.asarray(prof["loadings_comp1"], dtype=float)
    # sign-free agreement of the leading channel loading vector with the log solution
    load_match = float(abs(np.corrcoef(lead, log_load)[0, 1])) if lead.std() > 0 else float("nan")
    norm_block[name] = {
        "rank1_r2": float(prof["rank1_r2"]),
        "rank2_r2": float(prof["rank2_r2"]),
        "singular_values_first3": [float(v) for v in prof["singular_values"][:3]],
        "loading_agreement_with_log": load_match,
        "loadings_comp1": [float(v) for v in lead],
        "preserves_rank1": bool(prof["rank1_r2"] >= 0.9),
    }
# per-channel dynamic range, the diagnostic that explains the min-max collapse
norm_block["diagnostic_per_channel_range_of_linear_surface"] = {
    "range": [float(v) for v in (MLIN.max(axis=0) - MLIN.min(axis=0))],
    "range_ratio_to_ch0": [float(v / (MLIN.max(axis=0) - MLIN.min(axis=0))[0])
                           for v in (MLIN.max(axis=0) - MLIN.min(axis=0))],
    "range_of_log_surface": [float(v) for v in (BASE.max(axis=0) - BASE.min(axis=0))],
}
RESULT["normalisations"] = norm_block
preserve = [k for k, v in norm_block.items() if isinstance(v, dict) and "preserves_rank1" in v and v["preserves_rank1"]]
destroy = [k for k, v in norm_block.items() if isinstance(v, dict) and "preserves_rank1" in v and not v["preserves_rank1"]]
RESULT["normalisations_summary"] = {
    "preserve_rank1_R2_ge_0.9": preserve,
    "destroy_rank1_R2_lt_0.9": destroy,
}
LOG.append("normalisation battery: preserve R2>=0.9 -> %s ; destroy -> %s" % (preserve, destroy))
for name, blk in norm_block.items():
    if isinstance(blk, dict) and "rank1_r2" in blk:
        LOG.append("  %-20s rank1 R2=%.4f rank2=%.4f loading_match=%.3f"
                   % (name, blk["rank1_r2"], blk["rank2_r2"], blk["loading_agreement_with_log"]))

# =====================================================================
# 2. outlier treatment
# =====================================================================
out_block: dict[str, dict] = {}

# (a) drop the 3 replicates of any cell whose within-cell CV is in the top 1%
cv_cells = rep.groupby(KEY4)["response_value"].agg(["std", "mean"]).reset_index()
cv_cells["cv"] = cv_cells["std"] / cv_cells["mean"]
thr = float(cv_cells["cv"].quantile(0.99))
drop_cells = cv_cells[cv_cells["cv"] >= thr]
drop_keys = set(map(tuple, drop_cells[KEY4].values))
mask_drop = rep[KEY4].apply(tuple, axis=1).isin(drop_keys)
kept = rep[~mask_drop]
Za, n_drop_a, shp_a = rank1_table(kept)
Zb, n_drop_b, shp_b = rank1_table(kept, by=GRID)
out_block["drop_top1pct_cv_cells"] = {
    "definition": "remove every replicate of cells whose within-cell coefficient of variation "
                  "(replicate sd/mean, pandas ddof=1, computed on response_value) is >= the 99th percentile",
    "cv_threshold": thr,
    "n_cells_total": int(len(cv_cells)),
    "n_cells_dropped": int(len(drop_cells)),
    "n_rows_dropped": int(mask_drop.sum()),
    "cv_min_dropped": float(drop_cells["cv"].min()),
    "cv_max_kept": float(cv_cells[cv_cells["cv"] < thr]["cv"].max()),
    "dropped_cells": [[int(v) for v in k] for k in sorted(drop_keys)],
    "dropped_cells_per_channel": {str(c): int((drop_cells["output_channel_index"] == c).sum()) for c in CHANNELS},
    "rank1_r2_49x7": float(rank1_profile(Za)["rank1_r2"]),
    "rank2_r2_49x7": float(rank1_profile(Za)["rank2_r2"]),
    "rank1_r2_392x7": float(rank1_profile(Zb)["rank1_r2"]),
    "grid_cells_lost_with_all_replicates": {
        "49x7_table": [int(n_drop_a), list(shp_a)],
        "392x7_table": [int(n_drop_b), list(shp_b)],
    },
    "amplification": ratio_from(kept),
}

# (b) winsorise response_value at the 1st/99th percentile within each channel
lo = rep.groupby("output_channel_index")["response_value"].transform(lambda s: np.percentile(s, 1))
hi = rep.groupby("output_channel_index")["response_value"].transform(lambda s: np.percentile(s, 99))
rep_w = rep.copy()
rep_w["response_value"] = rep_w["response_value"].clip(lower=lo, upper=hi)
rep_w["ly"] = np.log(rep_w["response_value"])
n_wins = int((rep["response_value"] != rep_w["response_value"]).sum())
Za, _, shp_a = rank1_table(rep_w)
Zb, _, shp_b = rank1_table(rep_w, by=GRID)
out_block["winsorise_p1_p99_within_channel"] = {
    "definition": "clip response_value to the 1st/99th percentile computed within each of the 7 channels "
                  "(linear interpolation), then log-transform and average replicates",
    "n_values_clipped": n_wins,
    "fraction_clipped": float(n_wins / len(rep)),
    "table_shapes": {"49x7": list(shp_a), "392x7": list(shp_b)},
    "rank1_r2_49x7": float(rank1_profile(Za)["rank1_r2"]),
    "rank2_r2_49x7": float(rank1_profile(Za)["rank2_r2"]),
    "rank1_r2_392x7": float(rank1_profile(Zb)["rank1_r2"]),
    "amplification": ratio_from(rep_w),
}

# (c) drop the single largest and single smallest response_value per channel
keep_idx = []
dropped_rows = []
for c in CHANNELS:
    subc = rep[rep["output_channel_index"] == c]
    imax = subc["response_value"].idxmax()
    imin = subc["response_value"].idxmin()
    keep_idx.extend([i for i in subc.index if i not in (imax, imin)])
    dropped_rows += [("max", c, imax), ("min", c, imin)]
rep_d = rep.loc[keep_idx]
Za, n_drop_a, shp_a = rank1_table(rep_d)
Zb, n_drop_b, shp_b = rank1_table(rep_d, by=GRID)
rep_counts = rep_d.groupby(KEY4).size()
out_block["drop_global_minmax_per_channel"] = {
    "definition": "remove the single largest and single smallest response_value (row-level, raw scale) "
                  "independently within each channel; 14 rows removed; cell means recomputed from the "
                  "remaining 2 replicates for the affected cells",
    "n_rows_removed": int(len(rep) - len(rep_d)),
    "n_cells_with_2_replicates": int((rep_counts == 2).sum()),
    "min_replicates_left_in_a_cell": int(rep_counts.min()),
    "amax_values": {str(c): float(rep.loc[[i for t, cc, i in dropped_rows if cc == c and t == "max"][0], "response_value"])
                    for c in CHANNELS},
    "amin_values": {str(c): float(rep.loc[[i for t, cc, i in dropped_rows if cc == c and t == "min"][0], "response_value"])
                    for c in CHANNELS},
    "grid_cells_lost_with_all_replicates": [int(n_drop_a), int(n_drop_b)],
    "rank1_r2_49x7": float(rank1_profile(Za)["rank1_r2"]),
    "rank2_r2_49x7": float(rank1_profile(Za)["rank2_r2"]),
    "rank1_r2_392x7": float(rank1_profile(Zb)["rank1_r2"]),
    "amplification": ratio_from(rep_d),
}
out_block["baseline"] = {
    "rank1_r2_49x7": BASE_RANK1,
    "rank2_r2_49x7": float(BASE_INFO["rank2_r2"]),
    "amplification": BASE_RATIO,
}
RESULT["outlier_treatment"] = out_block
LOG.append("outlier battery (rank1 R2 / chan6-chan0 ratio / peak-subgrid ratio):")
for name in ["baseline", "drop_top1pct_cv_cells", "winsorise_p1_p99_within_channel", "drop_global_minmax_per_channel"]:
    b = out_block[name]
    r = b["amplification"]
    LOG.append("  %-30s R2=%.4f ratio=%.4f peak/sub=%.4f"
               % (name, b["rank1_r2_49x7"], r["ratio_a1_ch6_over_ch0"],
                  r["ratio_channel_to_peak"][6] / r["ratio_channel_to_peak"][5]))

# =====================================================================
# 3. exhaustive 63 two-block partitions of the 7 channels
# =====================================================================
# channel shape profiles on the (a1,bin) x a2 grid, exactly as in step1/t07
cell = cell_means(rep, "ly")
shapes = {}
for ch in CHANNELS:
    sub = cell[cell["output_channel_index"] == ch]
    P = sub.pivot_table(index=["input_axis_1_index", "signal_bin_index"],
                        columns="input_axis_2_index", values="ly").values.ravel()
    shapes[ch] = P - P.mean()
CORR = pd.DataFrame(shapes).corr().values

PAIRS = list(itertools.combinations(range(7), 2))
PI = np.array([p[0] for p in PAIRS])
PJ = np.array([p[1] for p in PAIRS])
R_VEC = np.array([CORR[i, j] for i, j in PAIRS])

N_PERM = 20000
rng = np.random.default_rng(2024)
PERMS = np.array([rng.permutation(7) for _ in range(N_PERM)])


def gap_vectorised(labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Observed within-minus-between gap and the full permutation null of it."""
    same = np.array([[labels[i] == labels[j] for i, j in PAIRS]])
    obs = float(R_VEC[same[0]].mean() - R_VEC[~same[0]].mean())
    LP = labels[PERMS]                      # (n_perm, 7) shuffled label vectors
    sm = LP[:, PI] == LP[:, PJ]             # (n_perm, 21)
    n_s = sm.sum(axis=1)
    n_d = 21 - n_s
    w = (sm * R_VEC).sum(axis=1) / n_s
    b = ((~sm) * R_VEC).sum(axis=1) / n_d
    return obs, w - b


# Every 2-block partition {A, B} of 7 channels has exactly one side containing channel 0,
# so enumerating the 2^6 - 1 = 63 proper subsets of {1..6} as A (with B = complement) is
# exhaustive and counts each partition exactly once. This includes the contiguous cuts.
PARTITIONS: list[tuple[str, tuple[int, ...]]] = []
for size in range(1, 7):
    for S in itertools.combinations(range(1, 7), size - 1):
        blk = (0,) + S
        others = [c for c in range(7) if c not in blk]
        label = ("cut=%d [0-%d | %d-6]" % (size, size - 1, size)) if blk == tuple(range(size)) else \
                ("blocks %s | %s" % (list(blk), others))
        PARTITIONS.append((label, blk))
assert len(PARTITIONS) == 63, len(PARTITIONS)
assert len(set(tuple(sorted(p[1])) for p in PARTITIONS)) == 63

part_rows = []
for name, blk in PARTITIONS:
    labels = np.array([0 if c in blk else 1 for c in range(7)])
    obs, null = gap_vectorised(labels)
    p = float((null >= obs).mean())
    part_rows.append({
        "partition": name,
        "first_block": [int(c) for c in blk],
        "labels": [int(v) for v in labels],
        "observed_gap": obs,
        "perm_p": p,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "n_perm": N_PERM,
    })
part_rows.sort(key=lambda d: -d["observed_gap"])
best = part_rows[0]
nominal = next(d for d in part_rows if d["labels"] == [0, 0, 0, 1, 1, 1, 1])
second = part_rows[1]

# cross-check the vectorised statistic against common.block_permutation (same method as t08)
chk = block_permutation(CORR, [0, 0, 0, 1, 1, 1, 1], n_perm=20000, seed=7)
check_ok = bool(abs(chk["observed_gap"] - nominal["observed_gap"]) < 1e-12)
within_null = float((part_rows[0]["observed_gap"] - part_rows[-1]["observed_gap"]))
RESULT["channel_block_partitions"] = {
    "definition": "channel shape = bin-averaged-over-conditions log profile on the (a1,bin)x(a2) grid, "
                  "mean-centred; CORR = 7x7 Pearson correlation across the 49 profile entries; "
                  "statistic = mean within-block CORR - mean between-block CORR; "
                  "permutation null = 20000 random shuffles of the 7 block labels (seed 2024)",
    "n_partitions": 63,
    "n_perm": N_PERM,
    "best_partition": best,
    "nominal_partition_ch0_2_vs_ch3_6": nominal,
    "second_best": second,
    "gap_difference_best_minus_nominal": float(best["observed_gap"] - nominal["observed_gap"]),
    "gap_difference_best_minus_second": float(best["observed_gap"] - second["observed_gap"]),
    "range_of_gaps_across_63_partitions": [float(part_rows[-1]["observed_gap"]), float(part_rows[0]["observed_gap"])],
    "best_vs_worst_gap_span": within_null,
    "vectorised_stat_matches_common_block_permutation": check_ok,
    "common_block_permutation_crosscheck": chk,
    "ranking": part_rows,
    "n_partitions_perm_p_lt_0.05": int(sum(d["perm_p"] < 0.05 for d in part_rows)),
    "n_partitions_perm_p_lt_0.025": int(sum(d["perm_p"] < 0.025 for d in part_rows)),
}
LOG.append("channel-block partitions: best = %s gap=%.4f p=%.5f ; nominal ch0-2|ch3-6 gap=%.4f p=%.5f"
           % (best["partition"], best["observed_gap"], best["perm_p"], nominal["observed_gap"], nominal["perm_p"]))
LOG.append("  second best = %s gap=%.4f p=%.5f ; gap span across all 63 = %.4f ; "
           "vectorised vs common.block_permutation observed-gap match = %s"
           % (second["partition"], second["observed_gap"], second["perm_p"], within_null, check_ok))
for d in part_rows[:8]:
    LOG.append("   top8: %-38s gap=%.4f p=%.5f (null sd=%.4f)" % (d["partition"], d["observed_gap"], d["perm_p"], d["null_sd"]))

# =====================================================================
# 4. bootstrap CIs over complete (a1,a2) grid conditions
# =====================================================================
N_BOOT = 4000
rng_b = np.random.default_rng(11)
boot_r2 = np.empty(N_BOOT)
boot_slope0 = np.empty(N_BOOT)
boot_slope6 = np.empty(N_BOOT)
boot_slope5 = np.empty(N_BOOT)
boot_slopes = np.empty((N_BOOT, 7))
for b in range(N_BOOT):
    draw = rng_b.integers(0, 49, 49)           # complete conditions, with replacement
    Bm = SUB[draw]                             # (49, 8, 7)
    M = Bm.reshape(392, 7)
    Xc = M - M.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    boot_r2[b] = 1.0 - (sv[1:] ** 2).sum() / (sv ** 2).sum()
    z = Bm.mean(axis=1)                        # bin-averaged (49,7), condition order = draw order
    a1d, a2d = COND_A1[draw], COND_A2[draw]
    # exact step1 estimator, evaluated on the drawn conditions: mean over the a2 levels present
    # of the within-level OLS slope on the a1 level
    per_level = []
    for g in range(7):
        m = a2d == g
        if m.sum() >= 2:
            d = a1d[m] - a1d[m].mean()
            if (d ** 2).sum() > 0:  # all drawn a1 levels identical -> slope undefined, level skipped
                per_level.append(((d[:, None] * z[m]).sum(axis=0)) / (d ** 2).sum())
    s1 = np.mean(per_level, axis=0)
    boot_slopes[b] = s1
    boot_slope0[b], boot_slope5[b], boot_slope6[b] = s1[0], s1[5], s1[6]


def ci(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "min": float(v.min()),
        "max": float(v.max()),
        "p2.5": float(np.percentile(v, 2.5)),
        "p50": float(np.percentile(v, 50)),
        "p97.5": float(np.percentile(v, 97.5)),
        "n_finite": int(len(v)),
    }


RESULT["bootstrap"] = {
    "definition": "4000 resamples of the 49 complete (a1,a2) grid conditions with replacement; "
                  "each resample carries all 8 signal bins of the drawn condition; the 392x7 matrix is "
                  "rebuilt and the rank-1 R2 and the two OLS channel slopes are recomputed",
    "n_resamples": N_BOOT,
    "n_units": 49,
    "rank1_r2": {**ci(boot_r2), "observed": BASE_RANK1,
                 "ci_contains_observed": bool(np.percentile(boot_r2, 2.5) <= BASE_RANK1 <= np.percentile(boot_r2, 97.5))},
    "ratio_a1_ch6_over_ch0": {**ci(boot_slope6 / boot_slope0),
                              "observed": BASE_RATIO["ratio_a1_ch6_over_ch0"],
                              "fraction_slope_ch0_negative": float((boot_slope0 < 0).mean()),
                              "fraction_ratio_negative": float(((boot_slope6 / boot_slope0) < 0).mean()),
                              "fraction_ratio_finite_and_gt_1": float(
                                  (np.isfinite(boot_slope6 / boot_slope0) & (boot_slope6 / boot_slope0 > 1)).mean()),
                              "note": "the ch0 slope is ~0.017 and can be negative by chance, so the ratio "
                                      "distribution is heavy-tailed and its percentile CI is unstable; the "
                                      "peak/subgrid ratio and the slope-ordering concordance are the stable summaries"},
    "slope_a1_ch0": {**ci(boot_slope0), "observed": BASE_RATIO["slope_a1"][0]},
    "slope_a1_ch6": {**ci(boot_slope6), "observed": BASE_RATIO["slope_a1"][6]},
    "ratio_a1_ch6_over_ch5_peak_subgrid": {**ci(boot_slope6 / boot_slope5),
                                           "observed": float(BASE_RATIO["ratio_channel_to_peak"][6] /
                                                             BASE_RATIO["ratio_channel_to_peak"][5])},
    "slope_ratio_rank_concordance": {
        "definition": "for each resample, the ordering of the 7 a1 slopes; fraction of resamples in "
                      "which the slope ordering is strictly increasing ch0<...<ch6 and in which ch6 is the maximum",
        "fraction_strictly_increasing": float((np.diff(boot_slopes, axis=1) > 0).all(axis=1).mean()),
        "fraction_ch6_is_max": float((np.argmax(boot_slopes, axis=1) == 6).mean()),
        "fraction_ch6_over_ch0_gt_1": float((boot_slope6 / boot_slope0 > 1).mean()),
    },
}
LOG.append("bootstrap (%d resamples of 49 complete conditions): rank1 R2 = %.4f [%.4f, %.4f] (observed %.4f)"
           % (N_BOOT, boot_r2.mean(), np.percentile(boot_r2, 2.5), np.percentile(boot_r2, 97.5), BASE_RANK1))
_b = RESULT["bootstrap"]
LOG.append("  ratio ch6/ch0 = %.3f [%.3f, %.3f] (observed %.3f) ; fraction of resamples with slope_ch0<=0 = %.4f"
           % (_b["ratio_a1_ch6_over_ch0"]["mean"], _b["ratio_a1_ch6_over_ch0"]["p2.5"],
              _b["ratio_a1_ch6_over_ch0"]["p97.5"], _b["ratio_a1_ch6_over_ch0"]["observed"],
              _b["ratio_a1_ch6_over_ch0"]["fraction_slope_ch0_negative"]))
LOG.append("  peak/subgrid ratio ch6/ch5 = %.3f [%.3f, %.3f] ; a1-slope ordering strictly increasing in %.4f of "
           "resamples, ch6 is the max in %.4f"
           % (_b["ratio_a1_ch6_over_ch5_peak_subgrid"]["mean"], _b["ratio_a1_ch6_over_ch5_peak_subgrid"]["p2.5"],
              _b["ratio_a1_ch6_over_ch5_peak_subgrid"]["p97.5"],
              _b["slope_ratio_rank_concordance"]["fraction_strictly_increasing"],
              _b["slope_ratio_rank_concordance"]["fraction_ch6_is_max"]))

# =====================================================================
# 5. complete-unit leave-out
# =====================================================================
loo_rep: dict[str, dict] = {}
for r in [1, 2, 3]:
    sub = rep[rep["replicate_id"] != r]
    Z, _, _ = rank1_table(sub)
    loo_rep["drop_replicate_%d" % r] = {
        "rank1_r2_49x7": float(rank1_profile(Z)["rank1_r2"]),
        "rank1_r2_392x7": float(rank1_profile(rank1_table(sub, by=GRID)[0])["rank1_r2"]),
        "amplification": ratio_from(sub),
        "n_rows_used": int(len(sub)),
    }

def score_held_out_channel(table: np.ndarray, held: int) -> tuple[float, np.ndarray]:
    cols = [c for c in CHANNELS if c != held]
    train, target = table[:, cols], table[:, held]
    mu = train.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(train - mu, full_matrices=False)
    latent = (train - mu) @ Vt[:1].T
    design = np.hstack([np.ones((len(target), 1)), latent])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(r2(target, design @ beta)), Vt[0]


B392, _, _ = rank1_table(rep, by=GRID)
loo_chan: dict[str, dict] = {}
pred_r2 = []
for held in CHANNELS:
    val, vt0 = score_held_out_channel(BASE, held)
    val392, _ = score_held_out_channel(B392, held)
    pred_r2.append(val)
    loo_chan["holdout_channel_%d" % held] = {
        "r2_vs_own_channel_variance_49x7": val,
        "r2_vs_own_channel_variance_392x7": val392,
        "latent_loading_of_held_channel_from_others": [float(v) for v in vt0],
        "n_train_channels": 6,
    }
RESULT["complete_unit_leaveout"] = {
    "definition": "rank-1 basis (leading right singular vector) and channel mean are estimated on the training "
                  "channels/units only; the held-out channel is predicted from the training latent coordinate "
                  "with an intercept refit on the held-out values; R2 is scored against the held-out channel's "
                  "own variance (no between-channel intercept credit)",
    "leave_one_replicate_id": loo_rep,
    "leave_one_channel": loo_chan,
    "leave_one_channel_summary": {
        "mean_r2": float(np.mean(pred_r2)),
        "min_r2": float(np.min(pred_r2)),
        "max_r2": float(np.max(pred_r2)),
        "channels_with_r2_lt_0.5": [int(c) for c, v in zip(CHANNELS, pred_r2) if v < 0.5],
    },
}
LOG.append("leave-one-replicate rank1 R2 (392x7 / 49x7): " + " ; ".join(
    "r%s=%.4f/%.4f" % (k[-1], v["rank1_r2_392x7"], v["rank1_r2_49x7"]) for k, v in loo_rep.items()))
LOG.append("leave-one-channel R2: " + " ; ".join("ch%d=%.4f" % (c, v) for c, v in zip(CHANNELS, pred_r2)))
LOG.append("  held-out channel mean R2=%.4f range [%.4f, %.4f]"
           % (np.mean(pred_r2), np.min(pred_r2), np.max(pred_r2)))

# =====================================================================
# 6. permutation null for the single-gain-axis claim
# =====================================================================
N_NULL = 500
rng_n = np.random.default_rng(2025)
null_r1 = np.empty(N_NULL)
null_r2 = np.empty(N_NULL)
for k in range(N_NULL):
    M = np.empty_like(BASE)
    for ch in CHANNELS:
        M[:, ch] = rng_n.permutation(BASE[:, ch])   # keeps per-channel mean and grid SD, destroys cross-channel pairing
    prof = rank1_profile(M)
    null_r1[k] = prof["rank1_r2"]
    null_r2[k] = prof["rank2_r2"]
n_ge = int((null_r1 >= BASE_RANK1).sum())
RESULT["single_gain_axis_null"] = {
    "definition": "500 synthetic 49x7 surfaces; within each channel the 49 (a1,a2) values are independently "
                  "permuted, exactly preserving that channel's mean and standard deviation but destroying any "
                  "cross-channel correspondence; rank-1 R2 recomputed each time",
    "n_simulations": N_NULL,
    "observed_rank1_r2": BASE_RANK1,
    "null_rank1_r2_mean": float(null_r1.mean()),
    "null_rank1_r2_sd": float(null_r1.std(ddof=1)),
    "null_rank1_r2_min": float(null_r1.min()),
    "null_rank1_r2_max": float(null_r1.max()),
    "null_rank1_r2_p50": float(np.percentile(null_r1, 50)),
    "null_rank1_r2_p95": float(np.percentile(null_r1, 95)),
    "null_rank2_r2_mean": float(null_r2.mean()),
    "n_null_ge_observed": n_ge,
    "empirical_p": float(n_ge / N_NULL),
    "empirical_p_plus_one": float((n_ge + 1) / (N_NULL + 1)),
    "p_resolution": 1.0 / N_NULL,
    "observed_exceeds_all_null": bool(n_ge == 0),
    "headroom_observed_minus_null_max": float(BASE_RANK1 - null_r1.max()),
}
LOG.append("null simulation (%d surfaces): null rank1 R2 mean=%.4f sd=%.4f range [%.4f, %.4f]; "
           "observed %.4f ; n_null>=observed = %d -> empirical p = %.4f (resolution %.4f)"
           % (N_NULL, null_r1.mean(), null_r1.std(ddof=1), null_r1.min(), null_r1.max(),
              BASE_RANK1, n_ge, n_ge / N_NULL, 1.0 / N_NULL))

# =====================================================================
# summary
# =====================================================================
RESULT["summary"] = {
    "rank1_claim_headline": {
        "value": BASE_RANK1,
        "holdout_channel_mean_r2": float(np.mean(pred_r2)),
        "bootstrap_ci": [RESULT["bootstrap"]["rank1_r2"]["p2.5"], RESULT["bootstrap"]["rank1_r2"]["p97.5"]],
        "null_p": RESULT["single_gain_axis_null"]["empirical_p_plus_one"],
        "rank1_r2_by_outlier_variant": {k: out_block[k]["rank1_r2_49x7"] for k in
                                        ["baseline", "drop_top1pct_cv_cells", "winsorise_p1_p99_within_channel",
                                         "drop_global_minmax_per_channel"]},
        "weakest_normalisation": min(
            [(k, v["rank1_r2"]) for k, v in norm_block.items() if isinstance(v, dict) and "rank1_r2" in v],
            key=lambda t: t[1]),
    },
    "amplification_claim_headline": {
        "value": BASE_RATIO["ratio_a1_ch6_over_ch0"],
        "ci95": [RESULT["bootstrap"]["ratio_a1_ch6_over_ch0"]["p2.5"],
                 RESULT["bootstrap"]["ratio_a1_ch6_over_ch0"]["p97.5"]],
        "peak_subgrid_ratio": RESULT["baseline"]["amplification"]["ratio_channel_to_peak"][6] /
                              RESULT["baseline"]["amplification"]["ratio_channel_to_peak"][5],
        "range_across_outlier_variants": [
            min(out_block[k]["amplification"]["ratio_a1_ch6_over_ch0"] for k in
                ["drop_top1pct_cv_cells", "winsorise_p1_p99_within_channel", "drop_global_minmax_per_channel"]),
            max(out_block[k]["amplification"]["ratio_a1_ch6_over_ch0"] for k in
                ["drop_top1pct_cv_cells", "winsorise_p1_p99_within_channel", "drop_global_minmax_per_channel"]),
        ],
    },
    "boundary_claim_headline": {
        "best_partition": best["partition"],
        "best_gap": best["observed_gap"],
        "best_perm_p": best["perm_p"],
        "nominal_gap": nominal["observed_gap"],
        "nominal_perm_p": nominal["perm_p"],
        "gap_difference_best_minus_nominal": float(best["observed_gap"] - nominal["observed_gap"]),
        "second_best_partition": second["partition"],
        "second_best_gap": second["observed_gap"],
        "indistinguishable_scale": float(second["null_sd"]),
    },
    "contradictions_and_caveats": [
        "per-channel min-max and per-channel z-score normalisations destroy the rank-1 structure "
        "(R2 %.4f and %.4f): the gain-axis claim is a property of the log/linear magnitude scale, not scale-free."
        % (norm_block["per_channel_minmax"]["rank1_r2"], norm_block["per_channel_zscore"]["rank1_r2"]),
        "the 29x amplification ratio rests on a noise-floor denominator: the peak/subgrid ratio "
        "(ch6/ch5, both on the shared rising trend) is only %.2f, and the bootstrap CI on the ratio is wide "
        "and can include zero denominators."
        % (RESULT["baseline"]["amplification"]["ratio_channel_to_peak"][6] /
           RESULT["baseline"]["amplification"]["ratio_channel_to_peak"][5]),
        "the channel-regime boundary at 0-2|3-6 is not uniquely identified: %d of the 63 two-block partitions "
        "reach perm p<0.05 and the best-vs-second-best gap difference is %.4f, below the permutation null sd "
        "(%.4f)." % (RESULT["channel_block_partitions"]["n_partitions_perm_p_lt_0.05"],
                     RESULT["channel_block_partitions"]["gap_difference_best_minus_second"], second["null_sd"]),
        "leave-one-channel out shows the rank-1 axis has no predictive content for channels 0-2 "
        "(R2 %.4f/%.4f/%.4f): the shared latent is carried by the high-gain channels 4-6."
        % tuple(pred_r2[:3]),
    ],
}

RESULT["environment"] = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "runtime_seconds": round(time.time() - T_START, 2),
    "t0": "2022-01-01",
}
LOG.append("total runtime %.1f s" % (time.time() - T_START))
save_json(RESULT, RESULTS / "t20_robustness.json")
flush(LOG, LOGS / "step5.log")
print("\n".join(LOG))