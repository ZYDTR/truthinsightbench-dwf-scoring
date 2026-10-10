#!/usr/bin/env python3
"""Change-point detection in increment variance for anonymous trajectories.

Method (pure numpy, contiguous time only):
  - 127 increments per trajectory.
  - 16-step blocks => 7 full 16-increment blocks + 1 tail block of 15
    (7*16 + 15 = 127). Candidate boundary s = 1..7 (after 16*s increments).
  - 8-step blocks => 15 full 8-increment blocks + 1 tail block of 7
    (15*8 + 7 = 127). Candidate boundary s = 1..15 (after 8*s increments).
  - V_pre = var(first segment), V_post = var(second segment), n_pre/n_post
    are the actual counts of increments in each segment (population variance,
    ddof=0). V_pool = (n_pre*V_pre + n_post*V_post)/(n_pre+n_post).
  - logLR = n_pre*log(V_pool/V_pre) + n_post*log(V_pool/V_post),
    variances floored at 1e-12.
  - cp estimate: boundary s* with max logLR, reported in steps as s* * block_size.
  - Label: nonstationary iff max logLR > 3.84 AND (V_post/V_pre >= 2 or
    V_pre/V_post >= 2), else stationary.

Robustness:
  a) Repeat with 8-step blocks.
  b) Threshold sensitivity (16-step): logLR-only vs logLR AND ratio >= k.
  c) Leave-one-block-out (16-step): exclude each of the 8 blocks, recompute
     best boundary and logLR on remaining 7 blocks (contiguous original order,
     n = actual increments in each block). Report fraction of nonstationary
     trajectories whose label is preserved in >= 6 of 7 exclusions, and the
     overall agreement fraction.
"""
import json
import math
import os

import numpy as np

ROOT = "/data/user/conversations/conv-48099158-ecf8-4e60-b17a-71d977ed1f64/project"
DATA = os.path.join(ROOT, "data", "anonymous_trajectories.npz")
OUT_RESULTS = os.path.join(ROOT, "output", "results")
CHIR2_1DF_095 = 3.84  # chi-squared, 1 df, 95% critical value
V_FLOOR = 1e-12


def var_from_cumsum(cum_s, cum_sq, a, b, n):
    """Population variance (ddof=0) of X[:, a:b] using precomputed cumulative
    sums. cum_s: (n_traj,), cum_sq: (n_traj,). a,b are global indices.
    Returns (n_traj,) array."""
    s = cum_s[:, b] - cum_s[:, a]
    sq = cum_sq[:, b] - cum_sq[:, a]
    v = (sq - s * s / n) / n
    return np.maximum(v, V_FLOOR)


def analyze(block_size, X, cum_s, cum_sq):
    """Two-segment LRT over contiguous blocks.

    X: (n_traj, 127) increments. cum_s, cum_sq: (n_traj, 128) cumulative sums
    (cum_s[:, i] = sum of X[:, :i]).
    Returns dict with per-trajectory logLR, V_pre, V_post, ratio, label,
    s_star, cp_steps.
    """
    n_traj, n_inc = X.shape
    n_blocks = math.ceil(n_inc / block_size)
    n = np.repeat(block_size, n_blocks)
    n[-1] = n_inc - (n_blocks - 1) * block_size
    n = np.asarray(n, dtype=float)

    # boundaries s = 1..k-1: pre = increments 0..16*s-1, post = 16*s..end
    # In terms of block boundaries, boundary s splits after block s-1.
    # pre spans blocks 0..s-1, post spans blocks s..k-1.
    # Global index of boundary s = sum(n[0:s]) = n_cum[s]
    n_cum = np.cumsum(n)  # (k,)
    boundaries = n_cum[:-1]  # (k-1,) global indices after each of blocks 0..k-2
    n_pre = boundaries  # (k-1,)
    n_post = n_inc - boundaries

    V_pre = np.empty((n_traj, len(boundaries)))
    V_post = np.empty((n_traj, len(boundaries)))
    for j, bnd in enumerate(boundaries):
        V_pre[:, j] = var_from_cumsum(cum_s, cum_sq, 0, int(bnd), n_pre[j])
        V_post[:, j] = var_from_cumsum(cum_s, cum_sq, int(bnd), n_inc, n_post[j])
    V_pool = (n_pre[np.newaxis, :] * V_pre + n_post[np.newaxis, :] * V_post) / n_inc
    V_pool = np.maximum(V_pool, V_FLOOR)
    V_pre = np.maximum(V_pre, V_FLOOR)
    V_post = np.maximum(V_post, V_FLOOR)

    logLR = (
        n_pre[np.newaxis, :] * np.log(V_pool / V_pre)
        + n_post[np.newaxis, :] * np.log(V_pool / V_post)
    )  # (n_traj, k-1)
    s_star = np.argmax(logLR, axis=1)  # (n_traj,)
    logLR_max = logLR[np.arange(n_traj), s_star]
    V_pre_star = V_pre[np.arange(n_traj), s_star]
    V_post_star = V_post[np.arange(n_traj), s_star]
    ratio = np.maximum(V_post_star / V_pre_star, V_pre_star / V_post_star)
    label = np.where(
        (logLR_max > CHIR2_1DF_095) & (ratio >= 2.0),
        "nonstationary",
        "stationary",
    )
    return {
        "cp_steps": (s_star + 1) * block_size,
        "logLR": logLR_max,
        "V_pre": V_pre_star,
        "V_post": V_post_star,
        "ratio": ratio,
        "label": label,
        "s_star": s_star,
        "blocks": X,  # not needed, just for reference
    }


def loo_analyze(block_size, X, cum_s, cum_sq, reference_label):
    """Leave-one-block-out. Exclude each of the 8 blocks (16-step), recompute
    best boundary and logLR on remaining 7 blocks. Returns (n_traj, 8) boolean
    matrix agree[i,e] = True if LOO-e label matches reference_label[i]."""
    n_traj, n_inc = X.shape
    n_blocks = 8
    n = np.repeat(block_size, n_blocks)
    n[-1] = n_inc - (n_blocks - 1) * block_size
    n = np.asarray(n, dtype=float)
    # block start indices (k+1 elements: starts of each block + end of last)
    n_cum = np.concatenate([[0], np.cumsum(n)])  # (k+1,)
    starts = n_cum  # (k+1,)

    agree = np.zeros((n_traj, n_blocks), dtype=bool)
    for e in range(n_blocks):
        keep = [b for b in range(n_blocks) if b != e]
        # Build a new trajectory from the kept blocks (contiguous, in order)
        # The kept blocks form a contiguous segment of the original trajectory,
        # EXCEPT the excluded block is in the middle. So we can't just use a
        # sub-slice. We need to concatenate the kept blocks.
        # For efficiency, precompute the concatenation.
        kept_X = np.concatenate([X[:, int(starts[b]):int(starts[b+1])] for b in keep], axis=1)
        # Now compute cumsums for kept_X
        kept_cum_s = np.cumsum(kept_X, axis=1)
        kept_cum_s = np.concatenate([np.zeros((n_traj, 1)), kept_cum_s], axis=1)  # (n_traj, n_kept+1)
        kept_cum_sq = np.cumsum(kept_X ** 2, axis=1)
        kept_cum_sq = np.concatenate([np.zeros((n_traj, 1)), kept_cum_sq], axis=1)
        n_kept = n[keep]
        n_kept_cum = np.concatenate([[0.0], np.cumsum(n_kept)])  # (8,)
        boundaries = n_kept_cum[1:-1]  # (6,) boundaries in the kept sequence
        n_pre = boundaries
        n_post = n_inc - n[keep].sum() - 0  # wait, this is wrong
        # The total number of increments in the kept sequence is n_inc - n[e]
        n_kept_total = n[keep].sum()
        n_post = n_kept_total - boundaries

        V_pre = np.empty((n_traj, len(boundaries)))
        V_post = np.empty((n_traj, len(boundaries)))
        n_kept_total_i = int(n_kept_total)
        for j, bnd in enumerate(boundaries):
            V_pre[:, j] = var_from_cumsum(kept_cum_s, kept_cum_sq, 0, int(bnd), n_pre[j])
            V_post[:, j] = var_from_cumsum(kept_cum_s, kept_cum_sq, int(bnd), n_kept_total_i, n_post[j])
        V_pool = (n_pre[np.newaxis, :] * V_pre + n_post[np.newaxis, :] * V_post) / n_kept_total
        V_pool = np.maximum(V_pool, V_FLOOR)
        V_pre = np.maximum(V_pre, V_FLOOR)
        V_post = np.maximum(V_post, V_FLOOR)

        logLR = (
            n_pre[np.newaxis, :] * np.log(V_pool / V_pre)
            + n_post[np.newaxis, :] * np.log(V_pool / V_post)
        )
        s_star = np.argmax(logLR, axis=1)
        logLR_max = logLR[np.arange(n_traj), s_star]
        V_pre_star = V_pre[np.arange(n_traj), s_star]
        V_post_star = V_post[np.arange(n_traj), s_star]
        ratio = np.maximum(V_post_star / V_pre_star, V_pre_star / V_post_star)
        label = np.where((logLR_max > CHIR2_1DF_095) & (ratio >= 2.0),
                         "nonstationary", "stationary")
        agree[:, e] = (label == reference_label)
    return agree


def main():
    os.makedirs(OUT_RESULTS, exist_ok=True)

    X = np.diff(np.load(DATA)["trajectories"], axis=1)
    cum_s = np.cumsum(X, axis=1)
    cum_s = np.concatenate([np.zeros((X.shape[0], 1)), cum_s], axis=1)
    cum_sq = np.cumsum(X ** 2, axis=1)
    cum_sq = np.concatenate([np.zeros((X.shape[0], 1)), cum_sq], axis=1)

    # ---------- main 16-step block analysis ----------
    res16 = analyze(16, X, cum_s, cum_sq)
    n_ns16 = int(np.sum(res16["label"] == "nonstationary"))

    def write_csv(path, res):
        with open(path, "w") as f:
            f.write("traj_id,cp_steps,logLR,V_pre,V_post,ratio,label\n")
            for i in range(len(res["cp_steps"])):
                f.write(
                    f"{i},{int(res['cp_steps'][i])},{res['logLR'][i]:.6f},"
                    f"{res['V_pre'][i]:.8f},{res['V_post'][i]:.8f},"
                    f"{res['ratio'][i]:.6f},{res['label'][i]}\n"
                )

    write_csv(os.path.join(OUT_RESULTS, "change_points.csv"), res16)

    # ---------- (a) 8-step block analysis ----------
    res8 = analyze(8, X, cum_s, cum_sq)
    n_ns8 = int(np.sum(res8["label"] == "nonstationary"))
    write_csv(os.path.join(OUT_RESULTS, "change_points_8.csv"), res8)

    # ---------- (b) threshold sensitivity (16-step) ----------
    logLR = res16["logLR"]
    ratio = res16["ratio"]
    mask_lr = logLR > CHIR2_1DF_095
    sens = {
        "logLR_only": int(np.sum(mask_lr)),
        "ratio2": int(np.sum(mask_lr & (ratio >= 2.0))),
        "ratio3": int(np.sum(mask_lr & (ratio >= 3.0))),
        "ratio5": int(np.sum(mask_lr & (ratio >= 5.0))),
        "ratio10": int(np.sum(mask_lr & (ratio >= 10.0))),
    }

    # ---------- (c) leave-one-block-out (16-step) ----------
    agree = loo_analyze(16, X, cum_s, cum_sq, res16["label"])
    agree_count = agree.sum(axis=1)
    nonstat_mask = res16["label"] == "nonstationary"
    n_nonstat = int(np.sum(nonstat_mask))
    nonstat_ge6 = float(np.mean(agree[nonstat_mask].sum(axis=1) >= 6)) if n_nonstat else 0.0
    overall_agree_pairs = float(agree.sum() / (X.shape[0] * 8))

    # ---------- aggregates ----------
    cp_steps = res16["cp_steps"]
    cp_median = float(np.median(cp_steps[nonstat_mask])) if n_nonstat else float("nan")
    cp_min = float(np.min(cp_steps))
    cp_max = float(np.max(cp_steps))
    pre_var_median = float(np.median(res16["V_pre"][nonstat_mask])) if n_nonstat else float("nan")
    post_var_median = float(np.median(res16["V_post"][nonstat_mask])) if n_nonstat else float("nan")
    ratio_median_nonstat = float(np.median(res16["ratio"][nonstat_mask])) if n_nonstat else float("nan")

    agg = {
        "n_nonstationary_16": int(n_ns16),
        "n_nonstationary_8": int(n_ns8),
        "cp_median": cp_median,
        "cp_min": cp_min,
        "cp_max": cp_max,
        "pre_var_median": pre_var_median,
        "post_var_median": post_var_median,
        "ratio_median_nonstat": ratio_median_nonstat,
        "threshold_sensitivity": sens,
        "leave_one_block_agreement": overall_agree_pairs,
        "leave_one_block_nonstat_ge6_frac": nonstat_ge6,
    }

    with open(os.path.join(OUT_RESULTS, "change_point_robustness.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print(json.dumps(agg, indent=2))
    print(f"nonstationary_16={n_ns16}, nonstationary_8={n_ns8}")
    print(f"leave_one_block_nonstat_ge6_frac={nonstat_ge6:.4f}")
    print(f"leave_one_block_overall_pairs={overall_agree_pairs:.4f}")


if __name__ == "__main__":
    main()
