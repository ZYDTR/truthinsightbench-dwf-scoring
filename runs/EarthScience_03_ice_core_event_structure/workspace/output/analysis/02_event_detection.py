#!/usr/bin/env python3
"""02_event_detection.py -- how reproducible are "sulfate events" across definitions?

DATA_GUIDE.md warning: "Peak detection depends on background and window
definitions, which must be reported and perturbed."

We therefore do not pick one detector.  We build a (background, threshold)
grid, count events at every grid point, and measure how well each variant
recovers the 15 curator-annotated horizons in ``c07_signal``.

A crucial subtlety that decides the interpretation: a *local-maximum* rule can
only ever return a local maximum.  We report the ceiling imposed by that
constraint explicitly, because it bounds recall independently of the threshold.

Outputs (under output/results/):
  detector_grid.csv
  annotated_event_excess.csv
  label_permutation_events.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, annotated_mask, blocks, load, log, save_csv, save_json  # noqa: E402

log("=" * 78)
log("02_event_detection.py -- detector sensitivity and annotation reproducibility")
log("=" * 78)

df = load()
n = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
ann = annotated_mask(df)
AI = np.where(ann)[0]
labels = df["c07_signal"].to_numpy()

rng = np.random.default_rng(SEED)
nan_so = np.isnan(SO)
# The single missing sulfate value is interpolated for detector runs only; the
# annotated-row statistics below are computed on the raw values.
SO_f = SO.copy()
SO_f[nan_so] = np.interp(np.where(nan_so)[0], np.where(~nan_so)[0], SO[~nan_so])


# --------------------------------------------------------------------------
# background definitions
# --------------------------------------------------------------------------
def bg_rolling(x, win, stat="median"):
    s = pd.Series(x)
    r = s.rolling(win, center=True, min_periods=max(2, win // 4))
    return (r.median() if stat == "median" else r.mean()).to_numpy()


def bg_linear(x, win, stat="median"):
    """Detrend by an ordinary least-squares line fitted inside the window."""
    out = np.full(len(x), np.nan)
    half = win // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        t = np.arange(lo, hi, dtype=float)
        if len(t) < 4:
            continue
        p = np.polyfit(t, x[lo:hi], 1)
        out[i] = np.polyval(p, i)
    return out


def bg_quantile(x, win, q=0.25):
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=max(2, win // 4)).quantile(q).to_numpy()


BACKGROUNDS = {
    "rolling_median": bg_rolling,
    "rolling_mean": lambda x, w: bg_rolling(x, w, "mean"),
    "rolling_q25": bg_quantile,
    "linear_detrend": bg_linear,
}

WINDOWS = [11, 21, 51, 101, 201, 401]
KS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

is_local_max = (SO_f > np.roll(SO_f, 1)) & (SO_f >= np.roll(SO_f, -1))
is_local_max[0] = False
is_local_max[-1] = False
ceiling = int(is_local_max[AI].sum())
log(f"local-maximum ceiling  : {ceiling}/{len(AI)} annotated horizons are strict local maxima of c04")
log(f"                         -> no local-maximum detector can recall more than {ceiling}/{len(AI)}")


def detect(excess, k, use_local_max=True):
    scale = np.nanstd(excess)
    m = excess > k * scale
    if use_local_max:
        m &= is_local_max
    return m


rows = []
for bname, bfun in BACKGROUNDS.items():
    for win in WINDOWS:
        bg = bfun(SO_f, win)
        excess = SO_f - bg
        for k in KS:
            sel = detect(excess, k)
            idx = np.where(sel)[0]
            hit = len(set(idx) & set(AI))
            fp = len(idx) - hit
            rows.append(
                {
                    "background": bname,
                    "window_samples": win,
                    "window_depth_m": float(win * 0.021),
                    "k_sigma": k,
                    "n_detected": int(sel.sum()),
                    "n_annotated_recalled": int(hit),
                    "recall_of_15": round(hit / 15, 4),
                    "false_positives": int(fp),
                    "precision": round(hit / len(idx), 4) if len(idx) else float("nan"),
                    "agreement_index": int(hit) - int(fp),
                }
            )
grid = pd.DataFrame(rows)
grid = grid.sort_values(["agreement_index", "n_annotated_recalled"], ascending=False)
save_csv(grid, "detector_grid.csv")

log("")
log("detector grid: %d (background, window, threshold) combinations" % len(grid))
log("best 12 by agreement index (recalled - false positives):")
for _, r in grid.head(12).iterrows():
    log(
        "  %-15s win=%3d k=%.1f -> n=%3d recalled=%2d/15 FP=%3d precision=%.2f"
        % (r.background, r.window_samples, r.k_sigma, r.n_detected,
           r.n_annotated_recalled, r.false_positives, r.precision)
    )

det = grid.n_detected
log("")
log("count spread across the whole grid : min=%d  median=%.0f  max=%d" % (det.min(), det.median(), det.max()))
log("recall spread                      : min=%d  max=%d  (of 15)" % (grid.n_annotated_recalled.min(), grid.n_annotated_recalled.max()))
log("variants with recall < 15          : %d/%d" % ((grid.n_annotated_recalled < 15).sum(), len(grid)))
log("variants with zero false positives : %d/%d" % ((grid.false_positives == 0).sum(), len(grid)))
best_recall = grid[grid.n_annotated_recalled == grid.n_annotated_recalled.max()]
log("variants reaching max recall %d     : %d, with %d-%d false positives"
    % (grid.n_annotated_recalled.max(), len(best_recall),
       best_recall.false_positives.min(), best_recall.false_positives.max()))

# --------------------------------------------------------------------------
# excess of the 15 annotated horizons, relative to a nominal background
# --------------------------------------------------------------------------
bg101 = bg_rolling(SO_f, 101)
exc101 = SO_f - bg101
rank_excess = pd.Series(-exc101).rank(method="min").to_numpy()
rank_raw = pd.Series(-SO_f).rank(method="min").to_numpy()

ex_rows = []
for i in AI:
    j = int(rank_excess[i])
    ex_rows.append(
        {
            "row": int(i),
            "depth_top_m": float(D[i]),
            "sulfate_ug_kg": float(SO[i]),
            "background_median_101": float(bg101[i]),
            "excess_ug_kg": float(exc101[i]),
            "rank_all_rows_by_excess": j,
            "n_rows": n,
            "rank_all_rows_by_raw": int(rank_raw[i]),
            "is_local_max": bool(is_local_max[i]),
            "is_window_max_pm5": bool(SO[i] >= np.nanmax(SO_f[max(0, i - 5):i + 6])),
            "is_window_max_pm50": bool(SO[i] >= np.nanmax(SO_f[max(0, i - 50):i + 51])),
            "label": labels[i],
        }
    )
ev = pd.DataFrame(ex_rows)
save_csv(ev, "annotated_event_excess.csv")
log("")
log("the 15 annotated horizons, ranked by excess over a 101-sample rolling-median background:")
for _, r in ev.iterrows():
    log(
        "  row %4d d=%6.2f sulf=%8.3f bg=%7.2f excess=%8.2f  rank %4d/%d  %s"
        % (r.row, r.depth_top_m, r.sulfate_ug_kg, r.background_median_101,
           r.excess_ug_kg, r.rank_all_rows_by_excess, n, r.label)
    )

n_top30 = int((ev.rank_all_rows_by_excess <= 30).sum())
log("")
log("annotated horizons inside the top-30 by excess : %d/15" % n_top30)
log("expected under a random selection of 15 of %d  : %.2f" % (n, 15 * 30 / n))

# permutation test on the enrichment
obs = n_top30
perm = np.array([len(set(rng.choice(n, 15, replace=False)) & set(range(30))) for _ in range(20000)])
p_perm = float((perm >= obs).mean())
log("permutation p(excess-top30 enrichment)        : %.5f  (20000 permutations, seed %d)" % (p_perm, SEED))

# --------------------------------------------------------------------------
# block-leave-one-out: is the enrichment carried by a single contiguous zone?
# --------------------------------------------------------------------------
block_rows = []
for k in (5, 10, 20):
    ws = grid[(grid.background == "rolling_median") & (grid.window_samples == 101) & (grid.k_sigma == 2.0)]
    base_ids = np.where(detect(SO_f - bg101, 2.0))[0]
    base_hit = len(set(base_ids) & set(AI))
    for b_i, b in enumerate(blocks(n, k)):
        keep = np.setdiff1d(np.arange(n), b)
        sub_hit = len(set(base_ids) & set(AI) & set(keep))
        tot_ann = len(set(AI) & set(keep))
        block_rows.append(
            {
                "n_blocks": k,
                "block_index": b_i,
                "row_start": int(b[0]),
                "row_end": int(b[-1]),
                "depth_start_m": float(D[b[0]]),
                "depth_end_m": float(D[b[-1]]),
                "annotated_in_block": int(len(set(AI) & set(b))),
                "recalled_in_block": int(len(set(base_ids) & set(AI) & set(b))),
                "recall_after_dropping_block": round(sub_hit / tot_ann, 4) if tot_ann else float("nan"),
            }
        )
blk = pd.DataFrame(block_rows)
save_csv(blk, "block_leave_one_out_detector.csv")
log("")
log("block leave-one-out at (rolling_median, win=101, k=2), baseline recall %d/15:" % base_hit)
for k in (5, 10, 20):
    sub = blk[blk.n_blocks == k]
    log(
        "  %2d contiguous blocks: recall after dropping a block ranges %.2f - %.2f"
        % (k, sub.recall_after_dropping_block.min(), sub.recall_after_dropping_block.max())
    )

summary = {
    "data_rows": n,
    "annotated_horizons": int(len(AI)),
    "local_max_ceiling": ceiling,
    "grid_combinations": len(grid),
    "detected_count_min": int(det.min()),
    "detected_count_median": float(det.median()),
    "detected_count_max": int(det.max()),
    "recall_min": int(grid.n_annotated_recalled.min()),
    "recall_max": int(grid.n_annotated_recalled.max()),
    "n_variants_with_full_recall": int((grid.n_annotated_recalled == 15).sum()),
    "n_top30_by_excess": n_top30,
    "expected_top30_by_chance": round(15 * 30 / n, 3),
    "permutation_p_top30": p_perm,
    "best_agreement_variant": grid.iloc[0][
        ["background", "window_samples", "k_sigma", "n_detected",
         "n_annotated_recalled", "false_positives"]
    ].to_dict(),
}
save_json(summary, "event_detection_summary.json")