"""Step 2 - Holdout validity of the latent gain axis (complete rows, columns,
whole grid regions) plus the leave-one-out sensitivity of the amplification.

Supports Finding 1.

Run: python3 output/analysis/step2_holdout_validity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    GRID,
    LOGS,
    RESULTS,
    cell_means,
    flush,
    holdout_block,
    holdout_full_factor,
    load_panels,
    r2,
    save_json,
)

LOG: list[str] = []
frames = load_panels(LOG)
rep = frames["rep"]

cell = cell_means(rep, "ly")
tab = cell.pivot_table(index=GRID, columns="output_channel_index", values="ly")
index_frame = tab.index.to_frame(index=False).reset_index(drop=True)
values = tab.values
LOG.append(f"holdout table: {values.shape} (complete grid conditions x 7 output channels)")

# ------- complete-factor holdouts: full input rows, full columns, grid regions
holdout = {"rank1": {}, "rank2": {}}
for factor, label in [("input_axis_1_index", "holdout_full_a1_row"),
                      ("input_axis_2_index", "holdout_full_a2_column")]:
    holdout["rank1"][label] = holdout_full_factor(values, index_frame, factor, k=1)
    holdout["rank2"][label] = holdout_full_factor(values, index_frame, factor, k=2)

# whole-region holdouts (a complete 3x3 corner block of the input grid, all bins)
i1 = index_frame["input_axis_1_index"].values
i2 = index_frame["input_axis_2_index"].values
regions = {
    "corner_low_a1_low_a2": (i1 < 3) & (i2 < 3),
    "corner_high_a1_high_a2": (i1 >= 4) & (i2 >= 4),
    "corner_low_a1_high_a2": (i1 < 3) & (i2 >= 4),
    "corner_high_a1_low_a2": (i1 >= 4) & (i2 < 3),
    "strip_a1_middle": i1 == 3,
    "strip_a2_middle": i2 == 3,
}


holdout["grid_regions"] = {
    name: {"rank1": holdout_block(values, ~sel, sel, k=1), "rank2": holdout_block(values, ~sel, sel, k=2)}
    for name, sel in regions.items()
}

# ------- held-out channel: latent from 6 channels predicts the 7th
chan_hold = {}
for held in range(7):
    cols = [c for c in range(7) if c != held]
    train = values[:, cols]
    mu = train.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(train - mu, full_matrices=False)
    latent = (train - mu) @ Vt[:1].T
    target = values[:, held]
    for k, tag in [(1, "rank1"), (2, "rank2")]:
        lat = (train - mu) @ Vt[:k].T
        design = np.hstack([np.ones((len(target), 1)), lat])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        chan_hold.setdefault(tag, {})[str(held)] = r2(target, design @ beta)
    chan_hold.setdefault("intercept_only_global_mean", {})[str(held)] = r2(
        target, np.full_like(target, train.mean())
    )
holdout["holdout_channel"] = chan_hold
holdout["holdout_channel_note"] = (
    "The single held-out channel is scored against its own variance, so no between-channel "
    "intercept benefit exists. This is the strictest form of the test."
)

# ------- cross-regime transfer: low channels -> high channels and back
def latent_from(cols):
    train = values[:, cols]
    mu = train.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(train - mu, full_matrices=False)
    return (train - mu) @ Vt[:1].T


cross = {}
for name, src, dsts in [
    ("low_ch0_2_to_high", [0, 1, 2], [4, 5, 6]),
    ("high_ch4_6_to_low", [4, 5, 6], [0, 1, 2]),
    ("middle_ch3_to_low", [3], [0, 1, 2]),
    ("middle_ch3_to_high", [3], [4, 5, 6]),
]:
    lat = latent_from(src)
    cross[name] = {}
    for d in dsts:
        target = values[:, d]
        design = np.hstack([np.ones((len(target), 1)), lat])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        cross[name][str(d)] = r2(target, design @ beta)
holdout["cross_regime_transfer"] = cross

# -------- leave-one-out on the amplification ratio
sub = rep
x = np.arange(7)


def _ols_slope(xs, ys) -> float:
    """Slope of the least-squares line; uses the actual level values so that
    leave-one-level-out fits (6 points) stay valid."""
    return float(np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 1)[0])


def slopes(df) -> np.ndarray:
    cm = df.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    out = []
    for ch in range(7):
        P = cm[cm["output_channel_index"] == ch].pivot_table(
            index="input_axis_2_index", columns="input_axis_1_index", values="ly").sort_index()
        xs = P.columns.values
        out.append(float(np.mean([_ols_slope(xs, P.loc[a].sort_index().values) for a in P.index])))
    return np.array(out)


full = slopes(sub)
loo = {"full_slopes": full.tolist(), "full_ratio": float(full[6] / full[0]), "drop_a1": {}, "drop_a2": {}, "drop_bin": {}, "drop_replicate": {}}
for d in range(7):
    s = slopes(sub[sub["input_axis_1_index"] != d])
    loo["drop_a1"][str(d)] = {"ratio": float(s[6] / s[0]), "slopes": s.tolist()}
for d in range(7):
    s = slopes(sub[sub["input_axis_2_index"] != d])
    loo["drop_a2"][str(d)] = {"ratio": float(s[6] / s[0])}
for d in range(8):
    s = slopes(sub[sub["signal_bin_index"] != d])
    loo["drop_bin"][str(d)] = {"ratio": float(s[6] / s[0])}
for r in [1, 2, 3]:
    s = slopes(sub[sub["replicate_id"] != r])
    loo["drop_replicate"][str(r)] = {"ratio": float(s[6] / s[0])}
loo["ratio_range_ex_bin_drops"] = [
    float(min(v["ratio"] for v in loo["drop_a2"].values())),
    float(max(v["ratio"] for v in loo["drop_a2"].values())),
]
holdout["amplification_leave_one_out"] = loo

# rank-1 R2 stability under the same hold-outs
r1_stab = {}
for d in range(8):
    cm = sub[sub["signal_bin_index"] != d].groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    M = cm.pivot_table(index=GRID, columns="output_channel_index", values="ly").values
    X = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    r1_stab[f"drop_bin_{d}"] = float(1 - (s[1:] ** 2).sum() / (s ** 2).sum())
for d in range(7):
    cm = sub[sub["input_axis_1_index"] != d].groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    M = cm.pivot_table(index=GRID, columns="output_channel_index", values="ly").values
    X = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    r1_stab[f"drop_a1_{d}"] = float(1 - (s[1:] ** 2).sum() / (s ** 2).sum())
holdout["rank1_stability"] = r1_stab

save_json(holdout, RESULTS / "t09_holdout_validity.json")

a1v = [v["r2_vs_channel_mean"] for v in holdout["rank1"]["holdout_full_a1_row"].values()]
a2v = [v["r2_vs_channel_mean"] for v in holdout["rank1"]["holdout_full_a2_column"].values()]
LOG.append(
    "holdout full a1 rows (rank1, vs per-channel mean): mean=%.4f min=%.4f (%s)"
    % (np.mean(a1v), np.min(a1v), np.round(a1v, 4).tolist())
)
LOG.append(
    "holdout full a2 columns (rank1, vs per-channel mean): mean=%.4f min=%.4f (%s)"
    % (np.mean(a2v), np.min(a2v), np.round(a2v, 4).tolist())
)
LOG.append(
    "holdout whole grid regions (rank1, vs per-channel mean): range %.4f - %.4f ; rank2 range %.4f - %.4f"
    % (
        min(v["rank1"]["r2_vs_channel_mean"] for v in holdout["grid_regions"].values()),
        max(v["rank1"]["r2_vs_channel_mean"] for v in holdout["grid_regions"].values()),
        min(v["rank2"]["r2_vs_channel_mean"] for v in holdout["grid_regions"].values()),
        max(v["rank2"]["r2_vs_channel_mean"] for v in holdout["grid_regions"].values()),
    )
)
LOG.append("holdout a1 rank2: %s" % {k: round(v["r2_vs_channel_mean"], 4) for k, v in holdout["rank2"]["holdout_full_a1_row"].items()})
LOG.append(
    "held-out channel rank1 R2: %s"
    % {k: round(v, 4) for k, v in holdout["holdout_channel"]["rank1"].items()}
)
LOG.append(
    "cross-regime transfer R2: %s"
    % {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in cross.items()}
)
LOG.append(
    "amplification ratio ch6/ch0 under leave-one-out: full=%.1f ; drop-a2 range %.1f-%.1f ; drop-bin range %.1f-%.1f"
    % (
        loo["full_ratio"],
        loo["ratio_range_ex_bin_drops"][0],
        loo["ratio_range_ex_bin_drops"][1],
        min(v["ratio"] for v in loo["drop_bin"].values()),
        max(v["ratio"] for v in loo["drop_bin"].values()),
    )
)
LOG.append(
    "rank-1 R2 stability: drop-bin range %.4f-%.4f ; drop-a1 range %.4f-%.4f"
    % (
        min(v for k, v in r1_stab.items() if k.startswith("drop_bin")),
        max(v for k, v in r1_stab.items() if k.startswith("drop_bin")),
        min(v for k, v in r1_stab.items() if k.startswith("drop_a1")),
        max(v for k, v in r1_stab.items() if k.startswith("drop_a1")),
    )
)

flush(LOG, LOGS / "step2.log")
print("\n".join(LOG))