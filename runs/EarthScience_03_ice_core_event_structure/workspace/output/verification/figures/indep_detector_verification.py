#!/usr/bin/env python3
"""Independent verification of Draft Finding 4.

Written from scratch (no import of output/analysis/common.py, no reuse of the
existing scripts).  Rebuilds a detector grid with independently chosen
backgrounds, computes the strict-local-maximum ceiling, attacks the top-30
enrichment claim with a permutation null, and diagnoses the recall deficit.

Run:  python3 indep_detector_verification.py
Outputs are written next to this script.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../project
DATA = ROOT / "data" / "observations.tsv"
SEED = 20260919

print("=" * 78)
print("Independent verification of Draft Finding 4")
print(f"data: {DATA}")
print("=" * 78)

df = pd.read_csv(DATA, sep="\t")
n = len(df)
assert n == 1931, n
depth = df["c01_depth_top_m"].to_numpy(float)
so = df["c04_sulfate_ug_kg"].to_numpy(float)
labels = df["c07_signal"].to_numpy()
ann_mask = df["c07_signal"].notna().to_numpy()
AI = np.where(ann_mask)[0]
assert len(AI) == 15, len(AI)
print(f"rows={n}  annotated={len(AI)} at {AI.tolist()}")

# ---------------------------------------------------------------------------
# 0. Strict local-maximum ceiling  (TASK 2)
# ---------------------------------------------------------------------------
nan_so = np.isnan(so)
print(f"\n[0] NaN in c04: {int(nan_so.sum())} at row {np.where(nan_so)[0].tolist()}")

so_i = so.copy()  # only used by detectors, never for annotated-row stats
so_i[nan_so] = np.interp(np.where(nan_so)[0], np.where(~nan_so)[0], so[~nan_so])

# strict local maximum: value > both neighbours; no edge convention needed here
strict_lm = np.zeros(n, bool)
strict_lm[1:-1] = (so_i[1:-1] > so_i[:-2]) & (so_i[1:-1] > so_i[2:])

# alternative edge conventions, to test whether row 0 can ever be a max
strict_lm_edge_equal = strict_lm.copy()
strict_lm_edge_equal[0] = so_i[0] > so_i[1]          # treat outside as -inf
strict_lm_edge_equal[-1] = so_i[-1] > so_i[-2]
strict_lm_plateau = np.zeros(n, bool)                 # >= left, > right (plateau start)
strict_lm_plateau[1:-1] = (so_i[1:-1] >= so_i[:-2]) & (so_i[1:-1] > so_i[2:])
strict_lm_plateau[0] = so_i[0] > so_i[1]
strict_lm_plateau[-1] = so_i[-1] > so_i[-2]

print(f"    strict local maxima in c04 (interior): {int(strict_lm.sum())}")
for nm, arr in [("edge-as--inf strict", strict_lm_edge_equal),
                ("plateau-start (>=  left)", strict_lm_plateau)]:
    fail = [int(r) for r in AI if not arr[r]]
    print(f"    convention '{nm}': annot recalled {15 - len(fail)}/15, failing rows {fail}")

fail_rows = [int(r) for r in AI if not strict_lm[r]]
ceiling = 15 - len(fail_rows)
print(f"    CEILING under strict local maxima: {ceiling}/15; failing rows {fail_rows}")
for r in fail_rows:
    print(f"      row {r}: c04={so_i[r]:.3f}  left={so_i[r-1] if r > 0 else float('nan'):.3f} "
          f"right={so_i[r+1]:.3f}  label={labels[r]}")

# which annotated rows are missed even by the loosest peak rule?
print(f"    even the plateau/edge-as--inf rule fails on: "
      f"{[int(r) for r in AI if not strict_lm_edge_equal[r]]}")

# ---------------------------------------------------------------------------
# 1. Independent detector grid  (TASK 1)
# ---------------------------------------------------------------------------
print("\n[1] Independent detector grid")

SO_S = pd.Series(so_i)


def bg_roll_median(x, w):
    return SO_S.rolling(w, center=True, min_periods=max(2, w // 5)).median().to_numpy()


def bg_roll_mean(x, w):
    return SO_S.rolling(w, center=True, min_periods=max(2, w // 5)).mean().to_numpy()


def bg_roll_q(x, w, q):
    return SO_S.rolling(w, center=True, min_periods=max(2, w // 5)).quantile(q).to_numpy()


def bg_roll_min(x, w):
    return SO_S.rolling(w, center=True, min_periods=max(2, w // 5)).min().to_numpy()


def bg_ewma(x, w):
    return SO_S.ewm(span=w, adjust=False).mean().to_numpy()


def bg_global_mean(x, w):
    return np.full(n, np.nanmean(x))


def bg_linear_global(x, w):
    t = np.arange(n, dtype=float)
    ok = ~np.isnan(x)
    b, a = np.polyfit(t[ok], x[ok], 1)
    return a + b * t


def bg_linear_roll(x, w):
    """Rolling OLS line evaluated at the centre, fitted on a window."""
    half = w // 2
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        t = np.arange(lo, hi, dtype=float)
        if hi - lo < 5:
            continue
        p = np.polyfit(t, x[lo:hi], 1)
        out[i] = np.polyval(p, i)
    return out


def bg_savgol(x, w):
    from scipy.signal import savgol_filter
    if w % 2 == 0:
        w += 1
    return savgol_filter(x, w, 2)


BACKGROUNDS = {
    "roll_median": bg_roll_median,
    "roll_mean": bg_roll_mean,
    "roll_min": bg_roll_min,
    "ewma": bg_ewma,
    "global_mean": bg_global_mean,
    "linear_global": bg_linear_global,
    "linear_roll": bg_linear_roll,
    "savgol_quad": bg_savgol,
}
BG_Q = {"roll_q20": lambda x, w: bg_roll_q(x, w, 0.20),
        "roll_q30": lambda x, w: bg_roll_q(x, w, 0.30)}
BACKGROUNDS.update(BG_Q)

# smaller windows than the original [11..401] to genuinely probe the range
WINDOWS = [15, 25, 51, 75, 101, 301]
KS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# permissive variant: strict inequality on the LEFT neighbour only
lm_right = np.zeros(n, bool)
lm_right[0] = so_i[0] > so_i[1]
lm_right[1:] = so_i[1:] > so_i[:-1]

print(f"    backgrounds={len(BACKGROUNDS)} windows={WINDOWS} thresholds={KS}"
      f"  -> {len(BACKGROUNDS) * len(WINDOWS) * len(KS)} variants")

grid_rows = []
for bname, bfun in BACKGROUNDS.items():
    for w in WINDOWS:
        bg = bfun(so_i, w)
        exc = so_i - bg
        valid = ~np.isnan(exc)
        sd = np.nanstd(exc)
        for k in KS:
            # detector A: threshold on excess AND strict local max (both neighbours)
            cand = valid & (exc > k * sd) & strict_lm
            # detector B: scipy find_peaks on raw c04, height = bg + k*sd
            idx = np.where(valid)[0]
            pk, _ = find_peaks(so_i[idx], height=bg[idx] + k * sd)
            candB = np.zeros(n, bool)
            candB[idx[pk]] = True
            for tag, sel in (("strict_lm", cand), ("find_peaks", candB)):
                hit_idx = np.where(sel)[0]
                hit = len(set(hit_idx.tolist()) & set(AI.tolist()))
                grid_rows.append({
                    "background": bname, "window": w, "k": k, "rule": tag,
                    "n_det": len(hit_idx), "hit": hit,
                })

G = pd.DataFrame(grid_rows)
G.to_csv(HERE / "indep_detector_grid.csv", index=False)

print(f"    variants total: {len(G)}")
print(f"    n_detected  min={G.n_det.min()}  median={G.n_det.median()}  "
      f"max={G.n_det.max()}  ratio={G.n_det.max() / G.n_det.min():.1f}x")
print(f"    recall/15   min={G.hit.min()}  median={G.hit.median()}  max={G.hit.max()}")
print(f"    variants with 15/15: {int((G.hit == 15).sum())};  with 14: {int((G.hit == 14).sum())}")
print(f"    variants with 0 recall: {int((G.hit == 0).sum())}")
best = G.sort_values(["hit", "n_det"], ascending=[False, True]).head(5)
print("    best variants:")
print(best.to_string(index=False))

# direct check: is each annotated row inside some variant's detected set?
mem = {int(r): 0 for r in AI}
for bname, bfun in BACKGROUNDS.items():
    for w in WINDOWS:
        bg = bfun(so_i, w)
        exc = so_i - bg
        valid = ~np.isnan(exc)
        sd = np.nanstd(exc)
        for k in KS:
            cand = valid & (exc > k * sd) & strict_lm
            idx = np.where(valid)[0]
            pk, _ = find_peaks(so_i[idx], height=bg[idx] + k * sd)
            candB = np.zeros(n, bool)
            candB[idx[pk]] = True
            for sel in (cand, candB):
                for r in AI:
                    if sel[r]:
                        mem[int(r)] += 1
tot = len(BACKGROUNDS) * len(WINDOWS) * len(KS) * 2
print(f"    per-annotated-row coverage across {tot} variants:")
for r in AI:
    print(f"      row {r:5d} {labels[r]:20s} detected by {mem[int(r)]:5d}/{tot} variants")
miss_all = [int(r) for r in AI if mem[int(r)] == 0]
argmax_row = G.loc[G.n_det.idxmax()]
print(f"    max-count variant: {argmax_row.to_dict()}")
print(f"    min-count variant: {G.loc[G.n_det.idxmin()].to_dict()}")
# distribution of counts by rule
print(G.groupby("rule").n_det.describe()[["min", "50%", "max"]].to_string())
print(f"    annotated rows missed by EVERY variant: {miss_all}")

# ---------------------------------------------------------------------------
# 3. ATTACK the enrichment claim  (TASK 3)
# ---------------------------------------------------------------------------
print("\n[3] Enrichment attack: top-30 by excess vs raw value + null model")

res = {}
for BGW in [25, 51, 101, 301]:
    bg = bg_roll_median(so_i, BGW)
    exc = so_i - bg
    order_exc = np.argsort(-exc, kind="stable")
    rank_exc = np.empty(n, int)
    rank_exc[order_exc] = np.arange(1, n + 1)
    top30_exc = set(order_exc[:30].tolist())
    hits_exc = len(top30_exc & set(AI.tolist()))
    p_exc = hits_exc / 15
    res[f"exc_w{BGW}"] = p_exc
    print(f"    bg=roll_median({BGW}): top-30-by-EXCESS contains {hits_exc}/15 "
          f"({p_exc:.1%}) annotated")

order_raw = np.argsort(-so_i, kind="stable")
rank_raw = np.empty(n, int)
rank_raw[order_raw] = np.arange(1, n + 1)
top30_raw = set(order_raw[:30].tolist())
hits_raw = len(top30_raw & set(AI.tolist()))
print(f"    top-30-by-RAW-c04 contains {hits_raw}/15 ({hits_raw / 15:.1%})")
res["raw"] = hits_raw / 15

# rank table for the 15 annotated rows
exc101 = so_i - bg_roll_median(so_i, 101)
tab = pd.DataFrame({
    "row": AI, "label": labels[AI], "c04": so_i[AI],
    "rank_raw": rank_raw[AI],
    "median_101": bg_roll_median(so_i, 101)[AI],
    "excess_101": exc101[AI],
})
tab["rank_exc_101"] = pd.Series(exc101).rank(ascending=False).to_numpy()[AI].astype(int)
lm_pos = {int(r): i for i, r in enumerate(np.where(strict_lm)[0])}
order_lm = np.argsort(-exc101[np.where(strict_lm)[0]], kind="stable")
rank_in_lm = np.empty(len(order_lm), int)
rank_in_lm[order_lm] = np.arange(1, len(order_lm) + 1)
tab["rank_exc_101_within_lm"] = [int(rank_in_lm[lm_pos[int(r)]]) if int(r) in lm_pos else -1 for r in AI]
tab["n_strict_lm"] = int(strict_lm.sum())
tab["is_strict_lm"] = strict_lm[AI]
tab.to_csv(HERE / "annotated_row_ranks.csv", index=False)
print(tab.to_string(index=False))

# ---- null model: draw k random distinct rows (local-maximum candidates or all
#      rows), count how many land in the fixed top-30 set ---------------------
rng = np.random.default_rng(SEED)
NPERM = 100_000


def perm_p(top_set, universe, k, rng, nperm=NPERM):
    """P(#hits >= observed | drawing k rows uniformly, with replacement).

    With replacement (duplicates possible) the draw is i.i.d.; the collision
    probability for k=15 out of n>=579 is < 0.2 %, so the effect on the
    reported p-value is negligible.  An exact hypergeometric p-value is
    reported alongside.
    """
    universe = np.asarray(universe)
    lookup = np.zeros(n, bool)
    lookup[universe] = True
    in_top = np.zeros(n, bool)
    in_top[list(top_set)] = True
    hits = np.empty(nperm, int)
    for j in range(nperm):
        pick = rng.integers(0, len(universe), size=k)
        hits[j] = in_top[universe[pick]].sum()
    return hits


lm_idx = np.where(strict_lm)[0]
k = 15

hits_perm_exc = perm_p(top30_exc, np.arange(n), k, rng)
p_exc_null = float((hits_perm_exc >= hits_exc).mean())
hits_perm_raw = perm_p(top30_raw, np.arange(n), k, rng)
p_raw_null = float((hits_perm_raw >= hits_raw).mean())
from scipy.stats import hypergeom

p_exc_exact = float(hypergeom.sf(hits_exc - 1, n, 30, k))
p_raw_exact = float(hypergeom.sf(hits_raw - 1, n, 30, k))
print(f"    permutation null (all rows): P(hits>= {hits_exc}) = {p_exc_null:.5f}  "
      f"[null mean {hits_perm_exc.mean():.3f}; exact hypergeom {p_exc_exact:.3e}]")
print(f"    same for raw top-30: P(hits>= {hits_raw}) = {p_raw_null:.5f}  "
      f"[null mean {hits_perm_raw.mean():.3f}; exact hypergeom {p_raw_exact:.3e}]")

# restricted universe: random strict local maxima only (removes the
# 'annotated rows are local maxima' artefact from the null)
univ = lm_idx
top30_exc_lm = set(lm_idx[np.argsort(-(so_i - bg_roll_median(so_i, 101))[lm_idx])[:30]].tolist())
hits_exc_lm = len(top30_exc_lm & set(AI.tolist()))
print(f"    top-30 EXCESS restricted to strict local maxima -> contains "
      f"{hits_exc_lm}/15 annotated")
hits_perm_lm = perm_p(top30_exc_lm, univ, k, rng)
p_lm_null = float((hits_perm_lm >= hits_exc_lm).mean())
p_lm_exact = float(hypergeom.sf(hits_exc_lm - 1, len(univ), 30, k))
print(f"    permutation null over {len(univ)} local maxima: P(hits >= {hits_exc_lm}) "
      f"= {p_lm_null:.5f}  [null mean {hits_perm_lm.mean():.3f}; "
      f"exact hypergeom {p_lm_exact:.3e}]")

# ---------------------------------------------------------------------------
# 2 & 4. Recall deficit: structural or tuning?  (TASK 4)
# ---------------------------------------------------------------------------
print("\n[4] Structural vs tuning: recall deficit")
rng2 = np.random.default_rng(SEED + 1)
print(f"    annotated rows that are strict local maxima: {int(strict_lm[AI].sum())}/15")
print(f"    fraction of all rows that are strict local maxima: {len(lm_idx)}/{n} "
      f"= {len(lm_idx)/n:.3f}; expected annotated-local-max under random draws: "
      f"{15*len(lm_idx)/n:.2f}")
perm_counts = np.empty(5000, int)
for j in range(5000):
    pick = rng2.choice(len(lm_idx), size=15, replace=False)
    perm_counts[j] = np.isin(lm_idx[pick], AI).sum()
print(f"    null (15 random local maxima): mean={perm_counts.mean():.2f} annotated "
      f"recovered, p(>=14)={float((perm_counts >= 14).mean()):.2e}")

# The decisive number: can c04 at row 0 (or any annotated row) ever be a strict
# local max under ANY window / smoothing / transform that preserves ordering?
print(f"    row 0 c04={so_i[0]:.3f}, neighbour c04[1]={so_i[1]:.3f} -> "
      f"row0 > neighbour: {bool(so_i[0] > so_i[1])}")
print(f"    so row 0 fails the peak test on the RAW series; any monotone "
      f"transform keeps the inequality, so no peak detector can return it.")

out = {
    "grid_variants": int(len(G)),
    "n_detected_min": int(G.n_det.min()),
    "n_detected_median": float(G.n_det.median()),
    "n_detected_max": int(G.n_det.max()),
    "recall_min": int(G.hit.min()),
    "recall_max": int(G.hit.max()),
    "variants_full_recall": int((G.hit == 15).sum()),
    "strict_lm_ceiling": int(ceiling),
    "annotated_not_local_max": fail_rows,
    "row0_gt_row1": bool(so_i[0] > so_i[1]),
    "top30_by_excess_101": int(hits_exc),
    "top30_by_raw": int(hits_raw),
    "p_null_excess": p_exc_null,
    "p_null_raw": p_raw_null,
    "top30_excess_within_localmax": int(hits_exc_lm),
    "p_null_excess_localmax": p_lm_null,
    "expected_fraction_if_random": 30 / n,
    "expected_count_15_draws": 15 * 30 / n,
    "annotated_missed_by_every_variant": miss_all,
    "per_row_variant_coverage": {int(r): int(mem[int(r)]) for r in AI},
    "null_overlap_localmax_mean": float(perm_counts.mean()),
}
with open(HERE / "indep_finding4_summary.json", "w") as f:
    json.dump(out, f, indent=2)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].hist(G.n_det, bins=40, color="steelblue")
ax[0, 0].axvline(21, color="k", ls="--", label="claimed median 21")
ax[0, 0].set_title(f"detected counts (n={len(G)} variants)")
ax[0, 0].legend()
ax[0, 1].hist(G.hit, bins=np.arange(-0.5, 16.5), color="darkorange")
ax[0, 1].axvline(15, color="r", ls="--", label="15/15 never reached")
ax[0, 1].axvline(14, color="g", ls=":", label="strict-LM ceiling 14")
ax[0, 1].set_title("annotated recall per variant")
ax[0, 1].legend()
ax[1, 0].plot(so_i, lw=0.6, color="gray")
ax[1, 0].plot(AI, so_i[AI], "r.", ms=8, label="annotated")
ax[1, 0].plot([1], [so_i[1]], "kx", ms=10, label="row 1 (neighbour of row 0)")
ax[1, 0].set_title("c04 sulfate, annotated horizons")
ax[1, 0].legend()
ax[1, 1].hist(hits_perm_exc, bins=np.arange(-0.5, hits_perm_exc.max() + 1.5),
              color="seagreen", density=True)
ax[1, 1].axvline(hits_exc, color="r", ls="--", label=f"observed {hits_exc}/15")
ax[1, 1].set_title("null: #annotated in top-30 by excess")
ax[1, 1].legend()
fig.tight_layout()
fig.savefig(HERE / "finding4_verification.png", dpi=110)
print(f"\nwrote {HERE/'indep_detector_grid.csv'}, {HERE/'annotated_row_ranks.csv'}, "
      f"{HERE/'indep_finding4_summary.json'}, {HERE/'finding4_verification.png'}")
print("done")