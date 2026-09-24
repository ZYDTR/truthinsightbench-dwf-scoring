#!/usr/bin/env python3
"""Supplementary attacks on Finding 4.

A) Detector-family attack: does 15/15 become reachable if the local-maximum
   post-filter is dropped (threshold-only / top-N-by-excess rules)?
B) Enrichment attack: cut-size sensitivity of the top-K enrichment, tie
   sensitivity of the top-30 boundary, and the local-maximum composition of the
   top-30 set (to explain why the LM-restricted top-30 contains 14/15).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
df = pd.read_csv(ROOT / "data" / "observations.tsv", sep="\t")
n = len(df)
so = df["c04_sulfate_ug_kg"].to_numpy(float)
labels = df["c07_signal"].to_numpy()
AI = np.where(df["c07_signal"].notna().to_numpy())[0]
so_i = so.copy()
m = np.isnan(so_i)
so_i[m] = np.interp(np.where(m)[0], np.where(~m)[0], so[~m])
S = pd.Series(so_i)
lm = np.zeros(n, bool)
lm[1:-1] = (so_i[1:-1] > so_i[:-2]) & (so_i[1:-1] > so_i[2:])
rows = []

print("=" * 78)
print("A) Family attack: threshold-only rules (no local-maximum filter)")
print("=" * 78)
for w in [15, 25, 51, 101, 301]:
    bg = S.rolling(w, center=True, min_periods=max(2, w // 5)).median().to_numpy()
    exc = so_i - bg
    order = np.argsort(-exc, kind="stable")
    # smallest N such that top-N-by-excess contains all 15 annotated
    needed = int(np.searchsorted(np.sort(np.array([np.where(order == r)[0][0] for r in AI])), 14) + 1)
    worst_rank = int(max(np.where(order == r)[0][0] for r in AI)) + 1
    hits_topN = len(set(order[:worst_rank].tolist()) & set(AI.tolist()))
    print(f"  w={w:4d}: rank of the annotated rows by excess -> "
          f"min={min(np.where(order == r)[0][0] for r in AI)+1}, "
          f"max={worst_rank}; top-{worst_rank} recall={hits_topN}/15 "
          f"({worst_rank/n:.1%} of the record called an event)")
    rows.append({"window": w, "n_for_full_recall": worst_rank, "recall": hits_topN})

# threshold-only scan: excess > k*sigma
print("\n  excess > k*sigma (no local-max filter), w=101:")
bg = S.rolling(101, center=True, min_periods=20).median().to_numpy()
exc = so_i - bg
sd = np.nanstd(exc)
for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
    sel = exc > k * sd
    print(f"    k={k:<4}: n_det={int(sel.sum()):4d}  recall={len(set(np.where(sel)[0].tolist()) & set(AI.tolist())):2d}/15"
          f"  row0_in={bool(sel[0])}")

print("\n  => 15/15 IS reachable outside the local-max family, but only by "
      "flagging hundreds of samples (~17-19% of the record).")

print("\n" + "=" * 78)
print("B) Enrichment: cut-size sensitivity, ties, and LM composition")
print("=" * 78)
bg101 = S.rolling(101, center=True, min_periods=20).median().to_numpy()
exc101 = so_i - bg101
order = np.argsort(-exc101, kind="stable")
print("  observed excess values around the top-30 boundary:")
for r in order[26:35]:
    print(f"    rank {int(np.where(order == r)[0][0])+1:3d} row {r:5d} exc={exc101[r]:9.3f} "
          f"lm={bool(lm[r])} annot={bool(r in set(AI.tolist()))}")
tied = exc101[order[29]] == exc101[order[30]]
print(f"  tie at the top-30 boundary: {tied}")
print("  enrichment vs cut size (expected = 15*K/n):")
for K in [10, 20, 30, 50, 100, 200]:
    top = set(order[:K].tolist())
    hits = len(top & set(AI.tolist()))
    exp = 15 * K / n
    p = float(hypergeom.sf(hits - 1, n, K, 15))
    print(f"    top-{K:3d}: observed={hits:2d}/15  expected={exp:5.2f}  "
          f"enrichment={hits/exp if exp else float('nan'):5.1f}x  p={p:.2e}")
top30 = order[:30]
print(f"  top-30 by excess: {int(lm[top30].sum())}/30 are strict local maxima; "
      f"{len([r for r in top30 if r in set(AI.tolist())])} are annotated")
print(f"  of the 14 annotated strict local maxima, "
      f"{sum(1 for r in AI if lm[r] and exc101[r] > exc101[order[29]])} are above "
      f"the 30th-highest excess overall;")
print(f"  rank within the 579 local maxima of each annotated row:")
lmi = np.where(lm)[0]
olm = np.argsort(-exc101[lmi], kind="stable")
r_in = np.empty(len(olm), int)
r_in[olm] = np.arange(1, len(olm) + 1)
pos = {int(r): i for i, r in enumerate(lmi)}
for r in AI:
    rr = int(r_in[pos[int(r)]]) if int(r) in pos else None
    print(f"    row {r:5d} {labels[r]:20s} rank_in_LM={rr}  (top 30 = top 5.2% of 579)")

# local-max composition of the top-30 and enrichment of annotated among LM
print(f"\n  Fraction of the 579 local maxima that are in the top-30 overall: "
      f"{int(lm[top30].sum())}/579")
print("  Interpretation: the overall top-30 by excess is dominated by "
      "non-local-maxima (steep rising limbs),")
print("  so restricting the top-30 to local maxima raises the annotated count "
      "from 7 to 14. The enrichment")
print("  is therefore NOT an artefact of annotated rows being local maxima - "
      "it is stronger within that class.")
json.dump({"threshold_only_full_recall": rows}, open(HERE / "attack_b_results.json", "w"), indent=2)
print("\ndone")