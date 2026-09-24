#!/usr/bin/env python3
"""Robustness of the tail detrended correlation to detrending convention/window,
p-value for prom~a in regime A, and admissibility criterion of the boundary grid."""
import json
import numpy as np
import pandas as pd
from math import erfc, sqrt

DATA = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv"
FIND = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/findings.json"

df = pd.read_csv(DATA, sep="\t")
prom = df.promoter_like_signal.to_numpy(float)
a = df.condition_a.to_numpy(float)
b = df.condition_b.to_numpy(float)


def pr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


print("=" * 78)
print("TAIL DETRENDED a~b: sensitivity to detrending convention and window")
print("=" * 78)
xC, yC = a[294:], b[294:]
print("  reported: 0.3051 (15-row centred rolling median, both channels)")
for w in (5, 7, 9, 11, 15, 21, 31, 51):
    for center in (True, False):
        rx = xC - pd.Series(xC).rolling(w, center=center, min_periods=1).median().to_numpy()
        ry = yC - pd.Series(yC).rolling(w, center=center, min_periods=1).median().to_numpy()
        print(f"    rolling median w={w:3d} center={str(center):5s} -> r={pr(rx, ry):+.4f}")
for w in (15, 51):
    rx = xC - pd.Series(xC).rolling(w, center=True, min_periods=1).mean().to_numpy()
    ry = yC - pd.Series(yC).rolling(w, center=True, min_periods=1).mean().to_numpy()
    print(f"    rolling MEAN   w={w:3d} center=True  -> r={pr(rx, ry):+.4f}")
# high-pass: subtract cubic polynomial fit
t = np.arange(len(xC), dtype=float)
for deg in (1, 2, 3, 5):
    ca = np.polyfit(t, xC, deg)
    cb = np.polyfit(t, yC, deg)
    print(f"    polynomial detrend deg={deg} -> r={pr(xC-np.polyval(ca,t), yC-np.polyval(cb,t)):+.4f}")

print()
print("  first-difference sensitivity: order-2 differences, and %-changes")
d1 = pr(np.diff(xC), np.diff(yC))
d2 = pr(np.diff(xC, 2), np.diff(yC, 2))
print(f"    order-1 diffs r={d1:+.4f}   order-2 diffs r={d2:+.4f}")
lx, ly = np.log1p(xC), np.log1p(yC)
print(f"    log1p levels r={pr(lx, ly):+.4f}   log1p diffs r={pr(np.diff(lx), np.diff(ly)):+.4f}")

print()
print("=" * 78)
print("p-value for prom~a in regime A (+0.1095) and a~b (+0.0062)")
print("=" * 78)
xA, yA, pA = a[:259], b[:259], prom[:259]
for name, r in [("prom~a", pr(pA, xA)), ("prom~b", pr(pA, yA)), ("a~b", pr(xA, yA))]:
    t = r * sqrt((259 - 2) / (1 - r ** 2))
    print(f"  {name}: r={r:+.6f}  t={t:+.4f}  two-sided p (normal)={erfc(abs(t)/sqrt(2)):.4f}")
print("  (the report says +0.1095 for prom~a is 'not distinguishable from zero at this n')")

print()
print("=" * 78)
print("ADMISSIBILITY CRITERION of the 7x7 boundary grid")
print("=" * 78)
with open(FIND) as f:
    fj = json.load(f)
bs = fj["finding1_regimes"]["boundary_sensitivity"]
admiss = {(v["a_end"], v["b_end"]) for v in bs}
Aends = [218, 238, 248, 258, 268, 278, 298]
Bends = [253, 273, 283, 293, 303, 313, 333]
diffs_in, diffs_out = [], []
for ae in Aends:
    for be in Bends:
        d = be - ae
        (diffs_in if (ae, be) in admiss else diffs_out).append(d)
print(f"  b_end - a_end for the 37 admitted: min={min(diffs_in)}, max={max(diffs_in)}")
print(f"  b_end - a_end for the 12 excluded: min={min(diffs_out)}, max={max(diffs_out)}")
print(f"  -> rule: admit iff b_end - a_end >= {min(diffs_in)} (gap >= 15 rows); "
      f"excluded iff gap <= {max(diffs_out)}")
print(f"  all 49 combos baseline_ab_r range: "
      f"[{min(pr(a[:ae+1], b[:ae+1]) for ae in Aends):+.6f}, "
      f"{max(pr(a[:ae+1], b[:ae+1]) for ae in Aends):+.6f}]")
print("  note: baseline r depends ONLY on a_end, not b_end -> range identical for 37 and 49")

print()
print("=" * 78)
print("IS regime C 'almost entirely a shared trajectory'? extra diagnostic")
print("=" * 78)
# partial correlation of a,b given the OTHER channel (prom) in the tail
r_ab, r_ap, r_bp = pr(xC, yC), pr(xC, prom[294:]), pr(yC, prom[294:])
part = (r_ab - r_ap * r_bp) / np.sqrt((1 - r_ap ** 2) * (1 - r_bp ** 2))
print(f"  tail r(a,b)={r_ab:+.6f}; partial r(a,b | promoter_like_signal)={part:+.6f}")
# correlation with row index (trend) in tail
t = np.arange(len(xC), dtype=float)
print(f"  tail r(a, row index)={pr(xC, t):+.6f}; r(b, row index)={pr(yC, t):+.6f}")
# how much of tail a-b association survives after removing a common linear trend from both
ra = xC - np.polyval(np.polyfit(t, xC, 1), t)
rb = yC - np.polyval(np.polyfit(t, yC, 1), t)
print(f"  tail after linear-trend removal both channels: r={pr(ra, rb):+.6f}")
print(f"  (first-difference r = {d1:+.6f}, a stronger high-pass)")

# ------------------------------------------------------------------
# Now read the ORIGINAL analysis.py (only for comparison of definitions)
# ------------------------------------------------------------------
print()
print("=" * 78)
print("COMPARISON WITH output/analysis/analysis.py DEFINITIONS (read after computing)")
print("=" * 78)
src = open("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/analysis/analysis.py").read()
import re
i = src.find("def part3_coupling")
print(src[i:i + 4200])