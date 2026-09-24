#!/usr/bin/env python3
"""Falsification probes for Finding 3:
 (1) fine-grained a~b correlation as the baseline cut point sweeps rows 200..340
 (2) what happens if the cut is a moving 259-row window (no regime assumption)
 (3) resolution limit of the lead-lag scan from the series autocorrelation
 (4) does the baseline null hold under alternative n by dropping sweep-adjacent rows
"""
import numpy as np, pandas as pd
df = pd.read_csv("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv", sep="\t")
a = df.condition_a.to_numpy(float); b = df.condition_b.to_numpy(float)
prom = df.promoter_like_signal.to_numpy(float)
def pr(x, y): return float(np.corrcoef(x, y)[0, 1])

print("(1) baseline a~b r for cut point a_end = 200..340 (step 4)")
for ae in range(200, 341, 4):
    print(f"   a_end={ae:3d} n={ae+1:3d} r={pr(a[:ae+1], b[:ae+1]):+.4f}", end="")
    if (ae//4) % 3 == 2: print()
print()
ae_vals = range(200, 341)
jump = [ae for ae in ae_vals if ae >= 259 and abs(pr(a[:ae+1], b[:ae+1])) > 0.3]
print(f"   first cut point with |r|>0.3 (scanning upward): a_end={min(jump) if jump else None}")
print(f"   max |r| for cuts <= 258: {max(abs(pr(a[:ae+1], b[:ae+1])) for ae in range(200,259)):.4f}")

print()
print("(2) moving 259-row window across the file: r(a,b) per window")
for start in range(0, 1367-259+1, 100):
    w = slice(start, start+259)
    print(f"   rows {start:4d}-{start+258:4d} r={pr(a[w], b[w]):+.4f}  prom_a={pr(prom[w], a[w]):+.4f}")

print()
print("(3) lead-lag resolution limit")
xC, yC = a[294:], b[294:]
ac_a = pr(xC[:-1], xC[1:]); ac_b = pr(yC[:-1], yC[1:])
print(f"   tail lag-1 autocorr: a={ac_a:.6f} b={ac_b:.6f}")
print(f"   implied decorrelation time ~ 1/(1-r): a={1/(1-ac_a):.0f} rows b={1/(1-ac_b):.0f} rows")
print(f"   tail length = {len(xC)} rows; scan window = 51 rows")
print("   -> both series are single slow sweeps; no lead-lag peak can be resolved inside +/-25 rows.")

print()
print("(4) baseline null under cuts placed BEFORE the sweep, dropping the last k rows")
for drop in (0, 5, 10, 20, 40):
    n = 259 - drop
    print(f"   rows 0..{n-1} (n={n}) r={pr(a[:n], b[:n]):+.5f}")

print()
print("(5) tail-level vs first-difference r: fraction of level r^2 removed")
r_lev = pr(xC, yC); r_dif = pr(np.diff(xC), np.diff(yC))
print(f"   level r={r_lev:.6f} (r2={r_lev**2:.6f}); diff r={r_dif:.6f} (r2={r_dif**2:.6f})")
print(f"   1 - r2_diff/r2_lev = {1-r_dif**2/r_lev**2:.6f}  <-- a defensible within-tail 'trajectory share'")
dx, dy = np.diff(xC), np.diff(yC)
print(f"   first-diff lag-1 autocorr: a={pr(dx[:-1],dx[1:]):.4f} b={pr(dy[:-1],dy[1:]):.4f}")
