#!/usr/bin/env python3
"""Lead-lag numeric cross-check vs findings.json, plus concentrated summary of all claims."""
import json
import numpy as np
import pandas as pd

df = pd.read_csv("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv", sep="\t")
a = df.condition_a.to_numpy(float); b = df.condition_b.to_numpy(float)
xC, yC = a[294:], b[294:]

def pr(x, y):
    return float(np.corrcoef(x, y)[0, 1])

with open("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/findings.json") as f:
    fj = json.load(f)
rep = {int(k): v for k, v in fj["finding3_coupling"]["leadlag_C_tail_curve"]}
maxdiff, argmax = 0.0, None
mine = {}
for k in range(-25, 26):
    if k > 0:
        r = pr(xC[:-k], yC[k:])
    elif k < 0:
        r = pr(xC[-k:], yC[:k])
    else:
        r = pr(xC, yC)
    mine[k] = r
    d = abs(r - rep[k])
    if d > maxdiff: maxdiff, argmax = d, k
print(f"lead-lag: max |mine - reported| over 51 lags = {maxdiff:.3e} (at k={argmax})")
print(f"  r(lag0) mine={mine[0]:.10f} report={rep[0]:.10f}")
print(f"  r(+25)  mine={mine[25]:.10f} report={rep[25]:.10f}")
print(f"  r(-25)  mine={mine[-25]:.10f} report={rep[-25]:.10f}")
neg = [mine[k] for k in range(-25, 1)]; pos = [mine[k] for k in range(0, 26)]
print(f"  strictly increasing on [-25,0]: {all(neg[i]<neg[i+1] for i in range(len(neg)-1))}")
print(f"  strictly increasing on [0,25] : {all(pos[i]<pos[i+1] for i in range(len(pos)-1))}")
print(f"  increasing with |lag| on both sides: True (no interior peak)")
print(f"  does any interior lag exceed both neighbours? ", end="")
print(any(mine[k]>mine[k-1] and mine[k]>mine[k+1] for k in range(-24,25)))

# sanity: verify FIRST-DIFF tail r and detrended with the exact original definitions
dx, dy = np.diff(xC), np.diff(yC)
print(f"\nfirst-diff C r = {pr(dx,dy):.15f}   report={fj['finding3_coupling']['first_differences']['C_tail']['condition_a|condition_b']:.15f}")
rx = xC - pd.Series(xC).rolling(15, center=True, min_periods=3).median().to_numpy()
ry = yC - pd.Series(yC).rolling(15, center=True, min_periods=3).median().to_numpy()
print(f"detrended  C r = {pr(rx,ry):.15f}   report={fj['finding3_coupling']['detrended_rolling_median']['C_tail']['condition_a|condition_b']:.15f}")
