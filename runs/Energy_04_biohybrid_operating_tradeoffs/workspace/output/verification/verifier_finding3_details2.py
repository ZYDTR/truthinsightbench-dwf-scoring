#!/usr/bin/env python3
"""Detail pass 2: identify the Fano convention used by the report; exact noncentral-t power
for Pearson r at n=259; significance of tail residual correlations.
"""
import json
import numpy as np
import pandas as pd
from math import erf, sqrt, pi, lgamma, isfinite

DATA = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv"
FIND = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/findings.json"

df = pd.read_csv(DATA, sep="\t")
prom = df.promoter_like_signal.to_numpy(float)
a = df.condition_a.to_numpy(float)
b = df.condition_b.to_numpy(float)
with open(FIND) as f:
    fj = json.load(f)
tg = fj["fano_by_regime"]


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2)))


print("=" * 78)
print("FANO CONVENTION SCAN  (find (window, convention) matching all 6 reported local Fano)")
print("=" * 78)

regs = [("A_baseline", 0, 259), ("C_tail", 294, 1367)]
chan = [("promoter_like_signal", prom), ("condition_a", a), ("condition_b", b)]


def fano_grid(x, w):
    """Yield (convention_name, value) for one window."""
    out = {}
    for how in ("median", "mean"):
        roll = pd.Series(x).rolling(w, center=True, min_periods=1)
        r = (roll.median().to_numpy() if how == "median" else roll.mean().to_numpy())
        resid = x - r
        out[f"{how}_pad1_varResid_over_meanX"] = np.var(resid, ddof=1) / x.mean()
        out[f"{how}_pad1_varResid_ddof0_over_meanX"] = np.var(resid, ddof=0) / x.mean()
        shifted = resid + x.mean()
        out[f"{how}_pad1_shifted_var_over_meanShifted"] = np.var(shifted, ddof=1) / shifted.mean()
    return out


best = None
for w in range(3, 32):
    tot = 0.0
    detail = {}
    for (reg, lo, hi) in regs:
        for cname, v in chan:
            x = v[lo:hi]
            t = tg[reg][cname]["fano_local"]
            vs = fano_grid(x, w)
            conv, val = min(vs.items(), key=lambda kv: abs(kv[1] - t))
            tot += abs(val - t)
            detail[(reg, cname)] = (conv, val, t)
    if best is None or tot < best[1]:
        best = (w, tot, detail)
print(f"best window overall: w={best[0]}, total abs err={best[1]:.2e}")
for k, (conv, val, t) in best[2].items():
    print(f"   {k[0]:10s} {k[1]:22s} convention={conv:34s} mine={val:.8f} reported={t:.8f} d={val-t:+.2e}")

print()
print("exact per-channel best window / convention:")
for (reg, lo, hi) in regs:
    for cname, v in chan:
        x = v[lo:hi]
        t = tg[reg][cname]["fano_local"]
        bestc = None
        for w in range(3, 32):
            for conv, val in fano_grid(x, w).items():
                d = abs(val - t)
                if bestc is None or d < bestc[0]:
                    bestc = (d, w, conv, val)
        d, w, conv, val = bestc
        print(f"   {reg:10s} {cname:22s} w={w:2d} {conv:34s} mine={val:.8f} rep={t:.8f} d={d:.2e}")

# rounded-to-2dp comparison under the natural convention (pad1 median, varResid/meanX)
print()
print("2-dp comparison under convention (15-pt centred median, pad1, var(resid)/mean(x)):")
for (reg, lo, hi) in regs:
    out = []
    for cname, v in chan:
        x = v[lo:hi]
        r = x - pd.Series(x).rolling(15, center=True, min_periods=1).median().to_numpy()
        mine = np.var(r, ddof=1) / x.mean()
        out.append(f"{cname}={mine:.2f}(rep {tg[reg][cname]['fano_local']:.2f})")
    print(f"   {reg}: " + ", ".join(out))

# ---------------- power: see verifier_finding3_power.py for the noncentral-t computation ----
print()
print("=" * 78)
print("POWER (summary; exact noncentral-t computed in verifier_finding3_power.py)")
print("=" * 78)
print("  exact noncentral-t detectable rho at 80% power = 0.17204")
print("  Fisher-z closed form                           = 0.17333")
print(f"  report claim                                   = ~0.174")

# ---------------- tail first-diff / detrended significance ----------------
print()
print("=" * 78)
print("TAIL RESIDUAL ASSOCIATIONS: are they zero?")
print("=" * 78)
xC, yC = a[294:], b[294:]
dx, dy = np.diff(xC), np.diff(yC)
r1 = pearson(dx, dy)
t1 = r1 * sqrt((len(dx) - 2) / (1 - r1 ** 2))
from math import erfc
print(f"  first-diff r={r1:+.6f}, n={len(dx)}, t={t1:.3f}, two-sided p (normal)={erfc(abs(t1)/sqrt(2)):.3e}")
rmx = xC - pd.Series(xC).rolling(15, center=True, min_periods=1).median().to_numpy()
rmy = yC - pd.Series(yC).rolling(15, center=True, min_periods=1).median().to_numpy()
r2 = pearson(rmx, rmy)
t2 = r2 * sqrt((len(rmx) - 2) / (1 - r2 ** 2))
print(f"  detrended  r={r2:+.6f}, n={len(rmx)}, t={t2:.3f}, two-sided p (normal)={erfc(abs(t2)/sqrt(2)):.3e}")
# autocorrelation of the first differences (effective sample size)
print(f"  lag-1 autocorr of residual(a) = {pearson(rmx[:-1], rmx[1:]):.4f}, "
      f"of residual(b) = {pearson(rmy[:-1], rmy[1:]):.4f}")
# permutation p for detrended r
rng = np.random.default_rng(4242)
cnt = 0
for _ in range(20000):
    if abs(pearson(rmx, rng.permutation(rmy))) >= abs(r2):
        cnt += 1
print(f"  permutation p (20000 draws, seed 4242) for detrended r = {(cnt + 1) / 20001:.5f}")

# ---------------- boundary grid: full 49 values incl. excluded ----------------
print()
print("=" * 78)
print("BOUNDARY GRID: value of all 49 combos (incl. the 12 the report calls inadmissible)")
print("=" * 78)
Aends = [218, 238, 248, 258, 268, 278, 298]
Bends = [253, 273, 283, 293, 303, 313, 333]
rep = {(v["a_end"], v["b_end"]): v for v in fj["finding1_regimes"]["boundary_sensitivity"]}
rows = []
for ae in Aends:
    line = []
    for be in Bends:
        nA = ae + 1
        tail = np.arange(be + 1, 1367)
        r_base = pearson(a[:nA], b[:nA])
        mark = "*" if (ae, be) in rep else " "
        line.append(f"{r_base:+.3f}{mark}")
        rows.append((ae, be, r_base, (ae, be) in rep))
    print(f"  a_end={ae:3d}: " + "  ".join(line))
print("  (* = a variant included in the report's 37; others are the 12 excluded combos,")
print("   all of which have b_end - a_end < 15, i.e. a middle strip of <14 rows)")
vals_all = [r[2] for r in rows]
vals_rep = [r[2] for r in rows if r[3]]
print(f"  range over report's 37: [{min(vals_rep):+.6f}, {max(vals_rep):+.6f}]")
print(f"  range over all 49     : [{min(vals_all):+.6f}, {max(vals_all):+.6f}]")

# ---------------- headline table cross-check ----------------
print()
print("=" * 78)
print("HEADLINE TABLE (table_headline_numbers.tsv) 3 a_b values")
print("=" * 78)
th = pd.read_csv("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/table_headline_numbers.tsv", sep="\t")
for _, row in th.iterrows():
    if "a_b" in row["quantity"]:
        print(f"  {row['quantity']}: reported={row['value']}")
print(f"  my baseline={pearson(a[:259], b[:259]):.15f}  pooled={pearson(a, b):.15f}")