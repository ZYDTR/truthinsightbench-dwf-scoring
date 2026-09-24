#!/usr/bin/env python3
"""Verifier part 2: exact-definition cross-checks for quantities the report
quotes with a specific construction (rows>=259 association, lag-1 autocorr,
Fano local, sign changes, boundary grid) plus framing checks."""
import json
import math
import os

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
cur = pd.read_csv(os.path.join(ROOT, "data", "time_response_curves.tsv"), sep="\t")
P = print


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def r2_ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    p = A @ c
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def lsd(v, w=15):
    v = np.asarray(v, float)
    tr = pd.Series(v).rolling(w, center=True, min_periods=3).median().values
    return float(np.std(v - tr))


P("=" * 78)
P("A. Finding-4 association, rows >= 259 (n=1108), report: r=0.99956/0.99878/0.99950, resid SD 38.1/35.3/36.3")
s = cur.loc[cur.index >= 259]
for raw, fit in [("condition_a", "fit_a"), ("condition_b", "fit_b"), ("promoter_like_signal", "condition_c")]:
    res = (s[fit] - s[raw]).values
    P(f"  {raw} vs {fit}: r={corr(s[raw], s[fit]):.6f} resid_sd(ddof=1)={res.std(ddof=1):.4f} "
      f"resid_sd(ddof=0)={res.std():.4f} mean={res.mean():.4f}")
P("  report says all at lag 0 (no lag improves them):")
for raw, fit in [("condition_a", "fit_a"), ("condition_b", "fit_b"), ("promoter_like_signal", "condition_c")]:
    best = max((corr(s[raw].values[k:] if k > 0 else s[raw].values[:len(s) + k],
                      s[fit].values[:len(s) - k] if k > 0 else s[fit].values[-k:] if k < 0 else s[fit].values),
                k) for k in range(-12, 13))
    P(f"    {raw}: best r={best[0]:.6f} at lag {best[1]}")

P("=" * 78)
P("B. Fano local with exact definition (std(v - rolling_median15,center,min_periods=3)^2 / mean)")
segs = {"A": cur.iloc[:259], "B": cur.iloc[259:294], "C": cur.iloc[294:], "WHOLE": cur}
for nm, sub in segs.items():
    vals = []
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        v = sub[c].values.astype(float)
        vals.append(lsd(v) ** 2 / v.mean())
    P(f"  {nm}: {['%.4f' % v for v in vals]}")
P("  report: A 0.98,1.02,0.95 ; C 1.94,1.11,1.50")

P("=" * 78)
P("C. Double-counting residual lag-1 autocorrelation, rows>=259 (report 0.48/0.47/0.74)")
for raw, fit in [("promoter_like_signal", "condition_c"), ("condition_a", "fit_a"), ("condition_b", "fit_b")]:
    r = (s[fit] - s[raw]).values
    P(f"  {raw}: lag1={float(np.corrcoef(r[:-1], r[1:])[0,1]):.4f}")

P("=" * 78)
P("D. Sign changes in regime C (report: 0,0,74,743,735,802)")
C = cur.iloc[294:]
for c in ["condition_c", "fit_a", "fit_b", "promoter_like_signal", "condition_a", "condition_b"]:
    d = np.diff(C[c].values.astype(float))
    sc = int(np.sum(np.diff(np.sign(d)) != 0))
    P(f"  {c}: sign_changes={sc} frac_desc={float(np.mean(d < 0)):.4f}")

P("=" * 78)
P("E. Boundary sensitivity recomputed from raw data (37 admissible A|B,B|C cuts)")
a_end, b_end = 258, 293
rows = []
for da in [-40, -20, -10, 0, 10, 20, 40]:
    for db in [-40, -20, -10, 0, 10, 20, 40]:
        a1, b1 = a_end + da, b_end + db
        if not (60 <= a1 < b1 - 10) or b1 >= len(cur) - 30:
            continue
        base = cur.iloc[:a1 + 1]
        tail = cur.iloc[b1 + 1:]
        rows.append((a1, b1, len(base), len(tail), corr(base.condition_a, base.condition_b),
                     corr(tail.condition_a, tail.condition_b),
                     r2_ols(np.arange(len(tail), dtype=float), tail.promoter_like_signal)))
df = pd.DataFrame(rows, columns=["a_end", "b_end", "nb", "nt", "base_r", "tail_r", "tail_r2"])
P(f"  n_variants={len(df)} (report 37)")
P(f"  baseline a~b r range: [{df.base_r.min():.6f}, {df.base_r.max():.6f}] (report [0.0062, 0.9522])")
P(f"  tail a~b r range:     [{df.tail_r.min():.6f}, {df.tail_r.max():.6f}] (report [0.679, 0.973])")
P(f"  tail trend R2 range:  [{df.tail_r2.min():.6f}, {df.tail_r2.max():.6f}] (report 'stayed high')")
P("  variants with baseline |r|>0.35:", df.loc[df.base_r.abs() > 0.35, ["a_end", "b_end", "base_r"]].values.tolist())
P("  a_end=268 variants:", df.loc[df.a_end == 268, ["b_end", "base_r"]].values.tolist())

P("=" * 78)
P("F. Regime-A internal R2 / 'pooled carried entirely by regime B'")
A = cur.iloc[:259]
P(f"  R2 categorical time within A: {r2_ols(pd.Series(A.time).astype('category').cat.codes.values.astype(float), A.promoter_like_signal):.6f}")
P(f"  R2 categorical time within C: {r2_ols(C.time.values.astype(float), C.promoter_like_signal):.6f}")
# concat A+C with time-categorical: between-group SS using the 3 distinct time labels
AC = pd.concat([A, C])
gm = AC.groupby("time")["promoter_like_signal"].mean()
pred = pd.Series(AC.time.values).map(gm).values
y = AC.promoter_like_signal.values
P(f"  R2 categorical time in A+C (3 labels): {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.6f}")

P("=" * 78)
P("G. File-wide rows at regime-A scale (promoter <= 7) outside A")
m = (cur.promoter_like_signal <= 7)
P("  rows with promoter<=7 at row>=259:", list(np.nonzero(m.values[259:])[0] + 259), "values:", cur.promoter_like_signal.values[m.values][len(A.condition_b):].tolist() if False else cur.promoter_like_signal.values[np.nonzero(m.values)[0]].tolist()[:5])
P("  n rows with promoter<=7 outside A:", int(m.values[259:].sum()))
P("  min promoter in C:", int(C.promoter_like_signal.min()))

P("=" * 78)
P("H. Lead-lag full curve shape (regime C)")
a = C.condition_a.values.astype(float); b = C.condition_b.values.astype(float)
ll = []
for lag in range(-25, 26):
    if lag < 0:
        x, y2 = a[-lag:], b[:lag]
    elif lag > 0:
        x, y2 = a[:-lag], b[lag:]
    else:
        x, y2 = a, b
    ll.append((lag, corr(x, y2)))
neg = [v for k, v in ll if k <= 0]
pos = [v for k, v in ll if k >= 0]
P("  monotone increasing for lag<=0 toward 0:", all(neg[i] <= neg[i + 1] + 1e-12 for i in range(len(neg) - 1)))
P("  monotone increasing for lag>=0:", all(pos[i] <= pos[i + 1] + 1e-12 for i in range(len(pos) - 1)))
P("  so r increases with |lag| only on the positive side; r(-25)=%.4f < r(0)=%.4f" % (ll[0][1], ll[25][1]))
P("  argmax:", max(ll, key=lambda t: t[1]))

P("=" * 78)
P("I. Perturbation pooled-SD definition check and table values")
pert = pd.read_csv(os.path.join(ROOT, "data", "perturbation_time_response.tsv"), sep="\t")
for cond in ["control", "condition_a", "condition_b"]:
    m = pert[f"{cond}_mean"].values; sd = pert[f"{cond}_sd"].values
    pooled = math.sqrt(np.mean(sd ** 2))
    t0, t6 = m[0], m[-1]
    P(f"  {cond}: t0={t0} t6={t6} change={t6-t0:+.8f} pct={100*(t6-t0)/t0:+.4f}% "
      f"pooled_sd={pooled:.6f} effect={abs(t6-t0)/pooled:.4f} strict_dec={bool(np.all(np.diff(m)<0))}")
    p = np.polyfit(pert.time.values.astype(float), np.log(m), 1)
    lr2 = 1 - ((np.log(m) - np.polyval(p, pert.time.values.astype(float))) ** 2).sum() / ((np.log(m) - np.log(m).mean()) ** 2).sum()
    P(f"      loglinear rate={p[0]:.6f} R2={lr2:.6f} half_life={math.log(2)/abs(p[0]):.4f} "
      f"ratio_to_control={np.round(m / pert.control_mean.values, 6).tolist()}")
    pl = np.polyfit(pert.time.values.astype(float), m, 1)
    lin_r2 = 1 - ((m - np.polyval(pl, pert.time.values.astype(float))) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    P(f"      linear rate={pl[0]:.6f} R2={lin_r2:.6f}")

P("=" * 78)
P("J. Framing scan of Result.md")
txt = open(os.path.join(ROOT, "output", "Result.md")).read()
for term in ["potency", "per unit dose", "per-unit-dose", "condition_c", "independent observation", "arm"]:
    hits = [ln.strip() for ln in txt.splitlines() if term.lower() in ln.lower()]
    P(f"  '{term}' appears {len(hits)} time(s):")
    for h in hits[:6]:
        P("     | " + h[:200])
P("=" * 78)
P("K. Cross-check declared <-> machine-readable integrity")
fj = json.load(open(os.path.join(ROOT, "output", "results", "findings.json")))
P("  findings.json data_integrity matches:", [(d["path"], d["match"], d["bytes"]) for d in fj["data_integrity"]])
P("  findings.json shapes:", fj["shapes"])
P("  findings.json missingness response_sd:", fj["missingness"]["dose_response.tsv"]["response_sd"], "(report: 3)")
P("  tail_decline keys:", {k: v for k, v in fj["tail_decline_order_null"].items() if k != "block_bootstrap_note"})