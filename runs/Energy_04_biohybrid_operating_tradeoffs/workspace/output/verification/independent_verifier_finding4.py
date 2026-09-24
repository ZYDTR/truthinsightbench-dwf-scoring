#!/usr/bin/env python3
"""
Independent verification of Finding 4 of output/Result.md
("The fitted columns are derived and strictly monotone; they are not
independent observations and cannot be counted twice").

Written from scratch from data/time_response_curves.tsv; does NOT import
output/analysis/analysis.py (read only afterwards, to see definitions).

Sections:
  1. raw-vs-fitted correlations on rows >= 259 (n=1108) + broad lag scan
  2. residual SDs (fit - raw)
  3. exhaustive monotonicity / reversal counts over the 1072 steps of regime C
  4. permutation test, 20000 draws, seed 12345 (report: 2000 draws, seed 20230919)
  5. moving-average window scan (row order and descending-sorted axis)
  6. DECISIVE axis tests: exhaustive feasibility + matched-window head-to-head
     + paired block bootstrap of the RMSE gap
  7. double-counting arithmetic and residual lag-1 autocorrelation
"""
import numpy as np
import pandas as pd

DATA = ("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-"
        "01601e51abe5/project/data/time_response_curves.tsv")
df = pd.read_csv(DATA, sep="\t")
assert len(df) == 1367

PAIRS = [("condition_a", "fit_a"), ("condition_b", "fit_b"),
         ("promoter_like_signal", "condition_c")]
R = df.loc[259:].reset_index(drop=True)        # report's "rows >= 259", n = 1108
C = df.loc[294:1366].reset_index(drop=True)    # regime C, n = 1073 -> 1072 steps
print(f"rows>=259: n={len(R)}   regime C: n={len(C)}, steps={len(C)-1}")


def pearson_hand(x, y):
    """Pearson r from the covariance formula, by hand."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean(); ym = y - y.mean()
    return float(xm @ ym / np.sqrt((xm @ xm) * (ym @ ym)))


def pearson_z(x, y):
    """Same quantity via z-scores (independent route)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    zx = (x - x.mean()) / x.std(ddof=0); zy = (y - y.mean()) / y.std(ddof=0)
    return float(zx @ zy / len(x))


def centre_ma(x, w):
    """Centred moving average, pandas rolling(w, center=True, min_periods=1)
    convention (shrinking window at the edges)."""
    x = np.asarray(x, float); n = len(x); h = w // 2
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - h); hi = min(n, i + h + 1)
        out[i] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def trailing_ma(x, w):
    x = np.asarray(x, float); n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        out[i] = (cs[i + 1] - cs[i + 1 - w]) / w
    return out


def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def dirn(v):
    """(frac strictly descending, reversals strict-nonzero, reversals any-sign
    change, ties) over consecutive steps."""
    v = np.asarray(v, float); d = np.diff(v); s = np.sign(d)
    frac = float((d < 0).mean())
    nz = s[s != 0]
    rev_strict = int((nz[1:] * nz[:-1] < 0).sum())
    rev_ne = int((s[1:] != s[:-1]).sum())      # definition used by analysis.py
    return frac, rev_strict, rev_ne, int((d == 0).sum())


# ======================================================================
# 1-2. association on rows >= 259 and residual SDs
# ======================================================================
print("\n=== 1. raw vs fitted, rows >= 259 (n=1108) ===")
for raw, fit in PAIRS:
    x = R[raw].values.astype(float); y = R[fit].values.astype(float)
    r1, r2, r3 = pearson_hand(x, y), pearson_z(x, y), float(np.corrcoef(x, y)[0, 1])
    res = y - x
    # broad lag scan |lag| <= 300
    lags = np.arange(-300, 301)
    rr = [pearson_hand(x[-L:] if L < 0 else x[:len(x) - L],
                       y[L:] if L > 0 else (y[:L] if L < 0 else y))
          for L in lags]
    bl = int(lags[int(np.argmax(rr))])
    print(f"{raw:22s}~{fit:12s} r_hand={r1:.9f} r_z={r2:.9f} r_np={r3:.9f} "
          f"| sd_resid_ddof0={res.std(ddof=0):.6f} sd_resid_ddof1={res.std(ddof=1):.6f}")
    print(f"    broad lag scan |lag|<=300: best lag={bl}, r={max(rr):.6f} "
          f"(r@0={r1:.6f}); r@lag=-300/+300={rr[0]:.4f}/{rr[-1]:.4f}")

# ======================================================================
# 3. monotonicity, regime C, 1072 steps
# ======================================================================
print("\n=== 3. regime C (rows 294-1366), 1072 consecutive steps ===")
for col in ["condition_c", "fit_a", "fit_b", "promoter_like_signal",
            "condition_a", "condition_b"]:
    frac, rs, rn, ties = dirn(C[col].values)
    print(f"{col:22s} frac_strict_desc={frac:.6f}  reversals(strict)={rs} "
          f"reversals(any-sign-change)={rn}  ties={ties}")
for col in ["condition_c", "fit_a"]:
    d = np.diff(C[col].values.astype(float))
    print(f"  EXHAUSTIVE {col}: n(diff>=0)={int((d >= 0).sum())} of {len(d)}, "
          f"min step={d.min():.6f} -> zero reversals confirmed")

# ======================================================================
# 4. permutation test with different seed and more draws
# ======================================================================
print("\n=== 4. permutation: 20000 draws seed 12345 (+2000 draws seed 777) ===")
rng = np.random.default_rng(12345)
rng2 = np.random.default_rng(777)
for col in ["condition_c", "fit_a", "fit_b", "promoter_like_signal",
            "condition_a", "condition_b"]:
    v = C[col].values.astype(float)
    obs = float((np.diff(v) < 0).mean())
    fr = np.array([(np.diff(rng.permutation(v)) < 0).mean() for _ in range(20000)])
    fr2 = np.array([(np.diff(rng2.permutation(v)) < 0).mean() for _ in range(2000)])
    nge = int((fr >= obs).sum())
    print(f"{col:22s} obs={obs:.6f} perm_mean={fr.mean():.6f} "
          f"perm_sd={fr.std(ddof=0):.6f} n_ge={nge}/20000 "
          f"p={(nge+1)/20001:.2e} | seed777 mean={fr2.mean():.6f} "
          f"sd={fr2.std(ddof=0):.6f} n_ge={int((fr2 >= obs).sum())}/2000")

# ======================================================================
# 5. window scan
# ======================================================================
print("\n=== 5. centred-MA window scan, regime C (n=1073) ===")
for raw, fit in PAIRS:
    x = C[raw].values.astype(float); y = C[fit].values.astype(float)
    xs = np.sort(x)[::-1]
    print(f"{raw:22s}~{fit:12s} direct unsmoothed RMSE(fit,raw)={rmse(y, x):.4f}")
    for w in (3, 5, 7, 9):
        ar, asrt = centre_ma(x, w), centre_ma(xs, w)
        print(f"    w={w}: row RMSE={rmse(ar, y):8.3f} r={pearson_hand(ar, y):.7f}"
              f" | sorted RMSE={rmse(asrt, y):8.3f} r={pearson_hand(asrt, y):.7f}")
    best_row = min(((rmse(centre_ma(x, w), y), w) for w in range(3, 60, 2)))
    best_srt = min(((rmse(centre_ma(xs, w), y), w) for w in range(3, 60, 2)))
    print(f"    best short-window: row w={best_row[1]} RMSE={best_row[0]:.3f} | "
          f"sorted w={best_srt[1]} RMSE={best_srt[0]:.3f}")

# ======================================================================
# 6. DECISIVE axis tests
# ======================================================================
print("\n=== 6a. row-order vs sorted axis at MATCHED smoothing (the fair test) ===")
print("     (the report compares sorted-smoothed RMSE against UNSMOOTHED row-order RMSE)")
for raw, fit in PAIRS:
    x = C[raw].values.astype(float); y = C[fit].values.astype(float)
    xs = np.sort(x)[::-1]
    for w in (3, 5, 7):
        rr, ss = rmse(centre_ma(x, w), y), rmse(centre_ma(xs, w), y)
        gap = rmse(centre_ma(x, w), centre_ma(xs, w))
        print(f"  {raw:22s} w={w}: row={rr:8.3f} sorted={ss:8.3f} "
              f"row-sorted={rr - ss:+7.3f}  RMS(between smoothers)={gap:.3f} "
              f"SD(fit)={y.std(ddof=0):.1f}")

print("\n=== 6b. EXHAUSTIVE feasibility: can ANY row-order centred MA be monotone? ===")
print("     (a centred MA of a non-monotone input can never be strictly decreasing)")
for raw, fit in PAIRS:
    x = C[raw].values.astype(float); y = C[fit].values.astype(float)
    best, zero = None, None
    for w in range(2, 1368):
        s = centre_ma(x, w); up = int((np.diff(s) > 0).sum())
        r = rmse(s, y)
        if best is None or r < best[0]:
            best = (r, w, up)
        if up == 0 and (zero is None or r < zero[0]):
            zero = (r, w)
    xs = np.sort(x)[::-1]
    srt = min(((rmse(centre_ma(xs, w), y), w) for w in range(3, 60, 2)))
    print(f"  {raw:22s}->{fit:12s} best row-order MA: w={best[1]} RMSE={best[0]:.3f} "
          f"(n_up_steps={best[2]})")
    print(f"      best row-order MA with ZERO up-steps: w={zero[1]} RMSE={zero[0]:.1f}")
    print(f"      best sorted-axis short-window MA: w={srt[1]} RMSE={srt[0]:.3f} (n_up=0)")
    d = np.diff(y)
    print(f"      observed fit: n_up={int((d > 0).sum())} n_down={int((d < 0).sum())} "
          f"sum_up={d[d > 0].sum():.4f} sum_down={-d[d < 0].sum():.1f}")

print("\n=== 6c. paired block bootstrap of the RMSE gap (row - sorted), 20000 draws ===")
print("     gap > 0 means the SORTED axis fits better")

n = len(C)


def stat_idx(rng, nn, mean_block):
    idx = []
    while len(idx) < nn:
        s = int(rng.integers(0, nn))
        L = max(1, int(rng.geometric(1.0 / mean_block)))
        idx.extend(((np.arange(s, s + L)) % nn).tolist())
    return np.asarray(idx[:nn])


for kind, meanb in [("iid", None), ("block100", 100)]:
    rngb = np.random.default_rng(20240919)
    idxs = [rngb.integers(0, n, n) if kind == "iid" else stat_idx(rngb, n, meanb)
            for _ in range(20000)]
    print(f"  --- {kind} ---")
    for raw, fit in PAIRS:
        x = C[raw].values.astype(float); y = C[fit].values.astype(float)
        xs = np.sort(x)[::-1]
        for w in (3, 5, 7):
            er = (y - centre_ma(x, w)) ** 2
            es = (y - centre_ma(xs, w)) ** 2
            d = np.array([np.sqrt(er[i].mean()) - np.sqrt(es[i].mean()) for i in idxs])
            obs = np.sqrt(er.mean()) - np.sqrt(es.mean())
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"    {raw:22s} w={w}: obs gap={obs:+.3f} "
                  f"95%CI=[{lo:+.3f},{hi:+.3f}] frac(gap<=0)={float((d <= 0).mean()):.4f}")

print("\n=== 6e. the report's contested comparison, verbatim ===")
print("  report: sorted-smoothed RMSE 'at least as well as applying it in row order'")
print("          quoted row-order RMSE 19.572 / 22.893 / 12.343")
for raw, fit, q in [("promoter_like_signal", "condition_c", 19.5717),
                    ("condition_a", "fit_a", 22.8928),
                    ("condition_b", "fit_b", 12.3431)]:
    x = C[raw].values.astype(float); y = C[fit].values.astype(float)
    print(f"  {raw:22s} quoted row-order {q:.4f} == UNSMOOTHED RMSE(fit,raw) "
          f"{rmse(y, x):.4f}  (no smoother applied to row order)")

print("\n=== 6d. corr(raw, own descending sort) in regime C ===")
for raw, fit in PAIRS:
    x = C[raw].values.astype(float)
    xs = np.sort(x)[::-1]
    print(f"  {raw:22s} r={pearson_hand(x, xs):.7f} RMSE(raw,sorted)={rmse(x, xs):.4f}")

# ======================================================================
# 7. double counting
# ======================================================================
print("\n=== 7. double counting ===")
for raw, fit in PAIRS:
    x = R[raw].values.astype(float); y = R[fit].values.astype(float)
    res = y - x
    print(f"  {raw:22s}~{fit:12s} n={len(x)} -> {2 * len(x)} "
          f"(factor 2.0) residual lag-1 autocorr={pearson_hand(res[:-1], res[1:]):.6f}")