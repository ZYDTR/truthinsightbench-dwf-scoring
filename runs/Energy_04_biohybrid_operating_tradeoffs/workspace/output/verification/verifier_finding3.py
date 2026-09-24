#!/usr/bin/env python3
"""
Independent verification of Finding 3 (regime-conditional condition_a-condition_b coupling)
from /data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/Result.md

Everything is recomputed from the raw TSV with hand-written numpy code.  No scipy/sklearn.
Two correlation methods are used everywhere possible:
  (M1) Pearson via the textbook formula  r = Sxy / sqrt(Sxx*Syy)
  (M2) Pearson via covariance matrix  r = cov(x,y)/sqrt(var(x)*var(y))  (different code path)
"""
import json
import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv"


def load():
    return pd.read_csv(DATA, sep="\t")


# ---------------- correlation machinery (independent implementations) -------------
def pearson_textbook(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxy = np.sum((x - xm) * (y - ym))
    sxx = np.sum((x - xm) ** 2)
    syy = np.sum((y - ym) ** 2)
    return sxy / np.sqrt(sxx * syy)


def pearson_covpath(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.cov(x, y, ddof=1)  # 2x2 covariance matrix
    return c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])


def check_two_methods(x, y, label, tol=1e-9):
    r1 = pearson_textbook(x, y)
    r2 = pearson_covpath(x, y)
    ok = abs(r1 - r2) < tol
    return r1, r2, ok


def rolling_median_centred(x, w=15):
    """Centred rolling median, window w, edges padded with the edge value.
    Implemented by hand (pandas rolling with center=True, min_periods=1 varies by edge
    convention, so we do it explicitly to control endpoints)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    half = w // 2
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def rolling_median_centred_pandas(x, w=15):
    """Same via pandas (different code path)."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(w, center=True, min_periods=1).median().to_numpy()


def fisher_ci(r, n, alpha=0.05):
    """Two-sided (1-alpha) Fisher-z CI for Pearson r."""
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = 1.959963984540054  # exact normal 0.975 quantile
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return lo, hi


def ols_slope_normaleq(x, y):
    """Slope via the normal equations (2x2 solve)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    return beta[1], beta[0]


def ols_slope_grid(x, y, lo=-0.1, hi=0.1, n=4000001):
    """Slope via brute-force grid search minimising SSE (completely different method).
    SSE(s) = Syy - 2 s Sxy + s^2 Sxx (centred form), minimised over a dense grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.linspace(lo, hi, n)
    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    sxy = np.sum((x - xbar) * (y - ybar))
    syy = np.sum((y - ybar) ** 2)
    sse = syy - 2 * grid * sxy + grid ** 2 * sxx
    return float(grid[np.argmin(sse)])


def diff_detectable_r(n, power=0.80, alpha=0.05, z_pow=None):
    """Smallest |r| detectable with given power at two-sided alpha, via Fisher z.
    Solve  n * (atanh(r))^2 = (z_{1-alpha/2} + z_power)^2  =>  r = tanh(sqrt(c/n)).
    Also do a brute-force search over a grid of r using the exact normal CDF-free
    criterion: power(r) = P(|Z| > zcrit) where Z ~ N(atanh(r)*sqrt(n-3), 1)."""
    zcrit = 1.959963984540054
    zpow = {0.80: 0.8416212335729143, 0.90: 1.2815515655446004}.get(power)
    if zpow is None:
        return None
    c = zcrit + zpow
    r_closed = np.tanh(np.sqrt(c * c / (n - 3)))
    # brute-force power computation on a grid, using math.erf directly (no scipy)
    from math import erf, sqrt
    grid = np.linspace(1e-6, 0.9, 900001)
    lam = np.arctanh(grid) * np.sqrt(n - 3)
    ncdf = np.vectorize(lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0))))
    power_grid = ncdf(-zcrit - lam) + (1.0 - ncdf(zcrit - lam))
    passed = np.nonzero(power_grid >= power)[0]
    r_brute = float(grid[passed[0]]) if len(passed) else None
    return r_closed, r_brute


def fano_local(x, w=15):
    """Fano after local detrending: remove centred rolling median, then var/mean of residual
    shifted to be non-negative?  Ambiguity: Fano = var/mean is only meaningful for counts.
    We follow the likely convention: residual 'counts' = x - rollmedian(x) + mean(x),
    so the residual series has the same mean as x and Fano = var(resid)/mean(x).
    We also report plain var(resid)/mean(resid_unshifted) alternatives."""
    x = np.asarray(x, dtype=float)
    rm = rolling_median_centred(x, w)
    resid = x - rm
    shifted = resid + x.mean()
    fano_shift = np.var(shifted, ddof=1) / shifted.mean()
    # alternative convention: Fano = var(resid)/mean(x)
    fano_alt = np.var(resid, ddof=1) / x.mean()
    return fano_shift, fano_alt


def main():
    df = load()
    n = len(df)
    res = {}

    prom = df["promoter_like_signal"].to_numpy(float)
    a = df["condition_a"].to_numpy(float)
    b = df["condition_b"].to_numpy(float)

    # ---- regimes by the report's stated row ranges
    regs = {
        "A_baseline": (0, 259),
        "B_sweep": (259, 294),
        "C_tail": (294, 1367),
        "WHOLE": (0, 1367),
    }
    pairs = [("promoter_like_signal", prom, "condition_a", a),
             ("promoter_like_signal", prom, "condition_b", b),
             ("condition_a", a, "condition_b", b)]

    print("=" * 78)
    print("1) Pearson correlations by regime (all 12 values)")
    print("=" * 78)
    corr = {}
    for rname, (lo, hi) in regs.items():
        corr[rname] = {}
        for n1, v1, n2, v2 in pairs:
            x, y = v1[lo:hi], v2[lo:hi]
            r1, r2, ok = check_two_methods(x, y, "x")
            corr[rname][f"{n1}|{n2}"] = (r1, len(x))
            flag = "OK" if ok else "!!TWO-METHOD-MISMATCH!!"
            print(f"  {rname:10s} n={len(x):5d}  {n1:22s}~{n2:22s} r={r1:+.6f} "
                  f"(cov-path {r2:+.6f}) {flag}")
    res["corr"] = {k: {kk: vv[0] for kk, vv in v.items()} for k, v in corr.items()}

    # ---- baseline CI, slope, range-span prediction
    print()
    print("=" * 78)
    print("2) Baseline regime A: a~b CI, OLS slope, prediction over full a range")
    print("=" * 78)
    xA, yA = a[0:259], b[0:259]
    rA = pearson_textbook(xA, yA)
    lo, hi = fisher_ci(rA, len(xA))
    print(f"  r_A(a~b)      = {rA:+.6f}   r^2 = {rA**2:.8f}")
    print(f"  95% CI (Fisher z) = [{lo:+.6f}, {hi:+.6f}]")
    print(f"  n = {len(xA)}")
    slope, intercept = ols_slope_normaleq(xA, yA)
    slope_g = ols_slope_grid(xA, yA)
    print(f"  OLS slope of b on a (normal eq) = {slope:+.8f}  intercept={intercept:+.8f}")
    print(f"  OLS slope (grid search)         = {slope_g:+.8f}   diff={abs(slope-slope_g):.2e}")
    a_range = xA.max() - xA.min()
    pred_change = slope * a_range
    sd_b = np.std(yA, ddof=1)
    print(f"  observed a range = [{xA.min():.0f}, {xA.max():.0f}] span={a_range:.0f}")
    print(f"  predicted b change over full a span = {pred_change:+.6f}")
    print(f"  SD(b) in regime A (ddof=1)          = {sd_b:.6f}")
    print(f"  ratio change/SD = {pred_change/sd_b:.5f} ({100*pred_change/sd_b:.2f}%)")
    res["baseline"] = dict(r=rA, ci=[lo, hi], n=len(xA), slope=slope, slope_grid=slope_g,
                           pred_change=pred_change, sd_b=sd_b)

    # ---- detectable |r|
    print()
    print("=" * 78)
    print("3) Smallest detectable |r| at 80% power, n=259")
    print("=" * 78)
    for p in (0.80, 0.90):
        rc, rg = diff_detectable_r(259, power=p)
        print(f"  power={p}: closed form = {rc:.6f}, grid search = {rg:.6f}")
    rc80, rg80 = diff_detectable_r(259, power=0.80)
    res["detectable_r_80"] = rc80

    # ---- first differences and detrended
    print()
    print("=" * 78)
    print("4) Tail regime C: first-difference and rolling-median detrended correlations")
    print("=" * 78)
    xC, yC = a[294:1367], b[294:1367]
    rC = pearson_textbook(xC, yC)
    dxC, dyC = np.diff(xC), np.diff(yC)
    rC_d = pearson_textbook(dxC, dyC)
    rC_d_cov = pearson_covpath(dxC, dyC)
    print(f"  levels        r = {rC:+.6f}")
    print(f"  first diffs   r = {rC_d:+.6f}  (cov-path {rC_d_cov:+.6f}, n={len(dxC)})")
    rx = xC - rolling_median_centred(xC, 15)
    ry = yC - rolling_median_centred(yC, 15)
    rC_det = pearson_textbook(rx, ry)
    # pandas cross-check of the rolling median itself
    rx_p = xC - rolling_median_centred_pandas(xC, 15)
    ry_p = yC - rolling_median_centred_pandas(yC, 15)
    rC_det_p = pearson_textbook(rx_p, ry_p)
    print(f"  detrended (mine)   r = {rC_det:+.6f}")
    print(f"  detrended (pandas) r = {rC_det_p:+.6f}")
    res["tail"] = dict(levels=rC, firstdiff=rC_d, detrended_hand=rC_det,
                       detrended_pandas=rC_det_p)

    # ---- trend share arithmetic + interpretation
    print()
    print("=" * 78)
    print("5) 'trend share' = 1 - (0.0062^2 / 0.6789^2)")
    print("=" * 78)
    pooled_r = pearson_textbook(a, b)
    share = 1 - (rA ** 2) / (pooled_r ** 2)
    print(f"  baseline r^2  = {rA**2:.10e}")
    print(f"  pooled   r^2  = {pooled_r**2:.10f}")
    print(f"  ratio         = {rA**2/pooled_r**2:.10f}")
    print(f"  share         = {share:.8f}")
    res["trend_share"] = share

    # ---- fano
    print()
    print("=" * 78)
    print("6) Fano factors after local detrending (15-pt centred rolling median)")
    print("=" * 78)
    for rname, (lo, hi) in [("A_baseline", (0, 259)), ("C_tail", (294, 1367))]:
        for name, v in [("promoter_like_signal", prom), ("condition_a", a), ("condition_b", b)]:
            f1, f2 = fano_local(v[lo:hi], 15)
            print(f"  {rname:10s} {name:22s} fano_shifted={f1:.6f}  fano_alt={f2:.6f}")
            res.setdefault("fano", {})[f"{rname}|{name}"] = (f1, f2)

    # ---- lead-lag
    print()
    print("=" * 78)
    print("7) Lead-lag scan regime C, lags -25..+25  (x shifted by k)")
    print("=" * 78)
    leadlag = {}
    for k in range(-25, 26):
        if k >= 0:
            xx, yy = xC[:len(xC) - k] if k > 0 else xC, yC[k:]
        else:
            xx, yy = xC[-k:], yC[:len(yC) + k]
        # simpler explicit slicing
        if k >= 0:
            xx = xC[0:len(xC) - k]; yy = yC[k:len(yC)]
        else:
            xx = xC[-k:len(xC)]; yy = yC[0:len(yC) + k]
        leadlag[k] = pearson_textbook(xx, yy)
    keys = sorted(leadlag)
    print("  k = -25..25, r values (condensed):")
    line = "   "
    for k in keys:
        line += f" {k}:{leadlag[k]:+.4f}"
        if (k % 5 == 4):
            print(line); line = "   "
    if line.strip():
        print(line)
    # monotonicity checks
    neg = [leadlag[k] for k in range(-25, 1)]     # increasing in k
    pos = [leadlag[k] for k in range(0, 26)]      # increasing in k
    inc_neg = all(neg[i] < neg[i + 1] for i in range(len(neg) - 1))
    inc_pos = all(pos[i] < pos[i + 1] for i in range(len(pos) - 1))
    print(f"  monotone increasing over k=-25..0: {inc_neg}")
    print(f"  monotone increasing over k=0..+25: {inc_pos}")
    print(f"  r(lag 0) = {leadlag[0]:+.6f}; r(lag+25) = {leadlag[25]:+.6f}; "
          f"r(lag-25) = {leadlag[-25]:+.6f}")
    res["leadlag"] = {str(k): v for k, v in leadlag.items()}

    # ---- boundary sensitivity, reproducing report's 7x7 grid / 37 variants
    print()
    print("=" * 78)
    print("8) Boundary sensitivity: 7x7 grid with the 12 excluded pairs")
    print("=" * 78)
    Aends = [218, 238, 248, 258, 268, 278, 298]
    Bends = [253, 273, 283, 293, 303, 313, 333]
    with open("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/findings.json") as f:
        fj = json.load(f)
    pairs_reported = {(v["a_end"], v["b_end"]): v for v in fj["finding1_regimes"]["boundary_sensitivity"]}
    bad = []
    vals = []
    for ae in Aends:
        for be in Bends:
            nA = ae + 1
            # baseline = rows [0, ae]; sweep+tail excluded per report; tail = ?
            # report's tail_n = 1367 - (be+1) so tail = rows [be+1, 1367)
            tail_rows = np.arange(be + 1, 1367)
            nb = len(tail_rows)
            if nb < 3 or nA < 3:
                continue
            r_base = pearson_textbook(a[:nA], b[:nA])
            r_tail = pearson_textbook(a[tail_rows], b[tail_rows])
            vals.append((ae, be, nA, nb, r_base, r_tail))
            if (ae, be) in pairs_reported:
                rep = pairs_reported[(ae, be)]
                okb = abs(rep["baseline_ab_r"] - r_base) < 1e-6
                okt = abs(rep["tail_ab_r"] - r_tail) < 1e-6
                okn = (rep["baseline_n"] == nA and rep["tail_n"] == nb)
                if not (okb and okt and okn):
                    bad.append((ae, be, rep["baseline_ab_r"], r_base, rep["tail_ab_r"], r_tail))
    # report re-computed max
    rmax = max(v[4] for v in vals); rmin = min(v[4] for v in vals)
    print(f"  n variants I computed = {len(vals)}  (report: 37)")
    print(f"  baseline_ab_r range = [{rmin:+.6f}, {rmax:+.6f}]")
    print(f"  report range        = [+0.006174, +0.952214]")
    print(f"  variants where my baseline r differs from reported by >1e-6: {len(bad)}")
    for x in bad[:10]:
        print("   ", x)
    res["boundary_range"] = [rmin, rmax]

    # ---- extra: leave-one-out of regime A rows to test null robustness
    print()
    print("=" * 78)
    print("9) Additional null-honesty checks on baseline a~b")
    print("=" * 78)
    # permutation test, different seed and more draws than report (which used none stated here)
    rng = np.random.default_rng(987654321)
    perm = np.empty(20000)
    for i in range(20000):
        perm[i] = pearson_textbook(xA, rng.permutation(yA))
    p_perm = (np.sum(np.abs(perm) >= abs(rA)) + 1) / (20000 + 1)
    print(f"  permutation test (20000 draws, seed 987654321): p = {p_perm:.5f}")
    print(f"  permutation |r| 95th pct = {np.quantile(np.abs(perm), 0.95):.5f}")
    # t-style p-value for r, computed by hand with a permutation-free method:
    # t = r sqrt((n-2)/(1-r^2)), then two-sided p via a beta-function-free approach:
    # use the exact null distribution of r under bivariate normal via numerical integration?
    # Simpler: report t and compare to a large-sample normal approx and an exact
    # permutation. Also bootstrap CI for r:
    bs_r = np.empty(20000)
    idx = rng.integers(0, len(xA), size=(20000, len(xA)))
    for i in range(20000):
        bs_r[i] = pearson_textbook(xA[idx[i]], yA[idx[i]])
    print(f"  bootstrap 95% CI for r (percentile, 20000 draws): "
          f"[{np.quantile(bs_r, 0.025):+.5f}, {np.quantile(bs_r, 0.975):+.5f}]")
    t_stat = rA * np.sqrt((len(xA) - 2) / (1 - rA ** 2))
    print(f"  t statistic = {t_stat:.6f} (df=257)")
    # rank correlation via Spearman computed by hand
    def spearman_hand(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return pearson_textbook(rx, ry)
    print(f"  Spearman (hand rank transform) a~b in A = {spearman_hand(xA, yA):+.6f}")
    print(f"  Spearman a~b in C = {spearman_hand(xC, yC):+.6f}")
    print(f"  Spearman a~b pooled = {spearman_hand(a, b):+.6f}")

    # effective sample size concern: autocorrelation of a and b in baseline
    def lag1(x):
        return pearson_textbook(x[:-1], x[1:])
    print(f"  lag-1 autocorr in A: a={lag1(xA):+.4f}, b={lag1(yA):+.4f}; in C: "
          f"a={lag1(xC):+.6f}, b={lag1(yC):+.6f}")

    with open("/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/verification/verifier_finding3_results.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print()
    print("Wrote /data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/verification/verifier_finding3_results.json")


if __name__ == "__main__":
    main()