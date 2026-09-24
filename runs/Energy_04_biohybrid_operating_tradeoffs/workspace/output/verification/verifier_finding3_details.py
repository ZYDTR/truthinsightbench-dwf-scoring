#!/usr/bin/env python3
"""Second-pass detail checks for Finding 3 verification:
Fano convention identification, null-honesty checks, trend-share interpretability,
boundary-grid pattern, exact power for r, lead-lag curve cross-check, Spearman ties.
"""
import json
import numpy as np
import pandas as pd
from math import erf, sqrt, pi

DATA = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/data/time_response_curves.tsv"
FIND = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project/output/results/findings.json"

df = pd.read_csv(DATA, sep="\t")
prom = df.promoter_like_signal.to_numpy(float)
a = df.condition_a.to_numpy(float)
b = df.condition_b.to_numpy(float)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    return np.sum((x - xm) * (y - ym)) / np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))


with open(FIND) as f:
    fj = json.load(f)


def rollmed(x, w, min_periods):
    return pd.Series(x).rolling(w, center=True, min_periods=min_periods).median().to_numpy()


# ---------------- Fano conventions ----------------
print("=" * 78)
print("FANO CONVENTION IDENTIFICATION")
print("=" * 78)
targets = fj["fano_by_regime"]


def fano_variants(x, w=15):
    """Return dict of candidate Fano values under different edge/mean conventions."""
    out = {}
    # v1: min_periods=1 pad; resid = x-med; var/mean(x)
    rm1 = rollmed(x, w, 1)
    r1 = x - rm1
    out["pad1_var_resid_over_meanx"] = np.var(r1, ddof=1) / x.mean()
    # v2: min_periods=w (NaN edges); drop NaN pairs; var/mean(x) over valid subset
    rm2 = rollmed(x, w, w)
    m = ~np.isnan(rm2)
    r2 = (x - rm2)[m]
    out["dropnan_var_resid_over_meanx_valid"] = np.var(r2, ddof=1) / x[m].mean()
    # v3: min_periods=w; fill NaN residual with 0; var over all, mean over all
    r3 = np.where(m, x - np.where(m, rm2, x), 0.0)
    out["nan2zero_var_over_meanx_all"] = np.var(r3, ddof=1) / x.mean()
    # v4: min_periods=w; residual on valid subset; ddof=0
    out["dropnan_var0_resid_over_meanx_valid"] = np.var(r2, ddof=0) / x[m].mean()
    # v5: pad1, var/mean of shifted series  (x - med + mean(x))
    s = r1 + x.mean()
    out["pad1_shifted_var_over_mean_of_shift"] = np.var(s, ddof=1) / s.mean()
    # v6: rolling MEAN instead of median (check the report is really a median)
    rmean = pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
    rm_ = x - rmean
    out["pad1_rollMEAN_var_over_meanx"] = np.var(rm_, ddof=1) / x.mean()
    return out


for reg, lo, hi in [("A_baseline", 0, 259), ("C_tail", 294, 1367)]:
    for cname, v in [("promoter_like_signal", prom), ("condition_a", a), ("condition_b", b)]:
        x = v[lo:hi]
        tgt = targets[reg][cname]["fano_local"]
        vs = fano_variants(x)
        best = min(vs.items(), key=lambda kv: abs(kv[1] - tgt))
        print(f"{reg:10s} {cname:22s} reported={tgt:.10f}")
        print(f"           best variant: {best[0]:38s} = {best[1]:.10f}  |diff|={abs(best[1]-tgt):.2e}")
        # also show plain var/mean for reference
        print(f"           fano_all reported={targets[reg][cname]['fano_all']:.6f} "
              f"recomputed={np.var(x, ddof=1)/x.mean():.6f}")

# ---------------- null honesty checks ----------------
print()
print("=" * 78)
print("NULL-HONESTY CHECKS on baseline a~b (regime A)")
print("=" * 78)
xA, yA = a[:259], b[:259]
print(f"  r = {pearson(xA, yA):+.6f};  Spearman with AVERAGE ranks (ties):",
      end=" ")
rx = pd.Series(xA).rank().to_numpy()
ry = pd.Series(yA).rank().to_numpy()
print(f"{pearson(rx, ry):+.6f}")
# conditional mean of b by a
print("  mean(condition_b | condition_a = k):")
for k in range(0, 5):
    sel = xA == k
    print(f"    a={k}: n={int(sel.sum()):3d}  mean_b={yA[sel].mean():.4f}  sd_b={yA[sel].std(ddof=1) if sel.sum()>1 else float('nan'):.4f}")
# rank-based / binarised checks
print(f"  P(b>0 | a>=1) = {np.mean(yA[xA>=1] > 0):.4f}   P(b>0 | a==0) = {np.mean(yA[xA==0] > 0):.4f}")
# a chi-square-free difference of proportions with a permutation p
obs = np.mean(yA[xA >= 1] > 0) - np.mean(yA[xA == 0] > 0)
rng = np.random.default_rng(20240101)
cnt = 0
for _ in range(20000):
    perm = rng.permutation(xA)
    d = np.mean(yA[perm >= 1] > 0) - np.mean(yA[perm == 0] > 0)
    if abs(d) >= abs(obs):
        cnt += 1
print(f"  diff in P(b>0) = {obs:+.4f}; permutation p (20000 draws, seed 20240101) = {(cnt+1)/20001:.4f}")
# nonlinear: correlation of ranks and of sqrt
print(f"  Pearson on sqrt-transformed: {pearson(np.sqrt(xA), np.sqrt(yA)):+.6f}")
# nearest-neighbour level check: mutual information style (binned)
xb = np.minimum(xA, 4).astype(int)
yb = np.digitize(yA, [0.5, 1.5, 2.5])
import itertools
n = len(xA)
Hx = -sum((np.sum(xb == i) / n) * np.log(np.sum(xb == i) / n) for i in range(5) if np.sum(xb == i) > 0)
Hy = -sum((np.sum(yb == j) / n) * np.log(np.sum(yb == j) / n) for j in range(4) if np.sum(yb == j) > 0)
Hxy = 0.0
for i, j in itertools.product(range(5), range(4)):
    p = np.mean((xb == i) & (yb == j))
    if p > 0:
        Hxy -= p * np.log(p)
print(f"  binned MI(a;b) = {Hxy - Hx - Hy:.5f} nats  (Hx={Hx:.4f}, Hy={Hy:.4f})")

# ---------------- trend-share interpretability ----------------
print()
print("=" * 78)
print("TREND-SHARE STATISTIC: arithmetic + alternative decompositions")
print("=" * 78)
rA = pearson(a[:259], b[:259])
rP = pearson(a, b)
rC = pearson(a[294:], b[294:])
dx, dy = np.diff(a[294:]), np.diff(b[294:])
rCf = pearson(dx, dy)


def rollmed1(x, w=15):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


rxC = a[294:] - rollmed1(a[294:])
ryC = b[294:] - rollmed1(b[294:])
rCd = pearson(rxC, ryC)
print(f"  reported formula 1 - (r_A^2/r_pool^2)          = {1 - rA**2/rP**2:.8f}")
print(f"  with rounded inputs 1-(0.0062^2/0.6789^2)      = {1-(0.0062**2)/(0.6789**2):.8f}")
print(f"  tail-levels analogue 1-(r_C^2/r_pool^2)        = {1 - rC**2/rP**2:.6f}")
print(f"  tail-first-diff analogue 1-(r_Cf^2/r_C^2)      = {1 - rCf**2/rC**2:.6f}")
print(f"  tail-detrended analogue 1-(r_Cd^2/r_C^2)       = {1 - rCd**2/rC**2:.6f}")
# variance decomposition within the tail: r2 of b~a vs b~a+trend proxies
# (a) demean within regime (removes between-regime means)
aC, bC = a[294:], b[294:]
r2_pooled = rP ** 2
# demean a and b by regime means -> pooled within-regime correlation
groups = np.concatenate([np.zeros(259), np.ones(35), 2 * np.ones(1073)])
ad = a - np.array([a[groups == g].mean() for g in range(3)])[groups.astype(int)]
bd = b - np.array([b[groups == g].mean() for g in range(3)])[groups.astype(int)]
r_within = pearson(ad, bd)
print(f"  r pooled                      = {rP:+.6f}  (r^2={rP**2:.6f})")
print(f"  r after regime-mean removal   = {r_within:+.6f}  (r^2={r_within**2:.6f})")
print(f"  share of r^2 removed by regime means alone = {1-r_within**2/rP**2:.6f}")
# residualise BOTH series on their own row-index trend then correlate (within-regime)
# r^2 of b on a controlling for trend: regress b on a + t (tail)
t = np.arange(len(aC), dtype=float)
X = np.column_stack([np.ones_like(t), aC, t])
beta, *_ = np.linalg.lstsq(X, bC, rcond=None)
fit = X @ beta
sse = np.sum((bC - fit) ** 2)
sst = np.sum((bC - bC.mean()) ** 2)
print(f"  tail: R^2 of b~a alone          = {rC**2:.6f}")
print(f"  tail: R^2 of b~a + linear row-t = {1-sse/sst:.6f}  (change {1-sse/sst-rC**2:+.6f})")
# partial correlation of a,b given row index within the tail
res_a = aC - np.column_stack([np.ones_like(t), t]) @ np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), aC, rcond=None)[0]
res_b = bC - np.column_stack([np.ones_like(t), t]) @ np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), bC, rcond=None)[0]
print(f"  tail: partial r(a,b | row-t)    = {pearson(res_a, res_b):+.6f}")

# ---------------- boundary grid pattern ----------------
print()
print("=" * 78)
print("BOUNDARY GRID PATTERN (per a_end)")
print("=" * 78)
bs = fj["finding1_regimes"]["boundary_sensitivity"]
for ae in sorted(set(v["a_end"] for v in bs)):
    sub = [v for v in bs if v["a_end"] == ae]
    print(f"  a_end={ae:3d} (n={sub[0]['baseline_n']:3d}) baseline_ab_r: " +
          ", ".join(f"{v['baseline_ab_r']:+.4f}" for v in sub))
print("  A|B cut at row 258 (true boundary) -> baseline r ~ 0.006; any a_end >= 268 includes")
print("  sweep rows 259-268 whose a,b blow up together, so r jumps to ~0.95.")

# ---------------- exact power for r (noncentral t via numerical integration) ----------------
print()
print("=" * 78)
print("EXACT POWER FOR PEARSON r (noncentral t), n=259, alpha=0.05 two-sided")
print("=" * 78)


def chi2_pdf(v, k):
    return np.where(v > 0, v ** (k / 2 - 1) * np.exp(-v / 2) / (2 ** (k / 2) * _gamma(k / 2)), 0.0)


def _gamma(z):
    # Lanczos approximation
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z < 0.5:
        return pi / (np.sin(pi * z) * _gamma(1 - z))
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t_ = z + g + 0.5
    return np.sqrt(2 * pi) * t_ ** (z + 0.5) * np.exp(-t_) * x


def ncdf(z):
    return 0.5 * (1.0 + erf(np.asarray(z) / sqrt(2.0)))


def nct_cdf(tval, df, delta, ngrid=400000, vmax=400.0):
    """CDF of noncentral t: int_0^inf Phi(t sqrt(v/df) - delta) chi2_df(v) dv."""
    v = np.linspace(1e-9, vmax, ngrid)
    w = chi2_pdf(v, df)
    integ = ncdf(tval * np.sqrt(v / df) - delta) * w
    return np.trapezoid(integ, v)


def power_exact(rho, n, alpha=0.05):
    df = n - 2
    delta = rho * np.sqrt(n) / np.sqrt(1 - rho ** 2)  # ncp for the t statistic
    # critical value: use alpha quantile of central t via bisection on nct_cdf(delta=0)
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if nct_cdf(mid, df, 0.0) < 1 - alpha / 2:
            lo = mid
        else:
            hi = mid
    tc = 0.5 * (lo + hi)
    return 1 - nct_cdf(tc, df, delta) + nct_cdf(-tc, df, delta), tc


# sanity: t critical for df=257 should be ~1.969
_, tc257 = power_exact(0.0, 259)
print(f"  critical |t| (df=257, alpha=0.05) = {tc257:.4f}  (textbook ~1.969)")
for rho in [0.17, 0.174, 0.18, 0.185, 0.19]:
    pw, _ = power_exact(rho, 259)
    print(f"  power(rho={rho:.3f}, n=259) = {pw:.4f}")
# root-find r giving 80% power
lo, hi = 0.01, 0.5
for _ in range(30):
    mid = 0.5 * (lo + hi)
    pw, _ = power_exact(mid, 259)
    if pw < 0.80:
        lo = mid
    else:
        hi = mid
print(f"  EXACT detectable rho at 80% power = {0.5*(lo+hi):.6f}")
print(f"  Fisher-z approximation             = 0.173331")
print(f"  report claim                       = ~0.174")
print(f"  rule-of-thumb 2.8/sqrt(n)          = {2.8/np.sqrt(259):.6f}")

# ---------------- leadlag cross-check vs findings.json ----------------
print()
print("=" * 78)
print("LEAD-LAG CROSS-CHECK vs findings.json curve")
print("=" * 78)
xC, yC = a[294:], b[294:]
reported_curve = dict({int(k): v for k, v in fj["finding3_coupling"]["leadlag_C_tail_curve"]})
maxdiff = 0.0
for k in range(-25, 26):
    if k >= 0:
        mine = pearson(xC[0:len(xC) - k], yC[k:])
    else:
        mine = pearson(xC[-k:], yC[0:len(yC) + k])
    maxdiff = max(maxdiff, abs(mine - reported_curve[k]))
print(f"  max |my r - reported r| over 51 lags = {maxdiff:.3e}")
print(f"  best reported lag = {fj['finding3_coupling']['leadlag_C_tail_best']}")

# ---------------- autocorrelation / effective n in tail ----------------
print()
print("=" * 78)
print("SMOOTHNESS CONTEXT (why lead-lag cannot resolve a delay)")
print("=" * 78)
print(f"  tail lag-1 autocorr: a={pearson(xC[:-1], xC[1:]):.6f}, b={pearson(yC[:-1], yC[1:]):.6f}")
print(f"  tail r at max lag searched (25) = {reported_curve[25]:.4f}; window +-25 is only "
      f"{50/len(xC)*100:.1f}% of the series length")

# ---------------- significance of tail first-difference r ----------------
print()
print("=" * 78)
print("IS THE TAIL RESIDUAL ASSOCIATION ZERO? (significance of first-diff r)")
print("=" * 78)
r = rCf
nn = len(dx)
tt = r * np.sqrt((nn - 2) / (1 - r ** 2))
print(f"  first-difference r = {r:+.6f}, n={nn}, t = {tt:.4f}")
# two-sided p via normal approx and via permutation with a different seed
from math import erfc
p_norm = erfc(abs(tt) / sqrt(2))
print(f"  p (normal approx) = {p_norm:.3e}")
rng = np.random.default_rng(555)
cnt = 0
for _ in range(20000):
    if abs(pearson(dx, rng.permutation(dy))) >= abs(r):
        cnt += 1
print(f"  permutation p (20000 draws, seed 555) = {(cnt+1)/20001:.6f}")
# detrended residual association significance
rd = rCd
nd = len(rxC)
td = rd * np.sqrt((nd - 2) / (1 - rd ** 2))
print(f"  detrended tail r = {rd:+.6f}, n={nd}, t = {td:.4f}, p(normal) = {erfc(abs(td)/sqrt(2)):.3e}")