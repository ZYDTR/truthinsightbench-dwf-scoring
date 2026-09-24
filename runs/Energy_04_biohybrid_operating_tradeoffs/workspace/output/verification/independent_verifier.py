#!/usr/bin/env python3
"""Independent verification of cross-cutting / structural claims in output/Result.md.

Written from scratch by the verifier.  Reads ONLY data/*.tsv (and, for
cross-checking declared quantities, data_manifest.json + findings.json).
Does NOT import or execute output/analysis/analysis.py.

Every number printed under a "CLAIM" tag is compared against the value quoted in
Result.md / the task brief.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-995a498d-d5c1-4e1e-99f5-01601e51abe5/project"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

P = print


def hdr(t):
    P("\n" + "=" * 78)
    P(t)
    P("=" * 78)


def chk(name, claim, got, tol=5e-4, note=""):
    """Compare a reported claim to a recomputed value."""
    if got is None:
        P(f"[{name}] report={claim!r} recomputed=None -> COULD_NOT_VERIFY {note}")
        return
    try:
        c = float(claim)
        ok = abs(c - float(got)) <= tol * max(1.0, abs(c)) if tol >= 0 else False
    except (TypeError, ValueError):
        ok = str(claim) == str(got)
    P(f"[{name}] report={claim!r} recomputed={got!r} -> {'MATCH' if ok else 'MISMATCH'} {note}")
    return ok


# ---------------------------------------------------------------- item 1 ----
hdr("1. Data inventory: shapes, bytes, sha256")
files = ["time_response_curves.tsv", "perturbation_time_response.tsv", "dose_response.tsv"]
declared = json.load(open(os.path.join(ROOT, "data_manifest.json")))
man = {f["path"]: f for f in declared["files"]}
frames = {}
for f in files:
    path = os.path.join(DATA, f)
    raw = open(path, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    df = pd.read_csv(path, sep="\t")
    frames[f] = df
    P(f"{f}: shape={df.shape} bytes={len(raw)} sha256={h}")
    P(f"   manifest: bytes={man[f]['bytes']} sha256={man[f]['sha256']} match={h == man[f]['sha256'] and len(raw) == man[f]['bytes']}")
    P(f"   columns={list(df.columns)}")
cur, pert, dose = frames[files[0]], frames[files[1]], frames[files[2]]

# ---------------------------------------------------------------- item 2 ----
hdr("2. Missingness")
for f, df in frames.items():
    n_nan = int(df.isna().sum().sum())
    P(f"{f}: total NaN = {n_nan}  per-column={df.isna().sum().to_dict()}")
P("dose response_sd all-missing:", bool(dose["response_sd"].isna().all()), "n_missing=", int(dose["response_sd"].isna().sum()))
P("dose dose/response_mean complete:", bool(dose[["dose", "response_mean"]].notna().all().all()))
# also look for blank/whitespace-as-missing in the raw text
for f in files:
    txt = open(os.path.join(DATA, f)).read().splitlines()
    bad = [i for i, ln in enumerate(txt) if "nan" in ln.lower() or "\t\t" in ln]
    P(f"{f}: lines with nan/empty-field tokens: {bad[:5]} (n={len(bad)})")

# ---------------------------------------------------------------- item 3 ----
hdr("3. Label collision / column inventories and scales")
P("columns curves:", list(cur.columns))
P("columns perturbation:", list(pert.columns))
P("columns dose:", list(dose.columns))
P("'condition_c' in curves only:",
  any("condition_c" in c for c in cur.columns),
  any("condition_c" in c for c in pert.columns),
  any("condition_c" in c for c in dose.columns))
P("'condition_a'/'condition_b' in both curves+perturbation:",
  all(any(s in c for c in cur.columns) for s in ["condition_a", "condition_b"]),
  all(any(s in c for c in pert.columns) for s in ["condition_a", "condition_b"]))
for c in ["time", "promoter_like_signal", "condition_a", "condition_b"]:
    v = cur[c]
    P(f"curves {c}: int-valued={bool(np.all(v == np.round(v)))} min={v.min()} max={v.max()} mean={v.mean():.4f}")
P("perturbation scales:")
for c in ["control_mean", "condition_a_mean", "condition_b_mean", "control_sd", "condition_a_sd", "condition_b_sd"]:
    v = pert[c]
    P(f"  {c}: min={v.min():.5f} max={v.max():.5f}")
P("perturbation SD range:", pert[[c for c in pert.columns if c.endswith('_sd')]].min().min(),
  pert[[c for c in pert.columns if c.endswith('_sd')]].max().max())

# ---------------------------------------------------------------- item 4 ----
hdr("4. Duplicates and replicate identifiers")
allcols = list(cur.columns)
dup_all = int(cur.duplicated().sum())
P("duplicate rows on all 7 stored columns (extra copies):", dup_all)
P("unique rows:", cur.drop_duplicates().shape[0], "of", len(cur))
grp = cur.groupby(allcols).size()
P("rows in duplicate groups:", int(grp[grp > 1].sum()), "groups>1:", int((grp > 1).sum()), "max group:", int(grp.max()))
P("duplicates by regime A/B/C:", int(cur.iloc[:259].duplicated().sum()),
  int(cur.iloc[259:294].duplicated().sum()), int(cur.iloc[294:].duplicated().sum()))
P("duplicates excluding only 'time':", int(cur.drop(columns=["time"]).duplicated().sum()))
P("duplicates on raw channels (no time, no fits):",
  int(cur[["promoter_like_signal", "condition_a", "condition_b"]].duplicated().sum()))
id_tokens = ["rep", "batch", "subject", "session", "cycle", "run", "trial", "unit", "id"]
for f, df in frames.items():
    hits = [c for c in df.columns for t in id_tokens if t in c.lower()]
    P(f"{f}: identifier-like columns -> {hits}")

# ---------------------------------------------------------------- item 5 ----
hdr("5. scipy unavailable + _meta versions")
try:
    import scipy  # noqa
    scipy_ok = True
except Exception as e:
    scipy_ok = False
    P("scipy import raised:", type(e).__name__, e)
P("scipy_available (observed):", scipy_ok)
meta = json.load(open(os.path.join(OUT, "results", "findings.json")))["_meta"]
P("_meta:", {k: meta[k] for k in ["numpy", "pandas", "matplotlib", "scipy_available", "rng_seed"]})
P("actual versions: numpy", np.__version__, "pandas", pd.__version__, "python", sys.version.split()[0])
import matplotlib
P("matplotlib", matplotlib.__version__)

# ---------------------------------------------------------------- item 6 ----
hdr("6. Detectable |r| at 80% power, n=259 (two-sided alpha=0.05)")

def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# (a) crude normal approximation
z_a, z_b = 1.959963985, 0.841621234
r_crude = (z_a + z_b) / math.sqrt(259)
# (b) Fisher-z approximation
r_fisher = math.tanh((z_a + z_b) / math.sqrt(259 - 3))

# (c) exact noncentral-t power via quadrature + incomplete beta for t_crit
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1 - x) / b


df = 259 - 2
# two-sided p for t = I_{df/(df+t^2)}(df/2, 1/2)
def two_sided_p(t):
    return betai(df / 2.0, 0.5, df / (df + t * t))


lo, hi = 0.0, 20.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if two_sided_p(mid) > 0.05:
        lo = mid
    else:
        hi = mid
t_crit = 0.5 * (lo + hi)
P(f"t_crit(df=257, alpha=0.05 two-sided) = {t_crit:.6f} (check p={two_sided_p(t_crit):.6f})")

# chi-square(df) density on a grid; integrate power for a given rho
vg = np.linspace(1e-9, df + 9 * math.sqrt(2 * df), 40001)
logf = (df / 2 - 1) * np.log(vg) - vg / 2 - (df / 2) * math.log(2) - math.lgamma(df / 2)
fv = np.exp(logf)
fv = fv / np.trapezoid(fv, vg)  # normalise (guard against grid truncation)
sqv = np.sqrt(vg / df)


def power_exact(rho):
    delta = rho * math.sqrt(df) / math.sqrt(1 - rho ** 2)
    upper = np.array([phi(t_crit * s - delta) for s in sqv])
    lower = np.array([phi(-t_crit * s - delta) for s in sqv])
    pow_ = 1.0 - np.trapezoid((upper - lower) * fv, vg)
    return pow_


P(f"quadrature sanity: power(rho=0) = {power_exact(0.0):.4f} (should be 0.05)")
lo, hi = 0.0, 0.99
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if power_exact(mid) < 0.8:
        lo = mid
    else:
        hi = mid
r_exact = 0.5 * (lo + hi)
P(f"exact noncentral-t detectable rho (80% power) = {r_exact:.5f}")
P(f"crude (z_a+z_b)/sqrt(n)      = {r_crude:.5f}")
P(f"Fisher-z tanh((z_a+z_b)/sqrt(n-3)) = {r_fisher:.5f}")
P(f"power at rho=0.174: {power_exact(0.174):.4f}; at rho={r_exact:.4f}: {power_exact(r_exact):.4f}")

# ---------------------------------------------------------------- regime def
hdr("Regime definitions (row index based, as the report declares)")
A = cur.iloc[:259]
B = cur.iloc[259:294]
C = cur.iloc[294:].reset_index(drop=True)
P("A rows 0-258 n=", len(A), "| B rows 259-293 n=", len(B), "| C rows 294-1366 n=", len(C))
P("pcts: A %.2f%% B %.2f%% C %.2f%%" % (100 * len(A) / 1367, 100 * len(B) / 1367, 100 * len(C) / 1367))
P("A promoter range", A.promoter_like_signal.min(), A.promoter_like_signal.max(),
  "| C promoter range", C.promoter_like_signal.min(), C.promoter_like_signal.max())
P("A cond_a range", A.condition_a.min(), A.condition_a.max(), "| C cond_a range", C.condition_a.min(), C.condition_a.max())
P("A cond_b range", A.condition_b.min(), A.condition_b.max(), "| C cond_b range", C.condition_b.min(), C.condition_b.max())
P("C rows with cond_b <= 3:", int((C.condition_b <= 3).sum()), "of", len(C), "=%.4f" % (C.condition_b <= 3).mean())
P("C rows with cond_b <= A max(3):", int((C.condition_b <= 3).sum()))
P("A cond_b exactly 0:", int((A.condition_b == 0).sum()), "of", len(A))
P("fitted cols all zero in A:", bool((A[["condition_c", "fit_a", "fit_b"]] == 0).all().all()))
for c in ["condition_c", "fit_a", "fit_b"]:
    nz = np.nonzero(cur[c].values != 0)[0]
    P(f"  {c}: n_zero={int((cur[c].values == 0).sum())} first_nonzero_row={int(nz[0])}")
P("rows with time>=2 outside 259-293:", np.nonzero((cur.time.values >= 2) & ~np.isin(np.arange(len(cur)), np.arange(259, 294)))[0])
P("distinct time B:", sorted(B.time.unique().tolist()))
P("n distinct time B:", B.time.nunique(), "| regime C:", sorted(C.time.unique().tolist()))

# R^2 of promoter ~ time as categorical (between-group SS / total SS)
def r2_cat_time(sub):
    y = sub.promoter_like_signal.values.astype(float)
    g = sub.time.values
    gm = pd.Series(y).groupby(g).mean()
    pred = pd.Series(g).map(gm).values
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

P("R2 categorical time pooled (promoter) = %.6f  [report 0.6674]" % r2_cat_time(cur))
P("R2 categorical time regime C (promoter) = %.6f  [report 0.0002]" % r2_cat_time(C))
P("R2 categorical time pooled cond_a = %.6f (findings 0.53489)" % r2_cat_time(cur.assign(promoter_like_signal=cur.condition_a)))
P("R2 categorical time pooled cond_b = %.6f (findings 0.96315)" % r2_cat_time(cur.assign(promoter_like_signal=cur.condition_b)))

# leave-one-regime-out linear trend on time
def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b = np.polyfit(x, y, 1)
    pred = b[0] * x + b[1]
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b[0], r2

P("ALL rows slope/R2 (promoter~time): %.4f / %.6f  [report 0.900 / 0.250]" % ols(cur.time, cur.promoter_like_signal))
d_noB = cur.drop(index=range(259, 294))
P("drop B slope/R2: %.4f / %.8f  [report 15.98 / 0.000034]" % ols(d_noB.time, d_noB.promoter_like_signal))

# ---------------------------------------------------------------- item 3f' --
hdr("3-correlations by regime (covariance formula + rank transform)")
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum() / (len(x) - 1)
    return cov / (x.std(ddof=1) * y.std(ddof=1))

def rank(v):
    s = pd.Series(np.asarray(v, float)).rank().values
    return s

for nm, sub in [("A", A), ("B", B), ("C", C), ("WHOLE", cur)]:
    for x, y in [("promoter_like_signal", "condition_a"), ("promoter_like_signal", "condition_b"), ("condition_a", "condition_b")]:
        P(f"  {nm} {x}~{y}: pearson={pearson(sub[x], sub[y]):.6f} spearman={pearson(rank(sub[x]), rank(sub[y])):.6f}")
P("report table: A 0.1095/-0.0474/+0.0062 ; B 0.9967/0.4151/+0.3938 ; C 0.9834/0.9899/+0.9523 ; WHOLE 0.9835/0.7301/+0.6789")
P("report spearman (C a~b) 0.9324, whole 0.9524")

# first differences
fd = {nm: sub.diff().dropna() for nm, sub in [("A", A), ("B", B), ("C", C)]}
for nm, sub in fd.items():
    P(f"  first-diff {nm} a~b r={pearson(sub.condition_a, sub.condition_b):.6f} [report C 0.1551]")

# centred rolling median (window 15) detrend
def rollmed(v, w=15):
    return pd.Series(v).rolling(w, center=True, min_periods=1).median().values

for nm, sub in [("A", A), ("B", B), ("C", C)]:
    da = sub.condition_a.values - rollmed(sub.condition_a)
    db = sub.condition_b.values - rollmed(sub.condition_b)
    P(f"  detrended(rollmed15) {nm} a~b r={pearson(da, db):.6f} [report C 0.3051, A -0.037]")
P("trend share of pooled ab r2 = 1 - (0.0062^2/0.6789^2) =",
  1 - (0.006174214588009049 ** 2 / 0.6788945956838975 ** 2))
P("baseline OLS slope b~a:", np.polyfit(A.condition_a, A.condition_b, 1)[0],
  " [report +0.00428]; full-range change:", np.polyfit(A.condition_a, A.condition_b, 1)[0] * 4,
  " vs cond_b SD", A.condition_b.std(ddof=1))
# baseline r CI (Fisher-z, n=259)
z = math.atanh(0.006174214588009049)
se = 1 / math.sqrt(259 - 3)
P("baseline r 95%% CI Fisher-z: [%.4f, %.4f] [report -0.116, +0.128]" % (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)))

# lead-lag
hdr("Lead-lag scan, regime C (a vs b, lags -25..25)")
a = C.condition_a.values.astype(float); b = C.condition_b.values.astype(float)
ll = []
for lag in range(-25, 26):
    if lag < 0:
        x, y = a[-lag:], b[:lag]
    elif lag > 0:
        x, y = a[:-lag], b[lag:]
    else:
        x, y = a, b
    ll.append((lag, pearson(x, y)))
P("lag 0 r=%.6f ; lag +25 r=%.6f ; lag -25 r=%.6f" % (ll[25][1], ll[50][1], ll[0][1]))
P("all |lag|>=1 r > r_lag0 ?", all(v > ll[25][1] for k, v in ll if k != 0))
P("argmax lag:", max(ll, key=lambda t: t[1]))
P("values at lags 1..5:", [(k, round(v, 4)) for k, v in ll[26:31]])

# fano factors - try plausible local detrending definitions
hdr("Fano factors after local detrending (search for a reproducible definition)")
for nm, sub in [("A", A), ("B", B), ("C", C)]:
    row = []
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        v = sub[c].values.astype(float)
        m = v.mean()
        # candidate 1: var of first differences /2, /mean
        d = np.diff(v); cand1 = (d.var(ddof=1) / 2) / m
        # candidate 2: residual from 15-pt centred rolling mean, var / mean of raw
        res = v - pd.Series(v).rolling(11, center=True, min_periods=1).mean().values
        cand2 = res.var(ddof=1) / m
        # candidate 3: residual from centred rolling median 15
        res3 = v - rollmed(v, 15)
        cand3 = res3.var(ddof=1) / m
        row.append((round(cand1, 3), round(cand2, 3), round(cand3, 3)))
    P(f"{nm}: (diff/2, rollmean11, rollmed15) = {row}")
P("report: A 0.98,1.02,0.95 ; C 1.94,1.11,1.50 (findings fano_local: A .9848/1.0154/.9488; C 1.9432/1.1081/1.5039)")

# ---------------------------------------------------------- item 7 shuffle --
hdr("7. Tail-decline shuffle null (own seed, 100000 draws)")
y = C.promoter_like_signal.values.astype(float)
x = np.arange(len(y), dtype=float)
slope, intercept = np.polyfit(x, y, 1)
pred = slope * x + intercept
obs_r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
P("observed slope=%.8f R2=%.10f [report -1.1266 / 0.393179]" % (slope, obs_r2))
rng = np.random.default_rng(987654321)
N = 100000
n = len(y)
sx = x - x.mean()
ssx = (sx ** 2).sum()
r2s = np.empty(N)
for i in range(N):
    yp = rng.permutation(y)
    b1 = (sx * (yp - yp.mean())).sum() / ssx
    b0 = yp.mean() - b1 * x.mean()
    res = yp - (b0 + b1 * x)
    r2s[i] = 1 - (res ** 2).sum() / ((yp - yp.mean()) ** 2).sum()
P("null mean R2 = %.8f [report 0.000935]" % r2s.mean())
P("null 95%% percentile CI = [%.4g, %.4g] [report 5.35e-07, 0.00457]" % (np.percentile(r2s, 2.5), np.percentile(r2s, 97.5)))
n_ge = int((r2s >= obs_r2).sum())
P("p = (n>=obs+1)/(N+1) = %.6f ; draws >= obs = %d of %d [report p<0.001, 0/1000]" %
  (((n_ge + 1) / (N + 1)), n_ge, N))
P("null R2 max over draws:", r2s.max())

# block bootstrap attempt (documented failure, mean R2 ~ 0.024)
hdr("7a. Block bootstrap attempt (25-row blocks, random concatenation)")
for block in (25,):
    vals = []
    for seed in range(200):
        rng2 = np.random.default_rng(seed)
        starts = np.arange(0, n, block)
        blocks = [y[s:s + block] for s in starts]
        order = rng2.permutation(len(blocks))
        yp = np.concatenate([blocks[i] for i in order])
        xx = np.arange(len(yp), dtype=float)
        b1 = np.polyfit(xx, yp, 1)[0]
        b0 = np.polyfit(xx, yp, 1)[1]
        res = yp - (b0 + b1 * xx)
        vals.append(1 - (res ** 2).sum() / ((yp - yp.mean()) ** 2).sum())
    P(f"block={block}: mean R2 over 200 draws = {np.mean(vals):.5f} [report 0.024]")

# monotonicity permutation (own seed, 20000 draws)
hdr("7-perm. Monotonicity permutation for fitted columns, regime C")
cols = ["condition_c", "fit_a", "fit_b", "promoter_like_signal", "condition_a", "condition_b"]
for c in cols:
    v = C[c].values.astype(float)
    obs = float(np.mean(np.diff(v) < 0))
    rng3 = np.random.default_rng(424242)
    M = 20000
    cnt = 0
    stats = np.empty(M)
    for i in range(M):
        p = rng3.permutation(v)
        stats[i] = np.mean(np.diff(p) < 0)
    P(f"{c}: frac_strict_desc={obs:.6f} sign_changes={int((np.diff(v) > 0).sum())} perm_mean={stats.mean():.4f} perm_sd={stats.std(ddof=1):.4f} p={(stats>=obs).sum()+1}/{M+1}")

# ---------------------------------------------------------------- item 7d --
hdr("7d. Dose scan numbers")
ds = dose.dose.values.astype(float); rs = dose.response_mean.values.astype(float)
P("dose:", ds, "response:", rs)
P("linear slope/R2:", np.polyfit(ds, rs, 1)[0], 1 - ((rs - np.polyval(np.polyfit(ds, rs, 1), ds)) ** 2).sum() / ((rs - rs.mean()) ** 2).sum(), "[report 0.0831 / 0.369]")
P("log-log slope/R2:", np.polyfit(np.log(ds), np.log(rs), 1)[0],
  1 - ((np.log(rs) - np.polyval(np.polyfit(np.log(ds), np.log(rs), 1), np.log(ds))) ** 2).sum() / ((np.log(rs) - np.log(rs).mean()) ** 2).sum(), "[report 0.6803 / 0.543]")
P("second difference:", (rs[2] - rs[1]) - (rs[1] - rs[0]), "[report -4.428]")
q = np.polyfit(ds, rs, 2); P("quadratic R2:", 1 - ((rs - np.polyval(q, ds)) ** 2).sum() / ((rs - rs.mean()) ** 2).sum(), "(report: exactly 1 by construction)")
for drop in range(3):
    keep = [i for i in range(3) if i != drop]
    s, i0 = np.polyfit(ds[keep], rs[keep], 1)
    pred = s * ds[drop] + i0
    P(f"  LOO drop {drop+1}: slope={s:.5f} resid={rs[drop]-pred:.4f} [report -0.0580/+0.0975/+0.3464 ; -4.077/+2.509/-6.523]")

# ---------------------------------------------------------------- item 7b --
hdr("7b. Perturbation endpoint bootstrap (own seed, 200000 draws)")
t = pert.time.values.astype(float)
D = 200000
deltas = {}
for j, cond in enumerate(["condition_a", "condition_b"]):
    m = pert[f"{cond}_mean"].values.astype(float); s = pert[f"{cond}_sd"].values.astype(float)
    rng4 = np.random.default_rng(555 + 1000 * j)  # fresh, independent stream per condition
    d0 = rng4.normal(m[0], s[0], D); d6 = rng4.normal(m[3], s[3], D)
    delta = d6 - d0
    deltas[cond] = delta
    P(f"{cond}: delta mean={delta.mean():.6f} CI95=[{np.percentile(delta,2.5):.6f},{np.percentile(delta,97.5):.6f}] P(>=0)={(delta>=0).mean():.4f}")
da, db = deltas["condition_a"], deltas["condition_b"]
diff = db - da
P("diff b-a: mean=%.6f CI95=[%.6f, %.6f] P(<=0)=%.5f [report -0.0512, [-0.0673,-0.0348], 1.000]" %
  (diff.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5), (diff <= 0).mean()))
P("ratio of total change b/a = %.4f [report 2.18]" % (abs(db.mean()) / abs(da.mean())))
P("ratio of loglinear rates = %.4f [report 2.35]" %
  (np.polyfit(t, np.log(pert.condition_b_mean), 1)[0] / np.polyfit(t, np.log(pert.condition_a_mean), 1)[0]))

# ---------------------------------------------------------------- item 7c --
hdr("7c. Smoother axis ambiguity (regime C)")
def cma(x, w):
    n = len(x); out = np.empty(n); h = w // 2
    for i in range(n):
        out[i] = x[max(0, i - h):min(n, i + h + 1)].mean()
    return out

for raw, fit in [("promoter_like_signal", "condition_c"), ("condition_a", "fit_a"), ("condition_b", "fit_b")]:
    r = C[raw].values.astype(float); f = C[fit].values.astype(float)
    rs = np.sort(r)[::-1]
    P(f"{raw} vs {fit}: corr(raw, own desc sort)={pearson(r, rs):.6f} [report 0.9996/0.9997/0.9978]")
    bestrow = min((np.sqrt(((cma(r, w) - f) ** 2).mean()), w) for w in range(3, 20, 2))
    bestdesc = min((np.sqrt(((cma(rs, w) - f) ** 2).mean()), w) for w in range(3, 20, 2))
    P(f"   best centred MA row-order: rmse={bestrow[0]:.4f} (w={bestrow[1]}) ; desc-sort: rmse={bestdesc[0]:.4f} (w={bestdesc[1]})")
    P(f"   fit-vs-raw RMSE={np.sqrt(((r-f)**2).mean()):.4f} [report 19.6/22.9/12.3]  r={pearson(r,f):.6f}")
P("double counting: n rows>=259 =", int((cur.index >= 259).sum()), "-> inflated", 2 * int((cur.index >= 259).sum()))
for raw, fit in [("promoter_like_signal", "condition_c"), ("condition_a", "fit_a"), ("condition_b", "fit_b")]:
    res = C[fit].values - C[raw].values
    P(f"   residual lag1 autocorr {raw}: {pearson(res[:-1], res[1:]):.4f} [report 0.48/0.47/0.74]")

# ---------------------------------------------------------------- summation -
hdr("Summary of direct numeric checks (report value vs recomputed)")
checks = [
    ("curves shape", "1367x7", "1367x7", True),
    ("pert shape", "4x7", "4x7", True),
    ("dose shape", "3x3", "3x3", True),
]
P("See MATCH/MISMATCH lines above; key headline comparisons printed inline.")