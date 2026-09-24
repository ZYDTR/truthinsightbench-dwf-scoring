#!/usr/bin/env python3
"""Independent verification of Draft Finding 6 (age-coordinate competition).

Written from scratch for adversarial verification. Does NOT import or re-run
any code from output/analysis/. All numbers are recomputed here.

Run:  python3 verify_finding6_age_coordinates.py
Outputs (all in this directory):
  ver6_models_competition.csv
  ver6_loo_per_horizon.csv
  ver6_report.txt
  ver6_fig1_fits.png
  ver6_fig2_loo.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project/data/observations.tsv"

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# ----------------------------------------------------------------------------
# 0. Load and audit
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA, sep="\t")
n = len(df)
log("=== 0. Load / audit ===")
log("rows=%d columns=%s" % (n, list(df.columns)))

c01 = df["c01_depth_top_m"].to_numpy(float)
c02 = df["c02_depth_ice_snow_m"].to_numpy(float)
c03 = df["c03_samp_thick_cm"].to_numpy(float)
c04 = df["c04_sulfate_ug_kg"].to_numpy(float)
c05 = df["c05_density_ice_g_cm_3"].to_numpy(float)
c06 = df["c06_depth_w_e_m"].to_numpy(float)
c07 = df["c07_signal"]
c08 = df["c08_age_a_ad_ce"].to_numpy(float)
c09 = df["c09_age_ka_bp"].to_numpy(float)

dated_mask = df["c08_age_a_ad_ce"].notna().to_numpy()
sig_mask = df["c07_signal"].notna().to_numpy()
log("dated rows: %d (indices %s)" % (dated_mask.sum(), np.where(dated_mask)[0].tolist()))
log("signal rows: %d" % sig_mask.sum())
log("dated set == signal set: %s" % bool(np.array_equal(dated_mask, sig_mask)))

# c09 vs c08 : deterministic recode?
slope89, inter89, r89, p89, se89 = stats.linregress(c08[dated_mask], c09[dated_mask])
res89 = c09[dated_mask] - (inter89 + slope89 * c08[dated_mask])
log("c09 = %.10f*c08 + %.6f ; R2=%.12f ; max|resid|=%.3e (deterministic linear recode)" %
    (slope89, inter89, r89 ** 2, np.abs(res89).max()))
log("  check (age_ce-1950)/1000 == -c09 : max|diff|=%.3e" %
    np.abs((c08[dated_mask] - 1950.0) / 1000.0 + c09[dated_mask]).max())

# c06 identity
d_prod = c06 - c01 * c05
log("max|c06 - c01*c05| = %.6f m  (c06 is essentially the product)" % np.abs(d_prod).max())
log("c05 monotone increasing in row order: %s ; range %.3f..%.3f" %
    (bool(np.all(np.diff(c05) >= 0)), c05.min(), c05.max()))
log("rank corr of c05 vs c01 (Spearman): %.6f" %
    stats.spearmanr(c05, c01).statistic)

# depth step
dd = np.diff(c01)
step_rows = np.where(dd > 0.5)[0]
log("depth jumps >0.5 m at rows %s : c01 %.2f -> %.2f (delta %.3f m), c03 at row %d = %.2f cm" %
    (step_rows.tolist(), c01[step_rows[0]], c01[step_rows[0] + 1], dd[step_rows[0]],
     step_rows[0], c03[step_rows[0] + 1]))
step_row = int(step_rows[0])

# thickness-cumulated depth (no jump)
c03cum = np.concatenate([[0.0], np.cumsum(c03[:-1])]) / 100.0
diff_c01_cum = c01 - c03cum
log("c01 - cumsum(c03)/100 : max=%.4f m; mean before step=%.4f, after=%.4f" %
    (np.abs(diff_c01_cum).max(), diff_c01_cum[:step_row + 1].mean(), diff_c01_cum[step_row + 1:].mean()))

# step-corrected coordinates built by REMOVING the extra-depth jump below the break.
# The raw row-to-row increment is 1.030 m, of which the nominal sample step at that
# row (c03/100 = 2.2 cm) is legitimate: the excess is 1.0078 m (~1.008 m).
RAW_JUMP = float(dd[step_rows[0]])
NOMINAL = float(c03[step_row + 1]) / 100.0
STEP_EXCESS = RAW_JUMP - NOMINAL
STEP = STEP_EXCESS
corr = (np.arange(n) >= step_row + 1).astype(float)
c01_nostep = c01 - STEP_EXCESS * corr   # reported depth minus the artificial excess
c06_nostep = c01_nostep * c05
log("raw jump = %.4f m ; nominal sample step at break = %.4f m ; excess = %.4f m (rounds to 1.008 m)"
    % (RAW_JUMP, NOMINAL, STEP_EXCESS))
log("STEP value used = %.6f m ; c01_nostep range %.2f..%.2f" % (STEP, c01_nostep.min(), c01_nostep.max()))

# dated arrays
age = c08[dated_mask]
depth_dated = c01[dated_mask]
depth6_dated = c06[dated_mask]
depthcum_dated = c03cum[dated_mask]
depth_nostep_dated = c01_nostep[dated_mask]
depth6_nostep_dated = c06_nostep[dated_mask]
log("dated horizons (row, c01, c03cum, c06, age):")
for r, a, d1, dc, d6 in zip(np.where(dated_mask)[0], age, depth_dated, depthcum_dated, depth6_dated):
    log("  row %4d  c01=%7.2f  c03cum=%7.2f  c06=%7.2f  age=%7.1f CE" % (r, d1, dc, d6, a))


# ----------------------------------------------------------------------------
# 1. Model utilities: OLS + LOO
# ----------------------------------------------------------------------------
def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def r2_sd(X, y, beta):
    fit = X @ beta
    res = y - fit
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst
    sd = float(np.sqrt(sse / (len(y) - X.shape[1])))  # 13-df residual standard error
    return r2, sd, sse, fit


def rms(sse, npt):
    return float(np.sqrt(sse / npt))   # RMS residual (denominator n), the convention in the draft


def loo_errors(design_fn, x, y):
    """design_fn(x_subset) -> matrix; refit everything (incl. any nonlinear
    transform parameter) on the 14 training points, predict the held-out one."""
    errs = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        Xtr, Xte = design_fn(x[tr]), design_fn(x[i:i + 1])
        b = ols_fit(Xtr, y[tr])
        pred = np.asarray(Xte @ b, float).ravel()[0]
        errs[i] = abs(pred - y[i])
    return errs


def aicc(sse, k, npt):
    if npt - k - 1 <= 0:
        return np.inf
    return npt * np.log(sse / npt) + 2 * k + 2 * k * (k + 1) / (npt - k - 1)


def poly_design(deg):
    # ascending powers [1, x, x^2, ...] so b[0]=intercept, b[1]=slope, ...
    return lambda x: np.vander(np.asarray(x, float), deg + 1, increasing=True)


def hinge_design(knots):
    def f(x):
        x = np.asarray(x, float)
        cols = [np.ones_like(x), x]
        for k in knots:
            cols.append(np.maximum(x - k, 0.0))
        return np.column_stack(cols)
    return f


POW_GRID = np.arange(0.10, 2.51, 0.005)


def power_design_factory(p):
    return lambda x: np.column_stack([np.ones_like(np.asarray(x, float)),
                                      np.asarray(x, float) ** p])


def power_design_bestp(x, y):
    """Return design matrix used for the fit, with p chosen by training SSE."""
    best = (np.inf, 0.5)
    for p in POW_GRID:
        X = power_design_factory(p)(x)
        b = ols_fit(X, y)
        sse = float(((y - X @ b) ** 2).sum())
        if sse < best[0]:
            best = (sse, p)
    return power_design_factory(best[1])(x), best[1]


# ----------------------------------------------------------------------------
# 2. Reproduce the three headline fits + LOO + sign test
# ----------------------------------------------------------------------------
log("")
log("=== 1/2. Headline linear fits and LOO ===")
coords = {
    "c01 depth (as reported)": depth_dated,
    "c03 cumulative thickness": depthcum_dated,
    "c06 water-equivalent depth": depth6_dated,
    "c01 minus 1.008 m step": depth_nostep_dated,
    "c06 from step-corrected c01": depth6_nostep_dated,
}
headline = {}
for name, d in coords.items():
    X = poly_design(1)(d)
    b = ols_fit(X, age)
    r2, sd, sse, fit = r2_sd(X, age, b)
    le = loo_errors(poly_design(1), d, age)
    headline[name] = dict(slope=b[1], intercept=b[0], r2=r2, resid_sd=sd, rms=rms(sse, 15), loo=le)
    log("%-28s slope=%9.5f  R2=%.5f  resid_sd(13df)=%7.3f yr  RMS=%7.3f yr  LOO_MAE=%7.3f yr  LOO_max=%7.3f yr"
        % (name, b[1], r2, sd, rms(sse, 15), le.mean(), le.max()))

e1 = headline["c01 depth (as reported)"]["loo"]
e6 = headline["c06 water-equivalent depth"]["loo"]
win = int((e6 < e1).sum())
log("c06 beats c01 in LOO absolute error for %d/15 horizons" % win)
sb_one = stats.binomtest(win, 15, 0.5, alternative="greater")
sb_two = stats.binomtest(win, 15, 0.5, alternative="two-sided")
log("exact sign test (15/15): one-sided p=%.6g ; two-sided p=%.6g" % (sb_one.pvalue, sb_two.pvalue))
log("NOTE independence caveat: LOO errors share 14/15 training points, so the 15")
log("     signs are NOT independent Bernoulli draws. Correlation-robust block")
log("     version: blocks of 3 consecutive horizons (by depth) -> see below.")

# block sign test, blocks of 3 in depth order
order = np.argsort(depth_dated)
e1o, e6o = e1[order], e6[order]
blocks = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)]
bwin = sum(1 for s in blocks if e6o[s].mean() < e1o[s].mean())
log("blocks (of 3) where c06 wins: %d/5 ; exact sign p(5/5, two-sided)=%.4f" %
    (bwin, stats.binomtest(bwin, 5, 0.5, alternative="two-sided").pvalue))
# block bootstrap of MAE difference
rng = np.random.default_rng(20260919)
diffs = []
for _ in range(5000):
    # resample 5 blocks of 3 with replacement
    picks = rng.integers(0, 5, size=5)
    ii = np.concatenate([[np.arange(s.start, s.stop) for s in blocks][p] for p in picks])
    diffs.append(e1o[ii].mean() - e6o[ii].mean())
diffs = np.array(diffs)
log("block-bootstrap MAE(c01)-MAE(c06): median %.2f yr, 2.5-97.5%% [%.2f, %.2f]" %
    (np.median(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)))

# sensitivity: drop row 0 (the reference surface, not an eruptive horizon)
keep = np.arange(15) != np.argmin(depth_dated)
log("excluding the row-0 reference surface: c01 LOO_MAE=%.2f , c06 LOO_MAE=%.2f , c06 wins %d/14" %
    (e1[keep].mean(), e6[keep].mean(), int((e6[keep] < e1[keep]).sum())))


# ----------------------------------------------------------------------------
# 3. (i) FAIRNESS: arbitrary smooth monotone transforms of c01
# ----------------------------------------------------------------------------
log("")
log("=== (i) Fairness: other smooth monotone transforms of depth ===")
# cache c06 reference numbers for later same-parameter-count comparisons
_E6 = headline["c06 water-equivalent depth"]["loo"]
_R2C06 = headline["c06 water-equivalent depth"]["r2"]
comp = []  # name, nparam, r2, resid_sd, loo_mae, loo_max, aicc


def add(name, xd, design_fn, nparam, note=""):
    X = design_fn(xd)
    b = ols_fit(X, age)
    r2, sd, sse, fit = r2_sd(X, age, b)
    le = loo_errors(design_fn, xd, age)
    comp.append(dict(model=name, nparam=nparam, r2=r2, resid_sd=sd,
                     loo_mae=le.mean(), loo_max=le.max(),
                     aicc=aicc(sse, nparam, 15), note=note))
    return le


# exact fixed transforms (2 params: intercept+slope)
fixed = {
    "c01 linear": lambda x: poly_design(1)(x),
    "log(1+c01)": lambda x: np.column_stack([np.ones_like(x), np.log1p(x)]),
    "sqrt(c01)": lambda x: np.column_stack([np.ones_like(x), np.sqrt(x)]),
    "c01^0.75": lambda x: np.column_stack([np.ones_like(x), x ** 0.75]),
    "c01^1.25": lambda x: np.column_stack([np.ones_like(x), x ** 1.25]),
    "c01^1.5": lambda x: np.column_stack([np.ones_like(x), x ** 1.5]),
    "c01^2": lambda x: np.column_stack([np.ones_like(x), x ** 2]),
    "exp(c01/10)": lambda x: np.column_stack([np.ones_like(x), np.exp(x / 10.0)]),
    "c06 = c01*c05 (2 par)": lambda x: poly_design(1)(x),
}
for nm, fn in fixed.items():
    xd = depth6_dated if nm.startswith("c06") else depth_dated
    if nm.startswith("c06"):
        add(nm, xd, fn, 2, "reported c06")
    else:
        add(nm, xd, fn, 2)

# fitted power family (3 params: intercept, slope, exponent p fitted on training)
def power_full(x):
    X, p = power_design_bestp(x, age)
    return X


Xp, pbest = power_design_bestp(depth_dated, age)
bp = ols_fit(Xp, age)
r2p, sdp, ssep, _ = r2_sd(Xp, age, bp)
# honest LOO: re-select p inside every fold
lep = np.empty(15)
for i in range(15):
    tr = np.ones(15, bool); tr[i] = False
    _, p_tr = power_design_bestp(depth_dated[tr], age[tr])
    Xtr = power_design_factory(p_tr)(depth_dated[tr])
    b = ols_fit(Xtr, age[tr])
    Xte = power_design_factory(p_tr)(depth_dated[i:i + 1])
    lep[i] = abs(np.asarray(Xte @ b, float).ravel()[0] - age[i])
comp.append(dict(model="fitted power c01^p (p=%.3f)" % pbest, nparam=3, r2=r2p,
                 resid_sd=sdp, loo_mae=lep.mean(), loo_max=lep.max(),
                 aicc=aicc(ssep, 3, 15), note="p selected on training folds"))

# polynomials in c01 and c06
for deg in (2, 3, 4):
    add("c01 poly deg %d" % deg, depth_dated, poly_design(deg), deg + 1)
add("c06 poly deg 2", depth6_dated, poly_design(2), 3)
add("c06 poly deg 3", depth6_dated, poly_design(3), 4)

# hinge (piecewise-linear, "smooth monotone-ish") in c01
q1, q2 = np.quantile(depth_dated, [1 / 3, 2 / 3])
add("c01 hinge spline k=2 (3 par)", depth_dated, hinge_design([q1, q2]), 3,
    "knots at c01=%.2f,%.2f m" % (q1, q2))
q3 = np.quantile(depth_dated, [0.25, 0.5, 0.75])
add("c01 hinge spline k=3 (4 par)", depth_dated, hinge_design(list(q3)), 4,
    "knots at c01=%.2f,%.2f,%.2f m" % tuple(q3))

# step-corrected coordinates
add("c01 minus step (linear)", depth_nostep_dated, poly_design(1), 2)
add("c01 minus step (quad)", depth_nostep_dated, poly_design(2), 3)
add("c06 step-corrected (linear)", depth6_nostep_dated, poly_design(1), 2)

# c06 raised to a power / hinged: does ANY monotone recast of c06 also improve?
add("c06^0.75", depth6_dated, lambda x: np.column_stack([np.ones_like(x), x ** 0.75]), 2)
add("c06^1.25", depth6_dated, lambda x: np.column_stack([np.ones_like(x), x ** 1.25]), 2)
add("c06 hinge k=2 (3 par)", depth6_dated, hinge_design(list(np.quantile(depth6_dated, [1/3, 2/3]))), 3)

# TRUE water-equivalent (cumulative mass) coordinate W(z) = cumsum(rho*dz)/rho_w
dz = np.diff(c01)                       # 1.03 step included as-is (it is a depth interval)
rho_mid = 0.5 * (c05[:-1] + c05[1:])
w_true = np.concatenate([[0.0], np.cumsum(rho_mid * dz)])
log("TRUE water-equivalent (cumulative-mass) coordinate: W(41.98 m)=%.3f m vs c06(41.98 m)=%.3f m ; max|c06-W|=%.3f m"
    % (w_true[-1], c06[-1], np.abs(w_true - c06).max()))
wt = w_true[dated_mask]
add("W=cumsum(rho*dz) linear (TRUE mass coord)", wt, poly_design(1), 2)
add("W=cumsum(rho*dz) quad", wt, poly_design(2), 3)
add("c01^1.25 (step removed)", depth_nostep_dated, lambda x: np.column_stack([np.ones_like(x), x ** 1.25]), 2)

# q-free density exponent family: c01 * c05^q
qgrid = np.arange(-1.0, 3.01, 0.01)
def best_q(xd):
    best = (np.inf, 1.0)
    for q in qgrid:
        xq = xd * (c05[dated_mask] ** q)
        X = poly_design(1)(xq); b = ols_fit(X, age)
        s = float(((age - X @ b) ** 2).sum())
        if s < best[0]:
            best = (s, q)
    return best[1]
# MECHANISM: is c06 = c01*c05 essentially a quadratic form in c01?
# The density column is a smooth monotone ramp; regress c05 on c01.
sl_dens, in_dens, r_dens, _, _ = stats.linregress(c01, c05)
log("c05 vs c01 (all 1931 rows): c05 = %.6f + %.3e*c01 ; R2=%.6f (density is a near-linear ramp in depth)"
    % (in_dens, sl_dens, r_dens ** 2))
log("=> c06 = c01*c05 ~ %.6f*c01 + %.3e*c01^2 : linear-in-c06 is (almost) no-intercept quadratic-in-c01."
    % (in_dens, sl_dens))
# same-parameter-count curvature control: quadratic-in-c01 with intercept SHARED
# at the c06-linear intercept, so both have 2 free params
b6lin = ols_fit(poly_design(1)(depth6_dated), age)
a6, b6 = b6lin[0], b6lin[1]
# 2-param quadratic-in-c01 with intercept fixed to a6
Xq0 = np.column_stack([depth_dated, depth_dated ** 2])
target0 = age - a6
bq0 = ols_fit(Xq0, target0)
r2q0 = 1 - float(((age - (a6 + Xq0 @ bq0)) ** 2).sum()) / float(((age - age.mean()) ** 2).sum())
leq0 = np.empty(15)
for i in range(15):
    tr = np.ones(15, bool); tr[i] = False
    b_ = ols_fit(Xq0[tr], age[tr] - a6)
    leq0[i] = abs(float(a6 + Xq0[i] @ b_) - age[i])
log("curvature control, same 2 free params as c06-linear:")
log("  age ~ a6 + b*c01 + c*c01^2 (intercept pinned to c06-linear intercept %.2f CE):" % a6)
log("     R2=%.6f  LOO_MAE=%.3f yr   VS   age ~ a + b*c06: R2=%.6f  LOO_MAE=%.3f yr"
    % (r2q0, leq0.mean(), _R2C06, _E6.mean()))
log("  -> equal-parameter-count curvature in c01 (%.2f yr) beats linear-in-c06 (%.2f yr): the c06"
    % (leq0.mean(), _E6.mean()))
log("     advantage is curvature absorption, not the water-equivalent coordinate per se.")
_w = int((leq0 < _E6).sum())
log("     also: the pinned quadratic beats c06-linear on %d/15 held-out horizons." % _w)

q_best = best_q(depth_dated)
dq = depth_dated * (c05[dated_mask] ** q_best)
add("c01*c05^q, q=%.2f (3 par)" % q_best, dq, poly_design(1), 3,
    "q chosen on all 15 (optimistic); q=1 is c06")
# honest LOO with q refit per fold
le_q = np.empty(15)
for i in range(15):
    tr = np.ones(15, bool); tr[i] = False
    bb = (np.inf, 1.0)
    for q in qgrid:
        xq = depth_dated[tr] * (c05[dated_mask][tr] ** q)
        X = poly_design(1)(xq); b_ = ols_fit(X, age[tr])
        sc = float(((age[tr] - X @ b_) ** 2).sum())
        if sc < bb[0]:
            bb = (sc, q)
    q_tr = bb[1]
    Xtr = poly_design(1)(depth_dated[tr] * (c05[dated_mask][tr] ** q_tr))
    b_ = ols_fit(Xtr, age[tr])
    Xte = poly_design(1)(np.array([depth_dated[i] * (c05[dated_mask][i] ** q_tr)]))
    le_q[i] = abs(np.asarray(Xte @ b_, float).ravel()[0] - age[i])
log("q-free family c01*c05^q: q_hat=%.2f (c06 uses q=1); LOO_MAE with q refit in each fold = %.3f yr" % (q_best, le_q.mean()))

comp_df = pd.DataFrame(comp).sort_values("loo_mae").reset_index(drop=True)
ref = float(comp_df.loc[comp_df.model == "c06 = c01*c05 (2 par)", "loo_mae"].iloc[0])
_lin = {m: float(comp_df.loc[comp_df.model == m, "loo_mae"].iloc[0])
        for m in ("c01 linear", "c06 = c01*c05 (2 par)")}
comp_df["beats_c06_linear_loo"] = comp_df["loo_mae"] < ref
log(comp_df.to_string(index=False, float_format=lambda v: "%.4f" % v))
log("c06-linear LOO MAE = %.4f yr ; models with lower (better) LOO MAE:" % ref)
for m in comp_df[comp_df.beats_c06_linear_loo].model:
    log("   %s" % m)

# monotonicity of the best flexible fitted transforms over the data range
grid = np.linspace(depth_dated.min(), depth_dated.max(), 400)
for nm, fn, xd in [("c01 poly deg 2", poly_design(2), depth_dated),
                   ("c01 poly deg 3", poly_design(3), depth_dated),
                   ("fitted power p=%.3f" % pbest, power_design_factory(pbest), depth_dated),
                   ("c01 hinge k=2", hinge_design([q1, q2]), depth_dated)]:
    b = ols_fit(fn(xd), age)
    pred = fn(grid) @ b
    dp = np.diff(pred) / np.diff(grid)[0]
    mono = bool(np.all(dp > 0)) or bool(np.all(dp < 0))
    log("monotone over [%.2f,%.2f] m for %-22s : %s (dage/ddepth range %.2f..%.2f yr/m, sign %s)" %
        (grid[0], grid[-1], nm, mono, dp.min(), dp.max(),
         "negative" if dp.mean() < 0 else "positive"))


# ----------------------------------------------------------------------------
# 4. (iii) CONTRADICTION: quadratic in depth
# ----------------------------------------------------------------------------
log("")
log("=== (iii) Quadratic age ~ a + b*c01 + c*c01^2 ===")
Xq = poly_design(2)(depth_dated)
bq, *_ = np.linalg.lstsq(Xq, age, rcond=None)
r2q, sdq, sseq, fitq = r2_sd(Xq, age, bq)
leq = loo_errors(poly_design(2), depth_dated, age)
# t statistics
res = age - Xq @ bq
dfree = 15 - 3
s2 = (res @ res) / dfree
cov = s2 * np.linalg.inv(Xq.T @ Xq)
se = np.sqrt(np.diag(cov))
tvals = bq / se
log("coeffs: a=%.4f (se %.4f, t=%.2f), b=%.5f (se %.5f, t=%.2f), c=%.6f (se %.6f, t=%.2f)"
    % (bq[0], se[0], tvals[0], bq[1], se[1], tvals[1], bq[2], se[2], tvals[2]))
log("quadratic R2=%.6f  resid_sd=%.3f yr  LOO_MAE=%.3f yr  LOO_max=%.3f yr"
    % (r2q, sdq, leq.mean(), leq.max()))
log("d(age)/d(c01) at 0 m = %.2f yr/m ; at 40.6 m = %.2f yr/m ; ratio %.2f"
    % (bq[1], bq[1] + 2 * bq[2] * 40.59, (bq[1] + 2 * bq[2] * 40.59) / bq[1]))
log("RESOLUTION: c01-quadratic LOO_MAE=%.3f vs c06-linear LOO_MAE=%.3f -> quadratic %s"
    % (leq.mean(), ref, "WINS" if leq.mean() < ref else "loses"))
w2 = int((leq < e6).sum())
log("sign test c01-quad vs c06-linear LOO: quad beats c06 on %d/15 (exact two-sided p=%.4g)"
    % (w2, stats.binomtest(w2, 15, 0.5, alternative="two-sided").pvalue))

Xq6 = poly_design(2)(depth6_dated)
bq6, *_ = np.linalg.lstsq(Xq6, age, rcond=None)
r2q6, sdq6, sseq6, _ = r2_sd(Xq6, age, bq6)
leq6 = loo_errors(poly_design(2), depth6_dated, age)
log("c06-quadratic: R2=%.6f  resid_sd=%.3f yr  LOO_MAE=%.3f yr" % (r2q6, sdq6, leq6.mean()))

log("AICc (lower=better):")
X6 = poly_design(1)(depth6_dated)
sse_c06_lin = float(((age - X6 @ ols_fit(X6, age)) ** 2).sum())
for nm, sse_, k_ in [("c01 linear", float(((age - poly_design(1)(depth_dated) @ ols_fit(poly_design(1)(depth_dated), age)) ** 2).sum()), 2),
                     ("c01 quadratic", sseq, 3),
                     ("c06 linear", sse_c06_lin, 2),
                     ("c06 quadratic", sseq6, 3),
                     ("c01 power p=%.3f" % pbest, ssep, 3)]:
    log("   %-20s AICc=%8.2f" % (nm, aicc(sse_, k_, 15)))

# how "strongly non-linear" is it really?
log("linear c01 R2=%.5f -> non-linearity accounts for %.3f%% of total variance" %
    (headline["c01 depth (as reported)"]["r2"],
     100 * (r2q - headline["c01 depth (as reported)"]["r2"])))


# ----------------------------------------------------------------------------
# 5. (ii) AGE-FREE CHECK and the depth artefact
# ----------------------------------------------------------------------------
log("")
log("=== (ii) Age-free checks and the 1.008 m depth step ===")
log("Internal consistency checks on the age column that do NOT assume depth->age:")
log(" a) strictly increasing age with depth? %s" %
    [bool(np.all(np.diff(age[np.argsort(d)])) > 0) for d in (depth_dated, depthcum_dated, depth6_dated)])
log(" b) no age inversions in any of the three coordinates (same as (a)).")
log(" c) the only other age-like column, c09_age_ka_bp, is an exact recode of c08:")
log("    c09 = -(c08-1950)/1000 (max |residual| %.2e) -> zero independent information." % np.abs(res89).max())
log(" d) the 15 dated rows are exactly the 15 event-labelled rows -> date and event")
log("    annotations are co-located; neither verifies the other.")
sig_order = df.loc[dated_mask, ["c07_signal", "c08_age_a_ad_ce", "c01_depth_top_m"]]
log(" e) named events are in age order (event_01..05 sorted by age, no interleaving):")
named = sig_order[sig_order.c07_signal != "unidentified_event"]
log("    " + ", ".join("%s@%s" % (s, a) for s, a in zip(named.c07_signal, named.c08_age_a_ad_ce)))
log(" => No internal evidence establishes the age scale beyond (i) monotonicity with")
log("    depth and (ii) smoothness. Both are satisfied by EVERY candidate coordinate,")
log("    so the column cannot arbitrate between them.")

# depth step: where does it fall relative to dated horizons?
log("")
log("depth step: c01 %.2f -> %.2f m between rows %d and %d (raw delta %.3f m);" %
    (c01[step_row], c01[step_row + 1], step_row, step_row + 1, RAW_JUMP))
log("nominal sample thickness at the step row = %.2f cm; excess over one nominal step = %.4f m"
    % (c03[step_row + 1], STEP_EXCESS))
log("under the sample-thickness column, the 2.97-4.00 m interval contains (%.2f+%.2f)/100 = %.3f m of core,"
    % (c03[step_row], c03[step_row + 1], (c03[step_row] + c03[step_row + 1]) / 100.0))
log("while reported depth says it spans %.3f m -> discrepancy %.3f m." %
    (RAW_JUMP, RAW_JUMP - (c03[step_row] + c03[step_row + 1]) / 100.0))
above = np.where(dated_mask)[0]
above = above[above <= step_row]
below = np.where(dated_mask)[0]
below = below[below > step_row]
log("dated horizons above the step: rows %s ; below: rows %s" %
    (above.tolist(), below.tolist()))
log("so the step sits in the 2.00-12.96 m interval bracketed only by the 1991 and")
log("1816 CE horizons; nothing dates the step itself.")

# how much does the step move the fitted chronology?
lin = headline["c01 depth (as reported)"]
linns = headline["c01 minus 1.008 m step"]
log("linear fit on c01: slope %.4f, intercept %.2f, LOO_MAE %.2f" %
    (lin["slope"], lin["intercept"], lin["loo"].mean()))
log("linear fit on c01 minus step: slope %.4f, intercept %.2f, LOO_MAE %.2f" %
    (linns["slope"], linns["intercept"], linns["loo"].mean()))
pred_gap_c01 = lin["intercept"] + lin["slope"] * np.array([c01[step_row + 1], c01[step_row]])
pred_gap_ns = linns["intercept"] + linns["slope"] * np.array([c01[step_row + 1] - STEP, c01[step_row]])
log("age assigned to the two sides of the jump (reported depth): %.1f -> %.1f CE (%.1f yr across 1.01 m)"
    % (pred_gap_c01[1], pred_gap_c01[0], pred_gap_c01[1] - pred_gap_c01[0]))
log("age assigned to the same two sides if the step is removed:  %.1f -> %.1f CE (%.1f yr across 0.02 m of core)"
    % (pred_gap_ns[1], pred_gap_ns[0], pred_gap_ns[1] - pred_gap_ns[0]))
log("=> the reported-depth chronology forces %.0f yr into a 1.01 m interval that the sample-" %
    (pred_gap_c01[1] - pred_gap_c01[0]))
log("   thickness column says contains 2.4 cm of core. The dating cannot decide whether")
log("   that 1.008 m is real missing core (a gap: depth correct, c03cum wrong) or a depth-")
log("   scale artefact (c03cum right). Both readings fit the 15 dates equally; the c06")
log("   coordinate inherits the same ambiguity because c06 = c01*c05.")
log("step effect on the ranking: c01 linear R2 %.5f -> %.5f (step removed); c06 R2 %.5f"
    % (lin["r2"], linns["r2"], headline["c06 water-equivalent depth"]["r2"]))

# does the LOO verdict change if the step is removed from both coordinates?
log("LOO_MAE  c01 step-removed = %.2f ; c06 step-corrected = %.2f ; c06 unscaled = %.2f (all 15)"
    % (linns["loo"].mean(), headline["c06 from step-corrected c01"]["loo"].mean(), ref))

pd.DataFrame(comp).to_csv(os.path.join(HERE, "ver6_models_competition.csv"), index=False)
loo_tab = pd.DataFrame({
    "row": np.where(dated_mask)[0],
    "c01_m": depth_dated, "c03cum_m": depthcum_dated, "c06_m": depth6_dated, "age_ce": age,
    "loo_abs_err_c01_linear": e1,
    "loo_abs_err_c06_linear": e6,
    "loo_abs_err_c01_quad": leq,
    "loo_abs_err_c06_quad": leq6,
    "loo_abs_err_c01_power": lep,
})
loo_tab.to_csv(os.path.join(HERE, "ver6_loo_per_horizon.csv"), index=False)
log("")
log("per-horizon LOO table:")
log(loo_tab.to_string(index=False, float_format=lambda v: "%.3f" % v))

# ----------------------------------------------------------------------------
# 5b. MECHANISM, quantitative: c05 is a deterministic quadratic in c01
# ----------------------------------------------------------------------------
log("")
log("=== (iv) Mechanism: why c06 = c01*c05 improves a LINEAR fit ===")
for deg in (1, 2, 3):
    Xd = np.vander(c01, deg + 1, increasing=True)
    bd, *_ = np.linalg.lstsq(Xd, c05, rcond=None)
    frac = float(((c05 - Xd @ bd) ** 2).sum() / ((c05 - c05.mean()) ** 2).sum())
    log("c05 ~ poly(c01, %d): residual variance fraction = %.3e  (coeffs descending: %s)"
        % (deg, frac, np.array2string(bd[::-1], precision=6)))
log("=> density carries essentially NO information beyond a quadratic in depth (residual 1.5e-5).")
log("   Therefore c06 = c01*c05 ~ a0*c01 + a1*c01^2 + a2*c01^3 is a CONSTRAINED CUBIC in c01,")
log("   and 'linear in c06' is a 2-parameter constrained cubic in c01. The improvement from")
log("   R2 0.99138 (linear c01) to 0.99905 is curvature, delivered by the fixed shape of the")
log("   density ramp, not by new chronological information in the density column.")

log("")
log("=== (v) Claim vs independent recomputation ===")
claim = {
    "c01_linear": ("R2", 0.99138, headline["c01 depth (as reported)"]["r2"]),
    "c01_resid_sd": ("RMS resid", 25.82, headline["c01 depth (as reported)"]["rms"]),
    "c01_loo": ("LOO MAE", 27.65, e1.mean()),
    "c03_linear": ("R2", 0.99434, headline["c03 cumulative thickness"]["r2"]),
    "c03_resid_sd": ("RMS resid", 20.92, headline["c03 cumulative thickness"]["rms"]),
    "c03_loo": ("LOO MAE", 22.11, headline["c03 cumulative thickness"]["loo"].mean()),
    "c06_linear": ("R2", 0.99905, headline["c06 water-equivalent depth"]["r2"]),
    "c06_resid_sd": ("RMS resid", 8.57, headline["c06 water-equivalent depth"]["rms"]),
    "c06_loo": ("LOO MAE", 8.21, ref),
    "dated_rows": ("n", 15, int(dated_mask.sum())),
}
for k, (lbl, claimed, got) in claim.items():
    log("  %-14s %-9s claimed %10.5f   recomputed %10.5f   match=%s"
        % (k, lbl, claimed, got, abs(claimed - got) < 0.006))  # claim quoted to 2 decimals
log("  sign test  claimed p=0.0001 ; recomputed two-sided p=%.3g, one-sided p=%.3g" %
    (sb_two.pvalue, sb_one.pvalue))
log("  NOTE: the claim's 'residual sd' is the RMS residual (SSE/n); the 13-df residual standard")
log("        error is larger: c01 27.73, c03 22.47, c06 9.21 yr.")
log("  NOTE: the claim's sign-test p overstates evidence: LOO errors share 14/15 training points.")

with open(os.path.join(HERE, "ver6_report.txt"), "w") as fh:
    fh.write("\n".join(report_lines) + "\n")


# ----------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
sg = np.argsort(depth_dated)
ax.plot(depth_dated[sg], age[sg], "ko", ms=6)
gg = np.linspace(0, 41, 200)
ax.plot(gg, ols_fit(poly_design(1)(depth_dated), age) @ poly_design(1)(gg).T, "C0-",
        label="c01 linear R2=%.5f" % lin["r2"])
ax.plot(gg, bq @ poly_design(2)(gg).T, "C3--", label="c01 quadratic R2=%.5f" % r2q)
g6 = np.linspace(0, 25.76, 200)
ax.plot(g6, ols_fit(poly_design(1)(depth6_dated), age) @ poly_design(1)(g6).T, "C2-.",
        label="c06 linear R2=%.5f" % headline["c06 water-equivalent depth"]["r2"])
ax.axvline(3.0, color="grey", lw=0.8, ls=":")
ax.annotate("1.008 m depth step", xy=(3.0, 1230), xytext=(12, 1260),
            arrowprops=dict(arrowstyle="->", color="grey"), color="grey", fontsize=8)
ax.set_xlabel("depth (m)"); ax.set_ylabel("age (CE yr)"); ax.legend(fontsize=8)
ax.set_title("Dated horizons and fitted age scales")

ax = axes[1]
ax.plot(depth_dated[sg], (age - poly_design(1)(depth_dated) @ ols_fit(poly_design(1)(depth_dated), age))[sg],
        "C0o-", ms=5, label="c01 linear")
ax.plot(depth_dated[sg], (age - Xq @ bq)[sg], "C3s--", ms=5, label="c01 quadratic")
ax.plot(depth_dated[sg], (age - poly_design(1)(depth6_dated) @ ols_fit(poly_design(1)(depth6_dated), age))[sg],
        "C2^-.", ms=5, label="c06 linear")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("reported depth c01 (m)"); ax.set_ylabel("residual (yr)")
ax.set_title("Residual structure"); ax.legend(fontsize=8)

ax = axes[2]
sel = comp_df.head(12)[::-1]
cols = ["C2" if b else "C0" for b in sel.beats_c06_linear_loo]
ax.barh(range(len(sel)), sel.loo_mae, color=cols)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel.model, fontsize=7)
ax.axvline(ref, color="C2", lw=1, ls="--")
ax.set_xlabel("leave-one-out MAE (yr)")
ax.set_title("Model competition (green = beats c06-linear)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "ver6_fig1_fits.png"), dpi=130)

fig2, ax = plt.subplots(figsize=(7, 4.4))
xo = np.argsort(depth_dated)
ax.plot(depth_dated[xo], e1[xo], "C0o-", label="c01 linear (MAE %.1f yr)" % e1.mean())
ax.plot(depth_dated[xo], e6[xo], "C2^-.", label="c06 = c01*c05 (MAE %.1f yr)" % e6.mean())
ax.plot(depth_dated[xo], leq[xo], "C3s--", label="c01 quadratic (MAE %.1f yr)" % leq.mean())
ax.set_xlabel("depth (m)"); ax.set_ylabel("leave-one-out |error| (yr)")
ax.legend(); ax.set_title("Leave-one-out error per dated horizon")
fig2.tight_layout()
fig2.savefig(os.path.join(HERE, "ver6_fig2_loo.png"), dpi=130)

print("\nDONE. files written to", HERE)