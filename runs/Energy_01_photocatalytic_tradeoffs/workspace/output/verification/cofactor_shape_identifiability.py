#!/usr/bin/env python3
"""
Independent analysis of cofactor_loading_response.tsv (4 loading settings).

Question: what response shape, if any, can 4 points identify? And is the peak
location (0.2% vs 1.0%) identifiable? And does a log-scale x-axis rescue
monotonicity?

Implemented from scratch against the raw TSV. No other output file was read.
"""
import json
import numpy as np
from scipy.optimize import least_squares

DATA = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/cofactor_loading_response.tsv"
OUT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/cofactor_shape_identifiability.json"

# ---------------------------------------------------------------- data
x = np.array([0.2, 1.0, 3.0, 8.0])
y = np.array([1081.0, 1217.0, 470.0, 268.0])
n = len(y)
tss = float(np.sum((y - y.mean()) ** 2))

# ---------------------------------------------------------------- helpers
def stats(rss, k, label, params):
    rss = float(max(rss, 0.0))
    rss_eff = max(rss, 1e-10)
    aic = n * np.log(rss_eff / n) + 2 * k
    if n - k - 1 > 0:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = None  # undefined (division by zero) when k >= n-1
    bic = n * np.log(rss_eff / n) + k * np.log(n)
    return dict(
        model=label, k=k, rss=rss, rmse=float(np.sqrt(rss / n)),
        r2=float(1 - rss / tss), aic=float(aic),
        aicc=(None if aicc is None else float(aicc)),
        aicc_note=("undefined: n-k-1=0 with n=4,k=3" if aicc is None else "ok"),
        bic=float(bic), params={kk: float(vv) for kk, vv in params.items()},
        exact_interpolation=bool(rss < 1e-6),
    )


def refine(fun, p0, lo, hi, n_starts=40, seed=0):
    """Multi-start bounded least squares on 4 points. Returns (params, rss)."""
    rng = np.random.default_rng(seed)
    cand = [np.asarray(p0, float)]
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    for _ in range(n_starts - 1):
        cand.append(lo + rng.random(len(lo)) * (hi - lo))
    best = None
    for c in cand:
        c = np.clip(c, lo, hi)
        try:
            s = least_squares(lambda p: fun(p) - y, c, bounds=(lo, hi),
                              method="trf", max_nfev=4000)
        except Exception:
            continue
        r = float(np.sum(s.fun ** 2))
        if best is None or r < best[1]:
            best = (s.x.copy(), r)
    return best


def fit_model(name, fun, p0, lo, hi, k, n_starts=40, seed=0, pnames=None):
    p, rss = refine(fun, p0, lo, hi, n_starts, seed)
    if pnames is None:
        pnames = [f"p{i}" for i in range(len(p))]
    return stats(rss, k, name, dict(zip(pnames, p)))

results = []

# 1. constant (k=1)
results.append(fit_model("constant", lambda p: np.full_like(x, p[0]),
                         [800.0], [0.0], [2000.0], 1, pnames=["c"]))

# 2. linear (k=2)
results.append(fit_model("linear a+b*x", lambda p: p[0] + p[1] * x,
                         [1200.0, -100.0], [-5000.0, -2000.0], [5000.0, 2000.0],
                         2, pnames=["a", "b"]))

# 3. monotone power law r = a*x^b (k=2)
results.append(fit_model("monotone power a*x^b", lambda p: p[0] * x ** p[1],
                         [1200.0, -0.3], [1e-6, -5.0], [1e6, 5.0], 2,
                         pnames=["a", "b"]))

# 4. monotone exponential decay r = a*exp(-b x) (k=2)
results.append(fit_model("monotone exp decay a*exp(-b*x)",
                         lambda p: p[0] * np.exp(-p[1] * x),
                         [1200.0, 0.2], [1e-6, 0.0], [1e6, 10.0], 2,
                         pnames=["a", "b"]))

# 4b. exponential decay + offset (k=3)
results.append(fit_model("exp decay + offset a*exp(-b*x)+c",
                         lambda p: p[0] * np.exp(-p[1] * x) + p[2],
                         [1000.0, 0.2, 0.0], [1e-6, 0.0, -2000.0], [1e6, 20.0, 2000.0],
                         3, pnames=["a", "b", "c"]))

# 5. saturating rise without inhibition, MM r = V*x/(K+x) (k=2)
results.append(fit_model("saturating rise V*x/(K+x)",
                         lambda p: p[0] * x / (p[1] + x),
                         [1500.0, 2.0], [1e-6, 1e-6], [1e6, 1e6], 2,
                         pnames=["V", "K"]))

# 6. substrate inhibition V*x/(K + x + x^2/Ki) (k=3)
results.append(fit_model("substrate inhibition V*x/(K+x+x^2/Ki)",
                         lambda p: p[0] * x / (p[1] + x + x ** 2 / p[2]),
                         [1500.0, 1.0, 5.0], [1e-6, 1e-6, 1e-6], [1e6, 1e6, 1e6],
                         3, n_starts=120, seed=1, pnames=["V", "K", "Ki"]))

# 7. plateau-then-linear-decline kink: r = a for x<=Lb, a+b*(x-Lb) for x>Lb (k=3)
def kink(p):
    a, b, lb = p
    return np.where(x <= lb, a, a + b * (x - lb))
results.append(fit_model("plateau then linear decline (kink)",
                         kink, [1200.0, -100.0, 1.0],
                         [0.0, -1e5, 0.2], [5000.0, 0.0, 8.0], 3,
                         n_starts=120, seed=2, pnames=["plateau_a", "slope_b<0", "break_Lb"]))

# 8. two-regime: flat up to break then exponential decline (k=3)
def flat_then_exp(p):
    a, b, lb = p
    return np.where(x <= lb, a, a * np.exp(-b * (x - lb)))
results.append(fit_model("flat then exponential decline (two-regime)",
                         flat_then_exp, [1200.0, 0.3, 1.0],
                         [0.0, 0.0, 0.2], [5000.0, 50.0, 8.0], 3,
                         n_starts=120, seed=3, pnames=["level_a", "decay_b", "break_Lb"]))

# 9. quadratic in x (k=3)
results.append(fit_model("quadratic c0+c1*x+c2*x^2",
                         lambda p: p[0] + p[1] * x + p[2] * x ** 2,
                         [1200.0, 100.0, -20.0], [-5000.0, -5000.0, -5000.0],
                         [5000.0, 5000.0, 5000.0], 3,
                         pnames=["c0", "c1", "c2"]))

# 10. log-quadratic peak model r = a - b*(ln x - ln P)^2, peak at P (k=3)
def logquad(p):
    a, b, P = p
    return a - b * (np.log(x) - np.log(P)) ** 2
res_lq = fit_model("log-quadratic peak a-b*(lnx-lnP)^2",
                   logquad, [1200.0, 200.0, 1.0],
                   [-5000.0, 1e-9, 0.02], [5000.0, 1e6, 200.0], 3,
                   n_starts=200, seed=4, pnames=["a", "b", "peak_P"])
results.append(res_lq)

# ---------------------------------------------------------------- peak-location tests
peak = {}

# (a) constrained log-quadratic: peak forced to 0.2 or to 1.0 (k=2 each)
for P in (0.2, 1.0):
    r = fit_model(f"log-quadratic peak FIXED at {P}%",
                  lambda p, P=P: p[0] - p[1] * (np.log(x) - np.log(P)) ** 2,
                  [1200.0, 200.0], [-5000.0, 1e-9], [5000.0, 1e6], 2,
                  n_starts=60, seed=5, pnames=["a", "b"])
    peak[f"logquad_P={P}"] = r

# free-fit peak location and profile
peak["logquad_free_peak_P"] = res_lq["params"]["peak_P"]
peak["logquad_free_RSS"] = res_lq["rss"]

grid = np.exp(np.linspace(np.log(0.05), np.log(30.0), 400))
prof = []
for P in grid:
    p, rss = refine(lambda p, P=P: p[0] - p[1] * (np.log(x) - np.log(P)) ** 2,
                    [1200.0, 200.0], [-5000.0, 1e-9], [5000.0, 1e6], 12, seed=6)
    prof.append((float(P), float(rss)))
prof = np.array(prof)
peak["logquad_profile_RSS_at_0.2"] = float(prof[np.argmin(abs(prof[:, 0] - 0.2)), 1])
peak["logquad_profile_RSS_at_1.0"] = float(prof[np.argmin(abs(prof[:, 0] - 1.0)), 1])
peak["logquad_profile_RSS_min"] = float(prof[:, 1].min())
peak["logquad_profile_P_at_RSS_min"] = float(prof[prof[:, 1].argmin(), 0])
# descriptive flat region: P with RSS <= 1.05*RSS_min (no sigma available)
flat = prof[prof[:, 1] <= 1.05 * prof[:, 1].min(), 0]
peak["logquad_profile_P_range_RSS<=1.05xmin"] = [float(flat.min()), float(flat.max())]

# (b) exact interpolant test: quartic polynomial constrained to pass through all
#     4 points AND have zero slope at a prescribed P -> 5 linear equations, 5 unknowns.
#     If a solution with a local max at P exists for both P=0.2 and P=1.0 then an
#     interpolating model gives RSS=0 for both -> peak location is vacuous for any
#     model flexible enough to interpolate.
def quartic_through_points_stationary_at(P):
    V = np.vander(x, 5, increasing=True)                    # 4 rows, 5 cols
    A = np.vstack([V, np.array([0, 1, 2 * P, 3 * P ** 2, 4 * P ** 3])])
    rhs = np.concatenate([y, [0.0]])
    try:
        c = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    pfun = lambda t: c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4
    d2 = 2 * c[2] + 6 * c[3] * P + 12 * c[4] * P ** 2
    rss = float(np.sum((pfun(x) - y) ** 2))
    return dict(P=float(P), rss=rss, second_derivative_at_P=float(d2),
                is_local_max=bool(d2 < 0),
                values_at_x=[float(v) for v in pfun(x)])

peak["quartic_interpolant_P=0.2"] = quartic_through_points_stationary_at(0.2)
peak["quartic_interpolant_P=1.0"] = quartic_through_points_stationary_at(1.0)

# ---------------------------------------------------------------- monotone / axis tests
axis = {}

# isotonic (best monotone non-increasing and non-decreasing) RSS via PAVA
def pava(yv, decreasing=True):
    yy = yv.copy()
    w = np.ones_like(yy)
    blocks = [[i] for i in range(len(yy))]
    while True:
        merged = False
        i = 0
        while i < len(blocks) - 1:
            m1 = np.sum(w[blocks[i]] * yy[blocks[i]]) / np.sum(w[blocks[i]])
            m2 = np.sum(w[blocks[i + 1]] * yy[blocks[i + 1]]) / np.sum(w[blocks[i + 1]])
            bad = (m1 < m2) if decreasing else (m1 > m2)
            if bad:
                blocks[i] = blocks[i] + blocks[i + 1]
                del blocks[i + 1]
                merged = True
            else:
                i += 1
        if not merged:
            break
    fit = np.zeros_like(yy)
    for b in blocks:
        fit[b] = np.sum(w[b] * yy[b]) / np.sum(w[b])
    return fit

fit_dec = pava(y, decreasing=True)
fit_inc = pava(y, decreasing=False)
rss_dec = float(np.sum((y - fit_dec) ** 2))
rss_inc = float(np.sum((y - fit_inc) ** 2))
axis["isotonic_decreasing_RSS"] = rss_dec
axis["isotonic_decreasing_fit"] = [float(v) for v in fit_dec]
axis["isotonic_increasing_RSS"] = rss_inc
axis["isotonic_decreasing_R2"] = float(1 - rss_dec / tss)
axis["isotonic_increasing_R2"] = float(1 - rss_inc / tss)

# same isotonic fit but with x replaced by log10(x): ordering of x is unchanged so
# the fitted values (and RSS) are IDENTICAL by construction -> invariant.
order = np.argsort(np.log10(x))
fit_dec_log = np.empty_like(fit_dec)
fit_dec_log[order] = pava(y[order], decreasing=True)
axis["isotonic_decreasing_RSS_log10_axis"] = float(np.sum((y - fit_dec_log) ** 2))
axis["isotonic_invariants_identical"] = bool(
    abs(axis["isotonic_decreasing_RSS_log10_axis"] - rss_dec) < 1e-9)

# permutation test: Spearman rho for the observed ordering, exact two-sided p,
# and the same test on the log10 axis (identical by monotone invariance).
def spearman(a, b):
    ra = np.argsort(np.argsort(a)) + 1.0
    rb = np.argsort(np.argsort(b)) + 1.0
    return float(np.corrcoef(ra, rb)[0, 1])

import itertools
perm_rhos = np.array([spearman(np.arange(n), np.array(p)) for p in itertools.permutations(range(n))])
obs_rho = spearman(np.arange(n), y)
p_two = float(np.mean(np.abs(perm_rhos) >= abs(obs_rho) - 1e-12))
p_one = float(np.mean(perm_rhos >= obs_rho - 1e-12))
axis["spearman_rho_linear_axis"] = obs_rho
axis["spearman_rho_log10_axis"] = spearman(np.log10(x), y)
axis["perm_exact_two_sided_p"] = p_two
axis["perm_exact_one_sided_p_decreasing"] = p_one
axis["n_permutations"] = int(len(perm_rhos))
axis["spearman_rho_ranked_by_axis_value"] = float(
    spearman(np.log10(x), y))

# ---------------------------------------------------------------- AICc set
valid = [r for r in results if r["aicc"] is not None]
best = min(valid, key=lambda r: r["aicc"])
for r in results:
    r["delta_aicc"] = (None if r["aicc"] is None else float(r["aicc"] - best["aicc"]))
within2 = [r["model"] for r in valid if r["aicc"] - best["aicc"] <= 2.0]
within2_plus_undefined = within2 + [r["model"] for r in results if r["aicc"] is None]
# also an AIC-only (uncorrected) within-2 set so 3-parameter models can be ranked
best_aic = min(results, key=lambda r: r["aic"])
within2_aic = [r["model"] for r in results if r["aic"] - best_aic["aic"] <= 2.0]
best_bic = min(results, key=lambda r: r["bic"])
within2_bic = [r["model"] for r in results if r["bic"] - best_bic["bic"] <= 2.0]

summary = dict(
    data=dict(loading_percent=x.tolist(), rate_umol_per_h=y.tolist(), n=n,
              tss=tss, note="one complete curve = one independent unit; the 4 points "
                            "are 4 loading conditions, not replicates. No error bars "
                            "or replicate variance exist in the TSV."),
    aicc_defined_note=("With n=4, AICc uses the correction 2k(k+1)/(n-k-1); for any "
                       "3-parameter model n-k-1=0 so AICc is formally +infinity "
                       "(undefined). Only the 1- and 2-parameter models have a usable AICc."),
    models=results,
    aicc_within_2_of_best=within2,
    aicc_undefined_models= [r["model"] for r in results if r["aicc"] is None],
    within2_if_undefined_treated_as_competitive=within2_plus_undefined,
    aic_uncorrected_within_2=within2_aic,
    bic_within_2=within2_bic,
    peak_location=peak,
    axis_sensitivity=axis,
)

with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------- console report
print("data:", list(zip(x, y)), " TSS=", tss)
print("\n%-45s %2s %10s %10s %8s %8s %10s %10s" %
      ("model", "k", "RSS", "R2", "AIC", "AICc", "dBIC", "BIC"))
for r in sorted(results, key=lambda r: r["bic"]):
    print("%-45s %2d %10.1f %10.4f %8.2f %8s %10.2f %10.2f" % (
        r["model"], r["k"], r["rss"], r["r2"], r["aic"],
        ("undef" if r["aicc"] is None else "%.2f" % r["aicc"]),
        r["bic"] - best_bic["bic"], r["bic"]))
print("\nAICc within 2 of best (defined-AICc models only):", within2)
print("models with undefined AICc:", [r["model"] for r in results if r["aicc"] is None])
print("AIC (uncorrected) within 2:", within2_aic)
print("BIC within 2:", within2_bic)
print("\nfree log-quadratic peak:", res_lq["params"])
print("RSS logquad P=0.2:", peak["logquad_P=0.2"]["rss"],
      " P=1.0:", peak["logquad_P=1.0"]["rss"],
      " free P:", peak["logquad_free_RSS"])
print("profile P at min RSS:", peak["logquad_profile_P_at_RSS_min"],
      " flat range (<1.05xmin):", peak["logquad_profile_P_range_RSS<=1.05xmin"])
print("quartic interpolant P=0.2:", peak["quartic_interpolant_P=0.2"])
print("quartic interpolant P=1.0:", peak["quartic_interpolant_P=1.0"])
print("\nisotonic decreasing RSS:", rss_dec, " R2:", axis["isotonic_decreasing_R2"])
print("isotonic increasing RSS:", rss_inc, " R2:", axis["isotonic_increasing_R2"])
print("isotonic RSS identical under log10 axis:", axis["isotonic_invariants_identical"])
print("Spearman rho (linear axis):", obs_rho, " (log10 axis):",
      axis["spearman_rho_log10_axis"], " exact two-sided p:", p_two)
print("\nwrote", OUT)