#!/usr/bin/env python3
"""
Independent verification agent: tests H1 (sharp cut-off in absorption fall-off),
H2 (optical-density -> absorptance transform) and H3 (extra red spectral component)
against the action-vs-absorption mismatch in optical_action_spectra.tsv.

All numbers are computed from the raw frozen TSV only.
Output: JSON in the agent's own output directory.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares, minimize_scalar, nnls

DATA = Path("/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/optical_action_spectra.tsv")
OUTDIR = Path("/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, sep="\t")
ab = df[df.measurement_role == "absorption"].sort_values("wavelength_nm").reset_index(drop=True)
ac = df[df.measurement_role == "action_response"].sort_values("wavelength_nm").reset_index(drop=True)
lam_a = ab.wavelength_nm.to_numpy(float)
A_a = ab.response_au_or_percent.to_numpy(float)
lam_y = ac.wavelength_nm.to_numpy(float)
y = ac.response_au_or_percent.to_numpy(float)

res = {"input": {"n_absorption_points": int(len(ab)), "n_action_points": int(len(ac)),
                 "action_wavelengths": lam_y.tolist(), "action_values": y.tolist(),
                 "absorption_range_nm": [float(lam_a.min()), float(lam_a.max())]}}

# ---------------------------------------------------------------- helpers
def aicc(rss, n, k):
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def linls(cols, target):
    coef, *_ = np.linalg.lstsq(cols, target, rcond=None)
    pred = cols @ coef
    rss = float(np.sum((target - pred) ** 2))
    return coef, rss, pred, target - pred

def runstats(r):
    r = np.asarray(r)
    n = len(r)
    npos = int(np.sum(r > 0)); nneg = int(np.sum(r < 0))
    runs = 1 + int(np.sum(np.sign(r[1:]) != np.sign(r[:-1])))
    exp_runs = 2.0 * npos * nneg / n + 1.0
    var_runs = (2.0 * npos * nneg * (2.0 * npos * nneg - n)) / (n ** 2 * (n - 1.0))
    z = (runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else np.nan
    dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    return {"n_runs": runs, "expected_runs": float(exp_runs), "runs_z": float(z),
            "durbin_watson": dw, "lag1_autocorr": ac1, "max_abs_resid": float(np.max(np.abs(r)))}

# noise-floor estimate: std of 2nd differences / sqrt(6) on the absorption curve
d2 = np.diff(A_a, 2)
res["noise_floor_estimate_2nd_diff"] = float(np.std(d2) / np.sqrt(6.0))

# ------------------------------------------------- action points vs absorption
A_at_action = np.interp(lam_y, lam_a, A_a)
res["input"]["absorption_at_action_nm"] = A_at_action.tolist()

# key ratios quoted in the task context, recomputed
def ratio(v, l0, l1):
    i0 = int(np.argmin(np.abs(lam_y - l0))); i1 = int(np.argmin(np.abs(lam_y - l1)))
    return float(v[i1] / v[i0]), i0, i1
r_act, i450, i650 = ratio(y, 450, 650)
r_abs, _, _ = ratio(A_at_action, 450, 650)
res["sanity"] = {"action_ratio_650_over_450": r_act, "absorption_ratio_650_over_450": r_abs,
                 "action_450": float(y[i450]), "action_650": float(y[i650]),
                 "A_450": float(A_at_action[i450]), "A_650": float(A_at_action[i650])}

# plain linear-in-A absorption-limited fit  y = s*A  (action points)
def fit_linear(lam, yv):
    Av = np.interp(lam, lam_a, A_a)
    cols = Av[:, None]
    coef, rss, pred, resid = linls(cols, yv)
    return {"s": float(coef[0]), "rss": rss, "rmse": float(np.sqrt(rss / len(yv))),
            "pred": pred.tolist(), "resid": resid.tolist(), "alpha": Av.tolist()}

lin_all = fit_linear(lam_y, y)
m5 = lam_y >= 450
lin_450 = fit_linear(lam_y[m5], y[m5])
res["linear_fit_all6"] = {k: lin_all[k] for k in ("s", "rss", "rmse", "pred", "resid")}
res["linear_fit_450_650"] = {k: lin_450[k] for k in ("s", "rss", "rmse", "pred", "resid")}
res["linear_fit_all6"]["pred650_over_actual650"] = float(lin_all["pred"][i650] / y[i650])
res["linear_fit_450_650"]["pred650_over_actual650"] = float(lin_450["pred"][-1] / y[i650])

# ================================================================ H1
# Fit absorption in 500-800 nm with (i) exponential, (ii) logistic, (iii) kinked piecewise linear.
win = (lam_a >= 500.0) & (lam_a <= 800.0)
x = lam_a[win]; z = A_a[win]; n = len(x)
res["H1"] = {"window_nm": [500.0, 800.0], "n_points": int(n),
             "curvature_note": "absorption is a single smooth spectrum; points 0.5 nm apart are not independent"}

# --- (i) exponential  z = c + a*exp(-(x-500)/tau)
def rss_exp(tau):
    cols = np.column_stack([np.ones_like(x), np.exp(-(x - 500.0) / tau)])
    _, rss, _, _ = linls(cols, z)
    return rss
taus = np.exp(np.linspace(np.log(2.0), np.log(800.0), 400))
rss_tau = np.array([rss_exp(t) for t in taus])
best_t = taus[int(np.argmin(rss_tau))]
opt = minimize_scalar(rss_exp, bounds=(2.0, 2000.0), method="bounded",
                      options={"xatol": 1e-6})
tau_e = float(opt.x)
cols_e = np.column_stack([np.ones_like(x), np.exp(-(x - 500.0) / tau_e)])
coef_e, rss_e, pred_e, resid_e = linls(cols_e, z)
res["H1"]["exponential"] = {
    "form": "A = c + a*exp(-(lam-500)/tau)", "k": 3,
    "c": float(coef_e[0]), "a": float(coef_e[1]), "tau_nm": tau_e,
    "rss": rss_e, "rmse": float(np.sqrt(rss_e / n)), "aicc": float(aicc(rss_e, n, 3)),
    "resid": runstats(resid_e)}

# --- (ii) logistic  z = c + a/(1+exp((x-lamc)/w))
def rss_log(p):
    lamc, w = p
    if w <= 0.05 or lamc < 490 or lamc > 810:
        return 1e9
    cols = np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(np.clip((x - lamc) / w, -500, 500)))])
    _, rss, _, _ = linls(cols, z)
    return rss
lamc_grid = np.linspace(505, 795, 120)
w_grid = np.exp(np.linspace(np.log(0.5), np.log(300), 120))
grid_l = np.array([[rss_log((lc, w)) for w in w_grid] for lc in lamc_grid])
i0, j0 = np.unravel_index(int(np.argmin(grid_l)), grid_l.shape)
p0 = [lamc_grid[i0], w_grid[j0]]
def _resid_log(p):
    lamc, w = p
    cols = np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(np.clip((x - lamc) / w, -500, 500)))])
    coef, *_ = np.linalg.lstsq(cols, z, rcond=None)
    return z - cols @ coef
lsol = least_squares(_resid_log, p0, method="lm", max_nfev=20000)
lamc_l, w_l = lsol.x
cols_l = np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(np.clip((x - lamc_l) / w_l, -500, 500)))])
coef_l, rss_l, pred_l, resid_l = linls(cols_l, z)
res["H1"]["logistic"] = {
    "form": "A = c + a/(1+exp((lam-lamc)/w))", "k": 4,
    "c": float(coef_l[0]), "a": float(coef_l[1]), "lamc_nm": float(lamc_l), "w_nm": float(w_l),
    "rss": rss_l, "rmse": float(np.sqrt(rss_l / n)), "aicc": float(aicc(rss_l, n, 4)),
    "resid": runstats(resid_l)}

# --- (iii) continuous piecewise-linear kink at free lamk
best_k = None
for lamk in x[1:-1]:
    x1 = np.minimum(x - 500.0, lamk - 500.0)
    x2 = np.maximum(x - lamk, 0.0)
    cols = np.column_stack([np.ones_like(x), x1, x2])
    coefk, rssk, _, _ = linls(cols, z)
    if best_k is None or rssk < best_k[0]:
        best_k = (rssk, float(lamk), coefk)
rss_k, lamk, coef_k = best_k
x1 = np.minimum(x - 500.0, lamk - 500.0); x2 = np.maximum(x - lamk, 0.0)
cols_k = np.column_stack([np.ones_like(x), x1, x2])
_, _, pred_k, resid_k = linls(cols_k, z)
res["H1"]["kinked_piecewise_linear"] = {
    "form": "b0 + m1*(lam-500) for lam<=lamk ; continuous kink then slope m2", "k": 4,
    "b0_at_500": float(coef_k[0]), "slope1_per_nm": float(coef_k[1]), "slope2_per_nm": float(coef_k[2]),
    "kink_nm": lamk, "rss": rss_k, "rmse": float(np.sqrt(rss_k / n)),
    "aicc": float(aicc(rss_k, n, 4)), "resid": runstats(resid_k)}

# --- extra descriptive: bi-exponential smooth two-component decay (k=5)
def rss_bi(p):
    t1, t2 = p
    if t1 <= 0 or t2 <= 0 or t1 >= t2:
        return 1e9
    cols = np.column_stack([np.ones_like(x), np.exp(-(x - 500.0) / t1), np.exp(-(x - 500.0) / t2)])
    _, rss, _, _ = linls(cols, z)
    return rss
tg = np.exp(np.linspace(np.log(3), np.log(600), 70))
grid_b = np.array([[rss_bi((t1, t2)) for t2 in tg] for t1 in tg])
i1, j1 = np.unravel_index(int(np.argmin(grid_b)), grid_b.shape)
def _rb(p, xx, zz):
    t1, t2 = np.sort(p)
    cols = np.column_stack([np.ones_like(xx), np.exp(-(xx - 500.0) / t1), np.exp(-(xx - 500.0) / t2)])
    coef, *_ = np.linalg.lstsq(cols, zz, rcond=None)
    return zz - cols @ coef
lsolb = least_squares(lambda p: _rb(p, x, z), [tg[i1], tg[j1]], method="lm", max_nfev=20000)
t1b, t2b = np.sort(lsolb.x)
cols_b = np.column_stack([np.ones_like(x), np.exp(-(x - 500.0) / t1b), np.exp(-(x - 500.0) / t2b)])
coef_b, rss_b, _, resid_b = linls(cols_b, z)
res["H1"]["biexponential_reference"] = {
    "form": "c + a1*exp(-(lam-500)/t1) + a2*exp(-(lam-500)/t2)", "k": 5,
    "c": float(coef_b[0]), "a1": float(coef_b[1]), "t1_nm": float(t1b),
    "a2": float(coef_b[2]), "t2_nm": float(t2b),
    "rss": rss_b, "rmse": float(np.sqrt(rss_b / n)), "aicc": float(aicc(rss_b, n, 5)),
    "resid": runstats(resid_b)}

aiccs = {k: res["H1"][k]["aicc"] for k in ("exponential", "logistic", "kinked_piecewise_linear")}
res["H1"]["delta_aicc_vs_best"] = {k: float(v - min(aiccs.values())) for k, v in aiccs.items()}
# robustness of the AIC ranking to pseudoreplication (601 points are NOT independent)
rss_e_ = res["H1"]["exponential"]["rss"]
robust = {}
for neff in (601, 301, 151, 61, 31, 15, 10):
    d_el = neff * np.log(rss_e_ / rss_l) + 2 * (3 - 4)          # exp vs logistic
    d_lk = neff * np.log(rss_l / rss_k) + 2 * (4 - 4)          # logistic vs kink
    robust[f"n_eff_{neff}"] = {"dAIC_exp_minus_logistic": float(d_el), "dAIC_logistic_minus_kink": float(d_lk)}
res["H1"]["aic_ranking_vs_effective_n"] = robust
# anchored-scale variant of the absorption-limited null (the ~4.5x figure)
res["sanity"]["anchored_scale_null"] = {
    "s_anchored_450_nm": float(y[i450] / A_at_action[i450]),
    "pred_650_nm": float(y[i450] / A_at_action[i450] * A_at_action[i650]),
    "pred650_over_actual650": float((y[i450] / A_at_action[i450] * A_at_action[i650]) / y[i650])}

# ================================================================ H2
# Treat measured absorption as OD with unknown scale k: alpha = 1 - 10^(-k*A); action = s*alpha.
Av = A_at_action.copy()
def rmse_for_alpha(alpha, yv):
    s = float(np.dot(alpha, yv) / np.dot(alpha, alpha))
    pred = s * alpha
    return s, float(np.sqrt(np.mean((yv - pred) ** 2))), pred
rmse_lin = float(np.sqrt(np.mean((y - lin_all["s"] * Av) ** 2)))
ks = np.logspace(-3, 2.0, 600)
best = None
for k in ks:
    alpha = 1.0 - 10.0 ** (-k * Av)
    s, r, pred = rmse_for_alpha(alpha, y)
    if np.isfinite(r) and (best is None or r < best[1]):
        best = (float(k), r, s, pred, alpha)
k_small = float(ks[0])
al_small = 1.0 - 10.0 ** (-k_small * Av)
s_small, r_small, _ = rmse_for_alpha(al_small, y)
# sample the curve at a few k for the record
sample = {}
for kk in [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    if kk == 0.0:
        s = lin_all["s"]; pred = s * Av; r = rmse_lin
    else:
        alpha = 1.0 - 10.0 ** (-kk * Av)
        s, r, pred = rmse_for_alpha(alpha, y)
    sample[f"{kk:g}"] = {"rmse": r, "s": s,
                         "pred650_over_actual650": float(pred[i650] / y[i650]),
                         "pred650_450_ratio": float(pred[i650] / pred[i450])}
res["H2"] = {"transform": "alpha = 1 - 10^(-k*A)", "n_action_points": int(len(y)),
             "k_grid_range": [1e-3, 100.0], "rmse_vs_k_samples": sample,
             "rmse_at_k_to_0_limit": rmse_lin,
             "rmse_at_k_1e-3": r_small, "s_at_k_1e-3": s_small,
             "s_eff_at_k_1e-3_equals_s_k_ln10": float(s_small * k_small * np.log(10.0)),
             "best_k_on_grid": best[0], "best_rmse_on_grid": best[1],
             "best_s": best[2],
             "plain_linear_rmse": rmse_lin,
             "observed_650_over_450_action_ratio": r_act,
             "monotone_check": {k: (1.0 - 10.0 ** (-k * Av[i650])) / (1.0 - 10.0 ** (-k * Av[i450])) for k in (0.1, 1.0, 10.0, 100.0)}}
res["H2"]["beats_plain_linear"] = bool(best[1] < rmse_lin - 1e-12)
res["H2"]["best_pred650_over_actual650"] = float(best[3][i650] / y[i650])

# secondary diagnostic: convex power law y = s*A^p (what kind of transform WOULD be needed?)
def rmse_pow(p):
    alpha = Av ** p
    s = float(np.dot(alpha, y) / np.dot(alpha, alpha))
    pred = s * alpha
    return s, float(np.sqrt(np.mean((y - pred) ** 2))), pred
ps = np.linspace(0.3, 6.0, 2000)
rr = [rmse_pow(p) for p in ps]
ip = int(np.argmin([q[1] for q in rr]))
res["H2"]["power_law_diagnostic"] = {"form": "y = s*A^p", "best_p": float(ps[ip]),
                                     "rmse": rr[ip][1], "s": rr[ip][0],
                                     "pred650_over_actual650": float(rr[ip][2][i650] / y[i650])}

# ================================================================ H3
# action = s*A + c*exp(-(lam-l0)/w).  Note exp(-(lam-l0)/w) = exp(l0/w)*exp(-lam/w):
# l0 and c are NOT separately identifiable; only C = c*exp(l0/w) and w are. Fit in (s, C, w).
def fit_h3(lam, yv, wgrid=None, nonneg=False):
    Avv = np.interp(lam, lam_a, A_a)
    if wgrid is None:
        wgrid = np.exp(np.linspace(np.log(3.0), np.log(2000.0), 800))
    best = None
    for w in wgrid:
        e = np.exp(-lam / w)
        cols = np.column_stack([Avv, e])
        if nonneg:
            coef, _ = nnls(cols, yv)
            s, C = float(coef[0]), float(coef[1])
            pred = cols @ coef
            rss = float(np.sum((yv - pred) ** 2))
        else:
            coef, rss_, pred, _ = linls(cols, yv)
            s, C = float(coef[0]), float(coef[1]); rss = rss_
        if best is None or rss < best["rss"]:
            best = {"s": s, "C": C, "w_nm": float(w), "rss": rss}
    return best
h3 = fit_h3(lam_y, y)
h3["rmse"] = float(np.sqrt(h3["rss"] / len(y)))
h3["aicc"] = float(aicc(h3["rss"], len(y), 3))
e650 = float(np.exp(-lam_y[i650] / h3["w_nm"]))
Avv = A_at_action
h3["pred"] = (h3["s"] * Avv + h3["C"] * np.exp(-lam_y / h3["w_nm"])).tolist()
h3["pred650_over_actual650"] = float(h3["pred"][i650] / y[i650])
# report in the task's (l0, c) parameterisation with an arbitrary l0 reference
for l0_ref in (600.0, 650.0):
    h3[f"equivalent_c_if_l0_{int(l0_ref)}"] = float(h3["C"] * np.exp(-l0_ref / h3["w_nm"]))
res["H3"] = {"form": "y = s*A + c*exp(-(lam-l0)/w); reparameterised y = s*A + C*exp(-lam/w) "
                     "because c and l0 are exactly degenerate (C = c*exp(l0/w))",
             "k": 3, "fit": h3,
             "linear_fit_aicc": float(aicc(lin_all["rss"], len(y), 1)),
             "linear_fit_rmse": lin_all["rmse"],
             "amplitude_sign": "negative (red-end deficit term)" if h3["C"] < 0 else "positive (extra red emitter)",
             "constrained_C_ge_0": fit_h3(lam_y, y, nonneg=True)}
res["H3"]["constrained_C_ge_0"]["rmse"] = float(np.sqrt(res["H3"]["constrained_C_ge_0"]["rss"] / len(y)))
res["H3"]["constrained_C_ge_0"]["pred650_over_actual650"] = float(
    (res["H3"]["constrained_C_ge_0"]["s"] * A_at_action[i650] +
     res["H3"]["constrained_C_ge_0"]["C"] * np.exp(-lam_y[i650] / res["H3"]["constrained_C_ge_0"]["w_nm"])) / y[i650])
res["H3"]["delta_aicc_vs_linear"] = float(h3["aicc"] - res["H3"]["linear_fit_aicc"])

# leave-one-out cross validation on the 6 action points (the honest test for a 3-param addition)
def loo(model):
    errs, store = [], {}
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        if model == "linear":
            linf = fit_linear(lam_y[tr], y[tr])
            p = float(np.interp(lam_y[i], lam_a, A_a)) * linf["s"]
        elif model == "h3":
            f = fit_h3(lam_y[tr], y[tr])
            p = f["s"] * A_at_action[i] + f["C"] * np.exp(-lam_y[i] / f["w_nm"])
        elif model == "pow":
            s, r, pr = rmse_pow_loo(lam_y[tr], y[tr], lam_y[i])
            p = pr
        errs.append(float(p - y[i]))
        store[int(lam_y[i])] = {"held_out": float(y[i]), "pred": float(p)}
    return {"loo_rmse": float(np.sqrt(np.mean(np.square(errs)))), "per_point": store}
def rmse_pow_loo(lam, yv, lam_pred):
    Avv = np.interp(lam, lam_a, A_a)
    best = None
    for p in np.linspace(0.3, 6.0, 600):
        alpha = Avv ** p
        s = float(np.dot(alpha, yv) / np.dot(alpha, alpha))
        rss = float(np.sum((yv - s * alpha) ** 2))
        if best is None or rss < best[0]:
            best = (rss, s, p)
    a = np.interp(lam_pred, lam_a, A_a) ** best[2]
    return best[1], best[0], float(best[1] * a)
res["H3"]["loo"] = {"linear": loo("linear"), "h3": loo("h3"), "power_law": loo("pow")}
# specific held-out 650 prediction from a fit to 400-600 only
tr5 = lam_y < 650
res["H3"]["holdout_650_from_fit_400_600"] = {
    "actual": float(y[i650]),
    "linear_pred": float(fit_linear(lam_y[tr5], y[tr5])["s"] * A_at_action[i650]),
    "h3_pred": float(fit_h3(lam_y[tr5], y[tr5])["s"] * A_at_action[i650] +
                     fit_h3(lam_y[tr5], y[tr5])["C"] * np.exp(-650.0 / fit_h3(lam_y[tr5], y[tr5])["w_nm"]))}

# ================================================================ REFINEMENTS
# --- H1a: can any sharp edge (small fixed width) describe the 500-800 nm fall-off?
sharp = {}
for wfix in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
    bestw = None
    for lamc in np.linspace(500, 800, 301):
        cols = np.column_stack([np.ones_like(x), 1.0/(1.0+np.exp(np.clip((x-lamc)/wfix,-500,500)))])
        _, rssw, _, _ = linls(cols, z)
        if bestw is None or rssw < bestw[0]:
            bestw = (rssw, float(lamc))
    sharp[f"w_fixed_{wfix:g}nm"] = {"best_lamc_nm": bestw[1], "rss": bestw[0],
                                    "rmse": float(np.sqrt(bestw[0]/n)),
                                    "aicc": float(aicc(bestw[0], n, 3))}
res["H1"]["fixed_width_logistic"] = sharp
# model-free sharpness of the real edge: max |dA/dlam| in window, and 90->10 width of the fitted logistic
dA = np.gradient(z, x)
res["H1"]["falloff_metrics"] = {
    "unfiltered_data_max_abs_2nm_diff_in_500_800": float(np.max(np.abs(np.diff(z)))),
    "smoothed_rolling20_max_abs_slope_per_nm": float(np.max(np.abs(
        np.gradient(np.convolve(z, np.ones(20) / 20, mode="valid"), x[9:-10])))),
    "max_abs_slope_per_nm_in_500_800": float(np.max(np.abs(dA))),
    "lambda_of_max_slope_nm": float(x[int(np.argmax(np.abs(dA)))]),
    "median_abs_slope_per_nm": float(np.median(np.abs(dA))),
    "sharp_edge_threshold_note": "a 1-nm edge spanning ~0.85 units needs |dA/dlam|~0.85/1nm; "
                                 "observed smoothed max slope is 1-2 orders of magnitude smaller at every wavelength",
    "fitted_logistic_90to10_width_nm": float(4.3944 * w_l),   # 2*ln9*w
    "fitted_logistic_width_param_w_nm": float(w_l),
    "fitted_exponential_tau_nm": tau_e,
    "interpretation_note": "sharp instrument-like cut-off would require w of a few nm; "
                           "data demand w ~ 80 nm and a kinked edge fits worse than a smooth curve"}

# --- H2a: analytic monotonicity. For alpha = 1-10^(-kA) (concave, alpha(0)=0, increasing),
# alpha(k*A650)/alpha(k*A450) >= A650/A450 for every k>0; saturation can never steepen the ratio.
res["H2"]["analytic_bound"] = {
    "claim": "alpha(k*Af)/(alpha(k*Ag)) >= Af/Ag for all k>0 (concavity/subadditivity of 1-10^-x)",
    "A450": float(A_at_action[i450]), "A650": float(A_at_action[i650]),
    "min_possible_predicted_650_over_450_ratio": float(A_at_action[i650] / A_at_action[i450]),
    "observed_action_650_over_450_ratio": r_act,
    "observed_is_below_bound": bool(r_act < A_at_action[i650] / A_at_action[i450]),
    "k_to_0_rmse_sequence": {}}
for kk in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    alpha = 1.0 - 10.0 ** (-kk * Av)
    s, r, pred = rmse_for_alpha(alpha, y)
    res["H2"]["analytic_bound"]["k_to_0_rmse_sequence"][f"{kk:g}"] = float(r)

# --- H3a: F-test, profile, and the sign-constrained version
rss_lin, kp = lin_all["rss"], 1
rss_h3, kh = h3["rss"], 3
dfn = kh - kp; dfd = len(y) - kh
F = ((rss_lin - rss_h3) / dfn) / (rss_h3 / dfd)
from scipy.stats import f as fdist
res["H3"]["F_test_h3_vs_linear"] = {"F": float(F), "dfn": int(dfn), "dfd": int(dfd),
                                    "p_value": float(fdist.sf(F, dfn, dfd)),
                                    "note": "6 action points; 3-parameter model has little power"}
# profile of RSS over w (C free, sign unconstrained) to expose non-identifiability
prof_w = {}
for w in (50, 100, 150, 200, 300, 400, 500, 640, 800, 1000, 1500):
    e = np.exp(-lam_y / w)
    cols = np.column_stack([A_at_action, e])
    coef, rssw, _, _ = linls(cols, y)
    prof_w[f"{w}nm"] = {"rss": rssw, "s": float(coef[0]), "C": float(coef[1]),
                        "rmse": float(np.sqrt(rssw / len(y)))}
res["H3"]["rss_profile_over_w"] = prof_w
# what the H3 fit implies as an *additive red emitter*: C>0 only
res["H3"]["verdict_numbers"] = {
    "unconstrained_C": h3["C"], "unconstrained_rmse": h3["rmse"],
    "constrained_C_ge_0_rmse": res["H3"]["constrained_C_ge_0"]["rmse"],
    "linear_rmse": lin_all["rmse"],
    "rmse_improvement_from_positive_red_component": float(lin_all["rmse"] - res["H3"]["constrained_C_ge_0"]["rmse"]),
    "pred650_over_actual650_positive_component": res["H3"]["constrained_C_ge_0"]["pred650_over_actual650"]}

with open(OUTDIR / "h1h2h3_optics_test.json", "w") as f:
    json.dump(res, f, indent=2)
print("H1 fixed-width logistic:")
print(json.dumps(sharp, indent=2))
print("H1 falloff metrics:")
print(json.dumps(res["H1"]["falloff_metrics"], indent=2))
print("H2 analytic bound:")
print(json.dumps(res["H2"]["analytic_bound"], indent=2))
print("H3 F / profile / verdict:")
print(json.dumps({k: res["H3"][k] for k in ("F_test_h3_vs_linear", "rss_profile_over_w", "verdict_numbers")}, indent=2))
print("DONE ->", OUTDIR / "h1h2h3_optics_test.json")