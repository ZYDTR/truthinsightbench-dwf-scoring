"""Audit scratch: re-derive Finding 4 numbers for the cofactor-loading scan.

Reads raw data + cited artifacts; prints report value vs recomputed value.
"""
import itertools
import json

import numpy as np
from scipy import optimize, stats

DATA = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/cofactor_loading_response.tsv"
OUT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output"

L = np.array([0.2, 1.0, 3.0, 8.0])
R = np.array([1081.0, 1217.0, 470.0, 268.0])

print("== basic ratios ==")
print("drop 1.0->3.0 pct:", round(100 * (1217 - 470) / 1217, 3), "(report 61.4)")
print("rate 8.0 / rate 0.2 pct:", round(100 * R[-1] / R[0], 3), "(report 24.8)")
print("low pair diff pct:", round(100 * abs(R[1] - R[0]) / R[0], 3), "(report 12.6)")

print("\n== Spearman / exact permutation ==")
sp = stats.spearmanr(L, R)
print("rho", sp.statistic, "asymptotic p", sp.pvalue, "(report -0.800 / 0.200)")
perms = [stats.spearmanr(L, p).statistic for p in itertools.permutations(R)]
n_ext = int(np.sum([abs(v) >= abs(sp.statistic) - 1e-12 for v in perms]))
print("n orderings", len(perms), "n at least as extreme", n_ext,
      "exact two-sided p", n_ext / len(perms), "(report 8 of 24, p=0.333)")

print("\n== leave-one-out ==")
for drop in range(4):
    keep = [i for i in range(4) if i != drop]
    s = stats.spearmanr(L[keep], R[keep])
    exact = np.mean([abs(stats.spearmanr(L[keep], p).statistic)
                     >= abs(s.statistic) - 1e-12 for p in itertools.permutations(R[keep])])
    print(f"drop {L[drop]}%: rho={s.statistic:+.1f} asympt p={s.pvalue:.3f} exact two-sided p={exact:.3f}")

print("\n== substrate-inhibition fit ==")
def model(x, V, K, Ki):
    return V * x / (K + x + x * x / Ki)

popt, _ = optimize.curve_fit(model, L, R, p0=[R.max() * 3, 1.0, 20.0], maxfev=200000)
V, K, Ki = popt
rss = float(((R - model(L, *popt)) ** 2).sum())
sst = float(((R - R.mean()) ** 2).sum())
print("V", V, "K", K, "Ki", Ki, "R2", 1 - rss / sst, "RMSE", np.sqrt(rss / 4),
      "peak sqrt(K*Ki)", np.sqrt(K * Ki), "(report 17486/2.599/0.0923/0.9901/39.8/0.490)")

print("\n== AICc / AIC / BIC of the five stage-4 models ==")
sis = json.load(open(f"{OUT}/results/loading_crossmodal.json"))["loading_analysis"]
print("ranking_by_aicc:", sis["ranking_by_aicc"])
print("aicc_undefined_models:", sis["aicc_undefined_models"])
for name, m in sis["models"].items():
    print(f"  {name}: k={m['k_params']} AIC={m['aic']:.3f} AICc={m['aicc']} BIC={m['bic']:.3f}")

print("\n== agent-B artifact (F039/F037) 3-parameter competitors ==")
aB = json.load(open(f"{OUT}/verification/cofactor_shape_identifiability_agentB.json"))
for name, rssv, k, aic in aB["indistinguishable_sets"]["aic_ranking"][:6]:
    print(f"  {name}: k={k} AIC={aic:.3f} RSS={rssv:.1f}")
print("by_aicc_within_2:", aB["indistinguishable_sets"]["by_aicc_within_2"])
print("by_aic_within_2:", aB["indistinguishable_sets"]["by_aic_within_2"])
f37 = json.load(open(f"{OUT}/verification/cofactor_shape_identifiability.json"))
d = {m["model"]: m for m in f37["models"]}
print("  F037 BIC substrate inhibition:", d["substrate inhibition V*x/(K+x+x^2/Ki)"]["bic"])
print("  F037 BIC flat-then-exp decline:", d["flat then exponential decline (two-regime)"]["bic"],
      " AIC:", d["flat then exponential decline (two-regime)"]["aic"])
print("  F037 BIC log-quadratic:", d["log-quadratic peak a-b*(lnx-lnP)^2"]["bic"],
      " AIC:", d["log-quadratic peak a-b*(lnx-lnP)^2"]["aic"])
print("  F037 logquad free peak:", f37["peak_location"]["logquad_free_peak_P"],
      "profile range 1.05x:", f37["peak_location"]["logquad_profile_P_range_RSS<=1.05xmin"])

print("\n== bootstrap reproduction (seed 20220101) ==")
resid = R - model(L, *popt)
rng = np.random.default_rng(20220101)
def model_peakparam(x, Vp, Lp, Ki):
    return Vp * x / (Lp * Lp / Ki + x + x * x / Ki)
pp0, _ = optimize.curve_fit(model_peakparam, L, R, p0=[V, float(np.sqrt(K * Ki)), Ki], maxfev=200000)
draws = rng.choice(resid, size=(5000, 4), replace=True)
Rb_all = model(L, *popt)[None, :] + draws

def residuals(p, x, y):
    return model_peakparam(x, *p) - y

peaks, fails = [], 0
for Rb in Rb_all:
    try:
        sol = optimize.least_squares(residuals, pp0, args=(L, Rb), method="lm", max_nfev=400)
        if sol.success and np.all(np.isfinite(sol.x)) and sol.x[1] > 0:
            peaks.append(float(sol.x[1]))
        else:
            fails += 1
    except Exception:
        fails += 1
peaks = np.array(peaks)
print("successful", len(peaks), "failed", fails,
      "median", np.median(peaks),
      "CI95", [np.percentile(peaks, 2.5), np.percentile(peaks, 97.5)],
      "frac<3%", np.mean(peaks < 3.0))
print("(report: 3660/5000, CI [0.453, 0.525], 100% below 3%)")

print("\n== scale ratios (loading vs timecourse) ==")
print("max ratio", R.max() / 0.04325, "min ratio", R.min() / 2.37)