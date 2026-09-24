#!/usr/bin/env python3
"""Independent recomputation of the cofactor-loading scan analysis.

Reads the raw TSV directly:
    data/cofactor_loading_response.tsv
columns: cofactor_loading_percent, product_evolution_rate_umol_h
conditions: 0.2, 1.0, 3.0, 8.0
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

ROOT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project"
TSV = os.path.join(ROOT, "data", "cofactor_loading_response.tsv")
OUT = os.path.join(ROOT, "output", "verification", "cofactor_loading_recomputed.json")

df = pd.read_csv(TSV, sep="\t")
df = df.sort_values("cofactor_loading_percent").reset_index(drop=True)
L = df["cofactor_loading_percent"].to_numpy(dtype=float)
R = df["product_evolution_rate_umol_h"].to_numpy(dtype=float)
print("data:\n", df.to_string(index=False))

res = {"conditions": L.tolist(), "rates": R.tolist()}

# --- peak loading -------------------------------------------------------
imax = int(np.argmax(R))
res["max_rate"] = float(R[imax])
res["max_rate_loading"] = float(L[imax])

# --- ratio 8% vs 0.2% ---------------------------------------------------
r02 = float(R[np.where(L == 0.2)[0][0]])
r8 = float(R[np.where(L == 8.0)[0][0]])
res["rate_at_8"] = r8
res["rate_at_0p2"] = r02
res["pct_8_of_0p2"] = 100.0 * r8 / r02

# --- Spearman (scipy asymptotic) ---------------------------------------
rho, pval = stats.spearmanr(L, R)
res["spearman_rho"] = float(rho)
res["spearman_p_asymptotic"] = float(pval)

# --- exact two-sided permutation p over all 4! orderings ----------------
# Two-sided criterion: |rho_perm| >= |rho_obs| (equivalently, since rho is
# a strictly monotone function of the rank-sum / inversion count, using the
# absolute rank correlation is the standard symmetric two-sided rule).
n = len(L)
rho_all = []
for perm in itertools.permutations(range(n)):
    r = stats.spearmanr(L, R[list(perm)]).statistic
    rho_all.append(r)
rho_all = np.array(rho_all)
assert len(rho_all) == 24
obs = abs(rho)
extreme = np.abs(rho_all) >= obs - 1e-12
res["perm_n_orderings"] = int(len(rho_all))
res["perm_n_at_least_as_extreme"] = int(extreme.sum())
res["perm_p_two_sided"] = float(extreme.mean())
res["perm_rho_values"] = sorted(np.round(rho_all, 6).tolist())

# --- leave-one-out ------------------------------------------------------
loo = {}
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    Ll, Rl = L[mask], R[mask]
    r, p = stats.spearmanr(Ll, Rl)
    dec = bool(np.all(np.diff(Rl) < 0))
    loo[f"leave_out_{L[i]}"] = {
        "remaining_conditions": Ll.tolist(),
        "remaining_rates": Rl.tolist(),
        "spearman_rho": float(r),
        "spearman_p_asymptotic": float(p),
        "strictly_decreasing": dec,
    }
res["leave_one_out"] = loo

# --- substrate-inhibition fit: rate = V*L/(K + L + L^2/Ki) --------------
def model(params, x):
    V, K, Ki = params
    return V * x / (K + x + x * x / Ki)

def resid(params, x, y):
    return model(params, x) - y

# multi-start least_squares (only a handful of starts -> fast)
best = None
starts = []
for V0 in (1000.0, 1500.0, 3000.0):
    for K0 in (0.05, 0.2, 1.0):
        for Ki0 in (0.1, 1.0, 5.0):
            starts.append((V0, K0, Ki0))
for s0 in starts:
    try:
        sol = least_squares(resid, s0, args=(L, R), method="lm", max_nfev=20000)
    except Exception:
        continue
    if not np.all(np.isfinite(sol.x)):
        continue
    ss = float(np.sum(sol.fun ** 2))
    if best is None or ss < best[0]:
        best = (ss, sol.x, sol.cost, sol.optimality)

ss_res, theta, cost, opt = best
V, K, Ki = (float(v) for v in theta)
pred = model(theta, L)
ss_tot = float(np.sum((R - R.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot
res["fit"] = {
    "V": V,
    "K": K,
    "Ki": Ki,
    "ss_res": ss_res,
    "ss_tot": ss_tot,
    "R2": float(r2),
    "peak_loading_sqrt_K_Ki": float(np.sqrt(K * Ki)),
    "predicted_rates": pred.tolist(),
    "residuals": (R - pred).tolist(),
    "n_starts_used": len(starts),
}
# numeric peak of the fitted function: sqrt(K*Ki) is the analytic argmax;
# confirm it and its predicted value on a fine grid.
grid = np.linspace(0.01, 20.0, 200001)
pg = model(theta, grid)
pk = float(grid[np.argmax(pg)])
res["fit"]["peak_loading_numeric"] = pk
res["fit"]["fit_at_analytic_peak"] = float(model(theta, np.array([np.sqrt(K * Ki)]))[0])
res["fit"]["fit_at_4_loadings"] = pred.tolist()
res["fit"]["predicted_at_analytic_peak_vs_grid_peak_diff"] = abs(
    pk - float(np.sqrt(K * Ki)))
# sanity: analytic peak value V/(1 + 2*sqrt(K/Ki))
res["fit"]["analytic_peak_value_formula"] = float(V / (1.0 + 2.0 * np.sqrt(K / Ki)))

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

print(json.dumps(res, indent=2))
print("\nwrote", OUT)