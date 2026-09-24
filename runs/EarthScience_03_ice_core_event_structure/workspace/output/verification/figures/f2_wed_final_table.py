#!/usr/bin/env python3
"""
Companion to f2_wed_verification.py: decisive numbers in the exact form the
claim requests, plus provenance refinement (which product base) and the
hard upper bound on any quadrature of the observed density profile.
Independent implementation; writes only to output/verification/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
OUT = os.path.join(BASE, "output/verification/figures")

d = pd.read_csv(os.path.join(BASE, "data/observations.tsv"), sep="\t")
c01 = d["c01_depth_top_m"].to_numpy(float)
c02 = d["c02_depth_ice_snow_m"].to_numpy(float)
c03 = d["c03_samp_thick_cm"].to_numpy(float)
rho = d["c05_density_ice_g_cm_3"].to_numpy(float)
c06 = d["c06_depth_w_e_m"].to_numpy(float)
n = len(c01)
dz = np.diff(c01)
dz3 = c03[:-1] / 100.0
rep = float(c06[-1])

R = {}

# ---- 1. four quadratures, 4 decimals ---------------------------------- #
W_trap = np.concatenate(([0.0], np.cumsum(0.5 * (rho[:-1] + rho[1:]) * dz)))
W_left = np.concatenate(([0.0], np.cumsum(rho[:-1] * dz)))
W_right = np.concatenate(([0.0], np.cumsum(rho[1:] * dz)))
W_c03l = np.concatenate(([0.0], np.cumsum(rho[:-1] * dz3)))
R["quadratures_base_m"] = {
    "trapezoid_on_c01_increments": round(float(W_trap[-1]), 4),
    "left_on_c01_increments": round(float(W_left[-1]), 4),
    "right_on_c01_increments": round(float(W_right[-1]), 4),
    "left_on_c03_thickness_increments": round(float(W_c03l[-1]), 4),
}
R["gaps_vs_reported_27p060_m"] = {k: round(v - 27.060, 4) for k, v in R["quadratures_base_m"].items()}
R["gap_pct_of_WED"] = {k: round(100 * (27.060 - v) / v, 4) for k, v in R["quadratures_base_m"].items()}

# ---- 2. attacks -------------------------------------------------------- #
si = int(np.argmax(dz))
att = {}
att["i_step"] = {
    "step_m": round(float(dz[si]), 4),
    "net_WED_needed_over_step_to_fill": round(float(rho[si] * dz[si]), 4),
    "WED_max_over_step_at_rho_max": round(float(rho.max() * dz[si]), 4),
    "W_left_plus_maxfill_m": round(float(W_left[-1] + rho.max() * dz[si]), 4),
    "residual_gap_after_maxfill_m": round(float(W_left[-1] + rho.max() * dz[si] - rep), 4),
    "residual_gap_after_maxfill_pct": round(float(100 * (W_left[-1] + rho.max() * dz[si] - rep) / W_left[-1]), 3),
}
shift_vals = {}
for s in (-2, -1, 1, 2):
    r = rho.copy()
    if s > 0:
        r[s:] = rho[:-s]
    else:
        r[:s] = rho[-s:]
    shift_vals[s] = float(np.sum(r[:-1] * dz))
att["ii_pairing"] = {
    "shift_minus2_m": round(shift_vals[-2], 4), "shift_minus1_m": round(shift_vals[-1], 4),
    "shift_plus1_m": round(shift_vals[1], 4), "shift_plus2_m": round(shift_vals[2], 4),
    "min_residual_gap_m": round(min(abs(v - rep) for v in shift_vals.values()), 4),
    "shift_effect_m": round(max(shift_vals.values()) - min(shift_vals.values()), 4),
}
z_mid = c01 + c03 / 200.0
att["iii_midpoint"] = {
    "trapezoid_m": round(float(W_trap[-1]), 4),
    "rho_evaluated_at_sample_midpoints_left_m": round(float(np.sum(rho[:-1] * dz)), 4),
    "midpoint_depth_product_zmidrho_left_m": round(float(np.sum(z_mid[:-1] * rho[:-1] * dz)), 1),
    "min_residual_gap_m": round(float(W_trap[-1] - rep), 4),
}
k = float(np.dot(c06, W_trap) / np.dot(W_trap, W_trap))
att["iv_scale_k"] = {
    "best_k_least_squares": round(k, 6),
    "1_over_k": round(1 / k, 6),
    "k_needed_to_map_WED_to_27p060": round(rep / W_trap[-1], 6),
    "k_implied_by_density_range_rho_max_over_mean": round(float(rho.max() / (W_trap[-1] / c01[-1])), 6),
    "max_abs_resid_at_best_k_m": round(float(np.max(np.abs(c06 - k * W_trap))), 4),
    "rmse_at_best_k_m": round(float(np.sqrt(np.mean((c06 - k * W_trap) ** 2))), 4),
    "ratio_c06_over_WED_min": round(float(np.nanmin(np.where(W_trap > 1e-9, c06 / W_trap, np.nan))), 4),
    "ratio_c06_over_WED_max": round(float(np.nanmax(np.where(W_trap > 1e-9, c06 / W_trap, np.nan))), 4),
}
secA = slice(0, 106)
secB = slice(106, n)
att["v_sections"] = {
    "A_WED_m": round(float(np.sum(rho[secA][:-1] * np.diff(c01[secA]))), 4),
    "A_c06_span_m": round(float(c06[105] - c06[0]), 4),
    "B_WED_m": round(float(np.sum(rho[secB][:-1] * np.diff(c01[secB]))), 4),
    "B_c06_span_m": round(float(c06[-1] - c06[106]), 4),
}
att["vi_hard_upper_bound"] = {
    "note": "any quadrature of int_0^41.98 rho dz with rho <= max observed density is bounded",
    "bound_m": round(float(rho.max() * c01[-1]), 4),
    "bound_plus_one_sample_thickness_m": round(float(rho.max() * (c01[-1] + c03[-1] / 100)), 4),
    "reported_m": rep,
    "reported_exceeds_bound_by_m": round(rep - float(rho.max() * c01[-1]), 4),
}
R["attacks"] = att

# ---- 3. decisive residuals and fits ----------------------------------- #
def fitstats(x):
    r = c06 - x
    xm, ym = x.mean(), c06.mean()
    b = float(np.sum((x - xm) * (c06 - ym)) / np.sum((x - xm) ** 2))
    a = float(ym - b * xm)
    r2 = 1.0 - float(np.sum((c06 - (a + b * x)) ** 2)) / float(np.sum((c06 - ym) ** 2))
    sse = float(np.sum(r ** 2))
    sst = float(np.sum((c06 - ym) ** 2))
    return {"max_abs_residual_m": round(float(np.max(np.abs(r))), 4),
            "rmse_m": round(float(np.sqrt(np.mean(r ** 2))), 4),
            "mean_residual_m": round(float(np.mean(r)), 5),
            "ols_a": round(a, 6), "ols_b": round(b, 6), "ols_r2": round(r2, 8),
            "r2_identity_line": round(1.0 - sse / sst, 6)}

prod01 = c01 * rho
prod02 = c02 * rho
R["decisive"] = {
    "c06_vs_c01xc05": fitstats(prod01),
    "c06_vs_c02xc05": fitstats(prod02),
    "c06_vs_integral_trapezoid": fitstats(W_trap),
    "c06_vs_integral_left": fitstats(W_left),
    "c06_vs_integral_c03_increments": fitstats(W_c03l),
}
R["residual_ratio_rmse_integral_over_product"] = round(
    R["decisive"]["c06_vs_integral_trapezoid"]["rmse_m"] /
    R["decisive"]["c06_vs_c01xc05"]["rmse_m"], 1)

# ---- 3b. provenance refinement: which base? --------------------------- #
scan = []
for dd in np.arange(0.0, 0.0301, 0.0005):
    cand = np.round((c01 + dd) * rho, 2)
    scan.append((round(float(dd), 4), int(np.sum(np.abs(c06 - cand) < 1e-9))))
best = max(scan, key=lambda t: t[1])
R["provenance_scan_offset"] = {
    "offsets_tested": [t[0] for t in scan],
    "n_exact_matches": [t[1] for t in scan],
    "best_offset_m": best[0], "best_exact": best[1],
    "n_at_offset_0": dict(scan)[0.0], "n_at_offset_0p01": dict(scan)[0.01],
    "n_at_offset_0p0105": dict(scan)[0.0105],
}
bias01 = c06 - prod01
R["provenance_scan_offset"]["mean_bias_vs_c01_product_m"] = round(float(bias01.mean()), 5)
R["provenance_scan_offset"]["implied_constant_offset_m"] = round(float(np.mean(bias01 / rho)), 5)

# ---- 4. gap reality statement ground truth ---------------------------- #
R["gap_summary"] = {
    "reported_m": rep,
    "WED_trapezoid_m": round(float(W_trap[-1]), 4),
    "abs_gap_m": round(rep - float(W_trap[-1]), 4),
    "gap_pct_of_WED": round(100 * (rep - float(W_trap[-1])) / float(W_trap[-1]), 3),
    "gap_pct_of_reported": round(100 * (rep - float(W_trap[-1])) / rep, 3),
    "max_WED_possible_m": round(float(rho.max() * c01[-1]), 4),
    "reported_minus_max_possible_m": round(rep - float(rho.max() * c01[-1]), 4),
}

with open(os.path.join(OUT, "f2_wed_final_table.json"), "w") as f:
    json.dump(R, f, indent=2)

with open(os.path.join(OUT, "f2_wed_final_table.txt"), "w") as f:
    f.write(json.dumps(R, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
offsets = [t[0] for t in scan]
counts = [t[1] for t in scan]
ax[0].plot(offsets, counts, "o-", ms=3)
ax[0].axvline(0.0, color="tab:red", ls="--", label="c01 (claim)")
ax[0].axvline(0.01, color="tab:green", ls="--", label="c02 = c01+0.01")
ax[0].axvline(0.0105, color="tab:orange", ls=":", label="sample midpoint c01+c03/200")
ax[0].set_xlabel("offset added to c01 before multiplying by c05 (m)")
ax[0].set_ylabel("rows where c06 == round((c01+offset)*c05, 2)")
ax[0].set_title("Provenance: the product base is c01 + ~0.010 m, not c01")
ax[0].legend(fontsize=8)

ax[1].plot(c01, c06, "k", lw=1.6, label="c06 reported")
ax[1].plot(c01, prod02, lw=1, label="c02*c05")
ax[1].plot(c01, prod01, lw=1, label="c01*c05")
ax[1].plot(c01, W_trap, lw=1.2, label="WED = int rho dz (trap)")
ax[1].set_xlabel("depth c01 (m)"); ax[1].set_ylabel("m"); ax[1].legend(fontsize=8)
ax[1].set_title("27.060 vs WED 21.222: gap = 5.838 m (27.5% of WED)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f2_wed_final_table.png"), dpi=120)
plt.close(fig)

print(json.dumps(R, indent=2))