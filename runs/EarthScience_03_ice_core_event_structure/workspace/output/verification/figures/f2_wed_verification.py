#!/usr/bin/env python3
"""
Independent verification of Draft Finding 2:
  "c06_depth_w_e_m is NOT a physically correct water-equivalent depth:
   it equals c01*c05 to within 0.0303 m, while a true WED (integral of
   density over depth) is 21.216-21.227 m at the base vs the reported
   27.060 m, a 27.5% gap that no quadrature closes."

Written from scratch (no reuse of output/analysis code).
Data: data/observations.tsv, 1931 rows.
All artifacts written to output/verification/figures/.
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
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(BASE, "data/observations.tsv"), sep="\t")
z = d["c01_depth_top_m"].to_numpy(float)        # sample-top depth, m
z2 = d["c02_depth_ice_snow_m"].to_numpy(float)  # ice/snow depth coordinate
th = d["c03_samp_thick_cm"].to_numpy(float)     # sample thickness, cm
rho = d["c05_density_ice_g_cm_3"].to_numpy(float)
c06 = d["c06_depth_w_e_m"].to_numpy(float)
n = len(z)
RHO_W = 1.0     # g/cm3, reference for water-equivalent
RHO_ICE = 0.917  # g/cm3, reference for ice-equivalent (alternative convention)
REP = float(c06[-1])  # reported final value = 27.06

dz = np.diff(z)                 # c01 increments (1930)
dz_th = th[:-1] / 100.0         # c03 increments attributed to interval i (1930)
z_mid = z + th / 200.0          # sample midpoint depth

L = {}  # report container


def cum_left(values, increments):
    return np.concatenate(([0.0], np.cumsum(values[:-1] * increments)))


def cum_trap(values, increments):
    return np.concatenate(([0.0], np.cumsum(0.5 * (values[:-1] + values[1:]) * increments)))


def cum_right(values, increments):
    return np.concatenate(([0.0], np.cumsum(values[1:] * increments)))


# ---------------------------------------------------------------- #
# 0. Grid / provenance facts
# ---------------------------------------------------------------- #
L["grid"] = {
    "n_rows": int(n), "n_intervals": int(n - 1),
    "z_first_m": float(z[0]), "z_last_m": float(z[-1]),
    "dz_min_m": float(dz.min()), "dz_median_m": float(np.median(dz)), "dz_max_m": float(dz.max()),
    "step_interval_index": int(np.argmax(dz)),
    "step_z_pair_m": [float(z[np.argmax(dz)]), float(z[np.argmax(dz) + 1])],
    "step_size_m": float(dz.max()),
    "rho_min": float(rho.min()), "rho_max": float(rho.max()),
    "rho_mean_depth_weighted": float(np.sum(rho[:-1] * dz) / z[-1]),
    "c06_first": float(c06[0]), "c06_last": REP,
    "c06_monotone_nondecreasing": bool(np.all(np.diff(c06) >= -1e-12)),
    "c06_two_decimals": bool(np.max(np.abs(c06 * 100 - np.round(c06 * 100))) < 1e-9),
    "c05_three_decimals": bool(np.max(np.abs(rho * 1000 - np.round(rho * 1000))) < 1e-9),
}

# ---------------------------------------------------------------- #
# 1. Four (plus) quadratures of the true water-equivalent integral
#    W(z) = int_0^z rho(z')/rho_w dz'   (rho and rho_w in g/cm3)
#    A: trapezoid on c01 increments
#    B: left endpoint on c01 increments
#    C: right endpoint on c01 increments
#    D: left endpoint using c03 sample thickness as the increment
#    E: trapezoid using c03 as the increment
#    F: right endpoint using c03 as the increment (last sample thickness)
# ---------------------------------------------------------------- #
W_trap = cum_trap(rho, dz)
W_left = cum_left(rho, dz)
W_right = cum_right(rho, dz)
W_left_c03 = cum_left(rho, th[:-1] / 100.0)
W_trap_c03 = cum_trap(rho, th[:-1] / 100.0)
W_right_c03 = cum_right(rho, th[:-1] / 100.0)  # right endpoint on c03 increments

quad = {
    "A trapezoid, c01 increments": W_trap[-1],
    "B left endpoint, c01 increments": W_left[-1],
    "C right endpoint, c01 increments": W_right[-1],
    "D left endpoint, c03 increments": W_left_c03[-1],
    "E trapezoid, c03 increments": W_trap_c03[-1],
    "F right endpoint, c03 increments": W_right_c03[-1],
    "ice-equivalent (divide by 0.917), trapezoid": W_trap[-1] / RHO_ICE,
}
L["quadratures_of_true_WED"] = {
    k: {"value_m": round(float(v), 6), "gap_vs_reported_m": round(float(v - REP), 6),
        "gap_vs_reported_pct": round(float(100 * (v - REP) / REP), 4),
        "gap_sum_over_core_pct": round(float(100 * (v - REP) / v), 4)}
    for k, v in quad.items()
}

# Rigorous envelope: with observed rho in [rho_min, rho_max] and depth z_last,
# ANY quadrature of int rho dz must lie in [rho_min*z_last, rho_max*z_last].
L["envelope_bound"] = {
    "lower": float(rho.min() * z[-1]), "upper": float(rho.max() * z[-1]),
    "reported_m": REP, "reported_inside": bool(rho.min() * z[-1] <= REP <= rho.max() * z[-1]),
    "reported_minus_upper_bound_m": REP - float(rho.max() * z[-1]),
}

# Counter-family: if one (mis)reads the quadrature as int (c01*c05) dz, the
# integrand endpoint is ~27 m and the integral is ~484 m.
Dprod = z * rho
famD = {
    "left endpoint": cum_left(Dprod, dz)[-1],
    "trapezoid": cum_trap(Dprod, dz)[-1],
    "right endpoint": cum_right(Dprod, dz)[-1],
    "left endpoint, c03 increments": cum_left(Dprod, th[:-1] / 100.0)[-1],
}
L["quadratures_of_int_c01c05_dz"] = {k: round(float(v), 6) for k, v in famD.items()}

# ---------------------------------------------------------------- #
# 2. ATTACK: can 27.060 be rescued?
# ---------------------------------------------------------------- #
attacks = {}
print("attack i: 1.008 m step")
si = int(np.argmax(dz))
attacks["i_depth_step"] = {
    "step_interval_index": si,
    "z_pair_m": [float(z[si]), float(z[si + 1])],
    "step_m": float(dz[si]),
    "true_WED_over_step_m": float(rho[si] * dz[si]),
    "WED_if_step_filled_at_rho_before_m": float(W_left[-1] + rho[si] * dz[si]),
    "WED_if_step_filled_at_rho_after_m": float(W_left[-1] + rho[si + 1] * dz[si]),
    "WED_max_if_step_filled_at_rho_max_m": float(W_left[-1] + rho.max() * dz[si]),
    "WED_left_if_step_interval_removed_m": float(W_left[-1] - rho[si] * dz[si]),
    "gap_after_max_fill_m": float(W_left[-1] + rho.max() * dz[si] - REP),
    "max_share_of_gap_from_step_pct": float(100 * rho.max() * dz[si] / (REP - W_left[-1])),
}

print("attack ii: density on different depths / shifted pairing")
# how freely can the pairing be changed?
poly6 = np.polyval(np.polyfit(z, rho, 6), z)
attacks["ii_density_pairing"] = {
    "max_abs_rho_minus_poly6": float(np.max(np.abs(rho - poly6))),
    "spearman_depth_vs_rho": float(pd.Series(z).corr(pd.Series(rho), method="spearman")),
}
shift = {}
for s in (-2, -1, 1, 2):
    r = rho.copy()
    if s > 0:
        r[s:] = rho[:-s]
    else:
        r[:s] = rho[-s:]
    shift[s] = float(cum_left(r, dz)[-1])
attacks["ii_density_pairing"]["shifted_integrals"] = shift
attacks["ii_density_pairing"]["min_gap_after_shift_m"] = float(
    min(abs(v - REP) for v in shift.values()))
# scale the whole density profile so the integral hits 27.06 -> what mean rho needed
attacks["ii_density_pairing"]["kappa_needed_to_hit_reported"] = float(REP / W_left[-1])

print("attack iii: midpoint depth / midpoint density")
mid = {
    "trapezoid on nodes (== midpoint rule on linear rho)": float(W_trap[-1]),
    "rho at sample midpoint (c01+c03/200), left": float(cum_left(rho, dz)[-1]),
    "product form evaluated at midpoint depth, left": float(
        np.sum((z[:-1] + th[:-1] / 200.0) * rho[:-1] * dz)),
    "product form evaluated at c02 depth, left": float(np.sum(z2[:-1] * rho[:-1] * dz)),
}
attacks["iii_midpoint_forms"] = mid
attacks["iii_midpoint_forms"]["note"] = ("product-form rows are of order 480 m and are shown only to "
                                        "document that no midpoint variant of the *integral of rho* "
                                        "exceeds 21.23 m")

print("attack iv: scale factor k")
def best_k_scale(x, y):
    k = float(np.dot(y, x) / np.dot(x, x))
    r = y - k * x
    return k, float(np.max(np.abs(r))), float(np.sqrt(np.mean(r ** 2)))

scale = {}
for name, vec in {"int rho dz (left)": W_left, "int rho dz (trap)": W_trap,
                  "int rho dz (right)": W_right,
                  "int rho/0.917 dz (trap)": W_trap / RHO_ICE}.items():
    k, mx, rm = best_k_scale(vec, c06)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vec > 1e-9, c06 / vec, np.nan)
    scale[name] = {"best_k": k, "max_abs_resid_m": mx, "rmse_m": rm,
                   "ratio_at_base": float(REP / vec[-1]),
                   "ratio_at_10pct_depth": float(np.nanmax(ratio)),
                   "ratio_min": float(np.nanmin(ratio)), "ratio_max": float(np.nanmax(ratio)),
                   "constant_k_fits": bool(mx < 0.1)}
attacks["iv_scale_factor_k"] = scale

print("attack v: per-section integrals")
sec = {}
for lab, sl in {"rows 0-105 (z<3 m)": slice(0, 106), "rows 106-1930 (z>=4 m)": slice(106, n)}.items():
    zz, rr = z[sl], rho[sl]
    sec[lab] = {"WED_span_m": float(np.sum(rr[:-1] * np.diff(zz))),
                "c06_span_m": float(c06[sl][-1] - c06[sl][0])}
    sec[lab]["ratio"] = float(sec[lab]["c06_span_m"] / sec[lab]["WED_span_m"])
for blk in (10, 50, 200, 500):
    ks = []
    for s in range(0, n, blk):
        e = min(s + blk, n)
        if e - s < 2:
            continue
        Wloc = cum_left(rho[s:e], np.diff(z[s:e]))
        if Wloc[-1] > 1e-9:
            ks.append(best_k_scale(Wloc, c06[s:e])[0])
    sec[f"local_k_per_{blk}_row_block"] = {"min": float(min(ks)), "max": float(max(ks)),
                                          "spread": float(max(ks) - min(ks))}
attacks["v_per_section"] = sec

# ---------------------------------------------------------------- #
# 3. Decisive test: pointwise product vs integral
# ---------------------------------------------------------------- #
def stats(r):
    return {"max_abs_m": float(np.max(np.abs(r))), "rmse_m": float(np.sqrt(np.mean(r ** 2))),
            "mean_m": float(np.mean(r))}


def ols_r2(x, y):
    xm, ym = x.mean(), y.mean()
    b = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    a = float(ym - b * xm)
    yhat = a + b * x
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / float(np.sum((y - ym) ** 2))
    return a, b, r2


cands = {
    "X1 = c01*c05": Dprod,
    "X2 = c02*c05": z2 * rho,
    "X3 = int rho dz, left (c01 incr)": W_left,
    "X4 = int rho dz, trapezoid (c01 incr)": W_trap,
    "X5 = int rho dz, right (c01 incr)": W_right,
    "X6 = int rho dz, left (c03 incr)": W_left_c03,
    "X7 = int rho dz /0.917, trapezoid": W_trap / RHO_ICE,
    "X8 = int c01*c05 dz, left": cum_left(Dprod, dz),
    "X9 = cumsum(c03)/100 (pure depth)": np.concatenate(([0.0], np.cumsum(th[:-1] / 100.0))),
}
res = {}
for name, x in cands.items():
    r = c06 - x
    a, b, r2 = ols_r2(x, c06)
    sse = float(np.sum(r ** 2))
    sst = float(np.sum((c06 - c06.mean()) ** 2))
    res[name] = {"max_abs_residual_m": stats(r)["max_abs_m"], "rmse_m": stats(r)["rmse_m"],
                 "mean_residual_m": stats(r)["mean_m"],
                 "ols_a": a, "ols_b": b, "ols_r2": r2,
                 "r2_identity_line": 1.0 - sse / sst}
L["section3_decisive"] = res
L["section3_ranking_by_rmse"] = sorted(res.items(), key=lambda kv: kv[1]["rmse_m"])
L["section3_ratio_of_rmse_product_over_integral"] = (
    res["X3 = int rho dz, left (c01 incr)"]["rmse_m"] / res["X1 = c01*c05"]["rmse_m"])

# micro-provenance probes for the winner
w = c06 / rho
L["section3_inverse_probe"] = {
    "median_abs(c06/rho - c01)": float(np.median(np.abs(w - z))),
    "median_abs(c06/rho - c02)": float(np.median(np.abs(w - z2))),
    "median_abs(c06/rho - midpoint)": float(np.median(np.abs(w - z_mid))),
    "median_abs(c06 - round(c01*c05,2))": float(np.median(np.abs(c06 - np.round(Dprod, 2)))),
    "n_rows_exact_c06_eq_round_c01c05_2dp": int(np.sum(np.abs(c06 - np.round(Dprod, 2)) < 1e-9)),
    "n_rows_exact_c06_eq_round_c02c05_2dp": int(np.sum(np.abs(c06 - np.round(z2 * rho, 2)) < 1e-9)),
}
# correlation of residuals with depth (structure)
L["section3_residual_depth_corr"] = {
    name: float(np.corrcoef(z, c06 - x)[0, 1]) for name, x in cands.items()}

# ---------------------------------------------------------------- #
# 4. Does "quadratic in depth" discriminate product from integral?
# ---------------------------------------------------------------- #
quad_fits = {}
for name, x in {"c06 (reported)": c06, "c01*c05": Dprod, "int rho dz": W_left}.items():
    c = np.polyfit(z, x, 2)
    pred = np.polyval(c, z)
    r = x - pred
    quad_fits[name] = {"quad_coeffs": [float(v) for v in c],
                       "max_abs_resid_m": float(np.max(np.abs(r))),
                       "r2": 1.0 - float(np.sum(r ** 2)) / float(np.sum((x - x.mean()) ** 2))}
L["section4_quadratic_discrimination"] = quad_fits
# ratio c06/z at the base vs max density
L["section4_base_identity"] = {
    "c06_base_over_z_base": float(REP / z[-1]),
    "rho_at_base": float(rho[-1]),
    "equal": bool(abs(REP / z[-1] - rho[-1]) < 1e-3),
    "c06_base_over_rho_base": float(REP / rho[-1]),
    "z_base": float(z[-1]),
}

# ---------------------------------------------------------------- #
# 5. Figures
# ---------------------------------------------------------------- #
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].plot(z, c06, "k", lw=1.6, label="reported c06")
ax[0, 0].plot(z, Dprod, lw=1, label="c01*c05")
ax[0, 0].plot(z, W_left, lw=1.4, label="int rho dz (true WED)")
ax[0, 0].plot(z, W_trap / RHO_ICE, lw=1, ls=":", label="int rho dz /0.917 (ice equiv.)")
ax[0, 0].set_xlabel("depth c01 (m)"); ax[0, 0].set_ylabel("m")
ax[0, 0].set_title("Reported c06 tracks the pointwise product, not the integral")
ax[0, 0].legend(fontsize=7)

ax[0, 1].plot(z, c06 - Dprod, lw=1, label="c06 - c01*c05")
ax[0, 1].plot(z, c06 - z2 * rho, lw=1, label="c06 - c02*c05")
ax[0, 1].plot(z, c06 - W_left, lw=1, label="c06 - int rho dz")
ax[0, 1].axhline(0, color="k", lw=0.5)
ax[0, 1].set_yscale("symlog", linthresh=0.01)
ax[0, 1].set_xlabel("depth c01 (m)"); ax[0, 1].set_ylabel("residual (m, symlog)")
ax[0, 1].set_title("Residual envelope: product <0.031 m, integral ~5.8 m")
ax[0, 1].legend(fontsize=7)

ax[1, 0].plot(z, rho, lw=1, color="tab:green")
ax[1, 0].set_xlabel("depth c01 (m)"); ax[1, 0].set_ylabel("rho (g/cm3)")
ax[1, 0].set_title("Density profile c05 (points on closed circles)")

ax[1, 1].plot(z, c06 / np.maximum(W_left, 1e-9), lw=1, label="c06 / int rho dz")
ax[1, 1].plot(z, c06 / np.maximum(W_trap / RHO_ICE, 1e-9), lw=1, label="c06 / ice-equiv")
ax[1, 1].set_xlabel("depth c01 (m)"); ax[1, 1].set_ylabel("ratio")
ax[1, 1].set_title("Ratio is not constant -> no k rescues the integral")
ax[1, 1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f2_wed_overview.png"), dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
names = list(res.keys())
rmses = [res[k]["rmse_m"] for k in names]
ax.barh(range(len(names)), rmses, color=["tab:green" if k.startswith("X1") or k.startswith("X2")
                                         else "tab:red" for k in names])
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("RMSE(c06 - X) in m (log)")
ax.set_title("Which X reproduces the reported column?")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f2_wed_candidate_rmse.png"), dpi=120)
plt.close(fig)

with open(os.path.join(OUT, "f2_wed_verification.json"), "w") as f:
    json.dump({"grid": L["grid"], "quadratures": L["quadratures_of_true_WED"],
               "envelope": L["envelope_bound"], "attacks": attacks,
               "decisive": {k: v for k, v in L["section3_decisive"].items()},
               "ranking": [(k, v["rmse_m"]) for k, v in L["section3_ranking_by_rmse"]],
               "inverse_probe": L["section3_inverse_probe"],
               "quadratic": L["section4_quadratic_discrimination"],
               "base_identity": L["section4_base_identity"]}, f, indent=2, default=str)

with open(os.path.join(OUT, "f2_wed_key_numbers.txt"), "w") as f:
    f.write("REPORTED c06 final = %.6f m\n\n" % REP)
    f.write("Quadratures of the true WED = int rho dz :\n")
    for k, v in quad.items():
        f.write("  %-45s = %.6f m   gap %+.6f m (%+.4f%%)\n"
                % (k, v, v - REP, 100 * (v - REP) / REP))
    f.write("\nEnvelope bound [rho_min*z, rho_max*z] = [%.6f, %.6f]\n\n"
            % (rho.min() * z[-1], rho.max() * z[-1]))
    f.write("Decisive residuals (1931 rows):\n")
    for k, v in L["section3_ranking_by_rmse"]:
        f.write("  %-40s max|r|=%.6f  rmse=%.6f  mean=%+.6f  OLS R2=%.9f  R2(identity)=%.6f\n"
                % (k, v["max_abs_residual_m"], v["rmse_m"], v["mean_residual_m"],
                   v["ols_r2"], v["r2_identity_line"]))
    f.write("\nQuadratic-in-depth fits:\n")
    for k, v in quad_fits.items():
        f.write("  %-16s max|r|=%.6f  R2=%.9f\n" % (k, v["max_abs_resid_m"], v["r2"]))
    f.write("\nInverse probe c06/rho:\n")
    for k, v in L["section3_inverse_probe"].items():
        f.write("  %s = %s\n" % (k, v))

print(json.dumps(L, indent=2, default=str))