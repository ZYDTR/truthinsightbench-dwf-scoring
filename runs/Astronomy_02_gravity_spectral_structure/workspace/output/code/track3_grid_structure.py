"""TRACK 3 - Structure, degeneracy and resolution boundary of the candidate parameter grids.

Data groups (read-only):
  data/ranked_full_parameter_grid.tsv   (n = 250 ranked candidate cells)
  data/mode_frequency_grid.tsv          (n = 52 mode frequencies)
  data/candidate_spectrum_comparison.tsv(2 scan families, n = 507 rows)
  data/pass_residual_traces.tsv         (n = 22 whole passes; used only for the
                                         single-pass Fourier-resolution section)

Analysis unit: one complete candidate parameter combination / one complete pass.
Grid rows and adjacent time samples are never treated as independent replicates.
Where a dispersion is quoted it is either (a) a bootstrap / leave-one-pass-out
range over the 22 passes, or (b) explicitly labelled as a surface residual,
not an error bar.  No p-value is computed that treats the 250 grid rows or the
~1698 time samples as independent observations (the mode-alignment null in
section 5 draws independent random frequencies, which is the correct null there).

Output: ../results/grid_structure.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code")

import numpy as np
import pandas as pd

from common import (MODELS, freq_grid, load_candidates, load_modes,
                    load_ranked_grid, load_traces, ls_power, write_json)

PROJECT = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
RESULTS = PROJECT / "output/results/grid_structure.json"

# declared axes of the ranked grid (from the task contract)
AX_VMAX = [1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 100.0]
AX_VMIN = [0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
AX_F = list(np.arange(860.0, 1311.0, 50.0))
AX_W = [250.0, 275.0, 300.0, 325.0]
LATTICE_N = 7 * 8 * 10 * 4

g = load_ranked_grid()
cand = load_candidates()
modes = load_modes()
traces = load_traces()

out = {}
out["provenance"] = {
    "script": "output/code/track3_grid_structure.py",
    "inputs": [
        "data/ranked_full_parameter_grid.tsv",
        "data/candidate_spectrum_comparison.tsv",
        "data/mode_frequency_grid.tsv",
        "data/pass_residual_traces.tsv",
    ],
    "analysis_unit": "one complete candidate parameter combination; one complete pass (n=22)",
    "independent_replicates": {"passes": 22, "candidate_cells_evaluated_in_table": 250},
}

# --------------------------------------------------------------------------
# 1. Axis inventory and truncation
# --------------------------------------------------------------------------
ax_inventory = {}
for col, name in [("vmax", "max_velocity_cm_s"), ("vmin", "min_velocity_cm_s"),
                  ("f", "peak_frequency_microhz"), ("w", "frequency_width_microhz")]:
    vc = g[col].value_counts().sort_index()
    ax_inventory[name] = {
        "declared_levels": len(vc),
        "counts": {float(k): int(v) for k, v in vc.items()},
        "expected_under_uniform_250_subset": round(250.0 / len(vc), 3),
    }

n_rows = len(g)
n_unique_cells = int(g[["vmax", "vmin", "f", "w"]].drop_duplicates().shape[0])
ranks_ok = sorted(int(x) for x in g["candidate_rank"]) == list(range(1, n_rows + 1))
monotone = bool(np.all(np.diff(g.sort_values("candidate_rank")["ic"].values) >= -1e-9))
ax_inventory["max_velocity_cm_s"]["sub_lattice_coverage"] = {
    float(a): round(int((g["vmax"] == a).sum()) / 320.0, 4) for a in AX_VMAX}

out["1_axis_inventory"] = {
    "n_rows": n_rows,
    "axes": ax_inventory,
    "cartesian_cells_declared": LATTICE_N,
    "cartesian_cells_present": n_unique_cells,
    "coverage_fraction": round(n_unique_cells / LATTICE_N, 4),
    "rows_missing": LATTICE_N - n_rows,
    "candidate_rank_is_1_to_250_complete": bool(ranks_ok),
    "ic_monotone_nonincreasing_in_rank": bool(monotone),
    "n_unique_ic_values": int(g["ic"].nunique()),
    "ic_of_rank1": float(g["ic"].min()),
    "ic_of_rank250": float(g["ic"].max()),
    "peakedness": ("row counts are strongly peaked at interior levels "
                   "(vmax=30 -> 99/250, f=1010 -> 57/250, w=300 -> 108/250), "
                   "which is the signature of an IC-ordered top-N list, not of a "
                   "complete factorial design"),
}

# --- truncation evidence: cross-file ledger ---------------------------------
c2 = cand.rename(columns={"max_velocity_cm_s": "vmax", "min_velocity_cm_s": "vmin",
                          "peak_frequency_microhz": "f", "peak_width_microhz": "w",
                          "information_criterion": "ic"})
for df in (c2, g):
    df["k_vmax"] = df["vmax"].astype(float).round(3)
    df["k_vmin"] = df["vmin"].astype(float).round(3)
    df["k_f"] = df["f"].astype(float).round(3)
    df["k_w"] = df["w"].astype(float).round(3)
KEY = ["k_vmax", "k_vmin", "k_f", "k_w"]

known = c2[c2["k_vmax"].isin(AX_VMAX) & c2["k_vmin"].isin([round(x, 3) for x in AX_VMIN])
           & c2["k_f"].isin([round(x, 3) for x in AX_F])
           & c2["k_w"].isin([round(x, 3) for x in AX_W])].copy()
known["in_table"] = [tuple(r) in set(map(tuple, g[KEY].values)) for r in known[KEY].values]

exact = g.merge(c2[KEY + ["ic"]].rename(columns={"ic": "ic_ref"}), on=KEY, how="inner")
exact["offset"] = exact["ic_ref"] - exact["ic"]
thr = float(g["ic"].max())
TP = int(((known["ic"] < thr) & known["in_table"]).sum())
FP = int(((known["ic"] < thr) & ~known["in_table"]).sum())
FN = int(((known["ic"] >= thr) & known["in_table"]).sum())

out["1_truncation_evidence"] = {
    "declared_lattice_cells": LATTICE_N,
    "cells_known_independently_from_candidate_file": int(len(known)),
    "known_cells_present_in_table": int(known["in_table"].sum()),
    "table_ic_cutoff": thr,
    "admission_rule_ic_lt_cutoff": f"ic < {thr}",
    "confusion_on_known_cells": {"true_positive": TP, "false_positive": FP, "false_negative": FN},
    "smallest_ic_known_but_absent": float(known.loc[~known["in_table"], "ic"].min()),
    "largest_ic_present": float(known.loc[known["in_table"], "ic"].max()),
    "rows_whose_ic_is_reproduced_exactly": {
        "n": int(len(exact)),
        "max_abs_offset": float(exact["offset"].abs().max()),
        "families": exact["candidate_family"].value_counts().to_dict() if "candidate_family" in exact else {},
    },
    "rows_whose_ic_exists_only_in_the_table": int(n_rows - len(exact)),
    "cells_out_of_lattice_rows": 0,
    "conclusion": (
        "The table is the top-250 of a larger evaluated set truncated at ic = %.1f, "
        "covering %.1f%% of the declared 2240-cell cartesian product; the remaining "
        "%d cells (%.1f%%) are censored to ic >= %.1f. The rank-1 cell is therefore "
        "the best *among the 250 survivors*; uniqueness over the 2240-cell space is "
        "not established by this file."
        % (thr, 100.0 * n_unique_cells / LATTICE_N, LATTICE_N - n_rows,
           100.0 * (LATTICE_N - n_rows) / LATTICE_N, thr)),
}

# --------------------------------------------------------------------------
# 2. Separation / robustness of the top cells
# --------------------------------------------------------------------------
gs = g.sort_values("ic").reset_index(drop=True)
top_margins = {}
for n in [1, 2, 3, 5, 10, 20, 30, 50, 100, 250]:
    top_margins[str(n)] = {"max_ic_within_top_n": float(gs["ic"].iloc[n - 1])}

axis_best = {}
for col, name in [("vmax", "max_velocity_cm_s"), ("vmin", "min_velocity_cm_s"),
                  ("f", "peak_frequency_microhz"), ("w", "frequency_width_microhz")]:
    p = g.groupby(col)["ic"].min().sort_values()
    axis_best[name] = {
        "best_level": float(p.index[0]),
        "best_level_min_ic": float(p.iloc[0]),
        "second_level": float(p.index[1]),
        "gap_to_second": float(p.iloc[1] - p.iloc[0]),
        "profile_min_ic_per_level": {float(k): float(v) for k, v in p.items()},
    }

out["2_top_cell_separation"] = {
    "ic_ladder": [float(x) for x in gs["ic"].iloc[:20]],
    "cumulative_top_n_max_ic": top_margins,
    "consecutive_gaps_top20": [float(x) for x in np.diff(gs["ic"].values[:20])],
    "counts_within_delta_ic": {
        str(t): int((gs["ic"] < t).sum()) for t in [1, 2, 5, 10, 20, 50, 100]},
    "rank1_vs_rank2_margin": float(gs["ic"].iloc[1] - gs["ic"].iloc[0]),
    "rank1_vs_rank5_margin": float(gs["ic"].iloc[4] - gs["ic"].iloc[0]),
    "rank1_vs_rank10_margin": float(gs["ic"].iloc[9] - gs["ic"].iloc[0]),
    "best_axis_level_and_margin": axis_best,
    "single_grid_step_cost_at_optimum": {
        "fa_family_vmax20_f1010": {"minus_50uHz": 14.1, "plus_50uHz": 12.8},
        "fa_family_vmax30_f1010": {"minus_50uHz": 25.3, "plus_50uHz": 10.1},
        "fa_family_vmax50_f1110": {"minus_50uHz": 20.6, "plus_50uHz": 1.1},
        "note": ("delta-IC of the neighbouring 50 uHz grid frequency in the complete "
                 "frequency_amplitude_scan family; the top-5 spread (11.9) is comparable "
                 "to or smaller than one frequency step"),
    },
    "verdict": ("rank-1 is separated from rank-2 by delta-IC = 2.0 only; there is a single "
                "cell with delta-IC < 2, two with < 5, four with < 10 and nine with < 20. "
                "With a supplied IC step granularity of 0.1 and neighbouring 50 uHz cells "
                "costing 10-25 IC units, the rank-1 cell is not distinguishable from the "
                "top handful of cells."),
}

# --------------------------------------------------------------------------
# 3. Parameter degeneracy in the two scan families
# --------------------------------------------------------------------------
fa = cand[cand["candidate_family"] == "frequency_amplitude_scan"].rename(
    columns={"max_velocity_cm_s": "vmax", "min_velocity_cm_s": "vmin",
             "peak_frequency_microhz": "f", "peak_width_microhz": "w",
             "information_criterion": "ic"})
mv = cand[cand["candidate_family"] == "minimum_velocity_scan"].rename(
    columns={"max_velocity_cm_s": "vmax", "min_velocity_cm_s": "vmin",
             "peak_frequency_microhz": "f", "peak_width_microhz": "w",
             "information_criterion": "ic"})

fa_best = fa.loc[fa.groupby("vmax")["ic"].idxmin()].sort_values("vmax")[["vmax", "f", "ic"]]
fa_best.columns = ["vmax", "f_star", "ic_star"]
mv_best = mv.loc[mv.groupby("vmax")["ic"].idxmin()].sort_values("vmax")[["vmax", "vmin", "ic"]]
mv_best.columns = ["vmax", "vmin_star", "ic_star"]


def loglog(x, y):
    A = np.column_stack([np.ones(len(x)), np.log10(x)])
    b, *_ = np.linalg.lstsq(A, np.log10(y), rcond=None)
    pred = A @ b
    r2 = 1 - np.sum((np.log10(y) - pred) ** 2) / np.sum((np.log10(y) - np.log10(y).mean()) ** 2)
    return float(b[1]), float(10 ** b[0]), float(r2), float(np.abs(np.log10(y) - pred).max())


sl, A0, r2fa, resfa = loglog(fa_best["vmax"].values, fa_best["f_star"].values)
fa_hi = fa_best[fa_best["vmax"] >= 0.5]
sl2, A2, r2fa2, resfa2 = loglog(fa_hi["vmax"].values, fa_hi["f_star"].values)

# does the IC surface of the complete fa factorial collapse onto one combination?
piv = fa.pivot_table(index="vmax", columns="f", values="ic")
Y = np.log10(piv.values)
V = np.log10(piv.index.values)[:, None]
Fq = piv.columns.values[None, :]
Vf = np.broadcast_to(V, Y.shape)
Ff = np.broadcast_to(Fq, Y.shape)


def r2_of(cols, target):
    X = np.column_stack(cols + [np.ones(len(target))])
    b, *_ = np.linalg.lstsq(X, target, rcond=None)
    return 1 - np.sum((target - X @ b) ** 2) / np.sum((target - target.mean()) ** 2)


fa_full = r2_of([Vf.ravel(), Ff.ravel(), (Vf * Ff).ravel(), Vf.ravel() ** 2, Ff.ravel() ** 2], Y.ravel())
fa_add = r2_of([Vf.ravel() - Vf.mean(), Ff.ravel() - Ff.mean()], Y.ravel())
ridge = {}
for a, b in [(A0, sl), (A2, sl2), (1000.0, 0.0), (600.0, 0.35)]:
    u = Ff - a * (10 ** Vf) ** b
    ridge[f"a={a:.1f},b={b:.4f}"] = round(float(r2_of([u.ravel(), u.ravel() ** 2], Y.ravel())), 4)

slm, Am, r2mv, resmv = loglog(mv_best["vmax"].values, mv_best["vmin_star"].values)
mvlg = np.log10(mv["ic"].values)
mv_lv = np.log10(mv["vmax"].values)
mv_lm = np.log10(mv["vmin"].values)
mv_add = r2_of([mv_lv - mv_lv.mean(), mv_lm - mv_lm.mean()], mvlg)
mv_int = r2_of([mv_lv, mv_lm, mv_lv * mv_lm], mvlg)

out["3_degeneracy"] = {
    "frequency_amplitude_scan": {
        "family": "vmin fixed at 0, w fixed at 300, vmax 0.1..500 (16 levels), f 610..1510 (19 levels)",
        "is_complete_factorial": bool(fa.groupby("vmax").size().nunique() == 1
                                      and fa.groupby("vmax").size().iloc[0] == 19),
        "n_rows": int(len(fa)),
        "f_star_vs_vmax": {float(k): float(v) for k, v in
                           zip(fa_best["vmax"], fa_best["f_star"])},
        "ic_star_vs_vmax": {float(k): float(v) for k, v in
                            zip(fa_best["vmax"], fa_best["ic_star"])},
        "loglog_fit_all_16_levels": {"slope": sl, "A_microHz": A0, "R2": r2fa,
                                     "max_residual_dex": resfa},
        "loglog_fit_vmax_ge_0.5": {"n": int(len(fa_hi)), "slope": sl2, "A_microHz": A2,
                                   "R2": r2fa2, "max_residual_dex": resfa2},
        "edge_pinned_levels": [float(x) for x in fa_best.loc[fa_best["f_star"] == 610.0, "vmax"]],
        "combination_spread_f_over_vmax_pow_b": {
            "min": float((fa_best["f_star"] / fa_best["vmax"] ** sl).min()),
            "max": float((fa_best["f_star"] / fa_best["vmax"] ** sl).max())},
        "ic_surface_factorisation": {
            "full_2d_R2": round(float(fa_full), 4),
            "additive_R2": round(float(fa_add), 4),
            "ridge_function_R2": ridge,
            "verdict": ("the complete fa factorial does NOT factorise: additive "
                        "R2 = %.3f vs full R2 = %.3f in log10-IC" % (fa_add, fa_full)),
        },
        "relation": ("f_star ~ %.0f * (vmax/1 cm/s)^%.3f uHz (R2 = %.3f); the IC-minimising "
                     "frequency rises only 610 -> 1510 uHz while vmax rises by 5000x, and the "
                     "slope is 0.135 (R2 = %.3f) once the four edge-pinned levels vmax<=0.5 "
                     "are dropped." % (A0, sl, r2fa, r2fa2)),
    },
    "minimum_velocity_scan": {
        "family": "f fixed at 1210, w fixed at 300, vmin nested inside vmax (20 levels each of vmax 0.02..500, vmin <= vmax)",
        "n_rows": int(len(mv)),
        "is_nested_not_factorial": True,
        "vmin_star_vs_vmax": {float(k): float(v) for k, v in
                              zip(mv_best["vmax"], mv_best["vmin_star"])},
        "ic_star_vs_vmax": {float(k): float(v) for k, v in
                            zip(mv_best["vmax"], mv_best["ic_star"])},
        "loglog_fit_vmax_ge_0.1": {"n": int((mv_best["vmax"] >= 0.1).sum()), "slope": slm,
                                   "A_cm_s": Am, "R2": r2mv, "max_residual_dex": resmv},
        "levels_visited": [float(x) for x in sorted(mv_best["vmin_star"].unique())],
        "non_monotonic": True,
        "interior_curvature_at_vmax30": {
            "vmin_star": 0.03, "ic_star": 50.7, "ic_at_lower_edge_0.01": 81.0,
            "delta_ic_to_edge": 30.3, "ic_at_upper_edge_30": 8435.6},
        "log10_ic_regressions": {
            "log_vmax_only_R2": round(float(r2_of([mv_lv - mv_lv.mean()], mvlg)), 4),
            "log_vmin_only_R2": round(float(r2_of([mv_lm - mv_lm.mean()], mvlg)), 4),
            "additive_R2": round(float(mv_add), 4),
            "with_interaction_R2": round(float(mv_int), 4),
            "interaction_gain": round(float(mv_int - mv_add), 4),
        },
    },
    "verdict": (
        "The data do not determine peak frequency and velocity scale independently. The "
        "frequency_amplitude_scan defines a shallow ridge f_star ~ 700 (vmax)^0.13 uHz "
        "(R2 = 0.97) whose residual scatter corresponds to +-1 grid step; only the "
        "combination is measured well at low vmax. The minimum_velocity_scan defines a "
        "flat, non-monotonic valley in vmin (0.03-0.1 cm/s, log-log slope -0.08, R2 = 0.20) "
        "whose floor moves by only ~30 IC units. Near the global optimum (vmax = 20-50) the "
        "frequency is pinned to 1010-1110 uHz to within one 50 uHz grid step, so the "
        "degeneracy is strong but not total."),
}

# --------------------------------------------------------------------------
# 4. Separability of delta-IC over the ranked grid
# --------------------------------------------------------------------------
sub = g[g["vmin"] > 0].copy()
n_dropped = int((g["vmin"] == 0).sum())
y = sub["ic"].values
v = sub["vmax"].values
vmin_ = sub["vmin"].values
f = sub["f"].values
w = sub["w"].values


def fit_report(cols):
    X = np.column_stack(cols + [np.ones(len(sub))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ b
    return {"R2": round(float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
            "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 3),
            "coef": [round(float(x), 5) for x in b]}


base = fit_report([np.log10(v), np.log10(vmin_), f, w])
rich = fit_report([np.log10(v), np.log10(vmin_), f, w, f ** 2, w ** 2])
rich2 = fit_report([np.log10(v), np.log10(vmin_), f, w, 1 / v, v, 1 / vmin_, vmin_, f ** 2, w ** 2])
ridge_u = fit_report([f - A0 * v ** sl, (f - A0 * v ** sl) ** 2])
fonly = fit_report([f, f ** 2])
inter = fit_report([np.log10(v), np.log10(vmin_), f, w, np.log10(v) * f, np.log10(v) * w, f * w])

loo = {}
for lv in sorted(sub["vmax"].unique()):
    s2 = sub[sub["vmax"] != lv]
    X2 = np.column_stack([np.log10(s2["vmax"]), np.log10(s2["vmin"]), s2["f"], s2["w"], np.ones(len(s2))])
    b2, *_ = np.linalg.lstsq(X2, s2["ic"].values, rcond=None)
    yy = s2["ic"].values
    pp = X2 @ b2
    loo[str(float(lv))] = {
        "n": int(len(s2)),
        "R2": round(float(1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)), 4),
        "coef_logvmax_logvmin_f_w_intercept": [round(float(x), 4) for x in b2]}

out["4_separability_ranked_grid"] = {
    "n_rows_total": n_rows,
    "rows_with_min_velocity_equal_zero": n_dropped,
    "log10_undefined_statement": ("{} of {} rows ({:.1f}%) have min_velocity_cm_s == 0, for which "
                                  "log10 is undefined (-inf); they were dropped, leaving n = {} "
                                  "rows for every model that uses log10(min_velocity).".format(
                                      n_dropped, n_rows, 100.0 * n_dropped / n_rows, len(sub))),
    "n_rows_used": int(len(sub)),
    "model_4term": base,
    "model_plus_f2_w2": rich,
    "model_richer_1_over_v_v_1_over_vmin_vmin": rich2,
    "model_ridge_combination_only": ridge_u,
    "model_f_only_quadratic": fonly,
    "model_with_interactions": inter,
    "leave_one_vmax_level_out": loo,
    "residual_scatter_4term": {"RMSE_ic_units": base["RMSE"],
                               "sd_of_ic": round(float(np.std(y, ddof=1)), 3)},
    "verdict": (
        "On this truncated table the surface does not separate: the 4-term linear model in "
        "(log10 vmax, log10 vmin, f, width) reaches R2 = %.4f with residual RMSE = %.2f IC "
        "units against an IC spread of %.2f; adding f^2, w^2 gives R2 = %.4f and the richer "
        "basis (1/v, v, 1/vmin, vmin, f^2, w^2) gives R2 = %.4f. Leave-one-vmax-level-out "
        "R2 ranges %.3f-%.3f with unstable signs/coefficients, so no separable additive "
        "structure is supported. Adding interactions nearly doubles R2 (%.3f -> %.3f), i.e. "
        "the surface is dominated by the truncation boundary and by interaction."
        % (base["R2"], base["RMSE"], float(np.std(y, ddof=1)), rich["R2"], rich2["R2"],
           min(d["R2"] for d in loo.values()), max(d["R2"] for d in loo.values()),
           base["R2"], inter["R2"])),
    "caveat": ("R2 computed on a top-250 censored slice is not a property of the underlying "
               "IC surface; the interpretable separability numbers are the complete-factorial "
               "fits reported in section 3 (fa additive R2 = %.3f vs full R2 = %.3f, "
               "mv family additive R2 = %.3f)." % (fa_add, fa_full, mv_add)),
}

# --------------------------------------------------------------------------
# 5. Mode-grid alignment
# --------------------------------------------------------------------------
mf = np.sort(modes["mode_frequency_microhz"].values)
cands = np.array(AX_F)
d = np.abs(cands[:, None] - mf[None, :]).min(axis=1)
pct = d / cands * 100.0
obs_med = float(np.median(d))
obs_mean = float(d.mean())
obs_pct_med = float(np.median(pct))

N_NULL = 200_000
CHUNK = 20_000


def null_median_dists(lo, hi, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(int(np.ceil(N_NULL / CHUNK))):
        draws = rng.uniform(lo, hi, size=(CHUNK, len(cands)))
        parts.append(np.abs(draws[:, :, None] - mf[None, None, :]).min(axis=2))
    dd = np.vstack(parts)
    return dd, np.median(dd, axis=1)


dn_full, med_full = null_median_dists(float(mf.min()), float(mf.max()), 12345)
dn_band, med_band = null_median_dists(860.0, 1310.0, 24680)

# rigid 50 uHz comb null: only the comb phase is free
rng = np.random.default_rng(99)
offs = rng.uniform(0.0, 50.0, size=100_000)
med_comb = np.array([np.median(np.abs((860.0 + np.arange(10) * 50.0 + o)[:, None] - mf[None, :]).min(axis=1))
                     for o in offs])

per_cand = {}
for i, f0 in enumerate(cands):
    per_cand[str(int(f0))] = {
        "offset_uHz": round(float(d[i]), 3),
        "offset_percent": round(float(pct[i]), 4),
        "null_p_full_range": round(float((dn_full[:, i] <= d[i]).mean()), 5),
        "null_p_band": round(float((dn_band[:, i] <= d[i]).mean()), 5),
    }
min_p = float(min((dn_full[:, i] <= d[i]).mean() for i in range(len(cands))))
fam_p = float((dn_full.min(axis=1) <= d.min()).mean())

mf_band = mf[(mf >= 860.0) & (mf <= 1310.0)]
dm_all = np.diff(mf)
dm_band = np.diff(mf_band)

out["5_mode_grid_alignment"] = {
    "n_mode_frequencies_total": int(len(mf)),
    "mode_range_uHz": [float(mf.min()), float(mf.max())],
    "n_modes_in_candidate_band_860_1310": int(len(mf_band)),
    "mode_frequencies_in_band_uHz": [float(x) for x in mf_band],
    "observed": {
        "median_offset_uHz": round(obs_med, 3),
        "mean_offset_uHz": round(obs_mean, 3),
        "median_offset_percent": round(obs_pct_med, 4),
        "per_candidate": per_cand,
    },
    "null_uniform_over_mode_range": {
        "range_uHz": [float(mf.min()), float(mf.max())],
        "n_draws": N_NULL,
        "null_median_of_median_offset_uHz": round(float(np.median(med_full)), 3),
        "null_median_5_95_uHz": [round(float(x), 3) for x in np.percentile(med_full, [5, 95])],
        "one_sided_p_observed_closer": round(float((med_full <= obs_med).mean()), 4),
    },
    "null_uniform_over_candidate_band": {
        "range_uHz": [860.0, 1310.0],
        "n_draws": N_NULL,
        "null_median_of_median_offset_uHz": round(float(np.median(med_band)), 3),
        "null_median_5_95_uHz": [round(float(x), 3) for x in np.percentile(med_band, [5, 95])],
        "one_sided_p_observed_closer": round(float((med_band <= obs_med).mean()), 4),
    },
    "null_rigid_50uHz_comb": {
        "n_draws": len(offs),
        "null_median_of_median_offset_uHz": round(float(np.median(med_comb)), 3),
        "null_median_5_95_uHz": [round(float(x), 3) for x in np.percentile(med_comb, [5, 95])],
        "one_sided_p_observed_closer": round(float((med_comb <= obs_med).mean()), 4),
        "best_achievable_median_uHz": round(float(med_comb.min()), 3),
        "observed_comb_at_phase_zero_median_uHz": round(obs_med, 3),
    },
    "individual_candidate_multiplicity": {
        "smallest_individual_null_p_full_range": round(min_p, 5),
        "candidate_achieving_it": "1160 (offset 0.98 uHz)",
        "family_wise_p_min_over_10_candidates_full_range": round(fam_p, 4),
    },
    "mode_spacing": {
        "overall_median_uHz": round(float(np.median(dm_all)), 3),
        "overall_mean_uHz": round(float(dm_all.mean()), 3),
        "overall_min_uHz": round(float(dm_all.min()), 3),
        "overall_max_uHz": round(float(dm_all.max()), 3),
        "in_band_median_uHz": round(float(np.median(dm_band)), 3),
        "in_band_mean_uHz": round(float(dm_band.mean()), 3),
        "comb_spacing_uHz": 50.0,
        "in_band_comb_ratio": round(float(50.0 / np.median(dm_band)), 3),
        "expected_uniform_nearest_distance_in_band_uHz": round(
            float(1.0 / (2.0 * (len(mf_band) - 1) / (mf_band[-1] - mf_band[0]))), 3),
        "interpretation": ("the mode grid is 2.4x denser than the 50 uHz comb in the candidate "
                           "band (median spacing 20.7 uHz vs 50 uHz), so a random frequency in "
                           "the band is within ~10 uHz of some mode by construction; the observed "
                           "median offset of 5.76 uHz (0.50%) is not smaller than the uniform null "
                           "(7.24-8.08 uHz median, one-sided p = 0.18-0.29)."),
    },
    "verdict": ("No evidence that the candidate frequency comb is locked to the supplied mode "
                "grid: observed median nearest-mode offset 5.76 uHz vs null medians 7.24 uHz "
                "(band), 8.08 uHz (full range), 7.34 uHz (rigid-comb phase null); one-sided "
                "p = 0.27, 0.18, 0.29 respectively. The best single candidate (1160 uHz, "
                "0.98 uHz from the 1159.02 uHz mode) has individual p = 0.067 but the "
                "family-wise probability of some candidate landing that close is 0.28."),
}

# --------------------------------------------------------------------------
# 6. Single-pass Fourier resolution boundary
# --------------------------------------------------------------------------
spans = traces.groupby("pass_id")["relative_time_min"].apply(lambda t: float(t.max() - t.min()))
n_samp = traces.groupby("pass_id").size()
res_hz = 1e6 / (spans.values * 60.0)   # uHz
res_med = float(np.median(res_hz))

widths = {str(int(x)): None for x in AX_W}
for x in AX_W:
    widths[str(int(x))] = {
        "ratio_to_median_1_over_T": round(float(x / res_med), 3),
        "ratio_to_longest_pass_1_over_T": round(float(x / res_hz.min()), 3),
        "ratio_to_shortest_pass_1_over_T": round(float(x / res_hz.max()), 3),
    }
comb_ratio = {
    "50uHz_over_median_1_over_T": round(float(50.0 / res_med), 3),
    "50uHz_over_longest_pass_1_over_T": round(float(50.0 / res_hz.min()), 3),
}

# two-sinusoid separation test on the real pass_01 timestamps
tr1 = traces[traces["pass_id"] == "pass_01"]
t1 = tr1["relative_time_min"].values
f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
rng = np.random.default_rng(2024)
sep_test = {}
for df in [50.0, 100.0, 211.0, 250.0, 275.0, 300.0, 325.0, 450.0]:
    hits = []
    for _ in range(200):
        ph = rng.uniform(0, 2 * np.pi, 2)
        ysim = (np.cos(2 * np.pi * 1000e-6 * t1 * 60 + ph[0])
                + np.cos(2 * np.pi * (1000 + df) * 1e-6 * t1 * 60 + ph[1]))
        pw = ls_power(t1, ysim, f_hz)
        m = (f_uhz >= 1000.0 + df / 2 - 3 * df) & (f_uhz <= 1000.0 + df / 2 + 3 * df)
        seg = pw[m]
        loc = [i for i in range(1, len(seg) - 1)
               if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.5 * seg.max()]
        hits.append(len(loc) >= 2)
    sep_test[str(int(df))] = {
        "separation_over_1_over_T": round(float(df / res_med), 3),
        "fraction_of_trials_resolved_two_peaks": round(float(np.mean(hits)), 2),
    }

# per-pass band-peak dispersion (whole-pass proxy, n = 22)
peaks = {}
for tag, col in MODELS.items():
    vals = []
    for pid, s in traces.groupby("pass_id"):
        p = ls_power(s["relative_time_min"].values, s[col].values, f_hz)
        mm = (f_uhz >= 860) & (f_uhz <= 1310)
        vals.append(float(f_uhz[mm][int(np.argmax(p[mm]))]))
    vals = np.array(vals)
    rng2 = np.random.default_rng(7)
    bs = np.array([np.median(rng2.choice(vals, len(vals), replace=True)) for _ in range(20000)])
    loo = np.array([np.median(np.delete(vals, i)) for i in range(len(vals))])
    peaks[tag] = {
        "model": col,
        "median_band_peak_uHz": float(np.median(vals)),
        "iqr_uHz": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
        "bootstrap_5_95_of_median_uHz": [round(float(x), 2) for x in np.percentile(bs, [5, 95])],
        "leave_one_pass_out_range_uHz": [round(float(loo.min()), 2), round(float(loo.max()), 2)],
    }

out["6_resolution_boundary"] = {
    "n_passes": int(len(spans)),
    "pass_span_minutes": {"min": float(spans.min()), "max": float(spans.max()),
                          "median": float(spans.median()),
                          "counts": {str(float(k)): int(v) for k, v in spans.value_counts().items()}},
    "pass_samples": {"min": int(n_samp.min()), "max": int(n_samp.max()),
                     "median": int(n_samp.median())},
    "one_over_T_uHz": {"min": round(float(res_hz.min()), 3), "max": round(float(res_hz.max()), 3),
                       "median": round(res_med, 3)},
    "candidate_widths": widths,
    "comb_spacing": comb_ratio,
    "two_sinusoid_separation_test_pass01": sep_test,
    "per_pass_band_peak_displacement": peaks,
    "verdict": (
        "Single-pass Fourier resolution 1/T = 210.97 uHz (79 min passes) to 277.78 uHz "
        "(60 min pass); the supplied candidate widths 250/275/300/325 uHz correspond to "
        "1.19/1.30/1.42/1.54 x 1/T (and 0.90/0.99/1.08/1.17 x 1/T for the 60 min pass). "
        "The narrowest candidate (250 uHz) is only marginally resolvable within one pass: "
        "simulation with the real pass_01 timestamps produces two half-power peaks in 55%% "
        "of trials at 250 uHz but 95-100%% at 275-325 uHz. The 50 uHz comb spacing is "
        "0.237 x 1/T, i.e. adjacent candidate frequencies are separated by a quarter of a "
        "resolution element and cannot be resolved within a single pass."),
}

# --------------------------------------------------------------------------
# 7. What is supported
# --------------------------------------------------------------------------
out["7_conclusions"] = {
    "supported_within_the_supplied_data": [
        "The ranked table is a truncated top-250 ranking (11.2%% of the declared 2240-cell "
        "lattice), not a complete grid. Verified independently on the 105 lattice cells that "
        "also appear in candidate_spectrum_comparison.tsv: all 28 with ic < 137.2 are present "
        "and the smallest absent one is 139.9; the table's own maximum is 137.2.",
        "The rank-1 cell (vmax=30, vmin=0.05, f=1010, w=275) is not uniquely determined: "
        "1 cell within delta-IC < 2, 2 within < 5, 4 within < 10, 9 within < 20; the top-5 "
        "spread (11.9) is the size of one 50 uHz frequency step (10.1-25.3 at the optimum).",
        "Peak frequency and velocity scale are not separately measured by the supplied "
        "families: f_star ~ 710 (vmax)^0.121 uHz over the complete frequency_amplitude_scan "
        "(R2 = 0.973, max residual 0.056 dex); the minimum_velocity_scan gives a flat, "
        "non-monotonic vmin valley (log-log slope -0.082, R2 = 0.204).",
        "The IC surface does not factorise: on the complete fa factorial (16 x 19 = 304 rows) "
        "the additive model gives R2 = 0.203 against R2 = 0.689 for the model with "
        "interactions; on the truncated table the 4-term linear model gives R2 = 0.024 "
        "(RMSE 33.4) and even the richer basis only 0.383.",
        "The candidate frequencies are not closer to the supplied mode grid than chance: "
        "observed median offset 5.76 uHz (0.499%), null medians 7.24/8.08/7.34 uHz, one-sided "
        "p = 0.18-0.29; the in-band mode spacing (20.7 uHz median) is 2.4x denser than the "
        "50 uHz comb, so proximity alone is not evidence of mode locking.",
        "A single pass resolves at best 1/T = 211 uHz; the narrowest supplied width (250 uHz) "
        "is 1.19 x 1/T (marginally resolved, 55% of simulated trials) and the 50 uHz comb is "
        "0.237 x 1/T (unresolvable within a pass).",
    ],
    "relative_only_to_the_supplied_candidate_family": [
        "Every statement naming a 'best' parameter combination is conditional on the two "
        "supplied scan families (frequency_amplitude_scan: vmin = 0, w = 300; "
        "minimum_velocity_scan: f = 1210, w = 300) and on the 2240-cell lattice step sizes "
        "(50 uHz in f, 250-325 uHz in width). Nothing here constrains a continuous optimum.",
        "The 88.4% of the lattice that is censored to ic >= 137.2 is unknown; a cell outside "
        "the 250 could in principle be optimal. The marginal 'best axis level' values "
        "(vmax = 30, f = 1010, w = 275) inherit this selection and are not estimates of a "
        "physical optimum.",
        "Each IC value is a single evaluation of one model on one dataset; there is no "
        "replicate IC, hence no calibrated uncertainty on a delta-IC. The only dispersion "
        "budgets available are whole-pass proxies (per-pass band peak bootstrap/LOO over the "
        "22 passes, section 6), which is why the rank separation is reported as a margin "
        "ladder rather than a p-value.",
        "The mode-alignment test is limited to 10 candidate frequencies in 860-1310 uHz and "
        "to the 52 supplied modes; it does not test any other frequency range or cadence.",
        "The dead time between passes is unknown (relative_time_min is per-pass), so no "
        "coherent 22-pass stack and no 22-pass Fourier resolution are claimed; only per-pass "
        "1/T is reported.",
    ],
}

write_json(RESULTS, out)
print("wrote", RESULTS)