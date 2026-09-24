#!/usr/bin/env python3
"""Step 08 - Final checks, consolidated result artifacts, and figures.

  * fairness check on the 13-level correspondence: is the identity mapping the
    best of the 13 cyclic shifts, or is the match an artefact of an offset?
  * consolidated machine-readable results (results.json, results_summary.csv)
  * figures for the registered findings

Outputs: evidence/results.json, evidence/results_summary.csv,
         evidence/fig_*.png
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
RNG = np.random.default_rng(99)
CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t")
d2 = pd.read_csv(DATA / "observation_set_2.tsv", sep="\t")
d3 = pd.read_csv(DATA / "observation_set_3.tsv", sep="\t")
d4 = pd.read_csv(DATA / "observation_set_4.tsv", sep="\t")
res: dict = {"task_id": "Chemistry_03_catalysis_response_landscape",
             "analysis_unit": "one complete reaction-condition observation "
                              "(panel 1) / one complete comparison series (panels 2-4)"}

# ----------------------------------------------------- cyclic-shift fairness
p1B6 = d1.groupby("numeric_feature_3")["numeric_feature_6"].mean().to_numpy()
p1B7 = d1.groupby("numeric_feature_3")["numeric_feature_7"].mean().to_numpy()
p3R = d3.groupby("numeric_feature_2")["numeric_feature_5"].mean().reindex(range(13)).to_numpy()
p4R = d4.groupby("numeric_feature_3")["numeric_feature_6"].mean().reindex(range(13)).to_numpy()
shifts = {}
for lab, x, y in [("p1B_f6_vs_p4R", p1B6, p4R), ("p1B_f6_vs_p3R", p1B6, p3R),
                  ("p1B_f7_vs_p4R", p1B7, p4R), ("p1B_f7_vs_p3R", p1B7, p3R)]:
    vals = [float(stats.spearmanr(x, np.roll(y, s)).statistic) for s in range(13)]
    shifts[lab] = {"rho_by_shift": vals, "best_shift": int(np.argmax(np.abs(vals))),
                   "best_abs_rho": float(max(np.abs(vals))),
                   "identity_shift0_rho": vals[0],
                   "identity_is_best": bool(np.argmax(np.abs(vals)) == 0)}
res["cyclic_shift_check"] = shifts

# ------------------------------------------------------------- results.json
ev = json.loads((EV / "03_core.json").read_text())
rob = json.loads((EV / "04_robustness.json").read_text())
inf = json.loads((EV / "05_interaction_inference.json").read_text())
cross = json.loads((EV / "06_crosspanel.json").read_text())
tb = json.loads((EV / "07_transfer_boundary.json").read_text())
prof = json.loads((EV / "02_structure.json").read_text())

res["panels"] = {
    "observation_set_1": {
        "n_rows": 1430, "n_cols": 8,
        "design": "complete 10 x 13 x 11 factorial, single replicate per cell",
        "identifier_columns": {"observation_index": "row counter 0..1429",
                               "numeric_feature_1": "run counter 1..1430",
                               "numeric_feature_5": "cell index 0..1429"},
        "factor_columns": {"numeric_feature_2": 10, "numeric_feature_3": 13,
                           "numeric_feature_4": 11},
        "responses": ["numeric_feature_6", "numeric_feature_7"],
        "missingness": "none",
    },
    "observation_set_2": {
        "n_rows": 41, "n_cols": 4,
        "design": "40 paired readings (two columns) + 1 single-valued column",
        "numeric_feature_3": "value 0 on row 0 only (40 of 41 entries missing)",
        "missingness": "structural; numeric_feature_3 is effectively empty",
    },
    "observation_set_3": {
        "n_rows": 104, "n_cols": 22,
        "design": "8 series x 13 conditions; identifier column numeric_feature_1 "
                  "present on the first row of each 13-row block (equals row//13)",
        "responses": ["numeric_feature_5 (duplicated exactly by numeric_feature_6)"],
        "missingness": "numeric_feature_7 fully empty; numeric_feature_1/8 and "
                       "numeric_feature_9..21 present only on the 8-row rank block",
    },
    "observation_set_4": {
        "n_rows": 91, "n_cols": 24,
        "design": "7 series x 13 conditions; identifier column numeric_feature_1 "
                  "present on the first row of each 13-row block (permuted labels)",
        "responses": ["numeric_feature_6"],
        "missingness": "numeric_feature_7/8/9 fully empty; numeric_feature_2 "
                       "single-valued; numeric_feature_10..23 present only on the 8-row rank block",
    },
}
res["resolved_identities"] = {
    "panel_3_response_identity": {
        "formula": "numeric_feature_5 = (numeric_feature_4 - numeric_feature_3) / "
                   "(numeric_feature_4 + numeric_feature_3)",
        "max_abs_residual": prof["panel_3"]["identity_C_equals_2B_over_AplusB_minus_1"]["max_abs_residual"],
        "n_rows_exact": 104,
    },
    "panel_4_response_identity": {
        "formula": "numeric_feature_6 = (numeric_feature_5 - numeric_feature_4) / "
                   "(numeric_feature_5 + numeric_feature_4)",
        "max_abs_residual": prof["panel_4"]["f6_identity"]["max_abs_residual"],
        "n_rows_exact": 91,
    },
    "panel_3_rank_block": {
        "columns": "numeric_feature_9..numeric_feature_21",
        "meaning": "ascending rank of the panel-3 response across the 8 series, "
                   "one column per condition in within-series order "
                   "(column j <-> condition CYCLE[j])",
        "verification": "13/13 columns reproduce the rank vector exactly",
    },
    "panel_4_rank_block": {
        "columns": "numeric_feature_11..numeric_feature_23",
        "meaning": "ascending rank of the panel-4 response across the 7 series, "
                   "one column per condition in within-series order "
                   "(column 10+j <-> condition CYCLE[j-1])",
        "verification": "13/13 columns reproduce the rank vector exactly",
    },
    "unresolved": ["observation_set_3.numeric_feature_7 is empty",
                   "observation_set_4.numeric_feature_7/8/9 are empty",
                   "observation_set_4.numeric_feature_10 is a permutation of 0..7 with "
                   "no stateable role",
                   "observation_set_4.numeric_feature_2 holds a single value 0"],
}

res["findings"] = {}

# ---- D1 : two responses, two structure regimes in the same factorial
f6, f7 = inf["tukey_and_bootstrap"]["numeric_feature_6"], inf["tukey_and_bootstrap"]["numeric_feature_7"]
res["findings"]["D1"] = {
    "title": "One factorial design hosts two different structure regimes",
    "object": "observation_set_1.tsv, complete 10x13x11 factorial, n=1430",
    "response_6": {"eta2_main": f6["eta2"]["ss_main"], "eta2_2way": f6["eta2"]["ss_2way"],
                   "eta2_3way": f6["eta2"]["ss_3way"],
                   "tukey_nonadditivity_F": f6["tukey_nonadditivity"]["F"],
                   "tukey_nonadditivity_p": f6["tukey_nonadditivity"]["p"],
                   "bootstrap_eta2_3way_ci95": f6["bootstrap_eta2"]["eta2_3way_ci95"]},
    "response_7": {"eta2_main": f7["eta2"]["ss_main"], "eta2_2way": f7["eta2"]["ss_2way"],
                   "eta2_3way": f7["eta2"]["ss_3way"],
                   "tukey_nonadditivity_F": f7["tukey_nonadditivity"]["F"],
                   "tukey_nonadditivity_p": f7["tukey_nonadditivity"]["p"],
                   "bootstrap_eta2_3way_ci95": f7["bootstrap_eta2"]["eta2_3way_ci95"]},
    "robustness": {
        "level_leaveout_eta2_2way_range_f6": rob["R2_level_leaveout"]["C:numeric_feature_6"]["eta2_2way_range"],
        "level_leaveout_eta2_3way_range_f6": rob["R2_level_leaveout"]["C:numeric_feature_6"]["eta2_3way_range"],
        "level_leaveout_eta2_2way_range_f7": rob["R2_level_leaveout"]["C:numeric_feature_7"]["eta2_2way_range"],
        "level_leaveout_eta2_3way_range_f7": rob["R2_level_leaveout"]["C:numeric_feature_7"]["eta2_3way_range"],
        "whole_level_holdout_mae_additive_f6": rob["R2_whole_condition_holdout"]["numeric_feature_6"]["mean_mae_additive"],
        "whole_level_holdout_mae_cell_f6": rob["R2_whole_condition_holdout"]["numeric_feature_6"]["mean_mae_cell"],
    },
    "counterevidence": inf["failed_test_preserved"],
}

# ---- D2 : the dominant factor is a rugged reproducible partition, not a gradient
res["findings"]["D2"] = {
    "title": "The dominant factor acts as a rugged level-specific partition, not a gradient",
    "object": "observation_set_1.tsv, factor numeric_feature_4 (11 levels), both responses",
    "level_means_f7": rob["R3_factorC_structure"]["level_means_f7"],
    "between_level_range": rob["R3_factorC_structure"]["between_level_range"],
    "within_level_sd_mean": rob["R3_factorC_structure"]["within_level_sd"]["mean"],
    "between_over_within_sd_ratio": rob["R3_factorC_structure"]["between_over_within_sd_ratio"],
    "linear_r2": rob["R3_factorC_structure"]["linear_in_level_index"]["r2"],
    "quadratic_r2": rob["R3_factorC_structure"]["quadratic_in_level_index"]["r2"],
    "best_two_group_partition_eta2": rob["R3_factorC_structure"]["best_two_group_partition"]["eta2"],
    "best_split_after_level_index": rob["R3_factorC_structure"]["best_two_group_partition"]["split_after_level_index"],
    "split_half_reproduction_pearson": rob["R3_factorC_structure"]["split_half_C_profile"]["pearson"],
    "split_half_reproduction_spearman": rob["R3_factorC_structure"]["split_half_C_profile"]["spearman"],
    "n_adjacent_pairs_with_nonoverlapping_ci": rob["R3_factorC_structure"]["n_adjacent_pairs_with_nonoverlapping_ci"],
    "profile_shared_with_other_response_pearson": rob["R3_factorC_structure"]["C_profile_correlation_between_responses"]["pearson"],
    "robustness": {
        "bootstrap_level_means_ci95": rob["R3_factorC_structure"]["bootstrap_level_means_ci95"],
        "n_levels": 11, "n_bootstrap": 2000,
    },
    "counterevidence": "the profile is response-specific: the 11-level profile of "
                       "numeric_feature_6 correlates only r=-0.33 with that of "
                       "numeric_feature_7, so factor C does not encode one latent "
                       "ordering shared by both responses",
}

# ---- D3 : exact response identity
res["findings"]["D3"] = {
    "title": "The response column of panels 3 and 4 is an exact algebraic function of two magnitudes",
    "object": "observation_set_3.tsv (n=104) and observation_set_4.tsv (n=91)",
    "panel_3": ev["response_identity"]["panel_3"],
    "panel_4": ev["response_identity"]["panel_4"],
    "robustness": {
        "n_rows_exact_to_1e-12_panel_3": rob["R1_identity"]["panel_3"]["n_exact_to_1e-12"],
        "n_rows_exact_to_1e-12_panel_4": rob["R1_identity"]["panel_4"]["n_exact_to_1e-12"],
        "residual_in_ulps_panel_4": rob["R1_identity"]["panel_4"]["residual_ulp_scale"],
        "alternative_forms_panel_3": {k: {"r2": v["r2"], "max_abs_resid": v["max_abs_resid"]}
                                      for k, v in rob["R1_identity"]["panel_3"]["alternative_forms"].items()},
    },
    "counterevidence": "the identity is a property of this frozen projection; it cannot "
                       "be attributed to the upstream measurement design from the "
                       "anonymous data alone",
}

# ---- D4 : transfer boundary
res["findings"]["D4"] = {
    "title": "Condition structure transfers to one panel and fails on another; response shape does not transfer between the two magnitude panels",
    "object": "cross-panel, levels of observation_set_1 factor numeric_feature_3 (13) vs "
              "the 13 conditions of observation_set_3 and observation_set_4",
    "level_correspondence": {
        "p1B_f6_vs_p4_condition_R": tb["level_correspondence"]["p1B_mean_f6|p4_cond_mean_R"],
        "p1B_f7_vs_p4_condition_R": tb["level_correspondence"]["p1B_mean_f7|p4_cond_mean_R"],
        "p1B_f6_vs_p3_condition_R": tb["level_correspondence"]["p1B_mean_f6|p3_cond_mean_R"],
        "p1B_f7_vs_p3_condition_R": tb["level_correspondence"]["p1B_mean_f7|p3_cond_mean_R"],
        "p3_condition_B_vs_p4_condition_B": tb["level_correspondence"]["p3_cond_mean_B|p4_cond_mean_B"],
        "p3_condition_R_vs_p4_condition_R": tb["level_correspondence"]["p3_cond_mean_R|p4_cond_mean_R"],
    },
    "cell_level_transfer": {
        "n_matched_cells": tb["n_matched_cells"],
        "response_raw_pearson": tb["response_shape_transfer"]["raw"]["pearson"],
        "response_condition_centred_pearson": tb["response_shape_transfer"]["condition_centred"]["pearson"],
        "response_z_rmse": tb["response_shape_transfer"]["z_scored_rmse"],
        "magnitude_B_spearman": tb["magnitude_transfer"]["B"]["spearman_raw"],
        "magnitude_B_between_condition_pearson": tb["magnitude_transfer"]["B"]["between_condition_pearson"],
        "magnitude_B_within_condition_pearson": tb["magnitude_transfer"]["B"]["within_condition_pearson_mean"],
        "magnitude_A_between_condition_pearson": tb["magnitude_transfer"]["A"]["between_condition_pearson"],
        "supplied_rank_block_pearson": tb["supplied_rank_block_comparison"]["pearson"],
    },
    "robustness": {
        "leave_one_level_out_p1f6_vs_p4": tb["leave_one_level_out"]["p1B_mean_f6|p4_cond_mean_R"],
        "leave_one_level_out_p1f6_vs_p3": tb["leave_one_level_out"]["p1B_mean_f6|p3_cond_mean_R"],
        "cyclic_shift_check": shifts,
        "leave_one_condition_out_mae_transfer": rob["R4_transfer"]["leave_one_condition_out"],
        "leave_one_series_out_condition_spearman_range": rob["R4_transfer"]["leave_one_series_out_condition_spearman"],
        "permutation_null_R_row_order": rob["R4_transfer"]["row_order"],
    },
    "counterevidence": "the response-shape failure is not evidence that the panels are "
                       "unrelated: the magnitude columns agree strongly across panels "
                       "(B: Spearman +0.567 at cell level, Pearson +0.912 between "
                       "condition means), so the two panels share scale but not shape",
}

# ---- D5 : panel 2 calibration
res["findings"]["D5"] = {
    "title": "Two paired readings agree almost perfectly but with a reproducibly compressed slope",
    "object": "observation_set_2.tsv, 41 paired readings, analysis unit = one comparison series",
    "pearson": rob["R5_panel2"]["all_41"]["pearson"],
    "spearman": ev["panel_2_calibration"]["spearman"],
    "ols_slope": rob["R5_panel2"]["all_41"]["slope"],
    "slope_ci95": rob["R5_panel2"]["all_41"]["slope_ci95"],
    "slope_ci_excludes_1": rob["R5_panel2"]["all_41"]["slope_ci_excludes_1"],
    "mean_difference": rob["R5_panel2"]["all_41"]["mean_diff"],
    "t_p_mean_difference_vs_0": rob["R5_panel2"]["all_41"]["t_p_vs_0"],
    "max_abs_difference": rob["R5_panel2"]["all_41"]["max_abs_diff"],
    "robustness": {
        "bootstrap_slope_ci95": rob["R5_panel2"]["bootstrap_slope_all"]["ci95"],
        "bootstrap_p_slope_below_1": rob["R5_panel2"]["bootstrap_slope_all"]["p_slope_lt_1"],
        "leave_one_pair_out_slope_range": [rob["R5_panel2"]["leave_one_pair_out_slope"]["min"],
                                           rob["R5_panel2"]["leave_one_pair_out_slope"]["max"]],
        "leave_one_pair_out_n_below_1": rob["R5_panel2"]["leave_one_pair_out_slope"]["n_below_1"],
        "slope_measured_subset_only": rob["R5_panel2"]["measured_rows_8_40"]["slope"],
        "slope_lattice_subset_only": rob["R5_panel2"]["lattice_rows_0_7"]["slope"],
    },
    "counterevidence": "the slope deficit is small (2.5%) and its practical impact is "
                       "limited: the largest single disagreement is 6.75 units and 22 of "
                       "528 pairs invert order (Spearman 0.983); a constant-offset "
                       "(zero-slope-difference) explanation is not rejected "
                       f"(mean difference p={rob['R5_panel2']['all_41']['t_p_vs_0']:.3f})",
}

with (EV / "results.json").open("w") as fh:
    json.dump(res, fh, indent=2, default=str)

# ------------------------------------------------------ flat summary table
rows = [
    ("D1", "observation_set_1", "eta2_main / eta2_2way / eta2_3way, numeric_feature_6",
     f"{f6['eta2']['ss_main']:.4f} / {f6['eta2']['ss_2way']:.4f} / {f6['eta2']['ss_3way']:.4f}",
     "103.83 SS on 1080 df; Tukey F=763.7"),
    ("D1", "observation_set_1", "eta2_main / eta2_2way / eta2_3way, numeric_feature_7",
     f"{f7['eta2']['ss_main']:.4f} / {f7['eta2']['ss_2way']:.4f} / {f7['eta2']['ss_3way']:.4f}",
     "6.02 SS on 1080 df; Tukey F=9.86, p=0.0017"),
    ("D2", "observation_set_1", "factor numeric_feature_4 level means, numeric_feature_7",
     f"range={rob['R3_factorC_structure']['between_level_range']:.4f}",
     f"linear R2={rob['R3_factorC_structure']['linear_in_level_index']['r2']:.5f}; "
     f"split-half r={rob['R3_factorC_structure']['split_half_C_profile']['pearson']:.4f}"),
    ("D3", "observation_set_3", "max |R - (B-A)/(A+B)|", "0.0 (104/104 exact)",
     "alternatives: A/B R2=0.9978 with max resid 0.0176"),
    ("D3", "observation_set_4", "max |R - (B-A)/(A+B)|", "1.11e-16 (91/91 exact)",
     "single float rounding unit"),
    ("D4", "cross-panel", "P1 factor B (13) vs P4 condition mean response",
     f"rho=+{tb['level_correspondence']['p1B_mean_f6|p4_cond_mean_R']['spearman_identity']:.4f}",
     f"perm p={tb['level_correspondence']['p1B_mean_f6|p4_cond_mean_R']['perm_p_two_sided']:.5f}"),
    ("D4", "cross-panel", "P1 factor B (13) vs P3 condition mean response",
     f"rho=+{tb['level_correspondence']['p1B_mean_f6|p3_cond_mean_R']['spearman_identity']:.4f}",
     f"perm p={tb['level_correspondence']['p1B_mean_f6|p3_cond_mean_R']['perm_p_two_sided']:.4f}"),
    ("D4", "cross-panel", "P3 vs P4 matched cells, response Pearson",
     f"{tb['response_shape_transfer']['raw']['pearson']:+.4f}",
     "magnitude B between-condition Pearson +0.912"),
    ("D5", "observation_set_2", "OLS slope of reading 2 on reading 1",
     f"{rob['R5_panel2']['all_41']['slope']:.4f}",
     f"CI95 [{rob['R5_panel2']['all_41']['slope_ci95'][0]:.4f}, "
     f"{rob['R5_panel2']['all_41']['slope_ci95'][1]:.4f}]; LOO all 41 below 1"),
]
pd.DataFrame(rows, columns=["finding", "panel", "quantity", "value", "robustness"]).to_csv(
    EV / "results_summary.csv", index=False)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
comp = ["main", "2-way", "3-way"]
v6 = [f6["eta2"]["ss_main"], f6["eta2"]["ss_2way"], f6["eta2"]["ss_3way"]]
v7 = [f7["eta2"]["ss_main"], f7["eta2"]["ss_2way"], f7["eta2"]["ss_3way"]]
x = np.arange(3)
ax[0].bar(x - 0.2, v6, 0.4, label="numeric_feature_6", color="#3b6ea5")
ax[0].bar(x + 0.2, v7, 0.4, label="numeric_feature_7", color="#c0603b")
ax[0].set_xticks(x); ax[0].set_xticklabels(comp)
ax[0].set_ylabel("share of total SS"); ax[0].set_ylim(0, 1)
ax[0].set_title("D1: same factorial, two regimes")
ax[0].legend(loc="upper right", frameon=False)

lv = np.arange(11)
lm = np.array(rob["R3_factorC_structure"]["level_means_f7"])
ci = np.array(rob["R3_factorC_structure"]["bootstrap_level_means_ci95"])
ax[1].errorbar(lv, lm, yerr=[lm - ci[:, 0], ci[:, 1] - lm], fmt="o-", ms=4,
               color="#3b6ea5", lw=1)
sl = rob["R3_factorC_structure"]["linear_in_level_index"]["slope"]
ic = lm.mean()
ax[1].plot(lv, ic + sl * (lv - lv.mean()), "--", color="#c0603b",
           label=f"linear fit R2={rob['R3_factorC_structure']['linear_in_level_index']['r2']:.4f}")
ax[1].set_xlabel("level of numeric_feature_4"); ax[1].set_ylabel("mean numeric_feature_7")
ax[1].set_title("D2: rugged, not a gradient"); ax[1].legend(frameon=False)

r3 = d3["numeric_feature_5"].to_numpy(float)
p3 = (d3["numeric_feature_4"] - d3["numeric_feature_3"]) / (d3["numeric_feature_4"] + d3["numeric_feature_3"])
r4 = d4["numeric_feature_6"].to_numpy(float)
p4 = (d4["numeric_feature_5"] - d4["numeric_feature_4"]) / (d4["numeric_feature_5"] + d4["numeric_feature_4"])
ax[2].scatter(p3, r3, s=8, alpha=0.7, label="panel 3 (n=104)", color="#3b6ea5")
ax[2].scatter(p4, r4, s=8, alpha=0.7, label="panel 4 (n=91)", color="#c0603b")
lim = [min(p3.min(), p4.min()), max(p3.max(), p4.max())]
ax[2].plot(lim, lim, "k--", lw=0.8, label="y = x")
ax[2].set_xlabel("(B - A) / (A + B)"); ax[2].set_ylabel("observed response")
ax[2].set_title("D3: exact identity"); ax[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(EV / "fig_findings_1_2_3.png")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
p4R = d4.groupby("numeric_feature_3")["numeric_feature_6"].mean().reindex(range(13)).to_numpy()
p3R = d3.groupby("numeric_feature_2")["numeric_feature_5"].mean().reindex(range(13)).to_numpy()
lv13 = np.arange(13)
ax[0].plot(lv13, (p1B6 - p1B6.mean()) / p1B6.std(), "o-", label="panel 1 f6 (factor B)", color="#3b6ea5")
ax[0].plot(lv13, (p4R - p4R.mean()) / p4R.std(), "s-", label="panel 4 condition mean R", color="#5a9e5a")
ax[0].plot(lv13, (p3R - p3R.mean()) / p3R.std(), "^--", label="panel 3 condition mean R", color="#c0603b")
ax[0].set_xlabel("level / condition index"); ax[0].set_ylabel("z-scored condition mean")
ax[0].set_title("D4: transfer to P4, failure on P3"); ax[0].legend(frameon=False, fontsize=7)

m = (d3.assign(pos=d3.index % 13, series=d3.index // 13)
     .rename(columns={"numeric_feature_3": "A3", "numeric_feature_4": "B3",
                      "numeric_feature_5": "R3"})
     .merge(d4.assign(pos=d4.index % 13, series=d4.index // 13)
            .rename(columns={"numeric_feature_4": "A4", "numeric_feature_5": "B4",
                             "numeric_feature_6": "R4"}),
            on=["series", "pos"]))
ax[1].scatter(m["R3"], m["R4"], s=9, alpha=0.75, color="#3b6ea5")
ax[1].set_xlabel("panel 3 response R"); ax[1].set_ylabel("panel 4 response R")
ax[1].set_title(f"D4: cell-level shape r={tb['response_shape_transfer']['raw']['pearson']:+.3f}")
ax[1].axhline(m["R4"].mean(), color="k", lw=0.5, ls=":")
ax[1].axvline(m["R3"].mean(), color="k", lw=0.5, ls=":")

ax[2].scatter(m["B3"], m["B4"], s=9, alpha=0.75, color="#5a9e5a")
lx, ly = np.log(m["B3"]), np.log(m["B4"])
sl2, ic2 = np.polyfit(lx, ly, 1)
xs = np.linspace(lx.min(), lx.max(), 50)
ax[2].plot(np.exp(xs), np.exp(sl2 * xs + ic2), "k--", lw=0.8)
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("panel 3 magnitude B"); ax[2].set_ylabel("panel 4 magnitude B")
ax[2].set_title(f"D4: magnitude transfers (log r={tb['magnitude_transfer']['B']['pearson_log']:+.3f})")
fig.tight_layout()
fig.savefig(EV / "fig_finding_4.png")
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
x2 = d2["numeric_feature_1"].to_numpy(float)
y2 = d2["numeric_feature_2"].to_numpy(float)
ax[0].scatter(x2, y2, s=16, color="#3b6ea5")
xs = np.linspace(x2.min(), x2.max(), 50)
ax[0].plot(xs, xs, "k--", lw=0.8, label="y = x")
slp, icp = np.polyfit(x2, y2, 1)
ax[0].plot(xs, slp * xs + icp, "-", color="#c0603b", lw=1,
           label=f"OLS slope {slp:.4f}")
ax[0].set_xlabel("reading 1"); ax[0].set_ylabel("reading 2")
ax[0].set_title("D5: paired readings"); ax[0].legend(frameon=False, fontsize=7)
ax[1].hist(x2 - y2, bins=14, color="#3b6ea5", edgecolor="white")
ax[1].axvline(0, color="k", lw=0.8, ls="--")
ax[1].set_xlabel("reading 1 - reading 2"); ax[1].set_ylabel("count")
ax[1].set_title(f"D5: differences, mean {rob['R5_panel2']['all_41']['mean_diff']:+.3f}")
fig.tight_layout()
fig.savefig(EV / "fig_finding_5.png")
plt.close(fig)

print("WROTE results.json, results_summary.csv, 3 figures")
print("\nCYCLIC SHIFT CHECK (is the identity mapping the best of 13 shifts?)")
for k, v in shifts.items():
    print(f"  {k:16s} identity_rho={v['identity_shift0_rho']:+.4f} "
          f"best_shift={v['best_shift']} best_abs_rho={v['best_abs_rho']:.4f} "
          f"identity_is_best={v['identity_is_best']}")