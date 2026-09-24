"""Step 7 - the shared-domain boundary between the two panels.

TruthInsightBench task: Information_03_signal_surface_invariants
Project: Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces
T0 = 2022-01-01

The two frozen panels do NOT cover the same domain:

    replicated  : signal_bin 0..7  x output_channel 0..6   (7x7x8x7x3 = 8232 rows)
    independent : signal_bin 0..6  x output_channel 0..3   (7x7x7x4x3 = 4116 rows)

Every cross-panel statement made so far (t11 direct replication, t12 alignment
search, t14 noise-scale sharing, t16 structural transfer) is restricted to the
1372 common (a1,a2,bin,chan) cells (bins 0..6 x channels 0..3). This script
quantifies how much of the apparent cross-panel disagreement can be caused by
that domain mismatch alone and states the applicability boundary.

It reports, with ACTUAL numbers:
  1. the shared / replicated-only / independent-only (a1,a2,bin,chan) cells;
  2. variance shares (sum of squared deviations from the grand log-mean) of the
     replicated-only region (bins=7 or chan in 4..6) and of the independent-only
     region (there is none - the independent panel is a strict sub-domain);
  3. mean and sd of log-response inside and outside the shared region for each
     panel, plus the raw-scale ranges;
  4. a domain-matched redo of the headline cross-panel comparison: the same
     statistics computed (A) on the common domain only and (B) with each panel
     collapsed over its own full marginal structure (bin and channel marginals
     averaged within panel);
  5. the bin-0 notch and the channel amplification re-measured inside the shared
     domain, so the report can separate "different domain" from "different law".

Only data/ is read.  Read-only with respect to every existing result file.

Run: python3 output/analysis/step7_domain_boundary.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401  (kept for parity with the other steps)
from common import (
    DATA,
    GRID,
    KEY4,
    LOGS,
    RESULTS,
    flush,
    load_panels,
    pearson,
    save_json,
    spearman,
)

LOG: list[str] = ["step7 domain boundary - shared-domain audit of the two panels"]

RESULT: dict = {
    "task": "t22_domain_boundary",
    "t0": "2022-01-01",
    "purpose": (
        "Quantify how much of the apparent cross-panel disagreement can be caused by the domain "
        "mismatch alone. The independent panel is a strict sub-domain of the replicated panel: "
        "bins 0..6 and channels 0..3 versus bins 0..7 and channels 0..6."
    ),
}

# ------------------------------------------------------------------ load
frames = load_panels(LOG)
rep, ind = frames["rep"], frames["ind"]

BINS_REP = sorted(int(b) for b in rep["signal_bin_index"].unique())
CHAN_REP = sorted(int(c) for c in rep["output_channel_index"].unique())
BINS_IND = sorted(int(b) for b in ind["signal_bin_index"].unique())
CHAN_IND = sorted(int(c) for c in ind["output_channel_index"].unique())
SHARED_BINS = sorted(set(BINS_REP) & set(BINS_IND))
SHARED_CHAN = sorted(set(CHAN_REP) & set(CHAN_IND))

RESULT["domain"] = {
    "replicated_bins": BINS_REP,
    "replicated_channels": CHAN_REP,
    "independent_bins": BINS_IND,
    "independent_channels": CHAN_IND,
    "shared_bins": SHARED_BINS,
    "shared_channels": SHARED_CHAN,
    "replicated_only_bins": sorted(set(BINS_REP) - set(BINS_IND)),
    "replicated_only_channels": sorted(set(CHAN_REP) - set(CHAN_IND)),
    "independent_only_bins": sorted(set(BINS_IND) - set(BINS_REP)),
    "independent_only_channels": sorted(set(CHAN_IND) - set(CHAN_REP)),
    "independent_is_subdomain_of_replicated": (
        set(BINS_IND) <= set(BINS_REP) and set(CHAN_IND) <= set(CHAN_REP)
    ),
}

# --------------------------------------------------- cells & variance shares
rep_cell = rep.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
ind_cell = ind.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
rep_cell["in_shared"] = rep_cell["signal_bin_index"].isin(SHARED_BINS) & rep_cell[
    "output_channel_index"].isin(SHARED_CHAN)
ind_cell["in_shared"] = ind_cell["signal_bin_index"].isin(SHARED_BINS) & ind_cell[
    "output_channel_index"].isin(SHARED_CHAN)

grand_rep = float(rep_cell["ly"].mean())
grand_ind = float(ind_cell["ly"].mean())


def ss_shares(cell_df: pd.DataFrame, grand: float) -> dict:
    """Sum-of-squared-deviations shares of the log-response cell means relative to
    that panel's own grand mean over its own full domain."""
    dev = (cell_df["ly"] - grand) ** 2
    total = float(dev.sum())
    out = {"total_ss_log": total, "n_cells": int(len(cell_df))}
    out["shared_ss"] = float(dev[cell_df["in_shared"]].sum())
    out["outside_ss"] = float(dev[~cell_df["in_shared"]].sum())
    out["shared_cells"] = int(cell_df["in_shared"].sum())
    out["outside_cells"] = int((~cell_df["in_shared"]).sum())
    out["shared_ss_share"] = float(out["shared_ss"] / total)
    out["outside_ss_share"] = float(out["outside_ss"] / total)
    return out


rep_var = ss_shares(rep_cell, grand_rep)
ind_var = ss_shares(ind_cell, grand_ind)

# Bin-wise and channel-wise marginal variance shares (each panel's own domain),
# so the "fraction of variance outside the shared domain" can be attributed to
# the bin axis (bin 7) and the channel axis (channels 4-6) separately.
rep_bin_mean = rep_cell.groupby("signal_bin_index", as_index=False)["ly"].mean()
rep_chan_mean = rep_cell.groupby("output_channel_index", as_index=False)["ly"].mean()
rep_var["bin_marginal_ss_share_bin7"] = float(
    ((rep_bin_mean["ly"] - grand_rep) ** 2)[rep_bin_mean["signal_bin_index"] == 7].sum()
    / ((rep_bin_mean["ly"] - grand_rep) ** 2).sum()
)
rep_var["chan_marginal_ss_share_ch4_6"] = float(
    ((rep_chan_mean["ly"] - grand_rep) ** 2)[rep_chan_mean["output_channel_index"] >= 4].sum()
    / ((rep_chan_mean["ly"] - grand_rep) ** 2).sum()
)
ind_var["bin_marginal_ss_share_bin7"] = None  # no bin 7 in the independent panel
ind_var["chan_marginal_ss_share_ch4_6"] = None  # no channels 4-6 in the independent panel

RESULT["variance_inside_outside_shared_region"] = {
    "definition": (
        "cell = replicate-averaged log response; deviations from that panel's own grand mean over its "
        "own full domain; SS shares therefore sum to 1 within a panel. The independent panel has no "
        "outside region (its domain is a strict subset of the replicated domain)."
    ),
    "replicated": {**rep_var, "grand_mean_log": grand_rep},
    "independent": {**ind_var, "grand_mean_log": grand_ind},
    "replicated_outside_region": "signal_bin_index == 7 OR output_channel_index in {4,5,6}",
}

# ------------------------------------------------- mean / sd inside & outside
def mean_sd(df: pd.DataFrame, mask: pd.Series) -> dict:
    v = df.loc[mask, "response_value"]
    ly = np.log(v)
    return {
        "n_rows": int(len(v)),
        "mean_log": float(ly.mean()),
        "sd_log": float(ly.std()),
        "min_raw": float(v.min()),
        "max_raw": float(v.max()),
    }


rep_shared_mask = rep_cell["in_shared"]
RESULT["panel_levels_inside_outside"] = {
    "replicated": {
        "inside_shared_region": mean_sd(rep[rep["signal_bin_index"].isin(SHARED_BINS) & rep[
            "output_channel_index"].isin(SHARED_CHAN)], pd.Series(True, index=rep[rep["signal_bin_index"].isin(SHARED_BINS) & rep["output_channel_index"].isin(SHARED_CHAN)].index)),
        "outside_shared_region": mean_sd(rep[~(rep["signal_bin_index"].isin(SHARED_BINS) & rep[
            "output_channel_index"].isin(SHARED_CHAN))], pd.Series(True, index=rep[~(rep["signal_bin_index"].isin(SHARED_BINS) & rep["output_channel_index"].isin(SHARED_CHAN))].index)),
        "whole_panel": mean_sd(rep, pd.Series(True, index=rep.index)),
    },
    "independent": {
        "inside_shared_region": mean_sd(ind, pd.Series(True, index=ind.index)),
        "outside_shared_region": None,  # independent panel lies entirely inside the shared region
        "whole_panel": mean_sd(ind, pd.Series(True, index=ind.index)),
    },
}

# ------------------------------------- domain-matched cross-panel comparison
def cross_panel(rep_df: pd.DataFrame, ind_df: pd.DataFrame, label: str, value: str = "ly") -> dict:
    """Cell-level Pearson/Spearman between the two panels on the cells they share,
    with the panels first collapsed to (a1,a2,bin,chan) cell means."""
    a = rep_df.groupby(KEY4, as_index=False)[value].mean().rename(columns={value: "ly_rep"})
    b = ind_df.groupby(KEY4, as_index=False)[value].mean().rename(columns={value: "ly_ind"})
    m = a.merge(b, on=KEY4, suffixes=("", "_r"))
    return {
        "definition": label,
        "n_cells": int(len(m)),
        "cell_pearson": pearson(m["ly_rep"], m["ly_ind"]),
        "cell_spearman": spearman(m["ly_rep"], m["ly_ind"]),
        "rep_log_mean": float(m["ly_rep"].mean()),
        "ind_log_mean": float(m["ly_ind"].mean()),
        "rep_log_sd": float(m["ly_rep"].std()),
        "ind_log_sd": float(m["ly_ind"].std()),
        "mean_level_offset_ind_minus_rep": float(m["ly_ind"].mean() - m["ly_rep"].mean()),
    }


rep_common = rep[rep["signal_bin_index"].isin(SHARED_BINS) & rep["output_channel_index"].isin(SHARED_CHAN)]
ind_common = ind[ind["signal_bin_index"].isin(SHARED_BINS) & ind["output_channel_index"].isin(SHARED_CHAN)]

def collapsed_axes_corr(rep_df: pd.DataFrame, ind_df: pd.DataFrame, label: str) -> dict:
    """Correlate the two panels after each is collapsed over its own bin x channel
    structure, leaving the 49 (a1,a2) cells. This is the comparison that survives
    the domain mismatch in the sense that bin-7/channels 4-6 information is simply
    averaged away rather than compared."""
    a = rep_df.groupby(["input_axis_1_index", "input_axis_2_index"], as_index=False)["ly"].mean()
    b = ind_df.groupby(["input_axis_1_index", "input_axis_2_index"], as_index=False)["ly"].mean()
    m = a.merge(b, on=["input_axis_1_index", "input_axis_2_index"], suffixes=("_rep", "_ind"))
    return {
        "definition": label,
        "n_cells": int(len(m)),
        "pearson": pearson(m["ly_rep"], m["ly_ind"]),
        "spearman": spearman(m["ly_rep"], m["ly_ind"]),
        "rep_mean_level": float(m["ly_rep"].mean()),
        "ind_mean_level": float(m["ly_ind"].mean()),
    }


RESULT["cross_panel_comparison"] = {
    "A_common_domain_only": cross_panel(
        rep_common, ind_common,
        "both panels restricted to bins 0..6 x channels 0..3 (the 1372 common cells); "
        "this is the same restriction used by t11/t12/t14/t16"),
    "B_marginal_collapsed_shared": collapsed_axes_corr(
        rep_common, ind_common,
        "both panels collapsed over their shared bin x channel structure to the 49 (a1,a2) cells; "
        "the bin/channel mismatch is averaged away rather than compared"),
    "B2_marginal_collapsed_unrestricted": collapsed_axes_corr(
        rep, ind,
        "each panel collapsed over its OWN full bin x channel structure to the 49 (a1,a2) cells; "
        "this uses the replicated panel's bins 0..7 and channels 0..6 and the independent panel's "
        "bins 0..6 and channels 0..3"),
}

# marginal-collapsed (bin, chan) structure: how different are the two panels'
# own marginal planes on the shared bins/channels?
r_bc = rep_common.groupby(["signal_bin_index", "output_channel_index"], as_index=False)["ly"].mean()
i_bc = ind_common.groupby(["signal_bin_index", "output_channel_index"], as_index=False)["ly"].mean()
r_bc = r_bc.merge(i_bc, on=["signal_bin_index", "output_channel_index"], suffixes=("_rep", "_ind"))
r_bc["rep_centred"] = r_bc["ly_rep"] - r_bc["ly_rep"].mean()
r_bc["ind_centred"] = r_bc["ly_ind"] - r_bc["ly_ind"].mean()
RESULT["cross_panel_comparison"]["marginal_plane_bin_x_chan"] = {
    "definition": (
        "the 28 shared (bin,chan) marginal means of each panel, centred; this is where the two "
        "panels differ most (replicated channel main effect 58.9% vs independent 0.18%)"),
    "n_cells": int(len(r_bc)),
    "pearson": pearson(r_bc["rep_centred"], r_bc["ind_centred"]),
    "spearman": spearman(r_bc["rep_centred"], r_bc["ind_centred"]),
    "rep_marginal_sd": float(r_bc["rep_centred"].std()),
    "ind_marginal_sd": float(r_bc["ind_centred"].std()),
}

# how much of the common-cell agreement is explained by the two panels' own
# bin x chan marginals? compare raw cell correlation with correlation after
# subtracting each panel's own (bin,chan) marginal.
def residualise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["signal_bin_index", "output_channel_index"])["ly"].transform("mean")
    return df.assign(ly_res=df["ly"] - g)


r_res = residualise(rep_common)
i_res = residualise(ind_common)
RESULT["cross_panel_comparison"]["C_marginal_removed"] = cross_panel(
    r_res, i_res, value="ly_res",
    label="common cells with each panel's own (bin,chan) marginal removed (interaction only); "
    "a non-zero value here means the two panels share cell-level structure beyond their marginals "
    "(but see t11: this is also where the replicate dispersion covariance lives)")  # noqa: E501

# ---------------------------------------------- shared-domain feature redo
# bin-0 notch inside the shared domain, both panels, on their own bin axis
def notch(df: pd.DataFrame) -> dict:
    s = df.groupby("signal_bin_index")["ly"].mean()
    b = s.index.values.astype(float)
    c2 = np.polyfit(b[1:], s.values[1:], 2)
    pred = float(np.polyval(c2, 0.0))
    return {
        "n_bins": int(len(s)),
        "bin0_observed": float(s.values[0]),
        "trend_predicted": pred,
        "excess": float(s.values[0] - pred),
    }


RESULT["shared_domain_feature_recheck"] = {
    "bin0_notch_replicated_common_bins": notch(rep_common),
    "bin0_notch_independent": notch(ind),
    "note": (
        "the notch is a bin-local feature; it is unchanged by the channel restriction and is the "
        "one feature that keeps its sign in the shared domain (t19)."
    ),
}

# channel amplification of the replicated panel inside the shared channel window
def slopes(df: pd.DataFrame) -> list:
    x = np.arange(7)
    cm = df.groupby(GRID + ["output_channel_index"], as_index=False)["ly"].mean()
    out = []
    for ch in sorted(cm["output_channel_index"].unique()):
        P = cm[cm["output_channel_index"] == ch].pivot_table(
            index="input_axis_2_index", columns="input_axis_1_index", values="ly"
        ).sort_index()
        out.append(float(np.mean([np.polyfit(P.columns.values, P.loc[a].values, 1)[0] for a in P.index])))
    return out


rep_slopes_shared = slopes(rep_common)
rep_slopes_full = slopes(rep)
RESULT["shared_domain_feature_recheck"]["replicated_a1_slopes"] = {
    "full_domain_channels_0_6": rep_slopes_full,
    "shared_domain_channels_0_3": rep_slopes_shared,
    "ratio_shared_ch3_over_ch0": float(rep_slopes_shared[3] / rep_slopes_shared[0]),
    "ratio_full_ch6_over_ch0": float(rep_slopes_full[6] / rep_slopes_full[0]),
}
ind_slopes = slopes(ind)
RESULT["shared_domain_feature_recheck"]["independent_a1_slopes_channels_0_3"] = ind_slopes

# --------------------------------------------------- what the mismatch costs
# Rebuild the "replicated as seen by the independent panel's domain" surfaces:
# (i) the replicated panel collapsed over its own 8 bins (bin average) and
# (ii) the independent panel collapsed over its 7 bins. Then a1/a2 slopes on the
# SAME channel window (0..3) and same bin window (0..6) can be compared without
# the domain mismatch; any remaining disagreement is a property of the panels.
def axis_profiles(df: pd.DataFrame, label: str) -> dict:
    d = df.groupby(["input_axis_1_index", "input_axis_2_index"], as_index=False)["ly"].mean()
    return {
        "label": label,
        "n_cells": int(len(d)),
        "a1_slope": float(np.mean([
            np.polyfit(g["input_axis_1_index"].values, g["ly"].values, 1)[0]
            for _, g in d.groupby("input_axis_2_index")
        ])),
        "a2_slope": float(np.mean([
            np.polyfit(g["input_axis_2_index"].values, g["ly"].values, 1)[0]
            for _, g in d.groupby("input_axis_1_index")
        ])),
        "a1_level_mean": [float(v) for v in d.groupby("input_axis_1_index")["ly"].mean().values],
        "grand_mean_log": float(d["ly"].mean()),
        "sd_log": float(d["ly"].std()),
    }


RESULT["marginal_collapsed_axes"] = {
    "definition": (
        "each panel collapsed over its own bin x channel structure; the only remaining axes are "
        "(a1,a2). This is the domain-free comparison: it uses the replicated panel's bins 0..7 and "
        "channels 0..6 and the independent panel's bins 0..6 and channels 0..3."
    ),
    "replicated_binchan_collapsed": axis_profiles(rep, "replicated, all 8 bins x all 7 channels"),
    "independent_binchan_collapsed": axis_profiles(ind, "independent, all 7 bins x all 4 channels"),
    "replicated_common_domain_collapsed": axis_profiles(rep_common, "replicated, bins 0..6 x channels 0..3"),
}

# variance the replicated panel holds outside the shared domain, relative to the
# per-channel variance scale that the cross-panel noise comparison (t14) uses
rep_chan_var = rep_cell.groupby("output_channel_index")["ly"].var()
RESULT["variance_concentration_check"] = {
    "definition": (
        "per-channel variance of the log cell means in the replicated panel; the panel's outside "
        "share is concentrated in the high-gain channels 4-6 and in bin 7"
    ),
    "per_channel_var_replicated": {str(k): float(v) for k, v in rep_chan_var.items()},
    "outside_share_of_total_ss": rep_var["outside_ss_share"],
    "outside_share_excluding_chan4_6": float(
        (rep_cell.loc[(~rep_cell["in_shared"]) & (rep_cell["output_channel_index"] < 4), "ly"]
         - grand_rep).pow(2).sum() / rep_var["total_ss_log"]
    ),
    "outside_share_excluding_bin7": float(
        (rep_cell.loc[(~rep_cell["in_shared"]) & (rep_cell["signal_bin_index"] < 7), "ly"]
         - grand_rep).pow(2).sum() / rep_var["total_ss_log"]
    ),
}

# --------------------------------------------------------------- reporting
lines = [
    "domains: replicated bins %s x channels %s ; independent bins %s x channels %s"
    % (BINS_REP, CHAN_REP, BINS_IND, CHAN_IND),
    "shared bins %s ; shared channels %s ; independent is a strict sub-domain: %s"
    % (SHARED_BINS, SHARED_CHAN, RESULT["domain"]["independent_is_subdomain_of_replicated"]),
    "replicated cells inside shared region %d / %d (%.4f) ; outside %d / %d (%.4f)"
    % (rep_var["shared_cells"], rep_var["n_cells"], 1 - rep_var["outside_ss_share"],
       rep_var["outside_cells"], rep_var["n_cells"], rep_var["outside_ss_share"]),
    "replicated outside SS share = %.4f (bin-7 marginal share %.4f of bin marginal SS; ch4-6 marginal share %.4f of chan marginal SS)"
    % (rep_var["outside_ss_share"], rep_var["bin_marginal_ss_share_bin7"], rep_var["chan_marginal_ss_share_ch4_6"]),
    "independent cells outside shared region: %d / %d (%.4f)"
    % (ind_var["outside_cells"], ind_var["n_cells"], ind_var["outside_ss_share"]),
    "log-response mean/sd: replicated inside %.4f/%.4f ; replicated outside %.4f/%.4f ; independent (all shared) %.4f/%.4f"
    % (RESULT["panel_levels_inside_outside"]["replicated"]["inside_shared_region"]["mean_log"],
       RESULT["panel_levels_inside_outside"]["replicated"]["inside_shared_region"]["sd_log"],
       RESULT["panel_levels_inside_outside"]["replicated"]["outside_shared_region"]["mean_log"],
       RESULT["panel_levels_inside_outside"]["replicated"]["outside_shared_region"]["sd_log"],
       RESULT["panel_levels_inside_outside"]["independent"]["inside_shared_region"]["mean_log"],
       RESULT["panel_levels_inside_outside"]["independent"]["inside_shared_region"]["sd_log"]),
    "cross-panel cell correlation on the common domain: pearson=%+.4f spearman=%+.4f (n=%d cells)"
    % (RESULT["cross_panel_comparison"]["A_common_domain_only"]["cell_pearson"],
       RESULT["cross_panel_comparison"]["A_common_domain_only"]["cell_spearman"],
       RESULT["cross_panel_comparison"]["A_common_domain_only"]["n_cells"]),
    "cross-panel (bin,chan) marginal plane correlation: pearson=%+.4f spearman=%+.4f (n=%d)"
    % (RESULT["cross_panel_comparison"]["marginal_plane_bin_x_chan"]["pearson"],
       RESULT["cross_panel_comparison"]["marginal_plane_bin_x_chan"]["spearman"],
       RESULT["cross_panel_comparison"]["marginal_plane_bin_x_chan"]["n_cells"]),
    "cross-panel after removing each panel's own (bin,chan) marginal: pearson=%+.4f spearman=%+.4f"
    % (RESULT["cross_panel_comparison"]["C_marginal_removed"]["cell_pearson"],
       RESULT["cross_panel_comparison"]["C_marginal_removed"]["cell_spearman"]),
    "cross-panel (a1,a2)-collapsed: shared domain pearson=%+.4f ; each panel's own full domain pearson=%+.4f"
    % (RESULT["cross_panel_comparison"]["B_marginal_collapsed_shared"]["pearson"],
       RESULT["cross_panel_comparison"]["B2_marginal_collapsed_unrestricted"]["pearson"]),
    "marginal-collapsed a1/a2 slopes: replicated %+.4f/%+.4f ; independent %+.4f/%+.4f"
    % (RESULT["marginal_collapsed_axes"]["replicated_binchan_collapsed"]["a1_slope"],
       RESULT["marginal_collapsed_axes"]["replicated_binchan_collapsed"]["a2_slope"],
       RESULT["marginal_collapsed_axes"]["independent_binchan_collapsed"]["a1_slope"],
       RESULT["marginal_collapsed_axes"]["independent_binchan_collapsed"]["a2_slope"]),
    "shared-domain a1 slopes over channels 0..3: replicated %s ; independent %s"
    % (np.round(rep_slopes_shared, 4).tolist(), np.round(ind_slopes, 4).tolist()),
    "bin0 notch: replicated (shared bins) observed=%.4f trend=%.4f excess=%+.4f ; independent observed=%.4f trend=%.4f excess=%+.4f"
    % (RESULT["shared_domain_feature_recheck"]["bin0_notch_replicated_common_bins"]["bin0_observed"],
       RESULT["shared_domain_feature_recheck"]["bin0_notch_replicated_common_bins"]["trend_predicted"],
       RESULT["shared_domain_feature_recheck"]["bin0_notch_replicated_common_bins"]["excess"],
       RESULT["shared_domain_feature_recheck"]["bin0_notch_independent"]["bin0_observed"],
       RESULT["shared_domain_feature_recheck"]["bin0_notch_independent"]["trend_predicted"],
       RESULT["shared_domain_feature_recheck"]["bin0_notch_independent"]["excess"]),
]

RESULT["verdict"] = {
    "domain_mismatch_alone_cannot_explain_the_disagreement": True,
    "reason": (
        "the entire disagreement is already measured inside the shared region: the bin-0 notch is "
        "opposite in sign within bins 0..6, and the cross-panel cell correlation on the 1372 common "
        "cells is ~0. Restricting or expanding the domain cannot remove those facts."
    ),
    "what_the_domain_mismatch_DOES_bound": (
        "the replicated panel's channel-amplification law (channels 0-6) and its peak channel 6, and "
        "its bin-7 level, are outside the independent panel's domain and are unsupported by it. The "
        "outside region carries %.4f of the replicated panel's cell-level SS, concentrated in "
        "channels 4-6; every cross-panel statement is therefore a statement about the low-gain, "
        "shared corner of the replicated surface." % rep_var["outside_ss_share"]
    ),
}

save_json(RESULT, RESULTS / "t22_domain_boundary.json")
flush(LOG + lines, LOGS / "step7.log")
print("\n".join(lines))