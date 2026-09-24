"""Step 1 - Replicated-panel surface geometry, inventory and variance decomposition.

Produces the evidence behind Finding 1 (single latent gain axis) and the
normalisation-perturbation result that defines its boundary condition.

Run: python3 output/analysis/step1_surface_geometry.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import (
    GRID,
    KEY4,
    LOGS,
    RESULTS,
    cell_means,
    flush,
    load_panels,
    pearson,
    r2,
    rank1_profile,
    save_json,
    separability,
    spearman,
)

LOG: list[str] = []
frames = load_panels(LOG)
rep, ind, fe = frames["rep"], frames["ind"], frames["independent_fit_error.tsv"]

# ---------------------------------------------------------------- inventory
inventory = {}
for name, df in [
    ("replicated_response_panel.tsv", frames["replicated_response_panel.tsv"]),
    ("independent_response_panel.tsv", frames["independent_response_panel.tsv"]),
    ("independent_fit_error.tsv", fe),
]:
    inventory[name] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        "levels": {c: [float(v) for v in sorted(df[c].unique())] for c in df.columns if c != "response_value" and c != "fit_error"},
        "value_column": "fit_error" if "fit_error" in df.columns else "response_value",
        "value_min": float(df[[c for c in df.columns if c.startswith(("response", "fit"))][0]].min()),
        "value_max": float(df[[c for c in df.columns if c.startswith(("response", "fit"))][0]].max()),
    }
inventory["replicated_response_panel.tsv"]["complete_factorial"] = bool(
    len(rep) == 7 * 7 * 8 * 7 * 3 and rep.duplicated(KEY4 + ["replicate_id"]).sum() == 0
)
inventory["independent_response_panel.tsv"]["complete_factorial"] = bool(
    len(ind) == 7 * 7 * 7 * 4 * 3 and ind.duplicated(KEY4 + ["replicate_id"]).sum() == 0
)
LOG.append(f"replicated panel complete 7x7x8x7x3 = {inventory['replicated_response_panel.tsv']['complete_factorial']}")
LOG.append(f"independent panel complete 7x7x7x4x3 = {inventory['independent_response_panel.tsv']['complete_factorial']}")
save_json(inventory, RESULTS / "t01_inventory.json")

# ------------------------------------------------- orthogonal decomposition
y = rep["ly"]
grand = y.mean()
ss_total = float(((y - grand) ** 2).sum())
n = len(y)
FAC = {
    "a1": "input_axis_1_index",
    "a2": "input_axis_2_index",
    "bin": "signal_bin_index",
    "chan": "output_channel_index",
}
levels = {k: rep[v].nunique() for k, v in FAC.items()}


def ss_terms(keys) -> float:
    """SS of the interaction subspace spanned by `keys` in a balanced design.

    The full factorial is complete, so the subspaces of a hierarchical ANOVA are
    mutually orthogonal and the SS of a subspace is the total SS of its cell
    means minus the SS of every strict sub-subspace.
    """
    count = n / float(np.prod([levels[k] for k in keys]))
    means = rep.groupby([FAC[k] for k in keys])["ly"].mean()
    total = float(((means - grand) ** 2).sum() * count)
    lower = set()
    for r in range(1, len(keys)):
        for combo in itertools.combinations(keys, r):
            lower.add(combo)
    return total - sum(ss_terms(list(c)) for c in lower)


decomp = {"total_ss_log": ss_total, "n": n, "main": {}, "two_way": {}, "three_way": {}, "four_way": None}
for k in FAC:
    decomp["main"][k] = ss_terms([k])
for combo in itertools.combinations(FAC, 2):
    decomp["two_way"]["x".join(combo)] = ss_terms(list(combo))
for combo in itertools.combinations(FAC, 3):
    decomp["three_way"]["x".join(combo)] = ss_terms(list(combo))
decomp["four_way"] = ss_terms(list(FAC))
cell_mean = rep.groupby([FAC[k] for k in FAC])["ly"].transform("mean")
decomp["replicate_pure_error"] = float(((y - cell_mean) ** 2).sum())
for group in ["main", "two_way", "three_way"]:
    for k, v in list(decomp[group].items()):
        decomp[group][k + "_pct"] = 100.0 * v / ss_total
for k in ["four_way", "replicate_pure_error"]:
    decomp[k + "_pct"] = 100.0 * decomp[k] / ss_total
decomp["sum_check"] = float(
    sum(decomp["main"][k] for k in FAC)
    + sum(decomp["two_way"][k] for k in decomp["two_way"] if not k.endswith("_pct"))
    + sum(decomp["three_way"][k] for k in decomp["three_way"] if not k.endswith("_pct"))
    + decomp["four_way"]
    + decomp["replicate_pure_error"]
)
save_json(decomp, RESULTS / "t02_variance_decomposition.json")
LOG.append(
    "variance decomposition (log scale): chan main %.2f%%, input main %.2f%%, two-way %.2f%%, "
    "three-way %.2f%%, replicate error %.2f%%"
    % (
        decomp["main"]["chan_pct"],
        sum(decomp["main"][k + "_pct"] for k in ["a1", "a2", "bin"]),
        sum(decomp["two_way"][k] for k in decomp["two_way"] if not k.endswith("_pct")) / ss_total * 100,
        sum(decomp["three_way"][k] for k in decomp["three_way"] if not k.endswith("_pct")) / ss_total * 100,
        decomp["replicate_pure_error_pct"],
    )
)

# ----------------------------------------------- rank-1 gain-axis structure
cell = cell_means(rep, "ly")
profiles = {}
for label, by in [("grid_a1a2", ["input_axis_1_index", "input_axis_2_index"]), ("grid_a1a2bin", GRID)]:
    table = cell.pivot_table(
        index=by, columns="output_channel_index", values="ly"
    )
    profiles[label] = rank1_profile(table.values)
    profiles[label]["shape"] = list(table.values.shape)
save_json(profiles, RESULTS / "t03_rank1_gain_axis.json")
LOG.append(
    "rank-1 gain axis: (a1,a2)x7ch R2=%.4f rank2=%.4f ; (a1,a2,bin)x7ch R2=%.4f rank2=%.4f"
    % (
        profiles["grid_a1a2"]["rank1_r2"],
        profiles["grid_a1a2"]["rank2_r2"],
        profiles["grid_a1a2bin"]["rank1_r2"],
        profiles["grid_a1a2bin"]["rank2_r2"],
    )
)

# per-channel first-order sensitivity to each input axis
x = np.arange(7)
sensitivity = {}
for ch in range(7):
    sub = cell[cell["output_channel_index"] == ch]
    pa = sub.pivot_table(index="input_axis_2_index", columns="input_axis_1_index", values="ly")
    pb = sub.pivot_table(index="input_axis_1_index", columns="input_axis_2_index", values="ly")
    sensitivity[ch] = {
        "slope_a1": float(np.mean([np.polyfit(x, pa.loc[a].sort_index().values, 1)[0] for a in pa.index])),
        "slope_a2": float(np.mean([np.polyfit(x, pb.loc[a].sort_index().values, 1)[0] for a in pb.index])),
        "grid_sd": float(profiles["grid_a1a2"]["singular_values"] and
                         cell[cell["output_channel_index"] == ch]
                         .pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                                      columns="output_channel_index", values="ly").values.std()),
    }
sens = pd.DataFrame(sensitivity).T
sens["abs_slope_a1_ratio_to_ch0"] = sens["slope_a1"].abs() / abs(sens.loc[0, "slope_a1"])
sens["grid_sd_ratio_to_ch0"] = sens["grid_sd"] / sens.loc[0, "grid_sd"]
sens.to_csv(RESULTS / "t04_channel_sensitivity.csv")
save_json(
    {
        "slope_a1": {int(k): float(v) for k, v in sens["slope_a1"].items()},
        "slope_a2": {int(k): float(v) for k, v in sens["slope_a2"].items()},
        "grid_sd": {int(k): float(v) for k, v in sens["grid_sd"].items()},
        "ratio_a1_ch6_over_ch0": float(sens.loc[6, "slope_a1"] / sens.loc[0, "slope_a1"]),
        "ratio_gridsd_ch6_over_ch0": float(sens.loc[6, "grid_sd"] / sens.loc[0, "grid_sd"]),
        "ratio_a2_ch6_over_ch0": float(sens.loc[6, "slope_a2"] / sens.loc[0, "slope_a2"]),
    },
    RESULTS / "t04_channel_sensitivity.json",
)
LOG.append(
    "channel amplification: a1 slope ch0=%.4f -> ch6=%.4f (x%.1f); grid sd ch0=%.4f -> ch6=%.4f (x%.1f); a2 slope changes sign (ch0=%+.4f -> ch6=%+.4f)"
    % (
        sens.loc[0, "slope_a1"], sens.loc[6, "slope_a1"], sens.loc[6, "slope_a1"] / sens.loc[0, "slope_a1"],
        sens.loc[0, "grid_sd"], sens.loc[6, "grid_sd"], sens.loc[6, "grid_sd"] / sens.loc[0, "grid_sd"],
        sens.loc[0, "slope_a2"], sens.loc[6, "slope_a2"],
    )
)

# ------------------------------- normalisation perturbation (boundary probe)
# Same rank-1 question, four defensible normalisations of the same surface.
norm = {}
linear = cell.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                          columns="output_channel_index", values="n_rep").shape and None
lin_cell = rep.groupby(GRID + ["output_channel_index"], as_index=False)["response_value"].mean()
Mlin = lin_cell.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                            columns="output_channel_index", values="response_value").values
Mlog = cell.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                        columns="output_channel_index", values="ly").values
Mnorm = Mlin / Mlin.mean(axis=0, keepdims=True)          # per-channel magnitude normalised
Mz = (Mlog - Mlog.mean(axis=0, keepdims=True)) / Mlog.std(axis=0, keepdims=True)
for label, M in [("linear", Mlin), ("log", Mlog), ("linear_per_channel_scaled", Mnorm), ("log_per_channel_z", Mz)]:
    norm[label] = rank1_profile(M)
    norm[label]["shape"] = list(M.shape)
norm["linear_without_ch6"] = rank1_profile(Mlin[:, :6])
norm["log_without_ch6"] = rank1_profile(Mlog[:, :6])
save_json(norm, RESULTS / "t05_normalisation_perturbation.json")
LOG.append(
    "normalisation perturbation of rank-1 R2: linear=%.4f, log=%.4f, linear/chan-mean=%.4f, log z-scored=%.4f"
    % (norm["linear"]["rank1_r2"], norm["log"]["rank1_r2"],
       norm["linear_per_channel_scaled"]["rank1_r2"], norm["log_per_channel_z"]["rank1_r2"])
)

# -------------------------------------- bin-0 boundary (notch structure)
bincell = cell.groupby("signal_bin_index")["ly"].mean()
xs = np.arange(1, 8)
trend = {deg: np.polyfit(xs, bincell.loc[1:].values, deg) for deg in (1, 2, 3)}
notch = {f"deg{deg}": float(bincell.loc[0] - np.polyval(co, 0)) for deg, co in trend.items()}
tab = cell.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                       columns="signal_bin_index", values="ly").sort_index()
per_cell = []
for row in tab.index:
    vals = tab.loc[row].sort_index().values
    co = np.polyfit(xs, vals[1:], 2)
    per_cell.append(float(vals[0] - np.polyval(co, 0)))
per_cell = np.array(per_cell)
loo = {}
for drop in range(1, 8):
    keep = [b for b in range(1, 8) if b != drop]
    co = np.polyfit(np.array(keep), bincell.loc[keep].values, 2)
    loo[str(drop)] = float(bincell.loc[0] - np.polyval(co, 0))
notch.update(
    {
        "bin0_observed": float(bincell.loc[0]),
        "trend_predicted_deg2": float(np.polyval(trend[2], 0)),
        "per_cell_excess_min": float(per_cell.min()),
        "per_cell_excess_max": float(per_cell.max()),
        "per_cell_fraction_positive": float((per_cell > 0).mean()),
        "per_cell_matrix": per_cell.reshape(7, 7).tolist(),
        "leave_one_bin_out": loo,
    }
)
save_json(notch, RESULTS / "t06_bin0_boundary.json")
LOG.append(
    "bin0 notch: observed=%.4f vs quadratic trend=%.4f, excess=%+.4f (x%.2f); positive in %.0f%% of the 49 grid cells"
    % (notch["bin0_observed"], notch["trend_predicted_deg2"], notch["deg2"], np.exp(notch["deg2"]),
       100 * notch["per_cell_fraction_positive"])
)

# ---------------------------- channel block structure on the grid shapes
shapes = {}
for ch in range(7):
    sub = cell[cell["output_channel_index"] == ch]
    P = sub.pivot_table(index=["input_axis_1_index", "signal_bin_index"],
                        columns="input_axis_2_index", values="ly").values.ravel()
    shapes[ch] = P - P.mean()
corr = pd.DataFrame(shapes).corr()
corr.to_csv(RESULTS / "t07_channel_shape_correlation.csv")
labels = [0, 0, 0, 1, 1, 1, 1]
from common import block_permutation

gaps = {}
for name, lab in [("block_ch0_2_vs_ch3_6", labels), ("block_ch0_2_ch3_ch4_6", [0, 0, 0, 1, 2, 2, 2])]:
    gaps[name] = block_permutation(corr.values, lab, n_perm=20000)
best = None
for cut in range(1, 7):
    lab = [0] * cut + [1] * (7 - cut)
    g = block_permutation(corr.values, lab, n_perm=20000)
    if best is None or g["observed_gap"] > best[1]:
        best = (cut, g["observed_gap"], lab, g)
gaps["best_2_block"] = {"cut": best[0], "labels": best[2], **best[3]}
save_json(gaps, RESULTS / "t08_channel_block_structure.json")
LOG.append(
    "channel block structure: best 2-block cut at ch<%d, gap=%.4f, permutation p=%.5f"
    % (best[0], best[1], best[3]["perm_p"])
)

flush(LOG, LOGS / "step1.log")
print("\n".join(LOG))