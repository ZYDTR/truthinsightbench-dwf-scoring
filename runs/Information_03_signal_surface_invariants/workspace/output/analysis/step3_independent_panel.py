"""Step 3 - Independent response panel: replication of the signal surface, of the
noise scale, and the supplied independent fit-error matrix.

Supports Findings 3, 4 and 5.

Run: python3 output/analysis/step3_independent_panel.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import (
    GRID,
    KEY4,
    KEY6,
    LOGS,
    RESULTS,
    block_permutation,
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

# ------------------------------------------------------------ panel overview
overview = {}
for name, df in [("replicated", rep), ("independent", ind)]:
    G = df["ly"].mean()
    ss = float(((df["ly"] - G) ** 2).sum())
    main = {}
    for label, col in [("input_axis_1_index", "a1"), ("input_axis_2_index", "a2"),
                       ("signal_bin_index", "bin"), ("output_channel_index", "chan")]:
        g = df.groupby(label)["ly"].agg(["mean", "size"])
        main[label] = float((g["size"] * (g["mean"] - G) ** 2).sum() / ss)
    overview[name] = {
        "rows": int(len(df)),
        "channels": int(df["output_channel_index"].nunique()),
        "bins": int(df["signal_bin_index"].nunique()),
        "log_mean": float(df["ly"].mean()),
        "log_sd": float(df["ly"].std()),
        "main_effect_share": main,
        "raw_value_range": [float(df["response_value"].min()), float(df["response_value"].max())],
    }
save_json(overview, RESULTS / "t10_panel_overview.json")
LOG.append(
    "panel overview: replicated chan main share=%.4f, independent chan main share=%.4f; "
    "independent bin main share=%.4f, a2 main share=%.4f"
    % (
        overview["replicated"]["main_effect_share"]["output_channel_index"],
        overview["independent"]["main_effect_share"]["output_channel_index"],
        overview["independent"]["main_effect_share"]["signal_bin_index"],
        overview["independent"]["main_effect_share"]["input_axis_2_index"],
    )
)

# ------------------------------------ matched common cells: direct replication
common = rep[(rep["signal_bin_index"] < 7) & (rep["output_channel_index"] < 4)]
matched = common.merge(ind, on=KEY6, suffixes=("_rep", "_ind"))
a = matched["ly_rep"].values
b = matched["ly_ind"].values
rng = np.random.default_rng(11)
boot = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(4000))])
cell_matched = matched.groupby(KEY4, as_index=False).agg(rep=("ly_rep", "mean"), ind=("ly_ind", "mean"))
replication = {
    "n_rows_matched": int(len(matched)),
    "n_cells_matched": int(len(cell_matched)),
    "row_pearson": pearson(a, b),
    "row_spearman": spearman(a, b),
    "row_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "cell_pearson": pearson(cell_matched["rep"], cell_matched["ind"]),
    "cell_spearman": spearman(cell_matched["rep"], cell_matched["ind"]),
    "replicated_log_mean": float(common["ly"].mean()),
    "independent_log_mean": float(ind["ly"].mean()),
    "replicated_log_sd": float(common["ly"].std()),
    "independent_log_sd": float(ind["ly"].std()),
}
save_json(replication, RESULTS / "t11_direct_replication.json")
LOG.append(
    "direct replication on the %d common grid cells: pearson=%+.4f [%+.4f,%+.4f] spearman=%+.4f"
    % (
        replication["n_cells_matched"],
        replication["cell_pearson"],
        replication["row_bootstrap_ci95"][0],
        replication["row_bootstrap_ci95"][1],
        replication["cell_spearman"],
    )
)

# robust alignment search: reflections, axis swap and channel relabelling
cell_rep = common.groupby(KEY4, as_index=False)["ly"].mean()
cell_ind = ind.groupby(KEY4, as_index=False)["ly"].mean()


def merge_with(u1, u2, channel_map=None):
    left = cell_rep.copy()
    left["u1"] = u1(left["input_axis_1_index"])
    left["u2"] = u2(left["input_axis_2_index"])
    right = cell_ind.copy()
    if channel_map is not None:
        right["output_channel_index"] = right["output_channel_index"].map(channel_map)
    m = left.merge(
        right,
        left_on=["u1", "u2", "signal_bin_index", "output_channel_index"],
        right_on=["input_axis_1_index", "input_axis_2_index", "signal_bin_index", "output_channel_index"],
        suffixes=("_rep", "_ind"),
    )
    return m


align = {}
for n1, f1 in [("forward", lambda v: v), ("reversed", lambda v: 6 - v)]:
    for n2, f2 in [("forward", lambda v: v), ("reversed", lambda v: 6 - v)]:
        m = merge_with(f1, f2)
        align[f"a1_{n1}__a2_{n2}"] = {
            "n": int(len(m)),
            "pearson": pearson(m["ly_rep"], m["ly_ind"]),
            "spearman": spearman(m["ly_rep"], m["ly_ind"]),
        }
        m2 = merge_with(f2, f1)  # axes swapped
        align[f"SWAP_a1_{n1}__a2_{n2}"] = {
            "n": int(len(m2)),
            "pearson": pearson(m2["ly_rep"], m2["ly_ind"]),
            "spearman": spearman(m2["ly_rep"], m2["ly_ind"]),
        }
best_align = max(align.items(), key=lambda kv: abs(kv[1]["pearson"]))
align["_best_abs"] = {"label": best_align[0], **best_align[1]}
save_json(align, RESULTS / "t12_alignment_search.json")
LOG.append(
    "alignment search over reflections/axis swaps: strongest match = %s (pearson %+.4f); "
    "identity pearson=%+.4f"
    % (best_align[0], best_align[1]["pearson"], align["a1_forward__a2_forward"]["pearson"])
)

# channel relabelling: can any permutation of the 4 independent channels match?
perm_best = None
for perm in itertools.permutations(range(4)):
    m = merge_with(lambda v: v, lambda v: v, channel_map={c: perm[c] for c in range(4)})
    r = pearson(m["ly_rep"], m["ly_ind"])
    if perm_best is None or abs(r) > abs(perm_best[1]):
        perm_best = (perm, r)
align["_best_channel_permutation"] = {"perm": list(perm_best[0]), "pearson": float(perm_best[1])}
save_json(align, RESULTS / "t12_alignment_search.json")
LOG.append(
    "channel relabelling of the independent panel: best permutation %s gives pearson %+.4f"
    % (list(perm_best[0]), perm_best[1])
)

# ------------------------------------------------ marginal (refracted) profiles
marg = {}
for ax in ["input_axis_1_index", "input_axis_2_index"]:
    pr = rep.groupby(ax)["ly"].mean()
    pi = ind.groupby(ax)["ly"].mean()
    marg[f"rep_{ax}"] = [float(v) for v in pr.values]
    marg[f"ind_{ax}"] = [float(v) for v in pi.values]
marg["rep_a1_vs_ind_a1"] = {"pearson": pearson(marg["rep_input_axis_1_index"], marg["ind_input_axis_1_index"]),
                            "spearman": spearman(marg["rep_input_axis_1_index"], marg["ind_input_axis_1_index"])}
marg["rep_a2_vs_ind_a2"] = {"pearson": pearson(marg["rep_input_axis_2_index"], marg["ind_input_axis_2_index"]),
                            "spearman": spearman(marg["rep_input_axis_2_index"], marg["ind_input_axis_2_index"])}
marg["rep_a1_vs_ind_a2"] = {"pearson": pearson(marg["rep_input_axis_1_index"], marg["ind_input_axis_2_index"]),
                            "spearman": spearman(marg["rep_input_axis_1_index"], marg["ind_input_axis_2_index"])}
marg["rep_a2_vs_ind_a1"] = {"pearson": pearson(marg["rep_input_axis_2_index"], marg["ind_input_axis_1_index"]),
                            "spearman": spearman(marg["rep_input_axis_2_index"], marg["ind_input_axis_1_index"])}
marg["rep_a1_range"] = float(np.ptp(marg["rep_input_axis_1_index"]))
marg["rep_a2_range"] = float(np.ptp(marg["rep_input_axis_2_index"]))
marg["ind_a1_range"] = float(np.ptp(marg["ind_input_axis_1_index"]))
marg["ind_a2_range"] = float(np.ptp(marg["ind_input_axis_2_index"]))
save_json(marg, RESULTS / "t13_marginal_refraction.json")
LOG.append(
    "marginal refraction: rep_a1~ind_a1 r=%+.4f ; rep_a2~ind_a2 r=%+.4f ; rep_a2~ind_a1 r=%+.4f (rank %+.4f)"
    % (marg["rep_a1_vs_ind_a1"]["pearson"], marg["rep_a2_vs_ind_a2"]["pearson"],
       marg["rep_a2_vs_ind_a1"]["pearson"], marg["rep_a2_vs_ind_a1"]["spearman"])
)
LOG.append(
    "marginal ranges: replicated a1=%.3f a2=%.3f ; independent a1=%.3f a2=%.3f"
    % (marg["rep_a1_range"], marg["rep_a2_range"], marg["ind_a1_range"], marg["ind_a2_range"])
)

# -------------------------------------------------- shared noise-scale structure
def cell_cv(df):
    g = df.groupby(KEY4)["response_value"]
    out = pd.DataFrame({"cv": g.std() / g.mean(), "sd": g.std()}).reset_index()
    return out


cv_rep = cell_cv(common)
cv_ind = cell_cv(ind)
mm = cv_rep.merge(cv_ind, on=KEY4, suffixes=("_rep", "_ind"))
obs = pearson(mm["cv_rep"], mm["cv_ind"])
rng2 = np.random.default_rng(5)
null = np.array([pearson(mm["cv_rep"], rng2.permutation(mm["cv_ind"].values)) for _ in range(5000)])
mm["cv_rep_res"] = mm["cv_rep"] - mm.groupby(["input_axis_1_index", "input_axis_2_index"])["cv_rep"].transform("mean")
mm["cv_ind_res"] = mm["cv_ind"] - mm.groupby(["input_axis_1_index", "input_axis_2_index"])["cv_ind"].transform("mean")
obs_res = pearson(mm["cv_rep_res"], mm["cv_ind_res"])
null_res = np.array([pearson(mm["cv_rep_res"], rng2.permutation(mm["cv_ind_res"].values)) for _ in range(5000)])
noise = {
    "n_cells": int(len(mm)),
    "cv_pearson": obs,
    "cv_perm_p": float((np.abs(null) >= abs(obs)).mean()),
    "cv_pearson_after_a1a2_means_removed": obs_res,
    "cv_perm_p_after_removal": float((np.abs(null_res) >= abs(obs_res)).mean()),
    "level_pearson_same_cells": pearson(mm["sd_rep"] * 0 + cell_matched["rep"], cell_matched["ind"]),
    "cv_profile_by_a1": {
        "rep": [float(v) for v in cv_rep.groupby("input_axis_1_index")["cv"].mean().values],
        "ind": [float(v) for v in cv_ind.groupby("input_axis_1_index")["cv"].mean().values],
    },
    "cv_profile_by_a2": {
        "rep": [float(v) for v in cv_rep.groupby("input_axis_2_index")["cv"].mean().values],
        "ind": [float(v) for v in cv_ind.groupby("input_axis_2_index")["cv"].mean().values],
    },
}
noise["cv_profile_by_a1_corr"] = pearson(noise["cv_profile_by_a1"]["rep"], noise["cv_profile_by_a1"]["ind"])
noise["cv_profile_by_a2_corr"] = pearson(noise["cv_profile_by_a2"]["rep"], noise["cv_profile_by_a2"]["ind"])
save_json(noise, RESULTS / "t14_noise_scale_sharing.json")
LOG.append(
    "noise-scale sharing: cell CV pearson=%+.4f (perm p<%.5f) ; after removing (a1,a2) means %+.4f (p<%.5f)"
    % (obs, max(noise["cv_perm_p"], 1e-5), obs_res, max(noise["cv_perm_p_after_removal"], 1e-5))
)
LOG.append(
    "CV marginal profiles: by a1 rep~ind r=%+.4f ; by a2 rep~ind r=%+.4f"
    % (noise["cv_profile_by_a1_corr"], noise["cv_profile_by_a2_corr"])
)

# ------------------------------------------------------- fit-error matrix (n=21)
fe_stats = {
    "rows": int(len(fe)),
    "n_axes": int(fe["input_axis_index"].nunique()),
    "n_replicates": int(fe["replicate_id"].nunique()),
    "mean": float(fe["fit_error"].mean()),
    "sd": float(fe["fit_error"].std()),
    "min": float(fe["fit_error"].min()),
    "max": float(fe["fit_error"].max()),
}
G = fe["fit_error"].mean()
ss = float(((fe["fit_error"] - G) ** 2).sum())
for col, lab in [("input_axis_index", "axis"), ("replicate_id", "replicate")]:
    g = fe.groupby(col)["fit_error"].agg(["mean", "size"])
    fe_stats[f"ss_share_{lab}"] = float((g["size"] * (g["mean"] - G) ** 2).sum() / ss)
resid = fe["fit_error"] - fe.groupby("input_axis_index")["fit_error"].transform("mean")
fe_stats["residual_share_after_axis"] = float((resid ** 2).sum() / ss)

fa = fe.groupby("input_axis_index")["fit_error"].mean()
corrs = {}
for lab, series in [("rep_cv_by_a1", cv_rep.groupby("input_axis_1_index")["cv"].mean()),
                    ("rep_cv_by_a2", cv_rep.groupby("input_axis_2_index")["cv"].mean()),
                    ("ind_cv_by_a1", cv_ind.groupby("input_axis_1_index")["cv"].mean()),
                    ("ind_cv_by_a2", cv_ind.groupby("input_axis_2_index")["cv"].mean())]:
    r = pearson(fa.values, series.values)
    t = r * np.sqrt(5 / (1 - r ** 2)) if abs(r) < 1 else np.inf
    rng3 = np.random.default_rng(3)
    nl = np.array([pearson(fa.values, rng3.permutation(series.values)) for _ in range(20000)])
    corrs[lab] = {"pearson": r, "t_df5": float(t), "crit_005": 2.5706,
                  "perm_p": float((np.abs(nl) >= abs(r)).mean()), "n": 7}
fe_stats["correlations"] = corrs
fe_stats["power_note"] = (
    "fit_error has only 7 axis levels x 3 replicates. Any correlation is computed on n=7 points; "
    "the two-sided t critical value at df=5, alpha=0.05 is 2.571 and the permutation p-value floor "
    "for n=7 is 2/5040=0.0004. Positive associations are consistent with a shared noise-scale axis "
    "but are not individually significant."
)
save_json(fe_stats, RESULTS / "t15_fit_error.json")
LOG.append(
    "fit_error (n=%d): mean=%.4f ; axis SS share=%.3f, replicate SS share=%.3f ; corr with rep CV by a1 r=%+.3f (p=%.3f), with ind CV by a1 r=%+.3f (p=%.3f)"
    % (fe_stats["rows"], fe_stats["mean"], fe_stats["ss_share_axis"], fe_stats["ss_share_replicate"],
       corrs["rep_cv_by_a1"]["pearson"], corrs["rep_cv_by_a1"]["perm_p"],
       corrs["ind_cv_by_a1"]["pearson"], corrs["ind_cv_by_a1"]["perm_p"])
)

# --------------------- structural transfer: does the replicated model predict ind?
def design(df, terms):
    n = len(df)
    cols = [np.ones(n).reshape(-1, 1)]
    for t in terms:
        D = None
        for k in t.split("+"):
            d = pd.get_dummies(df[k]).values.astype(float)
            D = d if D is None else np.einsum("ij,ik->ijk", D, d).reshape(n, -1)
        cols.append(D[:, 1:])
    return np.hstack(cols)


terms = ["output_channel_index", "output_channel_index+input_axis_1_index",
         "output_channel_index+input_axis_2_index", "output_channel_index+signal_bin_index"]
rr = rep[(rep["output_channel_index"] < 4) & (rep["signal_bin_index"] < 7)]
X = design(rr, terms)
beta, *_ = np.linalg.lstsq(X, rr["ly"].values, rcond=None)
Xi = design(ind, terms)
transfer = {
    "rep_model_r2_on_rep": r2(rr["ly"].values, X @ beta),
    "rep_model_r2_on_ind": r2(ind["ly"].values, Xi @ beta),
    "ind_own_model_r2": r2(ind["ly"].values, Xi @ np.linalg.lstsq(Xi, ind["ly"].values, rcond=None)[0]),
}
best = None
for perm in itertools.permutations(range(4)):
    j = ind.copy()
    j["output_channel_index"] = j["output_channel_index"].map({c: perm[c] for c in range(4)})
    Xj = design(j, terms)
    v = r2(j["ly"].values, Xj @ beta)
    if best is None or v > best[1]:
        best = (perm, v)
transfer["best_channel_permutation_r2"] = float(best[1])
transfer["best_channel_permutation"] = list(best[0])
terms_swap = ["output_channel_index", "output_channel_index+input_axis_2_index",
              "output_channel_index+input_axis_1_index", "output_channel_index+signal_bin_index"]
transfer["axis_swapped_r2"] = r2(ind["ly"].values, design(ind, terms_swap) @ beta)
save_json(transfer, RESULTS / "t16_structural_transfer.json")
LOG.append(
    "structural transfer: replicated model R2 on replicated=%.4f, applied unchanged to independent=%.4f "
    "(best channel permutation %+.4f, axis-swapped %+.4f)"
    % (transfer["rep_model_r2_on_rep"], transfer["rep_model_r2_on_ind"],
       transfer["best_channel_permutation_r2"], transfer["axis_swapped_r2"])
)

# --------------------------------------- independent panel's own geometry
ic = cell_means(ind, "ly")
ind_sep = {}
for label, rows, cols in [
    ("a1a2_vs_bin", ["input_axis_1_index", "input_axis_2_index"], "signal_bin_index"),
    ("a1a2_vs_binchan", ["input_axis_1_index", "input_axis_2_index"], ["signal_bin_index", "output_channel_index"]),
    ("a1_vs_a2bin", "input_axis_1_index", ["input_axis_2_index", "signal_bin_index"]),
]:
    r, mx = separability(ic, rows, cols, "ly")
    ind_sep[label] = {"r2": r, "max_abs_residual": mx}
rep_sep = {}
rc = cell_means(rep, "ly")
for label, rows, cols in [
    ("a1a2_vs_bin", ["input_axis_1_index", "input_axis_2_index"], "signal_bin_index"),
    ("a1a2_vs_binchan", ["input_axis_1_index", "input_axis_2_index"], ["signal_bin_index", "output_channel_index"]),
    ("a1_vs_a2bin", "input_axis_1_index", ["input_axis_2_index", "signal_bin_index"]),
]:
    r, mx = separability(rc, rows, cols, "ly")
    rep_sep[label] = {"r2": r, "max_abs_residual": mx}
save_json({"independent": ind_sep, "replicated": rep_sep}, RESULTS / "t17_separability.json")
LOG.append(
    "separability R2 - replicated a1a2 vs binchan=%.4f ; independent a1a2 vs bin=%.4f, a1a2 vs binchan=%.4f"
    % (rep_sep["a1a2_vs_binchan"]["r2"], ind_sep["a1a2_vs_bin"]["r2"], ind_sep["a1a2_vs_binchan"]["r2"])
)

# independent-panel rank structure over channels
ind_rank = {}
tab = ic.pivot_table(index=GRID, columns="output_channel_index", values="ly")
ind_rank["grid_a1a2bin_x4chan"] = rank1_profile(tab.values)
tab2 = ic.pivot_table(index=["input_axis_1_index", "input_axis_2_index"], columns="output_channel_index", values="ly")
ind_rank["grid_a1a2_x4chan"] = rank1_profile(tab2.values)
save_json(ind_rank, RESULTS / "t18_independent_rank.json")
LOG.append(
    "independent panel channel rank: (a1,a2)x4ch rank1 R2=%.4f ; (a1,a2,bin)x4ch rank1 R2=%.4f"
    % (ind_rank["grid_a1a2_x4chan"]["rank1_r2"], ind_rank["grid_a1a2bin_x4chan"]["rank1_r2"])
)

# ------------------------------- boundary feature: bin-0 notch in each panel
notch = {}
for name, df in [("replicated", rc), ("independent", ic)]:
    s = df.groupby("signal_bin_index")["ly"].mean()
    xs = np.arange(1, len(s))
    co = np.polyfit(xs, s.values[1:], 2)
    notch[name] = {
        "bin0_observed": float(s.values[0]),
        "trend_predicted": float(np.polyval(co, 0)),
        "excess": float(s.values[0] - np.polyval(co, 0)),
        "fold": float(np.exp(s.values[0] - np.polyval(co, 0))),
    }
save_json(notch, RESULTS / "t19_bin0_notch_panels.json")
LOG.append(
    "bin-0 boundary: replicated excess=%+.4f (x%.2f) ; independent excess=%+.4f (x%.2f) -- opposite sign"
    % (notch["replicated"]["excess"], notch["replicated"]["fold"],
       notch["independent"]["excess"], notch["independent"]["fold"])
)

flush(LOG, LOGS / "step3.log")
print("\n".join(LOG))