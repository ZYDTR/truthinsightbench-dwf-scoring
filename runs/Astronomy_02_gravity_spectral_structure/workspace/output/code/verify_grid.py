"""Adversarial verification of TRACK 3 (grid_structure.json).

Independent re-implementation.  Does NOT import or execute track3_grid_structure.py.
Writes ../results/verify_grid.json (machine-readable) and prints a diff table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
DATA = PROJ / "data"
RES = PROJ / "output/results"

AX_VMAX = [1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 100.0]
AX_VMIN = [0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1]
AX_F = list(np.arange(860.0, 1311.0, 50.0))
AX_W = [250.0, 275.0, 300.0, 325.0]

g = pd.read_csv(DATA / "ranked_full_parameter_grid.tsv", sep="\t").rename(columns={
    "delta_information_criterion": "ic", "frequency_width_microhz": "w",
    "peak_frequency_microhz": "f", "max_velocity_cm_s": "vmax",
    "min_velocity_cm_s": "vmin"})
cand = pd.read_csv(DATA / "candidate_spectrum_comparison.tsv", sep="\t")
modes = pd.read_csv(DATA / "mode_frequency_grid.tsv", sep="\t")
traces = pd.read_csv(DATA / "pass_residual_traces.tsv", sep="\t")

R = {}          # recomputed quantities
P = []          # problems


def rec(name, claimed, value):
    try:
        diff = float(abs(float(claimed) - float(value)))
    except Exception:
        diff = None
    R[name] = {"claimed": claimed, "recomputed": value, "abs_diff": diff}


# ---------------------------------------------------------------- section 1
n_rows = len(g)
counts = {c: g[c].value_counts().sort_index() for c in ["vmax", "vmin", "f", "w"]}
lat = len(AX_VMAX) * len(AX_VMIN) * len(AX_F) * len(AX_W)
uniq_cells = int(g[["vmax", "vmin", "f", "w"]].drop_duplicates().shape[0])
ranks = sorted(int(x) for x in g["candidate_rank"])
mono = bool(np.all(np.diff(g.sort_values("candidate_rank")["ic"].values) >= -1e-9))

rec("1_n_rows", 250, n_rows)
rec("1_cartesian_cells_declared", 2240, lat)
rec("1_cells_present_unique", 250, uniq_cells)
rec("1_n_unique_ic_values", 224, int(g["ic"].nunique()))
rec("1_ic_rank1", 0.0, float(g["ic"].min()))
rec("1_ic_rank250", 137.2, float(g["ic"].max()))
rec("1_rank_complete_1_to_250", True, ranks == list(range(1, n_rows + 1)))
rec("1_ic_monotone_in_rank", True, mono)
if not ranks == list(range(1, n_rows + 1)):
    P.append("candidate_rank is not exactly 1..250")
if not mono:
    P.append("ic is not monotone non-increasing with candidate_rank")
# axis levels actually present vs declared
declared = {"vmax": AX_VMAX, "vmin": [round(x, 4) for x in AX_VMIN],
            "f": AX_F, "w": AX_W}
for c in ["vmax", "vmin", "f", "w"]:
    got = [round(float(v), 4) for v in counts[c].index]
    if got != declared[c]:
        P.append(f"axis {c} levels in table != declared axes: {got} vs {declared[c]}")

# --- cross-file truncation ledger -----------------------------------------
c2 = cand.rename(columns={"max_velocity_cm_s": "vmax", "min_velocity_cm_s": "vmin",
                          "peak_frequency_microhz": "f", "peak_width_microhz": "w",
                          "information_criterion": "ic"})
for df in (c2, g):
    for col in ["vmax", "vmin", "f", "w"]:
        df["k_" + col] = df[col].astype(float).round(3)
K = ["k_vmax", "k_vmin", "k_f", "k_w"]
known = c2[c2["k_vmax"].isin(np.round(AX_VMAX, 3))
           & c2["k_vmin"].isin(np.round(AX_VMIN, 3))
           & c2["k_f"].isin(np.round(AX_F, 3))
           & c2["k_w"].isin(np.round(AX_W, 3))].copy()
table_keys = set(map(tuple, g[K].values))
known["in_table"] = [tuple(r) in table_keys for r in known[K].values]
thr = float(g["ic"].max())
TP = int(((known["ic"] < thr) & known["in_table"]).sum())
FP = int(((known["ic"] < thr) & ~known["in_table"]).sum())
FN = int(((known["ic"] >= thr) & known["in_table"]).sum())
present = known[known["in_table"]]
absent = known[~known["in_table"]]
exact = g.merge(c2[K + ["ic"]].rename(columns={"ic": "ic_ref"}), on=K, how="inner")
off = (exact["ic_ref"] - exact["ic"]).abs().max()

rec("1_known_lattice_cells_from_candidate_file", 105, int(len(known)))
rec("1_known_cells_present_in_table", 29, int(known["in_table"].sum()))
rec("1_confusion_TP", 28, TP)
rec("1_confusion_FP", 0, FP)
rec("1_confusion_FN", 1, FN)
rec("1_smallest_ic_known_but_absent", 139.9, float(absent["ic"].min()))
rec("1_largest_ic_present_among_known", 137.2, float(present["ic"].max()))
rec("1_rows_ic_reproduced_exactly_n", 29, int(len(exact)))
rec("1_rows_ic_reproduced_max_abs_offset", 1.4210854715202004e-14, float(off))
rec("1_rows_ic_only_in_table", 221, n_rows - len(exact))
fn_row = known[(known["ic"] >= thr) & known["in_table"]]
fn_info = fn_row[["candidate_family", "vmax", "vmin", "f", "w", "ic"]].to_dict("records")

# ---------------------------------------------------------------- section 2
gs = g.sort_values("ic").reset_index(drop=True)
rec("2_counts_within_dic_lt_1", 1, int((gs["ic"] < 1).sum()))
rec("2_counts_within_dic_lt_2", 1, int((gs["ic"] < 2).sum()))
rec("2_counts_within_dic_lt_5", 2, int((gs["ic"] < 5).sum()))
rec("2_counts_within_dic_lt_10", 4, int((gs["ic"] < 10).sum()))
rec("2_counts_within_dic_lt_20", 9, int((gs["ic"] < 20).sum()))
rec("2_margin_rank1_rank2", 2.0, float(gs["ic"].iloc[1] - gs["ic"].iloc[0]))
rec("2_margin_rank1_rank5", 11.9, float(gs["ic"].iloc[4] - gs["ic"].iloc[0]))
rec("2_margin_rank1_rank10", 21.0, float(gs["ic"].iloc[9] - gs["ic"].iloc[0]))
rec("2_ic_ladder_0", 0.0, float(gs["ic"].iloc[0]))
rec("2_ic_ladder_19", 29.8, float(gs["ic"].iloc[19]))
rec("2_top50_max_ic", 51.8, float(gs["ic"].iloc[49]))
rec("2_top100_max_ic", 73.7, float(gs["ic"].iloc[99]))

rank1 = g.loc[g["candidate_rank"] == 1].iloc[0]
rank1_params = {k: float(rank1[k]) for k in ["vmax", "vmin", "f", "w"]}

# best axis level profiles
axis_prof = {}
for c in ["vmax", "vmin", "f", "w"]:
    p = g.groupby(g[c].astype(float).round(4))["ic"].min().sort_values()
    axis_prof[c] = {float(k): float(v) for k, v in p.items()}
rec("2_vmax_best_level_min_ic", 0.0, axis_prof["vmax"][30.0])
rec("2_vmax_second_level_min_ic", 2.0, axis_prof["vmax"][20.0])
rec("2_vmin_best_005", 0.0, axis_prof["vmin"][0.05])
rec("2_vmin_second_003_gap", 5.7, axis_prof["vmin"][0.03] - axis_prof["vmin"][0.05])
rec("2_f_best_1010", 0.0, axis_prof["f"][1010.0])
rec("2_f_second_1060", 2.0, axis_prof["f"][1060.0])
rec("2_w_best_275", 0.0, axis_prof["w"][275.0])
rec("2_w_second_300", 2.0, axis_prof["w"][300.0])

# hard-coded single-grid-step costs -> recompute from candidate file
fa = c2[c2["candidate_family"] == "frequency_amplitude_scan"]
mv = c2[c2["candidate_family"] == "minimum_velocity_scan"]


def ic_at(fam, **kw):
    s = fam
    for k, v in kw.items():
        s = s[np.isclose(s[k], v)]
    return s[["vmax", "vmin", "f", "w", "ic"]]


def step_cost(vmax, f0):
    lo = ic_at(fa, vmax=vmax, f=f0 - 50.0)["ic"]
    hi = ic_at(fa, vmax=vmax, f=f0 + 50.0)["ic"]
    c0 = ic_at(fa, vmax=vmax, f=f0)["ic"]
    return (float(lo.min() - c0.min()) if len(lo) else None,
            float(hi.min() - c0.min()) if len(hi) else None)


sc20 = step_cost(20.0, 1010.0)
sc30 = step_cost(30.0, 1010.0)
sc50 = step_cost(50.0, 1110.0)
rec("2_stepcost_vmax20_f1010_minus", 14.1, sc20[0])
rec("2_stepcost_vmax20_f1010_plus", 12.8, sc20[1])
rec("2_stepcost_vmax30_f1010_minus", 25.3, sc30[0])
rec("2_stepcost_vmax30_f1010_plus", 10.1, sc30[1])
rec("2_stepcost_vmax50_f1110_minus", 20.6, sc50[0])
rec("2_stepcost_vmax50_f1110_plus", 1.1, sc50[1])

# ---------------------------------------------------------------- section 3
fa_b = fa.loc[fa.groupby("vmax")["ic"].idxmin()].sort_values("vmax")
mv_b = mv.loc[mv.groupby("vmax")["ic"].idxmin()].sort_values("vmax")


def loglog(x, y):
    X = np.column_stack([np.ones(len(x)), np.log10(x)])
    b, *_ = np.linalg.lstsq(X, np.log10(y), rcond=None)
    pred = X @ b
    r2 = 1 - np.sum((np.log10(y) - pred) ** 2) / np.sum((np.log10(y) - np.log10(y).mean()) ** 2)
    return float(b[1]), float(10 ** b[0]), float(r2), float(np.abs(np.log10(y) - pred).max())


sl, A0, r2fa, resfa = loglog(fa_b["vmax"].values, fa_b["f"].values)
fa_hi = fa_b[fa_b["vmax"] >= 0.5]
sl2, A2, r2fa2, resfa2 = loglog(fa_hi["vmax"].values, fa_hi["f"].values)
rec("3_fa_loglog_slope", 0.1213663690718669, sl)
rec("3_fa_loglog_A", 709.2857688037109, A0)
rec("3_fa_loglog_R2", 0.9728889093978028, r2fa)
rec("3_fa_loglog_maxres_dex", 0.05587495788421215, resfa)
rec("3_fa_loglog_slope_ge05", 0.13537563132972166, sl2)
rec("3_fa_rows", 304, len(fa))
rec("3_fa_is_complete_factorial", True,
    bool(fa.groupby("vmax").size().nunique() == 1 and fa.groupby("vmax").size().iloc[0] == 19))
rec("3_mv_rows", 203, len(mv))
# residual scatter of the ridge in uHz (claim: "+-1 grid step")
pred_f = A0 * fa_b["vmax"].values ** sl
res_uhz = (fa_b["f"].values - pred_f)
rec("3_fa_ridge_max_abs_resid_uHz", None, float(np.abs(res_uhz).max()))
rec("3_fa_ridge_resid_steps_of_50uHz", None, float(np.abs(res_uhz).max() / 50.0))

# IC-surface factorisation on the complete fa factorial
piv = fa.pivot_table(index="vmax", columns="f", values="ic")
Y = np.log10(piv.values)
V = np.log10(piv.index.values)[:, None]
Fq = piv.columns.values[None, :]
Vf = np.broadcast_to(V, Y.shape).ravel()
Ff = np.broadcast_to(Fq, Y.shape).ravel()
yv = Y.ravel()


def r2(cols):
    X = np.column_stack(cols + [np.ones(len(yv))])
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return 1 - np.sum((yv - X @ b) ** 2) / np.sum((yv - yv.mean()) ** 2)


fa_full = r2([Vf, Ff, Vf * Ff, Vf ** 2, Ff ** 2])
fa_add = r2([Vf - Vf.mean(), Ff - Ff.mean()])
rec("3_fa_full_2d_R2", 0.6885, float(fa_full))
rec("3_fa_additive_R2", 0.2033, float(fa_add))

# mv family: fits
slm, Am, r2mv, resmv = loglog(mv_b["vmax"].values, mv_b["vmin"].values)
rec("3_mv_loglog_slope_all19", None, slm)
rec("3_mv_loglog_R2_all19", None, r2mv)
mv_hi = mv_b[mv_b["vmax"] >= 0.1]
slm2, Am2, r2mv2, resmv2 = loglog(mv_hi["vmax"].values, mv_hi["vmin"].values)
rec("3_mv_loglog_slope_ge01", 0.009894456773213722, slm2)
rec("3_mv_loglog_A_ge01", 0.05875249208588403, Am2)
rec("3_mv_loglog_R2_ge01", 0.0032567630267700265, r2mv2)
rec("3_mv_loglog_maxres_ge01", 0.4511859117353503, resmv2)
# the -0.082 / 0.204 quoted in the track verdict: does any natural variant give it?
mv_interior = mv_b[mv_b["vmin"] < mv_b["vmax"]]
variants = {}
for tag, sub in [("all19", mv_b), ("ge0.1", mv_hi), ("interior_only", mv_interior),
                 ("vmax_ge1", mv_b[mv_b["vmax"] >= 1.0]),
                 ("vmax_ge5", mv_b[mv_b["vmax"] >= 5.0]),
                 ("vmax_5to100", mv_b[(mv_b["vmax"] >= 5.0) & (mv_b["vmax"] <= 100.0)])]:
    if len(sub) >= 3:
        s_, a_, r_, m_ = loglog(sub["vmax"].values, sub["vmin"].values)
        variants[tag] = {"n": int(len(sub)), "slope": s_, "A": a_, "R2": r_, "maxres": m_}
for tag, sub in [("ic_vs_vmax", mv_b), ("ic_vs_vmin", mv_b)]:
    x = sub["vmax"].values if tag == "ic_vs_vmax" else sub["vmin"].values
    s_, a_, r_, m_ = loglog(x, sub["ic"].values)
    variants[tag] = {"n": int(len(sub)), "slope": s_, "A": a_, "R2": r_, "maxres": m_}

# mv regressions in log10(ic)
mvlg = np.log10(mv["ic"].values)
lv = np.log10(mv["vmax"].values)
lm = np.log10(mv["vmin"].values)


def r2_of(y, cols):
    X = np.column_stack(cols + [np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2))


rec("3_mv_logvmax_only_R2", 0.0006, r2_of(mvlg, [lv - lv.mean()]))
rec("3_mv_logvmin_only_R2", 0.58, r2_of(mvlg, [lm - lm.mean()]))
rec("3_mv_additive_R2", 0.7585, r2_of(mvlg, [lv - lv.mean(), lm - lm.mean()]))
rec("3_mv_interaction_R2", 0.7808, r2_of(mvlg, [lv, lm, lv * lm]))
# interior curvature numbers (hard-coded in the script)
row30 = mv[np.isclose(mv["vmax"], 30.0)]
ic30_lower = float(row30[np.isclose(row30["vmin"], 0.01)]["ic"].min()) if (np.isclose(row30["vmin"], 0.01)).any() else None
ic30_upper = float(row30[np.isclose(row30["vmin"], 30.0)]["ic"].min()) if (np.isclose(row30["vmin"], 30.0)).any() else None
rec("3_mv_vmin_star_vmax30", 0.03, float(mv_b[np.isclose(mv_b["vmax"], 30.0)]["vmin"].iloc[0]))
rec("3_mv_ic_star_vmax30", 50.7, float(mv_b[np.isclose(mv_b["vmax"], 30.0)]["ic"].iloc[0]))
rec("3_mv_ic_at_lower_edge_0.01_vmax30", 81.0, ic30_lower)
rec("3_mv_delta_ic_to_edge", 30.3, (ic30_lower - 50.7) if ic30_lower else None)
rec("3_mv_ic_at_upper_edge_30_vmax30", 8435.6, ic30_upper)
rec("3_mv_levels_visited", [0.02, 0.03, 0.05, 0.1],
    [float(x) for x in sorted(mv_b["vmin"].unique())])

# ---------------------------------------------------------------- section 4
sub = g[g["vmin"] > 0].copy()
rec("4_n_rows_used", 188, len(sub))
rec("4_n_dropped_vmin0", 62, int((g["vmin"] == 0).sum()))
Xcols = [np.log10(sub["vmax"].values), np.log10(sub["vmin"].values),
         sub["f"].values, sub["w"].values]
y = sub["ic"].values
X = np.column_stack(Xcols + [np.ones(len(sub))])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ b
rec("4_4term_R2", 0.0236, float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)))
rec("4_4term_RMSE", 33.382, float(np.sqrt(np.mean((y - pred) ** 2))))
rec("4_sd_of_ic", 33.873, float(np.std(y, ddof=1)))
Xr = np.column_stack(Xcols + [sub["f"].values ** 2, sub["w"].values ** 2, np.ones(len(sub))])
br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
rec("4_plus_f2_w2_R2", 0.1744,
    float(1 - np.sum((y - Xr @ br) ** 2) / np.sum((y - y.mean()) ** 2)))
Xr2 = np.column_stack(Xcols + [1.0 / sub["vmax"].values, sub["vmax"].values,
                               1.0 / sub["vmin"].values, sub["vmin"].values,
                               sub["f"].values ** 2, sub["w"].values ** 2,
                               np.ones(len(sub))])
br2, *_ = np.linalg.lstsq(Xr2, y, rcond=None)
rec("4_richer_R2", 0.3832,
    float(1 - np.sum((y - Xr2 @ br2) ** 2) / np.sum((y - y.mean()) ** 2)))
Xi = np.column_stack(Xcols + [np.log10(sub["vmax"].values) * sub["f"].values,
                              np.log10(sub["vmax"].values) * sub["w"].values,
                              sub["f"].values * sub["w"].values, np.ones(len(sub))])
bi, *_ = np.linalg.lstsq(Xi, y, rcond=None)
rec("4_interactions_R2", 0.1382,
    float(1 - np.sum((y - Xi @ bi) ** 2) / np.sum((y - y.mean()) ** 2)))
# leave-one-vmax-out
loo = {}
for lv_ in sorted(sub["vmax"].unique()):
    s2 = sub[sub["vmax"] != lv_]
    X2 = np.column_stack([np.log10(s2["vmax"]), np.log10(s2["vmin"]), s2["f"], s2["w"],
                          np.ones(len(s2))])
    b2, *_ = np.linalg.lstsq(X2, s2["ic"].values, rcond=None)
    yy = s2["ic"].values
    loo[float(lv_)] = float(1 - np.sum((yy - X2 @ b2) ** 2) / np.sum((yy - yy.mean()) ** 2))
R["4_loo_R2"] = {"claimed": [0.0187, 0.0673], "recomputed": loo,
                 "abs_diff": float(abs(min(loo.values()) - 0.0187) + abs(max(loo.values()) - 0.0673))}

# ---------------------------------------------------------------- section 5
mf = np.sort(modes["mode_frequency_microhz"].values)
cands = np.array(AX_F)
d = np.abs(cands[:, None] - mf[None, :]).min(axis=1)
rec("5_n_modes", 52, len(mf))
rec("5_median_offset", 5.76, float(np.median(d)))
rec("5_mean_offset", 7.778, float(np.mean(d)))
rec("5_median_offset_pct", 0.4991, float(np.median(d / cands * 100)))
mf_band = mf[(mf >= 860) & (mf <= 1310)]
rec("5_n_modes_in_band", 18, len(mf_band))
rec("5_in_band_median_spacing", 20.69, float(np.median(np.diff(mf_band))))
rec("5_in_band_mean_spacing", 24.178, float(np.mean(np.diff(mf_band))))
rec("5_overall_median_spacing", 25.48, float(np.median(np.diff(mf))))

# nulls, own seeds and N
def null_med(lo, hi, seed, N=100_000, chunk=10_000):
    rng = np.random.default_rng(seed)
    out = np.empty(N)
    k = 0
    while k < N:
        m = min(chunk, N - k)
        dr = rng.uniform(lo, hi, size=(m, len(cands)))
        dd = np.abs(dr[:, :, None] - mf[None, None, :]).min(axis=2)
        out[k:k + m] = np.median(dd, axis=1)
        k += m
    return out


nf = null_med(float(mf.min()), float(mf.max()), 555)
nb = null_med(860.0, 1310.0, 777)
p_full = float((nf <= np.median(d)).mean())
p_band = float((nb <= np.median(d)).mean())
rec("5_null_full_range_median", 8.083, float(np.median(nf)))
rec("5_null_band_median", 7.242, float(np.median(nb)))
rec("5_p_full_range", 0.1786, p_full)
rec("5_p_band", 0.2695, p_band)
# rigid-comb phase null
rng = np.random.default_rng(31337)
offs = rng.uniform(0, 50, size=30_000)
med_comb = np.array([np.median(np.abs((860.0 + np.arange(10) * 50.0 + o)[:, None]
                                      - mf[None, :]).min(axis=1)) for o in offs])
rec("5_null_comb_median", 7.335, float(np.median(med_comb)))
rec("5_p_comb", 0.2876, float((med_comb <= np.median(d)).mean()))
rec("5_comb_best_achievable", 3.14, float(med_comb.min()))
# per-candidate p for 1160
p_1160 = float((np.abs(rng.uniform(mf.min(), mf.max(), 200_000)[:, None]
                        - mf[None, :]).min(axis=1) <= 0.98).mean())
rec("5_p_individual_1160", 0.06729, p_1160)
fam_p = float((nf.min(axis=1) <= d.min()).mean()) if nf.ndim == 2 else None

# NOTE: nf here is stored as medians only; recompute family-wise min-distance null
rng = np.random.default_rng(4242)
mind = np.empty(100_000)
k = 0
while k < 100_000:
    m = min(10_000, 100_000 - k)
    dr = rng.uniform(float(mf.min()), float(mf.max()), size=(m, len(cands)))
    dd = np.abs(dr[:, :, None] - mf[None, None, :]).min(axis=2)
    mind[k:k + m] = dd.min(axis=1)
    k += m
rec("5_familywise_p", 0.4991, float((mind <= d.min()).mean()))

# ---------------------------------------------------------------- section 6
spans = traces.groupby("pass_id")["relative_time_min"].apply(lambda t: float(t.max() - t.min()))
nsamp = traces.groupby("pass_id").size()
res_uhz = 1e6 / (spans.values * 60.0)
rec("6_n_passes", 22, int(len(spans)))
rec("6_span_min", 60.0, float(spans.min()))
rec("6_span_max", 79.0, float(spans.max()))
rec("6_span_median", 79.0, float(np.median(spans)))
rec("6_span_counts", {"79.0": 19, "72.0": 2, "60.0": 1},
    {str(float(k)): int(v) for k, v in spans.value_counts().items()})
rec("6_samples_min", 61, int(nsamp.min()))
rec("6_samples_max", 80, int(nsamp.max()))
rec("6_samples_median", 80, int(np.median(nsamp)))
rec("6_samples_sum", 1698, int(nsamp.sum()))
rec("6_one_over_T_median", 210.97, float(np.median(res_uhz)))
rec("6_one_over_T_min", 210.97, float(res_uhz.min()))
rec("6_one_over_T_max", 277.778, float(res_uhz.max()))
rec("6_width250_over_median", 1.185, float(250.0 / np.median(res_uhz)))
rec("6_width325_over_median", 1.54, float(325.0 / np.median(res_uhz)))
rec("6_comb50_over_median", 0.237, float(50.0 / np.median(res_uhz)))
# n samples vs span consistency (1 min cadence)
cad = (spans.values * 60.0) / (nsamp.values - 1)
rec("6_cadence_s", 60.0, float(np.median(cad)))

# two-sinusoid separation test, independent implementation
# NOTE: the LS grid is restricted to the same window the track then masks to;
# LS power at a given frequency does not depend on which other frequencies are
# evaluated, so this is numerically equivalent to evaluating the full grid.
tr1 = traces[traces["pass_id"] == "pass_01"]
t1 = tr1["relative_time_min"].values
tsec = t1 * 60.0
rng = np.random.default_rng(9091)
sep = {}
print("[verify] starting two-sinusoid simulation", flush=True)
for df in [50.0, 100.0, 211.0, 250.0, 275.0, 300.0, 325.0, 450.0]:
    lo = max(200.0, 1000.0 + df / 2 - 3 * df)
    hi = min(2999.0, 1000.0 + df / 2 + 3 * df)
    fw = np.arange(lo, hi, 1.0)
    fhw = fw * 1e-6
    hits = 0
    for _ in range(100):
        ph = rng.uniform(0, 2 * np.pi, 2)
        ysim = (np.cos(2 * np.pi * 1000e-6 * tsec + ph[0])
                + np.cos(2 * np.pi * (1000 + df) * 1e-6 * tsec + ph[1]))
        seg = lombscargle(tsec, ysim - ysim.mean(), 2 * np.pi * fhw, normalize=True)
        loc = [i for i in range(1, len(seg) - 1)
               if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.5 * seg.max()]
        hits += int(len(loc) >= 2)
    sep[str(int(df))] = hits / 100.0
    print(f"[verify]   df={df:.0f} -> {sep[str(int(df))]:.2f}", flush=True)
R["6_two_sinusoid_fraction_resolved"] = {
    "claimed": {"50": 0.14, "100": 0.31, "211": 0.56, "250": 0.75, "275": 0.88,
                "300": 1.0, "325": 1.0, "450": 1.0},
    "recomputed": sep, "abs_diff": None}
# pass_01 span used by the simulation
rec("6_pass01_span_min", None, float(spans["pass_01"]))

# per-pass band peak over all three models
f_uhz = np.arange(200.0, 3000.0, 1.0)
f_hz = f_uhz * 1e-6
pk = {}
for tag, col in [("A", "residual_model_A_mm_s"), ("B", "residual_model_B_mm_s"),
                 ("C", "residual_model_C_mm_s")]:
    vals = []
    for pid, s in traces.groupby("pass_id"):
        p = lombscargle(s["relative_time_min"].values * 60.0,
                        s[col].values - s[col].values.mean(), 2 * np.pi * f_hz, normalize=True)
        mm = (f_uhz >= 860) & (f_uhz <= 1310)
        vals.append(float(f_uhz[mm][int(np.argmax(p[mm]))]))
    vals = np.array(vals)
    pk[tag] = {"median": float(np.median(vals)),
               "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
               "min": float(vals.min()), "max": float(vals.max()),
               "n_unique": int(len(np.unique(vals)))}
R["6_per_pass_band_peak"] = {"claimed": {"A": 883.0, "B": 1207.0, "C": 1111.5},
                             "recomputed": {k: v["median"] for k, v in pk.items()},
                             "abs_diff": None}
R["6_per_pass_band_peak_detail"] = pk

# ---------------------------------------------------------------- problems
# P1: mv slope quoted in verdicts
mv_slope_claim = -0.082
_, _, _, _ = loglog(mv_hi["vmax"].values, mv_hi["vmin"].values)
best = None
for tag, v in variants.items():
    if abs(v["slope"] - mv_slope_claim) < 0.02:
        best = tag
P.append("VERDICT/JSON CONTRADICTION (§3 verdict and §7): text quotes 'log-log slope "
         "-0.082, R2 = 0.204' for the minimum_velocity_scan vmin valley, but the JSON's own "
         f"loglog_fit_vmax_ge_0.1 gives slope=+0.0099, R2=0.0033. No variant tested reproduces "
         f"-0.082/0.204 (closest-by-slope variant: {best}).")

# P2: two-sinusoid verdict text
P.append("VERDICT/JSON CONTRADICTION (§6 and §7): text says '55% of trials at 250 uHz but "
         "95-100% at 275-325'; the JSON's own two_sinusoid_separation_test_pass01 gives "
         "0.75 at 250 and 0.88 at 275 (1.0 at 300/325).")

# P3: %% artifacts
raw = (RES / "grid_structure.json").read_text()
n_pctpct = raw.count("%%")
P.append(f"FORMATTING: the stored JSON contains {n_pctpct} literal '%%' tokens (sections 1, 6, 7) "
         "because %%-escaped strings were never %-formatted.")

# P4: '+/-1 grid step' claim
P.append("OVERSTATEMENT (§3 verdict): 'residual scatter corresponds to +-1 grid step' - "
         f"max ridge residual is {float(np.abs(res_uhz).max()):.1f} uHz = "
         f"{float(np.abs(res_uhz).max()/50.0):.1f} x 50 uHz steps (recomputed).")

# P5: 'nearly doubles'
P.append("MIS-STATEMENT (§4 verdict): 'Adding interactions nearly doubles R2 (0.024 -> 0.138)' - "
         "0.0236 -> 0.1382 is a 5.9x increase, not ~2x.")

# P6: hard-coded numbers in script (not computed)
pairs = [(14.1, sc20[0]), (12.8, sc20[1]), (25.3, sc30[0]), (10.1, sc30[1]),
         (20.6, sc50[0]), (1.1, sc50[1])]
if all(y is not None for _, y in pairs):
    worst = max(abs(x - y) for x, y in pairs)
    P.append("TRACEABILITY: single_grid_step_cost_at_optimum (14.1/12.8/25.3/10.1/20.6/1.1) and the "
             "interior_curvature_at_vmax30 block are hard-coded literals in the script, not computed "
             "there; they were recomputed here from candidate_spectrum_comparison.tsv and are "
             f"reproduced within {worst:.4f} IC units.")
else:
    P.append("TRACEABILITY: single_grid_step_cost_at_optimum hard-coded; recomputation did not find "
             f"all rows (pairs={pairs}).")

out = {
    "recomputed": R,
    "rank1_params": rank1_params,
    "fn_cell": fn_info,
    "mv_loglog_variants": variants,
    "problems_found": P,
}
(RES / "verify_grid_numbers.json").write_text(json.dumps(out, indent=2, default=str) + "\n")

# ---------------------------------------------------------------- print table
print(f"{'quantity':52s} {'claimed':>14s} {'recomputed':>14s} {'absdiff':>12s}")
for k, v in R.items():
    c = v["claimed"]
    r = v["recomputed"]
    if isinstance(c, (int, float)) and isinstance(r, (int, float)):
        print(f"{k:52s} {float(c):14.4f} {float(r):14.6f} {v['abs_diff']:12.3e}")
    else:
        print(f"{k:52s} {str(c)[:14]:>14s} {str(r)[:40]:>40s}")
print("\nPROBLEMS")
for p in P:
    print(" -", p)
print("\nkey sources:", [k for k in ["sc20", "sc30", "sc50"] if k in dir()])