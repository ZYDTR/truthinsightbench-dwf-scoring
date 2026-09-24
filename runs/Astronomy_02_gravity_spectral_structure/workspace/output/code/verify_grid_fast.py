"""Fast subset of the TRACK 3 verification: everything except LS-heavy simulations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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

R, P = {}, []


def rec(name, claimed, value):
    if value is None:
        diff = None
    else:
        try:
            diff = float(abs(float(claimed) - float(value)))
        except Exception:
            diff = None
    R[name] = {"claimed": claimed, "recomputed": value, "abs_diff": diff}


# ---- grid integrity
rec("1_n_rows", 250, len(g))
rec("1_cartesian_cells_declared", 2240, 7 * 8 * 10 * 4)
rec("1_unique_cells", 250, int(g[["vmax", "vmin", "f", "w"]].drop_duplicates().shape[0]))
rec("1_ic_rank1", 0.0, float(g["ic"].min()))
rec("1_ic_rank250", 137.2, float(g["ic"].max()))
rec("1_n_unique_ic", 224, int(g["ic"].nunique()))
rec("1_rank_is_1_to_250", True,
    sorted(int(x) for x in g["candidate_rank"]) == list(range(1, 251)))
rec("1_ic_monotone", True,
    bool(np.all(np.diff(g.sort_values("candidate_rank")["ic"].values) >= -1e-9)))
for c, want in [("vmax", AX_VMAX), ("vmin", AX_VMIN), ("f", AX_F), ("w", AX_W)]:
    got = sorted(set(np.round(g[c].astype(float), 4)))
    if got != sorted(np.round(want, 4)):
        P.append(f"axis {c} levels in table != declared axes")

# ---- truncation ledger
c2 = cand.rename(columns={"max_velocity_cm_s": "vmax", "min_velocity_cm_s": "vmin",
                          "peak_frequency_microhz": "f", "peak_width_microhz": "w",
                          "information_criterion": "ic"})
for df in (c2, g):
    for col in ["vmax", "vmin", "f", "w"]:
        df["k_" + col] = df[col].astype(float).round(3)
K = ["k_vmax", "k_vmin", "k_f", "k_w"]
known = c2[c2["k_vmax"].isin(np.round(AX_VMAX, 3)) & c2["k_vmin"].isin(np.round(AX_VMIN, 3))
           & c2["k_f"].isin(np.round(AX_F, 3)) & c2["k_w"].isin(np.round(AX_W, 3))].copy()
keys = set(map(tuple, g[K].values))
known["in_table"] = [tuple(r) in keys for r in known[K].values]
thr = float(g["ic"].max())
present, absent = known[known["in_table"]], known[~known["in_table"]]
fn = known[(known["ic"] >= thr) & known["in_table"]]
rec("1_known_lattice_cells", 105, int(len(known)))
rec("1_known_present", 29, int(len(present)))
rec("1_TP", 28, int(((known["ic"] < thr) & known["in_table"]).sum()))
rec("1_FP", 0, int(((known["ic"] < thr) & ~known["in_table"]).sum()))
rec("1_FN", 1, int(len(fn)))
rec("1_smallest_ic_absent", 139.9, float(absent["ic"].min()))
rec("1_largest_ic_present", 137.2, float(present["ic"].max()))
rec("1_n_known_lattice_cells_in_cand", None, int(c2.shape[0]))
# how many distinct lattice cells does the candidate file itself cover?
c2k = c2[c2["k_vmax"].isin(np.round(AX_VMAX, 3)) & c2["k_vmin"].isin(np.round(AX_VMIN, 3))
         & c2["k_f"].isin(np.round(AX_F, 3)) & c2["k_w"].isin(np.round(AX_W, 3))]
rec("1_distinct_lattice_cells_in_candidate_file", 105,
    int(c2k[K].drop_duplicates().shape[0]))
# which cells are in the candidate file at all? fraction of full lattice evaluated
allcells = c2k[K].drop_duplicates()
rec("1_candidate_file_covers_lattice_fraction", None, round(len(allcells) / 2240.0, 4))
rec("1_FN_cell", None, fn[["candidate_family", "vmax", "vmin", "f", "w", "ic"]].to_dict("records"))
# how many of the 1990 missing lattice cells does the candidate file rule in/out?
missing = 2240 - len(allcells)
rec("1_lattice_cells_never_evaluated", None, int(missing))

# ---- separation
gs = g.sort_values("ic").reset_index(drop=True)
for t in [1, 2, 5, 10, 20, 50, 100]:
    rec(f"2_counts_dIC_lt_{t}", {1: 1, 2: 1, 5: 2, 10: 4, 20: 9, 50: 46, 100: 162}[t],
        int((gs["ic"] < t).sum()))
rec("2_margin_r1_r2", 2.0, float(gs["ic"].iloc[1] - gs["ic"].iloc[0]))
rec("2_margin_r1_r5", 11.9, float(gs["ic"].iloc[4] - gs["ic"].iloc[0]))
rec("2_margin_r1_r10", 21.0, float(gs["ic"].iloc[9] - gs["ic"].iloc[0]))
rec("2_top50_max_ic", 51.8, float(gs["ic"].iloc[49]))
rec("2_top100_max_ic", 73.7, float(gs["ic"].iloc[99]))
rank1 = g.loc[g["candidate_rank"] == 1].iloc[0]


# ---- step costs and interior curvature, recomputed from candidate file
def icsel(fam, **kw):
    s = fam
    for k, v in kw.items():
        s = s[np.isclose(s[k].astype(float), v)]
    return s


fa = c2[c2["candidate_family"] == "frequency_amplitude_scan"]
mv = c2[c2["candidate_family"] == "minimum_velocity_scan"]


def cost(vmax, f0):
    c0 = icsel(fa, vmax=vmax, f=f0)["ic"].min()
    lo = icsel(fa, vmax=vmax, f=f0 - 50.0)["ic"].min()
    hi = icsel(fa, vmax=vmax, f=f0 + 50.0)["ic"].min()
    return float(lo - c0), float(hi - c0)


pairs = {"vmax20_f1010": (14.1, 12.8, cost(20.0, 1010.0)),
         "vmax30_f1010": (25.3, 10.1, cost(30.0, 1010.0)),
         "vmax50_f1110": (20.6, 1.1, cost(50.0, 1110.0))}
for k, (cl, ch, (rl, rh)) in pairs.items():
    rec(f"2_stepcost_{k}_minus", cl, rl)
    rec(f"2_stepcost_{k}_plus", ch, rh)
row30 = icsel(mv, vmax=30.0)
mv_b = mv.loc[mv.groupby("vmax")["ic"].idxmin()].sort_values("vmax")
fa_b = fa.loc[fa.groupby("vmax")["ic"].idxmin()].sort_values("vmax")
b30 = mv_b[np.isclose(mv_b["vmax"], 30.0)].iloc[0]
rec("3_mv_vmin_star_vmax30", 0.03, float(b30["vmin"]))
rec("3_mv_ic_star_vmax30", 50.7, float(b30["ic"]))
rec("3_mv_ic_at_0.01_vmax30", 81.0, float(row30[np.isclose(row30["vmin"], 0.01)]["ic"].min()))
rec("3_mv_ic_at_30_vmax30", 8435.6, float(row30[np.isclose(row30["vmin"], 30.0)]["ic"].min()))

# ---- fa ridge fit
def loglog(x, y):
    X = np.column_stack([np.ones(len(x)), np.log10(x)])
    b, *_ = np.linalg.lstsq(X, np.log10(y), rcond=None)
    pr = X @ b
    r2 = 1 - np.sum((np.log10(y) - pr) ** 2) / np.sum((np.log10(y) - np.log10(y).mean()) ** 2)
    return float(b[1]), float(10 ** b[0]), float(r2), float(np.abs(np.log10(y) - pr).max())


sl, A0, r2fa, resfa = loglog(fa_b["vmax"].values, fa_b["f"].values)
fa_hi = fa_b[fa_b["vmax"] >= 0.5]
sl2, A2, r2fa2, resfa2 = loglog(fa_hi["vmax"].values, fa_hi["f"].values)
rec("3_fa_slope", 0.1213663690718669, sl)
rec("3_fa_A", 709.2857688037109, A0)
rec("3_fa_R2", 0.9728889093978028, r2fa)
rec("3_fa_maxres_dex", 0.05587495788421215, resfa)
rec("3_fa_slope_ge05", 0.13537563132972166, sl2)
rec("3_fa_R2_ge05", 0.9881479173689189, r2fa2)
rec("3_fa_n_rows", 304, len(fa))
rec("3_fa_complete_factorial", True,
    bool(fa.groupby("vmax").size().nunique() == 1 and fa.groupby("vmax").size().iloc[0] == 19))
rec("3_mv_n_rows", 203, len(mv))
resid_uhz = fa_b["f"].values - A0 * fa_b["vmax"].values ** sl
rec("3_fa_ridge_max_abs_resid_uHz", None, float(np.abs(resid_uhz).max()))
rec("3_fa_ridge_max_resid_steps", None, float(np.abs(resid_uhz).max() / 50.0))

# fa factorisation
piv = fa.pivot_table(index="vmax", columns="f", values="ic")
Y = np.log10(piv.values)
Vf = np.broadcast_to(np.log10(piv.index.values)[:, None], Y.shape).ravel()
Ff = np.broadcast_to(piv.columns.values[None, :], Y.shape).ravel()
yv = Y.ravel()


def r2c(cols):
    X = np.column_stack(cols + [np.ones(len(yv))])
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return float(1 - np.sum((yv - X @ b) ** 2) / np.sum((yv - yv.mean()) ** 2))


rec("3_fa_full_2d_R2", 0.6885, r2c([Vf, Ff, Vf * Ff, Vf ** 2, Ff ** 2]))
rec("3_fa_additive_R2", 0.2033, r2c([Vf - Vf.mean(), Ff - Ff.mean()]))

# mv loglog variants
slm, Am, r2mv, resmv = loglog(mv_b["vmax"].values, mv_b["vmin"].values)
rec("3_mv_slope_all19", None, slm)
rec("3_mv_R2_all19", None, r2mv)
mv_hi = mv_b[mv_b["vmax"] >= 0.1]
s_, a_, r_, m_ = loglog(mv_hi["vmax"].values, mv_hi["vmin"].values)
rec("3_mv_slope_ge01", 0.009894456773213722, s_)
rec("3_mv_A_ge01", 0.05875249208588403, a_)
rec("3_mv_R2_ge01", 0.0032567630267700265, r_)
rec("3_mv_maxres_ge01", 0.4511859117353503, m_)
variants = {}
for tag, sub in [("all19", mv_b), ("ge0.1", mv_hi),
                 ("interior_vmin<vmax", mv_b[mv_b["vmin"] < mv_b["vmax"]]),
                 ("vmax>=1", mv_b[mv_b["vmax"] >= 1.0]),
                 ("vmax>=5", mv_b[mv_b["vmax"] >= 5.0]),
                 ("icstar_vs_vmax", mv_b)]:
    x, y = ((sub["vmax"].values, sub["ic"].values) if tag == "icstar_vs_vmax"
            else (sub["vmax"].values, sub["vmin"].values))
    s2, a2, r2_, m2 = loglog(x, y)
    variants[tag] = {"n": int(len(sub)), "slope": s2, "A": a2, "R2": r2_, "maxres": m2}
# niche: vmin_star vs vmax only where the star is not on the lower edge of the
# visited set (i.e. an interior optimum)
rec("3_mv_slope_variants", None, variants)

# mv regressions
mvlg = np.log10(mv["ic"].values)
lv = np.log10(mv["vmax"].values)
lm = np.log10(mv["vmin"].values)


def r2of(y, cols):
    X = np.column_stack(cols + [np.ones(len(y))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2))


rec("3_mv_logvmax_R2", 0.0006, round(r2of(mvlg, [lv - lv.mean()]), 4))
rec("3_mv_logvmin_R2", 0.58, round(r2of(mvlg, [lm - lm.mean()]), 4))
rec("3_mv_additive_R2", 0.7585, round(r2of(mvlg, [lv - lv.mean(), lm - lm.mean()]), 4))
rec("3_mv_interaction_R2", 0.7808, round(r2of(mvlg, [lv, lm, lv * lm]), 4))

# ---- ranked-grid separability
sub = g[g["vmin"] > 0].copy()
y = sub["ic"].values
rec("4_n_used", 188, len(sub))
rec("4_n_dropped", 62, int((g["vmin"] == 0).sum()))
Xb = np.column_stack([np.log10(sub["vmax"]), np.log10(sub["vmin"]), sub["f"], sub["w"],
                      np.ones(len(sub))])
b, *_ = np.linalg.lstsq(Xb, y, rcond=None)
rec("4_4term_R2", 0.0236, round(float(1 - np.sum((y - Xb @ b) ** 2) / np.sum((y - y.mean()) ** 2)), 4))
rec("4_4term_RMSE", 33.382, round(float(np.sqrt(np.mean((y - Xb @ b) ** 2))), 3))
rec("4_sd_ic", 33.873, round(float(np.std(y, ddof=1)), 3))
Xr = np.column_stack([np.log10(sub["vmax"]), np.log10(sub["vmin"]), sub["f"], sub["w"],
                      sub["f"] ** 2, sub["w"] ** 2, np.ones(len(sub))])
br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
rec("4_f2w2_R2", 0.1744, round(float(1 - np.sum((y - Xr @ br) ** 2) / np.sum((y - y.mean()) ** 2)), 4))
X2 = np.column_stack([np.log10(sub["vmax"]), np.log10(sub["vmin"]), sub["f"], sub["w"],
                      1 / sub["vmax"], sub["vmax"], 1 / sub["vmin"], sub["vmin"],
                      sub["f"] ** 2, sub["w"] ** 2, np.ones(len(sub))])
b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
rec("4_richer_R2", 0.3832, round(float(1 - np.sum((y - X2 @ b2) ** 2) / np.sum((y - y.mean()) ** 2)), 4))
Xi = np.column_stack([np.log10(sub["vmax"]), np.log10(sub["vmin"]), sub["f"], sub["w"],
                      np.log10(sub["vmax"]) * sub["f"], np.log10(sub["vmax"]) * sub["w"],
                      sub["f"] * sub["w"], np.ones(len(sub))])
bi, *_ = np.linalg.lstsq(Xi, y, rcond=None)
rec("4_inter_R2", 0.1382, round(float(1 - np.sum((y - Xi @ bi) ** 2) / np.sum((y - y.mean()) ** 2)), 4))
# how much of the "surface" is just censoring?  compare with the raw IC order
rec("4_R2_of_rank_alone_on_ic", None,
    round(float(np.corrcoef(sub["candidate_rank"], y)[0, 1] ** 2), 4))
rec("4_ic_vs_rank_spearman", None,
    round(float(pd.Series(sub["ic"]).corr(pd.Series(sub["candidate_rank"]), method="spearman")), 4))

# ---- mode alignment
mf = np.sort(modes["mode_frequency_microhz"].values)
cands = np.array(AX_F)
d = np.abs(cands[:, None] - mf[None, :]).min(axis=1)
rec("5_n_modes", 52, len(mf))
rec("5_median_offset", 5.76, round(float(np.median(d)), 3))
rec("5_mean_offset", 7.778, round(float(np.mean(d)), 3))
rec("5_median_offset_pct", 0.4991, round(float(np.median(d / cands * 100)), 4))
mf_band = mf[(mf >= 860) & (mf <= 1310)]
rec("5_n_modes_in_band", 18, len(mf_band))
rec("5_inband_median_spacing", 20.69, round(float(np.median(np.diff(mf_band))), 3))
rec("5_overall_median_spacing", 25.48, round(float(np.median(np.diff(mf))), 3))
rec("5_expected_uniform_nearest_inband", 12.089,
    round(float(1.0 / (2.0 * (len(mf_band) - 1) / (mf_band[-1] - mf_band[0]))), 3))

# nulls
def null_med(lo, hi, seed, N=200_000, chunk=20_000):
    rng = np.random.default_rng(seed)
    out = np.empty(N)
    k = 0
    while k < N:
        m = min(chunk, N - k)
        dr = rng.uniform(lo, hi, size=(m, len(cands)))
        out[k:k + m] = np.median(np.abs(dr[:, :, None] - mf[None, None, :]).min(axis=2), axis=1)
        k += m
    return out


nf = null_med(float(mf.min()), float(mf.max()), 555)
nb = null_med(860.0, 1310.0, 777)
om = float(np.median(d))
rec("5_null_full_median", 8.083, round(float(np.median(nf)), 3))
rec("5_null_band_median", 7.242, round(float(np.median(nb)), 3))
rec("5_p_full", 0.1786, round(float((nf <= om).mean()), 4))
rec("5_p_band", 0.2695, round(float((nb <= om).mean()), 4))
rng = np.random.default_rng(31337)
offs = rng.uniform(0, 50, size=100_000)
med_comb = np.array([np.median(np.abs((860.0 + np.arange(10) * 50.0 + o)[:, None]
                                      - mf[None, :]).min(axis=1)) for o in offs])
rec("5_null_comb_median", 7.335, round(float(np.median(med_comb)), 3))
rec("5_p_comb", 0.2876, round(float((med_comb <= om).mean()), 4))
rec("5_comb_best", 3.14, round(float(med_comb.min()), 3))
rng2 = np.random.default_rng(4242)
mind = np.empty(200_000)
k = 0
while k < 200_000:
    m = min(20_000, 200_000 - k)
    dr = rng2.uniform(float(mf.min()), float(mf.max()), size=(m, len(cands)))
    mind[k:k + m] = np.abs(dr[:, :, None] - mf[None, None, :]).min(axis=2).min(axis=1)
    k += m
rec("5_familywise_p", 0.4991, round(float((mind <= d.min()).mean()), 4))
rec("5_min_individual_offset", None, round(float(d.min()), 3))
rec("5_candidate_at_min", None, int(cands[int(np.argmin(d))]))
rec("5_mode_at_1160", None, float(mf[np.argmin(np.abs(cands[:, None] - mf[None, :]), axis=None) % len(mf)]))

# ---- resolution boundary (non-LS part)
spans = traces.groupby("pass_id")["relative_time_min"].apply(
    lambda t: float(t.max() - t.min()))
nsamp = traces.groupby("pass_id").size()
res = 1e6 / (spans.values * 60.0)
rec("6_n_passes", 22, int(len(spans)))
rec("6_span_min", 60.0, float(spans.min()))
rec("6_span_max", 79.0, float(spans.max()))
rec("6_span_counts", {"79.0": 19, "72.0": 2, "60.0": 1},
    {str(float(k)): int(v) for k, v in spans.value_counts().items()})
rec("6_n_samples_min", 61, int(nsamp.min()))
rec("6_n_samples_max", 80, int(nsamp.max()))
rec("6_n_samples_sum", 1698, int(nsamp.sum()))
rec("6_1overT_median", 210.97, round(float(np.median(res)), 3))
rec("6_1overT_min", 210.97, round(float(res.min()), 3))
rec("6_1overT_max", 277.778, round(float(res.max()), 3))
rec("6_cadence_s", 60.0, round(float(np.median(spans.values * 60.0 / (nsamp.values - 1))), 4))
rec("6_w250_over_1overT", 1.185, round(float(250.0 / np.median(res)), 3))
rec("6_w325_over_1overT", 1.54, round(float(325.0 / np.median(res)), 3))
rec("6_comb50_over_1overT", 0.237, round(float(50.0 / np.median(res)), 3))
# top-of-band: is the 79-min pass the one used in the sim? pass_01 span
rec("6_pass01_span", None, float(spans["pass_01"]))

raw = (RES / "grid_structure.json").read_text()
rec("7_literal_pctpct_tokens_in_json", None, raw.count("%%"))

json.dump({"recomputed": R, "problems_found": P, "rank1_params":
           {k: float(rank1[k]) for k in ["vmax", "vmin", "f", "w"]}},
          open(RES / "verify_grid_fast.json", "w"), indent=2, default=str)

print(f"{'quantity':46s} {'claimed':>16s} {'recomputed':>16s} {'absdiff':>12s}")
for k, v in R.items():
    c, r = v["claimed"], v["recomputed"]
    if isinstance(c, (int, float)) and isinstance(r, (int, float)):
        print(f"{k:46s} {float(c):16.4f} {float(r):16.6f} {v['abs_diff']:12.3e}")
    else:
        print(f"{k:46s} {str(c)[:16]:>16s} {str(r)[:44]:>44s}")
print("\nrank1:", rank1[["vmax", "vmin", "f", "w", "ic"]].to_dict())
print("\nvariants:", json.dumps(variants, default=str))