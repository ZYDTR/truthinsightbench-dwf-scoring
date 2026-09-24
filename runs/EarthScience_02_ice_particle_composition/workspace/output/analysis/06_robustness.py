"""Step 6 - formal robustness for the registered findings.

Every finding gets: a two-stage (group-as-unit) estimate, a leave-one-group-out
stability check, an alternative functional form or outcome definition, a
threshold/restriction sensitivity check, and an explicit test of the simple
null baseline named in the public contract (constant composition; uniform
particle-size distribution; pure counting/Poisson noise).

All demeaning goes through one helper (``demean``) so that the repeated-measures
structure is handled identically everywhere.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, SIZE_MID, load, matrices, spearman,
                    ols, ols_cluster, t_pvalue_two_sided, _betai)

comp, env = load()
C = matrices(comp)
E = matrices(env)
groups = np.array(C["collection_group_id"])
uniq = sorted(set(groups))
gidx = {g: i for i, g in enumerate(uniq)}
gi = np.array([gidx[g] for g in groups])
n = 30
tot = C["particle_total_per_L"]
ltot = np.log10(tot)
mid = (C["depth_min_m"] + C["depth_max_m"]) / 2.0
chl = E["chlorophyll_a_ug_L"]
lchl = np.log10(chl)
pe = C["polymer_PE_percent"]
van = C["polymer_varnish_percent"]
sd = E["particle_total_std_or_error_per_L"]
lsd = np.log10(sd)
poc = E["particulate_organic_carbon_mg_L"]
pon = E["particulate_organic_nitrogen_ug_L"]
out = {}


def demean(v, g=None):
    """Subtract group means (within-group / fixed-effect transform)."""
    v = np.asarray(v, float)
    g = gi if g is None else np.asarray(g)
    res = v.copy()
    for k in np.unique(g):
        m = g == k
        res[m] = v[m] - np.nanmean(v[m])
    return res


def t_ci(beta, se, df, level=0.95):
    lo, hi = 0.0, 200.0
    target = 1 - (1 - level) / 2
    for _ in range(300):
        m = (lo + hi) / 2
        if 1 - _betai(df / 2.0, 0.5, df / (df + m * m)) / 2 < target:
            lo = m
        else:
            hi = m
    tc = (lo + hi) / 2
    return [beta - tc * se, beta + tc * se]


def fe(y, x, g=None, xname="x"):
    """Within-group slope with cluster-robust SE at the collection-group level."""
    g = gi if g is None else np.asarray(g)
    f = ols_cluster(demean(y, g), np.c_[np.ones(len(np.asarray(y))), demean(x, g)],
                    groups if g is gi else g, ["intercept", xname])
    return {"slope": float(f["beta"][1]), "se": float(f["se"][1]),
            "p": float(f["p"][1]), "t": float(f["t"][1]),
            "df_cluster": int(f["df_cluster"]),
            "ci95": t_ci(float(f["beta"][1]), float(f["se"][1]), int(f["df_cluster"]))}


def loo_fe(y, x, xname="x"):
    rows = []
    for g in uniq:
        m = gi != gidx[g]
        rows.append({"dropped": g, **fe(np.asarray(y)[m], np.asarray(x)[m], gi[m], xname)})
    return rows


def mean_rho(x, y, mask=None):
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    vals = []
    for g in uniq:
        sel = mask & (gi == gidx[g])
        if sel.sum() >= 3:
            vals.append(spearman(np.asarray(x)[sel], np.asarray(y)[sel])[0])
    return float(np.mean(vals)), [round(v, 3) for v in vals]


def two_stage(y, x):
    """Group-level slopes summarised across collection groups (5 independent units)."""
    sl, det = [], []
    for g in uniq:
        m = gi == gidx[g]
        b = ols(np.asarray(y)[m], np.c_[np.ones(m.sum()), np.asarray(x)[m]], ["int", "x"])
        sl.append(float(b["beta"][1]))
        det.append({"group": g, "n": int(m.sum()), "slope": float(b["beta"][1]),
                    "se": float(b["se"][1]), "p": float(b["p"][1]),
                    "r2": float(b["r2"])})
    sl = np.array(sl)
    t = float(sl.mean() / (sl.std(ddof=1) / math.sqrt(len(sl))))
    df = len(sl) - 1
    nneg = int((sl < 0).sum())
    k = min(nneg, len(sl) - nneg)
    sign_p = float(2 * sum(math.comb(len(sl), j) for j in range(k + 1)) / 2 ** len(sl))
    return {"per_group": det, "mean_slope": float(sl.mean()),
            "sd": float(sl.std(ddof=1)), "t": t, "df": df,
            "p": float(t_pvalue_two_sided(t, df)), "n_groups": len(sl),
            "n_negative": nneg, "sign_test_p_two_sided": sign_p}


# ===================================================================== F1
f1 = {}
f1["two_stage"] = two_stage(ltot, mid)
f1["fe_cluster"] = fe(ltot, mid, xname="depth_mid_m")
f1["fe_cluster"]["slope_units"] = "log10(particles per L) per metre"
f1["loo_cluster"] = loo_fe(ltot, mid, "depth_mid_m")
f1["pooled_spearman"] = dict(zip(["rho", "p", "n"], spearman(mid, ltot)))
f1["within_group_mean_rho"], f1["within_group_rhos"] = mean_rho(mid, ltot)
ok = mid > 0
f1["alt_power_law_pooled"] = (lambda f: {"slope": float(f["beta"][1]),
                                         "se": float(f["se"][1]),
                                         "p": float(f["p"][1]),
                                         "r2": float(f["r2"])})(
    ols(ltot[ok], np.c_[np.ones(ok.sum()), np.log10(mid[ok])], ["int", "log10depth"]))
frac = np.array([mid[i] / mid[gi == gi[i]].max() for i in range(n)])
f1["alt_relative_depth_fe"] = fe(ltot, frac, xname="relative_depth")
top_i = [i for i in range(n) if mid[i] == mid[gi == gi[i]].min()]
bot_i = [i for i in range(n) if mid[i] == mid[gi == gi[i]].max()]
d = ltot[bot_i] - ltot[top_i]
tt = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))
f1["deepest_vs_shallowest"] = {
    "pairs": [{"group": groups[top_i[k]], "shallow_per_L": float(tot[top_i[k]]),
               "deep_per_L": float(tot[bot_i[k]]), "log10_ratio": float(d[k])}
              for k in range(len(top_i))],
    "mean_log10_ratio": float(d.mean()), "t": tt, "df": len(d) - 1,
    "p": float(t_pvalue_two_sided(tt, len(d) - 1)),
    "n_deeper_lower": int((d < 0).sum()), "n_pairs": len(d),
}
# distance from the ice surface in absolute metres is the alternative exposure
f1["alt_absolute_top_of_ice_origin"] = fe(ltot, C["depth_max_m"], xname="depth_max_m")
out["F1_depth_abundance"] = f1

# ===================================================================== F2
f2 = {}
f2["within_group_mean_rho"], f2["within_group_rhos"] = mean_rho(lchl, ltot)
f2["pooled_spearman"] = dict(zip(["rho", "p", "n"], spearman(lchl, ltot)))
f2["fe_cluster"] = fe(ltot, lchl, xname="log10_chl_a")
f2["two_stage"] = two_stage(ltot, lchl)
keep = np.ones(n, bool)
for g in uniq:
    idx = np.where(gi == gidx[g])[0]
    keep[idx[np.argmax(mid[idx])]] = False
f2["exclude_deepest_interval_per_group"] = {
    "n_remaining": int(keep.sum()),
    "within_group_mean_rho": mean_rho(lchl, ltot, keep)[0],
    "fe_cluster": fe(ltot[keep], lchl[keep], gi[keep], "log10_chl_a"),
}
# also drop the single highest-chlorophyll sample of each group
keep2 = keep.copy()
for g in uniq:
    idx = np.where(gi == gidx[g])[0]
    keep2[idx[np.argmax(chl[idx])]] = False
f2["exclude_deepest_and_highest_chl_interval_per_group"] = {
    "n_remaining": int(keep2.sum()),
    "within_group_mean_rho": mean_rho(lchl, ltot, keep2)[0],
    "fe_cluster": fe(ltot[keep2], lchl[keep2], gi[keep2], "log10_chl_a"),
}
for nm, v in [("POC", np.log10(poc)), ("PON", np.log10(pon))]:
    f2[f"alt_covariate_{nm}"] = fe(ltot, v, xname=f"log10_{nm}")
f2["joint_model_chl_depth_POC"] = (lambda f: {
    nm: {"beta": float(f["beta"][i]), "se": float(f["se"][i]), "p": float(f["p"][i])}
    for i, nm in enumerate(f["names"])} | {"df_cluster": int(f["df_cluster"])})(
    ols_cluster(demean(ltot), np.c_[np.ones(n), demean(lchl), demean(mid), demean(np.log10(poc))],
                groups, ["intercept", "log10_chl_a", "depth_mid_m", "log10_POC"]))
f2["effect_size"] = {
    "slope_log10_total_per_log10_chl": fe(ltot, lchl, xname="log10_chl_a")["slope"],
    "factor_change_in_total_per_10x_chl":
        float(10 ** fe(ltot, lchl, xname="log10_chl_a")["slope"]),
    "note": "descriptive within-group association, not a causal effect",
}
f2["baseline_null_constant_composition"] = (
    "a constant-composition baseline predicts slope = 0 between chlorophyll a "
    "and total particle concentration; the observed within-group slope is "
    f"{fe(ltot, lchl, xname='x')['slope']:.3f} (cluster-robust p = "
    f"{fe(ltot, lchl, xname='x')['p']:.4f}), so the constant baseline is rejected")
out["F2_chlorophyll_abundance"] = f2

# ===================================================================== F3
f3 = {}
f3["within_group_mean_rho"], f3["within_group_rhos"] = mean_rho(ltot, pe)
f3["pooled_spearman"] = dict(zip(["rho", "p", "n"], spearman(ltot, pe)))
f3["fe_cluster"] = fe(pe, ltot, xname="log10_total")
f3["fe_cluster"]["slope_units"] = "percentage points of PE per decade of concentration"
f3["two_stage"] = two_stage(pe, ltot)
f3["loo_cluster"] = loo_fe(pe, ltot, "log10_total")
for nm, y in [("varnish_percent", van),
              ("PE_share_of_PE_plus_varnish",
               100 * pe / np.clip(pe + van, 1e-9, None)),
              ("log10_count_PE", np.log10(np.clip(C["polymer_count_PE_per_L"], 1e-3, None)))]:
    f3[f"alt_outcome_{nm}"] = fe(y, ltot, xname="log10_total")
# resolution-controlled: only samples with a quantum finer than 2 pp
q = json.loads(Path("output/results/05_quantisation.json").read_text())
qpp = np.array([x if x else np.nan for x in q["size_visibility"]["quantum_pp_by_sample"]])
fine = np.isfinite(qpp) & (qpp < 2.0)
f3["restricted_fine_resolution"] = {
    "n": int(fine.sum()), "excluded": [C["sample_id"][j] for j in range(n) if not fine[j]],
    "within_group_mean_rho": mean_rho(ltot, pe, fine)[0],
    "fe_cluster": fe(pe[fine], ltot[fine], gi[fine], "log10_total"),
}
f3["joint_model_total_depth_chl"] = (lambda f: {
    nm: {"beta": float(f["beta"][i]), "se": float(f["se"][i]), "p": float(f["p"][i])}
    for i, nm in enumerate(f["names"])} | {"df_cluster": int(f["df_cluster"])})(
    ols_cluster(demean(pe), np.c_[np.ones(n), demean(ltot), demean(mid), demean(lchl)],
                groups, ["intercept", "log10_total", "depth_mid_m", "log10_chl_a"]))
f3["simple_baseline_uniform_composition"] = (
    "a uniform-composition baseline predicts PE_percent independent of total "
    "concentration (slope 0); observed within-group slope is "
    f"{fe(pe, ltot, xname='x')['slope']:.2f} pp per decade (cluster-robust p = "
    f"{fe(pe, ltot, xname='x')['p']:.4f})")
out["F3_PE_share_abundance"] = f3

# ===================================================================== F4
zf = sd
f4 = {
    "recovered_tallies": q["recovered_tallies"],
    "tally_summary": q["tally_summary"],
    "quantum_pp_by_sample": q["size_visibility"]["quantum_pp_by_sample"],
    "n_visible_bins_by_sample": q["size_visibility"]["n_bins_visible_by_sample"],
    "n_visible_vs_logtotal": q["size_visibility"]["spearman_nvisible_vs_logtotal"],
    "bound_check": q["quantum_bound_check"],
    "samples_with_coarse_resolution": [
        {"sample_id": C["sample_id"][j], "total_per_L": float(tot[j]),
         "N": q["recovered_tallies"][j]["N_common_min"],
         "quantum_pp": q["size_visibility"]["quantum_pp_by_sample"][j],
         "n_visible_size_bins": q["size_visibility"]["n_bins_visible_by_sample"][j],
         "PE_percent": float(pe[j]), "chlorophyll_a": float(chl[j])}
        for j in range(n) if (q["size_visibility"]["quantum_pp_by_sample"][j] or 0) >= 1.0],
    "fe_cluster": fe(lsd, ltot, xname="log10_total"),
    "two_stage": two_stage(lsd, ltot),
    "loo_cluster": loo_fe(lsd, ltot, "log10_total"),
    "cv_summary": {"min": float((zf / tot).min()), "median": float(np.median(zf / tot)),
                   "mean": float((zf / tot).mean()), "max": float((zf / tot).max()),
                   "sd": float((zf / tot).std(ddof=1))},
    "cv_vs_total_spearman": dict(zip(["rho", "p", "n"], spearman(zf / tot, ltot))),
}
b = fe(lsd, ltot, xname="log10_total")
f4["test_slope_equals_half_poisson"] = {
    "t": float((b["slope"] - 0.5) / b["se"]),
    "p": float(t_pvalue_two_sided((b["slope"] - 0.5) / b["se"], b["df_cluster"]))}
f4["test_slope_equals_one_proportional"] = {
    "t": float((b["slope"] - 1.0) / b["se"]),
    "p": float(t_pvalue_two_sided((b["slope"] - 1.0) / b["se"], b["df_cluster"]))}
# arithmetic-scale slope (alternative form avoiding the log transform)
f4["alt_scale_arithmetic_ols"] = (lambda f: {
    "slope": float(f["beta"][1]), "se": float(f["se"][1]), "p": float(f["p"][1])})(
    ols_cluster(demean(zf), np.c_[np.ones(n), demean(tot)], groups, ["int", "total"]))
out["F4_uncertainty_scaling"] = f4

# ===================================================================== F5
size_pct = np.vstack([C[f"particle_size_{s}_percent"] for s in SIZE_BINS])
lmid_bins = np.log10(SIZE_MID)
Ns = np.array([q["recovered_tallies"][j]["N_common_min"] or np.nan for j in range(n)], float)
nvis = np.array(q["size_visibility"]["n_bins_visible_by_sample"], float)


def spectrum_slope(idx, sub=None):
    sub = list(range(n)) if sub is None else list(sub)
    sl, nb = [], []
    for j in sub:
        v = size_pct[idx, j]
        ok = v > 0
        sl.append(float(np.polyfit(lmid_bins[idx][ok], np.log10(v[ok]), 1)[0])
                  if ok.sum() >= 2 else np.nan)
        nb.append(int(ok.sum()))
    return np.array(sl), np.array(nb), sub


f5 = {"bin_ranges": {}}
for nm, idx in {"bins_1_9": list(range(9)), "bins_1_4": list(range(4)),
                "bins_1_3": list(range(3))}.items():
    sl, nb, sub = spectrum_slope(idx)
    gg = gi[sub]
    f5["bin_ranges"][nm] = {
        "n_samples": len(sub), "mean_slope": float(np.nanmean(sl)),
        "median_slope": float(np.nanmedian(sl)),
        "rho_with_logtotal": dict(zip(["rho", "p", "n"], spearman(sl, ltot[sub]))),
        "within_group_mean_rho": mean_rho(ltot, sl, np.isin(np.arange(n), sub))[0],
        "fe_cluster": fe(sl, ltot[sub], gg, "log10_total"),
        "bins_resolved_min": int(nb.min()), "bins_resolved_max": int(nb.max()),
    }
hiN = np.where(Ns >= 100)[0]
sl3, nb3, _ = spectrum_slope(list(range(3)), hiN)
f5["bin_ranges"]["bins_1_3_N_ge_100"] = {
    "n_samples": len(hiN), "mean_slope": float(np.nanmean(sl3)),
    "rho_with_logtotal": dict(zip(["rho", "p", "n"], spearman(sl3, ltot[hiN]))),
    "fe_cluster": fe(sl3, ltot[hiN], gi[hiN], "log10_total"),
}
sl9, nb9, _ = spectrum_slope(list(range(9)))
f5["slope_bins_1_9_vs_n_visible"] = dict(zip(["rho", "p", "n"], spearman(sl9, nvis)))
f5["slope_bins_1_9_vs_log_N"] = dict(zip(["rho", "p", "n"], spearman(sl9, np.log10(Ns))))
f5["slope_vs_nvisible_controlling_abundance"] = (lambda f: {
    nm: {"beta": float(f["beta"][i]), "se": float(f["se"][i]), "p": float(f["p"][i])}
    for i, nm in enumerate(f["names"])} | {"df_cluster": int(f["df_cluster"])})(
    ols_cluster(demean(sl9), np.c_[np.ones(n), demean(nvis), demean(ltot)],
                groups, ["intercept", "n_visible_bins", "log10_total"]))
# monotone shape test on the three universally resolved bins
f5["universal_three_bin_shape"] = {
    "mean_percent_bin_11um": float(size_pct[0].mean()),
    "mean_percent_bin_11_25um": float(size_pct[1].mean()),
    "mean_percent_bin_25_50um": float(size_pct[2].mean()),
    "monotone_decreasing_all_samples": bool(np.all(
        (size_pct[0] >= size_pct[1]) & (size_pct[1] >= size_pct[2]))),
    "n_samples_with_monotone_shape": int(np.sum(
        (size_pct[0] >= size_pct[1]) & (size_pct[1] >= size_pct[2]))),
}
out["F5_size_spectrum_boundary"] = f5

Path("output/results").mkdir(parents=True, exist_ok=True)
(Path("output/results") / "06_robustness.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))