"""Step 3 - targeted testing of the candidate relationships found in step 2.

Implements, for each candidate: (i) within-group (fixed-effect) estimation,
(ii) cluster-robust inference, (iii) leave-one-group-out stability,
(iv) within-group permutation null, (v) at least one alternative definition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, SIZE_MID, load, matrices, spearman,
                    pearson, ols, ols_cluster)

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
lratio = np.log(np.clip(pe, 1e-3, None) / np.clip(van, 1e-3, None))
sd = E["particle_total_std_or_error_per_L"]
out = {}


def demean(v):
    v = np.asarray(v, float)
    o = v.copy()
    for g in uniq:
        m = gi == gidx[g]
        o[m] = v[m] - np.nanmean(v[m])
    return o


def within_group_slopes(x, y, xname, yname):
    rows = []
    for g in uniq:
        m = gi == gidx[g]
        rho, p, nn = spearman(x[m], y[m])
        rows.append({"group": g, "n": nn, "spearman_rho": rho, "p": p})
    return {"x": xname, "y": yname, "per_group": rows,
            "mean_rho": float(np.mean([r["spearman_rho"] for r in rows])),
            "sign_consistency": int(sum(1 for r in rows
                                        if np.sign(r["spearman_rho"]) ==
                                        np.sign(np.mean([q["spearman_rho"] for q in rows]))))}


def perm_within(x, y, n_perm=20000, seed=3, stat="mean_rho"):
    rng = np.random.default_rng(seed)
    xd, yd = demean(x), demean(y)
    obs = float(np.corrcoef(xd, yd)[0, 1])
    null = np.empty(n_perm)
    xg = [xd[gi == gidx[g]].copy() for g in uniq]
    yg = [yd[gi == gidx[g]] for g in uniq]
    for i in range(n_perm):
        xs = np.concatenate([rng.permutation(v) for v in xg])
        null[i] = np.corrcoef(xs, yd)[0, 1]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"within_group_r_demeaned": obs, "p_perm": float(p), "n_perm": n_perm}


def fe_fit(y, x, name, add_intercept=True):
    """Fixed-effects (within-group) slope with cluster-robust SE."""
    yd, xd = demean(y), demean(x)
    X = np.c_[np.ones(n), xd] if add_intercept else xd[:, None]
    fit = ols_cluster(yd, X, groups, ["intercept", name])
    return {"slope": float(fit["beta"][1]), "se_cluster": float(fit["se"][1]),
            "t": float(fit["t"][1]), "p": float(fit["p"][1]),
            "df_cluster": int(fit["df_cluster"]), "n_groups": int(fit["n_clusters"]),
            "r2_within": float(fit["r2"])}


def loo_groups(y, x):
    """Refit pooled OLS slope dropping one group at a time."""
    res = []
    for g in uniq:
        m = gi != gidx[g]
        fit = ols(y[m], np.c_[np.ones(m.sum()), x[m]], ["int", "x"])
        res.append({"dropped": g, "slope": float(fit["beta"][1]),
                    "p": float(fit["p"][1])})
    return res


def auc_rank(x, y):
    """AUC of x separating top-tertile y from bottom-tertile y."""
    y = np.asarray(y, float)
    hi = y >= np.quantile(y, 2 / 3)
    lo = y <= np.quantile(y, 1 / 3)
    a = np.asarray(x, float)[hi]
    b = np.asarray(x, float)[lo]
    wins = sum((ai > bj) + 0.5 * (ai == bj) for ai in a for bj in b)
    return float(wins / (len(a) * len(b)))


# ---------------------------------------------------------------- C1: depth
out["C1_total_vs_depth"] = {
    "within_group_spearman": within_group_slopes(mid, ltot, "depth_mid_m", "log10_total"),
    "pooled_spearman": dict(zip(["rho", "p", "n"], spearman(mid, ltot))),
    "fe_fit": fe_fit(ltot, mid, "depth_mid_m"),
    "perm": perm_within(mid, ltot),
    "loo": loo_groups(ltot, mid),
    "alt_relative_depth": {
        "pooled": dict(zip(["rho", "p", "n"], spearman(
            np.concatenate([mid[gi == gidx[g]] / mid[gi == gidx[g]].max() for g in uniq]),
            ltot))),
        "within_group_mean_rho": float(np.mean([
            spearman(mid[gi == gidx[g]] / mid[gi == gidx[g]].max(),
                     ltot[gi == gidx[g]])[0] for g in uniq])),
    },
    "alt_spearman_bottom_only": {
        "rho_thickness": spearman(C["depth_max_m"] - C["depth_min_m"], ltot)[:2],
    },
}

# --------------------------------------------------------------- C2: chl
out["C2_chl_vs_total"] = {
    "within_group_spearman": within_group_slopes(lchl, ltot, "log10_chl_a", "log10_total"),
    "pooled_spearman": dict(zip(["rho", "p", "n"], spearman(lchl, ltot))),
    "fe_fit": fe_fit(ltot, lchl, "log10_chl_a"),
    "perm": perm_within(lchl, ltot),
    "loo": loo_groups(ltot, lchl),
    "depth_control": {
        "chl_vs_depth_within": within_group_slopes(mid, lchl, "depth_mid_m", "log10_chl_a"),
        "fe_partial_total_on_chl_given_depth": None,
    },
    "auc_chl_top_vs_bottom_tertile_of_total": auc_rank(lchl, ltot),
    "corr_chl_raw": dict(zip(["rho", "p", "n"], spearman(chl, tot))),
}
# both regressors, cluster-robust
yd, x1, x2 = demean(ltot), demean(lchl), demean(mid)
fit2 = ols_cluster(yd, np.c_[np.ones(n), x1, x2], groups, ["int", "lchl", "depth"])
out["C2_chl_vs_total"]["depth_control"]["fe_partial_total_on_chl_given_depth"] = {
    "beta_lchl": float(fit2["beta"][1]), "se": float(fit2["se"][1]),
    "p": float(fit2["p"][1]), "beta_depth": float(fit2["beta"][2]),
    "se_depth": float(fit2["se"][2]), "p_depth": float(fit2["p"][2]),
    "df_cluster": int(fit2["df_cluster"]),
}

# --------------------------------------------------- C3: vector correlation
def bray_curtis(M):
    """Bray-Curtis dissimilarity between sample columns of a taxa/bin matrix."""
    x = M.T.astype(float)                       # samples x features
    x = np.log10(np.clip(x, 1e-3, None))        # log-scale, avoids count-scale dominance
    s = x.sum(axis=1)
    D = np.zeros((x.shape[0], x.shape[0]))
    for a in range(x.shape[0]):
        for b in range(a + 1, x.shape[0]):
            D[a, b] = D[b, a] = np.abs(x[a] - x[b]).sum() / (s[a] + s[b])
    return D


def mantel(x, y, n_perm=20000, seed=11):
    """Mantel test: correlation between two sample-sample distance matrices.

    Used to ask whether the polymer assemblage and the particle-size
    assemblage of the same sample co-vary more than expected by chance.
    """
    Dx, Dy = bray_curtis(x), bray_curtis(y)
    iu = np.triu_indices(Dx.shape[0], 1)
    a, b = Dx[iu], Dy[iu]
    obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    order = np.arange(Dx.shape[0])
    null = np.empty(n_perm)
    for i in range(n_perm):
        p_ = rng.permutation(order)
        null[i] = np.corrcoef(a, Dy[np.ix_(p_, p_)][iu])[0, 1]
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)   # one-sided: co-variation
    return {"mantel_r": obs, "p_perm_one_sided": float(p),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975))}


size_cnt = np.vstack([C[f"particle_size_{s}_per_L"] for s in SIZE_BINS])
poly_cnt = np.vstack([C[f"polymer_count_{p}_per_L"] for p in POLYMERS])
out["C3_vector_links"] = {
    "polymer_vs_size_count_vector": mantel(poly_cnt.T, size_cnt.T),
    "polymer_percent_summary": {
        "sum_pct_check": float(np.abs(poly_cnt.sum(axis=0) / tot * 100 - 100).max()),
    },
}

# ------------------------------------------------- C4: PE share vs abundance
out["C4_PE_vs_abundance"] = {
    "within_group_spearman": within_group_slopes(ltot, pe, "log10_total", "PE_percent"),
    "pooled_spearman": dict(zip(["rho", "p", "n"], spearman(ltot, pe))),
    "fe_fit": fe_fit(pe, ltot, "log10_total"),
    "perm": perm_within(ltot, pe),
    "loo": loo_groups(pe, ltot),
    "alt_share_numerator": {
        "pooled_pe_count_vs_total": dict(zip(["rho", "p", "n"],
                                             spearman(ltot, np.log10(np.clip(
                                                 C["polymer_count_PE_per_L"], 1e-3, None))))),
    },
    "alt_dominance_ratio": {
        "pooled_pe_over_varnish_vs_total": dict(zip(["rho", "p", "n"],
                                                    spearman(ltot, lratio))),
        "within_group": within_group_slopes(ltot, lratio, "log10_total", "log(PE/varnish)"),
    },
    "confound_depth": fe_fit(pe, mid, "depth_mid_m"),
    "confound_chl": fe_fit(pe, lchl, "log10_chl_a"),
}
# PE share ~ log total + depth + chl jointly (within-group)
yd, X = demean(pe), np.c_[np.ones(n), demean(ltot), demean(mid), demean(lchl)]
fitp = ols_cluster(yd, X, groups, ["int", "ltot", "depth", "lchl"])
out["C4_PE_vs_abundance"]["joint_fe_model"] = {
    nm: {"beta": float(fitp["beta"][i]), "se": float(fitp["se"][i]),
         "p": float(fitp["p"][i])} for i, nm in enumerate(fitp["names"])
}
out["C4_PE_vs_abundance"]["joint_fe_model"]["df_cluster"] = int(fitp["df_cluster"])

# --------------------------------------------------- C5: uncertainty scaling
lsd = np.log10(sd)
out["C5_uncertainty"] = {
    "pooled_ols_slope": float(ols(lsd, np.c_[np.ones(n), ltot],
                                 ["int", "ltot"])["beta"][1]),
    "cluster_robust": (lambda f: {"slope": float(f["beta"][1]),
                                  "se": float(f["se"][1]), "p": float(f["p"][1]),
                                  "df": int(f["df_cluster"])})(
        ols_cluster(lsd, np.c_[np.ones(n), ltot], groups, ["int", "ltot"])),
    "cv_summary": {"min": float((sd / tot).min()), "median": float(np.median(sd / tot)),
                   "max": float((sd / tot).max())},
    "cv_vs_total_spearman": dict(zip(["rho", "p", "n"], spearman(sd / tot, ltot))),
    "loo": loo_groups(lsd, ltot),
    "alt_reduced_major_axis": None,
}
# RMA / geometric-mean slope (errors-in-variables style alternative)
var_x = float(np.var(ltot, ddof=1))
var_y = float(np.var(lsd, ddof=1))
cov_xy = float(np.cov(ltot, lsd, ddof=1)[0, 1])
out["C5_uncertainty"]["alt_reduced_major_axis"] = float(
    (var_y - var_x + np.sqrt((var_y - var_x) ** 2 + 4 * cov_xy ** 2)) / (2 * cov_xy))

# ------------------------------------------- C6: size spectrum curvature
size_pct = np.vstack([C[f"particle_size_{s}_percent"] for s in SIZE_BINS])
lmid_bins = np.log10(SIZE_MID)
slopes9, curvature = [], []
for j in range(n):
    v = size_pct[:9, j]
    ok = v > 0
    b = np.polyfit(lmid_bins[:9][ok], np.log10(v[ok]), 1)
    slopes9.append(b[0])
    sl_lo = np.polyfit(lmid_bins[:3][ok[:3]], np.log10(v[:3][ok[:3]]), 1)[0] \
        if ok[:3].sum() >= 2 else np.nan
    sl_hi = np.polyfit(lmid_bins[2:9][ok[2:9]], np.log10(v[2:9][ok[2:9]]), 1)[0] \
        if ok[2:9].sum() >= 2 else np.nan
    curvature.append((sl_lo, sl_hi))
slopes9 = np.array(slopes9)
sl_lo = np.array([c[0] for c in curvature])
sl_hi = np.array([c[1] for c in curvature])
out["C6_size_spectrum"] = {
    "slope_bins1_9": {"mean": float(np.mean(slopes9)), "median": float(np.median(slopes9)),
                      "sd": float(np.std(slopes9, ddof=1)),
                      "min": float(slopes9.min()), "max": float(slopes9.max())},
    "slope_shallow_bins1_3_median": float(np.nanmedian(sl_lo)),
    "slope_steep_bins3_9_median": float(np.nanmedian(sl_hi)),
    "curvature_test": {"n_steeper_than_shallow": int(np.nansum(sl_hi < sl_lo)),
                       "n_pairs": int(np.sum(np.isfinite(sl_lo) & np.isfinite(sl_hi)))},
    "slope_vs_log_total": dict(zip(["rho", "p", "n"], spearman(slopes9, ltot))),
    "within_group_spearman_slope_vs_total": within_group_slopes(ltot, slopes9,
                                                               "log10_total", "spectral_slope"),
    "slope_by_group": {g: [round(float(x), 3) for x in slopes9[gi == gidx[g]]]
                       for g in uniq},
}
# bin-resolved mean percent with bootstrap CI over the whole-group unit
rng = np.random.default_rng(5)
boot = np.array([[np.mean(size_pct[i][rng.integers(0, n, n)]) for i in range(22)]
                 for _ in range(4000)])
out["C6_size_spectrum"]["bin_mean_percent_ci95"] = [
    {"bin": SIZE_BINS[i], "mean": float(np.mean(size_pct[i])),
     "ci_lo": float(np.quantile(boot[:, i], 0.025)),
     "ci_hi": float(np.quantile(boot[:, i], 0.975)),
     "n_nonzero": int((size_pct[i] > 0).sum())} for i in range(22)]

# --------------------------------- C7: PE/varnish depth conflict
out["C7_PE_varnish_ratio_depth"] = {
    "within_group_spearman": within_group_slopes(mid, lratio, "depth_mid_m", "log(PE/varnish)"),
    "pooled_spearman": dict(zip(["rho", "p", "n"], spearman(mid, lratio))),
    "fe_fit": fe_fit(lratio, mid, "depth_mid_m"),
    "perm": perm_within(mid, lratio),
    "is_driven_by_chl": fe_fit(lratio, lchl, "log10_chl_a"),
}

Path("output/results").mkdir(parents=True, exist_ok=True)
(Path("output/results") / "03_candidates.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))