"""Independent adversarial verification of CLAIM F1.

Claim F1: within a collection group, total particle concentration
(particle_total_per_L) decreases with depth.

Everything here is written from scratch (pandas/numpy/scipy/statsmodels), reading
the raw TSVs directly. No existing analysis script is imported or re-run.
Outputs a single self-contained JSON with every number plus this source code.
"""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path

ROOT = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = ROOT / "output" / "results" / "verify_F1.json"

comp = pd.read_csv(ROOT / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(ROOT / "data" / "sample_environment_profiles.tsv", sep="\t")

# ---------------------------------------------------------------- assemble frame
df = comp[["sample_id", "collection_group_id", "depth_min_m", "depth_max_m",
           "particle_total_per_L"]].merge(
    env[["sample_id", "ice_regime", "particle_total_per_L",
         "particle_total_std_or_error_per_L"]],
    on="sample_id", suffixes=("", "_env"), validate="1:1")
assert np.allclose(df["particle_total_per_L"], df["particle_total_per_L_env"])
df = df.drop(columns=["particle_total_per_L_env"])
df["mid"] = (df["depth_min_m"] + df["depth_max_m"]) / 2.0
df["ltot"] = np.log10(df["particle_total_per_L"])
df["rel"] = df.groupby("collection_group_id")["mid"].transform(
    lambda v: (v - v.min()) / (v.max() - v.min()))
df["dmax"] = df["depth_max_m"]
groups = sorted(df["collection_group_id"].unique())
df["gcode"] = pd.Categorical(df["collection_group_id"], categories=groups, ordered=True)

res: dict = {}
res["data_structure"] = {
    "n_rows": int(len(df)),
    "n_groups": len(groups),
    "group_n": {g: int((df.collection_group_id == g).sum()) for g in groups},
    "ice_regime_by_group": {g: sorted(df.loc[df.collection_group_id == g, "ice_regime"].unique())
                            for g in groups},
    "depth_range_by_group_m": {g: [float(df.loc[df.collection_group_id == g, "mid"].min()),
                                   float(df.loc[df.collection_group_id == g, "mid"].max())]
                               for g in groups},
    "total_per_L_by_group": {g: [float(x) for x in df.loc[df.collection_group_id == g,
                                                          "particle_total_per_L"]]
                             for g in groups},
}

mid = df["mid"].to_numpy(float)
ltot = df["ltot"].to_numpy(float)
tot = df["particle_total_per_L"].to_numpy(float)
gid = df["collection_group_id"].to_numpy()
gi = pd.Categorical(df["collection_group_id"], categories=groups).codes.astype(int)
n = len(df)
G = len(groups)


# ---------------------------------------------------------------- helpers (mine)
def ols(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = X.shape[1]
    dof = len(y) - k
    s2 = resid @ resid / dof
    XtXinv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtXinv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta, "se": se, "t": t, "p": p, "dof": dof,
            "resid": resid, "XtXinv": XtXinv}


def demean(v, codes=gi):
    v = np.asarray(v, float)
    out = v.copy()
    for k in np.unique(codes):
        m = codes == k
        out[m] = v[m] - v[m].mean()
    return out


def cluster_ols(y, X, codes=gi):
    """Within-group (demeaned) slope with CR1 cluster-robust covariance.

    df uses G-1 (5 clusters -> 4) as the honest small-cluster convention.
    Length-agnostic so it works on subsets."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    codes = np.asarray(codes)
    n = len(y)
    Xd = np.c_[np.ones(n), demean(X, codes)]
    yd = demean(y, codes)
    fit = ols(yd, Xd)
    b = fit["beta"]
    r = fit["resid"]
    XtXinv = fit["XtXinv"]
    meat = np.zeros((2, 2))
    for k in np.unique(codes):
        m = codes == k
        u = (Xd[m] * r[m, None]).sum(axis=0)
        meat += np.outer(u, u)
    nc, kk = len(np.unique(codes)), Xd.shape[1]
    scale = (nc / (nc - 1)) * ((n - 1) / (n - kk))
    V = XtXinv @ (scale * meat) @ XtXinv
    se = np.sqrt(np.diag(V))
    dfc = nc - 1
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), dfc)
    lo = b - stats.t.ppf(0.975, dfc) * se
    hi = b + stats.t.ppf(0.975, dfc) * se
    # also the anticonservative n-k df variant
    p28 = 2 * stats.t.sf(np.abs(t), n - kk)
    return {"slope": float(b[1]), "se_cluster_df4": float(se[1]),
            "t": float(t[1]), "df_cluster": dfc, "p_cluster_df4": float(p[1]),
            "p_naive_df28": float(p28[1]),
            "ci95_df4": [float(lo[1]), float(hi[1])], "resid": r}


def spearman(x, y):
    s = stats.spearmanr(x, y)
    return float(s.statistic), float(s.pvalue)


def pearson(x, y):
    s = stats.pearsonr(x, y)
    return float(s.statistic), float(s.pvalue)


# ============================================================ 1. pooled tests
res["pooled"] = {
    "spearman_mid_vs_ltot": {"rho": spearman(mid, ltot)[0], "p_two_sided": spearman(mid, ltot)[1],
                             "p_one_sided_negative": spearman(mid, ltot)[1] / 2, "n": n},
    "pearson_mid_vs_ltot": {"r": pearson(mid, ltot)[0], "p": pearson(mid, ltot)[1]},
}
res["pooled"]["kendall_tau"] = dict(zip(["tau", "p"], [float(v) for v in
                                                      stats.kendalltau(mid, ltot)[:2]]))
res["pooled"]["note"] = ("pooled test mixes between-group (station) differences with "
                         "within-group depth; confounded by group identity.")

# ============================================================ 2. per-group slopes
rows = []
for g in groups:
    m = gid == g
    b = ols(ltot[m], np.c_[np.ones(m.sum()), mid[m]])
    rho, prho = spearman(mid[m], ltot[m])
    r, pr = pearson(mid[m], ltot[m])
    tau, ptau = stats.kendalltau(mid[m], ltot[m])
    # Theil-Sen style: median of all pairwise within-group slopes
    xx, yy = mid[m], ltot[m]
    pw = [(yy[j] - yy[i]) / (xx[j] - xx[i]) for i in range(len(xx)) for j in range(i + 1, len(xx))]
    rows.append({
        "group": g, "n": int(m.sum()), "ice_regime": df.loc[m, "ice_regime"].iloc[0],
        "mid_depth_min": float(mid[m].min()), "mid_depth_max": float(mid[m].max()),
        "ols_slope_per_m": float(b["beta"][1]), "ols_se": float(b["se"][1]),
        "ols_p": float(b["p"][1]), "ols_r2": float(1 - b["resid"] @ b["resid"] /
                                                   ((ltot[m] - ltot[m].mean()) @ (ltot[m] - ltot[m].mean()))),
        "spearman_rho": rho, "spearman_p_two_sided": prho,
        "pearson_r": r, "pearson_p": pr,
        "kendall_tau": float(tau), "kendall_p": ptau,
        "theil_sen_median_pairwise_slope": float(np.median(pw)),
        "total_first_vs_last_log10": float(ltot[m][np.argmax(mid[m])] - ltot[m][np.argmin(mid[m])]),
    })
res["per_group_slopes"] = rows

sl = np.array([r["ols_slope_per_m"] for r in rows])
se = np.array([r["ols_se"] for r in rows])
rho = np.array([r["spearman_rho"] for r in rows])
tt = sl.mean() / (sl.std(ddof=1) / math.sqrt(G))
nneg = int((sl < 0).sum())
k = min(nneg, G - nneg)
sign_p = float(2 * sum(math.comb(G, j) for j in range(k + 1)) / 2 ** G)
tt_rho = rho.mean() / (rho.std(ddof=1) / math.sqrt(G))
res["two_stage_group_as_unit"] = {
    "mean_ols_slope": float(sl.mean()), "sd_slope": float(sl.std(ddof=1)),
    "t": float(tt), "df": G - 1, "p": float(2 * stats.t.sf(abs(tt), G - 1)),
    "n_negative_slopes": nneg, "n_positive": int((sl > 0).sum()),
    "sign_test_p_two_sided": sign_p,
    "sign_test_p_one_sided_negative": float(sign_p / 2),
    "mean_spearman_rho": float(rho.mean()), "t_rho": float(tt_rho),
    "p_rho": float(2 * stats.t.sf(abs(tt_rho), G - 1)),
    "n_negative_rho": int((rho < 0).sum()),
    "sign_test_p_rho_two_sided": float(2 * sum(math.comb(G, j) for j in range(
        min(int((rho < 0).sum()), int((rho > 0).sum())) + 1)) / 2 ** G),
    "inverse_variance_weighted_mean_slope": float((sl / se ** 2).sum() / (1 / se ** 2).sum()),
}

# DerSimonian-Laird random-effects meta-analysis across the 5 group slopes
w = 1.0 / se ** 2
mu_fe = (sl * w).sum() / w.sum()
Q = float((w * (sl - mu_fe) ** 2).sum())
C = float(w.sum() - (w ** 2).sum() / w.sum())
tau2 = max(0.0, (Q - (G - 1)) / C) if C > 0 else 0.0
wr = 1.0 / (se ** 2 + tau2)
mu_re = float((sl * wr).sum() / wr.sum())
se_re = float(math.sqrt(1.0 / wr.sum()))
res["two_stage_group_as_unit"]["random_effects_DL"] = {
    "mu_re": mu_re, "se_re": se_re, "z": float(mu_re / se_re),
    "p": float(2 * stats.norm.sf(abs(mu_re / se_re))),
    "tau2": tau2, "Q": Q, "Q_df": G - 1, "Q_p": float(stats.chi2.sf(Q, G - 1)),
    "I2_percent": float(max(0.0, (Q - (G - 1)) / Q) * 100) if Q > 0 else 0.0,
}
res["two_stage_group_as_unit"]["stouffer_fisher_combine_group_rho"] = {}
zr = np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt(
    np.array([r["n"] for r in rows]) - 3)
Zst = float(zr.sum() / math.sqrt(G))
fisher = -2.0 * float(np.sum(np.log([r["spearman_p_two_sided"] for r in rows])))
res["two_stage_group_as_unit"]["stouffer_fisher_combine_group_rho"] = {
    "per_group_fisher_z": [float(x) for x in zr],
    "stouffer_Z": Zst, "stouffer_p_two_sided": float(2 * stats.norm.sf(abs(Zst))),
    "fisher_chi2": fisher, "fisher_df": 2 * G,
    "fisher_p": float(stats.chi2.sf(fisher, 2 * G)),
}

# ============================================================ 3. fixed-effect (within-group) slope
fe = cluster_ols(ltot, mid)
res["fixed_effect_within_group"] = {
    "slope_log10_per_m": fe["slope"], "se_cluster_CR1_df4": fe["se_cluster_df4"],
    "t": fe["t"], "df_cluster": fe["df_cluster"], "p": fe["p_cluster_df4"],
    "ci95": fe["ci95_df4"], "p_anticonservative_naive_df28": fe["p_naive_df28"],
    "units": "log10(particles/L) per metre",
    "equivalent_factor_change_per_metre": float(10 ** fe["slope"]),
}
# cross-check with statsmodels
Xsm = sm.add_constant(np.c_[mid])[:, :2]
msm = sm.OLS(ltot, Xsm).fit(cov_type="cluster", cov_kwds={"groups": gid})
res["fixed_effect_within_group"]["statsmodels_check_plain_ols_level"] = {
    "slope": float(msm.params[1]), "se_cluster": float(msm.bse[1]),
    "p_reported": float(msm.pvalues[1]), "df_used": float(msm.df_resid),
    "note": "level regression with group-clustered SE (between-group variation kept); "
            "this is NOT a within-group slope"}
fe_sm = sm.OLS(demean(ltot), sm.add_constant(demean(mid))).fit(
    cov_type="cluster", cov_kwds={"groups": gid})
res["fixed_effect_within_group"]["statsmodels_check_demeaned"] = {
    "slope": float(fe_sm.params[1]), "se_cluster": float(fe_sm.bse[1]),
    "p_reported_by_statsmodels": float(fe_sm.pvalues[1]),
    "df_used_by_statsmodels": float(fe_sm.df_resid),
    "agrees_with_my_CR1_se": bool(np.isclose(fe_sm.bse[1], fe["se_cluster_df4"], rtol=1e-6)),
}

# relative depth variant
ferel = cluster_ols(ltot, df["rel"].to_numpy(float))
res["fixed_effect_relative_depth"] = {
    "slope_per_full_group_depth_range": ferel["slope"], "se": ferel["se_cluster_df4"],
    "t": ferel["t"], "df": ferel["df_cluster"], "p": ferel["p_cluster_df4"],
    "ci95": ferel["ci95_df4"],
}
# depth_max variant
fedm = cluster_ols(ltot, df["dmax"].to_numpy(float))
res["fixed_effect_depth_max_m"] = {
    "slope": fedm["slope"], "se": fedm["se_cluster_df4"], "t": fedm["t"],
    "p": fedm["p_cluster_df4"], "df": fedm["df_cluster"], "ci95": fedm["ci95_df4"],
}

# ============================================================ 4. deepest vs shallowest
pairs = []
for g in groups:
    m = df.collection_group_id == g
    sub = df[m].sort_values("mid")
    sh, dp = sub.iloc[0], sub.iloc[-1]
    pairs.append({"group": g, "shallow_mid_m": float(sh["mid"]), "deep_mid_m": float(dp["mid"]),
                  "shallow_per_L": float(sh["particle_total_per_L"]),
                  "deep_per_L": float(dp["particle_total_per_L"]),
                  "log10_ratio_deep_over_shallow": float(dp["ltot"] - sh["ltot"]),
                  "raw_ratio": float(dp["particle_total_per_L"] / sh["particle_total_per_L"])})
d = np.array([p["log10_ratio_deep_over_shallow"] for p in pairs])
t_d = float(d.mean() / (d.std(ddof=1) / math.sqrt(G)))
res["deepest_vs_shallowest"] = {
    "pairs": pairs, "mean_log10_ratio": float(d.mean()), "sd": float(d.std(ddof=1)),
    "median_log10_ratio": float(np.median(d)),
    "t": t_d, "df": G - 1, "p_paired_t": float(2 * stats.t.sf(abs(t_d), G - 1)),
    "wilcoxon_stat": float(stats.wilcoxon(d).statistic),
    "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
    "n_deeper_lower": int((d < 0).sum()), "n_pairs": G,
    "sign_test_p_two_sided": float(2 * sum(math.comb(G, j) for j in range(
        min(int((d > 0).sum()), int((d < 0).sum())) + 1)) / 2 ** G),
    "geometric_mean_ratio_deep_over_shallow": float(10 ** d.mean()),
    "group_01_ratio_note": "group_01 is the one group where the deepest interval is HIGHER",
}
# all within-group non-adjacent depth contrasts (not just the extreme pair)
allc = []
for g in groups:
    m = df.collection_group_id == g
    sub = df[m].sort_values("mid")
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            allc.append(float(sub.iloc[j]["ltot"] - sub.iloc[i]["ltot"]))
allc = np.array(allc)
t_all = float(allc.mean() / (allc.std(ddof=1) / math.sqrt(len(allc))))
res["all_within_group_pairwise_contrasts"] = {
    "n_pairs": int(len(allc)), "mean_log10_diff_deeper_minus_shallower": float(allc.mean()),
    "t_naive_independent": t_all,
    "p_naive_independent": float(2 * stats.t.sf(abs(t_all), len(allc) - 1)),
    "note": "pairwise contrasts are not independent; shown for descriptive scale only",
}

# ============================================================ 5. robustness / adversarial
loo = []
for g in groups:
    m = gid != g
    keep_gi = pd.Categorical(df.loc[m, "collection_group_id"], categories=groups).codes.astype(int)
    codes = np.unique(keep_gi)
    remap = {c: i for i, c in enumerate(codes)}
    kg = np.array([remap[c] for c in keep_gi])
    f = cluster_ols(ltot[m], mid[m], kg)
    rho_l, p_l = spearman(mid[m], ltot[m])
    loo.append({"dropped_group": g, "slope": f["slope"], "se": f["se_cluster_df4"],
                "p": f["p_cluster_df4"], "df": f["df_cluster"],
                "ci95": f["ci95_df4"], "pooled_spearman_rho": rho_l,
                "pooled_spearman_p": p_l})
res["leave_one_group_out"] = loo

# excluding group_01 (the single landfast / positive-slope group)
m1 = gid != "group_01"
codes = np.unique(gi[m1]); remap = {c: i for i, c in enumerate(codes)}
kg = np.array([remap[c] for c in gi[m1]])
f_no1 = cluster_ols(ltot[m1], mid[m1], kg)
res["exclude_group_01"] = {
    "n": int(m1.sum()), "slope": f_no1["slope"], "se": f_no1["se_cluster_df4"],
    "t": f_no1["t"], "df": f_no1["df_cluster"], "p": f_no1["p_cluster_df4"],
    "ci95": f_no1["ci95_df4"],
    "pooled_spearman": dict(zip(["rho", "p", "n"], [spearman(mid[m1], ltot[m1])[0],
                                                    spearman(mid[m1], ltot[m1])[1], int(m1.sum())])),
    "pooled_pearson": dict(zip(["r", "p"], [pearson(mid[m1], ltot[m1])[0],
                                            pearson(mid[m1], ltot[m1])[1]])),
    "per_group_slopes": [r["ols_slope_per_m"] for r in rows if r["group"] != "group_01"],
    "two_stage_mean_of_remaining_4": float(np.mean([r["ols_slope_per_m"] for r in rows
                                                    if r["group"] != "group_01"])),
    "two_stage_t": float(np.mean([r["ols_slope_per_m"] for r in rows if r["group"] != "group_01"])
                         / (np.std([r["ols_slope_per_m"] for r in rows if r["group"] != "group_01"],
                                   ddof=1) / 2.0)),
}
# exclude the single most influential sample in turn (leave-one-sample-out)
loso = []
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    codes = np.unique(gi[m]); remap = {c: j for j, c in enumerate(codes)}
    kg = np.array([remap[c] for c in gi[m]])
    f = cluster_ols(ltot[m], mid[m], kg)
    loso.append({"dropped_sample": df.sample_id[i], "group": gid[i],
                 "slope": f["slope"], "p": f["p_cluster_df4"]})
res["leave_one_sample_out"] = {
    "rows": loso,
    "slope_min": float(min(x["slope"] for x in loso)),
    "slope_max": float(max(x["slope"] for x in loso)),
    "p_min": float(min(x["p"] for x in loso)), "p_max": float(max(x["p"] for x in loso)),
    "still_negative_everywhere": bool(all(x["slope"] < 0 for x in loso)),
    "any_p_below_0.05": bool(any(x["p"] < 0.05 for x in loso)),
}

# cluster-level permutation test for the FE slope (exact null: depth order random within group)
rng = np.random.default_rng(20180101)
obs = fe["slope"]
perm = np.empty(20000)
perm_rho = np.empty(20000)
for b in range(20000):
    lp = ltot.copy()
    for kk_ in np.unique(gi):
        mm = gi == kk_
        lp[mm] = ltot[mm][rng.permutation(mm.sum())]
    perm[b] = cluster_ols(lp, mid)["slope"]
    perm_rho[b] = spearman(mid, lp)[0]
res["cluster_permutation_test"] = {
    "statistic": "within-group demeaned slope of log10 total on mid-depth",
    "observed": obs, "n_permutations": 20000,
    "p_two_sided": float((np.abs(perm) >= abs(obs) - 1e-12).mean()),
    "p_one_sided_negative": float((perm <= obs + 1e-12).mean()),
    "perm_sd": float(perm.std(ddof=1)),
    "perm_quantiles": {q: float(np.quantile(perm, qq)) for q, qq in
                       [("2.5%", .025), ("25%", .25), ("50%", .5), ("75%", .75), ("97.5%", .975)]},
    "pooled_spearman_perm": {
        "observed_rho": spearman(mid, ltot)[0],
        "p_two_sided": float((np.abs(perm_rho) >= abs(spearman(mid, ltot)[0]) - 1e-12).mean())},
}
# cluster bootstrap CI for the FE slope (resample groups)
boot = np.empty(20000)
obs_groups = np.unique(gi)
for b in range(20000):
    pick = rng.choice(obs_groups, size=G, replace=True)
    idx = np.concatenate([np.where(gi == p)[0] for p in pick])
    codes = np.concatenate([np.full((gi == p).sum(), j) for j, p in enumerate(pick)])
    fit = cluster_ols(ltot[idx], mid[idx], codes)
    boot[b] = fit["slope"]
res["cluster_bootstrap"] = {
    "n_boot": 20000, "mean": float(boot.mean()), "sd": float(boot.std(ddof=1)),
    "ci95_percentile": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    "fraction_negative": float((boot < 0).mean()),
    "fraction_below_-1": float((boot < -1).mean()),
}

# same for the deepest-vs-shallowest contrast (bootstrap over the 5 group ratios)
dboot = rng.choice(d, size=(20000, G), replace=True).mean(axis=1)
res["deepest_vs_shallowest"]["bootstrap"] = {
    "ci95_percentile": [float(np.quantile(dboot, .025)), float(np.quantile(dboot, .975))],
    "fraction_negative": float((dboot < 0).mean()),
}

# ============================================================ 6. sensitivity floor / power
res["sensitivity"] = {
    "within_group_residual_sd_log10": float(fe["resid"].std(ddof=2)),
    "total_sd_log10": float(ltot.std(ddof=1)),
    "between_group_variance_share": float(
        1 - (fe["resid"] @ fe["resid"]) / ((ltot - ltot.mean()) @ (ltot - ltot.mean()))),
    "observed_ci95_for_slope": fe["ci95_df4"],
    "smallest_slope_excluded_by_ci": float(fe["ci95_df4"][0]),
    "largest_positive_slope_still_compatible": float(fe["ci95_df4"][1]),
    "t_crit_df4": float(stats.t.ppf(0.975, 4)),
    "note": ("with 5 clusters the 95% t critical value is ~2.78; the design can only "
             "detect slopes roughly |b| > 2.78 * SE. CI upper bound is positive, so a "
             "zero or even slightly positive within-group slope is compatible."),
}
# rough power: how often would we call p<0.05 given a true slope b, holding data fixed?
res["sensitivity"]["power_under_shifted_truth"] = {}
for b_true in [-2.0, -1.5, -1.0, -0.826, -0.5, -0.25]:
    shifted = fe["resid"] + b_true * demean(mid)
    cnt = 0
    reps = 2000
    for _ in range(reps):
        rs = rng.permutation(fe["resid"])
        ysim = rs + b_true * demean(mid)
        f2 = cluster_ols(ysim, mid)
        cnt += f2["p_cluster_df4"] < 0.05
    res["sensitivity"]["power_under_shifted_truth"][f"{b_true}"] = cnt / reps
    # add the permuted-residual noise on top of the systematic component

# alternative aggregation: regress group means (equal weight, n=5) -- the two-stage view
gmean_mid = np.array([mid[gid == g].mean() for g in groups])
gmean_l = np.array([ltot[gid == g].mean() for g in groups])
b_g = ols(gmean_l, np.c_[np.ones(G), gmean_mid])
res["group_mean_regression_n5"] = {
    "slope": float(b_g["beta"][1]), "se": float(b_g["se"][1]), "p": float(b_g["p"][1]),
    "r2": float(1 - b_g["resid"] @ b_g["resid"] /
                ((gmean_l - gmean_l.mean()) @ (gmean_l - gmean_l.mean()))),
    "pearson_r": pearson(gmean_mid, gmean_l)[0], "pearson_p": pearson(gmean_mid, gmean_l)[1],
    "spearman_rho": spearman(gmean_mid, gmean_l)[0], "spearman_p": spearman(gmean_mid, gmean_l)[1],
}
# is the trend driven by the single deepest interval in bulk groups (sediment-like rebound)?
keep_no_deepest = []
for g in groups:
    m = df.collection_group_id == g
    sub = df[m]
    keep_no_deepest.append(sub.index[sub["mid"] != sub["mid"].max()])
idx = np.concatenate(keep_no_deepest)
codes = pd.Categorical(df.loc[idx, "collection_group_id"], categories=groups).codes.astype(int)
uu = np.unique(codes); remap = {c: i for i, c in enumerate(uu)}
f_nd = cluster_ols(ltot[idx], mid[idx], np.array([remap[c] for c in codes]))
res["drop_deepest_interval_per_group"] = {
    "n": int(len(idx)), "slope": f_nd["slope"], "se": f_nd["se_cluster_df4"],
    "p": f_nd["p_cluster_df4"], "df": f_nd["df_cluster"], "ci95": f_nd["ci95_df4"],
    "pooled_spearman_rho": spearman(mid[idx], ltot[idx])[0],
    "pooled_spearman_p": spearman(mid[idx], ltot[idx])[1],
}
# drop the shallowest (surface) interval per group
keep_no_shallow = []
for g in groups:
    m = df.collection_group_id == g
    sub = df[m]
    keep_no_shallow.append(sub.index[sub["mid"] != sub["mid"].min()])
idx2 = np.concatenate(keep_no_shallow)
codes = pd.Categorical(df.loc[idx2, "collection_group_id"], categories=groups).codes.astype(int)
uu = np.unique(codes); remap = {c: i for i, c in enumerate(uu)}
f_ns = cluster_ols(ltot[idx2], mid[idx2], np.array([remap[c] for c in codes]))
res["drop_shallowest_interval_per_group"] = {
    "n": int(len(idx2)), "slope": f_ns["slope"], "se": f_ns["se_cluster_df4"],
    "p": f_ns["p_cluster_df4"], "df": f_ns["df_cluster"], "ci95": f_ns["ci95_df4"],
    "pooled_spearman_rho": spearman(mid[idx2], ltot[idx2])[0],
    "pooled_spearman_p": spearman(mid[idx2], ltot[idx2])[1],
}
# p-value across many variants (multiplicity-of-specification count)
res["specification_scan"] = {
    "variants_with_p_lt_0.05": [],
    "table": [],
}
spec = [
    ("pooled spearman (2-sided)", spearman(mid, ltot)[1]),
    ("pooled spearman (1-sided neg)", spearman(mid, ltot)[1] / 2),
    ("pooled pearson", pearson(mid, ltot)[1]),
    ("FE slope cluster SE df4", fe["p_cluster_df4"]),
    ("FE slope naive df28 (anticonservative)", fe["p_naive_df28"]),
    ("FE relative depth", ferel["p_cluster_df4"]),
    ("FE depth_max", fedm["p_cluster_df4"]),
    ("two-stage slope t-test", 2 * stats.t.sf(abs(tt), G - 1)),
    ("two-stage sign test", sign_p),
    ("deepest vs shallowest paired t", 2 * stats.t.sf(abs(t_d), G - 1)),
    ("deepest vs shallowest wilcoxon", float(stats.wilcoxon(d).pvalue)),
    ("drop deepest per group FE", f_nd["p_cluster_df4"]),
    ("drop shallowest per group FE", f_ns["p_cluster_df4"]),
    ("exclude group_01 FE", f_no1["p_cluster_df4"]),
    ("cluster permutation FE", float((np.abs(perm) >= abs(obs) - 1e-12).mean())),
    ("cluster permutation pooled spearman", float((np.abs(perm_rho) >= abs(spearman(mid, ltot)[0]) - 1e-12).mean())),
    ("Stouffer combined group rho", float(2 * stats.norm.sf(abs(Zst)))),
    ("Fisher combined group rho", float(stats.chi2.sf(fisher, 2 * G))),
    ("group-mean regression n=5 pearson", pearson(gmean_mid, gmean_l)[1]),
    ("group-mean regression n=5 spearman", spearman(gmean_mid, gmean_l)[1]),
]
for nm, p in spec:
    res["specification_scan"]["table"].append({"spec": nm, "p": float(p)})
    if p < 0.05:
        res["specification_scan"]["variants_with_p_lt_0.05"].append(nm)

# ------------------------------------------------------------------ code snapshot
res["code"] = Path(__file__).read_text()
res["software"] = {"numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": stats.__name__ and __import__("scipy").__version__,
                   "statsmodels": sm.__version__}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
print(json.dumps({k: v for k, v in res.items()
                  if k in ("pooled", "two_stage_group_as_unit", "fixed_effect_within_group",
                           "deepest_vs_shallowest", "exclude_group_01",
                           "cluster_permutation_test", "cluster_bootstrap",
                           "specification_scan", "group_mean_regression_n5")},
                 indent=2, default=str)[:6000])