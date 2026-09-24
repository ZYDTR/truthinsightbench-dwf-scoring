"""Figure 1 / Figure 2 production + independent adversarial re-testing of the two
within-group claims (F1: total particle concentration falls with depth inside a
collection group; F2: total particle concentration falls as chlorophyll a rises
inside a collection group).

Written from scratch: nothing is imported from output/analysis/common.py or from
any earlier verification script.  Statistics are implemented directly with
numpy/scipy; figures with matplotlib ('Agg').

Outputs
  output/figures/fig1_depth_abundance.png
  output/figures/fig2_chlorophyll_abundance.png
  output/results/fig1_depth_abundance.csv
  output/results/fig2_chlorophyll_abundance.csv
  output/results/10_figures_adversarial.json   (every computed number)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
DATA = PROJ / "data"
FIG = PROJ / "output" / "figures"
RES = PROJ / "output" / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- load + checks
comp = pd.read_csv(DATA / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(DATA / "sample_environment_profiles.tsv", sep="\t")
assert len(comp) == len(env) == 30
m = comp[["sample_id", "collection_group_id", "depth_min_m", "depth_max_m",
          "particle_total_per_L"]].merge(
    env[["sample_id", "collection_group_id", "ice_regime", "depth_min_m", "depth_max_m",
         "chlorophyll_a_ug_L", "particle_total_per_L",
         "particle_total_std_or_error_per_L"]],
    on="sample_id", suffixes=("", "_env"), validate="one_to_one")
assert np.allclose(m["depth_min_m"], m["depth_min_m_env"])
assert np.allclose(m["depth_max_m"], m["depth_max_m_env"])
assert np.allclose(m["particle_total_per_L"], m["particle_total_per_L_env"])
assert (m["collection_group_id"] == m["collection_group_id_env"]).all()
m = m.drop(columns=[c for c in m.columns if c.endswith("_env")])
assert not m[["particle_total_per_L", "chlorophyll_a_ug_L"]].isna().any().any()
assert (m["particle_total_per_L"] > 0).all() and (m["chlorophyll_a_ug_L"] > 0).all()

m["depth_mid_m"] = (m["depth_min_m"] + m["depth_max_m"]) / 2.0
m["log10_total"] = np.log10(m["particle_total_per_L"].to_numpy(float))
m["log10_chl_a"] = np.log10(m["chlorophyll_a_ug_L"].to_numpy(float))
m = m.sort_values(["collection_group_id", "depth_mid_m"]).reset_index(drop=True)
GROUPS = sorted(m["collection_group_id"].unique())
G = len(GROUPS)
gidx = m["collection_group_id"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(int)
n = len(m)
mid = m["depth_mid_m"].to_numpy(float)
ltot = m["log10_total"].to_numpy(float)
lchl = m["log10_chl_a"].to_numpy(float)
tot = m["particle_total_per_L"].to_numpy(float)
chl = m["chlorophyll_a_ug_L"].to_numpy(float)
sid = m["sample_id"].to_numpy()


# ------------------------------------------------------------------- estimators
def demean(v, g=gidx):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        s = g == k
        v[s] -= v[s].mean()
    return v


def ols(y, X):
    """OLS with classical and HC3 heteroskedasticity-robust covariance."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    nn, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    df = nn - k
    s2 = float(r @ r) / df
    cov_cl = s2 * XtXi
    # HC3: (1-h_ii)^-2 weighted meat
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    w = 1.0 / np.clip(1.0 - h, 1e-12, None) ** 2
    meat = (X * (w * r**2)[:, None]).T @ X
    cov_hc3 = XtXi @ meat @ XtXi
    se_cl = np.sqrt(np.maximum(np.diag(cov_cl), 0))
    se_hc3 = np.sqrt(np.maximum(np.diag(cov_hc3), 0))
    return {"beta": b, "resid": r, "df": df, "XtXi": XtXi, "cov_cl": cov_cl,
            "se_cl": se_cl, "se_hc3": se_hc3, "n": nn, "k": k}


def cr1(y, X, cl):
    """CR1 cluster-robust covariance, t inference with G-1 df."""
    f = ols(y, X)
    X = np.asarray(X, float)
    cl = np.asarray(cl)
    r = f["resid"]
    nn, k = X.shape
    Gc = len(np.unique(cl))
    meat = np.zeros((k, k))
    for kk in np.unique(cl):
        s = cl == kk
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (Gc / (Gc - 1.0)) * ((nn - 1.0) / (nn - k)) if Gc > 1 else 1.0
    cov = scale * f["XtXi"] @ meat @ f["XtXi"]
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    df = max(Gc - 1, 1)
    t = f["beta"] / se
    p = 2 * st.t.sf(np.abs(t), df)
    tc = st.t.ppf(0.975, df)
    return {"beta": f["beta"], "se": se, "t": t, "p": p, "df_cluster": df,
            "n_clusters": Gc, "ci95": np.c_[f["beta"] - tc * se, f["beta"] + tc * se],
            "se_cl": f["se_cl"], "p_cl": 2 * st.t.sf(np.abs(f["beta"] / f["se_cl"]),
                                                     f["df"]),
            "resid": f["resid"]}


def fe_slope(y, x, g=gidx):
    xd, yd = demean(x, g), demean(y, g)
    return float(xd @ yd / (xd @ xd))


def sub_cluster(mask):
    """Remap the global cluster index onto a boolean subset."""
    return pd.factorize(gidx[mask])[0]


def signflip_p(y, x, g=gidx):
    """Exact within-group sign-flip randomisation test of the FE slope."""
    obs = fe_slope(y, x, g)
    vals = []
    for mask in range(2 ** G):
        yv = np.asarray(y, float).copy()
        for k in range(G):
            if mask >> k & 1:
                yv[g == k] = -yv[g == k]
        vals.append(fe_slope(yv, x, g))
    vals = np.asarray(vals)
    return {"observed": obs, "n_configs": int(2 ** G),
            "p_one_sided": float((vals <= obs).mean()),
            "p_two_sided": float((np.abs(vals) >= abs(obs)).mean())}


def perm_within_group(y, x, g=gidx, B=20000, seed=7, drop=None):
    """Permute x within groups; recompute FE slope.  Null: no within-group
    association between x and y (depth exchangeable inside a group).

    Vectorised: lexsort by (group, random key) yields a within-group shuffle;
    groups stay contiguous so demeaning is a segment reduction.
    """
    rng = np.random.default_rng(seed)
    keep = np.ones(n, bool) if drop is None else (g != drop)
    gg, xx, yy = g[keep], x[keep], y[keep]
    # group sizes in ascending group order after lexsort
    sizes = np.array([int((gg == k).sum()) for k in range(G)])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    yd = demean(yy, gg)
    xd_obs = demean(xx, gg)
    obs = float(xd_obs @ yd / (xd_obs @ xd_obs))
    keys = rng.random((B, len(xx)))
    order = np.lexsort((keys, np.broadcast_to(gg, (B, len(gg)))), axis=1)
    X = xx[order]
    sizes = np.maximum(sizes, 1)   # unused trailing segments (drop=...) get size 0
    segmean = np.add.reduceat(X, starts, axis=1) / sizes
    Xd = X - np.repeat(segmean, sizes, axis=1)
    null = (Xd @ yd) / np.einsum("ij,ij->i", Xd, Xd)
    return {"observed": obs, "p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
            "p_one_sided_neg": float((null <= obs).mean()),
            "null_sd": float(null.std(ddof=1)), "B": B,
            "null_q": [float(np.quantile(null, q)) for q in (0.025, 0.5, 0.975)]}


def leave_one_group_out(y, x):
    out = {}
    for k, gk in enumerate(GROUPS):
        keep = gidx != k
        out[gk] = fe_slope(y[keep], x[keep], gidx[keep])
    return out


# ------------------------------------------------------------------ F1 numbers
f1 = {"claim": "within a collection group, total particle concentration decreases with depth",
      "per_group": {}, }
slopes = {}
for g in GROUPS:
    s = m["collection_group_id"] == g
    y = ltot[s]
    x = mid[s]
    fit = ols(y, np.c_[np.ones(s.sum()), x])
    b, se_c, se_h = fit["beta"][1], fit["se_cl"][1], fit["se_hc3"][1]
    df = fit["df"]
    tc = st.t.ppf(0.975, df)
    rho, rp = st.spearmanr(x, y)
    slopes[g] = b
    f1["per_group"][g] = {
        "n": int(s.sum()),
        "ols_slope_log10_per_m": float(b),
        "ols_se_classical": float(se_c),
        "ols_p_classical": float(2 * st.t.sf(abs(b / se_c), df)),
        "ols_se_hc3": float(se_h),
        "ols_ci95_hc3": [float(b - tc * se_h), float(b + tc * se_h)],
        "ci95_classical": [float(b - tc * se_c), float(b + tc * se_c)],
        "r2": float(1 - fit["resid"] @ fit["resid"] /
                    ((y - y.mean()) @ (y - y.mean()))),
        "spearman_rho": float(rho), "spearman_p": float(rp),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "depth_mid_range_m": [float(x.min()), float(x.max())],
        "ice_regime": str(m.loc[s, "ice_regime"].iloc[0]),
    }

fit_fe1 = cr1(demean(ltot), np.c_[np.ones(n), demean(mid)], gidx)
f1["fixed_effect_within_group"] = {
    "slope_log10_per_m": float(fit_fe1["beta"][1]),
    "se_CR1": float(fit_fe1["se"][1]), "t": float(fit_fe1["t"][1]),
    "p_CR1_df4": float(fit_fe1["p"][1]), "df_cluster": int(fit_fe1["df_cluster"]),
    "ci95_CR1": [float(fit_fe1["ci95"][1, 0]), float(fit_fe1["ci95"][1, 1])],
    "se_classical": float(fit_fe1["se_cl"][1]),
    "p_classical_df28": float(fit_fe1["p_cl"][1]),
    "r2_within": float(1 - fit_fe1["resid"] @ fit_fe1["resid"] /
                       (demean(ltot) @ demean(ltot))),
}
f1["signflip_exact"] = signflip_p(ltot, mid)
f1["within_group_permutation"] = perm_within_group(ltot, mid)
f1["permutation_drop_group_02"] = perm_within_group(ltot, mid, drop=1)
f1["permutation_raw_scale"] = perm_within_group(tot, mid, B=20000, seed=8)
# monotone rank-based within-group slope (Theil-Sen style median pairwise slope)
f1["theil_sen_within_group"] = {
    g: float(np.median([(ltot[i] - ltot[j]) / (mid[i] - mid[j])
                        for i, j in __import__("itertools").combinations(
                            np.where(gidx == k)[0], 2)]))
    for k, g in enumerate(GROUPS)}
f1["leave_one_group_out_FE"] = leave_one_group_out(ltot, mid)
xd, yd = demean(mid), demean(ltot)
num = {g: float(xd[gidx == k] @ yd[gidx == k]) for k, g in enumerate(GROUPS)}
den = float(xd @ xd)
f1["group_share_of_within_covariance"] = {
    "numerator_share": {g: num[g] / float(xd @ yd) for g in GROUPS},
    "denominator_share": {g: float(xd[gidx == k] @ xd[gidx == k]) / den
                          for k, g in enumerate(GROUPS)},
}
f1["group_mean_regression_n5"] = {
    "spearman_rho": float(st.spearmanr(
        [m.loc[m["collection_group_id"] == g, "depth_mid_m"].mean() for g in GROUPS],
        [m.loc[m["collection_group_id"] == g, "log10_total"].mean() for g in GROUPS])[0]),
    "note": "5 group means only; confounded with ice regime/station identity",
}
f1["two_stage_group_as_unit"] = {}
sl = np.array([f1["per_group"][g]["ols_slope_log10_per_m"] for g in GROUPS])
se = np.array([f1["per_group"][g]["ols_se_classical"] for g in GROUPS])
f1["two_stage_group_as_unit"] = {
    "mean_slope": float(sl.mean()), "sd_slope": float(sl.std(ddof=1)),
    "t": float(sl.mean() / (sl.std(ddof=1) / math.sqrt(G))), "df": G - 1,
    "p": float(2 * st.t.sf(abs(sl.mean() / (sl.std(ddof=1) / math.sqrt(G))), G - 1)),
    "n_negative": int((sl < 0).sum()),
    "sign_test_p_two_sided": float(st.binomtest(int((sl < 0).sum()), G, 0.5).pvalue),
    "inverse_variance_weighted": float((sl / se**2).sum() / (1 / se**2).sum()),
    "random_effects_DL": {},
}
# DerSimonian-Laird random effects on the 5 per-group slopes
w = 1.0 / se**2
mu_fixed = float((w * sl).sum() / w.sum())
Q = float((w * (sl - mu_fixed) ** 2).sum())
tau2 = max(0.0, (Q - (G - 1)) / (w.sum() - (w**2).sum() / w.sum()))
wr = 1.0 / (se**2 + tau2)
mu_re = float((wr * sl).sum() / wr.sum())
se_re = float(1.0 / math.sqrt(wr.sum()))
f1["two_stage_group_as_unit"]["random_effects_DL"] = {
    "mu": mu_re, "se": se_re, "z": mu_re / se_re,
    "p": float(2 * st.norm.sf(abs(mu_re / se_re))), "tau2": float(tau2),
    "Q": Q, "Q_p": float(st.chi2.sf(Q, G - 1)),
    "I2_percent": float(max(0.0, (Q - (G - 1)) / Q) * 100) if Q > 0 else 0.0,
}
f1["raw_scale_check"] = {
    "note": "arithmetic (not log10) within-group slope, to make sure the log "
            "transform is not creating the sign",
    "per_group_slope_per_L_per_m": {
        g: float(np.polyfit(mid[m["collection_group_id"] == g],
                            tot[m["collection_group_id"] == g], 1)[0])
        for g in GROUPS},
}

# ------------------------------------------------------------------ F2 numbers
f2 = {"claim": "within a collection group, total particle concentration decreases "
              "as chlorophyll a increases"}
f2["per_group"] = {}
for g in GROUPS:
    s = m["collection_group_id"] == g
    x, y = lchl[s], ltot[s]
    fit = ols(y, np.c_[np.ones(s.sum()), x])
    b, se_c, se_h = fit["beta"][1], fit["se_cl"][1], fit["se_hc3"][1]
    rho, rp = st.spearmanr(x, y)
    f2["per_group"][g] = {
        "n": int(s.sum()), "ols_slope_log10total_per_log10chl": float(b),
        "ols_se_classical": float(se_c),
        "ols_p_classical": float(2 * st.t.sf(abs(b / se_c), fit["df"])),
        "ols_se_hc3": float(se_h),
        "spearman_rho": float(rho), "spearman_p": float(rp),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "max_chl_ug_L": float(chl[s].max()),
    }
rhos = np.array([f2["per_group"][g]["spearman_rho"] for g in GROUPS])
f2["mean_within_group_spearman"] = {
    "mean_rho": float(rhos.mean()), "n_negative": int((rhos < 0).sum()),
    "per_group": {g: f2["per_group"][g]["spearman_rho"] for g in GROUPS},
    "stouffer_Z": float(rhos.sum() / math.sqrt(G)),
    "fisher_chi2": float(-2 * sum(math.log(st.spearmanr(
        lchl[m["collection_group_id"] == g], ltot[m["collection_group_id"] == g])[1])
        for g in GROUPS)),
}
fit_fe2 = cr1(demean(ltot), np.c_[np.ones(n), demean(lchl)], gidx)
f2["fixed_effect"] = {
    "slope_log10total_per_log10chl": float(fit_fe2["beta"][1]),
    "se_CR1": float(fit_fe2["se"][1]), "t": float(fit_fe2["t"][1]),
    "p_CR1_df4": float(fit_fe2["p"][1]), "df_cluster": int(fit_fe2["df_cluster"]),
    "ci95_CR1": [float(fit_fe2["ci95"][1, 0]), float(fit_fe2["ci95"][1, 1])],
    "se_classical": float(fit_fe2["se_cl"][1]),
    "p_classical_df28": float(fit_fe2["p_cl"][1]),
    "r2_within": float(1 - fit_fe2["resid"] @ fit_fe2["resid"] /
                       (demean(ltot) @ demean(ltot))),
    "factor_per_10x_chl": float(10 ** fit_fe2["beta"][1]),
}
f2["signflip_exact"] = signflip_p(ltot, lchl)
f2["within_group_permutation"] = perm_within_group(ltot, lchl)
f2["permutation_drop_group_02"] = perm_within_group(ltot, lchl, drop=1)
f2["theil_sen_within_group"] = {
    g: float(np.median([(ltot[i] - ltot[j]) / (lchl[i] - lchl[j])
                        for i, j in __import__("itertools").combinations(
                            np.where(gidx == k)[0], 2)]))
    for k, g in enumerate(GROUPS)}
f2["leave_one_group_out_FE"] = leave_one_group_out(ltot, lchl)
xd2, yd2 = demean(lchl), demean(ltot)
f2["group_share_of_within_covariance"] = {
    "numerator_share": {g: float(xd2[gidx == k] @ yd2[gidx == k] / float(xd2 @ yd2))
                        for k, g in enumerate(GROUPS)},
    "denominator_share": {g: float(xd2[gidx == k] @ xd2[gidx == k] / float(xd2 @ xd2))
                          for k, g in enumerate(GROUPS)},
}
# deepest interval of each group
deep = {}
for g in GROUPS:
    s = np.where(gidx == np.where(np.array(GROUPS) == g)[0][0])[0]
    deep[g] = int(s[np.argmax(mid[s])])
deep_idx = np.array([deep[g] for g in GROUPS])
f2["deepest_interval_diagnostics"] = {
    "n_deepest_is_max_chl": int(sum(
        chl[deep[g]] == chl[gidx == k].max() for k, g in enumerate(GROUPS))),
    "detail": {g: {"sample_id": str(sid[deep[g]]),
                   "depth_mid_m": float(mid[deep[g]]),
                   "chl_a": float(chl[deep[g]]),
                   "total_per_L": float(tot[deep[g]])} for g in GROUPS},
}
keep = np.ones(n, bool)
keep[deep_idx] = False
gc = sub_cluster(keep)
fit_d = cr1(demean(ltot[keep], gc), np.c_[np.ones(keep.sum()), demean(lchl[keep], gc)], gc)
f2["exclude_deepest_per_group"] = {
    "n": int(keep.sum()), "slope": float(fit_d["beta"][1]),
    "se_CR1": float(fit_d["se"][1]), "p_CR1": float(fit_d["p"][1]),
    "ci95": [float(fit_d["ci95"][1, 0]), float(fit_d["ci95"][1, 1])],
}
# joint models
Xj = np.c_[np.ones(n), demean(lchl), demean(mid)]
fj = cr1(demean(ltot), Xj, gidx)
f2["joint_chl_plus_depth"] = {
    "terms": ["intercept", "log10_chl_a", "depth_mid_m"],
    "beta": [float(v) for v in fj["beta"]],
    "se_CR1": [float(v) for v in fj["se"]],
    "p_CR1": [float(v) for v in fj["p"]],
}
poc = np.log10(m["particulate_organic_carbon_mg_L"].to_numpy(float)) if "particulate_organic_carbon_mg_L" in m else None
if poc is None:
    poc = np.log10(env.set_index("sample_id").loc[sid, "particulate_organic_carbon_mg_L"].to_numpy(float))
Xj2 = np.c_[np.ones(n), demean(lchl), demean(mid), demean(poc)]
fj2 = cr1(demean(ltot), Xj2, gidx)
f2["joint_chl_depth_POC"] = {
    "terms": ["intercept", "log10_chl_a", "depth_mid_m", "log10_POC"],
    "beta": [float(v) for v in fj2["beta"]],
    "se_CR1": [float(v) for v in fj2["se"]],
    "p_CR1": [float(v) for v in fj2["p"]],
}
# how much of the within-group chl-total association survives removing the
# 3 highest-chlorophyll intervals (012, 017, 024 = the deep chl maxima)
hi = np.argsort(chl)[-3:]
keeph = np.ones(n, bool)
keeph[hi] = False
gch = sub_cluster(keeph)
fith = cr1(demean(ltot[keeph], gch), np.c_[np.ones(keeph.sum()),
                                           demean(lchl[keeph], gch)], gch)
f2["exclude_3_highest_chl_intervals"] = {
    "excluded": [str(sid[i]) for i in hi], "n": int(keeph.sum()),
    "slope": float(fith["beta"][1]), "se_CR1": float(fith["se"][1]),
    "p_CR1": float(fith["p"][1]),
}
f2["chl_quantisation"] = {
    "unique_values": int(m["chlorophyll_a_ug_L"].nunique()),
    "min_step": float(np.diff(np.sort(m["chlorophyll_a_ug_L"].unique())).min()),
    "n_at_or_below_0p1": int((chl <= 0.1).sum()),
    "share_of_chl_variance_from_3_largest": float(
        np.sort(chl)[-3:].var() / chl.var()) if chl.var() > 0 else None,
}

# --------------------------------------------------------------- figure 1 plot
fig = plt.figure(figsize=(13, 3.6))
gs = GridSpec(2, 5, figure=fig, height_ratios=[2.7, 1.25], hspace=0.62, wspace=0.35)
order_all = {}
for i, g in enumerate(GROUPS):
    ax = fig.add_subplot(gs[0, i])
    s = m["collection_group_id"] == g
    sub = m[s].sort_values("depth_mid_m")
    order_all[g] = sub["sample_id"].tolist()
    x = sub["log10_total"].to_numpy(float)
    xr = sub["particle_total_per_L"].to_numpy(float)
    y = sub["depth_mid_m"].to_numpy(float)
    ax.plot(xr, y, "-", color="0.55", lw=0.9, zorder=1)
    ax.scatter(xr, y, s=22, color="#1f77b4", zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_title(f"{g}  (n={int(s.sum())})", fontsize=8)
    ax.invert_yaxis()
    ax.margins(x=0.22, y=0.10)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(
        base=10, subs=np.arange(2, 10) * 0.1, numticks=60))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(
        labelOnlyBase=True))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(labelsize=6)
    # label only the extreme points (max/min total, deepest interval), and keep
    # the text inside the axes box with explicit corner assignment
    lab_idx = {int(np.argmax(x)): 0.97, int(np.argmin(x)): 0.03, int(np.argmax(y)): 0.55}
    for li, xfrac in lab_idx.items():
        is_top = y[li] < 0.5 * (y.min() + y.max())   # shallower than the mid depth
        ax.annotate(sub["sample_id"].iloc[li], (xr[li], y[li]),
                    xytext=(xfrac, 0.02 if not is_top else 0.86),
                    textcoords="axes fraction", fontsize=5.0, color="#b2182b",
                    ha="left" if xfrac < 0.5 else "right", va="bottom", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#b2182b", lw=0.5,
                                    shrinkA=0.0, shrinkB=2.0))
    ax.set_xlabel("total particles / L (log10 scale)", fontsize=6.5)
    if i == 0:
        ax.set_ylabel("depth mid (m)", fontsize=7)
    ax.text(0.97, 0.03,
            f"OLS slope = {f1['per_group'][g]['ols_slope_log10_per_m']:+.2f}\n"
            f"(log10 / m)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color="0.25")
    ax.grid(alpha=0.25, lw=0.4)

axb = fig.add_subplot(gs[1, :])
xs = np.arange(G)
bs = np.array([f1["per_group"][g]["ols_slope_log10_per_m"] for g in GROUPS])
los = np.array([f1["per_group"][g]["ols_ci95_hc3"][0] for g in GROUPS])
his = np.array([f1["per_group"][g]["ols_ci95_hc3"][1] for g in GROUPS])
axb.errorbar(xs, bs, yerr=[bs - los, his - bs], fmt="o", ms=5, capsize=3,
             color="#1f77b4", lw=1.2, label="per-group OLS slope, 95% CI (HC3 robust)")
axb.axhline(0.0, color="k", lw=0.8, ls="--")
fe = f1["fixed_effect_within_group"]
axb.errorbar([G + 0.3], [fe["slope_log10_per_m"]],
             yerr=[[fe["slope_log10_per_m"] - fe["ci95_CR1"][0]],
                   [fe["ci95_CR1"][1] - fe["slope_log10_per_m"]]],
             fmt="D", ms=6, capsize=3, color="#b2182b", lw=1.6,
             label="pooled within-group slope, 95% CI (CR1 cluster-robust, df=4)")
axb.set_xticks(list(xs) + [G + 0.3])
axb.set_xticklabels(GROUPS + ["pooled\nFE"], fontsize=6.5)
axb.set_ylabel("slope of log10(total)\nper metre", fontsize=7)
axb.set_title("within-group depth slope of log10(total) (log10 particles/L per m); "
              "negative = decreasing with depth", fontsize=7.5)
axb.tick_params(labelsize=6)
axb.legend(fontsize=5.6, loc="lower left", framealpha=0.9)
axb.grid(alpha=0.25, lw=0.4)
fig.tight_layout()
fig.savefig(FIG / "fig1_depth_abundance.png", dpi=200)
plt.close(fig)

fig1_csv = m.copy()
fig1_csv["within_group_slope_log10_per_m"] = fig1_csv["collection_group_id"].map(slopes)
fig1_csv = fig1_csv[["collection_group_id", "sample_id", "depth_mid_m", "depth_min_m",
                     "depth_max_m", "particle_total_per_L", "log10_total",
                     "within_group_slope_log10_per_m"]]
fig1_csv.to_csv(RES / "fig1_depth_abundance.csv", index=False)

# --------------------------------------------------------------- figure 2 plot
fig = plt.figure(figsize=(13, 3.6))
gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.055], wspace=0.30)
norm = plt.Normalize(vmin=float(mid.min()), vmax=float(mid.max()))
cmap = plt.get_cmap("viridis")
last_sc = None
for i, g in enumerate(GROUPS):
    ax = fig.add_subplot(gs[0, i])
    s = m["collection_group_id"] == g
    sub = m[s]
    x = sub["chlorophyll_a_ug_L"].to_numpy(float)
    y = sub["particle_total_per_L"].to_numpy(float)
    d = sub["depth_mid_m"].to_numpy(float)
    fit = ols(np.log10(y), np.c_[np.ones(len(x)), np.log10(x)])
    b, a = fit["beta"][1], fit["beta"][0]
    gx = np.linspace(np.log10(x.min()), np.log10(x.max()), 50)
    ax.plot(10 ** gx, 10 ** (a + b * gx), "-", color="#333333", lw=1.0, zorder=2,
            label=f"OLS slope {b:+.2f}")
    sc = ax.scatter(x, y, c=d, cmap=cmap, norm=norm, s=34, zorder=3,
                    edgecolor="white", linewidth=0.5)
    di = deep[g]
    ax.scatter([chl[di]], [tot[di]], facecolor="none", edgecolor="#d62728",
               s=95, linewidth=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # thin out log tick labels: keep at most ~5 per decade
    for axis, sc_ in ((ax.xaxis, "log"), (ax.yaxis, "log")):
        axis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(
            labelOnlyBase=True, minor_thresholds=(3.0, 0.5)))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=7))
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=7))
    rho, rp = st.spearmanr(x, y)
    ax.text(0.04, 0.05, f"Spearman rho = {rho:+.3f}\np = {rp:.3f}",
            transform=ax.transAxes, fontsize=6, color="#b2182b", va="bottom")
    ax.set_title(f"{g}  (n={int(s.sum())})", fontsize=8)
    ax.set_xlabel("chlorophyll a (ug/L, log10)", fontsize=6.5)
    if i == 0:
        ax.set_ylabel("total particles / L (log10)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=5.4, loc="upper right", framealpha=0.9)
    last_sc = sc
cax = fig.add_subplot(gs[0, 5])
cb = fig.colorbar(last_sc, cax=cax)
cb.set_label("depth mid (m)", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.text(0.5, 0.005, "red open circle = deepest interval of the group",
         ha="center", fontsize=6, color="#d62728")
fig.tight_layout()
fig.savefig(FIG / "fig2_chlorophyll_abundance.png", dpi=200)
plt.close(fig)

f2_csv = m.copy()
f2_csv["is_deepest_interval_of_group"] = False
f2_csv.loc[deep_idx, "is_deepest_interval_of_group"] = True
f2_csv = f2_csv[["collection_group_id", "sample_id", "depth_mid_m",
                 "chlorophyll_a_ug_L", "log10_chl_a", "particle_total_per_L",
                 "log10_total", "is_deepest_interval_of_group"]]
f2_csv.to_csv(RES / "fig2_chlorophyll_abundance.csv", index=False)

# ------------------------------------------------------------------- output
summary = {
    "data_checks": {
        "n_intervals": n, "n_groups": G,
        "n_per_group": {g: int((m["collection_group_id"] == g).sum()) for g in GROUPS},
        "comp_vs_env_depth_and_total_agree": True,
        "log10_transform_ok_min_total": float(tot.min()),
        "min_chl": float(chl.min()),
    },
    "F1_depth": f1,
    "F2_chlorophyll": f2,
    "figures": {
        "fig1": str(FIG / "fig1_depth_abundance.png"),
        "fig2": str(FIG / "fig2_chlorophyll_abundance.png"),
    },
    "csvs": {
        "fig1": str(RES / "fig1_depth_abundance.csv"),
        "fig2": str(RES / "fig2_chlorophyll_abundance.csv"),
    },
}
(RES / "10_figures_adversarial.json").write_text(json.dumps(summary, indent=1))

print(json.dumps({
    "F1_per_group_slopes": {g: round(f1["per_group"][g]["ols_slope_log10_per_m"], 3)
                            for g in GROUPS},
    "F1_FE_slope_CR1": [round(fe["slope_log10_per_m"], 3), round(fe["se_CR1"], 3),
                        round(fe["p_CR1_df4"], 4), fe["ci95_CR1"]],
    "F1_signflip": f1["signflip_exact"],
    "F1_perm": {k: f1["within_group_permutation"][k] for k in
                ("observed", "p_two_sided", "p_one_sided_neg", "null_sd")},
    "F1_perm_drop_g02": {k: f1["permutation_drop_group_02"][k] for k in
                         ("observed", "p_two_sided", "p_one_sided_neg")},
    "F1_perm_raw_scale": {k: f1["permutation_raw_scale"][k] for k in
                          ("observed", "p_two_sided", "p_one_sided_neg")},
    "F1_theil_sen": {k: round(v, 3) for k, v in f1["theil_sen_within_group"].items()},
    "F1_LOO": {k: round(v, 3) for k, v in f1["leave_one_group_out_FE"].items()},
    "F1_cov_shares": {k: round(v, 3) for k, v in
                      f1["group_share_of_within_covariance"]["numerator_share"].items()},
    "F1_two_stage": {k: f1["two_stage_group_as_unit"][k]
                     for k in ("mean_slope", "p", "n_negative", "sign_test_p_two_sided",
                               "inverse_variance_weighted", "random_effects_DL")},
    "F2_per_group_rho": {g: [round(f2["per_group"][g]["spearman_rho"], 3),
                             round(f2["per_group"][g]["spearman_p"], 3)]
                         for g in GROUPS},
    "F2_mean_rho": round(float(rhos.mean()), 3),
    "F2_FE": [round(fit_fe2["beta"][1], 3), round(fit_fe2["se"][1], 3),
              round(fit_fe2["p"][1], 4), [round(v, 3) for v in fit_fe2["ci95"][1]]],
    "F2_signflip": f2["signflip_exact"],
    "F2_perm": {k: f2["within_group_permutation"][k] for k in
                ("observed", "p_two_sided", "p_one_sided_neg", "null_sd")},
    "F2_perm_drop_g02": {k: f2["permutation_drop_group_02"][k] for k in
                         ("observed", "p_two_sided", "p_one_sided_neg")},
    "F2_theil_sen": {k: round(v, 3) for k, v in f2["theil_sen_within_group"].items()},
    "F2_LOO": {k: round(v, 3) for k, v in f2["leave_one_group_out_FE"].items()},
    "F2_cov_shares": {k: round(v, 3) for k, v in
                      f2["group_share_of_within_covariance"]["numerator_share"].items()},
    "F2_excl_deepest": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in f2["exclude_deepest_per_group"].items()},
    "F2_joint_chl_depth": f2["joint_chl_plus_depth"],
    "F2_joint_chl_depth_POC": f2["joint_chl_depth_POC"],
    "F2_excl_3_hi_chl": f2["exclude_3_highest_chl_intervals"],
    "F2_deepest_is_max_chl": f2["deepest_interval_diagnostics"]["n_deepest_is_max_chl"],
    "fig_sizes": {p.name: p.stat().st_size for p in
                  [FIG / "fig1_depth_abundance.png", FIG / "fig2_chlorophyll_abundance.png"]},
}, indent=1))