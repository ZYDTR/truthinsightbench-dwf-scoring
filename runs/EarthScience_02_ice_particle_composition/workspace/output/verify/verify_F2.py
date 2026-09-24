"""Adversarial verification of CLAIM F2.

Claim: within a collection group, total particle concentration falls as
chlorophyll a rises; slope -1.143 log10(total) per log10(chl-a) (cluster-robust
SE 0.329, p=0.025, df=4); mean within-group Spearman = -0.496 (4/5 negative);
deepest-interval drop gives -1.782 p=0.003; joint model with depth+POC gives
-1.70 (p=0.014) and POC +1.27 (p=0.013).

We recompute everything from the raw TSVs. No code from output/analysis is used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
DATA = PROJ / "data"
OUT = PROJ / "output" / "verify" / "verify_F2.json"

comp = pd.read_csv(DATA / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(DATA / "sample_environment_profiles.tsv", sep="\t")
m = comp.merge(env, on=["sample_id", "collection_group_id"], suffixes=("", "_env"),
               validate="one_to_one")
# environment file also carries depth_min_m/depth_max_m and particle_total_per_L;
# verify the two files agree on those before using them.
assert np.allclose(m["depth_min_m"], m["depth_min_m_env"])
assert np.allclose(m["depth_max_m"], m["depth_max_m_env"])
assert np.allclose(m["particle_total_per_L"], m["particle_total_per_L_env"])
# cross-check the percent / _per_L pairing.  NOTE: contrary to the stated key
# structure, the COUNT columns are the primary quantities and the _percent
# columns are 2-dp-roundings of count/total*100 (max residual 0.23 pp over the
# table).  Direction matters for provenance, not for F2, which uses only
# particle_total_per_L and chlorophyll_a_ug_L.
deriv = {"n_pairs_checked": 0, "max_abs_count_minus_pct_over_100_times_total": 0.0,
         "max_abs_pct_minus_count_over_total_times_100": 0.0,
         "direction": "count primary, percent rounded from count"}
for col in [c for c in comp.columns if c.endswith("_per_L")
            and (c.startswith("polymer_count_") or c.startswith("particle_size_"))]:
    if col.startswith("polymer_count_"):
        pct = col.replace("polymer_count_", "polymer_").replace("_per_L", "_percent")
    else:
        pct = col.replace("_per_L", "_percent")
    if pct not in comp.columns:
        continue
    deriv["n_pairs_checked"] += 1
    deriv["max_abs_count_minus_pct_over_100_times_total"] = max(
        deriv["max_abs_count_minus_pct_over_100_times_total"],
        float(np.nanmax(np.abs(m[col] - m[pct] / 100.0 * m["particle_total_per_L"]))))
    deriv["max_abs_pct_minus_count_over_total_times_100"] = max(
        deriv["max_abs_pct_minus_count_over_total_times_100"],
        float(np.nanmax(np.abs(m[pct] - m[col] / m["particle_total_per_L"] * 100.0))))
assert deriv["max_abs_pct_minus_count_over_total_times_100"] < 1.0  # rounding only
# the particle_total_per_L column must agree between the two files
assert np.allclose(comp["particle_total_per_L"], env["particle_total_per_L"].to_numpy(float))

m["depth_mid_m"] = (m["depth_min_m"] + m["depth_max_m"]) / 2.0
m["group"] = m["collection_group_id"]
groups = sorted(m["group"].unique())
n = len(m)
tot = m["particle_total_per_L"].to_numpy(float)
chl = m["chlorophyll_a_ug_L"].to_numpy(float)
mid = m["depth_mid_m"].to_numpy(float)
poc = m["particulate_organic_carbon_mg_L"].to_numpy(float)
pon = m["particulate_organic_nitrogen_ug_L"].to_numpy(float)
sal = m["salinity_PSU"].to_numpy(float)
gi = m["group"].map({g: i for i, g in enumerate(groups)}).to_numpy(int)
ltot = np.log10(tot)
lchl = np.log10(chl)
lpoc = np.log10(poc)
lpon = np.log10(pon)

res = {
    "claim_id": "F2",
    "data": {
        "n_intervals": int(n),
        "n_groups": len(groups),
        "n_per_group": {g: int((m["group"] == g).sum()) for g in groups},
        "derived_column_check": deriv,
        "env_vs_comp_depth_and_total_consistent": True,
    },
}


def demean(v, g=gi):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        s = g == k
        v[s] -= np.nanmean(v[s])
    return v


def ols_fit(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    df = n - k
    s2 = float(r @ r) / df
    cov = s2 * XtXi
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * st.t.sf(np.abs(t), df)
    return {"beta": b, "se": se, "t": t, "p": p, "df": df, "resid": r, "XtXi": XtXi}


def cluster_fit(y, X, cl):
    """OLS + CR1 cluster-robust SE at group level, t with G-1 df."""
    f = ols_fit(y, X)
    X = np.asarray(X, float)
    r = f["resid"]
    cl = np.asarray(cl)
    G = len(np.unique(cl))
    k = X.shape[1]
    n = X.shape[0]
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        s = cl == g
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    cov = scale * f["XtXi"] @ meat @ f["XtXi"]
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = f["beta"] / se
    df = G - 1
    p = 2 * st.t.sf(np.abs(t), df)
    tc = st.t.ppf(0.975, df)
    return {"beta": f["beta"], "se": se, "t": t, "p": p, "df": df, "cov": cov,
            "ci95": np.c_[f["beta"] - tc * se, f["beta"] + tc * se]}


def fe(y, x, cl=None):
    """Within-group (demeaned) slope.  cl defaults to the global group index
    (use only on full-length vectors)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    cl = gi if cl is None else np.asarray(cl)
    assert len(y) == len(x) == len(cl)
    X = np.c_[np.ones(len(y)), demean(x, cl)]
    return cluster_fit(demean(y, cl), X, cl)


# ---------------------------------------------------------------- 1. raw within-group Spearman
rhos, rho_ps, rho_ns = [], [], []
for g in groups:
    s = m["group"] == g
    rho, p = st.spearmanr(lchl[s], ltot[s])
    rhos.append(rho)
    rho_ps.append(p)
    rho_ns.append(int(s.sum()))
res["within_group_spearman"] = {
    "per_group": {g: {"rho": float(r), "p": float(p), "n": int(nn)}
                  for g, r, p, nn in zip(groups, rhos, rho_ps, rho_ns)},
    "mean_rho": float(np.mean(rhos)),
    "mean_rho_rounded_by_claim": 0.496,
    "n_negative": int(sum(r < 0 for r in rhos)),
    "n_positive": int(sum(r > 0 for r in rhos)),
    "min_abs_n_for_rho": min(rho_ns),
}
# Fisher-z mean as an alternative aggregation
z = np.arctanh(rhos)
res["within_group_spearman"]["fisher_z_mean"] = float(np.tanh(z.mean()))

# groups with only 2 intervals cannot yield a rho (if any)
res["within_group_spearman"]["groups_with_n_lt_3"] = [g for g, nn in zip(groups, rho_ns) if nn < 3]

# ---------------------------------------------------------------- 2. fixed-effect slope (claim: -1.143, SE .329, p .025)
f = fe(ltot, lchl)
res["fe_slope_log10total_per_log10chl"] = {
    "slope": float(f["beta"][1]), "se_cluster": float(f["se"][1]),
    "t": float(f["t"][1]), "p_cluster": float(f["p"][1]), "df": int(f["df"]),
    "ci95_cluster": [float(f["ci95"][1, 0]), float(f["ci95"][1, 1])],
    "factor_change_per_10x_chl": float(10 ** f["beta"][1]),
}
# classical (non-clustered) SE for contrast
Xfe = np.c_[np.ones(n), demean(lchl)]
fo = ols_fit(demean(ltot), Xfe)
res["fe_slope_log10total_per_log10chl"]["se_classical"] = float(fo["se"][1])
res["fe_slope_log10total_per_log10chl"]["p_classical"] = float(fo["p"][1])
# R^2 of the within model
res["fe_slope_log10total_per_log10chl"]["r2_within"] = float(
    1 - (fo["resid"] @ fo["resid"]) / (demean(ltot) @ demean(ltot)))

# ---------------------------------------------------------------- 3. per-group OLS slopes
per_group_slope = {}
for g in groups:
    s = m["group"] == g
    yv = ltot[s] - ltot[s].mean()
    xv = lchl[s] - lchl[s].mean()
    if xv @ xv > 0:
        b = (xv @ yv) / (xv @ xv)
        r = np.corrcoef(xv, yv)[0, 1]
        # classical p for the slope
        nn = int(s.sum())
        se = math.sqrt((((yv - b * xv) ** 2).sum() / (nn - 2)) / (xv @ xv)) if nn > 2 else np.nan
        t = b / se if se and np.isfinite(se) else np.nan
        p = 2 * st.t.sf(abs(t), nn - 2) if np.isfinite(t) else np.nan
    else:
        b, r, p, se = np.nan, np.nan, np.nan, np.nan
    per_group_slope[g] = {"n": int(s.sum()), "slope": float(b), "r": float(r),
                          "se": float(se) if np.isfinite(se) else None,
                          "p": float(p) if np.isfinite(p) else None}
res["per_group_ols_slope"] = per_group_slope

# ---------------------------------------------------------------- 4. deepest interval diagnostic
deep_idx, top_idx = [], []
for g in groups:
    s = np.where(m["group"].to_numpy() == g)[0]
    deep_idx.append(s[np.argmax(mid[s])])
    top_idx.append(s[np.argmin(mid[s])])
deep_idx = np.array(deep_idx)
top_idx = np.array(top_idx)
res["deepest_intervals"] = {
    "detail": [{"group": groups[k], "sample": m["sample_id"].iloc[deep_idx[k]],
                "depth_mid_m": float(mid[deep_idx[k]]),
                "chl_a": float(chl[deep_idx[k]]), "total_per_L": float(tot[deep_idx[k]]),
                "rank_chl_within_group": int(
                    st.rankdata(chl[m["group"].to_numpy() == groups[k]])[
                        np.where(m["sample_id"].to_numpy() == m["sample_id"].iloc[deep_idx[k]])[0][0]
                        - np.where(m["group"].to_numpy() == groups[k])[0][0]]),
                "is_max_chl_in_group": bool(
                    chl[deep_idx[k]] == chl[m["group"].to_numpy() == groups[k]].max())}
               for k in range(len(groups))],
    "n_deepest_is_max_chl": int(sum(
        chl[deep_idx[k]] == chl[m["group"].to_numpy() == groups[k]].max()
        for k in range(len(groups)))),
    "n_deepest_is_min_total": int(sum(
        tot[deep_idx[k]] == tot[m["group"].to_numpy() == groups[k]].min()
        for k in range(len(groups)))),
    "mean_chl_deepest": float(chl[deep_idx].mean()),
    "mean_chl_non_deepest": float(np.delete(chl, deep_idx).mean()),
    "mean_log10total_deepest": float(ltot[deep_idx].mean()),
    "mean_log10total_non_deepest": float(np.delete(ltot, deep_idx).mean()),
    "median_chl_deepest": float(np.median(chl[deep_idx])),
    "median_chl_non_deepest": float(np.median(np.delete(chl, deep_idx))),
    "median_total_deepest": float(np.median(tot[deep_idx])),
    "median_total_non_deepest": float(np.median(np.delete(tot, deep_idx))),
}

# ---------------------------------------------------------------- 5. drop deepest interval per group
keep = np.ones(n, bool)
keep[deep_idx] = False
f2v = fe(ltot[keep], lchl[keep], gi[keep])
rhos_d = []
for g in groups:
    s = (m["group"] == g).to_numpy() & keep
    rhos_d.append(float(st.spearmanr(lchl[s], ltot[s])[0]) if s.sum() >= 3 else np.nan)
res["exclude_deepest_interval_per_group"] = {
    "n_remaining": int(keep.sum()),
    "mean_rho": float(np.nanmean(rhos_d)),
    "per_group_rho": {g: (None if not np.isfinite(r) else float(r))
                      for g, r in zip(groups, rhos_d)},
    "slope": float(f2v["beta"][1]), "se_cluster": float(f2v["se"][1]),
    "p_cluster": float(f2v["p"][1]), "df": int(f2v["df"]),
    "ci95_cluster": [float(f2v["ci95"][1, 0]), float(f2v["ci95"][1, 1])],
}
# also drop deepest + highest-chl interval
keep2 = keep.copy()
for g in groups:
    s = np.where((m["group"] == g).to_numpy())[0]
    keep2[s[np.argmax(chl[s])]] = False
f2v2 = fe(ltot[keep2], lchl[keep2], gi[keep2])
res["exclude_deepest_and_highest_chl_per_group"] = {
    "n_remaining": int(keep2.sum()), "slope": float(f2v2["beta"][1]),
    "se_cluster": float(f2v2["se"][1]), "p_cluster": float(f2v2["p"][1]),
    "df": int(f2v2["df"]),
}
# drop ONLY the highest-chl interval per group
keep3 = np.ones(n, bool)
for g in groups:
    s = np.where((m["group"] == g).to_numpy())[0]
    keep3[s[np.argmax(chl[s])]] = False
f2v3 = fe(ltot[keep3], lchl[keep3], gi[keep3])
res["exclude_highest_chl_interval_per_group"] = {
    "n_remaining": int(keep3.sum()), "slope": float(f2v3["beta"][1]),
    "se_cluster": float(f2v3["se"][1]), "p_cluster": float(f2v3["p"][1]),
    "df": int(f2v3["df"]),
}
# drop ONLY the single most extreme chl interval across all 30 (global max)
kglob = np.ones(n, bool)
kglob[np.argmax(chl)] = False
f2vg = fe(ltot[kglob], lchl[kglob], gi[kglob])
res["exclude_global_max_chl_interval"] = {
    "excluded_sample": m["sample_id"].iloc[int(np.argmax(chl))],
    "slope": float(f2vg["beta"][1]), "se_cluster": float(f2vg["se"][1]),
    "p_cluster": float(f2vg["p"][1]), "df": int(f2vg["df"]),
}
# drop only the single deepest interval GLOBALLY (sample_030)
k1 = np.ones(n, bool)
k1[int(np.argmax(mid))] = False
f2v1 = fe(ltot[k1], lchl[k1], gi[k1])
res["exclude_global_deepest_interval"] = {
    "excluded_sample": m["sample_id"].iloc[int(np.argmax(mid))],
    "slope": float(f2v1["beta"][1]), "se_cluster": float(f2v1["se"][1]),
    "p_cluster": float(f2v1["p"][1]), "df": int(f2v1["df"]),
}

# ---------------------------------------------------------------- 6. joint model chl + depth + POC
Xj = np.c_[np.ones(n), demean(lchl), demean(mid), demean(lpoc)]
fj = cluster_fit(demean(ltot), Xj, gi)
names = ["intercept", "log10_chl_a", "depth_mid_m", "log10_POC"]
res["joint_model_chl_depth_POC"] = {
    nm: {"beta": float(fj["beta"][i]), "se_cluster": float(fj["se"][i]),
         "t": float(fj["t"][i]), "p_cluster": float(fj["p"][i]),
         "ci95": [float(fj["ci95"][i, 0]), float(fj["ci95"][i, 1])]}
    for i, nm in enumerate(names)} | {"df": int(fj["df"])}
# alternative joint: chl + depth + PON
Xj2 = np.c_[np.ones(n), demean(lchl), demean(mid), demean(lpon)]
fj2 = cluster_fit(demean(ltot), Xj2, gi)
res["joint_model_chl_depth_PON"] = {
    nm: {"beta": float(fj2["beta"][i]), "se_cluster": float(fj2["se"][i]),
         "p_cluster": float(fj2["p"][i])}
    for i, nm in enumerate(["intercept", "log10_chl_a", "depth_mid_m", "log10_PON"])}
# joint chl + depth (no POC)
Xj3 = np.c_[np.ones(n), demean(lchl), demean(mid)]
fj3 = cluster_fit(demean(ltot), Xj3, gi)
res["joint_model_chl_depth_only"] = {
    nm: {"beta": float(fj3["beta"][i]), "se_cluster": float(fj3["se"][i]),
         "p_cluster": float(fj3["p"][i])}
    for i, nm in enumerate(["intercept", "log10_chl_a", "depth_mid_m"])}
# VIF between demeaned chl and demeaned depth in the joint model
Xd = np.c_[demean(lchl), demean(mid)]
c = np.corrcoef(Xd.T)[0, 1]
res["joint_model_diagnostics"] = {
    "corr_demeaned_logchl_depth": float(c),
    "vif_chl": float(1 / (1 - c ** 2)), "vif_depth": float(1 / (1 - c ** 2)),
}

# ---------------------------------------------------------------- 7. is chl a proxy for depth?
res["chl_vs_depth"] = {
    "pooled_spearman": dict(zip(["rho", "p", "n"], [float(x) for x in st.spearmanr(lchl, mid)])),
    "within_group_mean_spearman": float(np.mean(
        [st.spearmanr(lchl[(m["group"] == g).to_numpy()], mid[(m["group"] == g).to_numpy()])[0]
         for g in groups])),
    "per_group_spearman": {g: {"rho": float(st.spearmanr(
        lchl[(m["group"] == g).to_numpy()], mid[(m["group"] == g).to_numpy()])[0]),
        "p": float(st.spearmanr(lchl[(m["group"] == g).to_numpy()],
                                mid[(m["group"] == g).to_numpy()])[1])} for g in groups},
    "within_group_r2_demeaned": float(np.corrcoef(demean(lchl), demean(mid))[0, 1] ** 2),
}
# chl should be ~0 at depth if it were a pure surface signal: how much of the
# within-group chl variance is explained by (demeaned) depth vs by (demeaned) total
r_cd = np.corrcoef(demean(lchl), demean(mid))[0, 1]
r_ct = np.corrcoef(demean(lchl), demean(ltot))[0, 1]
r_dt = np.corrcoef(demean(mid), demean(ltot))[0, 1]
res["within_group_correlations"] = {
    "r_logchl_logtotal": float(r_ct), "r_logchl_depth": float(r_cd),
    "r_depth_logtotal": float(r_dt),
    "partial_r_logchl_logtotal_given_depth": float(
        (r_ct - r_cd * r_dt) / math.sqrt((1 - r_cd ** 2) * (1 - r_dt ** 2))),
    "partial_r_logchl_logtotal_given_POC": float(
        (r_ct - np.corrcoef(demean(lchl), demean(lpoc))[0, 1] * np.corrcoef(demean(ltot), demean(lpoc))[0, 1])
        / math.sqrt((1 - np.corrcoef(demean(lchl), demean(lpoc))[0, 1] ** 2)
                    * (1 - np.corrcoef(demean(ltot), demean(lpoc))[0, 1] ** 2))),
}

# ---------------------------------------------------------------- 8. two-stage / group-level
sl = np.array([per_group_slope[g]["slope"] for g in groups])
tt = float(sl.mean() / (sl.std(ddof=1) / math.sqrt(len(sl))))
res["two_stage_group_as_unit"] = {
    "mean_slope": float(sl.mean()), "sd": float(sl.std(ddof=1)), "t": tt,
    "df": len(sl) - 1, "p": float(2 * st.t.sf(abs(tt), len(sl) - 1)),
    "n_negative": int((sl < 0).sum()),
    "sign_test_p": float(st.binomtest(int((sl < 0).sum()), len(sl), 0.5,
                                      alternative="two-sided").pvalue),
}

# ---------------------------------------------------------------- 9. leave-one-group-out
loo = []
for g in groups:
    s = (m["group"] != g).to_numpy()
    fl = fe(ltot[s], lchl[s], gi[s])
    loo.append({"dropped": g, "slope": float(fl["beta"][1]),
                "se_cluster": float(fl["se"][1]), "p_cluster": float(fl["p"][1]),
                "df": int(fl["df"])})
res["leave_one_group_out"] = loo

# ---------------------------------------------------------------- 10. cluster bootstrap of slope
rng = np.random.default_rng(20260919)
B = 20000
boot = np.empty(B)
gidx_arr = np.array([np.where(m["group"].to_numpy() == g)[0] for g in groups], dtype=object)
for b in range(B):
    picks = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([gidx_arr[p] for p in picks])
    gg = np.concatenate([np.full(len(gidx_arr[p]), k) for k, p in enumerate(picks)])
    y = ltot[idx]
    x = lchl[idx]
    yd = y.copy()
    xd = x.copy()
    for k in np.unique(gg):
        s = gg == k
        yd[s] -= yd[s].mean()
        xd[s] -= xd[s].mean()
    denom = xd @ xd
    boot[b] = (xd @ yd) / denom if denom > 0 else np.nan
boot = boot[np.isfinite(boot)]
res["cluster_bootstrap_slope"] = {
    "n_boot": int(len(boot)), "mean": float(boot.mean()), "sd": float(boot.std(ddof=1)),
    "ci95_percentile": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "ci95_basic": [float(2 * f["beta"][1] - np.percentile(boot, 97.5)),
                   float(2 * f["beta"][1] - np.percentile(boot, 2.5))],
    "frac_boot_negative": float((boot < 0).mean()),
    "frac_boot_lt_0_ge_claim": float((boot <= 0).mean()),
}

# ---------------------------------------------------------------- 11. permutation tests of the FE slope
# 11a: unrestricted within-group sign-flip permutation of the response
K = 20000
null_a = np.empty(K)
for k in range(K):
    lp = np.empty(n)
    for g in groups:
        s = np.where(m["group"].to_numpy() == g)[0]
        lp[s] = ltot[rng.permutation(s)]
    yd = demean(lp)
    xd = demean(lchl)
    null_a[k] = (xd @ yd) / (xd @ xd)
obs = f["beta"][1]
res["permutation_fe_slope"] = {
    "observed": float(obs), "n_perm": K,
    "p_two_sided": float((np.abs(null_a) >= abs(obs)).mean()),
    "p_one_sided_negative": float((null_a <= obs).mean()),
    "null_ci95": [float(np.percentile(null_a, 2.5)), float(np.percentile(null_a, 97.5))],
}

# 11b: permutation of DEPTH-CONSTRAINED positions (same depth order kept; only
# chl reassigned) -- tests whether the chl-total link is more than order-preserving
# 11c: rank-based permutation within group of chl only (keeps chl marginal, breaks link)
# 11d: test the counter-hypothesis directly: is the chl effect explained by the
#      depth rank? Use within-group chl residuals after removing a monotone depth fit.
res["counter_hypothesis_tests"] = {}

# Spearman of chl with depth within each group, and whether deepest interval is
# the highest chl
deepest = []
for g in groups:
    s = np.where(m["group"].to_numpy() == g)[0]
    d = s[np.argmax(mid[s])]
    deepest.append(int(d))
deepest = np.array(sorted(deepest))
# restrict to non-deepest intervals only (all >= 4 per group except group p)
shallow = ~np.isin(np.arange(n), deepest)
fsd = fe(ltot[shallow], lchl[shallow], gi[shallow])
res["counter_hypothesis_tests"]["exclude_deepest_only"] = {
    "n": int(shallow.sum()), "slope": float(fsd["beta"][1]),
    "se_cluster": float(fsd["se"][1]), "p_cluster": float(fsd["p"][1]),
    "df": int(fsd["df"]),
}
# exclude the two deepest intervals per group (n = 30 - 10 = 20)
keep_d2 = np.ones(n, bool)
for g in groups:
    s = np.where(m["group"].to_numpy() == g)[0]
    keep_d2[s[np.argsort(mid[s])[-2:]]] = False
fd2 = fe(ltot[keep_d2], lchl[keep_d2], gi[keep_d2])
res["counter_hypothesis_tests"]["exclude_two_deepest_per_group"] = {
    "n": int(keep_d2.sum()), "slope": float(fd2["beta"][1]),
    "se_cluster": float(fd2["se"][1]), "p_cluster": float(fd2["p"][1]),
    "df": int(fd2["df"]),
}
# exclude only the three global deepest intervals
ord_mid = np.argsort(mid)[::-1]
keep_d3 = np.ones(n, bool)
keep_d3[ord_mid[:3]] = False
fd3 = fe(ltot[keep_d3], lchl[keep_d3], gi[keep_d3])
res["counter_hypothesis_tests"]["exclude_3_global_deepest"] = {
    "excluded": [m["sample_id"].iloc[i] for i in ord_mid[:3]],
    "slope": float(fd3["beta"][1]), "se_cluster": float(fd3["se"][1]),
    "p_cluster": float(fd3["p"][1]), "df": int(fd3["df"]),
}
# jackknife over the 6 "bottom two" intervals: drop both deepest of group_02,
# group_03 and group_05 (the high-chl bottom samples)
bot = np.ones(n, bool)
for g in ["group_02", "group_03", "group_05"]:
    s = np.where(m["group"].to_numpy() == g)[0]
    bot[s[np.argsort(mid[s])[-2:]]] = False
fb = fe(ltot[bot], lchl[bot], gi[bot])
res["counter_hypothesis_tests"]["exclude_2deepest_of_high_chl_groups"] = {
    "n": int(bot.sum()), "slope": float(fb["beta"][1]),
    "se_cluster": float(fb["se"][1]), "p_cluster": float(fb["p"][1]), "df": int(fb["df"]),
}

# ---------------------------------------------------------------- 12. does chl add anything beyond depth?
# F-test style: within-group R2 of ltot on depth vs on depth+chl
X1 = np.c_[np.ones(n), demean(mid)]
f1_ = ols_fit(demean(ltot), X1)
X2 = np.c_[np.ones(n), demean(mid), demean(lchl)]
f2_ = ols_fit(demean(ltot), X2)
ss1 = float(f1_["resid"] @ f1_["resid"])
ss2 = float(f2_["resid"] @ f2_["resid"])
F = ((ss1 - ss2) / 1) / (ss2 / (n - 3))
res["nested_model_chl_beyond_depth"] = {
    "r2_depth_only": float(1 - ss1 / (demean(ltot) @ demean(ltot))),
    "r2_depth_plus_chl": float(1 - ss2 / (demean(ltot) @ demean(ltot))),
    "F": float(F), "p_classical": float(st.f.sf(F, 1, n - 3)),
    "note": "classical F ignores clustering; cluster-robust p for chl is in joint_model_chl_depth_only",
}
# reverse: does depth add beyond chl?
X3 = np.c_[np.ones(n), demean(lchl)]
f3_ = ols_fit(demean(ltot), X3)
ss3 = float(f3_["resid"] @ f3_["resid"])
F2 = ((ss3 - ss2) / 1) / (ss2 / (n - 3))
res["nested_model_depth_beyond_chl"] = {
    "r2_chl_only": float(1 - ss3 / (demean(ltot) @ demean(ltot))),
    "F": float(F2), "p_classical": float(st.f.sf(F2, 1, n - 3)),
}

# ---------------------------------------------------------------- 13. influence: Cook's D style on FE model
Xfe2 = np.c_[np.ones(n), demean(lchl)]
ffit = ols_fit(demean(ltot), Xfe2)
resid = ffit["resid"]
h = np.einsum("ij,jk,ik->i", Xfe2, ffit["XtXi"], Xfe2)
mse = float(resid @ resid) / (n - 2)
cook = (resid ** 2 / (2 * mse)) * (h / (1 - h) ** 2)
order = np.argsort(cook)[::-1]
res["influence_cooksD"] = {
    "top10": [{"sample": m["sample_id"].iloc[i], "group": m["group"].iloc[i],
               "cooksD": float(cook[i]), "depth_mid": float(mid[i]),
               "chl": float(chl[i]), "total": float(tot[i]),
               "resid": float(resid[i]), "leverage": float(h[i])}
              for i in order[:10]],
    "sum_cooksD": float(cook.sum()),
}
# leave-one-interval-out slope range
loo_i = []
for i in range(n):
    s = np.arange(n) != i
    yd = demean(ltot[s], gi[s])
    xd = demean(lchl[s], gi[s])
    b = (xd @ yd) / (xd @ xd)
    loo_i.append(float(b))
res["leave_one_interval_out"] = {
    "slope_min": float(np.min(loo_i)), "slope_max": float(np.max(loo_i)),
    "slope_mean": float(np.mean(loo_i)),
    "most_influential_sample": m["sample_id"].iloc[int(np.argmax(np.abs(np.array(loo_i) - obs)))],
    "max_abs_delta": float(np.max(np.abs(np.array(loo_i) - obs))),
}

# ---------------------------------------------------------------- 14. weight-robustness: unweighted vs group-equal vs interval-equal
# (the FE estimator already equal-weights intervals; check group-equal weights)
w = np.zeros(n)
for g in groups:
    s = (m["group"] == g).to_numpy()
    w[s] = 1.0 / s.sum()
yd = demean(ltot)
xd = demean(lchl)
bw = (w * xd @ yd) / (w * xd @ xd)
res["group_equal_weight_slope"] = float(bw)

# ---------------------------------------------------------------- 15. alternative transforms
res["alt_specifications"] = {}
for nm, x_, y_, lab in [
    ("log10chl_vs_log10total", lchl, ltot, "baseline"),
    ("log10chl_vs_total", lchl, tot, "level outcome"),
    ("chl_vs_log10total", chl, ltot, "level predictor"),
    ("log10chl_vs_log10poc", lchl, np.log10(np.where(poc > 0, poc, np.nan)), "POC outcome"),
]:
    fa = fe(y_, x_)
    res["alt_specifications"][nm] = {"slope": float(fa["beta"][1]),
                                     "se_cluster": float(fa["se"][1]),
                                     "p_cluster": float(fa["p"][1]), "label": lab}

# ---------------------------------------------------------------- 16. quantisation / detection-limit check
res["data_quality"] = {
    "chl_min": float(chl.min()), "chl_unique": int(len(np.unique(chl))),
    "total_min": float(tot.min()), "total_max": float(tot.max()),
    "zero_or_neg_total": int((tot <= 0).sum()),
    "zero_or_neg_chl": int((chl <= 0).sum()),
    "chl_values": sorted(np.unique(chl).tolist()),
}
# robust rank correlation of the FE (pooled-demeaned) relationship
res["pooled_pearson_demeaned"] = float(np.corrcoef(demean(lchl), demean(ltot))[0, 1])
res["pooled_spearman_raw"] = dict(zip(["rho", "p"], [float(x) for x in st.spearmanr(lchl, ltot)]))

# ---------------------------------------------------------------- 17. bootstrap over groups for the rho mean
boot_rho = np.empty(B)
for b in range(B):
    picks = rng.integers(0, len(groups), len(groups))
    vals = [rhos[p] for p in picks]
    boot_rho[b] = np.mean(vals)
res["cluster_bootstrap_mean_within_rho"] = {
    "mean": float(boot_rho.mean()), "sd": float(boot_rho.std(ddof=1)),
    "ci95_percentile": [float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5))],
    "frac_negative": float((boot_rho < 0).mean()),
}

# ---------------------------------------------------------------- 18. sign test on group slopes
res["group_slope_sign_test"] = {
    "n_negative": int((sl < 0).sum()), "n_groups": len(sl),
    "p_two_sided_exact": float(st.binomtest(int((sl < 0).sum()), len(sl), 0.5,
                                            alternative="two-sided").pvalue),
}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))