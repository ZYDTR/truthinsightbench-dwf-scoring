"""Adversarial follow-up on F2: is the within-group chl-a / total-particle
association an artifact of the deepest interval of each group, and is chl-a
merely a depth proxy?

Independent implementation (numpy/scipy only).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify" / "verify_F2_adversarial.json"

comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(PROJ / "data" / "sample_environment_profiles.tsv", sep="\t")
m = comp.merge(env, on=["sample_id", "collection_group_id"], suffixes=("", "_e"))
m["depth_mid_m"] = (m["depth_min_m"] + m["depth_max_m"]) / 2.0
groups = sorted(m["collection_group_id"].unique())
gi = m["collection_group_id"].map({g: i for i, g in enumerate(groups)}).to_numpy(int)
n = len(m)
tot = m["particle_total_per_L"].to_numpy(float)
chl = m["chlorophyll_a_ug_L"].to_numpy(float)
mid = m["depth_mid_m"].to_numpy(float)
poc = m["particulate_organic_carbon_mg_L"].to_numpy(float)
ltot, lchl, lpoc = np.log10(tot), np.log10(chl), np.log10(poc)
sid = m["sample_id"].to_numpy()

res = {"claim_id": "F2", "purpose": "adversarial follow-up"}


def dm(v, g=gi):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        s = g == k
        v[s] -= v[s].mean()
    return v


def sl(y, x, g=gi):
    """Within-group slope (no SE)."""
    xd, yd = dm(x, g), dm(y, g)
    d = xd @ xd
    return float(xd @ yd / d) if d > 0 else np.nan


def wls_slope(y, x, g=gi):
    """Same as sl: FE slope is the x-variance-weighted mean of group slopes."""
    num = den = 0.0
    for k in np.unique(g):
        s = g == k
        xd = x[s] - x[s].mean()
        yd = y[s] - y[s].mean()
        num += xd @ yd
        den += xd @ xd
    return num / den


def cluster_se(y, x, g=gi):
    """CR1 cluster-robust SE of the FE slope, cluster = collection group."""
    xd, yd = dm(x, g), dm(y, g)
    X = np.c_[np.ones(len(xd)), xd]
    b = np.linalg.pinv(X.T @ X) @ X.T @ yd
    r = yd - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    G = len(np.unique(g))
    nn, k = X.shape
    meat = np.zeros((k, k))
    for kk in np.unique(g):
        s = g == kk
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (G / (G - 1.0)) * ((nn - 1.0) / (nn - k))
    cov = scale * XtXi @ meat @ XtXi
    se = math.sqrt(max(cov[1, 1], 0))
    t = b[1] / se
    return float(b[1]), se, float(t), 2 * st.t.sf(abs(t), G - 1), G - 1


# ===================================================== A. exact group sign-flip
# Under the sharp null "no within-group association", the sign of each group's
# y-residual pattern is exchangeable.  Sign-flipping per group gives 2^5 = 32
# equally likely FE slopes.  This is an exact finite permutation test that
# respects the cluster structure.
obs = wls_slope(ltot, lchl)
flips = []
for mask in range(2 ** len(groups)):
    yv = ltot.copy()
    for k in range(len(groups)):
        s = gi == k
        if mask >> k & 1:
            yv[s] = -yv[s]
    flips.append(wls_slope(yv, lchl))
flips = np.array(flips)
res["exact_group_signflip"] = {
    "observed_fe_slope": obs,
    "n_configurations": len(flips),
    "null_min": float(flips.min()), "null_max": float(flips.max()),
    "p_one_sided_le_obs": float((flips <= obs).mean()),
    "p_two_sided_symmetric": float((np.abs(flips) >= abs(obs)).mean()),
    "note": "group sign-flip null forces the group mean of log-total to be "
            "irrelevant, so it is a sharp test of the within-group slope only",
}
# Sign-flip of the group SLOPES (equivalent but shown for clarity)
gslopes = np.array([sl(ltot[gi == k], lchl[gi == k], np.zeros((gi == k).sum(), int))
                    for k in range(len(groups))])
gsub = lchl[gi == k]
gw = np.array([(lchl[gi == k] - lchl[gi == k].mean()).var() * (gi == k).sum()
               for k in range(len(groups))])
res["exact_group_signflip"]["group_slopes"] = gslopes.tolist()
res["exact_group_signflip"]["group_slope_weights"] = gw.tolist()
res["exact_group_signflip"]["weighted_mean_check"] = float((gslopes * gw).sum() / gw.sum())

# ===================================================== B. deepest-indicator model
deepest = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    deepest[s[np.argmax(mid[s])]] = True


def ols_with_se(y, X, g=gi):
    X = np.asarray(X, float)
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    r = y - X @ b
    G = len(np.unique(g))
    nn, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for kk in np.unique(g):
        s = g == kk
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (G / (G - 1.0)) * ((nn - 1.0) / (nn - k))
    cov = scale * XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = b / se
    return b, se, 2 * st.t.sf(np.abs(t), G - 1)


X = np.c_[np.ones(n), dm(lchl), deepest.astype(float)]
b, se, p = ols_with_se(dm(ltot), X)
res["deepest_indicator_model"] = {
    "terms": ["intercept", "log10_chl_a", "is_deepest_interval"],
    "beta": b.tolist(), "se_cluster": se.tolist(), "p_cluster": p.tolist(),
    "note": "if the chl association were purely a surface-vs-bottom contrast, "
            "the log10_chl_a coefficient would collapse once the deepest "
            "interval gets its own intercept",
}
# Same but with a full depth control (linear) + quadratic depth
X2 = np.c_[np.ones(n), dm(lchl), dm(mid), dm(mid) ** 2]
b2, se2, p2 = ols_with_se(dm(ltot), X2)
res["chl_with_depth_and_depth2"] = {
    "terms": ["intercept", "log10_chl_a", "depth_mid_m", "depth_mid_m^2"],
    "beta": b2.tolist(), "se_cluster": se2.tolist(), "p_cluster": p2.tolist(),
}
# depth tertile dummies + chl
tert = np.zeros(n, int)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    order = s[np.argsort(mid[s])]
    tert[order[1:]] = 1  # everything except the single shallowest of each group
    tert[order[-1]] = 2 if len(order) >= 3 else 1
Xt = np.c_[np.ones(n), dm(lchl), (tert == 1).astype(float), (tert == 2).astype(float)]
bt, set_, pt = ols_with_se(dm(ltot), Xt)
res["chl_with_depth_tertile_dummies"] = {
    "terms": ["intercept", "log10_chl_a", "depth_mid_tertile", "depth_deep_tertile"],
    "beta": bt.tolist(), "se_cluster": set_.tolist(), "p_cluster": pt.tolist(),
}

# ===================================================== C. group-wise contribution
contrib = []
for k in range(len(groups)):
    s = gi == k
    xd, yd = dm(lchl[s], np.zeros(s.sum(), int)), dm(ltot[s], np.zeros(s.sum(), int))
    contrib.append({"group": groups[k], "n": int(s.sum()),
                    "x_ss": float(xd @ xd), "xy": float(xd @ yd),
                    "share_of_denominator": float((xd @ xd) / (dm(lchl) @ dm(lchl))),
                    "share_of_numerator": float((xd @ yd) / (dm(lchl) @ dm(ltot))),
                    "group_slope": float(xd @ yd / (xd @ xd)),
                    "rho_spearman": float(st.spearmanr(lchl[s], ltot[s])[0])})
res["per_group_contribution"] = contrib

# ===================================================== D. targeted exclusions
def report(mask, label, extra=None):
    yy, xx, gg = ltot[mask], lchl[mask], gi[mask]
    s = wls_slope(yy, xx, gg)
    b1, se1, t1, p1, df1 = cluster_se(yy, xx, gg)
    d = {"label": label, "n": int(mask.sum()), "slope": s,
         "se_cluster": se1, "p_cluster": p1, "df": df1,
         "mean_within_rho": float(np.mean(
             [st.spearmanr(xx[gg == k], yy[gg == k])[0]
              for k in np.unique(gg) if (gg == k).sum() >= 3]))}
    if extra:
        d.update(extra)
    return d


res["targeted_exclusions"] = {}
res["targeted_exclusions"]["none"] = report(np.ones(n, bool), "all 30")
res["targeted_exclusions"]["drop_deepest_each"] = report(~deepest, "drop deepest of each group")
d2 = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    d2[s[np.argsort(mid[s])[-2:]]] = True
res["targeted_exclusions"]["drop_2_deepest_each"] = report(~d2, "drop 2 deepest of each group")
d3 = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    d3[s[np.argsort(mid[s])[-3:]]] = True
res["targeted_exclusions"]["drop_3_deepest_each"] = report(~d3, "drop 3 deepest of each group")
# drop the three big subsurface-chl-maximum samples only
scm = np.isin(sid, ["sample_012", "sample_017", "sample_024"])
res["targeted_exclusions"]["drop_3_strongest_SCM_samples"] = report(
    ~scm, "drop samples 012/017/024 (chl 1.74/4.19/6.64)")
res["targeted_exclusions"]["drop_both_012_024"] = report(
    ~np.isin(sid, ["sample_012", "sample_024"]), "drop samples 012/024")
res["targeted_exclusions"]["drop_024_only"] = report(~np.isin(sid, ["sample_024"]), "drop 024")
res["targeted_exclusions"]["drop_017_only"] = report(~np.isin(sid, ["sample_017"]), "drop 017")
res["targeted_exclusions"]["drop_012_only"] = report(~np.isin(sid, ["sample_012"]), "drop 012")
# surface only (first 4 intervals of each group)
surf = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    surf[s[np.argsort(mid[s])[:4]]] = True
res["targeted_exclusions"]["shallowest_4_each"] = report(surf, "shallowest 4 of each group")
surf3 = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    surf3[s[np.argsort(mid[s])[:3]]] = True
res["targeted_exclusions"]["shallowest_3_each"] = report(surf3, "shallowest 3 of each group")
# highest-chl interval per group removed (overlaps deepest in 4/5 groups)
hi = np.zeros(n, bool)
for k in range(len(groups)):
    s = np.where(gi == k)[0]
    hi[s[np.argmax(chl[s])]] = True
res["targeted_exclusions"]["drop_max_chl_each"] = report(~hi, "drop max-chl of each group")
res["targeted_exclusions"]["overlap_deepest_vs_maxchl"] = {
    "n_both": int((deepest & hi).sum()), "groups": groups,
    "same_in_n_groups": int(sum(
        deepest[np.where(gi == k)[0][np.argmax(mid[np.where(gi == k)[0]])]]
        == hi[np.where(gi == k)[0][np.argmax(chl[np.where(gi == k)[0]])]]
        for k in range(len(groups))))}

# ===================================================== E. is chl a depth proxy?
res["depth_proxy_tests"] = {
    "within_group_r2_logchl_on_depth": float(np.corrcoef(dm(lchl), dm(mid))[0, 1] ** 2),
    "within_group_r2_logtot_on_depth": float(np.corrcoef(dm(ltot), dm(mid))[0, 1] ** 2),
    "within_group_r2_logtot_on_logchl": float(np.corrcoef(dm(ltot), dm(lchl))[0, 1] ** 2),
    "within_group_r2_logtot_on_both": float(np.corrcoef(
        dm(ltot), np.c_[dm(lchl), dm(mid)] @ np.linalg.lstsq(
            np.c_[dm(lchl), dm(mid)], dm(ltot), rcond=None)[0])[0, 1] ** 2),
    "mean_within_group_spearman_chl_depth": float(np.mean(
        [st.spearmanr(lchl[gi == k], mid[gi == k])[0] for k in range(len(groups))])),
    "pooled_spearman_chl_depth": float(st.spearmanr(chl, mid)[0]),
    "mean_within_group_spearman_total_depth": float(np.mean(
        [st.spearmanr(ltot[gi == k], mid[gi == k])[0] for k in range(len(groups))])),
}
# chl vs depth after removing the deepest interval of each group
res["depth_proxy_tests"]["mean_within_group_spearman_chl_depth_no_deepest"] = float(np.mean(
    [st.spearmanr(lchl[gi == k][~deepest[gi == k]], mid[gi == k][~deepest[gi == k]])[0]
     for k in range(len(groups))]))
# slope of log-total on depth with and without the deepest intervals
res["depth_proxy_tests"]["fe_slope_logtot_on_depth"] = float(sl(ltot, mid))
res["depth_proxy_tests"]["fe_slope_logtot_on_depth_no_deepest"] = float(
    sl(ltot[~deepest], mid[~deepest], gi[~deepest]))
# chl as a proxy: within-group R2 of the deepest dummy on log-chl
X = np.c_[np.ones(n), deepest.astype(float)]
bb = np.linalg.lstsq(X, dm(lchl), rcond=None)[0]
res["depth_proxy_tests"]["within_group_r2_logchl_on_deepest_dummy"] = float(
    1 - ((dm(lchl) - X @ bb) @ (dm(lchl) - X @ bb)) / (dm(lchl) @ dm(lchl)))

# ===================================================== F. bootstrap diagnostics
rng = np.random.default_rng(7)
B = 40000
boot = np.full(B, np.nan)
nmiss = 0
for b in range(B):
    picks = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([np.where(gi == k)[0] for k in picks])
    gg = np.concatenate([np.full((gi == k).sum(), j) for j, k in enumerate(picks)])
    v = wls_slope(ltot[idx], lchl[idx], gg)
    if np.isfinite(v):
        boot[b] = v
    else:
        nmiss += 1
boot = boot[np.isfinite(boot)]
res["cluster_bootstrap_diagnostics"] = {
    "n_valid": int(len(boot)), "n_nan": int(nmiss),
    "sd": float(boot.std(ddof=1)),
    "q025": float(np.percentile(boot, 2.5)), "q975": float(np.percentile(boot, 97.5)),
    "frac_negative": float((boot < 0).mean()),
    "median": float(np.median(boot)),
    "n_distinct_groups_drawn_mean": 5 * (1 - (4 / 5) ** 5),
    "note": "with 5 clusters the bootstrap is lumpy; treat the interval as "
            "indicative only",
}
# cluster jackknife SE (delete-one-group) for the slope
jack = np.array([wls_slope(ltot[gi != k], lchl[gi != k], gi[gi != k])
                 for k in range(len(groups))])
jk_mean = jack.mean()
jk_se = math.sqrt((len(jack) - 1) / len(jack) * ((jack - jk_mean) ** 2).sum())
res["cluster_jackknife_slope"] = {
    "values": jack.tolist(), "mean": float(jk_mean), "se": float(jk_se),
    "t_vs_0": float(jk_mean / jk_se), "df": len(jack) - 1,
    "p": float(2 * st.t.sf(abs(jk_mean / jk_se), len(jack) - 1)),
}

# ===================================================== G. joint model robustness
def joint(mask, terms, label):
    yy = dm(ltot[mask], gi[mask])
    Xs = [np.ones(int(mask.sum()))] + [dm(v[mask], gi[mask]) for v in terms]
    X = np.c_[tuple(Xs)]
    b, se, p = ols_with_se(yy, X, gi[mask])
    return {"label": label, "n": int(mask.sum()),
            "terms": ["intercept"] + [t for t in ["log10_chl_a", "depth_mid_m", "log10_POC"][:len(terms)]],
            "beta": b.tolist(), "se": se.tolist(), "p": p.tolist()}


res["joint_models"] = {
    "full": joint(np.ones(n, bool), [lchl, mid, lpoc], "full n=30"),
    "no_deepest": joint(~deepest, [lchl, mid, lpoc], "no deepest"),
    "drop_3_scm": joint(~scm, [lchl, mid, lpoc], "drop 3 SCM samples"),
    "chl_depth": joint(np.ones(n, bool), [lchl, mid], "chl+depth"),
    "no_deepest_chl_depth": joint(~deepest, [lchl, mid], "no deepest chl+depth"),
}
# POC alone
res["poc_alone_fe"] = {"slope": float(sl(ltot, lpoc)),
                       "se_p_cluster": cluster_se(ltot, lpoc)[1:]}

# ===================================================== H. monotone / rank view
# rank-based FE: within-group ranks of chl vs within-group ranks of total
def within_rank(v, g=gi):
    out = np.empty(len(v))
    for k in np.unique(g):
        s = g == k
        out[s] = st.rankdata(v[s])
    return out


rk_chl = within_rank(lchl)
rk_tot = within_rank(ltot)
res["rank_views"] = {
    "fe_slope_rank_rank": float(sl(rk_tot, rk_chl)),
    "se_p_cluster": cluster_se(rk_tot, rk_chl)[1:],
    "spearman_rank_vs_log_agreement": float(st.spearmanr(rk_chl, lchl)[0]),
}
# Kendall tau within groups
taus = [st.kendalltau(lchl[gi == k], ltot[gi == k]) for k in range(len(groups))]
res["rank_views"]["within_group_kendall_tau"] = {groups[k]: {"tau": float(taus[k][0]),
                                                             "p": float(taus[k][1])}
                                                 for k in range(len(groups))}
res["rank_views"]["mean_tau"] = float(np.mean([t[0] for t in taus]))

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))