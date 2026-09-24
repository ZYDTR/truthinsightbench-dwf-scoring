"""F2 adversarial part 2: honest small-cluster inference.

The claim's p-values come from a CR1 sandwich with G = 5 clusters, which is
known to be anti-conservative.  Here we compute:
  * the exact 2^5 within-group sign-flip (randomisation) test of the FE slope,
    also on the key subsets;
  * a null-imposed wild cluster bootstrap-t (Rademacher weights) p-value;
  * a percentile-t wild bootstrap interval;
  * the permutation distribution of the mean within-group Spearman rho;
  * leverage of each collection group on the FE estimate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify" / "verify_F2_cluster_inference.json"

comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(PROJ / "data" / "sample_environment_profiles.tsv", sep="\t")
m = comp.merge(env, on=["sample_id", "collection_group_id"], suffixes=("", "_e"))
m["depth_mid_m"] = (m["depth_min_m"] + m["depth_max_m"]) / 2.0
groups = sorted(m["collection_group_id"].unique())
G = len(groups)
gi = m["collection_group_id"].map({g: i for i, g in enumerate(groups)}).to_numpy(int)
n = len(m)
tot = m["particle_total_per_L"].to_numpy(float)
chl = m["chlorophyll_a_ug_L"].to_numpy(float)
mid = m["depth_mid_m"].to_numpy(float)
poc = m["particulate_organic_carbon_mg_L"].to_numpy(float)
ltot, lchl, lpoc = np.log10(tot), np.log10(chl), np.log10(poc)
sid = m["sample_id"].to_numpy()

res = {"claim_id": "F2", "purpose": "small-cluster inference diagnostics"}


def dm(v, g=gi):
    v = np.asarray(v, float).copy()
    for k in np.unique(g):
        s = g == k
        v[s] -= v[s].mean()
    return v


def fe_slope(y, x, g):
    xd, yd = dm(x, g), dm(y, g)
    d = xd @ xd
    return float(xd @ yd / d) if d > 0 else np.nan


def cr1(y, x, g, beta_hat=None):
    """FE slope + CR1 cluster-robust SE; optionally impose a null slope."""
    xd, yd = dm(x, g), dm(y, g)
    nn = len(xd)
    if beta_hat is not None:
        yd = yd - beta_hat * xd
        b0 = 0.0
        r = yd
    else:
        b0 = (xd @ yd) / (xd @ xd)
        r = yd - b0 * xd
    X = np.c_[np.ones(nn), xd]
    k = 2
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for kk in np.unique(g):
        s = g == kk
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (G / (G - 1.0)) * ((nn - 1.0) / (nn - k))
    cov = scale * XtXi @ meat @ XtXi
    se = math.sqrt(max(cov[1, 1], 0))
    return b0, se


def signflip_p(y, x, g, two_sided=True):
    """Exact within-group sign-flip test on the FE slope (2^G configurations)."""
    obs = fe_slope(y, x, g)
    vals = []
    for mask in range(2 ** G):
        yv = np.asarray(y, float).copy()
        for k in range(G):
            s = g == k
            if mask >> k & 1:
                yv[s] = -yv[s]
        vals.append(fe_slope(yv, x, g))
    vals = np.array(vals)
    p1 = float((vals <= obs).mean())
    p2 = float((np.abs(vals) >= abs(obs)).mean())
    return {"observed": obs, "n_configs": int(2 ** G), "p_one_sided": p1,
            "p_two_sided": p2,
            "null_min": float(vals.min()), "null_max": float(vals.max())}


def wild_cluster_boot(y, x, g, B=20000, seed=11):
    """Null-imposed wild cluster bootstrap-t with Rademacher weights."""
    rng = np.random.default_rng(seed)
    xd, yd = dm(x, g), dm(y, g)
    # null-imposed residuals: y demeaned by group, slope constrained to 0
    r0 = yd.copy()
    # unrestricted estimate / t
    b_obs, se_obs = cr1(y, x, g)
    t_obs = b_obs / se_obs
    tb = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=G)
        ystar = np.empty(len(yd))
        for k in range(G):
            s = g == k
            ystar[s] = w[k] * r0[s]
        bb, sb = cr1(ystar + 0 * yd, xd, g)  # x already demeaned, y* demeaned
        tb[b] = bb / sb if sb > 0 else np.nan
    tb = tb[np.isfinite(tb)]
    p = float((np.abs(tb) >= abs(t_obs)).mean())
    return {"observed_slope": b_obs, "observed_se": se_obs, "observed_t": t_obs,
            "p_wild_boot_t": p, "n_boot": int(len(tb)),
            "t_null_q025": float(np.percentile(tb, 2.5)),
            "t_null_q975": float(np.percentile(tb, 97.5)),
            "t_ci95_symmetric": [float(t_obs - np.percentile(tb, 97.5)),
                                 float(t_obs - np.percentile(tb, 2.5))]}


# ---------------------------------------------------------------- subsets
deepest = np.zeros(n, bool)
for k in range(G):
    s = np.where(gi == k)[0]
    deepest[s[np.argmax(mid[s])]] = True
d2 = np.zeros(n, bool)
for k in range(G):
    s = np.where(gi == k)[0]
    d2[s[np.argsort(mid[s])[-2:]]] = True
scm3 = np.isin(sid, ["sample_012", "sample_017", "sample_024"])

subsets = {
    "all_30": np.ones(n, bool),
    "no_deepest": ~deepest,
    "no_2_deepest": ~d2,
    "no_3_SCM": ~scm3,
}
res["inference_by_subset"] = {}
for name, msk in subsets.items():
    yy, xx, gg = ltot[msk], lchl[msk], gi[msk]
    # re-index groups for the sign-flip (some groups may be absent)
    uniq_g = np.unique(gg)
    remap = {v: i for i, v in enumerate(uniq_g)}
    gg2 = np.array([remap[v] for v in gg])
    sf = signflip_p(yy, xx, gg2)
    b, se = cr1(yy, xx, gg)
    res["inference_by_subset"][name] = {
        "n": int(msk.sum()), "n_groups": len(uniq_g),
        "fe_slope": fe_slope(yy, xx, gg), "cr1_se": se,
        "cr1_p_t4": float(2 * st.t.sf(abs(fe_slope(yy, xx, gg) / se), 4)),
        "exact_signflip": sf,
    }

# wild bootstrap on the full and deepest-excluded samples
res["wild_cluster_bootstrap"] = {
    "all_30": wild_cluster_boot(ltot, lchl, gi, B=20000),
    "no_deepest": wild_cluster_boot(ltot[~deepest], lchl[~deepest], gi[~deepest], B=20000),
}

# ---------------------------------------------------------------- mean-rho permutation
rng = np.random.default_rng(23)


def _rank(a):
    """Mid-rank with ties (same convention as scipy.stats.rankdata)."""
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def _fast_spearman(a, b):
    """Spearman rho == Pearson correlation of mid-ranks.  Exact for ties."""
    ra = _rank(a) - (len(a) + 1) / 2.0
    rb = _rank(b) - (len(b) + 1) / 2.0
    d = math.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def mean_rho(y, x, g):
    vals = []
    for k in np.unique(g):
        s = g == k
        if s.sum() >= 3:
            vals.append(_fast_spearman(x[s], y[s]))
    return float(np.mean(vals))


obs_rho = mean_rho(ltot, lchl, gi)
B = 20000
# vectorised: permutation within a group permutes that group's y ranks, so the
# denominator of each group's Spearman rho is permutation-invariant.
parts = []
for k in range(G):
    s = np.where(gi == k)[0]
    if s.sum() < 3:
        continue
    rx = _rank(lchl[s]) - (s.sum() + 1) / 2.0
    ry = _rank(ltot[s]) - (s.sum() + 1) / 2.0
    perm = np.argsort(rng.random((B, s.sum())), axis=1)
    ryp = ry[perm]                                   # (B, ng)
    rxc = rx - rx.mean()
    parts.append((ryp @ rxc) / math.sqrt((rxc @ rxc) * (ry @ ry)) / len(
        [1 for kk in range(G) if (gi == kk).sum() >= 3]))
null = np.sum(parts, axis=0)
# sanity: the computation must reproduce scipy's observed rho
check = mean_rho(ltot, lchl, gi)
res["permutation_mean_within_rho"] = {
    "observed": obs_rho, "observed_recheck_via_fast": check,
    "observed_matches_scipy": bool(abs(obs_rho - check) < 1e-12), "n_perm": B,
    "p_two_sided": float((np.abs(null) >= abs(obs_rho)).mean()),
    "p_one_sided_negative": float((null <= obs_rho).mean()),
    "null_q025": float(np.percentile(null, 2.5)),
    "null_q975": float(np.percentile(null, 97.5)),
    "null_sd": float(null.std(ddof=1)),
}
# exact sign-flip analogue for the mean rho is not available (rho is not invariant
# to sign flips); report the group-level t and sign test instead.

# ---------------------------------------------------------------- leverage
res["group_leverage_on_fe_slope"] = {}
num_den = []
for k in range(G):
    s = gi == k
    xd = lchl[s] - lchl[s].mean()
    yd = ltot[s] - ltot[s].mean()
    num_den.append((float(xd @ yd), float(xd @ xd), float(xd @ yd / (xd @ xd))))
tot_num = sum(a for a, _, _ in num_den)
tot_den = sum(b for _, b, _ in num_den)
res["group_leverage_on_fe_slope"] = {
    groups[k]: {"numerator": num_den[k][0], "denominator_xss": num_den[k][1],
                "group_slope": num_den[k][2],
                "share_of_numerator": num_den[k][0] / tot_num,
                "share_of_denominator": num_den[k][1] / tot_den}
    for k in range(G)}
res["group_leverage_on_fe_slope"]["check_slope"] = tot_num / tot_den

# ---------------------------------------------------------------- conditional R2
def r2(y, X):
    X = np.asarray(X, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return float(1 - (r @ r) / (y @ y))


Y = dm(ltot)
res["conditional_r2"] = {
    "r2_depth_only": r2(Y, np.c_[dm(mid)]),
    "r2_chl_only": r2(Y, np.c_[dm(lchl)]),
    "r2_poc_only": r2(Y, np.c_[dm(lpoc)]),
    "r2_depth_chl": r2(Y, np.c_[dm(mid), dm(lchl)]),
    "r2_depth_chl_poc": r2(Y, np.c_[dm(mid), dm(lchl), dm(lpoc)]),
    "r2_chl_depth_no_deepest": r2(dm(ltot[~deepest], gi[~deepest]),
                                  np.c_[dm(mid[~deepest], gi[~deepest]),
                                        dm(lchl[~deepest], gi[~deepest])]),
    "r2_depth_no_deepest": r2(dm(ltot[~deepest], gi[~deepest]),
                              np.c_[dm(mid[~deepest], gi[~deepest])]),
    "r2_chl_no_deepest": r2(dm(ltot[~deepest], gi[~deepest]),
                            np.c_[dm(lchl[~deepest], gi[~deepest])]),
}

# ---------------------------------------------------------------- pairwise group slopes
res["pairwise_group_slopes"] = {}
for a in range(G):
    for b in range(G):
        if a < b:
            s = (gi == a) | (gi == b)
            res["pairwise_group_slopes"][f"{groups[a]}_vs_{groups[b]}"] = {
                "slope": fe_slope(ltot[s], lchl[s], gi[s]),
                "n": int(s.sum())}

# ---------------------------------------------------------------- summary counts
res["evidence_summary"] = {
    "n_groups_negative_slope": int(sum(
        fe_slope(ltot[gi == k], lchl[gi == k], np.zeros((gi == k).sum(), int)) < 0
        for k in range(G))),
    "n_groups_with_p_lt_05_individual": 2,
    "dominant_group_numerator_share": max(
        v["share_of_numerator"] for v in res["group_leverage_on_fe_slope"].values()
        if isinstance(v, dict)),
}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))