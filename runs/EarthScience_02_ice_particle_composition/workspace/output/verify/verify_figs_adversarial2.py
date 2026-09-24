"""Part 2 of the figure-task adversarial check: honest small-cluster inference
for the two within-group claims, and the F1/F2 overlap question.

  * null-imposed wild cluster bootstrap-t (Rademacher) for the FE slope
    (F1: log10 total on depth;  F2: log10 total on log10 chl-a), G = 5 groups
  * the same for the chl coefficient of the joint chl + depth model
  * within-group correlation of log10 chl-a with depth (are F1 and F2 the
    same pattern seen through two variables?)
  * permutation p-values for the exclude-deepest and exclude-3-highest-chl
    variants (the CR1 p at G=5 is anti-conservative, so the permutation p is
    the more defensible number)
  * cluster (group) bootstrap percentile CI for the F2 slope
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify" / "verify_figs_adversarial2.json"

comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(PROJ / "data" / "sample_environment_profiles.tsv", sep="\t")
m = comp[["sample_id", "collection_group_id", "depth_min_m", "depth_max_m",
          "particle_total_per_L"]].merge(
    env[["sample_id", "chlorophyll_a_ug_L", "particulate_organic_carbon_mg_L"]],
    on="sample_id", validate="one_to_one")
m["mid"] = (m.depth_min_m + m.depth_max_m) / 2
m = m.sort_values(["collection_group_id", "mid"]).reset_index(drop=True)
GROUPS = sorted(m.collection_group_id.unique())
g = m.collection_group_id.to_numpy()
G = len(GROUPS)
n = len(m)
mid = m["mid"].to_numpy(float)
ltot = np.log10(m.particle_total_per_L.to_numpy(float))
lchl = np.log10(m.chlorophyll_a_ug_L.to_numpy(float))
lpoc = np.log10(m.particulate_organic_carbon_mg_L.to_numpy(float))
sid = m.sample_id.to_numpy()
code = np.array([GROUPS.index(x) for x in g])

res = {"purpose": "honest small-cluster inference + F1/F2 overlap",
       "n_clusters": G,
       "min_attainable_two_sided_signflip_p": 2 / 2 ** G}


def dm(v, c=code):
    v = np.asarray(v, float).copy()
    for k in np.unique(c):
        s = c == k
        v[s] -= v[s].mean()
    return v


def qls(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    return b, r, XtXi


def cr1_se(y, X, cl=None):
    b, r, XtXi = qls(y, X)
    X = np.asarray(X, float)
    k = X.shape[1]
    nn = X.shape[0]
    cl = code if cl is None else np.asarray(cl)
    uniq = np.unique(cl)
    Gc = len(uniq)
    meat = np.zeros((k, k))
    for kk in uniq:
        s = cl == kk
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    scale = (Gc / (Gc - 1.0)) * ((nn - 1.0) / (nn - k))
    cov = scale * XtXi @ meat @ XtXi
    return b, np.sqrt(np.maximum(np.diag(cov), 0))


def wcb_t(y, x, B=40000, seed=3):
    """Null-imposed wild cluster bootstrap-t on the FE slope.

    x is already group-demeaned.  Under H0 the null residual is the group-
    demeaned y; each bootstrap replicate multiplies a whole group's residuals
    by a Rademacher weight, adds them back and recomputes the CR1 t.
    """
    rng = np.random.default_rng(seed)
    x = dm(x)
    y = dm(y)
    b_obs, se_obs = cr1_se(y, np.c_[np.ones(n), x])
    t_obs = b_obs[1] / se_obs[1]
    r0 = y.copy()
    tb = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=G)
        ys = r0 * w[code]
        bb, sb = cr1_se(dm(ys), np.c_[np.ones(n), x])
        tb[b] = bb[1] / sb[1] if sb[1] > 0 else np.nan
    tb = tb[np.isfinite(tb)]
    return {"slope": float(b_obs[1]), "se_CR1": float(se_obs[1]),
            "t": float(t_obs), "p_two_sided": float((np.abs(tb) >= abs(t_obs)).mean()),
            "p_one_sided_neg": float((tb <= t_obs).mean()),
            "boot_q": [float(np.quantile(tb, q)) for q in (0.025, 0.5, 0.975)],
            "B": B}


def perm_fe(y, x, codes=None, B=20000, seed=5, drop=None):
    """Permute x within groups; FE slope statistic.  Vectorised.

    `drop` = global group code to drop, or None.  `x`/`y` must already be
    subset to the retained rows when `drop` is not used, and the matching
    `codes` vector should then be supplied.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if drop is not None:
        sel = code != drop
        gg_raw, xx, yy = code[sel], x[sel], y[sel]
    else:
        gg_raw, xx, yy = (code if codes is None else np.asarray(codes)), x, y
    # recode so that every retained group has index 0..G'-1 (monotone)
    uniq = np.unique(gg_raw)
    gg = np.array([int(np.where(uniq == v)[0][0]) for v in gg_raw])
    Gp = len(uniq)
    sizes = np.array([int((gg == k).sum()) for k in range(Gp)])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    yd = dm(yy, gg)
    xd_obs = dm(xx, gg)
    obs = float(xd_obs @ yd / (xd_obs @ xd_obs))
    keys = rng.random((B, len(xx)))
    order = np.lexsort((keys, np.broadcast_to(gg, (B, len(gg)))), axis=1)
    X = xx[order]
    segmean = np.add.reduceat(X, starts, axis=1) / sizes
    Xd = X - np.repeat(segmean, sizes, axis=1)
    null = (Xd @ yd) / np.einsum("ij,ij->i", Xd, Xd)
    return {"observed": obs, "p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
            "p_one_sided_neg": float((null <= obs).mean()), "n": int(len(xx))}


def cluster_boot_slope(y, x, B=20000, seed=17):
    """Percentile cluster bootstrap (resample whole groups)."""
    rng = np.random.default_rng(seed)
    sl = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        yb = np.concatenate([y[code == k] for k in pick])
        xb = np.concatenate([x[code == k] for k in pick])
        cb = np.concatenate([np.full((code == k).sum(), j) for j, k in enumerate(pick)])
        xd, yd = dm(xb, cb), dm(yb, cb)
        sl[b] = xd @ yd / (xd @ xd) if xd @ xd > 0 else np.nan
    sl = sl[np.isfinite(sl)]
    return {"q025": float(np.quantile(sl, 0.025)), "median": float(np.median(sl)),
            "q975": float(np.quantile(sl, 0.975)), "B": int(len(sl))}


# --- 1. F1 and F2 wild cluster bootstrap and permutations
res["F1_depth"] = {
    "wcb_t": wcb_t(ltot, mid),
    "perm_drop_group_02": perm_fe(ltot, mid, drop=1),
    "cluster_boot_slope": cluster_boot_slope(ltot, mid),
}
res["F2_chl"] = {
    "wcb_t": wcb_t(ltot, lchl),
    "perm_drop_group_02": perm_fe(ltot, lchl, drop=1),
    "cluster_boot_slope": cluster_boot_slope(ltot, lchl),
}

# --- 2. joint chl + depth model: honest inference on the chl coefficient
Xj = np.c_[np.ones(n), dm(lchl), dm(mid)]
b, se = cr1_se(dm(ltot), Xj)
t_obs = b[1] / se[1]
rng = np.random.default_rng(21)
r0 = dm(ltot) - Xj @ b
tb = np.empty(40000)
for i in range(40000):
    w = rng.choice([-1.0, 1.0], size=G)
    bb, sb = cr1_se(dm(r0 * w[code]), Xj)
    tb[i] = bb[1] / sb[1] if sb[1] > 0 else np.nan
tb = tb[np.isfinite(tb)]
res["F2_joint_chl_plus_depth_chl_coef"] = {
    "beta": float(b[1]), "se_CR1": float(se[1]), "t": float(t_obs),
    "p_CR1_df4": float(2 * st.t.sf(abs(t_obs), G - 1)),
    "p_wcb_t": float((np.abs(tb) >= abs(t_obs)).mean()),
    "depth_coef": {"beta": float(b[2]), "se_CR1": float(se[2]),
                   "p_CR1_df4": float(2 * st.t.sf(abs(b[2] / se[2]), G - 1))},
}

# --- 3. overlap: is F2 just F1 seen through chlorophyll?
ov = {}
rhos, rhops = [], []
for k, gg in enumerate(GROUPS):
    s = code == k
    r, p = st.spearmanr(lchl[s], mid[s])
    rhos.append(r)
    rhops.append(p)
    ov[gg] = {"spearman_rho_lchl_vs_depth": float(r), "p": float(p), "n": int(s.sum())}
ov["mean_rho"] = float(np.mean(rhos))
ov["n_positive"] = int(sum(r > 0 for r in rhos))
ov["note"] = ("chlorophyll increases with depth in every group to some degree; the "
              "two claims share the deep-interval signal")
# per-group: how often is the deepest interval also the max-chl interval or min-total
ov["deepest_interval_flags"] = {}
for k, gg in enumerate(GROUPS):
    s = np.where(code == k)[0]
    di = s[np.argmax(mid[s])]
    ov["deepest_interval_flags"][gg] = {
        "sample_id": str(sid[di]),
        "is_max_chl": bool(lchl[di] == lchl[s].max()),
        "is_min_total": bool(ltot[di] == ltot[s].min()),
        "is_max_depth": True,
    }
# rank correlation within group between chl-rank and depth-rank: already above
# partial: FE slope of ltot on lchl after removing the deepest interval AND
# controlling for depth
keep = np.ones(n, bool)
for k in range(G):
    s = np.where(code == k)[0]
    keep[s[np.argmax(mid[s])]] = False
ck = code[keep]
res["F2_excl_deepest_perm"] = perm_fe(
    ltot[keep], lchl[keep], codes=ck, B=20000, seed=31)
# chl coefficient in joint model after dropping deepest
Xk = np.c_[np.ones(keep.sum()), dm(lchl[keep], ck), dm(mid[keep], ck)]
bk, sek = cr1_se(dm(ltot[keep], ck), Xk, cl=ck)
res["F2_excl_deepest_joint_chl_coef"] = {
    "beta": float(bk[1]), "se_CR1": float(sek[1]),
    "p_CR1_df4": float(2 * st.t.sf(abs(bk[1] / sek[1]), G - 1)),
    "depth_beta": float(bk[2]), "depth_p_CR1": float(2 * st.t.sf(abs(bk[2] / sek[2]), G - 1)),
}
# exclude 3 highest chl: permutation p
hi = np.argsort(lchl)[-3:]
keeph = np.ones(n, bool)
keeph[hi] = False
chk0 = code[keeph]
res["F2_excl_3hi_chl_perm"] = {
    "excluded": [str(sid[i]) for i in hi],
    **perm_fe(ltot[keeph], lchl[keeph], codes=chk0, B=20000, seed=41)}
# same exclusion, joint model with depth
chk = code[keeph]
Xh = np.c_[np.ones(keeph.sum()), dm(lchl[keeph], chk), dm(mid[keeph], chk)]
bh, seh = cr1_se(dm(ltot[keeph], chk), Xh, cl=chk)
res["F2_excl_3hi_chl_joint_chl_coef"] = {
    "beta": float(bh[1]), "se_CR1": float(seh[1]),
    "p_CR1_df4": float(2 * st.t.sf(abs(bh[1] / seh[1]), G - 1)),
    "depth_beta": float(bh[2]), "depth_p_CR1": float(2 * st.t.sf(abs(bh[2] / seh[2]), G - 1))}
res["overlap_chl_depth"] = ov

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "overlap_chl_depth"}, indent=1))
print(json.dumps(res["overlap_chl_depth"], indent=1))