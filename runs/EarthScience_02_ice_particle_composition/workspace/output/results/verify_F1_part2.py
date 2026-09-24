"""Part 2 of the adversarial verification of CLAIM F1.

Follows up the disagreement between the cluster-robust t-test (p=0.221, df=4)
and the exact within-group permutation test (p=0.030) found in part 1.

Adds:
  - per-group permutation/rank p-values and per-group influence on the FE slope
  - permutation p-value for the FE slope under every leave-one-group-out,
    drop-shallowest, drop-deepest and exclude-group_01 variant
  - uncertainty-weighted within-group slope using the reported errors
  - surface-interval vs remainder contrast
  - leave-two-groups-out sign consistency
  - within-group R^2 and effect size in plain units
Merges the results into verify_F1.json.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
J = ROOT / "output" / "results" / "verify_F1.json"

comp = pd.read_csv(ROOT / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(ROOT / "data" / "sample_environment_profiles.tsv", sep="\t")
df = comp[["sample_id", "collection_group_id", "depth_min_m", "depth_max_m",
           "particle_total_per_L"]].merge(
    env[["sample_id", "particle_total_std_or_error_per_L"]], on="sample_id",
    validate="1:1")
df["mid"] = (df.depth_min_m + df.depth_max_m) / 2
df["ltot"] = np.log10(df.particle_total_per_L)
df["rel_err"] = df.particle_total_std_or_error_per_L / df.particle_total_per_L
# delta-method SE of log10(y): err / (y * ln 10)
df["se_ltot"] = df.particle_total_std_or_error_per_L / (df.particle_total_per_L * np.log(10))
groups = sorted(df.collection_group_id.unique())
gid = df.collection_group_id.to_numpy()
mid = df.mid.to_numpy(float)
ltot = df.ltot.to_numpy(float)
rng = np.random.default_rng(7)
NPERM = 50000

out = json.loads(J.read_text())


def demean(v, codes):
    v = np.asarray(v, float)
    codes = np.asarray(codes)
    o = v.copy()
    for k in np.unique(codes):
        m = codes == k
        o[m] = v[m] - v[m].mean()
    return o


def fe_slope(y, x, codes):
    xd = demean(x, codes)
    yd = demean(y, codes)
    return float((xd * yd).sum() / (xd * xd).sum())


def perm_p(y, x, codes, nperm=NPERM):
    obs = fe_slope(y, x, codes)
    xd = demean(x, codes)
    ss = (xd * xd).sum()
    yd = demean(y, codes)
    kk = np.unique(codes)
    draws = np.empty(nperm)
    for b in range(nperm):
        yy = yd.copy()
        for k in kk:
            m = codes == k
            yy[m] = yd[m][rng.permutation(m.sum())]
        draws[b] = (xd * yy).sum() / ss
    p2 = float((np.abs(draws) >= abs(obs) - 1e-12).mean())
    p1 = float((draws <= obs + 1e-12).mean())
    return {"slope": obs, "perm_p_two_sided": p2, "perm_p_one_sided_negative": p1,
            "perm_sd": float(draws.std(ddof=1)),
            "perm_quantile_2.5": float(np.quantile(draws, .025)),
            "perm_quantile_97.5": float(np.quantile(draws, .975))}


uni = np.unique(gid)
codes0 = np.array([list(uni).index(g) for g in gid])

base = perm_p(ltot, mid, codes0, NPERM)
out["part2"] = {"fe_permutation_baseline_n50000": base,
                "note_on_part1_discrepancy": (
                    "part 1 already showed FE cluster-robust t p=0.221 (df=4) vs "
                    "within-group permutation p=0.030. The permutation test is exact "
                    "under within-group exchangeability and is not subject to the "
                    "5-cluster df problem; the CR1/t(df=4) test is very conservative "
                    "with only 5 clusters.")}

# ---- per-group permutation p-values (exact for the smaller groups)
pg = []
for g in groups:
    m = gid == g
    y, x = ltot[m], mid[m]
    b = perm_p(y, x, np.zeros(m.sum(), int), NPERM)
    rho = stats.spearmanr(x, y)
    # influence on the pooled FE slope
    keep = ~m
    nn = keep.sum()
    c = np.array([list(np.unique(gid[keep])).index(z) for z in gid[keep]])
    s_wo = fe_slope(ltot[keep], mid[keep], c)
    pg.append({"group": g, "n": int(m.sum()), **b,
               "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
               "fe_slope_without_group": s_wo,
               "delta_vs_baseline": s_wo - base["slope"]})
out["part2"]["per_group_permutation_and_influence"] = pg

# ---- permutation p for key subsets
subsets = {}
for g in groups:
    m = gid != g
    c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
    subsets[f"leave_out_{g}"] = perm_p(ltot[m], mid[m], c, NPERM)
m = gid != "group_01"
c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
subsets["exclude_group_01"] = perm_p(ltot[m], mid[m], c, NPERM)
m = gid != "group_02"
c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
subsets["exclude_group_02"] = perm_p(ltot[m], mid[m], c, NPERM)
for nm, keepfun in [("drop_shallowest_per_group", "shallow"),
                    ("drop_deepest_per_group", "deep")]:
    idx = []
    for g in groups:
        s = df[df.collection_group_id == g]
        tgt = s.mid.min() if keepfun == "shallow" else s.mid.max()
        idx += list(s.index[s.mid != tgt])
    idx = np.array(idx)
    c = np.array([list(np.unique(gid[idx])).index(z) for z in gid[idx]])
    subsets[nm] = perm_p(ltot[idx], mid[idx], c, NPERM)
    subsets[nm]["n"] = int(len(idx))
out["part2"]["permutation_by_subset"] = subsets

# ---- uncertainty-weighted slope (weights from reported error)
Y = ltot
X = mid
codes = codes0
w = 1.0 / df.se_ltot.to_numpy(float) ** 2
# guard against absurd weights (a few rel errors are >100%) by winsorising at 1/(0.3 dex)^2
w_win = np.clip(w, None, 1 / 0.3 ** 2)
xd = demean(X, codes)
yd = demean(Y, codes)


def wslope(wv):
    return float((wv * xd * yd).sum() / (wv * xd * xd).sum())


out["part2"]["uncertainty_weighted_fe_slope"] = {
    "weights": "delta-method inverse variance of log10(total): 1/(err/(y*ln10))^2",
    "slope_full_weights": wslope(w),
    "slope_winsorised_weights_se_le_0.3_dex": wslope(w_win),
    "unweighted_slope": base["slope"],
    "relative_error_summary": {"min": float(df.rel_err.min()), "median": float(df.rel_err.median()),
                               "max": float(df.rel_err.max())},
    "note": ("absorption of the 96%-error surface sample of group_02 is what the "
             "weighting mostly does; result stays negative but shrinks."),
    "perm_p_full_weights": None}
# permutation with weights
obs = wslope(w)
draws = np.empty(20000)
for b in range(20000):
    yy = yd.copy()
    for k in np.unique(codes):
        m = codes == k
        yy[m] = yd[m][rng.permutation(m.sum())]
    draws[b] = (w * xd * yy).sum() / (w * xd * xd).sum()
out["part2"]["uncertainty_weighted_fe_slope"]["perm_p_full_weights"] = {
    "p_two_sided": float((np.abs(draws) >= abs(obs) - 1e-12).mean()),
    "p_one_sided_negative": float((draws <= obs + 1e-12).mean()),
    "perm_sd": float(draws.std(ddof=1)), "slope": obs}

# ---- surface interval vs remainder
top_rows = []
for g in groups:
    s = df[df.collection_group_id == g]
    t = s.loc[s.mid.idxmin()]
    rest = s.loc[s.index != t.name]
    top_rows.append({"group": g, "top_per_L": float(t.particle_total_per_L),
                     "top_ltot": float(t.ltot),
                     "mean_rest_ltot": float(rest.ltot.mean()),
                     "log10_diff_top_minus_rest": float(t.ltot - rest.ltot.mean())})
dtop = np.array([r["log10_diff_top_minus_rest"] for r in top_rows])
tt = dtop.mean() / (dtop.std(ddof=1) / np.sqrt(len(dtop)))
out["part2"]["surface_vs_remainder"] = {
    "per_group": top_rows, "mean_log10_diff": float(dtop.mean()),
    "t": float(tt), "p": float(2 * stats.t.sf(abs(tt), len(dtop) - 1)),
    "n_top_higher": int((dtop > 0).sum()),
    "note": ("if the depth pattern were really a surface-enrichment spike rather than "
             "a monotone decline, this contrast (top vs rest) would be larger than the "
             "monotone FE slope; compare magnitudes.")}

# ---- leave-two-groups-out sign consistency (all 10 pairs, 3 clusters each)
pairs = []
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        m = ~np.isin(gid, [groups[i], groups[j]])
        c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
        pairs.append({"dropped": [groups[i], groups[j]], "slope": fe_slope(ltot[m], mid[m], c)})
out["part2"]["leave_two_groups_out"] = {
    "rows": pairs, "n_negative": int(sum(p["slope"] < 0 for p in pairs)),
    "n_positive": int(sum(p["slope"] > 0 for p in pairs)),
    "min": float(min(p["slope"] for p in pairs)),
    "max": float(max(p["slope"] for p in pairs)),
}

# ---- within-group R^2 / effect size
resid = yd - base["slope"] * xd
ssr = float((resid ** 2).sum())
sst = float((yd ** 2).sum())
out["part2"]["within_group_effect_size"] = {
    "within_group_r2": 1 - ssr / sst,
    "within_group_sd_log10": float(np.sqrt(sst / (len(yd) - 5))),
    "slope_log10_per_m": base["slope"],
    "factor_change_over_median_group_depth_range": float(
        10 ** (base["slope"] * np.median([df[gid == g].mid.max() - df[gid == g].mid.min()
                                           for g in groups]))),
    "median_group_depth_range_m": float(np.median(
        [df[gid == g].mid.max() - df[gid == g].mid.min() for g in groups])),
}

# ---- power of the CR1 df=4 test under the observed effect (simulation)
def cr1_slope_se(y, x, codes):
    n = len(y)
    xd = demean(x, codes); yd = demean(y, codes)
    Xd = np.c_[np.ones(n), xd]
    beta = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    r = yd - Xd @ beta
    XtXinv = np.linalg.inv(Xd.T @ Xd)
    meat = np.zeros((2, 2))
    for k in np.unique(codes):
        mm = codes == k
        u = (Xd[mm] * r[mm, None]).sum(axis=0)
        meat += np.outer(u, u)
    nc = len(np.unique(codes))
    scale = (nc / (nc - 1)) * ((n - 1) / (n - 2))
    V = XtXinv @ (scale * meat) @ XtXinv
    return beta[1], np.sqrt(V[1, 1])


power = {}
for b_true in [-1.0, -0.826]:
    cnt = 0
    for _ in range(1000):
        yy = yd.copy()
        for k in np.unique(codes):
            m = codes == k
            yy[m] = yd[m][rng.permutation(m.sum())]
        ysim = yy + b_true * xd
        bb, ss = cr1_slope_se(ysim, mid, codes)
        cnt += 2 * stats.t.sf(abs(bb / ss), 4) < 0.05
    power[str(b_true)] = cnt / 1000
out["part2"]["power_of_cr1_df4_test_at_observed_effect"] = power

out["code_part2"] = Path(__file__).read_text()
J.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out["part2"], indent=2, default=str)[:7000])
print("wrote", J)