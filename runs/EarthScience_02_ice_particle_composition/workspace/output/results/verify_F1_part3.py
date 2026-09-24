"""Part 3 of the adversarial verification of CLAIM F1.

  - raw-scale (arithmetic) within-group slope + permutation p
  - Theil-Sen median pairwise within-group slope (robust) + bootstrap CI
  - relative depth as mid/max(mid) per group
  - uncertainty-weighting fragility: error vs concentration structure
  - leave-one-sample-out permutation p-values
Merges results into verify_F1.json.
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
    env[["sample_id", "particle_total_std_or_error_per_L"]], on="sample_id", validate="1:1")
df["mid"] = (df.depth_min_m + df.depth_max_m) / 2
df["ltot"] = np.log10(df.particle_total_per_L)
df["relmid"] = df.mid / df.groupby("collection_group_id").mid.transform("max")
groups = sorted(df.collection_group_id.unique())
gid = df.collection_group_id.to_numpy()
mid = df.mid.to_numpy(float); ltot = df.ltot.to_numpy(float)
tot = df.particle_total_per_L.to_numpy(float)
rng = np.random.default_rng(99)
uni = np.unique(gid)
codes = np.array([list(uni).index(g) for g in gid])
out = json.loads(J.read_text())


def demean(v, c):
    v = np.asarray(v, float); c = np.asarray(c); o = v.copy()
    for k in np.unique(c):
        m = c == k; o[m] = v[m] - v[m].mean()
    return o


def fe_slope(y, x, c):
    xd, yd = demean(x, c), demean(y, c)
    return float((xd * yd).sum() / (xd * xd).sum())


def perm(y, x, c, nperm=20000):
    obs = fe_slope(y, x, c); xd = demean(x, c); yd = demean(y, c)
    ss = (xd * xd).sum(); draws = np.empty(nperm)
    for b in range(nperm):
        yy = yd.copy()
        for k in np.unique(c):
            m = c == k
            yy[m] = yd[m][rng.permutation(m.sum())]
        draws[b] = (xd * yy).sum() / ss
    return {"slope": obs, "perm_p_two_sided": float((np.abs(draws) >= abs(obs) - 1e-12).mean()),
            "perm_p_one_sided_negative": float((draws <= obs + 1e-12).mean()),
            "perm_sd": float(draws.std(ddof=1)),
            "ci95_perm": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


p3 = {}
# ----- raw scale
p3["raw_scale_arithmetic_fe"] = perm(tot, mid, codes)
p3["raw_scale_arithmetic_fe"]["units"] = "particles/L per metre"
p3["raw_scale_arithmetic_fe"]["ols_percent_change_of_mean_per_m"] = float(
    100 * p3["raw_scale_arithmetic_fe"]["slope"] / tot.mean())

# ----- Theil-Sen within-group median pairwise slope
pairs_all = []
for g in groups:
    m = gid == g
    x, y = mid[m], ltot[m]
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pairs_all.append((y[j] - y[i]) / (x[j] - x[i]))
pairs_all = np.array(pairs_all)
p3["theil_sen_within_group_median_pairwise_slope"] = {
    "median": float(np.median(pairs_all)), "mean": float(pairs_all.mean()),
    "n_pairs": int(len(pairs_all)),
    "bootstrap_ci95": [float(x) for x in np.percentile(
        [np.median(rng.choice(pairs_all, len(pairs_all), replace=True)) for _ in range(5000)],
        [2.5, 97.5])],
    "note": "all within-group unordered pairs of intervals; robust to a single deep point"}

# ----- relative depth as mid/max(mid)
p3["relative_depth_mid_over_max_fe"] = perm(ltot, df.relmid.to_numpy(float), codes)
p3["relative_depth_mid_over_max_fe"]["units"] = "log10 per 1.0 of mid/max(mid)"

# ----- error structure (explains the weighting fragility)
le = np.log10(df.particle_total_std_or_error_per_L.to_numpy(float))
p3["uncertainty_structure"] = {
    "corr_log10err_vs_log10total": dict(zip(["r", "p"], [float(x) for x in
        stats.pearsonr(le, ltot)[:2]])),
    "spearman_rho_log10err_vs_log10total": dict(zip(["rho", "p"], [float(x) for x in
        stats.spearmanr(le, ltot)[:2]])),
    "median_relative_error": float((df.particle_total_std_or_error_per_L /
                                    df.particle_total_per_L).median()),
    "n_samples_relative_error_gt_0.5": int(((df.particle_total_std_or_error_per_L /
                                            df.particle_total_per_L) > 0.5).sum()),
    "n_samples_err_gt_total": int((df.particle_total_std_or_error_per_L >
                                   df.particle_total_per_L).sum()),
    "note": ("err scales with concentration, so inverse-variance weighting on the log "
             "scale is nearly equivalent to a precision-weighted fit that downweights "
             "the deepest, low-count samples - exactly the samples that carry the "
             "negative slope. Because the reported errors are themselves very noisy "
             "(some exceed the value), such weighting is not a neutral arbitration."),
}
# weighted FE slope restricted to samples whose relative error <= median (best-determined half)
for q in [0.5, 0.7]:
    keep = (df.particle_total_std_or_error_per_L / df.particle_total_per_L) <= \
        (df.particle_total_std_or_error_per_L / df.particle_total_per_L).quantile(q)
    c = np.array([list(np.unique(gid[keep])).index(z) for z in gid[keep]])
    p3[f"restrict_to_best_determined_{int(q*100)}pct"] = {
        "n": int(keep.sum()), **perm(ltot[keep], mid[keep], c)}

# ----- LOSO permutation p (10000 perms each, 30 fits)
loso = []
for i in range(len(df)):
    m = np.ones(len(df), bool); m[i] = False
    c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
    r = perm(ltot[m], mid[m], c, 10000)
    loso.append({"dropped": df.sample_id[i], "group": gid[i], **r})
p3["loso_permutation"] = {
    "rows": loso,
    "p_two_sided_min": float(min(x["perm_p_two_sided"] for x in loso)),
    "p_two_sided_max": float(max(x["perm_p_two_sided"] for x in loso)),
    "n_p_two_sided_lt_0.05": int(sum(x["perm_p_two_sided"] < 0.05 for x in loso)),
    "slope_min": float(min(x["slope"] for x in loso)),
    "slope_max": float(max(x["slope"] for x in loso)),
    "slope_always_negative": bool(all(x["slope"] < 0 for x in loso)),
}

# ----- group_02 dominance: FE slope and permutation p without group_02, and with group_02 only
for nm, m in [("without_group_02", gid != "group_02"), ("only_group_02", gid == "group_02")]:
    c = np.array([list(np.unique(gid[m])).index(z) for z in gid[m]])
    p3[f"fe_{nm}"] = {"n": int(m.sum()), **perm(ltot[m], mid[m], c)}
    p3[f"fe_{nm}"]["pooled_spearman"] = dict(zip(
        ["rho", "p"], [float(x) for x in stats.spearmanr(mid[m], ltot[m])[:2]]))

out["part3"] = p3
out["code_part3"] = Path(__file__).read_text()
J.write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(p3, indent=2, default=str)[:6500])
print("wrote", J)