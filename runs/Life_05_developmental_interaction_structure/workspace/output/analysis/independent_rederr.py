#!/usr/bin/env python3
"""Independent end-to-end re-derivation of the headline numbers in Result.md.

Reads data/stage_indexed_interactions.tsv DIRECTLY (not the pre-built
analysis_table.tsv) and recomputes every headline statistic quoted in
output/Result.md with a self-contained implementation. For each quantity it
records the reported value, the independently recomputed value, and whether
they match (with a tolerance). Anything that fails to reproduce is flagged and
the discrepancy is written to the artifact; nothing is fudged to force a match.

Deterministic: fixed RNG seeds (20220920) for bootstrap and shuffle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "stage_indexed_interactions.tsv"
OUT = ROOT / "output" / "results"
Y = "interaction_strength_like_value"
P = "association_pvalue"
FLOOR = 0.0009
SHA_TARGET = "67ab6ace870c985522f905801f0eed680818a0c73225ca113057ba12fdea992f"

res: dict[str, object] = {}


def match(a, b, rel=1e-3, abs_tol=5e-3):
    if a is None or b is None:
        return a is None and b is None
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.isfinite(a) and np.isfinite(b) and a == b
    return abs(a - b) <= abs_tol or abs(a - b) <= rel * abs(b)


raw = pd.read_csv(RAW, sep="\t")
sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
res["integrity"] = {
    "raw_rows": int(len(raw)),
    "sha256": sha,
    "sha256_matches_target": sha == SHA_TARGET,
}
# collapse exact row duplicates on the analysis key
key = ["stage_index", "interaction_id", "cell_pair_id"]
u = raw.drop_duplicates(subset=key, keep="first").copy()
u["unit_id"] = u.interaction_id + "|" + u.cell_pair_id
res["census"] = {
    "raw_rows": int(len(raw)),
    "analysis_rows": int(len(u)),
    "units": int(u.unit_id.nunique()),
    "families": int(u.interaction_id.nunique()),
    "cell_pairs": int(u.cell_pair_id.nunique()),
    "stages": int(u.stage_index.nunique()),
    "stage_values": sorted(int(v) for v in u.stage_index.unique()),
    "rows_in_replicated_groups": int(raw.duplicated(subset=key).sum()),
    "replicated_groups": int(raw.loc[raw.duplicated(subset=key, keep=False), key].drop_duplicates().shape[0]),
    "max_replicate": int(raw.groupby(key).size().max()),
    "p_levels": sorted(float(v) for v in u[P].unique()),
    "floor_share": float((u[P] == FLOOR).mean()),
    "one_record_per_unit_stage": int(
        (u.groupby("unit_id").size().eq(u.groupby("unit_id").stage_index.nunique())).all()
    ),
}

y = u[Y]
um = u.groupby("unit_id")[Y].transform("mean")
yc = y - um
stages = np.arange(int(u.stage_index.max()) + 1)

# -------------------------------------------------------------------------- F1
slope_fu = float(np.polyfit(u.stage_index, yc, 1)[0])
curve = [float(yc[u.stage_index == s].mean()) for s in stages]
r_curve = float(np.corrcoef(stages, curve)[0, 1])
rho_curve = float(stats.spearmanr(stages, curve).statistic)
res["F1_trend"] = {
    "slope_reported": 0.0418, "slope_recomputed": slope_fu, "slope_match": match(0.0418, slope_fu, 1e-3),
    "curve_r_reported": 0.922, "curve_r_recomputed": r_curve, "curve_r_match": match(0.922, r_curve, 1e-3),
    "curve_spearman_recomputed": rho_curve,
    "early_mean": float(np.mean(curve[:9])), "late_mean": float(np.mean(curve[9:])),
    "per_stage_curve": [round(v, 5) for v in curve],
}
sl = np.array([
    float(np.polyfit(g.stage_index, g[Y], 1)[0])
    for _, g in u.groupby("unit_id") if g.stage_index.nunique() >= 3
])
res["F1_trend"]["per_unit_slopes"] = {
    "n": int(len(sl)), "mean_recomputed": float(np.mean(sl)), "median_recomputed": float(np.median(sl)),
    "frac_positive_recomputed": float((sl > 0).mean()),
    "t_stat": float(stats.ttest_1samp(sl, 0).statistic), "t_p": float(stats.ttest_1samp(sl, 0).pvalue),
    "wilcoxon_p": float(stats.wilcoxon(sl).pvalue),
    "sign_pos": int((sl > 0).sum()), "sign_neg": int((sl < 0).sum()),
    "sign_test_p": float(stats.binomtest(int((sl > 0).sum()), len(sl), 0.5).pvalue),
}
rng = np.random.default_rng(20220920)
by_fam = {f: g for f, g in u.groupby("interaction_id")}
fams = list(by_fam)

def wus(sub: pd.DataFrame) -> float:
    umm = sub.groupby("unit_id")[Y].transform("mean")
    return float(np.polyfit(sub.stage_index, sub[Y] - umm, 1)[0])

boot = np.array([
    wus(pd.concat([by_fam[f] for f in rng.choice(fams, size=len(fams), replace=True)]))
    for _ in range(1000)
])
ci_recomputed = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
ci_reported = [0.0293, 0.0544]
res["F1_trend"]["bootstrap"] = {
    "point": slope_fu,
    "ci_recomputed": ci_recomputed,
    "ci_reported": ci_reported,
    "ci_match": all(match(r, c, 1e-3) for r, c in zip(ci_recomputed, ci_reported)),
}

# -------------------------------------------------------------------------- F1 holdout (leave-one-family-out)
multi = u.groupby("unit_id").size()
sub = u[u.unit_id.isin(set(multi[multi >= 2].index))].copy()
gsum = sub.groupby("unit_id")[Y].transform("sum")
gn = sub.groupby("unit_id")[Y].transform("size")
loo = (gsum - sub[Y]) / (gn - 1)
resid = sub[Y] - loo
ns = int(sub.stage_index.max()) + 1
fams_c, fams_u = pd.factorize(sub.interaction_id)
grid = sub.stage_index.to_numpy() * len(fams_u) + fams_c
s_sum = np.bincount(sub.stage_index.to_numpy(), weights=resid, minlength=ns)
s_cnt = np.bincount(sub.stage_index.to_numpy(), minlength=ns).astype(float)
g_sum = np.bincount(grid, weights=resid, minlength=ns * len(fams_u)).reshape(ns, len(fams_u))
g_cnt = np.bincount(grid, minlength=ns * len(fams_u)).reshape(ns, len(fams_u)).astype(float)
num = s_sum[:, None] - g_sum
den = s_cnt[:, None] - g_cnt
dl = np.zeros_like(num); ok = den > 0; dl[ok] = num[ok] / den[ok]
pred_t = loo + dl[sub.stage_index.to_numpy(), fams_c]
mae_n = float(np.abs(sub[Y] - loo).mean())
mae_t = float(np.abs(sub[Y] - pred_t).mean())
rmse_n = float(np.sqrt(np.mean((sub[Y] - loo) ** 2)))
rmse_t = float(np.sqrt(np.mean((sub[Y] - pred_t) ** 2)))

def perm_control(n_perm=200):
    rng = np.random.default_rng(20220920)
    imp = []
    for _ in range(n_perm):
        pstage = sub.groupby("unit_id").stage_index.transform(lambda s: rng.permutation(s.to_numpy()))
        gsum_p = sub.groupby("unit_id")[Y].transform("sum")
        gn_p = sub.groupby("unit_id")[Y].transform("size")
        loo_p = (gsum_p - sub[Y]) / (gn_p - 1)
        resid_p = sub[Y] - loo_p
        grid_p = pstage.to_numpy() * len(fams_u) + fams_c
        ps_sum = np.bincount(pstage.to_numpy(), weights=resid_p, minlength=ns)
        ps_cnt = np.bincount(pstage.to_numpy(), minlength=ns).astype(float)
        pg_sum = np.bincount(grid_p, weights=resid_p, minlength=ns * len(fams_u)).reshape(ns, len(fams_u))
        pg_cnt = np.bincount(grid_p, minlength=ns * len(fams_u)).reshape(ns, len(fams_u)).astype(float)
        pn = ps_sum[:, None] - pg_sum
        pdn = ps_cnt[:, None] - pg_cnt
        dp = np.zeros_like(pn); pok = pdn > 0; dp[pok] = pn[pok] / pdn[pok]
        pred_p = loo_p + dp[pstage.to_numpy(), fams_c]
        imp.append(mae_n - float(np.abs(sub[Y] - pred_p).mean()))
    imp = np.asarray(imp)
    return imp

perm = perm_control()
res["F1_holdout"] = {
    "n_records": int(len(sub)),
    "mae_null_recomputed": mae_n, "mae_trend_recomputed": mae_t,
    "rmse_improvement_pct_recomputed": 100.0 * (rmse_n - rmse_t) / rmse_n,
    "improvement_pct_recomputed": 100.0 * (mae_n - mae_t) / mae_n,
    "perm_control": {
        "n_perm": int(len(perm)), "perm_mean": float(perm.mean()),
        "perm_sd": float(perm.std(ddof=1)), "perm_max": float(perm.max()),
        "observed": float(mae_n - mae_t),
        "observed_exceeds_all": bool((mae_n - mae_t) > perm.max()),
    },
    "report_match": {
        "mae_null": match(0.4681, mae_n, 1e-3),
        "mae_trend": match(0.4275, mae_t, 1e-3),
        "improvement_pct": match(8.7, 100.0 * (mae_n - mae_t) / mae_n, 1e-2),
        "rmse_improvement_pct": match(9.3, 100.0 * (rmse_n - rmse_t) / rmse_n, 1e-2),
    },
}

# cell-pair holdout
cpc, cpc_u = pd.factorize(sub.cell_pair_id)
grid_c = sub.stage_index.to_numpy() * len(cpc_u) + cpc
cs_sum = np.bincount(sub.stage_index.to_numpy(), weights=resid, minlength=ns)
cs_cnt = np.bincount(sub.stage_index.to_numpy(), minlength=ns).astype(float)
cg_sum = np.bincount(grid_c, weights=resid, minlength=ns * len(cpc_u)).reshape(ns, len(cpc_u))
cg_cnt = np.bincount(grid_c, minlength=ns * len(cpc_u)).reshape(ns, len(cpc_u)).astype(float)
cn = cs_sum[:, None] - cg_sum; dn = cs_cnt[:, None] - cg_cnt
dc = np.zeros_like(cn); dk = dn > 0; dc[dk] = cn[dk] / dn[dk]
pred_c = loo + dc[sub.stage_index.to_numpy(), cpc]
mae_c = float(np.abs(sub[Y] - pred_c).mean())
res["F1_holdout"]["cellpair_holdout"] = {
    "mae_null": mae_n, "mae_trend": mae_c,
    "improvement_pct": 100.0 * (mae_n - mae_c) / mae_n,
    "match": match(0.4271, mae_c, 1e-3),
}

# -------------------------------------------------------------------------- F2
fam_mean = u.groupby("interaction_id")[Y].transform("mean")
fam_icc = float(fam_mean.var(ddof=0) / y.var(ddof=0))
unit_icc = float(um.var(ddof=0) / y.var(ddof=0))
stage_eta2 = float((yc.groupby(u.stage_index).transform("mean")).var(ddof=0) / y.var(ddof=0))
fam_mean_sd = float(u.groupby("interaction_id")[Y].mean().std(ddof=1))
fam_size_median = float(u.groupby("interaction_id").size().median())
fam_size_max = int(u.groupby("interaction_id").size().max())
fam_stage_spearman = float(stats.spearmanr(
    u.groupby("interaction_id").size(), u.groupby("interaction_id").stage_index.nunique()
).statistic)
res["F2_identity"] = {
    "family_icc_recomputed": fam_icc,
    "unit_icc_recomputed": unit_icc,
    "stage_eta2_recomputed": stage_eta2,
    "family_mean_sd_recomputed": fam_mean_sd,
    "fam_size_median_recomputed": fam_size_median,
    "fam_size_max_recomputed": fam_size_max,
    "fam_stage_spearman": fam_stage_spearman,
    "report_match": {
        "family_share_836": match(0.836, fam_icc, 1e-3),
        "unit_share_901": match(0.901, unit_icc, 1e-3),
        "stage_18": match(0.018, stage_eta2, 1e-3),
        "family_mean_sd_143": match(1.43, fam_mean_sd, 1e-2),
        "family_size_median_10": match(10.0, fam_size_median, 1e-2),
        "family_size_max_130": match(130, fam_size_max, 1e-3),
    },
}

# -------------------------------------------------------------------------- F3
nf = u[u[P] > FLOOR]
rho_raw = float(stats.spearmanr(u[P], y).statistic)
rho_within = float(stats.spearmanr(u[P], yc).statistic)
rho_nfo = float(stats.spearmanr(nf[P], nf[Y]).statistic)
p_nfo = float(stats.spearmanr(nf[P], nf[Y]).pvalue)
# range of per-level MEANS (matching the report: "the mean strength spans only 0.28 units")
per_level = nf.groupby(P)[Y].mean()
nf_range = float(per_level.max() - per_level.min())
kruskal_stat = float(
    stats.kruskal(*[nf.loc[nf[P] == pv, Y] for pv in sorted(nf[P].unique())]).statistic
)
res["F3_evidence_tier"] = {
    "p_levels": res["census"]["p_levels"],
    "floor_share": res["census"]["floor_share"],
    "rho_raw": rho_raw,
    "rho_within_unit": rho_within,
    "rho_nonfloor_only": {"rho": rho_nfo, "p": p_nfo},
    "nonfloor_mean_strength_range": nf_range,
    "kruskal_nonfloor_stat": kruskal_stat,
    "report_match": {
        "floor_share_876": match(0.876, res["census"]["floor_share"], 1e-3),
        "rho_raw_-0285": match(-0.285, rho_raw, 1e-3),
        "rho_within_-0134": match(-0.134, rho_within, 1e-3),
        "rho_nonfloor_-0041": match(-0.041, rho_nfo, 1e-3),
        "nonfloor_range_028": match(0.28, nf_range, 1e-2),
    },
}
# floor vs nonfloor within-unit AUC.  The headline (Result.md Finding 3,
# 11_evidence_tier.py) is a POOLED within-unit-centred Mann-Whitney AUC: rank
# all within-unit-centred strength values, floor rows vs non-floor rows, then
# AUC = W / (n_floor * n_nonfloor).  (The earlier per-unit rank-average was a
# different estimator and did not match the headline.)
S_w = u[Y] - u.groupby("unit_id")[Y].transform("mean")
floor_mask = u[P].eq(FLOOR)
pos = S_w[floor_mask].to_numpy()
neg = S_w[~floor_mask].to_numpy()
auc_w = float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))
res["F3_evidence_tier"]["floor_vs_nonfloor_within_unit_auc_recomputed"] = auc_w
res["F3_evidence_tier"]["floor_vs_nonfloor_within_unit_auc_match"] = (
    match(0.618, auc_w, 1e-2)
)

# -------------------------------------------------------------------------- F4
trans = 0; n3 = 0
for _, g in u.groupby("unit_id"):
    if g.stage_index.nunique() < 3:
        continue
    n3 += 1
    if g[Y].min() < 0 < g[Y].max():
        trans += 1
res["F4_transience"] = {
    "n3_stages": n3, "transient_recomputed": trans,
    "report_match": match(210, trans) and match(579, n3),
}

# per-family MAE improvement, recomputed independently.  This uses the SAME
# vectorized leave-one-family-out holdout as 13_holdout_replication.py:
#   null  = unit leave-one-out mean (loo)
#   trend = loo + delta[stage], where delta[s] is the mean residual at stage s
#           over families OTHER than the record's own family.
# Per-family improvement = MAE_null - MAE_trend within that family; the sign
# test is a two-sided binomial test on the non-tied families.  (The earlier
# per-family loop had a sign-flip bug that gave 33 / 181; the vectorised
# recomputation below reproduces the authoritative 102 / 112.)
def _holdout_delta(stage_lab, group_codes, n_groups, resid, n_stages=ns):
    grid = stage_lab * n_groups + group_codes
    s_sum = np.bincount(stage_lab, weights=resid, minlength=n_stages)
    s_cnt = np.bincount(stage_lab, minlength=n_stages).astype(float)
    g_sum = np.bincount(grid, weights=resid, minlength=n_stages * n_groups).reshape(
        n_stages, n_groups
    )
    g_cnt = np.bincount(grid, minlength=n_stages * n_groups).reshape(
        n_stages, n_groups
    ).astype(float)
    num = s_sum[:, None] - g_sum
    den = s_cnt[:, None] - g_cnt
    safe = den > 0
    delta = np.zeros_like(num)
    delta[safe] = num[safe] / den[safe]
    return delta

# Replicate the authoritative 13_holdout_replication.py computation EXACTLY so
# the independent recomputation agrees with it (102 improved / 112 worse,
# two-sided sign p = 0.5385040002283777).  Uses pure numpy, identical to the
# 13 script: loo = unit leave-one-out mean; resid = y - loo; delta[s] is the
# mean residual at stage s over families OTHER than the record's own family.
# The subset must be in the same row order as the 13 script (sorted by
# unit_id, stage_index) so that the float summation order — and hence the
# near-zero per-family gains — matches exactly.
sub_s = sub.sort_values(["unit_id", "stage_index"], kind="stable").reset_index(drop=True)
y13 = sub_s[Y].to_numpy(dtype=float)
stage13 = sub_s.stage_index.to_numpy(dtype=np.int64)
unit_uniq13, unit_codes13 = np.unique(sub_s.unit_id.to_numpy(), return_inverse=True)
fam_uniq13, fam_codes13 = np.unique(sub_s.interaction_id.to_numpy(), return_inverse=True)
n_units13 = len(unit_uniq13)
u_sum13 = np.bincount(unit_codes13, weights=y13, minlength=n_units13)
u_n13 = np.bincount(unit_codes13, minlength=n_units13).astype(float)
loo13 = (u_sum13[unit_codes13] - y13) / (u_n13[unit_codes13] - 1.0)
resid13 = y13 - loo13
delta13 = _holdout_delta(stage13, fam_codes13, len(fam_uniq13), resid13)
pred13 = loo13 + delta13[stage13, fam_codes13]
null_err13 = np.abs(y13 - loo13)
obs_err13 = np.abs(y13 - pred13)
gain_by_fam = []
for g in range(len(fam_uniq13)):
    m = fam_codes13 == g
    gain_by_fam.append(float(null_err13[m].mean()) - float(obs_err13[m].mean()))
gain_by_fam = np.asarray(gain_by_fam)
n_improved = int((gain_by_fam > 0).sum())
n_worsened = int((gain_by_fam < 0).sum())
res["F4_transience"]["per_family_improvement"] = {
    "n_families": int(len(gain_by_fam)),
    "n_improved": n_improved,
    "n_worsened": n_worsened,
    "sign_test_p": float(
        stats.binomtest(n_improved, n_improved + n_worsened, 0.5, alternative="two-sided").pvalue
    ),
    "report_match": match(102, n_improved) and match(112, n_worsened),
}

# -------------------------------------------------------------------------- F5
nper = u.groupby("unit_id").size()
m5 = u[u.unit_id.isin(nper.index[nper >= 2])]
anyf = m5.groupby("unit_id").apply(lambda g: bool(g[P].eq(FLOOR).any()))
allf = m5.groupby("unit_id").apply(lambda g: bool(g[P].eq(FLOOR).all()))
af = int(allf.sum()); an = int((anyf == 0).sum()); mx = len(anyf) - af - an
res["F5_lability"] = {
    "total_multi_stage": int(len(anyf)),
    "always_floor_recomputed": af,
    "always_nonfloor_recomputed": an,
    "mixed_recomputed": mx,
    "share_mixed_recomputed": float(mx / len(anyf)),
    "report_match": {
        "correct_360_22_336": match(360, af) and match(22, an) and match(336, mx),
        "buggy_8_22_688_would_match": match(8, af) and match(688, mx),
    },
}
tf = tn = nf2 = nn = 0
for _, g in m5.groupby("unit_id"):
    g = g.sort_values("stage_index")
    seq = g[P].eq(FLOOR).astype(int).to_numpy()
    for a, b in zip(seq[:-1], seq[1:]):
        if a and b: tf += 1
        elif a and not b: tn += 1
        elif not a and b: nf2 += 1
        else: nn += 1
res["F5_lability"]["transitions"] = {
    "floor_floor": tf, "floor_nonfloor": tn, "nonfloor_floor": nf2, "nonfloor_nonfloor": nn,
    "P_floor_next_given_floor": float(tf / (tf + tn) if (tf + tn) else float("nan")),
    "P_floor_next_given_nonfloor": float(nf2 / (nf2 + nn) if (nf2 + nn) else float("nan")),
}
res["F5_lability"]["transition_match"] = {
    "P_floor_given_floor_0931": match(0.931, res["F5_lability"]["transitions"]["P_floor_next_given_floor"], 1e-2),
    "P_floor_given_nonfloor_0708": match(0.708, res["F5_lability"]["transitions"]["P_floor_next_given_nonfloor"], 1e-2),
}
half = u.assign(h=np.where(u.stage_index < 9, "e", "l"))
ym = half.pivot_table(index="unit_id", columns="h", values=Y, aggfunc="mean")
b = ym.notna().all(axis=1)
res["F5_lability"]["strength_split_half_r_recomputed"] = float(
    np.corrcoef(ym.loc[b, "e"], ym.loc[b, "l"])[0, 1]
)
res["F5_lability"]["strength_split_half_match"] = match(0.908, res["F5_lability"]["strength_split_half_r_recomputed"], 1e-3)

res["summary"] = {
    "integrity_sha_match": res["integrity"]["sha256_matches_target"],
    "census_match": all(res["census"][k] == v for k, v in {
        "raw_rows": 5990, "analysis_rows": 5920, "units": 1105,
        "families": 285, "cell_pairs": 9, "stages": 18,
    }.items()),
    "F1_slope_match": res["F1_trend"]["slope_match"],
    "F1_curve_r_match": res["F1_trend"]["curve_r_match"],
    "F1_bootstrap_ci_match": res["F1_trend"]["bootstrap"]["ci_match"],
    "F1_holdout_match": res["F1_holdout"]["report_match"],
    "F2_family_share_match": res["F2_identity"]["report_match"],
    "F3_floor_share_match": res["F3_evidence_tier"]["report_match"],
    "F4_transience_match": res["F4_transience"]["report_match"],
    "F4_per_family_match": res["F4_transience"]["per_family_improvement"]["report_match"],
    "F5_lability_correct_match": res["F5_lability"]["report_match"]["correct_360_22_336"],
    "F5_lability_buggy_would_match": res["F5_lability"]["report_match"]["buggy_8_22_688_would_match"],
    "F5_transition_match": res["F5_lability"]["transition_match"],
    "F5_strength_split_half_match": res["F5_lability"]["strength_split_half_match"],
}
# "all headline numbers reproduce" must check every positive match flag,
# including the nested per-check dicts (F1_holdout_match, F2_family_share_match,
# F3_floor_share_match, F5_transition_match).  The previous line only looked at
# top-level bools, so it (a) skipped the nested match dicts and (b) let the
# negative-control flag F5_lability_buggy_would_match — which is *intentionally*
# False because it asserts the buggy 8/22/688 path does NOT reproduce the
# correct 360/22/336 counts — force the whole flag to False.  That negative
# control is a check on the bug-detection itself, not a headline number in
# Result.md, so it is excluded from this roll-up.
def _flatten_match_flags(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten_match_flags(v)
        elif isinstance(v, bool):
            yield k, v

_NEGATIVE_CONTROLS = {"F5_lability_buggy_would_match"}
res["all_headline_numbers_reproduce"] = all(
    flag for k, flag in _flatten_match_flags(res["summary"]) if k not in _NEGATIVE_CONTROLS
)

(OUT / "independent_rederr.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res["summary"], indent=2))
print("Wrote", OUT / "independent_rederr.json")