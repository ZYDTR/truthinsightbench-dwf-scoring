#!/usr/bin/env python3
"""
11_evidence_tier.py

Independent characterisation of the association_pvalue column as a measurement
instrument, using the deduplicated analysis table.

Inputs : output/results/analysis_table.tsv  (5920 rows, unit_id included)
Outputs: output/results/evidence_tier.json

Sections
  1. Exact distribution of association_pvalue over the 6 levels, overall and per stage.
  2. Is interaction_strength_like_value ORDERED by the p level? three framings:
     (a) raw rank correlation, (b) within-unit centred, (c) non-floor levels only.
  3. Does the p level track ABSOLUTE magnitude |strength - unit mean|?
  4. Is the evidence tier labile across stages within a unit? (states, transitions,
     transition probabilities, permutation benchmarks, test-retest reliability)
  5. Verdict with exact numbers.

Convention used throughout:  P = association_pvalue (6-level ordinal report tier),
S = interaction_strength_like_value, floor = (P == 0.0009), unit = interaction x cell_pair.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

PROJ = "/data/user/conversations/conv-9877e7dd-aaaa-49c9-a430-e071a69303ee/project"
IN_TSV = os.path.join(PROJ, "output/results/analysis_table.tsv")
OUT_JSON = os.path.join(PROJ, "output/results/evidence_tier.json")

N_PERM = 1000
FLOOR = 0.0009
P_LEVELS = [0.0009, 0.01, 0.02, 0.03, 0.04, 0.05]
NONFLOOR = P_LEVELS[1:]


def f(x):
    """plain-float JSON-safe scalar (None for non-finite)"""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------
# 0. load
# ----------------------------------------------------------------------------
df = pd.read_csv(IN_TSV, sep="\t")
df["P"] = df["association_pvalue"].astype(float)
df["S"] = df["interaction_strength_like_value"].astype(float)
df["floor"] = (df["P"] == FLOOR).astype(int)

n_rows = len(df)
n_units = df["unit_id"].nunique()
n_stages = df["stage_index"].nunique()

# within-unit centring of strength (unit_mean subtracted)
df["S_unit_mean"] = df.groupby("unit_id")["S"].transform("mean")
df["S_w"] = df["S"] - df["S_unit_mean"]
df["absdev"] = df["S_w"].abs()
df["P_w"] = df["P"] - df.groupby("unit_id")["P"].transform("mean")
df["absdev_w"] = df["absdev"] - df.groupby("unit_id")["absdev"].transform("mean")
# alternative centring: family (interaction_id), for a robustness check
df["S_fam_mean"] = df.groupby("interaction_id")["S"].transform("mean")
df["S_f"] = df["S"] - df["S_fam_mean"]

out = {
    "meta": {
        "input_table": "output/results/analysis_table.tsv",
        "n_rows": int(n_rows),
        "n_units": int(n_units),
        "n_stages": int(n_stages),
        "n_p_levels": int(df["P"].nunique()),
        "floor_level": FLOOR,
        "n_perm": N_PERM,
        "notes": [
            "unit_id = interaction_id|cell_pair_id; repeated stages of a unit are dependent",
            "the p column is a 6-level ordinal report tier, not a continuous statistic",
            "within-unit centring subtracts the unit's own mean strength",
            "sign convention: negative Spearman(P, S) means weaker evidence (larger p) -> smaller strength",
        ],
    }
}

print("=" * 78)
print("11  EVIDENCE TIER: association_pvalue as a measurement instrument")
print("=" * 78)
print(f"rows={n_rows}  units={n_units}  stages={n_stages}")

# ----------------------------------------------------------------------------
# 1. distribution of the p tier
# ----------------------------------------------------------------------------
print("\n--- 1. DISTRIBUTION OF association_pvalue ---")
cnt = df["P"].value_counts().reindex(P_LEVELS).fillna(0).astype(int)
share = cnt / n_rows
floor_share = f(df["floor"].mean())

print("overall level counts / shares:")
for lv in P_LEVELS:
    print(f"   p={lv:<7g} n={cnt[lv]:5d}  share={share[lv]:.4f}")
print(f"   floor share overall = {floor_share:.4f}   non-floor share = {1 - floor_share:.4f}")

per_stage = {}
floor_share_by_stage = {}
for st, g in df.groupby("stage_index"):
    c = g["P"].value_counts().reindex(P_LEVELS).fillna(0).astype(int)
    per_stage[str(int(st))] = {
        "n_rows": int(len(g)),
        "n_units": int(g["unit_id"].nunique()),
        "counts": {f"{lv:g}": int(c[lv]) for lv in P_LEVELS},
        "shares": {f"{lv:g}": f(c[lv] / len(g)) for lv in P_LEVELS},
        "floor_share": f((g["P"] == FLOOR).mean()),
        "nonfloor_share": f((g["P"] != FLOOR).mean()),
    }
    floor_share_by_stage[int(st)] = f((g["P"] == FLOOR).mean())

fs_arr = np.array([floor_share_by_stage[s] for s in sorted(floor_share_by_stage)])
print("floor share by stage (0..17):")
print("   " + "  ".join(f"{s}:{floor_share_by_stage[s]:.3f}" for s in sorted(floor_share_by_stage)))
lo_st = int(np.argmin(fs_arr)); hi_st = int(np.argmax(fs_arr))
print(f"   min floor share = {fs_arr.min():.4f} at stage {lo_st};  max = {fs_arr.max():.4f} at stage {hi_st};"
      f"  range = {fs_arr.max() - fs_arr.min():.4f}")
print(f"   floor share early stages (0-3) mean={np.mean([floor_share_by_stage[s] for s in range(4)]):.4f},"
      f" middle (4-13) mean={np.mean([floor_share_by_stage[s] for s in range(4,14)]):.4f},"
      f" late (14-17) mean={np.mean([floor_share_by_stage[s] for s in range(14,18)]):.4f}")
# stage-level coupling of the floor share with stage-level mean strength
stg = df.groupby("stage_index").agg(floor_share=("floor", "mean"), mean_S=("S", "mean"))
rho_stg_couple = stats.spearmanr(stg["floor_share"], stg["mean_S"])
print(f"   across the 18 stages: Spearman(floor_share, mean strength) = {rho_stg_couple[0]:+.4f} "
      f"(p={rho_stg_couple[1]:.3g})")

out["section1_distribution"] = {
    "overall_counts": {f"{lv:g}": {"n": int(cnt[lv]), "share": f(share[lv])} for lv in P_LEVELS},
    "floor_share_overall": floor_share,
    "nonfloor_share_overall": f(1 - floor_share),
    "per_stage": per_stage,
    "floor_share_min_stage": lo_st, "floor_share_min": f(fs_arr.min()),
    "floor_share_max_stage": hi_st, "floor_share_max": f(fs_arr.max()),
    "floor_share_range": f(fs_arr.max() - fs_arr.min()),
    "floor_share_early_0_3_mean": f(np.mean([floor_share_by_stage[s] for s in range(4)])),
    "floor_share_mid_4_13_mean": f(np.mean([floor_share_by_stage[s] for s in range(4, 14)])),
    "floor_share_late_14_17_mean": f(np.mean([floor_share_by_stage[s] for s in range(14, 18)])),
    "stage_level_spearman_floor_share_vs_mean_strength": f(rho_stg_couple[0]),
    "stage_level_spearman_p": f(rho_stg_couple[1]),
    "level_usage_spearman_p_value_vs_count": f(stats.spearmanr(P_LEVELS, cnt.values)[0]),
}

# ----------------------------------------------------------------------------
# 2. is strength ORDERED by the p tier?  three framings
# ----------------------------------------------------------------------------
print("\n--- 2. IS STRENGTH ORDERED BY THE p LEVEL? (three framings) ---")

# (a) raw, all rows
rho_raw, p_raw = stats.spearmanr(df["P"], df["S"])
tau_raw, ptau_raw = stats.kendalltau(df["P"], df["S"])
pear_raw, ppear_raw = stats.pearsonr(df["P"], df["S"])
print(f"(a) raw all rows (n={n_rows}): Spearman rho={rho_raw:+.4f} (p={p_raw:.3g}) | Kendall tau={tau_raw:+.4f} "
      f"| Pearson r={pear_raw:+.4f}")
print("    sign reminder: rho<0 = larger p (weaker evidence) goes with smaller strength")
rho_floor_raw, p_floor_raw = stats.spearmanr(df["floor"], df["S"])
pb_floor_raw, ppb = stats.pointbiserialr(df["floor"], df["S"])
print(f"    floor-indicator vs strength: Spearman rho={rho_floor_raw:+.4f} (p={p_floor_raw:.3g}); "
      f"point-biserial r={pb_floor_raw:+.4f} (p={ppb:.3g})")

# (b) within-unit centred
rho_w, p_w = stats.spearmanr(df["P"], df["S_w"])
pear_w, ppear_w = stats.pearsonr(df["P"], df["S_w"])
rho_dem, p_dem = stats.spearmanr(df["P_w"], df["S_w"])
rho_f, p_f = stats.spearmanr(df["P"], df["S_f"])
print(f"(b) within-unit centred: Spearman(P, S-unit mean)={rho_w:+.4f} (p={p_w:.3g}); "
      f"Pearson={pear_w:+.4f}; Spearman(demeaned P, demeaned S)={rho_dem:+.4f} (p={p_dem:.3g})")
print(f"    (robustness) within-FAMILY centred: Spearman(P, S-family mean)={rho_f:+.4f} (p={p_f:.3g})")
per_unit_rhos = []
for uid, g in df.groupby("unit_id"):
    if len(g) >= 4 and g["P"].nunique() > 1 and g["S"].nunique() > 1:
        r, _ = stats.spearmanr(g["P"], g["S"])
        if np.isfinite(r):
            per_unit_rhos.append(r)
per_unit_rhos = np.array(per_unit_rhos)
print(f"    mean per-unit Spearman (units with >=4 stages and varying P, n={len(per_unit_rhos)}): "
      f"{per_unit_rhos.mean():+.4f} (median {np.median(per_unit_rhos):+.4f})")

# (c) only the 5 non-floor levels
nf = df[df["floor"] == 0]
rho_nf, p_nf = stats.spearmanr(nf["P"], nf["S"])
pear_nf, ppear_nf = stats.pearsonr(nf["P"], nf["S"])
rho_nf_w, p_nf_w = stats.spearmanr(nf["P"], nf["S_w"])
rho_nf_de, p_nf_de = stats.spearmanr(nf["P"] - nf.groupby("unit_id")["P"].transform("mean"),
                                     nf["S_w"])
print(f"(c) non-floor only (n={len(nf)}): Spearman(P,S)={rho_nf:+.4f} (p={p_nf:.3g}); Pearson={pear_nf:+.4f}; "
      f"Spearman(P, S-unit mean)={rho_nf_w:+.4f} (p={p_nf_w:.3g}); demeaned both={rho_nf_de:+.4f} (p={p_nf_de:.3g})")
# partial out stage within the non-floor rows
nf = nf.copy()
nf["S_res_stage"] = nf["S"] - nf.groupby("stage_index")["S"].transform("mean")
partial_nf = stats.spearmanr(nf["P"], nf["S_res_stage"])
print(f"    non-floor, strength residualised on stage: Spearman={partial_nf[0]:+.4f} (p={partial_nf[1]:.3g})")

# means per level, three framings
means_raw = df.groupby("P")["S"].agg(["size", "mean", "std"])
means_w = df.groupby("P")["S_w"].mean()
means_abs = df.groupby("P")["absdev"].mean()
mean_nf_range = f(means_raw.loc[NONFLOOR, "mean"].max() - means_raw.loc[NONFLOOR, "mean"].min())
sd_S = f(df["S"].std(ddof=1))
tbl = pd.DataFrame({
    "n": means_raw["size"].astype(int),
    "mean_strength_raw": means_raw["mean"],
    "mean_strength_within_unit": means_w,
    "mean_absdev": means_abs,
}).reindex(P_LEVELS)
print("\nmean strength by p level (all three framings + |deviation|):")
print(tbl.round(4).to_string())
print(f"    non-floor level means span only {mean_nf_range:.4f} strength units, vs SD(strength)={sd_S:.4f}")

# does ANY non-floor level differ from another?
groups = [nf.loc[nf["P"] == lv, "S"].values for lv in NONFLOOR]
kw = stats.kruskal(*groups)
kw_w = stats.kruskal(*[nf.loc[nf["P"] == lv, "S_w"].values for lv in NONFLOOR])
print(f"    Kruskal-Wallis across the 5 non-floor levels: raw H={kw.statistic:.3f} p={kw.pvalue:.3g}; "
      f"within-unit H={kw_w.statistic:.3f} p={kw_w.pvalue:.3g}")

# effect size: how much of strength variance does the p tier explain?
def eta2(frame, y, lvl):
    gm = frame[y].mean()
    ss_b = frame.groupby(lvl)[y].agg(["size", "mean"]).apply(lambda r: r["size"] * (r["mean"] - gm) ** 2, axis=1).sum()
    ss_t = ((frame[y] - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan

eta_raw = eta2(df, "S", "P")
eta_w = eta2(df, "S_w", "P")
eta_floor_raw = eta2(df, "S", "floor")
eta_floor_w = eta2(df, "S_w", "floor")
eta_nonfloor = eta2(nf, "S", "P")
print(f"    variance of strength explained by the full 6-level tier: eta2={eta_raw:.4f} (raw), "
      f"{eta_w:.4f} (within-unit)")
print(f"    variance of strength explained by the floor/nonfloor split alone: eta2={eta_floor_raw:.4f} (raw), "
      f"{eta_floor_w:.4f} (within-unit)")
print(f"    extra variance explained by the 5 non-floor levels beyond the floor split (non-floor rows only): "
      f"eta2={eta_nonfloor:.4f}")

# formal effect size of the only sizeable contrast: floor vs non-floor inside a unit
rows = []
for uid, g in df.groupby("unit_id"):
    flo = g.loc[g["floor"] == 1, "S"]
    nfl = g.loc[g["floor"] == 0, "S"]
    if len(flo) > 0 and len(nfl) > 0:
        rows.append((flo.mean(), nfl.mean(), len(flo), len(nfl)))
ub = pd.DataFrame(rows, columns=["m_floor", "m_non", "n_floor", "n_non"])
d_paired = ub["m_floor"] - ub["m_non"]
sd_w = f(df["S_w"].std(ddof=1))
d_pooled = float(df.loc[df["floor"] == 1, "S_w"].mean() - df.loc[df["floor"] == 0, "S_w"].mean())
pos = df.loc[df["floor"] == 1, "S_w"].values
neg = df.loc[df["floor"] == 0, "S_w"].values
auc_w = f(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))
auc_raw = f(stats.mannwhitneyu(df.loc[df["floor"] == 1, "S"], df.loc[df["floor"] == 0, "S"]).statistic
            / (int(df["floor"].sum()) * int((1 - df["floor"]).sum())))
wil = stats.wilcoxon(d_paired)
ptt = stats.ttest_rel(ub["m_floor"], ub["m_non"])
k_sign = int((d_paired > 0).sum())
sign_p = stats.binomtest(k_sign, len(d_paired), 0.5).pvalue
print(f"\nfloor vs non-floor INSIDE the same unit: {len(ub)} units contain both states")
print(f"    mean paired difference (floor - nonfloor) = {d_paired.mean():+.4f} strength units "
      f"(median {d_paired.median():+.4f}); floor higher in {k_sign}/{len(ub)} = {k_sign / len(ub):.4f}")
print(f"    Wilcoxon p={wil.pvalue:.3g}; paired t p={ptt.pvalue:.3g}; sign-test p={sign_p:.3g}")
print(f"    pooled within-unit difference = {d_pooled:+.4f}, within-unit SD of strength = {sd_w:.4f}, "
      f"Cohen's d = {d_pooled / sd_w:+.4f}")
print(f"    AUC(within-unit strength -> floor) = {auc_w:.4f}; AUC(raw strength -> floor) = {auc_raw:.4f}")

out["section2_ordering"] = {
    "a_raw_all_rows": {
        "n": int(n_rows),
        "spearman_rho": f(rho_raw), "spearman_p": f(p_raw),
        "kendall_tau": f(tau_raw), "kendall_p": f(ptau_raw),
        "pearson_r": f(pear_raw), "pearson_p": f(ppear_raw),
        "floor_indicator_spearman_rho": f(rho_floor_raw), "floor_indicator_spearman_p": f(p_floor_raw),
        "floor_indicator_point_biserial_r": f(pb_floor_raw), "floor_indicator_point_biserial_p": f(ppb),
    },
    "b_within_unit_centred": {
        "spearman_P_vs_S_minus_unitmean": f(rho_w), "p": f(p_w),
        "pearson_P_vs_S_minus_unitmean": f(pear_w), "pearson_p": f(ppear_w),
        "spearman_demeaned_both": f(rho_dem), "spearman_demeaned_both_p": f(p_dem),
        "spearman_within_family_centred": f(rho_f), "p_within_family": f(p_f),
        "mean_per_unit_spearman": f(per_unit_rhos.mean()) if len(per_unit_rhos) else None,
        "median_per_unit_spearman": f(np.median(per_unit_rhos)) if len(per_unit_rhos) else None,
        "n_units_used": int(len(per_unit_rhos)),
    },
    "c_nonfloor_only": {
        "n": int(len(nf)),
        "spearman_P_vs_S": f(rho_nf), "spearman_p": f(p_nf),
        "pearson_P_vs_S": f(pear_nf), "pearson_p": f(ppear_nf),
        "spearman_P_vs_S_minus_unitmean": f(rho_nf_w), "p": f(p_nf_w),
        "spearman_demeaned_both": f(rho_nf_de), "p_demeaned": f(p_nf_de),
        "spearman_strength_residualised_on_stage": f(partial_nf[0]), "p_stage_residualised": f(partial_nf[1]),
        "kruskal_across_5_levels_raw_H": f(kw.statistic), "kruskal_raw_p": f(kw.pvalue),
        "kruskal_across_5_levels_within_H": f(kw_w.statistic), "kruskal_within_p": f(kw_w.pvalue),
    },
    "mean_strength_by_level_table": {
        f"{lv:g}": {
            "n": int(means_raw.loc[lv, "size"]),
            "mean_strength_raw": f(means_raw.loc[lv, "mean"]),
            "sd_strength_raw": f(means_raw.loc[lv, "std"]),
            "mean_strength_within_unit": f(means_w.loc[lv]),
            "mean_absdev": f(means_abs.loc[lv]),
        } for lv in P_LEVELS
    },
    "nonfloor_level_mean_strength_range": f(mean_nf_range),
    "sd_strength_overall": sd_S,
    "eta2_strength_by_full_tier_raw": f(eta_raw),
    "eta2_strength_by_full_tier_within_unit": f(eta_w),
    "eta2_strength_by_floor_split_raw": f(eta_floor_raw),
    "eta2_strength_by_floor_split_within_unit": f(eta_floor_w),
    "eta2_nonfloor_levels_within_nonfloor_rows": f(eta_nonfloor),
    "effect_size_floor_vs_nonfloor_within_unit": {
        "n_units_with_both_states": int(len(ub)),
        "mean_paired_diff_floor_minus_nonfloor": f(d_paired.mean()),
        "median_paired_diff": f(d_paired.median()),
        "n_units_floor_higher": k_sign,
        "share_units_floor_higher": f(k_sign / len(ub)),
        "wilcoxon_p": f(wil.pvalue),
        "paired_t_p": f(ptt.pvalue),
        "sign_test_p": f(sign_p),
        "pooled_within_unit_diff": f(d_pooled),
        "within_unit_sd_strength": f(sd_w),
        "cohens_d_within": f(d_pooled / sd_w),
        "auc_within_unit_strength_for_floor": f(auc_w),
        "auc_raw_strength_for_floor": f(auc_raw),
    },
}

# ----------------------------------------------------------------------------
# 3. does p track the ABSOLUTE magnitude of strength?
# ----------------------------------------------------------------------------
print("\n--- 3. DOES p TRACK ABSOLUTE MAGNITUDE |strength - unit mean|? ---")
rho_abs, p_abs = stats.spearmanr(df["P"], df["absdev"])
pear_abs, ppear_abs = stats.pearsonr(df["P"], df["absdev"])
rho_abs_w, p_abs_w = stats.spearmanr(df["P_w"], df["absdev_w"])
rho_floor_abs, p_floor_abs = stats.spearmanr(df["floor"], df["absdev"])
pb_floor_abs, ppb_floor_abs = stats.pointbiserialr(df["floor"], df["absdev"])
print(f"Spearman(P, |S - unit mean|) = {rho_abs:+.4f} (p={p_abs:.3g}); Pearson = {pear_abs:+.4f} (p={ppear_abs:.3g})")
print(f"Spearman(demeaned P, demeaned |dev|) = {rho_abs_w:+.4f} (p={p_abs_w:.3g})")
print(f"floor-indicator vs |dev|: Spearman {rho_floor_abs:+.4f} (p={p_floor_abs:.3g}); "
      f"point-biserial {pb_floor_abs:+.4f} (p={ppb_floor_abs:.3g})")
abs_tbl = df.groupby("P")["absdev"].agg(["size", "mean", "median"]).reindex(P_LEVELS)
print("\nmean |strength - unit mean| by p level:")
print(abs_tbl.round(4).to_string())
print(f"    floor    mean|dev| = {abs_tbl.loc[FLOOR, 'mean']:.4f}")
print(f"    nonfloor mean|dev| = {df.loc[df.floor == 0, 'absdev'].mean():.4f}")
kw_abs = stats.kruskal(*[nf.loc[nf["P"] == lv, "absdev"].values for lv in NONFLOOR])
print(f"    Kruskal-Wallis across the 5 non-floor levels on |dev|: H={kw_abs.statistic:.3f} p={kw_abs.pvalue:.3g}")
# is |dev| ordered by tier?  compare with the same statistic on raw |S|
rho_absraw, p_absraw = stats.spearmanr(df["P"], df["S"].abs())
print(f"    (for reference) Spearman(P, |S| raw, no centring) = {rho_absraw:+.4f} (p={p_absraw:.3g})")

out["section3_absolute_magnitude"] = {
    "spearman_P_vs_absdev": f(rho_abs), "spearman_p": f(p_abs),
    "pearson_P_vs_absdev": f(pear_abs), "pearson_p": f(ppear_abs),
    "spearman_demeaned_P_vs_demeaned_absdev": f(rho_abs_w), "p": f(p_abs_w),
    "floor_indicator_vs_absdev_spearman": f(rho_floor_abs), "p": f(p_floor_abs),
    "floor_indicator_vs_absdev_point_biserial": f(pb_floor_abs), "p_point_biserial": f(ppb_floor_abs),
    "spearman_P_vs_raw_absS": f(rho_absraw), "p_raw_absS": f(p_absraw),
    "mean_absdev_by_level": {
        f"{lv:g}": {"n": int(abs_tbl.loc[lv, "size"]), "mean": f(abs_tbl.loc[lv, "mean"]),
                    "median": f(abs_tbl.loc[lv, "median"])} for lv in P_LEVELS
    },
    "mean_absdev_floor": f(abs_tbl.loc[FLOOR, "mean"]),
    "mean_absdev_nonfloor": f(df.loc[df.floor == 0, "absdev"].mean()),
    "kruskal_nonfloor_levels_H": f(kw_abs.statistic), "kruskal_nonfloor_levels_p": f(kw_abs.pvalue),
}

# ----------------------------------------------------------------------------
# 4. lability of the tier across stages within a unit
# ----------------------------------------------------------------------------
print("\n--- 4. LABILITY OF THE EVIDENCE TIER ACROSS STAGES WITHIN A UNIT ---")
d = df.sort_values(["unit_id", "stage_index"]).reset_index(drop=True)
n_stage_per_unit = d.groupby("unit_id")["stage_index"].nunique()
multi_ids = n_stage_per_unit[n_stage_per_unit >= 2]
print(f"units with >=2 observed stages: {len(multi_ids)} of {n_units}"
      f"  ({len(multi_ids) / n_units:.4f} of units)")
print(f"units with exactly 1 observed stage: {int((n_stage_per_unit == 1).sum())}")

state = d.groupby("unit_id").agg(
    n_stages=("stage_index", "nunique"), n_floor=("floor", "sum"), n_nonfloor=("floor", lambda x: int((1 - x).sum())))
mstate = state[state["n_stages"] >= 2].copy()
mstate["pattern"] = np.select([mstate["n_nonfloor"] == 0, mstate["n_floor"] == 0],
                              ["always_floor", "always_nonfloor"], default="mixed")
pat = mstate["pattern"].value_counts().to_dict()
print("state pattern across units observed at >=2 stages:")
for k in ["always_floor", "always_nonfloor", "mixed"]:
    print(f"    {k:16s} {pat.get(k, 0):4d}  ({pat.get(k, 0) / len(mstate):.4f})")

q = float(1 - df["floor"].mean())          # global non-floor rate
nvec = mstate["n_stages"].values
exp_af = float(np.sum((1 - q) ** nvec))
exp_an = float(np.sum(q ** nvec))
exp_mx = len(mstate) - exp_af - exp_an
print(f"iid benchmark (unit lengths held fixed, global non-floor rate q={q:.4f}):")
print(f"    expected always_floor={exp_af:.1f}, always_nonfloor={exp_an:.2f}, mixed={exp_mx:.1f}"
      f"   vs observed {pat.get('always_floor', 0)}, {pat.get('always_nonfloor', 0)}, {pat.get('mixed', 0)}")

mstate["p_nonfloor"] = mstate["n_nonfloor"] / mstate["n_stages"]
chi2 = float((((mstate["n_nonfloor"] - mstate["n_stages"] * q) ** 2) /
              (mstate["n_stages"] * q * (1 - q))).sum())
disp = chi2 / (len(mstate) - 1)
print(f"unit-level overdispersion of non-floor counts (units with >=2 stages): chi2={chi2:.1f} on df={len(mstate) - 1},"
      f" dispersion={disp:.3f} (>1 = non-floor rows cluster inside units)")
print(f"    per-unit non-floor share: mean={mstate['p_nonfloor'].mean():.4f} sd={mstate['p_nonfloor'].std():.4f};"
      f" share of ALL units never non-floor = {float((state['n_nonfloor'] == 0).mean()):.4f}")

# ---- transitions over consecutive observed stages, within a unit ----
d["nxt_floor"] = d.groupby("unit_id")["floor"].shift(-1)
d["nxt_stage"] = d.groupby("unit_id")["stage_index"].shift(-1)
d["gap"] = d["nxt_stage"] - d["stage_index"]
tr = d.dropna(subset=["nxt_floor"]).copy()
tr["gap"] = tr["gap"].astype(int)


def transitions(frame, label):
    a = frame["floor"].values.astype(int)      # 1 = floor, 0 = nonfloor
    b = frame["nxt_floor"].values.astype(int)
    c11 = int(((a == 1) & (b == 1)).sum())     # floor -> floor
    c10 = int(((a == 1) & (b == 0)).sum())     # floor -> nonfloor
    c01 = int(((a == 0) & (b == 1)).sum())     # nonfloor -> floor
    c00 = int(((a == 0) & (b == 0)).sum())     # nonfloor -> nonfloor
    n1, n0 = c11 + c10, c00 + c01
    return {
        "label": label, "n_pairs": int(len(a)),
        "floor_to_floor": c11, "floor_to_nonfloor": c10,
        "nonfloor_to_floor": c01, "nonfloor_to_nonfloor": c00,
        "n_now_floor": n1, "n_now_nonfloor": n0,
        "P_floor_next_given_floor": f(c11 / n1) if n1 else None,
        "P_nonfloor_next_given_floor": f(c10 / n1) if n1 else None,
        "P_floor_next_given_nonfloor": f(c01 / n0) if n0 else None,
        "P_nonfloor_next_given_nonfloor": f(c00 / n0) if n0 else None,
        "n_state_changes": c10 + c01,
    }


tr_all = transitions(tr, "all consecutive observed stage pairs within a unit")
tr_g1 = transitions(tr[tr["gap"] == 1], "gap==1 only (adjacent stages)")
print("\ntransitions over consecutive OBSERVED stages within a unit (nonfloor->floor / floor->nonfloor):")
for r in (tr_all, tr_g1):
    print(f"    [{r['label']}] n_pairs={r['n_pairs']}")
    print(f"       floor->floor        {r['floor_to_floor']:5d}")
    print(f"       floor->nonfloor     {r['floor_to_nonfloor']:5d}   P(nonfloor at next | floor now)   = "
          f"{r['P_nonfloor_next_given_floor']:.4f}")
    print(f"       nonfloor->nonfloor  {r['nonfloor_to_nonfloor']:5d}   P(nonfloor at next | nonfloor now)= "
          f"{r['P_nonfloor_next_given_nonfloor']:.4f}")
    print(f"       nonfloor->floor     {r['nonfloor_to_floor']:5d}   P(floor at next | nonfloor now)   = "
          f"{r['P_floor_next_given_nonfloor']:.4f}")

base_floor = float(df["floor"].mean())
print(f"\n    base rate P(floor) = {base_floor:.4f}")
print(f"    lift P(floor|floor)/base              = {tr_all['P_floor_next_given_floor'] / base_floor:.3f}")
print(f"    lift P(floor|nonfloor)/base           = {tr_all['P_floor_next_given_nonfloor'] / base_floor:.3f}"
      "   (<1 => non-floor state persists)")
print(f"    lift P(nonfloor|nonfloor)/P(nonfloor) = "
      f"{tr_all['P_nonfloor_next_given_nonfloor'] / (1 - base_floor):.3f}")

# ---- permutation benchmarks on the SAME within-unit ordered pairs ----
# (i) within-unit shuffle: preserves each unit's own floor/nonfloor mix, destroys time order
# (ii) global shuffle: also destroys the unit-level clustering
codes = pd.Categorical(d["unit_id"]).codes
base_order = np.lexsort((d["stage_index"].values, codes))
f_sorted = d["floor"].values.astype(int)[base_order]
# same_unit[i] refers to the PAIR (sorted row i, sorted row i+1), length n-1
same_unit = np.diff(codes[base_order]) == 0
gap_sorted = d["gap"].values[base_order][:-1]
gap1_pair = same_unit & (gap_sorted == 1)
a_idx = np.where(same_unit)[0]           # pairs used for the "all observed pairs" statistics
g1_idx = np.where(gap1_pair)[0]          # pairs used for the gap==1 statistics

n = len(d)
obs_c10_all = tr_all["floor_to_nonfloor"]
obs_c01_all = tr_all["nonfloor_to_floor"]
obs_chg_all = tr_all["n_state_changes"]
obs_c10_g1 = tr_g1["floor_to_nonfloor"]
obs_c01_g1 = tr_g1["nonfloor_to_floor"]
obs_chg_g1 = tr_g1["n_state_changes"]

rng = np.random.default_rng(20260920)
sim = np.zeros((N_PERM, 8))
for i in range(N_PERM):
    r = rng.random(n)
    order = np.lexsort((r, codes))          # random rank inside each unit
    lab_rand = np.empty(n, dtype=int)
    lab_rand[order] = f_sorted              # unit's own mix, randomly placed
    lab = lab_rand[base_order]              # back into unit-then-stage order
    a, b = lab[a_idx], lab[a_idx + 1]
    sim[i, 0] = int(((a == 1) & (b == 0)).sum())   # floor->nonfloor
    sim[i, 1] = int(((a == 0) & (b == 1)).sum())   # nonfloor->floor
    sim[i, 2] = int(((a == 1) & (b == 0)).sum() + ((a == 0) & (b == 1)).sum())  # changes
    a2, b2 = lab[g1_idx], lab[g1_idx + 1]
    sim[i, 3] = int(((a2 == 1) & (b2 == 0)).sum())
    sim[i, 4] = int(((a2 == 0) & (b2 == 1)).sum())
    sim[i, 5] = int(((a2 == 1) & (b2 == 0)).sum() + ((a2 == 0) & (b2 == 1)).sum())
    g = rng.permutation(f_sorted)
    ag, bg = g[a_idx], g[a_idx + 1]
    sim[i, 6] = int(((ag == 1) & (bg == 0)).sum() + ((ag == 0) & (bg == 1)).sum())
    gg, hh = g[g1_idx], g[g1_idx + 1]
    sim[i, 7] = int(((gg == 1) & (hh == 0)).sum() + ((gg == 0) & (hh == 1)).sum())


def pct(sim_col, obs):
    lo = float((sim_col <= obs).mean())
    hi = float((sim_col >= obs).mean())
    return {"obs": int(obs), "perm_mean": f(sim_col.mean()), "perm_sd": f(sim_col.std(ddof=1)),
            "p_perm_le_obs": f(lo), "p_perm_ge_obs": f(hi)}


print(f"\npermutation benchmarks (n_perm={N_PERM}, unit mixes preserved, labels reshuffled inside units):")
p_chg = pct(sim[:, 2], obs_chg_all)
p_chg_g1 = pct(sim[:, 5], obs_chg_g1)
p_c10 = pct(sim[:, 0], obs_c10_all)
p_c01 = pct(sim[:, 1], obs_c01_all)
p_chg_glob = pct(sim[:, 6], obs_chg_all)
p_chg_g1_glob = pct(sim[:, 7], obs_chg_g1)
print(f"    state changes, all in-unit observed pairs: observed={p_chg['obs']} vs "
      f"within-unit shuffle mean {p_chg['perm_mean']} (sd {p_chg['perm_sd']}), "
      f"p(<=obs)={p_chg['p_perm_le_obs']:.4f}, p(>=obs)={p_chg['p_perm_ge_obs']:.4f}")
print(f"    state changes, gap==1 pairs only        : observed={p_chg_g1['obs']} vs "
      f"mean {p_chg_g1['perm_mean']} (sd {p_chg_g1['perm_sd']}), "
      f"p(<=obs)={p_chg_g1['p_perm_le_obs']:.4f}")
print(f"    floor->nonfloor: observed={p_c10['obs']} vs mean {p_c10['perm_mean']} (sd {p_c10['perm_sd']})")
print(f"    nonfloor->floor: observed={p_c01['obs']} vs mean {p_c01['perm_mean']} (sd {p_c01['perm_sd']})")
print(f"    state changes under GLOBAL shuffle (destroys unit clustering too): observed={p_chg_glob['obs']} vs "
      f"mean {p_chg_glob['perm_mean']} (sd {p_chg_glob['perm_sd']}), p(<=obs)={p_chg_glob['p_perm_le_obs']:.4f}")
print(f"    gap==1 under GLOBAL shuffle: mean {p_chg_g1_glob['perm_mean']} (sd {p_chg_g1_glob['perm_sd']})")
persistence_ratio = float(p_chg["obs"] / float(p_chg["perm_mean"]))
print(f"    persistence ratio = observed changes / within-unit-shuffle expected changes = "
      f"{persistence_ratio:.4f}  (<1 => tier persists more than chance)")

# ---- test-retest reliability of the tier vs strength (units with >=4 stages) ----
rows = []
for uid, g in d.groupby("unit_id"):
    g = g.sort_values("stage_index")
    k = len(g)
    if k < 4:
        continue
    h = k // 2
    rows.append(dict(unit_id=uid, n=k,
                     P1=g["P"].values[:h].mean(), P2=g["P"].values[h:].mean(),
                     f1=g["floor"].values[:h].mean(), f2=g["floor"].values[h:].mean(),
                     s1=g["S"].values[:h].mean(), s2=g["S"].values[h:].mean()))
h = pd.DataFrame(rows)
r_s = stats.pearsonr(h["s1"], h["s2"])
r_s_sp = stats.spearmanr(h["s1"], h["s2"])
r_f_ = stats.pearsonr(h["f1"], h["f2"])
r_f_sp = stats.spearmanr(h["f1"], h["f2"])
r_p_ = stats.pearsonr(h["P1"], h["P2"])
r_p_sp = stats.spearmanr(h["P1"], h["P2"])
a1, a2 = (h["f1"] < 0.5), (h["f2"] < 0.5)
agree = float((a1 == a2).mean())
p1m, p2m = float(a1.mean()), float(a2.mean())
exp_agree = p1m * p2m + (1 - p1m) * (1 - p2m)
kappa = (agree - exp_agree) / (1 - exp_agree)
print(f"\ntest-retest reliability (units with >=4 observed stages, first half vs second half of stages, n={len(h)}):")
print(f"    strength   split-half r = {r_s[0]:.4f} (Spearman {r_s_sp[0]:.4f})")
print(f"    floor_frac split-half r = {r_f_[0]:.4f} (Spearman {r_f_sp[0]:.4f})")
print(f"    mean P     split-half r = {r_p_[0]:.4f} (Spearman {r_p_sp[0]:.4f})")
print(f"    early/late floor-class agreement = {agree:.4f}; chance={exp_agree:.4f}; kappa={kappa:.4f}")

out["section4_lability"] = {
    "n_units_multi_stage": int(len(mstate)),
    "n_units_single_stage": int((n_stage_per_unit == 1).sum()),
    "state_patterns": {
        "always_floor": int(pat.get("always_floor", 0)),
        "always_nonfloor": int(pat.get("always_nonfloor", 0)),
        "mixed": int(pat.get("mixed", 0)),
        "share_always_floor": f(pat.get("always_floor", 0) / len(mstate)),
        "share_always_nonfloor": f(pat.get("always_nonfloor", 0) / len(mstate)),
        "share_mixed": f(pat.get("mixed", 0) / len(mstate)),
    },
    "iid_benchmark_same_unit_lengths": {
        "global_nonfloor_rate_q": q,
        "expected_always_floor": f(exp_af), "expected_always_nonfloor": f(exp_an), "expected_mixed": f(exp_mx),
    },
    "unit_level_clustering": {
        "chi2": f(chi2), "df": int(len(mstate) - 1), "dispersion": f(disp),
        "per_unit_nonfloor_share_mean": f(mstate["p_nonfloor"].mean()),
        "per_unit_nonfloor_share_sd": f(mstate["p_nonfloor"].std()),
        "share_all_units_never_nonfloor": f(float((state["n_nonfloor"] == 0).mean())),
    },
    "transitions_all_consecutive_observed": tr_all,
    "transitions_gap1_only": tr_g1,
    "base_rate_floor": base_floor,
    "lift_floor_given_floor": f(tr_all["P_floor_next_given_floor"] / base_floor),
    "lift_floor_given_nonfloor": f(tr_all["P_floor_next_given_nonfloor"] / base_floor),
    "lift_nonfloor_given_nonfloor": f(tr_all["P_nonfloor_next_given_nonfloor"] / (1 - base_floor)),
    "permutation_benchmarks": {
        "n_perm": N_PERM,
        "scheme": "within-unit shuffle of the floor/nonfloor labels (unit mixes preserved); "
                  "global shuffle reported for contrast",
        "state_changes_all_pairs": p_chg,
        "state_changes_gap1_pairs": p_chg_g1,
        "floor_to_nonfloor_all_pairs": p_c10,
        "nonfloor_to_floor_all_pairs": p_c01,
        "state_changes_all_pairs_global_shuffle": p_chg_glob,
        "state_changes_gap1_pairs_global_shuffle": p_chg_g1_glob,
        "persistence_ratio_obs_over_expected_changes": f(persistence_ratio),
    },
    "test_retest_units_ge4_stages": {
        "n_units": int(len(h)),
        "strength_split_half_r": f(r_s[0]), "strength_split_half_spearman": f(r_s_sp[0]),
        "floor_frac_split_half_r": f(r_f_[0]), "floor_frac_split_half_spearman": f(r_f_sp[0]),
        "mean_P_split_half_r": f(r_p_[0]), "mean_P_split_half_spearman": f(r_p_sp[0]),
        "early_late_floor_class_agreement": f(agree), "chance_agreement": f(exp_agree),
        "cohen_kappa": f(kappa),
    },
}

# ----------------------------------------------------------------------------
# extra structure checks (stage effects, unit effects, step-level coupling)
# ----------------------------------------------------------------------------
print("\n--- EXTRA STRUCTURE CHECKS ---")
rho_p_stage = stats.spearmanr(df["P"], df["stage_index"])
rho_s_stage = stats.spearmanr(df["S"], df["stage_index"])
rho_floor_stage = stats.spearmanr(df["floor"], df["stage_index"])
print(f"Spearman(P, stage)      = {rho_p_stage[0]:+.4f} (p={rho_p_stage[1]:.3g})")
print(f"Spearman(floor, stage)  = {rho_floor_stage[0]:+.4f} (p={rho_floor_stage[1]:.3g})")
print(f"Spearman(strength, stage)= {rho_s_stage[0]:+.4f} (p={rho_s_stage[1]:.3g})")

# variance explained by unit / stage, restricted to units with >=3 stages (avoids the
# artefact that 1-row units are perfectly explained by their own mean)
dfm = df[df.groupby("unit_id")["unit_id"].transform("size") >= 3]


def r2_by(frame, y, key):
    gm = frame[y].mean()
    pred = frame.groupby(key)[y].transform("mean")
    return float(1 - ((frame[y] - pred) ** 2).sum() / ((frame[y] - gm) ** 2).sum())


r2_unit_floor = r2_by(dfm, "floor", "unit_id")
r2_unit_S = r2_by(dfm, "S", "unit_id")
r2_stage_floor = r2_by(df, "floor", "stage_index")
r2_stage_S = r2_by(df, "S", "stage_index")
print(f"units with >=3 stages (n={len(dfm)} rows, {dfm['unit_id'].nunique()} units):")
print(f"   strength variance explained by unit_id = {r2_unit_S:.4f};  floor/nonfloor variance explained = {r2_unit_floor:.4f}")
print(f"(all rows) floor variance explained by stage = {r2_stage_floor:.4f}; strength variance explained by stage = {r2_stage_S:.4f}")

# step-level: does strength move in the direction implied by the tier jump?
d2 = d.copy()
d2["dS"] = d2.groupby("unit_id")["S"].diff()
d2["dP"] = d2.groupby("unit_id")["P"].diff()
d2["dfloor"] = d2.groupby("unit_id")["floor"].diff()
dd = d2.dropna(subset=["dS", "dP"])
dd = dd[(dd["dS"] != 0) & (dd["dP"] != 0)]
rho_step = stats.spearmanr(dd["dS"], dd["dP"])
dl = d2.dropna(subset=["dS", "dfloor"])
dl = dl[dl["dfloor"] != 0]
# dfloor = floor(t) - floor(t-1):  +1 = nonfloor -> floor,  -1 = floor -> nonfloor
m_n2f = float(dl.loc[dl["dfloor"] == 1, "dS"].mean())     # strength change when evidence strengthens
m_f2n = float(dl.loc[dl["dfloor"] == -1, "dS"].mean())    # strength change when evidence weakens
n_n2f = int((dl["dfloor"] == 1).sum())
n_f2n = int((dl["dfloor"] == -1).sum())
print(f"\nstage-to-stage steps: Spearman(dStrength, dP) = {rho_step[0]:+.4f} (n={len(dd)})"
      "  (negative => strength falls when p rises)")
print(f"   steps nonfloor->floor (evidence strengthens): mean dStrength = {m_n2f:+.4f} (n={n_n2f})")
print(f"   steps floor->nonfloor (evidence weakens):     mean dStrength = {m_f2n:+.4f} (n={n_f2n})")
tstep = stats.ttest_ind(dl.loc[dl["dfloor"] == 1, "dS"], dl.loc[dl["dfloor"] == -1, "dS"], equal_var=False)
print(f"   difference of step means (n->f minus f->n) = {m_n2f - m_f2n:+.4f}; Welch t p={tstep.pvalue:.3g}")

out["extra_structure"] = {
    "spearman_P_vs_stage": f(rho_p_stage[0]), "spearman_P_vs_stage_p": f(rho_p_stage[1]),
    "spearman_floor_vs_stage": f(rho_floor_stage[0]), "spearman_floor_vs_stage_p": f(rho_floor_stage[1]),
    "spearman_strength_vs_stage": f(rho_s_stage[0]), "spearman_strength_vs_stage_p": f(rho_s_stage[1]),
    "units_ge3_stages": {"n_rows": int(len(dfm)), "n_units": int(dfm["unit_id"].nunique()),
                         "variance_strength_explained_by_unit": f(r2_unit_S),
                         "variance_floor_explained_by_unit": f(r2_unit_floor)},
    "all_rows": {"variance_floor_explained_by_stage": f(r2_stage_floor),
                 "variance_strength_explained_by_stage": f(r2_stage_S)},
    "step_level": {
        "n_steps_varying_both": int(len(dd)),
        "spearman_dS_vs_dP": f(rho_step[0]), "p": f(rho_step[1]),
        "mean_dS_nonfloor_to_floor": f(m_n2f), "n_nonfloor_to_floor_steps": n_n2f,
        "mean_dS_floor_to_nonfloor": f(m_f2n), "n_floor_to_nonfloor_steps": n_f2n,
        "difference_of_step_means_n2f_minus_f2n": f(m_n2f - m_f2n), "welch_t_p": f(tstep.pvalue),
    },
}

# ---- reconciliation with the existing main-analysis artifact ----
pv = d.pivot_table(index="unit_id", columns="stage_index", values="P", aggfunc="min")
obs_mask = pv.notna()
multi_mask = obs_mask.sum(axis=1) >= 2
mrow = pv[multi_mask]
om = obs_mask[multi_mask]                       # True where the stage is observed
n_observed = om.sum(axis=1).values
n_at_floor = ((mrow == FLOOR) & om).sum(axis=1).values
n_af = int((n_at_floor == n_observed).sum())
n_an = int((n_at_floor == 0).sum())
n_mx = int(len(mrow) - n_af - n_an)
# the uncorrected version, exactly as published by the main analysis
n_af_naive = int(((mrow == FLOOR)).all(axis=1).sum())
n_an_naive = int(((mrow != FLOOR)).all(axis=1).sum())
assert n_af == pat.get("always_floor", -1) and n_an == pat.get("always_nonfloor", -1) and n_mx == pat.get("mixed", -1), \
    "pivot reconciliation does not match the per-unit grouping counts"
print("\nreconciliation with output/results/evidence_coupling.json q2_evidence_stability:")
print(f"    correct (mask to observed stages): always_floor={n_af}, always_nonfloor={n_an}, mixed={n_mx}")
print(f"    main-analysis published          : always_floor=8, always_nonfloor=22, mixed=688")
print(f"    reproduced naive formula         : always_floor={n_af_naive}, always_nonfloor={n_an_naive}")
print("    cause: the published 'always_floor_p' uses .all(axis=1) over a NaN-filled 18-column pivot,")
print("           so it additionally requires ALL 18 stages to be observed (only 14 units qualify);")
print("           'always_nonfloor_p' accidentally survives because NaN==0.0009 evaluates False.")
out["reconciliation_with_main_analysis"] = {
    "main_analysis_artifact": "output/results/evidence_coupling.json",
    "published_state_patterns": {"always_floor": 8, "always_nonfloor": 22, "mixed": 688,
                                 "n_units_multi_stage": 718},
    "recomputed_correct_state_patterns": {"always_floor": n_af, "always_nonfloor": n_an, "mixed": n_mx,
                                          "n_units_multi_stage": int(len(mrow))},
    "reproduced_naive_formula": {"always_floor": n_af_naive, "always_nonfloor": n_an_naive},
    "diagnosis": ("the published always_floor count is a NaN-pivot artefact: sig.all(axis=1) is evaluated over "
                  "all 18 stage columns, so a unit must have all 18 stages observed AND all at the floor; only "
                  "14 units have all 18 stages and 8 of them are always-floor. The correct count, masking to "
                  f"observed stages, is {n_af}. always_nonfloor (22) is unaffected because NaN==0.0009 is False. "
                  "Transition counts are unaffected (they iterate over observed values only)."),
}

# ----------------------------------------------------------------------------
# 5. verdict
# ----------------------------------------------------------------------------
print("\n--- 5. VERDICT ---")
out["section5_verdict"] = {
    "call": ("the p column is a coarse REPORT TIER whose only reliable content is the floor vs non-floor "
             "dichotomy; above the floor the five levels carry essentially no information about strength, "
             "the tier does not track absolute magnitude, and it is highly labile across stages"),
    "key_numbers": {
        "floor_share": floor_share,
        "nonfloor_level_shares": {f"{lv:g}": f(share[lv]) for lv in NONFLOOR},
        "raw_spearman_P_vs_S": f(rho_raw),
        "within_unit_spearman_P_vs_S": f(rho_w),
        "within_family_spearman_P_vs_S": f(rho_f),
        "nonfloor_only_spearman_P_vs_S": f(rho_nf),
        "nonfloor_only_spearman_pvalue": f(p_nf),
        "nonfloor_only_within_unit_spearman": f(rho_nf_w),
        "nonfloor_only_within_unit_pvalue": f(p_nf_w),
        "nonfloor_level_mean_strength_range": f(mean_nf_range),
        "sd_strength_overall": sd_S,
        "eta2_strength_by_full_tier_raw": f(eta_raw),
        "eta2_strength_by_full_tier_within_unit": f(eta_w),
        "eta2_strength_by_floor_split_within_unit": f(eta_floor_w),
        "eta2_nonfloor_levels_within_nonfloor": f(eta_nonfloor),
        "kruskal_nonfloor_levels_p_raw": f(kw.pvalue),
        "floor_vs_nonfloor_within_unit_paired_diff": f(d_paired.mean()),
        "floor_vs_nonfloor_auc_within_unit": f(auc_w),
        "floor_vs_nonfloor_auc_raw": f(auc_raw),
        "cohens_d_within": f(d_pooled / sd_w),
        "spearman_P_vs_absdev": f(rho_abs),
        "mean_absdev_floor": f(abs_tbl.loc[FLOOR, "mean"]),
        "mean_absdev_nonfloor": f(df.loc[df.floor == 0, "absdev"].mean()),
        "mixed_fraction_of_multi_stage_units": f(pat.get("mixed", 0) / len(mstate)),
        "always_floor_fraction": f(pat.get("always_floor", 0) / len(mstate)),
        "P_floor_next_given_nonfloor": tr_all["P_floor_next_given_nonfloor"],
        "P_nonfloor_next_given_nonfloor": tr_all["P_nonfloor_next_given_nonfloor"],
        "base_rate_floor": base_floor,
        "persistence_ratio_changes": f(persistence_ratio),
        "strength_split_half_r": f(r_s[0]),
        "floor_frac_split_half_r": f(r_f_[0]),
        "floor_share_range_across_stages": f(fs_arr.max() - fs_arr.min()),
    },
    "evidence_for_carrying_information": [
        f"floor vs non-floor is a genuine but weak within-unit marker of strength: paired mean difference "
        f"{d_paired.mean():+.4f} strength units (floor higher in {k_sign}/{len(ub)} units, Wilcoxon p={wil.pvalue:.3g}), "
        f"within-unit AUC {auc_w:.4f}, Cohen's d {d_pooled / sd_w:+.4f}.",
        f"the tier is far from constant within a unit: only {pat.get('always_floor',0)}/{len(mstate)} multi-stage units "
        f"are always-floor and {pat.get('always_nonfloor',0)}/{len(mstate)} always-non-floor, so the non-floor levels "
        "are used (not a degenerate constant column).",
        f"state changes are rarer than a within-unit shuffle of the same labels predicts "
        f"({p_chg['obs']} observed vs {p_chg['perm_mean']} expected, ratio {persistence_ratio:.3f}), i.e. the tier "
        "has some within-unit temporal persistence.",
    ],
    "evidence_against_carrying_ordering_information": [
        f"the five non-floor levels are NOT ordered by strength: Spearman(P,S) among non-floor rows = {rho_nf:+.4f} "
        f"(p={p_nf:.3g}), within-unit {rho_nf_w:+.4f} (p={p_nf_w:.3g}); Kruskal-Wallis p={kw.pvalue:.3g}. "
        f"Their mean strength spans only {mean_nf_range:.4f} units vs SD(strength)={sd_S:.4f} and is non-monotone "
        "(p=0.04 has the highest mean, p=0.05 the lowest).",
        f"the full 6-level tier explains only {eta_w:.4f} of within-unit strength variance "
        f"(raw {eta_raw:.4f}), and {eta_nonfloor:.4f} among non-floor rows.",
        f"raw Spearman(P,S)={rho_raw:+.4f} is mostly between-unit structure: within-unit it drops to {rho_w:+.4f} "
        f"and raw AUC {auc_raw:.4f} falls to {auc_w:.4f} within unit.",
        f"|strength - unit mean| is essentially flat across levels (floor mean {abs_tbl.loc[FLOOR,'mean']:.4f} vs "
        f"non-floor {df.loc[df.floor==0,'absdev'].mean():.4f}); Spearman(P,|dev|)={rho_abs:+.4f}, Pearson={pear_abs:+.4f}.",
        f"the instrument's own threshold behaviour drifts by stage: floor share ranges {fs_arr.min():.4f} (stage {lo_st}) "
        f"to {fs_arr.max():.4f} (stage {hi_st}), range {fs_arr.max()-fs_arr.min():.4f}, while stage explains only "
        f"{r2_stage_floor:.4f} of floor variance.",
        f"test-retest reliability of the tier is low: split-half r={r_f_[0]:.4f} for floor fraction and {r_p_[0]:.4f} "
        f"for mean p, versus {r_s[0]:.4f} for strength.",
    ],
    "summary": ("Use the p column as a binary floor/non-floor evidence flag and nothing finer. Any analysis that "
                "treats the six levels as an ordered strength readout will be fitting noise above the floor; the "
                "within-unit floor/non-floor contrast, though statistically clear, is small "
                f"(~{d_paired.mean():.2f} strength units, d~{d_pooled / sd_w:.2f} within-unit SD)."),
    "conflicts_with_stated_facts": [
        "The frozen main-analysis artifact output/results/evidence_coupling.json reports "
        "always_floor_p=8 / always_nonfloor_p=22 / mixed=688 for multi-stage units. The correct counts "
        f"masking to observed stages are {n_af} / {n_an} / {n_mx}. The 8 is a NaN-pivot artefact "
        "(described in reconciliation_with_main_analysis); the stated 'fraction_mixed 0.9582' is therefore "
        f"also wrong (correct {n_mx / len(mrow):.4f}).",
        "The frozen input facts state '88% of rows at the floor'. The exact share is 0.8757 (5184/5920).",
    ],
}

with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2, allow_nan=False)

print(f"\nwrote {OUT_JSON}")