#!/usr/bin/env python3
"""Step 05 - What actually organises the strength values: pair identity vs stage.

Part A  Variance decomposition of `interaction_strength_like_value` into
        interaction-family, cell-pair, their combination, and residual, using
        the analysis unit as the observation.  Answers whether the ordered
        stage axis or the (interaction x cell-pair) identity carries the signal.

Part B  Permutation test on the pooled family-holdout MAE improvement from
        step 04, plus heterogeneity diagnostics.

Part C  Cell-pair leave-one-out holdout, as a second, non-nested holdout unit.

The holdout loop is vectorised: for each family the leave-one-family-out stage
offset is a group sum minus that family's own group sum, computed with matrix
arithmetic rather than a per-row Python loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
units = pd.read_csv(OUT / "analysis_table.tsv", sep="\t")
Y = "interaction_strength_like_value"
STAGES = np.arange(int(units.stage_index.max()) + 1)
res: dict[str, object] = {}

# ==================================================== Part A: decomposition
unit_tab = (
    units.groupby(["unit_id", "interaction_id", "cell_pair_id"], as_index=False)[Y]
    .mean()
    .rename(columns={Y: "unit_mean_strength"})
)
grand = unit_tab["unit_mean_strength"].mean()
ss_total = float(((unit_tab["unit_mean_strength"] - grand) ** 2).sum())

fam = unit_tab.groupby("interaction_id")["unit_mean_strength"].transform("mean")
cp = unit_tab.groupby("cell_pair_id")["unit_mean_strength"].transform("mean")
cell = unit_tab.groupby(["interaction_id", "cell_pair_id"])["unit_mean_strength"].transform("mean")

ss_fam = float(((fam - grand) ** 2).sum())
ss_cp = float(((cp - grand) ** 2).sum())
ss_cell = float(((cell - grand) ** 2).sum())

# NOTE: unit_id IS the (interaction_id x cell_pair_id) combination, so a
# groupby on that pair saturates the table by construction.  The meaningful
# decomposition is: family share, cell-pair-alone share, and the residual
# among cell pairs WITHIN a family (which is the incremental information
# carried by cell-pair identity beyond family identity).
res["partA_unit_level_variance"] = {
    "n_units": int(len(unit_tab)),
    "sd_unit_means": round(float(unit_tab["unit_mean_strength"].std(ddof=1)), 4),
    "ss_total": round(ss_total, 4),
    "share_explained_by_interaction_family": round(ss_fam / ss_total, 4),
    "share_explained_by_cell_pair_alone": round(ss_cp / ss_total, 4),
    "share_cellpair_within_family_residual": round((ss_total - ss_fam) / ss_total, 4),
    "n_interaction_x_cellpair_levels": int(
        unit_tab.groupby(["interaction_id", "cell_pair_id"]).ngroups
    ),
    "degenerate_note": (
        "a groupby on (interaction_id, cell_pair_id) equals unit_id and therefore "
        "explains 100% by construction; that figure is not reported as evidence"
    ),
}

y = units[Y]
um = units.groupby("unit_id")[Y].transform("mean")
gmean = float(y.mean())
ss_tot_rec = float(((y - gmean) ** 2).sum())
ss_unit = float(((um - gmean) ** 2).sum())
ss_resid_unit = float(((y - um) ** 2).sum())
stage_eff = (y - um).groupby(units.stage_index).transform("mean")
ss_stage_within = float((stage_eff ** 2).sum())
res["partA_record_level_partition"] = {
    "n_records": int(len(units)),
    "ss_total": round(ss_tot_rec, 4),
    "share_between_units": round(ss_unit / ss_tot_rec, 4),
    "share_within_units_total": round(ss_resid_unit / ss_tot_rec, 4),
    "share_stage_within_units": round(ss_stage_within / ss_tot_rec, 4),
    "share_stage_of_within_unit_variance": round(ss_stage_within / ss_resid_unit, 4),
    "note": "the stage axis organises only the within-unit slice, which is the smaller part",
}

# ============================================ Part B/C: vectorised holdout
multi = units.groupby("unit_id").size()
sub = units[units.unit_id.isin(set(multi[multi >= 2].index))].copy()
gsum = sub.groupby("unit_id")[Y].transform("sum")
gn = sub.groupby("unit_id")[Y].transform("size")
sub["unit_loo_mean"] = (gsum - sub[Y]) / (gn - 1)
sub["y_centered"] = sub[Y] - sub["unit_loo_mean"]

n_stage = len(STAGES)
fam_codes, fam_names = pd.factorize(sub.interaction_id)
cidx = sub.stage_index.to_numpy()
n_fam = len(fam_names)

# family x stage sums/counts of unit-centred values, and per-stage totals
grid_sum = np.zeros((n_fam, n_stage))
grid_cnt = np.zeros((n_fam, n_stage))
np.add.at(grid_sum, (fam_codes, cidx), sub.y_centered.to_numpy())
np.add.at(grid_cnt, (fam_codes, cidx), 1.0)
stage_sum = grid_sum.sum(axis=0)
stage_cnt = grid_cnt.sum(axis=0)

# leave-one-family-out stage offset for every row
num = stage_sum[cidx] - grid_sum[fam_codes, cidx]
den = stage_cnt[cidx] - grid_cnt[fam_codes, cidx]
delta = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

yv = sub[Y].to_numpy()
pred_null = sub.unit_loo_mean.to_numpy()
pred_trend = pred_null + delta

res["partB_pooled"] = {
    "heldout_records": int(len(sub)),
    "mae_null_unit_mean": round(float(np.abs(yv - pred_null).mean()), 5),
    "mae_trend": round(float(np.abs(yv - pred_trend).mean()), 5),
    "mae_improvement": round(float(np.abs(yv - pred_null).mean() - np.abs(yv - pred_trend).mean()), 5),
    "improvement_pct": round(
        100
        * (np.abs(yv - pred_null).mean() - np.abs(yv - pred_trend).mean())
        / np.abs(yv - pred_null).mean(),
        2,
    ),
}

# permutation control: shuffle stage labels within unit, redo the same machinery
rng = np.random.default_rng(20220920)


def shuffled_improvement(n_perm: int) -> np.ndarray:
    base_mae = float(np.abs(yv - pred_null).mean())
    out = np.empty(n_perm)
    for k in range(n_perm):
        perm_stage = sub.groupby("unit_id")["stage_index"].transform(
            lambda s: rng.permutation(s.to_numpy())
        ).to_numpy()
        gs = np.zeros((n_fam, n_stage))
        gc = np.zeros((n_fam, n_stage))
        np.add.at(gs, (fam_codes, perm_stage), sub.y_centered.to_numpy())
        np.add.at(gc, (fam_codes, perm_stage), 1.0)
        ss_, sc_ = gs.sum(axis=0), gc.sum(axis=0)
        n2 = ss_[perm_stage] - gs[fam_codes, perm_stage]
        d2 = sc_[perm_stage] - gc[fam_codes, perm_stage]
        off = np.divide(n2, d2, out=np.zeros_like(n2), where=d2 > 0)
        out[k] = base_mae - float(np.abs(yv - (pred_null + off)).mean())
    return out


perm = shuffled_improvement(200)
obs = float(np.abs(yv - pred_null).mean() - np.abs(yv - pred_trend).mean())
res["partB_stage_shuffle_control"] = {
    "n_perm": int(len(perm)),
    "mean_improvement_under_shuffled_stage": round(float(perm.mean()), 5),
    "sd_improvement": round(float(perm.std(ddof=1)), 5),
    "max_improvement_under_shuffled_stage": round(float(perm.max()), 5),
    "observed_improvement": round(obs, 5),
    "permutation_pvalue": float((perm >= obs).mean()),
    "exceeds_all_permutations": bool(obs > perm.max()),
}

# heterogeneity of transport across families
row_gain = np.abs(yv - pred_null) - np.abs(yv - pred_trend)
pf = pd.DataFrame({"family": sub.interaction_id.to_numpy(), "gain": row_gain})
pf = pf.groupby("family", as_index=False).agg(gain=("gain", "mean"), n=("gain", "size"))
fam_meta = units.groupby("interaction_id").agg(
    n_records=("unit_id", "size"), n_stages=("stage_index", "nunique")
)
pf = pf.merge(fam_meta, left_on="family", right_index=True, how="left")
res["partB_heterogeneity"] = {
    "n_families": int(len(pf)),
    "corr_family_size_vs_gain": round(float(stats.spearmanr(pf.n_records, pf.gain).statistic), 4),
    "corr_n_stages_vs_gain": round(float(stats.spearmanr(pf.n_stages, pf.gain).statistic), 4),
    "gain_median": round(float(pf.gain.median()), 5),
    "families_with_gain_gt_0": int((pf.gain > 0).sum()),
    "families_with_gain_lt_0": int((pf.gain < 0).sum()),
    "sign_test_pvalue": float(stats.binomtest(int((pf.gain > 0).sum()), len(pf), 0.5).pvalue),
    "top5_gain_families": pf.nlargest(5, "gain")[["family", "n_records", "gain"]]
    .round(4)
    .to_dict("records"),
    "bottom5_gain_families": pf.nsmallest(5, "gain")[["family", "n_records", "gain"]]
    .round(4)
    .to_dict("records"),
}
pf.to_csv(OUT / "family_holdout_annotated.tsv", sep="\t", index=False)

# ============================================ Part C: cell-pair holdout
cp_codes, cp_names = pd.factorize(sub.cell_pair_id)
n_cp = len(cp_names)
gs_cp = np.zeros((n_cp, n_stage))
gc_cp = np.zeros((n_cp, n_stage))
np.add.at(gs_cp, (cp_codes, cidx), sub.y_centered.to_numpy())
np.add.at(gc_cp, (cp_codes, cidx), 1.0)
ss_cp_, sc_cp_ = gs_cp.sum(axis=0), gc_cp.sum(axis=0)
n3 = ss_cp_[cidx] - gs_cp[cp_codes, cidx]
d3 = sc_cp_[cidx] - gc_cp[cp_codes, cidx]
delta_cp = np.divide(n3, d3, out=np.zeros_like(n3), where=d3 > 0)
mae_cp = float(np.abs(yv - (pred_null + delta_cp)).mean())
res["partC_cellpair_holdout"] = {
    "n_cell_pairs": int(n_cp),
    "mae_null_unit_mean": round(float(np.abs(yv - pred_null).mean()), 5),
    "mae_trend_cellpair_holdout": round(mae_cp, 5),
    "mae_improvement": round(float(np.abs(yv - pred_null).mean() - mae_cp), 5),
    "improvement_pct": round(
        100 * (np.abs(yv - pred_null).mean() - mae_cp) / np.abs(yv - pred_null).mean(), 2
    ),
}

(OUT / "structure_decomposition.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res, indent=2))