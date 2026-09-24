#!/usr/bin/env python3
"""Step 04 - Trajectory taxonomy, threshold sensitivity, and family-holdout
generalization of the stage trend.

Part A  Classify each analysis unit with >= 3 observed stages into a trajectory
        class (persistent / emerging / declining / transient sign-reversing /
        unresolved) using explicit thresholds, then re-run the taxonomy over a
        grid of thresholds to show which counts are threshold-stable.

Part B  Leave-one-family-out out-of-sample test.  For every observation
        (unit u, stage s) belonging to held-out family f, predict the value two
        ways:
          null   : the mean of u's OTHER observations (unit level only, no stage info)
          trend  : the null prediction plus delta[s], where delta[s] is the
                   stage offset estimated from all families EXCEPT f
        and compare held-out MAE / RMSE.  Because delta[s] excludes family f
        entirely, this measures whether the stage trend transports to unseen
        interaction families rather than memorising family identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
units = pd.read_csv(OUT / "analysis_table.tsv", sep="\t")
Y = "interaction_strength_like_value"
STAGES = np.arange(int(units.stage_index.max()) + 1)

res: dict[str, object] = {}

# =========================================================== Part A: taxonomy
MIN_STAGES = 3
series = {
    k: g.sort_values("stage_index")
    for k, g in units.groupby("unit_id")
    if g.stage_index.nunique() >= MIN_STAGES
}


def classify(g: pd.DataFrame, slope_t: float, flat_sd: float) -> str:
    yv = g[Y].to_numpy(dtype=float)
    sv = g.stage_index.to_numpy(dtype=float)
    slope = float(np.polyfit(sv, yv, 1)[0])
    if yv.min() < 0 < yv.max():
        return "transient_sign_reversing"
    if slope > slope_t:
        return "emerging"
    if slope < -slope_t:
        return "declining"
    if yv.std(ddof=1) <= flat_sd:
        return "persistent_level"
    return "unresolved_flat"


SLOPE_T, FLAT_SD = 0.15, 0.50
labels = {k: classify(g, SLOPE_T, FLAT_SD) for k, g in series.items()}
counts = pd.Series(labels).value_counts()
res["partA_taxonomy"] = {
    "n_units_taxonomised": int(len(labels)),
    "min_stages_required": MIN_STAGES,
    "slope_threshold": SLOPE_T,
    "flat_sd_threshold": FLAT_SD,
    "class_counts": {str(k): int(v) for k, v in counts.items()},
    "class_shares": {str(k): round(float(v / len(labels)), 4) for k, v in counts.items()},
}

# threshold sensitivity grid
grid = []
for st in (0.10, 0.15, 0.20, 0.30):
    for fs in (0.30, 0.50, 0.75):
        lab = {k: classify(g, st, fs) for k, g in series.items()}
        c = pd.Series(lab).value_counts()
        grid.append(
            {
                "slope_threshold": st,
                "flat_sd_threshold": fs,
                "transient_sign_reversing": int(c.get("transient_sign_reversing", 0)),
                "emerging": int(c.get("emerging", 0)),
                "declining": int(c.get("declining", 0)),
                "persistent_level": int(c.get("persistent_level", 0)),
                "unresolved_flat": int(c.get("unresolved_flat", 0)),
            }
        )
res["partA_threshold_sensitivity"] = grid

# where do the transient units sit in stage? (are reversals stage-clustered?)
trans_ids = [k for k, v in labels.items() if v == "transient_sign_reversing"]
trows = units[units.unit_id.isin(trans_ids)]
res["partA_transient_stage_profile"] = {
    "n_records": int(len(trows)),
    "stage_share_of_transient_vs_all": {
        str(int(s)): round(
            float((trows.stage_index == s).sum() / max((units.stage_index == s).sum(), 1)), 4
        )
        for s in STAGES
    },
}

# ================================================ Part B: family-holdout test
multi_units = units.groupby("unit_id").size()
eligible = set(multi_units[multi_units >= 2].index)
sub = units[units.unit_id.isin(eligible)].copy()

# unit mean excluding the row itself (leave-one-out within unit)
gsum = sub.groupby("unit_id")[Y].transform("sum")
gn = sub.groupby("unit_id")[Y].transform("size")
sub["unit_loo_mean"] = (gsum - sub[Y]) / (gn - 1)

# per-stage offset, centred within unit, then averaged over families
sub["y_centered"] = sub[Y] - sub["unit_loo_mean"]
fam_stage_centered = sub.groupby(["interaction_id", "stage_index"])["y_centered"].mean()

# delta[s] excluding held-out family f
fams = sorted(sub.interaction_id.unique())
tot = fam_stage_centered.groupby("stage_index").sum()
cnt = fam_stage_centered.groupby("stage_index").size()


def delta_excluding(f: str) -> dict[int, float]:
    out = {}
    fs = fam_stage_centered.loc[f] if f in fam_stage_centered.index.get_level_values(0) else None
    for s in STAGES:
        t = tot.get(s, 0.0)
        c = cnt.get(s, 0)
        if fs is not None and s in fs.index:
            t -= fs.loc[s]
            c -= 1
        out[int(s)] = float(t / c) if c > 0 else 0.0
    return out


rows = []
for f, g in sub.groupby("interaction_id"):
    d = delta_excluding(f)
    stage_names = pd.Index(sorted(d))
    off = g.stage_index.map(d)
    pred_null = g["unit_loo_mean"].to_numpy(dtype=float)
    pred_trend = pred_null + off.to_numpy(dtype=float)
    rows.append(
        pd.DataFrame(
            {
                "family": f,
                "y": g[Y].to_numpy(dtype=float),
                "pred_null": pred_null,
                "pred_trend": pred_trend,
                "stage_index": g.stage_index.to_numpy(),
                "unit_id": g.unit_id.to_numpy(),
            }
        )
    )
oos = pd.concat(rows, ignore_index=True)

# also a shuffled-stage null: same machinery, stage labels permuted within unit
rng = np.random.default_rng(20220920)
sh = oos.copy()
sh["y"] = np.nan
sh_rows = []
for u, g in sub.groupby("unit_id"):
    perm = rng.permutation(g.stage_index.to_numpy())
    gg = g.copy()
    gg["stage_index"] = perm
    sh_rows.append(gg)
shuffled = pd.concat(sh_rows)
sc = shuffled[Y] - shuffled.groupby("unit_id")["unit_loo_mean"].transform("mean")
sc_fs = shuffled.assign(_c=sc).groupby(["interaction_id", "stage_index"])["_c"].mean()
tot_s = sc_fs.groupby("stage_index").sum()
cnt_s = sc_fs.groupby("stage_index").size()
delta_s = {int(s): float(tot_s.get(s, 0.0) / cnt_s.get(s, 1)) for s in STAGES}
sh_pred = shuffled["unit_loo_mean"] + shuffled.stage_index.map(delta_s)


def errs(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    e = y - pred
    return {
        "mae": round(float(np.abs(e).mean()), 5),
        "rmse": round(float(np.sqrt((e ** 2).mean())), 5),
    }


base = errs(oos.pred_null.to_numpy(), oos.y.to_numpy())
trend = errs(oos.pred_trend.to_numpy(), oos.y.to_numpy())
shuf = errs(sh_pred.to_numpy(), shuffled[Y].to_numpy())

# paired improvement per held-out family
per_fam = []
for f, g in oos.groupby("family"):
    e0 = np.abs(g.y - g.pred_null)
    e1 = np.abs(g.y - g.pred_trend)
    per_fam.append({"family": f, "n": len(g), "mae_null": float(e0.mean()), "mae_trend": float(e1.mean())})
pf = pd.DataFrame(per_fam)
pf["improved"] = pf.mae_trend < pf.mae_null
res["partB_family_holdout"] = {
    "n_heldout_observations": int(len(oos)),
    "n_heldout_families": int(len(pf)),
    "n_units": int(oos.unit_id.nunique()),
    "null_unit_mean_only": base,
    "stage_trend_added": trend,
    "shuffled_stage_control": shuf,
    "mae_improvement_abs": round(base["mae"] - trend["mae"], 5),
    "mae_improvement_pct": round(100 * (base["mae"] - trend["mae"]) / base["mae"], 2),
    "families_improved": int(pf.improved.sum()),
    "families_improved_share": round(float(pf.improved.mean()), 4),
    "n_families_improved_significant": None,
}
# sign test over families
from scipy import stats as _st  # noqa: E402

res["partB_family_holdout"]["sign_test_families"] = {
    "n_improved": int(pf.improved.sum()),
    "n_worse": int((~pf.improved).sum()),
    "pvalue_two_sided": float(
        _st.binomtest(int(pf.improved.sum()), len(pf), 0.5).pvalue
    ),
}
res["partB_family_holdout"]["ribbon"] = {
    "per_family_mae_null_min": round(float(pf.mae_null.min()), 4),
    "per_family_mae_null_max": round(float(pf.mae_null.max()), 4),
    "per_family_mae_trend_min": round(float(pf.mae_trend.min()), 4),
    "per_family_mae_trend_max": round(float(pf.mae_trend.max()), 4),
}

pf.to_csv(OUT / "family_holdout_per_family.tsv", sep="\t", index=False)
delta_df = pd.DataFrame(
    {"stage_index": list(STAGES), "delta_within_unit_centered": [delta_excluding("__none__")[int(s)] for s in STAGES]}
)
delta_df.to_csv(OUT / "stage_delta_offsets.tsv", sep="\t", index=False)
res["partB_all_family_delta"] = {
    str(int(r.stage_index)): round(float(r.delta_within_unit_centered), 5)
    for r in delta_df.itertuples()
}

(OUT / "trajectories_holdout.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res, indent=2))