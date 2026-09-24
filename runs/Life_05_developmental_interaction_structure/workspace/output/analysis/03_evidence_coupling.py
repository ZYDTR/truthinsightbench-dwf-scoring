#!/usr/bin/env python3
"""Step 03 - How association evidence (p-value) relates to strength, and its lability.

Two questions:

  Q1  Does a stronger `interaction_strength_like_value` carry stronger
      association evidence (`association_pvalue`)?
  Q2  Is the evidence verdict stable for the same analysis unit across
      ordered stages?

`association_pvalue` takes only six discrete levels in this frozen table
(0.0009 is a floor / most-significant report level), so it is treated as an
ordinal evidence tier, not a continuous statistic.  Q1 is evaluated at three
levels of adjustment, because family identity explains most of the raw
strength variance and would otherwise masquerade as substantive coupling.
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
P = "association_pvalue"
P_FLOOR = 0.0009

units["unit_mean"] = units.groupby("unit_id")[Y].transform("mean")
units["fam_mean"] = units.groupby("interaction_id")[Y].transform("mean")
units["y_within_unit"] = units[Y] - units["unit_mean"]
units["y_within_family"] = units[Y] - units["fam_mean"]
units["is_floor_p"] = units[P] == P_FLOOR

res: dict[str, object] = {}

# ---------------------------------------------------- Q1: coupling estimates
def spearman(a, b) -> dict[str, float]:
    r = stats.spearmanr(a, b)
    return {"rho": round(float(r.statistic), 4), "pvalue": float(r.pvalue)}


res["q1_p_vs_strength"] = {
    "raw": spearman(units[P], units[Y]),
    "within_unit_centered": spearman(units[P], units["y_within_unit"]),
    "within_family_centered": spearman(units[P], units["y_within_family"]),
}
ns = units[~units["is_floor_p"]]
res["q1_p_vs_strength"]["restricted_to_nonfloor_p"] = {
    **spearman(ns[P], ns[Y]),
    "n": int(len(ns)),
}
ns_w = ns.copy()
ns_w["y_within_unit"] = ns_w[Y] - ns_w.groupby("unit_id")[Y].transform("mean")
res["q1_p_vs_strength"]["restricted_to_nonfloor_within_unit"] = {
    **spearman(ns_w[P], ns_w["y_within_unit"]),
    "n": int(len(ns_w)),
}

res["mean_strength_by_p_level"] = {
    str(k): {
        "n": int(v),
        "raw_mean": round(float(units.loc[units[P] == k, Y].mean()), 4),
        "within_unit_mean": round(float(units.loc[units[P] == k, "y_within_unit"].mean()), 4),
    }
    for k, v in units[P].value_counts().sort_index().items()
}

# floor vs non-floor contrast, adjusted for unit level, with a unit-blocked
# permutation test (shuffle the p-label within each unit)
floor = units["is_floor_p"].to_numpy()
yw = units["y_within_unit"].to_numpy()
obs_diff = float(yw[floor].mean() - yw[~floor].mean())
rng = np.random.default_rng(20220920)
unit_codes = units["unit_id"].to_numpy()
order = np.argsort(unit_codes, kind="stable")
groups = np.split(order, np.cumsum(units.groupby("unit_id").size().to_numpy())[:-1])
null = np.empty(2000)
for i in range(2000):
    perm = floor.copy()
    for idx in groups:
        if len(idx) > 1:
            perm[idx] = rng.permutation(perm[idx])
    null[i] = yw[perm].mean() - yw[~perm].mean()
res["floor_p_contrast_within_unit"] = {
    "observed_diff_floor_minus_nonfloor": round(obs_diff, 4),
    "permutation_p_two_sided": float((np.abs(null) >= abs(obs_diff)).mean()),
    "n_perm": int(len(null)),
    "cohens_d_approx_sd_units": round(float(yw.std(ddof=1)), 4),
}

# ------------------------------------------------- Q2: evidence lability
pv = units.pivot_table(index="unit_id", columns="stage_index", values=P, aggfunc="min")
multi = pv[pv.notna().sum(axis=1) >= 2]
sig = multi == P_FLOOR
state_counts = {
    "always_floor_p": int(sig.all(axis=1).sum()),
    "always_nonfloor_p": int((~sig).all(axis=1).sum()),
    "mixed_between_stages": int((sig.any(axis=1) & ~sig.all(axis=1)).sum()),
}
res["q2_evidence_stability"] = {
    "n_units_multi_stage": int(len(multi)),
    **state_counts,
    "fraction_mixed": round(state_counts["mixed_between_stages"] / len(multi), 4),
}

# consecutive-stage transition matrix (ordered by stage)
trans = np.zeros((2, 2), dtype=int)  # [prev 0=nonfloor,1=floor, next ...]
for _, row in multi.iterrows():
    seq = row.dropna().sort_index()
    for a, b in zip(seq[:-1], seq[1:]):
        trans[int(a == P_FLOOR), int(b == P_FLOOR)] += 1
res["q2_transition_counts_nonfloor_then_floor"] = {
    "nonfloor->nonfloor": int(trans[0, 0]),
    "nonfloor->floor": int(trans[0, 1]),
    "floor->nonfloor": int(trans[1, 0]),
    "floor->floor": int(trans[1, 1]),
}
p_from_nonfloor = trans[0, 1] / max(trans[0].sum(), 1)
p_from_floor = trans[1, 1] / max(trans[1].sum(), 1)
res["q2_transition_probabilities"] = {
    "P(floor at next | nonfloor now)": round(float(p_from_nonfloor), 4),
    "P(floor at next | floor now)": round(float(p_from_floor), 4),
}

# strength is far more stable than the verdict for the SAME units: compare
# split-half unit means against split-half unit evidence verdicts
half = units.assign(h=np.where(units.stage_index < 9, "early", "late"))
ym = half.pivot_table(index="unit_id", columns="h", values=Y, aggfunc="mean")
both = ym.notna().all(axis=1)
res["q2_contrast_strength_vs_evidence_stability"] = {
    "n_units_both_halves": int(both.sum()),
    "split_half_r_strength": round(float(np.corrcoef(ym.loc[both, "early"], ym.loc[both, "late"])[0, 1]), 4),
}
ev = half.pivot_table(index="unit_id", columns="h", values=P, aggfunc=lambda s: (s == P_FLOOR).mean())
both_ev = ev.notna().all(axis=1)
st = ev.loc[both_ev].map(lambda v: v >= 0.5)

# Cohen's kappa for the early-vs-late evidence verdict
tab = np.zeros((2, 2), dtype=float)
for a, b in zip(st["early"].astype(int), st["late"].astype(int)):
    tab[a, b] += 1
n = tab.sum()
expected = float(((tab.sum(axis=0) / n) * (tab.sum(axis=1) / n)).sum())
observed = float(np.trace(tab) / n)
kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
res["q2_contrast_strength_vs_evidence_stability"].update(
    {
        "n_units_both_halves_evidence": int(both_ev.sum()),
        "verdict_agreement_rate_early_vs_late": round(observed, 4),
        "expected_agreement_by_chance": round(expected, 4),
        "cohens_kappa_early_vs_late": round(float(kappa), 4),
    }
)

# ------------------------------------------------------- p-value level design
res["p_level_shares"] = {
    str(float(k)): round(float(v), 4) for k, v in units[P].value_counts(normalize=True).sort_index().items()
}
res["strength_value_granularity"] = {
    "n_unique_strength_values": int(units[Y].nunique()),
    "n_rows": int(len(units)),
    "n_unique_p_levels": int(units[P].nunique()),
    "note": "association_pvalue is a six-level ordinal report tier, not a continuous statistic",
}

(OUT / "evidence_coupling.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(json.dumps(res, indent=2))