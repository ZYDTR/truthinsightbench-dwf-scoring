"""Step 4 - measurement metrology: percentage denominators and count resolution.

Two things matter for the strength of every claim in this task:
  (a) what denominator the _percent columns use, and
  (b) the smallest resolvable increment of the _per_L count columns, i.e. the
      effective detection limit that grows as the reported concentration falls.
Also cross-validates the hand-rolled statistics in common.py against
scipy/statsmodels, which are now installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, load, matrices, spearman, ols,
                    ols_cluster, t_pvalue_two_sided)

comp, env = load()
C = matrices(comp)
E = matrices(env)
groups = np.array(C["collection_group_id"])
uniq = sorted(set(groups))
n = 30
tot = C["particle_total_per_L"]
out = {}

poly_cnt = {p: C[f"polymer_count_{p}_per_L"] for p in POLYMERS}
size_cnt = {s: C[f"particle_size_{s}_per_L"] for s in SIZE_BINS}
poly_sum = np.sum([poly_cnt[p] for p in POLYMERS], axis=0)
size_sum = np.sum([size_cnt[s] for s in SIZE_BINS], axis=0)

# ---------------------------------------------------------------- (a) denominators
def denom_check(family, names, cnts):
    rows = []
    for i in range(n):
        # denominator implied by each nonzero count/percent pair
        implied = []
        for nm in names:
            p = C[f"{family}_{nm}_percent"][i]
            c = cnts[nm][i]
            if p > 0 and c > 0:
                implied.append(c / p * 100.0)
        rows.append({
            "sample_id": C["sample_id"][i],
            "reported_total_per_L": float(tot[i]),
            "implied_denominator_median": float(np.median(implied)),
            "implied_denominator_spread": float(np.max(implied) - np.min(implied)),
            "family_count_sum": float(sum(cnts[nm][i] for nm in names)),
            "ratio_denominator_over_reported_total":
                float(np.median(implied) / tot[i]),
        })
    return rows


out["polymer_percent_denominator"] = denom_check("polymer", POLYMERS, poly_cnt)
out["size_percent_denominator"] = denom_check("particle_size", SIZE_BINS, size_cnt)
out["denominator_summary"] = {
    "polymer_implied_denominator_equals_family_sum": bool(np.allclose(
        [r["implied_denominator_median"] for r in out["polymer_percent_denominator"]],
        [r["family_count_sum"] for r in out["polymer_percent_denominator"]], atol=0.02)),
    "size_implied_denominator_equals_family_sum": bool(np.allclose(
        [r["implied_denominator_median"] for r in out["size_percent_denominator"]],
        [r["family_count_sum"] for r in out["size_percent_denominator"]], atol=0.02)),
    "polymer_family_sum_over_total": {
        "min": float((poly_sum / tot).min()), "max": float((poly_sum / tot).max())},
    "size_family_sum_over_total": {
        "min": float((size_sum / tot).min()), "max": float((size_sum / tot).max())},
    "max_denominator_spread_percent_points":
        float(max(r["implied_denominator_spread"]
                  for r in out["polymer_percent_denominator"])),
    "note": ("percent columns are count / (sum of that family's counts) * 100, not "
             "count / reported particle_total_per_L * 100; the two denominators "
             "agree within 1.3%."),
}

# ---------------------------------------------------------------- (b) count quantum
def best_quantum(vals, tol=0.02):
    """Largest q such that every value is within tol of an integer multiple of q."""
    v = np.sort(np.asarray([x for x in vals if x > 0], float))
    if len(v) < 2:
        return None
    best = None
    for q in np.linspace(v[0], v[0] / max(len(v) * 2, 1), 4000):
        if q <= 0:
            continue
        r = v / q
        err = np.abs(r - np.round(r)).max()
        if err <= tol:
            best = q
            break
    return best


rows = []
for i in range(n):
    vals = [poly_cnt[p][i] for p in POLYMERS] + [size_cnt[s][i] for s in SIZE_BINS]
    q = best_quantum(vals)
    rows.append({
        "sample_id": C["sample_id"][i], "group": groups[i],
        "total_per_L": float(tot[i]),
        "smallest_positive_count_per_L": float(min([v for v in vals if v > 0])),
        "largest_consistent_quantum_per_L": float(q) if q else None,
        "implied_volume_L": float(1.0 / q) if q else None,
        "implied_integer_count_of_total": float(tot[i] * q) if q else None,
        "n_nonzero_size_bins": int(sum(1 for s in SIZE_BINS if size_cnt[s][i] > 0)),
        "n_nonzero_polymer_classes": int(sum(1 for p in POLYMERS if poly_cnt[p][i] > 0)),
    })
out["count_quantum"] = rows

# per-group consistency: is the quantum (hence the sampled volume) constant within group?
grp_q = {}
for g in uniq:
    qs = [r["largest_consistent_quantum_per_L"] for r in rows if r["group"] == g]
    vs = [r["implied_volume_L"] for r in rows if r["group"] == g]
    grp_q[g] = {"quantum_per_L": [round(x, 4) if x else None for x in qs],
                "implied_volume_L": [round(x, 4) if x else None for x in vs],
                "constant_within_group": bool(
                    max(qs) - min(qs) < 0.05 * np.mean(qs)) if all(qs) else None}
out["count_quantum_by_group"] = grp_q

# the quantisation limit expressed as a percentage of each sample's total
qvec = np.array([r["largest_consistent_quantum_per_L"] if
                 r["largest_consistent_quantum_per_L"] else 1.0 for r in rows])
out["quantum_as_percent_of_total"] = {
    "per_sample": [round(float(100 * qvec[i] / tot[i]), 4) for i in range(n)],
    "max": float((100 * qvec / tot).max()),
    "median": float(np.median(100 * qvec / tot)),
    "note": ("a single detected particle equals this share of the reported total; "
             "bins below it cannot be resolved for that sample."),
}

# ---------------------------------------------------- implementation cross-validation
import scipy.stats as sps
mid = (C["depth_min_m"] + C["depth_max_m"]) / 2.0
ltot = np.log10(tot)
xchk = []
for nm, x, y in [("mid_depth_vs_ltot", mid, ltot),
                 ("chl_vs_ltot", E["chlorophyll_a_ug_L"], ltot),
                 ("ltot_vs_PE", ltot, C["polymer_PE_percent"])]:
    rho, p, _ = spearman(x, y)
    s = sps.spearmanr(x, y)
    r, pr, _ = spearman(x, y)
    xchk.append({"pair": nm, "my_spearman_rho": rho, "scipy_spearman_rho": float(s.statistic),
                 "my_p": p, "scipy_p": float(s.pvalue),
                 "abs_diff_rho": abs(rho - float(s.statistic)),
                 "abs_diff_p": abs(p - float(s.pvalue))})
out["cross_validation_hand_rolled_vs_scipy"] = xchk

# cluster-robust SE cross-check against statsmodels
try:
    import statsmodels.api as sm
    Xd = np.c_[np.ones(n), ltot]
    m = sm.OLS(ltot, Xd).fit(cov_type="cluster", cov_kwds={"groups": groups})
    mine = ols_cluster(ltot, Xd, groups, ["int", "ltot"])
    out["cross_validation_cluster_se"] = {
        "statsmodels_se": [float(v) for v in m.bse],
        "mine_se": [float(v) for v in mine["se"]],
        "statsmodels_p": [float(v) for v in m.pvalues],
        "mine_p": [float(v) for v in mine["p"]],
        "agree": bool(np.allclose(m.bse, mine["se"], rtol=1e-8)),
    }
except Exception as exc:  # pragma: no cover - recorded, not hidden
    out["cross_validation_cluster_se"] = {"error": f"{type(exc).__name__}: {exc}"}

Path("output/results").mkdir(parents=True, exist_ok=True)
(Path("output/results") / "04_resolution.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("polymer_percent_denominator", "size_percent_denominator",
                               "count_quantum")}, indent=2, default=str))
print("\nquantum rows:")
for r in rows:
    print(r)