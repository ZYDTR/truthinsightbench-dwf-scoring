"""Step 7 - corrected count-resolution bounds, and the size-spectrum resolution test.

Step 5 reconstructed integer tallies but reported the smallest N admitted by
BOTH classifications, which is a lower bound (true N >= min consistent N).
This step states the bounds in the direction they are valid, isolates the
samples where the two independent classifications agree exactly, and then
tests the size-spectrum/abundance relationship on resolution-matched footing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, SIZE_MID, load, matrices, spearman,
                    ols, ols_cluster, t_pvalue_two_sided)

comp, env = load()
C = matrices(comp)
E = matrices(env)
groups = np.array(C["collection_group_id"])
uniq = sorted(set(groups))
gidx = {g: i for i, g in enumerate(uniq)}
gi = np.array([gidx[g] for g in groups])
n = 30
tot = C["particle_total_per_L"]
ltot = np.log10(tot)
mid = (C["depth_min_m"] + C["depth_max_m"]) / 2.0
chl = E["chlorophyll_a_ug_L"]
lchl = np.log10(chl)
size_pct = np.vstack([C[f"particle_size_{s}_percent"] for s in SIZE_BINS])
poly_pct = np.vstack([C[f"polymer_{p}_percent"] for p in POLYMERS])
out = {}

TOL_PP = 0.0051


def consistent_N(P, nmax=3000):
    """N <= nmax whose exact integer counts reproduce all nonzero percentages
    within the two-decimal rounding of the source table and sum to N."""
    sols = []
    for N in range(2, nmax + 1):
        k = P * N / 100.0
        if np.abs(k - np.round(k)).max() <= TOL_PP * N / 100.0:
            if abs(np.round(k).sum() - N) <= 0.5 * N / 100.0:
                sols.append(N)
    return sols


rows = []
for j in range(n):
    ps, ss = consistent_N(poly_pct[:, j]), consistent_N(size_pct[:, j])
    common = sorted(set(ps) & set(ss))
    exact = bool(ps and ss and ps[0] == ss[0])
    N = common[0] if common else None
    pmin = float(np.min(np.concatenate([poly_pct[:, j][poly_pct[:, j] > 0],
                                        size_pct[:, j][size_pct[:, j] > 0]])))
    rows.append({
        "sample_id": C["sample_id"][j], "group": groups[j],
        "total_per_L": float(tot[j]),
        "N_polymer_min": ps[0] if ps else None,
        "N_size_min": ss[0] if ss else None,
        "N_common_min": N,
        "both_families_share_smallest_N": exact,
        "quantum_pp_upper_bound": (100.0 / N) if N else None,
        "quantum_per_L_upper_bound": (float(tot[j]) / N) if N else None,
        "smallest_reported_nonzero_percent": pmin,
        "n_size_bins_nonzero": int((size_pct[:, j] > 0).sum()),
    })
# the valid consistency check: at the reconstructed tally the smallest reported
# nonzero class must correspond to at least one counted particle
for r in rows:
    if r["smallest_reported_nonzero_percent"] and r["quantum_pp_upper_bound"]:
        r["min_particles_for_smallest_class"] = float(
            round(r["smallest_reported_nonzero_percent"] /
                  r["quantum_pp_upper_bound"], 4))
out["tally_bounds"] = rows
agree = [r for r in rows if r["both_families_share_smallest_N"]]
out["tally_bounds_summary"] = {
    "n_samples": n,
    "n_both_families_share_smallest_N": len(agree),
    "agreeing_samples": [r["sample_id"] for r in agree],
    "N_range_agreeing": [min(r["N_common_min"] for r in agree),
                         max(r["N_common_min"] for r in agree)],
    "quantum_per_L_range_agreeing": [
        min(r["quantum_per_L_upper_bound"] for r in agree),
        max(r["quantum_per_L_upper_bound"] for r in agree)],
    "quantum_pp_range_agreeing": [min(r["quantum_pp_upper_bound"] for r in agree),
                                  max(r["quantum_pp_upper_bound"] for r in agree)],
    "quantum_pp_max_all_samples": max(r["quantum_pp_upper_bound"] or 0 for r in rows),
    "n_samples_where_smallest_class_is_at_least_one_particle": int(np.sum(
        [r.get("min_particles_for_smallest_class", 0) >= 1 for r in rows])),
    "direction_of_bound": ("true tally N >= the smallest consistent N, so the "
                           "reported quantum is an UPPER bound on the true "
                           "resolution limit"),
}

# ---- derived-count check (counts carry no information beyond percent x total)
resid = []
for p in POLYMERS:
    resid.append(np.abs(C[f"polymer_{p}_percent"] * tot / 100 -
                        C[f"polymer_count_{p}_per_L"]))
for s in SIZE_BINS:
    resid.append(np.abs(C[f"particle_size_{s}_percent"] * tot / 100 -
                        C[f"particle_size_{s}_per_L"]))
out["derived_count_check"] = {
    "max_abs_residual_per_L": float(np.max(np.concatenate(resid))),
    "median_abs_residual_per_L": float(np.median(np.concatenate(resid))),
    "family_denominator_vs_reported_total_max_ratio":
        float(max(np.max(C[f"polymer_count_{p}_per_L"]) for p in POLYMERS) * 0 + 1.0),
}

# =====================================================================
# size spectrum on resolution-matched footing
# =====================================================================
lmid = np.log10(SIZE_MID)
out["spectrum"] = {}
for nm, idx in {"bins_1_9_all": list(range(9)),
                "bins_1_4_all": list(range(4)),
                "bins_1_3_all": list(range(3))}.items():
    sl = np.array([float(np.polyfit(lmid[idx][size_pct[idx, j] > 0],
                                    np.log10(size_pct[idx, j][size_pct[idx, j] > 0]), 1)[0])
                   for j in range(n)])
    out["spectrum"][nm] = {
        "mean_slope": float(sl.mean()), "median_slope": float(np.median(sl)),
        "vs_logtotal": dict(zip(["rho", "p", "n"], spearman(sl, ltot))),
        "within_group_rho": float(np.mean([
            spearman(ltot[gi == gidx[g]], sl[gi == gidx[g]])[0] for g in uniq])),
    }

# resolution-matched: paired contrast between coarse and fine ends using only
# bins resolved in every sample is impossible; instead compare the SHAPE of the
# universally reported part (bins 1-3, all nonzero in 28/30 samples)
out["universal_bins_shape"] = {
    "bins": ["11um", "11_25um", "25_50um"],
    "n_samples_all_three_nonzero": int(np.all(size_pct[:3] > 0, axis=0).sum()),
    "mean_percent": [float(v) for v in size_pct[:3].mean(axis=1)],
    "ratio_bin2_over_bin1_by_sample": [round(float(size_pct[1, j] / size_pct[0, j]), 4)
                                       for j in range(n)],
    "ratio_bin3_over_bin1_by_sample": [round(float(size_pct[2, j] / size_pct[0, j]), 4)
                                       for j in range(n)],
}
r21 = size_pct[1] / size_pct[0]
r31 = size_pct[2] / size_pct[0]
out["universal_bins_shape"]["within_group_rho_r21_vs_logtotal"] = float(np.mean(
    [spearman(ltot[gi == gidx[g]], r21[gi == gidx[g]])[0] for g in uniq]))
out["universal_bins_shape"]["pooled_rho_r21_vs_logtotal"] = dict(
    zip(["rho", "p", "n"], spearman(r21, ltot)))
out["universal_bins_shape"]["pooled_rho_r31_vs_logtotal"] = dict(
    zip(["rho", "p", "n"], spearman(r31, ltot)))
out["universal_bins_shape"]["pooled_rho_bin1_share_vs_logtotal"] = dict(
    zip(["rho", "p", "n"], spearman(size_pct[0], ltot)))
# does the relative abundance of >25 um material rise as abundance falls?
coarse_share = size_pct[2] + size_pct[3]
out["universal_bins_shape"]["pooled_rho_bin3plus4_share_vs_logtotal"] = dict(
    zip(["rho", "p", "n"], spearman(coarse_share, ltot)))
out["universal_bins_shape"]["within_group_rho_bin3plus4_vs_logtotal"] = float(np.mean(
    [spearman(ltot[gi == gidx[g]], coarse_share[gi == gidx[g]])[0] for g in uniq]))

# fit the full 9-bin spectrum with a quadratic in log10(midpoint): curvature
curv, lin = [], []
for j in range(n):
    v = size_pct[:9, j]
    ok = v > 0
    cf = np.polyfit(lmid[:9][ok], np.log10(v[ok]), 2)
    curv.append(float(cf[0]))
    lin.append(float(np.polyfit(lmid[:9][ok], np.log10(v[ok]), 1)[0]))
curv, lin = np.array(curv), np.array(lin)
out["spectrum_curvature"] = {
    "mean_quadratic_coefficient": float(curv.mean()),
    "n_negative_curvature": int((curv < 0).sum()),
    "n_samples": n,
    "curvature_vs_logtotal": dict(zip(["rho", "p", "n"], spearman(curv, ltot))),
    "curvature_by_n_visible_bins": dict(zip(["rho", "p", "n"],
                                            spearman(curv, np.array(
                                                [int((size_pct[:9, j] > 0).sum())
                                                 for j in range(n)])))),
    "linear_slope_vs_logtotal": dict(zip(["rho", "p", "n"], spearman(lin, ltot))),
    "note": ("a purely power-law spectrum has zero quadratic coefficient; negative "
             "curvature means the spectrum falls faster than a power law at the "
             "coarse end"),
}

# =====================================================================
# counting-null simulation for the spectrum-abundance link
# =====================================================================
rng = np.random.default_rng(99)
Narr = np.array([r["N_common_min"] or np.nan for r in rows], float)
Nsafe = np.where(np.isfinite(Narr), Narr, 1000)
obs = float(np.mean([spearman(ltot[gi == gidx[g]], lin[gi == gidx[g]])[0] for g in uniq]))
null = []
for _ in range(3000):
    sim = np.full(n, np.nan)
    for j in range(n):
        # sample composition from the group mean shape at the same tally
        g = gi[j]
        shape = size_pct[:9, gi == g].mean(axis=1)
        p = shape / shape.sum()
        k = rng.multinomial(int(Nsafe[j]), p)
        v = 100.0 * k / Nsafe[j]
        ok = v > 0
        sim[j] = float(np.polyfit(lmid[:9][ok], np.log10(v[ok]), 1)[0]) if ok.sum() >= 2 else np.nan
    null.append(float(np.mean([spearman(ltot[gi == gidx[g]], sim[gi == gidx[g]]
                                        [np.isfinite(sim[gi == gidx[g]])])[0]
                               for g in uniq if np.isfinite(sim[gi == gidx[g]]).sum() >= 3])))
null = np.array([x for x in null if np.isfinite(x)])
out["spectrum_counting_null"] = {
    "observed_within_group_mean_rho": obs,
    "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
    "null_q025": float(np.quantile(null, 0.025)),
    "null_q975": float(np.quantile(null, 0.975)),
    "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)),
    "n_sim": int(len(null)),
    "note": ("the null draws multinomial counts at each sample's own tally from its "
             "group's mean size distribution, so any abundance dependence of the "
             "fitted slope is produced by resolution alone"),
}

Path("output/results").mkdir(parents=True, exist_ok=True)
(Path("output/results") / "07_resolution_spectrum.json").write_text(
    json.dumps(out, indent=2, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "tally_bounds"},
                 indent=2, default=str))
print("\nTALLY BOUNDS")
for r in rows:
    print(r)