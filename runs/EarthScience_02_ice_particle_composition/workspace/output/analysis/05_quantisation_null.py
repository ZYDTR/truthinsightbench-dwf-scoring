"""Step 5 - count-quantisation metrology and the counting-noise null model.

Part 1 reconstructs, for every sample, the smallest integer particle tally N
consistent with BOTH the polymer-class percentages and the size-class
percentages (two independent classifications of the same counted particles).
The composition tables are counts/N, so a class below 100/N percentage points
cannot appear at all.

Part 2 asks whether the two strongest composition findings found in step 3
(PE share rising with abundance; steepening size spectrum with abundance)
survive that resolution boundary, by
  (a) simulating the pure multinomial counting null at the recovered tallies, and
  (b) refitting the size spectrum on a bin range that is resolved in every sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, SIZE_MID, load, matrices, spearman,
                    ols_cluster)

comp, env = load()
C = matrices(comp)
E = matrices(env)
groups = np.array(C["collection_group_id"])
uniq = sorted(set(groups))
n = 30
tot = C["particle_total_per_L"]
ltot = np.log10(tot)
mid = (C["depth_min_m"] + C["depth_max_m"]) / 2.0
out = {}

poly_pct = np.vstack([C[f"polymer_{p}_percent"] for p in POLYMERS])
size_pct = np.vstack([C[f"particle_size_{s}_percent"] for s in SIZE_BINS])

TOL_PP = 0.0051           # source table carries 2 decimals


def consistent_N(P, nmax=8000):
    """All N <= nmax whose implied integer counts reproduce every nonzero
    percentage within the 2-decimal rounding, and whose counts sum to N."""
    sols = []
    for N in range(2, nmax + 1):
        k = P * N / 100.0
        tol = TOL_PP * N / 100.0
        if np.abs(k - np.round(k)).max() <= tol:
            if abs(np.round(k).sum() - N) <= 0.5 * N / 100.0:
                sols.append(N)
    return sols


def smallest_common_N(j, nmax=8000):
    ps = set(consistent_N(poly_pct[:, j], nmax))
    ss = set(consistent_N(size_pct[:, j], nmax))
    common = sorted(ps & ss)
    return (common[0] if common else None), (min(ps) if ps else None), (min(ss) if ss else None)


tally = []
for j in range(n):
    Nc, Np, Ns = smallest_common_N(j)
    tally.append({
        "sample_id": C["sample_id"][j], "group": groups[j],
        "total_per_L": float(tot[j]),
        "N_common_min": Nc, "N_polymer_min": Np, "N_size_min": Ns,
        "families_agree": (Np is not None and Ns is not None and Np == Ns),
        "quantum_percent_pp": (100.0 / Nc) if Nc else None,
        "quantum_per_L": (float(tot[j]) / Nc) if Nc else None,
    })
out["recovered_tallies"] = tally
res = [t for t in tally if t["N_common_min"]]
out["tally_summary"] = {
    "n_samples_reconstructed": len(res),
    "n_families_agree": int(sum(1 for t in tally if t["families_agree"])),
    "N_min": min(t["N_common_min"] for t in res),
    "N_max": max(t["N_common_min"] for t in res),
    "quantum_pp_min": min(t["quantum_percent_pp"] for t in res),
    "quantum_pp_median": float(np.median([t["quantum_percent_pp"] for t in res])),
    "quantum_pp_max": max(t["quantum_percent_pp"] for t in res),
    "quantum_per_L_min": min(t["quantum_per_L"] for t in res),
    "quantum_per_L_median": float(np.median([t["quantum_per_L"] for t in res])),
    "quantum_per_L_max": max(t["quantum_per_L"] for t in res),
}

N_of = {t["sample_id"]: t["N_common_min"] for t in tally}
Nvec = np.array([N_of.get(C["sample_id"][j]) or np.nan for j in range(n)], float)
qvec = np.array([t["quantum_percent_pp"] or np.nan for t in tally], float)
out["quantum_vs_total"] = {
    "spearman_quantumpp_vs_logtotal": dict(zip(["rho", "p", "n"],
                                               spearman(np.log10(qvec), ltot))),
    "spearman_quantum_perL_vs_logtotal": dict(zip(["rho", "p", "n"],
                                                  spearman(np.log10(qvec / 100 * tot), ltot))),
    "interpretation": ("quantum_per_L is the concentration of one counted particle; "
                       "it is nearly constant across samples, implying a roughly "
                       "constant sampled volume, so low-abundance samples are counted "
                       "with few particles."),
}

# ---------------------------------------------------------------------------
# (a) multinomial counting null for the PE-share vs abundance relationship
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2024)
pe = C["polymer_PE_percent"]
in_grp = {g: (groups == g) for g in uniq}


def within_group_stat(x, y, mask=None):
    """Mean within-group Spearman rho, optionally restricted to a sample mask."""
    if mask is None:
        mask = np.ones(len(x), bool)
    rhos = []
    for g in uniq:
        sel = mask & in_grp[g]
        if sel.sum() >= 3:
            rhos.append(spearman(np.asarray(x)[sel], np.asarray(y)[sel])[0])
    return float(np.mean(rhos))


obs_stat = within_group_stat(ltot, pe)

# underlying composition: group-level mean of the reconstructed counts
sim_stats = []
valid = np.isfinite(Nvec)
for _ in range(4000):
    sim_pe = np.full(n, np.nan)
    for g in uniq:
        m = in_grp[g] & valid
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        # group-level pooled counts (rounded to integers) as the generator
        pooled = np.array([np.round(poly_pct[p][idx].mean() * Nvec[idx].mean() / 100)
                           for p in range(len(POLYMERS))], float)
        pooled = np.maximum(pooled, 0)
        pvec = pooled / pooled.sum()
        for j in idx:
            k = rng.multinomial(int(Nvec[j]), pvec)
            sim_pe[j] = 100.0 * k[0] / Nvec[j]
    sim_stats.append(within_group_stat(ltot, sim_pe))
sim_stats = np.array(sim_stats)
out["counting_null_PE_share"] = {
    "observed_mean_within_group_rho(log total, PE%)": obs_stat,
    "null_mean": float(sim_stats.mean()),
    "null_sd": float(sim_stats.std(ddof=1)),
    "null_q025": float(np.quantile(sim_stats, 0.025)),
    "null_q975": float(np.quantile(sim_stats, 0.975)),
    "p_two_sided": float((np.sum(np.abs(sim_stats) >= abs(obs_stat)) + 1) /
                         (len(sim_stats) + 1)),
    "n_sim": int(len(sim_stats)),
}

# resolution-restricted refit: samples whose quantum is finer than 2 pp
fine = np.isfinite(qvec) & (qvec < 2.0)
out["counting_null_PE_share"]["restricted_to_fine_resolution"] = {
    "n_samples": int(fine.sum()),
    "samples": [C["sample_id"][j] for j in range(n) if fine[j]],
    "mean_within_group_rho": within_group_stat(ltot, pe, mask=fine),
}
yd = pe - np.array([pe[in_grp[g]].mean() for g in groups])
xd = ltot - np.array([ltot[in_grp[g]].mean() for g in groups])
fit = ols_cluster(yd, np.c_[np.ones(n), xd], groups, ["int", "ltot"])
out["counting_null_PE_share"]["fe_all"] = {
    "slope": float(fit["beta"][1]), "se": float(fit["se"][1]), "p": float(fit["p"][1])}
fi = fine
go = np.array([groups[j] for j in range(n) if fi[j]])
yd2 = pe[fi] - np.array([pe[fi & in_grp[g]].mean() for g in groups[fi]])
xd2 = ltot[fi] - np.array([ltot[fi & in_grp[g]].mean() for g in groups[fi]])
if len(set(go)) > 2:
    fit2 = ols_cluster(yd2, np.c_[np.ones(fi.sum()), xd2], go, ["int", "ltot"])
    out["counting_null_PE_share"]["restricted_to_fine_resolution"]["fe"] = {
        "slope": float(fit2["beta"][1]), "se": float(fit2["se"][1]),
        "p": float(fit2["p"][1]), "df_cluster": int(fit2["df_cluster"])}

# ---------------------------------------------------------------------------
# (b) size-spectrum slope: is the steepening real or a truncation artefact?
# ---------------------------------------------------------------------------
lmid_bins = np.log10(SIZE_MID)


def fit_slope(pcts, bin_idx):
    """log-log OLS slope over the given bins, zeros excluded."""
    v = pcts[bin_idx]
    ok = v > 0
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(lmid_bins[bin_idx][ok], np.log10(v[ok]), 1)[0])


ranges = {
    "bins_1_9_full": list(range(9)),
    "bins_1_4_common": list(range(4)),
    "bins_1_3_common": list(range(3)),
    "bins_2_5": [1, 2, 3, 4],
    "bins_3_7": [2, 3, 4, 5, 6],
}
spec = {}
for nm, idx in ranges.items():
    sl = np.array([fit_slope(size_pct[:, j], idx) for j in range(n)])
    resolved = np.array([int((size_pct[idx, j] > 0).sum()) for j in range(n)])
    spec[nm] = {
        "mean_slope": float(np.nanmean(sl)),
        "median_slope": float(np.nanmedian(sl)),
        "slope_vs_logtotal_spearman": dict(zip(["rho", "p", "n"], spearman(sl, ltot))),
        "within_group_mean_rho": within_group_stat(ltot, sl),
        "n_resolvable_bins_per_sample_min": int(resolved.min()),
        "n_resolvable_bins_per_sample_max": int(resolved.max()),
        "slopes": [None if not np.isfinite(x) else round(float(x), 3) for x in sl],
    }
out["size_spectrum_by_bin_range"] = spec

# how many bins are above the per-sample quantum in each sample?
visible = np.zeros((22, n), bool)
for j in range(n):
    visible[:, j] = size_pct[:, j] > 0
out["size_visibility"] = {
    "n_bins_visible_min": int(visible.sum(axis=0).min()),
    "n_bins_visible_max": int(visible.sum(axis=0).max()),
    "n_bins_visible_by_sample": [int(v) for v in visible.sum(axis=0)],
    "spearman_nvisible_vs_logtotal": dict(zip(["rho", "p", "n"],
                                              spearman(visible.sum(axis=0).astype(float), ltot))),
    "quantum_pp_by_sample": [None if not np.isfinite(x) else round(float(x), 3) for x in qvec],
    "smallest_nonzero_percent_by_sample": [float(size_pct[:, j][size_pct[:, j] > 0].min())
                                           for j in range(n)],
}

# per-sample: observed smallest nonzero percent versus the quantum bound
out["quantum_bound_check"] = {
    "n_samples_smallest_nonzero_ge_quantum": int(np.sum(
        [out["size_visibility"]["smallest_nonzero_percent_by_sample"][j] >=
         (qvec[j] - 1e-9) for j in range(n) if np.isfinite(qvec[j])])),
    "n_samples_with_quantum": int(np.isfinite(qvec).sum()),
    "note": ("a class can only be reported if at least one particle was counted, so "
             "the smallest nonzero percent in a sample must be >= 100/N."),
}

Path("output/results").mkdir(parents=True, exist_ok=True)
(Path("output/results") / "05_quantisation.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "recovered_tallies"},
                 indent=2, default=str))
print("\nTALLIES")
for t in tally:
    print(t)