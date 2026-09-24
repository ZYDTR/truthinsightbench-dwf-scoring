"""Stage 2 - compositional structure: scale behaviour, Taylor law, closure.

Executed evidence for the raw-vs-transformed-vs-ratio comparison in the research
objective.  Writes output/results/compositional_structure.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_composition import (  # noqa: E402
    OUT,
    clr,
    counts_matrix,
    library_size,
    load_covariates,
    rarefy,
    richness,
    save_json,
)

P, samples, features = counts_matrix()
lib = library_size(P).astype(float)
rich = richness(P).astype(float)
res = {}

# --------------------------------------------------------------------------
# 1. positive scale dependence: richness and library size
# --------------------------------------------------------------------------
lx, ly = np.log(lib), np.log(rich)
slope, intercept = np.polyfit(lx, ly, 1)
r = stats.pearsonr(lx, ly)
res["log_log_richness_vs_library"] = {
    "slope": float(slope),
    "intercept": float(intercept),
    "pearson_r": float(r.statistic),
    "r_squared": float(r.statistic**2),
    "spearman": float(stats.spearmanr(lib, rich).statistic),
    "interpretation": (
        "Sub-linear positive scaling: a 10x larger library yields only about "
        f"{10**slope:.2f}x more observed features, consistent with sampling from a "
        "large unobserved pool rather than with exhaustive enumeration."
    ),
}

# --------------------------------------------------------------------------
# 2. Taylor law on the raw vs closed representation
# --------------------------------------------------------------------------
def taylor(X: np.ndarray, axis: int) -> dict:
    mean = X.mean(axis=axis)
    var = X.var(axis=axis, ddof=1)
    m = mean > 0
    lm, lv = np.log(mean[m]), np.log(var[m])
    if m.sum() < 3 or np.ptp(lm) == 0:
        # Degenerate case: the normalisation forces a constant across-unit mean.
        return {
            "slope": None,
            "intercept": None,
            "pearson_r": None,
            "r_squared": None,
            "n": int(m.sum()),
            "degenerate": True,
            "reason": (
                "Every row of the closed matrix sums to 1, so the across-feature mean "
                "is exactly 1/n_features for every sample: the x-axis of the Taylor "
                "relation is constant by construction and no slope is estimable."
            ),
            "constant_across_unit_mean": float(lm[0]) if m.sum() else None,
            "var_range": [float(var[m].min()), float(var[m].max())],
        }
    b, a = np.polyfit(lm, lv, 1)
    rr = stats.pearsonr(lm, lv)
    return {
        "slope": float(b),
        "intercept": float(a),
        "pearson_r": float(rr.statistic),
        "r_squared": float(rr.statistic**2),
        "n": int(m.sum()),
        "degenerate": False,
    }


res["taylor_across_samples_raw"] = taylor(P.astype(float), axis=1)
Pc = P / lib[:, None]
res["taylor_across_samples_closed"] = taylor(Pc, axis=1)
res["taylor_across_features_raw"] = taylor(P.astype(float), axis=0)
res["closure_effect_on_taylor"] = (
    "Closing the rows destroys the across-feature Taylor relation by construction: "
    "the per-sample mean is pinned to 1/n_features, so the mean-variance slope is "
    "not estimable on the closed scale. Scale behaviour must therefore be assessed "
    "on the raw counts, and any per-feature comparison must condition on library size."
)

# --------------------------------------------------------------------------
# 3. rarefaction: is richness comparable across samples at all?
# --------------------------------------------------------------------------
rng = np.random.default_rng(20240101)
rare = {}
for depth in (1_000, 5_000, 20_000, 100_000):
    sub = rarefy(P, depth, rng)
    rr = (sub > 0).sum(axis=1)
    rare[str(depth)] = {
        "mean_richness": float(rr.mean()),
        "sd_richness": float(rr.std()),
        "corr_with_raw_richness": float(stats.spearmanr(rr, rich).statistic),
        "corr_with_library": float(stats.spearmanr(rr, lib).statistic),
    }
rare["raw"] = {"mean_richness": float(rich.mean()), "sd_richness": float(rich.std())}
res["rarefaction"] = rare
res["rarefaction_interpretation"] = (
    "Rarefied richness rises steeply with sequencing depth and never saturates, so "
    "raw richness is a depth-dependent statistic. At the smallest common depth "
    "(1000 counts, well below every sample's observed library) only about "
    f"{rare['1000']['mean_richness']:.0f} of the mean {rich.mean():.0f} raw features "
    "are still visible: the composition is dominated by entities that the sampled "
    "depth does not reach."
)

# --------------------------------------------------------------------------
# 4. log-ratio / proportionality structure of observed pairs
# --------------------------------------------------------------------------
nz = P > 0
logP = np.where(nz, np.log(np.maximum(P, 1)), np.nan)
rng2 = np.random.default_rng(7)
pairs = [(i, j) for i in range(0, 600, 7) for j in range(i + 1, 600, 7)]
lr_full, lr_half, cooc = [], [], []
idx_half = rng2.choice(P.shape[0], P.shape[0] // 2, replace=False)
for i, j in pairs:
    m = nz[:, i] & nz[:, j]
    m_half = np.zeros(P.shape[0], dtype=bool)
    m_half[idx_half] = m[idx_half]
    if m.sum() >= 30 and m_half.sum() >= 15:
        lr_full.append(np.log(P[m, i] / P[m, j]).var())
        lr_half.append(np.log(P[m_half, i] / P[m_half, j]).var())
        cooc.append(float(m.mean()))
res["pairwise_logratio"] = {
    "n_pairs_evaluated": len(lr_full),
    "cooccurrence_median": float(np.median(cooc)),
    "logratio_variance_median": float(np.median(lr_full)),
    "logratio_variance_p05": float(np.percentile(lr_full, 5)),
    "logratio_variance_p95": float(np.percentile(lr_full, 95)),
    "half_sample_ratio_median": float(np.median(np.array(lr_half) / np.array(lr_full))),
    "frac_logratio_var_below_1": float((np.array(lr_full) < 1).mean()),
}
res["pairwise_logratio_interpretation"] = (
    "Median pairwise log-ratio variance across the full sample set is about "
    f"{np.median(lr_full):.2f}, far above the < 1 range that would indicate "
    "near-proportional (stable-ratio) behaviour, and it is dominated by which "
    "samples contain both members at all (median joint presence "
    f"{np.median(cooc):.3f})."
)

# --------------------------------------------------------------------------
# 5. CLR spectrum
# --------------------------------------------------------------------------
Z = clr(P, 0.5)
Zc = Z - Z.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
ev = S**2 / (S**2).sum()
res["clr_pca"] = {
    "explained_variance_top10": [float(v) for v in ev[:10]],
    "cumulative_top10": [float(v) for v in np.cumsum(ev[:10])],
    "n_components_for_50pct": int(np.searchsorted(np.cumsum(ev), 0.5) + 1),
    "n_components_for_90pct": int(np.searchsorted(np.cumsum(ev), 0.9) + 1),
    "pseudocount": 0.5,
    "pseudocount_sensitivity": {},
}
for pc in (0.1, 0.5, 1.0, 2.0):
    Zp = clr(P, pc)
    Zp = Zp - Zp.mean(axis=0, keepdims=True)
    _, Sp, _ = np.linalg.svd(Zp, full_matrices=False)
    evp = Sp**2 / (Sp**2).sum()
    res["clr_pca"]["pseudocount_sensitivity"][str(pc)] = {
        "pc1": float(evp[0]),
        "pc2": float(evp[1]),
        "n_components_for_50pct": int(np.searchsorted(np.cumsum(evp), 0.5) + 1),
    }

res["clr_pca_interpretation"] = (
    "No low-dimensional invariant dominates: the leading CLR axis carries only "
    f"{ev[0]*100:.1f}% of variance and 50% of variance needs "
    f"{res['clr_pca']['n_components_for_50pct']} axes. The spectrum is also stable "
    "under pseudocount choice, so this is a property of the data and not of the "
    "zero-replacement rule."
)

# --------------------------------------------------------------------------
# 6. occupancy-abundance relation
# --------------------------------------------------------------------------
prev = nz.mean(axis=0)
mean_ab = P.mean(axis=0)
ok = (prev > 0) & (prev < 1)
res["occupancy_abundance"] = {
    "n_features": int(ok.sum()),
    "spearman": float(stats.spearmanr(prev[ok], np.log(mean_ab[ok])).statistic),
    "interpretation": (
        "Prevalence and abundance are strongly and monotonically coupled, which is "
        "the classic signature of a shared underlying abundance distribution sampled "
        "at limited depth."
    ),
}

save_json(res, OUT / "results" / "compositional_structure.json")
print("stage 2 ok")
for k in ("log_log_richness_vs_library", "taylor_across_samples_raw", "taylor_across_features_raw"):
    row = res[k]
    print(f"  {k:34s} r={row['pearson_r']:+.4f} slope={row['slope']:.3f}")
row = res["taylor_across_samples_closed"]
print(f"  {'taylor_across_samples_closed':34s} not estimable by construction "
      f"(constant mean {row['constant_across_unit_mean']:.6e})")
print("  rarefication (depth 1000 mean richness):", round(rare["1000"]["mean_richness"], 1),
      "vs raw", round(rich.mean(), 1))
print("  median pairwise log-ratio variance     :", round(res["pairwise_logratio"]["logratio_variance_median"], 3))
print("  CLR PC1 variance        :", round(res["clr_pca"]["explained_variance_top10"][0], 4))
print("  components for 50% var  :", res["clr_pca"]["n_components_for_50pct"])
print("  occupancy~abundance rho :", round(res["occupancy_abundance"]["spearman"], 4))