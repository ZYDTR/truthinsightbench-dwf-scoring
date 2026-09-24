"""Step 2 - broad exploratory screen (printed, not registered).

Screens the candidate relationships named in the research objective:
abundance vs depth, composition vs depth/abundance, size structure,
uncertainty scaling, environmental covariates.  Uses within-group depth
permutations so that the repeated-measures structure is respected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (POLYMERS, SIZE_BINS, SIZE_MID, load, matrices, spearman,
                    ols, ols_cluster)

comp, env = load()
C = matrices(comp)
E = matrices(env)
E["ice_regime"] = [r["ice_regime"] for r in env]
groups = np.array(C["collection_group_id"])
uniq = sorted(set(groups))
tot = C["particle_total_per_L"]
mid = (C["depth_min_m"] + C["depth_max_m"]) / 2.0
ltot = np.log10(tot)
thick = C["depth_max_m"] - C["depth_min_m"]

print("=" * 78)
print("A. TOTAL CONCENTRATION vs DEPTH")
print("=" * 78)
for g in uniq:
    m = groups == g
    rho, p, n = spearman(mid[m], ltot[m])
    rho_t, p_t, _ = spearman(thick[m], ltot[m])
    print(f"{g}: n={n} rho(log10 total, mid-depth)={rho:+.3f} p={p:.4f} | "
          f"rho vs thickness={rho_t:+.3f} | totals={[int(x) for x in tot[m]]}")
rho, p, n = spearman(mid, ltot)
print(f"POOLED (ignores group): rho={rho:+.3f} p={p:.4f} n={n}")
rho, p, n = spearman(mid, ltot)
print(f"POOLED raw total: rho={spearman(mid, tot)[0]:+.3f} p={spearman(mid,tot)[1]:.4f}")

# within-group permutation of depth (group structure preserved)
rng = np.random.default_rng(7)
def stat_depth_perm(rng):
    vals = []
    for g in uniq:
        m = groups == g
        x = ltot[m].copy()
        rng.shuffle(x)
        vals.append(spearman(mid[m], x)[0])
    return np.mean(vals)
obs_mean_rho = np.mean([spearman(mid[groups == g], ltot[groups == g])[0] for g in uniq])
null = np.array([stat_depth_perm(rng) for _ in range(20000)])
p_perm = (np.sum(np.abs(null) >= abs(obs_mean_rho)) + 1) / (len(null) + 1)
print(f"mean within-group rho = {obs_mean_rho:+.3f}, within-group depth-permutation "
      f"p = {p_perm:.4f}  (null 2.5/97.5% = {np.quantile(null,.025):+.3f}/"
      f"{np.quantile(null,.975):+.3f})")

print()
print("Relative depth (mid/max depth in group) vs log total")
rel = np.zeros(len(tot))
for g in uniq:
    m = groups == g
    rel[m] = mid[m] / mid[m].max()
rho, p, n = spearman(rel, ltot)
print(f"  pooled rho={rho:+.3f} p={p:.4f}")
for g in uniq:
    m = groups == g
    print(f"  {g}: rho={spearman(rel[m], ltot[m])[0]:+.3f} rel_depth={[round(x,2) for x in rel[m]]}")

print()
print("=" * 78)
print("B. COMPOSITION vs DEPTH AND ABUNDANCE")
print("=" * 78)
pe = C["polymer_PE_percent"]
van = C["polymer_varnish_percent"]
print("PE + varnish share of total polymer per sample:")
s = pe + van
print("  min %.2f max %.2f median %.2f" % (s.min(), s.max(), np.median(s)))
ilr = np.log((np.clip(pe, 1e-6, None)) / (np.clip(van, 1e-6, None)))
print()
print("PE/varnish log-ratio vs depth, within group:")
for g in uniq:
    m = groups == g
    rho, p, n = spearman(mid[m], ilr[m])
    print(f"  {g}: rho={rho:+.3f} p={p:.4f} n={n}")
print("  pooled:", "%.3f (p=%.4f)" % spearman(mid, ilr)[:2])

print()
print("PE percent vs log10 total:")
print("  pooled Spearman rho=%.3f p=%.4f" % spearman(pe, ltot)[:2])
for g in uniq:
    m = groups == g
    print(f"  {g}: rho={spearman(pe[m], ltot[m])[0]:+.3f} p={spearman(pe[m], ltot[m])[1]:.3f}")

print()
print("Size-bin count columns: last bins all zero?")
for sb in SIZE_BINS[-6:]:
    v = C[f"particle_size_{sb}_per_L"]
    print(f"  {sb}: nonzero={int((v > 0).sum())}/30 max={v.max():.2f}")

print()
print("=" * 78)
print("C. SIZE STRUCTURE")
print("=" * 78)
size_pct = np.vstack([C[f"particle_size_{s}_percent"] for s in SIZE_BINS])  # 22 x 30
print("bin | mean percent | nonzero samples | min nonzero percent")
for i, s in enumerate(SIZE_BINS):
    v = size_pct[i]
    nz = v[v > 0]
    print(f"  {s:>10s} mean={v.mean():7.3f} nonzero={len(nz):2d} "
          f"min_nonzero={nz.min() if len(nz) else float('nan'):.3f} max={v.max():.3f}")

ltot_small = np.log10(np.clip(size_pct[:9], 1e-6, None))
print()
print("log10(percent+eps) vs log10(bin midpoint) slope per sample (bins 1-9),")
print("using only nonzero bins:")
slopes = []
for j in range(30):
    v = size_pct[:9, j]
    ok = v > 0
    if ok.sum() >= 3:
        sl = np.polyfit(np.log10(SIZE_MID[:9][ok]), np.log10(v[ok]), 1)[0]
    else:
        sl = np.nan
    slopes.append(sl)
slopes = np.array(slopes)
print("  slope mean=%.3f median=%.3f min=%.3f max=%.3f nan=%d"
      % (np.nanmean(slopes), np.nanmedian(slopes), np.nanmin(slopes),
         np.nanmax(slopes), int(np.isnan(slopes).sum())))
print("  slope by group:")
for g in uniq:
    m = groups == g
    print(f"    {g}: {np.round(slopes[m], 2)}")

print()
print("=" * 78)
print("D. UNCERTAINTY SCALING")
print("=" * 78)
sd = E["particle_total_std_or_error_per_L"]
cv = sd / tot
print("CV = sd/total: min=%.3f median=%.3f max=%.3f" % (cv.min(), np.median(cv), cv.max()))
print("CV by group:")
for g in uniq:
    m = groups == g
    print(f"  {g}: {np.round(cv[m], 2)}")
lsd = np.log10(sd)
print("log-log slope of sd vs total (pooled OLS):",
      "%.3f" % ols(lsd, np.c_[np.ones(30), ltot], ["int", "ltot"])["beta"][1])
fit = ols_cluster(lsd, np.c_[np.ones(30), ltot], groups, ["int", "ltot"])
print("  cluster-robust: beta=%.3f se=%.3f p=%.3f df=%d"
      % (fit["beta"][1], fit["se"][1], fit["p"][1], fit["df_cluster"]))
print("Poisson reference slope = 0.5; multiplicative-noise reference = 1.0")
print("corr(CV, log total) =", "%.3f" % spearman(cv, ltot)[0],
      "p=%.3f" % spearman(cv, ltot)[1])

print()
print("=" * 78)
print("E. ENVIRONMENTAL COVARIATES")
print("=" * 78)
cov_names = ["temperature_C", "salinity_PSU", "chlorophyll_a_ug_L",
             "particulate_organic_carbon_mg_L", "particulate_organic_nitrogen_ug_L",
             "carbon_to_nitrogen_ratio", "latitude_deg"]
for nm in cov_names:
    v = E[nm]
    rho, p, n = spearman(v, ltot)
    print(f"{nm:38s} vs log10 total: rho={rho:+.3f} p={p:.4f} n={n}")
print()
print("within-group Spearman vs log total (only covariates with within-group variance):")
for nm in cov_names:
    v = E[nm]
    vals = []
    for g in uniq:
        m = groups == g
        if len(set(v[m][np.isfinite(v[m])])) > 1:
            vals.append(spearman(v[m], ltot[m])[0])
    if vals:
        print(f"  {nm:38s} mean rho={np.mean(vals):+.3f} over {len(vals)} groups {np.round(vals,2)}")
    else:
        print(f"  {nm:38s} no within-group variance")

print()
print("chlorophyll_a vs depth within group:")
for g in uniq:
    m = groups == g
    print(f"  {g}: rho(mid depth, chl)={spearman(mid[m], E['chlorophyll_a_ug_L'][m])[0]:+.3f}",
          "chl=", np.round(E['chlorophyll_a_ug_L'][m], 2))

print()
print("=" * 78)
print("F. RICHNESS / DIVERSITY OF POLYMER ASSEMBLAGE")
print("=" * 78)
polym = np.vstack([C[f"polymer_{p}_percent"] for p in POLYMERS])
rich = (polym > 0).sum(axis=0)
print("richness (nonzero polymer classes): min=%d max=%d" % (rich.min(), rich.max()))
for g in uniq:
    m = groups == g
    print(f"  {g}: richness={rich[m]}  rho(depth)={spearman(mid[m], rich[m])[0]:+.3f} "
          f"rho(logTot)={spearman(ltot[m], rich[m])[0]:+.3f}")
print("pooled richness vs log total rho=%.3f p=%.4f" % spearman(rich, ltot)[:2])
print("pooled richness vs depth rho=%.3f p=%.4f" % spearman(mid, rich)[:2])
# Shannon on proportions
print()
print("=" * 78)
print("G. BOTTOM-INTERVAL SIGNATURE (deepest interval per group)")
print("=" * 78)
bot = np.zeros(30, bool)
top = np.zeros(30, bool)
for g in uniq:
    m = np.where(groups == g)[0]
    bot[m[np.argmax(mid[m])]] = True
    top[m[np.argmin(mid[m])]] = True
for nm, v in [("log10 total", ltot), ("PE percent", pe), ("varnish percent", van),
              ("richness", rich.astype(float)),
              ("chlorophyll_a", E["chlorophyll_a_ug_L"]),
              ("C:N", E["carbon_to_nitrogen_ratio"]),
              ("salinity", E["salinity_PSU"]),
              ("POC", E["particulate_organic_carbon_mg_L"])]:
    print(f"  {nm:22s} bottom mean={np.nanmean(v[bot]):8.3f}  rest mean={np.nanmean(v[~bot]):8.3f}")
print("  bottom samples:", [C['sample_id'][i] for i in range(30) if bot[i]])