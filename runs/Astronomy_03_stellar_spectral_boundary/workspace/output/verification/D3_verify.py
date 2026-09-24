#!/usr/bin/env python3
"""
Independent verification of FINDING D3.

CLAIM: The five columns of normalized_spectral_response.tsv relate to the
photon-flux grid via photon->energy conversion: response_level_k(wl) * wl is
proportional to F(T=5800 K, M_k, wl) with ordered composition indices
M = -0.90, -0.50, 0.00, +0.50, +0.90, single constant of proportionality
shared by all five levels and by the whole band ~330-850 nm.

Method (all from the raw TSVs; nothing under output/analysis or output/results
is imported or read):
  1. Build L_k(wl) = response_level_k(wl) * wl  (photon->energy conversion).
  2. Interpolate each grid spectrum onto the normalized file's wavelength axis
     (linear interp, no extrapolation) OR nearest-neighbour matching -- both used.
  3. For each level, scale-free best match over 105 (T,M) grid curves using a
     RELATIVE-DEVIATION statistic:  s(L, F) = median |1 - F/L|   (exact under
     any constant scaling, no correlation used anywhere).
  4. Calibration/null distribution: same statistic applied to all mismatched
     (level, curve) pairs -> shows whether the winner separates from the rest.
  5. Zero-free-parameter test: level_k(t)/level_1(t) vs F(T,M_k,t)/F(T,M_1,t),
     mean & max relative deviation, at all three temperatures (t = matched wl).
  6. Fitted constant c: level_k(t)*t / F(t) over the shared band; per level,
     per wavelength-block stability.
  7. Symmetric direction check: F/fit, c_hat = median(L_k/F), median(|1-F/c_hat|).

Writes /data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project/output/verification/D3_crossfile_relation.json
"""
import json
import hashlib
import platform
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project"
DATA = BASE + "/data"
OUT = BASE + "/output/verification"

# ----------------------------------------------------------------------------
# 0. Load raw data
# ----------------------------------------------------------------------------
grid = pd.read_csv(DATA + "/photon_flux_grid.tsv", sep="\t")
norm = pd.read_csv(DATA + "/normalized_spectral_response.tsv", sep="\t")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

file_hashes = {p: sha256(DATA + "/" + p) for p in
               ("photon_flux_grid.tsv", "normalized_spectral_response.tsv")}

wg = np.sort(grid["wavelength_nm"].unique())          # 176
wn = norm["wavelength_nm"].values.astype(float)       # 175
temps = np.array(sorted(grid["effective_temperature_K"].unique()))
mets = np.array(sorted(grid["metallicity_index"].unique()))

# dense per (T,M) spectra
scol = {}
for T in temps:
    for M in mets:
        d = grid[(grid.effective_temperature_K == T) &
                 (grid.metallicity_index == M)].sort_values("wavelength_nm")
        assert len(d) == len(wg)
        scol[(float(T), float(M))] = d["photon_flux_cm2"].values.astype(float)

# ----------------------------------------------------------------------------
# 1. L_k(wl) = photon->energy converted response
# ----------------------------------------------------------------------------
L = {k: norm["response_level_%d" % k].values.astype(float) * wn
     for k in range(1, 6)}

# ----------------------------------------------------------------------------
# 2. Match the two wavelength axes.
#    A) linear interpolation of the grid (176 pts) onto the norm axis (175 pts)
#    B) nearest-neighbour grid point per norm wavelength (no extrapolation)
# ----------------------------------------------------------------------------
def interp_spectrum(y, xs, xt):
    return np.interp(xt, xs, y)          # np.interp does not extrapolate

nn_idx = np.array([int(np.argmin(np.abs(wg - w))) for w in wn])
BAND = (wn >= 330.0) & (wn <= 850.0)   # first norm wl is 329.91 -> excluded
assert BAND.sum() >= 100, BAND.sum()

print("wavelength-axis match: nn index offsets per region")
d_idx = nn_idx - np.arange(len(wn))
print("  index offset min/max:", d_idx.min(), d_idx.max(),
      "| # rows with offset != 0:", int((d_idx != 0).sum()))
print("  band 330-850 nm rows:", int(BAND.sum()))

# ----------------------------------------------------------------------------
# 3+4. Scale-free search over all 105 (T,M) curves with a RELATIVE-DEVIATION
#      statistic (x2 scaling direction).  s = median |1 - F/L|  over the band.
#      No correlations used anywhere.
# ----------------------------------------------------------------------------
def stats_for_pair(Lk, F):
    """Lk, F = fluxes of the energy-converted level and the grid curve on the
    band; returns the two one-sided scale-free statistics."""
    r = Lk / F
    c_hat = np.median(r)                  # scale that minimises median rel dev
    eA = np.abs(1.0 - F / (Lk / c_hat))   # deviation of F from best-scaled L
    eB = np.abs(1.0 - Lk / (F * c_hat))   # deviation of L from best-scaled F
    return float(np.median(eA)), float(np.median(eB)), float(c_hat)

def spectrum_on_norm_axis(T, M, mode):
    y = scol[(float(T), float(M))]
    if mode == "interp":
        return interp_spectrum(y, wg, wn)
    return y[nn_idx]

modes = ("interp", "nn")
search = {m: {} for m in modes}
for mode in modes:
    for k in range(1, 6):
        Lk = L[k][BAND]
        rows = []
        for T in temps:
            for M in mets:
                F = spectrum_on_norm_axis(T, M, mode)[BAND]
                eA, eB, c = stats_for_pair(Lk, F)
                rows.append({"T": float(T), "M": float(M),
                             "med_rel_dev_Lk_vs_F": eA,
                             "med_rel_dev_F_vs_Lk": eB,
                             "c_hat_median_Lk_over_F": c})
        rows.sort(key=lambda r: r["med_rel_dev_Lk_vs_F"])
        search[mode][k] = rows

# calibration / null: all mismatched (level, curve) pairs
null_all = []
for mode in modes:
    for k in range(1, 6):
        null_all += [r["med_rel_dev_Lk_vs_F"] for r in search[mode][k][1:]]
null_all = np.array(null_all)

print("\n=== scale-free search, best (T,M) per level (median |1-F/(L/c)|, band 330-850) ===")
best_rows = []
for mode in modes:
    print("-- mode:", mode)
    for k in range(1, 6):
        r = search[mode][k][0]
        print("   L%d -> T=%.0f M=%+.2f   med_rel_dev=%.5f  (2nd: T=%.0f M=%+.2f "
              "%.5f)" % (k, r["T"], r["M"], r["med_rel_dev_Lk_vs_F"],
                         search[mode][k][1]["T"], search[mode][k][1]["M"],
                         search[mode][k][1]["med_rel_dev_Lk_vs_F"]))
        best_rows.append({"level": k, "mode": mode, **r})
print("null (all mismatched pairs, n=%d): median=%.5f  min=%.5f  p1=%.5f" %
      (len(null_all), np.median(null_all), null_all.min(),
       np.percentile(null_all, 1)))

# rank of the winner among all 105 for each level, and among *other* curves only
print("\nwinner separation (excluding the winner itself):")
for mode in modes:
    for k in range(1, 6):
        vals = np.array([r["med_rel_dev_Lk_vs_F"] for r in search[mode][k][1:]])
        w = search[mode][k][0]["med_rel_dev_Lk_vs_F"]
        print("   %s L%d: winner %.5f ; best-other %.5f (x%.2f) ; median-other %.4f (x%.0f)"
              % (mode, k, w, vals.min(), vals.min() / w, np.median(vals), np.median(vals) / w))

# ----------------------------------------------------------------------------
# Sanity check of the statistic itself on a KNOWN mismatch: same (T,M) curves
# are not available; use a permutation/negative control: for each level the
# best curve must beat curves that differ in T (all 35 M at other T's).
# ----------------------------------------------------------------------------
print("\nnegative control per level: best curve vs best curve at the OTHER temps (interp):")
for k in range(1, 6):
    best = search["interp"][k][0]
    Tb = best["T"]
    for T in temps:
        if T == Tb:
            continue
        vals = [r["med_rel_dev_Lk_vs_F"] for r in search["interp"][k] if r["T"] == T]
        print("   L%d best is T=%.0f; best at T=%.0f -> %.5f (x%.1f worse)"
              % (k, Tb, T, min(vals), min(vals) / best["med_rel_dev_Lk_vs_F"]))

# ----------------------------------------------------------------------------
# 5. Decisive zero-free-parameter test.
#    level_k / level_1  vs  F(T,M_k)/F(T,M_1)  on the nn-matched grid wavelengths.
#    No constant estimated; relative deviation = |ratio_num/ratio_den - 1|.
# ----------------------------------------------------------------------------
M_ASSIGN = {1: -0.90, 2: -0.50, 3: 0.00, 4: 0.50, 5: 0.90}
print("\n=== decisive test: level_k/level_1 vs F(T,M_k)/F(T,M_1), NO free parameter ===")
decisive = {}
for T in temps:
    per_level = {}
    for k in range(1, 6):
        r_num = L[k][BAND] / L[1][BAND]
        Fk = scol[(float(T), M_ASSIGN[k])][nn_idx][BAND]
        F1 = scol[(float(T), M_ASSIGN[1])][nn_idx][BAND]
        r_den = Fk / F1
        rel = np.abs(r_num / r_den - 1.0)
        per_level[k] = {"mean_rel_dev": float(rel.mean()),
                        "median_rel_dev": float(np.median(rel)),
                        "max_rel_dev": float(rel.max())}
        print("  T=%.0f L%d/L1 vs M=%+.2f/%.2f : mean=%.5f median=%.5f max=%.5f"
              % (T, k, M_ASSIGN[k], M_ASSIGN[1],
                 rel.mean(), np.median(rel), rel.max()))
    allrel = np.concatenate([np.abs((L[k][BAND] / L[1][BAND]) /
                                    (scol[(float(T), M_ASSIGN[k])][nn_idx][BAND] /
                                     scol[(float(T), M_ASSIGN[1])][nn_idx][BAND]) - 1.0)
                             for k in range(1, 6)])
    decisive[float(T)] = {"per_level": per_level,
                          "pooled_mean_rel_dev": float(allrel.mean()),
                          "pooled_median_rel_dev": float(np.median(allrel)),
                          "pooled_max_rel_dev": float(allrel.max())}
    print("  T=%.0f POOLED: mean=%.5f median=%.5f max=%.5f"
          % (T, allrel.mean(), np.median(allrel), allrel.max()))

# same test under the "interp" grid matching, for method-robustness
decisive_interp = {}
for T in temps:
    allrel = np.concatenate([np.abs((L[k][BAND] / L[1][BAND]) /
                                    (spectrum_on_norm_axis(T, M_ASSIGN[k], "interp")[BAND] /
                                     spectrum_on_norm_axis(T, M_ASSIGN[1], "interp")[BAND]) - 1.0)
                             for k in range(1, 6)])
    decisive_interp[float(T)] = {"pooled_mean_rel_dev": float(allrel.mean()),
                                 "pooled_median_rel_dev": float(np.median(allrel)),
                                 "pooled_max_rel_dev": float(allrel.max())}
print("  (interp-matched grid) pooled means:", {T: round(decisive_interp[T]["pooled_mean_rel_dev"], 5) for T in temps})

# ----------------------------------------------------------------------------
# 6. Fitted constant of proportionality (T=5800 K, assigned M's), band 330-850
# ----------------------------------------------------------------------------
T0 = 5800.0
print("\n=== constant of proportionality c_k = median(L_k*wl / F(T=5800,M_k)) ===")
consts = {}
for k in range(1, 6):
    F = scol[(T0, M_ASSIGN[k])][nn_idx][BAND]
    c = L[k][BAND] / F
    eA = np.abs(1.0 - F / (L[k][BAND] / np.median(c)))
    consts[k] = {"c_median": float(np.median(c)),
                 "c_mean": float(c.mean()),
                 "c_std": float(c.std()),
                 "c_p10": float(np.percentile(c, 10)),
                 "c_p90": float(np.percentile(c, 90)),
                 "med_rel_dev_Lk_vs_F": float(np.median(eA)),
                 "max_rel_dev_Lk_vs_F": float(eA.max())}
    print("  L%d: c=%.6e +- %.2e (spread %.1f%%), median|1-F/(L/c)|=%.4f max=%.4f"
          % (k, np.median(c), c.std(), 100 * c.std() / c.mean(), np.median(eA), eA.max()))
allc = np.concatenate([L[k][BAND] / scol[(T0, M_ASSIGN[k])][nn_idx][BAND] for k in range(1, 6)])
print("  pooled c = %.6e +- %.2e (spread %.2f%% across 5*121 points)"
      % (np.median(allc), allc.std(), 100 * allc.std() / allc.mean()))

# wavelength-block stability of c (contiguous-band holdout, per data guide warning)
edges = np.array([330, 380, 430, 480, 530, 580, 630, 680, 730, 780, 850])
blocks = []
for i in range(len(edges) - 1):
    m = BAND & (wn >= edges[i]) & (wn < edges[i + 1])
    row = {"lo_nm": float(edges[i]), "hi_nm": float(edges[i + 1]), "n": int(m.sum())}
    for k in range(1, 6):
        c = L[k][m] / scol[(T0, M_ASSIGN[k])][nn_idx][m]
        row["c_level_%d" % k] = float(np.median(c))
        row["w_med_rel_dev_level_%d" % k] = float(
            np.median(np.abs(1.0 - scol[(T0, M_ASSIGN[k])][nn_idx][m] / (L[k][m] / np.median(c)))))
    blocks.append(row)
print("\nper-block c stability (band holdout):")
for b in blocks:
    print("  %3.0f-%3.0f nm n=%3d  c_k/1e-13 = %s  worst-rel-dev = %.5f"
          % (b["lo_nm"], b["hi_nm"], b["n"],
             " ".join("%.4f" % (b["c_level_%d" % k] / 1e-13) for k in range(1, 6)),
             max(b["w_med_rel_dev_level_%d" % k] for k in range(1, 6))))

# ----------------------------------------------------------------------------
# 7. Symmetric direction: check c_hat = median(L/F) vs median(F/L) and also
#    verify the reverse statement  F <-> c*L  with a fold-crossed check:
#    fit c on even wavelengths, evaluate on odd wavelengths (band, interleaved).
# ----------------------------------------------------------------------------
print("\n=== fold check: fit c on even-index band points, evaluate on odd-index ===")
bandw = wn[BAND]
fold = {}
for k in range(1, 6):
    Feven = scol[(T0, M_ASSIGN[k])][nn_idx][BAND][::2]
    Leven = L[k][BAND][::2]
    c_fit = np.median(Leven / Feven)
    Fodd = scol[(T0, M_ASSIGN[k])][nn_idx][BAND][1::2]
    Lodd = L[k][BAND][1::2]
    rel = np.abs(1.0 - Fodd / (Lodd / c_fit))
    fold[k] = {"c_from_even": float(c_fit),
               "med_rel_dev_on_odd": float(np.median(rel)),
               "max_rel_dev_on_odd": float(rel.max())}
    print("  L%d: c_even=%.6e  odd median dev=%.5f max=%.5f"
          % (k, c_fit, np.median(rel), rel.max()))

# ----------------------------------------------------------------------------
# 8. What the runner-up NON-5800K candidate looks like for the pooled statement
#    (is 5800 uniquely selected by the shared-constant test?)
# ----------------------------------------------------------------------------
print("\n=== uniqueness of T: pooled single-constant fit over the five assigned M's, all T ===")
uniq = {}
for T in temps:
    allrel = np.concatenate([np.abs(1.0 - scol[(float(T), M_ASSIGN[k])][nn_idx][BAND] /
                                    (L[k][BAND] / np.median(L[k][BAND] / scol[(float(T), M_ASSIGN[k])][nn_idx][BAND])))
                             for k in range(1, 6)])
    uniq[float(T)] = {"pooled_med_rel_dev": float(np.median(allrel)),
                      "pooled_mean_rel_dev": float(allrel.mean()),
                      "pooled_max_rel_dev": float(allrel.max())}
    print("  T=%.0f pooled: median=%.5f mean=%.5f max=%.5f"
          % (T, np.median(allrel), allrel.mean(), allrel.max()))

# ----------------------------------------------------------------------------
# 9. Also test the claim's own 'roughly 330-850 nm' band versus full range and
#    versus narrower bands.
# ----------------------------------------------------------------------------
print("\n=== band sensitivity of the assignment fit (T=5800, assigned M, interp) ===")
band_tests = {}
for lo, hi in ((330, 850), (300, 850), (400, 850), (330, 700), (330, 600), (179.4, 850), (179.4, 329.9)):
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 5:
        continue
    allrel = np.concatenate([np.abs(1.0 - spectrum_on_norm_axis(T0, M_ASSIGN[k], "interp")[m] /
                                    (L[k][m] / np.median(L[k][m] / spectrum_on_norm_axis(T0, M_ASSIGN[k], "interp")[m])))
                             for k in range(1, 6)])
    band_tests["%g_%g" % (lo, hi)] = {"n": int(m.sum()),
                                      "median": float(np.median(allrel)),
                                      "mean": float(allrel.mean()),
                                      "max": float(allrel.max())}
    print("  %5.1f-%.1f nm n=%3d: median=%.5f mean=%.5f max=%.5f"
          % (lo, hi, m.sum(), np.median(allrel), allrel.mean(), allrel.max()))

# ----------------------------------------------------------------------------
# 10. TRANSFORM DISCRIMINATION: is the lambda-multiplication actually required?
#     Test T(x) in {level*wl (claim: photon->energy), level (identity),
#                   level/wl (energy->photon)} against F(T=5800, claimed M).
#     Only the claimed transform may pass; the others change the shape.
# ----------------------------------------------------------------------------
print("\n=== transform discrimination (T=5800, claimed M, interp, band 330-850) ===")
transforms = {"level*wl": lambda k: L[k],
              "level": lambda k: norm["response_level_%d" % k].values.astype(float),
              "level/wl": lambda k: norm["response_level_%d" % k].values.astype(float) / wn}
transform_fit = {}
for tname, fn in transforms.items():
    allrel = []
    for k in range(1, 6):
        x = fn(k)[BAND]
        F = spectrum_on_norm_axis(T0, M_ASSIGN[k], "interp")[BAND]
        allrel.append(np.abs(1.0 - F / (x / np.median(x / F))))
    allrel = np.concatenate(allrel)
    transform_fit[tname] = {"median": float(np.median(allrel)),
                            "mean": float(allrel.mean()),
                            "max": float(allrel.max())}
    print("  T(x)=%-9s median=%.5f mean=%.5f max=%.5f"
          % (tname, np.median(allrel), allrel.mean(), allrel.max()))

# and the same transform discrimination via the scale-free SEARCH (best (T,M)
# for every (transform, level) pair): does the claimed transform win globally?
print("\n=== global transform x (T,M) search: best achievable med_rel_dev per transform ===")
transform_search = {}
for tname, fn in transforms.items():
    per_level = []
    for k in range(1, 6):
        x = fn(k)[BAND]
        best = min((stats_for_pair(x, spectrum_on_norm_axis(T, M, "interp")[BAND])[0], float(T), float(M))
                   for T in temps for M in mets)
        per_level.append({"level": k, "best_med_rel_dev": best[0], "T": best[1], "M": best[2]})
    transform_search[tname] = per_level
    print("  %-9s best per level: %s" %
          (tname, " ".join("L%d:%.4f@T%.0f,M%+.2f" % (r["level"], r["best_med_rel_dev"], r["T"], r["M"])
                           for r in per_level)))

# ----------------------------------------------------------------------------
# 11. Is the claimed M SET distinguishable from neighbouring-grid alternatives?
#     Pooled single-constant fit at T=5800 with one component shifted by +-0.05
#     (the grid step), and the best possible per-level M at T=5800.
# ----------------------------------------------------------------------------
print("\n=== adjacent-metallicity discrimination at T=5800 (pooled over 5 levels, interp) ===")
def pooled_dev_T(T, mset):
    allrel = []
    for k, M in mset.items():
        x = L[k][BAND]
        F = spectrum_on_norm_axis(T, M, "interp")[BAND]
        allrel.append(np.abs(1.0 - F / (x / np.median(x / F))))
    return np.concatenate(allrel)

claimed_set = dict(M_ASSIGN)
base = pooled_dev_T(T0, claimed_set)
print("  claimed set        : median=%.5f mean=%.5f max=%.5f"
      % (np.median(base), base.mean(), base.max()))
adj = {}
for k in range(1, 6):
    for d in (-0.05, +0.05):
        mset = dict(claimed_set)
        newM = round(mset[k] + d, 2)
        if newM not in [round(m, 2) for m in mets]:
            continue
        mset[k] = newM
        v = pooled_dev_T(T0, mset)
        key = "L%d_shift_%+.2f" % (k, d)
        adj[key] = {"M": newM, "median": float(np.median(v)),
                    "mean": float(v.mean()), "ratio_to_claimed_median": float(np.median(v) / np.median(base))}
        print("  %-16s M=%+.2f: median=%.5f (x%.2f vs claimed)  mean=%.5f"
              % (key, newM, np.median(v), np.median(v) / np.median(base), v.mean()))

# best possible per-level M at T=5800 (already known from search); report the
# pooled dev for that set too
best_set = {k: search["nn"][k][0]["M"] for k in range(1, 6)}   # nn-mode per-level best M
bset = pooled_dev_T(T0, best_set)
print("  search-best set    : %s median=%.5f mean=%.5f"
      % (best_set, np.median(bset), bset.mean()))

# ----------------------------------------------------------------------------
# 12. Blue-edge boundary: per-wavelength median rel dev (claimed M, T=5800)
#     from 179 to 340 nm, to locate where the claimed band starts.
# ----------------------------------------------------------------------------
print("\n=== blue-edge boundary: per-wavelength |1-F/(L/c_k)| with c_k fitted on the 330-850 band ===")
c_band = {k: float(np.median(L[k][BAND] / spectrum_on_norm_axis(T0, M_ASSIGN[k], "interp")[BAND]))
          for k in range(1, 6)}
print("  band-fitted c_k:", {k: "%.6e" % v for k, v in c_band.items()})
edge_rows = []
for i in range(len(wn)):
    devs = [abs(1.0 - spectrum_on_norm_axis(T0, M_ASSIGN[k], "interp")[i] / (L[k][i] / c_band[k]))
            for k in range(1, 6)]
    edge_rows.append({"wl": float(wn[i]), "median_rel_dev": float(np.median(devs)),
                      "max_rel_dev": float(np.max(devs))})
for r in edge_rows:
    if r["wl"] <= 345 or r["wl"] >= 830:
        print("   %6.2f nm  median=%.5f  max=%.5f" % (r["wl"], r["median_rel_dev"], r["max_rel_dev"]))
good = [r["wl"] for r in edge_rows if r["median_rel_dev"] < 0.01]
first_good = min(good) if good else None
print("  first wavelength from the blue with median rel dev < 1%%:", first_good)
bad_red = [r["wl"] for r in edge_rows if r["wl"] > 330 and r["median_rel_dev"] >= 0.01]
print("  # band-red wavelengths (>330 nm) failing 1%%:", len(bad_red), bad_red[:10])
worst = sorted(edge_rows, key=lambda r: -r["median_rel_dev"])[:12]
print("  12 worst wavelengths:", [(round(r["wl"], 2), round(r["median_rel_dev"], 4)) for r in worst])

# ----------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------
result = {
    "finding_id": "D3",
    "verifier": "independent recomputation from raw TSVs (no output/analysis or output/results artefacts used)",
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__},
    "input_files": {
        "photon_flux_grid.tsv": file_hashes["photon_flux_grid.tsv"],
        "normalized_spectral_response.tsv": file_hashes["normalized_spectral_response.tsv"]},
    "wavelength_axis": {
        "grid_n": int(len(wg)), "norm_n": int(len(wn)),
        "nn_index_offset_regions": {"first_67_rows": int((d_idx[:67] != 0).sum()),
                                    "last_108_rows": int((d_idx[67:] != 0).sum())},
        "band_330_850_n": int(BAND.sum()),
        "max_abs_nn_wavelength_mismatch_nm": float(np.abs(wg[nn_idx] - wn).max())},
    "transform": "L_k(wl) = response_level_k(wl) * wl  (photon->energy; claimed by finding)",
    "statistic": "median over band of |1 - F/(L/c_hat)| with c_hat = median(L/F); scale-free, no correlation",
    "search_all_105_pairs": {"per_level_best": best_rows,
                             "null_mismatched_pairs": {"n": int(len(null_all)),
                                                       "median": float(np.median(null_all)),
                                                       "min": float(null_all.min()),
                                                       "p1": float(np.percentile(null_all, 1))}},
    "decisive_zero_free_parameter_test": decisive,
    "decisive_interp_matched": decisive_interp,
    "constant_of_proportionality_T5800_band330_850": {
        "per_level": consts,
        "c_final": float(np.median(allc)),
        "relative_spread_over_5x121_points": float(allc.std() / allc.mean()),
        "inter_level_c_ratios": {("L%d/L1" % k): float(consts[k]["c_median"] / consts[1]["c_median"]) for k in range(1, 6)}},
    "band_block_holdout_c_stability": blocks,
    "fold_check": fold,
    "temperature_uniqueness_pooled_single_constant": uniq,
    "band_sensitivity": band_tests,
    "transform_discrimination": {"fitted_with_claimed_M": transform_fit,
                                 "global_search_per_transform": transform_search},
    "adjacent_metallicity_discrimination": {
        "claimed_set": {str(k): v for k, v in claimed_set.items()},
        "claimed_pooled": {"median": float(np.median(base)), "mean": float(base.mean()),
                           "max": float(base.max())},
        "shifted_sets": adj,
        "search_best_set": {str(k): v for k, v in best_set.items()},
        "search_best_pooled": {"median": float(np.median(bset)), "mean": float(bset.mean()),
                               "max": float(bset.max())}},
    "blue_edge_boundary": {"per_wavelength_nm": edge_rows,
                           "first_wl_with_median_rel_dev_below_1pct_from_blue": first_good},
}

with open(OUT + "/D3_crossfile_relation.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote", OUT + "/D3_crossfile_relation.json")