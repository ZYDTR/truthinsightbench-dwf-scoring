"""Adversarial verification of CLAIM F5 (applicability boundary: size-spectrum
steepening at low abundance is a resolution artefact).

Written from scratch; only scipy/numpy used.  All outputs go to
output/verify_F5/verify_F5.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify_F5"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd

comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")

SIZE_LABELS = [
    "11um", "11_25um", "25_50um", "50_75um", "75_100um", "100_125um",
    "125_150um", "150_175um", "175_200um", "200_225um", "225_250um",
    "250_275um", "275_300um", "300_325um", "325_350um", "350_375um",
    "375_400um", "400_425um", "425_450um", "450_475um", "475_500um",
    "gt_500um",
]
MIDS = np.array([11, 18, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5,
                 237.5, 262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5,
                 462.5, 487.5, 600.0])

P = np.vstack([comp[f"particle_size_{s}_percent"].values for s in SIZE_LABELS]).T.astype(float)  # 30x22
C = np.vstack([comp[f"particle_size_{s}_per_L"].values for s in SIZE_LABELS]).T.astype(float)
tot = comp["particle_total_per_L"].values.astype(float)
ltot = np.log10(tot)
groups = comp["collection_group_id"].values
sids = comp["sample_id"].values
uniq = sorted(set(groups))
N = len(tot)
lmid = np.log10(MIDS)


def fit_slope(pcts, idx):
    """OLS slope of log10(percent) on log10(midpoint), zeros dropped (claim's est)."""
    v = np.asarray(pcts)[idx]
    ok = v > 0
    if ok.sum() < 2:
        return np.nan
    x = lmid[idx][ok]
    y = np.log10(v[ok])
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    if den == 0:
        return np.nan
    return float(((x - xm) * (y - ym)).sum() / den)


def n_nonzero(pcts, idx):
    return int((np.asarray(pcts)[idx] > 0).sum())


def within_group_mean_rho(x, y, mask=None):
    if mask is None:
        mask = np.ones(N, bool)
    rs = []
    for g in uniq:
        sel = mask & (groups == g)
        if sel.sum() >= 3:
            r, _ = stats.spearmanr(x[sel], y[sel])
            rs.append(r)
    return float(np.mean(rs)), len(rs)


def rho_p(x, y):
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


RANGES = {
    "bins_1_9": list(range(9)),
    "bins_1_4": list(range(4)),
    "bins_1_3": list(range(3)),
    "bins_2_5": [1, 2, 3, 4],
    "bins_3_7": [2, 3, 4, 5, 6],
    "bins_1_6_extra": list(range(6)),
    "bins_1_2_extra": [0, 1],
}

# ---------------------------------------------------------------- (a) table
slopes = {}
table = {}
for nm, idx in RANGES.items():
    sl = np.array([fit_slope(P[j], idx) for j in range(N)])
    slopes[nm] = sl
    nnz = np.array([n_nonzero(P[j], idx) for j in range(N)])
    r, p = rho_p(ltot, sl)
    wg, ng = within_group_mean_rho(ltot, sl)
    # cluster-robust slope of slope ~ log10 total (within-group demeaned, 5 clusters)
    ok = np.isfinite(sl)
    y = sl[ok]
    z = ltot[ok] - np.array([ltot[ok & (groups == g)].mean() for g in groups[ok]])
    yz = y - np.array([y[(groups[ok] == g)].mean() for g in groups[ok]])
    X = np.c_[np.ones(ok.sum()), z]
    b, *_ = np.linalg.lstsq(X, yz, rcond=None)
    e = yz - X @ b
    # cluster-robust (CR1) sandwich
    meat = np.zeros((2, 2))
    gv = groups[ok]
    for g in set(gv):
        sel = gv == g
        Xg, eg = X[sel], e[sel]
        meat += (Xg.T * eg) @ (Xg.T * eg).T
    XtXi = np.linalg.inv(X.T @ X)
    dfc = len(set(gv)) / (len(set(gv)) - 1) * (ok.sum() - 1) / max(ok.sum() - 2, 1)
    V = XtXi @ meat @ XtXi * dfc
    se = float(np.sqrt(V[1, 1]))
    tstat = float(b[1] / se) if se > 0 else np.nan
    pcl = float(2 * stats.t.sf(abs(tstat), len(set(gv)) - 1)) if np.isfinite(tstat) else np.nan
    ranks = stats.rankdata(sl)
    table[nm] = {
        "bins": [SIZE_LABELS[i] for i in idx],
        "mean_slope": float(np.nanmean(sl)),
        "median_slope": float(np.nanmedian(sl)),
        "sd_slope": float(np.nanstd(sl, ddof=1)),
        "n_defined": int(ok.sum()),
        "spearman_rho_vs_log10total": r,
        "spearman_p_vs_log10total": p,
        "pearson_r_vs_log10total": float(np.corrcoef(ltot[ok], sl[ok])[0, 1]),
        "within_group_mean_rho": wg,
        "n_groups_used": ng,
        "FE_cluster_slope_per_decade": float(b[1]),
        "FE_cluster_se": se,
        "FE_cluster_p": pcl,
        "n_nonzero_min": int(nnz.min()),
        "n_samples_with_all_bins_nonzero": int((nnz == len(idx)).sum()),
        "n_samples_with_ge3_nonzero": int((nnz >= 3).sum()),
        "slopes": [None if not np.isfinite(v) else round(float(v), 3) for v in sl],
        "n_nonzero_by_sample": nnz.tolist(),
    }

# sensitivity of the bins_1_3 result to the single 2-point sample (sample_024)
sl3 = slopes["bins_1_3"]
nnz3 = np.array([n_nonzero(P[j], range(3)) for j in range(N)])
keep = nnz3 == 3
r_all, p_all = rho_p(ltot, sl3)
r_k, p_k = rho_p(ltot[keep], sl3[keep])
# zero-replacement variants: replace bin3 zero with 1 particle (100/N) and with 0.5 pp
NZ = np.array([64, 139, 388, 219, 620, 1286, 1446, 2166, 2614, 2963])  # recovered tallies where known
# recovered tallies from the prior pipeline (05/07 results) keyed by sample
tally_map = {
    "sample_001": 388, "sample_011": 64, "sample_012": 139, "sample_013": 1286,
    "sample_014": 219, "sample_016": 620, "sample_020": 90, "sample_023": 234,
    "sample_024": 64, "sample_026": 165, "sample_028": 289, "sample_030": 1123,
    "sample_018": 144, "sample_019": 7038, "sample_022": 423,
}
def slope3_with_replacement(mode):
    out = np.full(N, np.nan)
    for j in range(N):
        v = P[j, :3].copy()
        if v.min() <= 0:
            if mode == "one_particle":
                # need N; fall back to quantum implied by smallest nonzero percent
                q = v[v > 0].min()
                v[v <= 0] = q
            elif mode == "half_min":
                v[v <= 0] = 0.5 * v[v > 0].min()
            elif mode == "reported_zero_as_is":
                continue
        out[j] = fit_slope(v, range(3))
    return out

repl = {}
for mode in ["one_particle", "half_min"]:
    s = slope3_with_replacement(mode)
    ok = np.isfinite(s)
    rr, pp = rho_p(ltot[ok], s[ok])
    wg, _ = within_group_mean_rho(ltot, s, mask=ok)
    repl[mode] = {"mean_slope": float(np.nanmean(s)),
                  "spearman_rho_vs_log10total": rr, "p": pp,
                  "within_group_mean_rho": wg, "n": int(ok.sum())}

table["bins_1_3_sensitivity"] = {
    "n_samples_with_3_nonzero_bins": int(keep.sum()),
    "rho_excluding_sample_024_2point_fit": r_k,
    "p_excluding_sample_024": p_k,
    "rho_zero_replacement_one_particle_quantum": repl["one_particle"],
    "rho_zero_replacement_half_of_smallest_nonzero": repl["half_min"],
    "note": ("the only zero inside bins 1-3 is sample_024 (total 33/L, the smallest "
             "sample); its 3-bin 'slope' is a 2-point line"),
}

# ---------------------------------------------------------------- (d) binary pattern
nnz9 = np.array([n_nonzero(P[j], range(9)) for j in range(N)])
r_nnz_ltot, p_nnz_ltot = rho_p(ltot, nnz9.astype(float))
r_nnz_sl, p_nnz_sl = rho_p(nnz9.astype(float), slopes["bins_1_9"])
# multiple regression: 9-bin slope ~ n_nonzero + log10 total (within-group demeaned, CR1)
def fe_reg(y, Xcols, names):
    ok = np.isfinite(y)
    Y = y[ok]
    Xs = [c[ok] for c in Xcols]
    zg = groups[ok]
    Z = np.column_stack([c - np.array([c[zg == g].mean() for g in zg]) for c in Xs])
    Yz = Y - np.array([Y[zg == g].mean() for g in zg])
    X = np.column_stack([np.ones(ok.sum()), Z])
    b, *_ = np.linalg.lstsq(X, Yz, rcond=None)
    e = Yz - X @ b
    meat = np.zeros((X.shape[1],) * 2)
    for g in set(zg):
        sel = zg == g
        Xg, eg = X[sel], e[sel]
        meat += (Xg.T * eg) @ (Xg.T * eg).T
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ meat @ XtXi * (len(set(zg)) / (len(set(zg)) - 1))
    se = np.sqrt(np.diag(V))
    t = b / se
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(np.abs(t), len(set(zg)) - 1)
    ss_tot = ((Yz - Yz.mean()) ** 2).sum()
    ss_res = (e ** 2).sum()
    return {"names": ["int"] + names, "coef": b.tolist(), "se": se.tolist(),
            "p": p.tolist(), "r2_within": float(1 - ss_res / ss_tot),
            "n": int(ok.sum())}

b_only = fe_reg(slopes["bins_1_9"], [nnz9.astype(float)], ["n_nonzero"])
b_tot = fe_reg(slopes["bins_1_9"], [ltot], ["log10_total"])
b_both = fe_reg(slopes["bins_1_9"], [nnz9.astype(float), ltot], ["n_nonzero", "log10_total"])
# reverse mediation: does ltot still predict n_nonzero controlling for nothing?
# noise-level test: among samples matched on n_nonzero, is slope still related to ltot?
sl9 = slopes["bins_1_9"]
strata = {}
for k in sorted(set(nnz9)):
    sel = (nnz9 == k) & np.isfinite(sl9)
    if sel.sum() >= 4:
        rr, pp = rho_p(ltot[sel], sl9[sel])
        strata[int(k)] = {"n": int(sel.sum()), "mean_slope": float(sl9[sel].mean()),
                          "sd_slope": float(sl9[sel].std(ddof=1)),
                          "rho_ltot_vs_slope": rr, "p": pp}
# partial spearman (rank residual) slope vs ltot controlling n_nonzero
rx = stats.rankdata(ltot); rn = stats.rankdata(nnz9.astype(float)); ry = stats.rankdata(sl9)
res_x = rx - np.polyval(np.polyfit(rn, rx, 1), rn)
res_y = ry - np.polyval(np.polyfit(rn, ry, 1), rn)
partial_r, partial_p = stats.pearsonr(res_x, res_y)

binary_pattern_test = {
    "rho_nnonzero_vs_log10total": [r_nnz_ltot, p_nnz_ltot],
    "rho_nnonzero_vs_slope9": [r_nnz_sl, p_nnz_sl],
    "partial_spearman_slope_vs_ltot_given_nnonzero": [float(partial_r), float(partial_p)],
    "fe_models": {"slope~n_nonzero": b_only, "slope~ltot": b_tot,
                  "slope~n_nonzero+ltot": b_both},
    "rho_slope_ltot_unconditional": [table["bins_1_9"]["spearman_rho_vs_log10total"],
                                     table["bins_1_9"]["spearman_p_vs_log10total"]],
    "rmse_nnonzero_only": float(np.sqrt(np.mean(
        (sl9 - np.polyval(np.polyfit(nnz9.astype(float), sl9, 1), nnz9.astype(float))) ** 2))),
    "rmse_ltot_only": float(np.sqrt(np.mean(
        (sl9 - np.polyval(np.polyfit(ltot, sl9, 1), ltot)) ** 2))),
    "strata_by_n_nonzero": strata,
}

# ---------------------------------------------------------------- (e) core-bin constraint
s3 = P[:, :3].sum(axis=1)
core_stats = {
    "sum_bins1_3_percent": {"min": float(s3.min()), "max": float(s3.max()),
                            "mean": float(s3.mean()), "sd": float(s3.std(ddof=1))},
    "bin1_share_of_core_mean": float((P[:, 0] / s3).mean()),
    "bin2_share_of_core_mean": float((P[:, 1] / s3).mean()),
    "bin3_share_of_core_mean": float((P[P[:, 2] > 0, 2] / s3[P[:, 2] > 0]).mean()),
    "sd_log_ratio_p1_over_p2": float(np.log10(P[:, 0] / P[:, 1]).std(ddof=1)),
    "sd_log_ratio_p2_over_p3": float(np.log10(P[P[:, 2] > 0, 1] / P[P[:, 2] > 0, 2]).std(ddof=1)),
    "observed_sd_slope_bins1_3": table["bins_1_3"]["sd_slope"],
    "note": ("core bins are compositional: they carry 82-100% of mass.  The 3-bin "
             "slope is a function of the two log-ratios, which are free given the sum; "
             "the constraint alone does not destroy power, it removes the level."),
}

# ---------------------------------------------------------------- (b)+(c) simulations
rng = np.random.default_rng(777)
NGRID = []


def tau_for_groups(gen_pct, nn):
    """Return concentration/overdispersion-free simple multinomial generator."""
    return None


def make_tally_map():
    """Recovered integer tallies where the prior pipeline found them; else fall
    back to the conservative upper bound N = 100/quantum_pp from 07 results.
    """
    tm = dict(tally_map)
    for j in range(N):
        if sids[j] not in tm:
            v = P[j][P[j] > 0]
            tm[sids[j]] = int(np.ceil(100.0 / v.min()))
    return tm


TALLY = make_tally_map()


def group_mean_pct(Pmat, g):
    return Pmat[groups == g].mean(axis=0)


def sim_once(seed, range_name, idx, gen="group", tilt_d=0.0, use_obs_scatter=False):
    """One realisation of the whole data set under a counting null plus an
    optional abundance-dependent tilt injected into the 3-bin sub-composition."""
    r = np.random.default_rng(seed)
    z = ltot - ltot.mean()
    out_pct = np.zeros((N, 22))
    for j in range(N):
        base = group_mean_pct(P, groups[j]).copy()
        if tilt_d != 0.0:
            # tilt the sub-composition over `idx` by exp(-d * z * (lmid - mean))
            c = lmid[idx] - lmid[idx].mean()
            base[idx] = base[idx] * np.exp(-tilt_d * z[j] * c)
            # renormalise the sub-composition back to its original mass
            base[idx] *= P[j][idx].sum() / base[idx].sum()
        base = np.maximum(base, 0)
        base = base / base.sum()
        n = TALLY[sids[j]]
        k = r.multinomial(n, base)
        v = 100.0 * k / n
        out_pct[j] = np.round(v, 2) if range_name != "raw" else v
    return out_pct


def sim_null_block(nsim, range_name, idx, seed0, gen="group", tally_noise=False):
    """Null distribution of the range-restricted slope and its abundance correlation."""
    sl_all = np.full((nsim, N), np.nan)
    for s in range(nsim):
        Ps = sim_once(seed0 + s, range_name, idx, gen=gen)
        for j in range(N):
            sl_all[s, j] = fit_slope(Ps[j], idx)
    return sl_all


def summarise_null(sl_all, idx):
    obs = np.array([fit_slope(P[j], idx) for j in range(N)])
    rs = np.array([rho_p(ltot, sl_all[s])[0] for s in range(len(sl_all))])
    wgs = np.array([within_group_mean_rho(ltot, sl_all[s])[0] for s in range(len(sl_all))])
    obs_r = rho_p(ltot, obs)[0]
    obs_wg, _ = within_group_mean_rho(ltot, obs)
    p_overall = (np.sum(np.abs(rs) >= abs(obs_r)) + 1) / (len(rs) + 1)
    p_within = (np.sum(np.abs(wgs) >= abs(obs_wg)) + 1) / (len(wgs) + 1)
    return {
        "observed_mean_slope": float(np.nanmean(obs)),
        "null_mean_slope": float(np.nanmean(sl_all)),
        "null_sd_slope": float(np.nanstd(sl_all)),
        "observed_rho_with_ltot": float(obs_r),
        "null_rho_mean": float(rs.mean()), "null_rho_sd": float(rs.std(ddof=1)),
        "null_rho_q025": float(np.quantile(rs, 0.025)),
        "null_rho_q975": float(np.quantile(rs, 0.975)),
        "p_two_sided_overall": float(p_overall),
        "fraction_of_obs_rho_explained_by_null": float(1 - abs(rs.mean()) / abs(obs_r)) if abs(obs_r) > 0 else None,
        "observed_within_group_mean_rho": float(obs_wg),
        "null_within_group_rho_mean": float(wgs.mean()),
        "null_within_group_rho_sd": float(wgs.std(ddof=1)),
        "p_two_sided_within_group": float(p_within),
        "fraction_within_group_explained": float(1 - abs(wgs.mean()) / abs(obs_wg)) if abs(obs_wg) > 0 else None,
        "n_sims": int(len(sl_all)),
    }


NSIM = 3000
null9 = sim_null_block(NSIM, "bins_1_9", list(range(9)), 10_000)
null3 = sim_null_block(NSIM, "bins_1_3", list(range(3)), 20_000)
null4 = sim_null_block(NSIM, "bins_1_4", list(range(4)), 30_000)

null_simulation = {
    "design": ("multinomial counts drawn at each sample's own recovered tally from its "
               "GROUP-MEAN size distribution (percentages rounded to 2 dp like the "
               "source table); seed 777, 3000 draws; far more than the 5-9 bin range "
               "used by the claim"),
    "bins_1_9": summarise_null(null9, list(range(9))),
    "bins_1_4": summarise_null(null4, list(range(4))),
    "bins_1_3": summarise_null(null3, list(range(3))),
    "tallies_used": TALLY,
}

# ---------------------------------------------------------------- power analysis
def power_curve(idx, d_grid, nsim, use_obs_scatter, alpha=0.05):
    """Power of the overall Spearman test (and of the within-group mean rho test)
    for a tilt of size d per decade of log10 total, injected into `idx`."""
    res = []
    z = ltot - ltot.mean()
    # observed scatter of the range-restricted slope (used as a pessimistic noise floor)
    obs_sl = np.array([fit_slope(P[j], idx) for j in range(N)])
    obs_sd = float(np.nanstd(obs_sl, ddof=1))
    for d in d_grid:
        rej = 0
        rej_wg = 0
        for s in range(nsim):
            r = np.random.default_rng(500_000 + s)
            sl = np.full(N, np.nan)
            Pgen = np.zeros((N, 22))
            for j in range(N):
                base = group_mean_pct(P, groups[j]).copy()
                c = lmid[idx] - lmid[idx].mean()
                base[idx] = base[idx] * np.exp(-d * z[j] * c)
                base[idx] *= P[j][idx].sum() / base[idx].sum()
                base = np.maximum(base, 0)
                base = base / base.sum()
                n = TALLY[sids[j]]
                k = r.multinomial(n, base)
                Pgen[j] = np.round(100.0 * k / n, 2)
                sl[j] = fit_slope(Pgen[j], idx)
            if use_obs_scatter:
                # add process scatter so total SD matches the observed SD
                base_sd = np.nanstd(sl, ddof=1)
                extra = np.sqrt(max(obs_sd ** 2 - base_sd ** 2, 0.0))
                sl = sl + r.normal(0, extra, N)
            _, p = rho_p(ltot, sl)
            if p < alpha:
                rej += 1
            wg, _ = within_group_mean_rho(ltot, sl)
            # permutation-free within-group p via t approx on mean of group rhos
            # (group rhos are ~independent, k=5 groups)
            t = wg * np.sqrt(3) / np.sqrt(max(1 - wg ** 2, 1e-9))  # mean of 5 group rhos, each n~5-7
            if 2 * stats.t.sf(abs(t), 3) < alpha:
                rej_wg += 1
        res.append({"tilt_per_decade": float(d),
                    "slope_change_across_observed_range": float(d * (ltot.max() - ltot.min())),
                    "power_overall_spearman": rej / nsim,
                    "power_within_group": rej_wg / nsim})
    return res


DGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
pc3 = power_curve(list(range(3)), DGRID, 1000, use_obs_scatter=False)
pc3s = power_curve(list(range(3)), DGRID, 1000, use_obs_scatter=True)
pc9 = power_curve(list(range(9)), DGRID, 1000, use_obs_scatter=True)


def mde(curve, key="power_overall_spearman", target=0.8):
    d = [c["tilt_per_decade"] for c in curve]
    pw = [c[key] for c in curve]
    for i in range(1, len(d)):
        if pw[i - 1] < target <= pw[i]:
            f = (target - pw[i - 1]) / max(pw[i] - pw[i - 1], 1e-12)
            return float(d[i - 1] + f * (d[i] - d[i - 1]))
    return None


power_analysis = {
    "design": ("synthetic abundance-dependent tilt exp(-d*z*(lmid-mean)) applied to the "
               "3-bin sub-composition (renormalised to each sample's observed core mass), "
               "then multinomial sampling at the recovered tally; 1000 sims per grid point, "
               "alpha=0.05 two-sided; d is the change in log-log slope per decade of total"),
    "observed_range_log10_total": float(ltot.max() - ltot.min()),
    "observed_sd_3bin_slope": table["bins_1_3"]["sd_slope"],
    "observed_sd_9bin_slope": table["bins_1_9"]["sd_slope"],
    "3bin_curve_counting_noise_only": pc3,
    "3bin_curve_with_observed_process_scatter": pc3s,
    "9bin_curve_with_observed_process_scatter": pc9,
    "MDE_3bin_counting_noise_only": mde(pc3),
    "MDE_3bin_with_process_scatter": mde(pc3s),
    "MDE_9bin_with_process_scatter": mde(pc9),
    "MDE_3bin_within_group": mde(pc3, key="power_within_group"),
    "note": ("'process scatter' version adds Gaussian noise so the simulated total SD of "
             "the slope equals the OBSERVED SD, i.e. it assumes all observed between-sample "
             "slope variation is noise.  That is the pessimistic (low-power) bound; the "
             "counting-noise-only version is the optimistic one."),
}

# ------------------------------------------------ what would the artefact predict for 3 bins?
# if the 9-bin steepening were purely the visibility artefact, the 3-bin slope (all bins
# essentially always visible) should show no trend: check the induced trend in the null
z9 = nnz9.astype(float)

result = {
    "claim_id": "F5",
    "bin_range_table": table,
    "binary_pattern_test": binary_pattern_test,
    "null_simulation": null_simulation,
    "power_analysis": power_analysis,
    "core_bin_constraint": core_stats,
}

with open(OUT / "verify_F5.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

# ---- console summary
print("=== (a) bin range table ===")
for nm in RANGES:
    t = table[nm]
    print(f"{nm:16s} mean={t['mean_slope']:+.2f} sd={t['sd_slope']:.2f} "
          f"rho={t['spearman_rho_vs_log10total']:+.3f} p={t['spearman_p_vs_log10total']:.3g} "
          f"wg_rho={t['within_group_mean_rho']:+.3f} FEslope={t['FE_cluster_slope_per_decade']:+.3f} "
          f"(p={t['FE_cluster_p']:.3g}) n_nonzero_min={t['n_nonzero_min']} "
          f"all_nonzero={t['n_samples_with_all_bins_nonzero']}/30")
print("\n=== (d) binary pattern ===")
print("rho(n_nonzero, ltot) =", binary_pattern_test["rho_nnonzero_vs_log10total"])
print("rho(n_nonzero, slope9) =", binary_pattern_test["rho_nnonzero_vs_slope9"])
print("partial slope~ltot|n_nonzero:", binary_pattern_test["partial_spearman_slope_vs_ltot_given_nnonzero"])
print("R2 within: n_nonzero only, ltot only, both:",
      b_only["r2_within"], b_tot["r2_within"], b_both["r2_within"])
print("both model coefs:", b_both["names"], np.round(b_both["coef"], 3), "p", np.round(b_both["p"], 4))
print("\n=== (c) null sim ===")
for k in ["bins_1_9", "bins_1_4", "bins_1_3"]:
    d = null_simulation[k]
    print(k, "obs_rho=%.3f null_mean=%.3f null_sd=%.3f p=%.3g frac_expl=%.2f wg_obs=%.3f wg_null=%.3f p_wg=%.3g"
          % (d["observed_rho_with_ltot"], d["null_rho_mean"], d["null_rho_sd"],
             d["p_two_sided_overall"], d["fraction_of_obs_rho_explained_by_null"],
             d["observed_within_group_mean_rho"], d["null_within_group_rho_mean"],
             d["p_two_sided_within_group"]))
print("\n=== (b) power ===")
print("MDE 3bin counting-only:", power_analysis["MDE_3bin_counting_noise_only"])
print("MDE 3bin +process scatter:", power_analysis["MDE_3bin_with_process_scatter"])
print("MDE 9bin +process scatter:", power_analysis["MDE_9bin_with_process_scatter"])
for c in pc3:
    print("  d=%.2f  P(rho)=%.2f  P(wg)=%.2f" % (c["tilt_per_decade"], c["power_overall_spearman"], c["power_within_group"]))
print("\n=== (e) core bins ===", json.dumps(core_stats, indent=1))
print("\n=== 1-3 sensitivity ===", json.dumps(table["bins_1_3_sensitivity"], indent=1))