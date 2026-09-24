"""Adversarial verification of CLAIM F3.

Claim: Within a collection group, polyethylene share (polymer_PE_percent) rises
with total particle concentration while the varnish share falls.

Everything below is written from scratch (numpy/scipy/pandas + matplotlib Agg).
The analysis unit is one depth interval (n=30), clustered in 5 collection
groups (group_01 n=5, group_02 n=7, group_03 n=7, group_04 n=5, group_05 n=6).
Intervals within a group are repeated measures -> cluster = group (G=5).

Tests requested:
  (a) abundance restriction (total_per_L > 1000 and > 3000)
  (b) PE% vs other covariates (chl-a, salinity, POC, C:N, depth), FE comparison
  (c) compositional approach: log(PE/varnish), ilr, logit(PE/(PE+V))
  (d) group_05 as genuine exception
  (e) permutation test with depth held fixed (partial correlation within group)

Plus self-initiated attacks:
  (f) quantisation / default-class artefact: recover the implicit number of
      analysed particles N from the granularity of the reported percentages
      and test whether PE% is a near-deterministic function of N (which would
      make "PE rises with concentration" mechanically a resolution artefact).
  (g) exact finite-sample inference: group sign-flip exact test, random-effects
      meta-analysis, influence diagnostics, measurement-error sensitivity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify_F3"
OUT.mkdir(parents=True, exist_ok=True)
RES = {}

# ----------------------------------------------------------------- load data
comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(PROJ / "data" / "sample_environment_profiles.tsv", sep="\t")
m = comp.merge(env, on=["sample_id", "collection_group_id"], suffixes=("_c", "_e"))
assert len(m) == 30

sid = m["sample_id"].to_numpy()
tot = m["particle_total_per_L_c"].to_numpy(float)
ltot = np.log10(tot)
PE = m["polymer_PE_percent"].to_numpy(float)
V = m["polymer_varnish_percent"].to_numpy(float)
PEc = m["polymer_count_PE_per_L"].to_numpy(float)
Vc = m["polymer_count_varnish_per_L"].to_numpy(float)
g = m["collection_group_id"].to_numpy()
groups = sorted(set(g))
gi = m["collection_group_id"].map({k: i for i, k in enumerate(groups)}).to_numpy(int)
dmid = 0.5 * (m["depth_min_m_c"] + m["depth_max_m_c"]).to_numpy(float)
chl = m["chlorophyll_a_ug_L"].to_numpy(float)
sal = m["salinity_PSU"].to_numpy(float)
poc = m["particulate_organic_carbon_mg_L"].to_numpy(float)
cn = m["carbon_to_nitrogen_ratio"].to_numpy(float)
tol_err = m["particle_total_std_or_error_per_L"].to_numpy(float)

G = len(groups)
N = len(m)


# ---------------------------------------------------------------- primitives
def demean_within(v, mask=None):
    v = np.asarray(v, float).astype(float).copy()
    if mask is None:
        mask = np.ones(len(v), bool)
    out = np.full(len(v), np.nan)
    for k in groups:
        s = (g == k) & mask
        if s.sum() >= 1:
            out[s] = v[s] - v[s].mean()
    return out


def fe_fit(y, Xcols, mask=None, cluster=True):
    """Fixed-effects (within-group demeaned) OLS on [1, X...].

    Returns dict with beta, cluster-robust (CR1, G-1 df) and classical SEs.
    Groups with <2 usable rows are dropped.
    """
    y = np.asarray(y, float)
    Xcols = [np.asarray(x, float) for x in Xcols]
    if mask is None:
        mask = np.ones(len(y), bool)
    mask = mask & np.isfinite(y) & np.all(np.isfinite(np.column_stack(Xcols)), axis=1)
    y = y[mask]
    X = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in Xcols])
    gg = g[mask]
    Xd = X.copy()
    yd = y.copy()
    kept = []
    for k in groups:
        s = gg == k
        if s.sum() >= 2:
            Xd[s] = X[s] - X[s].mean(axis=0)
            yd[s] = y[s] - y[s].mean()
            kept.append(k)
    if not kept:
        return None
    beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ yd
    resid = yd - Xd @ beta
    n, kpar = Xd.shape
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    out = {"beta": beta.tolist(), "n": int(n), "G": len(kept), "groups": kept}
    if cluster and len(kept) > 1:
        meat = np.zeros((kpar, kpar))
        for k in kept:
            s = gg == k
            u = Xd[s].T @ resid[s]
            meat += np.outer(u, u)
        scale = (len(kept) / (len(kept) - 1.0)) * ((n - 1.0) / (n - kpar))
        cov = scale * XtXi @ meat @ XtXi
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        df = len(kept) - 1
        t = beta / se
        p = 2 * st.t.sf(np.abs(t), df)
        out.update({"se": se.tolist(), "t": t.tolist(), "p": p.tolist(),
                    "df": df, "se_type": "cluster_CR1"})
    else:
        s2 = float(resid @ resid) / (n - kpar)
        cov = s2 * XtXi
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        t = beta / se
        p = 2 * st.t.sf(np.abs(t), n - kpar)
        out.update({"se": se.tolist(), "t": t.tolist(), "p": p.tolist(),
                    "df": int(n - kpar), "se_type": "classical"})
    return out


def fe_r2(y, x):
    """Within-group R^2 of a single FE regressor."""
    xd = demean_within(x)
    yd = demean_within(y)
    ok = np.isfinite(xd) & np.isfinite(yd)
    xd, yd = xd[ok], yd[ok]
    if (xd @ xd) == 0 or (yd @ yd) == 0:
        return np.nan
    return float((xd @ yd) ** 2 / ((xd @ xd) * (yd @ yd)))


def group_slope(y, x, k, rank=False):
    s = g == k
    xs, ys = np.asarray(x, float)[s], np.asarray(y, float)[s]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 3:
        return np.nan, np.nan
    if rank:
        xs, ys = st.rankdata(xs), st.rankdata(ys)
    return float(np.polyfit(xs, ys, 1)[0]), int(len(xs))


# ------------------------------------------------------------- 0. reproduce
pooled_rho_pe = st.spearmanr(ltot, PE)
pooled_rho_v = st.spearmanr(ltot, V)
pooled_rho_pv = st.spearmanr(PE, V)
fit0 = fe_fit(PE, [ltot])
per_group = {}
for k in groups:
    s = g == k
    rp = st.spearmanr(ltot[s], PE[s])
    rv = st.spearmanr(ltot[s], V[s])
    sl_ols, nk = group_slope(PE, ltot, k)
    sl_r, _ = group_slope(PE, ltot, k, rank=True)
    per_group[k] = {
        "n": int(s.sum()),
        "rho_logtot_PE": round(float(rp.statistic), 3),
        "p_rho_PE": round(float(rp.pvalue), 4),
        "rho_logtot_varnish": round(float(rv.statistic), 3),
        "ols_slope_PE_per_decade": round(sl_ols, 2),
        "rank_slope_PE_per_decade": round(sl_r, 2),
        "total_range": [float(tot[s].min()), float(tot[s].max())],
    }
mean_rho = float(np.mean([per_group[k]["rho_logtot_PE"] for k in groups]))
RES["0_reproduction"] = {
    "pooled_rho_logtot_PE": {"rho": round(float(pooled_rho_pe.statistic), 4),
                             "p": float(pooled_rho_pe.pvalue), "n": N},
    "pooled_rho_logtot_varnish": {"rho": round(float(pooled_rho_v.statistic), 4),
                                  "p": float(pooled_rho_v.pvalue)},
    "pooled_rho_PE_varnish": {"rho": round(float(pooled_rho_pv.statistic), 4),
                              "p": float(pooled_rho_pv.pvalue)},
    "mean_within_group_rho_PE": round(mean_rho, 3),
    "n_groups_positive_rho": int(sum(per_group[k]["rho_logtot_PE"] > 0 for k in groups)),
    "FE_slope_PE_per_decade": {"slope": round(fit0["beta"][1], 2),
                               "se_cluster": round(fit0["se"][1], 2),
                               "t": round(fit0["t"][1], 2),
                               "p": round(fit0["p"][1], 4),
                               "df": fit0["df"], "G": fit0["G"]},
    "std": 0.0,
    "per_group": per_group,
}

# ============================================================== (a) abundance
RES["a_abundance_restriction"] = {}
for thr in [0, 1000, 3000]:
    mask = tot > thr
    out = {"threshold_per_L": thr, "n_kept": int(mask.sum()),
           "groups_kept": [k for k in groups if (mask & (g == k)).sum() >= 2],
           "samples_kept": sid[mask].tolist()}
    f_pe = fe_fit(PE, [ltot], mask)
    f_v = fe_fit(V, [ltot], mask)
    pv = np.where((PE + V) > 0, PE / np.maximum(PE + V, 1e-12), np.nan)
    f_ratio = fe_fit(pv, [ltot], mask)
    out["FE_slope_PE"] = {"slope": round(f_pe["beta"][1], 2), "se": round(f_pe["se"][1], 2),
                          "t": round(f_pe["t"][1], 2), "p": round(f_pe["p"][1], 4),
                          "df": f_pe["df"], "G": f_pe["G"]}
    out["FE_slope_varnish"] = {"slope": round(f_v["beta"][1], 2), "se": round(f_v["se"][1], 2),
                               "t": round(f_v["t"][1], 2), "p": round(f_v["p"][1], 4)}
    out["FE_slope_PEshare_of_PEplusV"] = {"slope": round(f_ratio["beta"][1], 3),
                                          "se": round(f_ratio["se"][1], 3),
                                          "t": round(f_ratio["t"][1], 2),
                                          "p": round(f_ratio["p"][1], 4)}
    rh = st.spearmanr(ltot[mask], PE[mask])
    out["pooled_rho_PE"] = {"rho": round(float(rh.statistic), 3), "p": float(rh.pvalue),
                            "n": int(mask.sum())}
    rhv = st.spearmanr(ltot[mask], V[mask])
    out["pooled_rho_varnish"] = {"rho": round(float(rhv.statistic), 3), "p": float(rhv.pvalue)}
    per = {}
    for k in groups:
        s = (g == k) & mask
        if s.sum() >= 3:
            r = st.spearmanr(ltot[s], PE[s])
            sl, _ = group_slope(PE, ltot, k) if s.sum() == (g == k).sum() else (np.nan, np.nan)
            s_full = g == k
            sl_full = group_slope(PE, ltot, k)[0]
            per[k] = {"n": int(s.sum()), "rho": round(float(r.statistic), 3),
                      "p": round(float(r.pvalue), 4),
                      "ols_slope_full_group": round(sl_full, 2)}
    out["per_group_rho"] = per
    out["mean_within_group_rho"] = round(float(np.mean([per[k]["rho"] for k in per])), 3) if per else None
    RES["a_abundance_restriction"][f"thr_{thr}"] = out

# ============================================================ (b) covariates
RES["b_covariate_specificity"] = {}
covs = {
    "log10_total": ltot,
    "log10_chlorophyll_a": np.log10(np.clip(chl, 1e-12, None)),
    "salinity": sal,
    "POC_mg_L": poc,
    "C_to_N": cn,
    "depth_mid_m": dmid,
    "log10_PE_count": np.log10(np.clip(PEc, 1e-12, None)),
    "log10_varnish_count": np.log10(np.clip(Vc, 1e-12, None)),
}
for name, x in covs.items():
    f = fe_fit(PE, [x])
    ok = np.isfinite(x) & np.isfinite(PE)
    pr = st.spearmanr(x[ok], PE[ok])
    RES["b_covariate_specificity"][name] = {
        "FE_within_R2": round(fe_r2(PE, x), 3),
        "FE_slope_pp_per_unit": round(f["beta"][1], 3),
        "FE_t": round(f["t"][1], 2),
        "FE_p": round(f["p"][1], 4),
        "pooled_rho_vs_PE": round(float(pr.statistic), 3),
        "pooled_p": float(pr.pvalue),
    }
# joint FE model PE ~ ltot + depth
f_joint = fe_fit(PE, [ltot, dmid])
RES["b_covariate_specificity"]["joint_FE_PE_ltot_plus_depth"] = {
    "slope_ltot": round(f_joint["beta"][1], 2), "se_ltot": round(f_joint["se"][1], 2),
    "t_ltot": round(f_joint["t"][1], 2), "p_ltot": round(f_joint["p"][1], 4),
    "slope_depth": round(f_joint["beta"][2], 2), "se_depth": round(f_joint["se"][2], 2),
    "t_depth": round(f_joint["t"][2], 2), "p_depth": round(f_joint["p"][2], 4),
}

# ========================================================= (c) compositional
RES["c_compositional"] = {}
eps = 1e-6
lr = np.log((PE + eps) / (V + eps))
ilr = lr / np.sqrt(2.0)
logit = np.log((PE + 0.5) / (V + 0.5))
for name, vec in [("log_ratio_log_PE_over_varnish", lr),
                  ("ilr_2part", ilr),
                  ("logit_PE_over_PEplusV", logit),
                  ("share_PE_over_PEplusV", np.where((PE + V) > 0, PE / np.maximum(PE + V, 1e-12), np.nan))]:
    f = fe_fit(vec, [ltot])
    ok = np.isfinite(vec)
    pr = st.spearmanr(ltot[ok], vec[ok])
    per = {}
    for k in groups:
        s = (g == k) & ok
        if s.sum() >= 3:
            r = st.spearmanr(ltot[s], vec[s])
            per[k] = round(float(r.statistic), 3)
    RES["c_compositional"][name] = {
        "FE_slope_per_decade": round(f["beta"][1], 3),
        "se_cluster": round(f["se"][1], 3), "t": round(f["t"][1], 2),
        "p": round(f["p"][1], 4),
        "pooled_rho": round(float(pr.statistic), 3), "pooled_p": float(pr.pvalue),
        "per_group_rho": per,
        "mean_within_group_rho": round(float(np.mean(list(per.values()))), 3) if per else None,
    }
# zero-handling sensitivity for the log-ratio: add 0.5% pseudocount (half of
# the smallest observable non-zero class share) and drop the PE=0 sample.
lr2 = np.log((PE + 0.5) / (V + 0.5))
f2 = fe_fit(lr2, [ltot])
RES["c_compositional"]["log_ratio_pseudocount_0.5pct"] = {
    "FE_slope_per_decade": round(f2["beta"][1], 3), "se": round(f2["se"][1], 3),
    "t": round(f2["t"][1], 2), "p": round(f2["p"][1], 4)}
mask = PE > 0
f3 = fe_fit(lr, [ltot], mask)
RES["c_compositional"]["log_ratio_drop_PE_zero_sample"] = {
    "n": f3["n"], "FE_slope_per_decade": round(f3["beta"][1], 3),
    "se": round(f3["se"][1], 3), "t": round(f3["t"][1], 2), "p": round(f3["p"][1], 4)}

# ================================================================ (d) group_05
RES["d_group_05"] = {}
s5 = g == "group_05"
RES["d_group_05"]["description"] = {
    "n": int(s5.sum()),
    "mean_PE_pct": round(float(PE[s5].mean()), 2),
    "mean_varnish_pct": round(float(V[s5].mean()), 2),
    "mean_salinity_PSU": round(float(sal[s5].mean()), 3),
    "mean_total_per_L": round(float(tot[s5].mean()), 1),
    "total_range": [float(tot[s5].min()), float(tot[s5].max())],
    "rank_of_PE_within_group_ascending_total": [
        sid[s5][i] for i in np.argsort(tot[s5])],
    "PE_by_ascending_total": [round(float(x), 2) for x in PE[s5][np.argsort(tot[s5])]],
}
for k in groups:
    s = g == k
    RES["d_group_05"][f"contrast_{k}"] = {
        "mean_PE_pct": round(float(PE[s].mean()), 2),
        "mean_varnish_pct": round(float(V[s].mean()), 2),
        "mean_salinity": round(float(sal[s].mean()), 3),
        "mean_depth_m": round(float(dmid[s].mean()), 3),
        "mean_chl_a": round(float(chl[s].mean()), 3),
    }
# group_05: is its flat slope driven by one point? jackknife within group
jk = {}
base = group_slope(PE, ltot, "group_05")[0]
for i in np.where(s5)[0]:
    keep = np.ones(N, bool)
    keep[i] = False
    xs = ltot[keep & s5]
    ys = PE[keep & s5]
    jk[sid[i]] = round(float(np.polyfit(xs, ys, 1)[0]), 2)
RES["d_group_05"]["within_group_05_jackknife_slopes"] = {"full": round(base, 2), **jk}
# relative to its own abundance range: group_05 is the LOW-END group
RES["d_group_05"]["pooled_rho_removing_group_05"] = {
    "rho": round(float(st.spearmanr(ltot[~s5], PE[~s5]).statistic), 3),
    "p": float(st.spearmanr(ltot[~s5], PE[~s5]).pvalue)}
# does group_05 sit at low salinity -> opportunity for default-class artefact
RES["d_group_05"]["salinity_vs_mean_PE"] = {
    "note": "group_05 is freshest AND lowest mean total; both point to low-N regime"}

# =========================================== (e) permutation, depth held fixed
RES["e_permutation_depth_fixed"] = {}
rng = np.random.default_rng(20240919)
nperm = 20000


def resid_on_z(y, z):
    ok = np.isfinite(y) & np.isfinite(z)
    X = np.c_[np.ones(ok.sum()), z[ok]]
    b = np.linalg.pinv(X.T @ X) @ X.T @ y[ok]
    r = np.full(len(y), np.nan)
    r[ok] = y[ok] - X @ b
    return r


def partial_fe(x, y, z):
    xr = demean_within(resid_on_z(x, z))
    yr = demean_within(resid_on_z(y, z))
    ok = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[ok], yr[ok]
    return float(xr @ yr / math.sqrt((xr @ xr) * (yr @ yr)))


def rank_partial_fe(x, y, z):
    return partial_fe(st.rankdata(x), st.rankdata(y), st.rankdata(z))


# depth list-wise missing -> use all 30 (depth complete)
obs_p = partial_fe(ltot, PE, dmid)
obs_rp = rank_partial_fe(ltot, PE, dmid)
null_pe = np.empty(nperm)
null_depth = np.empty(nperm)
for i in range(nperm):
    yp = PE.copy()
    for k in groups:
        s = g == k
        yp[s] = rng.permutation(PE[s])
    null_pe[i] = partial_fe(ltot, yp, dmid)
    zp = dmid.copy()
    for k in groups:
        s = g == k
        zp[s] = rng.permutation(dmid[s])
    null_depth[i] = partial_fe(ltot, PE, zp)


def perm_summary(obs, null):
    return {"observed": round(float(obs), 3),
            "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)),
            "p_one_sided": float((np.sum(null >= obs) + 1) / (len(null) + 1)),
            "null_q025": round(float(np.quantile(null, 0.025)), 3),
            "null_q975": round(float(np.quantile(null, 0.975)), 3),
            "n_perm": int(len(null))}


RES["e_permutation_depth_fixed"]["FE_partial_r_ltot_PE_given_depth"] = perm_summary(obs_p, null_pe)
RES["e_permutation_depth_fixed"]["FE_partial_rank_r_given_depth"] = {
    "observed": round(float(obs_rp), 3)}
RES["e_permutation_depth_fixed"]["control_permute_depth_instead"] = perm_summary(obs_p, null_depth)
per = {}
for k in groups:
    s = g == k
    r = st.spearmanr(ltot[s], PE[s])
    rr = st.spearmanr(st.rankdata(ltot[s]), st.rankdata(PE[s]))
    d = dmid[s]
    zr = st.rankdata(d)
    xr = st.rankdata(ltot[s])
    yr = st.rankdata(PE[s])
    zc = zr - zr.mean()
    ex = xr - xr.mean() - (zc @ (xr - xr.mean())) / (zc @ zc) * zc
    ey = yr - yr.mean() - (zc @ (yr - yr.mean())) / (zc @ zc) * zc
    pr = ex @ ey / math.sqrt((ex @ ex) * (ey @ ey))
    n_k = int(s.sum())
    tstat = pr * math.sqrt((n_k - 3) / (1 - pr ** 2)) if abs(pr) < 1 else np.nan
    per[k] = {"n": n_k, "rho": round(float(r.statistic), 3),
              "partial_rho_given_depth": round(float(pr), 3),
              "partial_p": round(float(2 * st.t.sf(abs(tstat), n_k - 3)), 3),
              "depth_collinearity_rho": round(float(st.spearmanr(d, ltot[s]).statistic), 3),
              "depth_vs_PE_rho": round(float(st.spearmanr(d, PE[s]).statistic), 3)}
prs = [per[k]["partial_rho_given_depth"] for k in groups]
RES["e_permutation_depth_fixed"]["per_group_partial"] = per
RES["e_permutation_depth_fixed"]["mean_partial_rho"] = round(float(np.mean(prs)), 3)
RES["e_permutation_depth_fixed"]["t_test_partial_rhos"] = {
    "mean": round(float(np.mean(prs)), 3),
    "sd": round(float(np.std(prs, ddof=1)), 3),
    "t": round(float(st.ttest_1samp(prs, 0).statistic), 2),
    "p": round(float(st.ttest_1samp(prs, 0).pvalue), 4)}

# ============================================== (f) quantisation / low-N trap
RES["f_quantisation"] = {}
pol_cols = [c for c in comp.columns if c.startswith("polymer_") and c.endswith("_percent")]
size_cols = [c for c in comp.columns if c.startswith("particle_size_") and c.endswith("_percent")]


def recover_N(vals, nmax=2500):
    """Smallest N such that every reported percentage is consistent with an
    integer count N * pct/100 (rounding half-up, tolerance 0.5 of the last
    printed decimal)."""
    p = np.asarray([v for v in vals if v > 0], float)
    if len(p) == 0:
        return np.nan
    tol = 0.005 * (p.size and 1) + 1e-9
    for n in range(1, nmax + 1):
        k = p * n / 100.0
        if np.abs(k - np.round(k)).max() <= 0.005 * n / 100.0 + 1e-9:
            return n
    return np.nan


Nrec = []
for i in range(N):
    Nrec.append(recover_N(comp.loc[i, pol_cols].to_numpy(float)))
Nrec = np.array(Nrec, float)
RES["f_quantisation"]["recovered_N_polymer"] = {
    "values": {sid[i]: (None if not np.isfinite(Nrec[i]) else int(Nrec[i])) for i in range(N)},
    "note": ("N = minimal number of classified particles implied by the 2-dp "
             "percentages; NaN means >2500 or not identifiable. N is the "
             "polymer-only count, not particle_total_per_L.")}
ok = np.isfinite(Nrec) & (Nrec > 0)
RES["f_quantisation"]["conditional_N_polymer_vs_total"] = {
    "n_usable": int(ok.sum()),
    "poly_rho_logN_logtot": round(float(st.spearmanr(np.log10(Nrec[ok]), ltot[ok]).statistic), 3),
    "poly_p": float(st.spearmanr(np.log10(Nrec[ok]), ltot[ok]).pvalue),
    "range_N": [int(Nrec[ok].min()), int(Nrec[ok].max())],
    "range_total_of_those": [float(tot[ok].min()), float(tot[ok].max())],
}
# explicit low-N subset: samples whose identifiable N <= 700 (i.e. the regime
# where a single particle moves a class share by >= ~0.15 pp) -> is PE% still
# tracked by concentration there?
lowN = ok & (Nrec <= 700)
RES["f_quantisation"]["low_N_subset"] = {
    "samples": sid[lowN].tolist(),
    "N_values": Nrec[lowN].astype(int).tolist(),
    "total_values": tot[lowN].astype(int).tolist(),
}
f_low = fe_fit(PE, [ltot], lowN)
r_low = st.spearmanr(ltot[lowN], PE[lowN])
RES["f_quantisation"]["low_N_subset"]["FE_slope_PE"] = {
    "slope": round(f_low["beta"][1], 2), "se": round(f_low["se"][1], 2),
    "t": round(f_low["t"][1], 2), "p": round(f_low["p"][1], 4),
    "G": f_low["G"], "n": f_low["n"]}
RES["f_quantisation"]["low_N_subset"]["pooled_rho"] = {
    "rho": round(float(r_low.statistic), 3), "p": float(r_low.pvalue)}
# is PE% a deterministic function of N (default class)?
rN = st.spearmanr(Nrec[ok], PE[ok])
RES["f_quantisation"]["PE_vs_recovered_N"] = {
    "rho": round(float(rN.statistic), 3), "p": float(rN.pvalue),
    "within_group_FE_R2_PE_on_logN": round(fe_r2(PE, np.log10(np.clip(Nrec, 1e-9, None))), 3)}
# correlation of N with total vs PE within groups
for k in groups:
    s = (g == k) & ok
    if s.sum() >= 4:
        RES["f_quantisation"][f"N_vs_total_{k}"] = {
            "rho_logN_logtot": round(float(st.spearmanr(np.log10(Nrec[s]), ltot[s]).statistic), 3),
            "rho_N_PE": round(float(st.spearmanr(Nrec[s], PE[s]).statistic), 3)}

# =========================== (g) exact inference / robustness of the FE slope
RES["g_inference"] = {}
# exact group sign-flip on FE numerator contributions
num_g = []
den_g = []
for k in groups:
    s = g == k
    xd = ltot[s] - ltot[s].mean()
    yd = PE[s] - PE[s].mean()
    num_g.append(float(xd @ yd))
    den_g.append(float(xd @ xd))
num_g = np.array(num_g)
den = float(np.sum(den_g))
obs_slope = float(num_g.sum() / den)
flips = []
for mask in range(1 << G):
    sg = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(G)])
    flips.append(float((sg * num_g).sum() / den))
flips = np.array(flips)
RES["g_inference"]["exact_group_signflip_FE_slope"] = {
    "observed": round(obs_slope, 2),
    "p_one_sided": float((np.sum(flips >= obs_slope - 1e-9) + 1) / (len(flips) + 1)),
    "p_two_sided": float((np.sum(np.abs(flips) >= abs(obs_slope) - 1e-9) + 1) / (len(flips) + 1)),
    "n_patterns": int(len(flips)),
    "min_max_null": [round(float(flips.min()), 2), round(float(flips.max()), 2)]}
# group weights in the FE slope
weights = {}
for k in groups:
    s = g == k
    xd = ltot[s] - ltot[s].mean()
    weights[k] = round(float((xd @ xd) / den), 3)
RES["g_inference"]["FE_group_weights"] = weights
# leave-one-group-out and leave-one-sample-out
loo_g = {}
for kout in groups:
    mask = g != kout
    f = fe_fit(PE, [ltot], mask)
    loo_g[kout] = {"slope": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
                   "t": round(f["t"][1], 2), "p": round(f["p"][1], 4), "G": f["G"]}
RES["g_inference"]["leave_one_group_out"] = loo_g
loo_s = {}
for i in range(N):
    mask = np.ones(N, bool)
    mask[i] = False
    f = fe_fit(PE, [ltot], mask)
    loo_s[sid[i]] = {"slope": round(f["beta"][1], 2), "p": round(f["p"][1], 4)}
vals = np.array([v["slope"] for v in loo_s.values()])
RES["g_inference"]["leave_one_sample_out"] = {
    "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2),
    "max_abs_p": round(float(max(v["p"] for v in loo_s.values())), 4),
    "detail": loo_s}
# random-effects meta-analysis of per-group OLS slopes (DerSimonian-Laird)
sl, se = [], []
for k in groups:
    s = g == k
    x = ltot[s]
    y = PE[s]
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    r = y - X @ b
    s2 = float(r @ r) / (len(x) - 2)
    cov = s2 * np.linalg.pinv(X.T @ X)
    sl.append(b[1])
    se.append(math.sqrt(cov[1, 1]))
sl, se = np.array(sl), np.array(se)
w = 1 / se ** 2
fe_m = float((w @ sl) / w.sum())
Q = float(w @ (sl - fe_m) ** 2)
C = float(w.sum() - (w @ w) / w.sum())
tau2 = max(0.0, (Q - (G - 1)) / C) if C > 0 else 0.0
wr = 1 / (se ** 2 + tau2)
re_m = float((wr @ sl) / wr.sum())
se_re = math.sqrt(1 / wr.sum())
t_re = re_m / se_re
RES["g_inference"]["meta_analysis_group_slopes"] = {
    "per_group_slopes": [round(float(v), 2) for v in sl],
    "per_group_ses": [round(float(v), 2) for v in se],
    "Q": round(Q, 2), "Q_df": G - 1,
    "Q_p": round(float(st.chi2.sf(Q, G - 1)), 4),
    "tau2": round(tau2, 1),
    "fixed_effect_mean": round(fe_m, 2),
    "random_effects_mean_DL": round(re_m, 2), "se": round(se_re, 2),
    "t": round(t_re, 2), "p": round(float(2 * st.t.sf(abs(t_re), G - 1)), 4),
    "simple_mean": round(float(sl.mean()), 2),
    "simple_mean_t": round(float(st.ttest_1samp(sl, 0).statistic), 2),
    "simple_mean_p": round(float(st.ttest_1samp(sl, 0).pvalue), 4)}
# measurement error on concentration attenuates the slope -> sensitivity
rel = tol_err / tot
mask = rel <= 1.0
f = fe_fit(PE, [ltot], mask)
RES["g_inference"]["drop_samples_with_rel_SE_gt_1"] = {
    "dropped": sid[rel > 1].tolist(),
    "slope": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
    "t": round(f["t"][1], 2), "p": round(f["p"][1], 4), "G": f["G"], "n": f["n"]}
# errors-in-variables: if log10(total) has classical noise, slope is attenuated
RES["g_inference"]["attenuation_note"] = {
    "median_rel_reported_SE": round(float(np.median(rel)), 3),
    "note": "reported SE on total is large (median ~50% of total); classical "
            "measurement error in the regressor attenuates the FE slope, so the "
            "true slope could be larger, not smaller"}

# t-test on varnish too, sign convention
f_v = fe_fit(V, [ltot])
RES["g_inference"]["FE_slope_varnish_full"] = {
    "slope": round(f_v["beta"][1], 2), "se": round(f_v["se"][1], 2),
    "t": round(f_v["t"][1], 2), "p": round(f_v["p"][1], 4)}

# ================================================ summary verdict bookkeeping
RES["verdict_inputs"] = {
    "full_FE_slope": round(obs_slope, 2),
    "thr_1000_FE_slope": RES["a_abundance_restriction"]["thr_1000"]["FE_slope_PE"]["slope"],
    "thr_3000_FE_slope": RES["a_abundance_restriction"]["thr_3000"]["FE_slope_PE"]["slope"],
    "exact_signflip_p_two_sided": RES["g_inference"]["exact_group_signflip_FE_slope"]["p_two_sided"],
    "depth_partial_perm_p": RES["e_permutation_depth_fixed"]["FE_partial_r_ltot_PE_given_depth"]["p_two_sided"],
    "group05_slope": per_group["group_05"]["ols_slope_PE_per_decade"],
}

with open(OUT / "verify_F3.json", "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", OUT / "verify_F3.json")