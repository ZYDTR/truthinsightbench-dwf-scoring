"""Addendum to the F3 adversarial verification: extra attacks H1-H8."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats as st

PROJ = Path("/data/user/conversations/conv-db0bf41e-db48-4b3b-8351-ef78ee787f62/project")
OUT = PROJ / "output" / "verify_F3" / "verify_F3.json"

comp = pd.read_csv(PROJ / "data" / "particle_composition_profiles.tsv", sep="\t")
env = pd.read_csv(PROJ / "data" / "sample_environment_profiles.tsv", sep="\t")
m = comp.merge(env, on=["sample_id", "collection_group_id"], suffixes=("_c", "_e"))
sid = m["sample_id"].to_numpy()
tot = m["particle_total_per_L_c"].to_numpy(float)
ltot = np.log10(tot)
PE = m["polymer_PE_percent"].to_numpy(float)
V = m["polymer_varnish_percent"].to_numpy(float)
PEc = m["polymer_count_PE_per_L"].to_numpy(float)
Vc = m["polymer_count_varnish_per_L"].to_numpy(float)
g = m["collection_group_id"].to_numpy()
groups = sorted(set(g))
G = len(groups)
N = len(m)
dmid = 0.5 * (m["depth_min_m_c"] + m["depth_max_m_c"]).to_numpy(float)
chl = np.log10(np.clip(m["chlorophyll_a_ug_L"].to_numpy(float), 1e-12, None))
POC = m["particulate_organic_carbon_mg_L"].to_numpy(float)
sal = m["salinity_PSU"].to_numpy(float)
cn = m["carbon_to_nitrogen_ratio"].to_numpy(float)
# size descriptors
s11 = m["particle_size_11um_percent"].to_numpy(float)
mid_bins = np.array([11, 18, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5,
                     237.5, 262.5, 287.5, 312.5, 337.5, 362.5, 387.5, 412.5, 437.5,
                     462.5, 487.5, 600.0])
sz_cols = [c for c in comp.columns if c.startswith("particle_size_") and c.endswith("_percent")]
SZ = m[sz_cols].to_numpy(float)
SZ = SZ / np.maximum(SZ.sum(axis=1, keepdims=True), 1e-12)
mean_size = SZ @ mid_bins
frac_lt25 = SZ[:, 0] + SZ[:, 1]
frac_gt100 = SZ[:, 5:].sum(axis=1)


def dm(v, mask=None):
    v = np.asarray(v, float).astype(float).copy()
    if mask is None:
        mask = np.ones(len(v), bool)
    out = np.full(len(v), np.nan)
    for k in groups:
        s = (g == k) & mask
        out[s] = v[s] - v[s].mean()
    return out


def fe_fit(y, Xcols, mask=None):
    y = np.asarray(y, float)
    Xcols = [np.asarray(x, float) for x in Xcols]
    if mask is None:
        mask = np.ones(len(y), bool)
    mask = mask & np.isfinite(y) & np.all(np.isfinite(np.column_stack(Xcols)), axis=1)
    yy = y[mask]
    XX = np.column_stack([np.ones(mask.sum())] + [x[mask] for x in Xcols])
    gg = g[mask]
    Xd, yd = XX.copy(), yy.copy()
    kept = []
    for k in groups:
        s = gg == k
        if s.sum() >= 2:
            Xd[s] = XX[s] - XX[s].mean(axis=0)
            yd[s] = yy[s] - yy[s].mean()
            kept.append(k)
    beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ yd
    r = yd - Xd @ beta
    n, kp = Xd.shape
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((kp, kp))
    for k in kept:
        s = gg == k
        u = Xd[s].T @ r[s]
        meat += np.outer(u, u)
    Gd = len(kept)
    cov = (Gd / (Gd - 1.0)) * ((n - 1.0) / (n - kp)) * XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t = beta / se
    p = 2 * st.t.sf(np.abs(t), Gd - 1)
    return {"beta": beta, "se": se, "t": t, "p": p, "G": Gd, "n": int(n),
            "kept": kept, "Xd": Xd, "yd": yd, "gg": gg, "resid": r}


def fe_r2(y, x):
    xd, yd = dm(x), dm(y)
    ok = np.isfinite(xd) & np.isfinite(yd)
    return float((xd[ok] @ yd[ok]) ** 2 / ((xd[ok] @ xd[ok]) * (yd[ok] @ yd[ok])))


RES = json.load(open(OUT))
H = {}

# ---- H1: size-distribution confound -----------------------------------------
H["H1_size_confound"] = {}
H["H1_size_confound"]["FE_PE_on_size_descriptors"] = {
    "mean_particle_size_um": {"slope": None, "R2": round(fe_r2(PE, mean_size), 3)},
    "frac_lt_25um": {"R2": round(fe_r2(PE, frac_lt25), 3)},
    "frac_gt_100um": {"R2": round(fe_r2(PE, frac_gt100), 3)},
    "pct_11um_bin": {"R2": round(fe_r2(PE, s11), 3)},
    "R2_log10_total": round(fe_r2(PE, ltot), 3),
}
H["H1_size_confound"]["FE_total_on_size"] = {
    "mean_particle_size_um": {"R2": round(fe_r2(ltot, mean_size), 3)},
    "frac_lt_25um": {"R2": round(fe_r2(ltot, frac_lt25), 3)},
}
for nm, z in [("mean_size", mean_size), ("frac_lt_25um", frac_lt25),
              ("frac_gt_100um", frac_gt100), ("pct_11um", s11)]:
    f = fe_fit(PE, [ltot, z])
    H["H1_size_confound"][f"FE_PE_ltot_plus_{nm}"] = {
        "slope_ltot": round(f["beta"][1], 2), "se_ltot": round(f["se"][1], 2),
        "p_ltot": round(f["p"][1], 4),
        "slope_z": round(f["beta"][2], 2), "p_z": round(f["p"][2], 4)}
# full size composition: 22-bin CLR vs PE%, within group
Szp = np.clip(SZ, 1e-6, None)
clr = np.log(Szp) - np.log(Szp).mean(axis=1, keepdims=True)
# first 2 principal axes of CLR within group
clrd = clr.copy()
for _k in groups:
    _s = g == _k
    clrd[_s] = clr[_s] - clr[_s].mean(axis=0)
u, s_, vt = np.linalg.svd(clrd, full_matrices=False)
pc1 = clrd @ vt[0]
pc2 = clrd @ vt[1]
H["H1_size_confound"]["FE_PE_on_size_CLR_PC1_PC2"] = {
    "R2_PC1": round(fe_r2(PE, pc1), 3), "R2_PC2": round(fe_r2(PE, pc2), 3)}
f = fe_fit(PE, [ltot, pc1, pc2])
H["H1_size_confound"]["FE_PE_ltot_plus_sizePC1PC2"] = {
    "slope_ltot": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
    "p_ltot": round(f["p"][1], 4)}
H["H1_size_confound"]["R2_PE_on_log10total"] = round(fe_r2(PE, ltot), 3)

# ---- H2: PE and varnish are near-complementary -> not two independent tests --
H["H2_PE_varnish_complementarity"] = {
    "FE_corr_PE_V": round(float(np.corrcoef(dm(PE), dm(V))[0, 1]), 3),
    "R2_V_explained_by_PE_within_group": round(fe_r2(V, PE), 3),
    "R2_PE_explained_by_V_within_group": round(fe_r2(PE, V), 3),
    "note": ("pooled rho = -0.88 and the within-group correlation is similarly "
             "strong; 'varnish falls' is essentially the mirror of 'PE rises', "
             "so the two outcomes are not independent evidence."),
}
# per-group correlation
pgcv = {}
for k in groups:
    s = g == k
    pgcv[k] = round(float(st.spearmanr(PE[s], V[s]).statistic), 3)
H["H2_PE_varnish_complementarity"]["per_group_rho_PE_V"] = pgcv

# ---- H3: the log10 PE-count outcome is largely an identity -------------------
H["H3_log_pe_count_identity"] = {}
vec = np.log10(np.clip(PEc, 1e-12, None))
f = fe_fit(vec, [ltot])
f_pepct = fe_fit(np.log10(np.clip(PE, 1e-12, None)), [ltot])
H["H3_log_pe_count_identity"]["FE_log10_PEcount_on_log10_total"] = {
    "slope": round(f["beta"][1], 3), "se": round(f["se"][1], 3),
    "t": round(f["t"][1], 2), "p": round(f["p"][1], 4)}
H["H3_log_pe_count_identity"]["FE_log10_PEpercent_on_log10_total"] = {
    "slope": round(f_pepct["beta"][1], 3), "se": round(f_pepct["se"][1], 3),
    "t": round(f_pepct["t"][1], 2), "p": round(f_pepct["p"][1], 4)}
H["H3_log_pe_count_identity"]["identity_check_slope_minus_1"] = round(f["beta"][1] - 1.0, 3)
H["H3_log_pe_count_identity"]["note"] = ("log10 PE_count = log10 total + log10(PE%/100) "
                                         "exactly (verified to fp error), so the reported "
                                         "+1.17 slope is 1.0 mechanically plus the PE% "
                                         "trend; it is not independent evidence.")

# ---- H4: serial dependence within depth profiles ----------------------------
H["H4_serial_dependence"] = {}
# order by depth within group, lag-1 autocorr of PE and of FE residuals
acP, acR, acX = [], [], []
for k in groups:
    s = np.where(g == k)[0]
    o = s[np.argsort(dmid[s])]
    x = ltot[o]
    y = PE[o]
    xd = x - x.mean()
    yd = y - y.mean()
    b = (xd @ yd) / (xd @ xd)
    r = yd - b * xd
    if len(o) > 3:
        acP.append(np.corrcoef(y[:-1], y[1:])[0, 1])
        acR.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        acX.append(np.corrcoef(x[:-1], x[1:])[0, 1])
H["H4_serial_dependence"]["lag1_autocorr_PE"] = [round(float(v), 3) for v in acP]
H["H4_serial_dependence"]["lag1_autocorr_FE_residual"] = [round(float(v), 3) for v in acR]
H["H4_serial_dependence"]["lag1_autocorr_logtot"] = [round(float(v), 3) for v in acX]
H["H4_serial_dependence"]["mean_lag1_residual"] = round(float(np.mean(acR)), 3)
# block permutation: cyclically shift PE within each group (destroys the
# depth-concentration alignment while preserving each group's serial structure)
rng = np.random.default_rng(99)
obs = float(np.corrcoef(dm(ltot), dm(PE))[0, 1])
null = np.empty(5000)
for i in range(5000):
    yp = PE.copy()
    for k in groups:
        s = np.where(g == k)[0]
        o = s[np.argsort(dmid[s])]
        sh = rng.integers(1, len(o))
        yp[o] = np.roll(PE[o], int(sh))
    null[i] = np.corrcoef(dm(ltot), dm(yp))[0, 1]
H["H4_serial_dependence"]["cyclic_shift_permutation"] = {
    "observed_within_r": round(obs, 3),
    "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / 5001),
    "null_q025": round(float(np.quantile(null, 0.025)), 3),
    "null_q975": round(float(np.quantile(null, 0.975)), 3)}

# ---- H5: group_05 interaction test ------------------------------------------
H["H5_group05_interaction"] = {}
i5 = (g == "group_05").astype(float)
f = fe_fit(PE, [ltot, ltot * i5])
H["H5_group05_interaction"]["FE_PE_ltot_plus_ltot_x_group05"] = {
    "slope_ltot": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
    "p_ltot": round(f["p"][1], 4),
    "delta_slope_group05": round(f["beta"][2], 2), "se_delta": round(f["se"][2], 2),
    "p_interaction": round(f["p"][2], 4)}
f = fe_fit(PE, [ltot], g != "group_05")
H["H5_group05_interaction"]["FE_without_group05"] = {
    "slope": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
    "p": round(f["p"][1], 4), "G": f["G"]}
# group_05 depth is the dominant gradient there
s5 = g == "group_05"
H["H5_group05_interaction"]["group05_depth_vs_total"] = {
    "rho_depth_logtot": round(float(st.spearmanr(dmid[s5], ltot[s5]).statistic), 3),
    "rho_depth_PE": round(float(st.spearmanr(dmid[s5], PE[s5]).statistic), 3),
    "rho_logtot_PE": round(float(st.spearmanr(ltot[s5], PE[s5]).statistic), 3),
    "FE_PE_on_depth_group05": None}
_x = dmid[s5]
_y = PE[s5]
_X = np.c_[np.ones(len(_x)), _x]
_b = np.linalg.pinv(_X.T @ _X) @ _X.T @ _y
_r = _y - _X @ _b
_cov = (float(_r @ _r) / (len(_x) - 2)) * np.linalg.pinv(_X.T @ _X)
_se = math.sqrt(_cov[1, 1])
_t = _b[1] / _se
H["H5_group05_interaction"]["FE_PE_on_depth_group05"] = {
    "slope": round(float(_b[1]), 2), "se": round(float(_se), 2),
    "t": round(float(_t), 2),
    "p": round(float(2 * st.t.sf(abs(_t), len(_x) - 2)), 4), "df": len(_x) - 2}

# ---- H6: within-group rho excluding the low-abundance tail -------------------
H["H6_low_abundance_tail"] = {}
for lab, mask in [("all", np.ones(N, bool)), ("tot>1000", tot > 1000),
                  ("tot>500", tot > 500), ("tot<=500", tot <= 500)]:
    rh = {}
    for k in groups:
        s = (g == k) & mask
        if s.sum() >= 3:
            rh[k] = {"n": int(s.sum()),
                     "rho": round(float(st.spearmanr(ltot[s], PE[s]).statistic), 3),
                     "p": round(float(st.spearmanr(ltot[s], PE[s]).pvalue), 3)}
    f = fe_fit(PE, [ltot], mask)
    H["H6_low_abundance_tail"][lab] = {
        "per_group_rho": rh,
        "FE_slope": round(f["beta"][1], 2), "se": round(f["se"][1], 2),
        "p": round(f["p"][1], 4), "G": f["G"], "n": f["n"],
        "mean_rho": round(float(np.mean([v["rho"] for v in rh.values()])), 3) if rh else None,
        "n_positive": int(sum(v["rho"] > 0 for v in rh.values())),
        "n_negative": int(sum(v["rho"] < 0 for v in rh.values()))}
# the low tail is also the deepest-chl samples in some groups
lo = tot <= 500
H["H6_low_abundance_tail"]["low_tail_characteristics"] = {
    "samples": sid[lo].tolist(),
    "total": tot[lo].astype(int).tolist(),
    "chlorophyll_a": m["chlorophyll_a_ug_L"].to_numpy(float)[lo].tolist(),
    "depth_mid": dmid[lo].round(2).tolist(),
    "varnish_pct": V[lo].round(2).tolist(),
    "PE_pct": PE[lo].round(2).tolist()}

# ---- H7: two-stage group mean slope, sensitivity -----------------------------
H["H7_two_stage"] = {}
sl = {}
for k in groups:
    s = g == k
    sl[k] = float(np.polyfit(ltot[s], PE[s], 1)[0])
v = np.array(list(sl.values()))
H["H7_two_stage"] = {
    "group_slopes": {k: round(x, 2) for k, x in sl.items()},
    "mean": round(float(v.mean()), 2), "sd": round(float(v.std(ddof=1)), 2),
    "t": round(float(st.ttest_1samp(v, 0).statistic), 2),
    "p": round(float(st.ttest_1samp(v, 0).pvalue), 4),
    "sign_test_p_one_sided": (1 + 4 + 1) / 32,
    "jackknife_group_mean": {k: round(float(np.delete(v, i).mean()), 2)
                             for i, k in enumerate(sl)},
    "note": ("the two-stage mean uses OLS slopes; two of five groups (group_01 "
             "52.8, group_04 63.3) have n=5 and their slopes lean on a single "
             "low-abundance endpoint.")}

# ---- H8: rank-based / robust FE slope and outcome transforms ----------------
H["H8_robust_transforms"] = {}
for nm, vec in [("PE_probit_logit_pct", np.log(np.clip(PE, .01, 99.99) / (100 - np.clip(PE, .01, 99.99)))),
                ("log10_PE_pct", np.log10(np.clip(PE, 1e-3, None))),
                ("PE_rank_within_group", None),
                ("varnish_pct", V),
                ("log_ratio_pseudo0.5", np.log((PE + .5) / (V + .5)))]:
    if vec is None:
        continue
    f = fe_fit(vec, [ltot])
    H["H8_robust_transforms"][nm] = {
        "slope": round(f["beta"][1], 3), "se": round(f["se"][1], 3),
        "t": round(f["t"][1], 2), "p": round(f["p"][1], 4)}
# rank-within-group outcome
ranks = np.full(N, np.nan)
for k in groups:
    s = g == k
    ranks[s] = st.rankdata(PE[s])
f = fe_fit(ranks, [ltot])
H["H8_robust_transforms"]["PE_rank_within_group"] = {
    "slope": round(f["beta"][1], 3), "se": round(f["se"][1], 3),
    "t": round(f["t"][1], 2), "p": round(f["p"][1], 4)}

RES["h_addendum"] = H
with open(OUT, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("addendum written")
for sec in H:
    print(sec, "->", json.dumps(H[sec])[:400])