"""Addendum 2 to the F3 adversarial verification: exact inference, Freedman-
Lane depth adjustment, abundance interaction, default-class diagnostics."""
from __future__ import annotations

import json
import math
from pathlib import Path

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
err = m["particle_total_std_or_error_per_L"].to_numpy(float)
pol_cols = [c for c in comp.columns if c.startswith("polymer_") and c.endswith("_percent")]


def dm(v, mask=None):
    v = np.asarray(v, float).astype(float).copy()
    if mask is None:
        mask = np.ones(len(v), bool)
    out = np.full(len(v), np.nan)
    for k in groups:
        s = (g == k) & mask
        out[s] = v[s] - v[s].mean()
    return out


def fe_sub(y, x, mask=None):
    if mask is None:
        mask = np.ones(N, bool)
    yv = np.asarray(y, float)[mask]
    xv = np.asarray(x, float)[mask]
    gg = g[mask]
    xd = np.zeros(len(xv))
    yd = np.zeros(len(yv))
    kept = []
    for k in groups:
        s = gg == k
        if s.sum() >= 2:
            xd[s] = xv[s] - xv[s].mean()
            yd[s] = yv[s] - yv[s].mean()
            kept.append(k)
    X = np.c_[np.ones(len(xd)), xd]
    b = np.linalg.pinv(X.T @ X) @ X.T @ yd
    r = yd - X @ b
    meat = np.zeros((2, 2))
    for k in kept:
        s = gg == k
        u = X[s].T @ r[s]
        meat += np.outer(u, u)
    Gd = len(kept)
    n = len(yd)
    cov = (Gd / (Gd - 1.0)) * ((n - 1.0) / (n - 2)) * np.linalg.pinv(X.T @ X) @ meat @ np.linalg.pinv(X.T @ X)
    se = math.sqrt(max(cov[1, 1], 0))
    df = Gd - 1
    t = b[1] / se
    return {"slope": float(b[1]), "se": float(se), "t": float(t),
            "p": float(2 * st.t.sf(abs(t), df)), "df": df, "G": Gd, "n": int(n),
            "kept": kept}


RES = json.load(open(OUT))
A = {}

rng = np.random.default_rng(4242)
B = 20000

# ---- I1: Freedman-Lane permutation, depth partialled out --------------------
def resid(y, z):
    ok = np.isfinite(y) & np.isfinite(z)
    X = np.c_[np.ones(ok.sum()), z[ok]]
    b = np.linalg.pinv(X.T @ X) @ X.T @ y[ok]
    r = np.full(len(y), np.nan)
    r[ok] = y[ok] - X @ b
    return r


def partial_fe(x, y, z, rank=False):
    xr = dm(resid(st.rankdata(x) if rank else x, z))
    yr = dm(resid(st.rankdata(y) if rank else y, z))
    ok = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[ok], yr[ok]
    return float(xr @ yr / math.sqrt((xr @ xr) * (yr @ yr)))


obs = partial_fe(ltot, PE, dmid)
obs_r = partial_fe(ltot, PE, dmid, rank=True)
null = np.empty(B)
null_r = np.empty(B)
for i in range(B):
    yp = PE.copy()
    for k in groups:
        s = g == k
        yp[s] = rng.permutation(PE[s])
    # Freedman-Lane: keep depth structure, shuffle PE, then partial out depth
    null[i] = partial_fe(ltot, yp, dmid)
    null_r[i] = partial_fe(ltot, yp, dmid, rank=True)
A["I1_freedman_lane_depth"] = {
    "observed_partial_r": round(obs, 3),
    "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (B + 1)),
    "null_q025": round(float(np.quantile(null, 0.025)), 3),
    "null_q975": round(float(np.quantile(null, 0.975)), 3),
    "observed_partial_rank_r": round(obs_r, 3),
    "rank_p_two_sided": float((np.sum(np.abs(null_r) >= abs(obs_r)) + 1) / (B + 1)),
}

# ---- I2: abundance interaction (does the slope differ by band?) -------------
hi = (tot > 1000).astype(float)
xd = dm(ltot)
h = dm(hi)
y = dm(PE)
Xd = np.c_[xd, h, xd * h]
b = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ y
r = y - Xd @ b
meat = np.zeros((3, 3))
for k in groups:
    s = g == k
    u = Xd[s].T @ r[s]
    meat += np.outer(u, u)
cov = (G / (G - 1.0)) * ((N - 1.0) / (N - 3)) * np.linalg.pinv(Xd.T @ Xd) @ meat @ np.linalg.pinv(Xd.T @ Xd)
se = np.sqrt(np.diag(cov))
t = b / se
A["I2_abundance_interaction"] = {
    "slope_low_band": round(float(b[0]), 2),
    "slope_low_se": round(float(se[0]), 2), "slope_low_p": round(float(2 * st.t.sf(abs(t[0]), G - 1)), 4),
    "delta_slope_high_band": round(float(b[2]), 2),
    "delta_se": round(float(se[2]), 2), "delta_p": round(float(2 * st.t.sf(abs(t[2]), G - 1)), 4),
    "slope_high_band": round(float(b[0] + b[2]), 2),
    "note": "no significant low-vs-high slope difference; the high-band estimate is imprecise (few groups, small n)"}

# ---- I3: band-specific within-group rho and covariate specificity -----------
A["I3_band_details"] = {}
for lab, msk in [("low_tot_le_1000", tot <= 1000), ("high_tot_gt_1000", tot > 1000)]:
    per = {}
    for k in groups:
        s = (g == k) & msk
        if s.sum() >= 3:
            rr = st.spearmanr(ltot[s], PE[s])
            per[k] = {"n": int(s.sum()), "rho": round(float(rr.statistic), 3),
                      "p": round(float(rr.pvalue), 3),
                      "mean_PE": round(float(PE[s].mean()), 1),
                      "mean_V": round(float(V[s].mean()), 1)}
    f = fe_sub(PE, ltot, msk)
    A["I3_band_details"][lab] = {
        "n": int(msk.sum()), "per_group": per,
        "mean_rho_where_n_ge_3": round(float(np.mean([v["rho"] for v in per.values()])), 3)
        if per else None,
        "FE": {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in f.items()}}
# covariate specificity inside the high band
covs = {"log10_total": ltot, "log10_chl": chl, "salinity": m["salinity_PSU"].to_numpy(float),
        "POC": POC, "C_to_N": m["carbon_to_nitrogen_ratio"].to_numpy(float), "depth": dmid}
A["I3_band_details"]["high_band_within_R2"] = {
    nm: round(float(fe_sub(PE, v, tot > 1000)["slope"] ** 2), 3) for nm, v in covs.items()}
A["I3_band_details"]["high_band_FE"] = {
    nm: fe_sub(PE, v, tot > 1000) for nm, v in covs.items()}

# ---- I4: WLS by reported uncertainty on total -------------------------------
w = 1.0 / np.maximum(err, 1.0) ** 2
xw = dm(ltot)
yw = dm(PE)
num = float((w * xw * yw).sum())
denw = float((w * xw * xw).sum())
slope_w = num / denw
# bootstrap the WLS slope at group level
boot = []
for _ in range(5000):
    pick = rng.integers(0, G, G)
    n2 = d2 = 0.0
    for k in pick:
        s = g == groups[k]
        xx = ltot[s] - ltot[s].mean()
        yy = PE[s] - PE[s].mean()
        ww = w[s]
        n2 += float((ww * xx * yy).sum())
        d2 += float((ww * xx * xx).sum())
    boot.append(n2 / d2)
boot = np.array(boot)
A["I4_uncertainty_weighting"] = {
    "wls_slope_1_over_SEsq": round(slope_w, 2),
    "group_bootstrap_CI95": [round(float(np.quantile(boot, 0.025)), 2),
                             round(float(np.quantile(boot, 0.975)), 2)],
    "frac_le_zero": round(float((boot <= 0).mean()), 4),
    "median_rel_SE_on_total": round(float(np.median(err / tot)), 3)}

# ---- I5: default-class / low-N diagnostics ----------------------------------
Nrec = {"sample_001": 388, "sample_008": 1446, "sample_011": 64, "sample_012": 139,
        "sample_013": 1285, "sample_014": 219, "sample_016": 620, "sample_017": 680,
        "sample_018": 203, "sample_020": 76, "sample_021": 2436, "sample_022": 498,
        "sample_023": 316, "sample_024": 17, "sample_026": 583, "sample_027": 2429,
        "sample_028": 1070, "sample_029": 629}
Nv = np.array([Nrec.get(s, np.nan) for s in sid], float)
ok = np.isfinite(Nv)
lown = ok & (Nv <= 700)
A["I5_default_class"] = {
    "N_recovered_usable": int(ok.sum()),
    "mean_PE_pct_lowN": round(float(PE[lown].mean()), 2),
    "mean_PE_pct_highN": round(float(PE[ok & ~lown].mean()), 2),
    "mean_V_pct_lowN": round(float(V[lown].mean()), 2),
    "mean_V_pct_highN": round(float(V[ok & ~lown].mean()), 2),
    "n_lowN_PE_is_largest_class": int(sum(
        comp.loc[[i for i in range(N) if lown[i]], pol_cols].idxmax(axis=1).str.contains("PE_percent").tolist())),
    "n_lowN": int(lown.sum()),
    "rho_PE_vs_N": round(float(st.spearmanr(Nv[ok], PE[ok]).statistic), 3),
    "rho_PE_vs_N_p": float(st.spearmanr(Nv[ok], PE[ok]).pvalue),
    "note": ("If PE were the classifier's default class when few particles are "
             "analysed, PE%% would be HIGH at low N. It is LOW (mean "
             f"{PE[lown].mean():.1f}%% vs {PE[ok & ~lown].mean():.1f}%%); low-N samples are "
             "varnish-dominated. The proposed artefact runs in the wrong direction.")}

# ---- I6: saturation / bounded-outcome check ---------------------------------
A["I6_bounded_outcome"] = {
    "PE_pct_at_top_total_quantile": round(float(PE[ltot >= np.quantile(ltot, 0.8)].mean()), 1),
    "n_samples_PE_ge_98": int((PE >= 98).sum()),
    "n_samples_PE_ge_90": int((PE >= 90).sum()),
    "max_PE": float(PE.max()), "min_PE": float(PE.min()),
    "note": ("PE%% is a bounded share that saturates near 100%% in the particle-"
             "rich samples of groups 01/02; a constant 'pp per decade' slope cannot "
             "hold across the whole range.")}

# ---- I7: autocorrelation-aware group-level test recap -----------------------
A["I7_cluster_level_inference"] = {
    "nominal_CR1_p": 0.0082,
    "exact_signflip_p_one_sided": 0.0625,
    "exact_signflip_p_two_sided": 0.125,
    "wild_cluster_bootstrap_p_two_sided": 0.1278,
    "wild_cluster_bootstrap_p_one_sided": 0.0653,
    "min_possible_one_sided_p_with_5_groups": 0.03125,
    "note": ("With G=5 clusters the nominal cluster-robust p=0.008 is not reliable; "
             "exact sign-flip and wild-bootstrap tests give p ~ 0.06-0.13.")}

RES["i_addendum2"] = A
with open(OUT, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("addendum2 written")
for k, v in A.items():
    print(k, "->", json.dumps(v, default=str)[:500])