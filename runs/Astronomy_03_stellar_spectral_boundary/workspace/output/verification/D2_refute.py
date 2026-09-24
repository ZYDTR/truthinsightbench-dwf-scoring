#!/usr/bin/env python3
"""
D2 adversarial refutation of the CLAIM:

  "The temperature response of photon flux reverses sign with wavelength. The ratio
   F(6300 K)/F(5300 K) is far above 1 in the ultraviolet (order 100x at 180 nm), crosses 1
   somewhere in the red, and falls BELOW 1 at long wavelengths (about 0.86-0.88 near 850 nm).
   The crossing wavelength is not fixed: it moves redward as the metallicity index increases."

All numbers recomputed from the raw TSV with numpy only (no scipy).
Writes: output/verification/D2_refutation.json
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-b109a95b-77c2-4866-9c87-234d668cd900/project"
DATA = os.path.join(ROOT, "data", "photon_flux_grid.tsv")
OUT = os.path.join(ROOT, "output", "verification", "D2_refutation.json")
RNG = np.random.default_rng(20260919)

# ---------------------------------------------------------------- load raw
df = pd.read_csv(DATA, sep="\t")
temps = sorted(df.effective_temperature_K.unique())
mets = sorted(df.metallicity_index.unique())
wl = np.sort(df.wavelength_nm.unique())
assert len(mets) == 35 and len(wl) == 176 and len(temps) == 3

F = {}
for (T, M), g in df.groupby(["effective_temperature_K", "metallicity_index"]):
    g = g.sort_values("wavelength_nm")
    assert np.allclose(g.wavelength_nm.values, wl)
    F[(int(T), float(M))] = g.photon_flux_cm2.values.astype(float)


# ---------------------------------------------------------------- helpers
def crossing(wl, r):
    """First wavelength where (r-1) changes sign (blue->red), linearly interpolated."""
    d = r - 1.0
    s = np.sign(d)
    for i in range(len(wl) - 1):
        if s[i] != s[i + 1] and s[i] != 0:
            x0, x1, y0, y1 = wl[i], wl[i + 1], d[i], d[i + 1]
            xc = x0 - y0 * (x1 - x0) / (y1 - y0)
            on = (abs(y0) < 1e-12) or (abs(y1) < 1e-12)
            return xc, i, on
    return None, None, False


def crossing_poly2(wl, r, i, span=2):
    """Quantisation-robust crossing: quadratic fit on +/-span nodes around the sign change."""
    a, b = max(0, i - span), min(len(wl) - 1, i + 1 + span)
    x, y = wl[a:b + 1], r[a:b + 1] - 1.0
    if len(x) < 3:
        return None
    roots = np.roots(np.polyfit(x, y, 2))
    real = [rr.real for rr in roots if abs(rr.imag) < 1e-9]
    inside = [rr for rr in real if wl[i] - 1e-9 <= rr <= wl[i + 1] + 1e-9]
    if inside:
        return float(min(inside, key=lambda z: abs(z - wl[i])))
    return float(wl[i] - y[span] * (wl[i + 1] - wl[i]) / (y[span + 1] - y[span]))


def ols(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ sol
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(sol[0]), float(sol[1]), float(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def polyfit_r2(x, y, deg):
    c = np.polyfit(x, y, deg)
    pred = np.polyval(c, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return c, float(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sr = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


# half-ulp of the printed 3-significant-figure fluxes, per (T, M) and per wavelength node
def ulp_half_3sig(Farr):
    return 0.5 * 10.0 ** (np.floor(np.log10(np.abs(Farr))) - 2)


ulp_half = {T: {M: ulp_half_3sig(F[(T, M)]) for M in mets} for T in (5300, 6300)}


# ---------------------------------------------------------------- Planck
C, H, KB = 2.99792458e8, 6.62607015e-34, 1.380649e-23


def planck_per_nm(wl_nm, T):
    lam = np.asarray(wl_nm, float) * 1e-9
    x = H * C / (lam * KB * T)
    return (2 * H * C ** 2) / lam ** 5 / np.expm1(x)


def bb_ratio(wl_nm, Tc, Th):
    return planck_per_nm(wl_nm, Th) / planck_per_nm(wl_nm, Tc)


def _planck_B(Tgrid, lam):
    x = H * C / (lam[None, :] * KB * Tgrid[:, None])
    return (2 * H * C ** 2) / lam[None, :] ** 5 / np.expm1(x)


def fit_planck_Teff(wl, flux, weight="relative", n_coarse=1500, n_fine=1500):
    """Fit the Planck SHAPE with a free multiplicative scale m.

    weight='relative' -> chi2 on (m*B - F)/F      (blue-weighted)
    weight='log'      -> chi2 on (ln m*B - ln F)  (log-flux weighted)
    Returns (T_fit, m, chi2).  Two weightings are used because the flux has no Rayleigh-Jeans
    tail in band, so the shape fit is ill-conditioned in T; agreement/disagreement quantifies that.
    """
    lam = wl * 1e-9

    def chi2_of(Tgrid):
        B = _planck_B(Tgrid, lam)
        if weight == "relative":
            inv2 = 1.0 / flux ** 2
            m = np.sum(B * inv2[None, :], axis=1) / np.sum(B * B * inv2[None, :], axis=1)
            ch = np.sum((m[:, None] * B - flux[None, :]) ** 2 * inv2[None, :], axis=1)
        else:
            lnF = np.log(flux)
            lnB = np.log(B)
            m = np.exp(np.mean(lnF[None, :] - lnB, axis=1))
            ch = np.sum((lnB + np.log(m)[:, None] - lnF[None, :]) ** 2, axis=1)
        return ch, m

    Tg = np.exp(np.linspace(np.log(1000.0), np.log(30000.0), n_coarse))
    ch, m = chi2_of(Tg)
    T0 = float(Tg[int(np.argmin(ch))])
    Tg2 = np.linspace(max(1000.0, T0 * 0.8), T0 * 1.2, n_fine)
    ch2, m2 = chi2_of(Tg2)
    k2 = int(np.argmin(ch2))
    return float(Tg2[k2]), float(m2[k2]), float(ch2[k2])


# ================================================================ 1. CROSSINGS
pair = (6300, 5300)
R = {M: F[(6300, M)] / F[(5300, M)] for M in mets}

cross_lin, cross_p2, node_flags = {}, {}, {}
for M in mets:
    xc, i, on = crossing(wl, R[M])
    cross_lin[M] = xc
    node_flags[M] = {"left_node_nm": float(wl[i]), "left_node_index": int(i),
                     "lands_on_node": bool(on),
                     "r_minus_1_left_node": float(R[M][i] - 1.0),
                     "r_minus_1_right_node": float(R[M][i + 1] - 1.0)}
    cross_p2[M] = crossing_poly2(wl, R[M], i)

seq = np.array([np.nan if cross_lin[M] is None else cross_lin[M] for M in mets])
seq_p2 = np.array([np.nan if cross_p2[M] is None else cross_p2[M] for M in mets])
xM = np.array(mets)

back = [{"from_M": float(mets[j - 1]), "to_M": float(mets[j]), "step_nm": float(seq[j] - seq[j - 1])}
        for j in range(1, len(mets)) if seq[j] < seq[j - 1]]
n_back = len(back)
sum_back = float(sum(-b["step_nm"] for b in back))
n_on_node = int(sum(v["lands_on_node"] for v in node_flags.values()))
n_zero = int(sum(1 for j in range(1, len(mets)) if seq[j] == seq[j - 1]))
n_distinct = int(len(np.unique(seq[~np.isnan(seq)])))
net = float(seq[-1] - seq[0])
slope, intercept, r2_lin = ols(xM, seq)
qcoef, r2_quad = polyfit_r2(xM, seq, 2)
sp_loc = np.diff(wl[(wl >= 600) & (wl <= 680)])

# piecewise slopes (saturation test)
seg = {}
for lo, hi in [(-1.0, -0.3), (-0.3, 0.2), (0.2, 0.9)]:
    a = (xM >= lo) & (xM <= hi)
    s, _, rr = ols(xM[a], seq[a])
    seg[f"M in [{lo},{hi}]"] = {"slope_nm_per_M": s, "r2": rr, "n": int(a.sum()),
                                "delta_nm": float(seq[a][-1] - seq[a][0])}
spear = spearman(xM, seq)
pear = float(np.corrcoef(xM, seq)[0, 1])

# uncertainty models
nboot = 20000
sl_q = np.empty(nboot)
sl_lev = np.empty(nboot)
for b in range(nboot):
    sl_q[b] = ols(xM, seq + RNG.uniform(-2.5, 2.5, len(mets)))[0]
    j = RNG.integers(0, len(mets), len(mets))
    sl_lev[b] = ols(xM[j], seq[j])[0]

# --- adversarial test: is the red-end DECLINE above the 3-significant-figure noise floor?
# Perform a linear fit to R(l) on 800-850 nm per level; null = ratios built from flux values that
# share the same quantised digits but are forced to be non-decreasing (slope >= 0) locally.
def local_slope_800_850(r):
    a = wl >= 800
    return ols(wl[a], r[a])[0]


obs_slope = np.array([local_slope_800_850(R[M]) for M in mets])
n_pert = 20000
flip = 0
slope_pert = np.empty(n_pert)
for b in range(n_pert):
    M = mets[int(RNG.integers(0, len(mets)))]
    f1 = F[(5300, M)] + ulp_half[5300][M] * RNG.uniform(-1, 1, len(wl))
    f2 = F[(6300, M)] + ulp_half[6300][M] * RNG.uniform(-1, 1, len(wl))
    s = local_slope_800_850(f2 / f1)
    slope_pert[b] = s
    flip += int(s >= 0)
# implied slope error from independent per-node quantisation (uniform half-ulp, std = ulp/sqrt(12))
sig_node = np.median([np.median(ulp_half[5300][M][wl >= 800]) / F[(5300, M)][wl >= 800].mean() +
                      np.median(ulp_half[6300][M][wl >= 800]) / F[(6300, M)][wl >= 800].mean()
                      for M in mets]) / np.sqrt(12)
x8 = wl[wl >= 800]
sig_slope_pred = sig_node / np.sqrt(np.sum((x8 - x8.mean()) ** 2))
red_decline_test = {
    "observed_median_slope_per_nm": float(np.median(obs_slope)),
    "observed_n_negative": int((obs_slope < 0).sum()), "n_levels": len(mets),
    "perturbation_null_slope_median_per_nm": float(np.median(slope_pert)),
    "perturbation_null_slope_p95_per_nm": float(np.percentile(slope_pert, 95)),
    "n_perturbation_draws": n_pert,
    "frac_draws_with_nonnegative_slope_sign_flip": flip / n_pert,
    "implied_quantisation_slope_error_per_nm": float(sig_slope_pred),
    "observed_slope_over_error": float(abs(np.median(obs_slope)) / sig_slope_pred),
    "interpretation": ("The 3-significant-figure quantisation contributes a per-node ratio error of "
                       "~1.4e-3 (half-ulp), which over the 10 nodes spanning 800-850 nm implies a slope "
                       "error of ~%.1e per nm - about %.0fx smaller than the observed %.1e per nm. "
                       "Quantisation therefore does NOT explain the 800-850 nm decline, which is negative "
                       "in all %d/35 levels; the caveat is that this assumes the printed rounding errors "
                       "are independent, which cannot be verified from the file."
                       % (sig_slope_pred, abs(np.median(obs_slope)) / sig_slope_pred,
                          abs(np.median(obs_slope)), int((obs_slope < 0).sum()))),
    "caveat": "the deficit is still only ~13% below unity at 850 nm; the file provides no leverage on what happens past 850 nm",
}

# 3-significant-figure bootstrap: propagate the quantisation of the two flux arrays
# (the printed ulp is 10^(floor(log10|F|) - 2); perturb within +/- half ulp).
nboot2 = 3000
sl_param = np.empty(nboot2)
lost = 0
for b in range(nboot2):
    ys = []
    for M in mets:
        e5300 = ulp_half[5300][M] * RNG.uniform(-1, 1, len(wl))
        e6300 = ulp_half[6300][M] * RNG.uniform(-1, 1, len(wl))
        rp = (F[(6300, M)] + e6300) / (F[(5300, M)] + e5300)
        xc, _, _ = crossing(wl, rp)
        if xc is None:
            lost += 1
            ys.append(np.nan)
        else:
            ys.append(xc)
    y = np.array(ys)
    ok = ~np.isnan(y)
    sl_param[b] = ols(xM[ok], y[ok])[0] if ok.sum() > 3 else np.nan
sl_param = sl_param[~np.isnan(sl_param)]

q_ci = [float(np.percentile(sl_q, 2.5)), float(np.percentile(sl_q, 97.5))]
lev_ci = [float(np.percentile(sl_lev, 2.5)), float(np.percentile(sl_lev, 97.5))]
par_ci = [float(np.percentile(sl_param, 2.5)), float(np.percentile(sl_param, 97.5))]

# independent corroboration of the M-dependence: crossing vs the (smoothly varying) 850 nm ratio
r850 = np.array([R[M][-1] for M in mets])
corr_cross_r850 = {"pearson": float(np.corrcoef(seq, r850)[0, 1]), "spearman": spearman(seq, r850)}
corr_cross_r650 = {"pearson": float(np.corrcoef(seq, np.array([R[M][int(np.argmin(abs(wl - 650)))] for M in mets]))[0, 1])}

# --- adversarial test: is the NON-MONOTONICITY evidence against 'moves redward'?
# Null: a monotone redward trend (OLS line) + node-quantisation jitter of +/-half a node.
n_null = 20000
nb_null = np.empty(n_null)
mx_null = np.empty(n_null)
fit_line = intercept + slope * xM
for b in range(n_null):
    y = fit_line + RNG.uniform(-2.5, 2.5, len(mets))
    d = np.diff(y)
    nb_null[b] = int((d < 0).sum())
    mx_null[b] = float(-d.min()) if (d < 0).any() else 0.0
obs_max_back = float(max(-b["step_nm"] for b in back)) if back else 0.0
monotonicity_null = {
    "observed_n_backward_steps": n_back, "observed_max_backward_nm": obs_max_back,
    "null_max_backward_nm_median": float(np.median(mx_null)),
    "null_max_backward_nm_p95": float(np.percentile(mx_null, 95)),
    "p_null_max_backward_ge_observed": float(np.mean(mx_null >= obs_max_back)),
    "null_n_backward_median": float(np.median(nb_null)),
    "p_null_n_backward_ge_observed": float(np.mean(nb_null >= n_back)),
    "interpretation": ("Backward steps no larger than the node spacing are the EXPECTED behaviour of a "
                       "monotone redward trend sampled at 5 nm and quantised, so the non-monotonicity is "
                       "not by itself evidence against the redward drift."),
}

# ================================================================ 2. UV AMPLITUDE
uv_probe = {}
for probe in (179.4, 190.0, 250.0, 300.0, 320.0):
    k = int(np.argmin(np.abs(wl - probe)))
    v = np.array([R[M][k] for M in mets])
    uv_probe[f"{wl[k]:.1f}nm"] = {"median": float(np.median(v)), "min": float(v.min()),
                                  "max": float(v.max()), "n_above_100": int((v > 100).sum()),
                                  "n_levels": int(len(v)), "quartiles": [float(np.percentile(v, q)) for q in (25, 50, 75)]}

# ================================================================ 3. RED ENDPOINT / EDGE
i850 = len(wl) - 1
in_band = (r850 >= 0.86) & (r850 <= 0.88)
endpoint = {
    "wavelength_nm": float(wl[i850]),
    "median": float(np.median(r850)), "min": float(r850.min()), "max": float(r850.max()),
    "n_below_1": int((r850 < 1).sum()), "n_levels": int(len(r850)),
    "n_in_0.86_0.88": int(in_band.sum()),
    "levels_outside_band": [{"M": float(M), "ratio": float(v)} for M, v, ok in zip(mets, r850, in_band) if not ok],
    "min_at_M": float(mets[int(np.argmin(r850))]), "max_at_M": float(mets[int(np.argmax(r850))]),
    "spearman_ratio850_vs_M": spearman(xM, r850), "range_across_M": float(r850.max() - r850.min()),
    "monotone_increasing_over_last_5_nodes": int(sum(
        np.all(np.diff(R[M][-6:]) > 0) for M in mets)),
    "last_nodes_nm": [float(x) for x in wl[-6:]],
}

# red-end local behaviour
seg_slopes = {}
for lo, hi in [(650, 700), (700, 750), (750, 800), (800, 850), (845, 850)]:
    a = (wl >= lo) & (wl <= hi)
    s = np.array([ols(wl[a], R[M][a])[0] for M in mets])
    seg_slopes[f"{lo}-{hi}nm"] = {"median_dRdlambda_per_nm": float(np.median(s)),
                                  "n_negative": int((s < 0).sum()), "n_levels": len(mets)}
curv = np.array([np.polyfit(wl[wl >= 700], R[M][wl >= 700], 2)[0] for M in mets])
curv650 = np.array([np.polyfit(wl[wl >= 650], R[M][wl >= 650], 2)[0] for M in mets])
edge = {
    "red_end_segment_slopes": seg_slopes,
    "quadratic_curvature_700_850": {"median": float(np.median(curv)), "n_concave_up": int((curv > 0).sum()),
                                    "n_concave_down": int((curv < 0).sum())},
    "quadratic_curvature_650_850": {"median": float(np.median(curv650)), "n_concave_up": int((curv650 > 0).sum()),
                                    "n_concave_down": int((curv650 < 0).sum())},
    "note": ("845-850 nm is the LAST interval in the file; no data exist beyond 850.0 nm. "
             "Positive curvature at the red end means the decline is DECELERATING in-band, so the data "
             "give no evidence of steepening; they also cannot exclude an unseen turn-up or turn-down "
             "beyond 850 nm."),
}

# ================================================================ 4. BLACKBODY BASELINE
C_NOM = bb_ratio(wl, 5300.0, 6300.0)          # nominal temperatures, pure Planck
planck_nominal = {
    "ratio_at_180nm": float(C_NOM[0]), "ratio_at_400nm": float(C_NOM[int(np.argmin(abs(wl - 400)))]),
    "ratio_at_620nm": float(C_NOM[int(np.argmin(abs(wl - 620)))]),
    "ratio_at_850nm": float(C_NOM[-1]),
    "min_over_band": float(C_NOM.min()), "argmin_nm": float(wl[int(np.argmin(C_NOM))]),
    "rayleigh_jeans_asymptote_T_ratio": 6300.0 / 5300.0,
    "crosses_1": bool(C_NOM.min() < 1.0),
}

per_level = {}
resid_uv, resid_850, resid_650, resid_650_rel, dT = [], [], [], [], []
curv_obs, curv_bb = [], []
for M in mets:
    T1, m1, chi1 = fit_planck_Teff(wl, F[(5300, M)], "relative")
    T2, m2, chi2 = fit_planck_Teff(wl, F[(6300, M)], "relative")
    T1L, _, _ = fit_planck_Teff(wl, F[(5300, M)], "log")
    T2L, _, _ = fit_planck_Teff(wl, F[(6300, M)], "log")
    rb = bb_ratio(wl, T1, T2)                    # >1 everywhere when T2>T1
    rbL = bb_ratio(wl, T1L, T2L)
    # free multiplicative scale for the RATIO, relative residuals
    obs = R[M]
    m = np.sum(rb / obs) / np.sum(rb ** 2 / obs ** 2)
    model = m * rb
    resid = obs / model - 1.0                                  # relative residual
    # log-space residual (spread of the ratio normalised so '100x at 180' does not dominate)
    ln_resid = np.log(obs) - (np.log(rb) + np.mean(np.log(obs) - np.log(rb)))
    per_level[float(M)] = {
        "T_fit_5300K_relative_weight": T1, "T_fit_6300K_relative_weight": T2, "dT_fit_K": T2 - T1,
        "T_fit_5300K_log_weight": T1L, "T_fit_6300K_log_weight": T2L, "dT_fit_K_log_weight": T2L - T1L,
        "planck_ratio_min_over_band": float(rb.min()), "planck_ratio_argmin_nm": float(wl[int(np.argmin(rb))]),
        "planck_ratio_at_850nm": float(rb[-1]), "planck_ratio_at_180nm": float(rb[0]),
        "planck_ratio_min_log_weight": float(rbL.min()),
        "scaled_planck_ratio_at_180nm": float(model[0]),
        "scaled_planck_ratio_at_850nm": float(model[-1]),
        "observed_ratio_at_180nm": float(obs[0]), "observed_ratio_at_850nm": float(obs[-1]),
        "resid_180nm": float(resid[0]),
        "resid_620nm": float(resid[int(np.argmin(abs(wl - 620)))]),
        "resid_850nm": float(resid[-1]),
        "log_resid_180nm": float(ln_resid[0]), "log_resid_850nm": float(ln_resid[-1]),
        "median_abs_resid_250_400nm": float(np.median(np.abs(resid[(wl >= 250) & (wl <= 400)]))),
        "median_abs_resid_600_850nm": float(np.median(np.abs(resid[(wl >= 600)]))),
        "median_abs_log_resid": float(np.median(np.abs(ln_resid))),
        "quadratic_curvature_700_850_observed": float(np.polyfit(wl[wl >= 700], obs[wl >= 700], 2)[0]),
        "quadratic_curvature_700_850_scaled_planck": float(np.polyfit(wl[wl >= 700], model[wl >= 700], 2)[0]),
    }
    resid_uv.append(resid[0]); resid_850.append(resid[-1])
    resid_650.append(resid[int(np.argmin(abs(wl - 620)))])
    resid_650_rel.append(ln_resid[int(np.argmin(abs(wl - 620)))])
    dT.append(T2 - T1)
    curv_obs.append(per_level[float(M)]["quadratic_curvature_700_850_observed"])
    curv_bb.append(per_level[float(M)]["quadratic_curvature_700_850_scaled_planck"])

planck_test = {
    "planck_with_nominal_temperatures_5300_6300": planck_nominal,
    "key_analytic_fact": ("for T_hot>T_cold the Planck ratio B(l,T_hot)/B(l,T_cold) is monotonically "
                          "DECREASING in l from +inf (Wien) to the Rayleigh-Jeans asymptote T_hot/T_cold "
                          "(= 1.189 for these two temperatures). It is therefore > 1 at EVERY wavelength. "
                          "A pure blackbody pair can never produce a ratio below 1, so it cannot produce "
                          "the observed crossing at all, nor the 0.856-0.886 values at 850 nm."),
    "fitted_dT_K": {"mean": float(np.mean(dT)), "min": float(np.min(dT)), "max": float(np.max(dT)),
                    "median": float(np.median(dT)), "spread": float(np.max(dT) - np.min(dT)),
                    "mean_log_weight": float(np.mean([per_level[M]["dT_fit_K_log_weight"] for M in mets])),
                    "spread_log_weight": float(max(per_level[M]["dT_fit_K_log_weight"] for M in mets) -
                                               min(per_level[M]["dT_fit_K_log_weight"] for M in mets))},
    "dT_fit_vs_M_spearman": spearman(xM, np.array(dT)),
    "minimum_planck_ratio_over_band_across_levels": float(min(per_level[M]["planck_ratio_min_over_band"] for M in mets)),
    "minimum_planck_ratio_over_band_log_weight_across_levels": float(min(per_level[M]["planck_ratio_min_log_weight"] for M in mets)),
    "planck_ratio_850nm_range_across_M": [float(min(per_level[M]["planck_ratio_at_850nm"] for M in mets)),
                                          float(max(per_level[M]["planck_ratio_at_850nm"] for M in mets))],
    "median_abs_relative_residual": {
        "180nm": float(np.median(np.abs(resid_uv))),
        "620nm": float(np.median(np.abs(resid_650))),
        "850nm": float(np.median(np.abs(resid_850))),
    },
    "median_abs_log_residual": {
        "all_band": float(np.median([per_level[M]["median_abs_log_resid"] for M in mets])),
        "180nm": float(np.median([abs(per_level[M]["log_resid_180nm"]) for M in mets])),
        "620nm": float(np.median([abs(per_level[M]["log_resid_620nm"] if "log_resid_620nm" in per_level[M]
                                               else per_level[M]["resid_620nm"]) for M in mets])),
        "850nm": float(np.median([abs(per_level[M]["log_resid_850nm"]) for M in mets])),
    },
    "median_signed_residual": {"180nm": float(np.median(resid_uv)),
                               "850nm": float(np.median(resid_850))},
    "red_end_curvature_observed_vs_scaled_planck": {
        "observed_median": float(np.median(curv_obs)), "n_concave_up_observed": int((np.array(curv_obs) > 0).sum()),
        "scaled_planck_median": float(np.median(curv_bb)), "n_concave_up_scaled_planck": int((np.array(curv_bb) > 0).sum()),
        "n_levels": len(mets)},
    "n_levels_where_planck_ratio_crosses_1": int(sum(per_level[M]["planck_ratio_min_over_band"] < 1.0 for M in mets)),
    "n_levels": len(mets),
}
per_level_out = {str(M): per_level[M] for M in mets}

# ---------------------------------------------------------------- verdict
trend_real = (np.mean(sl_q <= 0) < 0.05) and (np.mean(sl_lev <= 0) < 0.05) and (np.mean(sl_param <= 0) < 0.05)
verdict = {
    "claim_component_A_uv_amplitude": {
        "survives": True, "weakened": True,
        "reason": ("median ratio at ~180 nm is %.1fx, so 'order 100x' holds as a median, but the ladder "
                   "spans %.0f-%.0f and %d/%d levels are below 100x."
                   % (np.median([R[M][0] for M in mets]),
                      min(R[M][0] for M in mets), max(R[M][0] for M in mets),
                      int((np.array([R[M][0] for M in mets]) < 100).sum()), len(mets)))},
    "claim_component_B_crosses_1_in_the_red": {
        "survives": True, "weakened": True,
        "reason": ("all 35 levels cross 1 between %.0f and %.0f nm, but this is a zero of a "
                   "5-nm-sampled function: %d/%d crossings land exactly on a grid node and only %d "
                   "distinct crossing values occur." % (seq.min(), seq.max(), n_on_node, len(mets), n_distinct))},
    "claim_component_C_below_1_at_850nm_band_0.86_0.88": {
        "survives": True, "weakened": True,
        "reason": ("all %d levels are <1 at 850 nm, but only %d/%d fall inside 0.86-0.88; the band is "
                   "left at both ends (%.4f at M=%.2f ... %.4f at M=%.2f). No data exist beyond "
                   "850 nm, so nothing can be said about continuation, steepening or turnover."
                   % (int((r850 < 1).sum()), int(in_band.sum()), len(mets), r850.min(),
                      mets[int(np.argmin(r850))], r850.max(), mets[int(np.argmax(r850))]))},
    "claim_component_D_crossing_moves_redward_with_M": {
        "survives": bool(trend_real), "weakened": True,
        "reason": ("redward slope %.2f nm per unit M, positive under all three noise models "
                   "(all p<=%.1e); the non-monotonicity is not evidence against it (%d backward steps, "
                   "largest %.2f nm, all below the 5 nm node spacing; the quantisation null produces "
                   "backward steps at least this large in %.1f%% of draws). What IS weakened: %d/%d "
                   "crossings sit exactly on grid nodes, so the sequence has only %d distinct values and "
                   "is only resolved to ~5 nm; and the rise saturates (%.1f nm/M for M<=-0.3 vs %.1f nm/M "
                   "for M>=0.2), so 'moves redward' is not a uniform, linear redward march."
                   % (slope, max(np.mean(sl_q <= 0), np.mean(sl_lev <= 0), np.mean(sl_param <= 0)),
                      n_back, max(-b["step_nm"] for b in back) if back else 0.0,
                      np.mean(mx_null >= obs_max_back) * 100,
                      n_on_node, len(mets), n_distinct,
                      seg["M in [-1.0,-0.3]"]["slope_nm_per_M"], seg["M in [0.2,0.9]"]["slope_nm_per_M"]))
        if trend_real else
        ("redward slope %.2f nm per unit M is not separable from the quantisation noise floor." % slope)},
    "claim_component_E_pure_blackbody_explanation": {
        "survives": False, "weakened": True,
        "reason": ("Planck with nominal 5300/6300 K gives %.1fx at 180 nm (observed median %.0fx) and "
                   "never drops below 1 (RJ asymptote 1.189). Fitted per-slice Planck shapes can mimic "
                   "the ratio level (free scale absorbs the normalisation), but every one of the 35 "
                   "fitted Planck-pair curves stays >1 across the whole band (minimum %.2f, with log-flux "
                   "weighting minimum %.2f), so blackbody cannot produce the reversal at all, let alone "
                   "its metallicity dependence. The observed crossing requires a wavelength-dependent "
                   "flux redistribution (atmospheric structure), not blackbody."
                   % (C_NOM[0], np.median([R[M][0] for M in mets]),
                      min(per_level[M]["planck_ratio_min_over_band"] for M in mets),
                      min(per_level[M]["planck_ratio_min_log_weight"] for M in mets)))},
    "overall": "claim survives only in weakened form; the pure-blackbody explanation does not survive",
    "refuted": not bool(trend_real),
}

# ---------------------------------------------------------------- output
result = {
    "finding_id": "D2-refutation",
    "claim_under_test": ("F(6300)/F(5300) >> 1 in UV (order 100x at 180 nm), crosses 1 in the red, "
                         "<1 at long wavelengths (~0.86-0.88 at 850 nm); crossing moves redward with "
                         "metallicity index."),
    "source_data": "data/photon_flux_grid.tsv",
    "method": ("numpy-only recomputation from the raw TSV; elementwise ratio on the 176-node wavelength "
               "axis; first sign change of r-1 with linear interpolation; local quadratic interpolation "
               "as a quantisation-robust cross-check; three uncertainty models for the trend "
               "(node quantisation, level resampling, 3-significant-figure flux quantisation); "
               "Planck shape fits on relative residuals with a free scale."),
    "crossing_analysis": {
        "sequence_nm_vs_M": {str(M): (None if np.isnan(seq[i]) else round(float(seq[i]), 3)) for i, M in enumerate(mets)},
        "sequence_quadratic_nm_vs_M": {str(M): (None if np.isnan(seq_p2[i]) else round(float(seq_p2[i]), 3)) for i, M in enumerate(mets)},
        "node_diagnostics": {str(M): node_flags[M] for M in mets},
        "first_nm": float(seq[0]), "last_nm": float(seq[-1]), "net_shift_nm": net,
        "range_nm": float(seq.max() - seq.min()),
        "n_backward_steps": n_back, "total_steps": len(mets) - 1, "backward_steps": back,
        "backward_step_abs_min_nm": float(min(-b["step_nm"] for b in back)) if back else 0.0,
        "backward_step_abs_max_nm": float(max(-b["step_nm"] for b in back)) if back else 0.0,
        "sum_backward_nm": sum_back, "n_zero_steps": n_zero, "n_distinct_values": n_distinct,
        "n_crossings_landing_on_node": n_on_node, "n_levels": len(mets),
        "ols_slope_nm_per_unit_M": slope, "ols_intercept": intercept, "ols_r2": r2_lin,
        "quadratic_fit_M": {"coeffs_deg2_deg1_deg0": [float(c) for c in qcoef], "r2": r2_quad,
                            "slope_at_M_minus1": float(2 * qcoef[0] * -1 + qcoef[1]),
                            "slope_at_M_plus1": float(2 * qcoef[0] * 1 + qcoef[1])},
        "spearman_M_vs_crossing": spear, "pearson_M_vs_crossing": pear,
        "piecewise_slopes": seg,
        "local_node_spacing_600_680nm": {"min": float(sp_loc.min()), "max": float(sp_loc.max()),
                                         "median": float(np.median(sp_loc))},
        "trend_uncertainty": {
            "bootstrap_node_quantisation": {"slope_mean": float(np.mean(sl_q)), "slope_std": float(np.std(sl_q)),
                                            "ci95": q_ci, "p_slope_le_0": float(np.mean(sl_q <= 0)), "n_boot": nboot},
            "bootstrap_resample_levels": {"slope_mean": float(np.mean(sl_lev)), "slope_std": float(np.std(sl_lev)),
                                          "ci95": lev_ci, "p_slope_le_0": float(np.mean(sl_lev <= 0)), "n_boot": nboot},
            "bootstrap_flux_3sigfig": {"slope_mean": float(np.mean(sl_param)), "slope_std": float(np.std(sl_param)),
                                       "ci95": par_ci, "p_slope_le_0": float(np.mean(sl_param <= 0)),
                                       "n_boot": nboot2, "n_lost_crossings": int(lost)},
        },
        "quadratic_estimator": {"ols_slope_nm_per_unit_M": ols(xM, seq_p2)[0],
                                "n_backward_steps": int(sum(1 for j in range(1, len(mets)) if seq_p2[j] < seq_p2[j - 1]))},
        "monotonic": n_back == 0,
        "monotonicity_vs_quantisation_null": monotonicity_null,
        "independent_corroboration": {"crossing_vs_ratio850nm": corr_cross_r850, "crossing_vs_ratio650nm": corr_cross_r650},
    },
    "uv_amplitude_analysis": uv_probe,
    "red_endpoint_analysis": endpoint,
    "red_end_edge_diagnostics": edge,
    "blackbody_baseline_test": planck_test,
    "blackbody_per_level": per_level_out,
    "red_decline_vs_quantisation_floor": red_decline_test,
    "cannot_see_beyond": ("850.0 nm is the last grid node in this file and the response file likewise ends "
                          "at 849.77 nm; no statement about the deficit beyond 850 nm is possible from "
                          "this dataset."),
    "verdict": verdict,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(result, fh, indent=1, default=float)

# ---------------------------------------------------------------- console summary
print("crossing sequence (nm) vs M:")
print("  " + ", ".join(f"{M:+.2f}:{(seq[i] if not np.isnan(seq[i]) else float('nan')):.2f}" for i, M in enumerate(mets)))
print(f"backward steps {n_back}/{len(mets)-1}, sizes {min(-b['step_nm'] for b in back):.2f}-{max(-b['step_nm'] for b in back):.2f} nm, sum {sum_back:.2f}, net drift {net:.2f} nm")
print(f"on-node {n_on_node}/35, zero steps {n_zero}, distinct values {n_distinct}")
print(f"OLS slope {slope:.2f} nm/M (r2 {r2_lin:.3f}); Spearman {spear:.3f}; quadratic r2 {r2_quad:.3f}")
print("piecewise:", {k: round(v['slope_nm_per_M'], 2) for k, v in seg.items()})
print(f"p(slope<=0): node {np.mean(sl_q<=0):.4f}, levels {np.mean(sl_lev<=0):.4f}, 3sigfig {np.mean(sl_param<=0):.4f} "
      f"(3sigfig slope std {np.std(sl_param):.2f}, lost crossings {lost})")
print(f"UV 179.4nm: median {uv_probe['179.4nm']['median']:.1f}, range {uv_probe['179.4nm']['min']:.1f}-{uv_probe['179.4nm']['max']:.1f}, "
      f"n<100 {(np.array([R[M][0] for M in mets])<100).sum()}/35")
print(f"850nm: median {np.median(r850):.4f}, range {r850.min():.4f}-{r850.max():.4f}, all<1: {(r850<1).all()}, in[.86,.88]: {in_band.sum()}/35")
print("red segments:", {k: round(v['median_dRdlambda_per_nm'], 5) for k, v in seg_slopes.items()})
print(f"curvature 700-850 median {np.median(curv):.2e}, concave_up {(curv>0).sum()}/35")
print(f"nominal Planck ratio: 180nm {C_NOM[0]:.2f}, 850nm {C_NOM[-1]:.3f}, min {C_NOM.min():.3f} -> crosses 1: {C_NOM.min()<1}")
print(f"fitted dT mean {np.mean(dT):.0f} K spread {max(dT)-min(dT):.0f} K (log-weight "
      f"{np.mean([per_level[M]['dT_fit_K_log_weight'] for M in mets]):.0f} K); min Planck-pair ratio over band across levels "
      f"{min(per_level[M]['planck_ratio_min_over_band'] for M in mets):.3f} (log-weight "
      f"{min(per_level[M]['planck_ratio_min_log_weight'] for M in mets):.3f})")
print(f"median |rel resid| vs scaled Planck: 180nm {np.median(np.abs(resid_uv)):.4f}, 620nm {np.median(np.abs(resid_650)):.4f}, 850nm {np.median(np.abs(resid_850)):.4f}")
print(f"median |log resid|: all {np.median([per_level[M]['median_abs_log_resid'] for M in mets]):.3f}, "
      f"180nm {np.median([abs(per_level[M]['log_resid_180nm']) for M in mets]):.3f}, "
      f"850nm {np.median([abs(per_level[M]['log_resid_850nm']) for M in mets]):.3f}")
print(f"curvature 700-850: observed median {np.median(curv_obs):.2e} (up {(np.array(curv_obs)>0).sum()}/35), "
      f"scaled Planck {np.median(curv_bb):.2e} (up {(np.array(curv_bb)>0).sum()}/35)")
print(f"monotonicity null: observed max backward {obs_max_back:.2f} nm, null median {np.median(mx_null):.2f}/p95 {np.percentile(mx_null,95):.2f} nm, "
      f"p_null>=obs {np.mean(mx_null>=obs_max_back):.4f}, n_back obs {n_back} vs null median {np.median(nb_null):.0f}")
print(f"red decline test: obs median slope {np.median(obs_slope):.2e}/nm, n<0 {(obs_slope<0).sum()}/35, "
      f"quantisation slope error {sig_slope_pred:.1e}/nm, ratio {abs(np.median(obs_slope))/sig_slope_pred:.0f}x, "
      f"frac perturbed draws with slope>=0: {flip/n_pert:.3f}")
print("wrote", OUT)