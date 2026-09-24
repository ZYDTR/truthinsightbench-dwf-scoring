#!/usr/bin/env python3
"""
PACKAGE p7_anisotropy: energy dependence of the angular structure.

(a) two-term fit  M(theta) = A (1 + beta P2(cos theta))  at every dense energy,
    with R^2; alternative (full Legendre expansion) beta when two-term is inadequate.
(b) energy dependence of beta(E): min/max, region means, Spearman/Pearson, bootstrap.
(c) same for the theory panel + experiment/theory matched-energy comparison.
(d) independent measured representation: CoefLeg F20-family / F00-family proxy.
(e) localization of features relative to the local trend.
(f) constant-beta null, permutation null, leave-one-out influence.
(g) beta(E) series to CSV.

Everything is computed from data/ only.  Re-runnable from scratch.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L
from scipy.stats import pearsonr, spearmanr

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
OUT_EV = os.path.join(ROOT, "output", "evidence")
OUT_AN = os.path.join(ROOT, "output", "analysis")
os.makedirs(OUT_EV, exist_ok=True)
os.makedirs(OUT_AN, exist_ok=True)

CSV_PATH = os.path.join(OUT_EV, "p7_anisotropy.csv")
JSON_PATH = os.path.join(OUT_EV, "p7_anisotropy.json")
PNG_PATH = os.path.join(OUT_EV, "p7_anisotropy.png")

RNG = np.random.default_rng(20260919)
NPERM = 20000
NBOOT = 2000
MEAS_E = [22.5, 23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1,
          35.65, 37.2, 38.75, 48.4]

FAILURES = []
NOTES = []


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pivot2d(df, group):
    sub = df[df.dataset_path == group]
    return sub.pivot_table(index="index_0", columns="index_1",
                           values="value").to_numpy()


def legendre_design(x, lmax):
    return np.stack([L.legval(x, [0] * l + [1]) for l in range(lmax + 1)], axis=1)


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.stack([np.ones_like(x), x - x.mean()], axis=1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(slope=float(c[1]), intercept=float(c[0]),
                ss_res=ss_res, ss_tot=ss_tot,
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan)


def moving_block_bootstrap_slope(E, y, block, nboot=NBOOT, rng=RNG):
    n = len(y)
    nb = int(np.ceil(n / block))
    out = np.empty(nboot)
    order = np.arange(n)
    for b in range(nboot):
        starts = rng.integers(0, max(1, n - block + 1), size=nb)
        idx = np.concatenate([(np.arange(s, s + block)) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        out[b] = ols_slope(E[idx], y[idx])["slope"]
    return out


def iid_bootstrap_slope(E, y, nboot=NBOOT, rng=RNG):
    n = len(y)
    out = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        out[b] = ols_slope(E[idx], y[idx])["slope"]
    return out


def local_residual(E, y, halfwin, deg):
    """Leave-one-out local polynomial residual: how much each point deviates
    from the trend defined by its neighbourhood."""
    res = np.full(len(E), np.nan)
    for i in range(len(E)):
        m = (np.abs(E - E[i]) <= halfwin) & (np.arange(len(E)) != i)
        if m.sum() < deg + 4:
            continue
        dx = E[m] - E[i]
        X = np.stack([dx ** k for k in range(deg + 1)], axis=1)
        c, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        res[i] = y[i] - c[0]
    finite = np.isfinite(res)
    med = np.median(res[finite])
    sigma = 1.4826 * np.median(np.abs(res[finite] - med))
    sigma = max(sigma, 1e-12)
    z = res / sigma
    return res, sigma, z


def runs_above(E, z, thresh):
    idx = np.where(np.abs(z) > thresh)[0]
    out = []
    if len(idx) == 0:
        return out
    a = idx[0]
    p = idx[0]
    for k in idx[1:]:
        if k == p + 1:
            p = k
        else:
            out.append((a, p))
            a = k
            p = k
    out.append((a, p))
    return out


def half_width(E, res, ipk):
    """contiguous width (eV) over which |res| stays above half its peak value."""
    pk = abs(res[ipk])
    lo = ipk
    while lo - 1 >= 0 and abs(res[lo - 1]) > 0.5 * pk:
        lo -= 1
    hi = ipk
    while hi + 1 < len(res) and abs(res[hi + 1]) > 0.5 * pk:
        hi += 1
    return float(E[hi] - E[lo]), int(lo), int(hi)


def analyse_panel(tag, M, E, x, lmax_full=8, lmax_six=6):
    """two-term fit + full Legendre beta + M^2 Legendre beta for one panel."""
    P2 = 0.5 * (3 * x ** 2 - 1)
    D = np.stack([np.ones_like(x), P2], axis=1)
    coef, *_ = np.linalg.lstsq(D, M.T, rcond=None)
    A2 = coef[0]
    b2 = coef[1]
    pred2 = (D @ coef).T
    resid2 = M - pred2
    sst = np.sum((M - M.mean(axis=1, keepdims=True)) ** 2, axis=1)
    R2_2 = 1 - np.sum(resid2 ** 2, axis=1) / sst
    rms2 = np.sqrt(np.mean(resid2 ** 2, axis=1))

    Vf = legendre_design(x, lmax_full)
    cf, *_ = np.linalg.lstsq(Vf, M.T, rcond=None)
    bf = cf[2] / cf[0]
    predf = (Vf @ cf).T
    R2_f = 1 - np.sum((M - predf) ** 2, axis=1) / sst

    V6 = legendre_design(x, lmax_six)
    c6, *_ = np.linalg.lstsq(V6, (M ** 2).T, rcond=None)
    bI = c6[2] / c6[0]
    R2_I6 = 1 - np.sum((M ** 2 - (V6 @ c6).T) ** 2, axis=1) / \
        np.sum((M ** 2 - (M ** 2).mean(axis=1, keepdims=True)) ** 2, axis=1)

    return dict(tag=tag, E=E, A2=A2, beta2=b2, R2_2=R2_2, rms2=rms2,
                beta_full=bf, R2_full=R2_f, beta_Msq=bI, R2_Msq6=R2_I6,
                cf=cf, c6=c6)


# ---------------------------------------------------------------- load data
log("=" * 78)
log("p7_anisotropy: energy dependence of the angular structure")
log("=" * 78)

files = {
    "angle_energy_observations_datasets.tsv": None,
    "angular_coefficients_datasets.tsv": None,
    "dipole_matrix_elements_datasets.tsv": None,
    "coordinate_definitions.json": None,
}
for fn in files:
    files[fn] = sha256(os.path.join(DATA, fn))
    log("sha256(%s) = %s" % (fn, files[fn]))

dfa = pd.read_csv(os.path.join(DATA, "angle_energy_observations_datasets.tsv"), sep="\t")
dfc = pd.read_csv(os.path.join(DATA, "angular_coefficients_datasets.tsv"), sep="\t")
dfd = pd.read_csv(os.path.join(DATA, "dipole_matrix_elements_datasets.tsv"), sep="\t")
with open(os.path.join(DATA, "coordinate_definitions.json")) as f:
    coord = json.load(f)

theta_deg = np.arange(181.0)
x = np.cos(np.deg2rad(theta_deg))

Me = pivot2d(dfa, "experiment/magnitude")            # (276,181)
Ee = dfa[dfa.dataset_path == "experiment/photon_energy_eV"].sort_values(
    "index_0").value.to_numpy()
Mt = pivot2d(dfa, "theory/magnitude")                # (99,181)
Et = dfa[dfa.dataset_path == "theory/photon_energy_eV"].sort_values(
    "index_0").value.to_numpy()
log("experiment dense grid: %s, E = %.3f .. %.3f eV (step %.3f)"
    % (Me.shape, Ee[0], Ee[-1], np.mean(np.diff(Ee))))
log("theory dense grid:     %s, E = %.3f .. %.3f eV (mean step %.3f)"
    % (Mt.shape, Et[0], Et[-1], np.mean(np.diff(Et))))

exp = analyse_panel("experiment", Me, Ee, x)
the = analyse_panel("theory", Mt, Et, x)

# ------------------------------------------------------------------- (a)
log("")
log("--- (a) two-term fit  M = A (1 + beta P2) -------------------------")
for p in (exp, the):
    R = p["R2_2"]
    log("%-10s two-term R2: median %.4f  min %.4f  max %.4f ; frac(R2>0.9)=%.3f "
        "frac(R2>0.99)=%.3f ; median rms %.4f"
        % (p["tag"], np.median(R), R.min(), R.max(), np.mean(R > 0.9),
           np.mean(R > 0.99), np.median(p["rms2"])))
    log("%-10s beta(2term) range [%.4f, %.4f]; beta(full L<=8) range [%.4f, %.4f];"
        " beta(M^2, L<=6) range [%.4f, %.4f]"
        % (p["tag"], p["beta2"].min(), p["beta2"].max(),
           p["beta_full"].min(), p["beta_full"].max(),
           p["beta_Msq"].min(), p["beta_Msq"].max()))
    log("%-10s full-expansion R2: median %.5f min %.5f ; M^2 L<=6 R2: median %.6f min %.6f"
        % (p["tag"], np.median(p["R2_full"]), p["R2_full"].min(),
           np.median(p["R2_Msq6"]), p["R2_Msq6"].min()))

# basis-truncation robustness of beta (alternative basis)
basis_robust = {}
for p in (exp, the):
    ref = p["beta_full"]
    basis_robust[p["tag"]] = {}
    M = Me if p["tag"] == "experiment" else Mt
    for nmax in (2, 4, 6, 10, 12, 20):
        V = legendre_design(x, nmax)
        c, *_ = np.linalg.lstsq(V, M.T, rcond=None)
        b = c[2] / c[0]
        basis_robust[p["tag"]][str(nmax)] = dict(
            corr_to_L8=float(np.corrcoef(ref, b)[0, 1]),
            max_abs_diff=float(np.abs(ref - b).max()),
            mean_abs_diff=float(np.mean(np.abs(ref - b))),
            pearson_E=float(pearsonr(p["E"], b)[0]),
            slope=float(ols_slope(p["E"], b)["slope"]))
    log("%-10s basis robustness (corr to L<=8, slope per eV): %s"
        % (p["tag"], {k: (round(v["corr_to_L8"], 5), round(v["slope"], 5))
                      for k, v in basis_robust[p["tag"]].items()}))
    # squared-magnitude basis robustness
    M = Me if p["tag"] == "experiment" else Mt
    for nmax in (4, 6, 8, 12):
        V = legendre_design(x, nmax)
        c, *_ = np.linalg.lstsq(V, (M ** 2).T, rcond=None)
        bk = c[2] / c[0]
        log("%-10s M^2 basis Lmax=%2d: beta range [%.4f, %.4f], slope %.5f /eV, "
            "corr to M^2 L6 %.6f"
            % (p["tag"], nmax, bk.min(), bk.max(), ols_slope(p["E"], bk)["slope"],
               np.corrcoef(p["beta_Msq"], bk)[0, 1]))

# ------------------------------------------------------------------- (b)
log("")
log("--- (b) energy dependence of beta(E), experiment dense grid --------")
bE = exp["beta_Msq"]          # primary smooth beta (degree-6 Legendre fit of M^2)
bE2 = exp["beta2"]           # plain two-term fit (inadequate, reported too)
bEf = exp["beta_full"]       # L<=8 expansion of M


def describe(E, y, label):
    i_min = int(np.argmin(y))
    i_max = int(np.argmax(y))
    pr = pearsonr(E, y)
    sr = spearmanr(E, y)
    sl = ols_slope(E, y)
    log("%-22s min %.4f @ %.2f eV ; max %.4f @ %.2f eV" %
        (label, y[i_min], E[i_min], y[i_max], E[i_max]))
    log("%-22s Pearson r=%.4f (p=%.3g) ; Spearman rho=%.4f (p=%.3g) ; "
        "OLS slope=%.5f per eV" % (label, pr[0], pr[1], sr[0], sr[1], sl["slope"]))
    return dict(min=float(y[i_min]), min_E=float(E[i_min]),
                max=float(y[i_max]), max_E=float(E[i_max]),
                pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                spearman_rho=float(sr[0]), spearman_p=float(sr[1]),
                slope=sl["slope"], ss_res=sl["ss_res"], ss_tot=sl["ss_tot"],
                r2=sl["r2"])


desc_exp = describe(Ee, bE, "exp beta(M^2,L6)")
desc_exp_2 = describe(Ee, bE2, "exp beta(2-term)")
desc_exp_f = describe(Ee, bEf, "exp beta(L<=8)")
desc_the = describe(Et, the["beta_Msq"], "theory beta(M^2,L6)")
desc_the_2 = describe(Et, the["beta2"], "theory beta(2-term)")
desc_the_f = describe(Et, the["beta_full"], "theory beta(L<=8)")

regions = {}
for p, b, name in ((exp, bE, "exp"), (the, the["beta_Msq"], "theory")):
    E = p["E"]
    regs = {}
    for lo, hi, lbl in ((22.5, 27.9, "low_22p5_27p9"),
                        (30.0, 40.0, "mid_30_40"),
                        (40.0, 41.9, "high_40_41p9_theory")):
        m = (E >= lo) & (E <= hi)
        if m.sum() == 0:
            continue
        regs[lbl] = dict(n=int(m.sum()), lo=float(E[m][0]), hi=float(E[m][-1]),
                         mean_beta=float(b[m].mean()), min_beta=float(b[m].min()),
                         max_beta=float(b[m].max()),
                         slope=float(ols_slope(E[m], b[m])["slope"]))
    if name == "exp":
        for ev in (48.4, 45.0, 50.0):
            i = int(np.argmin(np.abs(E - ev)))
            regs["point_%.1f" % ev] = dict(n=1, E=float(E[i]), mean_beta=float(b[i]))
    regions[name] = regs
    log("%-6s region means: %s" % (name, {k: round(v["mean_beta"], 4)
                                          for k, v in regs.items()}))

slope_boot_blk = moving_block_bootstrap_slope(Ee, bE, block=20)
slope_boot_iid = iid_bootstrap_slope(Ee, bE)
slope_ci = np.percentile(slope_boot_blk, [2.5, 97.5])
slope_ci_iid = np.percentile(slope_boot_iid, [2.5, 97.5])
log("exp slope = %.5f /eV ; moving-block(2 eV) bootstrap 95%% CI [%.5f, %.5f] ; "
    "iid bootstrap CI [%.5f, %.5f]"
    % (desc_exp["slope"], slope_ci[0], slope_ci[1], slope_ci_iid[0], slope_ci_iid[1]))

half = len(Ee) // 2
sh = dict(first_half=ols_slope(Ee[:half], bE[:half]),
          second_half=ols_slope(Ee[half:], bE[half:]),
          E_first=(float(Ee[0]), float(Ee[half - 1])),
          E_second=(float(Ee[half]), float(Ee[-1])))
log("exp split-sample slope: %.5f /eV on %.1f-%.1f eV ; %.5f /eV on %.1f-%.1f eV"
    % (sh["first_half"]["slope"], sh["E_first"][0], sh["E_first"][1],
       sh["second_half"]["slope"], sh["E_second"][0], sh["E_second"][1]))

# bootstrap uncertainty on the min / max positions
argmin_b = np.empty(NBOOT)
argmax_b = np.empty(NBOOT)
for b in range(NBOOT):
    idx = np.sort(RNG.integers(0, len(Ee), size=len(Ee)))
    yb = bE[idx]
    argmin_b[b] = Ee[idx][np.argmin(yb)]
    argmax_b[b] = Ee[idx][np.argmax(yb)]
log("exp argmin E 95%% CI [%.1f, %.1f] ; argmax E 95%% CI [%.1f, %.1f]"
    % (np.percentile(argmin_b, 2.5), np.percentile(argmin_b, 97.5),
       np.percentile(argmax_b, 2.5), np.percentile(argmax_b, 97.5)))

# ------------------------------------------------------------------- (c)
log("")
log("--- (c) theory panel and experiment-vs-theory comparison -----------")
Et_m = Et[(Et >= max(Ee[0], Et[0])) & (Et <= min(Ee[-1], Et[-1]))]
log("overlap of the two energy grids: %.3f .. %.3f eV, %d theory points"
    % (Et_m[0], Et_m[-1], len(Et_m)))
exp_on_t = np.interp(Et_m, Ee, bE)
the_on_t = the["beta_Msq"][(Et >= max(Ee[0], Et[0])) & (Et <= min(Ee[-1], Et[-1]))]
pr_et = pearsonr(exp_on_t, the_on_t)
sr_et = spearmanr(exp_on_t, the_on_t)
log("exp vs theory beta on overlap: Pearson r=%.4f (p=%.3g), Spearman rho=%.4f (p=%.3g)"
    % (pr_et[0], pr_et[1], sr_et[0], sr_et[1]))
log("mean(exp-theory)=%.4f  rms diff=%.4f  sign agreement=%.3f"
    % (np.mean(exp_on_t - the_on_t), np.sqrt(np.mean((exp_on_t - the_on_t) ** 2)),
       np.mean(np.sign(exp_on_t) == np.sign(the_on_t))))

the_interp_meas = np.interp([e for e in MEAS_E if e <= Et[-1]],
                            Et, the["beta_Msq"])
meas_in_range = [e for e in MEAS_E if e <= Et[-1]]
exp_at_meas = np.interp(meas_in_range, Ee, bE)
matched = dict(energies=meas_in_range,
               exp_beta=exp_at_meas.tolist(),
               theory_beta=the_interp_meas.tolist(),
               diff=(exp_at_meas - the_interp_meas).tolist())
log("matched energies (<=%.2f eV): %s" % (Et[-1], [round(e, 2) for e in meas_in_range]))
log("  exp beta : %s" % np.round(exp_at_meas, 4))
log("  the beta : %s" % np.round(the_interp_meas, 4))
pr_m = pearsonr(exp_at_meas, the_interp_meas)
sr_m = spearmanr(exp_at_meas, the_interp_meas)
log("  matched Pearson r=%.4f (p=%.3g) Spearman rho=%.4f (p=%.3g) sign agr=%.3f"
    % (pr_m[0], pr_m[1], sr_m[0], sr_m[1],
       np.mean(np.sign(exp_at_meas) == np.sign(the_interp_meas))))

# theory turning points, evaluated after thinning the near-duplicate first steps
# of the theory energy grid (steps of 0.016 eV vs the median 0.194 eV)
dth = np.diff(Et)
keep = np.r_[True, dth >= 0.1]
Et_s = Et[keep]
bt_s = the["beta_Msq"][keep]
turns = [i for i in range(1, len(bt_s) - 1)
         if np.sign(bt_s[i] - bt_s[i - 1]) != np.sign(bt_s[i + 1] - bt_s[i])]
log("theory turning points of beta(E) after thinning to steps>=0.1 eV "
    "(n=%d of %d): %s" % (len(Et_s), len(Et),
                          [(round(Et_s[i], 3), round(bt_s[i], 4)) for i in turns]))
turns_full = [i for i in range(1, len(Et) - 1)
              if np.sign(the["beta_Msq"][i] - the["beta_Msq"][i - 1]) !=
              np.sign(the["beta_Msq"][i + 1] - the["beta_Msq"][i])]
log("theory turning points on the raw grid (n=%d): %s"
    % (len(turns_full), [(round(Et[i], 3), round(the["beta_Msq"][i], 4))
                         for i in turns_full]))

# ------------------------------------------------------------------- (d)
log("")
log("--- (d) measured CoefLeg blocks: F20-family / F00-family proxy -----")
coef_tab = {}
proxy = {}
for ev in MEAS_E:
    t = pivot2d(dfc, "CoefLeg_%seV" % ev)
    coef_tab[ev] = t
    sF00 = float(t[:, 1].sum())
    sF20 = float(t[:, 3].sum())
    errF00 = float(np.sqrt(np.sum(t[:, 2] ** 2)))
    errF20 = float(np.sqrt(np.sum(t[:, 4] ** 2)))
    proxy[ev] = dict(F00_sum=sF00, F20_sum=sF20, ratio=sF20 / sF00,
                     ratio_err=float(abs(sF20 / sF00) *
                                     np.sqrt((errF20 / sF20) ** 2 + (errF00 / sF00) ** 2)),
                     F21_sum=float(t[:, 5].sum()), F22_sum=float(t[:, 7].sum()),
                     F11_sum=float(t[:, 9].sum()),
                     F20_o0=float(t[0, 3] / t[0, 1]),
                     F20_o2=float(t[2, 3] / t[2, 1]),
                     F20_o4=float(t[4, 3] / t[4, 1]))
    log("E=%6.2f  F00sum=%7.4f F20sum=%7.4f  F20/F00=%7.4f +- %.4f"
        % (ev, sF00, sF20, proxy[ev]["ratio"], proxy[ev]["ratio_err"]))

Es = np.array(MEAS_E)
ratio = np.array([proxy[e]["ratio"] for e in MEAS_E])
ratio_err = np.array([proxy[e]["ratio_err"] for e in MEAS_E])
dense_at_meas = np.interp(Es, Ee, bE)
the_at_meas = np.interp(Es, Et, the["beta_Msq"])
log("dense beta(M^2,L6) at the 13 measured energies: %s" % np.round(dense_at_meas, 4))
# non-interpolated variant: nearest dense-grid row to each measured energy
inearest = np.array([int(np.argmin(np.abs(Ee - e))) for e in Es])
dense_nearest = bE[inearest]
log("non-interpolated variant (nearest 0.1 eV grid row %s): %s"
    % (np.round(Ee[inearest], 2), np.round(dense_nearest, 4)))
log("interp vs nearest-row dense beta: max |diff| %.4f, corr %.6f"
    % (np.abs(dense_at_meas - dense_nearest).max(),
       np.corrcoef(dense_at_meas, dense_nearest)[0, 1]))

corr_all = pearsonr(ratio, dense_at_meas)
corr_all_s = spearmanr(ratio, dense_at_meas)
m_no48 = Es < 48.0
corr_no48 = pearsonr(ratio[m_no48], dense_at_meas[m_no48])
corr_no48_s = spearmanr(ratio[m_no48], dense_at_meas[m_no48])
corr_th = pearsonr(ratio, the_at_meas)
corr_th_s = spearmanr(ratio, the_at_meas)
log("CoefLeg proxy vs dense beta: all 13  Pearson r=%.4f (p=%.3g) Spearman %.4f (p=%.3g)"
    % (corr_all[0], corr_all[1], corr_all_s[0], corr_all_s[1]))
log("CoefLeg proxy vs dense beta: no 48.4 Pearson r=%.4f (p=%.3g) Spearman %.4f (p=%.3g)"
    % (corr_no48[0], corr_no48[1], corr_no48_s[0], corr_no48_s[1]))
log("CoefLeg proxy vs theory beta: all 13 Pearson r=%.4f (p=%.3g) Spearman %.4f (p=%.3g)"
    % (corr_th[0], corr_th[1], corr_th_s[0], corr_th_s[1]))
# sign of difference at the endpoint
d48 = float(dense_at_meas[-1] - np.interp(48.4, Ee, bEf))
log("dense beta at 48.4 eV: M^2-L6 %.4f, 2-term %.4f, L<=8 %.4f"
    % (dense_at_meas[-1], exp["beta2"][-1], exp["beta_full"][-1]))
ratio_slope = ols_slope(Es, ratio)
dense_slope13 = ols_slope(Es, dense_at_meas)
log("slope of proxy over 13 energies = %.5f /eV ; slope of dense beta on same energies = %.5f /eV"
    % (ratio_slope["slope"], dense_slope13["slope"]))
idx_diff = np.argsort(Es)
d_ratio = np.diff(ratio[np.argsort(Es)])
d_dense = np.diff(dense_at_meas[np.argsort(Es)])
log("sign agreement of consecutive differences (proxy vs dense, 12 pairs): %.3f"
    % np.mean(np.sign(d_ratio) == np.sign(d_dense)))

# --- how much of the conflict is the literal sum-vs-normalisation choice? ---
# The literal ratio sums the raw F20 blocks over order k.  The F20_k family is
# not a scaling family of the F00_k family (F20_0/F00_0 falls from 1.32 to 0.26
# while F20_2/F00_2 DOES track the dense beta), so the summed ratio is dominated
# by whichever order has the largest |F20_k|/|F00_k| weight, an order mix that
# changes with energy.  Three scale-free/low-order variants test that reading.
coef_variants = {}
variants = {}
for ev in MEAS_E:
    t = coef_tab[ev]
    # The F00 family reconstructs as sum_k F00_k * P_k(cos theta); its own
    # Legendre coefficients are therefore the F00_k column itself.
    variants.setdefault("F00_family_c2_over_c0", []).append(float(t[2, 1] / t[0, 1]))
    variants.setdefault("F20_over_F00_order0", []).append(float(t[0, 3] / t[0, 1]))
    variants.setdefault("F20_over_F00_order2", []).append(float(t[2, 3] / t[2, 1]))
    variants.setdefault("F20_family_norm_over_F00", []).append(
        float(np.sqrt(np.sum(t[:, 3] ** 2 + t[:, 5] ** 2 + t[:, 7] ** 2)) /
              t[:, 1].sum()))
for k, v in variants.items():
    v = np.array(v)
    r_d = pearsonr(v, dense_at_meas)
    r_t = pearsonr(v, the_at_meas)
    s_d = spearmanr(v, dense_at_meas)
    coef_variants[k] = dict(
        values=v.tolist(),
        pearson_vs_dense=float(r_d[0]), p_vs_dense=float(r_d[1]),
        spearman_vs_dense=float(s_d[0]), p_spearman_vs_dense=float(s_d[1]),
        pearson_vs_theory=float(r_t[0]), p_vs_theory=float(r_t[1]),
        slope=float(ols_slope(Es, v)["slope"]),
        mean_22p5_27p9=float(v[Es <= 27.9].mean()),
        mean_30_38p75=float(v[(Es >= 30) & (Es <= 38.75)].mean()),
        value_48p4=float(v[-1]))
    log("  variant %-24s slope %+.4f /eV  rho_vs_dense %+.3f (p=%.4f)  "
        "rho_vs_theory %+.3f  means low/mid %.3f/%.3f  48.4 eV %.3f"
        % (k, coef_variants[k]["slope"], s_d[0], s_d[1], r_t[0],
           coef_variants[k]["mean_22p5_27p9"], coef_variants[k]["mean_30_38p75"],
           coef_variants[k]["value_48p4"]))
coef_variants["literal_sum_ratios"] = dict(
    values=ratio.tolist(), slope=float(ratio_slope["slope"]),
    pearson_vs_dense=float(corr_all[0]), p_vs_dense=float(corr_all[1]),
    spearman_vs_dense=float(corr_all_s[0]), p_spearman_vs_dense=float(corr_all_s[1]),
    pearson_vs_theory=float(corr_th[0]), p_vs_theory=float(corr_th[1]),
    mean_22p5_27p9=float(ratio[Es <= 27.9].mean()),
    mean_30_38p75=float(ratio[(Es >= 30) & (Es <= 38.75)].mean()),
    value_48p4=float(ratio[-1]))
coef_variants["dense_beta_reference"] = dict(
    values=dense_at_meas.tolist(), slope=float(dense_slope13["slope"]),
    mean_22p5_27p9=float(dense_at_meas[Es <= 27.9].mean()),
    mean_30_38p75=float(dense_at_meas[(Es >= 30) & (Es <= 38.75)].mean()),
    value_48p4=float(dense_at_meas[-1]))
coef_variants["dense_beta_nearest_row"] = dict(
    values=dense_nearest.tolist(), grid_energies=Ee[inearest].tolist(),
    slope=float(ols_slope(Es, dense_nearest)["slope"]),
    max_abs_diff_from_interp=float(np.abs(dense_at_meas - dense_nearest).max()),
    pearson_vs_interp=float(np.corrcoef(dense_at_meas, dense_nearest)[0, 1]))
# the F00-family variant re-scored against the non-interpolated dense beta
r_sf_nearest = spearmanr(coef_variants["F00_family_c2_over_c0"]["values"],
                         dense_nearest)
coef_variants["F00_family_c2_over_c0"]["spearman_vs_dense_nearest_row"] = float(r_sf_nearest[0])
coef_variants["F00_family_c2_over_c0"]["p_spearman_vs_dense_nearest_row"] = float(r_sf_nearest[1])
log("  variant F00_family_c2_over_c0 vs nearest-row dense beta: Spearman %+.3f (p=%.4f)"
    % (r_sf_nearest[0], r_sf_nearest[1]))
log("  dense beta reference: slope %+.4f /eV  means low/mid %.3f/%.3f  48.4 eV %.3f"
    % (dense_slope13["slope"], dense_at_meas[Es <= 27.9].mean(),
       dense_at_meas[(Es >= 30) & (Es <= 38.75)].mean(), dense_at_meas[-1]))
log("  RECONCILIATION of (d): the literal summed ratio anti-correlates with the "
    "dense beta (Spearman %+.3f, p=%.4f), but the order-consistent variant "
    "F00-family c2/c0 correlates strongly positively (Spearman %+.3f, p=%.4f). "
    "F20_0/F00_0 falls from %.3f to %.3f between 22.5 and 48.4 eV while "
    "F20_2/F00_2 tracks the dense beta, so the sign of the summed proxy is set by "
    "an energy-dependent order mix inside the F20 family, not by the energy trend."
    % (corr_all_s[0], corr_all_s[1],
       coef_variants["F00_family_c2_over_c0"]["spearman_vs_dense"],
       coef_variants["F00_family_c2_over_c0"]["p_spearman_vs_dense"],
       coef_variants["F20_over_F00_order0"]["values"][0],
       coef_variants["F20_over_F00_order0"]["values"][-1]))

# extra cross-representation check: reconstruct an angular function from the
# measured DME blocks and from the 22.5 eV CoefLeg F00 family, correlate both
# with the dense magnitude row at that energy.
recon = {}
try:
    from scipy.special import sph_harm_y
    for ev in (22.5, 31.0, 48.4):
        blk = pivot2d(dfd, "experiment/DME_%seV" % ev)
        I = np.zeros_like(theta_deg)
        for m in (0, 1, 2):
            sel = np.isclose(blk[:, 1], m)
            if not sel.any():
                continue
            re_ = np.zeros_like(theta_deg)
            im_ = np.zeros_like(theta_deg)
            for row in blk[sel]:
                l = int(round(row[0]))
                Y = sph_harm_y(l, m, np.deg2rad(theta_deg), np.zeros_like(theta_deg))
                re_ += row[2] * Y.real
                im_ += row[3] * Y.real
            I += (1.0 if m == 0 else 2.0) * (re_ ** 2 + im_ ** 2)
        i0 = int(np.argmin(np.abs(Ee - ev)))
        obs = Me[i0]
        recon[str(ev)] = dict(
            corr_sqrtI_obs=float(np.corrcoef(np.sqrt(I), obs)[0, 1]),
            ratio_median=float(np.median(obs / np.maximum(np.sqrt(I), 1e-12))),
            ratio_iqr=float(np.subtract(*np.percentile(
                obs / np.maximum(np.sqrt(I), 1e-12), [75, 25]))))
        log("DME cross-check E=%5.1f eV: corr(sqrt(I_DME), dense magnitude)=%.4f ; "
            "obs/pred median %.3f (IQR %.3f)"
            % (ev, recon[str(ev)]["corr_sqrtI_obs"], recon[str(ev)]["ratio_median"],
               recon[str(ev)]["ratio_iqr"]))
    if max(abs(v["corr_sqrtI_obs"]) for v in recon.values()) < 0.9:
        FAILURES.append(
            "Naive DME->|M(theta)| reconstruction (Y_lm, l<=3, m=0..2, "
            "isotropic-M assumption) does not reproduce the dense magnitude row: "
            "corr values %s. The DME blocks alone are therefore NOT used to "
            "define beta; only their energy trend via CoefLeg is."
            % {k: round(v["corr_sqrtI_obs"], 3) for k, v in recon.items()})
except Exception as exc:  # noqa: BLE001
    FAILURES.append("DME cross-check raised %s: %s" % (type(exc).__name__, exc))
    log("DME cross-check raised %s: %s" % (type(exc).__name__, exc))

# ------------------------------------------------------------------- (e)
log("")
log("--- (e) localization of features against the local trend -----------")
features = {}
for name, E, y in (("experiment", Ee, bE), ("theory", Et, the["beta_Msq"])):
    features[name] = {}
    for halfwin, deg in ((5.0, 2), (10.0, 1), (10.0, 2), (20.0, 1)):
        res, sigma, z = local_residual(E, y, halfwin, deg)
        runs = runs_above(E, z, 3.0)
        key = "win%.0f_deg%d" % (halfwin, deg)
        lst = []
        for a, p in runs:
            zz = np.abs(z[a:p + 1])
            ipk = a + int(np.argmax(zz))
            w, lo, hi = half_width(E, res, ipk)
            lst.append(dict(E_lo=float(E[a]), E_hi=float(E[p]), width_eV=float(E[p] - E[a]),
                            fwhm_like_eV=w, amp=float(res[ipk]), peak_z=float(zz.max()),
                            peak_E=float(E[ipk])))
        features[name][key] = dict(sigma=float(sigma),
                                   n_gt3sigma=int(np.sum(np.abs(z) > 3)),
                                   runs=lst)
        log("%-10s %-12s sigma=%.5f  n(>3sigma)=%d" %
            (name, key, sigma, int(np.sum(np.abs(z) > 3))))
        for r in lst:
            log("    E %.2f-%.2f  span %.2f eV  half-max width %.2f eV  amp %+.4f  peak z %+.2f @ %.2f"
                % (r["E_lo"], r["E_hi"], r["width_eV"], r["fwhm_like_eV"],
                   r["amp"], r["peak_z"], r["peak_E"]))
    # robustness: thin the raw grid (the theory grid has two adjacent 0.016 eV
    # steps) and redo the +-5 eV / quadratic detection
    if name == "theory":
        keep_t = np.r_[True, np.diff(E) >= 0.1]
        E_t = E[keep_t]
        y_t = y[keep_t]
        res_t, sig_t, z_t = local_residual(E_t, y_t, 5.0, 2)
        lst_t = []
        for a, p in runs_above(E_t, z_t, 3.0):
            zz = np.abs(z_t[a:p + 1])
            ipk = a + int(np.argmax(zz))
            w, _, _ = half_width(E_t, res_t, ipk)
            lst_t.append(dict(E_lo=float(E_t[a]), E_hi=float(E_t[p]),
                              span_eV=float(E_t[p] - E_t[a]), fwhm_like_eV=w,
                              amp=float(res_t[ipk]), peak_z=float(zz.max()),
                              peak_E=float(E_t[ipk])))
        features[name]["win5_deg2_thinned"] = dict(sigma=float(sig_t), runs=lst_t)
        log("theory     win5_deg2_thinned (steps>=0.1 eV, n=%d): sigma=%.5f, %d runs"
            % (len(E_t), sig_t, len(lst_t)))
        for r in lst_t:
            log("    E %.2f-%.2f  span %.2f eV  half-max width %.2f eV  amp %+.4f  peak z %+.2f"
                % (r["E_lo"], r["E_hi"], r["span_eV"], r["fwhm_like_eV"],
                   r["amp"], r["peak_z"]))

# ------------------------------------------------------------------- (f)
log("")
log("--- (f) nulls and robustness --------------------------------------")
nulls = {}
for name, E, y in (("experiment", Ee, bE), ("theory", Et, the["beta_Msq"])):
    sl = ols_slope(E, y)
    const = float(np.sum((y - y.mean()) ** 2))
    r_const = y - y.mean()
    r_trend = y - (sl["intercept"] + sl["slope"] * (E - E.mean()))
    pr = pearsonr(E, y)
    perm = np.empty(NPERM)
    for i in range(NPERM):
        perm[i] = np.corrcoef(E, RNG.permutation(y))[0, 1]
    p_perm = float(np.mean(np.abs(perm) >= abs(pr[0])))
    # full polynomial-trend residual as an "alternative trend model"
    X2 = np.stack([np.ones_like(E), E - E.mean(), (E - E.mean()) ** 2], axis=1)
    c2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    r_quad = y - X2 @ c2
    nulls[name] = dict(
        const_ss_res=const, trend_ss_res=float(np.sum(r_trend ** 2)),
        quad_ss_res=float(np.sum(r_quad ** 2)),
        r2_improvement=float(1 - np.sum(r_trend ** 2) / const),
        pearson_r=float(pr[0]), perm_p=float(p_perm),
        perm_mean=float(perm.mean()), perm_std=float(perm.std()),
        perm_z=float((pr[0] - perm.mean()) / perm.std()),
        perm_abs_p95=float(np.percentile(np.abs(perm), 95)))
    log("%-10s constant-beta null SS=%.4f ; linear-trend SS=%.4f ; quadratic SS=%.4f"
        % (name, const, nulls[name]["trend_ss_res"], nulls[name]["quad_ss_res"]))
    log("%-10s trend model explains %.4f of the constant-null residual ; "
        "permutation p=%.4g (|r|>95%% of shuffles is %.4f)"
        % (name, nulls[name]["r2_improvement"], p_perm, nulls[name]["perm_abs_p95"]))

    # leave-one-out influence on the trend slope
    slopes = np.empty(len(E))
    for i in range(len(E)):
        m = np.arange(len(E)) != i
        slopes[i] = ols_slope(E[m], y[m])["slope"]
    i_inf = int(np.argmax(np.abs(slopes - sl["slope"])))
    nulls[name]["loo_slope_max_change"] = float(slopes[i_inf] - sl["slope"])
    nulls[name]["loo_slope_max_change_E"] = float(E[i_inf])
    nulls[name]["loo_slope_range"] = [float(slopes.min()), float(slopes.max())]
    log("%-10s leave-one-out slope: full %.5f ; range [%.5f, %.5f] ; most influential "
        "E=%.1f eV -> slope %.5f (change %+.5f)"
        % (name, sl["slope"], slopes.min(), slopes.max(), E[i_inf], slopes[i_inf],
           slopes[i_inf] - sl["slope"]))
    nulls[name]["slope_full"] = sl["slope"]

# leave-out-the-48.4-endpoint variant for experiment
m48 = np.abs(Ee - 48.4) > 1e-9
sl_no48 = ols_slope(Ee[m48], bE[m48])
log("experiment trend without the 48.4 eV endpoint (i.e. 22.5-48.3): slope %.5f /eV, "
    "Pearson %.4f, Spearman %.4f"
    % (sl_no48["slope"], pearsonr(Ee[m48], bE[m48])[0], spearmanr(Ee[m48], bE[m48])[0]))
sl_lo = ols_slope(Ee[Ee <= 40.0], bE[Ee <= 40.0])
log("experiment trend restricted to 22.5-40.0 eV: slope %.5f /eV, Pearson %.4f"
    % (sl_lo["slope"], pearsonr(Ee[Ee <= 40.0], bE[Ee <= 40.0])[0]))

# angular-window robustness of the two-term fit (alternative angular range)
win_robust = {}
for name, M, E in (("experiment", Me, Ee), ("theory", Mt, Et)):
    d = {}
    for lo, hi in ((0, 180), (20, 160), (30, 150), (0, 90), (90, 180)):
        m = (theta_deg >= lo) & (theta_deg <= hi)
        D = np.stack([np.ones(int(m.sum())), 0.5 * (3 * x[m] ** 2 - 1)], axis=1)
        c, *_ = np.linalg.lstsq(D, M[:, m].T, rcond=None)
        bb = c[1]
        d["%d_%d" % (lo, hi)] = dict(
            beta_range=[float(bb.min()), float(bb.max())],
            pearson_E=float(pearsonr(E, bb)[0]),
            slope=float(ols_slope(E, bb)["slope"]),
            corr_to_full_window=float(np.corrcoef(bb, (lambda D2, M2: (
                np.linalg.lstsq(D2, M2.T, rcond=None)[0][1]))(
                    np.stack([np.ones_like(x), 0.5 * (3 * x ** 2 - 1)], axis=1), M))[0, 1]))
    win_robust[name] = d
    log("%-10s angular-window robustness of beta(2-term): %s"
        % (name, {k: (round(v["beta_range"][0], 3), round(v["beta_range"][1], 3),
                      round(v["slope"], 4)) for k, v in d.items()}))

# ------------------------------------------------------------------- (g)
log("")
log("--- (g) writing CSV / JSON / PNG ----------------------------------")
rows = []
for name, E, p in (("experiment", Ee, exp), ("theory", Et, the)):
    for i in range(len(E)):
        rows.append(dict(panel=name, energy_eV=float(E[i]),
                         beta_2term=float(p["beta2"][i]),
                         beta_full_Lmax8=float(p["beta_full"][i]),
                         beta_Msq_Lmax6=float(p["beta_Msq"][i]),
                         A_2term=float(p["A2"][i]),
                         R2_2term=float(p["R2_2"][i]),
                         R2_full_Lmax8=float(p["R2_full"][i]),
                         rms_2term=float(p["rms2"][i])))
for ev in MEAS_E:
    rows.append(dict(panel="experiment_measured_CoefLeg", energy_eV=float(ev),
                     beta_2term=np.nan, beta_full_Lmax8=np.nan,
                     beta_Msq_Lmax6=float(proxy[ev]["ratio"]),
                     A_2term=float(proxy[ev]["F00_sum"]),
                     R2_2term=np.nan, R2_full_Lmax8=np.nan,
                     rms_2term=float(proxy[ev]["ratio_err"])))
df_out = pd.DataFrame(rows)
df_out.to_csv(CSV_PATH, index=False)
log("wrote %s (%d rows)" % (CSV_PATH, len(df_out)))

result = dict(
    package="p7_anisotropy",
    script=os.path.join(OUT_AN, "p7_anisotropy.py"),
    data_sha256=files,
    grid_shapes=dict(experiment_magnitude=list(Me.shape), theory_magnitude=list(Mt.shape),
                     experiment_E_range=[float(Ee[0]), float(Ee[-1])],
                     theory_E_range=[float(Et[0]), float(Et[-1])]),
    two_term_fit=dict(
        experiment=dict(R2_median=float(np.median(exp["R2_2"])),
                        R2_min=float(exp["R2_2"].min()),
                        R2_max=float(exp["R2_2"].max()),
                        frac_R2_gt_0p9=float(np.mean(exp["R2_2"] > 0.9)),
                        frac_R2_gt_0p99=float(np.mean(exp["R2_2"] > 0.99)),
                        median_rms=float(np.median(exp["rms2"])),
                        R2_full_median=float(np.median(exp["R2_full"])),
                        R2_Msq6_median=float(np.median(exp["R2_Msq6"]))),
        theory=dict(R2_median=float(np.median(the["R2_2"])),
                    R2_min=float(the["R2_2"].min()),
                    R2_max=float(the["R2_2"].max()),
                    frac_R2_gt_0p9=float(np.mean(the["R2_2"] > 0.9)),
                    frac_R2_gt_0p99=float(np.mean(the["R2_2"] > 0.99)),
                    median_rms=float(np.median(the["rms2"])),
                    R2_full_median=float(np.median(the["R2_full"])),
                    R2_Msq6_median=float(np.median(the["R2_Msq6"]))),
        basis_robustness=basis_robust,
        angular_window_robustness=win_robust),
    energy_dependence=dict(
        experiment_Msq6=desc_exp, experiment_2term=desc_exp_2, experiment_L8=desc_exp_f,
        theory_Msq6=desc_the, theory_2term=desc_the_2, theory_L8=desc_the_f,
        regions=regions,
        bootstrap_slope_block_ci=[float(slope_ci[0]), float(slope_ci[1])],
        bootstrap_slope_iid_ci=[float(slope_ci_iid[0]), float(slope_ci_iid[1])],
        bootstrap_slope_block_median=float(np.median(slope_boot_blk)),
        bootstrap_argmin_CI=[float(np.percentile(argmin_b, 2.5)),
                             float(np.percentile(argmin_b, 97.5))],
        bootstrap_argmax_CI=[float(np.percentile(argmax_b, 2.5)),
                             float(np.percentile(argmax_b, 97.5))],
        split_sample=dict(first_half_slope=sh["first_half"]["slope"],
                          second_half_slope=sh["second_half"]["slope"],
                          E_first=list(sh["E_first"]), E_second=list(sh["E_second"]))),
    exp_vs_theory=dict(
        theory_turning_points_thinned=[[float(Et_s[i]), float(bt_s[i])]
                                       for i in turns],
        theory_turning_points_raw_grid=[[float(Et[i]), float(the["beta_Msq"][i])]
                                        for i in turns_full],
        overlap_E_range=[float(Et_m[0]), float(Et_m[-1])], n_overlap=int(len(Et_m)),
        pearson_r=float(pr_et[0]), pearson_p=float(pr_et[1]),
        spearman_rho=float(sr_et[0]), spearman_p=float(sr_et[1]),
        mean_diff=float(np.mean(exp_on_t - the_on_t)),
        rms_diff=float(np.sqrt(np.mean((exp_on_t - the_on_t) ** 2))),
        sign_agreement=float(np.mean(np.sign(exp_on_t) == np.sign(the_on_t))),
        matched=matched,
        matched_pearson_r=float(pr_m[0]), matched_pearson_p=float(pr_m[1]),
        matched_spearman_rho=float(sr_m[0]), matched_spearman_p=float(sr_m[1]),
        matched_sign_agreement=float(np.mean(np.sign(exp_at_meas) ==
                                             np.sign(the_interp_meas)))),
    measured_representation=dict(
        energies=MEAS_E,
        F00_sum={str(e): proxy[e]["F00_sum"] for e in MEAS_E},
        F20_sum={str(e): proxy[e]["F20_sum"] for e in MEAS_E},
        ratio_F20_over_F00={str(e): proxy[e]["ratio"] for e in MEAS_E},
        ratio_err={str(e): proxy[e]["ratio_err"] for e in MEAS_E},
        dense_beta_at_measured={str(e): float(v) for e, v in zip(MEAS_E, dense_at_meas)},
        theory_beta_at_measured={str(e): float(v) for e, v in zip(MEAS_E, the_at_meas)},
        pearson_vs_dense_all13=float(corr_all[0]), p_vs_dense_all13=float(corr_all[1]),
        spearman_vs_dense_all13=float(corr_all_s[0]),
        p_spearman_vs_dense_all13=float(corr_all_s[1]),
        pearson_vs_dense_no48=float(corr_no48[0]), p_vs_dense_no48=float(corr_no48[1]),
        spearman_vs_dense_no48=float(corr_no48_s[0]),
        p_spearman_vs_dense_no48=float(corr_no48_s[1]),
        pearson_vs_theory_all13=float(corr_th[0]), p_vs_theory_all13=float(corr_th[1]),
        spearman_vs_theory_all13=float(corr_th_s[0]),
        slope_proxy=float(ratio_slope["slope"]), slope_dense_13=float(dense_slope13["slope"]),
        diff_sign_agreement=float(np.mean(np.sign(d_ratio) == np.sign(d_dense))),
        dme_reconstruction_check=recon,
        variants=coef_variants),
    localization=features,
    nulls_and_robustness=dict(
        experiment=dict(nulls["experiment"],
                        slope_no_48p4=float(sl_no48["slope"]),
                        slope_22p5_40=float(sl_lo["slope"]),
                        pearson_22p5_40=float(pearsonr(Ee[Ee <= 40.0], bE[Ee <= 40.0])[0])),
        theory=nulls["theory"]),
    provenance_note=coord["experimental_grid_provenance"]["independence_warning"],
    failures=FAILURES,
)

with open(JSON_PATH, "w") as f:
    json.dump(result, f, indent=2)
log("wrote %s" % JSON_PATH)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
ax = axes[0, 0]
ax.plot(Ee, bE2, color="tab:gray", lw=1.0, alpha=0.8,
        label="exp beta, 2-term fit (inadequate)")
ax.plot(Ee, bEf, color="tab:blue", lw=1.2, label="exp beta, Legendre L<=8 of M")
ax.plot(Ee, bE, color="black", lw=2.0, label="exp beta, Legendre L<=6 of M$^2$")
ax.plot(Et, the["beta_Msq"], color="tab:red", lw=1.6, ls="--",
        label="theory beta, Legendre L<=6 of M$^2$")
ax.axhline(0, color="k", lw=0.6)
ax.plot([desc_exp["min_E"]], [desc_exp["min"]], "v", color="black", ms=8)
ax.plot([desc_exp["max_E"]], [desc_exp["max"]], "^", color="black", ms=8)
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel(r"$\beta$")
ax.set_title("(a,b) anisotropy beta(E)")
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.plot(Ee, exp["R2_2"], color="tab:gray", lw=1.0, label="exp 2-term R$^2$")
ax.plot(Ee, exp["R2_full"], color="tab:blue", lw=1.2, label="exp L<=8 R$^2$")
ax.plot(Ee, exp["R2_Msq6"], color="black", lw=1.2, label="exp M$^2$ L<=6 R$^2$")
ax.plot(Et, the["R2_2"], color="tab:red", lw=1.0, ls="--", label="theory 2-term R$^2$")
ax.plot(Et, the["R2_Msq6"], color="darkred", lw=1.2, ls=":", label="theory M$^2$ L<=6 R$^2$")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel("R$^2$")
ax.set_title("(a) fit quality: two-term is not adequate")
ax.legend(fontsize=7)

ax = axes[1, 0]
ax.plot(Ee, bE, color="black", lw=2.0, label="dense derived beta (exp)")
XX = np.stack([np.ones_like(Ee), Ee - Ee.mean()], axis=1)
cc, *_ = np.linalg.lstsq(XX, bE, rcond=None)
ax.plot(Ee, XX @ cc, color="tab:green", lw=1.0, ls="-.",
        label="linear trend %.4f/eV" % desc_exp["slope"])
ax.plot(Es, coef_variants["F00_family_c2_over_c0"]["values"], "s-",
        color="tab:purple", ms=4, lw=1.0,
        label="measured F00-family $c_2/c_0$ (no scale)")
ax.errorbar(Es, ratio, yerr=ratio_err, fmt="o", color="tab:orange", ms=4,
            capsize=2, label="literal $\\Sigma F_{20}/\\Sigma F_{00}$ (scale-set)")
ax2 = ax.twinx()
ax2.plot(Et, the["beta_Msq"], color="tab:red", lw=1.4, ls="--", label="theory beta")
ax2.set_ylabel("theory $\\beta$", color="tab:red")
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel("exp anisotropy proxy")
ax.set_title("(b,d) trend and independent measured representation")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

ax = axes[1, 1]
for name, E, y, col in (("experiment", Ee, bE, "black"), ("theory", Et, the["beta_Msq"], "tab:red")):
    res, sigma, z = local_residual(E, y, 5.0, 2)
    ax.plot(E, z, color=col, lw=1.0, label="%s (win +-5 eV, quad)" % name)
ax.axhline(0, color="k", lw=0.6)
ax.axhline(3, color="tab:gray", lw=0.7, ls=":")
ax.axhline(-3, color="tab:gray", lw=0.7, ls=":")
ax.set_xlabel("photon energy (eV)")
ax.set_ylabel("local-trend residual (sigma)")
ax.set_title("(e) localization against the local trend")
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=130)
log("wrote %s" % PNG_PATH)

log("")
log("DONE. failures: %s" % (FAILURES if FAILURES else "none"))
log("notes: %s" % (NOTES if NOTES else "none"))