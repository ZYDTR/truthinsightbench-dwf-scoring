#!/usr/bin/env python3
"""
Independent test of kinetic-shape mechanisms M1-M5 on conversion_timecourses.tsv.

Implemented from scratch on the raw TSV (no reuse of any other output file).

  M1 INDUCTION / LAG then constant rate : y = r*max(t - t_lag, 0),  r, lag free
  M2 AUTOCATALYTIC / product-coupled    : y = y0*exp(k*t),          y0, k free
  M3 PURE POWER LAW                     : y = a*t^b                (b>1 = super-linear)
  M4 QUADRATIC                          : y = a*t^2 + b*t + c
  M5 SINGLE SHARED SHAPE rescaled       : rank-1 shape, one scale per curve

AICc (Gaussian RSS):
    AIC  = n*ln(RSS/n) + 2k
    AICc = AIC + 2k(k+1)/(n-k-1)          [n = 5, so k >= 4 is undefined]

Independent analysis unit = ONE complete curve.  Points along a curve are not
independent replicates; with n = 5 the AICc correction term is large and the
model ranking is weak evidence.  Everything below is reported with that caveat.
"""
import json
import numpy as np
from scipy.optimize import least_squares

DATA = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data/conversion_timecourses.tsv"
OUT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/mechanism_fits_results.json"

# ------------------------------------------------------------------ read data
formulations, times, ys = [], [], []
with open(DATA) as fh:
    fh.readline()
    for line in fh:
        if not line.strip():
            continue
        fid, tt_, yy_ = line.split()
        if fid not in formulations:
            formulations.append(fid)
            times.append([])
            ys.append([])
        times[-1].append(float(tt_))
        ys[-1].append(float(yy_))
t = np.array(times[0])
y = {f: np.array(v) for f, v in zip(formulations, ys)}
for f in formulations:
    assert np.allclose(times[formulations.index(f)], t), "ragged time grid"
n = len(t)

print("time grid t =", t)
for f in formulations:
    d = np.diff(y[f])
    print("  %s y = %s   increments = %s" % (f, y[f], np.round(d, 5)))


def aicc(rss, nobs, k):
    """AICc; returns (aicc, aic, correction).  Requires nobs-k-1 > 0."""
    if nobs - k - 1 <= 0:
        return None, None, None
    rss = max(rss, 1e-300)
    aic = nobs * np.log(rss / nobs) + 2.0 * k
    corr = 2.0 * k * (k + 1.0) / (nobs - k - 1.0)
    return aic + corr, aic, corr


# ==================================================================== fitters
def fit_m1(tt, yy, lag_lo=-1.5, lag_hi=5.0, n1=1301, n2=2001):
    """y = r*max(t-lag, 0).  For fixed lag, r is closed-form OLS through the
    kink; coarse grid then refined around the optimum (exact to ~1e-5 h)."""
    def scan(lags):
        L = np.maximum(tt[None, :] - lags[:, None], 0.0)
        den = np.sum(L * L, axis=1)
        num = L @ yy
        r = np.where(den > 1e-12, num / np.maximum(den, 1e-300), np.nan)
        res = r[:, None] * L - yy[None, :]
        rss = np.where(np.isfinite(r)[:, None], res ** 2, np.inf).sum(axis=1)
        return rss, r

    lags = np.linspace(lag_lo, lag_hi, n1)
    rss, r = scan(lags)
    i = int(np.argmin(rss))
    if 0 < i < len(lags) - 1:
        step = lags[1] - lags[0]
        lags2 = np.linspace(lags[i] - step, lags[i] + step, n2)
        rss2, r2 = scan(lags2)
        j = int(np.argmin(rss2))
        if rss2[j] < rss[i]:
            rss, r, lags, i = rss2, r2, lags2, j
    return dict(r=float(r[i]), lag=float(lags[i]), rss=float(rss[i]), k_par=2,
                name="M1_lag_rate")


def fit_m2(tt, yy):
    """y = y0*exp(k*t), y0 free (t = 0 anchor NOT forced to zero)."""
    tp, yp = tt[1:], yy[1:]                     # log-space init on t>0 only
    best = None
    starts = []
    for y0_0 in (0.005, 0.02, 0.05, 0.2, 0.5, 1.0, 3.0):
        for k0 in (-1.5, -0.5, 0.0, 0.2, 0.5, 0.8, 1.2):
            starts.append([y0_0, k0])
    A = np.vstack([np.ones_like(tp), tp]).T
    for k0 in np.linspace(-2, 2, 9):
        with np.errstate(divide="ignore"):
            lny = np.log(yp)
            coef, *_ = np.linalg.lstsq(A, lny - k0 * tp, rcond=None)
        starts.append([float(np.exp(coef[0])), float(coef[1] + k0)])

    def res(p):
        return p[0] * np.exp(p[1] * tt) - yy

    for s0 in starts:
        s0 = [min(max(s0[0], 1e-8), 1e3), min(max(s0[1], -3.0), 3.0)]
        try:
            s = least_squares(res, s0, bounds=([1e-8, -3.0], [1e3, 3.0]),
                              method="trf", max_nfev=3000)
        except Exception:
            continue
        c = 2.0 * s.cost
        if best is None or c < best[0] - 1e-15:
            best = (c, s.x.copy())
    rss, p = best
    return dict(y0=float(p[0]), k=float(p[1]), rss=float(rss), k_par=2,
                name="M2_exp")


def fit_m3(tt, yy, a_pen=False):
    """y = a*t^b, y(0)=0 imposed (b>0).  If a_pen=False, only t>0 points enter
    the residual (t=0 residual is identically 0 for a>0,b>0, so this is the same
    fit); if a_pen=True the t=0 point is added explicitly with residual 0."""
    tp, yp = tt[1:], yy[1:]
    best = None
    for b0 in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.5, 3.5):

        def res(p):
            return np.exp(p[0]) * tp ** p[1] - yp

        for A0 in (np.log(max(np.mean(yp), 1e-9)),):
            try:
                s = least_squares(res, [A0, b0],
                                  bounds=([-30.0, 0.02], [12.0, 6.0]),
                                  method="trf", max_nfev=3000)
            except Exception:
                continue
            c = 2.0 * s.cost
            if best is None or c < best[0] - 1e-15:
                best = (c, s.x.copy())
    rss, p = best
    a = float(np.exp(p[0]))
    b = float(p[1])
    return dict(a=a, b=b, rss=float(rss), k_par=2, name="M3_power")


def fit_m4(tt, yy):
    X = np.column_stack([tt ** 2, tt, np.ones_like(tt)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = X @ coef - yy
    return dict(a2=float(coef[0]), a1=float(coef[1]), a0=float(coef[2]),
                rss=float(r @ r), k_par=3, name="M4_quadratic")


def verify_global_minima(tt, yy, fits):
    """Independent brute-force check that the reported RSS minima are global.

    M1: lag grid, r closed form.  M2: k grid, y0 closed form.  M3: b grid, a
    closed form.  Also reports the residual of the best model found on the grid
    vs the optimiser result.
    """
    out = {}
    # --- M1
    lags = np.linspace(-3.0, 6.0, 90001)
    L = np.maximum(tt[None, :] - lags[:, None], 0.0)
    den = np.sum(L * L, axis=1)
    rr = np.where(den > 1e-14, (L @ yy) / np.maximum(den, 1e-300), np.nan)
    res = rr[:, None] * L - yy[None, :]
    rss = np.where(np.isfinite(rr)[:, None], res ** 2, np.inf).sum(axis=1)
    out["M1_global_grid_rss"] = float(rss.min())
    out["M1_reported_rss"] = float(fits["M1_lag_rate"]["rss"])
    # --- M2
    ks = np.linspace(-3.0, 3.0, 12001)
    E = np.exp(ks[:, None] * tt[None, :])
    y0 = (E @ yy) / np.maximum(np.sum(E * E, axis=1), 1e-300)
    res = y0[:, None] * E - yy[None, :]
    out["M2_global_grid_rss"] = float(np.min(np.sum(res ** 2, axis=1)))
    out["M2_reported_rss"] = float(fits["M2_exp"]["rss"])
    # --- M3
    bs = np.linspace(0.02, 6.0, 60001)
    P = tt[None, :] ** bs[:, None]
    den = np.sum(P * P, axis=1)
    a = (P @ yy) / np.maximum(den, 1e-300)
    res = a[:, None] * P - yy[None, :]
    rss = np.sum(res ** 2, axis=1)
    out["M3_global_grid_rss"] = float(rss.min())
    out["M3_reported_rss"] = float(fits["M3_power"]["rss"])
    return out


# ============================================================ per-curve M1-M4
MODELS = ["M1_lag_rate", "M2_exp", "M3_power", "M4_quadratic"]
results = {}
for f in formulations:
    fits = {"M1_lag_rate": fit_m1(t, y[f]),
            "M2_exp": fit_m2(t, y[f]),
            "M3_power": fit_m3(t, y[f]),
            "M4_quadratic": fit_m4(t, y[f])}
    results[f] = dict(
        data=dict(t=t.tolist(), y=y[f].tolist()),
        increments=np.diff(y[f]).tolist(),
        seg_ratio_last_over_first=float((y[f][4] - y[f][3]) / (y[f][1] - y[f][0])),
        fits=fits,
        global_minimum_check=verify_global_minima(t, y[f], fits),
        predicted_at_8h={},
    )

# predicted values at t = 8 h
for f in formulations:
    ff = results[f]["fits"]
    results[f]["predicted_at_8h"] = dict(
        M1_lag_rate=float(ff["M1_lag_rate"]["r"] * max(8.0 - ff["M1_lag_rate"]["lag"], 0.0)),
        M2_exp=float(ff["M2_exp"]["y0"] * np.exp(ff["M2_exp"]["k"] * 8.0)),
        M3_power=float(ff["M3_power"]["a"] * 8.0 ** ff["M3_power"]["b"]),
        M4_quadratic=float(ff["M4_quadratic"]["a2"] * 64.0
                           + ff["M4_quadratic"]["a1"] * 8.0
                           + ff["M4_quadratic"]["a0"]),
    )

# ================================================== M5 shared shape (group)
def rank1_normalised(norm_rows):
    """Best rank-1 fit of already-normalised curves (each row ends at 1).
    Returns shape s (s[-1]=1), per-row scale A, residual RSS."""
    Y = np.asarray(norm_rows)
    U, sv, Vt = np.linalg.svd(Y, full_matrices=False)
    s = Vt[0, :] / Vt[0, -1]
    A = (sv[0] * U[:, 0]) * Vt[0, -1]
    R = A[:, None] * s[None, :] - Y
    return s, A, R


GROUPS = {"group_1_2": ["formulation_1", "formulation_2"],
          "group_3_4": ["formulation_3", "formulation_4"]}
group_out = {}
for gname, mem in GROUPS.items():
    norm = {m: (y[m] / y[m][-1]) for m in mem}
    Yn = np.vstack([norm[m] for m in mem])                 # includes t=0, which is 0
    s_n, A_n, R_n = rank1_normalised(Yn)
    # same fit but ignoring the t=0 column (it is identically 0 for both rows)
    Ysub = Yn[:, 1:]
    U, sv, Vt = np.linalg.svd(Ysub, full_matrices=False)
    s_sub = Vt[0, :] / Vt[0, -1]
    A_sub = (sv[0] * U[:, 0]) * Vt[0, -1]
    R_sub = A_sub[:, None] * s_sub[None, :] - Ysub
    # absolute-units stochastic shared-shape fit: free shape on t=1..4
    # (normalised to 1 at t=4) + one free scale per curve
    Yabs = np.vstack([y[m] for m in mem])[:, 1:]
    U2, sv2, Vt2 = np.linalg.svd(Yabs, full_matrices=False)
    s_abs = Vt2[0, :] / Vt2[0, -1]
    A_abs = (sv2[0] * U2[:, 0]) * Vt2[0, -1]
    R_abs = A_abs[:, None] * s_abs[None, :] - Yabs
    cond = []
    for i, m in enumerate(mem):
        ai = float(np.dot(s_abs, y[m][1:]) / np.dot(s_abs, s_abs))
        ri = ai * s_abs - y[m][1:]
        cond.append(dict(curve=m, scale=ai, rss=float(ri @ ri)))
    d = Yn[0] - Yn[1]
    group_out[gname] = dict(
        members=mem,
        normalised_curves={m: norm[m].tolist() for m in mem},
        normalised_pointwise_diff=d.tolist(),
        normalised_max_abs_diff=float(np.max(np.abs(d))),
        normalised_rms_diff=float(np.sqrt(np.mean(d ** 2))),
        normalised_shared_shape=[0.0] + s_sub.tolist(),
        normalised_shared_scale=A_sub.tolist(),
        rss_shared_normalised_joint_t0_excluded=float(np.sum(R_sub ** 2)),
        rss_shared_normalised_per_curve_t0_excluded=[
            float(np.sum(R_sub[i] ** 2)) for i in range(len(mem))],
        rss_shared_normalised_joint_incl_t0=float(np.sum(R_n ** 2)),
        shared_shape_absolute=[0.0] + s_abs.tolist(),
        shared_scale_absolute=A_abs.tolist(),
        rss_shared_absolute_joint=float(np.sum(R_abs ** 2)),
        rss_shared_absolute_per_curve_conditional_scale=cond,
        k_shared_joint=3 + 2,           # 3 free shape values (t=1,2,3) + 2 scales
        n_points_joint=2 * (n - 1),
    )

# per-curve M5 row: shape taken from own group, scale refit to that curve.
K_EFF_M5 = 1.0 + 3.0 / 2.0     # own scale + shared shape amortised over 2 curves
for gname, mem in GROUPS.items():
    g = group_out[gname]
    s_abs = np.array(g["shared_shape_absolute"])
    for c in g["rss_shared_absolute_per_curve_conditional_scale"]:
        f = c["curve"]
        results[f]["fits"]["M5_shared_shape"] = dict(
            scale=float(c["scale"]), rss=float(c["rss"]), k_par=K_EFF_M5,
            shape=s_abs.tolist(),
            note=("non-parametric shape borrowed from %s; k_eff = 1 own scale + "
                  "3 shared shape values amortised over 2 curves" % gname))
        results[f]["predicted_at_8h"]["M5_shared_shape"] = None

results["_group_shared_shape"] = group_out

# ===================================================== per-formulation AICc
ALL = MODELS + ["M5_shared_shape"]
for f in formulations:
    table = {}
    for m in ALL:
        fit = results[f]["fits"][m]
        k = float(fit["k_par"])
        a, aic, corr = aicc(fit["rss"], n, k)
        table[m] = dict(rss=float(fit["rss"]), k=k, aic=float(aic),
                        aicc_correction=float(corr), aicc=float(a))
    best = min(table, key=lambda m: table[m]["aicc"])
    for m in table:
        table[m]["delta_aicc"] = float(table[m]["aicc"] - table[best]["aicc"])
    order = sorted(table, key=lambda m: table[m]["aicc"])
    w2 = [m for m in order if table[m]["delta_aicc"] <= 2.0]
    w2_core = [m for m in w2 if m != "M5_shared_shape"]
    # M5 k-sensitivity: what if the shared shape is charged fully (k=4 -> AICc
    # undefined at n=5) or partially (k=3)?
    def aicc_with_k(k):
        a, _, _ = aicc(results[f]["fits"]["M5_shared_shape"]["rss"], n, k)
        return a
    results[f]["aicc_table"] = table
    results[f]["aicc_order"] = order
    results[f]["best_model"] = best
    results[f]["aicc_is_defined"] = True
    results[f]["models_within_2_aicc"] = w2
    results[f]["n_models_within_2_of_5"] = len(w2)
    results[f]["models_within_2_aicc_excluding_M5"] = w2_core
    results[f]["n_models_within_2_excluding_M5"] = len(w2_core)
    rss5 = results[f]["fits"]["M5_shared_shape"]["rss"]
    results[f]["M5_k_sensitivity"] = {
        "k2.5_amortised_aicc": aicc_with_k(2.5),
        "d_aicc_vs_best_k2.5": (aicc_with_k(2.5) - table[best]["aicc"])
        if aicc_with_k(2.5) is not None else None,
        "k3.0_aicc": aicc_with_k(3.0),
        "k3.0_within_2": (bool(aicc_with_k(3.0) - table[best]["aicc"] <= 2.0)
                          if aicc_with_k(3.0) is not None else None),
        "k4.0_aicc": aicc_with_k(4.0),      # undefined: n-k-1 = 0
    }
    preds = {m: results[f]["predicted_at_8h"][m] for m in w2
             if results[f]["predicted_at_8h"][m] is not None}
    results[f]["predicted_at_8h_within2"] = preds
    if preds:
        v = np.array(list(preds.values()))
        results[f]["predicted_at_8h_within2_spread"] = dict(
            min=float(v.min()), max=float(v.max()), range=float(v.max() - v.min()),
            pct_of_min=float(100.0 * (v.max() - v.min()) / v.min()))
    else:
        results[f]["predicted_at_8h_within2_spread"] = None
    v4 = np.array([results[f]["predicted_at_8h"][m] for m in MODELS])
    results[f]["predicted_at_8h_all_M1_M4_spread"] = dict(
        min=float(v4.min()), max=float(v4.max()), range=float(v4.max() - v4.min()))
    # next-best rival outside the within-2 set (parametric, for falsification)
    rival = next((m for m in order if m not in w2 and m in MODELS), None)
    results[f]["next_best_rival"] = rival
    if rival is not None:
        results[f]["next_best_rival_delta_aicc"] = table[rival]["delta_aicc"]
        results[f]["next_best_rival_pred_at_8h"] = results[f]["predicted_at_8h"][rival]

# ================================================== robustness: drop t = 0
robust = {}
for f in formulations:
    tt, yy = t[1:], y[f][1:]
    fl = {"M1_lag_rate": fit_m1(tt, yy), "M2_exp": fit_m2(tt, yy),
          "M3_power": fit_m3(tt, yy), "M4_quadratic": fit_m4(tt, yy)}
    tab = {}
    for m, fi in fl.items():
        a, _, _ = aicc(fi["rss"], 4, float(fi["k_par"]))
        tab[m] = dict(rss=float(fi["rss"]), k=float(fi["k_par"]),
                      aicc=(None if a is None else float(a)))
    usable = [m for m in tab if tab[m]["aicc"] is not None]
    b = min(usable, key=lambda m: tab[m]["aicc"])
    for m in tab:
        tab[m]["delta_aicc"] = (None if tab[m]["aicc"] is None
                                else float(tab[m]["aicc"] - tab[b]["aicc"]))
    robust[f] = dict(
        table=tab, best=b,
        note="quadratic has k=3 and n=4 -> AICc undefined; excluded from ranking",
        within2=[m for m in sorted(usable, key=lambda x: tab[x]["aicc"])
                 if tab[m]["delta_aicc"] <= 2.0])

# ======================================================= group-level AICc
group_aicc = {}
for gname, mem in GROUPS.items():
    g = group_out[gname]
    Y = np.vstack([y[m] for m in mem])[:, 1:]         # 2 x 4, t = 1..4
    nn = Y.size                                       # 8
    # (a) shared rank-1 shape: 3 shape values + 2 scales = 5 params
    a_sh, _, _ = aicc(g["rss_shared_absolute_joint"], nn, 5)
    # (b) independent power laws: 2 params per curve = 4
    rss_pow = sum(fit_m3(t, y[m])["rss"] for m in mem)
    a_po, _, _ = aicc(rss_pow, nn, 4)
    # (c) independent quadratics: 3 per curve = 6
    rss_q = sum(fit_m4(t, y[m])["rss"] for m in mem)
    a_q, _, _ = aicc(rss_q, nn, 6)
    # (d) shared shape but only 3 effective params total (2 scales + 1 shape)
    a_sh3, _, _ = aicc(g["rss_shared_absolute_joint"], nn, 3)
    group_aicc[gname] = dict(
        n_points=nn, k_shared_shape=5, k_indep_power=4, k_indep_quad=6,
        rss_shared_shape=float(g["rss_shared_absolute_joint"]),
        rss_indep_power=float(rss_pow), rss_indep_quad=float(rss_q),
        rss_shared_shape_k3=float(g["rss_shared_absolute_joint"]),
        aicc_shared_shape=a_sh, aicc_indep_power=a_po, aicc_indep_quad=a_q,
        d_shared_minus_power=float(a_sh - a_po),
        d_shared_minus_quad=float(a_sh - a_q),
        note=("8 points but only 2 independent curves; this group-level AICc is "
              "indicative only and sits at the edge of AICc validity."))
results["_group_aicc"] = group_aicc

# ============ residual structure (is the winning model actually adequate?)
for f in formulations:
    for m in ALL:
        fit = results[f]["fits"][m]
        if m == "M1_lag_rate":
            pred = fit["r"] * np.maximum(t - fit["lag"], 0.0)
        elif m == "M2_exp":
            pred = fit["y0"] * np.exp(fit["k"] * t)
        elif m == "M3_power":
            pred = fit["a"] * t ** fit["b"]
        elif m == "M4_quadratic":
            pred = fit["a2"] * t ** 2 + fit["a1"] * t + fit["a0"]
        else:
            pred = fit["scale"] * np.array(fit["shape"])
        r = y[f] - pred
        fit["predicted"] = pred.tolist()
        fit["residuals"] = r.tolist()
        fit["residual_sign_pattern"] = "".join("+" if v > 1e-12 else ("-" if v < -1e-12 else "0")
                                               for v in r)
        fit["max_abs_residual_fraction_of_final"] = float(
            np.max(np.abs(r)) / y[f][-1])

# =============== parametric shared-SHAPE test (shared exponent within group)
def fit_shared_exponent(tt, Y):
    """Y ~ a_i * t**b with one shared b (scales the same for both curves).
    Scales closed-form given b; scan + refine b."""
    def rss_at(b):
        P = tt ** b
        den = float(np.dot(P, P))
        A = (Y @ P) / den
        R = A[:, None] * P[None, :] - Y
        return float(np.sum(R ** 2)), A

    bs = np.linspace(0.05, 4.0, 8001)
    vals = np.array([rss_at(b)[0] for b in bs])
    i = int(np.argmin(vals))
    lo, hi = bs[max(i - 1, 0)], bs[min(i + 1, len(bs) - 1)]
    bs2 = np.linspace(lo, hi, 4001)
    vals2 = np.array([rss_at(b)[0] for b in bs2])
    j = int(np.argmin(vals2))
    if vals2[j] < vals[i]:
        rss_b, A_b = rss_at(bs2[j])
        return float(bs2[j]), A_b, float(rss_b)
    rss_b, A_b = rss_at(bs[i])
    return float(bs[i]), A_b, float(rss_b)


shared_exp = {}
for gname, mem in GROUPS.items():
    Y = np.vstack([y[m] for m in mem])[:, 1:]         # t = 1..4, 2 x 4
    b_shared, A_sh, rss_sh = fit_shared_exponent(t[1:], Y)
    rss_ind = sum(fit_m3(t, y[m])["rss"] for m in mem)
    bs_ind = [fit_m3(t, y[m])["b"] for m in mem]
    nn = Y.size
    a_sh, _, _ = aicc(rss_sh, nn, 3)      # 2 scales + 1 shared exponent
    a_ind, _, _ = aicc(rss_ind, nn, 4)    # 2 params * 2 curves
    shared_exp[gname] = dict(
        shared_exponent_b=b_shared, scale_per_curve=A_sh.tolist(),
        rss_shared_exponent=float(rss_sh),
        independent_exponents=bs_ind, rss_independent=float(rss_ind),
        n_points=nn,
        aicc_shared_exponent=(None if a_sh is None else float(a_sh)),
        aicc_independent=(None if a_ind is None else float(a_ind)),
        delta_aicc_shared_minus_independent=float(a_sh - a_ind),
        aicc_shared_minus_independent_within_2=bool(a_sh - a_ind <= 2.0),
        exponent_difference=abs(bs_ind[0] - bs_ind[1]),
        note="2 independent curves only; a 'shared' exponent win is weak evidence.",
    )
results["_shared_exponent"] = shared_exp

# =========== M2 variant: exponential with the t=0 point used only for scoring
m2_alt = {}
for f in formulations:
    tp, yp = t[1:], y[f][1:]

    def res(p, tp=tp, yp=yp):
        return p[0] * np.exp(p[1] * tp) - yp

    best = None
    for y0_0 in (0.02, 0.1, 0.3, 0.6, 1.0, 2.0):
        for k0 in (0.0, 0.2, 0.4, 0.6, 0.9, 1.3):
            try:
                s = least_squares(res, [y0_0, k0], bounds=([1e-9, -3.0], [1e3, 4.0]),
                                  max_nfev=3000)
            except Exception:
                continue
            if best is None or 2 * s.cost < best[0]:
                best = (2 * s.cost, s.x.copy())
    y0, k = best[1]
    full = y0 * np.exp(k * t) - y[f]                    # scored on all 5 points
    m2_alt[f] = dict(
        y0=float(y0), k=float(k),
        rss_fit_points_only_t1_t4=float(best[0]),
        rss_scored_on_all_5=float(full @ full),
        implied_y_at_t0=float(y0),
        predicted_at_8h=float(y0 * np.exp(k * 8.0)),
        note=("exponential parameters from t=1..4; the measured y(0)=0 is used "
              "only for scoring.  Even so the model implies y(0)=%.4f" % y0))
results["_M2_alt_exclude_t0"] = m2_alt

# ====== local exponent diagnostic: is a single power law even admissible?
# For y = a*t^b, consecutive ratios give b = ln(y_j/y_i)/ln(t_j/t_i) exactly.
# A constant implied exponent = power law; drift = not a power law (model error).
local_exp = {}
for f in formulations:
    steps = [(t[k], t[k + 1]) for k in range(1, n - 1)]
    bvals = [float(np.log(y[f][k + 1] / y[f][k]) / np.log(t[k + 1] / t[k]))
             for k in range(1, n - 1)]
    local_exp[f] = dict(
        segment_pairs=["%g-%g" % s for s in steps],
        implied_exponent_b=bvals,
        spread=float(max(bvals) - min(bvals)),
        mean=float(np.mean(bvals)),
        note=("implied b must be exactly constant for a pure power law; this is "
              "a model-checking diagnostic independent of the AICc fits.  "
              "It uses ratios, so it is amplitude- and offset-free (given y(0)=0)."))
results["_local_exponent"] = local_exp

# ====== M5 k-assignment sensitivity: does the within-2 count depend on it?
for f in formulations:
    rss5 = results[f]["fits"]["M5_shared_shape"]["rss"]
    tab = results[f]["aicc_table"]
    k_primary = 2.5
    counts = {}
    for k_alt in (1.5, 2.0, 2.5, 3.0):
        a_alt, _, _ = aicc(rss5, n, k_alt)
        if a_alt is None:
            counts["k=%.1f" % k_alt] = None
            continue
        ref = {m: tab[m]["aicc"] for m in MODELS}          # best parametric
        ref["M5_shared_shape"] = a_alt
        b = min(ref, key=lambda m: ref[m])
        w2 = sorted([m for m in ref if ref[m] - ref[b] <= 2.0],
                    key=lambda m: ref[m])
        counts["k=%.1f" % k_alt] = dict(
            m5_aicc=float(a_alt), m5_delta_vs_best=float(a_alt - ref[b]),
            best=b, within2=w2, n_within2=len(w2))
    results[f]["M5_within2_count_vs_k"] = counts

# ============================== scale-invariant model comparison (key caveat)
# AICc here is computed on RSS in the curve's OWN units, so the four curves have
# wildly different amplitudes (9.48 -> 0.173).  Dividing each residual by the
# curve's own final value makes the comparison scale-free and is reported
# alongside the raw table.
scale_free = {}
for f in formulations:
    base = y[f][-1]
    tab = {}
    for m in ALL:
        fit = results[f]["fits"][m]
        kk = float(fit["k_par"])
        rss_n = float(fit["rss"]) / (base ** 2)
        a, _, _ = aicc(rss_n, n, kk)
        tab[m] = dict(rss_normalised=rss_n, k=kk,
                      aicc=(None if a is None else float(a)))
    usable = [m for m in tab if tab[m]["aicc"] is not None]
    b = min(usable, key=lambda m: tab[m]["aicc"])
    for m in tab:
        tab[m]["delta_aicc"] = (None if tab[m]["aicc"] is None
                                else float(tab[m]["aicc"] - tab[b]["aicc"]))
    w2 = [m for m in sorted(usable, key=lambda x: tab[x]["aicc"])
          if tab[m]["delta_aicc"] <= 2.0]
    scale_free[f] = dict(table=tab, best=b, within2=w2,
                         within2_excluding_M5=[m for m in w2 if m != "M5_shared_shape"],
                         b_var=np.var(y[f], ddof=0) / (base ** 2))
results["_scale_free_aicc"] = scale_free

# residual relative to signal variance (R^2-like, interpretable, unit-free)
for f in formulations:
    base = y[f][-1]
    for m in ALL:
        fit = results[f]["fits"][m]
        fit["rss_fraction_of_signal"] = float(fit["rss"] / np.sum(y[f] ** 2))
        fit["normalised_rmse_of_final_value"] = float(
            np.sqrt(fit["rss"] / n) / base)

# =============================================================== predictions
pred8 = {}
for f in formulations:
    pred8[f] = dict(
        within2=results[f]["predicted_at_8h_within2"],
        within2_spread=results[f]["predicted_at_8h_within2_spread"],
        all_M1_M4={m: results[f]["predicted_at_8h"][m] for m in MODELS},
        M5="non-parametric shared empirical shape; undefined beyond t=4 without an "
           "assumed functional form",
        next_rival=results[f]["next_best_rival"],
        next_rival_val=results[f].get("next_best_rival_pred_at_8h"),
        # reference only: linear extension of the final measured segment
        final_segment_linear_extrapolation=float(
            y[f][-1] + 4.0 * (y[f][-1] - y[f][-2])),
    )
results["_predictions_at_8h"] = pred8

# ==================================================================== report
def fmt(x):
    return "None" if x is None else ("%.4f" % x)


print("\n=== segment-rate ratios (last / first 1-h segment) ===")
for f in formulations:
    print("  %-14s %.3f" % (f, results[f]["seg_ratio_last_over_first"]))

for f in formulations:
    print("\n=== %s : AICc table (n = 5) ===" % f)
    tb = results[f]["aicc_table"]
    for m in results[f]["aicc_order"]:
        fi = results[f]["fits"][m]
        p = {kk: vv for kk, vv in fi.items()
             if kk not in ("name", "note", "shape", "rss", "k_par",
                           "predicted", "residuals", "residual_sign_pattern")}
        print("   %-18s k=%4.1f RSS=%.6g AICc=%9.3f dAICc=%7.3f | %s"
              % (m, tb[m]["k"], tb[m]["rss"], tb[m]["aicc"], tb[m]["delta_aicc"],
                 {kk: round(vv, 5) for kk, vv in p.items()}))
    print("   BEST                 : %s" % results[f]["best_model"])
    print("   within 2 AICc        : %s  (count %d of 5)"
          % (results[f]["models_within_2_aicc"], results[f]["n_models_within_2_of_5"]))
    print("   within 2 AICc (no M5): %s  (count %d)"
          % (results[f]["models_within_2_aicc_excluding_M5"],
             results[f]["n_models_within_2_excluding_M5"]))
    print("   next rival           : %s (dAICc %.3f)"
          % (results[f]["next_best_rival"], results[f].get("next_best_rival_delta_aicc", float("nan"))))
    print("   y(8h) leading cands  : %s" % {k: round(v, 4) for k, v in
                                             results[f]["predicted_at_8h_within2"].items()})
    sp = results[f]["predicted_at_8h_within2_spread"]
    if sp is None:
        print("   y(8h) within-2 spread: n/a (no parametric member in within-2 set)")
    else:
        print("   y(8h) within-2 spread: min %.4f max %.4f range %.4f (%.1f%% of min)"
              % (sp["min"], sp["max"], sp["range"], sp["pct_of_min"]))
    v4 = results[f]["predicted_at_8h_all_M1_M4_spread"]
    print("   y(8h) spread across ALL M1-M4: range %.4f (min %.4f max %.4f)"
          % (v4["range"], v4["min"], v4["max"]))

print("\n=== M5 normalised / shared-shape ===")
for gname, g in group_out.items():
    print("\n %s  (%s)" % (gname, " vs ".join(g["members"])))
    for m in g["members"]:
        print("   normalised %-14s %s" % (m, np.round(g["normalised_curves"][m], 4)))
    print("   pointwise diff       %s" % np.round(g["normalised_pointwise_diff"], 5))
    print("   max|diff| = %.4f   rms = %.4f" % (g["normalised_max_abs_diff"],
                                                 g["normalised_rms_diff"]))
    print("   shared shape (t=0..4) %s" % np.round(g["normalised_shared_shape"], 4))
    print("   scale per curve       %s" % np.round(g["normalised_shared_scale"], 5))
    print("   RSS shared shape (normalised, t=1..4) = %.6g ; per curve %s"
          % (g["rss_shared_normalised_joint_t0_excluded"],
             np.round(g["rss_shared_normalised_per_curve_t0_excluded"], 7)))
    print("   RSS shared shape (absolute units, t=1..4) = %.6g ; per curve %s"
          % (g["rss_shared_absolute_joint"],
             np.round([c["rss"] for c in g["rss_shared_absolute_per_curve_conditional_scale"]], 7)))

print("\n=== group-level AICc (8 points, only 2 independent curves) ===")
for gname, ga in group_aicc.items():
    print("  %s : shared-shape AICc %.3f (k=5) | indep power %.3f (k=4) | "
          "indep quad %.3f (k=6) | dShP %.3f dShQ %.3f"
          % (gname, ga["aicc_shared_shape"], ga["aicc_indep_power"],
             ga["aicc_indep_quad"], ga["d_shared_minus_power"], ga["d_shared_minus_quad"]))

print("\n=== robustness: refit on t = 1..4 only (n = 4) ===")
for f in formulations:
    print("  %-14s best=%s  within2=%s" % (f, robust[f]["best"],
                                           robust[f]["within2"]))

print("\n=== scale-free AICc (RSS normalised by each curve's final value) ===")
for f in formulations:
    sf = scale_free[f]
    print("  %-14s best=%s" % (f, sf["best"]))
    for m in sorted(sf["table"], key=lambda x: (sf["table"][x]["aicc"] is None,
                                                 sf["table"][x]["aicc"])):
        print("      %-18s k=%4.1f rss(norm)=%.3g dAICc=%s"
              % (m, sf["table"][m]["k"], sf["table"][m]["rss_normalised"],
                 "n/a" if sf["table"][m]["delta_aicc"] is None
                 else "%.3f" % sf["table"][m]["delta_aicc"]))
    print("      within2=%s ; within2 (no M5)=%s"
          % (sf["within2"], sf["within2_excluding_M5"]))

print("\n=== residual structure of each model (sign pattern, t=0..4) ===")
for f in formulations:
    print("  %s" % f)
    for m in ALL:
        fi = results[f]["fits"][m]
        print("      %-18s resid=%s  signs=%s  max|r|/y(4)=%.4f"
              % (m, np.round(fi["residuals"], 6), fi["residual_sign_pattern"],
                 fi["max_abs_residual_fraction_of_final"]))

print("\n=== shared-exponent power law within each group ===")
for gname, se in shared_exp.items():
    print("  %s : b_shared=%.4f  scales=%s  RSS_shared=%.4g (k=3) vs "
          "indep exponents %s RSS=%.4g (k=4)"
          % (gname, se["shared_exponent_b"], np.round(se["scale_per_curve"], 5),
             se["rss_shared_exponent"], np.round(se["independent_exponents"], 4),
             se["rss_independent"]))
    print("      exponents differ by %.4f ; dAICc(shared - indep) = %.4f"
          % (se["exponent_difference"], se["delta_aicc_shared_minus_independent"]))

print("\n=== M2 alternative: exponential fitted on t=1..4, scored on all 5 ===")
for f in formulations:
    a = m2_alt[f]
    print("  %-14s y0=%.5f k=%.5f  RSS(t1..4)=%.4g  RSS(all 5)=%.4g  "
          "implies y(0)=%.5f  y(8h)=%.4f"
          % (f, a["y0"], a["k"], a["rss_fit_points_only_t1_t4"],
             a["rss_scored_on_all_5"], a["implied_y_at_t0"], a["predicted_at_8h"]))

print("\n=== local exponent diagnostic (constancy => pure power law) ===")
for f in formulations:
    le = local_exp[f]
    print("  %-14s segments=%s  implied b=%s  spread=%.4f  mean=%.4f"
          % (f, le["segment_pairs"], np.round(le["implied_exponent_b"], 4),
             le["spread"], le["mean"]))

print("\n=== M5 within-2 count as a function of its effective k ===")
for f in formulations:
    print("  %s" % f)
    for kk, v in results[f]["M5_within2_count_vs_k"].items():
        if v is None:
            print("      %-6s n/a" % kk)
        else:
            print("      %-6s M5 dAICc=%7.3f best=%-16s within2=%s"
                  % (kk, v["m5_delta_vs_best"], v["best"], v["within2"]))

print("\n=== predicted values at t = 8 h ===")
for f in formulations:
    print("  %s" % f)
    for m in MODELS:
        print("     %-16s %10.4f" % (m, results[f]["predicted_at_8h"][m]))
    print("     %-16s %10.4f  (reference: linear extension of final segment)"
          % ("[final-seg lin]", pred8[f]["final_segment_linear_extrapolation"]))
    print("     within-2 spread: %s" % results[f]["predicted_at_8h_within2_spread"])

with open(OUT, "w") as fh:
    json.dump(results, fh, indent=2)
print("\nwrote", OUT)