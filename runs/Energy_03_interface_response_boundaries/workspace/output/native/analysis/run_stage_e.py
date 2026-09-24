"""Stage E: sigmoid light-off model, off-curve validation, family split, T50 model.

Fits a three-parameter logistic light-off model
    y(T) = y_max / (1 + exp(-(T - T50) / w))
by Gauss-Newton on a loss that is *relative* (log-space) so that the wide
dynamic range (0.06 -> 93) does not let the top decade dominate the fit.

Validation follows DATA_GUIDE.md: holdouts are whole curves or contiguous
setting blocks.  No random curve-point splitting is used anywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tib_stats as ts  # noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(parents=True, exist_ok=True)


def dump(name, payload):
    (RES / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[artifact] results/{name}")


# ---------------------------------------------------------------------------
# Logistic light-off fit
# ---------------------------------------------------------------------------
def logistic(T, ymax, T50, w):
    return ymax / (1.0 + np.exp(-(T - T50) / w))


def fit_logistic(T, y, *, ymax_fixed=None, max_iter=200):
    """Gauss-Newton / Levenberg-style fit minimising sum of squared log residuals.

    Parameters are (ymax, T50, w).  With fewer than 3 strictly-positive points
    the fit is not identifiable and the caller receives `identifiable=False`.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    if np.sum(y > 0) < 3:
        return {"identifiable": False, "reason": "fewer_than_3_positive_points"}

    res = {"identifiable": True, "ymax_fixed": ymax_fixed}
    free = [] if ymax_fixed is not None else ["ymax"]
    p = []
    if ymax_fixed is None:
        p.append(max(y.max() * 1.15, 1e-6))
    # crude T50/w start from the log-logistic transform on interior points
    pos = y > 0
    lv = np.log(np.clip(y[pos], 1e-12, None) /
                np.clip(np.maximum(y.max() * 1.05 - y[pos], 1e-12), 1e-12, None))
    if np.ptp(lv) > 0:
        st = ts.ols(np.column_stack([np.ones(pos.sum()), T[pos]]), lv)
        w0 = 1.0 / st["beta"][1] if st["beta"][1] > 1e-6 else 30.0
        T500 = -st["beta"][0] / st["beta"][1] if abs(st["beta"][1]) > 1e-9 else float(np.median(T))
    else:
        w0, T500 = 30.0, float(np.median(T))
    p += [float(np.clip(T500, T.min() - 100, T.max() + 100)), float(abs(w0))]
    p = np.array(p, float)
    lo_bound = [1e-6, -500.0, 1.0] if ymax_fixed is None else [-500.0, 1.0]
    hi_bound = [1e4, 2000.0, 1e3] if ymax_fixed is None else [2000.0, 1e3]
    p = np.clip(p, lo_bound, hi_bound)

    lam = 1e-3
    def unpack(pv):
        if ymax_fixed is None:
            return pv[0], pv[1], pv[2]
        return ymax_fixed, pv[0], pv[1]

    def cost(pv):
        ym, t50, w = unpack(pv)
        if w <= 0 or ym <= 0:
            return np.inf, None
        pred = logistic(T, ym, t50, w)
        if np.any(pred <= 0):
            return np.inf, None
        r = np.log(y) - np.log(pred) if np.all(y > 0) else None
        if r is None:
            m = y > 0
            r = np.log(y[m]) - np.log(pred[m])
        return float(r @ r), r

    c0, _ = cost(p)
    for _ in range(max_iter):
        ym, t50, w = unpack(p)
        pred = logistic(T, ym, t50, w)
        m = y > 0
        # numerical Jacobian in log-prediction space
        J = []
        eps = 1e-6
        for k in range(len(p)):
            pp = p.copy()
            pp[k] += eps * max(1.0, abs(p[k]))
            ymp, t50p, wp = unpack(pp)
            if wp <= 0 or ymp <= 0:
                J.append(np.zeros(m.sum()))
                continue
            J.append((np.log(logistic(T[m], ymp, t50p, wp)) -
                      np.log(pred[m])) / (eps * max(1.0, abs(p[k]))))
        J = np.column_stack(J)
        r = np.log(y[m]) - np.log(pred[m])
        try:
            step = np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        improved = False
        for _try in range(30):
            cand = p + step
            if ymax_fixed is None:
                cand[0] = max(cand[0], 1e-6)
            cand[-1] = max(cand[-1], 1e-3)
            c1, _ = cost(cand)
            if c1 < c0:
                p, c0 = cand, c1
                lam = max(lam * 0.5, 1e-9)
                improved = True
                break
            step = step * 0.5
        if not improved or np.max(np.abs(step)) < 1e-10:
            break

    ym, t50, w = unpack(p)
    pred = logistic(T, ym, t50, w)
    m = y > 0
    r = np.log(y[m]) - np.log(pred[m])
    rss = float(r @ r)
    n_used = int(m.sum())
    k = len(p) + 1  # + sigma
    aic, bic = ts.aic_bic(rss, n_used, k)
    # residual-based standard errors (Gauss-Newton covariance)
    eps = 1e-6
    J = []
    for kk in range(len(p)):
        pp = p.copy()
        pp[kk] += eps * max(1.0, abs(p[kk]))
        ymp, t50p, wp = unpack(pp)
        J.append((np.log(logistic(T[m], ymp, t50p, wp)) - np.log(pred[m])) /
                 (eps * max(1.0, abs(p[kk]))))
    J = np.column_stack(J)
    dof = max(n_used - len(p), 1)
    sigma2 = rss / dof
    try:
        cov = sigma2 * np.linalg.pinv(J.T @ J)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:  # pragma: no cover
        se = np.full(len(p), np.nan)
    names = (["ymax"] if ymax_fixed is None else []) + ["T50", "w"]
    params = dict(zip(names, [float(v) for v in p]))
    ses = dict(zip(names, [float(v) for v in se]))
    resid_all = np.where(y > 0, np.log(np.clip(y, 1e-12, None)) -
                         np.log(np.clip(logistic(T, ym, t50, w), 1e-12, None)), np.nan)

    # pseudo-R2 in log space on positive points
    tss = float(((np.log(y[m]) - np.log(y[m]).mean()) ** 2).sum())
    return {
        "identifiable": True,
        "ymax": float(ym), "T50": float(t50), "w": float(w),
        "param_se": ses, "params": params,
        "rss_log": rss, "n_points": n_used, "k_params": len(p),
        "aic": aic, "bic": bic,
        "r2_log": float(1.0 - rss / tss) if tss > 0 else float("nan"),
        "resid_sd_log": float(np.sqrt(sigma2)),
        "max_abs_resid_log": float(np.max(np.abs(r))),
        "resid_all_settings": [None if not np.isfinite(v) else float(v) for v in resid_all],
    }


def fit_3p_ymax_free(T, y):
    """Convenience wrapper on the shared setting range."""
    return fit_logistic(T, y)


# ---------------------------------------------------------------------------
# Stage E driver
# ---------------------------------------------------------------------------
def main():
    main_df = pd.read_csv(DATA / "anonymous_condition_response_curves.tsv", sep="\t")
    check_df = pd.read_csv(DATA / "independent_condition_check_curves.tsv", sep="\t")
    m_piv = main_df.pivot(index="condition_id", columns="temperature_like_setting",
                          values="rate_like_response")
    c_piv = check_df.pivot(index="condition_id", columns="temperature_like_setting",
                           values="rate_like_response")
    T = m_piv.columns.values.astype(float)
    Tc = c_piv.columns.values.astype(float)
    assert np.array_equal(T, Tc), "panels must share setting grid"

    out = {"per_curve_logistic": {"main": {}, "check": {}},
           "provenance": "logistic y=ymax/(1+exp(-(T-T50)/w)) fit by Gauss-Newton "
                         "on log-residuals; ymax free unless noted"}

    fits = {"main": {}, "check": {}}
    for lab, piv in (("main", m_piv), ("check", c_piv)):
        tt = T if lab == "main" else Tc
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            f = fit_3p_ymax_free(tt, y)
            fits[lab][cid] = f
            out["per_curve_logistic"][lab][cid] = f

    # ---- ymax-fixed sensitivity: is the ceiling actually 100? -------------
    sens = {"main": {}, "check": {}}
    for lab, piv in (("main", m_piv), ("check", c_piv)):
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            sens[lab][cid] = fit_logistic(T, y, ymax_fixed=100.0)
    out["ymax_fixed_100_sensitivity"] = sens
    out["ymax_free_summary"] = {}
    for lab in ("main", "check"):
        ym = [fits[lab][c]["ymax"] for c in fits[lab] if fits[lab][c].get("identifiable")]
        se = [fits[lab][c]["param_se"].get("ymax", np.nan) for c in fits[lab]
              if fits[lab][c].get("identifiable")]
        out["ymax_free_summary"][lab] = {
            "ymax_values": ym,
            "ymax_mean": float(np.mean(ym)), "ymax_sd": float(np.std(ym, ddof=1)),
            "ymax_min": float(np.min(ym)), "ymax_max": float(np.max(ym)),
            "mean_ymax_se": float(np.nanmean(se)),
            "includes_100_within_2se": bool(np.all(
                [abs(v - 100.0) <= 2 * s for v, s in zip(ym, se)])),
        }

    # ---- off-curve prediction: leave-one-curve-out is impossible for the
    # hidden conditions, so validate by *parameter transfer*: use each curve's
    # own T50 and each curve's own w to predict a DIFFERENT curve, and compare
    # with the reverse.  The decisive test is whether a global (T50,w) with
    # per-curve T50 alone predicts held-out curves.
    xfer = {}
    for lab, piv in (("main", m_piv), ("check", c_piv)):
        cids = list(piv.index)
        rows = []
        for src in cids:
            for dst in cids:
                if src == dst:
                    continue
                fs, fd = fits[lab][src], fits[lab][dst]
                if not (fs.get("identifiable") and fd.get("identifiable")):
                    continue
                y_dst = piv.loc[dst].values.astype(float)
                m = y_dst > 0
                # shared probe: same ymax, same w, source T50, target curve
                pred = logistic(T[m], np.mean([fs["ymax"], fd["ymax"]]),
                                fs["T50"], fd["w"])
                err = np.log(y_dst[m]) - np.log(pred)
                rows.append({"src": src, "dst": dst,
                             "rmse_log_using_src_T50_and_dst_w": float(np.sqrt((err @ err) / len(err)))})
        xfer[lab] = rows
    out["parameter_transfer"] = xfer
    out["parameter_transfer_summary"] = {
        lab: {"mean_rmse_log": float(np.mean([r["rmse_log_using_src_T50_and_dst_w"]
                                              for r in rows])),
              "n_pairs": len(rows)}
        for lab, rows in xfer.items()
    }

    # ---- T50 as the single varying parameter across curves ---------------
    # Fit a model with common ymax and common w but per-curve T50 (2 shared + n params)
    def fit_shared_shape(piv, tt):
        cids = list(piv.index)
        Y = {c: piv.loc[c].values.astype(float) for c in cids}

        def negcost(theta):
            ymax, w = theta[0], theta[1]
            t50 = theta[2:]
            rss = 0.0
            if ymax <= 0 or w <= 0:
                return 1e18, None
            preds = {}
            for i, c in enumerate(cids):
                y = Y[c]
                m = y > 0
                pred = logistic(tt[m], ymax, t50[i], w)
                if np.any(pred <= 0):
                    return 1e18, None
                r = np.log(y[m]) - np.log(pred)
                rss += float(r @ r)
                preds[c] = (r, m)
            return rss, preds

        # start from independent fits
        p = [float(np.mean([fits_lab[c]["ymax"] for c in cids])),
             float(np.mean([fits_lab[c]["w"] for c in cids]))] + \
            [float(fits_lab[c]["T50"]) for c in cids]
        p = np.array(p, float)
        c0, _ = negcost(p)
        lam = 1e-3
        for _ in range(400):
            J = []
            eps = 1e-6
            for k in range(len(p)):
                pp = p.copy()
                pp[k] += eps * max(1.0, abs(p[k]))
                c1, _ = negcost(pp)
                J.append(np.full(1, (c1 - c0) / (eps * max(1.0, abs(p[k])))))
            c1, _ = negcost(p)
            g = np.array([j[0] for j in J])
            if not np.all(np.isfinite(g)) or np.max(np.abs(g)) < 1e-12:
                break
            step = -g / (np.linalg.norm(g) + 1e-12) * 0.5
            improved = False
            for _t in range(40):
                cand = p + step
                cand[0] = max(cand[0], 1e-3)
                cand[1] = max(cand[1], 1e-3)
                cc, _ = negcost(cand)
                if cc < c0:
                    p, c0 = cand, cc
                    improved = True
                    break
                step = step * 0.5
            if not improved or np.max(np.abs(step)) < 1e-9:
                break
        rss, preds = negcost(p)
        n = sum(int(v[1].sum()) for v in preds.values())
        k = len(p) + 1
        aic, bic = ts.aic_bic(rss, n, k)
        per = {}
        for i, c in enumerate(cids):
            r, m = preds[c]
            per[c] = {"T50": float(p[2 + i]), "rmse_log": float(np.sqrt(r @ r / len(r))),
                      "max_abs_resid_log": float(np.max(np.abs(r)))}
        return {"shared_ymax": float(p[0]), "shared_w": float(p[1]),
                "per_curve": per, "rss": float(rss), "n_points": int(n),
                "k_params": int(len(p)), "aic": aic, "bic": bic}

    shared = {}
    for lab, piv in (("main", m_piv), ("check", c_piv)):
        fits_lab = fits[lab]
        ttx = T
        shared[lab] = fit_shared_shape(piv, ttx)
    out["shared_shape_model"] = shared

    # Compare: sum of independent 3-param fits vs shared-shape model
    cmp_rows = []
    for lab, piv in (("main", m_piv), ("check", c_piv)):
        ind_rss = sum(fits[lab][c]["rss_log"] for c in piv.index)
        ind_n = sum(fits[lab][c]["n_points"] for c in piv.index)
        ind_k = sum(fits[lab][c]["k_params"] for c in piv.index) + len(piv.index) * 0  # sigma per curve already counted
        ind_k = sum(fits[lab][c]["k_params"] + 1 for c in piv.index)
        ind_aic, ind_bic = ts.aic_bic(ind_rss, ind_n, ind_k)
        sh = shared[lab]
        cmp_rows.append({
            "panel": lab, "n_curves": len(piv.index),
            "independent_rss": float(ind_rss), "independent_k": ind_k,
            "independent_aic": ind_aic, "independent_bic": ind_bic,
            "shared_rss": sh["rss"], "shared_k": sh["k_params"] + 1,
            "shared_aic": sh["aic"], "shared_bic": sh["bic"],
            "delta_bic_shared_minus_independent": sh["bic"] - ind_bic,
            "shared_ymax": sh["shared_ymax"], "shared_w": sh["shared_w"],
        })
    out["shared_vs_independent"] = cmp_rows

    # F test: shared shape (2 + n params) vs independent (3n params), log space
    ftests = []
    for row, lab, piv in ((cmp_rows[0], "main", m_piv), (cmp_rows[1], "check", c_piv)):
        df1 = row["independent_k"] - 1 - (row["shared_k"] - 1)
        df2 = row["independent_rss"] and max(row["independent_rss"] / 1.0, 1e-12)
        df2n = row["n_curves"]
        # df for numerator = extra params = (3n) - (2+n) = 2n - 2
        n = row["n_curves"]
        df1 = 2 * n - 2
        df_resid = ind_n if False else None
        n_ind = sum(fits[lab][c]["n_points"] for c in piv.index)
        df2 = n_ind - 3 * n
        F = ((row["shared_rss"] - row["independent_rss"]) / df1) / (row["independent_rss"] / df2)
        p = float(ts.student_t_sf(np.sqrt(max(F, 0)), df2) * 2)
        ftests.append({"panel": lab, "F": float(F), "df1": df1, "df2": df2,
                       "p_two_sided": p})
    out["shared_shape_f_test"] = ftests

    dump("E_sigmoid_model.json", out)

    print("\n=== per-curve logistic fits (main) ===")
    print(f"{'curve':>18s} {'ymax':>9s} {'+-se':>8s} {'T50':>8s} {'+-se':>7s} {'w':>7s} {'R2log':>7s} {'maxres':>7s}")
    for c, f in out["per_curve_logistic"]["main"].items():
        print(f"{c:>18s} {f['ymax']:9.2f} {f['param_se'].get('ymax',float('nan')):8.2f} "
              f"{f['T50']:8.2f} {f['param_se'].get('T50',float('nan')):7.2f} {f['w']:7.2f} "
              f"{f['r2_log']:7.4f} {f['max_abs_resid_log']:7.4f}")
    print("\n=== per-curve logistic fits (check) ===")
    for c, f in out["per_curve_logistic"]["check"].items():
        print(f"{c:>18s} {f['ymax']:9.2f} {f['param_se'].get('ymax',float('nan')):8.2f} "
              f"{f['T50']:8.2f} {f['param_se'].get('T50',float('nan')):7.2f} {f['w']:7.2f} "
              f"{f['r2_log']:7.4f} {f['max_abs_resid_log']:7.4f}")
    print("\nymax free summary:", json.dumps(out["ymax_free_summary"], indent=2))
    print("\nshared shape:", json.dumps(shared, indent=2, default=str))
    print("\nshared vs independent:", json.dumps(cmp_rows, indent=2))
    print("\nF tests:", json.dumps(ftests, indent=2))
    print("\ntransfer summary:", json.dumps(out["parameter_transfer_summary"], indent=2))


if __name__ == "__main__":
    main()