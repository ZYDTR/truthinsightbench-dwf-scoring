"""Stage H: the decisive shape test.

Question: is the observed spread of apparent (wide-window) thermal exponents
across conditions a genuine difference in intrinsic activation behaviour, or
is it an artifact of comparing curves that sit at different positions on a
common saturating response?

Design: hold the ceiling at the data-supported value ymax=100 and ask whether
one shared width `w` plus a per-curve midpoint T50 explains the data as well
as per-curve `w` and T50.  If the shared-width model is not rejected, the
observed exponent spread is attributable to T50 placement rather than to a
different intrinsic exponent.  A synthetic-curve reconstruction closes the
loop: curves generated with ONE common intrinsic exponent and only T50 varying
are pushed back through the same estimation pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tib_stats as ts  # noqa: E402
from run_stage_e import fit_logistic, logistic  # noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
RES = Path(__file__).resolve().parents[1] / "results"


def dump(name, payload):
    (RES / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[artifact] results/{name}")


def load():
    m = pd.read_csv(DATA / "anonymous_condition_response_curves.tsv", sep="\t")
    c = pd.read_csv(DATA / "independent_condition_check_curves.tsv", sep="\t")
    return (m.pivot(index="condition_id", columns="temperature_like_setting",
                    values="rate_like_response"),
            c.pivot(index="condition_id", columns="temperature_like_setting",
                    values="rate_like_response"))


def numer_gd(cost, p0, *, iters=6000, base=0.4, ymax_floor=None):
    """Deterministic numerical-gradient descent with backtracking (no RNG)."""
    p = np.array(p0, float)
    c0 = cost(p)
    if not np.isfinite(c0):
        return p, c0
    for _ in range(iters):
        g = np.zeros_like(p)
        for k in range(len(p)):
            h = 1e-6 * max(1.0, abs(p[k]))
            pp = p.copy()
            pp[k] += h
            if ymax_floor is not None:
                pp = np.maximum(pp, ymax_floor)
            c1 = cost(pp)
            g[k] = (c1 - c0) / h if np.isfinite(c1) else 0.0
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        step = base / (1.0 + 0.001 * _)
        improved = False
        for _t in range(60):
            cand = p - step * g / (gn + 1e-300)
            if ymax_floor is not None:
                cand = np.maximum(cand, ymax_floor)
            cc = cost(cand)
            if np.isfinite(cc) and cc < c0:
                p, c0 = cand, cc
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return p, c0


def shared_width_model(piv, Tfit, ymax_fixed=100.0):
    cids = list(piv.index)
    starts = [fit_logistic(Tfit, piv.loc[c, Tfit].values.astype(float))
              for c in cids]
    w0 = float(np.mean([s["w"] for s in starts]))

    def rss_shared(p):
        w, t50s = p[0], p[1:]
        if w <= 0:
            return np.inf
        tot = 0.0
        for i, c in enumerate(cids):
            y = piv.loc[c, Tfit].values.astype(float)
            ok = y > 0
            pred = logistic(Tfit[ok], ymax_fixed, t50s[i], w)
            if np.any(pred <= 0):
                return np.inf
            r = np.log(y[ok]) - np.log(pred)
            tot += float(r @ r)
        return tot

    def rss_per(p):
        tot = 0.0
        for i, c in enumerate(cids):
            y = piv.loc[c, Tfit].values.astype(float)
            ok = y > 0
            pred = logistic(Tfit[ok], ymax_fixed, p[2 * i], p[2 * i + 1])
            if np.any(pred <= 0):
                return np.inf
            r = np.log(y[ok]) - np.log(pred)
            tot += float(r @ r)
        return tot

    p_shared = np.array([w0] + [s["T50"] for s in starts])
    p_shared, rss_s = numer_gd(rss_shared, p_shared, ymax_floor=np.array(
        [0.5] + [-400.0] * len(cids)))
    p_per = np.array([v for s in starts for v in (s["T50"], s["w"])])
    p_per, rss_p = numer_gd(rss_per, p_per, ymax_floor=np.array(
        [-400.0, 0.5] * len(cids)))

    n = int(sum((piv.loc[c, Tfit].values.astype(float) > 0).sum() for c in cids))
    k_s, k_p = 1 + len(cids) + 1, 2 * len(cids) + 1
    aic_s, bic_s = ts.aic_bic(rss_s, n, k_s)
    aic_p, bic_p = ts.aic_bic(rss_p, n, k_p)
    df1, df2 = k_p - k_s, n - k_p
    F = ((rss_s - rss_p) / df1) / (rss_p / df2)
    # nested F: per-curve-w is the unrestricted model
    p_val = float(2.0 * ts.student_t_sf(np.sqrt(max(F, 0.0)), df2))
    return {
        "ymax_fixed": ymax_fixed,
        "shared_width": {"params": k_s - 1, "rss": float(rss_s), "aic": aic_s, "bic": bic_s,
                         "w_K": float(p_shared[0]),
                         "T50_per_curve": {c: float(p_shared[1 + i])
                                           for i, c in enumerate(cids)}},
        "per_curve_width": {"params": k_p - 1, "rss": float(rss_p), "aic": aic_p, "bic": bic_p,
                            "T50_per_curve": {c: float(p_per[2 * i])
                                              for i, c in enumerate(cids)},
                            "w_per_curve": {c: float(p_per[2 * i + 1])
                                            for i, c in enumerate(cids)}},
        "n_points": n,
        "nested_F": {"F": float(F), "df1": df1, "df2": df2, "p_two_sided": p_val},
        "delta_bic_per_minus_shared": bic_p - bic_s,
        "per_curve_w_spread": {
            "min_K": float(min(p_per[2 * i + 1] for i in range(len(cids)))),
            "max_K": float(max(p_per[2 * i + 1] for i in range(len(cids)))),
            "sd_K": float(np.std([p_per[2 * i + 1] for i in range(len(cids))], ddof=1)),
        },
    }


def synthetic_reconstruction(piv, Tfit, T_all, ymax, w, t50s):
    """Push curves with ONE intrinsic exponent through the same estimator."""
    rows = []
    for cid in piv.index:
        y = piv.loc[cid].values.astype(float)
        syn = logistic(T_all, ymax, t50s[cid], w)
        m = (y > 0) & (syn > 0) & (T_all >= 140)
        if m.sum() < 4:
            continue
        f_obs = ts.ols(np.column_stack([np.ones(m.sum()), T_all[m]]), np.log(y[m]))
        f_syn = ts.ols(np.column_stack([np.ones(m.sum()), T_all[m]]), np.log(syn[m]))
        # narrow window on the observed curve
        mn = (y >= 1.0) & (y <= 20.0)
        f_nar = (ts.ols(np.column_stack([np.ones(mn.sum()), T_all[mn]]),
                        np.log(y[mn]))["beta"][1] if mn.sum() >= 4 else None)
        rows.append({
            "curve": cid,
            "T50_true_used": float(t50s[cid]),
            "wide_window_slope_observed": float(f_obs["beta"][1]),
            "wide_window_slope_synthetic_same_exponent": float(f_syn["beta"][1]),
            "narrow_window_slope_observed": (None if f_nar is None else float(f_nar)),
            "wide_minus_narrow_observed": (None if f_nar is None
                                           else float(f_obs["beta"][1] - f_nar)),
        })
    obs = [r["wide_window_slope_observed"] for r in rows]
    syn = [r["wide_window_slope_synthetic_same_exponent"] for r in rows]
    t50 = [r["T50_true_used"] for r in rows]
    return {
        "protocol": "synthetic curves y=ymax/(1+exp(-(T-T50_curve)/w_shared)) sampled on "
                    "the measured setting grid, then pushed through the identical "
                    "wide-window log-linear estimator",
        "generator_ymax": ymax, "generator_w": w,
        "generator_intrinsic_exponent_is_IDENTICAL_for_all_curves": True,
        "curves": rows,
        "spearman_observed_slope_vs_T50": ts.spearman(t50, obs),
        "spearman_synthetic_slope_vs_T50": ts.spearman(t50, syn),
        "spearman_observed_vs_synthetic": ts.spearman(obs, syn),
        "observed_slope_range": [float(min(obs)), float(max(obs))],
        "synthetic_slope_range": [float(min(syn)), float(max(syn))],
        "ratio_observed_range_over_synthetic_range":
            float((max(obs) - min(obs)) / (max(syn) - min(syn))) if max(syn) > min(syn) else None,
    }


def main():
    mp, cp = load()
    T = mp.columns.values.astype(float)
    Tfit = T[T >= 140]
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        res = shared_width_model(piv, Tfit, ymax_fixed=100.0)
        out[lab] = {"shape_test": res}
        syn = synthetic_reconstruction(piv, Tfit, T, 100.0,
                                       res["shared_width"]["w_K"],
                                       res["shared_width"]["T50_per_curve"])
        out[lab]["synthetic_reconstruction"] = syn
    dump("H_shape_test.json", out)

    print("===== H: shared-width vs per-curve-width, ceiling fixed at 100 =====")
    for lab in ("main", "check"):
        r = out[lab]["shape_test"]
        print(f"\n[{lab}] n_points={r['n_points']}")
        print(f"  shared w + per-curve T50 : k={r['shared_width']['params']:>2d} "
              f"RSS={r['shared_width']['rss']:.5f} BIC={r['shared_width']['bic']:.2f} "
              f"w={r['shared_width']['w_K']:.2f} K")
        print(f"  per-curve w and T50      : k={r['per_curve_width']['params']:>2d} "
              f"RSS={r['per_curve_width']['rss']:.5f} BIC={r['per_curve_width']['bic']:.2f}")
        print(f"  nested F={r['nested_F']['F']:.3f} df1={r['nested_F']['df1']} "
              f"df2={r['nested_F']['df2']} p={r['nested_F']['p_two_sided']:.5f}")
        print(f"  delta BIC (per - shared) = {r['delta_bic_per_minus_shared']:+.3f}")
        print(f"  per-curve w spread = {r['per_curve_w_spread']}")
        s = out[lab]["synthetic_reconstruction"]
        print(f"  synthetic check: obs slope range={s['observed_slope_range']}")
        print(f"                   syn slope range={s['synthetic_slope_range']}")
        print(f"                   obs/syn range ratio="
              f"{s['ratio_observed_range_over_synthetic_range']:.3f}")
        print(f"                   Spearman(obs slope, syn slope)="
              f"{s['spearman_observed_vs_synthetic']}")
        print(f"                   Spearman(T50, syn slope)={s['spearman_synthetic_slope_vs_T50']}")


if __name__ == "__main__":
    main()