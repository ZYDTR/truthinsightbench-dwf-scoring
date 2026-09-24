"""Stage G: robustness suite.

G1. Multi-cut-point contiguous-setting holdout (240/260/280/300 K cutoffs).
G2. Whole-curve leave-one-out with shared-shape refit (complete-unit holdout).
G3. Separation/SNR metric locating the low-temperature resolvability boundary.
G4. Outlier / zero-response treatment sensitivity.
G5. Alternative sigmoid parameterisation (Gompertz, Hill) as an alternative model.
"""

from __future__ import annotations

import json
import sys
from math import comb
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


# --------------------------------------------------------------------------
def g1_multicut(mp, cp):
    out = {"protocol": "fit on T<=cut, predict the remaining settings; "
                       "logistic vs unsaturated exponential vs persistence",
           "cuts": {}}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        res = {}
        for cut in (240.0, 260.0, 280.0, 300.0):
            train, test = T <= cut, T > cut
            if test.sum() == 0:
                continue
            rows = []
            for cid in piv.index:
                y = piv.loc[cid].values.astype(float)
                if np.sum((y > 0) & train) < 3:
                    continue
                fl = fit_logistic(T[train], y[train])
                fe = ts.ols(np.column_stack([np.ones((train & (y > 0)).sum()),
                                             T[train & (y > 0)]]),
                            np.log(y[train & (y > 0)]))
                m_te = y[test] > 0
                if m_te.sum() == 0:
                    continue
                def rl(pred):
                    p = np.clip(np.asarray(pred, float)[m_te], 1e-12, None)
                    r = np.log(y[test][m_te]) - np.log(p)
                    return float(np.sqrt(r @ r / len(r)))
                pl = logistic(T[test], fl["ymax"], fl["T50"], fl["w"])
                pe = np.exp(fe["beta"][0] + fe["beta"][1] * T[test])
                pp = np.full(test.sum(), y[train][-1])
                rows.append({"curve": cid, "rmse_logistic": rl(pl),
                             "rmse_exponential": rl(pe), "rmse_persistence": rl(pp)})
            if not rows:
                continue
            d_exp = [r["rmse_exponential"] - r["rmse_logistic"] for r in rows]
            d_per = [r["rmse_persistence"] - r["rmse_logistic"] for r in rows]
            n = len(rows)
            k_exp = int(sum(x > 0 for x in d_exp))
            k_per = int(sum(x > 0 for x in d_per))
            res[str(int(cut))] = {
                "n_curves": n, "n_held_out_settings": int(test.sum()),
                "n_logistic_better_than_exponential": k_exp,
                "n_logistic_better_than_persistence": k_per,
                "median_delta_rmse_exponential": float(np.median(d_exp)),
                "median_delta_rmse_persistence": float(np.median(d_per)),
                "sign_test_p_exponential": sum(comb(n, i) for i in range(k_exp, n + 1)) / 2 ** n,
                "sign_test_p_persistence": sum(comb(n, i) for i in range(k_per, n + 1)) / 2 ** n,
                "detail": rows,
            }
        out["cuts"][lab] = res
    dump("G1_multicut_holdout.json", out)
    return out


# --------------------------------------------------------------------------
def g2_whole_curve_loo(mp, cp):
    """Complete-unit holdout: refit a shared-shape model without the held-out curve."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        cids = list(piv.index)
        rows = []
        for hold in cids:
            train_c = [c for c in cids if c != hold]
            # shared (ymax, w) from training curves; T50 refit on the held-out
            # curve's own low-T points only (T<=260) to keep it a genuine
            # forward prediction of the two highest settings.
            src = [fit_logistic(T, piv.loc[c].values.astype(float)) for c in train_c]
            ymax = float(np.mean([s["ymax"] for s in src]))
            w = float(np.mean([s["w"] for s in src]))
            y = piv.loc[hold].values.astype(float)
            tr = T <= 260.0
            fh = fit_logistic(T[tr], y[tr], ymax_fixed=ymax)
            te = T > 260.0
            m = y[te] > 0
            pred = logistic(T[te][m], ymax, fh["T50"], w)
            r = np.log(y[te][m]) - np.log(np.clip(pred, 1e-12, None))
            # baseline: exponential extrapolation from the same training points
            mm = tr & (y > 0)
            fe = ts.ols(np.column_stack([np.ones(mm.sum()), T[mm]]), np.log(y[mm]))
            pe = np.exp(fe["beta"][0] + fe["beta"][1] * T[te][m])
            re = np.log(y[te][m]) - np.log(np.clip(pe, 1e-12, None))
            rows.append({
                "held_out_curve": hold,
                "n_train_curves": len(train_c),
                "shared_ymax_from_train": ymax, "shared_w_from_train": w,
                "own_T50_from_T<=260": fh["T50"],
                "rmse_log_transferred_shape": float(np.sqrt((r @ r) / len(r))),
                "rmse_log_exponential": float(np.sqrt((re @ re) / len(re))),
                "logistic_better": bool((r @ r) < (re @ re)),
            })
        n = len(rows)
        k = int(sum(r["logistic_better"] for r in rows))
        out[lab] = {
            "protocol": "shared ymax+w from all OTHER curves; held-out curve's own "
                        "T50 from T<=260 only; predict T=280,300,320",
            "n_curves": n, "n_logistic_better": k,
            "sign_test_p": sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n,
            "mean_rmse_log_transferred": float(np.mean([r["rmse_log_transferred_shape"] for r in rows])),
            "mean_rmse_log_exponential": float(np.mean([r["rmse_log_exponential"] for r in rows])),
            "per_curve": rows,
        }
    dump("G2_whole_curve_holdout.json", out)
    return out


# --------------------------------------------------------------------------
def g3_separation(mp, cp):
    """Locate the setting below which condition ordering stops being resolvable."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for t in T:
            v = piv[t].values.astype(float)
            pos = v[v > 0]
            rows.append({
                "setting": float(t),
                "response_range": float(v.max() - v.min()),
                "response_max": float(v.max()),
                "response_mean": float(v.mean()),
                "n_zero": int((v == 0).sum()),
                "cv_positive": float(np.std(pos, ddof=1) / np.mean(pos)) if pos.size > 1 else None,
                "range_over_max": float((v.max() - v.min()) / v.max()) if v.max() > 0 else None,
            })
        # separation growth relative to the lowest setting
        base = rows[0]["response_range"] if rows[0]["response_range"] > 0 else 1e-12
        for r in rows:
            r["range_relative_to_first_setting"] = float(r["response_range"] / base)
        # rank stability between adjacent settings
        adj = []
        for i in range(len(T) - 1):
            r1, r2 = piv[T[i]].rank().values, piv[T[i + 1]].rank().values
            rho, p = ts.spearman(r1, r2)
            adj.append({"from": float(T[i]), "to": float(T[i + 1]),
                        "spearman": rho, "p": p})
        # cumulative pairwise reversal census as a function of the lower bound
        cens = []
        cids = list(piv.index)
        for lo in T:
            sel = [t for t in T if t >= lo]
            n_rev = 0
            tot = 0
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    d = piv.loc[cids[i], sel].values - piv.loc[cids[j], sel].values
                    s = np.sign(d)
                    s = s[s != 0]
                    tot += 1
                    if s.size > 1 and (np.diff(s) != 0).any():
                        n_rev += 1
            cens.append({"lower_bound_setting": float(lo), "n_settings": len(sel),
                         "pairs_with_reversal": n_rev, "n_pairs": tot})
        out[lab] = {"per_setting": rows, "adjacent_rank_stability": adj,
                    "reversal_census_by_lower_bound": cens}
    dump("G3_separation_boundary.json", out)
    return out


# --------------------------------------------------------------------------
def g4_outliers(mp, cp):
    """Zero/outlier treatment sensitivity."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            full = fit_logistic(T, y)
            trimmed = fit_logistic(T[1:], y[1:])  # drop the 120 K onset point
            outl = None
            m = y > 0
            if m.sum() >= 4:
                pred = logistic(T[m], full["ymax"], full["T50"], full["w"])
                r = np.log(y[m]) - np.log(np.clip(pred, 1e-12, None))
                sd = r.std(ddof=1)
                keep = np.abs(r - r.mean()) <= 2.0 * sd
                if keep.sum() >= 4:
                    sel = np.nonzero(m)[0]
                    idx = sel[keep]
                    outl = fit_logistic(T[idx], y[idx])
            rows.append({
                "curve": cid,
                "T50_full": full["T50"], "w_full": full["w"], "ymax_full": full["ymax"],
                "T50_drop120K": trimmed["T50"], "w_drop120K": trimmed["w"],
                "delta_T50_drop120K": trimmed["T50"] - full["T50"],
                "delta_w_drop120K": trimmed["w"] - full["w"],
                "n_points_dropped_2sd": int((y > 0).sum() - 0)
                if outl is None else int(len(T) - int(0)),
                "T50_after_2sd_trim": None if outl is None else outl["T50"],
                "delta_T50_after_2sd_trim": None if outl is None else outl["T50"] - full["T50"],
                "r2_log_full": full["r2_log"],
            })
        dT = [r["delta_T50_drop120K"] for r in rows]
        dW = [r["delta_w_drop120K"] for r in rows]
        out[lab] = {
            "protocol": "refit after dropping the 120 K onset setting and after a "
                        "2-sigma log-residual trim",
            "n_curves": len(rows),
            "mean_abs_delta_T50_drop120K": float(np.mean(np.abs(dT))),
            "max_abs_delta_T50_drop120K": float(np.max(np.abs(dT))),
            "mean_abs_delta_w_drop120K": float(np.mean(np.abs(dW))),
            "max_abs_delta_w_drop120K": float(np.max(np.abs(dW))),
            "per_curve": rows,
        }
    dump("G4_outlier_sensitivity.json", out)
    return out


# --------------------------------------------------------------------------
def g5_alt_sigmoid(mp, cp):
    """Alternative saturation models: Gompertz and Hill, against logistic."""
    def gompertz(T, ymax, T50, w):
        return ymax * np.exp(-np.exp(-(T - T50) / w))

    def hill(T, ymax, T50, w):
        # Hill in T with the same (T50, w) semantics: exponent fixed by w
        n = max(20.0 / w, 0.5)
        return ymax * (T ** n) / (T50 ** n + T ** n)

    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = y > 0
            log_y = np.log(y[m])
            res = {}
            for name, fn in (("logistic", logistic), ("gompertz", gompertz), ("hill", hill)):
                # grid + refinement over T50/w with ymax from the data scale
                best = None
                ymax_grid = np.linspace(max(y.max(), 1.0),
                                        max(y.max() * 3.0, 3.0), 40)
                t50_grid = np.linspace(T.min() - 60, T.max() + 120, 60)
                w_grid = np.linspace(5.0, 90.0, 40)
                for ym in ymax_grid:
                    for t50 in t50_grid:
                        for w in w_grid:
                            pred = fn(T[m], ym, t50, w)
                            if np.any(pred <= 0):
                                continue
                            r = log_y - np.log(pred)
                            s = float(r @ r)
                            if best is None or s < best[0]:
                                best = (s, ym, t50, w)
                rss, ym, t50, w = best
                n = int(m.sum())
                aic, bic = ts.aic_bic(rss, n, 4)
                tss = float(((log_y - log_y.mean()) ** 2).sum())
                res[name] = {"rss_log": rss, "ymax": float(ym), "T50": float(t50),
                             "w": float(w), "aic": aic, "bic": bic,
                             "r2_log": float(1 - rss / tss)}
            rows.append({"curve": cid, **{f"{k}_r2_log": v["r2_log"] for k, v in res.items()},
                         **{f"{k}_bic": v["bic"] for k, v in res.items()},
                         "best_model": min(res, key=lambda k: res[k]["bic"]),
                         "full": res})
        from collections import Counter
        cnt = Counter(r["best_model"] for r in rows)
        out[lab] = {
            "protocol": "coarse grid search over (ymax,T50,w) in log space; "
                        "3 models, 4 parameters each including sigma",
            "model_win_counts_by_BIC": dict(cnt),
            "n_curves": len(rows),
            "mean_r2_log": {mn: float(np.mean([r[f"{mn}_r2_log"] for r in rows]))
                            for mn in ("logistic", "gompertz", "hill")},
            "per_curve": rows,
        }
    dump("G5_alternative_sigmoids.json", out)
    return out


def main():
    mp, cp = load()
    g1_multicut(mp, cp)
    g2_whole_curve_loo(mp, cp)
    g3_separation(mp, cp)
    g4_outliers(mp, cp)
    g5_alt_sigmoid(mp, cp)
    for f in ("G1_multicut_holdout", "G2_whole_curve_holdout",
              "G3_separation_boundary", "G4_outlier_sensitivity",
              "G5_alternative_sigmoids"):
        d = json.loads((RES / f"{f}.json").read_text())
        print(f"\n===== {f} =====")
        if f == "G1_multicut_holdout":
            for lab, cuts in d["cuts"].items():
                for cut, v in cuts.items():
                    print(f"  {lab} cut<={cut}: logistic>exp {v['n_logistic_better_than_exponential']}"
                          f"/{v['n_curves']} (p={v['sign_test_p_exponential']:.4f}) "
                          f"| >persist {v['n_logistic_better_than_persistence']}/{v['n_curves']} "
                          f"(p={v['sign_test_p_persistence']:.4f})")
        elif f == "G2_whole_curve_holdout":
            for lab, v in d.items():
                print(f"  {lab}: logistic>exp {v['n_logistic_better']}/{v['n_curves']} "
                      f"p={v['sign_test_p']:.4f} meanRMSE transferred="
                      f"{v['mean_rmse_log_transferred']:.3f} vs exp={v['mean_rmse_log_exponential']:.3f}")
        elif f == "G3_separation_boundary":
            for lab, v in d.items():
                print(f"  {lab} adjacent rank stability:")
                for a in v["adjacent_rank_stability"]:
                    print(f"    {a['from']:.0f}->{a['to']:.0f} rho={a['spearman']:+.3f} p={a['p']}")
                print(f"  {lab} reversal census by lower bound:")
                for cn in v["reversal_census_by_lower_bound"]:
                    print(f"    T>={cn['lower_bound_setting']:.0f}: {cn['pairs_with_reversal']}"
                          f"/{cn['n_pairs']}")
        elif f == "G4_outlier_sensitivity":
            for lab, v in d.items():
                print(f"  {lab}: mean|dT50| drop120K={v['mean_abs_delta_T50_drop120K']:.2f} K "
                      f"max={v['max_abs_delta_T50_drop120K']:.2f} K | "
                      f"mean|dw|={v['mean_abs_delta_w_drop120K']:.2f} K")
        else:
            for lab, v in d.items():
                print(f"  {lab}: BIC wins {v['model_win_counts_by_BIC']} "
                      f"meanR2={v['mean_r2_log']}")


if __name__ == "__main__":
    main()