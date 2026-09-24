"""Stage F: falsification tests.

F1. Contiguous-setting holdout (fit on T<=280, predict T=300 and T=320).
    This is the task-mandated forward holdout; it directly tests whether the
    saturating (logistic) description beats an unsaturated exponential one.
F2. Whole-curve holdout on the check panel using main-panel-derived parameters.
F3. Curve-level crossing census with a paired sign test.
F4. Onset-descriptor vs apparent-exponent relationship (model-free).
F5. Apparent activation energy / exponent distribution and its origin.
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
    mp = m.pivot(index="condition_id", columns="temperature_like_setting",
                 values="rate_like_response")
    cp = c.pivot(index="condition_id", columns="temperature_like_setting",
                 values="rate_like_response")
    return m, c, mp, cp


# --------------------------------------------------------------------------
def f1_forward_holdout(mp, cp):
    """Fit on settings <= 280 K; predict 300 K and 320 K."""
    out = {"protocol": "fit on T<=280 (9 settings), predict T=300 and T=320",
           "main": {}, "check": {}, "summary": {}}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        train = T <= 280.0
        test = T > 280.0
        rows = {}
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            if np.sum(y > 0) < 3:
                continue
            # --- logistic forward prediction
            fl = fit_logistic(T[train], y[train])
            pred_log = logistic(T[test], fl["ymax"], fl["T50"], fl["w"])
            # --- unsaturated exponential (log-linear) forward prediction
            m_tr = train & (y > 0)
            fe = ts.ols(np.column_stack([np.ones(m_tr.sum()), T[m_tr]]),
                        np.log(y[m_tr]))
            pred_exp = np.exp(fe["beta"][0] + fe["beta"][1] * T[test])
            # --- last-value persistence baseline (predicts no further change)
            last = y[train][-1]
            pred_pers = np.full(test.sum(), last)
            m_te = (y[test] > 0)  # both held-out settings are positive for all curves
            def rmse_log(pred):
                p = np.clip(np.asarray(pred, float)[m_te], 1e-12, None)
                r = np.log(y[test][m_te]) - np.log(p)
                return float(np.sqrt(r @ r / len(r)))
            rows[cid] = {
                "observed_300": float(y[T == 300][0]),
                "observed_320": float(y[T == 320][0]),
                "pred_logistic_300": float(pred_log[0]),
                "pred_logistic_320": float(pred_log[1]),
                "pred_exponential_300": float(pred_exp[0]),
                "pred_exponential_320": float(pred_exp[1]),
                "pred_persistence_320": float(last),
                "rmse_log_logistic": rmse_log(pred_log),
                "rmse_log_exponential": rmse_log(pred_exp),
                "rmse_log_persistence": rmse_log(pred_pers),
                "logistic_beats_exponential": bool(rmse_log(pred_log) < rmse_log(pred_exp)),
                "logistic_beats_persistence": bool(rmse_log(pred_log) < rmse_log(pred_pers)),
                "fit_ymax": fl["ymax"], "fit_T50": fl["T50"], "fit_w": fl["w"],
            }
        out[lab] = rows
        n_le = sum(v["logistic_beats_exponential"] for v in rows.values())
        n_lp = sum(v["logistic_beats_persistence"] for v in rows.values())
        # paired comparison across held-out curves: logistic vs exponential error
        d = [v["rmse_log_exponential"] - v["rmse_log_logistic"] for v in rows.values()]
        out["summary"][lab] = {
            "n_curves": len(rows),
            "n_logistic_beats_exponential": n_le,
            "n_logistic_beats_persistence": n_lp,
            "mean_rmse_log_logistic": float(np.mean([v["rmse_log_logistic"] for v in rows.values()])),
            "mean_rmse_log_exponential": float(np.mean([v["rmse_log_exponential"] for v in rows.values()])),
            "mean_rmse_log_persistence": float(np.mean([v["rmse_log_persistence"] for v in rows.values()])),
            "paired_delta_rmse": d,
            "median_delta": float(np.median(d)),
        }
    dump("F1_forward_holdout.json", out)
    return out


# --------------------------------------------------------------------------
def f2_cross_panel(mp, cp):
    """Which check curves are genuinely new, and do main-derived parameters transfer?"""
    out = {}
    # exact duplicate detection
    dups = []
    for ci in cp.index:
        for mi in mp.index:
            if np.array_equal(cp.loc[ci].values.astype(float),
                              mp.loc[mi].values.astype(float)):
                dups.append({"check_curve": ci, "identical_main_curve": mi,
                             "n_settings_matched": int(len(cp.columns))})
    out["exact_duplicates"] = dups
    dup_check = {d["check_curve"] for d in dups}
    out["n_check_curves"] = int(len(cp.index))
    out["n_exactly_duplicated"] = len(dup_check)
    out["n_genuinely_new_check_curves"] = int(len(cp.index) - len(dup_check))
    out["genuinely_new_check_curves"] = [c for c in cp.index if c not in dup_check]

    # Parameter transfer: main -> check for the genuinely new curves.
    T = mp.columns.values.astype(float)
    transfers = []
    for ci in cp.index:
        if ci in dup_check:
            continue
        y = cp.loc[ci].values.astype(float)
        fc = fit_logistic(T, y)
        for mi in mp.index:
            ym = mp.loc[mi].values.astype(float)
            fm = fit_logistic(T, ym)
            m = y > 0
            # transfer ONLY the shape (w) and amplitude (ymax) from main; keep
            # the check curve's own T50 -> tests whether shape is conserved
            pred = logistic(T[m], fm["ymax"], fc["T50"], fm["w"])
            r = np.log(y[m]) - np.log(np.clip(pred, 1e-12, None))
            transfers.append({
                "check_curve": ci, "main_source": mi,
                "rmse_log_transfer_shape": float(np.sqrt(r @ r / len(r))),
                "rmse_log_own_fit": float(np.sqrt(
                    ((np.log(y[T > 0]) - np.log(logistic(T[T > 0], fc["ymax"],
                                                         fc["T50"], fc["w"]))) ** 2).mean())),
                "main_w": fm["w"], "main_ymax": fm["ymax"],
                "check_w": fc["w"], "check_ymax": fc["ymax"], "check_T50": fc["T50"],
                "delta_w": fc["w"] - fm["w"],
            })
    out["shape_transfer_main_to_new_check_curves"] = transfers
    if transfers:
        dw = [t["delta_w"] for t in transfers]
        out["shape_transfer_summary"] = {
            "n_pairs": len(transfers),
            "mean_abs_delta_w_K": float(np.mean(np.abs(dw))),
            "mean_delta_w_K": float(np.mean(dw)),
            "max_abs_delta_w_K": float(np.max(np.abs(dw))),
        }
    dump("F2_cross_panel_integrity.json", out)
    return out


# --------------------------------------------------------------------------
def f3_crossing_census(mp, cp):
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        cids = list(piv.index)
        T = piv.columns.values.astype(float)
        pairs = []
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                d = piv.loc[cids[i]].values.astype(float) - piv.loc[cids[j]].values.astype(float)
                s = np.sign(d)
                s = s[s != 0]
                flips = int((np.diff(s) != 0).sum()) if s.size > 1 else 0
                pairs.append({"a": cids[i], "b": cids[j], "n_reversals": flips,
                              "diff_at_first_setting": float(d[0]),
                              "diff_at_last_setting": float(d[-1]),
                              "direction_flipped_net": bool(np.sign(d[0]) != np.sign(d[-1]))})
        n_net = sum(p["direction_flipped_net"] for p in pairs)
        # exact paired sign test on "net direction flipped" over pairs
        # under H0 each pair flips with prob 0.5 -> binomial; enumerate exactly
        from math import comb as _comb
        n = len(pairs)
        # one-sided p for >= n_net successes at p=0.5
        p_one = sum(_comb(n, k) for k in range(n_net, n + 1)) / (2 ** n)
        # rank-order stability between first and last setting
        r1 = piv[T[0]].rank().values
        r2 = piv[T[-1]].rank().values
        out[lab] = {
            "n_curves": len(cids), "n_pairs": n,
            "n_pairs_with_any_reversal": int(sum(p["n_reversals"] > 0 for p in pairs)),
            "n_pairs_with_net_direction_flip": int(n_net),
            "sign_test_p_one_sided": float(p_one),
            "spearman_rank_first_vs_last": ts.spearman(r1, r2),
            "kendall_tau_b_first_vs_last": float(ts.kendall_tau_b(r1, r2)),
            "pairs": pairs,
        }
    dump("F3_crossing_census.json", out)
    return out


# --------------------------------------------------------------------------
def f4_onset_vs_exponent(mp, cp):
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = T >= 140
            fe = ts.ols(np.column_stack([np.ones(m.sum()), T[m]]), np.log(y[m]))
            fa = ts.ols(np.column_stack([np.ones(m.sum()), 1.0 / T[m]]), np.log(y[m]))
            # model-free T10 by linear interpolation in log-response
            idx = np.nonzero(y >= 10)[0]
            if idx.size == 0:
                t10 = None
            else:
                i = int(idx[0])
                if i == 0 or y[i - 1] <= 0:
                    t10 = float(T[i])
                else:
                    frac = ((np.log(10) - np.log(y[i - 1])) /
                            (np.log(y[i]) - np.log(y[i - 1])))
                    t10 = float(T[i - 1] + frac * (T[i] - T[i - 1]))
            fw = None
            ok = (y > 0) & (y < 100)
            if ok.sum() >= 5:
                st = ts.ols(np.column_stack([np.ones(ok.sum()), T[ok]]),
                            np.log(y[ok] / (100 - y[ok])))
                fw = {"slope_per_K": float(st["beta"][1]),
                      "w_K": float(1.0 / st["beta"][1]),
                      "T50_K": float(-st["beta"][0] / st["beta"][1]),
                      "r2": float(st["r2"]), "n": int(ok.sum())}
            rows.append({
                "curve": cid, "T10_K": t10,
                "exp_slope_per_K": float(fe["beta"][1]),
                "exp_r2": float(fe["r2"]),
                "arrhenius_slope": float(fa["beta"][1]),
                "apparent_Ea_kJ_per_mol_if_T_is_K": float(-fa["beta"][1] * 8.314 / 1000.0),
                "arrhenius_r2": float(fa["r2"]),
                "logit_ceiling100": fw,
                "y_at_last": float(y[-1]),
            })
        t10s = [r["T10_K"] for r in rows if r["T10_K"] is not None]
        sls = [r["exp_slope_per_K"] for r in rows if r["T10_K"] is not None]
        eas = [r["apparent_Ea_kJ_per_mol_if_T_is_K"] for r in rows]
        out[lab] = {
            "curves": rows,
            "spearman_T10_vs_exp_slope": ts.spearman(t10s, sls),
            "kendall_T10_vs_exp_slope": float(ts.kendall_tau_b(t10s, sls)),
            "pearson_T10_vs_exp_slope": [float(np.corrcoef(t10s, sls)[0, 1]),
                                         ts.fisher_z_pvalue(float(np.corrcoef(t10s, sls)[0, 1]), len(t10s))],
            "apparent_Ea_range_kJ": [float(min(eas)), float(max(eas))],
            "apparent_Ea_mean_kJ": float(np.mean(eas)),
            "apparent_Ea_sd_kJ": float(np.std(eas, ddof=1)),
            "exp_slope_range_per_K": [float(min(r["exp_slope_per_K"] for r in rows)),
                                      float(max(r["exp_slope_per_K"] for r in rows))],
            "max_over_min_slope": float(max(r["exp_slope_per_K"] for r in rows) /
                                        min(r["exp_slope_per_K"] for r in rows)),
        }
    dump("F4_onset_vs_exponent.json", out)
    return out


# --------------------------------------------------------------------------
def f5_saturation_ceiling(mp, cp):
    """Ymax identifiability and the design's saturation boundary."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            free = fit_logistic(T, y)
            fixed = fit_logistic(T, y, ymax_fixed=100.0)
            rows.append({
                "curve": cid,
                "ymax_free": free["ymax"], "ymax_free_se": free["param_se"].get("ymax"),
                "ymax_identifiable": bool(free["param_se"].get("ymax", np.inf) < 0.5 * free["ymax"]),
                "T50_free": free["T50"], "w_free": free["w"],
                "r2_log_free": free["r2_log"],
                "T50_fixed100": fixed["T50"], "w_fixed100": fixed["w"],
                "r2_log_fixed100": fixed["r2_log"],
                "delta_r2_fixed_minus_free": fixed["r2_log"] - free["r2_log"],
                "T50_exceeds_design_max": bool(fixed["T50"] > T.max()),
                "y_at_320": float(y[-1]),
                "fraction_of_100_at_320": float(y[-1] / 100.0),
            })
        out[lab] = {
            "curves": rows,
            "n_ymax_identifiable": int(sum(r["ymax_identifiable"] for r in rows)),
            "n_T50_above_design_max": int(sum(r["T50_exceeds_design_max"] for r in rows)),
            "max_design_setting": float(T.max()),
            "mean_delta_r2_fixed_minus_free": float(np.mean(
                [r["delta_r2_fixed_minus_free"] for r in rows])),
            "max_delta_r2_fixed_minus_free": float(np.max(
                [r["delta_r2_fixed_minus_free"] for r in rows])),
        }
    dump("F5_saturation_boundary.json", out)
    return out


def main():
    _, _, mp, cp = load()
    f1 = f1_forward_holdout(mp, cp)
    f2 = f2_cross_panel(mp, cp)
    f3 = f3_crossing_census(mp, cp)
    f4 = f4_onset_vs_exponent(mp, cp)
    f5 = f5_saturation_ceiling(mp, cp)

    print("\n=== F1 forward holdout (fit T<=280, predict 300/320) ===")
    print(json.dumps(f1["summary"], indent=2))
    print("\n=== F2 cross-panel integrity ===")
    print("exact duplicates:", json.dumps(f2["exact_duplicates"], indent=2))
    print("genuinely new check curves:", f2["genuinely_new_check_curves"])
    print("shape transfer:", json.dumps(f2.get("shape_transfer_summary"), indent=2))
    print("\n=== F3 crossing census ===")
    for k, v in f3.items():
        print(f"{k}: curves={v['n_curves']} pairs={v['n_pairs']} "
              f"any_reversal={v['n_pairs_with_any_reversal']} "
              f"net_flip={v['n_pairs_with_net_direction_flip']} "
              f"p={v['sign_test_p_one_sided']:.6f} "
              f"spearman_first_last={v['spearman_rank_first_vs_last']}")
    print("\n=== F4 onset vs exponent ===")
    for k, v in f4.items():
        print(f"{k}: spearman(T10, exp_slope)={v['spearman_T10_vs_exp_slope']}")
        print(f"     apparent Ea range={v['apparent_Ea_range_kJ']} mean={v['apparent_Ea_mean_kJ']:.2f} "
              f"sd={v['apparent_Ea_sd_kJ']:.2f} kJ/mol; slope max/min={v['max_over_min_slope']:.2f}")
    print("\n=== F5 saturation boundary ===")
    for k, v in f5.items():
        print(f"{k}: ymax identifiable {v['n_ymax_identifiable']}/{len(v['curves'])}; "
              f"T50>max_setting {v['n_T50_above_design_max']}/{len(v['curves'])}; "
              f"mean delta R2 (ymax=100 vs free)={v['mean_delta_r2_fixed_minus_free']:.5f}")


if __name__ == "__main__":
    main()