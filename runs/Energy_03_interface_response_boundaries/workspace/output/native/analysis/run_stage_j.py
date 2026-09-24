"""Stage J: robustness checks for the curvature claim (registered finding 1).

1. Curvature is not an artifact of the zero/low-onset 120 K setting: refit with
   the 120 K point excluded and with T>=140 only.
2. Curvature does not depend on the choice of concave functional form: compare
   quadratic-in-T, log-quadratic, and a saturating hyperbola, and compare
   against a model-free discrete second-difference test on the raw data.
3. Curvature survives whole-curve and contiguous-setting holdout: fit the
   curvature coefficient on T<=280 and test the sign on held-out T=300,320.
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


def quadr(T, y):
    X = np.column_stack([np.ones_like(T), T, T ** 2])
    f = ts.ols(X, np.log(y))
    p1 = f["p"][2] / 2 if f["beta"][2] < 0 else 1.0 - f["p"][2] / 2
    return {"c": float(f["beta"][2]), "se": float(f["se"][2]),
            "t": float(f["t"][2]), "p_one_sided_c_less_0": float(p1),
            "r2": float(f["r2"]), "n": int(f["n"])}


def main():
    mp, cp = load()
    out = {}

    # ---- 1. onset exclusion ------------------------------------------------
    ex = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            r = {"curve": cid}
            for tag, sel in (("all_11_settings", T >= 120),
                             ("T_ge_140", T >= 140),
                             ("drop_120_and_140", T >= 160)):
                m = sel & (y > 0)
                r[tag] = quadr(T[m], y[m]) if m.sum() >= 4 else None
            rows.append(r)
        ex[lab] = rows
    out["onset_exclusion"] = ex

    # ---- 2. alternative curvature forms -----------------------------------
    alt = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            m = (T >= 140) & (y > 0)
            tt, yy = T[m], y[m]
            q = quadr(tt, yy)
            # log-quadratic in log T
            f_lt = ts.ols(np.column_stack([np.ones_like(tt), np.log(tt),
                                           np.log(tt) ** 2]), np.log(yy))
            # hyperbolic saturation: y = A*T/(K+T)  -> 1/y linear in 1/T
            inv = ts.ols(np.column_stack([np.ones_like(tt), 1.0 / tt]), 1.0 / yy)
            # model-free discrete second difference of log y at interior points
            ly = np.log(yy)
            d2 = np.diff(ly, 2)
            rows.append({
                "curve": cid,
                "quadratic_c": q["c"], "quadratic_p_one_sided": q["p_one_sided_c_less_0"],
                "logT_quadratic_coef": float(f_lt["beta"][2]),
                "logT_quadratic_r2": float(f_lt["r2"]),
                "reciprocal_fit_r2": float(inv["r2"]),
                "n_negative_second_differences": int((d2 < 0).sum()),
                "n_second_differences": int(len(d2)),
                "all_second_differences_negative": bool((d2 < 0).all()),
                "max_second_difference": float(d2.max()),
            })
        out_key = alt.setdefault(lab, {"curves": rows})
    out["alternative_forms"] = alt

    # ---- 3. holdout sign test on the curvature coefficient -----------------
    hold = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            tr = (T <= 280) & (y > 0)
            te = (T > 280) & (y > 0)
            if tr.sum() < 4 or te.sum() < 1:
                continue
            fit_tr = quadr(T[tr], y[tr])
            # predicted log-response at held-out settings from the training fit
            Xtr = np.column_stack([np.ones(tr.sum()), T[tr], T[tr] ** 2])
            f = ts.ols(Xtr, np.log(y[tr]))
            pred = f["beta"][0] + f["beta"][1] * T[te] + f["beta"][2] * T[te] ** 2
            obs = np.log(y[te])
            err = obs - pred
            # baseline: pure exponential (no curvature) trained on the same points
            f2 = ts.ols(np.column_stack([np.ones(tr.sum()), T[tr]]), np.log(y[tr]))
            pred2 = f2["beta"][0] + f2["beta"][1] * T[te]
            err2 = obs - pred2
            rows.append({
                "curve": cid,
                "c_train": fit_tr["c"],
                "c_train_negative": bool(fit_tr["c"] < 0),
                "rmse_holdout_quadratic": float(np.sqrt((err @ err) / len(err))),
                "rmse_holdout_exponential": float(np.sqrt((err2 @ err2) / len(err2))),
                "quadratic_better_on_holdout": bool((err @ err) < (err2 @ err2)),
                "mean_holdout_log_residual_quadratic": float(err.mean()),
            })
        hold[lab] = {"curves": rows,
                     "n_c_negative_in_training": int(sum(r["c_train_negative"] for r in rows)),
                     "n_quadratic_better_on_holdout": int(sum(r["quadratic_better_on_holdout"] for r in rows)),
                     "n_curves": len(rows)}
    out["holdout_curvature"] = hold

    dump("J_curvature_robustness.json", out)

    print("=== onset exclusion (quadratic curvature c, one-sided p for c<0) ===")
    for lab, rows in ex.items():
        for r in rows:
            line = f"  {lab}:{r['curve']:>18s}"
            for tag in ("all_11_settings", "T_ge_140", "drop_120_and_140"):
                v = r[tag]
                line += (f" | {tag}: c={v['c']:+.2e} p={v['p_one_sided_c_less_0']:.4f}"
                         if v else f" | {tag}: n/a")
            print(line)
    print("\n=== alternative curvature forms ===")
    for lab, d in alt.items():
        for r in d["curves"]:
            print(f"  {lab}:{r['curve']:>18s} logT-quad coef={r['logT_quadratic_coef']:+.4f} "
                  f"(R2={r['logT_quadratic_r2']:.5f}) reciprocal-R2={r['reciprocal_fit_r2']:.5f} "
                  f"neg 2nd-diffs={r['n_negative_second_differences']}/{r['n_second_differences']} "
                  f"all_neg={r['all_second_differences_negative']}")
    print("\n=== holdout sign test ===")
    for lab, d in hold.items():
        print(f"  {lab}: c<0 in training for {d['n_c_negative_in_training']}/{d['n_curves']}; "
              f"quadratic beats exponential on holdout for "
              f"{d['n_quadratic_better_on_holdout']}/{d['n_curves']}")


if __name__ == "__main__":
    main()