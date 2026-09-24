"""Stage I: consolidation -- block effects, single-ordering-parameter test,
and the numeric tables that back the report.
"""

from __future__ import annotations

import csv
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
TAB = Path(__file__).resolve().parents[1] / "results"


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


def block_test(mp):
    """Does the block label carry any systematic parameter signal?"""
    T = mp.columns.values.astype(float)
    fits = {c: fit_logistic(T, mp.loc[c].values.astype(float)) for c in mp.index}
    b1 = [c for c in mp.index if c.startswith("condition_block1")]
    b2 = [c for c in mp.index if c.startswith("condition_block2")]
    out = {"block1_conditions": b1, "block2_conditions": b2}
    for pname in ("T50", "w", "ymax"):
        v1 = [fits[c][pname] for c in b1]
        v2 = [fits[c][pname] for c in b2]
        obs = float(np.mean(v2) - np.mean(v1))
        vals = np.array(v1 + v2, float)
        # exact complete enumeration over which 4 of the 8 are "block2"
        stats = []
        idx = range(len(vals))
        for combi in __import__("itertools").combinations(idx, len(b2)):
            stats.append(np.mean(vals[list(combi)]) - np.mean(vals[[i for i in idx if i not in combi]]))
        stats = np.array(stats)
        p_two = float((np.abs(stats) >= abs(obs) - 1e-12).mean())
        out[pname] = {
            "block1_values": v1, "block2_values": v2,
            "mean_block1": float(np.mean(v1)), "mean_block2": float(np.mean(v2)),
            "observed_difference": obs,
            "exact_two_sided_p": p_two,
            "n_enumerated_assignments": int(len(stats)),
            "blocks_fully_separated": bool(min(v2) > max(v1) or min(v1) > max(v2)),
        }
    # overall: is the block label recovered by any parameter?
    return out


def ordering_by_single_parameter(mp, cp):
    """Is T50 alone the ordering parameter? Predict held-out ordering."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        T = piv.columns.values.astype(float)
        fits = {c: fit_logistic(T, piv.loc[c].values.astype(float)) for c in piv.index}
        cids = list(piv.index)
        # predicted ordering by fitted T50; observed ordering at each setting
        pred_order = sorted(cids, key=lambda c: fits[c]["T50"])
        rows = []
        for t in T:
            obs_order = list(piv[t].sort_values().index)
            # rank agreement
            ro = {c: i for i, c in enumerate(obs_order)}
            rp = {c: i for i, c in enumerate(pred_order)}
            rho, p = ts.spearman([ro[c] for c in cids], [rp[c] for c in cids])
            kend = ts.kendall_tau_b([ro[c] for c in cids], [rp[c] for c in cids])
            rows.append({"setting": float(t), "spearman_order_vs_T50": rho,
                         "p": p, "kendall": float(kend)})
        out[lab] = {"T50_by_curve": {c: fits[c]["T50"] for c in cids},
                    "predicted_order_by_T50": pred_order,
                    "agreement_vs_setting": rows}
    return out


def exact_pair_reversal_census(mp, cp):
    """Exact count of sign reversals, per pair, for the report table."""
    out = {}
    for lab, piv in (("main", mp), ("check", cp)):
        cids = list(piv.index)
        T = list(piv.columns.values.astype(float))
        rows = []
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                d = piv.loc[cids[i]].values.astype(float) - piv.loc[cids[j]].values.astype(float)
                s = np.sign(d)
                s = s[s != 0]
                flips = int((np.diff(s) != 0).sum()) if s.size > 1 else 0
                rows.append({"a": cids[i], "b": cids[j], "n_reversals": flips,
                             "d_at_120": float(d[0]), "d_at_320": float(d[-1])})
        out[lab] = {"pairs": rows,
                    "n_pairs": len(rows),
                    "n_with_reversal": int(sum(r["n_reversals"] > 0 for r in rows)),
                    "n_flipping_net_direction": int(sum(
                        np.sign(r["d_at_120"]) != np.sign(r["d_at_320"]) for r in rows))}
    return out


def write_curve_table(mp, cp):
    T = list(mp.columns.values.astype(float))
    path = TAB / "curve_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "condition_id", "T50_K", "w_K", "ymax_free",
                    "ymax_free_se", "r2_log_logistic",
                    "wide_window_slope_per_K", "arrhenius_slope_K",
                    "apparent_Ea_kJ_per_mol", "arrhenius_r2", "B_320_over_140"])
        for lab, piv in (("main", mp), ("check", cp)):
            for cid in piv.index:
                y = piv.loc[cid].values.astype(float)
                tt = np.asarray(T, float)
                fl = fit_logistic(tt, y)
                m = tt >= 140
                fe = ts.ols(np.column_stack([np.ones(m.sum()), tt[m]]), np.log(y[m]))
                fa = ts.ols(np.column_stack([np.ones(m.sum()), 1.0 / tt[m]]), np.log(y[m]))
                w.writerow([lab, cid, f"{fl['T50']:.4f}", f"{fl['w']:.4f}",
                            f"{fl['ymax']:.4f}", f"{fl['param_se'].get('ymax', float('nan')):.4f}",
                            f"{fl['r2_log']:.6f}",
                            f"{fe['beta'][1]:.6f}", f"{fa['beta'][1]:.4f}",
                            f"{-fa['beta'][1] * 8.314 / 1000.0:.4f}",
                            f"{fa['r2']:.6f}",
                            f"{y[tt == 320][0] / y[tt == 140][0]:.4f}"])
    print(f"[artifact] results/curve_parameters.csv")


def write_wide_table(mp):
    """Figure-backing table: the full 8 x 11 matrix with ranks."""
    T = list(mp.columns.values.astype(float))
    path = TAB / "response_matrix_with_ranks.csv"
    ranks = mp.rank(axis=0, method="average")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition_id"] + [f"y@{int(t)}" for t in T] +
                   [f"rank@{int(t)}" for t in T])
        for cid in mp.index:
            w.writerow([cid] + [f"{v:.5f}" for v in mp.loc[cid].values] +
                       [f"{v:.1f}" for v in ranks.loc[cid].values])
    print(f"[artifact] results/response_matrix_with_ranks.csv")


def main():
    mp, cp = load()
    bt = block_test(mp)
    ob = ordering_by_single_parameter(mp, cp)
    rc = exact_pair_reversal_census(mp, cp)
    out = {"block_test": bt, "ordering_by_single_parameter": ob,
           "reversal_census": rc}
    dump("I_consolidation.json", out)
    write_curve_table(mp, cp)
    write_wide_table(mp)

    print("\n=== block test ===")
    for p in ("T50", "w", "ymax"):
        d = bt[p]
        print(f"  {p}: block1 mean={d['mean_block1']:.3f} block2 mean={d['mean_block2']:.3f} "
              f"diff={d['observed_difference']:+.3f} exact p={d['exact_two_sided_p']:.4f} "
              f"separated={d['blocks_fully_separated']}")
    print("\n=== ordering predicted by T50 alone ===")
    for lab in ("main", "check"):
        print(f"  [{lab}] predicted order: {ob[lab]['predicted_order_by_T50']}")
        for r in ob[lab]["agreement_vs_setting"]:
            print(f"    T={r['setting']:5.0f} spearman={r['spearman_order_vs_T50']:+.3f} "
                  f"p={r['p']:.4f} kendall={r['kendall']:+.3f}")
    print("\n=== reversal census ===")
    for lab, v in rc.items():
        print(f"  {lab}: {v['n_with_reversal']}/{v['n_pairs']} pairs reverse; "
              f"{v['n_flipping_net_direction']} flip net direction")


if __name__ == "__main__":
    main()