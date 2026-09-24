"""Final verification pass: recompute every number that the report cites,
from the raw data, and write one consolidated machine-readable summary.

This is the artifact the report's numeric claims resolve to.  Every value here
is recomputed from data/*.tsv in this run; nothing is copied from earlier
stage outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
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

EXPECTED = {
    "anonymous_condition_response_curves.tsv":
        "b3668160fac559700404181f686acec1394cfcdaf2ef34d38a75834556f89726",
    "independent_condition_check_curves.tsv":
        "1a79f8b8bf11fe116930ddc9e4b3a50baefe441235f41cf0bd8a1f81ae335237",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    S = {"schema": "tib-verification-summary", "task_id": "Energy_03_interface_response_boundaries"}

    # ---- integrity --------------------------------------------------------
    S["data_integrity"] = {f: {"sha256": sha256(DATA / f),
                               "matches_manifest": sha256(DATA / f) == v}
                           for f, v in EXPECTED.items()}
    assert all(v["matches_manifest"] for v in S["data_integrity"].values())

    m = pd.read_csv(DATA / "anonymous_condition_response_curves.tsv", sep="\t")
    c = pd.read_csv(DATA / "independent_condition_check_curves.tsv", sep="\t")
    mp = m.pivot(index="condition_id", columns="temperature_like_setting",
                 values="rate_like_response")
    cp = c.pivot(index="condition_id", columns="temperature_like_setting",
                 values="rate_like_response")
    T = mp.columns.values.astype(float)
    S["design"] = {
        "main_rows": int(len(m)), "check_rows": int(len(c)),
        "main_conditions": list(mp.index), "check_conditions": list(cp.index),
        "settings": [float(t) for t in T],
        "n_settings": int(len(T)),
        "main_is_balanced_complete": bool(mp.notna().all().all()),
        "check_is_balanced_complete": bool(cp.notna().all().all()),
        "no_missing_imposed": bool(m.notna().all().all() and c.notna().all().all()),
        "response_max_all_panels": float(max(m.rate_like_response.max(),
                                            c.rate_like_response.max())),
        "n_zero_responses": int((m.rate_like_response == 0).sum()
                                + (c.rate_like_response == 0).sum()),
    }

    # ---- D1 curvature -----------------------------------------------------
    curv = {}
    for lab, piv in (("main", mp), ("check", cp)):
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            sel = (T >= 140) & (y > 0)
            f = ts.ols(np.column_stack([np.ones(sel.sum()), T[sel], T[sel] ** 2]),
                       np.log(y[sel]))
            d2 = np.diff(np.log(y[sel]), 2)
            rows.append({"curve": cid, "c": float(f["beta"][2]),
                         "se": float(f["se"][2]), "t": float(f["t"][2]),
                         "p_one_sided_lt0": float(f["p"][2] / 2),
                         "r2": float(f["r2"]),
                         "n_neg_second_diffs": int((d2 < 0).sum()),
                         "n_second_diffs": int(len(d2))})
        curv[lab] = rows
    allc = [r["c"] for lab in curv for r in curv[lab]]
    S["D1_curvature"] = {
        "model": "log(y) = a + b*T + c*T^2 estimated per curve on T>=140 K",
        "curves": curv,
        "n_curves_total": len(allc),
        "n_with_negative_c": int(sum(x < 0 for x in allc)),
        "c_range": [float(min(allc)), float(max(allc))],
        "max_one_sided_p": float(max(r["p_one_sided_lt0"] for lab in curv for r in curv[lab])),
        "logistic_ymax_mean_main": None,  # filled below
    }

    # forward-holdout confirmation of curvature
    hold = {}
    for lab, piv in (("main", mp), ("check", cp)):
        n_quad_better = 0
        n_c_neg = 0
        deltas = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            tr, te = (T <= 280) & (y > 0), (T > 280) & (y > 0)
            fq = ts.ols(np.column_stack([np.ones(tr.sum()), T[tr], T[tr] ** 2]),
                        np.log(y[tr]))
            fe = ts.ols(np.column_stack([np.ones(tr.sum()), T[tr]]), np.log(y[tr]))
            eq = np.log(y[te]) - (fq["beta"][0] + fq["beta"][1] * T[te] + fq["beta"][2] * T[te] ** 2)
            ee = np.log(y[te]) - (fe["beta"][0] + fe["beta"][1] * T[te])
            rmq, rme = float(np.sqrt((eq @ eq) / len(eq))), float(np.sqrt((ee @ ee) / len(ee)))
            n_quad_better += int(rmq < rme)
            n_c_neg += int(fq["beta"][2] < 0)
            deltas.append(rme - rmq)
        hold[lab] = {"n_curves": len(piv.index), "n_c_negative_in_training": n_c_neg,
                     "n_quadratic_better_on_holdout": n_quad_better,
                     "mean_rmse_gain": float(np.mean(deltas)),
                     "sign_test_p": sum(comb(len(piv.index), i)
                                        for i in range(n_quad_better, len(piv.index) + 1))
                                     / 2 ** len(piv.index)}
    S["D1_curvature_holdout"] = hold

    # ---- D2 exponent spread vs width --------------------------------------
    d2 = {}
    for lab, piv in (("main", mp), ("check", cp)):
        rows = []
        for cid in piv.index:
            y = piv.loc[cid].values.astype(float)
            sel = T >= 140
            f = ts.ols(np.column_stack([np.ones(sel.sum()), T[sel]]), np.log(y[sel]))
            fa = ts.ols(np.column_stack([np.ones(sel.sum()), 1.0 / T[sel]]), np.log(y[sel]))
            fl = fit_logistic(T, y)
            m140 = (T >= 1.0) & (y >= 1.0) & (y <= 20.0)
            fn = (ts.ols(np.column_stack([np.ones(m140.sum()), T[m140]]),
                         np.log(y[m140]))["beta"][1] if m140.sum() >= 4 else None)
            rows.append({"curve": cid,
                         "wide_slope": float(f["beta"][1]),
                         "wide_r2": float(f["r2"]),
                         "narrow_slope": None if fn is None else float(fn),
                         "apparent_Ea_kJ_per_mol": float(-fa["beta"][1] * 8.314 / 1000),
                         "arrhenius_r2": float(fa["r2"]),
                         "T50": fl["T50"], "w": fl["w"], "ymax_free": fl["ymax"]})
        ws = [r["wide_slope"] for r in rows]
        d2[lab] = {"curves": rows,
                   "wide_slope_range": [float(min(ws)), float(max(ws))],
                   "wide_slope_max_over_min": float(max(ws) / min(ws))}
    S["D2_exponent_spread"] = d2
    rows = d2["main"]["curves"]
    S["D2_relationship"] = {
        "spearman_slope_vs_T50": ts.spearman([r["T50"] for r in rows], [r["wide_slope"] for r in rows]),
        "spearman_slope_vs_w": ts.spearman([r["w"] for r in rows], [r["wide_slope"] for r in rows]),
        "spearman_wide_vs_narrow_slope": ts.spearman(
            [r["wide_slope"] for r in rows if r["narrow_slope"] is not None],
            [r["narrow_slope"] for r in rows if r["narrow_slope"] is not None]),
        "T50_range": [float(min(r["T50"] for r in rows)), float(max(r["T50"] for r in rows))],
        "design_T_max": float(T.max()),
        "n_T50_above_design_max": int(sum(r["T50"] > T.max() for r in rows)),
    }

    # ---- D3 ordering / crossings -----------------------------------------
    d3 = {}
    for lab, piv in (("main", mp), ("check", cp)):
        cids = list(piv.index)
        def census(sel_T):
            nrev = 0
            tot = 0
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    dd = piv.loc[cids[i], sel_T].values - piv.loc[cids[j], sel_T].values
                    s = np.sign(dd)
                    s = s[s != 0]
                    tot += 1
                    if s.size > 1 and (np.diff(s) != 0).any():
                        nrev += 1
            return nrev, tot
        rev_all, tot = census(list(T))
        rev_mid, _ = census([t for t in T if 140 <= t <= 300])
        r_first = piv[T[0]].rank().values
        r_last = piv[T[-1]].rank().values
        r_mid = piv[140.0].rank().values
        r_mid2 = piv[300.0].rank().values
        adj = []
        for i in range(len(T) - 1):
            rho, p = ts.spearman(piv[T[i]].rank().values, piv[T[i + 1]].rank().values)
            adj.append({"from": float(T[i]), "to": float(T[i + 1]),
                        "spearman": rho, "p": p})
        d3[lab] = {
            "n_pairs": tot, "n_pairs_reversing_120_320": rev_all,
            "n_pairs_reversing_140_300": rev_mid,
            "spearman_rank_120_vs_320": ts.spearman(r_first, r_last),
            "spearman_rank_140_vs_300": ts.spearman(r_mid, r_mid2),
            "adjacent_rank_stability": adj,
            "min_adjacent_spearman": min(a["spearman"] for a in adj),
        }
        # exact sign test: under H0 of no systematic reversal, half the pairs flip
        po = sum(comb(tot, i) for i in range(rev_all, tot + 1)) / 2 ** tot
        d3[lab]["sign_test_p_one_sided_all_flip"] = float(po)
    S["D3_ordering"] = d3

    # ---- D4 panel correspondence ------------------------------------------
    dup = []
    for ci in cp.index:
        for mi in mp.index:
            if np.array_equal(cp.loc[ci].values.astype(float),
                              mp.loc[mi].values.astype(float)):
                dup.append({"check_curve": ci, "identical_main_curve": mi,
                            "n_settings_matched": int(len(T)),
                            "max_abs_difference": 0.0})
    dup_curves = {d["check_curve"] for d in dup}
    # how many exact value coincidences exist across panels beyond the full dups?
    exact_val_matches = 0
    for t in T:
        vals = pd.concat([mp[t], cp[t]])
        _, counts = np.unique(vals.values, return_counts=True)
        exact_val_matches += int(sum(k - 1 for k in counts if k > 1))
    S["D4_panel_correspondence"] = {
        "n_check_curves": int(len(cp.index)),
        "exact_full_curve_duplicates": dup,
        "n_exactly_duplicated_check_curves": len(dup_curves),
        "n_genuinely_new_check_curves": int(len(cp.index) - len(dup_curves)),
        "genuinely_new_check_curves": [x for x in cp.index if x not in dup_curves],
        "exact_value_repeats_within_settings_across_all_12_curves": exact_val_matches,
        "interpretation_note": "identical to all 11 significant digits at every "
                               "setting; independent measurement cannot produce this",
    }
    # does concavity hold on the genuinely-new curves alone?
    new_only = {}
    for ci in S["D4_panel_correspondence"]["genuinely_new_check_curves"]:
        y = cp.loc[ci].values.astype(float)
        sel = (T >= 140) & (y > 0)
        f = ts.ols(np.column_stack([np.ones(sel.sum()), T[sel], T[sel] ** 2]),
                   np.log(y[sel]))
        new_only[ci] = {"c": float(f["beta"][2]), "p_one_sided_lt0": float(f["p"][2] / 2),
                        "r2": float(f["r2"])}
    S["D4_panel_correspondence"]["curvature_on_genuinely_new_curves_only"] = new_only

    # ---- null: block labels ------------------------------------------------
    fits = {cid: fit_logistic(T, mp.loc[cid].values.astype(float)) for cid in mp.index}
    b1 = [c for c in mp.index if c.startswith("condition_block1")]
    b2 = [c for c in mp.index if c.startswith("condition_block2")]
    blk = {}
    for pname in ("T50", "w", "ymax"):
        v1 = [fits[c][pname] for c in b1]
        v2 = [fits[c][pname] for c in b2]
        obs = float(np.mean(v2) - np.mean(v1))
        vals = np.array(v1 + v2, float)
        stats = np.array([np.mean(vals[list(cb)]) -
                          np.mean(vals[[i for i in range(len(vals)) if i not in cb]])
                          for cb in combinations(range(len(vals)), len(b2))])
        blk[pname] = {"block1": v1, "block2": v2,
                      "mean_difference": obs,
                      "exact_two_sided_p": float((np.abs(stats) >= abs(obs) - 1e-12).mean()),
                      "n_enumerated": int(len(stats))}
    S["D5_block_null"] = blk

    (RES / "results.json").write_text(json.dumps(S, indent=2, default=str), encoding="utf-8")
    print("[artifact] results/results.json")
    print(json.dumps(S, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()