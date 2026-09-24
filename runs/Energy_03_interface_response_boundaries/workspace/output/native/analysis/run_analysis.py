"""Stage A-D: integrity, ordering/crossing structure, scaling law, family split.

Task: Energy_03_interface_response_boundaries
Sole direct evidence: data/anonymous_condition_response_curves.tsv and
data/independent_condition_check_curves.tsv (hashes verified against
data_manifest.json before use).

Design facts that drive the whole analysis (from DATA_GUIDE.md /
data_manifest.json):
  * independent analysis unit = one complete condition curve;
  * adjacent temperature settings are ordered scan points of the SAME curve,
    so curve-point splitting is invalid and is never used here;
  * all holdouts are whole-curve or contiguous-setting holdouts.

Outputs JSON artifacts into ../results/ for every stage.
"""

from __future__ import annotations

import hashlib
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
LOG = Path(__file__).resolve().parents[1] / "logs"
LOG.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA = {
    "anonymous_condition_response_curves.tsv":
        "b3668160fac559700404181f686acec1394cfcdaf2ef34d38a75834556f89726",
    "independent_condition_check_curves.tsv":
        "1a79f8b8bf11fe116930ddc9e4b3a50baefe441235f41cf0bd8a1f81ae335237",
}

T_SHARED = (140.0, 320.0)  # setting range usable for growth ratios


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, payload) -> None:
    (RES / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[artifact] results/{name}")


# ---------------------------------------------------------------------------
# Stage A -- integrity and description
# ---------------------------------------------------------------------------
def stage_a():
    receipt = {}
    for fname, want in EXPECTED_SHA.items():
        got = sha256(DATA / fname)
        receipt[fname] = {"sha256": got, "matches_manifest": got == want}
        assert got == want, f"hash mismatch for {fname}"

    main = pd.read_csv(DATA / "anonymous_condition_response_curves.tsv", sep="\t")
    check = pd.read_csv(DATA / "independent_condition_check_curves.tsv", sep="\t")

    inv = {}
    for name, df in (("anonymous_condition_response_curves.tsv", main),
                     ("independent_condition_check_curves.tsv", check)):
        temps = sorted(df.temperature_like_setting.unique())
        cids = sorted(df.condition_id.unique())
        pivot = df.pivot(index="condition_id", columns="temperature_like_setting",
                         values="rate_like_response").reindex(cids, columns=temps)
        inv[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "n_unique_conditions": len(cids),
            "condition_ids": cids,
            "n_unique_settings": len(temps),
            "settings": temps,
            "setting_step": float(np.diff(temps)[0]) if len(temps) > 1 else None,
            "balanced_panel": bool(pivot.notna().all().all()),
            "missing_cells": int(pivot.isna().sum().sum()),
            "duplicate_(condition,setting)_rows": int(df.duplicated(
                ["condition_id", "temperature_like_setting"]).sum()),
            "response_min": float(df.rate_like_response.min()),
            "response_max": float(df.rate_like_response.max()),
            "n_zero_responses": int((df.rate_like_response == 0).sum()),
            "zero_response_rows": df.loc[df.rate_like_response == 0,
                                         ["condition_id", "temperature_like_setting"]]
                                     .to_dict("records"),
        }
    art = {"integrity": receipt, "inventory": inv}
    dump("A_integrity_inventory.json", art)
    return main, check, art


# ---------------------------------------------------------------------------
# Stage B -- within-curve monotonicity, ordering, crossings
# ---------------------------------------------------------------------------
def stage_b(main: pd.DataFrame, check: pd.DataFrame):
    out = {"datasets": {}}
    for label, df in (("main", main), ("check", check)):
        temps = sorted(df.temperature_like_setting.unique())
        cids = sorted(df.condition_id.unique())
        piv = df.pivot(index="condition_id", columns="temperature_like_setting",
                       values="rate_like_response").reindex(cids, columns=temps)

        mono = {c: ts.within_curve_monotone(piv.loc[c].values) for c in cids}

        # Pairwise ordering: does any pair reverse relative order across settings?
        pairs = {}
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                d = piv.loc[a].values - piv.loc[b].values
                sgn = np.sign(d)
                sgn = sgn[sgn != 0]
                n_flips = int((np.diff(sgn) != 0).sum()) if sgn.size else 0
                pairs[f"{a}|{b}"] = {
                    "n_settings_with_tie": int((d == 0).sum()),
                    "n_sign_changes": n_flips,
                    "final_direction_at_320": ("a>b" if d[-1] > 0 else
                                               ("a<b" if d[-1] < 0 else "tie")),
                    "monotone_difference": bool(np.all(np.diff(d) >= 0) or
                                                np.all(np.diff(d) <= 0)),
                }

        # Rank of each curve at each setting
        ranks = piv.rank(axis=0, method="average")
        ordering = {str(t): list(ranks.index[np.argsort(ranks[t].values, kind="stable")])
                    for t in temps}
        rank_vectors = {c: [float(ranks.loc[c, t]) for t in temps] for c in cids}
        fully_invariant_rank = {c: bool(len(set(rank_vectors[c])) == 1) for c in cids}

        # Nonparametric crossing diagnostic: first T reaching integer response k
        trans = {c: ts.segments_with_T(piv.loc[c].values, temps) for c in cids}
        vectors = {c: [trans[c].get(k) for k in range(1, 11)] for c in cids}
        distinct = len({tuple(v) for v in vectors.values()})

        out["datasets"][label] = {
            "conditions": cids,
            "settings": temps,
            "within_curve_non_decreasing": mono,
            "n_non_monotone_curves": int(sum(not v for v in mono.values())),
            "pairwise_ordering": pairs,
            "n_pairs_with_any_sign_change": int(sum(
                v["n_sign_changes"] > 0 for v in pairs.values())),
            "ordering_at_each_setting": ordering,
            "rank_of_curve_across_settings": rank_vectors,
            "curves_with_invariant_rank": fully_invariant_rank,
            "crossing_diagnostic_threshold_settings": vectors,
            "n_distinct_threshold_vectors": distinct,
        }
    dump("B_ordering_crossings.json", out)
    return out


# ---------------------------------------------------------------------------
# Stage C -- growth ratios, curve independence, transition temperatures
# ---------------------------------------------------------------------------
def loglinear_fit(T, y):
    """OLS of log(y) on T; returns slope/intercept with inference."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    fit = ts.ols(np.column_stack([np.ones_like(T), T]), np.log(y))
    return {
        "intercept_log": float(fit["beta"][0]),
        "slope_per_K": float(fit["beta"][1]),
        "slope_se": float(fit["se"][1]),
        "slope_p": float(fit["p"][1]),
        "r2_log_space": float(fit["r2"]),
        "resid_sd_log": float(fit["resid_sd"]),
        "rss": float(fit["rss"]),
        "n": int(fit["n"]),
    }


def transition_stats(T, y):
    """Data-driven onset statistics (no model imposed)."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    steps = ts.segments_with_T(y, T)
    ratios = {str(int(T[i + 1])): float(y[i + 1] / y[i]) for i in range(len(T) - 1)
              if y[i] > 0}
    # T at which the ratio from the previous setting first drops below 1.5
    first_below = None
    for i in range(len(T) - 1):
        if y[i] > 0 and (y[i + 1] / y[i]) < 1.5:
            first_below = float(T[i])
            break
    return {
        "T_at_first_positive": float(T[np.nonzero(y > 0)[0][0]]),
        "T_reaching_k": {str(k): v for k, v in steps.items()},
        "growth_ratio_by_upper_T": ratios,
        "T_of_first_ratio_below_1.5": first_below,
        "max_ratio": float(max(ratios.values())) if ratios else None,
        "min_ratio": float(min(ratios.values())) if ratios else None,
    }


def stage_c(main: pd.DataFrame, check: pd.DataFrame):
    out = {"datasets": {}}
    fits = {}
    for label, df in (("main", main), ("check", check)):
        temps = sorted(df.temperature_like_setting.unique())
        cids = sorted(df.condition_id.unique())
        piv = df.pivot(index="condition_id", columns="temperature_like_setting",
                       values="rate_like_response").reindex(cids, columns=temps)
        per = {}
        for c in cids:
            y = piv.loc[c].values.astype(float)
            T = np.asarray(temps, float)
            m = T >= T_SHARED[0]
            rec = {"transition": transition_stats(T, y),
                   "loglinear_full_range": loglinear_fit(T[m], y[m])}
            rec["B_ratio_320_over_140"] = float(y[T == 320][0] / y[T == 140][0])
            per[c] = rec
            fits[(label, c)] = rec
        out["datasets"][label] = per

        # Within-block exponent comparison (block = first 4 conditions).
        # The check panel has only 4 conditions, so it forms a single group.
        groups = ({"block1": cids[:4], "block2": cids[4:]} if len(cids) >= 8
                  else {"panel": cids})
        blocks = {}
        for name, grp in groups.items():
            if not grp:
                continue
            slopes = [per[c]["loglinear_full_range"]["slope_per_K"] for c in grp]
            blocks[name] = {
                "conditions": list(grp),
                "slopes_per_K": slopes,
                "mean_slope": float(np.mean(slopes)),
                "sd_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
                "range_slope": float(max(slopes) - min(slopes)),
                "max_over_min_slope_ratio": float(max(slopes) / min(slopes)),
            }
        out["datasets"][label]["blocks"] = blocks
        if len(cids) == 8:
            b1 = blocks["block1"]["slopes_per_K"]
            b2 = blocks["block2"]["slopes_per_K"]
            out["datasets"][label]["cross_block_slope_test"] = {
                "mean_block1": float(np.mean(b1)),
                "mean_block2": float(np.mean(b2)),
                "ratio_block2_over_block1": float(np.mean(b2) / np.mean(b1)),
                "all_block2_slopes_exceed_all_block1_slopes":
                    bool(min(b2) > max(b1)),
                "permutation_gap": permutation_exponent_split(b1, b2),
            }
    dump("C_growth_and_exponents.json", out)
    return out, fits


def permutation_exponent_split(b1, b2):
    """Exact complete-enumeration test for the 4+4 exponent split."""
    vals = np.array(list(b1) + list(b2), float)
    T_obs = float(np.sum(b2))
    res = ts.permutation_p(direction=+1, T_obs=T_obs, values=vals,
                           target=len(b2), exact=True)
    res["observed_statistic_sum_block2"] = T_obs
    return res


# ---------------------------------------------------------------------------
# Stage D -- model competition on the pooled design
# ---------------------------------------------------------------------------
def stage_d(main: pd.DataFrame):
    temps = sorted(main.temperature_like_setting.unique())
    cids = sorted(main.condition_id.unique())
    piv = main.pivot(index="condition_id", columns="temperature_like_setting",
                     values="rate_like_response").reindex(cids, columns=temps)
    block_of = {c: (0 if i < 4 else 1) for i, c in enumerate(cids)}

    T, y, cid, blk = [], [], [], []
    for c in cids:
        for t in temps:
            T.append(t)
            y.append(piv.loc[c, t])
            cid.append(c)
            blk.append(block_of[c])
    T = np.array(T, float)
    y = np.array(y, float)
    blk = np.array(blk, float)
    n = len(y)

    # Only strictly positive responses can enter a log-scale model.
    pos = y > 0
    logy = np.log(y[pos])

    def design(kind):
        cols = [np.ones(n)]
        names = ["intercept"]
        if kind in ("curve_intercepts", "curve_intercepts_block_slope",
                    "curve_intercepts_curve_slopes"):
            for c in cids[1:]:
                cols.append((np.array(cid) == c).astype(float))
                names.append(f"curve[{c}]")
        if kind in ("curve_intercepts", "curve_intercepts_block_slope"):
            cols.append(T)
            names.append("T")
        if kind == "curve_intercepts_block_slope":
            cols.append(blk * T)
            names.append("T:block2")
        if kind == "curve_intercepts_curve_slopes":
            for c in cids[1:]:
                cols.append((np.array(cid) == c).astype(float) * T)
                names.append(f"T:curve[{c}]")
        if kind == "single_power_law":
            cols.append(T)
            names.append("T")
        return np.column_stack(cols), names

    models = {}
    kinds = ["single_power_law", "curve_intercepts", "curve_intercepts_block_slope",
             "curve_intercepts_curve_slopes"]
    for kind in kinds:
        X, names = design(kind)
        Xp = X[pos]
        fit = ts.ols(Xp, logy)
        aic, bic = ts.aic_bic(fit["rss"], int(fit["n"]), X.shape[1] + 1)
        models[kind] = {
            "n_points_used": int(fit["n"]),
            "k_parameters_excl_sigma": int(X.shape[1]),
            "rss_log_space": float(fit["rss"]),
            "r2": float(fit["r2"]),
            "adj_r2": float(fit["adj_r2"]),
            "resid_sd_log": float(fit["resid_sd"]),
            "aic": aic,
            "bic": bic,
            "parameter_names": names,
        }

    # Nested F tests
    def nested_f(small, big):
        rss_s, rss_b = models[small]["rss_log_space"], models[big]["rss_log_space"]
        k_s, k_b = models[small]["k_parameters_excl_sigma"], models[big]["k_parameters_excl_sigma"]
        df1 = k_b - k_s
        df2 = models[big]["n_points_used"] - k_b
        F = ((rss_s - rss_b) / df1) / (rss_b / df2)
        p = float(ts.student_t_sf(np.sqrt(F), df2)) if F > 0 else 1.0
        p = float(2.0 * ts.student_t_sf(np.sqrt(F), df2))  # two-sided on t = sqrt(F)
        return {"F": float(F), "df1": df1, "df2": df2, "p_two_sided": p,
                "delta_bic": float(models[small]["bic"] - models[big]["bic"])}

    tests = {
        "shared_slope_vs_block_slope_interaction":
            nested_f("curve_intercepts", "curve_intercepts_block_slope"),
        "curve_intercepts_vs_per_curve_slopes":
            nested_f("curve_intercepts", "curve_intercepts_curve_slopes"),
        "single_global_rate_vs_curve_intercepts":
            nested_f("single_power_law", "curve_intercepts"),
    }

    # Residual structure of the two leading models
    resid_summary = {}
    for kind in ("curve_intercepts", "curve_intercepts_block_slope"):
        X, _ = design(kind)
        fit = ts.ols(X[pos], logy)
        r = fit["resid"]
        resid_summary[kind] = {
            "mean_resid_log": float(r.mean()),
            "sd_resid_log": float(r.std(ddof=1)),
            "max_abs_resid_log": float(np.max(np.abs(r))),
            "resid_by_curve": {
                c: float(np.mean(r[np.array(cid)[pos] == c])) for c in cids
            },
            "resid_trend_vs_T": float(np.corrcoef(T[pos], r)[0, 1]),
        }

    out = {"models": models, "nested_tests": tests, "residual_summary": resid_summary,
           "n_zero_response_points_excluded_from_log_models": int((~pos).sum())}
    dump("D_model_competition.json", out)
    return out


def main():
    main_df, check_df, a = stage_a()
    b = stage_b(main_df, check_df)
    c, fits = stage_c(main_df, check_df)
    d = stage_d(main_df)
    print("\n=== STAGE A ===")
    print(json.dumps(a["inventory"], indent=2)[:1800])
    print("\n=== STAGE B (summary) ===")
    for k, v in b["datasets"].items():
        print(k, "non-monotone curves:", v["n_non_monotone_curves"],
              "| pairs with sign change:", v["n_pairs_with_any_sign_change"],
              "| distinct threshold vectors:", v["n_distinct_threshold_vectors"],
              "| invariant-rank curves:", sum(v["curves_with_invariant_rank"].values()),
              "/", len(v["curves_with_invariant_rank"]))
    print("\n=== STAGE C (main slopes) ===")
    for cid_, rec in c["datasets"]["main"].items():
        if not isinstance(rec, dict) or "loglinear_full_range" not in rec:
            continue
        print(f"{cid_:>18s} slope={rec['loglinear_full_range']['slope_per_K']:.5f}/K "
              f"R2={rec['loglinear_full_range']['r2_log_space']:.4f} "
              f"B(y320/y140)={rec['B_ratio_320_over_140']:.2f}")
    print("cross-block slope test:",
          json.dumps(c["datasets"]["main"].get("cross_block_slope_test"), indent=2))
    print("\ncheck panel slopes:")
    for cid_, rec in c["datasets"]["check"].items():
        if not isinstance(rec, dict) or "loglinear_full_range" not in rec:
            continue
        print(f"{cid_:>18s} slope={rec['loglinear_full_range']['slope_per_K']:.5f}/K "
              f"R2={rec['loglinear_full_range']['r2_log_space']:.4f}")
    print("\n=== STAGE D ===")
    for k, v in d["models"].items():
        print(f"{k:>34s} k={v['k_parameters_excl_sigma']:>3d} "
              f"R2={v['r2']:.5f} AIC={v['aic']:.2f} BIC={v['bic']:.2f}")
    print(json.dumps(d["nested_tests"], indent=2))


if __name__ == "__main__":
    main()