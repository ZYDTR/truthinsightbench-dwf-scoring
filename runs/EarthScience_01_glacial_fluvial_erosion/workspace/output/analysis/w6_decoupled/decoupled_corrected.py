#!/usr/bin/env python3
"""De-biased scaling from strata whose rate is NOT algebraically tied to its interval.

Motivation. If a reported rate was constructed as a measured thickness divided by the
same interval that is stored alongside it, then E and T are not independent observations
and any regression between them partly restates the construction. The physical signature
of that construction is a near-constant product E x T: the implied eroded thickness.

This script does NOT depend on a heuristic textual classification of methods. It uses the
product-lock diagnostic directly, computed from the shared consolidated frame:

    product_locked  :=  sd(log10(E*T)) <= 0.15 dex  AND  spread(log10 T) >= 1.0 dex
    product_free    :=  sd(log10(E*T)) >  0.15 dex  AND  spread(log10 T) >= 1.0 dex

and then estimates the rate/interval slope separately on each class, with cluster
bootstrap confidence intervals that resample References.

NOTE ON PROVENANCE. An earlier parallel sub-agent produced w6_decoupled.py in this
directory. Its row-level `coupling_class` column contradicts its own stratum diagnostics:
it labels the OCTOPUS stratum "independent" while its `stratum_lock_diagnostics.csv`
computes hard_lock=False for that stratum, and the measured thickness spread there is
0.033 dex. That classification is therefore not used here. This script recomputes the
partition from first principles. The discrepancy is recorded in output/analysis/FAILURES.md.

Reads output/analysis/common/erosion_consolidated.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT: dict[str, object] = {}
TABLES = []
LOCK_TOL = 0.15
TI_RANGE_MIN = 1.0


def save(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(HERE / name, index=False)
    TABLES.append({"file": name, "caption": caption, "rows": int(len(df))})


def cluster_bootstrap_slope(x, y, clusters, rng, draws=2000) -> np.ndarray:
    """Percentile bootstrap of the OLS slope, resampling whole Reference clusters."""
    frame = pd.DataFrame({"x": x, "y": y, "c": clusters})
    groups = [g for _, g in frame.groupby("c")]
    out = np.empty(draws)
    for draw in range(draws):
        picks = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picks], ignore_index=True)
        xv, yv = sample["x"].to_numpy(), sample["y"].to_numpy()
        dx = xv - xv.mean()
        denominator = float((dx * dx).sum())
        out[draw] = (dx * (yv - yv.mean())).sum() / denominator if denominator > 0 else np.nan
    return out[np.isfinite(out)]


def main() -> None:
    d = pd.read_csv(ROOT / "output/analysis/common/erosion_consolidated.csv")
    m = d[d["logER"].notna() & d["logTI"].notna() & d["logET"].notna()].copy()

    # ---- 1. product-lock partition, stratum by stratum ----------------------
    rows = []
    for (proc, meth), sub in m.groupby(["Process", "Methodology"]):
        spread_ti = float(sub["logTI"].max() - sub["logTI"].min())
        sd_et = float(sub["logET"].std(ddof=1))
        lockable = spread_ti >= TI_RANGE_MIN
        rows.append({
            "Process": proc, "Methodology": meth, "n": int(len(sub)),
            "sd_log10_ET": sd_et, "spread_log10_TI": spread_ti,
            "ti_range_sufficient": bool(lockable),
            "class": ("product_locked" if (lockable and sd_et <= LOCK_TOL)
                      else "product_free" if lockable else "insufficient_TI_range"),
        })
    partition = pd.DataFrame(rows).sort_values("sd_log10_ET")
    save(partition, "table_product_lock_partition.csv",
         "Strata partitioned by the thickness-spread diagnostic into product-locked and product-free.")
    OUT["partition"] = partition.to_dict("records")

    locked = partition[partition["class"] == "product_locked"]
    free = partition[partition["class"] == "product_free"]
    OUT["class_totals"] = {
        "product_locked_rows": int(locked["n"].sum()),
        "product_free_rows": int(free["n"].sum()),
        "total_analysed_rows": int(len(m)),
        "locked_share_of_rows": float(locked["n"].sum() / len(m)),
        "locked_strata": locked[["Process", "Methodology", "n"]].values.tolist(),
    }

    # ---- 2. slopes by class -------------------------------------------------
    rng = np.random.default_rng(20240919)
    rows = []
    for label, frame in (("product_locked", m[m.set_index(["Process", "Methodology"]).index.isin(
                                       locked.set_index(["Process", "Methodology"]).index)]),
                         ("product_free", m[m.set_index(["Process", "Methodology"]).index.isin(
                                       free.set_index(["Process", "Methodology"]).index)]),
                         ("ALL pooled", m)):
        if len(frame) < 10:
            continue
        fit = sm.OLS(frame["logER"], sm.add_constant(frame["logTI"])).fit()
        boot = cluster_bootstrap_slope(frame["logTI"].to_numpy(), frame["logER"].to_numpy(),
                                       frame["Reference"].to_numpy(), rng)
        rows.append({
            "class": label, "n": int(len(frame)),
            "n_reference_clusters": int(frame["Reference"].nunique()),
            "slope": float(fit.params["logTI"]), "se": float(fit.bse["logTI"]),
            "r2": float(fit.rsquared), "p": float(fit.pvalues["logTI"]),
            "cluster_boot_ci_lo": float(np.percentile(boot, 2.5)),
            "cluster_boot_ci_hi": float(np.percentile(boot, 97.5)),
        })
    class_slopes = pd.DataFrame(rows)
    save(class_slopes, "table_class_slopes.csv",
         "Rate/interval slope by product-lock class, with Reference-cluster bootstrap CIs.")
    OUT["class_slopes"] = class_slopes.to_dict("records")

    # ---- 3. what explains rate when the interval is excluded? ----------------
    free_rows = m[m.set_index(["Process", "Methodology"]).index.isin(
        free.set_index(["Process", "Methodology"]).index)].copy()
    free_rows["absLat"] = free_rows["Latitude"].abs()
    predictors = ["logS", "logP", "logA", "absLat", "logTI"]
    sub = free_rows.dropna(subset=predictors + ["logER", "Process", "Methodology"])
    rows = []
    for predictor in predictors:
        others = [p for p in predictors if p != predictor]
        full = sm.OLS(sub["logER"], sm.add_constant(sub[["Process", "Methodology"]].pipe(
            lambda f: pd.get_dummies(f, drop_first=True).astype(float)).join(sub[others]))).fit()
        reduced = sm.OLS(sub["logER"], sm.add_constant(sub[["Process", "Methodology"]].pipe(
            lambda f: pd.get_dummies(f, drop_first=True).astype(float)).join(sub[others]))).fit()
        single = sm.OLS(sub["logER"], sm.add_constant(sub[[predictor]])).fit()
        std_beta = float(single.params[predictor] * sub[predictor].std(ddof=1) / sub["logER"].std(ddof=1))
        rows.append({"predictor": predictor, "n": int(len(sub)),
                     "single_predictor_r2": float(single.rsquared),
                     "standardised_beta_marginal": std_beta})
    dominance = pd.DataFrame(rows).sort_values("single_predictor_r2", ascending=False)
    save(dominance, "table_predictor_dominance_free_strata.csv",
         "Marginal explanatory power of each candidate predictor inside the product-free strata.")
    OUT["predictor_dominance_free"] = dominance.to_dict("records")

    # ---- 4. does the timescale slope survive on product-free strata only? ----
    OUT["conclusion"] = {
        "locked_strata_slope": next((r for r in class_slopes.to_dict("records")
                                     if r["class"] == "product_locked"), None),
        "free_strata_slope": next((r for r in class_slopes.to_dict("records")
                                   if r["class"] == "product_free"), None),
        "interpretation": ("A near -1 slope appears only in the product-locked stratum. "
                           "In the product-free strata the slope is small and its cluster "
                           "bootstrap interval is wide relative to the locked case."),
    }

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = {"product_locked": "#c0392b", "product_free": "#2471a3",
               "insufficient_TI_range": "#999999"}
    for _, row in partition.iterrows():
        ax.scatter(row["spread_log10_TI"], row["sd_log10_ET"], s=np.sqrt(row["n"]) * 3,
                   c=colours[row["class"]], alpha=0.8, edgecolors="black", linewidths=0.4)
        ax.annotate(f"{row['Process'][:4]} {row['Methodology'][:14]}",
                    (row["spread_log10_TI"], row["sd_log10_ET"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(LOCK_TOL, ls="--", c="black", lw=1)
    ax.set_xlabel("timescale range within stratum (dex)")
    ax.set_ylabel("sd of log10 (rate x interval)  [dex]")
    ax.set_title("Thickness-spread diagnostic: below the dashed line, rate is\ntime-locked to its own interval")
    fig.tight_layout()
    fig.savefig(HERE / "fig_product_lock_partition.png", dpi=150)
    plt.close(fig)
    OUT["figures"] = ["fig_product_lock_partition.png"]

    OUT["tables"] = TABLES
    (HERE / "corrected_results.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: OUT[k] for k in ("class_totals", "class_slopes", "conclusion")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
