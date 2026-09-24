#!/usr/bin/env python3
"""Falsification tests for the competing explanations of erosion-rate scale dependence.

Each test is designed to try to BREAK the coupling explanation rather than confirm it.

Writes results into output/analysis/w0_main/ (decisive_results.json, CSVs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = {}
TABLES = []


def save(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(HERE / name, index=False)
    TABLES.append({"file": name, "caption": caption, "rows": int(len(df))})


def main() -> None:
    d = pd.read_csv(ROOT / "output/analysis/common/erosion_consolidated.csv")

    # ---- T1: is the thickness spread the discriminator between coupled and free strata?
    rows = []
    for (proc, meth), sub in d[d.logET.notna()].groupby(["Process", "Methodology"]):
        rows.append({
            "Process": proc, "Methodology": meth, "n": int(len(sub)),
            "sd_log10_ET": float(sub["logET"].std(ddof=1)),
            "median_ET_mm": float(sub["ET_mm"].median()),
            "implied_depth_m": float(sub["ET_mm"].median() / 1000.0),
        })
    t1 = pd.DataFrame(rows).sort_values("sd_log10_ET")
    save(t1, "table_thickness_spread_diagnostic.csv",
         "Spread of implied eroded thickness per stratum; a collapsed spread marks a rate time-locked to its own interval.")

    # ---- T2: within-reference slopes in the coupled stratum ------------------
    cd = d[(d.Methodology == "Cosmogenic detrital") & (d.Process == "Nonglacial")
           & d.logER.notna() & d.logTI.notna()]
    rows = []
    for ref, sub in cd.groupby("Reference"):
        if len(sub) < 10:
            continue
        fit = sm.OLS(sub["logER"], sm.add_constant(sub["logTI"])).fit()
        rows.append({"Reference": ref, "n": int(len(sub)),
                     "slope": float(fit.params["logTI"]), "r2": float(fit.rsquared)})
    t2 = pd.DataFrame(rows).sort_values("n", ascending=False)
    save(t2, "table_within_reference_slopes.csv",
         "Slope of log10 rate vs log10 interval recomputed inside each individual study of the coupled stratum.")
    OUT["within_reference_slopes"] = {
        "n_references": int(len(t2)), "median_slope": float(t2["slope"].median()),
        "q25": float(t2["slope"].quantile(0.25)), "q75": float(t2["slope"].quantile(0.75)),
        "median_r2": float(t2["r2"].median()),
    }

    # ---- T3: thickness concentration of the coupled stratum ------------------
    e = cd["ET_mm"].dropna()
    OUT["coupled_thickness"] = {
        "n": int(len(e)), "median_mm": float(e.median()),
        "iqr_mm": [float(e.quantile(0.25)), float(e.quantile(0.75))],
        "fraction_within_620_680mm": float(((e >= 620) & (e <= 680)).mean()),
        "implied_depth_m": float(e.median() / 1000.0),
        "implied_attenuation_length_over_density_cm": float(e.median() / 10.0),
    }

    # ---- T4: matched-timescale process contrast on common support ------------
    rows = []
    for meth, sub in d[d.logER.notna() & d.logTI.notna()].groupby("Methodology"):
        g = sub[sub.Process == "Glacial"]
        n = sub[sub.Process == "Nonglacial"]
        if len(g) < 5 or len(n) < 10:
            continue
        lo = max(g["logTI"].min(), n["logTI"].min())
        hi = min(g["logTI"].max(), n["logTI"].max())
        gg = g[(g.logTI >= lo) & (g.logTI <= hi)]
        nn = n[(n.logTI >= lo) & (n.logTI <= hi)]
        if len(gg) < 5 or len(nn) < 10:
            continue
        u, p = stats.mannwhitneyu(gg["logER"], nn["logER"])
        rows.append({
            "Methodology": meth, "overlap_lo_dex": float(lo), "overlap_hi_dex": float(hi),
            "n_glacial": int(len(gg)), "n_nonglacial": int(len(nn)),
            "median_ER_glacial": float(gg["Erosion rate (mm/yr)"].median()),
            "median_ER_nonglacial": float(nn["Erosion rate (mm/yr)"].median()),
            "ratio_median": float(gg["Erosion rate (mm/yr)"].median() / nn["Erosion rate (mm/yr)"].median()),
            "median_logER_diff_dex": float(gg["logER"].median() - nn["logER"].median()),
            "mannwhitney_p": float(p),
        })
    t4 = pd.DataFrame(rows)
    save(t4, "table_common_support_process.csv",
         "Glacial vs nonglacial comparison restricted to the timescale range both processes actually sample.")
    OUT["common_support"] = t4.to_dict("records")

    # ---- T5: compiler effect ------------------------------------------------
    d["is_this_study"] = d["Compilation"] == "This study"
    rows = []
    for proc, sub in d[d.logER.notna()].groupby("Process"):
        a = sub[sub.is_this_study]["logER"]
        b = sub[~sub.is_this_study]["logER"]
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"Process": proc, "n_this_study": int(len(a)), "n_prior": int(len(b)),
                     "median_this_study_dex": float(a.median()), "median_prior_dex": float(b.median()),
                     "mean_diff_dex": float(a.mean() - b.mean()), "welch_p": float(p)})
    t5 = pd.DataFrame(rows)
    save(t5, "table_compiler_effect.csv",
         "Rows contributed by the present compilation versus rows inherited from prior compilations.")
    OUT["compiler_effect"] = t5.to_dict("records")

    OUT["tables"] = TABLES
    (HERE / "decisive_results.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: OUT[k] for k in ("within_reference_slopes", "coupled_thickness")}, indent=2))


if __name__ == "__main__":
    main()
