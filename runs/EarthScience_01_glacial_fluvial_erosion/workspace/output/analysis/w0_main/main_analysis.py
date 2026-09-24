#!/usr/bin/env python3
"""Core cross-cutting analysis: does observation timescale, process category, or
measurement method govern reported erosion rates, and how much of the apparent
timescale dependence is mathematical coupling rather than geology?

Reads only output/analysis/common/erosion_consolidated.csv (built from data/raw/*.tsv).

Writes results.json, several CSV tables and PNG figures into output/analysis/w0_main/.
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
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = ROOT / "output" / "analysis" / "common"
RESULTS = {}
TABLES = []


def read() -> pd.DataFrame:
    df = pd.read_csv(COMMON / "erosion_consolidated.csv")
    df["logER"] = pd.to_numeric(df["logER"], errors="coerce")
    df["logTI"] = pd.to_numeric(df["logTI"], errors="coerce")
    df["ET_mm"] = pd.to_numeric(df["ET_mm"], errors="coerce")
    return df


def ols(frame: pd.DataFrame, formula: str) -> dict:
    fit = smf.ols(formula, data=frame).fit()
    return {
        "formula": formula,
        "n": int(fit.nobs),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "params": {k: float(v) for k, v in fit.params.items()},
        "bse": {k: float(v) for k, v in fit.bse.items()},
        "pvalues": {k: float(v) for k, v in fit.pvalues.items()},
    }


def bootstrap_slope(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    """Vectorised ordinary-least-squares slope over `draws` row-resamples of (x, y)."""
    n = len(x)
    idx = rng.integers(0, n, size=(draws, n))
    xb = x[idx]
    yb = y[idx]
    xm = xb.mean(axis=1, keepdims=True)
    ym = yb.mean(axis=1, keepdims=True)
    dx = xb - xm
    dy = yb - ym
    denominator = (dx * dx).sum(axis=1)
    slope = np.where(denominator > 0, (dx * dy).sum(axis=1) / denominator, np.nan)
    return slope[np.isfinite(slope)]


def save_table(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(HERE / name, index=False)
    TABLES.append({"file": name, "caption": caption, "rows": int(len(df)), "columns": list(df.columns)})


def main() -> None:
    df = read()
    RESULTS["input_rows"] = {"glacial": int((df.Process == "Glacial").sum()),
                             "nonglacial": int((df.Process == "Nonglacial").sum())}

    # ---- 1. Coverage of the analysed variables -------------------------------
    coverage = (
        df.groupby(["Process", "Methodology"], as_index=False)
        .agg(n=("logER", "size"),
             n_logTI=("logTI", lambda s: int(s.notna().sum())),
             median_ER_mm_yr=("Erosion rate (mm/yr)", "median"),
             median_TI_yr=("Time interval (yr)", "median"),
             min_TI_yr=("Time interval (yr)", "min"),
             max_TI_yr=("Time interval (yr)", "max"))
        .sort_values(["Process", "n"], ascending=[True, False])
    )
    save_table(coverage, "table_coverage_by_stratum.csv",
               "Rows, timescale coverage and median values per process x methodology stratum.")

    # ---- 2. Mathematical coupling: the E = D / T identity --------------------
    ident = df[df["ET_equals_exhumation"].notna()].copy()
    by_method = (
        ident.groupby(["Process", "Methodology"], as_index=False)
        .agg(n_identity_checked=("ET_equals_exhumation", "size"),
             n_identity_exact=("ET_equals_exhumation", "sum"),
             median_ratio=("ET_vs_exhumation_ratio", "median"))
    )
    by_method["fraction_exact"] = by_method["n_identity_exact"] / by_method["n_identity_checked"]
    save_table(by_method, "table_coupling_identity.csv",
               "Fraction of rows where Erosion rate x Time interval equals Exhumation x 1000 (unit identity).")
    RESULTS["identity_test"] = by_method.to_dict("records")

    # ---- 3. ET constancy per stratum (signature of a fixed measured thickness)
    et = df[df["ET_mm"].notna() & (df["ET_mm"] > 0)].copy()
    et_stats = (
        et.groupby(["Process", "Methodology"], as_index=False)
        .agg(n=("ET_mm", "size"),
             median_ET_mm=("ET_mm", "median"),
             q25_ET_mm=("ET_mm", lambda s: float(s.quantile(0.25))),
             q75_ET_mm=("ET_mm", lambda s: float(s.quantile(0.75))),
             sd_log10_ET=("logET", lambda s: float(np.nanstd(s, ddof=1))),
             cv_ET=("ET_mm", lambda s: float(s.std(ddof=1) / s.mean())))
    )
    save_table(et_stats, "table_ET_constancy.csv",
               "Spread of the implied eroded thickness E x T per stratum; a near-constant E x T indicates a fixed measured thickness.")
    RESULTS["ET_constancy"] = et_stats.to_dict("records")

    # ---- 4. Pooled vs method-stratified scale dependence --------------------
    m = df[df["logER"].notna() & df["logTI"].notna()].copy()
    models = [
        ols(m, "logER ~ 1"),
        ols(m, "logER ~ logTI"),
        ols(m, "logER ~ C(Process)"),
        ols(m, "logER ~ C(Methodology)"),
        ols(m, "logER ~ C(Compilation)"),
        ols(m, "logER ~ C(Reference)"),
        ols(m, "logER ~ logTI + C(Methodology)"),
        ols(m, "logER ~ logTI * C(Methodology)"),
        ols(m, "logER ~ logTI + C(Methodology) + C(Process)"),
        ols(m, "logER ~ logTI * C(Methodology) + C(Process)"),
        ols(m, "logER ~ C(Process) * C(Methodology)"),
        ols(m, "logER ~ logS + logP + logA + Latitude"),
        ols(m, "logER ~ logTI + logS + logP + logA + C(Methodology)"),
    ]
    model_table = pd.DataFrame([{k: v for k, v in row.items() if k not in ("params", "bse", "pvalues")}
                                for row in models])
    save_table(model_table, "table_model_comparison.csv",
               "Competing explanations for log10 erosion rate ranked by fit.")
    RESULTS["models"] = models

    # Incremental R^2 (added-last) for the timescale and process terms.
    full = smf.ols("logER ~ logTI + logS + logP + logA + C(Methodology) + C(Process)", data=m).fit()
    RESULTS["incremental_r2"] = {
        "full_r2": float(full.rsquared),
        "without_logTI": float(smf.ols("logER ~ logS + logP + logA + C(Methodology) + C(Process)", data=m).fit().rsquared),
        "without_process": float(smf.ols("logER ~ logTI + logS + logP + logA + C(Methodology)", data=m).fit().rsquared),
        "without_methodology": float(smf.ols("logER ~ logTI + logS + logP + logA + C(Process)", data=m).fit().rsquared),
    }

    # ---- 5. Per-stratum slope with bootstrap CIs ----------------------------
    rng = np.random.default_rng(20240919)
    rows = []
    for (process, method), sub in m.groupby(["Process", "Methodology"]):
        if len(sub) < 5:
            rows.append({"Process": process, "Methodology": method, "n": len(sub),
                         "slope": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "r2": np.nan, "p": np.nan, "note": "n<5, not fitted"})
            continue
        fit = sm.OLS(sub["logER"], sm.add_constant(sub["logTI"])).fit()
        boots = bootstrap_slope(sub["logTI"].to_numpy(), sub["logER"].to_numpy(), rng, 2000)
        rows.append({
            "Process": process, "Methodology": method, "n": int(len(sub)),
            "slope": float(fit.params["logTI"]), "se": float(fit.bse["logTI"]),
            "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "r2": float(fit.rsquared), "p": float(fit.pvalues["logTI"]),
            "note": "consistent with -1 (mechanical)" if np.percentile(boots, 97.5) < -0.9
                    else ("consistent with 0 (no scale relation)" if np.percentile(boots, 2.5) < 0 < np.percentile(boots, 97.5)
                          else "intermediate"),
        })
    slopes = pd.DataFrame(rows).sort_values(["Process", "n"], ascending=[True, False])
    save_table(slopes, "table_stratum_slopes.csv",
               "log10(erosion rate) vs log10(timescale) slope per stratum with 2000-resample bootstrap CI.")
    RESULTS["stratum_slopes"] = slopes.to_dict("records")

    pooled = sm.OLS(m["logER"], sm.add_constant(m["logTI"])).fit()
    RESULTS["pooled_slope"] = {
        "n": int(pooled.nobs), "slope": float(pooled.params["logTI"]),
        "se": float(pooled.bse["logTI"]), "r2": float(pooled.rsquared),
        "p": float(pooled.pvalues["logTI"]),
        "ci": [float(pooled.conf_int().loc["logTI", 0]), float(pooled.conf_int().loc["logTI", 1])],
    }

    # ---- 6. Matched-timescale process comparison ----------------------------
    matched_rows = []
    for method, sub in m.groupby("Methodology"):
        g = sub[sub.Process == "Glacial"]
        nn = sub[sub.Process == "Nonglacial"]
        if len(g) < 5 or len(nn) < 20:
            continue
        diffs = []
        for _, row in g.iterrows():
            dist = (nn["logTI"] - row["logTI"]).abs()
            idx = dist.nsmallest(5).index
            diffs.append(row["logER"] - nn.loc[idx, "logER"].mean())
        diffs = np.array(diffs)
        t, p = stats.ttest_1samp(diffs, 0.0)
        boot = np.array([np.mean(rng.choice(diffs, len(diffs))) for _ in range(2000)])
        matched_rows.append({
            "Methodology": method, "n_glacial_matched": int(len(diffs)),
            "mean_diff_dex": float(diffs.mean()), "median_diff_dex": float(np.median(diffs)),
            "ratio_glacial_over_nonglacial": float(10 ** diffs.mean()),
            "t": float(t), "p": float(p),
            "boot_ci_lo": float(np.percentile(boot, 2.5)), "boot_ci_hi": float(np.percentile(boot, 97.5)),
        })
    matched = pd.DataFrame(matched_rows)
    save_table(matched, "table_matched_process.csv",
               "Glacial minus nonglacial log10 erosion rate at matched timescale (5 nearest neighbours in log10 time interval), per methodology.")
    RESULTS["matched_process"] = matched.to_dict("records")

    # ---- 7. Alternatives: the geological-controls-only model -----------------
    geological = m.dropna(subset=["logS", "logP", "logA", "Latitude"])
    RESULTS["geological_controls"] = ols(geological, "logER ~ logS + logP + logA + Latitude")
    RESULTS["geological_controls_n"] = int(len(geological))

    # ---- Figures ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for process, colour in (("Glacial", "#1f5fa8"), ("Nonglacial", "#c1663a")):
        sub = m[m.Process == process]
        axes[0].scatter(sub["logTI"], sub["logER"], s=6, alpha=0.35, c=colour, label=process)
    axes[0].set_xlabel("log10 time interval (yr)")
    axes[0].set_ylabel("log10 erosion rate (mm/yr)")
    axes[0].set_title("All rows: no single stable relation")
    axes[0].legend(loc="lower left", fontsize=8)

    show = slopes[slopes["n"] >= 5]
    axes[1].errorbar(show["slope"], np.arange(len(show)),
                     xerr=[show["slope"] - show["ci_lo"], show["ci_hi"] - show["slope"]],
                     fmt="o", ms=4, capsize=3, ecolor="#666666")
    axes[1].axvline(-1, ls="--", c="crimson", lw=1)
    axes[1].axvline(0, ls=":", c="black", lw=1)
    axes[1].set_yticks(np.arange(len(show)))
    axes[1].set_yticklabels([f"{r.Process[:4]} {r.Methodology}" for r in show.itertuples()], fontsize=7)
    axes[1].set_xlabel("slope of log10 rate vs log10 interval")
    axes[1].set_title("Method-stratified slopes (-1 = mechanical coupling)")
    fig.tight_layout()
    fig.savefig(HERE / "figure_scale_by_stratum.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    et_stats_sorted = et_stats.sort_values("sd_log10_ET")
    ax.barh(np.arange(len(et_stats_sorted)), et_stats_sorted["sd_log10_ET"], color="#4a7c59")
    ax.set_yticks(np.arange(len(et_stats_sorted)))
    ax.set_yticklabels([f"{r.Process[:4]} {r.Methodology}" for r in et_stats_sorted.itertuples()], fontsize=7)
    ax.set_xlabel("sd of log10 (erosion rate x time interval)")
    ax.set_title("Spread of implied eroded thickness: near 0 means rate is set by a fixed depth")
    fig.tight_layout()
    fig.savefig(HERE / "figure_ET_spread.png", dpi=150)
    plt.close(fig)

    RESULTS["tables"] = TABLES
    RESULTS["figures"] = ["figure_scale_by_stratum.png", "figure_ET_spread.png"]
    (HERE / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: RESULTS[k] for k in
                      ("input_rows", "pooled_slope", "incremental_r2", "geological_controls")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
