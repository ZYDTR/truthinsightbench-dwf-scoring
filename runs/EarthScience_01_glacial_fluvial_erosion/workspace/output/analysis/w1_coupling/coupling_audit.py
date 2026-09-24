#!/usr/bin/env python3
"""Audit how much of the erosion-rate / timescale relation is mathematical coupling.

Three independent lines of evidence are assembled, each able to fail on its own:
  A. the unit identity E [mm/yr] x T [yr] == Exhumation [m] x 1000, wherever both columns exist;
  B. the collapse of the implied thickness E x T onto a constant within a stratum;
  C. a bottom-up reconstruction of the coupled stratum from the raw prior-compilation columns,
     using the published derivation script as the method definition.

Reads data/raw/glacial_erosion_Earth.tsv, data/raw/nonglacial_erosion_Earth.tsv and
data/raw/prior compilations/OCTOPUS/{octopusdata.tsv,octopusages.tsv}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "data" / "raw"
OUT: dict[str, object] = {}
TABLES = []


def save(df: pd.DataFrame, name: str, caption: str) -> None:
    df.to_csv(HERE / name, index=False)
    TABLES.append({"file": name, "caption": caption, "rows": int(len(df))})


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column].astype(str).str.strip().replace({"": np.nan, "NA": np.nan}),
                         errors="coerce")


def main() -> None:
    g = pd.read_csv(RAW / "glacial_erosion_Earth.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = pd.read_csv(RAW / "nonglacial_erosion_Earth.tsv", sep="\t", dtype=str, keep_default_na=False)
    g["Process"] = "Glacial"
    n["Process"] = "Nonglacial"

    frames = []
    for frame in (g, n):
        f = pd.DataFrame({
            "Process": frame["Process"],
            "Methodology": frame["Methodology"],
            "Reference": frame["Reference"],
            "ER": numeric(frame, "Erosion rate (mm/yr)"),
            "TI": numeric(frame, "Time interval (yr)"),
            "Exhumation": numeric(frame, "Exhumation (m)"),
            "SedimentFlux": numeric(frame, "Sediment flux (km3/yr)"),
            "Area": numeric(frame, "Area (km2)"),
        })
        f["ET"] = f["ER"] * f["TI"]
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    # ---- A. exact unit identity --------------------------------------------
    chk = d[d["ET"].notna() & d["Exhumation"].notna()].copy()
    chk["ratio"] = chk["ET"] / (chk["Exhumation"] * 1000.0)
    chk["exact"] = np.isclose(chk["ET"], chk["Exhumation"] * 1000.0, rtol=1e-3)
    tA = (chk.groupby(["Process", "Methodology"], as_index=False)
          .agg(n_checked=("exact", "size"), n_exact=("exact", "sum"),
               median_ratio=("ratio", "median")))
    tA["fraction_exact"] = tA["n_exact"] / tA["n_checked"]
    save(tA, "table_A_unit_identity.csv",
         "Rows where Erosion rate x Time interval reproduces Exhumation x 1000 exactly.")
    OUT["A_unit_identity"] = {
        "rows_with_both_columns": int(len(chk)),
        "rows_satisfying_identity": int(chk["exact"].sum()),
        "fraction": float(chk["exact"].mean()),
        "by_stratum": tA.to_dict("records"),
    }

    # ---- B. implied thickness collapse --------------------------------------
    tB = (d[d["ET"].notna() & (d["ET"] > 0)]
          .groupby(["Process", "Methodology"], as_index=False)
          .agg(n=("ET", "size"), median_ET_mm=("ET", "median"),
               q25_mm=("ET", lambda s: float(s.quantile(0.25))),
               q75_mm=("ET", lambda s: float(s.quantile(0.75))),
               sd_log10_ET=("ET", lambda s: float(np.log10(s).std(ddof=1)))))
    save(tB, "table_B_thickness_collapse.csv",
         "Spread of implied eroded thickness per stratum; a collapsed spread flags a rate time-locked to its own interval.")
    OUT["B_thickness_collapse"] = tB.to_dict("records")

    # ---- C. bottom-up reconstruction of the coupled stratum ------------------
    octo = pd.read_csv(RAW / "prior compilations/OCTOPUS/octopusdata.tsv", sep="\t",
                       dtype=str, keep_default_na=False)
    ages = pd.read_csv(RAW / "prior compilations/OCTOPUS/octopusages.tsv", sep="\t")
    be = numeric(octo, "be10nc")
    prod = (numeric(octo, "beprod") * numeric(octo, "betopo")
            * numeric(octo, "beself") * numeric(octo, "besnow"))
    e_reported = numeric(octo, "ebe_mmkyr") / 1000.0  # mm/kyr -> mm/yr
    exposure = pd.to_numeric(ages["Be10_exposure_yr"], errors="coerce")

    recon = pd.DataFrame({"be10nc": be, "P_total_relative": prod,
                          "E_reported_mm_yr": e_reported, "exposure_yr": exposure})
    recon["ET_mm"] = recon["E_reported_mm_yr"] * recon["exposure_yr"]
    valid = recon.dropna(subset=["ET_mm"])
    OUT["C_reconstruction"] = {
        "rows_pair_aligned": int(len(recon)),
        "rows_with_product": int(len(valid)),
        "median_ET_mm": float(valid["ET_mm"].median()),
        "q25_ET_mm": float(valid["ET_mm"].quantile(0.25)),
        "q75_ET_mm": float(valid["ET_mm"].quantile(0.75)),
        "cv_ET": float(valid["ET_mm"].std(ddof=1) / valid["ET_mm"].mean()),
        "implied_Lambda_over_rho_cm": float(valid["ET_mm"].median() / 10.0),
    }

    # Sensitivity of the implied attenuation length to assumed rock density.
    sea_level_production = 4.30  # atoms/g/yr, as used in the published derivation script
    p_abs = sea_level_production * prod
    rows = []
    for rho in (2.50, 2.60, 2.65, 2.70):
        e_pred = p_abs * 160.0 / (rho * be) * 10.0  # cm/yr -> mm/yr
        ratio = e_pred / recon["E_reported_mm_yr"]
        rows.append({"assumed_rho_g_cm3": rho, "assumed_Lambda_g_cm2": 160.0,
                     "median_Epred_over_Ereported": float(ratio.median()),
                     "q25": float(ratio.quantile(0.25)), "q75": float(ratio.quantile(0.75)),
                     "fraction_within_5pct": float(((ratio > 0.95) & (ratio < 1.05)).mean())})
    tC = pd.DataFrame(rows)
    save(tC, "table_C_density_sensitivity.csv",
         "Reconstruction of the reported rate from raw nuclide columns under four assumed rock densities.")

    # Does the reported rate equal the analytic steady-state form?
    recon["E_analytic"] = p_abs * 160.0 / (2.65 * be) * 10.0
    sub = recon.dropna(subset=["E_analytic", "E_reported_mm_yr"])
    sub = sub[(sub["E_analytic"] > 0) & (sub["E_reported_mm_yr"] > 0)]
    fit = sm.OLS(np.log10(sub["E_reported_mm_yr"]), sm.add_constant(np.log10(sub["E_analytic"]))).fit()
    OUT["C_analytic_recovery"] = {
        "n": int(fit.nobs), "slope": float(fit.params.iloc[1]),
        "se": float(fit.bse.iloc[1]), "r2": float(fit.rsquared),
        "spearman_rho": float(sub["E_analytic"].corr(sub["E_reported_mm_yr"], method="spearman")),
    }

    # If E is set by a fixed depth D over interval T, then log E = log D - log T exactly.
    # Report observed vs mechanically-implied slope for the coupled stratum.
    cd = d[(d.Methodology == "Cosmogenic detrital") & (d.Process == "Nonglacial")]
    cd = cd[(cd["ER"] > 0) & (cd["TI"] > 0)]
    fit_obs = sm.OLS(np.log10(cd["ER"]), sm.add_constant(np.log10(cd["TI"]))).fit()
    OUT["C_coupled_slope"] = {
        "n": int(fit_obs.nobs), "observed_slope": float(fit_obs.params.iloc[1]),
        "observed_r2": float(fit_obs.rsquared),
        "mechanically_implied_slope": -1.0,
        "mechanically_implied_r2_if_depth_fixed": 1.0,
    }

    # Contact test: does the same stratum's slope survive if thickness is allowed to vary?
    vol = d[(d.Methodology == "Volumetric") & (d["ER"] > 0) & (d["TI"] > 0)]
    fit_vol = sm.OLS(np.log10(vol["ER"]), sm.add_constant(np.log10(vol["TI"]))).fit()
    OUT["D_free_stratum_slope"] = {
        "stratum": "Volumetric (pooled processes)",
        "n": int(fit_vol.nobs), "slope": float(fit_vol.params.iloc[1]),
        "se": float(fit_vol.bse.iloc[1]), "r2": float(fit_vol.rsquared),
        "sd_log10_ET": float(np.log10(vol["ET"]).std(ddof=1)),
    }
    tD = (vol.groupby("Process", as_index=False)
          .agg(n=("ET", "size"), median_ET_mm=("ET", "median"),
               sd_log10_ET=("ET", lambda s: float(np.log10(s).std(ddof=1)))))
    save(tD, "table_D_free_stratum_thickness.csv",
         "Thickness spread in the Volumetric stratum, where rate and interval are free to vary independently.")

    OUT["tables"] = TABLES
    (HERE / "results.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(OUT, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
