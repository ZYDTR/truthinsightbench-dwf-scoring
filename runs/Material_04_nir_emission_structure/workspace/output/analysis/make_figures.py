#!/usr/bin/env python3
"""
make_figures.py - produce supporting evidence figures (optional) for
Material_04_nir_emission_structure.

Figures (only if matplotlib is available):
  Figure 1: emission_structure.png
    * one panel per design_series: aggregate-state spectra (state_1..state_5)
      overlaid with peak markers
    * plus one dilute-state panel per design_series
  Figure 2: response_composition.png
    * long_wavelength_response vs composition_fraction per design_series,
      with a linear (OLS) fit line per series

Fallback (matplotlib NOT available):
  * record the real ImportError
  * skip figure generation
  * write output/evidence/figure_availability.tsv naming the machine-readable
    TSVs that serve as evidence for each (would-be) figure.

Prints to stdout what was actually produced.
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd

# --- guarded matplotlib import (module level so plot helpers can use it) ---
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_OK = True
    MATPLOTLIB_VER = matplotlib.__version__
    MATPLOTLIB_ERR = ""
except ImportError as e:
    MATPLOTLIB_OK = False
    MATPLOTLIB_VER = ""
    MATPLOTLIB_ERR = f"{e.__class__.__name__}: {e}"

BASE = Path("/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project")
DATA = BASE / "data"
OUTPUT = BASE / "output"
FIGS_DIR = OUTPUT / "figures"
EVID_DIR = OUTPUT / "evidence"
ANALYSIS_DIR = OUTPUT / "analysis"

AGG_TSV = DATA / "aggregate_state_emission_spectra.tsv"
DIL_TSV = DATA / "dilute_state_emission_spectra.tsv"
RESP_TSV = DATA / "long_wavelength_response.tsv"

FIG1_PNG = FIGS_DIR / "emission_structure.png"
FIG2_PNG = FIGS_DIR / "response_composition.png"
FIGMETA_JSON = FIGS_DIR / "figmeta.json"
FIGAVAIL_TSV = EVID_DIR / "figure_availability.tsv"

STATE_IDS = [f"state_{i}" for i in range(1, 6)]


def load_data():
    agg = pd.read_csv(AGG_TSV, sep="\t")
    dil = pd.read_csv(DIL_TSV, sep="\t")
    resp = pd.read_csv(RESP_TSV, sep="\t")
    return agg, dil, resp


def peak_of(sub):
    """Return (wavelength_nm, intensity) of the argmax point of a 1-D spectrum."""
    m = sub["relative_emission_au"].max()
    row = sub.loc[sub["relative_emission_au"] == m].iloc[0]
    return float(row["wavelength_nm"]), float(m)


def build_figmeta(agg, dil, resp):
    """Describe each produced figure (panels, x/y units, source data)."""
    series = sorted(agg["design_series"].unique())
    figmeta = {}

    f1_panels = []
    for s in series:
        f1_panels.append({
            "label": f"aggregate:{s}",
            "type": "aggregate",
            "x_unit": "wavelength_nm",
            "y_unit": "relative_emission_au",
            "curves": STATE_IDS,
            "source_tsv": str(AGG_TSV),
        })
    for s in sorted(dil["design_series"].unique()):
        f1_panels.append({
            "label": f"dilute:{s}",
            "type": "dilute",
            "x_unit": "wavelength_nm",
            "y_unit": "relative_emission_au",
            "curves": [s],
            "source_tsv": str(DIL_TSV),
        })
    figmeta["emission_structure.png"] = {
        "panels": f1_panels,
        "x_unit": "wavelength_nm",
        "y_unit": "relative_emission_au",
        "source_data": [str(AGG_TSV), str(DIL_TSV)],
    }

    f2_panels = []
    for s in sorted(resp["design_series"].unique()):
        f2_panels.append({
            "label": f"response:{s}",
            "type": "response",
            "x_unit": "composition_fraction",
            "y_unit": "long_wavelength_response_au",
            "has_linear_fit": True,
            "source_tsv": str(RESP_TSV),
        })
    figmeta["response_composition.png"] = {
        "panels": f2_panels,
        "x_unit": "composition_fraction",
        "y_unit": "long_wavelength_response_au",
        "source_data": [str(RESP_TSV)],
    }
    return figmeta


def plot_figure1(agg, dil):
    """Figure 1: aggregate-state spectra (state_1..state_5) + dilute panel
    per design_series, with peak markers."""
    series = sorted(agg["design_series"].unique())
    n_series = len(series)
    # grid: rows = series, col0 = aggregate (5 states), col1 = dilute
    fig, axes = plt.subplots(n_series, 2, figsize=(14, 4.5 * n_series),
                             squeeze=False)
    for i, s in enumerate(series):
        # --- aggregate panel ---
        ax = axes[i, 0]
        for st in STATE_IDS:
            sub = agg[(agg["design_series"] == s) & (agg["aggregate_state_id"] == st)]
            sub = sub.sort_values("wavelength_nm")
            ax.plot(sub["wavelength_nm"], sub["relative_emission_au"],
                    label=st, lw=1.0)
            pw, pi = peak_of(sub)
            ax.plot([pw], [pi], marker="o", ms=4, zorder=5)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("relative emission (a.u.)")
        ax.set_title(f"Aggregate emission - {s}")
        ax.legend(fontsize=7)

        # --- dilute panel ---
        ax2 = axes[i, 1]
        ds = dil[dil["design_series"] == s].sort_values("wavelength_nm")
        ax2.plot(ds["wavelength_nm"], ds["relative_emission_au"], lw=1.0)
        pw, pi = peak_of(ds)
        ax2.plot([pw], [pi], marker="o", ms=4, zorder=5, label="peak")
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_ylabel("relative emission (a.u.)")
        ax2.set_title(f"Dilute emission - {s}")
        ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG1_PNG, dpi=150)
    plt.close(fig)


def plot_figure2(resp):
    """Figure 2: long_wavelength_response vs composition_fraction per
    design_series, with a linear fit line."""
    series = sorted(resp["design_series"].unique())
    n_series = len(series)
    fig, axes = plt.subplots(n_series, 1, figsize=(8, 4.0 * n_series),
                             squeeze=False)
    if n_series == 1:
        axes = [axes[0]]
    for i, s in enumerate(series):
        ax = axes[i]
        sub = resp[resp["design_series"] == s]
        x = sub["composition_fraction"].to_numpy(dtype=float)
        y = sub["long_wavelength_response_au"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.0, label=f"{s} data")
        # OLS linear fit
        coef, intercept = np.polyfit(x, y, 1)
        xgrid = np.linspace(x.min(), x.max(), 50)
        ax.plot(xgrid, coef * xgrid + intercept, "--", lw=1.5,
                label=f"linear fit (slope={coef:.1f})")
        ax.set_xlabel("composition_fraction")
        ax.set_ylabel("long_wavelength_response (a.u.)")
        ax.set_title(f"Long-wavelength response vs composition - {s}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG2_PNG, dpi=150)
    plt.close(fig)


def write_figmeta(figmeta):
    FIGMETA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(FIGMETA_JSON, "w") as f:
        json.dump(figmeta, f, indent=2)


def write_figure_availability(error_text, agg, dil, resp):
    """Fallback: record matplotlib unavailability and the machine-readable
    TSVs that stand in for each figure."""
    series_agg = sorted(agg["design_series"].unique())
    series_dil = sorted(dil["design_series"].unique())
    series_resp = sorted(resp["design_series"].unique())

    f1_evidence = [
        str(AGG_TSV),
        str(DIL_TSV),
        str(EVID_DIR / "spectral_summary.tsv"),
        str(EVID_DIR / "state_by_state.tsv"),
        str(EVID_DIR / "state_effects.tsv"),
        str(EVID_DIR / "dilute_summary.tsv"),
        str(EVID_DIR / "dilute_vs_aggregate.tsv"),
    ]
    f2_evidence = [
        str(RESP_TSV),
        str(EVID_DIR / "composition_effects.tsv"),
    ]

    rows = []
    for s in series_agg:
        rows.append({
            "figure_id": "figure_1_emission_structure",
            "panel": f"aggregate:{s}",
            "status": "skipped_no_matplotlib",
            "matplotlib_available": "False",
            "matplotlib_version": "",
            "machine_readable_evidence_tsv": ";".join(f1_evidence),
        })
    for s in series_dil:
        rows.append({
            "figure_id": "figure_1_emission_structure",
            "panel": f"dilute:{s}",
            "status": "skipped_no_matplotlib",
            "matplotlib_available": "False",
            "matplotlib_version": "",
            "machine_readable_evidence_tsv": ";".join(f1_evidence),
        })
    for s in series_resp:
        rows.append({
            "figure_id": "figure_2_response_composition",
            "panel": f"response:{s}",
            "status": "skipped_no_matplotlib",
            "matplotlib_available": "False",
            "matplotlib_version": "",
            "machine_readable_evidence_tsv": ";".join(f2_evidence),
        })

    df = pd.DataFrame(rows)
    df.to_csv(FIGAVAIL_TSV, sep="\t", index=False)
    return df


def main():
    print("=== make_figures.py ===")
    print("Base:", BASE)
    agg, dil, resp = load_data()
    print(f"Loaded aggregate rows={len(agg)} series={sorted(agg['design_series'].unique())}")
    print(f"Loaded dilute rows={len(dil)} series={sorted(dil['design_series'].unique())}")
    print(f"Loaded response rows={len(resp)} series={sorted(resp['design_series'].unique())}")

    produced = []
    errors = []

    if MATPLOTLIB_OK:
        print(f"matplotlib available (version {MATPLOTLIB_VER}).")
        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        # Figure 1
        plot_figure1(agg, dil)
        produced.append(f"figure_1: {FIG1_PNG}")
        # Figure 2
        plot_figure2(resp)
        produced.append(f"figure_2: {FIG2_PNG}")
        # figmeta
        figmeta = build_figmeta(agg, dil, resp)
        write_figmeta(figmeta)
        produced.append(f"figmeta: {FIGMETA_JSON}")
        print("Figure branch complete.")
    else:
        error_text = f"ImportError: {MATPLOTLIB_ERR}"
        print("matplotlib is NOT available.")
        print("Real error:", error_text)
        errors.append(error_text)
        figavail = write_figure_availability(error_text, agg, dil, resp)
        produced.append(f"fallback_tsv: {FIGAVAIL_TSV}")
        print("Figures skipped (matplotlib unavailable).")
        print(f"Wrote fallback TSV with {len(figavail)} rows.")

    print("\n=== PRODUCED ===")
    for p in produced:
        print(p)
    if errors:
        print("\n=== ERRORS ===")
        for err in errors:
            print(err)
    else:
        print("\n=== ERRORS: none ===")


if __name__ == "__main__":
    main()
