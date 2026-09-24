#!/usr/bin/env python3
"""Aggregate the 13 per-file DNP results (written by the parallel sub-agents)
into the final trend/correlation/figure deliverables.

This is a single SEQUENTIAL reduction over the per-file outputs; it is not
parallelizable (it consumes all 13 results jointly). Input:
  output/results/perfile/*.json
Output:
  output/results/epr_metrics.csv, nmr_metrics.csv, summary.json
  output/figures/epr_echo.png, nmr_perindirect.png, nmr_structure.png, comparison.png
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent  # output/
PERFILE = BASE / "results" / "perfile"
OUT = BASE / "results"
FIG = BASE / "figures"

INV_NS = [0, 132, 220, 440, 880, 2200]


def load_perfile():
    rows = []
    for p in sorted(PERFILE.glob("*.json")):
        with open(p) as fh:
            d = json.load(fh)
        rows.append(d)
    return rows


# Explicit mapping (matches data/comparison_manifest.json)
NAME_TO_NS = {
    "20240522_125527_dnp_echo_pp_length_2": 0,
    "20240522_130805_dnp_echo_pp_length_2": 132,
    "20240522_130914_dnp_echo_pp_length_2": 220,
    "20240522_131018_dnp_echo_pp_length_2": 440,
    "20240522_131142_dnp_echo_pp_length_2": 880,
    "20240522_131404_dnp_echo_pp_length_2": 2200,
    "20240523_1308_dnpdata_dnp_contact_swp": 0,
    "20240522_1404_dnpdata_dnp_contact_swp": 132,
    "20240522_1434_dnpdata_dnp_contact_swp": 220,
    "20240522_1505_dnpdata_dnp_contact_swp": 440,
    "20240522_1535_dnpdata_dnp_contact_swp": 880,
    "20240523_0954_dnpdata_dnp_contact_swp": 2200,
}


def main():
    rows = load_perfile()
    epr_rows, nmr_rows, ref_rows = [], [], []
    for d in rows:
        name = d["file"].replace(".mat", "").replace(".json", "")
        ns = NAME_TO_NS.get(name)
        if d["type"] == "epr":
            d2 = dict(d); d2["inv_ns"] = ns
            epr_rows.append(d2)
        else:
            d2 = dict(d); d2["inv_ns"] = ns
            if d.get("is_reference"):
                ref_rows.append(d2)
            else:
                nmr_rows.append(d2)

    # sort by inversion time
    epr_rows.sort(key=lambda r: (r["inv_ns"], r["file"]))
    nmr_rows.sort(key=lambda r: (r["inv_ns"], r["file"]))

    # ---- write CSVs (subset of fields) ----
    epr_fields = ["inv_ns", "file", "f0_pts_per_sample", "echo_start", "echo_end",
                  "echo_len", "echo_area_env", "echo_area_signed", "echo_peak",
                  "echo_tau_pts", "n_avgs", "points"]
    with open(OUT / "epr_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=epr_fields)
        w.writeheader()
        for r in epr_rows:
            w.writerow({k: r.get(k) for k in epr_fields})

    nmr_fields = ["inv_ns", "file", "n_indirect", "points_direct", "total_area",
                  "early_area", "late_area", "mid_area", "early_fraction",
                  "late_fraction", "fid_tau_pts_mean"]
    with open(OUT / "nmr_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=nmr_fields)
        w.writeheader()
        for r in nmr_rows:
            w.writerow({k: r.get(k) for k in nmr_fields})

    # ---- correlations / summary ----
    inv = np.array([r["inv_ns"] for r in epr_rows], dtype=float)
    epr_areas = np.array([r["echo_area_env"] for r in epr_rows])
    epr_peaks = np.array([r["echo_peak"] for r in epr_rows])
    nmr_areas = np.array([r["total_area"] for r in nmr_rows])
    nmr_early = np.array([r["early_fraction"] for r in nmr_rows])
    nmr_late = np.array([r["late_fraction"] for r in nmr_rows])
    nmr_tau = np.array([r["fid_tau_pts_mean"] for r in nmr_rows])
    corr_epr_inv = float(np.corrcoef(epr_areas, inv)[0, 1])
    corr_nmr_inv = float(np.corrcoef(nmr_areas, inv)[0, 1])
    corr_epr_nmr = float(np.corrcoef(epr_areas, nmr_areas)[0, 1])

    # robustness: fixed inner window recompute is not available from per-file JSON
    # (envelope not stored); we report the monotonic order instead.
    epr_mono = all(epr_areas[i] > epr_areas[i + 1] for i in range(len(epr_areas) - 1))

    summary = {
        "task_id": "Physics_01_dynamic_nuclear_polarization_echoes",
        "t0": "2024-06-26",
        "method": "per-file parallel sub-agent processing (process_one.py) + sequential aggregation",
        "inversion_times_ns": INV_NS,
        "epr": [ {k: r[k] for k in epr_fields} for r in epr_rows ],
        "nmr": [ {k: r[k] for k in nmr_fields} for r in nmr_rows ],
        "nmr_reference": [ {k: r[k] for k in nmr_fields} for r in ref_rows ],
        "correlations": {
            "epr_area_vs_inv": corr_epr_inv,
            "nmr_area_vs_inv": corr_nmr_inv,
            "epr_area_vs_nmr_area": corr_epr_nmr,
        },
        "epr_area_monotonic_decreasing": bool(epr_mono),
        "epr_area_2200_over_0": float(epr_areas[-1] / epr_areas[0]),
        "epr_peak_2200_over_0": float(epr_peaks[-1] / epr_peaks[0]),
        "nmr_total_area_peak_ns": int(nmr_rows[np.argmax(nmr_areas)]["inv_ns"]),
        "nmr_total_area_440_over_0": float(nmr_areas[3] / nmr_areas[0]),
        "nmr_early_fraction_constant": bool(np.allclose(nmr_early, nmr_early[0], rtol=1e-3, atol=1e-3)),
        "nmr_fid_tau_range": [float(nmr_tau.min()), float(nmr_tau.max())],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote epr_metrics.csv, nmr_metrics.csv, summary.json")
    print("corr(EPR,inv)=%.4f  corr(NMR,inv)=%.4f  corr(EPR,NMR)=%.4f" % (corr_epr_inv, corr_nmr_inv, corr_epr_nmr))
    print("EPR area monotonic decreasing:", epr_mono, "| 2200/0 =", round(epr_areas[-1]/epr_areas[0], 4))
    print("NMR peak at", summary["nmr_total_area_peak_ns"], "ns | 440/0 =", round(summary["nmr_total_area_440_over_0"], 4))
    print("NMR early-frac constant:", summary["nmr_early_fraction_constant"], "| FID tau range =", summary["nmr_fid_tau_range"])

    # ---- figures ----
    inv_arr = np.array([r["inv_ns"] for r in epr_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv_arr, epr_areas, "o-", color="C0", label="integrated echo envelope area")
    ax.plot(inv_arr, epr_peaks, "s-", color="C1", label="echo peak envelope")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("EPR echo signal")
    ax.set_title("EPR echo vs phase-inversion time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "epr_echo.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv_arr, nmr_areas, "o-", color="C2", label="integrated NMR area")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("NMR integrated area")
    ax.set_title("NMR integrated area vs phase-inversion time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "nmr_structure.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(inv_arr, epr_areas / epr_areas[0], "o-", color="C0", label="EPR echo (norm/0)")
    ax.plot(inv_arr, nmr_areas / nmr_areas[0], "s-", color="C2", label="NMR area (norm/0)")
    ax.set_xlabel("phase-inversion time (ns)")
    ax.set_ylabel("normalized signal")
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_title("Normalized EPR vs NMR signal vs phase-inversion time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "comparison.png", dpi=150)
    plt.close(fig)

    # per-indirect-point structure: peak envelope vs indirect index for each condition.
    idx = np.arange(10)
    colors = {"0": "C0", "132": "C1", "440": "C2", "2200": "C3"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in nmr_rows:
        c = colors.get(str(r["inv_ns"]), "none")
        if c == "none":
            continue
        ax.plot(idx, np.array(r["per_indirect_peak"])[:10] / 1e6,
                "o-", color=c, label=f'{r["inv_ns"]} ns')
    ax.set_xlabel("indirect index (0-9)")
    ax.set_ylabel("per-indirect peak envelope (x1e6)")
    ax.set_title("NMR per-indirect-point envelope structure by phase-inversion time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(FIG / "nmr_perindirect.png", dpi=150)
    plt.close(fig)

    print("wrote epr_echo.png, nmr_structure.png, comparison.png, nmr_perindirect.png")
    return summary


if __name__ == "__main__":
    main()
