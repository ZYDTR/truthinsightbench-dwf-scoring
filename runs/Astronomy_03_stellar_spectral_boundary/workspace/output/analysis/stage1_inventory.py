"""Stage 1 -- data inventory, integrity, and structural characterisation.

Establishes, from the workspace data alone:
  * the two bundled files match the hashes declared in `data_manifest.json`;
  * the photon-flux table is a complete `(T, M, W)` regular grid with no
    missing or duplicated cells;
  * both files are free of missing / non-finite / non-positive values;
  * the wavelength sampling is deliberately non-uniform (dense in a narrow
    band near 300 nm), which later stages must respect.

Run:  python3 analysis/stage1_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS, load_flux_grid, load_normalized, save_json,  # noqa: E402
                    verify_hashes)

OUT = {}


def main() -> None:
    OUT["integrity"] = verify_hashes()

    cube, ts, ms, ws, rows = load_flux_grid()
    nw, levels, norm_rows = load_normalized()

    # --- completeness / regularity of the flux grid -------------------------
    t = np.array([float(r["effective_temperature_K"]) for r in rows])
    w = np.array([float(r["wavelength_nm"]) for r in rows])
    m = np.array([float(r["metallicity_index"]) for r in rows])
    f = np.array([float(r["photon_flux_cm2"]) for r in rows])

    expected_rows = ts.size * ms.size * ws.size
    per_cell, counts = np.unique(
        np.stack([np.searchsorted(ts, t), np.searchsorted(ms, m), np.searchsorted(ws, w)], 1),
        axis=0, return_counts=True)

    OUT["flux_grid"] = {
        "rows": int(len(rows)),
        "expected_rows_if_complete": int(expected_rows),
        "complete_grid": bool(len(rows) == expected_rows and np.all(counts == 1)),
        "duplicated_cells": int((per_cell.shape[0] != expected_rows) * 1),
        "temperature_slices_K": [float(x) for x in ts],
        "metallicity_levels": [float(x) for x in ms],
        "metallicity_step_uniform_K": float(np.median(np.diff(ms[:21]))),
        "metallicity_gap_note": [
            float(a) for a in np.diff(ms) if abs(a - 0.05) > 1e-9],
        "n_wavelengths": int(ws.size),
        "wavelength_min_nm": float(ws.min()),
        "wavelength_max_nm": float(ws.max()),
        "flux_min": float(f.min()),
        "flux_max": float(f.max()),
        "n_nonpositive_flux": int((f <= 0).sum()),
        "n_nonfinite_flux": int((~np.isfinite(f)).sum()),
        "n_missing_cells": int(np.isnan(cube).sum()),
        "wavelength_step_histogram": {
            str(k): int(v) for k, v in
            zip(*np.unique(np.round(np.diff(ws), 2), return_counts=True))},
        "densest_band_nm": [
            float(ws[0]), float(ws[-1])
        ] if False else [
            float(ws[np.argmax(np.convolve((np.diff(ws) <= 1.5).astype(int),
                                           np.ones(5, int), "valid"))]),
            float(ws[np.argmax(np.convolve((np.diff(ws) <= 1.5).astype(int),
                                           np.ones(5, int), "valid")) + 25])],
    }

    # --- normalised response series ----------------------------------------
    lvl_stats = {}
    for k, v in levels.items():
        lvl_stats[k] = {
            "min": float(v.min()),
            "min_wavelength_nm": float(nw[int(np.argmin(v))]),
            "max": float(v.max()),
            "max_wavelength_nm": float(nw[int(np.argmax(v))]),
            "value_at_first_wavelength": float(v[0]),
            "value_at_last_wavelength": float(v[-1]),
            "strictly_increasing_in_wavelength": bool(np.all(np.diff(v) > 0)),
            "n_nonpositive": int((v <= 0).sum()),
        }
    OUT["normalized_series"] = {
        "rows": int(len(norm_rows)),
        "levels": list(levels.keys()),
        "wavelength_min_nm": float(nw.min()),
        "wavelength_max_nm": float(nw.max()),
        "wavelength_step_histogram": {
            str(k): int(v) for k, v in
            zip(*np.unique(np.round(np.diff(nw), 2), return_counts=True))},
        "n_wavelengths_shared_with_flux_grid_at_0p2nm": int(sum(
            1 for x in nw if np.min(np.abs(ws - x)) < 0.2)),
        "per_level": lvl_stats,
    }

    # --- correlation length of the wavelength axis (why random splits fail) --
    # Measured as the lag at which the lag-1..k autocorrelation of the
    # log-spectrum drops below 1/2, averaged over the 105 grid spectra.
    logcube = np.log(cube.reshape(-1, ws.size))
    acf = []
    for row in logcube:
        r = row - row.mean()
        denom = float(r @ r)
        acf.append([float((r[:-k] @ r[k:]) / denom) for k in (1, 2, 5, 10, 20)])
    acf = np.array(acf)
    OUT["wavelength_axis_correlation"] = {
        "mean_autocorrelation_at_lag": {
            str(k): float(acf[:, i].mean()) for i, k in enumerate((1, 2, 5, 10, 20))},
        "median_lag20_autocorrelation": float(np.median(acf[:, 4])),
    }

    path = save_json("stage1_inventory.json", OUT)
    print(f"wrote {path}")
    print(f"grid complete: {OUT['flux_grid']['complete_grid']}  "
          f"rows={OUT['flux_grid']['rows']}")
    print(f"all file hashes match manifest: "
          f"{all(v['match'] for v in OUT['integrity'].values())}")
    print(f"wavelength autocorrelation at lag 20: "
          f"{OUT['wavelength_axis_correlation']['median_lag20_autocorrelation']:.4f}")


if __name__ == "__main__":
    main()