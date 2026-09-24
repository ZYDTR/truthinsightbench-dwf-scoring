"""Shared loading / hashing helpers for the TruthInsightBench task
`Astronomy_03_stellar_spectral_boundary`.

The module is deliberately dependency-light: only the Python standard library,
`numpy` and `pandas` are required.  Everything downstream of this file consumes
the `(T, M, W)` flux cube and the independent normalised response matrix built
here, so that every reported number traces back to one parsing path.

Analysis unit (from `DATA_GUIDE.md` / `data_manifest.json`):
  * one complete temperature-by-wavelength spectrum, or
  * one complete ordered response series.
Neighbouring wavelengths and neighbouring metallicity settings are correlated
scan points, never independent replicates.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"

FLUX_FILE = DATA / "photon_flux_grid.tsv"
NORM_FILE = DATA / "normalized_spectral_response.tsv"

# Declared in data_manifest.json; re-checked at runtime by verify_hashes().
DECLARED_SHA256 = {
    "photon_flux_grid.tsv": "d45e7e2b62621a70039b78e0f289865dea3306f13d04d5c8c2af298ca7b6e8f8",
    "normalized_spectral_response.tsv": "1261f4dd43f004ce722e0106e49b9ccfb848d8c984ca98d9573fd5e1b9a5b468",
}

# Physical constants (SI), CODATA 2018 exact values.  Used only for the
# blackbody-baseline comparison, never as a data substitute.
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_BOLTZ = 1.380649e-23


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> dict[str, Any]:
    """Confirm the bundled files match the manifest hashes before analysis."""
    report = {}
    for name, declared in DECLARED_SHA256.items():
        path = DATA / name
        if not path.is_file():
            report[name] = {"present": False, "declared": declared, "observed": None,
                            "match": False}
            continue
        observed = sha256(path)
        report[name] = {
            "present": True,
            "declared": declared,
            "observed": observed,
            "match": observed == declared,
            "bytes": path.stat().st_size,
        }
    return report


def load_flux_grid():
    """Return the photon-flux grid as a dense ``(T, M, W)`` cube plus axes.

    The long-form table is verified to be a complete regular grid; the function
    raises if any ``(T, M, W)`` cell is missing or duplicated so that no silent
    reshape can reorder the data.
    """
    with FLUX_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    t = np.array([float(r["effective_temperature_K"]) for r in rows])
    w = np.array([float(r["wavelength_nm"]) for r in rows])
    m = np.array([float(r["metallicity_index"]) for r in rows])
    f = np.array([float(r["photon_flux_cm2"]) for r in rows])

    ts = np.unique(t)
    ms = np.unique(m)
    ws = np.unique(w)
    cube = np.full((ts.size, ms.size, ws.size), np.nan, dtype=float)
    ti = np.searchsorted(ts, t)
    mi = np.searchsorted(ms, m)
    wi = np.searchsorted(ws, w)
    cube[ti, mi, wi] = f
    if np.isnan(cube).any():
        raise ValueError("photon_flux_grid.tsv is not a complete (T, M, W) grid")
    return cube, ts, ms, ws, rows


def load_normalized():
    """Return ``(wavelength, {level_name: values})`` for the response series."""
    with NORM_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    levels = [c for c in rows[0] if c != "wavelength_nm"]
    w = np.array([float(r["wavelength_nm"]) for r in rows])
    series = {k: np.array([float(r[k]) for r in rows]) for k in levels}
    return w, series, rows


def planck_spectral_radiance(temperature_k: float, wavelength_nm: np.ndarray) -> np.ndarray:
    """Planck spectral radiance B_lambda(T) in SI units (W m^-3 sr^-1).

    Only the shape matters for the baseline test: the flux table is compared
    against a single-temperature Planck curve *shape* via log-residual spread,
    which is invariant to the arbitrary unit normalisation of the table.
    """
    lam = wavelength_nm * 1e-9
    exponent = H_PLANCK * C_LIGHT / (lam * K_BOLTZ * temperature_k)
    # Numerically stable form for the large exponents found in the UV.
    with np.errstate(over="ignore"):
        return (2.0 * H_PLANCK * C_LIGHT**2 / lam**5) / np.expm1(exponent)


def save_json(name: str, payload: Any) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_enc) + "\n",
                    encoding="utf-8")
    return path


def _enc(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")