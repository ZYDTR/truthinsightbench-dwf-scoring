"""Shared analysis utilities for the time-resolved gravity residual task.

Task: Astronomy_02_gravity_spectral_structure (T0 = 2022-01-01)
Data: ../data/*.tsv (read-only task dataset; workspace is the only evidence).

All statistics are computed on the declared independent analysis unit:
one complete time-resolved pass (n = 22) or one complete candidate spectral
parameter combination. Adjacent time samples are NOT treated as replicates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = PROJECT / "output"

MODELS = {
    "A": "residual_model_A_mm_s",
    "B": "residual_model_B_mm_s",
    "C": "residual_model_C_mm_s",
}

# Candidate band derived from the evaluated family's peak-frequency axis.
CAND_BAND = (860.0, 1310.0)


def load_traces() -> pd.DataFrame:
    return pd.read_csv(DATA / "pass_residual_traces.tsv", sep="\t")


def load_candidates() -> pd.DataFrame:
    return pd.read_csv(DATA / "candidate_spectrum_comparison.tsv", sep="\t")


def load_ranked_grid() -> pd.DataFrame:
    df = pd.read_csv(DATA / "ranked_full_parameter_grid.tsv", sep="\t")
    return df.rename(
        columns={
            "delta_information_criterion": "ic",
            "frequency_width_microhz": "w",
            "peak_frequency_microhz": "f",
            "max_velocity_cm_s": "vmax",
            "min_velocity_cm_s": "vmin",
        }
    )


def load_modes() -> pd.DataFrame:
    return pd.read_csv(DATA / "mode_frequency_grid.tsv", sep="\t")


def load_uniform() -> pd.DataFrame:
    return pd.read_csv(DATA / "uniform_alternative_comparison.tsv", sep="\t")


def per_pass_rms(traces: pd.DataFrame) -> pd.DataFrame:
    """Whole-pass RMS residual per model. One row per pass (the analysis unit)."""
    g = traces.groupby("pass_id")
    out = pd.DataFrame({tag: g[col].apply(lambda v: float(np.sqrt(np.mean(v**2))))
                        for tag, col in MODELS.items()})
    return out


def freq_grid(lo=200.0, hi=3000.0, step=1.0):
    """Frequency grid in microHz and Hz."""
    f_uhz = np.arange(lo, hi, step)
    return f_uhz, f_uhz * 1e-6


def ls_power(t_min: np.ndarray, y: np.ndarray, f_hz: np.ndarray) -> np.ndarray:
    """Normalized Lomb-Scargle power. t_min in minutes -> converted to seconds."""
    t_sec = np.asarray(t_min, float) * 60.0
    y = np.asarray(y, float)
    return lombscargle(t_sec, y - y.mean(), 2 * np.pi * f_hz, normalize=True)


def ar1_params(y: np.ndarray):
    y = np.asarray(y, float)
    y = y - y.mean()
    den = float(np.dot(y[:-1], y[:-1]))
    a = float(np.clip(np.dot(y[:-1], y[1:]) / den, -0.99, 0.99)) if den > 0 else 0.0
    return a, float(np.std(y))


def ar1_surrogate(n: int, a: float, s: float, rng) -> np.ndarray:
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-9)), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def band_power_percentile(t_min, y, f_uhz, f_hz, band, n_surrogate=200, rng=None):
    """Observed band-mean power and its percentile against an AR(1) red-noise null."""
    if rng is None:
        rng = np.random.default_rng(20220101)
    m = (f_uhz >= band[0]) & (f_uhz <= band[1])
    obs = float(ls_power(t_min, y, f_hz)[m].mean())
    a, s = ar1_params(y)
    n = len(y)
    null = np.array([
        float(ls_power(t_min, ar1_surrogate(n, a, s, rng), f_hz)[m].mean())
        for _ in range(n_surrogate)
    ])
    return obs, float((null < obs).mean()), null


def min_sinusoid_amplitude(t_min, y, f_uhz, detrend=True):
    """Least-squares sinusoid amplitude at one frequency (mm/s)."""
    t_sec = np.asarray(t_min, float) * 60.0
    y = np.asarray(y, float)
    off = 0.0
    if detrend:
        p = np.polyfit(t_sec - t_sec.mean(), y, 1)
        off = np.polyval(p, t_sec - t_sec.mean())
        y = y - off
    w = 2 * np.pi * f_uhz * 1e-6
    X = np.column_stack([np.cos(w * t_sec), np.sin(w * t_sec)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.hypot(beta[0], beta[1]))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")