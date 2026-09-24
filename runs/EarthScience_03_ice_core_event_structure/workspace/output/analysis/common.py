#!/usr/bin/env python3
"""common.py -- shared loading / IO helpers for the ice-core event-structure task.

Task: EarthScience_03_ice_core_event_structure.  T0 = 2019-01-01.

The workspace ``data/observations.tsv`` is the only direct evidence used by any
script in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT / "data" / "observations.tsv"
OUTPUT = PROJECT / "output"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

NUMERIC = [
    "c01_depth_top_m",
    "c02_depth_ice_snow_m",
    "c03_samp_thick_cm",
    "c04_sulfate_ug_kg",
    "c05_density_ice_g_cm_3",
    "c06_depth_w_e_m",
    "c08_age_a_ad_ce",
    "c09_age_ka_bp",
]
SIGNAL = "c07_signal"

# Recovered from the data itself: c08 + 1000 * c09 == 1950 exactly on every
# annotated row, i.e. c09 is "ka before 1950" and c08 is "CE" on the same scale.
BP_DATUM_CE = 1950.0

SEED = 20190101  # fixed seed (T0) for every resampling step


def load() -> pd.DataFrame:
    """Load the frozen record; numeric columns coerced, empty strings -> NaN."""
    raw = pd.read_csv(DATA_FILE, sep="\t", dtype=str, keep_default_na=False)
    df = raw.copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(raw[col].str.strip(), errors="coerce")
    df[SIGNAL] = raw[SIGNAL].str.strip()
    df["row"] = np.arange(len(df))
    return df


def annotated_mask(df: pd.DataFrame) -> np.ndarray:
    return df[SIGNAL].to_numpy() != ""


def blocks(n: int, k: int) -> list[np.ndarray]:
    """k contiguous, near-equal-length index blocks covering range(n)."""
    return [b for b in np.array_split(np.arange(n), k) if len(b) > 0]


def save_json(obj, name: str) -> Path:
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_encode) + "\n", encoding="utf-8")
    log(f"[artifact] {path.relative_to(OUTPUT).as_posix()}")
    return path


def save_csv(frame: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    frame.to_csv(path, index=False)
    log(f"[artifact] {path.relative_to(OUTPUT).as_posix()}")
    return path


def _encode(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def log(message) -> None:
    text = str(message)
    print(text)
    with (RESULTS / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")