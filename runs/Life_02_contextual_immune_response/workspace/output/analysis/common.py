"""Shared helpers for the TruthInsightBench Life_02 analysis pipeline.

Task: Life_02_contextual_immune_response
All paths resolve relative to this file so every script is runnable from any
working directory.  No network access, no external data: the four TSV files in
``data/`` are the complete evidentiary boundary of this task.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent      # output/analysis
OUTPUT = HERE.parent                        # output
ROOT = OUTPUT.parent                        # project root
DATA = ROOT / "data"
RESULTS = OUTPUT / "results"
FIGURES = OUTPUT / "figures"
LOGS = OUTPUT / "logs"

DATA_FILES = {
    "multi_condition": "multi_condition_response_replicates.tsv",
    "compartment": "compartment_response_replicates.tsv",
    "intervention": "mechanistic_intervention_replicates.tsv",
    "time_resolved": "time_resolved_response.tsv",
}

# Values in the compartment and intervention files are condition-role means of
# an underlying ratio/count readout; no transformation is applied before use.


def ensure_dirs() -> None:
    for directory in (RESULTS, FIGURES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.DataFrame:
    """Load one bundled scientific TSV by short name."""
    frame = pd.read_csv(DATA / DATA_FILES[name], sep="\t")
    frame.columns = [str(column) for column in frame.columns]
    return frame


def jsonable(obj):
    """Recursively convert numpy/pandas objects into plain JSON values."""
    if isinstance(obj, dict):
        return {str(key): jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def write_json(name: str, payload) -> Path:
    ensure_dirs()
    path = RESULTS / name
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] results/{name}")
    return path


def write_table(frame: pd.DataFrame, name: str, *, sep: str = ",") -> Path:
    ensure_dirs()
    path = RESULTS / name
    frame.to_csv(path, sep=sep, index=False, float_format="%.6g")
    print(f"[write] results/{name}  ({len(frame)} rows)")
    return path


def mean_ci(values, *, confidence: float = 0.95) -> dict:
    """Normal-theory mean with a two-sided confidence interval."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "sd": None}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "ci_low": None, "ci_high": None, "sd": None}
    sd = float(array.std(ddof=1))
    # t critical value without scipy dependency drift: use the t distribution.
    from scipy import stats as _stats

    critical = float(_stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    half = critical * sd / np.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_level": confidence,
        "t_critical": critical,
    }


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) for k raters x n items.

    ``rank_matrix`` has one row per rater and one column per item.
    """
    matrix = np.asarray(rank_matrix, dtype=float)
    k, n = matrix.shape
    rank_sums = matrix.sum(axis=0)
    mean_sum = rank_sums.mean()
    s_stat = float(((rank_sums - mean_sum) ** 2).sum())
    return 12.0 * s_stat / (k ** 2 * (n ** 3 - n))


def spearman(a, b) -> float:
    from scipy import stats as _stats

    result = _stats.spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(result.statistic)


def pformat(value) -> str:
    """Format a p-value for compact reporting."""
    if value is None:
        return "NA"
    value = float(value)
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"