"""Shared loading and compositional-analysis utilities.

Task: Math_02_compositional_invariants
All functions operate on the frozen workspace data under ``data/`` only.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = PROJECT / "output"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_counts() -> pd.DataFrame:
    """feature_id / sample_id / relative_count long table."""
    return pd.read_csv(DATA / "anonymous_feature_counts.tsv", sep="\t")


def load_covariates() -> pd.DataFrame:
    return pd.read_csv(DATA / "anonymous_sample_covariates.tsv", sep="\t")


def load_bias_metrics() -> pd.DataFrame:
    return pd.read_csv(DATA / "bias_scenario_metrics.tsv", sep="\t")


def counts_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    """Dense sample x feature integer matrix (zeros for unobserved pairs)."""
    fc = load_counts()
    M = fc.pivot(index="sample_id", columns="feature_id", values="relative_count")
    M = M.fillna(0.0)
    return M.values.astype(np.int64), list(M.index), list(M.columns)


def bias_wide() -> pd.DataFrame:
    """scenario_id / metric_id wide table with each condition as a column."""
    bm = load_bias_metrics()
    w = bm.pivot_table(
        index=["scenario_id", "metric_id"],
        columns="condition_id",
        values="metric_value",
    ).reset_index()
    w["metric_number"] = w["metric_id"].str.extract(r"(\d+)").astype(int)
    return w


# --------------------------------------------------------------------------
# compositional utilities
# --------------------------------------------------------------------------
def closure(P: np.ndarray) -> np.ndarray:
    """Row-normalise an unconstrained count matrix to a closed composition."""
    return P / P.sum(axis=1, keepdims=True)


def clr(P: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Centred log-ratio transform with multiplicative replacement of zeros."""
    X = np.log(P + pseudocount)
    return X - X.mean(axis=1, keepdims=True)


def richness(P: np.ndarray) -> np.ndarray:
    return (P > 0).sum(axis=1)


def library_size(P: np.ndarray) -> np.ndarray:
    return P.sum(axis=1)


def rarefy(P: np.ndarray, depth: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial thinning to a common depth (equal-effort richness control)."""
    q = P / P.sum(axis=1, keepdims=True)
    return np.array([rng.multinomial(depth, qi) for qi in q])


def exact_fraction(x: float, cap: int = 5000) -> tuple[int | None, int | None]:
    """Smallest exact numerator/denominator (tolerance 1e-12) or (None, None)."""
    for d in range(1, cap + 1):
        n = round(x * d)
        if abs(n / d - x) < 1e-12:
            return n, d
    return None, None


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman partial correlation of x,y controlling z (rank-linear residualisation)."""
    from scipy import stats

    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


def cliffs_delta_paired(a, b) -> float:
    d = np.sign(np.asarray(a) - np.asarray(b))
    return float((d > 0).mean() - (d < 0).mean())


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def integrity_receipt() -> dict:
    manifest = json.loads((PROJECT / "data_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = DATA / row["path"]
        rows.append(
            {
                "path": row["path"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "matches_manifest": sha256(path) == row["sha256"],
            }
        )
    return {"files": rows, "all_match": all(r["matches_manifest"] for r in rows)}