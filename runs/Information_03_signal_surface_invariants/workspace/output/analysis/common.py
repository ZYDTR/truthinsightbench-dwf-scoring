"""Shared loading, hashing, statistics and JSON helpers.

TruthInsightBench task: Information_03_signal_surface_invariants
Project: Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces

Only the standard library + numpy + pandas are assumed (scipy is not installed in
this environment; rank-based statistics are implemented directly).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
RESULTS = OUTPUT / "results"
FIGURES = RESULTS / "figures"
LOGS = OUTPUT / "logs"

KEY6 = [
    "input_axis_1_index",
    "input_axis_2_index",
    "signal_bin_index",
    "output_channel_index",
    "replicate_id",
]
KEY4 = KEY6[:4]
GRID = ["input_axis_1_index", "input_axis_2_index", "signal_bin_index"]

T0 = "2022-01-01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_panels(log: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the three frozen task tables and record their hashes."""
    frames = {}
    for name in [
        "replicated_response_panel.tsv",
        "independent_response_panel.tsv",
        "independent_fit_error.tsv",
    ]:
        path = DATA / name
        df = pd.read_csv(path, sep="\t")
        frames[name] = df
        if log is not None:
            log.append(f"loaded {name}: shape={df.shape} sha256={sha256(path)}")
    rep = frames["replicated_response_panel.tsv"].copy()
    ind = frames["independent_response_panel.tsv"].copy()
    rep["ly"] = np.log(rep["response_value"])
    ind["ly"] = np.log(ind["response_value"])
    frames["rep"] = rep
    frames["ind"] = ind
    return frames


def cell_means(df: pd.DataFrame, value: str, by: list[str] = GRID + ["output_channel_index"]) -> pd.DataFrame:
    """Replicate-averaged response surface. The analysis unit is one complete grid
    condition *with* its replicate structure, so replicate means are intermediate
    quantities and replicate dispersion is analysed separately."""
    out = df.groupby(by, as_index=False).agg(**{value: (value, "mean"), "n_rep": (value, "size")})
    return out


def spearman(a, b) -> float:
    """Spearman rho without scipy."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().values
    rb = pd.Series(np.asarray(b, dtype=float)).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1])


def r2(obs, pred) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(1.0 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def separability(df: pd.DataFrame, rowfac, colfac, value: str) -> tuple[float, float]:
    """R^2 and max |residual| of the additive (separable) model
    M[row, col] = row_mean + col_mean - grand_mean, fitted on a complete table."""
    table = df.pivot_table(index=rowfac, columns=colfac, values=value)
    M = table.values
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    grand = M.mean()
    resid = M - (row + col - grand)
    return r2(M.ravel(), (row + col - grand).ravel()), float(np.abs(resid).max())


def rank1_profile(M: np.ndarray) -> dict:
    """SVD of a (cells x channels) matrix after centring each channel.
    Returns singular values, the rank-1/rank-2 R^2 and the leading loadings."""
    M = np.asarray(M, dtype=float)
    X = M - M.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    total = (s ** 2).sum()
    return {
        "singular_values": [float(v) for v in s],
        "explained_variance": [float(v ** 2 / total) for v in s],
        "rank1_r2": float(1.0 - (s[1:] ** 2).sum() / total),
        "rank2_r2": float(1.0 - (s[2:] ** 2).sum() / total),
        "loadings_comp1": [float(v) for v in Vt[0]],
    }


def block_permutation(
    matrix: np.ndarray, labels, n_perm: int = 20000, seed: int = 7
) -> dict:
    """Within-block minus between-block mean correlation of a square correlation
    matrix, with a permutation null obtained by shuffling the block labels."""
    R = np.asarray(matrix, dtype=float)
    lab = np.asarray(labels)

    def stat(l):
        w, b = [], []
        k = len(l)
        for i in range(k):
            for j in range(i + 1, k):
                (w if l[i] == l[j] else b).append(R[i, j])
        return float(np.mean(w) - np.mean(b))

    observed = stat(lab)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(lab)) for _ in range(n_perm)])
    return {
        "observed_gap": observed,
        "perm_p": float((null >= observed).mean()),
        "null_mean": float(null.mean()),
        "n_perm": n_perm,
    }


def holdout_block(
    table: np.ndarray, mask_train, mask_test, k: int = 1
) -> dict:
    """Score one held-out block with two explicit baselines.

    ``r2_vs_channel_mean`` divides the residual SS by the deviation of the held-out
    block from *its own per-channel means*. This removes the trivial between-channel
    level difference (58.9% of total variance in the replicated panel) and is the
    honest test of whether the latent structure transfers. ``r2_vs_global_mean``
    uses a single grand mean as the baseline and is therefore dominated by the
    intercept; it is reported only for completeness.

    The rank-k channel basis is estimated on the training data only; the latent
    coordinate and an intercept are refit on the held-out block.
    """
    values = np.asarray(table, dtype=float)
    train, test = values[mask_train], values[mask_test]
    mu = train.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(train - mu, full_matrices=False)
    latent = (test - mu) @ Vt[:k].T
    design = np.hstack([np.ones((len(test), 1)), latent])
    beta, *_ = np.linalg.lstsq(design, test, rcond=None)
    pred = design @ beta
    ss_res = float(((test - pred) ** 2).sum())
    ss_chan = float(((test - test.mean(axis=0, keepdims=True)) ** 2).sum())
    ss_glob = float(((test - test.mean()) ** 2).sum())
    return {
        "n_rows_held_out": int(mask_test.sum()),
        "r2_vs_channel_mean": 1.0 - ss_res / ss_chan,
        "r2_vs_global_mean": 1.0 - ss_res / ss_glob,
    }


def holdout_full_factor(table: np.ndarray, index_frame: pd.DataFrame, factor: str, k: int = 1) -> dict:
    """Hold out every complete level of one factor (full input rows / columns)."""
    out = {}
    for level in sorted(pd.unique(index_frame[factor])):
        mask = (index_frame[factor] == level).values
        out[str(level)] = holdout_block(table, ~mask, mask, k=k)
    return out


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")


def flush(log: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(log) + "\n", encoding="utf-8")