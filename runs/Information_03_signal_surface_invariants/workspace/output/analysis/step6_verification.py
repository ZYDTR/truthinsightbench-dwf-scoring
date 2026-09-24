#!/usr/bin/env python3
"""step6_verification.py -- adversarial recomputation (TruthInsightBench Information_03).

Purpose: try to REFUTE the headline numbers A-E using code that is independent of
output/analysis/common.py.  This module imports nothing from the project: cell
means, SVD rank R2, OLS slopes, Pearson, average-rank Spearman, polynomial trend
fits and permutation nulls are all re-implemented below from plain numpy/pandas.

Only the frozen task tables in data/ are read (no external data, no merging of
external sources).  Any section that raises is caught, its real traceback is kept
in the JSON under "errors", and the remaining sections still run.

Outputs:
  output/results/t21_verification.json
  output/logs/step6.log
"""

from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/data/user/conversations/conv-7cd614b8-05bc-4b7d-9604-8b94ad1eeb44/project")
DATA = PROJECT / "data"
OUTPUT = PROJECT / "output"
RESULTS = OUTPUT / "results"
LOGS = OUTPUT / "logs"

T0 = "2022-01-01"

KEY4 = ["input_axis_1_index", "input_axis_2_index", "signal_bin_index", "output_channel_index"]
KEY5 = KEY4 + ["replicate_id"]

LOG: list[str] = []
ERRORS: list[dict] = []


def log(msg: str) -> None:
    LOG.append(msg)
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# statistics from scratch
# --------------------------------------------------------------------------- #
def pearson(x, y) -> float:
    """Pearson product-moment correlation written out longhand."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = math_sqrt((dx * dx).sum() * (dy * dy).sum())
    if denom == 0.0:
        return float("nan")
    return float((dx * dy).sum() / denom)


def math_sqrt(v: float) -> float:
    # np.sqrt is fine but keep a single square-root path for clarity
    return float(np.sqrt(v))


def average_rank(x) -> np.ndarray:
    """Average ranks (ties share the mean rank), implemented explicitly."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    return pearson(average_rank(x), average_rank(y))


def slope_ols_simple(x, y) -> float:
    """OLS slope of y on a single regressor x, longhand."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    return float((dx * (y - y.mean())).sum() / (dx * dx).sum())


def slope_ols_multi(design: np.ndarray, y) -> np.ndarray:
    """Least-squares coefficients via normal equations (no scipy)."""
    beta, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
    return beta


def poly_design(x, deg: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.vstack([x ** k for k in range(deg + 1)]).T  # [1, x, x^2, ...]


def poly_fit(x, y, deg: int) -> np.ndarray:
    return slope_ols_multi(poly_design(x, deg), y)


def poly_eval(coef, x) -> float:
    """Horner evaluation."""
    out = 0.0
    for c in np.asarray(coef, dtype=float)[::-1]:
        out = out * float(x) + float(c)
    return float(out)


def svd_rank_r2(M: np.ndarray, centre_columns: bool = True) -> dict:
    """Rank-k R^2 of a (cells x channels) matrix.

    R^2 is computed twice with independent formulas: (a) from the singular-value
    spectrum, (b) from the explicit reconstruction residual.
    """
    M = np.asarray(M, dtype=float)
    X = M - M.mean(axis=0, keepdims=True) if centre_columns else M
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    total_ss = float((X ** 2).sum())
    spec = [float(v ** 2 / total_ss) for v in s]
    rank1_spec = float(1.0 - (s[1:] ** 2).sum() / total_ss)
    rank2_spec = float(1.0 - (s[2:] ** 2).sum() / total_ss)
    # explicit reconstruction
    recon1 = np.outer(U[:, 0] * s[0], Vt[0])
    recon2 = recon1 + np.outer(U[:, 1] * s[1], Vt[1])
    rank1_res = float(1.0 - ((X - recon1) ** 2).sum() / total_ss)
    rank2_res = float(1.0 - ((X - recon2) ** 2).sum() / total_ss)
    return {
        "shape": [int(M.shape[0]), int(M.shape[1])],
        "singular_values": [float(v) for v in s],
        "squared_share_by_spectrum": spec,
        "rank1_r2_spectrum": rank1_spec,
        "rank2_r2_spectrum": rank2_spec,
        "rank1_r2_reconstruction": rank1_res,
        "rank2_r2_reconstruction": rank2_res,
        "formula_agreement": bool(abs(rank1_spec - rank1_res) < 1e-9 and abs(rank2_spec - rank2_res) < 1e-9),
    }


def perm_null(x, y, n_perm: int, seed: int) -> dict:
    """Two-sided permutation p-value for pearson(x, y) by shuffling y.

    Permuting y leaves its mean and sd untouched, so the permuted correlation is
    (x . y_perm - n*mx*my) / (n*sx*sy); this is evaluated in chunks with matrix
    products instead of one np.corrcoef call per permutation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    obs = pearson(x, y)
    mx, my = x.mean(), y.mean()
    sx = np.sqrt(((x - mx) ** 2).sum())
    sy = np.sqrt(((y - my) ** 2).sum())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        yp = y[rng.permutation(n)]
        null[i] = (yp @ x - n * mx * my) / (sx * sy)
    p = float((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1)
    return {
        "observed": obs,
        "perm_p_two_sided": p,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "n_perm": int(n_perm),
    }


def verdict(claim: float, observed: float, tol: float) -> dict:
    diff = observed - claim
    ok = bool(abs(diff) <= tol)
    return {
        "claim": float(claim),
        "observed": float(observed),
        "difference": float(diff),
        "tolerance": float(tol),
        "verdict": "MATCH" if ok else "DISCREPANCY",
    }


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_tables() -> dict:
    out = {}
    for name in [
        "replicated_response_panel.tsv",
        "independent_response_panel.tsv",
        "independent_fit_error.tsv",
    ]:
        path = DATA / name
        df = pd.read_csv(path, sep="\t")
        out[name] = df
        log(f"loaded {name}: shape={df.shape} sha256={sha256(path)}")
    rep = out["replicated_response_panel.tsv"]
    ind = out["independent_response_panel.tsv"]
    # completeness check (independent of common.py)
    for tag, df, expect in [("rep", rep, {"signal_bin_index": 8, "output_channel_index": 7}),
                            ("ind", ind, {"signal_bin_index": 7, "output_channel_index": 4})]:
        counts = df.groupby(KEY5).size()
        log(
            f"{tag}: rows={len(df)} cells={df.groupby(KEY4).ngroups} "
            f"replicate counts min={int(counts.min())} max={int(counts.max())} "
            f"bins={df['signal_bin_index'].nunique()} chans={df['output_channel_index'].nunique()}"
        )
    rep = rep.copy()
    ind = ind.copy()
    rep["ly"] = np.log(rep["response_value"].values)
    ind["ly"] = np.log(ind["response_value"].values)
    out["rep"] = rep
    out["ind"] = ind
    return out


# --------------------------------------------------------------------------- #
# A: rank-1 R^2 of the 49x7 log surface
# --------------------------------------------------------------------------- #
def check_A(rep: pd.DataFrame) -> dict:
    cell = rep.groupby(["input_axis_1_index", "input_axis_2_index", "output_channel_index"],
                       as_index=False)["ly"].mean()
    M = cell.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                         columns="output_channel_index", values="ly").values
    primary = svd_rank_r2(M, centre_columns=True)
    variants = {}
    variants["uncentred_columns"] = svd_rank_r2(M, centre_columns=False)
    # 392x7 grid (a1,a2,bin) for context only
    cell3 = rep.groupby(["input_axis_1_index", "input_axis_2_index", "signal_bin_index",
                         "output_channel_index"], as_index=False)["ly"].mean()
    M3 = cell3.pivot_table(index=["input_axis_1_index", "input_axis_2_index", "signal_bin_index"],
                           columns="output_channel_index", values="ly").values
    variants["grid_a1a2bin_392x7"] = svd_rank_r2(M3, centre_columns=True)
    # per-channel z-scored variant (a scale perturbation of the same claim)
    Mz = (M - M.mean(axis=0, keepdims=True)) / M.std(axis=0, keepdims=True)
    variants["per_channel_zscore"] = svd_rank_r2(Mz, centre_columns=True)
    log(
        "A: 49x7 log rank1 R2=%.6f rank2 R2=%.6f (spectrum vs reconstruction agree=%s)"
        % (primary["rank1_r2_spectrum"], primary["rank2_r2_spectrum"], primary["formula_agreement"])
    )
    log(
        "A variants: uncentred rank1=%.4f ; 392x7 rank1=%.4f rank2=%.4f ; per-channel z rank1=%.4f"
        % (variants["uncentred_columns"]["rank1_r2_spectrum"],
           variants["grid_a1a2bin_392x7"]["rank1_r2_spectrum"],
           variants["grid_a1a2bin_392x7"]["rank2_r2_spectrum"],
           variants["per_channel_zscore"]["rank1_r2_spectrum"])
    )
    return {
        "definition": (
            "replicate-averaged natural-log response; 49 (a1,a2) rows x 7 channels; columns "
            "centred before SVD; rank-k R2 = 1 - sum_{i>=k} s_i^2 / sum_i s_i^2"
        ),
        "primary": primary,
        "rank1": verdict(0.9550, primary["rank1_r2_spectrum"], 5e-5),
        "rank2": verdict(0.9902, primary["rank2_r2_spectrum"], 5e-5),
        "variants_not_the_claim": variants,
    }


# --------------------------------------------------------------------------- #
# B: chan6/chan0 mean OLS slope ratio of log response on a1
# --------------------------------------------------------------------------- #
def check_B(rep: pd.DataFrame) -> dict:
    per_channel = {}
    for ch in range(7):
        sub = rep[rep["output_channel_index"] == ch]
        P = sub.pivot_table(index="input_axis_2_index", columns="input_axis_1_index",
                            values="ly").values  # rows a2, cols a1
        x = np.arange(P.shape[1], dtype=float)
        slopes = [slope_ols_simple(x, P[a, :]) for a in range(P.shape[0])]
        per_channel[ch] = float(np.mean(slopes))

    # adversarial variants of the same estimand
    variants = {}
    pooled = {}
    for ch in range(7):
        sub = rep[rep["output_channel_index"] == ch]
        x = sub["input_axis_1_index"].values.astype(float)
        y = sub["ly"].values
        pooled[ch] = slope_ols_simple(x, y)
    variants["pooled_cells_slope_ratio_ch6_over_ch0"] = float(pooled[6] / pooled[0])
    # both-axis model: ly ~ 1 + a1 + a2 + bin per channel
    both = {}
    for ch in range(7):
        sub = rep[rep["output_channel_index"] == ch]
        D = np.column_stack([
            np.ones(len(sub)),
            sub["input_axis_1_index"].values.astype(float),
            sub["input_axis_2_index"].values.astype(float),
            sub["signal_bin_index"].values.astype(float),
        ])
        both[ch] = float(slope_ols_multi(D, sub["ly"].values)[1])
    variants["a1_slope_controlling_a2_and_bin_ratio_ch6_over_ch0"] = float(both[6] / both[0])
    # per-a2 slope ratios (distribution, not just the mean)
    ratios_by_a2 = []
    for a2 in range(7):
        s6 = slope_ols_simple(np.arange(7.0),
                              rep[(rep["output_channel_index"] == 6) & (rep["input_axis_2_index"] == a2)]
                              .groupby("input_axis_1_index")["ly"].mean().sort_index().values)
        s0 = slope_ols_simple(np.arange(7.0),
                              rep[(rep["output_channel_index"] == 0) & (rep["input_axis_2_index"] == a2)]
                              .groupby("input_axis_1_index")["ly"].mean().sort_index().values)
        ratios_by_a2.append(float(s6 / s0))
    variants["ratio_per_a2_row"] = ratios_by_a2
    variants["ratio_per_a2_row_min"] = float(np.min(ratios_by_a2))
    variants["ratio_per_a2_row_max"] = float(np.max(ratios_by_a2))

    ratio = per_channel[6] / per_channel[0]
    log("B: mean a1 slope (logged, by-a2 averaged) ch0=%.6f ch6=%.6f ratio=%.4f"
        % (per_channel[0], per_channel[6], ratio))
    log("B variants: pooled-cell ratio=%.4f ; controlling a2+bin ratio=%.4f ; per-a2 row range=[%.3f, %.3f]"
        % (variants["pooled_cells_slope_ratio_ch6_over_ch0"],
           variants["a1_slope_controlling_a2_and_bin_ratio_ch6_over_ch0"],
           variants["ratio_per_a2_row_min"], variants["ratio_per_a2_row_max"]))
    return {
        "definition": (
            "per channel: OLS slope of log(response) on input_axis_1_index fitted within each "
            "a2 row of the replicate-averaged 7x7 surface, then averaged over the 7 a2 rows; "
            "ratio = slope(chan6)/slope(chan0)"
        ),
        "slope_a1_per_channel": {str(k): v for k, v in per_channel.items()},
        "primary": verdict(29.2, ratio, 0.05),
        "variants_not_the_claim": variants,
    }


# --------------------------------------------------------------------------- #
# C: cross-panel cell-level correspondence on the 1372 shared cells
# --------------------------------------------------------------------------- #
def shared_cells(rep: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    r = rep.groupby(KEY4, as_index=False).agg(ly_rep=("ly", "mean"), y_rep=("response_value", "mean"))
    i = ind.groupby(KEY4, as_index=False).agg(ly_ind=("ly", "mean"), y_ind=("response_value", "mean"))
    m = r.merge(i, on=KEY4, how="inner").sort_values(KEY4).reset_index(drop=True)
    return m


def check_C(rep: pd.DataFrame, ind: pd.DataFrame) -> dict:
    m = shared_cells(rep, ind)
    r_log = pearson(m["ly_rep"], m["ly_ind"])
    r_lin = pearson(m["y_rep"], m["y_ind"])
    rho = spearman(m["ly_rep"], m["ly_ind"])
    n_perm = 20000
    nulltest = perm_null(m["ly_rep"].values, m["ly_ind"].values, n_perm, 20220101)
    p = nulltest["perm_p_two_sided"]
    log("C: shared cells=%d ; pearson(log)=%+.6f pearson(linear)=%+.6f spearman(log)=%+.6f perm p=%.4f"
        % (len(m), r_log, r_lin, rho, p))
    log("C scale context: rep log sd=%.4f ind log sd=%.4f ; rep log mean=%.4f ind log mean=%.4f"
        % (m["ly_rep"].std(ddof=1), m["ly_ind"].std(ddof=1), m["ly_rep"].mean(), m["ly_ind"].mean()))
    return {
        "definition": "replicate-averaged log response matched on (a1,a2,bin,chan), 7x7x7x4=1372 cells",
        "n_cells": int(len(m)),
        "pearson_log": r_log,
        "pearson_linear": r_lin,
        "spearman_log": rho,
        "perm_p_two_sided": p,
        "perm_null_mean": nulltest["null_mean"],
        "perm_null_sd": nulltest["null_sd"],
        "perm_n": n_perm,
        "log_sd_rep": float(m["ly_rep"].std(ddof=1)),
        "log_sd_ind": float(m["ly_ind"].std(ddof=1)),
        "primary": verdict(0.0168, r_log, 5e-5),
    }


# --------------------------------------------------------------------------- #
# D: within-cell coefficient-of-variation sharing
# --------------------------------------------------------------------------- #
def panel_cell_cv(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEY4)["response_value"]
    out = pd.DataFrame({
        "sd": g.std(ddof=1),
        "mean": g.mean(),
    }).reset_index()
    out["cv"] = out["sd"] / out["mean"]
    # log-space dispersion for the adversarial variant
    gl = df.groupby(KEY4)["ly"]
    sdl = gl.std(ddof=1).rename("sd_log").reset_index()
    out = out.merge(sdl, on=KEY4)
    return out


def check_D(rep: pd.DataFrame, ind: pd.DataFrame) -> dict:
    a = panel_cell_cv(rep)
    b = panel_cell_cv(ind)
    m = a.merge(b, on=KEY4, suffixes=("_rep", "_ind")).sort_values(KEY4).reset_index(drop=True)
    r_cv = pearson(m["cv_rep"], m["cv_ind"])
    n_perm = 20000
    nulltest = perm_null(m["cv_rep"].values, m["cv_ind"].values, n_perm, 11)
    p = nulltest["perm_p_two_sided"]
    log("D: shared cells=%d ; pearson(CV)=%.6f perm p=%.5f ; mean CV rep=%.4f ind=%.4f"
        % (len(m), r_cv, p, m["cv_rep"].mean(), m["cv_ind"].mean()))
    return {
        "definition": "per-cell sd/mean of the 3 replicates, linear space, matched on the 1372 shared cells",
        "n_cells": int(len(m)),
        "pearson_cv": r_cv,
        "perm_p_two_sided": p,
        "perm_null_mean": nulltest["null_mean"],
        "perm_null_sd": nulltest["null_sd"],
        "perm_n": n_perm,
        "mean_cv_rep": float(m["cv_rep"].mean()),
        "mean_cv_ind": float(m["cv_ind"].mean()),
        "frame": m,
        "primary": verdict(0.3822, r_cv, 5e-5),
    }


# --------------------------------------------------------------------------- #
# E: signal_bin 0 excess over a quadratic trend fitted to bins 1-7
# --------------------------------------------------------------------------- #
def bin_means(rep: pd.DataFrame) -> pd.Series:
    return rep.groupby("signal_bin_index")["ly"].mean().sort_index()


def check_E(rep: pd.DataFrame) -> dict:
    bm = bin_means(rep)
    xs = np.arange(1, 8, dtype=float)
    co = poly_fit(xs, bm.loc[1:].values, 2)
    excess = float(bm.loc[0] - poly_eval(co, 0.0))
    log("E: bin0 mean=%.6f ; quadratic(1-7) prediction at bin0=%.6f ; excess=%+.6f (x%.4f)"
        % (bm.loc[0], poly_eval(co, 0.0), excess, np.exp(excess)))
    log("E bin means: " + ", ".join("bin%d=%.4f" % (k, v) for k, v in bm.items()))
    return {
        "definition": "mean log response per signal_bin over all a1,a2,chan,replicates; degree-2 fit on bins 1-7 evaluated at bin 0",
        "bin_means": {str(k): float(v) for k, v in bm.items()},
        "quadratic_coef_bins1_7": [float(c) for c in co],
        "predicted_bin0": poly_eval(co, 0.0),
        "excess": excess,
        "excess_ratio_exp": float(np.exp(excess)),
        "primary": verdict(0.9482, excess, 5e-5),
    }


# --------------------------------------------------------------------------- #
# R1: is the near-zero cross-panel correspondence a scale artefact?
# --------------------------------------------------------------------------- #
def demean_axes(values, keys_frame: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Subtract the group mean of `values` within each combination of `keys`.

    Implemented with explicit integer codes and boolean masks so it does not lean
    on groupby-transform semantics.
    """
    vals = np.asarray(values, dtype=float).ravel()
    codes = np.zeros(vals.size, dtype=np.int64)
    nlev = 1
    for k in keys:
        c = pd.factorize(keys_frame[k].values)[0].astype(np.int64)
        codes = codes * (c.max() + 1) + c
        nlev *= int(c.max() + 1)
    out = vals.copy()
    for c in range(nlev):
        mask = codes == c
        if mask.any():
            out[mask] -= vals[mask].mean()
    return out


def check_R1(rep: pd.DataFrame, ind: pd.DataFrame, m: pd.DataFrame) -> dict:
    out = {}
    n_perm = 20000

    def run(tag, xr, xi, seed):
        res = perm_null(xr, xi, n_perm, seed)
        res["material_positive"] = bool(res["observed"] >= 0.10 and res["perm_p_two_sided"] < 0.001)
        log("R1 %s: pearson=%+.6f perm p=%.5f (null sd=%.4f) material_positive=%s"
            % (tag, res["observed"], res["perm_p_two_sided"], res["null_sd"], res["material_positive"]))
        return res

    # (i) per-channel centring, both panels, on the shared cells
    xr = (m["ly_rep"] - m.groupby("output_channel_index")["ly_rep"].transform("mean")).values
    xi = (m["ly_ind"] - m.groupby("output_channel_index")["ly_ind"].transform("mean")).values
    out["i_per_channel_centred"] = run("i(per-channel centred)", xr, xi, 101)
    # alternative per-channel centring using each panel's own full cell set
    r_full = rep.groupby(KEY4, as_index=False)["ly"].mean()
    i_full = ind.groupby(KEY4, as_index=False)["ly"].mean()
    r_full["ly_c"] = r_full["ly"] - r_full.groupby("output_channel_index")["ly"].transform("mean")
    i_full["ly_c"] = i_full["ly"] - i_full.groupby("output_channel_index")["ly"].transform("mean")
    m2 = r_full[KEY4 + ["ly_c"]].merge(i_full[KEY4 + ["ly_c"]], on=KEY4, suffixes=("_rep", "_ind"))
    out["i_alt_per_channel_centred_own_panel_means"] = {
        "pearson": pearson(m2["ly_c_rep"], m2["ly_c_ind"]), "n_cells": int(len(m2))}

    # (ii) variance standardising both panels (global z across the shared cells)
    zr = ((m["ly_rep"] - m["ly_rep"].mean()) / m["ly_rep"].std(ddof=1)).values
    zi = ((m["ly_ind"] - m["ly_ind"].mean()) / m["ly_ind"].std(ddof=1)).values
    out["ii_variance_standardised_global"] = run("ii(global z)", zr, zi, 102)
    zr_c = xr / m.groupby("output_channel_index")["ly_rep"].transform("std").values
    zi_c = xi / m.groupby("output_channel_index")["ly_ind"].transform("std").values
    out["ii_alt_variance_standardised_within_channel"] = run("ii(channel z)", zr_c, zi_c, 103)

    # (iii) rank transform of both panels
    out["iii_rank_transformed_pooled"] = run("iii(pooled ranks)", average_rank(m["ly_rep"]),
                                             average_rank(m["ly_ind"]), 104)
    rr = m.groupby("output_channel_index")["ly_rep"].transform(lambda s: average_rank(s.values))
    ri = m.groupby("output_channel_index")["ly_ind"].transform(lambda s: average_rank(s.values))
    out["iii_alt_rank_within_channel"] = run("iii(within-channel ranks)", rr.values, ri.values, 105)

    # (iv) replicate-level deviations from each panel's own cell mean
    r = rep.copy()
    i = ind.copy()
    r["cell_mean"] = r.groupby(KEY4)["ly"].transform("mean")
    i["cell_mean"] = i.groupby(KEY4)["ly"].transform("mean")
    r["dev"] = r["ly"] - r["cell_mean"]
    i["dev"] = i["ly"] - i["cell_mean"]
    md = r[KEY5 + ["dev"]].merge(i[KEY5 + ["dev"]], on=KEY5, suffixes=("_rep", "_ind"))
    out["iv_replicate_deviations"] = run("iv(replicate deviations)", md["dev_rep"], md["dev_ind"], 106)
    out["iv_replicate_deviations"]["n_rows"] = int(len(md))

    # (v) decisive variant: remove the independent panel's (a1,a2) marginal means
    # from BOTH panels, then re-standardise; a common (a1,a2) trend can no longer
    # be the source of any positive correlation.
    work = m.copy()
    AX2 = ["input_axis_1_index", "input_axis_2_index"]
    for col, tag in [("ly_rep", "v_ind_panel_a1a2_structure_removed"),
                     ("y_rep", "vi_linear_control_same_treatment")]:
        other = "ly_ind" if col == "ly_rep" else "y_ind"
        vals_r = demean_axes(work[col].values, work, AX2)
        vals_i = demean_axes(work[other].values, work, AX2)
        vals_r = demean_axes(vals_r, work, ["output_channel_index"])
        vals_i = demean_axes(vals_i, work, ["output_channel_index"])
        vals_r = (vals_r - vals_r.mean()) / vals_r.std(ddof=1)
        vals_i = (vals_i - vals_i.mean()) / vals_i.std(ddof=1)
        out[tag] = run(("v(ind a1a2 removed, both z)" if col == "ly_rep"
                        else "vi(linear control, same treatment)"), vals_r, vals_i,
                       107 if col == "ly_rep" else 108)

    # (vii) locate the small positive signal: is it the pooled (a1,a2) structure
    # rather than cell-level agreement?  49-point marginal correspondence, then a
    # 196-point (a1,a2,chan) correspondence, both in log space.
    marg_r = m.groupby(["input_axis_1_index", "input_axis_2_index"])["ly_rep"].mean()
    marg_i = m.groupby(["input_axis_1_index", "input_axis_2_index"])["ly_ind"].mean()
    out["vii_a1a2_marginal_49_points"] = {
        "pearson": pearson(marg_r.values, marg_i.values),
        "spearman": spearman(marg_r.values, marg_i.values),
        "n_points": int(len(marg_r)),
    }
    c_r = m.groupby(["input_axis_1_index", "input_axis_2_index", "output_channel_index"])["ly_rep"].mean()
    c_i = m.groupby(["input_axis_1_index", "input_axis_2_index", "output_channel_index"])["ly_ind"].mean()
    out["vii_a1a2_chan_cell_196_points"] = {
        "pearson_pooled_over_bins": pearson(c_r.values, c_i.values),
        "n_points": int(len(c_r)),
    }
    log("R1 vii: (a1,a2) marginal 49-point pearson=%+.4f ; (a1,a2,chan) 196-point pearson=%+.4f"
        % (out["vii_a1a2_marginal_49_points"]["pearson"],
           out["vii_a1a2_chan_cell_196_points"]["pearson_pooled_over_bins"]))

    substantial = {k: v for k, v in out.items()
                   if isinstance(v, dict) and v.get("material_positive")}
    out["_summary"] = {
        "variants_checked": [k for k in out if not k.startswith("_")],
        "any_material_positive": bool(substantial),
        "material_variants": list(substantial),
    }
    log("R1 summary: any variant with substantial positive correlation: %s %s"
        % (bool(substantial), list(substantial)))
    return out


# --------------------------------------------------------------------------- #
# R2: is the CV sharing an artefact of the multiplicative scale?
# --------------------------------------------------------------------------- #
def check_R2(rep: pd.DataFrame, ind: pd.DataFrame, mm: pd.DataFrame) -> dict:
    out = {}
    n_perm = 20000

    def run(tag, xr, xi, seed):
        res = perm_null(xr, xi, n_perm, seed)
        log("R2 %s: pearson=%+.6f perm p=%.5f (null sd=%.4f)"
            % (tag, res["observed"], res["perm_p_two_sided"], res["null_sd"]))
        return res

    # linear-space sd instead of CV
    out["linear_sd"] = run("linear sd", mm["sd_rep"].values, mm["sd_ind"].values, 201)
    out["log_space_sd"] = run("log-space sd", mm["sd_log_rep"], mm["sd_log_ind"], 202)
    out["log_of_cv"] = run("log cv", np.log(mm["cv_rep"]), np.log(mm["cv_ind"]), 203)

    def demean_by_a1a2(frame, col):
        return frame[col] - frame.groupby(["input_axis_1_index", "input_axis_2_index"])[col].transform("mean")

    cv_r_res = demean_by_a1a2(mm, "cv_rep")
    cv_i_res = demean_by_a1a2(mm, "cv_ind")
    out["cv_after_a1a2_means_removed"] = run("cv minus (a1,a2) means", cv_r_res, cv_i_res, 204)
    sd_r_res = demean_by_a1a2(mm, "sd_rep")
    sd_i_res = demean_by_a1a2(mm, "sd_ind")
    out["linear_sd_after_a1a2_means_removed"] = run("linear sd minus (a1,a2) means", sd_r_res, sd_i_res, 205)

    def demean_by(frame, keys, col):
        return frame[col] - frame.groupby(keys)[col].transform("mean")

    out["cv_after_a1a2_and_channel_means_removed"] = run(
        "cv minus (a1,a2,chan) means",
        demean_by(mm, ["input_axis_1_index", "input_axis_2_index", "output_channel_index"], "cv_rep"),
        demean_by(mm, ["input_axis_1_index", "input_axis_2_index", "output_channel_index"], "cv_ind"),
        206,
    )
    out["linear_sd_after_a1a2_and_channel_means_removed"] = run(
        "linear sd minus (a1,a2,chan) means",
        demean_by(mm, ["input_axis_1_index", "input_axis_2_index", "output_channel_index"], "sd_rep"),
        demean_by(mm, ["input_axis_1_index", "input_axis_2_index", "output_channel_index"], "sd_ind"),
        207,
    )
    for k, v in out.items():
        if isinstance(v, dict) and "observed" in v:
            v["abs_observed"] = abs(v["observed"])
    out["_summary"] = {
        "cv_on_original_scale_pearson": pearson(mm["cv_rep"], mm["cv_ind"]),
        "linear_sd_pearson": out["linear_sd"]["observed"],
        "log_space_sd_pearson": out["log_space_sd"]["observed"],
        "cv_after_a1a2_removal": out["cv_after_a1a2_means_removed"]["observed"],
        "linear_sd_after_a1a2_removal": out["linear_sd_after_a1a2_means_removed"]["observed"],
        "cv_survives_a1a2_removal": bool(
            0 < out["cv_after_a1a2_means_removed"]["observed"]
            and out["cv_after_a1a2_means_removed"]["perm_p_two_sided"] < 0.001),
        "linear_sd_survives_a1a2_removal": bool(
            0 < out["linear_sd_after_a1a2_means_removed"]["observed"]
            and out["linear_sd_after_a1a2_means_removed"]["perm_p_two_sided"] < 0.001),
        "sharing_survives_on_linear_sd_scale": bool(
            0 < out["linear_sd"]["observed"] and out["linear_sd"]["perm_p_two_sided"] < 0.001),
        "material_threshold": "|r| >= 0.10 with permutation p < 0.001",
    }
    out["_summary"]["sharing_present_at_at_least_one_dispersion_scale"] = bool(
        out["_summary"]["sharing_survives_on_linear_sd_scale"]
        or out["_summary"]["cv_survives_a1a2_removal"]
        or out["_summary"]["log_space_sd_pearson"] >= 0.10)
    log("R2 summary: %s" % json.dumps(out["_summary"], default=float))
    return out


# --------------------------------------------------------------------------- #
# R3: is the bin-0 excess an artefact of the trend model?
# --------------------------------------------------------------------------- #
def check_R3(rep: pd.DataFrame) -> dict:
    bm = bin_means(rep)
    estimates = {}

    def excess_for(tag, deg, fit_bins):
        xs = np.array(fit_bins, dtype=float)
        co = poly_fit(xs, bm.loc[fit_bins].values, deg)
        e = float(bm.loc[0] - poly_eval(co, 0.0))
        estimates[tag] = {
            "degree": deg,
            "fit_bins": list(fit_bins),
            "predicted_bin0": poly_eval(co, 0.0),
            "excess": e,
            "ratio_exp": float(np.exp(e)),
        }
        return e

    vals = []
    for deg in (1, 2, 3):
        for fit_bins, lab in [(list(range(1, 8)), "bins1_7"),
                              (list(range(1, 7)), "bins1_6"),
                              (list(range(2, 8)), "bins2_7")]:
            vals.append(excess_for(f"deg{deg}_{lab}", deg, fit_bins))

    # direct neighbour comparison, no trend model at all
    direct_1_2 = float(bm.loc[0] - 0.5 * (bm.loc[1] + bm.loc[2]))
    direct_1_3 = float(bm.loc[0] - np.mean([bm.loc[1], bm.loc[2], bm.loc[3]]))
    estimates["direct_mean_bins1_2"] = {"excess": direct_1_2, "ratio_exp": float(np.exp(direct_1_2)),
                                        "degree": None, "fit_bins": None}
    estimates["direct_mean_bins1_3"] = {"excess": direct_1_3, "ratio_exp": float(np.exp(direct_1_3)),
                                        "degree": None, "fit_bins": None}
    vals += [direct_1_2, direct_1_3]

    # per-cell version of the extreme variants: does the sign hold in all 49 cells?
    tab = rep.pivot_table(index=["input_axis_1_index", "input_axis_2_index"],
                          columns="signal_bin_index", values="ly").sort_index()
    frac_pos = {}
    cell_specs = [
        ("deg2_bins1_7", 2, list(range(1, 8))),
        ("deg2_bins2_7", 2, list(range(2, 8))),
        ("deg1_bins1_7", 1, list(range(1, 8))),
        ("mean_bins1_2", None, None),
    ]
    for tag, deg, fit_bins in cell_specs:
        per_cell = []
        for row in tab.index:
            v = tab.loc[row].sort_index().values
            if deg is None:
                e = float(v[0] - 0.5 * (v[1] + v[2]))
            else:
                co = poly_fit(np.array(fit_bins, dtype=float), v[fit_bins], deg)
                e = float(v[0] - poly_eval(co, 0.0))
            per_cell.append(e)
        per_cell = np.array(per_cell)
        frac_pos[tag] = {
            "min": float(per_cell.min()),
            "max": float(per_cell.max()),
            "mean": float(per_cell.mean()),
            "fraction_positive": float((per_cell > 0).mean()),
            "n_cells": int(per_cell.size),
        }

    log("R3: excess range across %d trend/comparison variants = [%+.4f, %+.4f]"
        % (len(vals), min(vals), max(vals)))
    log("R3 all estimates: " + ", ".join(
        "%s=%+.4f" % (k, v["excess"]) for k, v in estimates.items()))
    log("R3 per-cell fraction positive: " + ", ".join(
        "%s=%.2f" % (k, v["fraction_positive"]) for k, v in frac_pos.items()))
    return {
        "definition": "bin-0 mean log response minus the predicted value at bin 0 of a polynomial trend fitted to the listed bins",
        "estimates": estimates,
        "range_excess": [float(min(vals)), float(max(vals))],
        "range_ratio_exp": [float(np.exp(min(vals))), float(np.exp(max(vals)))],
        "all_positive": bool(min(vals) > 0),
        "per_cell": frac_pos,
        "primary": verdict(0.9482, estimates["deg2_bins1_7"]["excess"], 5e-5),
    }


def check_R3_independent(ind: pd.DataFrame) -> dict:
    bm = ind.groupby("signal_bin_index")["ly"].mean().sort_index()
    out = {}
    for deg in (1, 2, 3):
        xs = np.arange(1, 7, dtype=float)
        co = poly_fit(xs, bm.loc[1:].values, deg)
        out[f"deg{deg}_bins1_6"] = {
            "bin0_observed": float(bm.loc[0]),
            "predicted_bin0": poly_eval(co, 0.0),
            "excess": float(bm.loc[0] - poly_eval(co, 0.0)),
        }
    d12 = float(bm.loc[0] - 0.5 * (bm.loc[1] + bm.loc[2]))
    out["direct_mean_bins1_2"] = {"excess": d12}
    out["bin_means"] = {str(k): float(v) for k, v in bm.items()}
    log("R3 supplementary (independent panel, bins 0-6): quadratic excess=%+.4f ; linear=%+.4f ; "
        "direct bins1-2=%+.4f"
        % (out["deg2_bins1_6"]["excess"], out["deg1_bins1_6"]["excess"], d12))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    log("step6_verification: adversarial recomputation, independent of output/analysis/common.py")
    tables = load_tables()
    rep, ind = tables["rep"], tables["ind"]

    payload: dict = {
        "task": "Information_03_signal_surface_invariants",
        "step": "step6_verification",
        "t0": T0,
        "independence": {
            "imports_common_py": False,
            "note": (
                "no project module imported; cell means, SVD rank R2 (two independent formulas), "
                "OLS slopes (simple and multi-regressor), Pearson, average-rank Spearman, "
                "polynomial least-squares fits and permutation nulls are re-implemented in this file"
            ),
        },
        "inputs": {
            name: {
                "sha256": sha256(DATA / name),
                "rows": int(df.shape[0]),
                "columns": list(df.columns),
            }
            for name, df in tables.items() if name in (
                "replicated_response_panel.tsv", "independent_response_panel.tsv",
                "independent_fit_error.tsv")
        },
    }

    def section(key, fn):
        try:
            payload[key] = fn()
        except Exception:
            tb = traceback.format_exc()
            ERRORS.append({"section": key, "error": tb})
            payload[key] = {"error": tb.strip().splitlines()[-1], "traceback": tb}
            log(f"!! section {key} failed:\n{tb}")

    section("A_rank1_r2", lambda: check_A(rep))
    section("B_slope_ratio", lambda: check_B(rep))
    section("C_cross_panel_correlation", lambda: check_C(rep, ind))
    section("D_cv_sharing", lambda: check_D(rep, ind))
    section("E_bin0_excess", lambda: check_E(rep))
    if "D_cv_sharing" in payload and "frame" in payload["D_cv_sharing"]:
        mm = payload["D_cv_sharing"].pop("frame")
    else:
        mm = None
    if "C_cross_panel_correlation" in payload:
        m = shared_cells(rep, ind)
    else:
        m = None
    if m is not None:
        section("R1_scale_artefact", lambda: check_R1(rep, ind, m))
    if mm is not None:
        section("R2_multiplicative_artefact", lambda: check_R2(rep, ind, mm))
    section("R3_trend_model_artefact", lambda: check_R3(rep))
    section("R3s_independent_panel_bin0", lambda: check_R3_independent(ind))

    # ------------------------------------------------ headline verdict table
    headline = {}
    for key, lab in [("A_rank1_r2", "rank1"), ("B_slope_ratio", "primary"),
                     ("C_cross_panel_correlation", "primary"), ("D_cv_sharing", "primary"),
                     ("E_bin0_excess", "primary")]:
        blk = payload.get(key, {})
        if isinstance(blk, dict) and isinstance(blk.get(lab), dict):
            headline[key] = blk[lab]
            if key == "A_rank1_r2" and isinstance(blk.get("rank2"), dict):
                headline["A_rank2_r2"] = blk["rank2"]
    payload["headline_verdicts"] = headline
    payload["n_discrepancies"] = sum(1 for v in headline.values() if v["verdict"] != "MATCH")
    payload["errors"] = ERRORS
    payload["log"] = LOG

    save = RESULTS / "t21_verification.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "step6.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("wrote", save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())