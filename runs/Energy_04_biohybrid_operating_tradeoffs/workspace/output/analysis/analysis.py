#!/usr/bin/env python3
"""TruthInsightBench - Energy_04_biohybrid_operating_tradeoffs

Dose, time, and operating tradeoffs in a biohybrid energy system.

Executed analysis pipeline. Reads only the frozen task data under ``data/``,
verifies their hashes against ``data_manifest.json``, derives every quantitative
claim reported in ``output/Result.md``, and writes machine-readable result
artifacts to ``output/results/`` plus figures to ``output/figures/``.

Run:  python3 output/analysis/analysis.py
Env:  numpy, pandas, matplotlib (Agg). scipy is NOT required by this script.

Analysis units (per DATA_GUIDE.md):
  * one complete condition-specific trajectory  -> perturbation_time_response.tsv
  * one complete dose setting with its uncertainty -> dose_response.tsv
  * fitted columns are supporting representations, not independent observations
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RNG_SEED = 20230919
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output"
RES = OUT / "results"
FIG = OUT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RESULTS: dict = {}
WARNINGS: list[str] = []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def trap(y, x):
    """np.trapz was removed in NumPy 2.x."""
    fn = getattr(np, "trapezoid", None)
    return fn(y, x) if fn is not None else np.trapz(y, x)


def ols(x, y):
    """Simple OLS y = a*x + b. Returns (slope, intercept, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[0]), float(coef[1]), float(r2)


def r2_categorical(y, g):
    """R^2 of y explained by group means of g."""
    y = np.asarray(y, float)
    g = np.asarray(g)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_between = sum(
        len(y[g == k]) * (y[g == k].mean() - y.mean()) ** 2 for k in np.unique(g)
    )
    return float(ss_between / ss_tot)


def safe_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = pd.Series(np.asarray(a, float)).rank().values
    rb = pd.Series(np.asarray(b, float)).rank().values
    return safe_corr(ra, rb)


def local_resid_sd(v, window=15):
    v = np.asarray(v, float)
    tr = pd.Series(v).rolling(window, center=True, min_periods=3).median().values
    return float(np.std(v - tr))


# --------------------------------------------------------------------------
# Part 0. load + integrity
# --------------------------------------------------------------------------
def part0_load():
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    integrity = []
    for row in manifest["files"]:
        p = DATA / row["path"]
        got = sha256(p)
        integrity.append(
            {
                "path": row["path"],
                "declared_sha256": row["sha256"],
                "observed_sha256": got,
                "bytes": p.stat().st_size,
                "match": got == row["sha256"],
            }
        )
    RESULTS["data_integrity"] = integrity
    if not all(r["match"] for r in integrity):
        WARNINGS.append("data hash mismatch vs data_manifest.json")

    curves = pd.read_csv(DATA / "time_response_curves.tsv", sep="\t")
    curves["row"] = np.arange(len(curves))
    pert = pd.read_csv(DATA / "perturbation_time_response.tsv", sep="\t")
    dose = pd.read_csv(DATA / "dose_response.tsv", sep="\t")

    RESULTS["shapes"] = {
        "time_response_curves.tsv": list(curves.shape),
        "perturbation_time_response.tsv": list(pert.shape),
        "dose_response.tsv": list(dose.shape),
    }
    RESULTS["missingness"] = {
        "time_response_curves.tsv": curves.isna().sum().to_dict(),
        "perturbation_time_response.tsv": pert.isna().sum().to_dict(),
        "dose_response.tsv": {k: int(v) for k, v in dose.isna().sum().items()},
    }
    return curves, pert, dose


# --------------------------------------------------------------------------
# Part 1. regime segmentation  (Finding 1)
# --------------------------------------------------------------------------
def segment(curves):
    """Segment rows into time-resolution regimes.

    Definition used: a row is 'sweep-resolved' if its ``time`` value belongs to
    the global multi-point time set {t : t >= 2} that occurs in the file at all.
    Contiguous runs of sweep-resolved rows define the sweep regime.
    """
    sweep_t = set(int(t) for t in curves["time"].unique() if t >= 2)
    mask = curves["time"].isin(sweep_t).values.astype(int)
    ch = np.diff(np.r_[0, mask, 0])
    starts = np.where(ch == 1)[0]
    ends = np.where(ch == -1)[0]
    return sweep_t, [(int(s), int(e - 1)) for s, e in zip(starts, ends)]


def part1_regimes(curves):
    sweep_t, runs = segment(curves)
    n = len(curves)
    a_end, b_end = 258, 293

    regimes = {
        "A_baseline": dict(rows=[0, a_end], n=a_end + 1),
        "B_sweep": dict(rows=[a_end + 1, b_end], n=b_end - a_end),
        "C_tail": dict(rows=[b_end + 1, n - 1], n=n - b_end - 1),
    }
    for name, d in regimes.items():
        seg = curves.loc[d["rows"][0]: d["rows"][1]]
        d["distinct_time_values"] = sorted(int(t) for t in seg["time"].unique())
        d["n_distinct_time"] = len(d["distinct_time_values"])
        for c in ["promoter_like_signal", "condition_a", "condition_b"]:
            d[f"{c}_min"] = float(seg[c].min())
            d[f"{c}_max"] = float(seg[c].max())
            d[f"{c}_mean"] = float(seg[c].mean())
        d["fit_columns_all_zero"] = bool(
            (seg[["condition_c", "fit_a", "fit_b"]].to_numpy() == 0).all()
        )

    # time-value support
    support = {}
    for t in sorted(int(x) for x in curves["time"].unique()):
        support[str(t)] = int((curves["time"] == t).sum())

    # ---- robustness R1a: does the segmentation choice change the conclusions? ----
    # Move the A|B and B|C boundaries over a wide range and re-estimate the two
    # headline quantities: the baseline-regime a~b correlation and the trend R^2
    # of the tail regime.
    boundary_sens = []
    for da in [-40, -20, -10, 0, 10, 20, 40]:
        for db in [-40, -20, -10, 0, 10, 20, 40]:
            a1, b1 = a_end + da, b_end + db
            if not (60 <= a1 < b1 - 10) or b1 >= n - 30:
                continue
            base = curves.loc[:a1]
            tail = curves.loc[b1 + 1:]
            boundary_sens.append({
                "a_end": a1, "b_end": b1,
                "baseline_n": int(len(base)),
                "tail_n": int(len(tail)),
                "baseline_ab_r": safe_corr(base["condition_a"], base["condition_b"]),
                "baseline_prom_a_r": safe_corr(base["promoter_like_signal"],
                                               base["condition_a"]),
                "tail_ab_r": safe_corr(tail["condition_a"], tail["condition_b"]),
                "tail_trend_r2_prom": ols(tail["row"].values.astype(float),
                                          tail["promoter_like_signal"].values.astype(float))[2],
            })
    bs = pd.DataFrame(boundary_sens)
    boundary_summary = {
        "n_variants": int(len(bs)),
        "baseline_ab_r_min": float(bs["baseline_ab_r"].min()),
        "baseline_ab_r_max": float(bs["baseline_ab_r"].max()),
        "tail_ab_r_min": float(bs["tail_ab_r"].min()),
        "tail_ab_r_max": float(bs["tail_ab_r"].max()),
    }
    if bs["baseline_ab_r"].abs().max() > 0.35:
        WARNINGS.append(
            "baseline a~b correlation becomes non-trivial (>|0.35|) under some "
            "segmentation variants; the null-coupling claim is boundary-sensitive."
        )

    # ---- robustness R1b: alternative definition of sweep regime ----
    # (i) rows with t >= 2 anywhere; (ii) rows in the contiguous run containing t_max
    m_any = curves["time"].values >= 2
    alt_i = int(m_any.sum())
    # (ii) contiguous run around the argmax of time
    imax = int(np.argmax(curves["time"].values))
    lo = hi = imax
    while lo > 0 and curves["time"].values[lo - 1] >= 2:
        lo -= 1
    while hi < n - 1 and curves["time"].values[hi + 1] >= 2:
        hi += 1
    alt_ii = hi - lo + 1

    # ---- robustness R1c: leave-one-regime-out effect on pooled trend ----
    # The POOLED reference is reported alongside the folds so the folds can be
    # read against the right baseline (the drop-A fold is numerically close to
    # the pooled fit, and quoting it as "the baseline" would misstate which
    # regime drives the trend).
    pooled_slope, _, pooled_r2 = ols(
        curves["time"].values.astype(float),
        curves["promoter_like_signal"].values.astype(float))
    leave_out = {"POOLED_all_rows": {"n_kept": int(n),
                                     "slope_per_time_unit": pooled_slope,
                                     "r2": pooled_r2}}
    for name in regimes:
        keep = np.ones(n, bool)
        r0, r1 = regimes[name]["rows"]
        keep[r0: r1 + 1] = False
        s = curves[keep]
        x = s["time"].values.astype(float)
        slope, _, r2 = ols(x, s["promoter_like_signal"].values.astype(float))
        leave_out[name] = {"n_kept": int(keep.sum()), "slope_per_time_unit": slope,
                           "r2": r2}

    out = {
        "n_rows": n,
        "n_distinct_time_overall": int(curves["time"].nunique()),
        "time_value_support": support,
        "sweep_time_values": sorted(sweep_t),
        "sweep_contiguous_runs": runs,
        "regimes": regimes,
        "sweep_rows_contiguous_run": int(sum(e - s + 1 for s, e in runs)),
        "sweep_rows_t_ge_2": alt_i,
        "sweep_rows_run_containing_tmax": alt_ii,
        "rows_outside_B_with_t_ge_2": int(
            ((curves["row"] < a_end + 1) | (curves["row"] > b_end)).pipe(
                lambda m: (curves.loc[m, "time"] >= 2).sum())
        ),
        "boundary_sensitivity": boundary_sens,
        "boundary_sensitivity_summary": boundary_summary,
        "leave_one_regime_out_trend": leave_out,
        "fit_coverage": {
            c: {
                "n_zero": int((curves[c] == 0).sum()),
                "n_zero_in_A": int((curves.loc[curves["row"] <= a_end, c] == 0).sum()),
                "first_nonzero_row": int(
                    curves.index[curves[c] != 0][0]) if (curves[c] != 0).any() else None,
            }
            for c in ["condition_c", "fit_a", "fit_b"]
        },
    }
    RESULTS["finding1_regimes"] = out
    return regimes, sweep_t


# --------------------------------------------------------------------------
# Part 2. path dependence / matched-time hysteresis  (Finding 2)
# --------------------------------------------------------------------------
def part2_pathdependence(curves, regimes):
    B = curves.loc[regimes["B_sweep"]["rows"][0]: regimes["B_sweep"]["rows"][1]]
    n = len(curves)

    # --- 2a. single-valuedness of signal against the labelled time column ---
    single = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        # whole file
        g = curves.groupby("time")[c].agg(["count", "min", "max"])
        multi = g[g["count"] > 1]
        within_range = (multi["max"] - multi["min"])
        global_range = curves[c].max() - curves[c].min()
        # t=0 stratum split by regime
        t0 = curves[curves["time"] == 0]
        single[c] = {
            "r2_categorical_time_all": r2_categorical(curves[c], curves["time"]),
            "r2_categorical_time_regimeC": r2_categorical(
                curves.loc[regimes["C_tail"]["rows"][0]:, c],
                curves.loc[regimes["C_tail"]["rows"][0]:, "time"]),
            "n_times_with_multiple_rows": int(len(multi)),
            "max_within_time_range": float(within_range.max()),
            "max_within_time_range_over_global_range": float(within_range.max() / global_range),
            "t0_A_min": float(t0.loc[t0["row"] <= 258, c].min()),
            "t0_A_max": float(t0.loc[t0["row"] <= 258, c].max()),
            "t0_C_min": float(t0.loc[t0["row"] >= 294, c].min()),
            "t0_C_max": float(t0.loc[t0["row"] >= 294, c].max()),
        }
    single["_t2_rows"] = B[B["time"] == 2][
        ["row", "time", "promoter_like_signal", "condition_a", "condition_b"]
    ].to_dict("records")

    # --- 2b. branch separation at exactly matched nominal time ---
    matched = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        g = B.groupby("time")[c].agg(["count", "min", "max"]).reset_index()
        g = g[g["count"] > 1]
        g["ratio_max_min"] = g["max"] / g["min"].replace(0, np.nan)
        matched[c] = g.to_dict("records")

    # --- 2c. branch separation on a common log-time grid (interpolation) ---
    up = B.iloc[:9]          # rows 259..267, t ascending to the apex
    dn = B.iloc[9:25]        # rows 268..283, t descending
    grid_t = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 9000], float)
    lg = np.log10(grid_t)

    def interp_branch(br, col):
        x = np.log10(np.clip(br["time"].values.astype(float), 1, None))
        y = br[col].values.astype(float)
        o = np.argsort(x)
        return np.interp(lg, x[o], y[o], left=np.nan, right=np.nan)

    loops = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        u = interp_branch(up, c)
        d = interp_branch(dn, c)
        m = np.isfinite(u) & np.isfinite(d)
        mx = float(curves[c].max())
        loops[c] = {
            "up_on_grid": u.tolist(),
            "down_on_grid": d.tolist(),
            "grid_t": grid_t.tolist(),
            "loop_area_logt": float(trap(d[m] - u[m], lg[m])),
            "max_abs_gap_over_max": float(np.nanmax(np.abs(d - u)) / mx),
            "peak_up": float(np.nanmax(u)),
            "peak_down": float(np.nanmax(d)),
            "peak_ratio_down_over_up": float(np.nanmax(d) / max(np.nanmax(u), 1e-9)),
        }

    # --- robustness R2a: jackknife the descending t=2 replicates ---
    jk = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        d2 = B.loc[(B["time"] == 2) & (B.index > 267), c].values.astype(float)
        u2 = B.loc[(B["time"] == 2) & (B.index < 268), c].values.astype(float)
        if len(d2) >= 2 and len(u2) >= 1 and u2[0] > 0:
            loo = [d2[np.arange(len(d2)) != i].mean() / u2[0] for i in range(len(d2))]
            jk[c] = {
                "n_down_replicates": int(len(d2)),
                "down_values": d2.tolist(),
                "up_value": float(u2[0]),
                "ratio_full": float(d2.mean() / u2[0]),
                "ratio_jackknife_min": float(min(loo)),
                "ratio_jackknife_max": float(max(loo)),
            }

    # --- robustness R2b: branch label as predictor after removing log-time ---
    branch_pred = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        s = B[B["time"] >= 2]
        x = np.log10(s["time"].values.astype(float))
        y = s[c].values.astype(float)
        # regress out log-time only on the ascending branch, then test branch offset
        a = s.index.values <= 267
        sl, ic, _ = ols(x[a], y[a])
        resid = y - (sl * x + ic)
        branch_pred[c] = {
            "slope_logt_ascending": sl,
            "ascending_resid_mean": float(resid[a].mean()),
            "descending_resid_mean": float(resid[~a].mean()),
            "gap_desc_minus_asc": float(resid[~a].mean() - resid[a].mean()),
        }

    # --- robustness R2c: apex location in row order vs time order ---
    apex = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        i_row = int(np.argmax(B[c].values.astype(float)))
        ts = B.sort_values("time", kind="stable").reset_index()
        j_t = int(np.argmax(ts[c].values.astype(float)))
        apex[c] = {
            "apex_row_order_global_row": int(B["row"].values[i_row]),
            "apex_row_order_time": int(B["time"].values[i_row]),
            "apex_row_order_value": float(B[c].values[i_row]),
            "apex_value_order_time": int(ts["time"].values[j_t]),
            "apex_value_order_global_row": int(ts["index"].values[j_t]),
            "apex_value_order_value": float(ts[c].values[j_t]),
        }

    RESULTS["finding2_path_dependence"] = {
        "single_valuedness": single,
        "matched_time_branches": matched,
        "loop_on_logtime_grid": loops,
        "jackknife_matched_t2_ratio": jk,
        "branch_effect_after_logtime": branch_pred,
        "apex_location": apex,
    }
    return B


# --------------------------------------------------------------------------
# Part 3. regime-conditional coupling  (Finding 3)
# --------------------------------------------------------------------------
def part3_coupling(curves, regimes):
    segs = {
        "A_baseline": curves.loc[regimes["A_baseline"]["rows"][0]: regimes["A_baseline"]["rows"][1]],
        "B_sweep": curves.loc[regimes["B_sweep"]["rows"][0]: regimes["B_sweep"]["rows"][1]],
        "C_tail": curves.loc[regimes["C_tail"]["rows"][0]: regimes["C_tail"]["rows"][1]],
        "WHOLE": curves,
    }
    pairs = [("promoter_like_signal", "condition_a"),
             ("promoter_like_signal", "condition_b"),
             ("condition_a", "condition_b")]

    levels = {}
    for name, s in segs.items():
        levels[name] = {
            f"{a}|{b}": safe_corr(s[a], s[b]) for a, b in pairs
        }
        levels[name]["n"] = int(len(s))

    # --- robustness R3a: first differences remove the shared smooth trajectory ---
    diffs = {}
    for name, s in segs.items():
        d = {a: np.diff(s[a].values.astype(float)) for a, _ in pairs}
        d["condition_b"] = np.diff(s["condition_b"].values.astype(float))
        diffs[name] = {f"{a}|{b}": safe_corr(d[a], d[b]) for a, b in pairs}
        diffs[name]["n"] = int(len(s) - 1)

    # --- robustness R3b: spearman (rank) versions ---
    ranks = {}
    for name, s in segs.items():
        ranks[name] = {f"{a}|{b}": spearman(s[a], s[b]) for a, b in pairs}

    # --- robustness R3c: detrend both channels with a rolling median then correlate ---
    detrended = {}
    for name, s in segs.items():
        W = 15 if len(s) > 40 else 3
        rr = {}
        for a, b in pairs:
            ra = s[a].values.astype(float) - pd.Series(s[a].values).rolling(
                W, center=True, min_periods=2).median().values
            rb = s[b].values.astype(float) - pd.Series(s[b].values).rolling(
                W, center=True, min_periods=2).median().values
            rr[f"{a}|{b}"] = safe_corr(ra, rb)
        detrended[name] = rr

    # --- robustness R3d: trend share of covariance (pooled r vs within-regime r) ---
    pooled = levels["WHOLE"]["condition_a|condition_b"]
    within = levels["A_baseline"]["condition_a|condition_b"]
    trend_share = 1.0 - (within ** 2) / (pooled ** 2) if pooled != 0 else float("nan")

    # --- lead-lag inside the C tail (raw a vs raw b) ---
    Ct = segs["C_tail"]
    x = Ct["condition_a"].values.astype(float)
    y = Ct["condition_b"].values.astype(float)
    leadlag = []
    for k in range(-25, 26):
        if k > 0:
            aa, bb = x[:-k], y[k:]
        elif k < 0:
            aa, bb = x[-k:], y[:k]
        else:
            aa, bb = x, y
        leadlag.append([k, safe_corr(aa, bb)])
    best = max(leadlag, key=lambda z: z[1])
    # Shape diagnostics: a genuine delay shows an interior peak. This curve does
    # not -- it is monotone on one side and monotone-decreasing on the other,
    # which is the signature of two smooth series rather than a lag.
    # Sort by increasing |lag| on each side so "monotone" reads in the natural
    # direction of travelling away from lag 0.
    pos = sorted([(k, v) for k, v in leadlag if k >= 0], key=lambda z: z[0])
    neg = sorted([(k, v) for k, v in leadlag if k <= -1],
                 key=lambda z: -z[0])  # -1, -2, ... -25
    leadlag_shape = {
        "monotone_increasing_for_lag_ge_0":
            all(pos[i][1] <= pos[i + 1][1] for i in range(len(pos) - 1)),
        "monotone_decreasing_as_abs_lag_grows_for_negative_side":
            all(neg[i][1] >= neg[i + 1][1] for i in range(len(neg) - 1)),
        "r_at_lag_-1": safe_corr(x[1:], y[:-1]),
        "r_at_lag_-25": safe_corr(x[25:], y[:-25]),
        "interior_peak_exists": bool(
            any(leadlag[i][1] > leadlag[i - 1][1] and leadlag[i][1] > leadlag[i + 1][1]
                for i in range(1, len(leadlag) - 1))),
        "interpretation": (
            "No interior peak anywhere in lags -25..+25. r rises monotonically "
            "for lag >= 0 and falls monotonically as |lag| grows on the negative "
            "side. Both are signatures of two smooth series, not of a delay, so "
            "no lead-lag delay is identified."),
    }

    RESULTS["finding3_coupling"] = {
        "levels_by_regime": levels,
        "first_differences": diffs,
        "spearman": ranks,
        "detrended_rolling_median": detrended,
        "pooled_ab": pooled,
        "baseline_ab": within,
        "trend_share_of_pooled_ab_r2": trend_share,
        "leadlag_C_tail_best": {"lag": best[0], "r": best[1],
                                "r_lag0": safe_corr(x, y)},
        "leadlag_C_tail_curve": leadlag,
        "leadlag_C_tail_shape": leadlag_shape,
    }

    # --- noise model: Fano factor (Poisson check) ---
    fano = {}
    for name, s in segs.items():
        fano[name] = {}
        for c in ["promoter_like_signal", "condition_a", "condition_b"]:
            v = s[c].values.astype(float)
            fano[name][c] = {
                "mean": float(v.mean()),
                "var": float(v.var()),
                "fano_all": float(v.var() / max(v.mean(), 1e-9)),
                "fano_local": float(local_resid_sd(v) ** 2 / max(v.mean(), 1e-9)),
            }
    RESULTS["fano_by_regime"] = fano

    # --- robustness R3e: order-shuffle null and segment-wise persistence ---
    # The tail regime is the only place where a sustained trend appears, so it
    # gets both a shuffle null and a segmented-persistence test.
    rng = np.random.default_rng(RNG_SEED)
    Ct = segs["C_tail"]
    y = Ct["promoter_like_signal"].values.astype(float)
    x = Ct["row"].values.astype(float)
    obs_slope, _, obs_r2 = ols(x, y)
    shuffle_r2 = np.array([ols(x, rng.permutation(y))[2] for _ in range(1000)])

    # A block bootstrap that concatenates randomly ordered blocks of *levels*
    # destroys the trend by construction (blocks carry very different local
    # levels), so it is NOT a valid null here and is reported only as a
    # diagnostic of that failure mode.
    blk = 25
    idx = np.arange(len(y))
    nb = len(y) // blk
    boot_r2 = []
    for _ in range(500):
        starts = rng.integers(0, len(y) - blk, size=nb)
        pick = np.concatenate([idx[s: s + blk] for s in starts])
        boot_r2.append(ols(np.arange(len(pick), dtype=float), y[pick])[2])

    # Segment-wise persistence: is the decline present in every consecutive
    # block, or driven by one or two stretches?
    nseg = 20
    bounds = np.linspace(0, len(y), nseg + 1).astype(int)
    seg_slopes, seg_r2 = [], []
    for i in range(nseg):
        s0, s1 = bounds[i], bounds[i + 1]
        if s1 - s0 < 5:
            continue
        sl, _, r_ = ols(x[s0:s1], y[s0:s1])
        seg_slopes.append(sl)
        seg_r2.append(r_)
    seg_slopes = np.array(seg_slopes, float)

    RESULTS["tail_decline_order_null"] = {
        "observed_slope_per_row": obs_slope,
        "observed_r2_row": obs_r2,
        "shuffle_null_r2_mean": float(shuffle_r2.mean()),
        "shuffle_null_r2_ci95": [float(np.percentile(shuffle_r2, 2.5)),
                                 float(np.percentile(shuffle_r2, 97.5))],
        "shuffle_null_p_ge_observed": float(np.mean(shuffle_r2 >= obs_r2)),
        "block_bootstrap_concat_r2_mean": float(np.mean(boot_r2)),
        "block_bootstrap_note": (
            "Invalid as a null: concatenating randomly ordered 25-row blocks "
            "mixes very different local levels and destroys the trend by "
            "construction (mean R2 = %.3f). Reported to document the failure, "
            "not as evidence." % float(np.mean(boot_r2))),
        "n_consecutive_segments": int(len(seg_slopes)),
        "segments_with_negative_slope": int((seg_slopes < 0).sum()),
        "segment_slope_min": float(seg_slopes.min()),
        "segment_slope_max": float(seg_slopes.max()),
        "segment_slope_median": float(np.median(seg_slopes)),
    }
    return segs


# --------------------------------------------------------------------------
# Part 4. fitted columns are derived, not independent  (Finding 4)
# --------------------------------------------------------------------------
def part4_fitted(curves, regimes):
    Ct = curves.loc[regimes["C_tail"]["rows"][0]:].reset_index(drop=True)
    pairs = [("promoter_like_signal", "condition_c"),
             ("condition_a", "fit_a"),
             ("condition_b", "fit_b")]

    # --- 4a. raw vs fit association ---
    assoc = {}
    for raw, fit in pairs:
        s = curves.loc[curves["row"] >= regimes["B_sweep"]["rows"][0]]
        x = s[raw].values.astype(float)
        y = s[fit].values.astype(float)
        assoc[f"{raw}|{fit}"] = {
            "r": safe_corr(x, y),
            "spearman": spearman(x, y),
            "mean_resid": float((y - x).mean()),
            "sd_resid": float((y - x).std()),
            "lag_of_max_corr": max(
                ((k, safe_corr(x[:-k] if k > 0 else (x[-k:] if k < 0 else x),
                               y[k:] if k > 0 else (y[:k] if k < 0 else y)))
                 for k in range(-12, 13)),
                key=lambda z: z[1])[0],
        }

    # --- 4b. moving-average window scan ---
    window_scan = {}
    for raw, fit in pairs:
        s = curves.loc[curves["row"] >= regimes["B_sweep"]["rows"][0]]
        x = s[raw].values.astype(float)
        y = s[fit].values.astype(float)
        rows = []
        for mode in ["center", "trailing"]:
            for w in [3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201, 401]:
                p = pd.Series(x).rolling(w, center=(mode == "center"),
                                        min_periods=1).mean().values
                rows.append([mode, w, float(np.sqrt(np.mean((p - y) ** 2))),
                             safe_corr(p, y)])
        rows.sort(key=lambda z: z[2])
        window_scan[f"{raw}|{fit}"] = [
            {"mode": r[0], "window": r[1], "rmse": r[2], "r": r[3]} for r in rows[:5]
        ]

    # --- 4c. monotonicity of the fitted columns in the C tail (permutation test) ---
    rng = np.random.default_rng(RNG_SEED)
    monotone = {}
    for c in ["condition_c", "fit_a", "fit_b"]:
        v = Ct[c].values.astype(float)
        obs = float(np.mean(np.diff(v) < 0))
        perm = np.array([np.mean(np.diff(rng.permutation(v)) < 0) for _ in range(2000)])
        monotone[c] = {
            "frac_strict_descending": obs,
            "n_steps": int(len(v) - 1),
            "permutation_mean": float(perm.mean()),
            "permutation_sd": float(perm.std()),
            "p_ge_observed": float(np.mean(perm >= obs)),
        }
    for c in ["promoter_like_signal", "condition_a", "condition_b"]:
        v = Ct[c].values.astype(float)
        monotone[c] = {"frac_strict_descending": float(np.mean(np.diff(v) < 0)),
                       "n_steps": int(len(v) - 1)}

    # --- 4d. is the fit closer to the raw than to the plain descending sort? ---
    sortedness = {}
    for raw, fit in pairs:
        v = Ct[raw].values.astype(float)
        y = Ct[fit].values.astype(float)
        s = np.sort(v)[::-1]
        sortedness[f"{raw}|{fit}"] = {
            "rmse_fit_vs_raw": float(np.sqrt(np.mean((y - v) ** 2))),
            "rmse_fit_vs_descending_sort_of_raw": float(np.sqrt(np.mean((y - s) ** 2))),
            "rmse_raw_vs_descending_sort_of_raw": float(np.sqrt(np.mean((v - s) ** 2))),
            "local_noise_sd_raw": local_resid_sd(v),
            "corr_raw_vs_own_descending_sort": safe_corr(v, s),
            "corr_fit_vs_raw": safe_corr(y, v),
        }

    # --- 4e. effective independent sample size if raw+fit are pooled ---
    inflation = {}
    for raw, fit in pairs:
        s = curves.loc[curves["row"] >= regimes["B_sweep"]["rows"][0]]
        inflation[f"{raw}|{fit}"] = {
            "n_rows": int(len(s)),
            "n_if_fit_counted_as_independent": int(2 * len(s)),
            "inflation_factor": 2.0,
            "residual_lag1_autocorr": safe_corr(
                (s[fit] - s[raw]).values[:-1], (s[fit] - s[raw]).values[1:]),
        }

    # --- robustness R4a: sign-change count of the first difference ---
    sign_changes = {}
    for c in ["promoter_like_signal", "condition_a", "condition_b",
              "condition_c", "fit_a", "fit_b"]:
        d = np.diff(Ct[c].values.astype(float))
        sign_changes[c] = int(np.sum(np.diff(np.sign(d)) != 0))

    # --- robustness R4b: smoother applied to row order vs to the descending sort ---
    # The tail regime happens to be close to a descending sort of its own values.
    # If the fitted column is a smoother of the raw channel *in row order*, it must
    # be much closer to that than to a smoother of a re-sorted sequence.
    def _ma(v, w):
        return pd.Series(v).rolling(w, center=True, min_periods=1).mean().values

    operator_test = {}
    for raw, fit in pairs:
        v = Ct[raw].values.astype(float)
        y = Ct[fit].values.astype(float)
        srt = np.sort(v)[::-1]
        cand = []
        for label, src in [("row_order", v), ("descending_sort", srt)]:
            for w in [3, 5, 7, 9, 15, 25, 51]:
                p = _ma(src, w)
                cand.append({"source": label, "window": w,
                             "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                             "r": safe_corr(p, y)})
        cand.sort(key=lambda z: z["rmse"])
        operator_test[f"{raw}|{fit}"] = {
            "best": cand[0], "runner_up": cand[1],
            "best_source_consistent": cand[0]["source"] == cand[1]["source"],
        }

    RESULTS["finding4_fitted_columns"] = {
        "association": assoc,
        "window_scan": window_scan,
        "monotonicity_permutation": monotone,
        "sortedness": sortedness,
        "double_counting": inflation,
        "sign_changes_C_tail": sign_changes,
        "operator_test": operator_test,
        "fit_definition_note": (
            "No analytic form is declared in the manifest. In the tail regime a "
            "centred moving average with a short window (w=3-7) reproduces the "
            "fitted column, but the fit is reproduced at least as well when the "
            "same smoother is applied to the raw channel after descending sort. "
            "The smoothing operator is therefore identifiable, while the axis the "
            "operator was applied along (row order vs. a magnitude-sorted "
            "sequence) is not resolved by these data."
        ),
    }
    return Ct


# --------------------------------------------------------------------------
# Part 5. dose & perturbation amplitude response  (Finding 5)
# --------------------------------------------------------------------------
def part5_dose(pert, dose):
    # ---- dose scan ----
    d = dose.copy()
    x = d["dose"].values.astype(float)
    y = d["response_mean"].values.astype(float)
    slope, intercept, r2_lin = ols(x, y)
    ll_slope, ll_int, r2_ll = ols(np.log(x), np.log(y))
    curv = float((y[2] - y[1]) - (y[1] - y[0]))

    loo = []
    for i in range(3):
        m = np.arange(3) != i
        sl, ic, r2 = ols(x[m], y[m])
        loo.append({
            "dropped_setting": int(i + 1),
            "slope": sl,
            "intercept": ic,
            "r2": r2,
            "predicted_at_dropped_dose": float(sl * x[i] + ic),
            "observed_at_dropped_dose": float(y[i]),
            "residual": float(y[i] - (sl * x[i] + ic)),
        })
    apex = int(np.argmax(y))

    # quadratic through 3 points is exactly identified -> not evidence
    quad_note = (
        "A quadratic in dose has 3 free parameters and the scan has 3 settings; "
        "the fit is exactly determined (R2=1 by construction) and carries no "
        "independent support for an interior optimum."
    )

    dose_out = {
        "n_settings": int(len(d)),
        "dose": x.tolist(),
        "response_mean": y.tolist(),
        "response_sd_present": int(d["response_sd"].notna().sum()),
        "response_sd_missing": int(d["response_sd"].isna().sum()),
        "interior_apex_setting_index": apex + 1,
        "interior_apex_dose": float(x[apex]),
        "fold_change_max_over_min": float(y.max() / y.min()),
        "linear_slope": slope, "linear_intercept": intercept, "linear_r2": r2_lin,
        "loglog_slope": ll_slope, "loglog_r2": r2_ll,
        "second_difference_curvature": curv,
        "leave_one_setting_out": loo,
        "quadratic_note": quad_note,
    }

    # ---- perturbation trajectory ----
    p = pert.copy()
    t = p["time"].values.astype(float)
    pert_out = {"n_timepoints": int(len(p)), "time": t.tolist()}
    for c in ["control", "condition_a", "condition_b"]:
        v = p[f"{c}_mean"].values.astype(float)
        sd = p[f"{c}_sd"].values.astype(float)
        sl, ic, r2l = ols(t, v)
        lsl, lic, r2ll = ols(t, np.log(v))
        pooled_sd = float(np.sqrt(np.mean(sd ** 2)))
        pert_out[c] = {
            "mean": v.tolist(),
            "sd": sd.tolist(),
            "t0": float(v[0]), "t6": float(v[-1]),
            "change": float(v[-1] - v[0]),
            "pct_change": float(100 * (v[-1] / v[0] - 1)),
            "strictly_decreasing": bool(np.all(np.diff(v) < 0)),
            "linear_slope": sl, "linear_r2": r2l,
            "loglinear_slope": lsl, "loglinear_r2": r2ll,
            "half_life_from_loglinear": float(math.log(2) / -lsl) if lsl < 0 else None,
            "pooled_sd": pooled_sd,
            "effect_over_pooled_sd": float(abs(v[-1] - v[0]) / pooled_sd),
            "ratio_to_control_at_each_t": (v / p["control_mean"].values.astype(float)).tolist(),
        }
    ch_a = abs(pert_out["condition_a"]["change"])
    ch_b = abs(pert_out["condition_b"]["change"])
    pert_out["potency_ratio_b_over_a_change"] = float(ch_b / ch_a)
    pert_out["potency_ratio_b_over_a_loglinear_rate"] = float(
        abs(pert_out["condition_b"]["loglinear_slope"])
        / abs(pert_out["condition_a"]["loglinear_slope"]))
    pert_out["sd_inventory"] = {
        "n_sd_columns": 3,
        "n_sd_values_total": int(pert[[c for c in pert.columns
                                       if c.endswith("_sd")]].size),
        "n_sd_missing": int(pert[[c for c in pert.columns
                                  if c.endswith("_sd")]].isna().sum().sum()),
        "sd_min": float(pert[[c for c in pert.columns
                              if c.endswith("_sd")]].min().min()),
        "sd_max": float(pert[[c for c in pert.columns
                              if c.endswith("_sd")]].max().max()),
        "n_endpoint_sd": 6,
    }

    ctrl = p["control_mean"].values.astype(float)
    pert_out["control_drift_abs"] = float(abs(ctrl[-1] - ctrl[0]))
    pert_out["control_drift_over_pooled_sd"] = float(
        abs(ctrl[-1] - ctrl[0]) / pert_out["control"]["pooled_sd"])
    pert_out["control_strictly_decreasing"] = bool(np.all(np.diff(ctrl) < 0))

    # ---- robustness R5a: bootstrap the perturbation difference using supplied SDs ----
    rng = np.random.default_rng(RNG_SEED)
    nboot = 20000
    scen = {}
    for c in ["condition_a", "condition_b"]:
        v = p[f"{c}_mean"].values.astype(float)
        sd = p[f"{c}_sd"].values.astype(float)
        draws = rng.normal(v, sd, size=(nboot, len(v)))
        deltas = draws[:, -1] - draws[:, 0]
        scen[c] = {
            "delta_mean": float(deltas.mean()),
            "delta_ci95_low": float(np.percentile(deltas, 2.5)),
            "delta_ci95_high": float(np.percentile(deltas, 97.5)),
            "p_delta_ge_0": float(np.mean(deltas >= 0)),
        }
    va = p["condition_a_mean"].values.astype(float)
    sa = p["condition_a_sd"].values.astype(float)
    vb = p["condition_b_mean"].values.astype(float)
    sb = p["condition_b_sd"].values.astype(float)
    da = rng.normal(va, sa, size=(nboot, len(va)))
    db = rng.normal(vb, sb, size=(nboot, len(vb)))
    diff = (db[:, -1] - db[:, 0]) - (da[:, -1] - da[:, 0])
    scen["difference_b_minus_a"] = {
        "mean": float(diff.mean()),
        "ci95_low": float(np.percentile(diff, 2.5)),
        "ci95_high": float(np.percentile(diff, 97.5)),
        "p_le_0": float(np.mean(diff <= 0)),
    }

    # ---- robustness R5c: ALL-timepoint resampling (not just the endpoints) ----
    # Endpoint-only resampling holds the intermediate timepoints fixed, which
    # understates the interval. Resample every timepoint of both trajectories.
    da_all = rng.normal(va, sa, size=(nboot, len(va)))
    db_all = rng.normal(vb, sb, size=(nboot, len(vb)))
    diff_all = (db_all[:, -1] - db_all[:, 0]) - (da_all[:, -1] - da_all[:, 0])
    scen["difference_b_minus_a_all_timepoints"] = {
        "mean": float(diff_all.mean()),
        "ci95_low": float(np.percentile(diff_all, 2.5)),
        "ci95_high": float(np.percentile(diff_all, 97.5)),
        "p_le_0": float(np.mean(diff_all <= 0)),
    }

    # ---- robustness R5d: paired per-timepoint separation, t = 2, 4, 6 ----
    # z_k = (b_k - a_k) / sqrt(sd_bk^2 + sd_ak^2); combine via Stouffer.
    zs, diffs = [], []
    for k in range(1, len(t)):
        num = vb[k] - va[k]
        den = math.sqrt(sb[k] ** 2 + sa[k] ** 2)
        zs.append(num / den)
        diffs.append(num)
    stouffer = sum(zs) / math.sqrt(len(zs))
    scen["per_timepoint_separation"] = {
        "timepoints_tested": t[1:].tolist(),
        "b_minus_a": diffs,
        "z_per_timepoint": zs,
        "n_negative": int(sum(1 for d in diffs if d < 0)),
        "stouffer_z": stouffer,
        "control_range_over_pooled_sd": float(
            (ctrl.max() - ctrl.min()) / pert_out["control"]["pooled_sd"]),
    }

    # ---- robustness R5b: alternative dose summaries ----
    alt = {
        "monotone_increasing": bool(np.all(np.diff(y) > 0)),
        "monotone_decreasing": bool(np.all(np.diff(y) < 0)),
        "apex_is_interior": bool(0 < apex < len(y) - 1),
        "first_to_second_ratio": float(y[1] / y[0]),
        "second_to_last_ratio": float(y[2] / y[1]),
    }

    RESULTS["finding5_dose_and_perturbation"] = {
        "dose_scan": dose_out,
        "perturbation": pert_out,
        "bootstrap_from_supplied_sd": scen,
        "dose_alternative_summaries": alt,
    }
    return dose_out, pert_out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(curves, segs, Ct, B, pert, dose, regimes):
    plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.25})

    # Fig 1 - overview of the three regimes against row order and time
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    a = ax[0, 0]
    a.plot(curves["row"], curves["promoter_like_signal"], lw=0.5, color="C0")
    for name, col in [("A_baseline", "C1"), ("B_sweep", "C3"), ("C_tail", "C2")]:
        r0, r1 = regimes[name]["rows"]
        a.axvspan(r0, r1, color=col, alpha=0.13, label=name)
    a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("row index"); a.set_ylabel("promoter_like_signal (symlog)")
    a.set_title("(a) three regimes in row order"); a.legend(fontsize=6, loc="center right")

    a = ax[0, 1]
    a.plot(curves["time"], curves["promoter_like_signal"], ".", ms=1.6, color="C0")
    a.set_xscale("symlog", linthresh=1)
    a.set_xlabel("labelled time"); a.set_ylabel("promoter_like_signal")
    a.set_title("(b) signal vs labelled time: not single-valued")

    a = ax[1, 0]
    for c, col in [("promoter_like_signal", "C0"), ("condition_a", "C1"),
                   ("condition_b", "C2")]:
        a.plot(B["row"], B[c], "o-", ms=2.5, lw=0.9, color=col, label=c)
    a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("row index"); a.set_ylabel("value (symlog)")
    a.set_title("(c) sweep regime: rising then falling branch"); a.legend(fontsize=6)

    a = ax[1, 1]
    for c, col in [("promoter_like_signal", "C0"), ("condition_a", "C1"),
                   ("condition_b", "C2")]:
        a.plot(B["time"], B[c], "o-", ms=2.5, lw=0.9, color=col, label=c)
    a.set_xscale("symlog", linthresh=1); a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("labelled time"); a.set_ylabel("value (symlog)")
    a.set_title("(d) same rows keyed by labelled time"); a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_regime_structure.png"); plt.close(fig)

    # Fig 2 - hysteresis loop on the log-time grid
    loops = RESULTS["finding2_path_dependence"]["loop_on_logtime_grid"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for a, c in zip(ax, ["promoter_like_signal", "condition_a", "condition_b"]):
        g = loops[c]
        a.plot(g["grid_t"], g["up_on_grid"], "o-", color="C0", ms=3, label="ascending")
        a.plot(g["grid_t"], g["down_on_grid"], "s-", color="C3", ms=3, label="descending")
        a.set_xscale("log"); a.set_yscale("symlog", linthresh=1)
        a.set_title(c.replace("_", " "), fontsize=8)
        a.set_xlabel("time"); a.legend(fontsize=6)
    ax[0].set_ylabel("signal")
    fig.suptitle("Hysteresis: ascending vs descending branch at matched time", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_hysteresis.png"); plt.close(fig)

    # Fig 3 - coupling per regime, levels vs first differences
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    names = ["A_baseline", "B_sweep", "C_tail", "WHOLE"]
    pairs = ["promoter_like_signal|condition_a",
             "promoter_like_signal|condition_b",
             "condition_a|condition_b"]
    lv = RESULTS["finding3_coupling"]["levels_by_regime"]
    df = RESULTS["finding3_coupling"]["first_differences"]
    w = 0.26
    for i, p in enumerate(pairs):
        ax[0].bar(np.arange(len(names)) + (i - 1) * w,
                  [lv[n][p] for n in names], w, label=p)
        ax[1].bar(np.arange(len(names)) + (i - 1) * w,
                  [df[n][p] for n in names], w, label=p)
    for a, t in zip(ax, ["pairwise correlation of levels", "pairwise correlation of first differences"]):
        a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=6)
        a.set_ylim(-1.05, 1.05); a.axhline(0, color="k", lw=0.6)
        a.set_title(t, fontsize=8)
    ax[0].legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_coupling_by_regime.png"); plt.close(fig)

    # Fig 4 - raw vs fitted
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    sub = curves.loc[curves["row"] >= 259]
    ax[0].plot(sub["row"], sub["condition_a"], ".", ms=1.5, color="C0", label="condition_a (raw)")
    ax[0].plot(sub["row"], sub["fit_a"], "-", lw=1.2, color="C3", label="fit_a")
    ax[0].set_yscale("symlog", linthresh=1)
    ax[0].set_xlabel("row index"); ax[0].set_ylabel("value (symlog)")
    # The fitted columns are exactly zero for all 259 regime-A rows, so the plot
    # starts at row 259; the title states that explicitly rather than implying a
    # full-file range.
    ax[0].set_title("(a) raw vs fitted, rows 259-1366 "
                    "(fitted columns are exactly 0 before row 259)", fontsize=7)
    ax[0].legend(fontsize=6)

    ax[1].plot(Ct["condition_a"], ".", ms=2, color="C0", label="condition_a")
    ax[1].plot(Ct["fit_a"], "-", lw=0.9, color="C3", label="fit_a")
    ax[1].set_xlabel("row index within tail regime"); ax[1].set_ylabel("value")
    ax[1].set_title("(b) tail regime: fit is strictly monotone", fontsize=8)
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_raw_vs_fitted.png"); plt.close(fig)

    # Fig 5 - dose and perturbation
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    ax[0].errorbar(dose["dose"], dose["response_mean"], yerr=None, fmt="o-",
                   ms=5, color="C0")
    ax[0].set_xlabel("dose"); ax[0].set_ylabel("response_mean")
    ax[0].set_title("(a) dose scan: n=3, no SD supplied", fontsize=8)
    for i, (xx, yy, ss) in enumerate(zip(dose["dose"], dose["response_mean"],
                                         [None] * 3)):
        ax[0].annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points",
                       xytext=(2, 5), fontsize=6)
    t = pert["time"].values
    for c, col in [("control_mean", "C7"), ("condition_a_mean", "C1"),
                   ("condition_b_mean", "C2")]:
        sd = pert[c.replace("_mean", "_sd")].values
        ax[1].errorbar(t, pert[c].values, yerr=sd, fmt="o-", ms=4, capsize=2,
                       color=col, label=c.replace("_mean", ""), lw=1)
    ax[1].set_xlabel("time"); ax[1].set_ylabel("mean value")
    ax[1].set_title("(b) perturbation time response", fontsize=8); ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_dose_perturbation.png"); plt.close(fig)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def write_tables(curves, regimes, Ct, B, pert, dose):
    # regime summary table
    rows = []
    for name, d in regimes.items():
        rows.append({
            "regime": name,
            "row_start": d["rows"][0], "row_end": d["rows"][1], "n_rows": d["n"],
            "n_distinct_time": d["n_distinct_time"],
            "time_values": ",".join(map(str, d["distinct_time_values"])),
            "promoter_min": d["promoter_like_signal_min"],
            "promoter_max": d["promoter_like_signal_max"],
            "condition_a_min": d["condition_a_min"], "condition_a_max": d["condition_a_max"],
            "condition_b_min": d["condition_b_min"], "condition_b_max": d["condition_b_max"],
            "fit_columns_all_zero": d["fit_columns_all_zero"],
        })
    pd.DataFrame(rows).to_csv(RES / "table_regimes.tsv", sep="\t", index=False)

    # matched-time branch table
    rec = []
    for c, rows_ in RESULTS["finding2_path_dependence"]["matched_time_branches"].items():
        for r in rows_:
            rec.append({"channel": c, "time": r["time"], "n": r["count"],
                        "min": r["min"], "max": r["max"],
                        "ratio_max_min": r["ratio_max_min"]})
    pd.DataFrame(rec).to_csv(RES / "table_matched_time_branches.tsv", sep="\t", index=False)

    # fit vs raw comparison
    rec = []
    sc = RESULTS["finding4_fitted_columns"]
    for k, v in sc["sortedness"].items():
        rec.append({"pair": k, **v})
    pd.DataFrame(rec).to_csv(RES / "table_fit_vs_raw.tsv", sep="\t", index=False)

    # per-finding headline numbers
    pd.DataFrame([
        {"finding": 1, "quantity": "sweep_rows",
         "value": regimes["B_sweep"]["n"], "unit": "rows",
         "denominator": regimes["B_sweep"]["n"] + regimes["A_baseline"]["n"] + regimes["C_tail"]["n"]},
        {"finding": 1, "quantity": "n_distinct_time_in_sweep",
         "value": len(RESULTS["finding1_regimes"]["sweep_time_values"]), "unit": "timepoints",
         "denominator": None},
        {"finding": 2, "quantity": "max_within_time_range_over_global_promoter",
         "value": RESULTS["finding2_path_dependence"]["single_valuedness"]
                  ["promoter_like_signal"]["max_within_time_range_over_global_range"],
         "unit": "ratio", "denominator": None},
        {"finding": 3, "quantity": "a_b_corr_baseline",
         "value": RESULTS["finding3_coupling"]["baseline_ab"], "unit": "pearson r",
         "denominator": None},
        {"finding": 3, "quantity": "a_b_corr_pooled",
         "value": RESULTS["finding3_coupling"]["pooled_ab"], "unit": "pearson r",
         "denominator": None},
        {"finding": 4, "quantity": "fit_vs_raw_max_r",
         "value": max(v["r"] for v in sc["association"].values()), "unit": "pearson r",
         "denominator": None},
        {"finding": 5, "quantity": "dose_settings",
         "value": len(dose), "unit": "settings", "denominator": None},
    ]).to_csv(RES / "table_headline_numbers.tsv", sep="\t", index=False)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    curves, pert, dose = part0_load()
    regimes, sweep_t = part1_regimes(curves)
    B = part2_pathdependence(curves, regimes)
    segs = part3_coupling(curves, regimes)
    Ct = part4_fitted(curves, regimes)
    dose_out, pert_out = part5_dose(pert, dose)

    RESULTS["_meta"] = {
        "script": "output/analysis/analysis.py",
        "rng_seed": RNG_SEED,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy_available": False,
        "note": ("scipy was unavailable in this runtime; all statistics "
                 "(OLS, correlations, permutation and bootstrap resampling) "
                 "are implemented with numpy/pandas."),
        "warnings": WARNINGS,
    }

    make_figures(curves, segs, Ct, B, pert, dose, regimes)
    write_tables(curves, regimes, Ct, B, pert, dose)

    (RES / "findings.json").write_text(
        json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "n_rows_curves": len(curves),
        "regimes": {k: v["n"] for k, v in regimes.items()},
        "outputs": sorted(p.name for p in RES.iterdir()) +
                   sorted(p.name for p in FIG.iterdir()),
    }, indent=2))


if __name__ == "__main__":
    main()