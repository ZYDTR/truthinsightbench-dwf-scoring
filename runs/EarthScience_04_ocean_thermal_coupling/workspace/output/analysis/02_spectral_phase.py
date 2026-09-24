#!/usr/bin/env python3
"""Step 2 - Spectral structure, dominant period, and phase stability.

Scientific question
-------------------
Does the d18O proxy contain a resolvable periodic component, and is that
component phase-stable through the record or an artefact of one interval?

Design
------
* Lomb-Scargle periodogram on the irregular age axis (no resampling artefacts).
* Least-squares sinusoid scanning with an F-style 95% period interval.
* AR(1) red-noise Monte-Carlo null: the decisive test.  Sedimentary proxy
  series are strongly autocorrelated, so a periodogram peak of R^2~0.40 is
  only meaningful if it exceeds what AR(1) noise produces at the same
  sampling and autocorrelation.
* Sliding-window phase/amplitude to test phase stability.
* Block holdout against a gradient-matched persistence baseline.

Outputs
-------
output/results/02_spectral_phase.json
output/results/02_periodogram.csv
output/results/02_sliding_window.csv
output/results/02_block_holdout.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20220101)

COLUMNS = ["depth_m", "age_ka_bp_1", "age_ka_bp_2", "d18o_pdb"]


def load_clean() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(ROOT / "data" / "observations.tsv", sep="\t")
    df.columns = COLUMNS
    df = df.dropna(subset=["d18o_pdb"]).reset_index(drop=True)
    return (df["age_ka_bp_1"].to_numpy(),
            df["d18o_pdb"].to_numpy(),
            df["depth_m"].to_numpy())


def harmonic_design(t: np.ndarray, period: float, n_harm: int = 2,
                    trend: bool = True) -> np.ndarray:
    w = 2.0 * np.pi / period
    cols = [np.ones_like(t)]
    if trend:
        cols.append(t)
    for k in range(1, n_harm + 1):
        cols.extend([np.cos(k * w * t), np.sin(k * w * t)])
    return np.vstack(cols).T


def fit_rss(t: np.ndarray, y: np.ndarray, period: float, n_harm: int = 2) -> float:
    A = harmonic_design(t, period, n_harm)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(resid @ resid)


def amplitude_phase(t: np.ndarray, y: np.ndarray, period: float) -> tuple[float, float]:
    """Amplitude and phase of the FUNDAMENTAL only, trend + 2 harmonics controlled."""
    A = harmonic_design(t, period, n_harm=2, trend=True)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    # columns: 1, t, cos(wt), sin(wt), cos(2wt), sin(2wt)
    amp = float(np.hypot(beta[2], beta[3]))
    phase = float(np.arctan2(-beta[3], beta[2]))
    return amp, phase


def ar1_surrogate(age: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """AR(1) surrogates on the irregular age grid.

    Fit AR(1) on the *evenly resampled* series to get rho at the median step,
    then generate surrogate residuals on the same irregular grid by
    exponential decay of correlation with age lag.  Mean and sd are preserved.
    """
    step = float(np.median(np.diff(age)))
    t_even = np.arange(age.min(), age.max(), step)
    y_even = np.interp(t_even, age, y)
    y_even = y_even - np.polyval(np.polyfit(t_even, y_even, 1), t_even)  # detrend
    rho = float(np.corrcoef(y_even[:-1], y_even[1:])[0, 1])
    sd = float(np.std(y_even, ddof=1))

    # Cholesky of the exponential-decay correlation matrix on the irregular grid
    lag = np.abs(age[:, None] - age[None, :]) / step
    corr = rho ** lag
    corr[np.diag_indices_from(corr)] += 1e-10
    L = np.linalg.cholesky(corr)
    out = np.empty((n, age.size))
    for i in range(n):
        out[i] = L @ RNG.standard_normal(age.size) * sd
    return out, rho


def main() -> None:
    age, y, depth = load_clean()
    n = age.size
    sst = float(np.sum((y - y.mean()) ** 2))

    # ---------------- Lomb-Scargle periodogram ---------------------------
    y_det = ss.detrend(y)
    freqs = np.linspace(1 / 400.0, 1 / 5.0, 8000)
    power = ss.lombscargle(age - age.mean(), y_det, 2 * np.pi * freqs, normalize=True)
    periods = 1.0 / freqs
    order = np.argsort(power)[::-1]
    # distinct peaks (>=10% period separation)
    peaks: list[dict] = []
    for i in order:
        if all(abs(periods[i] - p["period_ka"]) / p["period_ka"] > 0.10 for p in peaks):
            peaks.append({"period_ka": float(periods[i]), "power": float(power[i])})
        if len(peaks) >= 6:
            break
    pd.DataFrame({"period_ka": periods, "ls_power": power}).to_csv(
        OUT / "02_periodogram.csv", index=False)

    # ---------------- period scan + F-style interval ---------------------
    grid = np.linspace(40.0, 300.0, 4001)
    rss = np.array([fit_rss(age, y, P, n_harm=2) for P in grid])
    best_idx = int(np.argmin(rss))
    best_period = float(grid[best_idx])
    p_params = 6  # intercept, trend, 2 x (cos, sin)
    s2 = rss[best_idx] / (n - p_params)
    f_crit = 3.0
    within = grid[rss <= rss[best_idx] * (1 + f_crit * p_params / (n - p_params))]
    r2_best = 1.0 - rss[best_idx] / sst

    # fundamental-only R^2
    A_f = harmonic_design(age, best_period, n_harm=1, trend=True)
    b_f, *_ = np.linalg.lstsq(A_f, y, rcond=None)
    r2_fund = 1.0 - float(((y - A_f @ b_f) ** 2).sum()) / sst

    # ---------------- AR(1) red-noise null ---------------------------------
    n_surr = 400
    surr, rho = ar1_surrogate(age, y, n_surr)
    # fit the SAME period search to each surrogate so the null accounts for
    # the look-elsewhere effect of scanning 4001 periods
    scan_grid = np.linspace(40.0, 300.0, 401)
    surr_r2 = np.empty(n_surr)
    for i in range(n_surr):
        rs = np.array([fit_rss(age, surr[i], P, n_harm=2) for P in scan_grid])
        surr_r2[i] = 1.0 - rs.min() / float(np.sum((surr[i] - surr[i].mean()) ** 2))
    p_emp = float((np.sum(surr_r2 >= r2_best) + 1) / (n_surr + 1))

    # ---------------- sliding window phase / amplitude -----------------------
    win = 150.0
    rows = []
    for lo in np.arange(age.min(), age.max() - win + 1e-9, 10.0):
        m = (age >= lo) & (age < lo + win)
        if m.sum() < 50:
            continue
        amp, ph = amplitude_phase(age[m], y[m], best_period)
        # fundamental amplitude relative to window sd (scale-free measure)
        rows.append({
            "window_start_ka": float(lo),
            "window_end_ka": float(lo + win),
            "n": int(m.sum()),
            "amplitude": amp,
            "phase_rad": ph,
            "phase_wrapped_rad": float(ph % (2 * np.pi)),
            "window_sd": float(np.std(y[m], ddof=1)),
            "amplitude_fraction_of_sd": amp / float(np.std(y[m], ddof=1)),
        })
    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / "02_sliding_window.csv", index=False)
    ph_circ = np.angle(np.mean(np.exp(1j * sw["phase_rad"].to_numpy())))
    ph_r = np.abs(np.mean(np.exp(1j * sw["phase_rad"].to_numpy())))

    # ---------------- block holdout vs persistence baseline ------------------
    K = 6
    edges = np.linspace(age.min(), age.max(), K + 1)
    hold = []
    for i in range(K):
        te = (age >= edges[i]) & (age < edges[i + 1])
        tr = ~te
        A = harmonic_design(age, best_period, n_harm=2, trend=True)
        beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        pred = A[te] @ beta
        r2_out = 1.0 - float(((y[te] - pred) ** 2).sum()) / float(
            ((y[te] - y[te].mean()) ** 2).sum())
        rmse_out = float(np.sqrt(np.mean((y[te] - pred) ** 2)))

        # Serial-correlation-matched baseline: nearest-neighbour persistence
        # (the random-walk / red-noise benchmark).  Deliberately CLAMPED at the
        # training-range edges -- an unclamped linear extrapolation blows up for
        # the first and last blocks and would make the baseline artificially bad.
        atr, ytr = age[tr], y[tr]
        pos = np.clip(np.searchsorted(atr, age[te]), 1, atr.size - 1)
        take_hi = (age[te] - atr[pos - 1]) > (atr[pos] - age[te])
        pers = np.where(take_hi, ytr[pos], ytr[pos - 1])
        r2_per = 1.0 - float(((y[te] - pers) ** 2).sum()) / float(
            ((y[te] - y[te].mean()) ** 2).sum())
        rmse_per = float(np.sqrt(np.mean((y[te] - pers) ** 2)))

        hold.append({
            "block": i + 1,
            "age_start_ka": float(edges[i]),
            "age_end_ka": float(edges[i + 1]),
            "n_held_out": int(te.sum()),
            "r2_out_of_sample": r2_out,
            "rmse_out_of_sample": rmse_out,
            "r2_persistence_baseline": r2_per,
            "rmse_persistence_baseline": rmse_per,
            "sinusoid_beats_persistence": bool(rmse_out < rmse_per),
        })
    hd = pd.DataFrame(hold)
    hd.to_csv(OUT / "02_block_holdout.csv", index=False)

    result = {
        "step": "02_spectral_phase",
        "n_rows_used": int(n),
        "lomb_scargle_top_peaks": peaks,
        "period_scan": {
            "grid_ka": [40.0, 300.0],
            "best_period_ka": best_period,
            "r2_trend_plus_2harmonics": r2_best,
            "r2_fundamental_only": r2_fund,
            "approx_95pct_period_interval_ka": [
                float(within.min()), float(within.max())],
            "scan_grid_resolution_ka": float(grid[1] - grid[0]),
        },
        "ar1_red_noise_null": {
            "n_surrogates": n_surr,
            "fitted_ar1_rho_at_median_step": rho,
            "null_r2_mean": float(surr_r2.mean()),
            "null_r2_sd": float(surr_r2.std(ddof=1)),
            "null_r2_p95": float(np.percentile(surr_r2, 95)),
            "null_r2_max": float(surr_r2.max()),
            "observed_r2": r2_best,
            "empirical_p_value": p_emp,
            "instruction": "empirical p uses the same 401-period scan on each surrogate",
        },
        "phase_stability": {
            "n_windows": int(len(sw)),
            "window_width_ka": win,
            "phase_circular_mean_rad": float(ph_circ),
            "phase_resultant_length_R": float(ph_r),
            "phase_sd_rad": float(sw["phase_rad"].std(ddof=1)),
            "amplitude_min": float(sw["amplitude"].min()),
            "amplitude_max": float(sw["amplitude"].max()),
            "amplitude_ratio_max_over_min": float(
                sw["amplitude"].max() / sw["amplitude"].min()),
        },
        "block_holdout": {
            "n_blocks": K,
            "median_r2_out_of_sample": float(hd["r2_out_of_sample"].median()),
            "median_r2_persistence": float(hd["r2_persistence_baseline"].median()),
            "mean_rmse_out_of_sample": float(hd["rmse_out_of_sample"].mean()),
            "mean_rmse_persistence": float(hd["rmse_persistence_baseline"].mean()),
            "n_blocks_sinusoid_beats_persistence": int(
                hd["sinusoid_beats_persistence"].sum()),
            "per_block": hold,
        },
    }
    (OUT / "02_spectral_phase.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "block_holdout"},
                     indent=2))
    print("\nblock holdout:\n", hd.to_string(index=False))


if __name__ == "__main__":
    main()