#!/usr/bin/env python3
"""Adversarial calibration of the two stacked-peak nulls claimed by TRACK 2.

(i)  AR(1) stacked-max null
(ii) circular-shift ('phase-randomised') stacked-max null
Both are applied to (a) the real traces (observed p) and (b) AR(1) surrogate
*trace sets* (calibration: under a correct null the p-values should be uniform).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
DATA = PROJ / "data"
RES = PROJ / "output" / "results"
MODELS = {"A": "residual_model_A_mm_s", "B": "residual_model_B_mm_s", "C": "residual_model_C_mm_s"}
SEED = 20220101


def my_ls(t_sec, Y, f_hz):
    t_sec = np.asarray(t_sec, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Y.shape[0]
    Yc = Y - Y.mean(axis=0, keepdims=True)
    var = np.mean(Yc * Yc, axis=0)
    F = np.asarray(f_hz, float).size
    out = np.empty((Y.shape[1], F))
    for lo in range(0, F, 700):
        hi = min(lo + 700, F)
        w = 2.0 * np.pi * f_hz[lo:hi]
        wt = np.outer(w, t_sec)
        c, s = np.cos(wt), np.sin(wt)
        CC = (c * c).mean(axis=1)
        SS = (s * s).mean(axis=1)
        CS = (c * s).mean(axis=1)
        tau = 0.5 * np.arctan2(2.0 * CS, CC - SS)
        c2 = np.cos(wt - tau[:, None])
        s2 = np.sin(wt - tau[:, None])
        cc = (c2 * c2).mean(axis=1)
        ss = (s2 * s2).mean(axis=1)
        YC = (Yc.T @ c2.T) / n
        YS = (Yc.T @ s2.T) / n
        out[:, lo:hi] = (YC * YC / cc[None, :] + YS * YS / ss[None, :]) / var[:, None]
    return out


def ar1_fit(y):
    y = np.asarray(y, float); y = y - y.mean()
    a = float(np.clip(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]), -0.99, 0.99))
    return a, float(np.std(y))


def ar1_gen(n, a, s, rng):
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-12)), n)
    return lfilter([1.0], [1.0, -a], e)


def stacked_max(series_by_pass, tmin_by_pass, f_hz):
    acc = None
    for p, y in series_by_pass.items():
        P = my_ls(tmin_by_pass[p] * 60.0, y, f_hz)[0]
        acc = P if acc is None else acc + P
    return (acc / len(series_by_pass)).max()


def main():
    t0 = time.time()
    tr = pd.read_csv(DATA / "pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {p: {k: tr.loc[tr.pass_id == p, c].to_numpy() for k, c in MODELS.items()} for p in passes}
    f_uhz = np.arange(200.0, 3000.0, 1.0)
    f_hz = f_uhz * 1e-6
    N = 100
    rng = np.random.default_rng(SEED + 2024)
    out = {"n_realisations": N}

    for k in MODELS:
        # observed stacked max
        obs_stack = None
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            obs_stack = P if obs_stack is None else obs_stack + P
        obs_stack /= len(passes)
        obs_peak = float(obs_stack.max())

        ar1_null = np.empty(N)
        circ_null = np.empty(N)
        p_circ_given_ar1 = np.empty(N)   # phase-randomised p computed on AR(1) data
        for r in range(N):
            sur = {}
            for p in passes:
                a, s = ar1_fit(YS[p][k])
                sur[p] = ar1_gen(len(YS[p][k]), a, s, rng)
            ar1_null[r] = stacked_max(sur, TS, f_hz)
            circ = {p: np.roll(sur[p], int(rng.integers(1, len(sur[p])))) for p in passes}
            circ_null[r] = stacked_max(circ, TS, f_hz)
            # calibrate the phase-randomised p itself, using a fresh independent
            # shift-only null computed from the same AR(1) data set
            cs = np.empty(40)
            for j in range(40):
                cj = {p: np.roll(sur[p], int(rng.integers(1, len(sur[p])))) for p in passes}
                cs[j] = stacked_max(cj, TS, f_hz)
            p_circ_given_ar1[r] = float((cs >= ar1_null[r]).mean())
        out[k] = {
            "observed_stacked_peak_power": obs_peak,
            "ar1_null_stacked_peak_q50_q95_q99_max": [float(np.percentile(ar1_null, q)) for q in (50, 95, 99, 100)],
            "p_observed_vs_ar1_null": float((ar1_null >= obs_peak).mean()),
            "circshift_null_stacked_peak_q50_q95_q99_max": [float(np.percentile(circ_null, q)) for q in (50, 95, 99, 100)],
            "p_observed_vs_circshift_null": float((circ_null >= obs_peak).mean()),
            "calibration_phaseRand_p_on_AR1_data_q2.5_q50_q97.5": [
                float(np.percentile(p_circ_given_ar1, q)) for q in (2.5, 50, 97.5)],
            "calibration_frac_p_lt_0.05_on_AR1_data": float((p_circ_given_ar1 < 0.05).mean()),
        }
        print(k, json.dumps(out[k], indent=1), flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    (RES / "_verify_spectral_nullcal.json").write_text(json.dumps(out, indent=2) + "\n")
    print("runtime", out["runtime_sec"])


if __name__ == "__main__":
    main()