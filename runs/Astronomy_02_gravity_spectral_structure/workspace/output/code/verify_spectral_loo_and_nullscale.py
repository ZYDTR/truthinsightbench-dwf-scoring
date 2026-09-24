#!/usr/bin/env python3
"""(a) reproduce the track's mean-of-per-pass-maxima AR(1) null on the FULL grid
        with my own code (to show the mismatch is the statistic, not my LS);
    (b) leave-one-pass-out robustness of model A's stacked low-frequency excess.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
MODELS = {"A": "residual_model_A_mm_s", "B": "residual_model_B_mm_s", "C": "residual_model_C_mm_s"}


def my_ls(t_sec, Y, f_hz):
    t_sec = np.asarray(t_sec, float); Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Y.shape[0]
    Yc = Y - Y.mean(axis=0, keepdims=True); var = np.mean(Yc * Yc, axis=0)
    F = np.asarray(f_hz, float).size
    out = np.empty((Y.shape[1], F))
    for lo in range(0, F, 700):
        hi = min(lo + 700, F)
        w = 2.0 * np.pi * f_hz[lo:hi]; wt = np.outer(w, t_sec)
        c, s = np.cos(wt), np.sin(wt)
        CC = (c * c).mean(1); SS = (s * s).mean(1); CS = (c * s).mean(1)
        tau = 0.5 * np.arctan2(2 * CS, CC - SS)
        c2 = np.cos(wt - tau[:, None]); s2 = np.sin(wt - tau[:, None])
        cc = (c2 * c2).mean(1); ss = (s2 * s2).mean(1)
        YC = (Yc.T @ c2.T) / n; YS = (Yc.T @ s2.T) / n
        out[:, lo:hi] = (YC * YC / cc[None, :] + YS * YS / ss[None, :]) / var[:, None]
    return out


def ar1_fit(y):
    y = np.asarray(y, float); y = y - y.mean()
    return float(np.clip(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]), -0.99, 0.99)), float(np.std(y))


def ar1_track(n, a, s, rng):   # exact common.ar1_surrogate
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-9)), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def main():
    t0 = time.time()
    tr = pd.read_csv(PROJ / "data/pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {p: {k: tr.loc[tr.pass_id == p, c].to_numpy() for k, c in MODELS.items()} for p in passes}
    fu = np.arange(200.0, 3000.0, 1.0); f_hz = fu * 1e-6
    R = 250
    rng = np.random.default_rng(1717)
    out = {}
    for k in MODELS:
        mom = np.zeros(R)     # mean over passes of per-pass max  (TRACK's sec4b stat)
        mms = np.zeros(f_hz.size)  # sum of per-pass spectra (correct stat numerator)
        for p in passes:
            y = YS[p][k]; a, s = ar1_fit(y)
            S = np.column_stack([ar1_track(len(y), a, s, rng) for _ in range(R)])
            P = my_ls(TS[p] * 60.0, S, f_hz)
            mom += P.max(axis=1)
            mms += P.mean(axis=0)
        mom /= len(passes)
        mms /= len(passes)
        out[k] = {
            "my_mean_of_per_pass_maxima_q50_q95_q99": [float(np.percentile(mom, q)) for q in (50, 95, 99)],
            "my_max_of_pass_mean_q50_q95_q99": [float(np.percentile(mms, q)) for q in (50, 95, 99)],
            "track_reported_ar1_null_q50_q50_q99_max": {
                "A": [0.24247, 0.26988, 0.28120, 0.29890],
                "B": [0.06560, 0.07419, 0.07919, 0.08362],
                "C": [0.13402, 0.14831, 0.15372, 0.15618]}[k],
        }
        print(k, json.dumps(out[k]), flush=True)
    out["n_draws"] = R

    # (b) leave-one-pass-out, model A, full grid
    loo = {}
    stack_all = None
    for p in passes:
        P = my_ls(TS[p] * 60.0, YS[p]["A"], f_hz)[0]
        stack_all = P if stack_all is None else stack_all + P
    stack_all /= len(passes)
    ipk = int(np.argmax(stack_all)); fpk = float(fu[ipk])
    for p in passes:
        others = [q for q in passes if q != p]
        acc = None
        for q in others:
            P = my_ls(TS[q] * 60.0, YS[q]["A"], f_hz)[0]
            acc = P if acc is None else acc + P
        acc /= len(others)
        j = int(np.argmax(acc))
        loo[p] = {"peak_uhz": float(fu[j]), "peak_power": float(acc[j]),
                  "power_at_489": float(acc[int(np.argmin(np.abs(fu - 489.0)))])}
    pk = np.array([loo[p]["peak_power"] for p in passes])
    pw489 = np.array([loo[p]["power_at_489"] for p in passes])
    out["model_A_LOO"] = {
        "full_stacked_peak_uhz": fpk, "full_stacked_peak_power": float(stack_all[ipk]),
        "loo_peak_power_min": float(pk.min()), "loo_peak_power_max": float(pk.max()),
        "loo_power_at_489_min": float(pw489.min()), "loo_power_at_489_max": float(pw489.max()),
        "loo_peak_freq_range": [float(min(loo[p]["peak_uhz"] for p in passes)),
                                float(max(loo[p]["peak_uhz"] for p in passes))],
    }
    print("LOO", json.dumps(out["model_A_LOO"]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_loo_and_nullscale.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()