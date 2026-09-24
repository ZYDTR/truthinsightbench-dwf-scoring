#!/usr/bin/env python3
"""Locality check for the stacked peaks + verification of my AR(1) convention.

(i)  For model A, is the stacked maximum at 489 uHz a localized line or part of
     a broad low-frequency excess?  Compare with the ensemble AR(1) level.
(ii) Re-run the matched-statistic AR(1) stacked-max null using the TRACK's exact
     surrogate convention (x[0]=0 recursion) to make sure the mismatch result is
     not an artefact of my lfilter initial condition.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

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


def ar1_track(n, a, s, rng):
    """exactly common.ar1_surrogate: x[0]=0, x[i]=a x[i-1] + e[i]"""
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
    out = {}

    # (i) locality: stacked spectrum level around the peak vs local continuum
    for k in MODELS:
        acc = None
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            acc = P if acc is None else acc + P
        acc /= len(passes)
        i = int(np.argmax(acc)); fp = fu[i]
        loc = np.abs(fu - fp) <= 210.97
        near = (fu >= fp - 210.97) & (fu <= fp + 210.97)
        outer = (fu >= max(200.0, fp - 700)) & (fu <= min(3000.0, fp + 700)) & ~near
        out.setdefault("locality", {})[k] = {
            "stacked_peak_uhz": float(fp),
            "power_at_peak": float(acc[i]),
            "median_power_within_1_res_of_peak": float(np.median(acc[near])),
            "median_power_outer_700uhz": float(np.median(acc[outer])) if outer.sum() else None,
            "peak_over_local_median": float(acc[i] / np.median(acc[near])),
            "peak_over_outer_median": float(acc[i] / np.median(acc[outer])) if outer.sum() else None,
            "band_860_1310_median": float(np.median(acc[(fu >= 860) & (fu <= 1310)])),
        }
        print("locality", k, json.dumps(out["locality"][k]), flush=True)

    # (ii) matched AR(1) stacked-max null with the TRACK's exact surrogate
    R = 150
    rng = np.random.default_rng(9091)
    mm = {}
    for k in MODELS:
        acc = None
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            acc = P if acc is None else acc + P
        acc /= len(passes)
        obs = float(acc.max())
        null = np.empty(R)
        for r in range(R):
            a_acc = np.zeros(f_hz.size)
            for p in passes:
                y = YS[p][k]; a, s = ar1_fit(y)
                a_acc += my_ls(TS[p] * 60.0, ar1_track(len(y), a, s, rng), f_hz)[0]
            null[r] = (a_acc / len(passes)).max()
        mm[k] = {"observed_stacked_peak": obs,
                 "track_convention_matched_AR1_q50_q95_q99_max": [float(np.percentile(null, q)) for q in (50, 95, 99, 100)],
                 "p_matched_AR1_track_convention": float((null >= obs).mean()),
                 "n_draws": R}
        print("matched-trackconv", k, json.dumps(mm[k]), flush=True)
    out["matched_AR1_track_convention"] = mm
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_locality.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()