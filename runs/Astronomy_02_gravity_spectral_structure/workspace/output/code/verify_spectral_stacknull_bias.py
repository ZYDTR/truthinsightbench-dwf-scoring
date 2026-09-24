#!/usr/bin/env python3
"""Quantify the statistic mismatch in TRACK 2 sec4b.

Track code builds the AR(1) stacked-peak null as
    (1/22) * sum_p max_f P_{p,r}(f)          <- MEAN OF PER-PASS MAXIMA
but the observed statistic is
    max_f (1/22) * sum_p P_p(f)              <- MAX OF PASS-MEANS
By Jensen, mean-of-maxima >= max-of-means, so the reported p is biased high.
Here both statistics are evaluated on the SAME AR(1) surrogate draws.
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


def ar1_gen(n, a, s, rng):
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-12)), n)
    return lfilter([1.0], [1.0, -a], e)


def main():
    t0 = time.time()
    tr = pd.read_csv(PROJ / "data/pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {p: {k: tr.loc[tr.pass_id == p, c].to_numpy() for k, c in MODELS.items()} for p in passes}
    NU = np.arange(860.0, 1311.0, 1.0)
    rng = np.random.default_rng(31337)
    R = 200
    out = {}
    for k in MODELS:
        mean_of_max = np.zeros(R)     # track's stat
        max_of_mean = np.zeros((R, NU.size))     # correct stat
        for p in passes:
            y = YS[p][k]
            a, s = ar1_fit(y)
            S = np.column_stack([ar1_gen(len(y), a, s, rng) for _ in range(R)])
            P = my_ls(TS[p] * 60.0, S, NU * 1e-6)      # (R, 451)
            mean_of_max += P.max(axis=1)
            max_of_mean += P
        mean_of_max /= len(passes)
        max_of_mean = (max_of_mean / len(passes)).max(axis=1)
        out[k] = {
            "null_mean_of_per_pass_maxima_q50_q95_q99": [float(np.percentile(mean_of_max, q)) for q in (50, 95, 99)],
            "null_max_of_pass_mean_q50_q95_q99": [float(np.percentile(max_of_mean, q)) for q in (50, 95, 99)],
            "median_ratio_meanmax_over_maxmean": float(np.median(mean_of_max / np.maximum(max_of_mean, 1e-12))),
            "n_draws": R,
        }
        print(k, json.dumps(out[k]), flush=True)
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_stacknull_bias.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()