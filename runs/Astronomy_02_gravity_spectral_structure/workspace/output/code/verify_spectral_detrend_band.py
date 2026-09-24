#!/usr/bin/env python3
"""Two extra adversarial checks:
 (1) is model A's stacked 489-uHz maximum a broadband leftover trend rather than
     a line?  compare with per-pass linear detrending;
 (2) matched-statistic AR(1) null for the *band-limited* (860-1310 uHz) stacked
     maximum that TRACK 2 reports in sec4b ("stacked_candidate_band_peak").
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
BAND = (860.0, 1310.0)


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
    fu = np.arange(200.0, 3000.0, 1.0); f_hz = fu * 1e-6
    R = 200
    rng = np.random.default_rng(777)
    out = {}

    # (1) stacked spectrum with per-pass linear detrend
    det = {}
    for k in MODELS:
        acc = None
        for p in passes:
            y = YS[p][k]
            t = TS[p] * 60.0
            tc = t - t.mean()
            y = y - np.polyval(np.polyfit(tc, y, 1), tc)
            P = my_ls(t, y, f_hz)[0]
            acc = P if acc is None else acc + P
        acc /= len(passes)
        i = int(np.argmax(acc))
        det[k] = {"detrended_stacked_peak_uhz": float(fu[i]),
                  "detrended_peak_power": float(acc[i]),
                  "detrended_power_489": float(acc[int(np.argmin(np.abs(fu - 489.0)))]),
                  "detrended_median_low_200_700": float(np.median(acc[(fu >= 200) & (fu <= 700)]))}
        print("detrend", k, json.dumps(det[k]), flush=True)
    out["per_pass_linear_detrend"] = det

    # (2) band-limited matched-statistic AR(1) null
    mb = (fu >= BAND[0]) & (fu <= BAND[1])
    res = {}
    for k in MODELS:
        obs = np.zeros(int(mb.sum()))
        for p in passes:
            obs += my_ls(TS[p] * 60.0, YS[p][k], (fu * 1e-6)[mb])[0]
        obs /= len(passes)
        obs_peak = float(obs.max())
        f_obs = float(fu[mb][int(np.argmax(obs))])
        null = np.empty(R)
        for r in range(R):
            acc = np.zeros(int(mb.sum()))
            for p in passes:
                y = YS[p][k]; a, s = ar1_fit(y)
                acc += my_ls(TS[p] * 60.0, ar1_gen(len(y), a, s, rng), (fu * 1e-6)[mb])[0]
            null[r] = (acc / len(passes)).max()
        res[k] = {"observed_band_stacked_peak_uhz": f_obs,
                  "observed_band_stacked_peak_power": obs_peak,
                  "matched_AR1_null_q50_q95_q99_max": [float(np.percentile(null, q)) for q in (50, 95, 99, 100)],
                  "p_matched_AR1_band_peak": float((null >= obs_peak).mean()),
                  "track_reported_band_peak_uhz": {"A": 860.0, "B": 1310.0, "C": 1228.0}[k],
                  "track_reported_p_vs_matched_ar1": {"A": 1.0, "B": 1.0, "C": 0.994}[k]}
        print("band", k, json.dumps(res[k]), flush=True)
    out["band_limited_matched_null"] = res
    out["runtime_sec"] = round(time.time() - t0, 1)
    (PROJ / "output/results/_verify_spectral_detrend_band.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()