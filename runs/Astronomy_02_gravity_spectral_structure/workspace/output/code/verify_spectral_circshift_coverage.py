#!/usr/bin/env python3
"""How much does TRACK 2's circular-shift (called "periodogram-preserving") null
actually explore the statistic?  Because the observed per-pass y is deterministic
and np.roll only cycles sample indices, the set of distinct shifted series has at
most n_p elements per pass, and all of them are just relabellings of the SAME
time stamps.  We enumerate the full shift orbit per pass/model and measure
D_t = max_f |P_shift(f) - P_obs(f)| / max_f P_obs(f).
For a genuine periodogram-preserving surrogate (phase randomisation of the
Fourier transform with the same time sampling) D_t is driven only by the
phase change.  We compare both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
MODELS = {"A": "residual_model_A_mm_s", "B": "residual_model_B_mm_s", "C": "residual_model_C_mm_s"}
COLS = dict(MODELS)


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


def main():
    tr = pd.read_csv(PROJ / "data/pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    f = np.arange(200.0, 3000.0, 1.0) * 1e-6
    out = {}
    for k in MODELS:
        orbit_frac = []      # fraction of the n shifts that are distinct series
        D = []
        for p in passes:
            g = tr[tr.pass_id == p]
            t = g["relative_time_min"].to_numpy() * 60.0
            y = g[COLS[k]].to_numpy()
            n = len(y)
            shifts = np.unique([tuple(np.roll(y, s)) for s in range(n)])
            orbit_frac.append(len(shifts) / n)
            P0 = my_ls(t, y, f)[0]
            dmax = []
            for j in range(min(n, 20)):
                ys = np.roll(y, j + 1)
                P = my_ls(t, ys, f)[0]
                dmax.append(float(np.max(np.abs(P - P0)) / P0.max()))
            D.append(float(np.median(dmax)))
        out[k] = {
            "mean_fraction_of_distinct_series_in_roll_orbit": float(np.mean(orbit_frac)),
            "median_D_relative_LS_change_over_shifts": float(np.median(D)),
            "q90_D": float(np.percentile(D, 90)),
            "note": "D small => shift genuinely preserves the periodogram; D large => "
                    "the null does not stay on the observed periodogram",
        }
        print(k, json.dumps(out[k]), flush=True)
    (PROJ / "output/results/_verify_spectral_circshift_coverage.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()