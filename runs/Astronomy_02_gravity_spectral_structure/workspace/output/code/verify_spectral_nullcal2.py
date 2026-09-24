#!/usr/bin/env python3
"""Adversarial calibration of TRACK 2's two stacked-peak nulls + the pass-peak
support statistic's chance expectation under BOTH a uniform-frequency null and
a red-noise AR(1) null.

Batched LS; own implementation; does not run the track's script.
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
        CC = (c * c).mean(axis=1); SS = (s * s).mean(axis=1); CS = (c * s).mean(axis=1)
        tau = 0.5 * np.arctan2(2.0 * CS, CC - SS)
        c2 = np.cos(wt - tau[:, None]); s2 = np.sin(wt - tau[:, None])
        cc = (c2 * c2).mean(axis=1); ss = (s2 * s2).mean(axis=1)
        YC = (Yc.T @ c2.T) / n; YS = (Yc.T @ s2.T) / n
        out[:, lo:hi] = (YC * YC / cc[None, :] + YS * YS / ss[None, :]) / var[:, None]
    return out


def ar1_fit(y):
    y = np.asarray(y, float); y = y - y.mean()
    a = float(np.clip(np.dot(y[:-1], y[1:]) / np.dot(y[:-1], y[:-1]), -0.99, 0.99))
    return a, float(np.std(y))


def ar1_gen(n, a, s, rng):
    e = rng.normal(0.0, s * np.sqrt(max(1 - a * a, 1e-12)), n)
    return lfilter([1.0], [1.0, -a], e)


def main():
    t0 = time.time()
    tr = pd.read_csv(DATA / "pass_residual_traces.tsv", sep="\t")
    passes = sorted(tr.pass_id.unique())
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {p: {k: tr.loc[tr.pass_id == p, c].to_numpy() for k, c in MODELS.items()} for p in passes}
    f_uhz = np.arange(200.0, 3000.0, 1.0)
    f_hz = f_uhz * 1e-6
    RES_ELEM = 210.9704641350211
    N_SH = 50          # circular-shift realisations per pass per rep
    N_AR1 = 50         # independent AR(1) stacked realisations per rep
    N_REP = 20         # number of calibration repetitions
    rng = np.random.default_rng(20240919)
    out = {"n_shift_per_pass": N_SH, "n_ar1_per_rep": N_AR1, "n_rep": N_REP}

    # ---- observed quantities
    obs = {}
    for k in MODELS:
        obs_stack = None
        pass_peaks = []
        for p in passes:
            P = my_ls(TS[p] * 60.0, YS[p][k], f_hz)[0]
            obs_stack = P if obs_stack is None else obs_stack + P
            pass_peaks.append(float(f_uhz[int(np.argmax(P))]))
        obs_stack /= len(passes)
        ipk = int(np.argmax(obs_stack))
        fpk = float(f_uhz[ipk])
        supp = int(np.sum(np.abs(np.array(pass_peaks) - fpk) <= RES_ELEM))
        obs[k] = {"stacked_peak_uhz": fpk, "stacked_peak_power": float(obs_stack[ipk]),
                  "pass_peak_support_within_1res": supp,
                  "pass_peak_freqs": pass_peaks,
                  "n_pass_peaks_in_1res_of_stacked": supp}
    out["observed"] = obs
    print("observed:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pass_peak_freqs"} for k, v in obs.items()}), flush=True)

    # ---- calibration reps
    for k in MODELS:
        p_circ_on_ar1, p_ar1_on_ar1, supp_ar1, supp_unif = [], [], [], []
        for rep in range(N_REP):
            # draw one AR(1) "trace set"
            sur = {}
            for p in passes:
                a, s = ar1_fit(YS[p][k])
                sur[p] = ar1_gen(len(YS[p][k]), a, s, rng)
            # (a) observed stacked max of this surrogate set + circular-shift null
            acc_obs = None
            acc_sh = None
            acc_ar1 = None
            for p in passes:
                y = sur[p]; n = len(y)
                acc_obs = my_ls(TS[p] * 60.0, y, f_hz)[0] if acc_obs is None else acc_obs + my_ls(TS[p] * 60.0, y, f_hz)[0]
                sh = np.empty((n, N_SH))
                for j in range(N_SH):
                    sh[:, j] = np.roll(y, int(rng.integers(1, n)))
                PS = my_ls(TS[p] * 60.0, sh, f_hz)
                acc_sh = PS if acc_sh is None else acc_sh + PS
                a, s = ar1_fit(y)
                S = np.column_stack([ar1_gen(n, a, s, rng) for _ in range(N_AR1)])
                PA = my_ls(TS[p] * 60.0, S, f_hz)
                acc_ar1 = PA if acc_ar1 is None else acc_ar1 + PA
                # pass-peak support under AR(1) for this pass (use the AR(1) draws)
                for j in range(N_AR1):
                    pass
            obs_max = float((acc_obs / len(passes)).max())
            sh_max = (acc_sh / len(passes)).max(axis=1)
            p_circ_on_ar1.append(float((sh_max >= obs_max).mean()))
            # AR(1) null: per-pass independent draws; stacked max distribution
            # (need argmax per pass for support; recompute cheaply)
            ar1_max = (acc_ar1 / len(passes)).max(axis=1)
            p_ar1_on_ar1.append(float((ar1_max >= obs_max).mean()))
            print(f"  {k} rep {rep} obs_max={obs_max:.4f} p_circ={p_circ_on_ar1[-1]:.2f}", flush=True)
        out[k] = {
            "calib_phaseRand_p_on_AR1_data_q2.5_q50_q97.5": [float(np.percentile(p_circ_on_ar1, q)) for q in (2.5, 50, 97.5)],
            "calib_phaseRand_frac_p_lt_0.05_on_AR1_data": float(np.mean(np.array(p_circ_on_ar1) < 0.05)),
            "calib_AR1null_p_on_AR1_data_q2.5_q50_q97.5": [float(np.percentile(p_ar1_on_ar1, q)) for q in (2.5, 50, 97.5)],
            "calib_AR1null_frac_p_lt_0.05_on_AR1_data": float(np.mean(np.array(p_ar1_on_ar1) < 0.05)),
        }
        print(k, json.dumps(out[k], indent=1), flush=True)

    # ---- pass-peak support: FULL-PIPELINE AR(1) chance null for the observed
    #      statistic (find stacked peak of the surrogate set, then count how many
    #      of that set's pass peaks fall within 1 resolution element of it).
    supp_out = {}
    N_DRAW = 40
    for k in MODELS:
        fpk = obs[k]["stacked_peak_uhz"]
        counts, margins = [], []
        for rep in range(40):
            spec = []
            peaks = []
            for p in passes:
                y = YS[p][k]; n = len(y)
                a, s = ar1_fit(y)
                S = np.column_stack([ar1_gen(n, a, s, rng) for _ in range(N_DRAW)])
                P = my_ls(TS[p] * 60.0, S, f_hz)      # (N_DRAW, F)
                spec.append(P)
                peaks.append(f_uhz[np.argmax(P, axis=1)])
            spec = np.stack(spec)                       # (P, N_DRAW, F)
            peaks = np.stack(peaks)                     # (P, N_DRAW)
            for j in range(N_DRAW):
                St = spec[:, j, :].mean(axis=0)
                fj = f_uhz[int(np.argmax(St))]
                counts.append(int(np.sum(np.abs(peaks[:, j] - fj) <= RES_ELEM)))
                margins.append(abs(fj - fpk))
        counts = np.array(counts)
        supp_out[k] = {
            "observed_count_of_22": obs[k]["pass_peak_support_within_1res"],
            "ar1_fullpipeline_null_mean_count_per_22": float(counts.mean()),
            "ar1_fullpipeline_null_q2.5_q50_q97.5_count_per_22": [
                float(np.percentile(counts, 2.5)), float(np.percentile(counts, 50)),
                float(np.percentile(counts, 97.5))],
            "ar1_fullpipeline_frac_ge_observed": float((counts >= obs[k]["pass_peak_support_within_1res"]).mean()),
            "n_pipeline_realisations": int(counts.size),
            "median_stacked_peak_offset_from_489uhz": float(np.median(margins)),
            "uniform_null_expected_count_per_22": 22 * min(1.0, 2 * RES_ELEM / 2800.0),
        }
        print("support", k, json.dumps(supp_out[k]), flush=True)
    out["pass_peak_support_chance"] = supp_out

    out["runtime_sec"] = round(time.time() - t0, 1)
    (RES / "_verify_spectral_nullcal2.json").write_text(json.dumps(out, indent=2) + "\n")
    print("runtime", out["runtime_sec"])


if __name__ == "__main__":
    main()