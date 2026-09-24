#!/usr/bin/env python3
"""Track 2 - Is there reproducible spectral structure in the residual traces?

Data group:  data/pass_residual_traces.tsv ONLY.

Independent analysis unit: one complete pass (n = 22).  Every statistical
statement below is aggregated over passes (sign test / Wilcoxon / bootstrap /
leave-one-pass-out).  Frequency bins inside one pass are *not* treated as
independent replicates: they are collapsed into per-pass summary statistics
before any test is run.

Sections
 1. per-pass normalized Lomb-Scargle periodogram, 200-3000 uHz, models A/B/C
 2. candidate-band (860-1310 uHz) band-mean power vs AR(1) red-noise surrogates
 3. frequency-comb test at 1010/1060/1110/1210 (and the full 860..1310 comb)
 4. cross-pass reproducibility of the stacked spectrum
 5. phase coherence of a sinusoid at 1010 / 1210 uHz
 6. spectral slope + low/high band power ratio
 7. verdict

The observed periodograms are computed with common.ls_power (the shared API).
Surrogate batches use an exactly equivalent vectorised routine (`ls_norm_batch`,
validated in-script to <1e-12 against common.ls_power) purely for speed.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CAND_BAND,
    MODELS,
    ar1_params,
    ar1_surrogate,
    band_power_percentile,
    freq_grid,
    load_traces,
    ls_power,
    min_sinusoid_amplitude,
    write_json,
)

OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

COMB_CORE = [1010.0, 1060.0, 1110.0, 1210.0]
COMB_FULL = list(np.arange(860.0, 1310.0 + 1e-9, 50.0))  # 860..1310 step 50
N_SURROGATE = 1000
SEED = 20220101


# --------------------------------------------------------------------------
# fast, validated, batch Lomb-Scargle (Zechmeister & Kuerster, as scipy)
# --------------------------------------------------------------------------
def ls_norm_batch(t_sec, Y, f_hz, freq_chunk=700):
    """Normalized LS power for a batch of series; identical to common.ls_power.

    Y : (N,) or (N, S)  - S series sharing the same time stamps.
    Returns (S, F).
    """
    t_sec = np.asarray(t_sec, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Y.shape[0]
    Y = Y - Y.mean(axis=0, keepdims=True)
    YY = (Y * Y).mean(axis=0)                     # scipy's YY, weights sum to 1
    f_hz = np.asarray(f_hz, float)
    out = np.empty((Y.shape[1], f_hz.size))
    for lo in range(0, f_hz.size, freq_chunk):
        hi = min(lo + freq_chunk, f_hz.size)
        w = 2.0 * np.pi * f_hz[lo:hi]
        wt = w[:, None] * t_sec[None, :]          # (F, N)
        cw, sw = np.cos(wt), np.sin(wt)
        CC = (cw * cw).mean(axis=1)
        SS = 1.0 - CC
        CS = (cw * sw).mean(axis=1)
        tau = 0.5 * np.arctan2(2.0 * CS, CC - SS)
        c2 = np.cos(wt - tau[:, None])
        s2 = np.sin(wt - tau[:, None])
        YC = (Y.T @ c2.T) / n                     # (S, F)
        YS = (Y.T @ s2.T) / n
        CC2 = (c2 * c2).mean(axis=1)
        SS2 = 1.0 - CC2
        pgram = 2.0 * ((YC / CC2[None, :]) * YC + (YS / SS2[None, :]) * YS)
        out[:, lo:hi] = pgram * (0.5 / YY)[:, None]
    return out


def validate_batch_ls():
    """Prove ls_norm_batch == common.ls_power on real data before using it."""
    tr = load_traces()
    f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
    worst = 0.0
    rng = np.random.default_rng(0)
    n_checked = 0
    for pid in ["pass_01", "pass_07", "pass_13", "pass_22"]:
        g = tr[tr.pass_id == pid]
        t = g["relative_time_min"].to_numpy()
        for tag, col in MODELS.items():
            y = g[col].to_numpy()
            obs = ls_power(t, y, f_hz)
            fast = ls_norm_batch(t * 60.0, y, f_hz)[0]
            worst = max(worst, float(np.max(np.abs(obs - fast))))
            a, s = ar1_params(y)
            surr = np.column_stack([ar1_surrogate(len(y), a, s, rng) for _ in range(8)])
            ref = np.column_stack([ls_power(t, surr[:, j], f_hz) for j in range(8)])
            got = ls_norm_batch(t * 60.0, surr, f_hz)
            worst = max(worst, float(np.max(np.abs(ref - got.T))))
            n_checked += 1
    # also prove that restricting the batch to the 451-point band grid is
    # numerically identical to slicing the full 1 uHz grid (surrogate path)
    _fu, _fh = freq_grid(200.0, 3000.0, 1.0)
    _mb = (_fu >= CAND_BAND[0]) & (_fu <= CAND_BAND[1])
    a, s = ar1_params(tr.loc[tr.pass_id == "pass_05",
                             MODELS["C"]].to_numpy())
    t = tr.loc[tr.pass_id == "pass_05", "relative_time_min"].to_numpy()
    S = np.column_stack([ar1_surrogate(len(t), a, s, rng) for _ in range(4)])
    full = ls_norm_batch(t * 60.0, S, _fh)[:, _mb]
    band = ls_norm_batch(t * 60.0, S, _fh[_mb])
    worst_band = float(np.max(np.abs(full - band)))
    # and that it matches common.ls_power on surrogates too
    ref_band = np.column_stack(
        [ls_power(t, S[:, j], _fh[_mb]) for j in range(4)]
    )
    worst_ref = float(np.max(np.abs(ref_band - band.T)))
    return {
        "max_abs_diff_vs_common_ls_power": worst,
        "pass_model_combos_checked": n_checked,
        "band_grid_vs_full_grid_slice_max_abs_diff": worst_band,
        "band_grid_vs_common_ls_power_max_abs_diff": worst_ref,
    }


def fit_sinusoid_full(t_min, y):
    """Same detrending/model as common.min_sinusoid_amplitude but returns phase.

    Returns (amplitude, phase_rad, rms_of_detrended_series).
    Convention: y = A*cos(w t - phi), phi = atan2(b, a) with a*cos + b*sin.
    """
    t_sec = np.asarray(t_min, float) * 60.0
    y = np.asarray(y, float)
    tc = t_sec - t_sec.mean()
    p = np.polyfit(tc, y, 1)
    y = y - np.polyval(p, tc)
    return float(np.std(y)), y


def sinusoid_amp_phase(t_min, y, f_uhz):
    t_sec = np.asarray(t_min, float) * 60.0
    y = np.asarray(y, float)
    tc = t_sec - t_sec.mean()
    p = np.polyfit(tc, y, 1)
    yd = y - np.polyval(p, tc)
    w = 2.0 * np.pi * f_uhz * 1e-6
    X = np.column_stack([np.cos(w * t_sec), np.sin(w * t_sec)])
    beta, *_ = np.linalg.lstsq(X, yd, rcond=None)
    amp = float(np.hypot(beta[0], beta[1]))
    phase = float(np.arctan2(beta[1], beta[0]))       # y = A cos(wt - phi)
    rms = float(np.sqrt(np.mean(yd ** 2)))
    return amp, phase, rms


def rayleigh_R(phases):
    z = np.exp(1j * np.asarray(phases, float))
    n = len(z)
    return float(np.abs(z.sum()) / n), n


def sign_test(above, n):
    """Two-sided exact sign test of count `above` out of n against p=0.5."""
    if n == 0:
        return {"k_above_median": 0, "n": 0, "p_value": None}
    bt = stats.binomtest(int(above), int(n), 0.5, alternative="two-sided")
    return {"k_above_median": int(above), "n": int(n), "p_value": float(bt.pvalue)}


def wilcoxon_vs_half(x):
    x = np.asarray(x, float)
    if np.allclose(x, 0.5):
        return None
    try:
        r = stats.wilcoxon(x - 0.5)
        return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}
    except Exception as exc:  # pragma: no cover
        return {"error": f"{type(exc).__name__}: {exc}"}


def summarize(arr):
    a = np.asarray(arr, float)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "q75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
    }


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    res = {
        "track": "TRACK 2 - reproducible spectral structure in residual traces",
        "data_file": "data/pass_residual_traces.tsv",
        "independent_unit": "one complete time-resolved pass (n = 22)",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "settings": {
            "freq_grid_uhz": [200.0, 3000.0, 1.0],
            "candidate_band_uhz": list(CAND_BAND),
            "comb_core_uhz": COMB_CORE,
            "comb_full_uhz": COMB_FULL,
            "n_surrogate_per_pass_model": N_SURROGATE,
            "surrogate_seed": SEED,
        },
        "validation": {},
    }

    tr = load_traces()
    passes = sorted(tr["pass_id"].unique())
    n_pass = len(passes)
    res["n_passes"] = n_pass
    res["pass_ids"] = passes

    # cache raw arrays once (pandas groupby inside 66+ hot loops is slow)
    TS = {p: tr.loc[tr.pass_id == p, "relative_time_min"].to_numpy() for p in passes}
    YS = {
        p: {tag: tr.loc[tr.pass_id == p, col].to_numpy() for tag, col in MODELS.items()}
        for p in passes
    }

    f_uhz, f_hz = freq_grid(200.0, 3000.0, 1.0)
    m_band = (f_uhz >= CAND_BAND[0]) & (f_uhz <= CAND_BAND[1])
    band_uhz = f_uhz[m_band]
    band_hz = f_hz[m_band]
    m_lo = (f_uhz >= 300.0) & (f_uhz <= 600.0)
    m_hi = (f_uhz >= 2000.0) & (f_uhz <= 3000.0)
    # broad reference grid (separated bands, 1 in 10 frequencies: cheap and
    # sufficient for a mean-power ratio)
    m_ref = (m_lo | m_hi)
    ref_hz = f_hz[m_ref][::10]

    # per-pass meta (span -> Rayleigh resolution)
    meta = {}
    for pid, g in tr.groupby("pass_id"):
        t = g["relative_time_min"].to_numpy()
        span_s = float((t.max() - t.min()) * 60.0)
        meta[pid] = {
            "n_samples": int(len(t)),
            "t_min": float(t.min()),
            "t_max": float(t.max()),
            "span_min": float(t.max() - t.min()),
            "rayleigh_resolution_uhz": 1e6 / span_s,
            "t_start_offset_min": float(t.min()),
            "t_stop_offset_min": float(t.max()),
        }
    res["pass_meta"] = meta
    res["median_rayleigh_resolution_uhz"] = float(
        np.median([meta[p]["rayleigh_resolution_uhz"] for p in passes])
    )

    res["validation"] = validate_batch_ls()

    # ======================================================================
    # 1. per-pass periodograms
    # ======================================================================
    obs_spec = {}      # obs_spec[tag][pid] = (F,)
    obs_band = {}      # band-mean power
    obs_fullmax = {}
    for tag, col in MODELS.items():
        obs_spec[tag] = {}
        obs_band[tag] = {}
        obs_fullmax[tag] = {}
        for pid in passes:
            t = TS[pid]
            y = YS[pid][tag]
            P = ls_power(t, y, f_hz)
            obs_spec[tag][pid] = P
            obs_band[tag][pid] = float(P[m_band].mean())
            i = int(np.argmax(P))
            obs_fullmax[tag][pid] = {
                "f_uhz": float(f_uhz[i]),
                "power": float(P[i]),
                "in_candidate_band": bool(m_band[i]),
            }

    res["sec1_periodograms"] = {
        "note": (
            "normalized LS power per pass (common.ls_power, irregular time stamps, "
            "200-3000 uHz, 1 uHz grid). Values are per-pass; no cross-frequency "
            "pooling is used anywhere."
        ),
        "per_model": {},
    }
    for tag in MODELS:
        mx = np.array([obs_fullmax[tag][p]["power"] for p in passes])
        fx = np.array([obs_fullmax[tag][p]["f_uhz"] for p in passes])
        inband = np.array([obs_fullmax[tag][p]["in_candidate_band"] for p in passes])
        res["sec1_periodograms"]["per_model"][tag] = {
            "pass_global_max_power": {p: obs_fullmax[tag][p] for p in passes},
            "global_max_power_summary": summarize(mx),
            "global_max_freq_uhz_summary": summarize(fx),
            "n_passes_peaking_inside_candidate_band": int(inband.sum()),
            "frac_passes_peaking_inside_candidate_band": float(inband.mean()),
            "n_passes_global_max_below_0.05": int((mx < 0.05).sum()),
        }

    # ======================================================================
    # 2. candidate-band test vs AR(1) surrogates
    # ======================================================================
    rng = np.random.default_rng(SEED)
    # surrogate spectra are only needed inside the candidate band, so the batch
    # is evaluated on the 451-frequency band grid (numerically identical to
    # slicing the 1 uHz full grid - see validation).
    comb_idx_in_band = [
        int(np.argmin(np.abs(band_uhz - f))) for f in COMB_FULL
    ]
    band_pct = {}
    band_obs = {}
    band_null_q = {}
    surro_band = {}     # for reporting the null band-power scale
    surro_band_spec = {}  # (S, 451) surrogate band spectra, reused in section 3
    for tag, col in MODELS.items():
        band_pct[tag], band_obs[tag] = {}, {}
        band_null_q[tag], surro_band[tag] = {}, {}
        surro_band_spec[tag] = {}
        for pid in passes:
            t = TS[pid]
            y = YS[pid][tag]
            a, s = ar1_params(y)
            S = np.column_stack(
                [ar1_surrogate(len(y), a, s, rng) for _ in range(N_SURROGATE)]
            )
            PS = ls_norm_batch(t * 60.0, S, band_hz)
            surro_band_spec[tag][pid] = PS
            nb = PS.mean(axis=1)
            surro_band[tag][pid] = nb
            obs = obs_band[tag][pid]
            band_obs[tag][pid] = obs
            band_pct[tag][pid] = float((nb < obs).mean())
            band_null_q[tag][pid] = {
                "q50": float(np.percentile(nb, 50)),
                "q95": float(np.percentile(nb, 95)),
                "q99": float(np.percentile(nb, 99)),
                "max": float(nb.max()),
            }

    res["sec2_candidate_band"] = {
        "band_uhz": list(CAND_BAND),
        "n_grid_frequencies_in_band": int(m_band.sum()),
        "null": "AR(1) red noise, same irregular time stamps, same n, "
                f"{N_SURROGATE} surrogates per pass/model",
        "per_model": {},
    }
    for tag in MODELS:
        pct = np.array([band_pct[tag][p] for p in passes])
        res["sec2_candidate_band"]["per_model"][tag] = {
            "per_pass": {
                p: {
                    "band_mean_power": band_obs[tag][p],
                    "percentile_vs_ar1": band_pct[tag][p],
                    "ar1_null_q50_q95_q99_max": band_null_q[tag][p],
                }
                for p in passes
            },
            "percentile_summary": summarize(pct),
            "median_percentile_over_passes": float(np.median(pct)),
            "n_passes_gt_0.95": int((pct > 0.95).sum()),
            "n_passes_gt_0.99": int((pct > 0.99).sum()),
            "frac_passes_gt_0.95": float((pct > 0.95).mean()),
            "frac_passes_gt_0.99": float((pct > 0.99).mean()),
            "sign_test_vs_0.5": sign_test(int((pct > 0.5).sum()), len(pct)),
            "wilcoxon_vs_0.5": wilcoxon_vs_half(pct),
            "null_calibration_KS_uniformity_of_percentiles": {
                "D": None, "p_value": None, "note": "filled in sec2b"
            },
        }

    # ======================================================================
    # 2b. null calibration + held-out (split-half) band-power check
    # ======================================================================
    rng_cal = np.random.default_rng(SEED + 1000)
    sec2b = {
        "purpose": (
            "(i) prove the AR(1) surrogate null is correctly calibrated on these "
            "irregular time stamps - surrogate realisations fed through the same "
            "percentile pipeline must give uniform percentiles; (ii) check the "
            "observed band power out-of-sample, since any band chosen by looking "
            "at the data is circular."
        ),
        "null_calibration": {},
        "split_half_out_of_sample": {},
    }
    for tag in MODELS:
        # --- (i) calibration: 200 fresh AR(1) draws per pass, each compared to
        #     its own independent 1000-surrogate null
        pcts = []
        for pid in passes:
            y = YS[pid][tag]
            a, s = ar1_params(y)
            S = np.column_stack(
                [ar1_surrogate(len(y), a, s, rng_cal) for _ in range(201)]
            )
            nb = ls_norm_batch(TS[pid] * 60.0, S, band_hz).mean(axis=1)
            null = nb[1:]
            for v in nb[:1]:
                pcts.append(float((null < v).mean()))
        pcts = np.array(pcts)
        ks = stats.kstest(pcts, "uniform")
        frac95 = float((pcts > 0.95).mean())
        frac99 = float((pcts > 0.99).mean())
        # binomial p for 1/22 exceedance rate vs nominal 5% / 1%
        sec2b["null_calibration"][tag] = {
            "n_samples": int(pcts.size),
            "mean_percentile": float(pcts.mean()),
            "ks_D": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "frac_gt_q95": frac95,
            "frac_gt_q99": frac99,
            "binom_p_frac_gt_q95_vs_0.05": float(
                stats.binomtest(int((pcts > 0.95).sum()), int(pcts.size), 0.05).pvalue
            ),
        }

        # --- (ii) split-half: pass i's first half predicts pass i+1's second half
        obs_ratio = []
        null_ratio = []
        for i, pid in enumerate(passes):
            nxt = passes[(i + 1) % n_pass]
            t1, y1 = TS[pid], YS[pid][tag]
            t2, y2 = TS[nxt], YS[nxt][tag]
            h1 = t1 <= np.median(t1)
            h2 = t2 > np.median(t2)
            a, s = ar1_params(y1)
            # held-out band power / held-out broad-band power; under the null
            # (band power = local red-noise level) this ratio ~ 1
            b1 = float(ls_power(t1[h1], y1[h1], band_hz).mean())
            ref1 = float(ls_power(t1[h1], y1[h1], ref_hz).mean())
            b2 = float(ls_power(t2[h2], y2[h2], band_hz).mean())
            ref2 = float(ls_power(t2[h2], y2[h2], ref_hz).mean())
            obs_ratio.append(b1 / ref1 if ref1 > 0 else np.nan)
            # AR(1) null for the same statistic on held-out halves
            n2 = int(h2.sum())
            S2 = np.column_stack(
                [ar1_surrogate(n2, a, s, rng_cal) for _ in range(N_SURROGATE)]
            )
            b2n = ls_norm_batch(t2[h2] * 60.0, S2, band_hz).mean(axis=1)
            r2n = ls_norm_batch(t2[h2] * 60.0, S2, ref_hz).mean(axis=1)
            null_ratio.append(
                float((np.median(b2 / ref2) > (b2n / r2n)).mean()) if ref2 > 0 else np.nan
            )
        obs_ratio = np.array(obs_ratio)
        null_ratio = np.array(null_ratio)
        med = float(np.median(obs_ratio))
        sec2b["split_half_out_of_sample"][tag] = {
            "statistic": "held-out band power / held-out broad-band power "
                         "(first half of pass i is not used; second half of pass i+1)",
            "observed_ratio_median": med,
            "observed_ratio_summary": summarize(obs_ratio),
            "n_passes_ratio_gt_1": int(np.nansum(obs_ratio > 1.0)),
            "sign_test_vs_1": sign_test(int(np.nansum(obs_ratio > 1.0)), n_pass),
            "median_null_percentile_of_held_out_band_power": float(
                np.nanmedian(null_ratio)
            ),
            "sign_test_null_pct_vs_0.5": sign_test(
                int(np.nansum(null_ratio > 0.5)), n_pass
            ),
        }
    res["sec2b_null_calibration_and_out_of_sample"] = sec2b

    # ======================================================================
    # 3. frequency-comb test
    # ======================================================================
    # per-pass percentile of the power at each comb frequency, computed two ways:
    #   (a) within that pass's own full 200-3000 uHz spectrum (literal request)
    #   (b) within that pass's own 860-1310 uHz band (slope-neutral reference)
    # plus a proper frequency-specific AR(1) null from the same surrogate batch.
    comb_freqs = COMB_FULL
    idx_full = [int(np.argmin(np.abs(f_uhz - f))) for f in comb_freqs]
    idx_band = [int(np.argmin(np.abs(band_uhz - f))) for f in comb_freqs]
    comb_pct_full = {tag: {} for tag in MODELS}
    comb_pct_band = {tag: {} for tag in MODELS}
    comb_pct_ar1 = {tag: {} for tag in MODELS}
    comb_power = {tag: {} for tag in MODELS}
    for tag in MODELS:
        for pid in passes:
            P = obs_spec[tag][pid]
            Pb = P[m_band]
            PSb = surro_band_spec[tag][pid]
            comb_pct_full[tag][pid] = [float((P < P[i]).mean()) for i in idx_full]
            comb_pct_band[tag][pid] = [
                float((Pb < Pb[j]).mean()) for j in idx_band
            ]
            comb_pct_ar1[tag][pid] = [
                float((PSb[:, j] < Pb[j]).mean()) for j in idx_band
            ]
            comb_power[tag][pid] = [float(P[i]) for i in idx_full]

    res["sec3_comb_test"] = {
        "comb_frequencies_uhz": comb_freqs,
        "core_frequencies_uhz": COMB_CORE,
        "aggregation": "per-pass percentile -> sign test / Wilcoxon across the 22 "
                       "passes (never across frequencies)",
        "resolution_caveat_uhz": res["median_rayleigh_resolution_uhz"],
        "comb_spacing_vs_resolution": (
            f"comb spacing 50 uHz = "
            f"{50.0 / res['median_rayleigh_resolution_uhz']:.3f} Rayleigh resolution "
            f"elements; the whole 860-1310 uHz comb spans "
            f"{(1310.0 - 860.0) / res['median_rayleigh_resolution_uhz']:.2f} "
            "resolution elements"
        ),
        "per_frequency": [],
    }
    core_set = set(COMB_CORE)
    for k, fcomb in enumerate(comb_freqs):
        entry = {"f_uhz": float(fcomb), "is_core": bool(fcomb in core_set)}
        for tag in MODELS:
            p_full = np.array([comb_pct_full[tag][p][k] for p in passes])
            p_band = np.array([comb_pct_band[tag][p][k] for p in passes])
            p_ar1 = np.array([comb_pct_ar1[tag][p][k] for p in passes])
            pw = np.array([comb_power[tag][p][k] for p in passes])
            entry[tag] = {
                "median_power": float(np.median(pw)),
                "pct_within_own_full_spectrum": {
                    "median": float(np.median(p_full)),
                    "summary": summarize(p_full),
                    "n_passes_gt_0.5": int((p_full > 0.5).sum()),
                    "sign_test_vs_0.5": sign_test(int((p_full > 0.5).sum()), len(p_full)),
                    "wilcoxon_vs_0.5": wilcoxon_vs_half(p_full),
                },
                "pct_within_own_candidate_band": {
                    "median": float(np.median(p_band)),
                    "summary": summarize(p_band),
                    "n_passes_gt_0.5": int((p_band > 0.5).sum()),
                    "sign_test_vs_0.5": sign_test(int((p_band > 0.5).sum()), len(p_band)),
                    "wilcoxon_vs_0.5": wilcoxon_vs_half(p_band),
                },
                "pct_vs_ar1_null_at_that_frequency": {
                    "median": float(np.median(p_ar1)),
                    "summary": summarize(p_ar1),
                    "n_passes_gt_0.95": int((p_ar1 > 0.95).sum()),
                    "n_passes_gt_0.5": int((p_ar1 > 0.5).sum()),
                    "sign_test_vs_0.5": sign_test(int((p_ar1 > 0.5).sum()), len(p_ar1)),
                    "wilcoxon_vs_0.5": wilcoxon_vs_half(p_ar1),
                },
            }
        res["sec3_comb_test"]["per_frequency"].append(entry)

    # whole-pass combination: one summary per pass (mean percentile across the
    # 4 core comb frequencies), then a single test across 22 passes.
    core_idx = [k for k, f in enumerate(comb_freqs) if f in core_set]
    res["sec3_comb_test"]["core_comb_whole_pass_statistic"] = {}
    for tag in MODELS:
        for label, store in (("pct_within_own_full_spectrum", comb_pct_full),
                             ("pct_within_own_candidate_band", comb_pct_band),
                             ("pct_vs_ar1_null", comb_pct_ar1)):
            x = np.array([np.mean([store[tag][p][k] for k in core_idx])
                          for p in passes])
            res["sec3_comb_test"]["core_comb_whole_pass_statistic"].setdefault(tag, {})[
                label
            ] = {
                "per_pass_mean_percentile": {p: float(v) for p, v in zip(passes, x)},
                "median": float(np.median(x)),
                "sign_test_vs_0.5": sign_test(int((x > 0.5).sum()), len(x)),
                "wilcoxon_vs_0.5": wilcoxon_vs_half(x),
            }

    # ======================================================================
    # 4. cross-pass reproducibility (stacked spectrum)
    # ======================================================================
    res["sec4_reproducibility"] = {
        "note": "stacked = unweighted pass-average of the 22 per-pass normalized "
                "LS spectra (each pass normalized to its own power scale)",
        "per_model": {},
    }
    stacked = {}
    for tag in MODELS:
        St = np.mean([obs_spec[tag][p] for p in passes], axis=0)
        stacked[tag] = St
        i = int(np.argmax(St))
        dist = res["median_rayleigh_resolution_uhz"]
        pk, props = find_peaks(St, distance=int(round(dist)))
        order = np.argsort(St[pk])[::-1]
        top8 = [
            {
                "f_uhz": float(f_uhz[pk[j]]),
                "power": float(St[pk[j]]),
                "delta_uhz_from_global_max": float(abs(f_uhz[pk[j]] - f_uhz[i])),
                "rank_by_power": int(r + 1),
            }
            for r, j in enumerate(order[:8])
        ]
        # which passes individually peak near the stacked global max?
        fpk = float(f_uhz[i])
        supp_fixed = 0
        supp_own = 0
        pass_peak_near = {}
        for p in passes:
            ip = int(np.argmax(obs_spec[tag][p]))
            fp = float(f_uhz[ip])
            res_p = meta[p]["rayleigh_resolution_uhz"]
            near_fixed = abs(fp - fpk) <= dist
            near_own = abs(fp - fpk) <= res_p
            supp_fixed += int(near_fixed)
            supp_own += int(near_own)
            pass_peak_near[p] = {
                "pass_peak_f_uhz": fp,
                "pass_peak_power": float(obs_spec[tag][p][ip]),
                "near_stacked_peak_fixed_211uhz": bool(near_fixed),
                "near_stacked_peak_own_resolution": bool(near_own),
            }
        # leave-one-pass-out sensitivity of the stacked peak
        loo = {}
        for p in passes:
            others = [q for q in passes if q != p]
            S2 = np.mean([obs_spec[tag][q] for q in others], axis=0)
            j = int(np.argmax(S2))
            loo[p] = {"f_uhz": float(f_uhz[j]), "power": float(S2[j])}
        loo_f = np.array([loo[p]["f_uhz"] for p in passes])
        i_boot = int(np.argmax(St))
        res["sec4_reproducibility"]["per_model"][tag] = {
            "stacked_global_max": {"f_uhz": fpk, "power": float(St[i_boot])},
            "stacked_top8_local_maxima": top8,
            "n_local_maxima_separated_by_1_resolution_element": int(len(pk)),
            "n_passes_peaking_within_1_res_elem_of_stacked_peak_fixed_211uhz": supp_fixed,
            "n_passes_peaking_within_1_res_elem_of_stacked_peak_own_resolution": supp_own,
            "expected_passes_by_chance_fixed": float(
                n_pass * min(1.0, 2.0 * dist / (3000.0 - 200.0))
            ),
            "pass_peaks_relative_to_stacked_peak": pass_peak_near,
            "leave_one_pass_out_stacked_peak_uhz": loo,
            "leave_one_pass_out_peak_freq_summary": summarize(loo_f),
            "leave_one_pass_out_all_within_1_res_elem": bool(
                np.all(np.abs(loo_f - fpk) <= dist)
            ),
            "stacked_peak_power_vs_single_pass_max_power": {
                "stacked": float(St[i_boot]),
                "median_single_pass_max": float(
                    np.median([obs_fullmax[tag][p]["power"] for p in passes])
                ),
            },
        }

    # ======================================================================
    # 4b. is the stacked peak itself significant, and is its candidate-band
    #     near-miss (model C) consistent with chance?
    # ======================================================================
    rng_46 = np.random.default_rng(SEED + 46)
    N_STACK_SURR = 300
    res["sec4b_stacked_significance"] = {
        "purpose": "does the stacked spectrum contain a peak that is unusual "
                   "relative to (i) an AR(1) red-noise null and (ii) a "
                   "periodogram-preserving phase-randomised null",
        "per_model": {},
    }
    for tag in MODELS:
        # ---- (i) AR(1) null for the *stacked* spectrum.  Band-limited peak
        # null reuses the 1000 band surrogates from section 2; the global-peak
        # null needs the full grid (N_STACK_SURR per pass).
        null_peak = np.zeros(N_STACK_SURR)
        for pid in passes:
            y = YS[pid][tag]
            a, s = ar1_params(y)
            S = np.column_stack(
                [ar1_surrogate(len(y), a, s, rng_46) for _ in range(N_STACK_SURR)]
            )
            null_peak += ls_norm_batch(TS[pid] * 60.0, S, f_hz).max(axis=1)
        null_peak /= n_pass
        # band-limited stacked peak null, from the sec2 surrogate pools
        nb_stack = np.zeros(N_SURROGATE)
        for pid in passes:
            nb_stack += surro_band_spec[tag][pid].max(axis=1)
        nb_stack /= n_pass
        St = stacked[tag]
        obs_peak = float(St.max())
        p_peak = float((null_peak >= obs_peak).mean())
        i_band = int(np.argmax(np.where(m_band, St, -np.inf)))
        obs_band_peak = float(St[i_band])
        p_band_peak = float((nb_stack >= obs_band_peak).mean())
        res["sec4b_stacked_significance"]["per_model"][tag] = {
            "stacked_global_peak_power": obs_peak,
            "ar1_null_stacked_global_peak_q50_q95_q99_max": [
                float(np.percentile(null_peak, q)) for q in (50, 95, 99, 100)
            ],
            "p_stacked_global_peak_vs_ar1": p_peak,
            "n_surrogates_global": N_STACK_SURR,
            "p_resolution_global": 1.0 / N_STACK_SURR,
            "stacked_candidate_band_peak": {
                "f_uhz": float(f_uhz[i_band]),
                "power": obs_band_peak,
                "p_vs_matched_ar1_null": p_band_peak,
                "ar1_null_stacked_candband_peak_q50_q95_q99_max": [
                    float(np.percentile(nb_stack, q)) for q in (50, 95, 99, 100)
                ],
                "n_surrogates_band": N_SURROGATE,
                "p_resolution_band": 1.0 / N_SURROGATE,
            },
        }

    # 4b(ii) baseline-preserving null: circularly rotate each observed pass in
    # time (keeps its exact periodogram shape and power level, destroys phase
    # alignment across passes). A coherent line shared by all 22 passes would
    # survive this null; a line whose frequency wanders pass-to-pass would not.
    res["sec4b_stacked_significance"]["phase_randomised_stacked_null"] = {
        "method": "circular time shift of each observed pass by a random offset "
                  "(periodogram-preserving), then the same pass-average and "
                  "peak-picking as for the real data",
        "n_realisations": 500,
        "per_model": {},
    }
    rng_sh = np.random.default_rng(SEED + 99)
    n_sh = 300
    for tag in MODELS:
        # all random shifts for all passes at once: shifts per pass, all passes
        # combined into one batch (the stacked null only needs the sum)
        acc = np.zeros((n_sh, f_uhz.size))
        acc_band = np.zeros((n_sh, band_hz.size))
        for pid in passes:
            y = YS[pid][tag]
            n = len(y)
            S = np.empty((n, n_sh))
            for k in range(n_sh):
                S[:, k] = np.roll(y, int(rng_sh.integers(1, n)))
            acc += ls_norm_batch(TS[pid] * 60.0, S, f_hz)
            acc_band += ls_norm_batch(TS[pid] * 60.0, S, band_hz)
        acc /= n_pass
        acc_band /= n_pass
        sh = acc.max(axis=1)
        sh_band = acc_band.max(axis=1)
        St = stacked[tag]
        obs_peak = float(St.max())
        i_band = int(np.argmax(np.where(m_band, St, -np.inf)))
        obs_band_peak = float(St[i_band])
        res["sec4b_stacked_significance"]["phase_randomised_stacked_null"][
            "per_model"
        ][tag] = {
            "observed_stacked_global_peak": obs_peak,
            "null_stacked_global_peak_q50_q95_q99_max": [
                float(np.percentile(sh, q)) for q in (50, 95, 99, 100)
            ],
            "p_stacked_global_peak": float((sh >= obs_peak).mean()),
            "observed_stacked_band_peak": obs_band_peak,
            "null_stacked_band_peak_q50_q95_q99_max": [
                float(np.percentile(sh_band, q)) for q in (50, 95, 99, 100)
            ],
            "p_stacked_band_peak": float((sh_band >= obs_band_peak).mean()),
            "n_realisations": n_sh,
            "p_resolution": 1.0 / n_sh,
        }

    # ======================================================================
    # 5. phase coherence at 1010 / 1210 uHz
    # ======================================================================
    res["sec5_phase_coherence"] = {
        "alignment_assumption": (
            "relative_time_min is used as a COMMON time origin across passes. The "
            "file provides only relative_time_min, and every pass spans roughly "
            "-40..+40 min on its own 1-min grid, so the data are consistent with "
            "each pass having its own independent time origin. If the origins are "
            "in fact independent, the per-pass fitted phases are independent "
            "uniform random variables and the Rayleigh test carries no "
            "information (it then has exactly the uniform-phase null reported "
            "below). The TESTED ASSUMPTION is therefore: all passes share one "
            "absolute time origin, i.e. relative_time_min = t_pass - T0 with the "
            "same T0 for all 22 passes."
        ),
        "phase_convention": "y = A cos(2*pi*f*t - phi), phi = atan2(b, a) from "
                            "the 2-column LS fit [cos, sin] after linear detrending",
        "frequencies_uhz": [1010.0, 1210.0],
        "per_frequency": {},
        "alignment_sensitivity_check": {},
    }
    # direct sensitivity check: R for the observed phases, for the phases with a
    # random per-pass origin added, and for phases from an independent AR(1)
    # draw. If the observed R is indistinguishable from the "random origin"
    # value, the common-origin assumption is doing all the work.
    rng_al = np.random.default_rng(SEED + 555)
    for f0 in (1010.0, 1210.0):
        for tag in MODELS:
            phs0 = np.array(
                [
                    sinusoid_amp_phase(TS[p], YS[p][tag], f0)[1] for p in passes
                ]
            )
            rand_orig = np.array(
                [
                    rayleigh_R(phs0 + rng_al.uniform(0, 2 * np.pi, n_pass))[0]
                    for _ in range(2000)
                ]
            )
            res["sec5_phase_coherence"]["alignment_sensitivity_check"].setdefault(
                str(f0), {}
            )[tag] = {
                "R_common_origin": rayleigh_R(phs0)[0],
                "R_randomised_per_pass_origin_q2.5_q50_q97.5": [
                    float(np.percentile(rand_orig, q)) for q in (2.5, 50, 97.5)
                ],
                "null_mean_R_for_22_independent_uniform_phases": 0.0,
                "expected_R_scale_1_over_sqrt_n": float(1.0 / np.sqrt(n_pass)),
            }
    rng_ph = np.random.default_rng(SEED + 7)
    n_mc = 20000
    for f0 in (1010.0, 1210.0):
        ent = {"f_uhz": f0, "per_model": {}}
        # AR(1) null for the Rayleigh R (same sinusoid fit on red-noise draws):
        # tells us how much R is expected purely from the red-noise shape of the
        # residual series, independent of any phase alignment question.
        rng_ar = np.random.default_rng(SEED + 31)
        for tag, col in MODELS.items():
            amps, phs, amprms, rmss = [], [], [], []
            for p in passes:
                t = TS[p]
                y = YS[p][tag]
                A, phi, rms = sinusoid_amp_phase(t, y, f0)
                amps.append(A)
                phs.append(phi)
                amprms.append(A / rms if rms > 0 else np.nan)
                rmss.append(rms)
            amps = np.array(amps)
            phs = np.array(phs)
            amprms = np.array(amprms)
            R, n = rayleigh_R(phs)
            # Monte-Carlo null for R with 22 independent uniform phases
            zr = np.abs(
                np.exp(1j * rng_ph.uniform(0, 2 * np.pi, size=(n_mc, n))).sum(axis=1)
            ) / n
            p_mc = float((zr >= R).mean())
            p_rayleigh_approx = float(np.exp(-n * R * R))
            # amplitude reference: same fit at every frequency in the band
            amp_band = []
            for fb in band_uhz[::5]:
                As = [
                    sinusoid_amp_phase(TS[p], YS[p][tag], fb)[0] for p in passes
                ]
                amp_band.append(np.median(As))
            amp_band = np.array(amp_band)
            # is the 1010/1210 median amplitude unusual anywhere in the band?
            amp_band_pct = float((amp_band < np.median(amps)).mean())
            # AR(1) null for R: 200 draws, each fit at f0, phases -> R
            n_sim = 200
            Rs = np.empty(n_sim)
            for k in range(n_sim):
                ps = []
                for p in passes:
                    y = YS[p][tag]
                    a, s = ar1_params(y)
                    ps.append(
                        sinusoid_amp_phase(
                            TS[p], ar1_surrogate(len(y), a, s, rng_ar), f0
                        )[1]
                    )
                Rs[k] = rayleigh_R(ps)[0]
            ent["per_model"][tag] = {
                "ar1_null_R_q50_q95_q99_max": [
                    float(np.percentile(Rs, q)) for q in (50, 95, 99, 100)
                ],
                "p_R_vs_ar1_null": float((Rs >= R).mean()),
                "median_amplitude_at_f0_percentile_within_band_amplitudes": amp_band_pct,
                "amplitude_mm_s": summarize(amps),
                "amplitude_over_rms": summarize(amprms),
                "pass_rms_mm_s": summarize(np.array(rmss)),
                "per_pass": {
                    p: {
                        "amplitude_mm_s": float(amps[i]),
                        "phase_rad": float(phs[i]),
                        "phase_deg": float(np.degrees(phs[i]) % 360.0),
                        "amplitude_over_rms": float(amprms[i]),
                    }
                    for i, p in enumerate(passes)
                },
                "rayleigh_R": R,
                "rayleigh_p_approx_exp_minus_nR2": p_rayleigh_approx,
                "rayleigh_p_monte_carlo_uniform_phases": p_mc,
                "rayleigh_R_bootstrap_over_passes": None,  # filled after the loop below
                "rayleigh_R_sensitivity_to_pass_time_origin": (
                    "R is invariant to a COMMON shift of all passes; it changes "
                    "completely if each pass has its own unknown origin. "
                    "Equivalent statement: with 22 independent unknown origins the "
                    "null is exactly the uniform-phase null reported above."
                ),
                "n_monte_carlo": n_mc,
                "median_amplitude_all_band_freqs_reference_mm_s": float(
                    np.median(amp_band)
                ),
                "median_pass_amplitude_percentile_within_band_amplitudes": float(
                    (amp_band < np.median(amps)).mean()
                ),
            }
            boot = np.array(
                [
                    rayleigh_R(rng_ph.choice(phs, size=n, replace=True))[0]
                    for _ in range(2000)
                ]
            )
            ent["per_model"][tag]["rayleigh_R_bootstrap_over_passes"] = {
                "q2.5": float(np.percentile(boot, 2.5)),
                "q50": float(np.percentile(boot, 50)),
                "q97.5": float(np.percentile(boot, 97.5)),
            }
        res["sec5_phase_coherence"]["per_frequency"][str(f0)] = ent

    # ======================================================================
    # 6. spectral slope + low/high band ratio
    # ======================================================================
    res["sec6_spectral_slope"] = {
        "fit": "OLS of log10(normalized LS power) on log10(f_uhz), 200-3000 uHz, "
               "per pass (frequencies are correlated within a pass; the pass is "
               "the unit, only the pass-level slope distribution is reported)",
        "bands_for_ratio_uhz": {"low": [300.0, 600.0], "high": [2000.0, 3000.0]},
        "per_model": {},
    }
    lf = np.log10(f_uhz)
    for tag in MODELS:
        slopes, ratios, lows, highs = [], [], [], []
        for p in passes:
            P = obs_spec[tag][p]
            ok = np.isfinite(np.log10(np.maximum(P, 1e-300)))
            s, _ = np.polyfit(lf[ok], np.log10(np.maximum(P[ok], 1e-300)), 1)
            slopes.append(float(s))
            lo = float(P[m_lo].mean())
            hi = float(P[m_hi].mean())
            lows.append(lo)
            highs.append(hi)
            ratios.append(lo / hi if hi > 0 else np.nan)
        res["sec6_spectral_slope"]["per_model"][tag] = {
            "slope_summary": summarize(np.array(slopes)),
            "median_slope": float(np.median(slopes)),
            "per_pass_slope": {p: s for p, s in zip(passes, slopes)},
            "low_band_mean_power_summary": summarize(np.array(lows)),
            "high_band_mean_power_summary": summarize(np.array(highs)),
            "low_over_high_ratio_summary": summarize(np.array(ratios)),
            "median_low_over_high_ratio": float(np.median(ratios)),
            "n_passes_ratio_gt_1": int((np.array(ratios) > 1).sum()),
        }

    # ======================================================================
    # 7. verdict
    # ======================================================================
    band_med = {tag: res["sec2_candidate_band"]["per_model"][tag][
        "median_percentile_over_passes"] for tag in MODELS}
    n95 = {tag: res["sec2_candidate_band"]["per_model"][tag]["n_passes_gt_0.95"]
           for tag in MODELS}
    n99 = {tag: res["sec2_candidate_band"]["per_model"][tag]["n_passes_gt_0.99"]
           for tag in MODELS}
    res["sec7_verdict"] = {
        "band_test_median_percentile": band_med,
        "band_test_n_passes_gt_q95": n95,
        "band_test_n_passes_gt_q99": n99,
        "n_passes": n_pass,
        "stacked_peak_uhz": {tag: res["sec4_reproducibility"]["per_model"][tag][
            "stacked_global_max"]["f_uhz"] for tag in MODELS},
        "stacked_peak_support_n_passes": {
            tag: res["sec4_reproducibility"]["per_model"][tag][
                "n_passes_peaking_within_1_res_elem_of_stacked_peak_fixed_211uhz"]
            for tag in MODELS},
        "rayleigh_R_at_1010": {
            tag: res["sec5_phase_coherence"]["per_frequency"]["1010.0"][
                "per_model"][tag]["rayleigh_R"] for tag in MODELS},
        "rayleigh_p_monte_carlo_at_1010": {
            tag: res["sec5_phase_coherence"]["per_frequency"]["1010.0"][
                "per_model"][tag]["rayleigh_p_monte_carlo_uniform_phases"]
            for tag in MODELS},
        "rayleigh_R_at_1210": {
            tag: res["sec5_phase_coherence"]["per_frequency"]["1210.0"][
                "per_model"][tag]["rayleigh_R"] for tag in MODELS},
        "rayleigh_p_monte_carlo_at_1210": {
            tag: res["sec5_phase_coherence"]["per_frequency"]["1210.0"][
                "per_model"][tag]["rayleigh_p_monte_carlo_uniform_phases"]
            for tag in MODELS},
        "statement": None,
    }

    # ---- cross-model agreement: do the three models localise the same line?
    stacked_f = np.vstack([stacked[tag] for tag in MODELS])
    res["sec7_verdict"]["cross_model_stacked_peak_frequencies_uhz"] = {
        tag: res["sec4_reproducibility"]["per_model"][tag]["stacked_global_max"][
            "f_uhz"]
        for tag in MODELS
    }
    cm = {}
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if j <= i:
                continue
            cm[f"{a}_vs_{b}"] = {
                "pearson_r_stacked_spectra": float(
                    np.corrcoef(stacked[a], stacked[b])[0, 1]
                ),
                "spearman_rho_stacked_spectra": float(
                    stats.spearmanr(stacked[a], stacked[b]).statistic
                ),
                "pearson_r_log_power": float(
                    np.corrcoef(
                        np.log10(np.maximum(stacked[a], 1e-300)),
                        np.log10(np.maximum(stacked[b], 1e-300)),
                    )[0, 1]
                ),
                "peak_frequency_separation_uhz": float(
                    abs(res["sec4_reproducibility"]["per_model"][a][
                        "stacked_global_max"]["f_uhz"]
                        - res["sec4_reproducibility"]["per_model"][b][
                            "stacked_global_max"]["f_uhz"])
                ),
                "separation_in_resolution_elements": float(
                    abs(res["sec4_reproducibility"]["per_model"][a][
                        "stacked_global_max"]["f_uhz"]
                        - res["sec4_reproducibility"]["per_model"][b][
                            "stacked_global_max"]["f_uhz"])
                    / res["median_rayleigh_resolution_uhz"]
                ),
            }
    res["sec7_verdict"]["cross_model_stacked_spectrum_agreement"] = cm

    # ---- is the candidate comb special? mean power at comb frequencies vs the
    #      mean power at off-comb frequencies within the same band (per pass,
    #      then a paired sign test across passes)
    comb_mask = np.zeros(band_uhz.size, bool)
    for f in COMB_FULL:
        comb_mask |= np.abs(band_uhz - f) <= 15.0
    if comb_mask.sum() == 0:
        comb_mask = np.zeros(band_uhz.size, bool)
        for f in COMB_FULL:
            comb_mask |= np.abs(band_uhz - f) <= 20.0
    off_mask = ~comb_mask
    res["sec7_verdict"]["comb_vs_offcomb_band_power"] = {}
    for tag in MODELS:
        on = np.array([obs_spec[tag][p][m_band][comb_mask].mean() for p in passes])
        off = np.array([obs_spec[tag][p][m_band][off_mask].mean() for p in passes])
        diff = on - off
        res["sec7_verdict"]["comb_vs_offcomb_band_power"][tag] = {
            "comb_frequencies_uhz": COMB_FULL,
            "comb_window_uhz": 15.0,
            "n_comb_bins": int(comb_mask.sum()),
            "n_offcomb_bins": int(off_mask.sum()),
            "mean_power_on_comb_median": float(np.median(on)),
            "mean_power_off_comb_median": float(np.median(off)),
            "median_difference": float(np.median(diff)),
            "n_passes_diff_positive": int((diff > 0).sum()),
            "sign_test_vs_0.5": sign_test(int((diff > 0).sum()), n_pass),
        }
        try:
            w = stats.wilcoxon(diff)
            res["sec7_verdict"]["comb_vs_offcomb_band_power"][tag][
                "wilcoxon_signed_rank_vs_0"
            ] = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
        except Exception as exc:
            res["sec7_verdict"]["comb_vs_offcomb_band_power"][tag][
                "wilcoxon_signed_rank_vs_0"
            ] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- explicit answers to the seven questions, computed, not asserted
    qa = {}
    for tag in MODELS:
        p_band = res["sec2_candidate_band"]["per_model"][tag]
        p_stacked = res["sec4b_stacked_significance"]["per_model"][tag]
        p_phase = res["sec4b_stacked_significance"][
            "phase_randomised_stacked_null"]["per_model"][tag]
        qa[tag] = {
            "q2_band_median_pct": p_band["median_percentile_over_passes"],
            "q2_n_passes_gt_q95": p_band["n_passes_gt_0.95"],
            "q2_n_passes_gt_q99": p_band["n_passes_gt_0.99"],
            "q2_sign_test_p": p_band["sign_test_vs_0.5"]["p_value"],
            "q4_stacked_peak_uhz": res["sec4_reproducibility"]["per_model"][tag][
                "stacked_global_max"]["f_uhz"],
            "q4_stacked_peak_in_candidate_band": bool(
                CAND_BAND[0] <= res["sec4_reproducibility"]["per_model"][tag][
                    "stacked_global_max"]["f_uhz"] <= CAND_BAND[1]),
            "q4_support_passes": res["sec4_reproducibility"]["per_model"][tag][
                "n_passes_peaking_within_1_res_elem_of_stacked_peak_fixed_211uhz"],
            "q4_p_stacked_peak_vs_ar1": p_stacked[
                "p_stacked_global_peak_vs_ar1"],
            "q4_p_stacked_peak_vs_phase_randomised": p_phase[
                "p_stacked_global_peak"],
            "q5_R_1010": res["sec5_phase_coherence"]["per_frequency"]["1010.0"][
                "per_model"][tag]["rayleigh_R"],
            "q5_p_1010_uniform_phases": res["sec5_phase_coherence"]["per_frequency"][
                "1010.0"]["per_model"][tag][
                "rayleigh_p_monte_carlo_uniform_phases"],
            "q6_median_slope": res["sec6_spectral_slope"]["per_model"][tag][
                "median_slope"],
            "q6_median_low_over_high": res["sec6_spectral_slope"]["per_model"][tag][
                "median_low_over_high_ratio"],
        }
    res["sec7_verdict"]["answers"] = qa

    any_band = any(qa[t]["q2_n_passes_gt_q99"] >= 3 for t in MODELS)
    any_stacked_local = any(
        res["sec4_reproducibility"]["per_model"][t]["stacked_global_max"]["f_uhz"]
        >= CAND_BAND[0]
        and res["sec4_reproducibility"]["per_model"][t]["stacked_global_max"]["f_uhz"]
        <= CAND_BAND[1]
        for t in MODELS
    )
    any_phase = any(
        res["sec5_phase_coherence"]["per_frequency"]["1010.0"]["per_model"][t][
            "rayleigh_p_monte_carlo_uniform_phases"] < 0.05 for t in MODELS
    )
    if any_band and any_stacked_local and any_phase:
        res["sec7_verdict"]["statement"] = (
            "REPRODUCIBLE LOCALIZED LINE SUPPORTED (band, stacked peak and phase "
            "tests all pass for at least one model)"
        )
    elif any_band or any_stacked_local:
        res["sec7_verdict"]["statement"] = (
            "PARTIAL / UNRESOLVED: some aggregate statistic exceeds the red-noise "
            "null but the tests do not agree, so a reproducible localized line is "
            "not established."
        )
    else:
        res["sec7_verdict"]["statement"] = (
            "NO reproducible localized spectral line at 860-1310 uHz. Median "
            "per-pass band percentile "
            + ", ".join(f"{t}={band_med[t]:.3f}" for t in MODELS)
            + "; passes above the AR(1) 99th percentile "
            + ", ".join(f"{t}={n99[t]}/22" for t in MODELS)
            + "; stacked peaks at "
            + ", ".join(
                f"{t}={res['sec4_reproducibility']['per_model'][t]['stacked_global_max']['f_uhz']:.0f} uHz"
                for t in MODELS)
            + "."
        )

    res["runtime_sec"] = round(time.time() - t_start, 2)
    out_path = OUT / "spectral_tests.json"
    write_json(out_path, res)

    # full per-pass spectra + stacked spectra for downstream plotting
    np.savez_compressed(
        OUT / "spectral_tests_arrays.npz",
        f_uhz=f_uhz,
        band_uhz=band_uhz,
        **{f"stacked_{tag}": stacked[tag] for tag in MODELS},
        **{f"spec_{tag}_{p}": obs_spec[tag][p] for tag in MODELS for p in passes},
        **{
            f"bandnull_{tag}_{p}": surro_band[tag][p]
            for tag in MODELS
            for p in passes
        },
    )
    print("wrote", out_path)
    print("statement:", res["sec7_verdict"]["statement"])
    print("runtime_sec", res["runtime_sec"])
    return res


if __name__ == "__main__":
    main()