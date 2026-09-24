#!/usr/bin/env python3
"""
Dimension tag: s10  (drift-rate re-derivation from pixel data)
Task: Astronomy_01_radio_burst_drift  (T0 = 2022-10-13)

Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n = 5 events).
Pixels are nested repeats, never treated as independent samples.

Loads the five normalized dynamic spectra and re-derives drift rates with
several independent pixel-level methods, then compares to the reported values.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage, stats

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
NV = os.path.join(ROOT, "data", "normalized_views")
RES = os.path.join(ROOT, "output", "results")
os.makedirs(RES, exist_ok=True)

EPS = 1.0e-9  # background is exactly 1.0


def load_event(i):
    df = pd.read_csv(os.path.join(NV, f"event_{i}_dynamic_spectrum.tsv"), sep="\t")
    f = df.iloc[:, 0].to_numpy(dtype=float)
    tnames = list(df.columns[1:])
    t = np.array([float(s.replace("time_s_", "")) for s in tnames], dtype=float)
    Z = df.iloc[:, 1:].to_numpy(dtype=float)
    return f, t, Z


def theil_sen(x, y):
    """Theil-Sen slope + intercept and a slope stderr estimate."""
    res = stats.theilslopes(y, x, 0.95)
    slope, intercept, lo, hi = res[0], res[1], res[2], res[3]
    # stderr from the 95% CI width (normal approx)
    se = (hi - lo) / (2 * 1.959964)
    return slope, intercept, se, lo, hi


def huber_fit(x, y, sigma=1.5, iters=50):
    X = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        s = s if s > 0 else 1e-12
        u = np.abs(r) / s
        w = np.ones_like(u)
        m = u > sigma
        w[m] = sigma / u[m]
        W = np.sqrt(w)[:, None]
        beta = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)[0]
    r = y - X @ beta
    return beta[0], beta[1], r


def main():
    meta = pd.read_csv(os.path.join(NV, "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    print("=" * 78)
    print("s10: drift-rate re-derivation from pixel data  (unit = one event, n=5)")
    print("=" * 78)
    print(meta.to_string(index=False))
    print()

    grid_rows = []
    centroid_rows = []   # per-event column centroids (for sub-structure tests)
    results = []

    for _, m in meta.iterrows():
        eid = m["event_id"]
        i = int(eid.split("_")[1])
        f, t, Z = load_event(i)
        nF, nT = Z.shape
        df_axis = np.diff(f)
        dt_axis = np.diff(t)
        assert np.all(df_axis < 0), "frequency axis not descending"
        assert np.all(dt_axis > 0), "time axis not ascending"
        df = float(np.median(np.abs(df_axis)))
        dt = float(np.median(dt_axis))
        span_f = float(f[0] - f[-1])
        span_t = float(t[-1] - t[0])
        reported = float(m["reported_event_drift_rate_MHz_s"])

        S = Z - 1.0                       # background-subtracted intensity
        S[S < 0] = 0.0
        w_tot = S.sum()

        # ---------- 2. signal characterisation (per event) ----------
        frac = {}
        for thr in [EPS, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5 - EPS]:
            frac[f"frac_gt_1+{thr:.2f}"] = float(np.mean(Z > 1.0 + thr))
        mask = Z > 1.0 + EPS
        n_sig = int(mask.sum())
        sat = int(np.sum(Z >= 1.5 - 1e-9))
        print(f"--- {eid} ---  grid {nF}x{nT}, df={df:.6f} MHz, dt={dt:.6f} s")
        print(f"    f: {f[0]:.4f} -> {f[-1]:.4f} MHz (descending), span {span_f:.4f}")
        print(f"    t: {t[0]:.4f} -> {t[-1]:.4f} s, span {span_t:.4f} s")
        print(f"    reported drift = {reported:.4f} MHz/s")
        print(f"    signal pixels (>1.0+eps): {n_sig} = {100.0*n_sig/Z.size:.2f}% of grid")
        print(f"    saturated pixels (==1.5): {sat} ({100.0*sat/Z.size:.3f}%)")
        print("    occupied fraction by threshold: " +
              ", ".join(f"{k.split('+')[-1]}:{100*v:.2f}%" for k, v in frac.items()))
        print(f"    total background-subtracted intensity = {w_tot:.2f}")
        print(f"    max pixel = {Z.max():.4f}, min pixel = {Z.min():.4f}")

        # ---------- 3a. per-column intensity-weighted mean frequency ----------
        col_cnt = mask.sum(axis=0)
        col_w = S.sum(axis=0)
        cen = np.full(nT, np.nan)
        for j in range(nT):
            if col_w[j] > 0:
                cen[j] = np.sum(S[:, j] * f) / col_w[j]
        good = np.isfinite(cen)
        # analysis subset: columns with meaningful signal
        thr_w = 0.05 * np.nanmax(col_w)
        good_w = good & (col_w >= max(thr_w, 1e-9))
        print(f"    columns with any signal: {int(good.sum())}/{nT}; "
              f"with >=5% of peak weight: {int(good_w.sum())}")

        # Method A: Theil-Sen robust linear fit of centroid vs time (all signal cols)
        sA, bA, seA, loA, hiA = theil_sen(t[good], cen[good])
        # Method A': same but only well-populated columns
        sAw, bAw, seAw, loAw, hiAw = theil_sen(t[good_w], cen[good_w])
        # Method A'': Huber
        sAh, bAh, rAh = huber_fit(t[good], cen[good])

        # ---------- 3b. intensity-weighted 2-D second moment / PCA ----------
        mask_good = mask
        # Use signal mask pixels for the "mask PCA"
        fy, tx = np.nonzero(mask_good)
        fy = f[fy]
        tx = t[tx]
        mx, my = tx.mean(), fy.mean()
        cov = np.cov(np.vstack([tx - mx, fy - my]))
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, np.argmax(evals)]      # principal axis
        sB = v[1] / v[0]
        # eigenvalue-ratio: elongation of the signal cloud
        elong = float(np.max(evals) / max(np.min(evals), 1e-30))

        # weighted PCA (weights = background-subtracted intensity)
        wy = np.repeat((S * mask).sum(axis=1), mask.sum(axis=1)) if False else None
        rows, cols = np.nonzero(mask)
        wts = S[rows, cols]
        W = wts / wts.sum()
        mwx = np.sum(W * t[cols])
        mwy = np.sum(W * f[rows])
        cxx = np.sum(W * (t[cols] - mwx) ** 2)
        cyy = np.sum(W * (f[rows] - mwy) ** 2)
        cxy = np.sum(W * (t[cols] - mwx) * (f[rows] - mwy))
        evals_w, evecs_w = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
        vw = evecs_w[:, np.argmax(evals_w)]
        sBw = vw[1] / vw[0]
        elong_w = float(max(evals_w) / max(min(evals_w), 1e-30))

        # ---------- 3c. slope scan (Radon-like) maximising contained signal ----------
        slopes = np.linspace(-40.0, 5.0, 9001)   # MHz/s
        band = 0.35  # MHz half-width (approx 16.5 bins)
        # vectorised band-containment via cumulative sums on the ascending-f grid
        f_asc = f[::-1].copy()
        S_asc = S[::-1, :].copy()
        C = np.vstack([np.zeros((1, nT)), np.cumsum(S_asc, axis=0)])  # (nF+1, nT)
        Ftr = mwy + slopes[:, None] * (t[None, :] - mwx)              # (nS, nT)
        ia = np.clip(np.searchsorted(f_asc, Ftr - band), 0, nF)
        ib = np.clip(np.searchsorted(f_asc, Ftr + band, side="right"), 0, nF)
        jj = np.broadcast_to(np.arange(nT)[None, :], Ftr.shape)
        scan = (C[ib, jj] - C[ia, jj]).sum(axis=1)
        kbest = int(np.argmax(scan))
        best = (float(slopes[kbest]), float(scan[kbest]))
        sC = float(best[0])
        # parabolic refinement around the maximum
        kb = kbest
        if 0 < kb < len(slopes) - 1:
            y0, y1, y2 = scan[kb - 1], scan[kb], scan[kb + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                sC = float(slopes[kb] + delta * (slopes[1] - slopes[0]))
        # ------------- Method D: argmax-ridge tracking (independent of weights) ---
        ridx = np.argmax(Z, axis=0)
        rid_f = f[ridx]
        rid_ok = col_cnt > 0
        # restrict to columns with a real peak (max > 1.05)
        peak_ok = (Z.max(axis=0) > 1.05) & rid_ok
        sD, bD, seD, loD, hiD = theil_sen(t[peak_ok], rid_f[peak_ok])
        sDh, bDh, rDh = huber_fit(t[peak_ok], rid_f[peak_ok])

        # ---------- 5. sub-structure: connected components ----------
        struct8 = np.ones((3, 3), dtype=int)
        lab, ncc = ndimage.label(mask, structure=struct8)
        sizes = np.bincount(lab.ravel())[1:] if ncc else np.array([])
        order = np.argsort(sizes)[::-1]
        top = sizes[order][:6]
        # components carrying >=1% of total weight
        comp_w = np.zeros(ncc + 1)
        for c in range(1, ncc + 1):
            comp_w[c] = S[lab == c].sum()
        big = np.where(comp_w[1:] >= 0.01 * max(w_tot, 1e-9))[0] + 1
        n_big = int(len(big))
        # peak count in the time-integrated spectrum
        prof = S.sum(axis=1)
        _, n_freq_peaks = ndimage.label(prof > 0.5 * prof.max())
        n_freq_peaks = int(n_freq_peaks)
        # per-column multi-modality: number of columns whose vertical profile has
        # >=2 separated runs above 50% of that column's max
        n_multi = 0
        for j in range(nT):
            if col_w[j] <= 0:
                continue
            p = S[:, j]
            if p.max() <= 0:
                continue
            above = p > 0.5 * p.max()
            if above.sum() == 0:
                continue
            n_runs = int(ndimage.label(above)[1])
            if n_runs >= 2:
                n_multi += 1
        # residual scatter of the centroid about the robust line (single-ridge test)
        resid = cen[good] - (sA * t[good] + bA)
        med_r = float(np.median(np.abs(resid)))
        mad_r = float(1.4826 * med_r)
        # band occupancy: number of frequency bins between the 5th and 95th pct of
        # the signal-mask frequencies per column, median over columns
        band_f = []
        for j in range(nT):
            fj = f[mask[:, j]]
            if fj.size >= 5:
                band_f.append(np.percentile(fj, 95) - np.percentile(fj, 5))
        med_band = float(np.median(band_f)) if band_f else float("nan")

        print(f"    centroid slope (Theil-Sen, all cols)   = {sA:9.4f} MHz/s  "
              f"+-{seA:.4f}  (95%CI {loA:.3f},{hiA:.3f})")
        print(f"    centroid slope (Theil-Sen, >=5% cols)   = {sAw:9.4f} MHz/s  "
              f"+-{seAw:.4f}")
        print(f"    centroid slope (Huber, all cols)        = {sAh:9.4f} MHz/s")
        print(f"    mask-PCA slope                          = {sB:9.4f} MHz/s  "
              f"(axis elongation {elong:.1f})")
        print(f"    weighted-PCA slope                      = {sBw:9.4f} MHz/s  "
              f"(elongation {elong_w:.1f})")
        print(f"    slope-scan (band +-{band} MHz)           = {sC:9.4f} MHz/s  "
              f"(contained {100*best[1]/w_tot:.1f}% of weight)")
        print(f"    argmax-ridge Theil-Sen                  = {sD:9.4f} MHz/s  "
              f"+-{seD:.4f}   (Huber {sDh:.4f})")
        print(f"    connected components (8-conn, >1.0)     = {ncc}  "
              f"(top sizes {list(top.astype(int))})")
        print(f"    components >=1% of weight               = {n_big}")
        print(f"    separated peaks in time-integrated spec = {n_freq_peaks}")
        print(f"    columns with >=2 vertical runs (>50%max)= {n_multi}/{nT}")
        print(f"    centroid residual MAD about fit         = {mad_r:.4f} MHz "
              f"(median |resid| {med_r:.4f})")
        print(f"    median per-column 5-95% frequency band  = {med_band:.4f} MHz")
        print()

        row = dict(
            event_id=eid, reported_drift=reported,
            slope_centroid_theilsen=sA, se_centroid_theilsen=seA,
            ci_lo_centroid=loA, ci_hi_centroid=hiA,
            slope_centroid_theilsen_goodcols=sAw, se_goodcols=seAw,
            slope_centroid_huber=sAh,
            slope_mask_pca=sB, mask_pca_elongation=elong,
            slope_weighted_pca=sBw, weighted_pca_elongation=elong_w,
            slope_scan=sC, slope_scan_contained_frac=float(best[1] / w_tot),
            slope_argmax_theilsen=sD, se_argmax=seD, slope_argmax_huber=sDh,
            n_signal_pixels=n_sig, frac_grid_signal=float(n_sig / Z.size),
            n_saturated=sat, total_weight=float(w_tot),
            n_cols_any_signal=int(good.sum()), n_cols_ge5pct=int(good_w.sum()),
            n_components_8conn=ncc,
            comp_sizes_top6=[int(x) for x in top],
            n_components_ge1pct_weight=n_big,
            n_ti_peaks=n_freq_peaks, n_multirun_columns=n_multi,
            centroid_resid_mad=mad_r, median_col_band_p95_p05=med_band,
            df_MHz=df, dt_s=dt, span_f_MHz=span_f, span_t_s=span_t,
            **frac,
        )
        results.append(row)
        grid_rows.append(dict(event_id=eid, **{k: v for k, v in frac.items()}))
        centroid_rows.append(pd.DataFrame(dict(
            event_id=eid, time_s=t, col_weight=col_w, col_count=col_cnt,
            centroid_MHz=cen, argmax_MHz=f[ridx],
            fitA_MHz=sA * t + bA, in_fitA=good, resid_MHz=cen - (sA * t + bA))))

    R = pd.DataFrame(results)
    # -------- 4. comparison to reported values --------
    for meth in ["slope_centroid_theilsen", "slope_centroid_theilsen_goodcols",
                 "slope_centroid_huber", "slope_mask_pca", "slope_weighted_pca",
                 "slope_scan", "slope_argmax_theilsen"]:
        R[f"diff_{meth}"] = R[meth] - R["reported_drift"]
        R[f"relerr_{meth}"] = R[f"diff_{meth}"] / R["reported_drift"]
    print("=" * 78)
    print("COMPARISON: re-derived minus reported (MHz/s) and relative error")
    print("=" * 78)
    cols = ["event_id", "reported_drift", "slope_centroid_theilsen",
            "diff_slope_centroid_theilsen", "relerr_slope_centroid_theilsen",
            "slope_mask_pca", "diff_slope_mask_pca", "relerr_slope_mask_pca",
            "slope_scan", "slope_argmax_theilsen"]
    pd.set_option("display.width", 250)
    print(R[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    summary = {}
    for meth in ["slope_centroid_theilsen", "slope_mask_pca", "slope_scan",
                 "slope_argmax_theilsen"]:
        rel = R[f"relerr_{meth}"].to_numpy()
        summary[meth] = dict(
            max_abs_rel_err=float(np.max(np.abs(rel))),
            median_abs_rel_err=float(np.median(np.abs(rel))),
            median_diff=float(np.median(R[f"diff_{meth}"])),
            all_within_10pct=bool(np.all(np.abs(rel) < 0.10)),
            within_10pct_count=int(np.sum(np.abs(rel) < 0.10)),
        )
    print("Summary across methods (unit = 5 events):")
    print(json.dumps(summary, indent=2))

    R.to_csv(os.path.join(RES, "s10_drift_rederived_per_event.csv"), index=False)
    with open(os.path.join(RES, "s10_drift_rederived_per_event.json"), "w") as fh:
        json.dump(dict(
            analysis_unit="one complete dynamic-spectrum event (n=5 events)",
            methods={
                "A": "per-time-column intensity-weighted mean frequency, "
                     "Theil-Sen robust linear fit vs time",
                "A2": "same on columns carrying >=5% of peak column weight",
                "A3": "Huber M-estimator fit of centroid vs time",
                "B": "unweighted signal-mask PCA principal-axis slope (df/dt)",
                "B2": "intensity-weighted PCA principal-axis slope",
                "C": "slope scan (Radon-like) maximising background-subtracted "
                     "signal within +-0.35 MHz of a straight ridge",
                "D": "per-column argmax frequency, Theil-Sen fit",
            },
            per_event=R.to_dict(orient="records"),
            method_summary=summary,
        ), fh, indent=2)
    pd.concat(centroid_rows).to_csv(
        os.path.join(RES, "s10_per_column_centroids.csv"), index=False)
    print("\nwrote", os.path.join(RES, "s10_drift_rederived_per_event.csv"))


if __name__ == "__main__":
    main()