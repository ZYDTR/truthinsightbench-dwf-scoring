#!/usr/bin/env python3
"""
Dimension tag: s17-v2  (ROBUSTNESS of dynamic-spectrum drift results + the
                        DECISIVE test of recoverability of the reported rates)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).
Pixels and adjacent directional coordinates are NESTED REPEATS, never an
event-level sample size.

v2 fixes a fatal flaw of the v1 line scan: a straight line with a steep negative
slope anchored at the bottom-left corner (f0 = f_min at t0) exits the 8-17 MHz
grid immediately, so every sampled column was degenerate/Nan.  v2 uses a
coverage-matched Radon/Hough formulation instead:

  R(m, phase) = mean background-subtracted intensity along the line
                f(t) = phase + m*(t - t0), averaged over the time columns where
                that line is inside the frequency grid.
  H(m)        = max over phase of R(m, phase)   [best-phase detector]
                restricted to phases where >= 25% of the time columns contribute.

  The DECISIVE statistic is the percentile of the REPORTED slope in the set
  {H(m)} over 401 slopes spanning -40..+40 MHz/s, plus a length-matched subset.
  A calibration control injects a synthetic ridge with the reported slope and
  amplitude A into the REAL image and re-runs the whole pipeline, which measures
  the power of the estimator to recover the reported rate if it were present.

Also: threshold sweep of a fixed pixel estimator, connected components /
multi-run structure, full leave-one-out over the 5 events, and figures.
Memory discipline: one 425x425 grid in memory at a time, freed after use.
"""
import gc
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy import ndimage, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
NV = os.path.join(ROOT, "data", "normalized_views")
RES = os.path.join(ROOT, "output", "results")
FIG = os.path.join(RES, "figures")
LOG = os.path.join(ROOT, "output", "logs")
for d in (RES, FIG, LOG):
    os.makedirs(d, exist_ok=True)

EPS = 1.0e-9
THRESHOLDS = [("1.00+eps", 1.0 + EPS), ("1.01", 1.01), ("1.02", 1.02),
              ("1.05", 1.05), ("1.10", 1.10), ("1.20", 1.20)]
THR_PRIMARY = "1.05"
SCAN_THRESHOLDS = ["1.01", "1.05", "1.20"]
SLOPES = np.linspace(-40.0, 40.0, 401)   # 0.2 MHz/s steps (reported slopes are
                                         # off-grid, so the nearest node is used
                                         # and that node's slope is reported back)
MIN_COL_FRAC = 0.25          # min fraction of time columns on the line
MATCH_TOL = 0.30             # +-30% length matching for the matched percentile
INJ_AMPS = [0.02, 0.05, 0.10, 0.25]
INJ_SIGMA_MHZ = 0.40
ERRORS = []


def log(msg=""):
    print(msg, flush=True)


def load_event(i):
    df = pd.read_csv(os.path.join(NV, f"event_{i}_dynamic_spectrum.tsv"), sep="\t")
    f = df.iloc[:, 0].to_numpy(dtype=float)
    t = np.array([float(s.replace("time_s_", "")) for s in df.columns[1:]], dtype=float)
    Z = df.iloc[:, 1:].to_numpy(dtype=float)
    return f, t, Z


def theil_sen(x, y):
    res = stats.theilslopes(y, x, 0.95)
    slope, intercept, lo, hi = res[0], res[1], res[2], res[3]
    se = (hi - lo) / (2.0 * 1.959964)
    p = float(2.0 * stats.norm.sf(abs(slope) / se)) if se > 0 else np.nan
    return float(slope), float(intercept), float(se), float(lo), float(hi), p


def rate_estimate(Z, f, t, thr):
    """Fixed simple estimator: intensity-weighted mean frequency per time column,
    then robust Theil-Sen slope vs time.  Returns dict."""
    mask = Z > thr
    S = np.where(mask, Z - 1.0, 0.0)
    nT = Z.shape[1]
    colw = S.sum(axis=0)
    good = colw > 0
    cen = np.full(nT, np.nan)
    if good.any():
        cen[good] = (S[:, good] * f[:, None]).sum(axis=0) / colw[good]
    ncol = int(good.sum())
    if ncol >= 3 and np.unique(cen[good]).size > 2:
        sl, ic, se, lo, hi, p = theil_sen(t[good], cen[good])
        rho, sp = stats.spearmanr(t[good], cen[good])
    else:
        sl = ic = se = lo = hi = p = np.nan
        rho = sp = np.nan
    return dict(mask=mask, S=S, cen=cen, good=good, rate=sl, inter=ic, se=se,
                ci_lo=lo, ci_hi=hi, p_ts=p, rho=float(rho), p_rho=float(sp),
                n_cols=ncol)


def radon_scan(S_asc, f_asc, t, slopes, df, min_col_frac=MIN_COL_FRAC):
    """Coverage-matched Radon/Hough scan.

    For every slope m, shear the image by phase = f - m*(t - t0) and compute the
    mean background-subtracted intensity of every phase bin, averaged over the
    time columns that contribute to it.  H(m) is the maximum over phase bins that
    are supported by >= min_col_frac * nT time columns.

    Returns (H, argphase, support_frac_at_best, R_rows) where R_rows is a compact
    list of (m, phase_index, mean_value, n_columns) for the phase curve.
    """
    nF, nT = S_asc.shape
    f0 = float(f_asc[0])
    nph = nF
    jj = np.arange(nT, dtype=np.int64)[None, :]
    H = np.full(slopes.size, np.nan)
    argph = np.full(slopes.size, -1, dtype=int)
    supp = np.full(slopes.size, np.nan)
    curves = []
    for k, m in enumerate(slopes):
        idx = np.rint((f_asc[:, None] - m * (t[None, :] - t[0]) - f0) / df).astype(np.int64)
        ok = (idx >= 0) & (idx < nph)
        flat = idx[ok] * nT + np.broadcast_to(jj, idx.shape)[ok]
        vals = S_asc[ok]
        s = np.bincount(flat, weights=vals, minlength=nph * nT).astype(np.float64)
        c = np.bincount(flat, minlength=nph * nT).astype(np.float64)
        Ssum = s.reshape(nph, nT).sum(axis=1)
        Csum = c.reshape(nph, nT).sum(axis=1)
        meanph = np.full(nph, np.nan)
        nzc = Csum > 0
        # FIXED denominator nT, not the data-dependent Csum: a data-dependent
        # denominator makes the statistic non-monotone in the image (adding
        # signal can add empty/low bins and LOWER the mean), which invalidates
        # any percentile or injection argument.  With nT the statistic is a
        # monotone (linear) functional of the image.
        meanph[nzc] = Ssum[nzc] / float(nT)
        okph = Csum >= min_col_frac * nT
        if okph.any():
            vals_ok = np.where(okph, meanph, -np.inf)
            kb = int(np.argmax(vals_ok))
            H[k] = float(meanph[kb])
            argph[k] = kb
            supp[k] = float(Csum[kb] / nT)
        curves.append((meanph, Csum))
    return H, argph, supp, curves


def pct_of(score_null, value):
    finite = np.isfinite(score_null)
    return float(stats.percentileofscore(score_null[finite], value, kind="mean"))


def main():
    meta = pd.read_csv(os.path.join(NV, "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    log("=" * 88)
    log("s17-v2  ROBUSTNESS + DECISIVE REPORTED-SLOPE RECOVERY TEST (Radon/Hough)")
    log("ANALYSIS UNIT = one complete dynamic-spectrum event (n_events = 5)")
    log("=" * 88)
    log(meta.to_string(index=False))
    log()

    rate_rows, comp_rows, scan_summary, inj_rows, block_rows = [], [], [], [], []
    store = {}

    for _, mrow in meta.iterrows():
        eid = mrow["event_id"]
        i = int(eid.split("_")[1])
        reported = float(mrow["reported_event_drift_rate_MHz_s"])
        f, t, Z = load_event(i)
        nF, nT = Z.shape
        df = float(np.median(np.abs(np.diff(f))))
        dt = float(np.median(np.diff(t)))
        fmin, fmax = float(f.min()), float(f.max())
        t0, t1 = float(t[0]), float(t[-1])
        Dt, Df = t1 - t0, fmax - fmin
        f_asc = f[::-1].copy()
        rep_cross_frac = min(1.0, Df / (abs(reported) * Dt))
        log("#" * 88)
        log(f"### {eid}  grid {nF}x{nT}  f {fmin:.4f}..{fmax:.4f} MHz (Df={Df:.4f})"
            f"  t {t0:.4f}..{t1:.4f} s (Dt={Dt:.4f})")
        log(f"### reported drift {reported:.4f} MHz/s -> implied ridge displacement"
            f" |m|*Dt = {abs(reported)*Dt:.3f} MHz = {100*abs(reported)*Dt/Df:.1f}%"
            f" of the total band; implied band-crossing time {Df/abs(reported):.3f} s"
            f" = {100*rep_cross_frac:.1f}% of the window")

        st = dict(event_id=eid, reported=reported, f=f, t=t, Dt=Dt, Df=Df,
                  Z=Z.astype(np.float32).copy(), thr={}, radon={}, inj={})
        st["rep_cross_frac"] = rep_cross_frac

        for lab, thr in THRESHOLDS:
            r = rate_estimate(Z, f, t, thr)
            mask, S = r["mask"], r["S"]
            nmask = int(mask.sum())
            frac_grid = float(nmask / Z.size)
            dif = r["rate"] - reported
            relerr = dif / reported

            lab_img, ncc = ndimage.label(mask, structure=np.ones((3, 3), int))
            sizes = np.sort(np.bincount(lab_img.ravel())[1:])[::-1] if ncc else np.array([], int)
            runs_list = np.zeros(nT, dtype=int)
            n_multirun = 0
            for j in range(nT):
                col = mask[:, j]
                if not col.any():
                    continue
                _, nr = ndimage.label(col)
                runs_list[j] = int(nr)
                if nr >= 2:
                    n_multirun += 1
            rr, cc = np.nonzero(mask)
            wts = S[rr, cc]
            slope_pca = elong = mwx = mwy = np.nan
            if wts.sum() > 0:
                W = wts / wts.sum()
                mwx = float(np.sum(W * t[cc]))
                mwy = float(np.sum(W * f[rr]))
                cxx = float(np.sum(W * (t[cc] - mwx) ** 2))
                cyy = float(np.sum(W * (f[rr] - mwy) ** 2))
                cxy = float(np.sum(W * (t[cc] - mwx) * (f[rr] - mwy)))
                ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
                v = evec[:, int(np.argmax(ev))]
                slope_pca = float(v[1] / v[0]) if v[0] != 0 else np.nan
                elong = float(ev.max() / max(ev.min(), 1e-30))

            rate_rows.append(dict(
                event_id=eid, threshold=lab, thr_value=thr, reported_drift=reported,
                rate_theilsen=r["rate"], se=r["se"], ci_lo=r["ci_lo"],
                ci_hi=r["ci_hi"], p_ts=r["p_ts"], diff=float(dif),
                relerr=float(relerr), spearman_rho=r["rho"], spearman_p=r["p_rho"],
                n_cols_signal=r["n_cols"], n_px_mask=nmask, frac_grid_mask=frac_grid,
                mask_weight=float(S.sum()), pca_slope=slope_pca, pca_elong=elong))
            comp_rows.append(dict(
                event_id=eid, threshold=lab, n_components=int(ncc),
                comp_sizes_top10=str([int(x) for x in sizes[:10]]),
                largest_frac=float(sizes[0] / nmask) if ncc and nmask else np.nan,
                n_multirun_cols=n_multirun, frac_multirun_cols=float(n_multirun / nT),
                max_runs_per_col=int(runs_list.max()) if nT else 0,
                frac_grid_mask=frac_grid, n_px_mask=nmask))

            # empirical displacement of the bulk centroid over the window
            cen_ok = r["good"]
            cen_range = (float(np.nanmax(r["cen"][cen_ok]) - np.nanmin(r["cen"][cen_ok]))
                         if cen_ok.sum() > 1 else np.nan)
            st["thr"][lab] = dict(thr=thr, S=S, cen=r["cen"], good=cen_ok,
                                  rate=r["rate"], ncc=int(ncc),
                                  n_multirun=n_multirun, frac_grid=frac_grid,
                                  cen_range_MHz=cen_range, slope_pca=slope_pca,
                                  elong=elong)

            if lab in SCAN_THRESHOLDS:
                H, argph, supp, curves = radon_scan(S[::-1, :], f_asc, t, SLOPES, df)
                krep = int(np.argmin(np.abs(SLOPES - reported)))
                rep_node = float(SLOPES[krep])
                Hrep = float(H[krep])
                # value at the max-chord anchor phase (line entering at the corner)
                ph_anchor = (fmax if reported < 0 else fmin) - fmin
                ph_idx = min(max(int(round(ph_anchor / df)), 0), nF - 1)
                meanph_a, csum_a = curves[krep]
                anchor_val = float(meanph_a[ph_idx])
                anchor_cnt = float(csum_a[ph_idx] / nT)
                pct_all = pct_of(H, Hrep)
                length = np.minimum(1.0, Df / (np.abs(SLOPES) * Dt + 1e-12))
                sel = np.abs(length - rep_cross_frac) <= MATCH_TOL * rep_cross_frac
                pct_matched = pct_of(H[sel], Hrep) if sel.sum() >= 5 else np.nan
                kbest = int(np.nanargmax(H))
                st["radon"][lab] = dict(H=H.copy(), argph=argph.copy(), supp=supp.copy(),
                                        Hrep=Hrep, rep_node=rep_node,
                                        pct_all=pct_all,
                                        pct_matched=pct_matched,
                                        argmax_slope=float(SLOPES[kbest]),
                                        argmax_H=float(H[kbest]),
                                        anchor_val=anchor_val,
                                        anchor_cnt=anchor_cnt,
                                        n_matched=int(sel.sum()))
                log("-" * 88)
                log(f"[{eid} | thr>{thr:.4f}] mask {nmask}px={100*frac_grid:.2f}%  "
                    f"cols {r['n_cols']}/{nT}  components {ncc}  multirun "
                    f"{n_multirun}/{nT}  centroid range {cen_range:.4f} MHz")
                log(f"   FITTED rate (weighted centroid, Theil-Sen) = {r['rate']:+8.4f}"
                    f" +- {r['se']:.4f} MHz/s (p={r['p_ts']:.2e}); reported"
                    f" {reported:+.4f}; diff {dif:+.4f}; relerr {relerr:+.3f};"
                    f" Spearman rho {r['rho']:+.4f}")
                log(f"   RADON/HOUGH: H at the node nearest the reported slope ({rep_node:+.2f}"
                    f" vs {reported:+.4f} MHz/s) = {Hrep:.5f}; percentile among"
                    f" {int(np.isfinite(H).sum())} slopes = {pct_all:.2f}"
                    f"   (length-matched pct {pct_matched:.2f}, n={int(sel.sum())})")
                log(f"      argmax H at {SLOPES[kbest]:+.2f} MHz/s (H={H[kbest]:.5f});"
                    f" phase support at best {100*supp[kbest]:.0f}% of columns;"
                    f" anchor-phase value {anchor_val:.5f} over {100*anchor_cnt:.0f}% cols")

        # NOTE: the ridge-free (row-rolled) surrogate control lives in
        # s19_surrogate_null.py, which runs it for all 5 events at threshold 1.05.

        # ---------------- injection control (calibrates detectability) ---------
        for amp in INJ_AMPS:
            Zs = (1.0 + (Z - 1.0)
                  + amp * np.exp(-0.5 * ((f[:, None] - (fmax + reported * (t[None, :] - t0)))
                                         / INJ_SIGMA_MHZ) ** 2))
            rj = rate_estimate(Zs, f, t, 1.0 + EPS)
            S_inj = rj["S"]
            H, argph, supp, _ = radon_scan(S_inj[::-1, :], f_asc, t, SLOPES, df)
            Hrep = float(H[int(np.argmin(np.abs(SLOPES - reported)))])
            pct = pct_of(H, Hrep)
            kbest = int(np.nanargmax(H))
            inj_rows.append(dict(event_id=eid, injected_amplitude=amp,
                                 injected_slope=reported, sigma_MHz=INJ_SIGMA_MHZ,
                                 recovered_rate=rj["rate"], recovered_se=rj["se"],
                                 diff=float(rj["rate"] - reported),
                                 pct_of_reported=pct,
                                 radon_argmax_slope=float(SLOPES[kbest]),
                                 H_reported=Hrep, H_argmax=float(H[kbest])))
            log(f"   INJECT ridge amp {amp:.2f} at reported slope -> recovered"
                f" {rj['rate']:+8.4f} MHz/s, radon pct of reported {pct:.2f},"
                f" radon argmax {SLOPES[kbest]:+.2f}")
        del Zs, S_inj
        gc.collect()

        # ------------- block diagnostic (coherence across column blocks) ------
        rp = st["thr"][THR_PRIMARY]
        t_ = t
        edges = [(0, 85), (85, 170), (170, 255), (255, 340), (340, 425)]
        med = []
        for a, b in edges:
            sel = rp["good"][a:b]
            med.append(float(np.median(rp["cen"][a:b][sel])) if sel.any() else np.nan)
        tot = med[-1] - med[0]
        block_rows.append(dict(
            event_id=eid, block_medians_MHz=str([float(x) for x in med]),
            first_last_shift_MHz=float(tot),
            peak_to_peak_MHz=float(np.nanmax(med) - np.nanmin(med)),
            implied_slope=float(tot / (t_[edges[-1][0]] - t_[edges[0][0]]))))

        store[eid] = st
        del Z
        gc.collect()

    ord_ids = list(store.keys())
    R = pd.DataFrame(rate_rows)
    C = pd.DataFrame(comp_rows)
    Ssum = pd.DataFrame(scan_summary)
    INJ = pd.DataFrame(inj_rows)
    B = pd.DataFrame(block_rows)

    # ---------------------------------------------------------------- LOO ----
    log()
    log("=" * 88)
    log("FULL LEAVE-ONE-OUT over the 5 events (5 folds) for every event-level stat")
    log("=" * 88)

    def agg(name, vals, boundary=None, unit="", mode="median"):
        """mode='median' for a continuous statistic; 'count' for the sum over the
        5 events; 'fraction' for the mean over the 5 events.  Using the median
        on a 0/1 array silently reports the majority vote, so counts/fractions
        must be asked for explicitly."""
        v = np.asarray(vals, dtype=float)
        red = {"median": np.nanmedian, "count": np.nansum,
               "fraction": np.nanmean}[mode]
        if v.size == 0 or not np.isfinite(v).any():
            log(f"{name:40s} SKIPPED: no finite values (n={v.size})")
            return dict(statistic=name, unit=unit, aggregation=mode,
                        full_sample=np.nan, loo_min=np.nan, loo_max=np.nan,
                        loo_range=np.nan, range_over_full=np.nan, sign_flip=False,
                        crosses_boundary=False, flagged=False,
                        decision_boundary=boundary, fold_values="[]")
        full = float(red(v))
        folds = np.array([red(np.delete(v, k)) for k in range(len(v))], float)
        rng = float(np.nanmax(folds) - np.nanmin(folds))
        rel = float(rng / abs(full)) if np.isfinite(full) and full != 0 else np.inf
        sign_flip = bool(np.any(np.sign(folds) != np.sign(full)))
        cross = (bool(np.any((folds - boundary) * (full - boundary) < 0))
                 if boundary is not None else False)
        flag = sign_flip or cross
        tag = "stable"
        if flag:
            parts = []
            if sign_flip:
                parts.append("sign flip")
            if cross:
                parts.append("crosses boundary %.2f" % boundary)
            tag = "FLAGGED: " + "; ".join(parts)
        log(f"{name:40s} full={full:+13.6f} {unit:14s} LOO [{np.nanmin(folds):+13.6f},"
            f"{np.nanmax(folds):+13.6f}] width {rng:.6f}  {tag}")
        return dict(statistic=name, unit=unit, aggregation=mode, full_sample=full,
                    loo_min=float(np.nanmin(folds)), loo_max=float(np.nanmax(folds)),
                    loo_range=rng, range_over_full=rel, sign_flip=sign_flip,
                    crosses_boundary=cross, flagged=flag,
                    decision_boundary=boundary,
                    fold_values=str([round(float(x), 6) for x in folds]))

    rows = []
    g = R[R.threshold == THR_PRIMARY].set_index("event_id").loc[ord_ids]
    pct = [store[e]["radon"][THR_PRIMARY]["pct_all"] for e in ord_ids]
    pctm = [store[e]["radon"][THR_PRIMARY]["pct_matched"] for e in ord_ids]
    argm = [store[e]["radon"][THR_PRIMARY]["argmax_slope"] for e in ord_ids]
    rep = g["reported_drift"].to_numpy()
    fit = g["rate_theilsen"].to_numpy()
    rows.append(agg("median_fitted_rate_theilsen", fit, unit="MHz/s"))
    rows.append(agg("median_abs_bias_vs_reported", np.abs(fit - rep), unit="MHz/s"))
    rows.append(agg("median_fitted_over_reported_ratio", fit / rep, boundary=0.0))
    rows.append(agg("n_events_same_sign_as_reported",
                    np.sign(fit) == np.sign(rep), boundary=2.5, unit="count/5",
                    mode="count"))
    rows.append(agg("frac_events_same_sign_as_reported",
                    np.sign(fit) == np.sign(rep), boundary=0.5, unit="fraction",
                    mode="fraction"))
    rows.append(agg("median_radon_pct_of_reported", pct, boundary=50.0, unit="pct"))
    rows.append(agg("median_radon_pct_matched", pctm, boundary=50.0, unit="pct"))
    rows.append(agg("n_events_radon_pct_extreme_p10",
                    [(p <= 10 or p >= 90) for p in pct], boundary=2.5, unit="count/5",
                    mode="count"))
    rows.append(agg("n_events_radon_pct_extreme_p05",
                    [(p <= 5 or p >= 95) for p in pct], boundary=2.5, unit="count/5",
                    mode="count"))
    rows.append(agg("median_radon_argmax_slope", argm, boundary=0.0, unit="MHz/s"))
    # displacement test: reported implies |m|*Dt of motion, pixels show cen_range
    obs = np.array([store[e]["thr"][THR_PRIMARY]["cen_range_MHz"] for e in ord_ids])
    imp = np.abs(rep) * np.array([store[e]["Dt"] for e in ord_ids])
    rows.append(agg("median_observed_centroid_range", obs, unit="MHz"))
    rows.append(agg("median_implied_ridge_displacement", imp, unit="MHz"))
    rows.append(agg("median_displacement_ratio_implied_over_observed", imp / obs,
                    boundary=1.0))
    rows.append(agg("n_events_implied_exceedsWholeBand",
                    imp > np.array([store[e]["Df"] for e in ord_ids]),
                    boundary=2.5, unit="count/5", mode="count"))
    rows.append(agg("frac_events_implied_displacement_gt_band",
                    imp / np.array([store[e]["Df"] for e in ord_ids]),
                    boundary=1.0, unit="fraction", mode="fraction"))
    rows.append(agg("median_mask_fraction", g["frac_grid_mask"], unit="frac"))
    rows.append(agg("median_pca_elongation", g["pca_elong"], unit="ratio"))
    rows.append(agg("median_n_components",
                    C[C.threshold == THR_PRIMARY]["n_components"], unit="count"))
    rows.append(agg("median_frac_multirun_columns",
                    C[C.threshold == THR_PRIMARY]["frac_multirun_cols"], unit="frac"))
    rows.append(agg("median_injection_pct_at_amp_0.10",
                    INJ[INJ.injected_amplitude == 0.10]["pct_of_reported"],
                    boundary=90.0, unit="pct"))
    inj_med = INJ.groupby("injected_amplitude")["pct_of_reported"].median()
    rows.append(agg("median_injection_pct_at_amp_0.02",
                    INJ[INJ.injected_amplitude == 0.02]["pct_of_reported"],
                    boundary=90.0, unit="pct"))
    rows.append(agg("median_injection_pct_at_amp_0.05",
                    INJ[INJ.injected_amplitude == 0.05]["pct_of_reported"],
                    boundary=90.0, unit="pct"))
    rows.append(agg("median_injection_pct_at_amp_0.25",
                    INJ[INJ.injected_amplitude == 0.25]["pct_of_reported"],
                    boundary=90.0, unit="pct"))
    rows.append(agg("min_amp_with_median_pct_ge90",
                    [float(inj_med[inj_med >= 90].index.min())
                     if (inj_med >= 90).any() else np.nan] * 5, unit="amplitude"))
    LOO = pd.DataFrame(rows)

    for e in ord_ids:
        s = store[e]["radon"][THR_PRIMARY]
        Ssum = pd.concat([Ssum, pd.DataFrame([dict(
            event_id=e, threshold=THR_PRIMARY, reported_drift=store[e]["reported"],
            H_reported=s["Hrep"], pct_of_reported=s["pct_all"],
            pct_matched=s["pct_matched"], argmax_slope=s["argmax_slope"],
            argmax_H=s["argmax_H"], anchor_val=s["anchor_val"],
            anchor_support_frac=s["anchor_cnt"], n_matched=s["n_matched"],
            cen_range_MHz=store[e]["thr"][THR_PRIMARY]["cen_range_MHz"],
            implied_displacement_MHz=abs(store[e]["reported"]) * store[e]["Dt"],
            observed_displacement_MHz=store[e]["thr"][THR_PRIMARY]["cen_range_MHz"],
            fitted_rate=store[e]["thr"][THR_PRIMARY]["rate"])])], ignore_index=True)
    Ssum = Ssum.drop_duplicates(subset=["event_id", "threshold"], keep="last")

    # ------------------------------------------------------------- saving ----
    R.to_csv(os.path.join(RES, "s17_threshold_rates.csv"), index=False)
    C.to_csv(os.path.join(RES, "s17_mask_components.csv"), index=False)
    Ssum.to_csv(os.path.join(RES, "s17_line_scan_summary.csv"), index=False)
    INJ.to_csv(os.path.join(RES, "s17_injection_control.csv"), index=False)
    B.to_csv(os.path.join(RES, "s17_block_shifts.csv"), index=False)
    LOO.to_csv(os.path.join(RES, "s17_leave_one_out.csv"), index=False)
    with open(os.path.join(RES, "s17_leave_one_out.json"), "w") as fh:
        json.dump(json.loads(LOO.to_json(orient="records")), fh, indent=1)

    summary = dict(
        analysis_unit="one complete dynamic-spectrum event (n_events=5); pixels are nested repeats",
        scan="coverage-matched Radon/Hough: H(m)=max over phase of the mean intensity along f(t)=phase+m*(t-t0), phases with >=25% column support",
        thresholds=[l for l, _ in THRESHOLDS],
        scan_thresholds=SCAN_THRESHOLDS,
        slope_grid=[float(SLOPES[0]), float(SLOPES[-1]), int(SLOPES.size)],
        injection_control=dict(amplitudes=INJ_AMPS, sigma_MHz=INJ_SIGMA_MHZ,
                               slope="the event's reported slope",
                               phase="f_max at t_min for negative reported slopes"),
        per_event={
            e: dict(reported=store[e]["reported"],
                    fitted_rate_primary=store[e]["thr"][THR_PRIMARY]["rate"],
                    fitted_rate_by_threshold={l: store[e]["thr"][l]["rate"]
                                              for l, _ in THRESHOLDS},
                    mask_fraction_by_threshold={l: store[e]["thr"][l]["frac_grid"]
                                                for l, _ in THRESHOLDS},
                    n_components_by_threshold={l: store[e]["thr"][l]["ncc"]
                                               for l, _ in THRESHOLDS},
                    asserted_rate_hz=store[e]["Dt"], Dt=store[e]["Dt"],
                    Df=store[e]["Df"],
                    implied_displacement_MHz=abs(store[e]["reported"]) * store[e]["Dt"],
                    observed_centroid_range_MHz=store[e]["thr"][THR_PRIMARY]["cen_range_MHz"],
                    radon_pct_of_reported_primary=store[e]["radon"][THR_PRIMARY]["pct_all"],
                    radon_pct_matched_primary=store[e]["radon"][THR_PRIMARY]["pct_matched"],
                    radon_argmax_primary=store[e]["radon"][THR_PRIMARY]["argmax_slope"],
                    radon_by_threshold={l: dict(
                        pct_all=store[e]["radon"][l]["pct_all"],
                        pct_matched=store[e]["radon"][l]["pct_matched"],
                        argmax=store[e]["radon"][l]["argmax_slope"],
                        H_reported=store[e]["radon"][l]["Hrep"])
                        for l in SCAN_THRESHOLDS})
            for e in ord_ids},
        threshold_rates=json.loads(R.to_json(orient="records")),
        mask_components=json.loads(C.to_json(orient="records")),
        line_scan_summary=json.loads(Ssum.to_json(orient="records")),
        injection_control_table=json.loads(INJ.to_json(orient="records")),
        block_shifts=json.loads(B.to_json(orient="records")),
        leave_one_out=json.loads(LOO.to_json(orient="records")),
        errors=ERRORS)
    with open(os.path.join(RES, "s17_robustness_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ------------------------------------------------------------ figures ----
    log()
    log("FIGURES")
    fig, axes = plt.subplots(1, 5, figsize=(27, 6.4), dpi=125)
    for ax, e in zip(axes, ord_ids):
        s = store[e]
        f_, t_ = s["f"], s["t"]
        bg = np.clip(s["Z"] - 1.0, 0, None)
        ax.imshow(bg, origin="lower", aspect="auto",
                  extent=[t_[0], t_[-1], f_.min(), f_.max()], cmap="magma",
                  norm=PowerNorm(0.4, vmin=0, vmax=0.5))
        repv = s["reported"]
        tline = np.array([t_[0], t_[-1]])
        ax.plot(tline, f_.max() + repv * (tline - t_[0]), "c--", lw=2.2,
                label=f"reported {repv:+.2f} MHz/s")
        rp_ = s["thr"][THR_PRIMARY]
        g_ = rp_["good"]
        if g_.any():
            b = np.median(rp_["cen"][g_]) - rp_["rate"] * np.median(t_[g_])
            ax.plot(tline, rp_["rate"] * tline + b, "w-", lw=1.8,
                    label=f"fitted {rp_['rate']:+.2f} MHz/s")
        ax.plot(t_[g_], rp_["cen"][g_], "-", color="lime", lw=0.8, alpha=0.9,
                label="centroid track")
        ax.set_title(f"{e}: reported {repv:+.2f}, fitted {rp_['rate']:+.2f} MHz/s\n"
                     f"implied displacement {abs(repv)*s['Dt']:.1f} MHz vs observed"
                     f" centroid range {rp_['cen_range_MHz']:.2f} MHz", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (MHz)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("s17(a) event images with the reported-slope line (cyan dashed) and"
                 " the fitted ridge (white); unit = one event (n=5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "s17_perevent_ridge.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.6), dpi=125, sharex=True)
    for ax, e in zip(axes, ord_ids):
        for lab in SCAN_THRESHOLDS:
            H = store[e]["radon"][lab]["H"]
            ax.plot(SLOPES, H, lw=1.2, label=f"thr {lab}")
            k = int(np.nanargmax(H))
            ax.plot([SLOPES[k]], [H[k]], "v", ms=7, color=ax.lines[-1].get_color())
        ax.axvline(store[e]["reported"], color="k", ls="--", lw=2,
                   label=f"reported {store[e]['reported']:+.2f}")
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_title(f"{e}: reported at pct "
                     f"{store[e]['radon'][THR_PRIMARY]['pct_all']:.1f} (thr 1.05)", fontsize=10)
        ax.set_xlabel("slope (MHz/s)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("H(m) = max-phase mean line intensity")
    fig.suptitle("s17(b) coverage-matched Radon/Hough null distributions; triangles ="
                 " argmax of each curve", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s17_slope_scan_null.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6), dpi=125)
    x = range(len(THRESHOLDS))
    xl = [l for l, _ in THRESHOLDS]
    ax = axes[0]
    for e in ord_ids:
        ax.plot(x, [store[e]["thr"][l]["rate"] for l, _ in THRESHOLDS], "o-", label=e)
        ax.axhline(store[e]["reported"], color="gray", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("pixel threshold"); ax.set_ylabel("fitted rate (MHz/s)")
    ax.set_title("(c1) fitted vs reported (grey dotted)"); ax.legend(fontsize=8)
    ax = axes[1]
    for e in ord_ids:
        ax.plot(x, [100 * store[e]["thr"][l]["frac_grid"] for l, _ in THRESHOLDS],
                "o-", label=e)
    ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("pixel threshold"); ax.set_ylabel("% of grid above threshold")
    ax.set_title("(c2) mask fraction"); ax.legend(fontsize=8)
    ax = axes[2]
    for e in ord_ids:
        ax.plot(x, [store[e]["thr"][l]["ncc"] for l, _ in THRESHOLDS], "o-", label=e)
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("pixel threshold"); ax.set_ylabel("8-connected components (log)")
    ax.set_title("(c3) fragmentation"); ax.legend(fontsize=8)
    ax = axes[3]
    for e in ord_ids:
        ax.plot(x, [store[e]["radon"][l]["pct_all"] if l in SCAN_THRESHOLDS else np.nan
                    for l, _ in THRESHOLDS], "o-", label=e)
    for e in ord_ids:
        ax.plot(x, [store[e]["radon"][l]["pct_matched"] if l in SCAN_THRESHOLDS else np.nan
                    for l, _ in THRESHOLDS], "s--", alpha=0.6, color="gray")
    ax.axhspan(0, 10, color="green", alpha=0.12)
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_ylim(0, 100); ax.set_xticks(list(x)); ax.set_xticklabels(xl)
    ax.set_xlabel("pixel threshold"); ax.set_ylabel("percentile of reported slope")
    ax.set_title("(c4) decisive test vs threshold\n(green = 'recovered'; grey dashed"
                 " = length-matched)"); ax.legend(fontsize=8)
    fig.suptitle("s17(c) threshold sensitivity; unit = one event (n=5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s17_threshold_curves.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=125)
    ax = axes[0]
    for e in ord_ids:
        sub = INJ[INJ.event_id == e].sort_values("injected_amplitude")
        ax.plot(sub.injected_amplitude, sub.recovered_rate, "o-", label=e)
        ax.axhline(store[e]["reported"], color="gray", ls=":", lw=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("injected ridge amplitude (above background 1.0)")
    ax.set_ylabel("recovered rate (MHz/s)")
    ax.set_title("(d1) injection control: recovered rate vs injected amplitude\n"
                 "(grey dotted = that event's reported rate)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for e in ord_ids:
        sub = INJ[INJ.event_id == e].sort_values("injected_amplitude")
        ax.plot(sub.injected_amplitude, sub.pct_of_reported, "o-", label=e)
    ax.axhspan(90, 100, color="green", alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("injected ridge amplitude")
    ax.set_ylabel("Radon percentile of the reported slope")
    ax.set_title("(d2) injection control: detectability of the reported slope")
    ax.legend(fontsize=8)
    fig.suptitle("s17(d) power calibration on the REAL images; unit = one event (n=5)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "s17_injection_control.png"))
    plt.close(fig)

    for e in ord_ids:
        store[e]["thr"].clear()
        store[e]["radon"].clear()
        store[e]["Z"] = None
    store.clear()
    gc.collect()
    log("saved s17_*.csv/json + 4 figures; errors: "
        + (json.dumps(ERRORS) if ERRORS else "none"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERRORS.append(traceback.format_exc())
        log("FATAL:\n" + traceback.format_exc())
        sys.exit(1)