#!/usr/bin/env python3
"""S04 - Polarization-channel contrast within anonymous source group.

Dimension sub-question: "Does the polarization channel alter the conclusion?"

Analysis unit = one complete source-polarization series (6 of them in
drift_population_series.tsv).  Bins inside a series are NOT independent
samples: every series is one physical-source observation, and 608/948 bins are
empty.  All uncertainty is therefore obtained by resampling BINS (multinomial
shape bootstrap / Poisson total bootstrap), never by treating bins as iid
events.

Two bootstraps are used deliberately:

  * multinomial (conditional on the observed total N) -> shape statistics
    (weighted median, IQR, sigma, fraction above threshold, mode).  Its
    sampling error is the error of the distribution SHAPE given the series
    total.  It holds N fixed by construction, so it can never return a
    non-degenerate CI for a total and is not used for ratios.
  * Poisson (unconditional) -> totals and channel ratios.

The whole bootstrap is vectorised: the multinomial draws are one
(n_draws, n_bins) array and the log-normal MLE is a fixed-iteration Newton
solve with analytic gradients, applied to every row at once.  Deterministic
seed, so repeated runs are identical.

Workbook cross-check of the drift_regime split (verified here as a hard
equality test, see *_structure_check.csv):
  fast regime == the '(-35<s<-10)' drift block of one sheet
  slow regime == the concat of the '(-10<s<0)' and '(0<s<20)' blocks of the
                 same sheet.  The seam is a bookkeeping artefact (last bin of
                 block 1 at x=2.986996, first bin of block 2 at x=0.219728 are
                 not monotone), so raw bin centres contain a 2.77 MHz/s
                 artificial gap; edges are rebuilt block-by-block so no bin
                 gets a spurious negative width.

Outputs: output/results/s04_*.csv|json|npz, log at output/logs/s04.log
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized_views"
OUT = ROOT / "output" / "results"
NPZ = OUT / "s03_workbook_blocks.npz"

SEED = 20221013
NBOOT = 4000
NPERM = 4000
RNG = np.random.default_rng(SEED)

SHEET_OF_FAST = {
    "population_series_1": "Sheet_025",
    "population_series_2": "Sheet_027",
    "population_series_3": "Sheet_029",
    "population_series_4": "Sheet_031",
    "population_series_5": "Sheet_033",
    "population_series_6": "Sheet_035",
}
SLOW_SHEET_PREFIX = {
    "population_series_1": "Sheet_024",
    "population_series_2": "Sheet_026",
    "population_series_3": "Sheet_028",
    "population_series_4": "Sheet_030",
    "population_series_5": "Sheet_032",
    "population_series_6": "Sheet_034",
}


# --------------------------------------------------------------------------
# vectorised log-normal MLE (bin centres as values), Newton with analytic grad
# --------------------------------------------------------------------------
def lognormal_mle_newton(lx, w, n_iter=200, tol=1e-12):
    """Weighted MLE of (mu, sigma) for log-normal on values lx with weights w.

    lx: (..., K) log of bin centres (NaN-padded rows allowed via w==0)
    w:  (..., K) non-negative weights (zeros ignored)
    Fully vectorised over leading axes; returns mu, sigma with same shape[:-1].
    """
    lx = np.asarray(lx, float)
    w = np.asarray(w, float)
    W = w.sum(axis=-1)
    mu = (w * lx).sum(axis=-1) / W
    var0 = (w * (lx - mu[..., None]) ** 2).sum(axis=-1) / W
    s = np.sqrt(np.maximum(var0, 1e-9))
    for _ in range(n_iter):
        z = (lx - mu[..., None]) / s[..., None]
        wzz = (w * z * z).sum(axis=-1)
        g_mu = -((w * z).sum(axis=-1)) / s
        g_s = (W - wzz) / s
        h_mu = W / s ** 2
        h_ss = (-W + 3.0 * wzz) / s ** 2
        h_ms = -2.0 * ((w * z).sum(axis=-1)) / s ** 2
        det = h_mu * h_ss - h_ms ** 2
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        d_mu = (h_ss * g_mu - h_ms * g_s) / det
        d_s = (-h_ms * g_mu + h_mu * g_s) / det
        step = np.maximum(np.abs(d_mu) / np.maximum(np.abs(mu), 1e-12),
                          np.abs(d_s) / np.maximum(np.abs(s), 1e-12))
        mu = mu - d_mu
        s = np.maximum(s - d_s, 1e-9)
        if np.all(step < tol):
            break
    return mu, s


def fit_lognormal_binned_mle(x, n):
    """Point-estimate wrapper used for the per-series table."""
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    m = n > 0
    if m.sum() < 2:
        return np.nan, np.nan
    mu, s = lognormal_mle_newton(np.log(x[m]), n[m])
    return float(mu), float(s)


def fit_lognormal_binned_mle_nm(x, n):
    """Independent Nelder-Mead fit, used only to cross-check the Newton solve."""
    x = np.asarray(x, float)
    n = np.asarray(n, float)
    m = n > 0
    lx, w = np.log(x[m]), n[m]

    def nll(p):
        mu, ls = p[0], p[1]
        s = np.exp(ls)
        z = (lx - mu) / s
        return float(np.sum(w * (np.log(s) + 0.5 * np.log(2 * np.pi) + 0.5 * z * z)))

    mu0 = float((w * lx).sum() / w.sum())
    s0 = float(np.sqrt(max((w * (lx - mu0) ** 2).sum() / w.sum(), 1e-9)))
    r = optimize.minimize(nll, [mu0, np.log(s0)], method="Nelder-Mead",
                          options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000})
    return float(r.x[0]), float(np.exp(r.x[1]))


def fit_lognormal_interval_mle(edges, n):
    """MLE integrating the log-normal density over each bin (point estimate)."""
    lo, hi = np.log(np.maximum(edges[:-1], 1e-12)), np.log(edges[1:])
    n = np.asarray(n, float)
    m = n > 0
    lo, hi, w = lo[m], hi[m], n[m]
    ctr = 0.5 * (lo + hi)

    def nll(p):
        mu, ls = p[0], p[1]
        s = np.exp(ls)
        pbin = np.diff(stats.norm.cdf((np.stack([lo, hi]) - mu) / s), axis=0)[0]
        return float(-np.sum(w * np.log(np.clip(pbin, 1e-300, None))))

    mu0 = float((w * ctr).sum() / w.sum())
    s0 = float(np.sqrt(max((w * (ctr - mu0) ** 2).sum() / w.sum(), 1e-9)))
    r = optimize.minimize(nll, [mu0, np.log(s0)], method="Nelder-Mead",
                          options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000})
    return float(r.x[0]), float(np.exp(r.x[1]))


# --------------------------------------------------------------------------
# distribution-free summaries
# --------------------------------------------------------------------------
def wquantile(x, n, q):
    x, n = np.asarray(x, float), np.asarray(n, float)
    m = n > 0
    x, n = x[m], n[m]
    if n.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    x, n = x[o], n[o]
    return float(np.interp(q, np.cumsum(n) / n.sum(), x))


def summaries(x, n, edges, thresh=10.0):
    x, n = np.asarray(x, float), np.asarray(n, float)
    N = float(n.sum())
    q1, q3 = wquantile(x, n, 0.25), wquantile(x, n, 0.75)
    muA, sA = fit_lognormal_binned_mle(x, n)
    muB, sB = fit_lognormal_interval_mle(edges, n)
    m = n > 0
    lx, ln = np.log(x[m]), n[m]
    muC = float((ln * lx).sum() / ln.sum())
    sC = float(np.sqrt(max((ln * (lx - muC) ** 2).sum() / ln.sum(), 1e-12)))
    return {
        "events": N,
        "n_bins": int(len(x)),
        "n_empty_bins": int((n == 0).sum()),
        "range_min": float(x[m].min()) if m.any() else np.nan,
        "range_max": float(x[m].max()) if m.any() else np.nan,
        "weighted_mean": float((x * n).sum() / N) if N > 0 else np.nan,
        "weighted_median": wquantile(x, n, 0.5),
        "weighted_q25": q1,
        "weighted_q75": q3,
        "weighted_iqr": q3 - q1,
        "mode_bin_center": float(x[int(np.argmax(n))]) if N > 0 else np.nan,
        "mode_bin_count": float(n.max()),
        "frac_gt_10": float(n[x > thresh].sum() / N) if N > 0 else np.nan,
        "mu_binned_mle": muA, "sigma_binned_mle": sA,
        "median_binned_mle": float(np.exp(muA)), "mode_binned_mle": float(np.exp(muA - sA ** 2)),
        "mu_interval_mle": muB, "sigma_interval_mle": sB,
        "median_interval_mle": float(np.exp(muB)), "mode_interval_mle": float(np.exp(muB - sB ** 2)),
        "mu_moment": muC, "sigma_moment": sC, "median_moment": float(np.exp(muC)),
    }


def vector_bootstrap_stats(x, n, draws, thresh=10.0):
    """Shape statistics for many bootstrap replicates at once.

    x: (K,) sorted-able bin centres of NON-EMPTY-mappable bins (we pass all bins)
    draws: (B, K) resampled counts
    Returns dict stat -> (B,) array.
    """
    x = np.asarray(x, float)
    o = np.argsort(x)
    xs = x[o]
    D = np.asarray(draws, float)[:, o]
    tot = D.sum(axis=1)
    tot_safe = np.where(tot <= 0, 1.0, tot)
    cum = np.cumsum(D, axis=1)
    out = {}

    def wq(q):
        tgt = q * tot
        idx = (cum < tgt[:, None]).sum(axis=1)
        idx = np.clip(idx, 0, len(xs) - 1)
        return xs[idx]

    out["weighted_median"] = wq(0.5)
    out["weighted_q25"] = wq(0.25)
    out["weighted_q75"] = wq(0.75)
    out["weighted_iqr"] = out["weighted_q75"] - out["weighted_q25"]
    out["frac_gt_10"] = D[:, xs > thresh].sum(axis=1) / tot_safe
    out["mode_bin_center"] = xs[np.argmax(D, axis=1)]

    lx = np.log(np.maximum(xs, 1e-300))
    mu = (D * lx).sum(axis=1) / tot_safe
    var = (D * (lx - mu[:, None]) ** 2).sum(axis=1) / tot_safe
    out["mu_moment"] = mu
    out["sigma_moment"] = np.sqrt(np.maximum(var, 1e-12))
    out["median_moment"] = np.exp(mu)

    with np.errstate(divide="ignore", invalid="ignore"):
        mu2, s2 = lognormal_mle_newton(np.broadcast_to(lx, D.shape), D, n_iter=120)
    out["mu_binned_mle"] = mu2
    out["sigma_binned_mle"] = s2
    out["median_binned_mle"] = np.exp(mu2)
    out["mode_binned_mle"] = np.exp(mu2 - s2 ** 2)
    return out


# --------------------------------------------------------------------------
# workbook edges
# --------------------------------------------------------------------------
def load_cache():
    z = np.load(NPZ, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def block_edges(sheet, cache):
    """Edges for one workbook sheet, built block-by-block then concatenated."""
    keys = [k for k in cache if k.split("__")[0] == sheet]
    edges = []
    for k in keys:
        x = np.asarray(cache[k][0], float)
        h = float(np.median(np.diff(x))) if len(x) > 1 else 0.1
        edges.append(np.concatenate([x - h / 2.0, [x[-1] + h / 2.0]]))
    return np.concatenate(edges)


def block_xy(sheet, cache):
    keys = [k for k in cache if k.split("__")[0] == sheet]
    x = np.concatenate([np.asarray(cache[k][0], float) for k in keys])
    y = np.concatenate([np.asarray(cache[k][1], float) for k in keys])
    return x, y


def main() -> None:
    df = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    cache = load_cache()

    # ------------------------------------------------------------------
    # 0. structural verification: fast == 1 block, slow == 2 blocks
    # ------------------------------------------------------------------
    struct = []
    for pid, sub in df.groupby("population_series_id"):
        for regime, s in sub.groupby("drift_regime"):
            sheet = SHEET_OF_FAST[pid] if regime == "fast" else SLOW_SHEET_PREFIX[pid]
            xw, yw = block_xy(sheet, cache)
            xs, ns = s.sort_values("distribution_bin_center").distribution_bin_center.values, \
                s.sort_values("distribution_bin_center").event_count.values
            struct.append({
                "population_series_id": pid, "regime": regime, "sheet": sheet,
                "n_bins_view": len(xs), "n_bins_workbook": len(xw),
                "bins_identical": bool(len(xs) == len(xw) and np.allclose(xs, xw, atol=1e-9)),
                "counts_identical": bool(len(ns) == len(yw) and np.allclose(ns, yw)),
                "counts_sum_view": float(ns.sum()), "counts_sum_workbook": float(yw.sum()),
                "monotone_bins": bool(len(xs) < 2 or np.all(np.diff(xs) > 0)),
            })
    struct_df = pd.DataFrame(struct)
    print("=== S04.0 structural verification of drift_regime vs workbook blocks ===")
    print(struct_df.to_string(index=False))
    struct_df.to_csv(OUT / "s04_structure_check.csv", index=False)
    n_ok = int((struct_df.bins_identical & struct_df.counts_identical).sum())
    print(f"exact view<->workbook block matches: {n_ok}/12")

    # ------------------------------------------------------------------
    # cross-check Newton MLE against Nelder-Mead on every series
    # ------------------------------------------------------------------
    chk = []
    for pid, sub in df.groupby("population_series_id"):
        for regime, s in sub.groupby("drift_regime"):
            x, n = s.distribution_bin_center.values, s.event_count.values
            a = fit_lognormal_binned_mle(x, n)
            b = fit_lognormal_binned_mle_nm(x, n)
            chk.append({"population_series_id": pid, "regime": regime,
                        "mu_newton": a[0], "mu_nm": b[0], "dmu": abs(a[0] - b[0]),
                        "sigma_newton": a[1], "sigma_nm": b[1], "dsigma": abs(a[1] - b[1])})
    chk_df = pd.DataFrame(chk)
    print("\n=== S04.0b Newton MLE vs Nelder-Mead cross-check ===")
    print(f"max |dmu| = {chk_df.dmu.max():.3e}   max |dsigma| = {chk_df.dsigma.max():.3e}")
    chk_df.to_csv(OUT / "s04_fit_crosscheck.csv", index=False)

    # ------------------------------------------------------------------
    # 1..4 per-series summaries + vectorised multinomial bootstrap
    # ------------------------------------------------------------------
    rows, boot_rows = [], []
    for pid, sub in df.groupby("population_series_id"):
        grp, ch = sub.anonymous_source_group.iloc[0], sub.polarization_channel.iloc[0]
        for regime, s in sub.groupby("drift_regime"):
            s = s.sort_values("distribution_bin_center")
            x = s.distribution_bin_center.values.astype(float)
            n = s.event_count.values.astype(float)
            sheet = SHEET_OF_FAST[pid] if regime == "fast" else SLOW_SHEET_PREFIX[pid]
            edges = block_edges(sheet, cache)
            if len(edges) != len(x) + 1:
                raise RuntimeError(f"edges {len(edges)} != bins+1 {len(x)+1} for {pid}/{regime}")
            base = summaries(x, n, edges)
            base.update({"population_series_id": pid, "group": grp,
                         "channel": ch, "regime": regime, "workbook_sheet": sheet})
            rows.append(base)

            N = int(n.sum())
            draws = RNG.multinomial(N, n / N, size=NBOOT).astype(float)
            vb = vector_bootstrap_stats(x, n, draws)
            for k, v in vb.items():
                v = np.asarray(v, float)
                v = v[np.isfinite(v)]
                lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
                boot_rows.append({"population_series_id": pid, "group": grp, "channel": ch,
                                  "regime": regime, "stat": k, "boot": "multinomial",
                                  "median_boot": float(np.median(v)) if len(v) else np.nan,
                                  "ci_lo": float(lo), "ci_hi": float(hi),
                                  "ci_width": float(hi - lo)})

    sum_df = pd.DataFrame(rows)
    boot_df = pd.DataFrame(boot_rows)
    sum_df.to_csv(OUT / "s04_series_summaries.csv", index=False)
    boot_df.to_csv(OUT / "s04_series_bootstrap_ci.csv", index=False)

    print("\n=== S04.1 per-series summaries (unit = one source-polarization series) ===")
    cols = ["group", "channel", "regime", "events", "n_bins", "n_empty_bins",
            "weighted_median", "weighted_iqr", "mode_bin_center", "frac_gt_10",
            "sigma_binned_mle", "sigma_interval_mle", "median_binned_mle"]
    print(sum_df[cols].to_string(index=False))

    # paired shape CI table: channel_l vs channel_r within group+regime
    print("\n=== S04.1b multinomial-bootstrap 95% CI on shape stats ===")
    piv = boot_df.pivot_table(index=["group", "regime", "stat"],
                              columns="channel",
                              values=["ci_lo", "ci_hi", "median_boot"])
    for stat in ["weighted_median", "sigma_binned_mle", "frac_gt_10", "weighted_iqr"]:
        print(f"\n-- {stat} --")
        sub = boot_df[boot_df.stat == stat]
        t = sub.pivot(index=["group", "regime"], columns="channel",
                      values=["median_boot", "ci_lo", "ci_hi"])
        print(t.to_string())
    piv.to_csv(OUT / "s04_shape_ci_pivot.csv")

    # ------------------------------------------------------------------
    # 2. paired channel totals + ratios (Poisson bootstrap, scale corrected)
    # ------------------------------------------------------------------
    ratio_rows = []
    for regime in ["fast", "slow"]:
        for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
            a = sum_df[(sum_df.group == grp) & (sum_df.regime == regime) & (sum_df.channel == "channel_l")]
            b = sum_df[(sum_df.group == grp) & (sum_df.regime == regime) & (sum_df.channel == "channel_r")]
            if a.empty or b.empty:
                continue
            va = df[(df.population_series_id == a.population_series_id.iloc[0])
                    & (df.drift_regime == regime)].event_count.values.astype(float)
            vb_ = df[(df.population_series_id == b.population_series_id.iloc[0])
                     & (df.drift_regime == regime)].event_count.values.astype(float)
            Na, Nb = va.sum(), vb_.sum()
            A = RNG.poisson(np.tile(va, (NBOOT, 1))).sum(axis=1)
            B = RNG.poisson(np.tile(vb_, (NBOOT, 1))).sum(axis=1)
            A = np.where(A <= 0, 1, A)
            B = np.where(B <= 0, 1, B)
            ratio = B / A                     # unconditional ratio of totals
            scaled = (B / Nb) / (A / Na)      # density-normalised (bin-count differs)
            r_lo, r_hi = np.percentile(ratio, [2.5, 97.5])
            s_lo, s_hi = np.percentile(scaled, [2.5, 97.5])
            ratio_rows.append({
                "group": grp, "regime": regime,
                "series_l": a.population_series_id.iloc[0], "series_r": b.population_series_id.iloc[0],
                "events_l": Na, "events_r": Nb,
                "n_bins_l": int(a.n_bins.iloc[0]), "n_bins_r": int(b.n_bins.iloc[0]),
                "ratio_r_over_l": Nb / Na,
                "ratio_ci_lo": float(r_lo), "ratio_ci_hi": float(r_hi),
                "ratio_excludes_1": bool(r_lo > 1 or r_hi < 1),
                "scaled_ratio": (Nb / len(vb_)) / (Na / len(va)),
                "scaled_ratio_ci_lo": float(s_lo), "scaled_ratio_ci_hi": float(s_hi),
                "scaled_excludes_1": bool(s_lo > 1 or s_hi < 1),
                "insufficient": bool(min(Na, Nb) < 100),
            })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(OUT / "s04_channel_ratio.csv", index=False)
    print("\n=== S04.2 paired channel totals and Poisson-bootstrap ratios ===")
    print(ratio_df[["group", "regime", "events_l", "events_r", "ratio_r_over_l",
                    "ratio_ci_lo", "ratio_ci_hi", "ratio_excludes_1",
                    "scaled_ratio", "scaled_ratio_ci_lo", "scaled_ratio_ci_hi",
                    "insufficient"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3/5. shape contrast on a COMMON grid + permutation test
    # ------------------------------------------------------------------
    shape_rows, gridded = [], {}
    for regime in ["fast", "slow"]:
        for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
            sel = df[(df.anonymous_source_group == grp) & (df.drift_regime == regime)]
            chans = sorted(sel.polarization_channel.unique())
            cl, cr = chans[0], chans[1]
            sl = sel[sel.polarization_channel == cl]
            sr = sel[sel.polarization_channel == cr]
            lo = max(sl.distribution_bin_center.min(), sr.distribution_bin_center.min())
            hi = min(sl.distribution_bin_center.max(), sr.distribution_bin_center.max())
            grid = np.linspace(lo + 1e-6, hi - 1e-6, 40)
            edges = np.r_[grid[0] - (grid[1] - grid[0]) / 2,
                          (grid[:-1] + grid[1:]) / 2,
                          grid[-1] + (grid[-1] - grid[-2]) / 2]
            hl, _ = np.histogram(sl.distribution_bin_center, bins=edges, weights=sl.event_count.values)
            hr, _ = np.histogram(sr.distribution_bin_center, bins=edges, weights=sr.event_count.values)
            hl, hr = hl.astype(float), hr.astype(float)
            Nl, Nr = hl.sum(), hr.sum()
            if Nl < 1 or Nr < 1:
                continue
            fl, fr = hl / Nl, hr / Nr
            tv = 0.5 * np.abs(fl - fr).sum()
            pooled = hl + hr
            p = pooled / pooled.sum()
            null_tv = np.empty(NPERM)
            for i in range(NPERM):
                aa = RNG.multinomial(int(Nl), p)
                bb = RNG.multinomial(int(Nr), p)
                null_tv[i] = 0.5 * np.abs(aa / Nl - bb / Nr).sum()
            pval = float((null_tv >= tv).mean())
            shape_rows.append({
                "group": grp, "regime": regime, "channel_l": cl, "channel_r": cr,
                "grid_lo": float(grid[0]), "grid_hi": float(grid[-1]), "n_grid": len(grid),
                "N_l": float(Nl), "N_r": float(Nr),
                "tv_distance": float(tv), "null_tv_median": float(np.median(null_tv)),
                "null_tv_p95": float(np.percentile(null_tv, 95)),
                "p_permutation": pval,
                "tv_over_null_median": float(tv / np.median(null_tv)),
                "insufficient": bool(min(Nl, Nr) < 100),
            })
            gridded[f"{grp}|{regime}|{cl}"] = hl
            gridded[f"{grp}|{regime}|{cr}"] = hr
            gridded[f"{grp}|{regime}|grid"] = grid
    shape_df = pd.DataFrame(shape_rows)
    shape_df.to_csv(OUT / "s04_shape_contrast.csv", index=False)
    np.savez_compressed(OUT / "s04_common_grid_hist.npz", **gridded)
    print("\n=== S04.3 shape contrast on common grid (TV vs permutation null) ===")
    print(shape_df.to_string(index=False))

    # ------------------------------------------------------------------
    # JSON bundle
    # ------------------------------------------------------------------
    bundle = {
        "dimension": "polarization channel contrast",
        "analysis_unit": "one complete source-polarization series",
        "seed": SEED, "n_boot": NBOOT, "n_perm": NPERM,
        "series_summaries": json.loads(sum_df.to_json(orient="records")),
        "channel_ratio": json.loads(ratio_df.to_json(orient="records")),
        "shape_contrast": json.loads(shape_df.to_json(orient="records")),
        "structure_check": json.loads(struct_df.to_json(orient="records")),
        "fit_crosscheck": json.loads(chk_df.to_json(orient="records")),
        "caveats": [
            "Bins are not independent samples; they are nested repeats inside one series.",
            "Only 6 source-polarization series exist, i.e. 3 paired comparisons per regime.",
            "Multinomial bootstrap CIs describe shape conditional on the series total.",
            "Poisson bootstrap CIs describe totals and are used for the channel ratios.",
            "The slow regime concatenates two workbook blocks with a bookkeeping seam.",
            "Event counts are drifting sub-burst counts, not independent burst events.",
        ],
    }
    with open(OUT / "s04_channel_contrast.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print("\nwrote s04_channel_contrast.json and s04_*.csv / _common_grid_hist.npz")


if __name__ == "__main__":
    main()