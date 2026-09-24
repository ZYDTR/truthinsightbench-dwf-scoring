#!/usr/bin/env python3
"""chpol_s01 - Polarization-channel contrast within anonymous source group.

Dimension sub-question: "Does the polarization channel alter the conclusion?"

ANALYSIS UNIT (binding): one complete source-polarization series.  The
normalized view exposes 6 of them (3 anonymous groups x 2 channels).  Bins
inside a series are NESTED REPEATS, not independent samples: 608/948 bins are
empty and every series is a single physical-source observation.  All
uncertainty is therefore obtained by resampling BINS, never by treating bins
as iid events.

Two bootstraps are used deliberately and are reported as distinct things:

  * MULTINOMIAL, conditional on the observed series total N -> sampling error
    of the distribution SHAPE (weighted median, IQR, sigma, fraction above
    threshold, mode).  It holds N fixed by construction, so it can never
    return a non-degenerate CI for a total and is NOT used for ratios.
  * POISSON, unconditional -> sampling error of the TOTALS, used for the
    channel ratios.

The bootstrap is fully vectorised: the multinomial draws form one
(n_draws, n_bins) array and the log-normal MLE is a fixed-iteration Newton
solve with analytic gradients applied to all rows at once.  Seeded and
deterministic.

STRUCTURE VERIFIED HERE AS HARD EQUALITY TESTS (see *_structure_check.csv)
  * fast regime == the '(-35<s<-10)' drift block of one workbook sheet
    (exact bin centres and exact counts, 6/6).
  * slow regime == the concatenation of the '(-10<s<0)' AND '(0<s<20)' blocks
    of the SAME sheet (exact counts, 6/6; bin centres not monotone because of
    a bookkeeping seam at x=2.986996 -> x=0.219728, so the raw series carries
    a ~2.77 MHz/s artificial gap; edges are rebuilt block-by-block so no bin
    receives a spurious negative width).
  * orientation series -> workbook: orientation_series_1..6 == Sheet_009..014,
    equal after dropping the first bin of each block (x=-34/-9/1), which the
    view does not contain.
  * C-variant sheets 046/047 are a channel_l group_b variant whose block
    totals differ from the non-C channel_l group_b blocks by a roughly
    constant ~ -22 / -4 events.

Outputs: output/results/chpol_*.csv|json|npz, log at output/logs/chpol_s01.log
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
    "population_series_1": "Sheet_025", "population_series_2": "Sheet_027",
    "population_series_3": "Sheet_029", "population_series_4": "Sheet_031",
    "population_series_5": "Sheet_033", "population_series_6": "Sheet_035",
}
SLOW_SHEET_PREFIX = {
    "population_series_1": "Sheet_024", "population_series_2": "Sheet_026",
    "population_series_3": "Sheet_028", "population_series_4": "Sheet_030",
    "population_series_5": "Sheet_032", "population_series_6": "Sheet_034",
}
ORIENT_SHEET = {
    "orientation_series_1": "Sheet_009", "orientation_series_2": "Sheet_010",
    "orientation_series_3": "Sheet_011", "orientation_series_4": "Sheet_012",
    "orientation_series_5": "Sheet_013", "orientation_series_6": "Sheet_014",
}
C_VARIANT = {"Sheet_046": "group_b C channel_l", "Sheet_047": "group_b C channel_l"}
GROUP_B_L = {"fast": "Sheet_029", "slow": ("Sheet_028",)}


# ==========================================================================
# vectorised log-normal MLE / binned Newton solve
# ==========================================================================
def lognormal_mle_newton(lx, w, n_iter=120, tol=1e-12):
    """Weighted MLE of (mu, sigma) of a log-normal.

    lx: (..., K) log of bin centres; w: (..., K) weights (0 = ignored).
    Vectorised over all leading axes.  Returns (mu, sigma) of shape lx.shape[:-1].
    """
    lx = np.asarray(lx, float)
    w = np.asarray(w, float)
    W = w.sum(axis=-1)
    Ws = np.where(W <= 0, 1.0, W)
    mu = (w * lx).sum(axis=-1) / Ws
    var = (w * (lx - mu[..., None]) ** 2).sum(axis=-1) / Ws
    s = np.sqrt(np.maximum(var, 1e-9))
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
    x, n = np.asarray(x, float), np.asarray(n, float)
    m = n > 0
    if m.sum() < 2:
        return np.nan, np.nan
    mu, s = lognormal_mle_newton(np.log(x[m]), n[m])
    return float(mu), float(s)


def fit_lognormal_binned_mle_nm(x, n):
    """Independent Nelder-Mead fit; used only to validate the Newton solve."""
    x, n = np.asarray(x, float), np.asarray(n, float)
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


def fit_gamma_moments(x, n):
    """Method-of-moments gamma shape k for a robustness check on the LN shape."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    N = n.sum()
    if N <= 0:
        return np.nan, np.nan
    mean = (x * n).sum() / N
    var = (n * (x - mean) ** 2).sum() / N
    if var <= 0 or mean <= 0:
        return np.nan, np.nan
    theta = var / mean
    return float(mean / theta), float(theta)


# ==========================================================================
# distribution-free summaries
# ==========================================================================
def wquantile(x, n, q):
    x, n = np.asarray(x, float), np.asarray(n, float)
    m = n > 0
    x, n = x[m], n[m]
    if n.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(n[o]) / n.sum(), x[o]))


def summaries(x, n, edges, thresh=10.0):
    x, n = np.asarray(x, float), np.asarray(n, float)
    N = float(n.sum())
    m = n > 0
    q1, q3 = wquantile(x, n, 0.25), wquantile(x, n, 0.75)
    muA, sA = fit_lognormal_binned_mle(x, n)
    muB, sB = fit_lognormal_interval_mle(edges, n)
    lx, ln = np.log(x[m]), n[m]
    muC = float((ln * lx).sum() / ln.sum())
    sC = float(np.sqrt(max((ln * (lx - muC) ** 2).sum() / ln.sum(), 1e-12)))
    k, _ = fit_gamma_moments(x, n)
    return {
        "events": N, "n_bins": int(len(x)), "n_empty_bins": int((n == 0).sum()),
        "range_min": float(x[m].min()) if m.any() else np.nan,
        "range_max": float(x[m].max()) if m.any() else np.nan,
        "weighted_mean": float((x * n).sum() / N) if N > 0 else np.nan,
        "weighted_median": wquantile(x, n, 0.5),
        "weighted_q25": q1, "weighted_q75": q3, "weighted_iqr": q3 - q1,
        "mode_bin_center": float(x[int(np.argmax(n))]) if N > 0 else np.nan,
        "mode_bin_count": float(n.max()),
        "frac_gt_10": float(n[x > thresh].sum() / N) if N > 0 else np.nan,
        "mu_binned_mle": muA, "sigma_binned_mle": sA,
        "median_binned_mle": float(np.exp(muA)),
        "mode_binned_mle": float(np.exp(muA - sA ** 2)),
        "mu_interval_mle": muB, "sigma_interval_mle": sB,
        "median_interval_mle": float(np.exp(muB)),
        "mode_interval_mle": float(np.exp(muB - sB ** 2)),
        "mu_moment": muC, "sigma_moment": sC, "median_moment": float(np.exp(muC)),
        "gamma_shape_k": k,
    }


def vector_bootstrap_stats(x, draws, thresh=10.0):
    """Shape statistics for many bootstrap replicates at once.  draws: (B, K)."""
    x = np.asarray(x, float)
    o = np.argsort(x)
    xs = x[o]
    D = np.asarray(draws, float)[:, o]
    tot = D.sum(axis=1)
    tot_safe = np.where(tot <= 0, 1.0, tot)
    cum = np.cumsum(D, axis=1)
    out = {}

    def wq(q):
        idx = np.clip((cum < (q * tot)[:, None]).sum(axis=1), 0, len(xs) - 1)
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
        mu2, s2 = lognormal_mle_newton(np.broadcast_to(lx, D.shape), D, n_iter=150)
    out["mu_binned_mle"] = mu2
    out["sigma_binned_mle"] = s2
    out["median_binned_mle"] = np.exp(mu2)
    out["mode_binned_mle"] = np.exp(mu2 - s2 ** 2)
    return out


# ==========================================================================
# workbook helpers
# ==========================================================================
def load_cache():
    z = np.load(NPZ, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def block_keys(sheet, cache):
    return [k for k in cache if k.split("__")[0] == sheet]


def bin_edges(x, h):
    """Monotone edges for a set of bin centres x with per-bin width h.

    Interior edges are the midpoints between consecutive SORTED centres; the
    outer edges extend by half the local width.  For uniformly spaced bins
    this reproduces the original grid exactly.  For the slow regime, whose
    raw centres are non-monotone at the block seam, this yields exactly
    len(x)+1 strictly increasing edges instead of a spurious negative-width
    bin."""
    x = np.asarray(x, float)
    h = np.asarray(h, float)
    o = np.argsort(x)
    xs, hs = x[o], h[o]
    mid = (xs[:-1] + xs[1:]) / 2.0
    return np.concatenate([[xs[0] - hs[0] / 2.0], mid, [xs[-1] + hs[-1] / 2.0]])


def block_edges(sheet, cache):
    """Edges matching the (sorted) bins of a whole sheet's concatenated blocks."""
    xs, hs = [], []
    for k in block_keys(sheet, cache):
        x = np.asarray(cache[k][0], float)
        h = float(np.median(np.diff(x))) if len(x) > 1 else 0.1
        xs.append(x)
        hs.append(np.full(len(x), h))
    return bin_edges(np.concatenate(xs), np.concatenate(hs))


def block_xy(sheet, cache):
    x = np.concatenate([np.asarray(cache[k][0], float) for k in block_keys(sheet, cache)])
    y = np.concatenate([np.asarray(cache[k][1], float) for k in block_keys(sheet, cache)])
    return x, y


def main() -> None:
    df = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    ori = pd.read_csv(DATA / "orientation_count_series.tsv", sep="\t")
    cache = load_cache()

    # ------------------------------------------------------------------
    # 0. structure verification
    # ------------------------------------------------------------------
    struct = []
    for pid, sub in df.groupby("population_series_id"):
        for regime, s in sub.groupby("drift_regime"):
            sheet = SHEET_OF_FAST[pid] if regime == "fast" else SLOW_SHEET_PREFIX[pid]
            s = s.sort_values("distribution_bin_center")
            xw, yw = block_xy(sheet, cache)
            xs, ns = s.distribution_bin_center.values, s.event_count.values
            struct.append({
                "population_series_id": pid, "regime": regime, "sheet": sheet,
                "n_bins_view": len(xs), "n_bins_workbook": len(xw),
                "bins_identical_ordered": bool(len(xs) == len(xw) and np.allclose(xs, xw, atol=1e-9)),
                "bins_identical_sorted": bool(len(xs) == len(xw) and np.allclose(np.sort(xs), np.sort(xw), atol=1e-9)),
                "counts_identical": bool(len(ns) == len(yw) and np.allclose(ns, yw)),
                "counts_sum_view": float(ns.sum()), "counts_sum_workbook": float(yw.sum()),
            })
    struct_df = pd.DataFrame(struct)
    print("=== chpol.0 drift_regime vs workbook blocks (hard equality test) ===")
    print(struct_df.to_string(index=False))
    struct_df.to_csv(OUT / "chpol_structure_check.csv", index=False)
    print(f"exact matches: {(struct_df.bins_identical_ordered & struct_df.counts_identical).sum()}/6 fast, "
          f"{(struct_df.bins_identical_sorted & struct_df.counts_identical).sum()}/6 slow (sorted)")

    # orientation -> workbook
    orows = []
    for sid, sheet in ORIENT_SHEET.items():
        v = ori[ori.orientation_series_id == sid].sort_values("relative_coordinate")
        x, y = block_xy(sheet, cache)
        keep = np.array([not np.isclose(xx, k) for k, xx in zip(np.round(x, 9),
                                                               np.round(x, 9))]) & np.isin(np.round(x, 9), np.round(v.relative_coordinate.values, 9))
        match = np.isin(np.round(x, 9), np.round(v.relative_coordinate.values, 9))
        orows.append({"orientation_series_id": sid, "sheet": sheet,
                      "view_n": len(v), "workbook_n": len(x),
                      "dropped_bins": int((~match).sum()),
                      "x_equal": bool(np.allclose(x[match], v.relative_coordinate.values)),
                      "y_equal": bool(np.allclose(y[match], v.event_count.values)),
                      "total_view": float(v.event_count.sum()),
                      "total_workbook": float(y.sum())})
    ori_df = pd.DataFrame(orows)
    print("\n=== chpol.0b orientation series vs workbook sheets ===")
    print(ori_df.to_string(index=False))
    ori_df.to_csv(OUT / "chpol_structure_check_orientation.csv", index=False)

    # Newton vs Nelder-Mead
    chk = []
    for pid, sub in df.groupby("population_series_id"):
        for regime, s in sub.groupby("drift_regime"):
            x, n = s.distribution_bin_center.values, s.event_count.values
            a, b = fit_lognormal_binned_mle(x, n), fit_lognormal_binned_mle_nm(x, n)
            chk.append({"population_series_id": pid, "regime": regime,
                        "mu_newton": a[0], "mu_nm": b[0], "abs_dmu": abs(a[0] - b[0]),
                        "sigma_newton": a[1], "sigma_nm": b[1], "abs_dsigma": abs(a[1] - b[1])})
    chk_df = pd.DataFrame(chk)
    print("\n=== chpol.0c Newton vs Nelder-Mead MLE cross-check ===")
    print(f"max |dmu|={chk_df.abs_dmu.max():.3e}  max |dsigma|={chk_df.abs_dsigma.max():.3e}")
    chk_df.to_csv(OUT / "chpol_fit_crosscheck.csv", index=False)

    # ------------------------------------------------------------------
    # 1..4 per-series summaries + vectorised multinomial bootstrap
    # ------------------------------------------------------------------
    rows, boot_rows = [], []
    for pid, sub in df.groupby("population_series_id"):
        grp, ch = sub.anonymous_source_group.iloc[0], sub.polarization_channel.iloc[0]
        for regime, s in sub.groupby("drift_regime"):
            s = s.sort_values("distribution_bin_center")
            x, n = s.distribution_bin_center.values.astype(float), s.event_count.values.astype(float)
            sheet = SHEET_OF_FAST[pid] if regime == "fast" else SLOW_SHEET_PREFIX[pid]
            edges = block_edges(sheet, cache)
            if len(edges) != len(x) + 1:
                raise RuntimeError(f"edges {len(edges)} != bins+1 {len(x)+1} for {pid}/{regime}")
            base = summaries(x, n, edges)
            base.update({"population_series_id": pid, "group": grp, "channel": ch,
                         "regime": regime, "workbook_sheet": sheet})
            rows.append(base)
            draws = RNG.multinomial(int(n.sum()), n / n.sum(), size=NBOOT).astype(float)
            for k, v in vector_bootstrap_stats(x, draws).items():
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
    sum_df.to_csv(OUT / "chpol_series_summaries.csv", index=False)
    boot_df.to_csv(OUT / "chpol_series_bootstrap_ci.csv", index=False)

    print("\n=== chpol.1 per-series summaries (unit = one source-polarization series) ===")
    print(sum_df[["group", "channel", "regime", "events", "n_bins", "n_empty_bins",
                  "weighted_median", "weighted_iqr", "mode_bin_center", "frac_gt_10",
                  "sigma_binned_mle", "sigma_interval_mle", "median_binned_mle"]].to_string(index=False))

    print("\n=== chpol.1b multinomial-bootstrap 95% CI, paired by group+regime ===")
    for stat in ["weighted_median", "sigma_binned_mle", "frac_gt_10", "weighted_iqr", "mode_bin_center"]:
        t = (boot_df[boot_df.stat == stat]
             .pivot(index=["group", "regime"], columns="channel",
                    values=["median_boot", "ci_lo", "ci_hi"]))
        print(f"\n-- {stat} --")
        print(t.to_string())

    # ------------------------------------------------------------------
    # 2. paired channel totals + Poisson-bootstrap ratios
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
            A = np.maximum(RNG.poisson(np.tile(va, (NBOOT, 1))).sum(axis=1), 1)
            B = np.maximum(RNG.poisson(np.tile(vb_, (NBOOT, 1))).sum(axis=1), 1)
            ratio = B / A
            scaled = (B / Nb) / (A / Na)
            r_lo, r_hi = np.percentile(ratio, [2.5, 97.5])
            s_lo, s_hi = np.percentile(scaled, [2.5, 97.5])
            ratio_rows.append({
                "group": grp, "regime": regime,
                "series_l": a.population_series_id.iloc[0], "series_r": b.population_series_id.iloc[0],
                "events_l": Na, "events_r": Nb,
                "ratio_r_over_l": Nb / Na, "ratio_ci_lo": float(r_lo), "ratio_ci_hi": float(r_hi),
                "ratio_excludes_1": bool(r_lo > 1 or r_hi < 1),
                "scaled_ratio": (Nb / len(vb_)) / (Na / len(va)),
                "scaled_ratio_ci_lo": float(s_lo), "scaled_ratio_ci_hi": float(s_hi),
                "scaled_excludes_1": bool(s_lo > 1 or s_hi < 1),
                "n_events_min": float(min(Na, Nb)),
                "insufficient": bool(min(Na, Nb) < 100),
            })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(OUT / "chpol_channel_ratio.csv", index=False)
    print("\n=== chpol.2 paired channel totals and Poisson-bootstrap ratios ===")
    print(ratio_df[["group", "regime", "events_l", "events_r", "ratio_r_over_l", "ratio_ci_lo",
                    "ratio_ci_hi", "ratio_excludes_1", "scaled_ratio", "scaled_ratio_ci_lo",
                    "scaled_ratio_ci_hi", "insufficient"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 3/5. shape contrast on a COMMON grid + permutation test
    # ------------------------------------------------------------------
    shape_rows, gridded = [], {}
    for regime in ["fast", "slow"]:
        for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
            sel = df[(df.anonymous_source_group == grp) & (df.drift_regime == regime)]
            cl, cr = "channel_l", "channel_r"
            sl, sr = sel[sel.polarization_channel == cl], sel[sel.polarization_channel == cr]
            lo = max(sl.distribution_bin_center.min(), sr.distribution_bin_center.min())
            hi = min(sl.distribution_bin_center.max(), sr.distribution_bin_center.max())
            grid = np.linspace(lo + 1e-6, hi - 1e-6, 40)
            gstep = grid[1] - grid[0]
            edges = np.r_[grid[0] - gstep / 2, (grid[:-1] + grid[1:]) / 2, grid[-1] + gstep / 2]
            hl, _ = np.histogram(sl.distribution_bin_center, bins=edges, weights=sl.event_count.values)
            hr, _ = np.histogram(sr.distribution_bin_center, bins=edges, weights=sr.event_count.values)
            hl, hr = hl.astype(float), hr.astype(float)
            Nl, Nr = hl.sum(), hr.sum()
            if Nl < 1 or Nr < 1:
                continue
            tv = 0.5 * np.abs(hl / Nl - hr / Nr).sum()
            pooled = hl + hr
            p = pooled / pooled.sum()
            null_tv = np.empty(NPERM)
            for i in range(NPERM):
                aa = RNG.multinomial(int(Nl), p)
                bb = RNG.multinomial(int(Nr), p)
                null_tv[i] = 0.5 * np.abs(aa / Nl - bb / Nr).sum()
            shape_rows.append({
                "group": grp, "regime": regime, "channel_l": cl, "channel_r": cr,
                "grid_lo": float(grid[0]), "grid_hi": float(grid[-1]), "n_grid": len(grid),
                "N_l": float(Nl), "N_r": float(Nr), "tv_distance": float(tv),
                "null_tv_median": float(np.median(null_tv)),
                "null_tv_p95": float(np.percentile(null_tv, 95)),
                "p_permutation": float((null_tv >= tv).mean()),
                "tv_over_null_median": float(tv / np.median(null_tv)),
                "insufficient": bool(min(Nl, Nr) < 100),
            })
            gridded[f"{grp}|{regime}|{cl}"] = hl
            gridded[f"{grp}|{regime}|{cr}"] = hr
            gridded[f"{grp}|{regime}|grid"] = grid
    shape_df = pd.DataFrame(shape_rows)
    shape_df.to_csv(OUT / "chpol_shape_contrast.csv", index=False)
    np.savez_compressed(OUT / "chpol_common_grid_hist.npz", **gridded)
    print("\n=== chpol.3 shape contrast on common grid (TV vs permutation null) ===")
    print(shape_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. orientation channel pairs + C-variant sheets
    # ------------------------------------------------------------------
    ori_pairs = [
        ("anonymous_group_1", "orientation_series_1", "orientation_series_2"),
        ("anonymous_group_2", "orientation_series_3", "orientation_series_4"),
        ("anonymous_group_3", "orientation_series_5", "orientation_series_6"),
    ]
    orows2 = []
    for grp, sl_, sr_ in ori_pairs:
        a = ori[ori.orientation_series_id == sl_].sort_values("relative_coordinate")
        b = ori[ori.orientation_series_id == sr_].sort_values("relative_coordinate")
        va, vb_ = a.event_count.values.astype(float), b.event_count.values.astype(float)
        xg = a.relative_coordinate.values
        Na, Nb = va.sum(), vb_.sum()
        tv = 0.5 * np.abs(va / Na - vb_ / Nb).sum()
        A = np.maximum(RNG.poisson(np.tile(va, (NBOOT, 1))).sum(axis=1), 1)
        B = np.maximum(RNG.poisson(np.tile(vb_, (NBOOT, 1))).sum(axis=1), 1)
        r = B / A
        # coordinate-level signed contrast, weighted by the pair mean
        wmean_pair = (va + vb_) / 2
        neg = xg < -10
        orows2.append({
            "group": grp, "series_l": sl_, "series_r": sr_,
            "events_l": Na, "events_r": Nb, "ratio_r_over_l": Nb / Na,
            "ratio_ci_lo": float(np.percentile(r, 2.5)), "ratio_ci_hi": float(np.percentile(r, 97.5)),
            "tv_distance": float(tv),
            "share_lt_-10_l": float(va[neg].sum() / Na), "share_lt_-10_r": float(vb_[neg].sum() / Nb),
            "com_l": float((xg * va).sum() / Na), "com_r": float((xg * vb_).sum() / Nb),
            "peak_coord_l": float(xg[int(np.argmax(va))]), "peak_coord_r": float(xg[int(np.argmax(vb_))]),
            "n_nonzero_coords_in_both": int(((va > 0) & (vb_ > 0)).sum()),
            "n_coords": int(len(xg)),
        })
    ori_pair_df = pd.DataFrame(orows2)
    ori_pair_df.to_csv(OUT / "chpol_orientation_pairs.csv", index=False)
    print("\n=== chpol.5 orientation channel pairs (unit = one orientation series) ===")
    print(ori_pair_df.to_string(index=False))

    # C-variant sheets 046/047 vs the non-C group_b channel_l blocks
    crows = []
    gb_l_sheets = {"(-35<s<-10)": "Sheet_029", "(-10<s<0)": "Sheet_028", "(0<s<20)": "Sheet_028"}
    ref = {"(-35<s<-10)": 0, "(-10<s<0)": 0, "(0<s<20)": 1}
    for sheet in ["Sheet_046", "Sheet_047"]:
        keys = block_keys(sheet, cache)
        labels = ["(-35<s<-10)", "(-10<s<0)", "(0<s<20)"]
        for k, lab in zip(keys, labels):
            x, y = np.asarray(cache[k][0], float), np.asarray(cache[k][1], float)
            rs = gb_l_sheets[lab]
            rx, ry = block_xy(rs, cache)
            rk = block_keys(rs, cache)
            rblk = np.asarray(cache[rk[ref[lab]]][1], float)
            crows.append({
                "sheet": sheet, "block": lab, "n_bins": len(x), "workbook_total": float(y.sum()),
                "group_b_channel_l_total": float(rblk.sum()),
                "difference": float(y.sum() - rblk.sum()),
                "n_bins_ref": len(rblk),
                "label_says": C_VARIANT.get(sheet, ""),
            })
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "chpol_group_b_C_variant.csv", index=False)
    print("\n=== chpol.5b group_b 'C' variant sheets vs group_b channel_l ===")
    print(cdf.to_string(index=False))

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
        "orientation_pairs": json.loads(ori_pair_df.to_json(orient="records")),
        "group_b_C_variant": json.loads(cdf.to_json(orient="records")),
        "structure_check_drift": json.loads(struct_df.to_json(orient="records")),
        "structure_check_orientation": json.loads(ori_df.to_json(orient="records")),
        "fit_crosscheck": json.loads(chk_df.to_json(orient="records")),
        "caveats": [
            "Bins are nested repeats inside one series, not independent samples.",
            "Only 6 source-polarization series exist: 3 paired comparisons per regime.",
            "Event counts are drifting sub-burst counts, not independent burst events.",
            "Multinomial bootstrap CIs describe shape conditional on the series total.",
            "Poisson bootstrap CIs describe totals and are used for the channel ratios.",
            "The slow regime concatenates two workbook blocks with a bookkeeping seam.",
        ],
    }
    with open(OUT / "chpol_channel_contrast.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print("\nwrote chpol_*.csv / chpol_channel_contrast.json / chpol_common_grid_hist.npz")


if __name__ == "__main__":
    main()