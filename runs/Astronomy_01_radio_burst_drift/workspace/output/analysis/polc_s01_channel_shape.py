#!/usr/bin/env python3
"""POLC s01 - polarization-channel contrast with an uncertainty-aware shape comparison.

Dimension: finish the polarization-channel (channel_l vs channel_r) contrast for the
six anonymous (group, regime) cells using the ORIGINAL published bin grids.

Analysis unit = one complete source-polarization series (a whole binned drift-rate
profile for one (group, channel, regime)).  Bins are NESTED REPEATS of that series,
never independent samples; every uncertainty statement below is therefore about
COUNTING NOISE inside one series, not about source-to-source scatter.

Inputs (all under <root>/data or <root>/output/results):
  data/normalized_views/drift_population_series.tsv          the 12 published series
  output/results/s11_blocks.csv                              tidy workbook blocks
Outputs: output/results/polc_*.csv|json, output/results/figures/polc_*.png

Nothing is re-parsed from the workbook here: the workbook block table s11_blocks.csv
is reused, and every block used is re-verified elementwise against the normalized
view before it enters an analysis (see step 1 of main()).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(divide="ignore", invalid="ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized_views"
RES = ROOT / "output" / "results"
FIG = RES / "figures"
LOG = ROOT / "output" / "logs"

RNG = np.random.default_rng(20221013)
NBOOT = 5000          # Poisson noise bootstrap replicates (>= 2000 required)
NNULL = 5000          # null (common-shape) Monte Carlo replicates
THRESH = 10.0         # MHz/s, for "fraction of events above 10"

# (group, regime) -> (channel_l sheet, channel_r sheet); slow sheets carry 2 blocks each
CELL_SHEETS = {
    ("anonymous_group_1", "fast"): ("Sheet_025", "Sheet_027"),
    ("anonymous_group_1", "slow"): ("Sheet_024", "Sheet_026"),
    ("anonymous_group_2", "fast"): ("Sheet_029", "Sheet_031"),
    ("anonymous_group_2", "slow"): ("Sheet_028", "Sheet_030"),
    ("anonymous_group_3", "fast"): ("Sheet_033", "Sheet_035"),
    ("anonymous_group_3", "slow"): ("Sheet_032", "Sheet_034"),
}
SERIES_ID = {  # (group, regime, channel) -> population_series_id
    ("anonymous_group_1", "fast", "channel_l"): "population_series_1",
    ("anonymous_group_1", "fast", "channel_r"): "population_series_2",
    ("anonymous_group_2", "fast", "channel_l"): "population_series_3",
    ("anonymous_group_2", "fast", "channel_r"): "population_series_4",
    ("anonymous_group_3", "fast", "channel_l"): "population_series_5",
    ("anonymous_group_3", "fast", "channel_r"): "population_series_6",
    ("anonymous_group_1", "slow", "channel_l"): "population_series_1",
    ("anonymous_group_1", "slow", "channel_r"): "population_series_2",
    ("anonymous_group_2", "slow", "channel_l"): "population_series_3",
    ("anonymous_group_2", "slow", "channel_r"): "population_series_4",
    ("anonymous_group_3", "slow", "channel_l"): "population_series_5",
    ("anonymous_group_3", "slow", "channel_r"): "population_series_6",
}
COARSE = {  # group, channel -> coarse sheet
    ("anonymous_group_1", "channel_l"): "Sheet_039",
    ("anonymous_group_1", "channel_r"): "Sheet_040",
    ("anonymous_group_2", "channel_l"): "Sheet_041",
    ("anonymous_group_2", "channel_r"): "Sheet_042",
    ("anonymous_group_3", "channel_l"): "Sheet_043",
    ("anonymous_group_3", "channel_r"): "Sheet_044",
}


# ---------------------------------------------------------------------------
# weighted statistics on a published (original) bin grid, counts as weights
# ---------------------------------------------------------------------------
def _clean(centers, counts):
    c = np.asarray(centers, float)
    w = np.asarray(counts, float)
    ok = np.isfinite(c) & np.isfinite(w) & (w > 0)
    o = np.argsort(c[ok], kind="stable")
    return c[ok][o], w[ok][o]


def weighted_quantile(centers, counts, qs, convention="midpoint"):
    """Quantile of the discrete distribution that puts mass w_i at center c_i.

    'midpoint': cumulative weight is located at the middle of each bin's mass
                (np.interp between the half-weight cumulative positions);
    'step'    : smallest observed centre whose cumulative weight reaches q*W.
    Both are reported so the reader can see the convention sensitivity.
    """
    c, w = _clean(centers, counts)
    qs = np.atleast_1d(np.asarray(qs, float))
    if len(c) == 0:
        return np.full(len(qs), np.nan)
    W = w.sum()
    if convention == "midpoint":
        cw = np.cumsum(w) - 0.5 * w
        return np.interp(qs * W, cw, c)
    cs = np.cumsum(w)
    idx = np.searchsorted(cs, qs * W, side="left").clip(0, len(c) - 1)
    return c[idx]


def density_quantile(centers, counts, widths, qs):
    """Quantile assuming a uniform event density inside each published bin.

    This is the bin-width-aware counterpart of `weighted_quantile` and is the
    robustness check for the fact that the two channels' published bin widths
    are not identical (e.g. 1.035 vs 1.023 MHz/s in group_1 fast).
    """
    c = np.asarray(centers, float)
    w = np.asarray(counts, float)
    wd = np.asarray(widths, float)
    ok = np.isfinite(c) & np.isfinite(w) & (w > 0) & np.isfinite(wd) & (wd > 0)
    c, w, wd = c[ok], w[ok], wd[ok]
    o = np.argsort(c, kind="stable")
    c, w, wd = c[o], w[o], wd[o]
    if len(c) == 0:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    lo = c - wd / 2.0
    hi = c + wd / 2.0
    W = w.sum()
    csum = np.cumsum(w)
    out = []
    for q in np.atleast_1d(qs):
        t = q * W
        i = int(min(np.searchsorted(csum, t, side="left"), len(c) - 1))
        prev = csum[i] - w[i]
        frac = 0.0 if w[i] <= 0 else (t - prev) / w[i]
        out.append(lo[i] + np.clip(frac, 0.0, 1.0) * (hi[i] - lo[i]))
    return np.array(out)


def profile_stats(centers, counts, widths):
    """All weighted summaries required for one published series."""
    c, w = _clean(centers, counts)
    N = w.sum()
    if N <= 0:
        return None
    q = weighted_quantile(c, w, [0.25, 0.5, 0.75])
    qs = weighted_quantile(c, w, [0.25, 0.5, 0.75], "step")
    qd = density_quantile(centers, counts, widths, [0.25, 0.5, 0.75])
    imax = int(np.argmax(w))
    return dict(
        n_events=int(N), n_bins=int(len(c)),
        n_occupied_bins=int((w > 0).sum()),
        median=float(q[1]), q25=float(q[0]), q75=float(q[2]), iqr=float(q[2] - q[0]),
        median_step=float(qs[1]), q25_step=float(qs[0]), q75_step=float(qs[2]),
        median_density=float(qd[1]), iqr_density=float(qd[2] - qd[0]),
        mode_bin_center=float(c[imax]), mode_bin_count=int(w[imax]),
        frac_gt_10=float(w[c > THRESH].sum() / N),
        min_center=float(c.min()), max_center=float(c.max()),
    )


# ---------------------------------------------------------------------------
# re-binning onto a common edge grid (overlap-proportional, no count invented)
# ---------------------------------------------------------------------------
def overlap_rebin(centers, counts, widths, edges):
    """Split each source bin across target bins in proportion to x-overlap."""
    edges = np.asarray(edges, float)
    cc = np.asarray(centers, float)
    ww = np.full(len(cc), float(widths)) if np.ndim(widths) == 0 else np.asarray(widths, float)
    out = np.zeros(len(edges) - 1)
    for c, k, d in zip(cc, np.asarray(counts, float), ww):
        if not np.isfinite(c) or not np.isfinite(d) or d <= 0 or k <= 0:
            continue
        lo, hi = c - d / 2.0, c + d / 2.0
        i0 = np.searchsorted(edges, lo, "right") - 1
        i1 = np.searchsorted(edges, hi, "right") - 1
        for t in range(max(i0, 0), min(i1, len(out) - 1) + 1):
            ov = min(hi, edges[t + 1]) - max(lo, edges[t])
            if ov > 0:
                out[t] += k * ov / d
    return out


def common_edges(centers_a, centers_b, width_a, width_b, step):
    """Uniform common grid covering the overlap of the two published supports.

    `width_a` / `width_b` may be a scalar or a per-bin array (each bin's support
    is [center - width/2, center + width/2]).
    """
    ca = np.asarray(centers_a, float)
    cb = np.asarray(centers_b, float)
    wa = np.full(len(ca), float(width_a)) if np.ndim(width_a) == 0 else np.asarray(width_a, float)
    wb = np.full(len(cb), float(width_b)) if np.ndim(width_b) == 0 else np.asarray(width_b, float)
    la = float(np.min(ca - wa / 2.0))
    ha = float(np.max(ca + wa / 2.0))
    lb = float(np.min(cb - wb / 2.0))
    hb = float(np.max(cb + wb / 2.0))
    lo, hi = max(la, lb), min(ha, hb)
    n = max(int(np.ceil((hi - lo) / step)), 4)
    e = lo + np.arange(n + 1) * (hi - lo) / n
    return e


def g_stat(obs_a, obs_b, pbar):
    """G statistic of a common shape pbar given two count vectors."""
    ea = obs_a.sum() * pbar
    eb = obs_b.sum() * pbar
    return float(2 * np.sum(
        np.where(obs_a > 0, obs_a * np.log(obs_a / np.maximum(ea, 1e-12)), 0.0)
        + np.where(obs_b > 0, obs_b * np.log(obs_b / np.maximum(eb, 1e-12)), 0.0)))


def window_table(yl, cl, yr, cr, rng, cell_label="", n_mc=5000):
    """Channel x drift-window contingency test, with the ratio held FIXED.

    The correct null for a window-restricted 2x2 table (channel x window) is a
    constant l/r RATIO across windows, not an equal split: the channels are known
    a priori to have different totals (this IS the count contrast). The exact
    conditional test given both margins is a binomial on the cell counts with
    p = N_l/(N_l+N_r); the Monte-Carlo G uses the same fixed-ratio null on a
    window x window-set grid.
    """
    from scipy.stats import binomtest
    windows = [("-2 < s < 2", -2.0, 2.0), ("0 < s < 2", 0.0, 2.0),
               ("2 < s < 12", 2.0, 12.0), ("s > 12", 12.0, np.inf)]
    rows = []
    for i, (lab, lo, hi) in enumerate(windows):
        Nwl = float(yl[(cl > lo) & (cl < hi)].sum())
        Nwr = float(yr[(cr > lo) & (cr < hi)].sum())
        rows.append(dict(cell=cell_label, window=lab, lo=lo, hi=hi,
                         events_l=Nwl, events_r=Nwr,
                         ratio_l_over_r=(Nwl / Nwr if Nwr else np.nan)))
    W = pd.DataFrame(rows)
    tot_l, tot_r = float(W.events_l.sum()), float(W.events_r.sum())
    W["expected_l_under_constant_ratio"] = tot_l * (W.events_l + W.events_r) / (tot_l + tot_r)
    W["expected_r_under_constant_ratio"] = tot_r * (W.events_l + W.events_r) / (tot_l + tot_r)
    p_l = tot_l / (tot_l + tot_r)
    W["binom_p_constant_ratio"] = [
        float(binomtest(int(r.events_l), int(r.events_l + r.events_r), p_l).pvalue)
        if (r.events_l + r.events_r) > 0 else np.nan for r in W.itertuples()]
    # joint Monte-Carlo G on the window grid under the constant-ratio null
    K = len(W)
    t = np.array([r.events_l + r.events_r for r in W.itertuples()])
    prob = t / t.sum() if t.sum() > 0 else np.full(K, 1.0 / K)
    obs_l = np.array([r.events_l for r in W.itertuples()])
    obs_r = np.array([r.events_r for r in W.itertuples()])
    g_obs = g_stat(obs_l, obs_r, prob)
    g_mc = np.empty(n_mc)
    for b in range(n_mc):
        sl = RNG.poisson(tot_l * prob)
        sr = RNG.poisson(tot_r * prob)
        g_mc[b] = g_stat(sl, sr, prob)
    W["G_joint_window"] = g_obs
    W["p_G_joint_window_monte_carlo"] = float((g_mc >= g_obs).mean())
    W["G_joint_null_median"] = float(np.median(g_mc))
    W["G_joint_null_p95"] = float(np.percentile(g_mc, 95))
    return W


def tv_distance(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return np.nan
    return float(0.5 * np.abs(p / sp - q / sq).sum())


# ---------------------------------------------------------------------------
# input assembly
# ---------------------------------------------------------------------------
def load_all():
    pop = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    blk = pd.read_csv(RES / "s11_blocks.csv")
    blocks = {}
    for (sh, lab), g in blk.groupby(["sheet", "block_label"], sort=False):
        blocks.setdefault(sh, []).append(
            (lab, g["x"].to_numpy(float), g["y"].to_numpy(float)))
    return pop, blocks


def verify_against_views(pop, blocks, log):
    """Re-verify every block used here against the published normalized view."""
    checks = []
    for (grp, reg), (shl, shr) in CELL_SHEETS.items():
        for ch, sh in (("channel_l", shl), ("channel_r", shr)):
            pid = SERIES_ID[(grp, reg, ch)]
            v = pop[(pop.population_series_id == pid) & (pop.drift_regime == reg)]
            bx = np.concatenate([b[1] for b in blocks[sh]])
            by = np.concatenate([b[2] for b in blocks[sh]])
            vo = v.sort_values("distribution_bin_center")
            # normalized view: fast is already sorted; slow is a raw concatenation
            vx_file = v["distribution_bin_center"].to_numpy(float)
            vy_file = v["event_count"].to_numpy(float)
            dx_file = float(np.max(np.abs(vx_file - bx)))
            dy_file = float(np.max(np.abs(vy_file - by)))
            vx_s, vy_s = vo["distribution_bin_center"].to_numpy(float), vo["event_count"].to_numpy(float)
            dx_s = float(np.max(np.abs(vx_s - np.sort(bx)))) if len(vx_s) == len(bx) else np.nan
            checks.append(dict(group=grp, regime=reg, channel=ch, sheet=sh,
                               n_view=len(vx_file), n_blocks=len(bx),
                               max_abs_dx_file_order=dx_file,
                               max_abs_dy_file_order=dy_file,
                               max_abs_dx_sorted=dx_s,
                               view_total=float(vy_file.sum()),
                               blocks_total=float(by.sum())))
    df = pd.DataFrame(checks)
    df.to_csv(RES / "polc_s00_reverify.csv", index=False)
    df.to_json(RES / "polc_s00_reverify.json", orient="records", indent=2)
    log(f"[verify] workbook blocks vs normalized views, elementwise, FILE ORDER: "
        f"max |dy| = {df.max_abs_dy_file_order.max():.3g} (counts, exact 0), "
        f"max |dx| = {df.max_abs_dx_file_order.max():.3g} (bin centres, float-write rounding); "
        f"identical within 1e-9 = "
        f"{bool((df.max_abs_dy_file_order < 1e-9).all() and (df.max_abs_dx_file_order < 1e-9).all())}; "
        f"all 12 series' totals match = "
        f"{bool((df.view_total == df.blocks_total).all())}")
    return df


def cell_arrays(blocks, grp, reg, ch):
    """Original published (centers, counts, widths) for one series, file order."""
    sh = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
    cs, ys, ws = [], [], []
    for lab, x, y in blocks[sh]:
        dx = np.full(len(x), float(np.median(np.diff(x))) if len(x) > 1 else np.nan)
        cs.append(x)
        ys.append(y)
        ws.append(dx)
    return np.concatenate(cs), np.concatenate(ys), np.concatenate(ws)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fh = open(LOG / "polc_s01.log", "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        fh.write(s + "\n")
        fh.flush()

    pop, blocks = load_all()
    log("=== POLC s01: polarization-channel shape contrast ===")
    verify_against_views(pop, blocks, log)

    # ---------------- 1. per-series weighted statistics ----------------
    rows, cells = [], {}
    for (grp, reg) in CELL_SHEETS:
        for ch in ("channel_l", "channel_r"):
            c, y, w = cell_arrays(blocks, grp, reg, ch)
            st = profile_stats(c, y, w)
            st.update(group=grp, regime=reg, channel=ch,
                      sheet=CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1],
                      n_source_blocks=int(len(blocks[CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]])),
                      total_events=int(y.sum()))
            rows.append(st)
            cells[(grp, reg, ch)] = (c, y, w)
    stats = pd.DataFrame(rows)
    stats = stats[["group", "regime", "channel", "sheet", "n_source_blocks",
                   "total_events", "n_events", "n_bins", "n_occupied_bins",
                   "q25", "median", "q75", "iqr", "q25_step", "median_step",
                   "q75_step", "median_density", "iqr_density",
                   "mode_bin_center", "mode_bin_count", "frac_gt_10",
                   "min_center", "max_center"]]
    stats.to_csv(RES / "polc_s01_series_stats.csv", index=False)
    stats.to_json(RES / "polc_s01_series_stats.json", orient="records", indent=2)
    log("\n=== 1. weighted statistics on the ORIGINAL published grids ===")
    log(stats.to_string(index=False))

    # slow regimes have two overlapping source blocks -> report them separately too
    sub_rows = []
    for (grp, reg) in CELL_SHEETS:
        if reg != "slow":
            continue
        for ch in ("channel_l", "channel_r"):
            sh = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
            for bi, (lab, x, y) in enumerate(blocks[sh]):
                w = np.full(len(x), float(np.median(np.diff(x))))
                st = profile_stats(x, y, w)
                st.update(group=grp, regime=reg, channel=ch, sheet=sh,
                          block_index=bi, block_label=lab)
                sub_rows.append(st)
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(RES / "polc_s02_slow_subblocks.csv", index=False)
    sub.to_json(RES / "polc_s02_slow_subblocks.json", orient="records", indent=2)
    log("\n=== 1b. the slow series decomposed into its two source blocks ===")
    log(sub[["group", "channel", "block_index", "block_label", "n_events", "median",
             "iqr", "mode_bin_center", "frac_gt_10", "min_center", "max_center"]].to_string(index=False))

    # ---------------- 2. paired channel contrast + Poisson bootstrap ----------------
    cont_rows, knit = [], []
    for (grp, reg) in CELL_SHEETS:
        cl, yl, wl = cells[(grp, reg, "channel_l")]
        cr, yr, wr = cells[(grp, reg, "channel_r")]
        sl = profile_stats(cl, yl, wl)
        sr = profile_stats(cr, yr, wr)
        # common grid for shape statistics
        step = 0.5 if reg == "fast" else 0.25
        edges = common_edges(cl, cr, wl, wr, step)
        pl = overlap_rebin(cl, yl, wl, edges)
        pr = overlap_rebin(cr, yr, wr, edges)
        tv_obs = tv_distance(pl, pr)
        Nl, Nr = yl.sum(), yr.sum()
        dmed_obs = sl["median"] - sr["median"]
        diqr_obs = sl["iqr"] - sr["iqr"]
        dfrac_obs = sl["frac_gt_10"] - sr["frac_gt_10"]

        # --- (a) Poisson counting-noise bootstrap around the OBSERVED profiles ---
        boot_dmed = np.empty(NBOOT)
        boot_diqr = np.empty(NBOOT)
        boot_dfrac = np.empty(NBOOT)
        boot_tv = np.empty(NBOOT)
        for b in range(NBOOT):
            yl_s = RNG.poisson(yl)
            yr_s = RNG.poisson(yr)
            sl_s = profile_stats(cl, yl_s, wl)
            sr_s = profile_stats(cr, yr_s, wr)
            boot_dmed[b] = sl_s["median"] - sr_s["median"]
            boot_diqr[b] = sl_s["iqr"] - sr_s["iqr"]
            boot_dfrac[b] = sl_s["frac_gt_10"] - sr_s["frac_gt_10"]
            boot_tv[b] = tv_distance(overlap_rebin(cl, yl_s, wl, edges),
                                     overlap_rebin(cr, yr_s, wr, edges))

        # --- (b) null Monte Carlo: SAME shape (pooled over the COMMON support),
        #     each channel keeping its own in-range total ---
        ctr_e = edges[:-1] + np.diff(edges) / 2
        in_l = (cl >= edges[0]) & (cl <= edges[-1])
        in_r = (cr >= edges[0]) & (cr <= edges[-1])
        nl_in = pl.sum()
        nr_in = pr.sum()
        pooled = (pl + pr) / (pl.sum() + pr.sum())
        nu_dmed = np.empty(NNULL)
        nu_tv = np.empty(NNULL)
        for b in range(NNULL):
            yl_s = RNG.poisson(nl_in * pooled)
            yr_s = RNG.poisson(nr_in * pooled)
            nu_tv[b] = tv_distance(yl_s, yr_s)
            ml = weighted_quantile(ctr_e, yl_s, [0.5])[0]
            mr = weighted_quantile(ctr_e, yr_s, [0.5])[0]
            nu_dmed[b] = ml - mr

        ci = np.percentile(boot_dmed, [2.5, 97.5])
        ci_iqr = np.percentile(boot_diqr, [2.5, 97.5])
        ci_frac = np.percentile(boot_dfrac, [2.5, 97.5])
        ci_tv = np.percentile(boot_tv, [2.5, 97.5])
        p_shift = float((np.abs(nu_dmed) >= abs(dmed_obs)).mean())
        p_shape_tv = float((nu_tv >= tv_obs).mean())
        rec = dict(
            group=grp, regime=reg,
            total_events_channel_l=int(Nl), total_events_channel_r=int(Nr),
            count_ratio_l_over_r=float(Nl / Nr) if Nr else np.nan,
            median_l=sl["median"], median_r=sr["median"], median_diff=dmed_obs,
            median_diff_ci_lo=float(ci[0]), median_diff_ci_hi=float(ci[1]),
            iqr_l=sl["iqr"], iqr_r=sr["iqr"], iqr_diff=diqr_obs,
            iqr_diff_ci_lo=float(ci_iqr[0]), iqr_diff_ci_hi=float(ci_iqr[1]),
            frac_gt_10_l=sl["frac_gt_10"], frac_gt_10_r=sr["frac_gt_10"],
            frac_gt_10_diff=dfrac_obs,
            frac_gt_10_diff_ci_lo=float(ci_frac[0]), frac_gt_10_diff_ci_hi=float(ci_frac[1]),
            common_grid_n_bins=int(len(pl)), common_grid_step=float(step),
            tv_observed=tv_obs, tv_noise_ci_lo=float(ci_tv[0]), tv_noise_ci_hi=float(ci_tv[1]),
            tv_null_median=float(np.median(nu_tv)), tv_null_p95=float(np.percentile(nu_tv, 95)),
            p_shape_same_tv=p_shape_tv,
            p_median_shift_null=p_shift,
            null_median_diff_p2_5=float(np.percentile(nu_dmed, 2.5)),
            null_median_diff_p97_5=float(np.percentile(nu_dmed, 97.5)),
            noise_only_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
            events_l_in_common_support=float(nl_in),
            events_r_in_common_support=float(nr_in),
            events_l_outside_common_support=float(Nl - nl_in),
            events_r_outside_common_support=float(Nr - nr_in),
        )
        cont_rows.append(rec)
        knit.append(dict(group=grp, regime=reg, channel_l_median=sl["median"],
                         channel_r_median=sr["median"],
                         median_density_l=sl["median_density"],
                         median_density_r=sr["median_density"],
                         median_step_l=sl["median_step"], median_step_r=sr["median_step"]))
        log(f"\n--- {grp} / {reg}: N_l={int(Nl)} N_r={int(Nr)} ratio_l/r={Nl/Nr:.3f}")
        log(f"    median l={sl['median']:.4f} r={sr['median']:.4f} diff={dmed_obs:+.4f} "
            f"95% Poisson-noise CI [{ci[0]:+.4f},{ci[1]:+.4f}]")
        log(f"    density-median diff={sl['median_density']-sr['median_density']:+.4f}  "
            f"step-median diff={sl['median_step']-sr['median_step']:+.4f}")
        log(f"    IQR l={sl['iqr']:.4f} r={sr['iqr']:.4f} diff={diqr_obs:+.4f} CI [{ci_iqr[0]:+.4f},{ci_iqr[1]:+.4f}]")
        log(f"    frac>10 l={sl['frac_gt_10']:.4f} r={sr['frac_gt_10']:.4f} diff={dfrac_obs:+.4f} "
            f"CI [{ci_frac[0]:+.4f},{ci_frac[1]:+.4f}]")
        log(f"    TV={tv_obs:.4f} noise 95% [{ci_tv[0]:.4f},{ci_tv[1]:.4f}] "
            f"null median={np.median(nu_tv):.4f} p95={np.percentile(nu_tv,95):.4f} p_shape={p_shape_tv:.4f}")
        log(f"    p(median shift | same shape)={p_shift:.4f}")
        gc.collect()

    cont = pd.DataFrame(cont_rows)
    # column order for readability
    first = ["group", "regime", "total_events_channel_l", "total_events_channel_r",
             "count_ratio_l_over_r", "events_l_in_common_support", "events_r_in_common_support",
             "events_l_outside_common_support", "events_r_outside_common_support",
             "median_l", "median_r", "median_diff", "median_diff_ci_lo", "median_diff_ci_hi",
             "iqr_l", "iqr_r", "iqr_diff", "iqr_diff_ci_lo", "iqr_diff_ci_hi",
             "frac_gt_10_l", "frac_gt_10_r", "frac_gt_10_diff",
             "frac_gt_10_diff_ci_lo", "frac_gt_10_diff_ci_hi",
             "common_grid_n_bins", "common_grid_step", "tv_observed",
             "tv_noise_ci_lo", "tv_noise_ci_hi", "tv_null_median", "tv_null_p95",
             "p_shape_same_tv", "p_median_shift_null", "null_median_diff_p2_5",
             "null_median_diff_p97_5", "noise_only_excludes_zero"]
    cont = cont[first]
    cont.to_csv(RES / "polc_s03_channel_contrast.csv", index=False)
    cont.to_json(RES / "polc_s03_channel_contrast.json", orient="records", indent=2)
    pd.DataFrame(knit).to_csv(RES / "polc_s03_median_conventions.csv", index=False)
    log("\n=== 2. paired channel contrast (bootstrap = COUNTING NOISE ONLY) ===")
    log(cont[["group", "regime", "total_events_channel_l", "total_events_channel_r",
              "median_diff", "median_diff_ci_lo", "median_diff_ci_hi",
              "p_median_shift_null", "tv_observed", "tv_null_p95", "p_shape_same_tv"]].to_string(index=False))
    log("\n  median-definition sensitivity (diff l-r):")
    kd = pd.DataFrame(knit)
    kd["midpoint_diff"] = kd.channel_l_median - kd.channel_r_median
    kd["density_diff"] = kd.median_density_l - kd.median_density_r
    kd["step_diff"] = kd.median_step_l - kd.median_step_r
    log(kd[["group", "regime", "midpoint_diff", "density_diff", "step_diff"]].to_string(index=False))

    # ---------------- 3. group_2 fast: is the ~6.9 count ratio + shape compatible? ----------
    grp, reg = "anonymous_group_2", "fast"
    cl, yl, wl = cells[(grp, reg, "channel_l")]
    cr, yr, wr = cells[(grp, reg, "channel_r")]
    Nl, Nr = int(yl.sum()), int(yr.sum())
    lo_support = max(cl.min() - np.median(wl) / 2, cr.min() - np.median(wr) / 2)
    hi_support = min(cl.max() + np.median(wl) / 2, cr.max() + np.median(wr) / 2)
    below = float(yl[cl < cr.min() - np.median(wr) / 2].sum())
    g2_rows = []
    g2 = dict(
        N_l=Nl, N_r=Nr, ratio=float(Nl / Nr),
        ratio_poisson_rel_se=float(np.sqrt(1.0 / Nl + 1.0 / Nr)),
        ratio_ci_lo=float(Nl / Nr * (1 - 1.96 * np.sqrt(1.0 / Nl + 1.0 / Nr))),
        ratio_ci_hi=float(Nl / Nr * (1 + 1.96 * np.sqrt(1.0 / Nl + 1.0 / Nr))),
        support_common_lo=float(lo_support), support_common_hi=float(hi_support),
        channel_l_events_below_common_support=below,
        channel_l_events_below_common_support_frac=float(below / Nl),
        channel_r_min_center=float(cr.min()), channel_l_min_center=float(cl.min()),
    )
    # two-sample homogeneity test on a 0.5 MHz/s common grid, Monte Carlo null
    step = 0.5
    edges = common_edges(cl, cr, wl, wr, step)
    pl = overlap_rebin(cl, yl, wl, edges)
    pr = overlap_rebin(cr, yr, wr, edges)
    n_l_common, n_r_common = pl.sum(), pr.sum()
    keep = (pl + pr) > 0
    pbar = (pl + pr)[keep] / (pl + pr)[keep].sum()
    exp_l = n_l_common * pbar
    exp_r = n_r_common * pbar
    g_obs = float(2 * np.sum(
        np.where(pl[keep] > 0, pl[keep] * np.log(pl[keep] / np.maximum(exp_l, 1e-12)), 0.0)
        + np.where(pr[keep] > 0, pr[keep] * np.log(pr[keep] / np.maximum(exp_r, 1e-12)), 0.0)))
    g_mc = np.empty(NNULL)
    tv_mc = np.empty(NNULL)
    for b in range(NNULL):
        sl_ = RNG.poisson(n_l_common * pbar)
        sr_ = RNG.poisson(n_r_common * pbar)
        e_l = sl_.sum() * pbar
        e_r = sr_.sum() * pbar
        g_mc[b] = 2 * np.sum(
            np.where(sl_ > 0, sl_ * np.log(sl_ / np.maximum(e_l, 1e-12)), 0.0)
            + np.where(sr_ > 0, sr_ * np.log(sr_ / np.maximum(e_r, 1e-12)), 0.0))
        tv_mc[b] = tv_distance(sl_, sr_)
    tv_obs = tv_distance(pl, pr)
    # full marginal test: EVERYTHING channel l holds, vs r's 52 events spread with
    # l's shape over l's whole support.  This is the only comparison in which the
    # factor-6.9 COUNT contrast itself is under test.
    efull = np.concatenate([[cl.min() - np.median(wl) / 2], cl + np.median(wl) / 2])
    pfull = overlap_rebin(cl, yl, wl, efull)
    pfull = pfull / pfull.sum()
    cfull = efull[:-1] + np.diff(efull) / 2
    exp_r_full = Nr * np.interp(cr, cfull, pfull)
    g_obs_full = float(2 * np.sum(
        np.where(yr > 0, yr * np.log(yr / np.maximum(exp_r_full, 1e-12)), 0.0)))
    g_mc_full = np.empty(NNULL)
    for b in range(NNULL):
        yr_full = RNG.poisson(Nr * pfull)
        expf = Nr * pfull
        g_mc_full[b] = 2 * np.sum(
            np.where(yr_full > 0, yr_full * np.log(yr_full / np.maximum(expf, 1e-12)), 0.0))
    log(f"    [group_2 fast] marginal G (r's 52 events vs l's shape over l's full "
        f"support, {len(cfull)} bins) = {g_obs_full:.2f}; null median = "
        f"{np.median(g_mc_full):.2f}, p95 = {np.percentile(g_mc_full, 95):.2f}; "
        f"Monte-Carlo p = {(g_mc_full >= g_obs_full).mean():.4f}")

    g2w = window_table(yl, cl, yr, cr, RNG, cell_label=f"{grp}|{reg}")
    g2w.to_csv(RES / "polc_s04b_group2_windows.csv", index=False)
    g2w.to_json(RES / "polc_s04b_group2_windows.json", orient="records", indent=2)
    log("    [group_2 fast] window-restricted count ratios and channel x window "
        "homogeneity (G, 5000-rep Poisson Monte Carlo):")
    log(g2w.head(8).to_string(index=False))

    # the same test restricted to the COMMON support (> channel_r's first bin):
    # the r profile simply has no bins below 7.28 MHz/s, so any window crossing
    # that edge compares l-events that have no r counterpart at all.
    in_l2 = cl >= lo_support
    in_r2 = cr >= lo_support
    g2c = window_table(yl[in_l2], cl[in_l2], yr[in_r2], cr[in_r2], RNG,
                       cell_label=f"{grp}|{reg}|common_support_only")
    g2c.to_csv(RES / "polc_s04c_group2_windows_common.csv", index=False)
    g2c.to_json(RES / "polc_s04c_group2_windows_common.json", orient="records", indent=2)
    log(f"\n    [group_2 fast] SAME test restricted to the common support "
        f"(s >= {lo_support:.3f} MHz/s; l keeps {int(yl[in_l2].sum())}, r keeps {int(yr[in_r2].sum())}):")
    log(g2c.head(8).to_string(index=False))

    g2.update(
        common_grid_n_bins=int(keep.sum()), G_stat=g_obs,
        G_null_median=float(np.median(g_mc)), G_null_p95=float(np.percentile(g_mc, 95)),
        p_G_monte_carlo=float((g_mc >= g_obs).mean()),
        TV_obs=tv_obs, TV_null_median=float(np.median(tv_mc)),
        TV_null_p95=float(np.percentile(tv_mc, 95)),
        p_TV_monte_carlo=float((tv_mc >= tv_obs).mean()),
        TV_detectable_p95=float(np.percentile(tv_mc, 95)),
        n_l_common=float(n_l_common), n_r_common=float(n_r_common),
        G_marginal_full_support=g_obs_full,
        G_marginal_null_median=float(np.median(g_mc_full)),
        G_marginal_null_p95=float(np.percentile(g_mc_full, 95)),
        p_G_marginal_monte_carlo=float((g_mc_full >= g_obs_full).mean()),
    )
    # what would the count ratio be if channel r had the same shape as l over l's support?
    frac_below = below / Nl
    g2["r_expected_below_if_same_shape"] = float(Nr * frac_below)
    g2["ratio_restricted_to_common_support"] = float(pl.sum() / pr.sum())
    g2_rows.append(g2)
    g2df = pd.DataFrame(g2_rows)
    g2df.to_csv(RES / "polc_s04_group2_fast_test.csv", index=False)
    g2df.to_json(RES / "polc_s04_group2_fast_test.json", orient="records", indent=2)
    log("\n=== 3. group_2 fast: 357 (l) vs 52 (r) ===")
    for k, v in g2.items():
        log(f"    {k} = {v}")

    # ---------------- 4. paired workbook blocks that differ ONLY by channel ----------
    pair_rows = []
    for (grp, reg) in CELL_SHEETS:
        shl, shr = CELL_SHEETS[(grp, reg)]
        for bi in range(max(len(blocks[shl]), len(blocks[shr]))):
            lab_l, xl, yl_ = blocks[shl][bi] if bi < len(blocks[shl]) else (None, None, None)
            lab_r, xr, yr_ = blocks[shr][bi] if bi < len(blocks[shr]) else (None, None, None)
            if lab_l is None or lab_r is None:
                continue
            same_grid = bool(len(xl) == len(xr) and np.allclose(xl, xr, atol=1e-12))
            dxl = float(np.median(np.diff(xl))); dxr = float(np.median(np.diff(xr)))
            # common-grid homogeneity test, 0.5 (fast) / 0.25 (slow) MHz/s
            step = 0.5 if reg == "fast" else 0.25
            edges = common_edges(xl, xr, dxl, dxr, step)
            pl = overlap_rebin(xl, yl_, np.full(len(xl), dxl), edges)
            pr = overlap_rebin(xr, yr_, np.full(len(xr), dxr), edges)
            common_lo, common_hi = float(edges[0]), float(edges[-1])
            keep = (pl + pr) > 0
            pl, pr = pl[keep], pr[keep]
            pbar = (pl + pr) / (pl + pr).sum()
            G = float(2 * np.sum(
                np.where(pl > 0, pl * np.log(pl / np.maximum(pl.sum() * pbar, 1e-12)), 0.0)
                + np.where(pr > 0, pr * np.log(pr / np.maximum(pr.sum() * pbar, 1e-12)), 0.0)))
            gmc = np.empty(2000)
            for b in range(2000):
                sl_ = RNG.poisson(pl.sum() * pbar); sr_ = RNG.poisson(pr.sum() * pbar)
                e1 = sl_.sum() * pbar; e2 = sr_.sum() * pbar
                gmc[b] = 2 * np.sum(
                    np.where(sl_ > 0, sl_ * np.log(sl_ / np.maximum(e1, 1e-12)), 0.0)
                    + np.where(sr_ > 0, sr_ * np.log(sr_ / np.maximum(e2, 1e-12)), 0.0))
            pair_rows.append(dict(
                group=grp, regime=reg, block_index=bi,
                sheet_l=shl, sheet_r=shr, label_l=lab_l, label_r=lab_r,
                n_l=len(xl), n_r=len(xr), events_l=float(yl_.sum()), events_r=float(yr_.sum()),
                ratio_l_over_r=float(yl_.sum() / yr_.sum()) if yr_.sum() else np.nan,
                identical_x_grid=same_grid,
                dx_l=dxl, dx_r=dxr,
                xmin_l=float(xl.min()), xmax_l=float(xl.max()),
                xmin_r=float(xr.min()), xmax_r=float(xr.max()),
                common_support_lo=common_lo, common_support_hi=common_hi,
                n_common_bins=int(len(pl)),
                G_stat=G, p_G_monte_carlo=float((gmc >= G).mean()),
                G_null_p95=float(np.percentile(gmc, 95)),
                TV=float(tv_distance(pl, pr)), TV_null_p95=float(np.percentile(
                    [tv_distance(RNG.poisson(pl.sum() * pbar), RNG.poisson(pr.sum() * pbar))
                     for _ in range(500)], 95)),
            ))
    # ---- window x channel contingency for EVERY cell (same fixed-ratio null) ----
    wall = []
    for (grp, reg) in CELL_SHEETS:
        cl, yl, wl = cells[(grp, reg, "channel_l")]
        cr, yr, wr = cells[(grp, reg, "channel_r")]
        wall.append(window_table(yl, cl, yr, cr, RNG, cell_label=f"{grp}|{reg}"))
    wall = pd.concat(wall, ignore_index=True)
    wall["group"] = wall["cell"].str.split("|").str[0]
    wall["regime"] = wall["cell"].str.split("|").str[1]
    wall.to_csv(RES / "polc_s08_window_channel.csv", index=False)
    wall.to_json(RES / "polc_s08_window_channel.json", orient="records", indent=2)
    log("\n=== 4c. channel x drift-window homogeneity for all six cells ===")
    log(wall[["group", "regime", "window", "events_l", "events_r", "ratio_l_over_r",
              "expected_l_under_constant_ratio", "binom_p_constant_ratio",
              "G_joint_window", "p_G_joint_window_monte_carlo"]].to_string(index=False))

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(RES / "polc_s05_block_pairs.csv", index=False)
    pairs.to_json(RES / "polc_s05_block_pairs.json", orient="records", indent=2)
    log("\n=== 4. paired workbook blocks differing only by channel ===")
    log(pairs.to_string(index=False))

    # ---------------- 5. coarse workbook histograms vs the fine blocks ----------
    coarse_rows = []
    for (grp, reg) in CELL_SHEETS:
        for ch in ("channel_l", "channel_r"):
            shc = COARSE[(grp, ch)]
            shf = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
            for bi, (lab_f, xf, yf) in enumerate(blocks[shf]):
                # match a coarse block by the FULL label: the coarse sheets carry
                # exactly the three labels of the fine sheet (verified on Sheet_039)
                cands = [(l, x, y) for (l, x, y) in blocks[shc] if l == lab_f]
                if not cands:
                    cands = [(l, x, y) for (l, x, y) in blocks[shc]
                             if l.replace(" ", "") == lab_f.replace(" ", "")]
                if not cands:
                    continue
                lab_c, xc, yc = cands[0]
                dxc = float(np.median(np.diff(xc)))
                dxf = float(np.median(np.diff(xf)))
                # rebin fine onto the coarse edges; use ONLY coarse bins FULLY inside the fine
                # support, so no coarse bin is compared against a partial expectation
                e = np.concatenate([[xc[0] - dxc / 2], xc + dxc / 2])
                pr = overlap_rebin(xf, yf, np.full(len(xf), dxf), e)
                flo = xf.min() - dxf / 2
                fhi = xf.max() + dxf / 2
                cl_edges = np.concatenate([[xc[0] - dxc / 2], xc + dxc / 2])
                fully_in = (cl_edges[:-1] >= flo - 1e-12) & (cl_edges[1:] <= fhi + 1e-12)
                obs_all, exp_all = yc[fully_in], pr[fully_in]
                # 2xK contingency G between the coarse and the re-binned fine
                # histogram, restricted to the coarse bins FULLY inside the fine
                # support, with its own Poisson Monte-Carlo null
                G = g_stat(obs_all, exp_all, exp_all / max(exp_all.sum(), 1e-12))
                gmc_c = np.empty(2000)
                pb = exp_all / max(exp_all.sum(), 1e-12)
                for b in range(2000):
                    gmc_c[b] = g_stat(RNG.poisson(obs_all.sum() * pb),
                                      RNG.poisson(exp_all.sum() * pb), pb)
                coarse_rows.append(dict(
                    group=grp, regime=reg, channel=ch,
                    coarse_sheet=shc, fine_sheet=shf, fine_block_index=bi,
                    label_coarse=lab_c, label_fine=lab_f,
                    n_coarse_bins=len(xc), n_fine_bins=len(xf),
                    events_coarse=float(yc.sum()), events_fine=float(yf.sum()),
                    total_diff=float(yc.sum() - yf.sum()),
                    total_rel_diff=float((yc.sum() - yf.sum()) / yf.sum()),
                    n_bins_fully_inside_fine_support=int(fully_in.sum()),
                    events_coarse_inside=float(obs_all.sum()),
                    events_fine_inside=float(exp_all.sum()),
                    G_coarse_vs_fine=G, G_null_p95=float(np.percentile(gmc_c, 95)),
                    p_G_monte_carlo=float((gmc_c >= G).mean()),
                    coarse_lo=float(xc.min()), coarse_hi=float(xc.max()),
                    fine_lo=float(xf.min()), fine_hi=float(xf.max()),
                    max_rel_bin_dev=float(np.max(np.abs(obs_all - exp_all) / np.maximum(exp_all, 1.0)))
                    if len(obs_all) else np.nan,
                ))
    coarse = pd.DataFrame(coarse_rows)
    coarse.to_csv(RES / "polc_s06_coarse_vs_fine.csv", index=False)
    coarse.to_json(RES / "polc_s06_coarse_vs_fine.json", orient="records", indent=2)
    log("\n=== 5. coarse (1.75 MHz/s) vs fine published histograms ===")
    log(coarse.to_string(index=False))

    # ---- alignment-free CDF comparison + edge-offset scan ----
    # If the coarse and fine histograms describe the same quantity, their CDFs
    # must agree whatever the bin registration; the offset scan asks whether some
    # shift of the coarse edges reconciles the two binned shapes.
    cdf_rows = []
    for (grp, reg) in CELL_SHEETS:
        for ch in ("channel_l", "channel_r"):
            shc = COARSE[(grp, ch)]
            shf = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
            for bi, (lab_f, xf, yf) in enumerate(blocks[shf]):
                cands = [(l, x, y) for (l, x, y) in blocks[shc] if l == lab_f]
                if not cands:
                    continue
                lab_c, xc, yc = cands[0]
                dxc = float(np.median(np.diff(xc)))
                dxf = float(np.median(np.diff(xf)))
                lo = max(xc.min() - dxc / 2, xf.min() - dxf / 2)
                hi = min(xc.max() + dxc / 2, xf.max() + dxf / 2)
                grid = np.linspace(lo, hi, 400)
                cf = np.interp(grid, xf, np.cumsum(yf) / yf.sum()) if len(xf) else np.full(len(grid), np.nan)
                cc = np.interp(grid, xc, np.cumsum(yc) / yc.sum()) if len(xc) else np.full(len(grid), np.nan)
                ks = float(np.max(np.abs(cf - cc)))
                # offset scan: shift the coarse edges by delta inside +/- one bin
                best = None
                for delta in np.linspace(-dxc / 2, dxc / 2, 25):
                    e = np.concatenate([[xc[0] - dxc / 2 + delta], xc + dxc / 2 + delta])
                    pr = overlap_rebin(xf, yf, np.full(len(xf), dxf), e)
                    fn = pr / max(pr.sum(), 1e-12)
                    cn = yc / max(yc.sum(), 1e-12)
                    d = float(np.abs(fn - cn).sum())  # L1, aligned bin-by-bin
                    if best is None or d < best[0]:
                        best = (d, float(delta))
                cdf_rows.append(dict(
                    group=grp, regime=reg, channel=ch, coarse_sheet=shc,
                    fine_sheet=shf, fine_block_index=bi, label=lab_f,
                    cdf_max_abs_diff_within_overlap=ks,
                    best_offset_L1=best[0], best_offset_delta=best[1],
                    best_offset_as_fraction_of_coarse_bin=best[1] / dxc,
                    L1_at_zero_offset=float(np.abs(
                        overlap_rebin(xf, yf, np.full(len(xf), dxf),
                                      np.concatenate([[xc[0] - dxc / 2], xc + dxc / 2])
                                      ) / max(yf.sum(), 1e-12) - yc / max(yc.sum(), 1e-12)).sum()),
                ))
    cdf = pd.DataFrame(cdf_rows)
    cdf.to_csv(RES / "polc_s09_coarse_cdf.csv", index=False)
    cdf.to_json(RES / "polc_s09_coarse_cdf.json", orient="records", indent=2)
    log("\n=== 5b. alignment-free CDF distance and coarse-edge offset scan ===")
    log(cdf.to_string(index=False))

    # ---- wide translation scan: can ANY origin convention reconcile coarse & fine? ----
    scan_rows = []
    for (grp, reg) in CELL_SHEETS:
        for ch in ("channel_l", "channel_r"):
            shc = COARSE[(grp, ch)]
            shf = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
            for bi, (lab_f, xf, yf) in enumerate(blocks[shf]):
                cands = [(l, x, y) for (l, x, y) in blocks[shc] if l == lab_f]
                if not cands:
                    continue
                lab_c, xc, yc = cands[0]
                dxc = float(np.median(np.diff(xc)))
                dxf = float(np.median(np.diff(xf)))
                cn = yc / max(yc.sum(), 1e-12)
                best = None
                curve = []
                for delta in np.arange(-4.0, 4.001, 0.05):
                    e = np.concatenate([[xc[0] - dxc / 2 + delta], xc + dxc / 2 + delta])
                    pr = overlap_rebin(xf, yf, np.full(len(xf), dxf), e)
                    fn = pr / max(pr.sum(), 1e-12)
                    d = float(np.abs(fn - cn).sum())
                    # CDF distance on the common support of the shifted grid
                    grid = np.linspace(max(e[0], xf.min() - dxf / 2),
                                       min(e[-1], xf.max() + dxf / 2), 300)
                    if grid[-1] > grid[0]:
                        cf = np.interp(grid, xf, np.cumsum(yf) / yf.sum())
                        cc = np.interp(grid, e[:-1] + np.diff(e) / 2,
                                       np.cumsum(pr) / max(pr.sum(), 1e-12))
                        ks = float(np.max(np.abs(cf - cc)))
                    else:
                        ks = np.nan
                    curve.append((delta, d, ks))
                    if best is None or d < best[1]:
                        best = (delta, d, ks)
                scan_rows.append(dict(
                    group=grp, regime=reg, channel=ch, coarse_sheet=shc,
                    fine_sheet=shf, label=lab_f,
                    best_translation_MHz_s=float(best[0]), min_L1=float(best[1]),
                    KS_at_best=float(best[2]),
                    L1_at_zero_offset=float([c[1] for c in curve if abs(c[0]) < 1e-9][0]),
                    n_offsets_tested=len(curve),
                    L1_below_0p10_offsets=int(sum(1 for c in curve if c[1] < 0.10)),
                ))
    scan = pd.DataFrame(scan_rows)
    scan.to_csv(RES / "polc_s11_translation_scan.csv", index=False)
    scan.to_json(RES / "polc_s11_translation_scan.json", orient="records", indent=2)
    log("\n=== 5d. wide translation scan: is the coarse/fine gap a pure origin shift? ===")
    log(scan.to_string(index=False))
    log(f"    offsets tested: {scan.n_offsets_tested.iloc[0]} per pair, range -4..+4 MHz/s at 0.05; "
        f"pairs where ANY offset brings L1 below 0.10: "
        f"{int((scan.L1_below_0p10_offsets > 0).sum())}/{len(scan)}")

    # ---- first-bin pile-up diagnostic on the published fine grids ----
    pile = []
    for (grp, reg) in CELL_SHEETS:
        for ch in ("channel_l", "channel_r"):
            sh = CELL_SHEETS[(grp, reg)][0 if ch == "channel_l" else 1]
            for bi, (lab, x, y) in enumerate(blocks[sh]):
                if y.sum() <= 0:
                    continue
                pile.append(dict(group=grp, regime=reg, channel=ch, block_index=bi,
                                 label=lab, n_events=float(y.sum()),
                                 first_bin_center=float(x.min()),
                                 first_bin_count=float(y[np.argmin(x)]),
                                 first_bin_fraction=float(y[np.argmin(x)] / y.sum()),
                                 max_bin_center=float(x[np.argmax(y)]),
                                 max_bin_fraction=float(y.max() / y.sum())))
    pile = pd.DataFrame(pile)
    pile.to_csv(RES / "polc_s10_bin_pileup.csv", index=False)
    pile.to_json(RES / "polc_s10_bin_pileup.json", orient="records", indent=2)
    log("\n=== 5c. edge pile-up diagnostic (fraction of a source block in one bin) ===")
    log(pile.to_string(index=False))

    # ---------------- 6. channel count-ratio summary across groups ----------------
    rr = []
    for (grp, reg) in CELL_SHEETS:
        Nl = float(cells[(grp, reg, "channel_l")][1].sum())
        Nr = float(cells[(grp, reg, "channel_r")][1].sum())
        rr.append(dict(group=grp, regime=reg, events_l=Nl, events_r=Nr,
                       ratio_l_over_r=Nl / Nr,
                       ratio_rel_se=float(np.sqrt(1 / Nl + 1 / Nr)),
                       ln_ratio=float(np.log(Nl / Nr)),
                       ln_ratio_se=float(np.sqrt(1 / Nl + 1 / Nr))))
    rat = pd.DataFrame(rr)
    rat.to_csv(RES / "polc_s07_count_ratios.csv", index=False)
    rat.to_json(RES / "polc_s07_count_ratios.json", orient="records", indent=2)
    log("\n=== 6. channel count ratios (l/r) per group and regime ===")
    log(rat.to_string(index=False))
    log(f"    ratio spread across groups (fast): "
        f"{rat[rat.regime=='fast'].ratio_l_over_r.min():.3f} .. {rat[rat.regime=='fast'].ratio_l_over_r.max():.3f}")

    # ---------------- 7. figures ----------------
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), dpi=130)
    for i, grp in enumerate(sorted({g for g, _ in CELL_SHEETS})):
        for j, reg in enumerate(["fast", "slow"]):
            ax = axes[i, j]
            for ch, col in (("channel_l", "#1f77b4"), ("channel_r", "#d62728")):
                c, y, w = cells[(grp, reg, ch)]
                step = 0.5 if reg == "fast" else 0.25
                ce = common_edges(c, c, w, w, step)
                p = overlap_rebin(c, y, w, ce)
                f = p / p.sum()
                ax.step(ce[:-1], f, where="post", color=col, lw=1.6,
                        label=f"{ch} (N={int(y.sum())})")
            ax.set_xlabel("published drift-rate bin coordinate (MHz/s)")
            ax.set_ylabel("fraction of events per bin")
            ax.set_title(f"{grp} / {reg}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("POLC fig1: paired polarization-channel drift-rate profiles "
                 "(same source, two channels; original bins re-binned onto a common grid for display only)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "polc_fig1_channel_profiles.png")
    plt.close(fig)

    # bootstrap interval panel
    order = [(g, r) for g in sorted({g for g, _ in CELL_SHEETS}) for r in ["fast", "slow"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), dpi=130)
    ax = axes[0]
    ys = np.arange(len(order))
    obs = [float(cont[(cont.group == g) & (cont.regime == r)].median_diff.iloc[0]) for g, r in order]
    lo = [float(cont[(cont.group == g) & (cont.regime == r)].median_diff_ci_lo.iloc[0]) for g, r in order]
    hi = [float(cont[(cont.group == g) & (cont.regime == r)].median_diff_ci_hi.iloc[0]) for g, r in order]
    nlo = [float(cont[(cont.group == g) & (cont.regime == r)].null_median_diff_p2_5.iloc[0]) for g, r in order]
    nhi = [float(cont[(cont.group == g) & (cont.regime == r)].null_median_diff_p97_5.iloc[0]) for g, r in order]
    ax.errorbar(obs, ys - 0.12, xerr=[np.array(obs) - np.array(lo), np.array(hi) - np.array(obs)],
                fmt="o", color="#1f77b4", capsize=4, label="observed diff, 95% Poisson-noise CI")
    for k, (g, r) in enumerate(order):
        ax.plot([nlo[k], nhi[k]], [k + 0.12] * 2, color="0.45", lw=3, solid_capstyle="butt",
                label="95% band under same-shape null" if k == 0 else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g.replace('anonymous_group_','g')}\n{r}" for g, r in order], fontsize=8)
    ax.set_xlabel("weighted median difference  channel_l - channel_r  (MHz/s)")
    ax.set_title("POLC fig2a: median shift vs counting noise and vs the\nsame-shape null (paired source, two channels)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    tv = [float(cont[(cont.group == g) & (cont.regime == r)].tv_observed.iloc[0]) for g, r in order]
    tvlo = [float(cont[(cont.group == g) & (cont.regime == r)].tv_noise_ci_lo.iloc[0]) for g, r in order]
    tvhi = [float(cont[(cont.group == g) & (cont.regime == r)].tv_noise_ci_hi.iloc[0]) for g, r in order]
    tvn = [float(cont[(cont.group == g) & (cont.regime == r)].tv_null_p95.iloc[0]) for g, r in order]
    ax.errorbar(tv, ys, xerr=[np.array(tv) - np.array(tvlo), np.array(tvhi) - np.array(tv)],
                fmt="o", color="#2ca02c", capsize=4, label="observed TV, 95% noise CI")
    for k in range(len(ys)):
        ax.plot([tvn[k]], [k], marker="x", color="k", ms=7,
                label="null p95 (same shape)" if k == 0 else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g.replace('anonymous_group_','g')}/{r}" for g, r in order], fontsize=8)
    ax.set_xlabel("total-variation distance between channel profiles")
    ax.set_title("POLC fig2b: shape distance vs same-shape null")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "polc_fig2_bootstrap_intervals.png")
    plt.close(fig)

    # group_2 fast detail
    grp, reg = "anonymous_group_2", "fast"
    cl, yl, wl = cells[(grp, reg, "channel_l")]
    cr, yr, wr = cells[(grp, reg, "channel_r")]
    edges = common_edges(cl, cr, wl, wr, 0.5)
    pl = overlap_rebin(cl, yl, wl, edges)
    pr = overlap_rebin(cr, yr, wr, edges)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=130)
    ctr = edges[:-1] + np.diff(edges) / 2
    wdt = np.diff(edges) * 0.4
    ax.bar(ctr - wdt / 2, pl / pl.sum(), width=wdt, color="#1f77b4", label=f"channel_l (N={int(yl.sum())})")
    ax.bar(ctr + wdt / 2, pr / pr.sum(), width=wdt, color="#d62728", label=f"channel_r (N={int(yr.sum())})")
    ax.axvline(cr.min() - np.median(wr) / 2, color="0.4", ls="--", lw=1.2,
               label=f"channel_r lowest bin ({cr.min():.2f} MHz/s)")
    ax.set_xlabel("published drift-rate bin coordinate (MHz/s)")
    ax.set_ylabel("fraction of events per 0.5 MHz/s bin")
    ax.set_title(f"POLC fig3: group_2 fast, 357 vs 52 events; G={g2['G_stat']:.1f}, "
                 f"Monte-Carlo p={g2['p_G_monte_carlo']:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "polc_fig3_group2_fast.png")
    plt.close(fig)

    log("\nwrote polc_s00..s07 tables and 3 figures")
    fh.close()


if __name__ == "__main__":
    main()