#!/usr/bin/env python3
"""chpol_s02 - Polarization-channel contrast, decomposed into the real source blocks.

WHY THIS SCRIPT EXISTS
chpol_s01 showed that the `drift_regime` split in drift_population_series.tsv is
not a regime decomposition of one population:

  fast == the '(-35<s<-10)' block of one workbook sheet (exact, 6/6)
  slow == the RAW CONCATENATION of the '(-10<s<0)' and '(0<s<20)' blocks of the
          same sheet (exact, 6/6 including order and value)

The two slow blocks are DISJOINT in the orientation coordinate s but OVERLAP
heavily in drift rate (both start near 0.15-0.26 MHz/s and the first runs to
1.95-2.99 while the second runs to 10.3-11.7).  Concatenating them therefore
mixes two different physical sub-populations on one drift axis, and the raw
concatenation is not even monotonically ordered.  Any shape statistic computed
on 'slow' as a single distribution is a mixture artefact.

So all channel inference here is done per (group, s-range) BLOCK, which is the
finest unit the source workbook actually supports.  The three s-ranges are
disjoint event sets, so a block is a genuine sub-population; the concatenated
'slow' number is reported too, clearly flagged as a mixture.

ANALYSIS UNIT: one complete source-polarization series (6), or equivalently one
complete source block x channel.  Bins are nested repeats, not samples.

PAIRED DESIGN: the l and r blocks of a group cover the SAME events observed in
two polarization channels.  Per block the paired total T = n_l + n_r is
therefore the natural conditioning quantity, and under a channel-independent
null the split is Binomial(T, p) with a single p.  That gives an exact test of
(a) whether p differs from 1/2 and (b) whether p is constant across s-ranges.
Poisson resampling is used for unconditional totals and ratios.

Outputs: output/results/chpol2_*, log at output/logs/chpol_s02.log
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
RNG = np.random.default_rng(SEED)

# group -> channel -> sheet holding the three s-range blocks, in workbook order
SHEETS = {
    ("anonymous_group_1", "channel_l"): "Sheet_009",
    ("anonymous_group_1", "channel_r"): "Sheet_010",
    ("anonymous_group_2", "channel_l"): "Sheet_011",
    ("anonymous_group_2", "channel_r"): "Sheet_012",
    ("anonymous_group_3", "channel_l"): "Sheet_013",
    ("anonymous_group_3", "channel_r"): "Sheet_014",
}
SHEETS_C = {  # 'group_b C' variant sheets, workbook declares BOTH as channel_l
    "C_variant_A": "Sheet_046",
    "C_variant_B": "Sheet_047",
}
S_RANGE = ["(-35,-10)", "(-10,0)", "(0,20)"]
FAST_SHEETS = {  # group -> channel -> the fast-regime sheet (s < -10), for the regime view
    ("anonymous_group_1", "channel_l"): "Sheet_025",
    ("anonymous_group_1", "channel_r"): "Sheet_027",
    ("anonymous_group_2", "channel_l"): "Sheet_029",
    ("anonymous_group_2", "channel_r"): "Sheet_031",
    ("anonymous_group_3", "channel_l"): "Sheet_033",
    ("anonymous_group_3", "channel_r"): "Sheet_035",
}


# ==========================================================================
# log-normal fit (Newton, validated against Nelder-Mead in chpol_s01)
# ==========================================================================
def lognormal_mle(x, n):
    """(mu, sigma) weighted MLE on log(bin centre); plain scalar version."""
    x, n = np.asarray(x, float), np.asarray(n, float)
    m = n > 0
    if m.sum() < 3:
        return np.nan, np.nan
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


def wq(x, n, q):
    x, n = np.asarray(x, float), np.asarray(n, float)
    m = n > 0
    if not m.any():
        return np.nan
    x, n = x[m], n[m]
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(n[o]) / n.sum(), x[o]))


def shape_stats(x, n, thresh=10.0):
    x, n = np.asarray(x, float), np.asarray(n, float)
    N = float(n.sum())
    m = n > 0
    mu, s = lognormal_mle(x, n)
    lx, ln = np.log(x[m]), n[m]
    mu_m = float((ln * lx).sum() / ln.sum()) if m.any() else np.nan
    sig_m = (float(np.sqrt(max((ln * (lx - mu_m) ** 2).sum() / ln.sum(), 1e-12)))
             if m.any() else np.nan)
    q25, q75 = wq(x, n, 0.25), wq(x, n, 0.75)
    return {
        "events": N, "n_bins": int(len(x)), "n_empty_bins": int((n == 0).sum()),
        "n_nonempty_bins": int(m.sum()),
        "weighted_mean": float((x * n).sum() / N) if N > 0 else np.nan,
        "weighted_median": wq(x, n, 0.5), "weighted_iqr": q75 - q25,
        "mode_bin_center": float(x[int(np.argmax(n))]) if N > 0 else np.nan,
        "peak_count": float(n.max()),
        "frac_gt_10": float(n[x > thresh].sum() / N) if N > 0 else np.nan,
        "mu_ln": mu, "sigma_ln": s,
        "median_ln": float(np.exp(mu)) if np.isfinite(mu) else np.nan,
        "mode_ln": float(np.exp(mu - s ** 2)) if np.isfinite(mu) else np.nan,
        "sigma_moment": sig_m,
        # coefficient of variation of the non-zero part, a shape-only summary
        "cv_nonzero": (float(np.sqrt((ln * (x[m] - (x[m] * ln).sum() / ln.sum()) ** 2).sum() / ln.sum())
                            / ((x[m] * ln).sum() / ln.sum())) if m.any() else np.nan),
    }


def grid_edges(x, h=0.02):
    x = np.asarray(x, float)
    o = np.argsort(x)
    xs = x[o]
    mid = (xs[:-1] + xs[1:]) / 2.0
    return np.concatenate([[xs[0] - h], mid, [xs[-1] + h]])


# ==========================================================================
def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    cache = {k: z[k] for k in z.keys()}

    def blocks_of(sheet):
        ks = [k for k in cache if k.split("__")[0] == sheet]
        return [(np.asarray(cache[k][0], float), np.asarray(cache[k][1], float)) for k in ks]

    # ------------------------------------------------------------------
    # 0. hard equality verification of the whole block layout
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    ori = pd.read_csv(DATA / "orientation_count_series.tsv", sep="\t")
    oser = {("anonymous_group_1", "channel_l"): "orientation_series_1",
            ("anonymous_group_1", "channel_r"): "orientation_series_2",
            ("anonymous_group_2", "channel_l"): "orientation_series_3",
            ("anonymous_group_2", "channel_r"): "orientation_series_4",
            ("anonymous_group_3", "channel_l"): "orientation_series_5",
            ("anonymous_group_3", "channel_r"): "orientation_series_6"}

    ver = []
    for (grp, ch), sheet in SHEETS.items():
        bl = blocks_of(sheet)
        # orientation view == the three blocks with the first bin of each dropped
        ox = np.concatenate([b[0][1:] for b in bl])
        oy = np.concatenate([b[1][1:] for b in bl])
        v = ori[ori.orientation_series_id == oser[(grp, ch)]].sort_values("relative_coordinate")
        # drift view: fast == block0 of the fast sheet; slow == concat(block1, block2)
        fb = blocks_of(FAST_SHEETS[(grp, ch)])
        fview = df[(df.anonymous_source_group == grp) & (df.polarization_channel == ch)
                   & (df.drift_regime == "fast")].sort_values("distribution_bin_center")
        sview = df[(df.anonymous_source_group == grp) & (df.polarization_channel == ch)
                   & (df.drift_regime == "slow")].sort_values("distribution_bin_center")
        slow_cat_x = np.concatenate([bl[1][0], bl[2][0]])
        slow_cat_y = np.concatenate([bl[1][1], bl[2][1]])
        ver.append({
            "group": grp, "channel": ch, "sheet": sheet, "fast_sheet": FAST_SHEETS[(grp, ch)],
            "orient_x_equal": bool(np.allclose(ox, v.relative_coordinate.values)),
            "orient_y_equal": bool(np.allclose(oy, v.event_count.values)),
            "orient_dropped_events": float(np.concatenate([b[1] for b in bl]).sum()
                                           - np.concatenate([b[1][1:] for b in bl]).sum()),
            "fast_x_equal": bool(np.allclose(np.sort(fview.distribution_bin_center.values),
                                             np.sort(fb[0][0]))),
            "fast_counts_equal": bool(np.allclose(fview.event_count.values, fb[0][1])),
            "fast_events": float(fb[0][1].sum()),
            "slow_is_ordered_concat": bool(np.allclose(
                sview.distribution_bin_center.values, slow_cat_x)),
            "slow_counts_equal": bool(np.allclose(sview.event_count.values, slow_cat_y)),
            "slow_events": float(slow_cat_y.sum()),
            "blockA_events": float(bl[1][1].sum()), "blockB_events": float(bl[2][1].sum()),
            "blockA_drift_min": float(bl[1][0].min()), "blockA_drift_max": float(bl[1][0].max()),
            "blockB_drift_min": float(bl[2][0].min()), "blockB_drift_max": float(bl[2][0].max()),
        })
    ver_df = pd.DataFrame(ver)
    print("=== chpol2.0 layout verification (all must be True) ===")
    print(ver_df[["group", "channel", "sheet", "orient_x_equal", "orient_y_equal",
                  "fast_counts_equal", "slow_is_ordered_concat", "slow_counts_equal"]].to_string(index=False))
    ok = bool(ver_df[["orient_x_equal", "orient_y_equal", "fast_counts_equal",
                      "slow_is_ordered_concat", "slow_counts_equal"]].all().all())
    print(f"ALL layout equalities hold: {ok}")
    ver_df.to_csv(OUT / "chpol2_layout_verification.csv", index=False)

    print("\n=== chpol2.0b drift overlap of the two 'slow' blocks ===")
    print(ver_df[["group", "channel", "blockA_events", "blockB_events",
                  "blockA_drift_min", "blockA_drift_max",
                  "blockB_drift_min", "blockB_drift_max"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 1. per-block paired channel table
    # ------------------------------------------------------------------
    rows = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for ch in ["channel_l", "channel_r"]:
            bl = blocks_of(SHEETS[(grp, ch)])
            for lab, (x, y) in zip(S_RANGE, bl):
                st = shape_stats(x, y)
                st.update({"group": grp, "channel": ch, "s_range": lab,
                           "sheet": SHEETS[(grp, ch)],
                           "drift_min": float(x.min()), "drift_max": float(x.max())})
                rows.append(st)
        fb = blocks_of(FAST_SHEETS[(grp, "channel_l")])[0]
        fb_r = blocks_of(FAST_SHEETS[(grp, "channel_r")])[0]
    blk = pd.DataFrame(rows)

    # paired pivot
    pair = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for lab in S_RANGE:
            a = blk[(blk.group == grp) & (blk.s_range == lab) & (blk.channel == "channel_l")].iloc[0]
            b = blk[(blk.group == grp) & (blk.s_range == lab) & (blk.channel == "channel_r")].iloc[0]
            pair.append({
                "group": grp, "s_range": lab,
                "n_l": a.events, "n_r": b.events, "pair_total": a.events + b.events,
                "channel_r_fraction": b.events / (a.events + b.events),
                "ratio_r_over_l": b.events / a.events if a.events > 0 else np.nan,
                "median_l": a.weighted_median, "median_r": b.weighted_median,
                "median_diff_r_minus_l": b.weighted_median - a.weighted_median,
                "sigma_l": a.sigma_ln, "sigma_r": b.sigma_ln,
                "sigma_ratio_r_over_l": b.sigma_ln / a.sigma_ln if a.sigma_ln else np.nan,
                "frac_gt10_l": a.frac_gt_10, "frac_gt10_r": b.frac_gt_10,
                "drift_min": min(a.drift_min, b.drift_min),
                "drift_max": max(a.drift_max, b.drift_max),
                "insufficient": bool(min(a.events, b.events) < 100),
            })
    pair = pd.DataFrame(pair)
    # exact binomial test of the channel split per (group, s-range)
    pair["binom_p_vs_half"] = [
        stats.binomtest(int(r.n_r), int(r.pair_total), 0.5).pvalue if r.pair_total > 0 else np.nan
        for r in pair.itertuples()]
    pair["binom_p_bonferroni9"] = np.minimum(pair.binom_p_vs_half * 9, 1.0)
    blk.to_csv(OUT / "chpol2_block_stats.csv", index=False)
    pair.to_csv(OUT / "chpol2_block_paired.csv", index=False)

    print("\n=== chpol2.1 per-block shape statistics (unit = one source block x channel) ===")
    print(blk[["group", "channel", "s_range", "events", "n_nonempty_bins",
               "weighted_median", "weighted_iqr", "sigma_ln", "frac_gt_10",
               "drift_min", "drift_max"]].to_string(index=False))

    print("\n=== chpol2.2 paired channel split per block (binomial vs 1/2) ===")
    print(pair[["group", "s_range", "n_l", "n_r", "ratio_r_over_l",
                "channel_r_fraction", "binom_p_vs_half", "binom_p_bonferroni9",
                "median_l", "median_r", "insufficient"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 2. is the channel fraction CONSTANT across s-ranges? (partially exact)
    # ------------------------------------------------------------------
    het = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        sub = pair[pair.group == grp]
        r = sub.n_r.values.astype(int)
        T = sub.pair_total.values.astype(int)
        p_hat = r.sum() / T.sum()
        # conditional binomial likelihood: p constant vs p free
        ll0 = np.sum(stats.binom.logpmf(r, T, p_hat))
        ll1 = np.sum(stats.binom.logpmf(r, T, np.clip(r / T, 1e-12, 1 - 1e-12)))
        lr = 2 * (ll1 - ll0)
        # parametric bootstrap of the LR statistic under 'p constant'
        sim = RNG.binomial(np.tile(T, (NBOOT, 1)), p_hat)
        ll0s = np.sum(stats.binom.logpmf(sim, np.tile(T, (NBOOT, 1)), p_hat), axis=1)
        ll1s = np.sum(stats.binom.logpmf(sim, np.tile(T, (NBOOT, 1)),
                                         np.clip(sim / np.tile(T, (NBOOT, 1)), 1e-12, 1 - 1e-12)), axis=1)
        lr_null = 2 * (ll1s - ll0s)
        het.append({
            "group": grp, "n_blocks": len(sub),
            "p_common_mle": p_hat,
            "p_block_fast": sub.channel_r_fraction.iloc[0],
            "p_block_mid": sub.channel_r_fraction.iloc[1],
            "p_block_pos": sub.channel_r_fraction.iloc[2],
            "spread_max_minus_min": sub.channel_r_fraction.max() - sub.channel_r_fraction.min(),
            "LR_stat": float(lr), "df": 2,
            "LR_p_chi2": float(1 - stats.chi2.cdf(lr, 2)),
            "LR_p_bootstrap": float((lr_null >= lr).mean()),
        })
    het = pd.DataFrame(het)
    het.to_csv(OUT / "chpol2_channel_fraction_heterogeneity.csv", index=False)
    print("\n=== chpol2.3 is the channel fraction constant across s-ranges? ===")
    print(het.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. unconditional Poisson bootstrap: totals and ratios
    # ------------------------------------------------------------------
    pr = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for lab in S_RANGE:
            a = blk[(blk.group == grp) & (blk.s_range == lab) & (blk.channel == "channel_l")].iloc[0]
            b = blk[(blk.group == grp) & (blk.s_range == lab) & (blk.channel == "channel_r")].iloc[0]
            xa = blocks_of(SHEETS[(grp, "channel_l")])[S_RANGE.index(lab)][1]
            xb = blocks_of(SHEETS[(grp, "channel_r")])[S_RANGE.index(lab)][1]
            A = np.maximum(RNG.poisson(np.tile(xa, (NBOOT, 1))).sum(axis=1), 1)
            B = np.maximum(RNG.poisson(np.tile(xb, (NBOOT, 1))).sum(axis=1), 1)
            r = B / A
            lo, hi = np.percentile(r, [2.5, 97.5])
            pr.append({"group": grp, "s_range": lab, "n_l": a.events, "n_r": b.events,
                       "ratio_r_over_l": b.events / a.events,
                       "ratio_ci_lo": float(lo), "ratio_ci_hi": float(hi),
                       "excludes_1": bool(lo > 1 or hi < 1),
                       "insufficient": bool(min(a.events, b.events) < 100)})
    pr = pd.DataFrame(pr)
    pr.to_csv(OUT / "chpol2_poisson_ratio.csv", index=False)
    print("\n=== chpol2.4 Poisson-bootstrap channel ratios per block ===")
    print(pr.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. multinomial shape bootstrap per block (median, sigma, frac>10)
    # ------------------------------------------------------------------
    bs_rows = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for ch in ["channel_l", "channel_r"]:
            bl = blocks_of(SHEETS[(grp, ch)])
            for lab, (x, y) in zip(S_RANGE, bl):
                N, tot = int(y.sum()), y.sum()
                if N < 2:
                    continue
                draws = RNG.multinomial(N, y / tot, size=NBOOT).astype(float)
                o = np.argsort(x)
                xs, D = x[o], draws[:, o]
                cum = np.cumsum(D, axis=1)
                T = np.where(D.sum(axis=1) <= 0, 1.0, D.sum(axis=1))
                med = xs[np.clip((cum < (0.5 * T)[:, None]).sum(axis=1), 0, len(xs) - 1)]
                q25 = xs[np.clip((cum < (0.25 * T)[:, None]).sum(axis=1), 0, len(xs) - 1)]
                q75 = xs[np.clip((cum < (0.75 * T)[:, None]).sum(axis=1), 0, len(xs) - 1)]
                f10 = D[:, xs > 10].sum(axis=1) / T
                for name, v in [("weighted_median", med), ("weighted_iqr", q75 - q25),
                                ("frac_gt_10", f10), ("mode_bin_center", xs[np.argmax(D, axis=1)])]:
                    lo, hi = np.percentile(v, [2.5, 97.5])
                    bs_rows.append({"group": grp, "channel": ch, "s_range": lab,
                                    "stat": name, "point": float(np.median(v)),
                                    "ci_lo": float(lo), "ci_hi": float(hi),
                                    "boot": "multinomial"})
    bs = pd.DataFrame(bs_rows)
    bs.to_csv(OUT / "chpol2_shape_bootstrap.csv", index=False)
    print("\n=== chpol2.5 multinomial shape bootstrap (95% CI) per block ===")
    for name in ["weighted_median", "frac_gt_10"]:
        t = bs[bs.stat == name].pivot_table(index=["group", "s_range"], columns="channel",
                                            values=["point", "ci_lo", "ci_hi"])
        print(f"\n-- {name} --")
        print(t.to_string())

    # do the paired channel CIs overlap?  explicit statement
    ov = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for lab in S_RANGE:
            for name in ["weighted_median", "weighted_iqr", "frac_gt_10"]:
                a = bs[(bs.group == grp) & (bs.s_range == lab) & (bs.channel == "channel_l")
                       & (bs.stat == name)]
                b = bs[(bs.group == grp) & (bs.s_range == lab) & (bs.channel == "channel_r")
                       & (bs.stat == name)]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                ov.append({"group": grp, "s_range": lab, "stat": name,
                           "l_point": a.point, "l_lo": a.ci_lo, "l_hi": a.ci_hi,
                           "r_point": b.point, "r_lo": b.ci_lo, "r_hi": b.ci_hi,
                           "ci_overlap": bool(min(a.ci_hi, b.ci_hi) >= max(a.ci_lo, b.ci_lo)),
                           "n_events_min": int(min(blk[(blk.group == grp) & (blk.s_range == lab)
                                                       & (blk.channel == 'channel_l')].events.iloc[0],
                                                   blk[(blk.group == grp) & (blk.s_range == lab)
                                                       & (blk.channel == 'channel_r')].events.iloc[0]))})
    ov = pd.DataFrame(ov)
    ov.to_csv(OUT / "chpol2_shape_ci_overlap.csv", index=False)
    print("\n=== chpol2.6 multinomial shape CIs: do channels overlap? ===")
    print(ov.groupby(["stat", "ci_overlap"]).size().to_string())

    # ------------------------------------------------------------------
    # 5. common-grid TV contrast on the disjoint s-range blocks
    # ------------------------------------------------------------------
    grid_rows, gsave = [], {}
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for lab in S_RANGE:
            bl = blocks_of(SHEETS[(grp, "channel_l")])[S_RANGE.index(lab)]
            br = blocks_of(SHEETS[(grp, "channel_r")])[S_RANGE.index(lab)]
            lo = max(bl[0].min(), br[0].min())
            hi = min(bl[0].max(), br[0].max())
            if not (hi > lo):
                continue
            g = np.linspace(lo + 1e-9, hi - 1e-9, 40)
            e = grid_edges(g, h=(g[1] - g[0]) / 2)
            hl, _ = np.histogram(bl[0], bins=e, weights=bl[1])
            hr, _ = np.histogram(br[0], bins=e, weights=br[1])
            hl, hr = hl.astype(float), hr.astype(float)
            Nl, Nr = hl.sum(), hr.sum()
            if Nl < 1 or Nr < 1:
                continue
            tv = 0.5 * np.abs(hl / Nl - hr / Nr).sum()
            pooled = hl + hr
            p = pooled / pooled.sum()
            null = np.empty(NBOOT)
            for i in range(NBOOT):
                aa = RNG.multinomial(int(Nl), p)
                bb = RNG.multinomial(int(Nr), p)
                null[i] = 0.5 * np.abs(aa / Nl - bb / Nr).sum()
            grid_rows.append({
                "group": grp, "s_range": lab, "N_l": float(Nl), "N_r": float(Nr),
                "drift_lo": float(lo), "drift_hi": float(hi),
                "tv": float(tv), "null_median": float(np.median(null)),
                "null_p95": float(np.percentile(null, 95)),
                "tv_over_null": float(tv / np.median(null)),
                "p_perm": float((null >= tv).mean()),
                "insufficient": bool(min(Nl, Nr) < 100),
            })
            gsave[f"{grp}|{lab}|l"] = hl
            gsave[f"{grp}|{lab}|r"] = hr
            gsave[f"{grp}|{lab}|grid"] = g
    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(OUT / "chpol2_grid_tv.csv", index=False)
    np.savez_compressed(OUT / "chpol2_grid_hist.npz", **gsave)
    print("\n=== chpol2.7 common-grid TV distance vs permutation null ===")
    print(gdf.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. C-variant sheets 046/047 versus the group_b blocks
    # ------------------------------------------------------------------
    cv = []
    bl_b_l = blocks_of(SHEETS[("anonymous_group_2", "channel_l")])
    bl_b_r = blocks_of(SHEETS[("anonymous_group_2", "channel_r")])
    for key, sheet in SHEETS_C.items():
        bl = blocks_of(sheet)
        for i, lab in enumerate(S_RANGE):
            cv.append({
                "variant": key, "sheet": sheet, "s_range": lab,
                "workbook_label_channel": "channel_l",
                "n": float(bl[i][1].sum()),
                "group_b_channel_l": float(bl_b_l[i][1].sum()),
                "group_b_channel_r": float(bl_b_r[i][1].sum()),
                "diff_vs_l": float(bl[i][1].sum() - bl_b_l[i][1].sum()),
                "diff_vs_r": float(bl[i][1].sum() - bl_b_r[i][1].sum()),
                "ratio_vs_l": float(bl[i][1].sum() / bl_b_l[i][1].sum()),
                "x_identical_to_l": bool(np.allclose(bl[i][0], bl_b_l[i][0])),
                "x_identical_to_r": bool(np.allclose(bl[i][0], bl_b_r[i][0])),
                "n_bins": len(bl[i][0]),
            })
    cv = pd.DataFrame(cv)
    cv.to_csv(OUT / "chpol2_C_variant.csv", index=False)
    print("\n=== chpol2.8 'group_b C' variant sheets vs group_b channel_l / channel_r ===")
    print(cv.to_string(index=False))

    # which non-C channel does each C sheet mimic?  compare r/l ratio per block
    # treat C_variant_A as candidate channel_l and C_variant_B as candidate channel_r
    print("\n--- C-variant as an independent replication of the group_b channel pair ---")
    rep = []
    for i, lab in enumerate(S_RANGE):
        nA = float(blocks_of(SHEETS_C["C_variant_A"])[i][1].sum())
        nB = float(blocks_of(SHEETS_C["C_variant_B"])[i][1].sum())
        nl, nr = float(bl_b_l[i][1].sum()), float(bl_b_r[i][1].sum())
        rep.append({"s_range": lab, "A_events": nA, "B_events": nB,
                    "original_l": nl, "original_r": nr,
                    "ratio_B_over_A": nB / nA, "ratio_r_over_l": nr / nl,
                    "diff_of_ratios": nB / nA - nr / nl,
                    "binomial_p_A_vs_half": stats.binomtest(int(nB), int(nA + nB), 0.5).pvalue,
                    "binomial_p_orig": stats.binomtest(int(nr), int(nl + nr), 0.5).pvalue})
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "chpol2_C_variant_replication.csv", index=False)
    print(rep.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. regime-level (concatenated) numbers, flagged as a mixture
    # ------------------------------------------------------------------
    mix = []
    for grp in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        for ch in ["channel_l", "channel_r"]:
            bl = blocks_of(SHEETS[(grp, ch)])
            cat_x = np.concatenate([b[0] for b in bl])
            cat_y = np.concatenate([b[1] for b in bl])
            st = shape_stats(cat_x, cat_y)
            mix.append({"group": grp, "channel": ch, "view": "orientation_merged",
                        "events": st["events"], "weighted_median": st["weighted_median"],
                        "sigma_ln": st["sigma_ln"], "frac_gt_10": st["frac_gt_10"],
                        "monotone": bool(np.all(np.diff(cat_x) > 0)),
                        "mixture_of_disjoint_s_ranges": True})
            fs = blocks_of(FAST_SHEETS[(grp, ch)])[0]
            stf = shape_stats(fs[0], fs[1])
            mix.append({"group": grp, "channel": ch, "view": "fast_block",
                        "events": stf["events"], "weighted_median": stf["weighted_median"],
                        "sigma_ln": stf["sigma_ln"], "frac_gt_10": stf["frac_gt_10"],
                        "monotone": bool(np.all(np.diff(fs[0]) > 0)),
                        "mixture_of_disjoint_s_ranges": False})
    mix = pd.DataFrame(mix)
    mix.to_csv(OUT / "chpol2_mixture_flag.csv", index=False)
    print("\n=== chpol2.9 concatenated 'view' numbers, flagged as mixtures ===")
    print(mix.to_string(index=False))

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    bundle = {
        "dimension": "polarization channel contrast, decomposed into source blocks",
        "analysis_unit": "one complete source block x polarization channel",
        "seed": SEED, "n_boot": NBOOT,
        "layout_verified_all_true": ok,
        "block_stats": json.loads(blk.to_json(orient="records")),
        "block_paired": json.loads(pair.to_json(orient="records")),
        "channel_fraction_heterogeneity": json.loads(het.to_json(orient="records")),
        "poisson_ratio": json.loads(pr.to_json(orient="records")),
        "grid_tv": json.loads(gdf.to_json(orient="records")),
        "C_variant": json.loads(cv.to_json(orient="records")),
        "C_variant_replication": json.loads(rep.to_json(orient="records")),
        "mixture_flag": json.loads(mix.to_json(orient="records")),
        "caveats": [
            "Bins are nested repeats inside one series, not independent samples.",
            "The 'slow' regime is a raw concatenation of two disjoint s-range blocks "
            "whose drift supports overlap; it is a mixture, not a single distribution.",
            "Only 3 paired source groups exist; block-level n can be as low as 52 events.",
            "Multinomial CIs describe shape conditional on the block total.",
            "The l and r blocks observe the same events, so they are paired, not independent.",
        ],
    }
    with open(OUT / "chpol2_block_contrast.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print("\nwrote chpol2_*.csv and chpol2_block_contrast.json")


if __name__ == "__main__":
    main()