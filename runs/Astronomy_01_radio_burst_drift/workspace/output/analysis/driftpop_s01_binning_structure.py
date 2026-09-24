#!/usr/bin/env python3
"""driftpop_s01 - Exact binning structure and provenance of drift_population_series.tsv.

Dimension: is the drift-rate distribution one population, several populations, a
continuous distribution, or insufficient data?

This script establishes, numerically and without any model:
  (1) per population_series_id x drift_regime: number of bins, bin width(s),
      bin edges, covered range, empty-bin count, count concentration;
  (2) the provenance of every bin: which deidentified workbook block(s) compose
      the series, and whether the composition is an exact concatenation;
  (3) that the 'fast'/'slow' regime labels correspond exactly to the orientation
      sub-ranges of the workbook (sum-matching against the orientation blocks),
      i.e. the regime split is a *design* split, not a data-driven split.

Analysis unit for everything printed here: one source-group x polarization
series (population_series_id). Counts inside a series are events (bursts), not
independent units in the sense of the task contract; they are nested repeats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "output" / "results"
OUT.mkdir(parents=True, exist_ok=True)

POP = DATA / "normalized_views" / "drift_population_series.tsv"
BLOCK_CSV = OUT / "s03_workbook_blocks.csv"
BLOCK_NPZ = OUT / "s03_workbook_blocks.npz"

# Workbook block sums for the orientation-range blocks (group x channel),
# taken from the raw workbook by driftpop_s01b (verified there); keyed here by
# (source group, channel, orientation label).
ORIENT_SHEETS = {
    ("anonymous_group_1", "channel_l"): "Sheet_009",
    ("anonymous_group_1", "channel_r"): "Sheet_010",
    ("anonymous_group_2", "channel_l"): "Sheet_011",
    ("anonymous_group_2", "channel_r"): "Sheet_012",
    ("anonymous_group_3", "channel_l"): "Sheet_013",
    ("anonymous_group_3", "channel_r"): "Sheet_014",
}


def load_blocks() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Join the s03 block table to the s03 npz arrays (same row order)."""
    df = pd.read_csv(BLOCK_CSV)
    z = np.load(BLOCK_NPZ)
    arrs = {}
    for i, key in enumerate(z.files):
        arrs[i] = z[key]
    assert len(df) == len(arrs), (len(df), len(arrs))
    # sanity: key encodes sheet name and file order -> same enumeration
    for i, key in enumerate(z.files):
        assert key.startswith(f"Sheet_{int(df.iloc[i]['sheet'].split('_')[1]):03d}__")
        assert key.split("__")[1] == str(i)
    return df, arrs


def bin_edges(centers: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Edges for an internally-uniform grid: center +- slope/2, slope from LS.

    Returns (edges, width, max residual of the uniform-grid fit).
    """
    c = np.sort(np.asarray(centers, float))
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5]), 1.0, 0.0
    idx = np.arange(len(c), dtype=float)
    A = np.vstack([idx, np.ones_like(idx)]).T
    slope, intercept = np.linalg.lstsq(A, c, rcond=None)[0]
    resid = float(np.max(np.abs(c - (slope * idx + intercept))))
    edges = np.concatenate([[c[0] - slope / 2], c + slope / 2])
    return edges, float(slope), resid


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main() -> None:
    pop = pd.read_csv(POP, sep="\t")
    bdf, barrs = load_blocks()

    # ---------------------------------------------------------------- structure
    rows = []
    edges_by_series: dict[str, np.ndarray] = {}
    counts_by_series: dict[str, np.ndarray] = {}
    for (pid, regime), g in pop.groupby(["population_series_id", "drift_regime"]):
        g = g.sort_values("distribution_bin_center")
        c = g.distribution_bin_center.to_numpy(float)
        y = g.event_count.to_numpy(float)
        e, w, resid = bin_edges(c)
        d = np.diff(c)
        rows.append(
            dict(
                population_series_id=pid,
                anonymous_source_group=g.anonymous_source_group.iloc[0],
                polarization_channel=g.polarization_channel.iloc[0],
                drift_regime=regime,
                n_bins=len(c),
                bin_width_median=w,
                bin_width_min=float(d.min()) if len(d) else np.nan,
                bin_width_max=float(d.max()) if len(d) else np.nan,
                n_distinct_widths=int(len(np.unique(np.round(d, 6)))),
                grid_residual_max=resid,
                center_min=float(c.min()),
                center_max=float(c.max()),
                edge_min=float(e[0]),
                edge_max=float(e[-1]),
                n_empty=int((y == 0).sum()),
                frac_empty=float((y == 0).mean()),
                total_events=int(y.sum()),
                max_bin_count=int(y.max()),
                max_bin_center=float(c[int(np.argmax(y))]),
                gini=gini(y),
                frac_top1_bin=float(y.max() / y.sum()),
                frac_top5_bins=float(np.sort(y)[-5:].sum() / y.sum()),
                frac_in_5pct_bins=float(np.sort(y)[-max(1, int(round(0.05 * len(y)))):].sum() / y.sum()),
            )
        )
        edges_by_series[f"{pid}|{regime}"] = e
        counts_by_series[f"{pid}|{regime}"] = y

    struct = pd.DataFrame(rows)
    struct.to_csv(OUT / "driftpop_s01_binning_structure.csv", index=False)

    print("=== binning structure per (population_series_id, drift_regime) ===")
    cols = ["population_series_id", "drift_regime", "n_bins", "bin_width_median",
            "n_distinct_widths", "center_min", "center_max", "n_empty", "frac_empty",
            "total_events", "max_bin_count", "max_bin_center", "gini",
            "frac_top1_bin", "frac_top5_bins", "frac_in_5pct_bins"]
    print(struct[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # slow-combined widths (why the min inter-center spacing is misleading)
    print("\n=== why min inter-center spacing is NOT the bin width (ps1 slow) ===")
    g1 = pop[(pop.population_series_id == "population_series_1") & (pop.drift_regime == "slow")]
    c1 = np.sort(g1.distribution_bin_center.to_numpy(float))
    print("first 8 centers:", np.round(c1[:8], 6))
    print("first 7 spacings:", np.round(np.diff(c1)[:7], 6))
    print("two interleaved 0.10-wide grids offset by ~0.05 -> alternating spacing;")
    print("legitimate resolution of the merged series is ~0.10 MHz/s, not 0.04.")

    # ---------------------------------------------------------------- provenance
    blocks = {}
    for i, (_, r) in enumerate(bdf.iterrows()):
        blocks[i] = dict(sheet=r["sheet"], label=r["label"], n=int(r["n"]),
                         y_sum=float(r["y_sum"]), x=barrs[i][0], y=barrs[i][1])

    FAST_SHEET = {1: "Sheet_025", 2: "Sheet_027", 3: "Sheet_029",
                  4: "Sheet_031", 5: "Sheet_033", 6: "Sheet_035"}
    SLOW_SHEETS = {1: "Sheet_024", 2: "Sheet_026", 3: "Sheet_028",
                   4: "Sheet_030", 5: "Sheet_032", 6: "Sheet_034"}

    prov = []
    for k in range(1, 7):
        pid = f"population_series_{k}"
        grp = pop[pop.population_series_id == pid].anonymous_source_group.iloc[0]
        ch = pop[pop.population_series_id == pid].polarization_channel.iloc[0]

        # fast: must equal ONE workbook block exactly
        yfast = counts_by_series[f"{pid}|fast"]
        cfast = np.sort(pop[(pop.population_series_id == pid) & (pop.drift_regime == "fast")]
                        .distribution_bin_center.to_numpy(float))
        cand = [b for b in blocks.values()
                if b["sheet"] == FAST_SHEET[k] and len(b["x"]) == len(cfast)]
        assert len(cand) == 1, (pid, [(b["sheet"], b["label"]) for b in cand])
        b = cand[0]
        o = np.argsort(b["x"])
        ok_exact = bool(np.allclose(b["x"][o], cfast, atol=1e-9)
                        and np.allclose(b["y"][o], yfast, atol=1e-9))
        prov.append(dict(population_series_id=pid, drift_regime="fast",
                         source_group=grp, channel=ch,
                         sheet=b["sheet"], labels=b["label"], n_blocks=1,
                         concat_exact=ok_exact,
                         block_events=[float(b["y"].sum())],
                         n_bins=[len(b["x"])],
                         bin_width=[float(np.median(np.diff(np.sort(b["x"]))))]))

        # slow: must equal concatenation of the two blocks in its sheet
        yslow = counts_by_series[f"{pid}|slow"]
        cslow = np.sort(pop[(pop.population_series_id == pid) & (pop.drift_regime == "slow")]
                        .distribution_bin_center.to_numpy(float))
        cands = sorted([b for b in blocks.values() if b["sheet"] == SLOW_SHEETS[k]],
                       key=lambda d: d["label"])
        xx = np.concatenate([d["x"] for d in cands])
        yy = np.concatenate([d["y"] for d in cands])
        o = np.argsort(xx)
        xx, yy = xx[o], yy[o]
        ok_exact = bool(len(xx) == len(cslow)
                        and np.allclose(xx, cslow, atol=1e-9)
                        and np.allclose(yy, yslow, atol=1e-9))
        prov.append(dict(population_series_id=pid, drift_regime="slow",
                         source_group=grp, channel=ch,
                         sheet=SLOW_SHEETS[k], labels=" || ".join(d["label"] for d in cands),
                         n_blocks=len(cands), concat_exact=ok_exact,
                         block_events=[float(d["y"].sum()) for d in cands],
                         n_bins=[len(d["x"]) for d in cands],
                         bin_width=[float(np.median(np.diff(np.sort(d["x"])))) for d in cands]))

    provdf = pd.DataFrame(prov)
    provdf.to_csv(OUT / "driftpop_s01_provenance.csv", index=False)
    print("\n=== provenance: which workbook blocks compose each series ===")
    print(provdf.to_string(index=False))
    assert provdf.concat_exact.all(), "provenance concatenation not exact!"
    print("\nAll 12 series are EXACT re-expressions of workbook blocks "
          "(fast = 1 block, slow = concatenation of 2 blocks).")

    # ------------------------------------------ regime == orientation sub-range
    orient_sums = {}
    for (grp, ch), sheet in ORIENT_SHEETS.items():
        for b in blocks.values():
            if b["sheet"] != sheet:
                continue
            orient_sums[(grp, ch, b["label"])] = b["y_sum"]
    print("\n=== orientation block totals by group x channel x sub-range ===")
    for kk, vv in sorted(orient_sums.items()):
        print(f"  {kk}: {vv:.0f}")

    match_rows = []
    for k in range(1, 7):
        pid = f"population_series_{k}"
        grp = pop[pop.population_series_id == pid].anonymous_source_group.iloc[0]
        ch = pop[pop.population_series_id == pid].polarization_channel.iloc[0]
        cands = sorted([b for b in blocks.values() if b["sheet"] == SLOW_SHEETS[k]],
                       key=lambda d: d["label"])
        fastb = [b for b in blocks.values() if b["sheet"] == FAST_SHEET[k]][0]
        got = [fastb["y_sum"], cands[0]["y_sum"], cands[1]["y_sum"]]
        # labels in the workbook use 'group_a/b/c' style
        gl = {"anonymous_group_1": "group_a", "anonymous_group_2": "group_b",
              "anonymous_group_3": "group_c"}[grp]
        cl = ch.replace("channel_", "channel_")
        lab_lo = f"{gl} (-35<s<-10), {cl}"
        lab_mid = f"{gl}(-10<s<0), {cl}"
        lab_hi = f"{gl}(0<s<20), {cl}"
        expected = [orient_sums.get((grp, ch, lab_lo)),
                    orient_sums.get((grp, ch, lab_mid)),
                    orient_sums.get((grp, ch, lab_hi))]
        match_rows.append(dict(
            population_series_id=pid, group=grp, channel=ch,
            fast_block_sum=got[0], orient_m35_m10=expected[0],
            fast_matches_orient_m35_m10=bool(expected[0] is not None and got[0] == expected[0]),
            slow_blockA_sum=got[1], orient_m10_0=expected[1],
            slowA_matches_orient_m10_0=bool(expected[1] is not None and got[1] == expected[1]),
            slow_blockB_sum=got[2], orient_0_20=expected[2],
            slowB_matches_orient_0_20=bool(expected[2] is not None and got[2] == expected[2]),
        ))
    mdf = pd.DataFrame(match_rows)
    mdf.to_csv(OUT / "driftpop_s01_regime_orientation_map.csv", index=False)
    print("\n=== regime label <-> orientation sub-range (sum matching) ===")
    print(mdf.to_string(index=False))
    print("\nALL fast sums match -35<s<-10:",
          bool(mdf.fast_matches_orient_m35_m10.all()))
    print("ALL slow block A sums match -10<s<0:",
          bool(mdf.slowA_matches_orient_m10_0.all()))
    print("ALL slow block B sums match 0<s<20:",
          bool(mdf.slowB_matches_orient_0_20.all()))
    print("\nINTERPRETATION: 'fast' = orientation sub-range -35<s<-10 (1.0 MHz/s bins);")
    print("'slow' = merged (-10<s<0) and (0<s<20) sub-ranges (0.10 MHz/s bins).")
    print("The regime split is a design partition of the source data, not a")
    print("data-driven separation of drift rates.")

    # ------------------------------------------------------------------- overlap
    ov = []
    for pid, g in pop.groupby("population_series_id"):
        f = np.sort(g[g.drift_regime == "fast"].distribution_bin_center.to_numpy(float))
        s = np.sort(g[g.drift_regime == "slow"].distribution_bin_center.to_numpy(float))
        lo, hi = max(f.min(), s.min()), min(f.max(), s.max())
        f_in = ((f >= lo) & (f <= hi)).sum()
        s_in = ((s >= lo) & (s <= hi)).sum()
        ov.append(dict(population_series_id=pid, fast_min=f.min(), fast_max=f.max(),
                       slow_min=s.min(), slow_max=s.max(),
                       overlap_lo=lo, overlap_hi=hi,
                       overlap_width=hi - lo,
                       fast_bins_in_overlap=int(f_in), slow_bins_in_overlap=int(s_in)))
    ovdf = pd.DataFrame(ov)
    ovdf.to_csv(OUT / "driftpop_s01_regime_overlap.csv", index=False)
    print("\n=== fast/slow center range overlap ===")
    print(ovdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # total events per regime, all series + per series
    tot = pop.groupby(["drift_regime"]).event_count.sum().to_dict()
    print("\n=== total events per regime (all 6 series) ===")
    print(tot)
    tot_ser = pop.groupby(["population_series_id", "drift_regime"]).event_count.sum().unstack()
    print(tot_ser.to_string())
    print("\neffective independent units: 6 source-group x channel series "
          "(and 5 complete dynamic-spectrum events in the sibling data family).")

    summary = dict(
        series=struct.to_dict(orient="records"),
        provenance=provdf.to_dict(orient="records"),
        regime_orientation=mdf.to_dict(orient="records"),
        overlap=ovdf.to_dict(orient="records"),
        total_events_by_regime={k: int(v) for k, v in tot.items()},
    )
    (OUT / "driftpop_s01_binning_structure.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")
    print("\nwrote:", OUT / "driftpop_s01_binning_structure.csv", "and .json")


if __name__ == "__main__":
    main()