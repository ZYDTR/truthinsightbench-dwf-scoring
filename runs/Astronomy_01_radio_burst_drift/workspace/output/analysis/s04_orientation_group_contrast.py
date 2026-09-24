#!/usr/bin/env python3
"""S04 - Directional/orientation count series and anonymous source-group contrasts.

Dimension: directional/orientation count series and source-group conditional
differences (TruthInsightBench Astronomy_01_radio_burst_drift, T0=2022-10-13).

Analysis unit (binding): one complete source-polarization series.  The 52
relative coordinates inside a series are NESTED REPEATS, not independent
samples: every series-level or group-level statistic below is computed on the
series (or group of series) as the unit, and permutation/bootstrap resampling
is done at the SERIES level only.

Steps
  A  load orientation_count_series.tsv, establish totals and exact coordinate grid
  B  per-series shape statistics (peak, zero fraction, centre of mass, local maxima)
  C  pairwise series comparison (Pearson / Spearman / total-variation / total ratio)
  D  reconstruct the normalized-series -> workbook mapping: concatenate the three
     directional blocks of Sheet_009..Sheet_014 (x=-34..-10, -9..0, 1..20) and
     compare with each normalized 52-point series (grid = -33..20 with the three
     zero boundary coordinates dropped).  Also characterise the 'group_b C'
     variant blocks of Sheet_046/047.
  E  cross-view consistency: the orientation block -34..-10 count of each series
     should equal its drift-regime 'fast' total from drift_population_series.tsv
  F  group contrasts with distribution-free statistics: total-variation distance
     between normalized profiles, centre-of-mass, region shares, plus exact
     permutation tests at the series level (6 units -> only 90 distinct
     assignments of 6 series to 3 groups of 2; p_min = 1/90)
  G  series-level bootstrap of group means (2 series per group - state the limit)
  H  figures

Deterministic: no RNG seeds needed (exact permutation enumeration, deterministic
bootstrap enumeration).
"""
from __future__ import annotations

import json
import re
import zipfile
from itertools import permutations
from pathlib import Path
import xml.etree.ElementTree as ET

# --- timed entry point ----------------------------------------------------- #
# Full workbook scan is not needed: only the small directional sheets are read
# directly from the xlsx zip stream (openpyxl on this 32 MB file is minutes-slow
# here; direct XML parsing of 8 sheets takes < 1 s).

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized_views"
XLSX = ROOT / "data" / "source_data" / "all_source_data_deidentified.xlsx"
RESULTS = ROOT / "output" / "results"
FIGS = RESULTS / "figures"
LOGS = ROOT / "output" / "logs"
for d in (RESULTS, FIGS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

MSG = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# Workbook direction blocks of Sheet_009..Sheet_014 are labelled
# '<group> (s-range), channel_x' in the order (fast, mid, slow s-range).
ORI_SHEETS = [
    ("Sheet_009", "anonymous_group_1", "channel_l"),
    ("Sheet_010", "anonymous_group_1", "channel_r"),
    ("Sheet_011", "anonymous_group_2", "channel_l"),
    ("Sheet_012", "anonymous_group_2", "channel_r"),
    ("Sheet_013", "anonymous_group_3", "channel_l"),
    ("Sheet_014", "anonymous_group_3", "channel_r"),
]
C_SHEETS = [("Sheet_046", "channel_l"), ("Sheet_047", "channel_l")]
DROP_COORDS = (-34.0, -9.0, 1.0)  # zero boundary bins dropped by the normalization


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# raw xlsx reader (zip + ElementTree).  openpyxl in this environment takes
# minutes on this 32 MB workbook, so the orientation sheets are read directly.
# --------------------------------------------------------------------------- #
def xlsx_reader(path: Path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("xl/workbook.xml").decode("utf-8"))
    rid = {s.attrib["name"]: s.attrib[REL + "id"] for s in root.find(MSG + "sheets")}
    rels = dict(
        re.findall(
            r'Id="(rId\d+)"[^>]*Target="([^"]+)"',
            z.read("xl/_rels/workbook.xml.rels").decode("utf-8"),
        )
    )

    def sheet_rows(sheet: str):
        r = ET.fromstring(z.read("xl/" + rels[rid[sheet]]))
        for rowel in r.find(MSG + "sheetData"):
            cells = {}
            for c in rowel:
                col = re.match(r"([A-Z]+)", c.attrib["r"]).group(1)
                v = c.find(MSG + "v")
                if v is None:
                    continue
                t = c.attrib.get("t", "n")
                cells[col] = ("S", int(v.text)) if t == "s" else float(v.text)
            yield int(rowel.attrib["r"]), cells

    def blocks(sheet: str):
        out, cur = [], None
        for _, cells in sheet_rows(sheet):
            b = cells.get("B")
            if b is not None and b[0] == "S":
                cur = {"x": [], "y": []}
                out.append(cur)
            x, y = cells.get("C"), cells.get("D")
            if cur is not None and isinstance(x, float) and isinstance(y, float):
                cur["x"].append(x)
                cur["y"].append(y)
        return [(np.array(b["x"], float), np.array(b["y"], float)) for b in out]

    return blocks


# --------------------------------------------------------------------------- #
def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two non-negative profiles."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum())


def com(coords: np.ndarray, y: np.ndarray) -> float:
    w = np.asarray(y, float)
    return float((coords * w).sum() / w.sum())


def local_maxima(coords: np.ndarray, y: np.ndarray) -> list[float]:
    out = []
    for i in range(1, len(y) - 1):
        if y[i] > 0 and y[i] > y[i - 1] and y[i] > y[i + 1]:
            out.append(float(coords[i]))
    return out


def pseudof(profiles: np.ndarray, groups: np.ndarray) -> float:
    """PERMANOVA pseudo-F for a 3-group, 2-replicates-per-group design."""
    grand = profiles.mean(axis=0)
    sst = float(((profiles - grand) ** 2).sum())
    ssw = 0.0
    for g in np.unique(groups):
        blk = profiles[groups == g]
        ssw += float(((blk - blk.mean(axis=0)) ** 2).sum())
    k = len(np.unique(groups))
    n = len(profiles)
    return ((sst - ssw) / (k - 1)) / (ssw / (n - k))


def main() -> None:
    report: dict = {}

    # ---------------- A. normalized orientation series ---------------------- #
    ori = pd.read_csv(DATA / "orientation_count_series.tsv", sep="\t")
    piv = ori.pivot(index="relative_coordinate", columns="orientation_series_id",
                    values="event_count")
    sid_order = [f"orientation_series_{i}" for i in range(1, 7)]
    piv = piv[sid_order]
    coords = piv.index.to_numpy(float)
    profiles = piv.to_numpy(float)  # 52 x 6
    totals = profiles.sum(axis=0)
    log("=== A. orientation series totals (analysis unit = series) ===")
    for s, t in zip(sid_order, totals):
        log(f"  {s}: total={t:.0f}")
    log(f"  coordinates: n={len(coords)} min={coords.min()} max={coords.max()}")
    log(f"  coordinate set == -33..20 minus (-34,-9,1)? "
        f"{sorted(coords.tolist()) == sorted(set(range(-33, 21)) - {-9, 1})}")
    report["series_totals"] = {s: float(t) for s, t in zip(sid_order, totals)}
    report["n_coordinates"] = int(len(coords))

    # ---------------- B. shape statistics ---------------------------------- #
    rows = []
    for j, s in enumerate(sid_order):
        y = profiles[:, j]
        nz = int((y == 0).sum())
        lm = local_maxima(coords, y)
        rows.append(
            {
                "orientation_series_id": s,
                "total_events": float(y.sum()),
                "n_coordinates": len(y),
                "n_zero_bins": nz,
                "zero_fraction": nz / len(y),
                "peak_coordinate": float(coords[int(np.argmax(y))]),
                "peak_count": float(y.max()),
                "centre_of_mass": com(coords, y),
                "median_coordinate": float(coords[np.searchsorted(np.cumsum(y), y.sum() / 2)]),
                "n_local_maxima": len(lm),
                "local_maxima": ";".join(f"{v:g}" for v in lm),
                "fast_regime_total_s_le_-10": float(y[coords <= -10].sum()),
                "slow_regime_total_s_gt_-10": float(y[coords > -10].sum()),
                "share_fast_s_le_-10": float(y[coords <= -10].sum() / y.sum()),
                "share_mid_-10_lt_s_le_0": float(y[(coords > -10) & (coords <= 0)].sum() / y.sum()),
                "share_s_gt_0": float(y[coords > 0].sum() / y.sum()),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "s04_orientation_series_stats.csv", index=False)
    log("\n=== B. per-series shape statistics (unit = series) ===")
    log(stats.to_string(index=False))

    # structurally zero across all six series
    zero_all = [float(c) for c, m in zip(coords, (profiles == 0).all(axis=1)) if m]
    log(f"\n  coordinates that are ZERO in every series (coverage gaps): {zero_all}")
    log(f"  coordinates absent from the grid entirely: {sorted(DROP_COORDS)}")
    report["structurally_zero_coords"] = zero_all
    report["grid_absent_coords"] = sorted(DROP_COORDS)

    # ---------------- C. pairwise series comparison ------------------------ #
    pw_rows = []
    for a in range(6):
        for b in range(a + 1, 6):
            ya, yb = profiles[:, a], profiles[:, b]
            pear = float(np.corrcoef(ya, yb)[0, 1])
            spear = float(pd.Series(ya).corr(pd.Series(yb), method="spearman"))
            pw_rows.append(
                {
                    "series_a": sid_order[a],
                    "series_b": sid_order[b],
                    "total_a": float(ya.sum()),
                    "total_b": float(yb.sum()),
                    "total_ratio_b_over_a": float(yb.sum() / ya.sum()),
                    "pearson_r_counts": pear,
                    "spearman_rho_counts": spear,
                    "pearson_r_log1p_counts": float(np.corrcoef(np.log1p(ya), np.log1p(yb))[0, 1]),
                    "tv_distance_normalized": tv(ya, yb),
                    "max_abs_count_difference": float(np.abs(ya - yb).max()),
                }
            )
    pwd = pd.DataFrame(pw_rows)
    pwd.to_csv(RESULTS / "s04_orientation_pairwise.csv", index=False)
    log("\n=== C. pairwise series comparison (unit = series) ===")
    log(pwd.to_string(index=False))
    log(f"\n  total ratio max/min across series = {totals.max() / totals.min():.2f}")

    # ---------------- D. workbook mapping ---------------------------------- #
    blocks = xlsx_reader(XLSX)
    map_rows = []
    for sheet, grp, ch in ORI_SHEETS:
        bs = blocks(sheet)
        xs, ys = [], []
        for bx, by in bs:
            for x, y in zip(bx, by):
                if x in DROP_COORDS:
                    continue
                xs.append(x)
                ys.append(y)
        xa, ya = np.array(xs, float), np.array(ys, float)
        order = np.argsort(xa)
        xa, ya = xa[order], ya[order]
        assert np.allclose(xa, coords), f"grid mismatch {sheet}"
        j = int(np.argmax([np.array_equal(ya, profiles[:, k]) for k in range(6)]))
        exact = np.array_equal(ya, profiles[:, j])
        map_rows.append(
            {
                "workbook_sheet": sheet,
                "workbook_group": grp,
                "workbook_channel": ch,
                "block_sums": ";".join(f"{int(b[1].sum())}" for b in bs),
                "assembled_total": float(ya.sum()),
                "normalized_series": sid_order[j] if exact else None,
                "exact_match": bool(exact),
            }
        )
    mp = pd.DataFrame(map_rows)
    mp.to_csv(RESULTS / "s04_orientation_mapping.csv", index=False)
    log("\n=== D. normalized orientation series -> workbook mapping "
        "(assembled = 3 blocks, x in -34..20, boundary zeros dropped) ===")
    log(mp.to_string(index=False))
    log(f"  all six series exactly recovered: {bool(mp['exact_match'].all())}")

    # group_b 'C' variant blocks
    crows = []
    for sheet, ch in C_SHEETS:
        bs = blocks(sheet)
        base = blocks("Sheet_011" if sheet == "Sheet_046" else "Sheet_012")
        for i, (bx, by) in enumerate(bs):
            ax, ay = base[i]
            common = [k for k, v in enumerate(bx) if v in set(ax.tolist())]
            idx = [list(ax).index(bx[k]) for k in common]
            crows.append(
                {
                    "variant_sheet": sheet,
                    "group": "anonymous_group_2",
                    "channel": ch,
                    "block_index": i,
                    "n_points": len(bx),
                    "x_range": f"{bx.min():g}..{bx.max():g}",
                    "x_step": float(np.median(np.diff(bx))) if len(bx) > 1 else np.nan,
                    "variant_total": float(by.sum()),
                    "base_total": float(ay.sum()),
                    "total_ratio_variant_over_base": float(by.sum() / ay.sum()),
                    "pearson_r_on_common_grid": float(np.corrcoef(ay[idx], by[common])[0, 1]),
                }
            )
    cd = pd.DataFrame(crows)
    cd.to_csv(RESULTS / "s04_group_b_C_variant.csv", index=False)
    log("\n=== D2. group_b 'C' variant blocks (Sheet_046/047) vs group_b base "
        "(Sheet_011/012) ===")
    log(cd.to_string(index=False))

    # ---------------- E. cross-view consistency with drift regimes ---------- #
    # NB the regime split runs at s=-10 (fast = s<=-10), not s<-10: coordinate
    # -10 carries counts in every series, and the earlier '< -10' reading of the
    # regime boundary loses them (it is why the first cross-check run disagreed).
    pop = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
    pser = (pop[pop.drift_regime == "fast"].groupby("population_series_id")["event_count"]
            .sum().to_dict())
    sser = (pop[pop.drift_regime == "slow"].groupby("population_series_id")["event_count"]
            .sum().to_dict())
    xrows = []
    for j, s in enumerate(sid_order):
        orient_fast = float(profiles[coords <= -10, j].sum())
        orient_slow = float(profiles[coords > -10, j].sum())
        drift_fast = float(pser[f"population_series_{j + 1}"])
        drift_slow = float(sser[f"population_series_{j + 1}"])
        # the normalized view zeroes/drops the two interior boundary bins x=-9 and
        # x=1; recover them from the workbook block sums to close the identity
        bs = blocks(ORI_SHEETS[j][0])
        dropped = float(bs[1][1][0]) + float(bs[2][1][0])  # y(x=-9) + y(x=1)
        xrows.append(
            {
                "orientation_series_id": s,
                "population_series_id": f"population_series_{j + 1}",
                "orientation_fast_region_s_le_-10": orient_fast,
                "drift_regime_fast_total": drift_fast,
                "fast_equal": orient_fast == drift_fast,
                "orientation_slow_region_s_gt_-10": orient_slow,
                "workbook_dropped_boundary_bins": dropped,
                "slow_reconstructed": orient_slow + dropped,
                "drift_regime_slow_total": drift_slow,
                "slow_equal": orient_slow + dropped == drift_slow,
            }
        )
    xd = pd.DataFrame(xrows)
    xd.to_csv(RESULTS / "s04_orientation_vs_drift_consistency.csv", index=False)
    log("\n=== E. cross-view check vs drift regimes (unit = series) ===")
    log("  fast regime = orientation block s<=-10 (matches Sheet_025/027/029/031/033/035);")
    log("  slow regime = orientation s>-10 PLUS the two interior boundary bins x=-9 and x=1")
    log("  that the normalized orientation view pins to zero.")
    log(xd.to_string(index=False))
    log(f"  fast totals identical for all six series: {bool(xd['fast_equal'].all())}")
    log(f"  slow totals identical for all six series: {bool(xd['slow_equal'].all())}")

    # ---------------- F. group contrasts ------------------------------------ #
    groups = np.array(["anonymous_group_1"] * 2 + ["anonymous_group_2"] * 2
                      + ["anonymous_group_3"] * 2)
    P = profiles / profiles.sum(axis=0, keepdims=True)  # 52 x 6 normalized profiles

    grows = []
    for g in ["anonymous_group_1", "anonymous_group_2", "anonymous_group_3"]:
        cols = [j for j in range(6) if groups[j] == g]
        pool = profiles[:, cols].sum(axis=1)
        grows.append(
            {
                "group": g,
                "channel_l_series": sid_order[cols[0]],
                "channel_r_series": sid_order[cols[1]],
                "group_total": float(pool.sum()),
                "mean_of_series_totals": float(totals[cols].mean()),
                "tv_between_its_two_channels": tv(profiles[:, cols[0]], profiles[:, cols[1]]),
                "com_channel_l": com(coords, profiles[:, cols[0]]),
                "com_channel_r": com(coords, profiles[:, cols[1]]),
                "com_pooled": com(coords, pool),
                "peak_coordinate_pooled": float(coords[int(np.argmax(pool))]),
                "share_fast_s_le_-10_pooled": float(pool[coords <= -10].sum() / pool.sum()),
                "share_mid_-10_s_le_0_pooled": float(pool[(coords > -10) & (coords <= 0)].sum() / pool.sum()),
                "share_s_gt_0_pooled": float(pool[coords > 0].sum() / pool.sum()),
            }
        )
    gd = pd.DataFrame(grows)
    gd.to_csv(RESULTS / "s04_orientation_group_stats.csv", index=False)
    log("\n=== F. group summary (unit = group of 2 series; coordinates nested) ===")
    log(gd.to_string(index=False))

    # group x coordinate table and TV matrix
    gcols = {"anonymous_group_1": [0, 1], "anonymous_group_2": [2, 3],
             "anonymous_group_3": [4, 5]}
    gp = {g: profiles[:, c].sum(axis=1) for g, c in gcols.items()}
    gt = pd.DataFrame({"relative_coordinate": coords,
                       **{g: v for g, v in gp.items()}})
    gt.to_csv(RESULTS / "s04_orientation_group_by_coordinate.csv", index=False)

    gnames = list(gcols)
    tvmat = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            tvmat[a, b] = tv(gp[gnames[a]], gp[gnames[b]])
    log("\n  group x group total-variation distance between pooled profiles:")
    log(pd.DataFrame(tvmat, index=gnames, columns=gnames).to_string())

    # exact permutation tests at the SERIES level (6 units, 3 groups of 2)
    perms = sorted({p for p in permutations([0, 0, 1, 1, 2, 2])})
    assert len(perms) == 90
    obs_f = pseudof(P.T, groups)
    obs_com = np.array([com(coords, profiles[:, c].sum(axis=1)) for c in gcols.values()])
    obs_com_range = float(obs_com.max() - obs_com.min())
    obs_tv = float(max(tvmat[i, j] for i in range(3) for j in range(i + 1, 3)))
    f_perm, com_perm, tv_perm = [], [], []
    for p in perms:
        g = np.array(p)
        f_perm.append(pseudof(P.T, np.array([f"g{v}" for v in p])))
        cs = [com(coords, profiles[:, [k for k in range(6) if g[k] == v]].sum(axis=1))
              for v in range(3)]
        com_perm.append(max(cs) - min(cs))
        pooled = [profiles[:, [k for k in range(6) if g[k] == v]].sum(axis=1) for v in range(3)]
        tv_perm.append(max(tv(pooled[i], pooled[j]) for i in range(3) for j in range(i + 1, 3)))
    f_perm, com_perm, tv_perm = map(np.array, (f_perm, com_perm, tv_perm))
    p_f = float((f_perm >= obs_f).mean())
    p_com = float((com_perm >= obs_com_range).mean())
    p_tv = float((tv_perm >= obs_tv).mean())
    log("\n=== F2. exact permutation tests, resampling at the SERIES level ===")
    log(f"  units = 6 series; 3 groups x 2 series; 90 distinct assignments, p_min=1/90={1/90:.4f}")
    log(f"  PERMANOVA pseudo-F on 52-dim normalized profiles: F={obs_f:.2f}  p={p_f:.4f}")
    log(f"  centre-of-mass range across group means: {obs_com_range:.3f}  p={p_com:.4f}")
    log(f"  max pairwise TV between group profiles: {obs_tv:.3f}  p={p_tv:.4f}")
    # 2-group contrast (2 vs 2 series -> 6 distinct partitions, p_min=1/6)
    log("  two-group contrasts (4 series, 6 distinct partitions, p_min=1/6):")
    two = {}
    for i in range(3):
        for j in range(i + 1, 3):
            units = gcols[gnames[i]] + gcols[gnames[j]]  # 4 series, 2 per group
            pi = P[:, gcols[gnames[i]]].mean(axis=1)
            pj = P[:, gcols[gnames[j]]].mean(axis=1)
            obs = tv(pi, pj)
            null = []
            for p in sorted(set(permutations([0, 0, 1, 1]))):
                sel = [units[k] for k in range(4)]
                a = P[:, [sel[k] for k in range(4) if p[k] == 0]].mean(axis=1)
                b = P[:, [sel[k] for k in range(4) if p[k] == 1]].mean(axis=1)
                null.append(tv(a, b))
            pv = float((np.array(null) >= obs).mean())
            two[f"{gnames[i]}__vs__{gnames[j]}"] = {"tv": float(obs), "p_exact": pv}
            log(f"    {gnames[i]} vs {gnames[j]}: TV={obs:.3f} p={pv:.3f}")

    # nesting-violating contrast (coordinates treated as independent) - flagged invalid
    from scipy.stats import chi2_contingency
    chi2 = pchi = dof = float("nan")
    exp = None
    chi_note = ""
    tab = np.vstack([gp[g] for g in gnames])
    try:
        chi2, pchi, dof, exp = chi2_contingency(tab)
        chi_note = "computed"
    except ValueError as exc:  # structural zeros -> undefined expected frequencies
        chi_note = f"chi2_contingency ValueError: {exc}; retried on non-structural-zero columns"
        keep = tab.sum(axis=0) > 0
        chi2, pchi, dof, exp = chi2_contingency(tab[:, keep])
    log("\n  [INVALID reference only] chi-square treating the 52 coordinates as "
        "independent replicates:")
    log(f"    note: {chi_note}")
    log(f"    chi2={chi2:.1f} dof={dof} p={pchi:.3e}  "
        f"(expected counts <5: {(exp < 5).mean() * 100:.0f}% of cells) - "
        "inflates significance by n=52 per series; not used for any claim.")

    # ---------------- G. series-level bootstrap ----------------------------- #
    boot = []
    for g in gnames:
        cols = gcols[g]
        for pick in [(cols[0], cols[0]), (cols[0], cols[1]), (cols[1], cols[0]), (cols[1], cols[1])]:
            pooled = profiles[:, list(pick)].sum(axis=1)
            boot.append({"group": g, "resampled_series": ";".join(sid_order[k] for k in pick),
                         "com": com(coords, pooled),
                         "share_fast_s_le_-10": float(pooled[coords <= -10].sum() / pooled.sum())})
    bd = pd.DataFrame(boot)
    bd.to_csv(RESULTS / "s04_orientation_series_bootstrap.csv", index=False)
    log("\n=== G. series-level bootstrap (2 units per group -> only 4 resamples) ===")
    log(bd.groupby("group")[["com", "share_fast_s_le_-10"]].agg(["min", "max"]).to_string())
    # leave-one-channel-out (complete-unit leave-out robustness check)
    loo = []
    for g in gnames:
        for drop in gcols[g]:
            keep = [k for k in range(6) if k != drop]
            gg = np.array([g if k in gcols[g] else ("other" if k in gcols[gnames[0]] else "other2")
                           for k in keep])
            loo.append({"group": g, "dropped_series": sid_order[drop],
                        "remaining_total": float(totals[keep].sum()),
                        "group_fast_share": float(profiles[coords <= -10, gcols[g] if False else [k for k in gcols[g] if k != drop]].sum()
                                                  / profiles[:, [k for k in gcols[g] if k != drop]].sum())})
    lod = pd.DataFrame(loo)
    lod.to_csv(RESULTS / "s04_orientation_leave_one_series_out.csv", index=False)
    log("\n  leave-one-series-out fast-share per group (complete-unit leave-out):")
    log(lod.to_string(index=False))

    # ---------------- summary json ------------------------------------------ #
    summary = {
        "dimension": "directional/orientation count series and source-group conditional differences",
        "analysis_unit": "one complete source-polarization series (52 nested coordinates)",
        "series_totals": report["series_totals"],
        "structurally_zero_coords": zero_all,
        "grid_absent_coords": sorted(DROP_COORDS),
        "total_ratio_max_over_min": float(totals.max() / totals.min()),
        "workbook_mapping_exact": bool(mp["exact_match"].all()),
        "workbook_mapping": {r.workbook_sheet: r.normalized_series for r in mp.itertuples()},
        "orientation_fast_block_equals_drift_fast_total": bool(xd["equal"].all()),
        "group_stats": gd.to_dict(orient="records"),
        "group_tv_matrix": {gnames[i]: {gnames[j]: float(tvmat[i, j]) for j in range(3)}
                            for i in range(3)},
        "permutation_tests_series_level": {
            "n_units": 6,
            "n_permutations": 90,
            "p_min": 1 / 90,
            "permanova_pseudo_F": {"stat": float(obs_f), "p": p_f},
            "com_range": {"stat": obs_com_range, "p": p_com},
            "max_pairwise_tv": {"stat": obs_tv, "p": p_tv},
            "two_group": two,
        },
        "chi_square_invalid_reference": {"chi2": float(chi2), "dof": int(dof), "p": float(pchi),
                                         "reason": "coordinates are nested repeats, not replicates"},
        "independent_units": {
            "series": 6,
            "independent_sources": 3,
            "note": ("channels l/r within a group are two polarizations of the same source; "
                     "group-level claims rest on 3 independent sources and cannot be tested "
                     "with 3 groups (no residual df)"),
        },
    }
    (RESULTS / "s04_orientation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n")

    # ---------------- H. figures -------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    style = {0: "-", 1: "--"}
    for j, s in enumerate(sid_order):
        g = j // 2
        ax.plot(coords, profiles[:, j], style[j % 2], color=cmap(g), lw=1.4,
                label=f"{s} ({gnames[g]}, {'l' if j % 2 == 0 else 'r'})")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("relative coordinate s (directional count series)")
    ax.set_ylabel("event count (symlog)")
    ax.set_title("S04 - six orientation count series (unit = series, 52 nested coordinates)")
    ax.axvspan(-33, -10.5, color="grey", alpha=0.08)
    ax.axvspan(-0.5, 0.5, color="grey", alpha=0.08)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "s04_orientation_profiles.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    for g in gnames:
        ax.plot(coords, gp[g] / gp[g].sum(), "o-", ms=3, color=cmap(gnames.index(g)),
                label=g)
    ax.set_yscale("log")
    ax.set_xlabel("relative coordinate s")
    ax.set_ylabel("normalized profile (log)")
    ax.set_title("group-pooled orientation profiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    im = ax.imshow(tvmat, cmap="viridis", vmin=0)
    ax.set_xticks(range(3), gnames, rotation=20)
    ax.set_yticks(range(3), gnames)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{tvmat[i, j]:.2f}", ha="center", va="center",
                    color="w" if tvmat[i, j] < tvmat.max() * 0.6 else "k", fontsize=9)
    ax.set_title("total-variation distance between group profiles")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    labels = [f"{sid_order[j]}\n{groups[j].split('_')[-2]}-{groups[j].split('_')[-1]}"
              for j in range(6)]
    cols = [cmap(j // 2) for j in range(6)]
    ax.bar(range(6), totals, color=cols)
    for j, t in enumerate(totals):
        ax.text(j, t, f"{t:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(6), [sid_order[j] for j in range(6)], rotation=20, fontsize=7)
    ax.set_ylabel("series total events")
    ax.set_title("series totals (35x spread; left/right bar color = group)")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 1]
    ax.hist(f_perm, bins=30, color="grey", alpha=0.8)
    ax.axvline(obs_f, color="crimson", lw=2,
               label=f"observed F={obs_f:.1f}, p={p_f:.4f} (p_min=1/90)")
    ax.set_xlabel("PERMANOVA pseudo-F over 90 series-level permutations")
    ax.set_ylabel("count")
    ax.set_title("permutation null (6 series, 3 groups x 2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "s04_orientation_group_contrast.png", dpi=150)
    plt.close(fig)

    log("\n=== H. wrote figures ===")
    log(f"  {FIGS / 's04_orientation_profiles.png'}")
    log(f"  {FIGS / 's04_orientation_group_contrast.png'}")
    log("\nDONE")


if __name__ == "__main__":
    main()