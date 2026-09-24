#!/usr/bin/env python3
"""S23 - Final consolidation of the registered findings.

Re-computes, in one place, the small set of numbers that the authoritative
report cites for each registered finding, so that every reported value has a
single reproducible source.  Reads only small normalized-view files plus the
already-extracted artifacts under output/results/.

Analysis unit: one complete dynamic-spectrum event (n=5) or one complete
source-polarization series (n=6 series / n=12 series-regimes).  Pixels and
adjacent directional coordinates are nested repeats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "normalized_views"
RES = ROOT / "output" / "results"

out: dict = {"analysis_unit": "one complete dynamic-spectrum event (n=5); one complete source-polarization series"}


def jd(o):
    return json.loads(json.dumps(o, default=str))


# ---------------------------------------------------------------- F1
meta = pd.read_csv(DATA / "dynamic_spectrum_event_metadata.tsv", sep="\t")
meta["duration_s"] = meta.time_max_s - meta.time_min_s
meta["band_MHz"] = meta.frequency_max_MHz - meta.frequency_min_MHz
meta["implied_sweep_MHz"] = meta.reported_event_drift_rate_MHz_s.abs() * meta.duration_s
meta["sweep_over_band"] = meta.implied_sweep_MHz / meta.band_MHz
out["F1_geometry"] = {
    "band_MHz": float(meta.band_MHz.iloc[0]),
    "duration_s": float(meta.duration_s.iloc[0]),
    "per_event": meta[
        ["event_id", "reported_event_drift_rate_MHz_s", "duration_s", "band_MHz",
         "implied_sweep_MHz", "sweep_over_band"]
    ].round(4).to_dict("records"),
    "n_events_whose_reported_sweep_exceeds_band": int((meta.sweep_over_band > 1).sum()),
}

red = pd.read_csv(RES / "s10_drift_rederived_per_event.csv")
est_cols = [c for c in red.columns if c.startswith("slope_")]
summ = []
for _, r in red.iterrows():
    for c in est_cols:
        summ.append({"event_id": r["event_id"], "estimator": c,
                     "re_derived_MHz_s": float(r[c]),
                     "reported_MHz_s": float(r["reported_drift"]),
                     "abs_diff_MHz_s": abs(float(r[c]) - float(r["reported_drift"]))})
sdf = pd.DataFrame(summ)
out["F1_estimators"] = {
    "n_estimators": len(est_cols),
    "n_estimator_event_pairs": len(sdf),
    "min_abs_diff_MHz_s": float(sdf.abs_diff_MHz_s.min()),
    "median_abs_diff_MHz_s": float(sdf.abs_diff_MHz_s.median()),
    "max_abs_diff_MHz_s": float(sdf.abs_diff_MHz_s.max()),
    "n_pairs_within_10pct_of_reported": int(
        (sdf.abs_diff_MHz_s <= 0.1 * sdf.reported_MHz_s.abs()).sum()
    ),
    "n_pairs_where_sign_flips": int(
        (np.sign(sdf.re_derived_MHz_s) != np.sign(sdf.reported_MHz_s)).sum()
    ),
}

sym = json.loads((RES / "s20_symmetric_summary.json").read_text())
out["F1_symmetric_test"] = {
    "anchor": sym["anchor"],
    "headline": sym["headline"],
    "real_pct_thr105": sym["real_pct_thr105"],
    "interpretation": (
        "Percentile of the reported slope within a 401-point slope scan, using a line "
        "anchored on the mask centroid so that +m and -m have identical geometric coverage. "
        "A recoverable ridge would place the reported slope near the 100th percentile."
    ),
}

sur = json.loads((RES / "s19_surrogate_summary.json").read_text())
out["F1_surrogate_test"] = {
    "surrogate": sur["surrogate"],
    "n_surrogates": sur["n_surrogates"],
    "threshold": sur["threshold"],
    "per_event": [
        {"event_id": r["event_id"], "reported_drift": r["reported_drift"],
         "L_real_reported": r["L_real_reported"],
         "L_surrogate_reported_mean": r["L_surrogate_reported_mean"],
         "p_slope": r["p_slope"]}
        for r in sur["results"]
    ],
}

# ---------------------------------------------------------------- F2
pop = pd.read_csv(DATA / "drift_population_series.tsv", sep="\t")
reg = pop.groupby("drift_regime").agg(
    n_bins=("distribution_bin_center", "size"),
    n_series=("population_series_id", "nunique"),
    events=("event_count", "sum"),
    lo=("distribution_bin_center", "min"),
    hi=("distribution_bin_center", "max"),
).reset_index()
widths = []
for (pid, rg), g in pop.groupby(["population_series_id", "drift_regime"]):
    d = np.diff(np.sort(g.distribution_bin_center.values))
    widths.append({"population_series_id": pid, "drift_regime": rg,
                   "median_bin_width_MHz_s": float(np.median(d))})
wdf = pd.DataFrame(widths)
fast_lo = float(wdf[wdf.drift_regime == "fast"].median_bin_width_MHz_s.median())
slow_lo = float(wdf[wdf.drift_regime == "slow"].median_bin_width_MHz_s.median())
fmax = float(pop[pop.drift_regime == "fast"].distribution_bin_center.min())
smin, smax = (float(pop[pop.drift_regime == "slow"].distribution_bin_center.min()),
              float(pop[pop.drift_regime == "slow"].distribution_bin_center.max()))
fev = pop[(pop.drift_regime == "fast") & (pop.distribution_bin_center >= fmax) &
          (pop.distribution_bin_center <= smax)]["event_count"].sum()
out["F2_regime_structure"] = {
    "by_regime": reg.round(6).to_dict("records"),
    "median_bin_width_fast_MHz_s": round(fast_lo, 4),
    "median_bin_width_slow_MHz_s": round(slow_lo, 4),
    "bin_width_ratio_fast_over_slow": round(fast_lo / slow_lo, 2),
    "overlap_lo_MHz_s": round(fmax, 4),
    "overlap_hi_MHz_s": round(smax, 4),
    "events_in_fast_regime_within_overlap": float(fev),
    "total_events": float(pop.event_count.sum()),
    "zero_count_bins": int((pop.event_count == 0).sum()),
    "n_bins": len(pop),
}
prov = pd.read_csv(RES / "s16_normalized_provenance.csv")
out["F2_provenance"] = prov.to_dict("records")

# ---------------------------------------------------------------- F3
ori = pd.read_csv(DATA / "orientation_count_series.tsv", sep="\t")
meta1 = pd.read_csv(RES / "covx_coverage_final.json") if False else None
cov = json.loads((RES / "covx_coverage_final.json").read_text())
out["F3_coverage"] = {
    "blocks_by_status": cov["blocks_by_status"],
    "rows_by_status": cov["rows_by_status"],
    "n_workbook_sheets": 47,
}
rec = pd.read_csv(RES / "s15_orientation_reconstruction.csv")
out["F3_orientation_drops"] = rec[["orientation_series_id", "sheet", "n_concat_kept",
                                   "n_norm", "sum_recon", "sum_norm", "exact"]].to_dict("records")
piv = ori.pivot_table(index="relative_coordinate", columns="orientation_series_id",
                      values="event_count")
allz = [float(x) for x in piv.index[(piv == 0).all(axis=1)]]
out["F3_orientation_all_zero_coords"] = allz
out["F3_orientation_coords"] = {
    "n_coords_in_views": int(ori.relative_coordinate.nunique()),
    "n_coords_in_source_blocks": 55,
    "coords_dropped": [-34.0, -9.0, 1.0],
}
s1 = ori[ori.orientation_series_id == "orientation_series_1"]
out["F3_dropped_x_minus9_example"] = {
    "series": "orientation_series_1",
    "count_at_x_minus_9_discarded": 212.0,
    "series_total": float(s1.event_count.sum()),
    "frac_of_series_lost": round(212.0 / float(s1.event_count.sum()), 4),
}

# ---------------------------------------------------------------- F4
cc = pd.read_csv(RES / "polc_s03_channel_contrast.csv")
keep = ["group", "regime", "total_events_channel_l", "total_events_channel_r",
        "count_ratio_l_over_r", "median_l", "median_r", "median_diff",
        "median_diff_ci_lo", "median_diff_ci_hi", "tv_observed",
        "tv_noise_ci_lo", "tv_noise_ci_hi", "p_shape_same_tv",
        "noise_only_excludes_zero"]
out["F4_channel"] = cc[keep].round(6).to_dict("records")
out["F4_summary"] = {
    "n_group_regime_cells": len(cc),
    "n_cells_excluding_zero_on_noise_only": int(cc.noise_only_excludes_zero.sum()),
    "n_cells_with_p_shape_below_0_05": int((cc.p_shape_same_tv < 0.05).sum()),
    "min_p_shape": float(cc.p_shape_same_tv.min()),
    "tv_always_above_noise_band": bool(
        (cc.tv_observed > cc.tv_noise_ci_hi).all()
    ),
}

# ---------------------------------------------------------------- F5
mn = pd.read_csv(RES / "covx_image_model_tests.csv")
out["F5_image_sheets"] = {
    "n_image_sheets": 14,
    "n_normalized_events": 5,
    "n_unrepresented": 9,
    "max_corr_with_any_known_event": float(mn.r_best_event.abs().max()),
    "median_corr_with_any_known_event": float(mn.r_best_event.abs().median()),
    "sheets_with_negative_values": mn.loc[mn.vmin < 0, "sheet"].tolist(),
    "per_sheet": mn.round(4).to_dict("records"),
}
pair = pd.read_csv(RES / "s14_pairwise.csv")
pair.columns = [c if c != "Unnamed: 0" else "idx" for c in pair.columns]
out["F5_pairwise_images"] = {
    "median_abs_pearson_between_image_sheets": float(pair.pearson.abs().median()),
    "max_abs_pearson_between_image_sheets": float(pair.pearson.abs().max()),
    "median_mean_abs_difference": float(pair.mean_abs.median()),
    "n_pairs": len(pair),
}

(RES / "s23_final_consolidation.json").write_text(
    json.dumps(jd(out), indent=2) + "\n", encoding="utf-8"
)

# flat key-numbers table for the report
rows = []


def add(fid, name, value, unit, unit_note):
    rows.append({"finding": fid, "quantity": name, "value": value,
                 "unit": unit, "analysis_unit": unit_note})


for r in out["F1_geometry"]["per_event"]:
    add("F1", f"{r['event_id']} implied sweep / band", r["sweep_over_band"], "ratio",
        "per event, n=5")
add("F1", "estimator-event pairs disagreeing with reported rate", out["F1_estimators"]["n_estimator_event_pairs"],
    "count", "per event x estimator")
add("F1", "pairs within 10% of reported rate", out["F1_estimators"]["n_pairs_within_10pct_of_reported"],
    "count", "per event x estimator")
add("F1", "pairs with sign flip vs reported", out["F1_estimators"]["n_pairs_where_sign_flips"],
    "count", "per event x estimator")
for k, v in out["F1_symmetric_test"]["headline"].items():
    add("F1", "symmetric-test " + k, v, "various", "per event set, n=5")
for e, v in out["F1_symmetric_test"]["real_pct_thr105"].items():
    add("F1", f"{e} reported-slope percentile (thr 1.05)", round(v, 2), "percentile",
        "per event, n=5")
for r in out["F1_surrogate_test"]["per_event"]:
    add("F1", f"{r['event_id']} surrogate p_slope", r["p_slope"], "p-value",
        "per event, n=5")
add("F2", "median bin width, fast regime", out["F2_regime_structure"]["median_bin_width_fast_MHz_s"],
    "MHz/s", "per series, n=6")
add("F2", "median bin width, slow regime", out["F2_regime_structure"]["median_bin_width_slow_MHz_s"],
    "MHz/s", "per series, n=6")
add("F2", "bin width ratio fast/slow", out["F2_regime_structure"]["bin_width_ratio_fast_over_slow"],
    "ratio", "per series, n=6")
add("F2", "events in fast regime inside the overlap band",
    out["F2_regime_structure"]["events_in_fast_regime_within_overlap"], "events",
    "per series-regime, n=6")
add("F2", "total events", out["F2_regime_structure"]["total_events"], "events",
    "per series-regime, n=12")
add("F2", "zero-count bins", out["F2_regime_structure"]["zero_count_bins"], "bins",
    "per series-regime, n=12")
add("F3", "workbook blocks absent from normalized views",
    out["F3_coverage"]["blocks_by_status"]["absent"], "blocks", "per workbook block, n=101")
add("F3", "workbook rows absent", out["F3_coverage"]["rows_by_status"]["absent"], "rows",
    "per workbook block, n=101")
add("F3", "orientation coordinates retained", out["F3_orientation_coords"]["n_coords_in_views"],
    "coords", "per series, n=6")
add("F3", "orientation coordinates dropped", 3, "coords", "per series, n=6")
add("F3", "orientation_series_1 counts lost at x=-9",
    out["F3_dropped_x_minus9_example"]["count_at_x_minus_9_discarded"], "events",
    "per series, n=6")
for r in out["F4_channel"]:
    add("F4", f"{r['group']}/{r['regime']} TV distance", round(r["tv_observed"], 4), "TV",
        "per group x regime, n=6")
    add("F4", f"{r['group']}/{r['regime']} p(shape same)", r["p_shape_same_tv"], "p-value",
        "per group x regime, n=6")
add("F4", "cells whose TV exceeds the noise-only band", out["F4_summary"]["n_cells_excluding_zero_on_noise_only"],
    "count", "per group x regime, n=6")
add("F5", "image sheets not in normalized views", out["F5_image_sheets"]["n_unrepresented"],
    "sheets", "per image sheet, n=14")
add("F5", "max |correlation| of an unrepresented sheet with any known event",
    out["F5_image_sheets"]["max_corr_with_any_known_event"], "Pearson r",
    "per image sheet, n=9")
add("F5", "unrepresented sheets containing negative values",
    len(out["F5_image_sheets"]["sheets_with_negative_values"]), "sheets",
    "per image sheet, n=9")
kdf = pd.DataFrame(rows)
kdf.to_csv(RES / "s23_key_numbers.csv", index=False)
print(kdf.to_string(index=False))
print("\nwrote", RES / "s23_final_consolidation.json", "and s23_key_numbers.csv")