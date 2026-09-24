#!/usr/bin/env python3
"""
Dimension tag: s21  (consolidated leave-one-out table for the whole dimension)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).

Reads the s17/s18/s19/s20 result tables and emits ONE leave-one-out table with
min / max / range for every event-level summary statistic computed anywhere in
this dimension, plus explicit flags for sign flips and decision-boundary
crossings.  Nothing is recomputed from pixels here; this is pure aggregation of
already-written machine-readable results.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
RES = os.path.join(ROOT, "output", "results")
OUT = []


def loo_of(name, vals, boundary=None, unit="", mode="median", note=""):
    v = np.asarray(vals, dtype=float)
    red = {"median": np.nanmedian, "count": np.nansum, "fraction": np.nanmean}[mode]
    if v.size == 0 or not np.isfinite(v).any():
        return None
    full = float(red(v))
    folds = np.array([red(np.delete(v, k)) for k in range(len(v))], float)
    rng = float(folds.max() - folds.min())
    sign_flip = bool(np.any(np.sign(folds) != np.sign(full)) and full != 0)
    cross = bool(np.any((folds - boundary) * (full - boundary) < 0)) if boundary is not None else False
    return dict(source=name, statistic=name, unit=unit, aggregation=mode,
                n_events=len(v), full_sample=full, loo_min=float(folds.min()),
                loo_max=float(folds.max()), loo_range=rng,
                range_over_full=(float(rng / abs(full)) if full != 0 else np.inf),
                sign_flip=sign_flip, crosses_boundary=cross,
                flagged=bool(sign_flip or cross),
                decision_boundary=boundary,
                fold_values=str([round(float(x), 6) for x in folds]), note=note)


def main():
    # ---- s17: threshold sweep / structure / displacement / injection ---------
    R = pd.read_csv(os.path.join(RES, "s17_threshold_rates.csv"))
    C = pd.read_csv(os.path.join(RES, "s17_mask_components.csv"))
    INJ = pd.read_csv(os.path.join(RES, "s17_injection_control.csv"))
    orphans = []
    for lab in R.threshold.unique():
        g = R[R.threshold == lab].set_index("event_id")
        gc_ = C[C.threshold == lab].set_index("event_id")
        rows = [
            loo_of(f"s17[{lab}] fitted_rate_theilsen", g["rate_theilsen"], unit="MHz/s",
                   note="weighted-centroid Theil-Sen, all time columns"),
            loo_of(f"s17[{lab}] abs_bias_vs_reported", (g["rate_theilsen"] - g["reported_drift"]).abs(),
                   unit="MHz/s"),
            loo_of(f"s17[{lab}] mask_fraction", g["frac_grid_mask"], unit="fraction"),
            loo_of(f"s17[{lab}] n_components_8conn", gc_["n_components"], unit="count"),
            loo_of(f"s17[{lab}] frac_multirun_cols", gc_["frac_multirun_cols"], unit="fraction"),
            loo_of(f"s17[{lab}] pca_slope", g["pca_slope"], boundary=0.0, unit="MHz/s"),
            loo_of(f"s17[{lab}] frac_events_identical_sign",
                   (np.sign(g["rate_theilsen"]) == np.sign(g["reported_drift"])).astype(float),
                   boundary=0.5, unit="fraction", mode="fraction"),
        ]
        orphans += [r for r in rows if r]
    for amp in sorted(INJ.injected_amplitude.unique()):
        s = INJ[INJ.injected_amplitude == amp]
        orphans.append(loo_of(f"s17[inj amp {amp}] median_pct_of_reported",
                              s["pct_of_reported"], boundary=90.0, unit="pct"))
    # displacement ratio per event (from metadata + measured centroid range)
    meta = pd.read_csv(os.path.join(ROOT, "data", "normalized_views",
                                    "dynamic_spectrum_event_metadata.tsv"), sep="\t")
    Df = float(meta.frequency_max_MHz.iloc[0] - meta.frequency_min_MHz.iloc[0])
    disp = np.abs(meta.reported_event_drift_rate_MHz_s) * (meta.time_max_s - meta.time_min_s)
    orphans.append(loo_of("s17 implied_displacement/band", disp / Df, boundary=1.0,
                          unit="fraction", note="reported |m|*Dt / 8.969 MHz"))
    orphans.append(loo_of("s17 implied_displacement_MHz", disp, unit="MHz"))

    # ---- s18 and s20 already carry their own LOO tables ---------------------
    s18 = pd.read_csv(os.path.join(RES, "s18_leave_one_out.csv"))
    s20 = pd.read_csv(os.path.join(RES, "s20_symmetric_leave_one_out.csv"))

    # ---- s19 surrogate nulls -----------------------------------------------
    s19 = pd.read_csv(os.path.join(RES, "s19_surrogate_null.csv"))
    orphans.append(loo_of("s19 n_events_reject_p_slope_0.05",
                          (s19.p_slope <= 0.05).astype(float), boundary=2.5,
                          unit="count/5", mode="count",
                          note="ridge-free row-rolled surrogate, L(reported)"))
    orphans.append(loo_of("s19 n_events_reject_p_any_0.05",
                          (s19.p_any <= 0.05).astype(float), boundary=2.5,
                          unit="count/5", mode="count",
                          note="any coherent straight ridge, all slopes"))
    orphans.append(loo_of("s19 n_events_reject_p_len_0.05",
                          (s19.p_len <= 0.05).astype(float), boundary=2.5,
                          unit="count/5", mode="count"))
    orphans.append(loo_of("s19 median_p_slope", s19.p_slope, boundary=0.05, unit="p"))

    A = pd.DataFrame([r for r in orphans if r])
    A.insert(0, "dimension", "s17/s19-consolidated")
    s18b = s18.copy(); s18b.insert(0, "dimension", "s18-decisive")
    s20b = s20.copy(); s20b.insert(0, "dimension", "s20-symmetric")
    # unify the columns
    cols = ["dimension", "statistic", "unit", "aggregation", "full_sample",
            "loo_min", "loo_max", "loo_range", "crosses_boundary"]
    for df in (A, s18b, s20b):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
    ALL = pd.concat([A[cols], s18b[cols], s20b[cols]], ignore_index=True)
    ALL.to_csv(os.path.join(RES, "s21_consolidated_leave_one_out.csv"), index=False)
    with open(os.path.join(RES, "s21_consolidated_leave_one_out.json"), "w") as fh:
        json.dump(json.loads(ALL.to_json(orient="records")), fh, indent=1)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    print(ALL.to_string(index=False))
    print()
    print("FLAGGED (sign flip or decision-boundary crossing):")
    print(ALL[ALL.crosses_boundary == True].to_string(index=False))  # noqa: E712
    print()
    print("n statistics:", len(ALL), " flagged:", int(ALL.crosses_boundary.sum()))
    print("saved s21_consolidated_leave_one_out.csv/.json")


if __name__ == "__main__":
    main()