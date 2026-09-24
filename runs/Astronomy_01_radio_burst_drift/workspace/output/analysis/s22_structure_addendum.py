#!/usr/bin/env python3
"""
Dimension tag: s22  (structure addendum: how diffuse are the thresholded masks?)

Task: TruthInsightBench Astronomy_01_radio_burst_drift (T0 = 2022-10-13)
Analysis unit: ONE COMPLETE DYNAMIC-SPECTRUM EVENT (n_events = 5).

Adds two descriptive numbers the component table alone does not make obvious:
  * mean / median number of separated vertical runs per time column
  * median span (max-min frequency) of the signal in a time column, in MHz
    and as a fraction of the 8.969 MHz band
Both are computed per event and leave-one-out aggregated.  They quantify "single
ridge vs diffuse field" in the data's own terms.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = "/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project"
NV = os.path.join(ROOT, "data", "normalized_views")
RES = os.path.join(ROOT, "output", "results")
THRESH = 1.05

rows = []
arrays = {}
for i in range(1, 6):
    df = pd.read_csv(os.path.join(NV, f"event_{i}_dynamic_spectrum.tsv"), sep="\t")
    f = df.iloc[:, 0].to_numpy(float)
    Z = df.iloc[:, 1:].to_numpy(float)
    del df
    m = Z > THRESH
    runs, spans, nsig, mass = [], [], [], 0.0
    for j in range(m.shape[1]):
        if m[:, j].any():
            _, nr = ndimage.label(m[:, j])
            runs.append(int(nr))
            nsig.append(int(m[:, j].sum()))
            fj = f[m[:, j]]
            spans.append(float(fj.max() - fj.min()))
            mass += float((Z[:, j][m[:, j]] - 1.0).sum())
    Df = float(f.max() - f.min())
    rows.append(dict(
        event_id=f"event_{i}", threshold=THRESH, n_time_cols=int(m.shape[1]),
        n_cols_with_signal=len(runs),
        mean_runs_per_col=float(np.mean(runs)), median_runs_per_col=float(np.median(runs)),
        mean_signal_bins_per_col=float(np.mean(nsig)),
        median_col_span_MHz=float(np.median(spans)),
        mean_col_span_MHz=float(np.mean(spans)),
        band_MHz=Df, median_col_span_frac_of_band=float(np.median(spans) / Df)))
    arrays[f"event_{i}"] = dict(mean_runs=np.mean(runs), med_span=np.median(spans) / Df)
    del Z
    print(f"event_{i}: {len(runs)}/425 cols with signal; runs/col mean"
          f" {np.mean(runs):.1f} median {np.median(runs):.0f}; median column span"
          f" {np.median(spans):.3f} MHz = {100*np.median(spans)/Df:.1f}% of the band;"
          f" mean signal bins/col {np.mean(nsig):.1f}")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, "s22_mask_diffuseness.csv"), index=False)

vg = R.mean_runs_per_col.to_numpy()
sp = R.median_col_span_frac_of_band.to_numpy()
loo = []
for name, v, unit in [("mean_runs_per_col", vg, "runs"), ("median_col_span_frac_of_band", sp, "fraction")]:
    full = float(np.median(v))
    folds = np.array([np.median(np.delete(v, k)) for k in range(len(v))])
    loo.append(dict(statistic=name, unit=unit, full_sample=full,
                    loo_min=float(folds.min()), loo_max=float(folds.max()),
                    loo_range=float(folds.max() - folds.min()),
                    fold_values=str([round(float(x), 6) for x in folds])))
    print(f"LOO {name}: full {full:.4f} range [{folds.min():.4f},{folds.max():.4f}]"
          f" width {folds.max()-folds.min():.4f}")
pd.DataFrame(loo).to_csv(os.path.join(RES, "s22_mask_diffuseness_loo.csv"), index=False)
with open(os.path.join(RES, "s22_mask_diffuseness.json"), "w") as fh:
    json.dump(dict(analysis_unit="one complete dynamic-spectrum event (n=5)",
                   threshold=THRESH,
                   per_event=json.loads(R.to_json(orient="records")),
                   leave_one_out=loo), fh, indent=1)
print("saved s22_mask_diffuseness.csv/.json/.csv(loo)")