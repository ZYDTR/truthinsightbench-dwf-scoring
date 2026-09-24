#!/usr/bin/env python3
"""Frequency-domain agreement analysis between reference and forecast responses."""
import json
import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/data/normalized_views"
OUT = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/output"

FILES = {
    "control_a": f"{DATA}/control_a_frequency_response.tsv",
    "control_b": f"{DATA}/control_b_frequency_response.tsv",
    "signed_control": f"{DATA}/signed_control_frequency_response.tsv",
}

results = {}
for key, path in FILES.items():
    df = pd.read_csv(path, sep="\t")
    ref = df["reference_frequency"].to_numpy(dtype=float)
    fc = df["forecast_frequency"].to_numpy(dtype=float)
    diff = fc - ref
    abs_err = np.abs(diff)

    exact_mask = np.abs(ref - fc) < 1e-6
    differing = []
    for idx in df.index[~exact_mask]:
        differing.append({
            "index": int(idx),
            "reference": float(df.loc[idx, "reference_frequency"]),
            "forecast": float(df.loc[idx, "forecast_frequency"]),
        })

    results[key] = {
        "n_pairs": int(len(df)),
        "exact_match_count": int(int(exact_mask.sum())),
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mean_relative_error": float(np.mean(np.abs(diff) / np.abs(ref))),
        "bias": float(np.mean(diff)),
        "differing_rows": differing,
    }

out_path = f"{OUT}/frequency_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
