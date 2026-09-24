#!/usr/bin/env python3
"""Time-domain trajectory agreement analysis.

For each condition TSV (columns: time, drive, reference_response, forecast_response)
compute error statistics between reference_response and forecast_response.
"""
import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/data/normalized_views"
OUTPUT_DIR = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project/output"
JSON_PATH = os.path.join(OUTPUT_DIR, "trajectory_results.json")

def analyze(path, label):
    df = pd.read_csv(path, sep="\t")
    ref = df["reference_response"].to_numpy(dtype=float)
    fc = df["forecast_response"].to_numpy(dtype=float)
    n = len(ref)
    abs_err = np.abs(ref - fc)
    mse = float(np.mean((ref - fc) ** 2))
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(ref, fc)[0, 1])
    ref_std = float(np.std(ref))
    nrmse_std = float(rmse / ref_std) if ref_std != 0 else float("nan")
    return {
        "n_points": int(n),
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
        "mse": mse,
        "rmse": rmse,
        "pearson_corr": corr,
        "nrmse_std": nrmse_std,
        "first_point": {
            "reference_response": float(ref[0]),
            "forecast_response": float(fc[0]),
            "identical": bool(ref[0] == fc[0]),
        },
        "last_point": {
            "reference_response": float(ref[-1]),
            "forecast_response": float(fc[-1]),
        },
        "first_time": float(df["time"].iloc[0]),
        "last_time": float(df["time"].iloc[-1]),
    }

def main():
    results = {}
    for i in (1, 2, 3):
        path = os.path.join(DATA_DIR, f"trajectory_condition_{i:02d}.tsv")
        label = f"condition_{i:02d}"
        results[label] = analyze(path, label)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
