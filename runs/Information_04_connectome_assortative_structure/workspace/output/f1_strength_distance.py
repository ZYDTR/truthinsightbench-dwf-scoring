#!/usr/bin/env python3
"""F1: strength-vs-distance association and per-system robustness.

Only pandas/numpy (no scipy). Computes overall Pearson r between
node_strength and mean_connection_distance, t-stat, two-sided p via
normal CDF, per-system (region_index // 200) r values, and a
robustness check (all 4 per-system r same sign and > 0.5).
"""
import json
import math

import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data/network_node_summary.tsv"
OUT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output/f1_strength_distance_result.json"


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = math.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    r = np.dot(dx, dy) / denom if denom != 0 else 0.0
    if np.isclose(r, 1.0):
        r = 1.0
    elif np.isclose(r, -1.0):
        r = -1.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # two-sided p via normal CDF: p = 2 * P(Z > |t|) = 1 - erf(|t|/sqrt(2))
    p = 1.0 - math.erf(abs(t) / math.sqrt(2.0))
    if p < 1e-16:
        p = 0.0
    return float(r), float(t), float(p), int(n)


def main():
    df = pd.read_csv(DATA, sep="\t")
    df["system"] = df["region_index"] // 200

    r, t, p, n = pearson(df["node_strength"], df["mean_connection_distance"])

    per_system_r = {}
    for s in range(4):
        sub = df[df["system"] == s]
        rs, _, _, _ = pearson(sub["node_strength"], sub["mean_connection_distance"])
        per_system_r[int(s)] = rs

    robust = all(rs > 0.5 for rs in per_system_r.values()) and len({
        (rs > 0) for rs in per_system_r.values()}) == 1

    result = {
        "overall": {"r": r, "t": t, "p": p, "n": n},
        "per_system_r": per_system_r,
        "robust_same_direction": bool(robust),
        "n_nodes": int(len(df)),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
