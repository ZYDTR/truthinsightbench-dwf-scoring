#!/usr/bin/env python3
"""F3: null test on homophily.

Pearson r (plus t-statistic, two-sided p via normal CDF, n) of
mean_homophily against each of the 9 node attribute columns, using
network_node_summary.tsv joined to network_profile_a.tsv on region_index.
Analysis unit: 4 systems = region_index // 200 (200 nodes each, verified).
Only pandas/numpy (no scipy).
"""
import json
import math

import numpy as np
import pandas as pd

DATA_NODE = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data/network_node_summary.tsv"
DATA_PROFILE = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data/network_profile_a.tsv"
OUT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output/f3_homophily_null_result.json"

COLUMNS = [
    "node_strength",
    "mean_connection_distance",
    "coord_x",
    "coord_y",
    "coord_z",
    "annotation_1",
    "annotation_2",
    "annotation_3",
    "annotation_4",
    "annotation_5",
]


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(x))
    dx, dy = x - x.mean(), y - y.mean()
    dxx, dyy = float(np.dot(dx, dx)), float(np.dot(dy, dy))
    if dxx == 0.0 or dyy == 0.0:
        return 0.0, 0.0, 1.0, n
    r = float(np.dot(dx, dy) / (math.sqrt(dxx) * math.sqrt(dyy)))
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0 - 1e-15:
        r = 1.0 if r >= 0.0 else -1.0
        t = math.inf if r > 0 else -math.inf
    else:
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
    # two-sided p via normal CDF: p = 2 * P(Z > |t|) = 1 - erf(|t|/sqrt(2))
    p = 1.0 - math.erf(abs(t) / math.sqrt(2.0))
    if not math.isfinite(p):
        p = 0.0
    return float(r), float(t), float(p), n


def main():
    a = pd.read_csv(DATA_NODE, sep="\t")
    b = pd.read_csv(DATA_PROFILE, sep="\t")
    df = a.merge(b, on="region_index", how="inner")
    assert df["region_index"].is_unique, "region_index not unique after join"
    assert len(df) == 800, "expected 800 nodes after join"
    df["system"] = df["region_index"] // 200
    assert set(df["system"].unique()) == {0, 1, 2, 3}, "expected 4 systems"
    assert (df.groupby("system").size() == 200).all(), "expected 200 nodes per system"
    df = df.sort_values("region_index").reset_index(drop=True)

    per_col = {}
    for c in COLUMNS:
        r, t, p, n = pearson(df["mean_homophily"], df[c])
        per_col[c] = {"r": r, "t": t, "p": p, "n": n}

    max_abs_r = max(abs(v["r"]) for v in per_col.values())
    largest_r_col = max(per_col, key=lambda c: abs(per_col[c]["r"]))
    all_null_lt_0_15 = all(abs(v["r"]) < 0.15 for v in per_col.values())

    result = {
        "per_col": per_col,
        "max_abs_r": max_abs_r,
        "all_null_lt_0.15": bool(all_null_lt_0_15),
        "n_nodes": int(len(df)),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
