#!/usr/bin/env python3
"""Step 01 - Structural profiling of the four anonymous observation panels.

Goal: identify identifier columns, design structure (factorial / series),
duplicate columns, constant or empty columns, and the analysis unit.
No scientific modelling happens here; this fixes the data inventory that all
later steps rely on.

Outputs: evidence/01_profile.json, evidence/01_column_profile.csv
"""
from __future__ import annotations

import json
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

SETS = [1, 2, 3, 4]


def load(i: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"observation_set_{i}.tsv", sep="\t")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


report: dict = {"files": {}, "panels": {}}

for i in SETS:
    p = DATA / f"observation_set_{i}.tsv"
    report["files"][p.name] = {
        "bytes": p.stat().st_size,
        "sha256": sha256(p),
    }

# ---------------------------------------------------------------- column rows
rows = []
for i in SETS:
    df = load(i)
    n = len(df)
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        rows.append(
            {
                "set": i,
                "column": c,
                "n": n,
                "n_missing": int(s.isna().sum()),
                "n_unique": int(nn.nunique()),
                "min": float(nn.min()) if len(nn) else np.nan,
                "max": float(nn.max()) if len(nn) else np.nan,
                "mean": float(nn.mean()) if len(nn) else np.nan,
                "std": float(nn.std(ddof=1)) if len(nn) > 1 else np.nan,
                "is_integer_valued": bool(len(nn) and np.allclose(nn % 1, 0)),
                "is_constant": bool(len(nn) and nn.nunique() == 1),
                "is_empty": bool(len(nn) == 0),
            }
        )
colprof = pd.DataFrame(rows)
colprof.to_csv(EV / "01_column_profile.csv", index=False)

for i in SETS:
    df = load(i)
    info: dict = {"n_rows": len(df), "n_cols": df.shape[1]}

    # ---- identifier detection: a column that is a bijection with row order
    ident = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == len(df) and s.nunique() == len(df):
            if s.is_monotonic_increasing or s.is_monotonic_decreasing:
                ident.append(c)
    info["row_identifier_columns"] = ident

    # ---- exact duplicate / affine-duplicate column pairs
    dup = []
    cols = [c for c in df.columns]
    for a, b in itertools.combinations(cols, 2):
        sa, sb = df[a], df[b]
        mask = sa.notna() & sb.notna()
        if mask.sum() < 2:
            continue
        if mask.sum() == len(df) and np.allclose(sa.to_numpy(float), sb.to_numpy(float)):
            dup.append({"a": a, "b": b, "kind": "exact", "n_overlap": int(mask.sum())})
            continue
        va, vb = sa[mask].to_numpy(float), sb[mask].to_numpy(float)
        if np.allclose(va, vb):
            dup.append({"a": a, "b": b, "kind": "exact_on_overlap", "n_overlap": int(mask.sum())})
            continue
        if va.std() > 0 and vb.std() > 0:
            r = float(np.corrcoef(va, vb)[0, 1])
            if abs(r) > 0.999999:
                dup.append(
                    {
                        "a": a,
                        "b": b,
                        "kind": "affine",
                        "n_overlap": int(mask.sum()),
                        "r": round(r, 8),
                        "slope": float(np.polyfit(va, vb, 1)[0]),
                        "intercept": float(np.polyfit(va, vb, 1)[1]),
                    }
                )
    info["duplicate_column_pairs"] = dup

    # ---- missingness blocks: which rows are complete
    feat = [c for c in df.columns if c != "observation_index"]
    complete = df[feat].notna().all(axis=1)
    info["rows_complete_all_features"] = int(complete.sum())
    info["complete_row_ranges"] = []
    if complete.any():
        idx = np.flatnonzero(complete.to_numpy())
        start = idx[0]
        prev = idx[0]
        for v in idx[1:]:
            if v != prev + 1:
                info["complete_row_ranges"].append([int(start), int(prev)])
                start = v
            prev = v
        info["complete_row_ranges"].append([int(start), int(prev)])

    # ---- which columns share identical missingness pattern
    pat: dict[str, list[str]] = {}
    for c in feat:
        key = "".join("1" if v else "0" for v in df[c].notna().to_numpy())
        pat.setdefault(key, []).append(c)
    info["missingness_pattern_groups"] = [v for v in pat.values() if len(v) > 1]
    info["n_distinct_missingness_patterns"] = len(pat)

    report["panels"][f"set_{i}"] = info

# -------------------------------------------------- set 1 factorial structure
d1 = load(1)
f = {
    "numeric_feature_2": int(d1["numeric_feature_2"].nunique()),
    "numeric_feature_3": int(d1["numeric_feature_3"].nunique()),
    "numeric_feature_4": int(d1["numeric_feature_4"].nunique()),
}
grid = d1.groupby(["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]).size()
report["set_1_design"] = {
    "factor_levels": f,
    "product_of_levels": int(np.prod(list(f.values()))),
    "n_rows": int(len(d1)),
    "is_complete_factorial": bool(
        len(grid) == int(np.prod(list(f.values()))) and (grid == 1).all()
    ),
    "cells": int(len(grid)),
    "cell_count_min": int(grid.min()),
    "cell_count_max": int(grid.max()),
    "factor_level_values": {
        c: sorted(int(v) for v in d1[c].unique()) for c in f
    },
    "feature_2_formula": "checking f2 == row index",
    "f1_equals_f5_plus_1": bool(
        np.array_equal(d1["numeric_feature_1"].to_numpy(), d1["numeric_feature_5"].to_numpy() + 1)
    ),
}

print(json.dumps(report["set_1_design"], indent=2))
print()

# ------------------------------------------------------- set 2 / 3 / 4 series
for i in (2, 3, 4):
    df = load(i)
    feat = [c for c in df.columns if c != "observation_index"]
    sub = df[[c for c in feat if df[c].notna().sum() >= 2]]
    if len(sub.columns) >= 2:
        cc = sub.corr(numeric_only=True)
        report["panels"][f"set_{i}"]["pearson_corr_matrix"] = json.loads(
            cc.round(6).to_json()
        )

with (EV / "01_profile.json").open("w") as fh:
    json.dump(report, fh, indent=2, default=str)

# --------------------------------------------------------------- console view
for i in SETS:
    info = report["panels"][f"set_{i}"]
    print("=" * 78)
    print(f"SET {i}: rows={info['n_rows']} cols={info['n_cols']}")
    print("  row-id columns     :", info["row_identifier_columns"])
    print("  complete-all rows  :", info["rows_complete_all_features"],
          "ranges:", info["complete_row_ranges"][:6])
    print("  missingness groups :", info["missingness_pattern_groups"])
    for d in info["duplicate_column_pairs"]:
        print("  duplicate pair     :", d)