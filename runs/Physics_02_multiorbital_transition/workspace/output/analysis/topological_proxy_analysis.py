#!/usr/bin/env python3
"""Topological-proxy analysis: entanglement-spectrum ground-state degeneracy
and gap-to-second-level versus control_ratio (U/W).

Data provenance:
  WS/data/normalized_views/control_vs_topological_proxy.tsv
  TSV columns: control_ratio (U/W), spectrum_value, degeneracy
  48 controls (U/W = 0.1..8.0 step 0.1), 50 states each (2400 rows).
"""
import json
import os
import sys

import pandas as pd

WS = "/data/user/conversations/conv-af56a8cc-3c22-4b90-8a53-8142e2d7b40e/project"
TSL_PATH = os.path.join(
    WS, "data", "normalized_views", "control_vs_topological_proxy.tsv"
)
OUT_CSV = os.path.join(
    WS, "output", "results", "topological_proxy_per_control.csv"
)
OUT_JSON = os.path.join(
    WS, "output", "results", "topological_proxy_summary.json"
)
EPS = 1e-9  # degeneracy tolerance for "within low_value"

rng = {"0.1": 0.1, "1.0": 1.0, "2.0": 2.0, "4.0": 4.0, "8.0": 8.0}


def main():
    df = pd.read_csv(TSL_PATH, sep="\t")
    df["control_ratio"] = df["control_ratio"].astype(float)
    df["spectrum_value"] = df["spectrum_value"].astype(float)
    df["degeneracy"] = df["degeneracy"].astype(int)
    df = df.sort_values(["control_ratio", "spectrum_value"]).reset_index(drop=True)

    rows = []
    for u, g in df.groupby("control_ratio", sort=True):
        low_value = g["spectrum_value"].min()
        low_mask = (g["spectrum_value"] - low_value).abs() <= EPS
        low_deg = int(g.loc[low_mask, "degeneracy"].sum())
        distinct = sorted(g["spectrum_value"].unique())
        if len(distinct) >= 2:
            gap_to_2nd = float(distinct[1] - low_value)
        else:
            gap_to_2nd = float("nan")
        rows.append(
            {
                "control_ratio": u,
                "low_value": float(low_value),
                "low_deg": low_deg,
                "gap_to_2nd": gap_to_2nd,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    deg1 = out.loc[out["low_deg"] == 1, "control_ratio"].tolist()
    deg2 = out.loc[out["low_deg"] == 2, "control_ratio"].tolist()
    first_deg2 = (
        float(out.loc[out["low_deg"] == 2, "control_ratio"].min())
        if not (out["low_deg"] == 2).any()
        else None
    )

    below_1e3 = out[out["gap_to_2nd"] < 1e-3]
    below_1e2 = out[out["gap_to_2nd"] < 1e-2]

    low_values = {}
    for key, u in rng.items():
        v = out.loc[out["control_ratio"] == u, "low_value"]
        low_values["u_" + key] = (float(v.iloc[0]) if len(v) else None)

    summary = {
        "n_controls": int(len(out)),
        "deg_1_controls": deg1,
        "deg_2_controls": deg2,
        "first_deg2_control": first_deg2,
        "gap_below_1e3_u_min": (float(below_1e3["control_ratio"].min())
                                if len(below_1e3) else None),
        "gap_below_1e3_u_max": (float(below_1e3["control_ratio"].max())
                                if len(below_1e3) else None),
        "low_values": low_values,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)

    # console report
    print("per-control (head/tail):")
    print(out.head(3).to_string(index=False))
    print("...")
    print(out.tail(3).to_string(index=False))
    print(f"\nn_controls={len(out)}")
    print(f"deg==1 controls: {len(deg1)}  -> first={deg1[0] if deg1 else None}")
    print(f"deg==2 controls: {len(deg2)}  -> first={deg2[0] if deg2 else None}")
    print(f"first control with low_deg==2: {first_deg2}")
    if len(below_1e3):
        print(
            f"gap_to_2nd < 1e-3: U/W in [{below_1e3['control_ratio'].min()}, "
            f"{below_1e3['control_ratio'].max()}] "
            f"({len(below_1e3)} controls)"
        )
    else:
        print("gap_to_2nd < 1e-3: no controls")
    if len(below_1e2):
        print(
            f"gap_to_2nd < 1e-2: U/W in [{below_1e2['control_ratio'].min()}, "
            f"{below_1e2['control_ratio'].max()}] "
            f"({len(below_1e2)} controls)"
        )
    else:
        print("gap_to_2nd < 1e-2: no controls")
    print("low_value at U/W = 0.1, 1.0, 2.0, 4.0, 8.0:")
    for u in (0.1, 1.0, 2.0, 4.0, 8.0):
        v = out.loc[out["control_ratio"] == u, "low_value"].iloc[0]
        print(f"  U/W={u:.1f}: low_value={v:.6f}")
    print(f"\nWrote: {OUT_CSV}\nWrote: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
