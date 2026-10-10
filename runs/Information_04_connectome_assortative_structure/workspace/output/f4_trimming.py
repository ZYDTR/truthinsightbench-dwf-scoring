#!/usr/bin/env python3
"""Trimming-sensitivity summary (pandas only)."""
import json
import pandas as pd

DATA = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/data/trimming_sensitivity.tsv"
OUT = "/data/user/conversations/conv-88510242-47d4-46dc-a7c4-b833ab46f409/project/output/f4_trimming_result.json"

trim_cols = [f"trim_percent_{i}" for i in range(1, 20)]

df = pd.read_csv(DATA, sep="\t")

# Group by (system_index, annotation_index); each pair has two result_kind rows.
pairs = {}
for (sys_id, ann_id), g in df.groupby(["system_index", "annotation_index"]):
    key = f"{sys_id}/{ann_id}"
    z = g[g["result_kind"] == "assort_z"][trim_cols].values[0]
    p = g[g["result_kind"] == "assort_p_fdr"][trim_cols].values[0]

    pairs[key] = {
        "assort_z": {
            "tp1": float(z[0]),
            "tp_peak": float(z.max()),
            "tp_last": float(z[-1]),
            "declines_with_trim": bool(z[0] > z[-1]),
        },
        "assort_p_fdr": {
            "min_p": float(p.min()),
            "n_sig": int((p < 0.05).sum()),
            "ever_sig": bool((p < 0.05).any()),
        },
    }

all_decline = all(v["assort_z"]["declines_with_trim"] for v in pairs.values())
never_sig = [k for k, v in pairs.items() if not v["assort_p_fdr"]["ever_sig"]]

result = {
    "n_pairs": len(pairs),
    "pairs": pairs,
    "all_assort_z_decline": all_decline,
    "never_significant_pairs": never_sig,
    "n_rows": int(len(df)),
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
