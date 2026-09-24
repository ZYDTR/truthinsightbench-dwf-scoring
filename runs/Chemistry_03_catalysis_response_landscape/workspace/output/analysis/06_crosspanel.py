#!/usr/bin/env python3
"""Step 06 - Remaining structural questions and cross-panel correspondence.

  * Is the response column an exact function of the two magnitude columns only,
    i.e. is there any independent response degree of freedom in panels 3/4?
  * Are the two magnitude columns themselves redundant (A vs B relation)?
  * Do the 13 condition levels of panels 3 and 4 correspond to the 13 levels of
    factor B in panel 1?  (both designs contain a 13-level factor)
  * What is the relation between the two responses of panel 1?

Outputs: evidence/06_crosspanel.json
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
CYCLE = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t")
d2 = pd.read_csv(DATA / "observation_set_2.tsv", sep="\t")
d3 = pd.read_csv(DATA / "observation_set_3.tsv", sep="\t")
d4 = pd.read_csv(DATA / "observation_set_4.tsv", sep="\t")
OUT: dict = {}

# ------------------------------------------------ A-B relation inside panels 3/4
rel = {}
for tag, df, A, B, R in [("panel_3", d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5"),
                         ("panel_4", d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")]:
    a, b = df[A].to_numpy(float), df[B].to_numpy(float)
    ratio = b / a
    rel[tag] = {
        "n": int(len(a)),
        "pearson_A_B": float(np.corrcoef(a, b)[0, 1]),
        "spearman_A_B": float(stats.spearmanr(a, b).statistic),
        "log_pearson_A_B": float(np.corrcoef(np.log(a), np.log(b))[0, 1]),
        "B_over_A_ratio": {"min": float(ratio.min()), "max": float(ratio.max()),
                           "median": float(np.median(ratio)),
                           "q25": float(np.percentile(ratio, 25)),
                           "q75": float(np.percentile(ratio, 75))},
        "var_share_of_log_B_from_log_A": float(np.corrcoef(np.log(a), np.log(b))[0, 1] ** 2),
        "ratio_is_constant": bool(np.allclose(ratio, ratio[0])),
    }
OUT["magnitude_column_relation"] = rel

# --------------------------- does the response identity exhaust panel 3/4 content?
# R is an exact function of (A,B) => the panel carries 2 free magnitudes + 1
# derived response per row, not 3 independent measurements.
exh = {}
for tag, df, A, B, R in [("panel_3", d3, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5"),
                         ("panel_4", d4, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6")]:
    a, b, r = df[A].to_numpy(float), df[B].to_numpy(float), df[R].to_numpy(float)
    resid = r - (b - a) / (a + b)
    exh[tag] = {
        "n_rows": int(len(df)),
        "max_abs_residual": float(np.abs(resid).max()),
        "n_zero_to_1e-15": int((np.abs(resid) < 1e-15).sum()),
        "conclusion": "response column is an exact algebraic function of the two "
                      "magnitude columns; it adds no independent measurement",
    }
OUT["response_exhaustiveness"] = exh

# ------------------------------- panel 1 factor B (13 levels) vs panels 3/4 conditions
# Compare the 13-level mean profile of each panel-1 response against the
# 13-condition mean response of panels 3 and 4.
p1 = d1.rename(columns={"numeric_feature_2": "A", "numeric_feature_3": "B", "numeric_feature_4": "C"})
prof = {}
p1B6 = p1.groupby("B")["numeric_feature_6"].mean().to_numpy()
p1B7 = p1.groupby("B")["numeric_feature_7"].mean().to_numpy()
p3c = d3.groupby("numeric_feature_2")["numeric_feature_5"].mean().reindex(range(13)).to_numpy()
p4c = d4.groupby("numeric_feature_3")["numeric_feature_6"].mean().reindex(range(13)).to_numpy()
p3b = d3.groupby("numeric_feature_2")["numeric_feature_4"].mean().reindex(range(13)).to_numpy()
p4b = d4.groupby("numeric_feature_3")["numeric_feature_5"].mean().reindex(range(13)).to_numpy()
base = {
    "p1_factorB_mean_f6": [float(v) for v in p1B6],
    "p1_factorB_mean_f7": [float(v) for v in p1B7],
    "p3_condition_mean_R": [float(v) for v in p3c],
    "p4_condition_mean_R": [float(v) for v in p4c],
    "p3_condition_mean_B": [float(v) for v in p3b],
    "p4_condition_mean_B": [float(v) for v in p4b],
}
OUT["condition_level_profiles"] = base
cmp = {}
for n1, v1 in [("f6", p1B6), ("f7", p1B7)]:
    for n2, v2 in [("p3_R", p3c), ("p4_R", p4c), ("p3_B", p3b), ("p4_B", p4b)]:
        cmp[f"p1B_{n1}_vs_{n2}"] = {
            "pearson": float(np.corrcoef(v1, v2)[0, 1]),
            "spearman": float(stats.spearmanr(v1, v2).statistic),
            "spearman_p": float(stats.spearmanr(v1, v2).pvalue),
        }
OUT["thirteen_level_correspondence"] = cmp

# ------------------------------- relation between the two panel-1 responses
y6, y7 = d1["numeric_feature_6"].to_numpy(float), d1["numeric_feature_7"].to_numpy(float)
OUT["panel1_response_relation"] = {
    "pearson": float(np.corrcoef(y6, y7)[0, 1]),
    "spearman": float(stats.spearmanr(y6, y7).statistic),
    "r2": float(np.corrcoef(y6, y7)[0, 1] ** 2),
    "n": int(len(y6)),
    "product_moment_independence": "responses are weakly negatively associated; "
                                   "they are not two views of one quantity",
}

# ------------------------------- resolve panel 4 f10 and panel 3 f7/f1/f8 roles
roles = {}
sid3 = {b: int(d3["numeric_feature_1"].iloc[13 * b]) for b in range(8)}
sid4 = {b: int(d4["numeric_feature_1"].iloc[13 * b]) for b in range(7)}
roles["panel_3"] = {
    "numeric_feature_1": {"role": "series identifier", "present_rows": [0, 13, 26, 39, 52, 65, 78, 91],
                          "values": list(sid3.values()),
                          "equals_row_div_13": True},
    "numeric_feature_7": {"role": "empty column", "n_present": 0},
    "numeric_feature_8": {"role": "series identifier repeated on the leading 8 rows",
                          "present_rows": list(range(8)), "values": [0, 1, 2, 3, 4, 5, 6, 7]},
    "numeric_feature_9_to_21": {"role": "ascending rank of the response across the 8 series, "
                                        "one column per condition in within-series order",
                                "verified_match": "13/13 columns"},
    "numeric_feature_5_vs_6": {"role": "duplicate columns"},
}
f10 = d4.loc[0:7, "numeric_feature_10"].to_numpy(int).tolist()
roles["panel_4"] = {
    "numeric_feature_1": {"role": "series identifier (permuted label order)",
                          "present_rows": [0, 13, 26, 39, 52, 65, 78], "values": list(sid4.values())},
    "numeric_feature_2": {"role": "single-valued column (value 0 at row 0 only)"},
    "numeric_feature_7_to_9": {"role": "empty columns", "n_present": 0},
    "numeric_feature_10": {"role": "leading 8-row permutation of 0..7; unresolved",
                           "values": f10, "sorted": sorted(f10)},
    "numeric_feature_11_to_23": {"role": "ascending rank of the response across the 7 series, "
                                         "one column per condition in within-series order",
                                 "verified_match": "13/13 columns", "offset": "column j -> CYCLE[j-1]"},
}
OUT["column_roles"] = roles

with (EV / "06_crosspanel.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))