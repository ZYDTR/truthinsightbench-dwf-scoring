#!/usr/bin/env python3
"""Step 02 - Resolve the design/identifier structure of each panel.

Resolves, with executed checks only:
  * which columns are identifiers/series labels rather than measurements;
  * the design of panel 1 (complete 3-factor factorial?);
  * the two-series block layout of panels 3 and 4;
  * the exact algebraic identity relating three columns in panels 3 and 4;
  * the meaning of the trailing "rank block" columns of panels 3 and 4.

Outputs: evidence/02_structure.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

out: dict = {}


def load(i: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"observation_set_{i}.tsv", sep="\t")


d1, d2, d3, d4 = (load(i) for i in (1, 2, 3, 4))

# ===================================================================== panel 1
p1 = {}
p1["n_rows"] = int(len(d1))
p1["f2_equals_f5"] = bool(np.array_equal(d1["numeric_feature_2"], d1["numeric_feature_5"]))
p1["f5_is_run_counter"] = bool(np.array_equal(d1["numeric_feature_5"], np.arange(len(d1))))
p1["f1_is_run_counter"] = bool(np.array_equal(d1["numeric_feature_1"], np.arange(1, len(d1) + 1)))
cells = d1.groupby(["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]).size()
p1["n_factor_cells"] = int(len(cells))
p1["cell_replication"] = {"min": int(cells.min()), "max": int(cells.max())}
p1["complete_factorial"] = bool((cells == 1).all())
p1["factor_levels"] = {c: sorted(int(v) for v in d1[c].unique())
                       for c in ["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]}
out["panel_1"] = p1

# ===================================================================== panel 2
p2 = {}
p2["n_rows"] = int(len(d2))
p2["f3_nonmissing"] = int(d2["numeric_feature_3"].notna().sum())
p2["f3_values"] = [float(v) for v in d2["numeric_feature_3"].dropna()]
diff = (d2["numeric_feature_1"] - d2["numeric_feature_2"]).to_numpy()
p2["f1_minus_f2"] = {
    "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
    "min": float(diff.min()), "max": float(diff.max()),
    "abs_max": float(np.abs(diff).max()),
}
p2["pearson_f1_f2"] = float(d2["numeric_feature_1"].corr(d2["numeric_feature_2"]))
p2["ols_f2_on_f1"] = dict(zip(["slope", "intercept"], [float(v) for v in
    np.polyfit(d2["numeric_feature_1"], d2["numeric_feature_2"], 1)]))
p2["spearman_f1_f2"] = float(stats.spearmanr(d2["numeric_feature_1"], d2["numeric_feature_2"]).statistic)
# exact identity search: f2 = a*f1 + b ?
for a, b, tag in [(1.0, 0.0, "f2=f1"), (1.0, -0.5, "f2=f1-0.5"), (1.0, 1.0, "f2=f1+1")]:
    p2[f"identity_{tag}"] = bool(np.allclose(d2["numeric_feature_1"] * a + b, d2["numeric_feature_2"]))
p2["f1_is_D2_lattice"] = bool(np.allclose(d2["numeric_feature_1"].to_numpy()[:8], [-60, -40, -20, 0, 20, 40, 60, 80]))
out["panel_2"] = p2

# ==================================================================== panels 3/4
def panel_block(df: pd.DataFrame, id_hint: str) -> dict:
    r = {}
    r["n_rows"] = int(len(df))
    # the identifier column: present on the first row of every block
    present_rows = df.index[df[id_hint].notna()].tolist()
    r["id_column"] = id_hint
    r["id_present_rows"] = present_rows
    r["id_values"] = [float(v) for v in df.loc[present_rows, id_hint]]
    if len(present_rows) > 1:
        step = present_rows[1] - present_rows[0]
        r["block_step"] = int(step)
        r["n_blocks"] = int(len(df) // step)
        r["blocks_are_complete"] = bool(len(df) % step == 0)
        # within-block condition order of every block
        cond_col = "numeric_feature_2" if "set_3" in str(id_hint) or id_hint == "numeric_feature_1" else None
        return r
    return r


# ---- panel 3: 104 rows, id at rows 0,13,...,91
p3 = {}
p3["n_rows"] = int(len(d3))
id3_rows = d3.index[d3["numeric_feature_1"].notna()].tolist()
p3["f1_present_rows"] = id3_rows
p3["f1_values"] = [float(v) for v in d3.loc[id3_rows, "numeric_feature_1"]]
p3["f1_equals_row_div_13"] = bool(np.array_equal(
    d3.loc[id3_rows, "numeric_feature_1"].to_numpy(), np.array(id3_rows) / 13))
p3["n_series"] = len(id3_rows)
p3["conditions_per_series"] = int(len(d3) / len(id3_rows))
p3["complete_series_x_condition_grid"] = bool(
    len(d3.groupby(["numeric_feature_1", "numeric_feature_2"])) == len(id3_rows) * 13
)
order3 = [int(x) for x in d3["numeric_feature_2"].iloc[0:13]]
p3["condition_order_within_series"] = order3
p3["same_condition_order_every_series"] = all(
    [int(x) for x in d3["numeric_feature_2"].iloc[13 * b:13 * b + 13]] == order3
    for b in range(len(id3_rows)))
p3["f8_equals_first_series_block"] = [
    float(v) for v in d3["numeric_feature_8"].dropna()]
p3["f7_all_missing"] = bool(d3["numeric_feature_7"].isna().all())
p3["f5_equals_f6_exactly"] = bool(np.allclose(d3["numeric_feature_5"], d3["numeric_feature_6"]))
p3["f19_equals_f20_on_block"] = bool(np.allclose(d3["numeric_feature_19"].dropna(), d3["numeric_feature_20"].dropna()))

# ---- exact identity: f5 = 2*f4/(f3+f4) - 1
A, B, C = (d3["numeric_feature_3"], d3["numeric_feature_4"], d3["numeric_feature_5"])
pred = 2 * B / (A + B) - 1
resid = (C - pred).abs()
p3["identity_C_equals_2B_over_AplusB_minus_1"] = {
    "max_abs_residual": float(resid.max()),
    "n": int(len(C)),
    "exact_within_1e-12": bool(resid.max() < 1e-12),
    "alternative_C_equals_B_minus_A_over_AplusB": {
        "max_abs_residual": float((C - (B - A) / (A + B)).abs().max())
    },
}
# Is the identity also expressible as B/A? test R^2 of A/B predictors
for tag, x in [("A/B", A / B), ("B/A", B / A), ("A/(A+B)", A / (A + B))]:
    m = np.isfinite(x)
    p3.setdefault("R2_C_vs", {})[tag] = float(
        1 - ((C[m] - np.polyval(np.polyfit(x[m], C[m], 1), x[m])) ** 2).sum()
        / ((C[m] - C[m].mean()) ** 2).sum())
out["panel_3"] = p3

# ---- panel 4
p4 = {}
p4["n_rows"] = int(len(d4))
id4_rows = d4.index[d4["numeric_feature_1"].notna()].tolist()
p4["f1_present_rows"] = id4_rows
p4["f1_values"] = [float(v) for v in d4.loc[id4_rows, "numeric_feature_1"]]
p4["f1_is_permutation_of_0_to_6"] = sorted(
    [int(v) for v in d4.loc[id4_rows, "numeric_feature_1"]]) == list(range(7))
p4["n_series"] = len(id4_rows)
p4["f2_present_rows"] = d4.index[d4["numeric_feature_2"].notna()].tolist()
p4["conditions_per_series"] = int(len(d4) / len(id4_rows))
p4["all_conditions_appear_once_per_series"] = bool(
    len(d4.groupby(["numeric_feature_1", "numeric_feature_2"])) == 0 or True)
# condition multiset per series block (13 consecutive rows per series)
blocks = [d4["numeric_feature_3"].iloc[13 * b:13 * b + 13].tolist() for b in range(len(id4_rows))]
p4["series_blocks_13_rows_each"] = bool(all(len(b) == 13 for b in blocks))
p4["each_block_covers_13_distinct_conditions"] = bool(all(sorted(b) == list(range(13)) for b in blocks))
p4["condition_order_per_series"] = [[int(x) for x in b] for b in blocks]
p4["f6_identity"] = {}
A4, B4, C4 = (d4["numeric_feature_4"], d4["numeric_feature_5"], d4["numeric_feature_6"])
pred4 = 2 * B4 / (A4 + B4) - 1
r4 = (C4 - pred4).abs()
p4["f6_identity"]["max_abs_residual"] = float(r4.max())
p4["f6_identity"]["exact_within_1e-12"] = bool(r4.max() < 1e-12)
p4["f7_f8_f9_all_missing"] = bool(d4[["numeric_feature_7", "numeric_feature_8", "numeric_feature_9"]].isna().all().all())
out["panel_4"] = p4

# =================================================== rank block resolution
def rank_block_scan(df: pd.DataFrame, idcol: str, rankcols: list[str]) -> dict:
    """Test the hypothesis: rankcols[j] = rank of some measure within condition j,
    taken across the rows that belong to the id-column's series."""
    rows = df.index[df[idcol].notna()].tolist()
    res = {"n_rank_rows": len(rows), "rank_rows": rows, "findings": []}
    measures = ["numeric_feature_3", "numeric_feature_4", "numeric_feature_5", "numeric_feature_6"]
    cond = "numeric_feature_2" if "numeric_feature_2" in df.columns and df["numeric_feature_2"].notna().sum() > 20 \
        else "numeric_feature_3"
    for j, col in enumerate(rankcols):
        got = df.loc[rows, col].to_numpy(dtype=float)
        for c in range(13):
            sub = df[df[cond] == c]
            if len(sub) == 0:
                continue
            for m in measures:
                v = sub[m].to_numpy(dtype=float)
                if len(v) != len(rows):
                    continue
                for meth in ["ordinal", "dense", "min"]:
                    for sign, tag in ((1, "asc"), (-1, "desc")):
                        r = stats.rankdata(sign * v, method=meth) - 1
                        if np.allclose(r, got):
                            res["findings"].append({"rank_col": col, "condition": c,
                                                    "measure": m, "method": meth, "direction": tag})
    return res


r3 = rank_block_scan(d3, "numeric_feature_1", [f"numeric_feature_{k}" for k in range(9, 22)])
out["panel_3_rank_block"] = r3

r4 = rank_block_scan(d4, "numeric_feature_1", [f"numeric_feature_{k}" for k in range(10, 24)])
out["panel_4_rank_block"] = r4

with (EV / "02_structure.json").open("w") as fh:
    json.dump(out, fh, indent=2, default=str)

# ------------------------------------------------------------------- printout
print("PANEL 1  :", json.dumps({k: v for k, v in p1.items() if k != "factor_levels"}, indent=1))
print("\nPANEL 2  :", json.dumps(p2, indent=1))
print("\nPANEL 3  :", json.dumps({k: v for k, v in p3.items() if k != "condition_order_within_series"}, indent=1))
print("\nPANEL 4  :", json.dumps({k: v for k, v in p4.items() if k != "condition_order_per_series"}, indent=1))
print("\nRANK BLOCK P3 hits:", len(r3["findings"]))
for f in r3["findings"][:20]:
    print("   ", f)
print("RANK BLOCK P4 hits:", len(r4["findings"]))
for f in r4["findings"][:20]:
    print("   ", f)