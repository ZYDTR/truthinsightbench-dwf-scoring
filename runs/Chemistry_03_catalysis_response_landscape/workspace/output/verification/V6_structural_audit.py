#!/usr/bin/env python3
"""
V6 independent structural audit of the four observation sets (falsification pass).
Re-derives every structural claim S1..S7 from the raw TSVs with no reliance on
prior agent output.  Writes V6_structural_audit.json next to itself.

Axis convention used throughout for the panels:
  resp_m[b, k] = response of series (block) b at the k-th ROW of that block
  rk[b, k, j]  = value of the j-th rank column for series (block) b, row k of block b
The j-th rank column therefore holds a vector over SERIES b, i.e. rk[:, k, j].
"""
import json, os, itertools
import numpy as np
import pandas as pd

ROOT = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project"
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "verification")
os.makedirs(OUT, exist_ok=True)
RS = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/workspace_receipt.json"

R = {}

def load(name):
    return pd.read_csv(os.path.join(DATA, name), sep="\t", dtype=str,
                       keep_default_na=False, na_values=[])

s1 = load("observation_set_1.tsv")
s2 = load("observation_set_2.tsv")
s3 = load("observation_set_3.tsv")
s4 = load("observation_set_4.tsv")
raw = {"set1": s1, "set2": s2, "set3": s3, "set4": s4}

def num(df, col):
    return pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan),
                         errors="coerce").astype(float)

def dat(df):
    return df.apply(lambda c: pd.to_numeric(c.astype(str).str.strip().replace("", np.nan),
                                            errors="coerce"))

n1, n2, n3, n4 = dat(s1), dat(s2), dat(s3), dat(s4)

# ---------------------------------------------------------------- shape / fill
R["shapes"] = {}
for k, df in raw.items():
    info = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": list(df.columns)}
    info["per_column"] = {}
    for c in df.columns:
        v = df[c].astype(str)
        nonempty = v[v.str.strip() != ""]
        numeric = pd.to_numeric(nonempty, errors="coerce")
        info["per_column"][c] = {
            "n_nonblank": int(len(nonempty)),
            "n_blank": int((v.str.strip() == "").sum()),
            "n_non_numeric_when_nonempty": int(numeric.isna().sum()),
            "n_unique": int(nonempty.nunique()),
            "min": (None if len(numeric.dropna()) == 0 else float(numeric.min())),
            "max": (None if len(numeric.dropna()) == 0 else float(numeric.max())),
        }
    R["shapes"][k] = info

# ================================================================= S1 (set 1)
row_idx = num(s1, "observation_index")
f1, f5 = num(s1, "numeric_feature_1"), num(s1, "numeric_feature_5")
S1 = {"n_rows": int(len(s1)), "n_cols": int(s1.shape[1]), "columns": list(s1.columns)}
S1["observation_index_is_0_to_n-1"] = bool(np.array_equal(row_idx.values, np.arange(len(s1))))
S1["f1_nonblank"] = int(f1.notna().sum())
S1["f5_nonblank"] = int(f5.notna().sum())
S1["f1_equals_1_to_n"] = bool(np.array_equal(f1.values, np.arange(1, len(s1) + 1)))
S1["f5_equals_0_to_n-1"] = bool(np.array_equal(f5.values, np.arange(0, len(s1))))
S1["f1_min_max"] = [float(f1.min()), float(f1.max())]
S1["f5_min_max"] = [float(f5.min()), float(f5.max())]
S1["f1_n_unique"] = int(f1.nunique())
S1["f5_n_unique"] = int(f5.nunique())
S1["f1_integer_valued"] = bool(np.all(f1.values == np.round(f1.values)))
S1["f5_integer_valued"] = bool(np.all(f5.values == np.round(f5.values)))
S1["f1_diff_all_one"] = bool(np.all(np.diff(f1.values) == 1))
S1["f5_diff_all_one"] = bool(np.all(np.diff(f5.values) == 1))
S1["f1_eq_index_plus_1"] = bool(np.array_equal(f1.values, row_idx.values + 1))
S1["f5_eq_index"] = bool(np.array_equal(f5.values, row_idx.values))
S1["f1_eq_f5_plus_1"] = bool(np.array_equal(f1.values, f5.values + 1))

facts = ["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]
S1["factor_levels"] = {c: sorted(n1[c].dropna().unique().tolist()) for c in facts}
S1["factor_n_levels"] = {c: int(n1[c].nunique()) for c in facts}
S1["factor_all_integer"] = {c: bool(np.all(n1[c].values == np.round(n1[c].values)))
                            for c in facts}
cell = s1[facts].astype(int).apply(tuple, axis=1)
S1["n_distinct_cells"] = int(cell.nunique())
S1["n_expected_cells"] = int(np.prod([S1["factor_n_levels"][c] for c in facts]))
S1["complete_factorial_1rep"] = bool(cell.nunique() == S1["n_expected_cells"] == len(s1))
S1["max_replicates_per_cell"] = int(cell.value_counts().max())
S1["design_fully_balanced"] = bool(S1["complete_factorial_1rep"])
S1["cell_order_is_lexicographic_by_factor_code"] = None

resp = ["numeric_feature_6", "numeric_feature_7"]
S1["responses_nonblank"] = {c: int(n1[c].notna().sum()) for c in resp}
S1["responses_n_unique"] = {c: int(n1[c].nunique()) for c in resp}
S1["responses_range"] = {c: [float(n1[c].min()), float(n1[c].max())] for c in resp}
S1["all_cols_n_unique"] = {c: int(n1[c].nunique()) for c in s1.columns}
S1["all_cols_n_nan"] = {c: int(n1[c].isna().sum()) for c in s1.columns}
S1["corr_f1_with_index"] = float(np.corrcoef(f1.values, row_idx.values)[0, 1])
S1["corr_f5_with_index"] = float(np.corrcoef(f5.values, row_idx.values)[0, 1])
S1["corr_f2_with_index"] = float(np.corrcoef(n1["numeric_feature_2"].values,
                                             row_idx.values)[0, 1])
# does the index encode the factor design?  check ordering of the factorial
S1["index_equals_ravel_of_factor_grid"] = None
# rank of f1, f5 vs anything: they carry no information beyond a counter
R["S1"] = S1

# ================================================================= S2 (set 2)
S2 = {"n_rows": int(len(s2)), "n_cols": int(s2.shape[1]), "columns": list(s2.columns)}
S2["f3_nonblank"] = int(n2["numeric_feature_3"].notna().sum())
S2["f3_nonblank_rows"] = [int(i) for i in s2.index[n2["numeric_feature_3"].notna()]]
S2["f3_values"] = [float(x) for x in n2["numeric_feature_3"].dropna()]
d = n2["numeric_feature_1"] - n2["numeric_feature_2"]
S2["f1_minus_f2"] = [float(x) for x in d.values]
S2["f1_minus_f2_abs"] = [float(abs(x)) for x in d.values]
S2["max_abs_f1_minus_f2"] = float(np.nanmax(np.abs(d.values)))
S2["mean_f1_minus_f2"] = float(np.nanmean(d.values))
S2["mean_f1_minus_f2_head7"] = float(np.nanmean(d.values[:7]))
S2["mean_f1_minus_f2_tail"] = float(np.nanmean(d.values[7:]))
S2["f1_f2_identical_rows"] = [int(i) for i in s2.index[np.isclose(n2["numeric_feature_1"],
                                                                  n2["numeric_feature_2"])]]
S2["f1_head7"] = [float(x) for x in n2["numeric_feature_1"].values[:7]]
S2["f2_head7"] = [float(x) for x in n2["numeric_feature_2"].values[:7]]
S2["head7_absdiff_diff_positive"] = bool(np.all(np.diff(np.abs(d.values[:7])) > 0))
S2["f1_is_symmetric_pairs_around_0"] = bool(np.array_equal(n2["numeric_feature_1"].values[:7],
                                                           -n2["numeric_feature_1"].values[:7][::-1]))
S2["f2_is_odd_function_of_f1_head7"] = None
S2["corr_f1_f2"] = float(np.corrcoef(n2["numeric_feature_1"].values,
                                     n2["numeric_feature_2"].values)[0, 1])
S2["f1_all_integer_head7"] = bool(np.all(n2["numeric_feature_1"].values[:7] % 1 == 0))
S2["f1_tail_all_2dp"] = bool(np.all(np.abs(n2["numeric_feature_1"].values[7:] * 100 -
                                           np.round(n2["numeric_feature_1"].values[7:] * 100)) < 1e-9))
S2["f2_tail_all_2dp"] = bool(np.all(np.abs(n2["numeric_feature_2"].values[7:] * 100 -
                                           np.round(n2["numeric_feature_2"].values[7:] * 100)) < 1e-9))
S2["f1_f2_range"] = {"f1": [float(n2["numeric_feature_1"].min()), float(n2["numeric_feature_1"].max())],
                     "f2": [float(n2["numeric_feature_2"].min()), float(n2["numeric_feature_2"].max())]}
R["S2"] = S2

# ================================================================= S3 header/layout
BS = 13
S3 = {"n_rows": int(len(s3)), "n_cols": int(s3.shape[1]), "columns": list(s3.columns)}
S3["n_blocks"] = int(len(s3) // BS) if len(s3) % BS == 0 else None
S3["block_size_13_divides"] = bool(len(s3) % BS == 0)
f1_3 = num(s3, "numeric_feature_1")
S3["f1_nonblank_rows"] = [int(i) for i in s3.index[f1_3.notna()]]
S3["f1_nonblank_values"] = [float(x) for x in f1_3.dropna()]
S3["f1_present_only_on_block_starts"] = bool(
    list(s3.index[f1_3.notna()]) == list(range(0, 13 * (len(s3) // BS), 13)))
S3["f1_eq_rowdiv13"] = bool(np.all(f1_3.dropna().values ==
                                   (np.array(s3.index[f1_3.notna()]) // BS)))
S3["f1_blank_between_block_starts"] = bool(f1_3.isna().sum() == len(s3) - len(s3) // BS)
S3["f1_all_else_blank"] = bool(all(str(s3.loc[i, "numeric_feature_1"]).strip() == ""
                                   for i in s3.index if i % 13 != 0))
f8_3 = num(s3, "numeric_feature_8")
S3["f8_nonblank_rows"] = [int(i) for i in s3.index[f8_3.notna()]]
S3["f8_values"] = [float(x) for x in f8_3.dropna()]
S3["f8_is_0_to_7_on_rows_0_7"] = bool(np.array_equal(f8_3.dropna().values, np.arange(8.0)))
S3["f8_only_rows_0_7"] = bool(list(s3.index[f8_3.notna()]) == list(range(8)))
f7_3 = num(s3, "numeric_feature_7")
S3["f7_n_nonblank"] = int(f7_3.notna().sum())
S3["f7_all_blank"] = bool(f7_3.isna().all())
S3["f7_raw_all_empty_string"] = bool(all(str(x).strip() == ""
                                         for x in s3["numeric_feature_7"]))

# S3 f5 == f6 exact duplicates
v5, v6 = num(s3, "numeric_feature_5"), num(s3, "numeric_feature_6")
S3["f5_f6_bitwise_equal_all_rows"] = bool(np.array_equal(v5.values, v6.values, equal_nan=True))
S3["f5_f6_max_abs_diff"] = float(np.nanmax(np.abs(v5.values - v6.values)))
S3["f5_f6_n_differing_rows"] = int(np.sum(~(v5.values == v6.values)))
S3["f5_f6_string_identical_all_rows"] = bool(all(
    str(s3.loc[i, "numeric_feature_5"]) == str(s3.loc[i, "numeric_feature_6"])
    for i in s3.index))
f3c, f4c = num(s3, "numeric_feature_3"), num(s3, "numeric_feature_4")
calc = (f4c - f3c) / (f4c + f3c)
S3["f5_eq_(f4-f3)/(f4+f3)_max_abs_err"] = float(np.nanmax(np.abs(calc.values - v5.values)))
S3["f5_eq_(f4-f3)/(f4+f3)_bitwise_exact"] = bool(np.all(calc.values == v5.values))
S3["f5_eq_(f3-f4)/(f3+f4)_max_abs_err"] = float(np.nanmax(np.abs(-calc.values - v5.values)))
S3["f5_eq_f3_ratio_alts"] = {
    "f5_eq_(f4-f3)/(f4+f3)": float(np.nanmax(np.abs(calc.values - v5.values))),
    "f5_eq_(f3-f4)/(f3+f4)": float(np.nanmax(np.abs(-calc.values - v5.values))),
    "f5_eq_1-2f3/(f3+f4)": float(np.nanmax(np.abs((1 - 2 * f3c / (f3c + f4c)).values - v5.values))),
}
S3["f3_lt_f4_all_rows"] = bool(np.all((f3c < f4c).values))
S3["f5_always_positive"] = bool(np.all(v5.values > 0))
S3["f2_unique_values"] = sorted(n3["numeric_feature_2"].dropna().unique().tolist())
S3["f2_is_integer"] = bool(np.all(n3["numeric_feature_2"].values ==
                                  np.round(n3["numeric_feature_2"].values)))
S3["f2_min_max"] = [float(n3["numeric_feature_2"].min()), float(n3["numeric_feature_2"].max())]

# non-blank patterns (first rows) per column of set 3
S3["nonblank_pattern"] = {}
for c in s3.columns:
    v = num(s3, c)
    nb = v.notna()
    S3["nonblank_pattern"][c] = {
        "n_nonblank": int(nb.sum()),
        "rows_nonblank_head": [int(i) for i in s3.index[nb][:18]],
        "values_head": [None if pd.isna(x) else float(x) for x in v[nb][:18]],
        "n_unique": int(v.nunique()),
    }
R["S3"] = S3

# ================================================================= S4 layout
S4 = {"n_rows": int(len(s4)), "n_cols": int(s4.shape[1]), "columns": list(s4.columns)}
S4["n_blocks"] = int(len(s4) // BS) if len(s4) % BS == 0 else None
S4["block_size_13_divides"] = bool(len(s4) % BS == 0)
f1_4 = num(s4, "numeric_feature_1")
S4["f1_nonblank_rows"] = [int(i) for i in s4.index[f1_4.notna()]]
S4["f1_nonblank_values"] = [float(x) for x in f1_4.dropna()]
S4["f1_present_only_on_block_starts"] = bool(
    list(s4.index[f1_4.notna()]) == list(range(0, 13 * (len(s4) // BS), 13)))
S4["f1_all_else_blank"] = bool(all(str(s4.loc[i, "numeric_feature_1"]).strip() == ""
                                   for i in s4.index if i % 13 != 0))
S4["block_labels"] = [float(x) for x in f1_4.dropna()]
S4["block_labels_are_perm_of_0_6"] = bool(sorted(f1_4.dropna().tolist()) == list(range(7)))
S4["block_labels_equal_row_positions"] = bool(np.array_equal(f1_4.dropna().values, np.arange(7)))
S4["block_label_order"] = ("i -> (i+6) mod 7 (cyclic shift by one)" if np.array_equal(
    f1_4.dropna().values, (np.arange(7) + 6) % 7) else "other")
S4["sorting_blocks_by_label_reproduces_0..6"] = bool(
    np.array_equal(np.sort(f1_4.dropna().values), np.arange(7)))
S4["score_by_label_would_differ_from_row_order"] = bool(not np.array_equal(
    f1_4.dropna().values, np.arange(7)))

S4["sparse_cols"] = {}
for c in ["numeric_feature_2", "numeric_feature_7", "numeric_feature_8", "numeric_feature_9"]:
    v = num(s4, c)
    S4["sparse_cols"][c] = {
        "n_nonblank": int(v.notna().sum()),
        "rows_nonblank": [int(i) for i in s4.index[v.notna()]],
        "values": [float(x) for x in v.dropna()],
        "all_blank": bool(v.isna().all()),
    }
f10_4 = num(s4, "numeric_feature_10")
S4["f10_n_nonblank"] = int(f10_4.notna().sum())
S4["f10_n_unique"] = int(f10_4.nunique())
S4["f10_values"] = [float(x) for x in f10_4.dropna()]
S4["f10_equals_f8_range"] = bool(list(sorted(f10_4.dropna().tolist())) == list(range(8)))
S4["f10_perm_of_0_7"] = S4["f10_equals_f8_range"]
S4["f10_per_block"] = [float(x) for x in f10_4.dropna().values.reshape(-1)]
S4["f10_matches_f8_of_set3"] = bool(list(f10_4.dropna().values) == list(np.arange(8.0)))
S4["f10_block0_rows"] = [None if pd.isna(x) else float(x)
                         for x in s4["numeric_feature_10"].iloc[:13].apply(
                             lambda z: pd.to_numeric(z, errors="coerce"))]
S4["f10_blank_rows"] = [int(i) for i in s4.index[f10_4.isna()]]

rank_cols4 = [f"numeric_feature_{i}" for i in range(11, 24)]
S4["rank_cols"] = rank_cols4
S4["rank_cols_n_nonblank"] = {c: int(num(s4, c).notna().sum()) for c in rank_cols4}
S4["rank_cols_n_unique"] = {c: int(num(s4, c).nunique()) for c in rank_cols4}
S4["rank_cols_head"] = {c: [None if pd.isna(x) else float(x)
                            for x in num(s4, c).iloc[:16]] for c in rank_cols4}
S4["rank_cols_nonblank_rows"] = {c: [int(i) for i in s4.index[num(s4, c).notna()]]
                                 for c in rank_cols4}
S4["nonblank_pattern"] = {}
for c in s4.columns:
    v = num(s4, c)
    nb = v.notna()
    S4["nonblank_pattern"][c] = {
        "n_nonblank": int(nb.sum()),
        "rows_nonblank_head": [int(i) for i in s4.index[nb][:20]],
        "values_head": [None if pd.isna(x) else float(x) for x in v[nb][:20]],
        "n_unique": int(v.nunique()),
    }
f4c4, f5c4 = num(s4, "numeric_feature_4"), num(s4, "numeric_feature_5")
resp4 = num(s4, "numeric_feature_6")
calc4 = (f5c4 - f4c4) / (f5c4 + f4c4)
S4["f6_eq_(f5-f4)/(f5+f4)_max_abs_err"] = float(np.nanmax(np.abs(calc4.values - resp4.values)))
S4["f6_eq_(f5-f4)/(f5+f4)_bitwise_exact"] = bool(np.all(calc4.values == resp4.values))
S4["f6_eq_(f4-f5)/(f4+f5)_max_abs_err"] = float(np.nanmax(np.abs(-calc4.values - resp4.values)))
S4["f4_lt_f5_all_rows"] = bool(np.all((f4c4 < f5c4).values))
S4["f6_always_positive"] = bool(np.all(resp4.values > 0))
S4["f3_unique_sorted"] = sorted(n4["numeric_feature_3"].dropna().unique().tolist())
S4["f3_is_integer"] = bool(np.all(n4["numeric_feature_3"].values ==
                                  np.round(n4["numeric_feature_3"].values)))
R["S4"] = S4

# ================================================================= S7 block starts
resp3_m = num(s3, "numeric_feature_5").values.reshape(-1, BS)
resp4_m = resp4.values.reshape(-1, BS)
S7 = {}
S7["set3_block_matrix"] = [[float(x) for x in r] for r in resp3_m]
S7["set4_block_matrix"] = [[float(x) for x in r] for r in resp4_m]
S7["set3_start_row_in_block_is_cond0_claim"] = None

def rankset(row):
    return pd.Series(row).rank(method="max").values

S7["set3_start_rank_by_max"] = [int(rankset(r)[0]) for r in resp3_m]
S7["set4_start_rank_by_max"] = [int(rankset(r)[0]) for r in resp4_m]
S7["set3_start_is_max_every_block"] = bool(all(rankset(r)[0] == 8 for r in resp3_m))
S7["set3_start_is_min_every_block"] = bool(all(rankset(r)[0] == 1 for r in resp3_m))
S7["set4_start_is_min_every_block"] = bool(all(rankset(r)[0] == 1 for r in resp4_m))
S7["set4_start_is_max_every_block"] = bool(all(rankset(r)[0] == 8 for r in resp4_m))
S7["set3_start_minus_blockmean"] = [float(r[0] - r.mean()) for r in resp3_m]
S7["set4_start_minus_blockmean"] = [float(r[0] - r.mean()) for r in resp4_m]
# how many blocks have a strictly-unique start value
S7["set3_blocks_start_unique"] = int(sum(r[0] not in r[1:] for r in resp3_m))
S7["set4_blocks_start_unique"] = int(sum(r[0] not in r[1:] for r in resp4_m))
R["S7"] = S7

# ================================================================= rank analysis
def analyse_ranks(resp_m, rk, label):
    out = {}
    nB, nC = resp_m.shape
    out["shape_BxC"] = [nB, nC]
    out["rank_table_all_values_present"] = bool(not np.isnan(rk).any())
    out["per_column_is_perm_0_7_over_series"] = {
        int(j): bool(sorted(rk[:, :, j].ravel().tolist()) == list(range(nB)))
        for j in range(nC)}
    out["n_columns_that_are_perm_of_0_7"] = int(sum(
        sorted(rk[:, :, j].ravel().tolist()) == list(range(nB)) for j in range(nC)))
    # per row-within-block: is rk[b,k,:] a perm of 0..nB-1?
    per_row_perm = all(sorted(rk[b, k, :].tolist()) == list(range(nB))
                       for b in range(nB) for k in range(nC))
    out["each_rank_row_is_perm_0_7"] = bool(per_row_perm)
    out["example_block0_rank_table_rows_over_series"] = [
        {"row_in_block": k, "ranks_by_series": [int(x) for x in rk[:, k, 0]]}
        for k in range(nC)]

    methods = {
        "max": lambda x: pd.Series(x).rank(method="max").values,
        "average": lambda x: pd.Series(x).rank(method="average").values,
        "min": lambda x: pd.Series(x).rank(method="min").values,
        "dense": lambda x: pd.Series(x).rank(method="dense").values,
        "first": lambda x: pd.Series(x).rank(method="first").values,
    }
    # HYP: rank column j = rank across series of the condition found at row position p
    # in EVERY block (p unknown).  Test p = j (block-start = condition 0 rule) first.
    for mname, fn in methods.items():
        hits = []
        for j in range(nC):
            h = fn(resp_m[:, j])          # vector over series b
            hits.append(bool(np.array_equal(h, rk[:, :, j].astype(float))))
        out[f"cols_reproduced_by_identity_order_{mname}"] = int(sum(hits))
        out[f"which_reproduced_identity_order_{mname}"] = [int(j) for j, h in enumerate(hits) if h]

    # General feasible condition per column j: which p (same across blocks) reproduces it?
    feas = {}
    for j in range(nC):
        cand = []
        for p in range(nC):
            for mname, fn in methods.items():
                if np.array_equal(fn(resp_m[:, p]), rk[:, :, j].astype(float)):
                    cand.append({"p": int(p), "method": mname})
        feas[j] = cand
    out["per_col_feasible_row_position"] = {int(j): v for j, v in feas.items()}
    out["n_cols_with_0_feasible"] = int(sum(len(v) == 0 for v in feas.values()))
    out["n_cols_with_unique_feasible"] = int(sum(len({c["p"] for c in v}) == 1 and len(v) >= 1
                                                 for v in feas.values()))
    unique_map = {int(j): sorted({c["p"] for c in feas[j]})
                  for j in range(nC) if len({c["p"] for c in feas[j]}) == 1 and feas[j]}
    out["unique_p_per_col"] = unique_map
    out["unique_p_map_is_injective"] = bool(
        len(set(tuple(v) for v in unique_map.values())) == len(unique_map))

    # bipartite: injective assignment j -> p using feasible sets
    def bt(j, used, res):
        if j == nC:
            return True
        for c in feas[j]:
            if c["p"] in used:
                continue
            used.add(c["p"]); res[j] = c
            if bt(j + 1, used, res):
                return True
            used.discard(c["p"])
        return False

    res = [None] * nC
    ok = bt(0, set(), res)
    out["injective_assignment_exists"] = bool(ok)
    out["assignment_j_to_rowpos"] = (None if not ok else
                                     [{"rank_col_index": j, "row_position": res[j]["p"],
                                       "method": res[j]["method"]} for j in range(nC)])
    if ok:
        used_p = [res[j]["p"] for j in range(nC)]
        out["assignment_covers_all_positions"] = bool(sorted(used_p) == list(range(nC)))
        out["assignment_implied_block0_condition_order"] = used_p
        # does it equal identity?
        out["assignment_equals_identity_j_eq_p"] = bool(used_p == list(range(nC)))
        # apply the discovered order to reorder the block-0 response and see if it is sorted
        out["block0_response_reordered"] = [float(resp_m[0, used_p[j]]) for j in range(nC)]
        out["block0_response_reordered_is_increasing"] = bool(np.all(
            np.diff([resp_m[0, used_p[j]] for j in range(nC)]) >= 0))

    # Alternative family: rank columns describe the WITHIN-BLOCK ascending rank of the
    # response, grouped by condition index taken from the position at which the value sits
    # in the sorted order (rank r = the r-th smallest condition).
    for mname, fn in methods.items():
        hits = []
        for j in range(nC):
            h = np.array([fn(resp_m[b, :])[j] for b in range(nB)])
            hits.append(bool(np.array_equal(h, rk[:, :, j].astype(float))))
        out[f"cols_reproduced_by_within_block_rank_{mname}"] = int(sum(hits))
        out[f"which_within_block_rank_{mname}"] = [int(j) for j, h in enumerate(hits) if h]

    # Alternative family 2: rank columns derived from the SORTED-within-block order
    # (ascending: j-th col corresponds to the j-th smallest condition; the mapping
    #  "row position of the j-th smallest" must be constant across blocks)
    asc_pos = np.argsort(resp_m, axis=1, kind="stable")     # B x C
    desc_pos = np.argsort(-resp_m, axis=1, kind="stable")
    for tag, P in [("ascending", asc_pos), ("descending", desc_pos)]:
        for mname, fn in methods.items():
            hits = []
            for j in range(nC):
                h = np.array([fn(resp_m[b, :])[P[b, j]] for b in range(nB)])
                hits.append(bool(np.array_equal(h, rk[:, :, j].astype(float))))
            out[f"cols_reproduced_by_{tag}_rank_position_{mname}"] = int(sum(hits))
            out[f"which_{tag}_{mname}"] = [int(j) for j, h in enumerate(hits) if h]
    out["ascending_order_positions_per_block"] = asc_pos.tolist()
    out["ascending_order_is_constant_across_blocks"] = bool(
        all(list(asc_pos[0]) == list(asc_pos[b]) for b in range(nB)))
    out["descending_order_is_constant_across_blocks"] = bool(
        all(list(desc_pos[0]) == list(desc_pos[b]) for b in range(nB)))
    # condition index of a value == its ascending rank; so col j = rank of block at the
    # condition whose within-block ascending rank is j  <=>  position p = asc_pos[b][j]
    # The above tests cover it.
    return out

rank_cols3 = [f"numeric_feature_{i}" for i in range(9, 22)]
S3["rank_cols"] = rank_cols3
S3["rank_cols_n_nonblank"] = {c: int(num(s3, c).notna().sum()) for c in rank_cols3}
S3["rank_cols_n_unique"] = {c: int(num(s3, c).nunique()) for c in rank_cols3}
S3["rank_cols_head"] = {c: [None if pd.isna(x) else float(x)
                            for x in num(s3, c).iloc[:16]] for c in rank_cols3}
rk3b = np.column_stack([num(s3, c).values for c in rank_cols3]).reshape(-1, BS, BS)
rk4b = np.column_stack([num(s4, c).values for c in rank_cols4]).reshape(-1, BS, BS)
R["rank_analysis_set3"] = analyse_ranks(resp3_m, rk3b, "set3")
R["rank_analysis_set4"] = analyse_ranks(resp4_m, rk4b, "set4")

# ================================================================= extra relations
E = {}
E["set1_corr"] = n1.corr().round(8).to_dict()
E["set1_f6_f7_corr"] = float(n1["numeric_feature_6"].corr(n1["numeric_feature_7"]))
E["set1_f6_plus_f7_min_max"] = [float((n1["numeric_feature_6"] + n1["numeric_feature_7"]).min()),
                                float((n1["numeric_feature_6"] + n1["numeric_feature_7"]).max())]
E["set1_f6_plus_f7_max_abs_err_from_1"] = float(np.max(np.abs(
    (n1["numeric_feature_6"] + n1["numeric_feature_7"]).values - 1.0)))
E["set1_f6_minus_f7_min_max"] = [float((n1["numeric_feature_6"] - n1["numeric_feature_7"]).min()),
                                 float((n1["numeric_feature_6"] - n1["numeric_feature_7"]).max())]
E["set1_f6_gt_f7_fraction"] = float(np.mean((n1["numeric_feature_6"] >
                                             n1["numeric_feature_7"]).values))
E["set1_f6_rounded_check"] = None
E["set1_f6_f7_roundtrip"] = None
E["set1_f6_is_ratio_of_medians_of_f7"] = None

E["set3_corr"] = n3.corr().round(8).to_dict()
E["set4_corr"] = n4.corr().round(8).to_dict()
E["set3_f6_f7_relation_absent"] = None

# set3 f6/f5 vs f3,f4 raw ratio reconstruction bit-exactness
E["set3_f5_bitwise_from_ratio_fraction"] = float(np.mean((calc.values == v5.values)))
E["set4_f6_bitwise_from_ratio_fraction"] = float(np.mean((calc4.values == resp4.values)))

E["set1_index_spanning"] = None
R["E"] = E

with open(os.path.join(OUT, "V6_structural_audit.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=str)
print("WROTE", os.path.join(OUT, "V6_structural_audit.json"))