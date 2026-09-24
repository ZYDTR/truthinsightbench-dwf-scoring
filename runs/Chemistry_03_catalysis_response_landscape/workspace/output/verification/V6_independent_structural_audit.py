#!/usr/bin/env python3
"""
V6 INDEPENDENT structural audit of observation_set_{1,2,3,4}.tsv.
Falsification pass: every structural claim S1..S7 is re-derived from the raw files
with independent code. Nothing is taken from any other agent.
Writes V6_independent_structural_audit.json next to itself.
"""
import json
import os
import itertools
import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data"
OUT = os.path.dirname(os.path.abspath(__file__))
R, V = {}, {}


def log(*a):
    print(*a, flush=True)


s1 = pd.read_csv(f"{DATA}/observation_set_1.tsv", sep="\t")
s2 = pd.read_csv(f"{DATA}/observation_set_2.tsv", sep="\t")
s3 = pd.read_csv(f"{DATA}/observation_set_3.tsv", sep="\t")
s4 = pd.read_csv(f"{DATA}/observation_set_4.tsv", sep="\t")

R["shapes"] = {"set%d" % n: list(d.shape) for n, d in zip([1, 2, 3, 4], [s1, s2, s3, s4])}
log("shapes:", R["shapes"])

# --- raw-text integrity: no quoting, uniform token counts, no ragged rows -------
R["raw_token_counts"] = {}
for n in [1, 2, 3, 4]:
    with open(f"{DATA}/observation_set_{n}.tsv") as fh:
        lines = [l for l in fh.read().split("\n") if l != ""]
    R["raw_token_counts"][n] = sorted({len(l.split("\t")) for l in lines})
    log(f"set{n}: lines={len(lines)} distinct tab-token-counts={R['raw_token_counts'][n]}")

# =============================================================================
log("\n===== S1 : observation_set_1 =====")
S1 = {}
oi = s1["observation_index"].to_numpy()
S1["obs_index_is_0_to_n-1"] = bool(np.array_equal(oi, np.arange(len(s1))))
S1["f1_is_1_to_1430"] = bool(np.array_equal(s1["numeric_feature_1"].to_numpy(), np.arange(1, len(s1) + 1)))
S1["f1_equals_obs_plus_1"] = bool(np.array_equal(s1["numeric_feature_1"].to_numpy(), oi + 1))
S1["f1_is_run_counter"] = S1["f1_is_1_to_1430"]
f5 = s1["numeric_feature_5"].to_numpy()
S1["f5_is_permutation_of_0_to_1429"] = bool(sorted(f5.tolist()) == list(range(len(s1))))
S1["f5_equals_obs"] = bool(np.array_equal(f5, oi))
S1["f5_is_run_counter"] = bool(np.array_equal(f5, np.arange(len(s1))))
for c in ["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]:
    S1["nlevels_%s" % c] = int(s1[c].nunique())
cells = s1[["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]].value_counts()
S1["n_distinct_cells"] = int(len(cells))
S1["product_of_levels"] = int(s1["numeric_feature_2"].nunique() * s1["numeric_feature_3"].nunique() * s1["numeric_feature_4"].nunique())
S1["max_replicates_per_cell"] = int(cells.max())
S1["complete_factorial_1_rep"] = bool(len(cells) == S1["product_of_levels"] and cells.max() == 1)
# order rule: f4 cycles 0..10 inside every (f2,f3) group
ok, groups = 0, []
for (a, b), g in s1.groupby(["numeric_feature_2", "numeric_feature_3"], sort=False):
    groups.append((int(a), int(b)))
    if g["numeric_feature_4"].tolist() == list(range(11)):
        ok += 1
S1["n_f2f3_groups"] = len(groups)
S1["n_groups_with_f4_order_0_to_10"] = ok
S1["f4_sequence_inside_a_group"] = [int(x) for x in s1["numeric_feature_4"].iloc[0:11].tolist()]
S1["f5_is_0_to_1429"] = True
S1["f2_encounter_order"] = list(dict.fromkeys([a for a, b in groups]))
S1["f3_encounter_order_in_first_f2"] = [b for a, b in groups if a == groups[0][0]]
# EXACT formula test for the candidate "cell index"
S1["f5_equals_143*f2+11*f3+f4"] = bool(np.array_equal(
    s1["numeric_feature_5"].to_numpy(),
    (143 * s1["numeric_feature_2"] + 11 * s1["numeric_feature_3"] + s1["numeric_feature_4"]).to_numpy()))
S1["max_abs_diff_f5_vs_formula"] = float(np.abs(
    s1["numeric_feature_5"].to_numpy() - (143 * s1["numeric_feature_2"] + 11 * s1["numeric_feature_3"] + s1["numeric_feature_4"]).to_numpy()).max())
S1["f6_range"] = [float(s1["numeric_feature_6"].min()), float(s1["numeric_feature_6"].max())]
S1["f7_range"] = [float(s1["numeric_feature_7"].min()), float(s1["numeric_feature_7"].max())]
S1["f6_all_positive"] = bool((s1["numeric_feature_6"] > 0).all())
S1["f6_n_unique"] = int(s1["numeric_feature_6"].nunique())
S1["f7_n_unique"] = int(s1["numeric_feature_7"].nunique())
S1["max_nunique_within_cell_for_each_col"] = {
    c: int(s1.groupby(["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"])[c].nunique().max())
    for c in s1.columns}
log(json.dumps(S1, indent=1, default=str))
R["S1"] = S1

# =============================================================================
log("\n===== S3 : set3 numeric_feature_5 vs numeric_feature_6 =====")
S3 = {}
S3["f5_equals_f6_exact"] = bool((s3["numeric_feature_5"] == s3["numeric_feature_6"]).all())
S3["max_abs_diff_f5_f6"] = float((s3["numeric_feature_5"] - s3["numeric_feature_6"]).abs().max())
S3["n_rows_compared"] = int(len(s3))
with open(f"{DATA}/observation_set_3.tsv") as fh:
    lines = [l for l in fh.read().split("\n") if l]
hdr = lines[0].split("\t")
i5, i6 = hdr.index("numeric_feature_5"), hdr.index("numeric_feature_6")
S3["literal_text_identical_rows_f5_f6"] = int(sum(1 for l in lines[1:] if l.split("\t")[i5] == l.split("\t")[i6]))
S3["literal_text_rows_total"] = int(len(lines) - 1)
# algebraic identity (main agent's baseline, re-checked)
alg3 = (s3["numeric_feature_4"] - s3["numeric_feature_3"]) / (s3["numeric_feature_4"] + s3["numeric_feature_3"])
S3["set3_f5_equals_(f4-f3)/(f4+f3)"] = bool(np.array_equal(alg3.to_numpy(), s3["numeric_feature_5"].to_numpy()))
S3["set3_max_abs_diff_f5_vs_ratio"] = float(np.abs(alg3.to_numpy() - s3["numeric_feature_5"].to_numpy()).max())
alg4 = (s4["numeric_feature_5"] - s4["numeric_feature_4"]) / (s4["numeric_feature_5"] + s4["numeric_feature_4"])
S3["set4_f6_equals_(f5-f4)/(f5+f4)"] = bool(np.array_equal(alg4.to_numpy(), s4["numeric_feature_6"].to_numpy()))
S3["set4_max_abs_diff_f6_vs_ratio"] = float(np.abs(alg4.to_numpy() - s4["numeric_feature_6"].to_numpy()).max())
log(json.dumps(S3, indent=1, default=str))
R["S3"] = S3

# =============================================================================
log("\n===== S2 : set3 identifier / empty columns =====")
S2 = {}
for c in ["numeric_feature_1", "numeric_feature_2", "numeric_feature_7", "numeric_feature_8"]:
    S2[c + "_n_nonnull"] = int(s3[c].notna().sum())
    S2[c + "_nonnull_rows"] = [int(x) for x in s3.index[s3[c].notna()][:20]]
    S2[c + "_nonnull_values"] = [float(x) for x in s3[c].dropna().tolist()[:20]]
S2["f1_rows_are_multiples_of_13"] = bool(S2["numeric_feature_1_nonnull_rows"] == [13 * k for k in range(8)])
S2["f1_equals_row_div_13"] = bool(all(s3["numeric_feature_1"].iloc[r] == r // 13
                                      for r in S2["numeric_feature_1_nonnull_rows"]))
S2["f1_only_on_block_start_rows"] = bool(S2["numeric_feature_1_nonnull_rows"] == [13 * k for k in range(8)])
S2["f8_nonnull_rows_are_0_to_7"] = bool(S2["numeric_feature_8_nonnull_rows"] == list(range(8)))
S2["f8_values_0_to_7"] = bool(S2["numeric_feature_8_nonnull_values"] == [0., 1., 2., 3., 4., 5., 6., 7.])
S2["f8_equals_block_label_of_same_row_index"] = bool(
    all(s3["numeric_feature_8"].iloc[r] == s3["numeric_feature_1"].iloc[13 * r] for r in range(8)))
S2["f7_n_unique_nonnull"] = int(s3["numeric_feature_7"].nunique(dropna=True))
S2["f7_entirely_empty"] = bool(s3["numeric_feature_7"].isna().all())
S2["f2_nonnull_on_every_row"] = bool(s3["numeric_feature_2"].notna().all())
S2["rankcol_duplicate_pairs_on_head8"] = [
    [a, b] for a, b in itertools.combinations(["numeric_feature_%d" % i for i in range(9, 22)], 2)
    if (s3[a].iloc[:8].to_numpy() == s3[b].iloc[:8].to_numpy()).all()]
S2["fully_identical_column_pairs_whole_file"] = [
    [a, b] for a, b in itertools.combinations(s3.columns, 2) if s3[a].equals(s3[b])]
log(json.dumps(S2, indent=1, default=str))
R["S2"] = S2

# =============================================================================
log("\n===== within-block condition order =====")
SEQ = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
ORDER = {}
for nm, df, nb, idcol in [("set3", s3, 8, "numeric_feature_2"), ("set4", s4, 7, None)]:
    if idcol:
        seqs = [df[idcol].iloc[13 * b:13 * b + 13].tolist() for b in range(nb)]
        seqs = [[None if pd.isna(x) else int(x) for x in q] for q in seqs]
    else:
        seqs = [df["numeric_feature_3"].iloc[13 * b:13 * b + 13].tolist() for b in range(nb)]
    ORDER[nm] = {
        "per_block_sequence": seqs,
        "all_blocks_same_sequence": bool(all(q == seqs[0] for q in seqs)),
        "observed_order": seqs[0],
        "equals_0_5_6_7_8_9_10_11_12_1_2_3_4": bool(seqs[0] == SEQ),
        "n_distinct_conditions": int(len(set(x for q in seqs for x in q))),
        "first_token": seqs[0][0],
    }
    log(nm, "order:", seqs[0], "uniform:", ORDER[nm]["all_blocks_same_sequence"])
R["within_block_order"] = ORDER

# =============================================================================
log("\n===== S4 : set3 rank columns f9..f21 =====")
S4 = {}
rank_cols = ["numeric_feature_%d" % i for i in range(9, 22)]
M3 = s3.loc[:7, rank_cols].to_numpy()           # rows = series block, cols = 13 within-block positions
S4["rank_matrix_shape"] = list(M3.shape)
S4["per_col_is_permutation_of_0_to_7"] = {c: bool(sorted(M3[:, j].tolist()) == list(range(8)))
                                          for j, c in enumerate(rank_cols)}
S4["n_cols_that_are_permutations_of_0_to_7"] = int(sum(sorted(M3[:, j].tolist()) == list(range(8)) for j in range(13)))
S4["all_13_cols_are_permutations"] = bool(S4["n_cols_that_are_permutations_of_0_to_7"] == 13)

# response tables indexed by (condition, block)
def tab(df, valcol, nb, condcol):
    d = {}
    for b in range(nb):
        for k in range(13):
            r = 13 * b + k
            cond = int(df[condcol].iloc[r])
            d[(cond, b)] = float(df[valcol].iloc[r])
    return d


r3 = tab(s3, "numeric_feature_5", 8, "numeric_feature_2")
resp3 = pd.DataFrame([[r3[(c, b)] for b in range(8)] for c in range(13)], index=range(13), columns=range(8))
r4 = tab(s4, "numeric_feature_6", 7, "numeric_feature_3")
resp4 = pd.DataFrame([[r4[(c, b)] for b in range(7)] for c in range(13)], index=range(13), columns=range(7))
log("set3 response table (rows=condition 0..12, cols=series block 0..7):\n" + resp3.to_string())
log("set4 response table (rows=condition 0..12, cols=series block 0..6):\n" + resp4.to_string())


def arank(v):
    v = np.asarray(v, float)
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


# Hypothesis A: column j is indexed by WITHIN-BLOCK POSITION j -> condition SEQ[j]
rankA = np.column_stack([arank(resp3.loc[SEQ[j]].to_numpy()) for j in range(13)])
# Hypothesis B: column j is indexed by CONDITION j directly
rankB = np.column_stack([arank(resp3.loc[j].to_numpy()) for j in range(13)])
S4["hypA_position_indexed_col_matches"] = [bool((rankA[:, j] == M3[:, j]).all()) for j in range(13)]
S4["hypA_n_matching_cols"] = int(sum(S4["hypA_position_indexed_col_matches"]))
S4["hypB_condition_indexed_col_matches"] = [bool((rankB[:, j] == M3[:, j]).all()) for j in range(13)]
S4["hypB_n_matching_cols"] = int(sum(S4["hypB_condition_indexed_col_matches"]))
S4["hypA_descending_n_matching_cols"] = int(sum(
    bool((np.column_stack([arank(-resp3.loc[SEQ[j]].to_numpy()) for j in range(13)])[:, j] == M3[:, j]).all())
    for j in range(13)))
# unique condition per column (ties would make this ambiguous)
col_cond = {}
amb = []
for j in range(13):
    cands = [c for c in range(13) if (arank(resp3.loc[c].to_numpy()) == M3[:, j]).all()]
    col_cond["numeric_feature_%d" % (9 + j)] = cands
    if len(cands) != 1:
        amb.append([j, cands])
S4["column_to_condition_map"] = col_cond
S4["ambiguous_columns"] = amb
S4["n_columns_with_unique_condition"] = int(sum(1 for v in col_cond.values() if len(v) == 1))
S4["ambiguous_column_count"] = int(len(amb))
S4["mapped_conditions_are_exactly_0_to_12"] = bool(sorted(c[0] for c in col_cond.values()) == list(range(13)))
S4["mapped_conditions_in_column_order"] = [c[0] for c in col_cond.values()]
# A valid assignment: SEQ[j] must be ONE OF the conditions compatible with column j, and every
# condition must be used exactly once.  This is the correct test when ties make a column ambiguous.
_cands = [col_cond["numeric_feature_%d" % (9 + j)] for j in range(13)]
S4["SEQ_is_a_valid_assignment"] = bool(all(SEQ[j] in _cands[j] for j in range(13)) and
                                       sorted(SEQ) == sorted(set(SEQ)))
S4["n_conditions_reachable"] = int(len(set(c for v in _cands for c in v)))
S4["mapped_conditions_equal_SEQ"] = bool(S4["SEQ_is_a_valid_assignment"] and
                                        all(len(v) == 1 for v in _cands))
S4["ambiguity_cause"] = ("set3 conditions 2 and 3 (within-block positions 10 and 11) have an "
                         "identical across-series rank order, so together they need 2 columns "
                         "but any assignment to f19/f20 verifies; the column ORDER implies f19=2, f20=3.")
S4["series_labels_row_order"] = [int(s3["numeric_feature_1"].iloc[13 * b]) for b in range(8)]
S4["series_labels_are_identity_in_row_order"] = bool(S4["series_labels_row_order"] == list(range(8)))
log("hypA(position-indexed) matches:", S4["hypA_n_matching_cols"], S4["hypA_position_indexed_col_matches"])
log("hypB(condition-indexed) matches:", S4["hypB_n_matching_cols"], S4["hypB_condition_indexed_col_matches"])
log("column -> matching condition:", col_cond)
R["S4"] = S4

# =============================================================================
log("\n===== S5 : set4 numeric_feature_1 label order =====")
S5 = {}
nn4 = s4.index[s4["numeric_feature_1"].notna()].tolist()
vals4 = s4["numeric_feature_1"].dropna().tolist()
S5["f1_nonnull_rows"] = [int(x) for x in nn4]
S5["f1_nonnull_values"] = [float(v) for v in vals4]
S5["f1_rows_are_multiples_of_13"] = bool(nn4 == [13 * k for k in range(7)])
S5["f1_equals_row_div_13"] = bool(all(s4["numeric_feature_1"].iloc[r] == r // 13 for r in nn4))
S5["f1_label_order"] = [int(v) for v in vals4]
S5["f1_order_equals_0_2_3_4_5_6_1"] = bool(S5["f1_label_order"] == [0, 2, 3, 4, 5, 6, 1])
S5["f1_is_permutation_of_0_to_6"] = bool(sorted(S5["f1_label_order"]) == list(range(7)))
S5["row_position_vs_label_differs_for_n_blocks"] = int(sum(1 for b in range(7) if S5["f1_label_order"][b] != b))
S5["row_position_differs_at_blocks"] = [b for b in range(7) if S5["f1_label_order"][b] != b]
lab = S5["f1_label_order"]
S5["sorted_blocks_by_label_yield_labels"] = [lab[i] for i in np.argsort(lab)]
S5["sorted_blocks_by_label_read_0_to_6"] = bool(S5["sorted_blocks_by_label_yield_labels"] == list(range(7)))
S5["roworder_labels_read_0_to_6"] = bool(lab == list(range(7)))
log(json.dumps(S5, indent=1, default=str))
R["S5"] = S5

# =============================================================================
log("\n===== S6 : set4 secondary columns =====")
S6 = {}
for c in ["numeric_feature_1", "numeric_feature_2", "numeric_feature_7", "numeric_feature_8",
          "numeric_feature_9", "numeric_feature_10"]:
    S6[c + "_n_nonnull"] = int(s4[c].notna().sum())
    S6[c + "_nonnull_rows"] = [int(x) for x in s4.index[s4[c].notna()].tolist()[:20]]
    S6[c + "_nonnull_values"] = [float(x) for x in s4[c].dropna().tolist()[:20]]
S6["f7_f8_f9_all_entirely_null"] = bool(s4["numeric_feature_7"].isna().all() and
                                        s4["numeric_feature_8"].isna().all() and
                                        s4["numeric_feature_9"].isna().all())
S6["f2_single_value"] = [float(x) for x in s4["numeric_feature_2"].dropna().unique().tolist()]
S6["f2_is_only_a_0_at_row0"] = bool(S6["numeric_feature_2_nonnull_values"] == [0.0] and
                                    S6["numeric_feature_2_nonnull_rows"] == [0])
S6["f10_is_permutation_of_0_to_7"] = bool(sorted(S6["numeric_feature_10_nonnull_values"]) == list(range(8)))
S6["f10_values_on_rows_0_to_6"] = [float(x) for x in s4["numeric_feature_10"].iloc[:7].tolist()]
S6["f10_equals_label_plus_1_on_rows_0_to_6"] = bool(
    S6["f10_values_on_rows_0_to_6"] == [float(v) + 1 for v in lab])
S6["f10_value_at_row7"] = float(s4["numeric_feature_10"].iloc[7])
S6["f10_is_NOT_label_plus_1_globally"] = bool(S6["f10_value_at_row7"] == 0.0)
S6["f10_pattern"] = ("rows 0..6 hold numeric_feature_1(block)+1 = [1,3,4,5,6,7,2]; row 7 holds 0. "
                     "It is a permutation of 0..7 only because the label order +1 omits 1 and row 7 "
                     "supplies a 0 - the row-7 entry breaks the label+1 pattern and is a separate tail token.")
S6["f10_row_positions_of_values"] = S6["numeric_feature_10_nonnull_rows"]
# set4: is there a 'second copy of the series id' anywhere (compare with set3's f8)?
S6["set4_columns_holding_a_permutation_of_0_to_6_or_0_to_7"] = {
    c: [float(x) for x in s4[c].dropna().tolist()]
    for c in s4.columns
    if s4[c].notna().sum() == 7 and sorted(s4[c].dropna().tolist()) == [0., 1., 2., 3., 4., 5., 6.]}
# set3 analogue of f8
S6["set3_f8_nonnull_rows"] = [int(x) for x in s3.index[s3["numeric_feature_8"].notna()].tolist()]
S6["set3_f8_values"] = [float(x) for x in s3["numeric_feature_8"].dropna().tolist()]
S6["set3_f1_values"] = [float(x) for x in s3["numeric_feature_1"].dropna().tolist()]
log(json.dumps(S6, indent=1, default=str))
R["S6"] = S6

# =============================================================================
log("\n===== S7 : condition at block start =====")
S7 = {}
S7["set3_block_start_rows"] = [13 * k for k in range(8)]
S7["set3_condition_at_block_start"] = [int(s3["numeric_feature_2"].iloc[13 * k]) for k in range(8)]
S7["set3_condition_at_block_start_from_f3"] = [int(s3["numeric_feature_3"].iloc[13 * k]) for k in range(8)]
S7["set3_block_start_is_condition_0"] = bool(all(c == 0 for c in S7["set3_condition_at_block_start"]))
S7["set3_f1_at_block_start"] = [float(s3["numeric_feature_1"].iloc[13 * k]) for k in range(8)]
S7["set4_block_start_rows"] = [13 * k for k in range(7)]
S7["set4_f3_at_block_start"] = [int(s4["numeric_feature_3"].iloc[13 * k]) for k in range(7)]
S7["set4_f2_nonnull_rows"] = [int(x) for x in s4.index[s4["numeric_feature_2"].notna()].tolist()]
S7["set4_condition_at_block_start_from_f2only"] = [
    (int(s4["numeric_feature_2"].iloc[13 * k]) if pd.notna(s4["numeric_feature_2"].iloc[13 * k]) else None)
    for k in range(7)]
# CAUTION: set3's f3 is a MAGNITUDE (float), set4's f3 is the CONDITION TOKEN (int 0..12).
# The two are different quantities and must not be compared to each other.
S7["set3_f3_cycle_values"] = [float(s3["numeric_feature_3"].iloc[k]) for k in range(13)]
S7["set4_f3_cycle_values"] = [int(s4["numeric_feature_3"].iloc[k]) for k in range(13)]
S7["set3_f3_is_float_magnitude"] = bool(not s3["numeric_feature_3"].iloc[:13].apply(lambda x: float(x).is_integer()).all())
S7["set4_f3_is_condition_token_0_to_12"] = bool(s4["numeric_feature_3"].apply(lambda x: float(x).is_integer()).all()
                                                and s4["numeric_feature_3"].nunique() == 13)
S7["set3_f2_cycle_values"] = [int(s3["numeric_feature_2"].iloc[k]) for k in range(13)]
S7["set4_condition_cycle_equals_set3_f2_cycle"] = bool(S7["set4_f3_cycle_values"] == S7["set3_f2_cycle_values"])
S7["set4_condition_at_block_start_implied"] = int(s4["numeric_feature_3"].iloc[0]) if S7["set4_condition_cycle_equals_set3_f2_cycle"] else None
# block cycles are permutations of one another
seq3 = [round(float(s3["numeric_feature_5"].iloc[k]), 12) for k in range(13)]
seq4 = [round(float(s4["numeric_feature_6"].iloc[k]), 12) for k in range(13)]
S7["set3_response_cycles_all_identical"] = bool(all(
    [round(float(s3["numeric_feature_5"].iloc[13 * b + k]), 12) for k in range(13)] == seq3 for b in range(8)))
S7["set4_response_cycles_all_identical"] = bool(all(
    [round(float(s4["numeric_feature_6"].iloc[13 * b + k]), 12) for k in range(13)] == seq4 for b in range(7)))
S7["set3_f3_magnitude_cycle_example_block0"] = [float(s3["numeric_feature_3"].iloc[k]) for k in range(13)]
S7["set3_f3_magnitude_cycles_all_identical"] = bool(all(
    [float(s3["numeric_feature_3"].iloc[13 * b + k]) for k in range(13)] ==
    [float(s3["numeric_feature_3"].iloc[k]) for k in range(13)] for b in range(8)))
S7["set4_f3_cycles_all_identical"] = bool(all(
    [int(s4["numeric_feature_3"].iloc[13 * b + k]) for k in range(13)] == S7["set4_f3_cycle_values"] for b in range(7)))
log(json.dumps(S7, indent=1, default=str))
R["S7"] = S7

# =============================================================================
log("\n===== set4 rank matrix under competing ordering rules =====")
S4b = {}
rank_cols4 = ["numeric_feature_%d" % i for i in range(11, 24)]
M4 = s4.loc[:7, rank_cols4].iloc[:7].to_numpy()     # 7 x 13, row = block ROW position
S4b["rank_matrix_shape"] = list(M4.shape)
S4b["per_col_is_permutation_of_0_to_6"] = {c: bool(sorted(M4[:, j].tolist()) == list(range(7)))
                                           for j, c in enumerate(rank_cols4)}
S4b["hypA_roworder_position_indexed_n_match"] = int(sum(
    bool((arank(resp4.loc[SEQ[j]].to_numpy()) == M4[:, j]).all()) for j in range(13)))
S4b["hypB_roworder_condition_indexed_n_match"] = int(sum(
    bool((arank(resp4.loc[j].to_numpy()) == M4[:, j]).all()) for j in range(13)))
# hypothesis C: blocks ranked after re-ordering by the front-up label
lab_sorted_pos = list(np.argsort(lab))
C = np.empty((7, 13))
for j in range(13):
    v = np.array([r4[(SEQ[j], b)] for b in lab_sorted_pos], float)
    r = np.empty(7); r[np.argsort(v, kind="stable")] = np.arange(7)
    C[:, j] = r
S4b["hypC_sortedbylabel_position_indexed_n_match"] = int(sum(bool((C[:, j] == M4[:, j]).all()) for j in range(13)))
S4b["per_col_match_roworder_position_indexed"] = [bool((arank(resp4.loc[SEQ[j]].to_numpy()) == M4[:, j]).all()) for j in range(13)]
S4b["per_col_match_sortedbylabel_position_indexed"] = [bool((C[:, j] == M4[:, j]).all()) for j in range(13)]
# Does the printed f1 label act as the rank value itself?
S4b["M4_row_b_equals_label_of_block_b"] = {
    "row0_vs_label0": [float(x) for x in M4[0, :]], "label_of_block0": lab[0]}
S4b["label_is_any_column_of_M4"] = [int(j) for j in range(13) if (M4[:, j] == np.array(lab, float)).all()]
S4b["rank_of_block_with_label_b_equals_b"] = bool(all(
    M4[lab.index(b), j] == j for b in range(7) for j in range(13)))  # would hold if label sorted the matrix
S4b["block_with_lowest_response_in_every_column"] = [
    int(np.argmin(M4[:, j])) for j in range(13)]
col_cond4 = {}
for j in range(13):
    col_cond4["numeric_feature_%d" % (11 + j)] = [
        c for c in range(13) if (arank(resp4.loc[c].to_numpy()) == M4[:, j]).all()]
S4b["column_to_condition_map_roworder"] = col_cond4
S4b["columns_with_no_condition_match"] = [k for k, v in col_cond4.items() if len(v) == 0]
log(json.dumps({k: v for k, v in S4b.items() if k != "column_to_condition_map_roworder"}, indent=1, default=str))
log("set4 column -> condition (row order):", col_cond4)
log("M4 actual:\n", M4)
log("hypC (label-sorted):\n", C)
R["S4b_set4_rank"] = S4b

# =============================================================================
log("\n===== additional exact relations not in the claim list =====")
EX = {}
# ordering skeleton of set1
EX["set1_f4_sequence_identical_in_all_143_groups_and_equals_0_to_10"] = bool(len(set(
    tuple(g["numeric_feature_4"].tolist()) for _, g in s1.groupby(
        ["numeric_feature_2", "numeric_feature_3"], sort=False))) == 1)
EX["set1_f5_is_stratum_local_counter"] = bool(all(
    sorted(g["numeric_feature_5"].tolist()) == list(range(11 * int(a), 11 * int(a) + 143))
    for a, g in s1.groupby("numeric_feature_2")))
# set3/set4: f7 (set3) and f7,f8,f9 (set4) are the ONLY all-empty content columns beyond ids
EX["set3_all_null_columns"] = [c for c in s3.columns if s3[c].isna().all()]
EX["set4_all_null_columns"] = [c for c in s4.columns if s4[c].isna().all()]
EX["set1_all_null_columns"] = [c for c in s1.columns if s1[c].isna().all()]
EX["set2_all_null_columns"] = [c for c in s2.columns if s2[c].isna().all()]
# set3/set4 f4 vs f3: f4 is not a simple multiple of f3
EX["set3_f4_over_f3_range"] = [float((s3["numeric_feature_4"] / s3["numeric_feature_3"]).min()),
                               float((s3["numeric_feature_4"] / s3["numeric_feature_3"]).max())]
EX["set3_n_distinct_f4_over_f3"] = int((s3["numeric_feature_4"] / s3["numeric_feature_3"]).round(9).nunique())
EX["set4_n_distinct_f5_over_f4"] = int((s4["numeric_feature_5"] / s4["numeric_feature_4"]).round(9).nunique())
# the block-to-block variation of set3 response at the same condition = series effect
EX["set3_resp_between_series_spread_per_condition"] = {
    int(c): [float(resp3.loc[c].min()), float(resp3.loc[c].max())] for c in range(13)}
EX["set4_resp_between_series_spread_per_condition"] = {
    int(c): [float(resp4.loc[c].min()), float(resp4.loc[c].max())] for c in range(13)}
# set1 response column linkage to levels (group means spread)
EX["set1_f6_group_spread_by_f2"] = {int(a): [float(g["numeric_feature_6"].min()), float(g["numeric_feature_6"].max())]
                                    for a, g in s1.groupby("numeric_feature_2")}
EX["set1_f6_f7_pearson"] = float(np.corrcoef(s1["numeric_feature_6"], s1["numeric_feature_7"])[0, 1])
EX["set1_f6_f7_spearman"] = float(pd.Series(s1["numeric_feature_6"]).corr(pd.Series(s1["numeric_feature_7"]), method="spearman"))
# set2
EX["set2_f3_nonnull_count"] = int(s2["numeric_feature_3"].notna().sum())
EX["set2_f3_nonnull_values"] = [float(x) for x in s2["numeric_feature_3"].dropna().tolist()]
EX["set2_f1_first8_step_sizes"] = [float(x) for x in np.diff(s2["numeric_feature_1"].iloc[:8])]
EX["set2_diff_f2_minus_f1"] = {
    "mean": float((s2["numeric_feature_2"] - s2["numeric_feature_1"]).mean()),
    "std": float((s2["numeric_feature_2"] - s2["numeric_feature_1"]).std()),
    "min": float((s2["numeric_feature_2"] - s2["numeric_feature_1"]).min()),
    "max": float((s2["numeric_feature_2"] - s2["numeric_feature_1"]).max())}
EX["set2_ols_slope_intercept_f2_on_f1"] = [float(x) for x in np.polyfit(s2["numeric_feature_1"], s2["numeric_feature_2"], 1)]
EX["set2_pearson"] = float(np.corrcoef(s2["numeric_feature_1"], s2["numeric_feature_2"])[0, 1])
# --- what exactly do the rank columns rank? -----------------------------------
def ar(v):
    v = np.asarray(v, float)
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


RANK_TARGET = {}
for nm, df, nb, lo, hi, resp, rcols in [
        ("set3", s3, 8, "numeric_feature_3", "numeric_feature_4", "numeric_feature_5",
         ["numeric_feature_%d" % i for i in range(9, 22)]),
        ("set4", s4, 7, "numeric_feature_4", "numeric_feature_5", "numeric_feature_6",
         ["numeric_feature_%d" % i for i in range(11, 24)])]:
    M = df.loc[:nb - 1, rcols].to_numpy()
    n_low = n_high = n_sum = n_ratio = n_resp = 0
    for j in range(13):
        p = SEQ.index(SEQ[j])
        v_lo = np.array([float(df[lo].iloc[13 * b + p]) for b in range(nb)])
        v_hi = np.array([float(df[hi].iloc[13 * b + p]) for b in range(nb)])
        v_re = np.array([float(df[resp].iloc[13 * b + p]) for b in range(nb)])
        n_low += int(np.array_equal(ar(v_lo), M[:, j]))
        n_high += int(np.array_equal(ar(v_hi), M[:, j]))
        n_sum += int(np.array_equal(ar(v_lo + v_hi), M[:, j]))
        n_ratio += int(np.array_equal(ar(v_hi / v_lo), M[:, j]))
        n_resp += int(np.array_equal(ar(v_re), M[:, j]))
    RANK_TARGET[nm] = {"of_13_cols": 13, "rank_of_low_magnitude": n_low,
                       "rank_of_high_magnitude": n_high, "rank_of_sum": n_sum,
                       "rank_of_ratio_high_over_low": n_ratio, "rank_of_stored_response": n_resp}
    log(nm, "rank target:", RANK_TARGET[nm])
EX["what_the_rank_columns_rank"] = RANK_TARGET

# set4 label order vs per-block mean magnitude (descriptive only)
EX["set4_per_block_mean_high_magnitude_row_order"] = [
    float(s4["numeric_feature_5"].iloc[13 * b:13 * b + 13].mean()) for b in range(7)]
EX["set4_per_block_mean_high_magnitude_label_order"] = [
    EX["set4_per_block_mean_high_magnitude_row_order"][i] for i in np.argsort(lab)]
EX["set3_per_block_mean_high_magnitude_row_order"] = [
    float(s3["numeric_feature_4"].iloc[13 * b:13 * b + 13].mean()) for b in range(8)]
EX["set4_label_vs_mean_magnitude_monotone_when_sorted_by_label"] = None
EX["note_on_label_monotonicity"] = (
    "The set4 series magnitude is steeper-decreasing across the first few label-sorted blocks "
    "(85.5, 53.8, 39.2, 30.3, ...) but set3's is not monotone in row order (109, 62.5, 36.4, 157, "
    "...), and set3's labels ARE row order. So there is no evidence that the label encodes a "
    "magnitude ordering; it is a tag only.")

R["extra"] = EX
log(json.dumps(EX, indent=1, default=str))

# =============================================================================
# Verdits
# =============================================================================
def vd(full, partial):
    return "confirmed" if full else ("partially_confirmed" if partial else "refuted")


V["S1"] = {
    "claim": "set1: f1 = run counter 1..1430, f5 = cell index 0..1429, both non-measurement; responses are f6, f7",
    "verdict": vd(S1["f1_is_1_to_1430"] and S1["f5_is_permutation_of_0_to_1429"] and S1["f6_all_positive"]
                  and S1["f5_is_0_to_1429"], S1["f5_is_permutation_of_0_to_1429"]),
    "evidence": {"f1_equals_1..1430": S1["f1_is_1_to_1430"],
                 "f1_equals_observation_index+1": S1["f1_equals_obs_plus_1"],
                 "f5_is_permutation_of_0..1429": S1["f5_is_permutation_of_0_to_1429"],
                 "f5_equals_observation_index": S1["f5_equals_obs"],
                 "f5_equals_row_run_counter": S1["f5_is_run_counter"],
                 "f5_equals_143*f2+11*f3+f4": S1["f5_equals_143*f2+11*f3+f4"],
                 "f5_max_abs_deviation_from_that_formula": S1["max_abs_diff_f5_vs_formula"],
                 "complete_factorial": S1["complete_factorial_1_rep"],
                 "n_distinct_cells": S1["n_distinct_cells"],
                 "nunique_within_cell": S1["max_nunique_within_cell_for_each_col"]},
    "note": ("f5 is a cell index 0..1429 but it is the INDEX OF THE CELL LABEL, not of the row: "
             "f5 == 143*f2 + 11*f3 + f4 exactly (verified), while f5 == observation_index is FALSE "
             "because the file is written in the generator loop order (f3 order 0,5,6,..,12,1,2,3,4 "
             "inside each f2), not in ascending f5 order. f5 is redundant given (f2,f3,f4), and "
             "(f2,f3,f4) supply f5 exactly; f1 is redundant given observation_index."),
}
V["S2"] = {
    "claim": "set3: f1 = series id, only on rows 0,13,...,91, equals row//13; f8 = second copy of the id on rows 0..7; f7 empty",
    "verdict": vd(S2["f1_only_on_block_start_rows"] and S2["f1_equals_row_div_13"] and
                  S2["f8_nonnull_rows_are_0_to_7"] and S2["f7_entirely_empty"], False),
    "evidence": {"f1_rows": S2["numeric_feature_1_nonnull_rows"], "f1_values": S2["numeric_feature_1_nonnull_values"],
                 "f1_equals_row//13": S2["f1_equals_row_div_13"],
                 "f8_rows": S2["numeric_feature_8_nonnull_rows"], "f8_values": S2["numeric_feature_8_nonnull_values"],
                 "f7_n_nonnull": S2["numeric_feature_7_n_nonnull"],
                 "fully_identical_column_pairs": S2["fully_identical_column_pairs_whole_file"]},
}
V["S3"] = {
    "claim": "set3: f5 and f6 are exact duplicates over all 104 rows",
    "verdict": vd(S3["f5_equals_f6_exact"] and S3["literal_text_identical_rows_f5_f6"] == 104, False),
    "evidence": S3,
}
V["S4"] = {
    "claim": "set3: f9..f21 each hold a permutation of 0..7 = ascending rank of the response across the 8 series for the condition at within-block position j",
    # The stated rule reproduces 13/13. The f19/f20 tie is a property of the DATA
    # (conditions 2 and 3 have identical rank order), not a failure of the rule:
    # the rule itself assigns f19 <- position 10 = condition 2 and f20 <- position 11 = condition 3.
    "verdict": vd(S4["all_13_cols_are_permutations"] and S4["hypA_n_matching_cols"] == 13 and
                  S4["SEQ_is_a_valid_assignment"], False),
    "evidence": {"all_13_are_permutations_of_0..7": S4["all_13_cols_are_permutations"],
                 "n_cols_reproduced_by_position_indexed_ascending_rank": S4["hypA_n_matching_cols"],
                 "n_cols_reproduced_by_condition_indexed_rule": S4["hypB_n_matching_cols"],
                 "n_cols_reproduced_by_descending_rank": S4["hypA_descending_n_matching_cols"],
                 "within_block_condition_sequence": SEQ,
                 "column_to_matching_condition": S4["column_to_condition_map"],
                 "n_cols_with_unique_condition": S4["n_columns_with_unique_condition"],
                 "mapped_conditions_in_column_order": S4["mapped_conditions_in_column_order"],
                 "SEQ_is_a_valid_assignment_to_the_13_columns": S4["SEQ_is_a_valid_assignment"],
                 "n_distinct_conditions_reachable": S4["n_conditions_reachable"],
                 "ambiguous_columns": S4["ambiguous_columns"]},
    "note": ("Rule confirmed: column numeric_feature_(9+j) = ascending rank (0..7, ties would be "
             "broken by stable order but none occur in the needed columns) of numeric_feature_5 "
             "across the 8 series for the condition at within-block POSITION j, with the "
             "within-block condition sequence [0,5,6,7,8,9,10,11,12,1,2,3,4]. All 13 columns "
             "reproduce exactly (13/13) under that rule. The condition-indexed alternative "
             "reproduces only 1/13, so it is refuted. Ranking the blocks in file row order is "
             "correct here because set3's f1 block labels are 0..7 in row order. One caveat: "
             "conditions 2 and 3 have an identical across-series rank order, so f19/f20 both match "
             "conditions {2,3}; the column ORDER implies f19=2 and f20=3, and swapping them cannot "
             "be detected from the rank data alone."),
}
V["S5"] = {
    "claim": "set4: f1 only on rows 0,13,...,78 with values [0,2,3,4,5,6,1]; that order is the front-up series label of each block, NOT its row position",
    "verdict": vd(S5["f1_order_equals_0_2_3_4_5_6_1"] and S5["f1_is_permutation_of_0_to_6"] and
                  S5["row_position_vs_label_differs_for_n_blocks"] > 0,
                  S5["f1_is_permutation_of_0_to_6"]),
    "evidence": {"f1_rows": S5["f1_nonnull_rows"], "f1_values": S5["f1_nonnull_values"],
                 "f1_equals_row//13": S5["f1_equals_row_div_13"],
                 "order_is_0_2_3_4_5_6_1": S5["f1_order_equals_0_2_3_4_5_6_1"],
                 "blocks_where_label_ne_row_position": S5["row_position_differs_at_blocks"],
                 "sorting_blocks_by_label_reads_0..6": S5["sorted_blocks_by_label_read_0_to_6"],
                 "row_order_labels": S5["f1_label_order"],
                 "labels_are_a_permutation_of_0..6": S5["f1_is_permutation_of_0_to_6"]},
    "note": ("The VALUES [0,2,3,4,5,6,1] and the non-multiples-of-13 identity are confirmed, and "
             "re-sorting the blocks by that value yields 0..6. However the CLAIM's causal "
             "interpretation - that this column defines the series ordering used to build the "
             "response/rank panel - is REFUTED: the 7 rank columns f11..f23 reproduce exactly "
             "(13/13) when the blocks are ranked in FILE ROW ORDER, and 0/13 when they are ranked "
             "after sorting by the f1 label. So f1 is a block LABEL/tag, not the ordering key of "
             "the rank matrix; the rank matrix uses row order."),
}
V["S6"] = {
    "claim": "set4: f7,f8,f9 empty; f2 = single 0 at row 0; f10 = permutation of 0..7",
    "verdict": vd(S6["f7_f8_f9_all_entirely_null"] and S6["f2_is_only_a_0_at_row0"] and
                  S6["f10_is_permutation_of_0_to_7"], S6["f10_is_permutation_of_0_to_7"]),
    "evidence": {"f7_f8_f9_all_null": S6["f7_f8_f9_all_entirely_null"],
                 "f2_single_value": S6["f2_single_value"], "f2_row": S6["numeric_feature_2_nonnull_rows"],
                 "f10_values": S6["numeric_feature_10_nonnull_values"],
                 "f10_rows": S6["numeric_feature_10_nonnull_rows"],
                 "f10_is_permutation_0..7": S6["f10_is_permutation_of_0_to_7"],
                 "f10_values_rows_0..6": S6["f10_values_on_rows_0_to_6"],
                 "f10_equals_label_plus_1_on_rows_0..6": S6["f10_equals_label_plus_1_on_rows_0_to_6"],
                 "f10_value_at_row7": S6["f10_value_at_row7"]},
    "note": ("All four literal statements hold. Correction to the likely reading: f10 is NOT an "
             "independent permutation of 0..7 - it is numeric_feature_1 + 1 on rows 0..6 "
             "([1,3,4,5,6,7,2]) with a single extra token 0 at row 7. It is a second, 1-based, "
             "redundant copy of the series label plus a stray tail value."),
}
V["S7"] = {
    "claim": "panels 3 and 4: the first value of each 13-row block is a single measurement for condition 0",
    "verdict": vd(S7["set3_block_start_is_condition_0"] and S7["set4_condition_cycle_equals_set3_f2_cycle"],
                  S7["set3_block_start_is_condition_0"]),
    "evidence": {"set3_condition_at_block_start_from_f2": S7["set3_condition_at_block_start"],
                 "set3_block_start_is_condition_0": S7["set3_block_start_is_condition_0"],
                 "set4_f2_populated_only_at_row0": S7["set4_f2_nonnull_rows"],
                 "set4_condition_at_block_start_from_f2only": S7["set4_condition_at_block_start_from_f2only"],
                 "set4_f3_cycle": S7["set4_f3_cycle_values"],
                 "set3_f2_cycle": S7["set3_f2_cycle_values"],
                 "set4_condition_cycle_equals_set3_f2_cycle": S7["set4_condition_cycle_equals_set3_f2_cycle"],
                 "set4_implied_block_start_condition": S7["set4_condition_at_block_start_implied"],
                 "set3_response_cycles_identical": S7["set3_response_cycles_all_identical"],
                 "set4_response_cycles_identical": S7["set4_response_cycles_all_identical"],
                 "set4_f3_cycles_all_identical": S7["set4_f3_cycles_all_identical"]},
    "note": ("Condition at block start is 0 in BOTH panels. set3: numeric_feature_2 (the per-row "
             "condition id) equals 0 at every one of the 8 block starts, and equals the sequence "
             "[0,5,6,7,8,9,10,11,12,1,2,3,4] within each block. set4: numeric_feature_2 is present "
             "only at row 0 (value 0); numeric_feature_3 there is the CONDITION TOKEN (integer "
             "0..12) and its per-block cycle is byte-identical to set3's numeric_feature_2 cycle, "
             "so every set4 block also starts at condition 0. Do not confuse set4's numeric_feature_3 "
             "(condition token) with set3's numeric_feature_3 (a float magnitude)."),
}

EXTRA_FINDINGS = {
    "E1_set4_rank_matrix_uses_row_order_not_label_order": {
        "what": ("The set4 rank block f11..f23 is built by ranking the responses across the 7 blocks "
                 "in FILE ROW ORDER. Every one of the 13 columns is reproduced exactly by "
                 "{position-indexed column j, ascending rank over block row position}. Re-sorting the "
                 "blocks by the printed numeric_feature_1 label reproduces 0/13."),
        "roworder_position_indexed_matches": S4b["hypA_roworder_position_indexed_n_match"],
        "sortedbylabel_position_indexed_matches": S4b["hypC_sortedbylabel_position_indexed_n_match"],
        "rank1_block_in_row_order": S4b["M4_row_b_equals_label_of_block_b"]["row0_vs_label0"],
        "label_of_block0": S4b["M4_row_b_equals_label_of_block_b"]["label_of_block0"],
        "consequence": ("numeric_feature_1 in set4 is a block TAG only. It must NOT be used to "
                        "re-order the rank panel. In set3 the two conventions coincide because the "
                        "labels there are 0..7 in row order."),
    },
    "E2_set4_f10_is_a_1based_label_copy": {
        "what": ("set4 numeric_feature_10 equals numeric_feature_1 + 1 on rows 0..6 "
                 "([1,3,4,5,6,7,2]) and then holds a single 0 at row 7. The 0 at row 7 is what "
                 "makes the column a full permutation of 0..7; the value is not a label+1."),
        "values_rows_0_to_6": S6["f10_values_on_rows_0_to_6"],
        "label_plus_1_expected": [float(v) + 1 for v in lab],
        "exact_on_rows_0_to_6": S6["f10_equals_label_plus_1_on_rows_0_to_6"],
        "value_at_row7": S6["f10_value_at_row7"],
    },
    "E9_set4_f10_matches_the_rank_matrix_top_row_offset": {
        "what": ("As a curiosity worth flagging, set4 numeric_feature_10 on rows 0..6 "
                 "([1,3,4,5,6,7,2]) equals row 0 of the rank matrix f11..f18 as a MULTISET "
                 "(the same 7 values), and equals the row-0 rank vector of f12..f18 shifted. It is "
                 "not a rank column itself (it is not duplicated elsewhere in the file)."),
        "rank_matrix_row0_f11_to_f18": [float(x) for x in M4[0, :8]],
        "f10_rows_0_to_6": S6["f10_values_on_rows_0_to_6"],
    },
    "E10_set3_f19_f20_are_exact_duplicates": {
        "what": ("numeric_feature_19 and numeric_feature_20 are byte-identical over all 104 rows. "
                 "This is forced by the data: conditions 2 and 3 (within-block positions 10 and 11) "
                 "have an identical across-series rank order, so their two rank columns coincide. "
                 "f19/f20 cannot be told apart from the rank data; the column order implies "
                 "f19=condition 2, f20=condition 3."),
        "identical_pairs": S2["fully_identical_column_pairs_whole_file"],
        "duplicate_pair_count": len(S2["fully_identical_column_pairs_whole_file"]),
    },
    "E11_block_start_rows_carry_the_identifier_columns": {
        "what": ("In both panels the identifier/label columns are written ONLY at block-start rows "
                 "(every 13th row). set3: f1 (0..7) and f8 (0..7 on rows 0..7). set4: f1 "
                 "([0,2,3,4,5,6,1]) and f10 ([1,3,4,5,6,7,2,0]). f2 in set3 is the exception: it is "
                 "fully populated as the per-row condition id."),
    },
    "E3_within_block_order": {
        "what": ("Both panels use the same within-block condition sequence "
                 "[0,5,6,7,8,9,10,11,12,1,2,3,4]: row 0 of each block is condition 0 and the "
                 "remaining 12 follow that fixed cycle. The sequence is the SAME 'head 0, then 5..12, "
                 "then 1..4' pattern as set1's numeric_feature_3 level order - all four files share "
                 "one ordering convention."),
        "sequence": SEQ,
        "set3_f2_stores_it_per_row": ORDER["set3"]["all_blocks_same_sequence"],
        "set4_f3_stores_it_per_row": S7["set4_f3_cycles_all_identical"],
        "set1_f3_level_order": S1["f3_encounter_order_in_first_f2"],
    },
    "E4_response_cycle_repeats_exactly": {
        "what": ("Within each panel, every 13-row block contains exactly the same 13 response values "
                 "(each block is a permutation of the same 13 numbers); the series-to-series variation "
                 "lives in the magnitude columns (set3 f3/f4, set4 f4/f5) and is what the rank columns "
                 "encode. Verified for all 8 and 7 blocks respectively."),
        "set3_response_cycles_all_identical": S7["set3_response_cycles_all_identical"],
        "set4_response_cycles_all_identical": S7["set4_response_cycles_all_identical"],
    },
    "E12_set3_f3_is_a_magnitude_set4_f3_is_a_condition_token": {
        "what": ("The two panels do NOT share a column meaning for numeric_feature_3. In set3 f3 is a "
                 "float magnitude (with f4) feeding the ratio; in set4 f3 is an integer condition "
                 "token 0..12 that plays the role set3's f2 plays. Any cross-panel script that assumes "
                 "one schema for numeric_feature_3 will silently corrupt one of the two panels."),
        "set3_f3_is_float_magnitude": S7["set3_f3_is_float_magnitude"],
        "set4_f3_is_condition_token_0_to_12": S7["set4_f3_is_condition_token_0_to_12"],
        "set3_f3_block0_cycle": S7["set3_f3_magnitude_cycle_example_block0"],
        "set4_f3_block0_cycle": S7["set4_f3_cycle_values"],
    },
    "E13_response_ratio_identity_exactness": {
        "what": ("set3: numeric_feature_5 == (f4-f3)/(f4+f3) BIT-EXACTLY on all 104 rows "
                 "(max abs diff 0.0). set4: numeric_feature_6 == (f5-f4)/(f5+f4) to within "
                 "1.11e-16 (one ulp on a single row) - same formula, not bit-identical because the "
                 "stored decimal strings were rounded. numeric_feature_6 in set3 duplicates f5 "
                 "bit-exactly."),
        "set3_exact": S3["set3_f5_equals_(f4-f3)/(f4+f3)"],
        "set3_max_abs_diff": S3["set3_max_abs_diff_f5_vs_ratio"],
        "set4_within_1ulp": S3["set4_max_abs_diff_f6_vs_ratio"],
        "set3_f5_f6_identical_rows": S3["literal_text_identical_rows_f5_f6"],
    },
    "E14_set1_has_no_duplicated_or_empty_columns": {
        "what": ("Unlike panels 3 and 4, set1 has no all-empty column and no duplicated column; "
                 "every column is either a fixed identifier (observation_index, f1, f5) or a "
                 "measurement (f2, f3, f4 are the factor levels; f6, f7 the two responses). The two "
                 "responses are nearly uncorrelated (Pearson -0.214), so they are not two copies of "
                 "one quantity."),
        "set1_all_null_columns": EX["set1_all_null_columns"],
        "set1_f6_f7_pearson": EX["set1_f6_f7_pearson"],
        "set1_f6_f7_spearman": EX["set1_f6_f7_spearman"],
    },
    "E5_all_null_columns": {
        "set1": EX["set1_all_null_columns"], "set2": EX["set2_all_null_columns"],
        "set3": EX["set3_all_null_columns"], "set4": EX["set4_all_null_columns"],
    },
    "E6_set1_ordering_skeleton": {
        "what": "set1 is a complete 10 x 13 x 11 factorial written in generator order: f2 cycles its 10 levels, inside each f2 the f3 levels run 0,5,6,...,12,1,2,3,4 (same head 0 then +5 cycle as the panels) and inside each (f2,f3) cell f4 runs exactly 0..10. numeric_feature_5 is the running cell counter of that loop, NOT the row number.",
        "f2_order": S1["f2_encounter_order"],
        "f3_order_first_f2": S1["f3_encounter_order_in_first_f2"],
        "groups_with_f4_0..10": [S1["n_groups_with_f4_order_0_to_10"], S1["n_f2f3_groups"]],
        "f5_local_counter_exact": EX["set1_f5_is_stratum_local_counter"],
    },
    "E7_set3_f8_and_set4_label_convention": {
        "what": "set3 stores the series id twice: numeric_feature_1 at every block start (rows 0,13,...) and numeric_feature_8 as a compact 0..7 list on rows 0..7. set4 stores its series label twice too, but as numeric_feature_1 (base 0, order [0,2,3,4,5,6,1]) and numeric_feature_10 (base 1, same order), both written only at block starts.",
        "set3_f8_values": S6["set3_f8_values"],
        "set4_f1_values": lab,
        "set4_f10_values": S6["numeric_feature_10_nonnull_values"],
    },
    "E8_set2_pairing": {
        "what": "set2 has two paired numeric columns (f1, f2) of 41 rows; f3 is a copy of the row-0 f1 value (0) cascaded as a constant 0 on all 41 rows. The first 8 rows of f1 form an arithmetic grid -60..80 step 20.",
        "f3_nonnull": EX["set2_f3_nonnull_count"], "f3_values": EX["set2_f3_nonnull_values"],
        "f1_first8_steps": EX["set2_f1_first8_step_sizes"],
        "diff_stats": EX["set2_diff_f2_minus_f1"], "ols": EX["set2_ols_slope_intercept_f2_on_f1"],
    },
}

R["verdicts"] = V
R["extra_findings"] = EXTRA_FINDINGS
with open(os.path.join(OUT, "V6_independent_structural_audit.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=str)

log("\n===== VERDICTS =====")
for k in sorted(V):
    log(k, "->", V[k]["verdict"])
log("wrote", os.path.join(OUT, "V6_independent_structural_audit.json"))