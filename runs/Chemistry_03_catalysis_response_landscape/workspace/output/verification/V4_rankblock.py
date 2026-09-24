"""V4 step 2: determine empirically what the supplied rank blocks mean.

set3: 8 blocks x 13 conditions. rank block candidates numeric_feature_8..21 (non-null rows 0..7 only)
set4: 7 blocks x 13 conditions. rank block candidates numeric_feature_10..23 (non-null rows 0..7 only)

Hypothesis: column j of the rank block gives the across-series rank of the response under the
condition that occupies block-position j.  We verify by matching against ranks computed directly
from the response columns.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/"
s3 = pd.read_csv(D + "observation_set_3.tsv", sep="\t")
s4 = pd.read_csv(D + "observation_set_4.tsv", sep="\t")

ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]   # observed within-block row order (both panels)


def cells(df, condcol, sercol, resp):
    """long cell table: series, condition, response"""
    rec = []
    for b in range(len(df) // 13):
        sub = df.iloc[b * 13:(b + 1) * 13]
        ser = sub[sercol].dropna().iloc[0]            # series index only on first row of block
        for _, r in sub.iterrows():
            rec.append((int(ser), int(r[condcol]), float(r[resp])))
    return pd.DataFrame(rec, columns=["series", "condition", "response"])


c3 = cells(s3, "numeric_feature_2", "numeric_feature_1", "numeric_feature_5")
c4 = cells(s4, "numeric_feature_3", "numeric_feature_1", "numeric_feature_6")
print("cells3", c3.shape, "series", sorted(c3.series.unique()), "conds", sorted(c3.condition.unique()))
print("cells4", c4.shape, "series", sorted(c4.series.unique()), "conds", sorted(c4.condition.unique()))
print("resp3 == (f4-f3)/(f4+f3) exact:",
      bool(np.allclose(s3.numeric_feature_5, (s3.numeric_feature_4 - s3.numeric_feature_3) / (s3.numeric_feature_4 + s3.numeric_feature_3))))
print("resp4 == (f5-f4)/(f5+f4) exact:",
      bool(np.allclose(s4.numeric_feature_6, (s4.numeric_feature_5 - s4.numeric_feature_4) / (s4.numeric_feature_5 + s4.numeric_feature_4))))


def computed_rank_cols(cells_df, nser):
    cols = {}
    for cond, grp in cells_df.groupby("condition"):
        v = grp.sort_values("series")["response"].values
        assert len(v) == nser
        cols[cond] = rankdata(v, method="average").astype(float)   # 1-based
    return cols


def supplied_block(df, first_col, ncols, nrows):
    """return matrix nrows x ncols of the supplied rank block"""
    cols = list(df.columns[first_col:first_col + ncols])
    return cols, df[cols].values[:nrows]


def match_columns(supplied, computed, nser, nconds, label):
    """Find, for each supplied column, which condition's computed rank vector it equals (0-based ranks)."""
    sup0 = supplied - 1.0     # convert to 0-based if 1-based
    mapping = {}
    for j in range(nconds):
        hits = [c for c in range(nconds) if np.allclose(sup0[:, j], computed[c], atol=1e-9)]
        mapping[j] = hits
    print(f"\n--- {label}: supplied col j (block position {ORDER[j]}) -> matching conditions (0-based ranks) ---")
    for j, h in mapping.items():
        print(f"  j={j:2d} (blockcond {ORDER[j]:2d}) -> conditions {h}")
    # also try 0-based supplied
    mapping0 = {}
    for j in range(nconds):
        hits = [c for c in range(nconds) if np.allclose(supplied[:, j], computed[c], atol=1e-9)]
        mapping0[j] = hits
    print(f"--- {label}: same but supplied treated as already 0-based ---")
    for j, h in mapping0.items():
        print(f"  j={j:2d} -> {h}")
    return mapping, mapping0


# ---- set3: 8 series, ranks 0..7 ----
cols3, sup3 = supplied_block(s3, 8, 13, 8)   # f8 is series index; f9..f21 are the 13 rank cols
print("\nset3 supplied rank cols:", cols3)
comp3 = computed_rank_cols(c3, 8)
m3_1, m3_0 = match_columns(sup3, comp3, 8, 13, "set3 (f9..f21)")

# ---- set4: 7 series ----
cols4, sup4 = supplied_block(s4, 10, 13, 8)
print("\nset4 supplied rank cols:", cols4)
print("set4 supplied first 8 rows:\n", sup4)
print("unique values per set4 rank column:", [sorted(set(sup4[:, j])) for j in range(13)])
print("set4 rank-block row 7:", sup4[7, :])
print("set4 f10 (rows0..7):", s4.numeric_feature_10.values[:8])
comp4 = computed_rank_cols(c4, 7)
# try rows 0..6 only
m4_1, m4_0 = match_columns(sup4[:7, :], comp4, 7, 13, "set4 rows0-6 (f11..f23)")