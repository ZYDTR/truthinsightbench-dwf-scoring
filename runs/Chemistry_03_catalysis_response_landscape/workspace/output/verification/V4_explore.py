"""V4 exploration: determine raw layout of panels 1,3,4 before the formal checks."""
import numpy as np
import pandas as pd

D = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 120)

s1 = pd.read_csv(D + "observation_set_1.tsv", sep="\t")
s2 = pd.read_csv(D + "observation_set_2.tsv", sep="\t")
s3 = pd.read_csv(D + "observation_set_3.tsv", sep="\t")
s4 = pd.read_csv(D + "observation_set_4.tsv", sep="\t")

print("shapes", s1.shape, s2.shape, s3.shape, s4.shape)

# ---- panel 1 ----
print("\n=== panel1 level counts ===")
for c in ["numeric_feature_2", "numeric_feature_3", "numeric_feature_4", "numeric_feature_5"]:
    print(c, sorted(s1[c].unique()))
print("nf1 range", s1.numeric_feature_1.min(), s1.numeric_feature_1.max())
print(s1[["numeric_feature_6", "numeric_feature_7"]].describe())
# completeness of the 3-way factorial
g = s1.groupby(["numeric_feature_2", "numeric_feature_3", "numeric_feature_4"]).size()
print("cell sizes unique:", g.unique(), "n cells", len(g))

# ---- panel 3 blocks ----
print("\n=== panel3 f1 per row ===")
print(s3["numeric_feature_1"].dropna().to_dict())
print("f2 (candidate condition) first 40:", s3.numeric_feature_2.values[:40])
print("f2 range", s3.numeric_feature_2.min(), s3.numeric_feature_2.max())

# block structure guess: 13-row blocks
def block_report(df, condcol, name):
    n = len(df)
    print(f"\n--- {name}: {n} rows, condcol={condcol} ---")
    orders = []
    for b in range(n // 13):
        sub = df.iloc[b * 13:(b + 1) * 13]
        order = list(sub[condcol].astype(int).values)
        orders.append(order)
        print(f"block {b}: {order}  perm={sorted(order)==list(range(13))}")
    same = all(o == orders[0] for o in orders)
    print("all blocks same order:", same, "order0 =", orders[0])
    return orders

o3 = block_report(s3, "numeric_feature_2", "set3")
o4 = block_report(s4, "numeric_feature_3", "set4")
print("\npanel3 order:", o3[0])
print("panel4 order:", o4[0])
print("same order? ", o3[0] == o4[0])

# rank-block permutation structure
print("\n=== set3 rank block candidate columns f8..f21 over rows 0..7 ===")
for c in s3.columns[8:]:
    v = s3[c].values[:8]
    print(c, v, "perm0..7" if sorted(v) == list(range(8)) else "NOT-perm")
print("\n=== set4 rank block candidate columns f9..f23 over rows 0..7 ===")
for c in s4.columns[9:]:
    v = s4[c].values[:8]
    print(c, v, "perm0..7" if sorted(v) == list(range(8)) else ("perm0..6(rows0-6)" if sorted(v[:7]) == list(range(7)) else "neither"))