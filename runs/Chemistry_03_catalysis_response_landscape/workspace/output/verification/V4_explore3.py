"""V4 step 4: brute-force search for what set4's supplied rank block (f11..f23) encodes."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

D = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/"
s3 = pd.read_csv(D + "observation_set_3.tsv", sep="\t")
s4 = pd.read_csv(D + "observation_set_4.tsv", sep="\t")
ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]

R3 = s3[s3.columns[9:22]].values[:8]     # 8 x 13, set3 supplied rank block
R4 = s4[s4.columns[11:24]].values[:7]    # 7 x 13, set4 supplied rank block


def cells(df, condcol, sercol, cols):
    rec = []
    for b in range(len(df) // 13):
        sub = df.iloc[b * 13:(b + 1) * 13]
        ser = sub[sercol].dropna().iloc[0]
        for _, row in sub.iterrows():
            rec.append([int(ser), int(row[condcol])] + [float(row[c]) for c in cols])
    return pd.DataFrame(rec, columns=["series", "condition"] + list(cols))


c3 = cells(s3, "numeric_feature_2", "numeric_feature_1", ["numeric_feature_3", "numeric_feature_4", "numeric_feature_5"])
c4 = cells(s4, "numeric_feature_3", "numeric_feature_1", ["numeric_feature_4", "numeric_feature_5", "numeric_feature_6"])

print("=== A) does R4 match R3 columns (shared series 0..6)? ===")
same = []
for j in range(13):
    for k in range(13):
        if np.allclose(R4[:, j], R3[:7, k]):
            same.append((j, k))
print("exact matching (R4 col j == R3 col k) pairs:", same)
best = []
for j in range(13):
    for k in range(13):
        r = spearmanr(R4[:, j], R3[:7, k]).statistic
        best.append((r, j, k))
best.sort(reverse=True)
print("top-5 Spearman pairs R4 col vs R3 col:", [(round(r, 3), j, k) for r, j, k in best[:5]])

print("\n=== B) search over candidate quantities in set4 for each R4 column ===")
cand = {}
for cond, g in c4.groupby("condition"):
    g = g.sort_values("series")
    v4, v5, v6 = g.numeric_feature_4.values, g.numeric_feature_5.values, g.numeric_feature_6.values
    cand[("f4", cond)] = v4
    cand[("f5", cond)] = v5
    cand[("f6", cond)] = v6
    cand[("logf4", cond)] = np.log(v4)
    cand[("logf5", cond)] = np.log(v5)
    cand[("diff", cond)] = v5 - v4
    cand[("ratio56", cond)] = v5 / v4
print("candidate keys:", {k[0] for k in cand})

for j in range(13):
    hits = []
    for (nm, cond), v in cand.items():
        if np.allclose(rankdata(v) - 1, R4[:, j]):
            hits.append((nm, cond))
    if hits:
        print(f"  R4 col j={j:2d} (blockcond {ORDER[j]:2d}) EXACT match: {hits}")

print("\n  (rank-correlation view, max |rho| per column)")
for j in range(13):
    bs = []
    for (nm, cond), v in cand.items():
        bs.append((spearmanr(v, R4[:, j]).statistic, nm, cond))
    bs.sort()
    print(f"  j={j:2d}: best {[(round(r,3), nm, cond) for r, nm, cond in bs[-3:]]}")

print("\n=== C) is R4 consistent with a rank among 7 series of ANY per-condition scalar? ===")
# For each R4 column, check the implied ordering against f6 ordering per condition
for j in range(13):
    implied = np.argsort(R4[:, j])   # series sorted from lowest rank to highest
    for cond in range(13):
        g = c4[c4.condition == cond].sort_values("series")
        ord_f6 = np.argsort(g.numeric_feature_6.values)
        if np.array_equal(implied, ord_f6):
            print(f"  j={j} equals ordering of f6 under condition {cond}")
        ord_f5 = np.argsort(g.numeric_feature_5.values)
        if np.array_equal(implied, ord_f5):
            print(f"  j={j} equals ordering of f5 under condition {cond}")
        ord_f4 = np.argsort(g.numeric_feature_4.values)
        if np.array_equal(implied, ord_f4):
            print(f"  j={j} equals ordering of f4 under condition {cond}")

print("\n=== D) exact reproduction attempt: R4 vs R3 with 7-series-subset columns ===")
# maybe R4 == R3 restricted to some 7 of the 8 series, some column permutation
for drop in range(8):
    keep = [i for i in range(8) if i != drop]
    match = sum(1 for j in range(13) if any(np.allclose(R4[:, j], R3[keep, k]) for k in range(13)))
    print(f"  drop series {drop}: {match}/13 columns exact-match R3")

print("\n=== E) set4 rank-block row7 value across columns ===")
R4f = s4[s4.columns[11:24]].values
print("row 7 :", R4f[7])
print("non-null rows of set4 rank cols:", np.where(~np.isnan(R4f[:, 0]))[0])
print("non-null rows of set3 rank cols:", np.where(~np.isnan(R3[:, 0]))[0])
print("full shape set4 rank f11..f23:", R4f.shape, " set3 f9..f21:", R3.shape)