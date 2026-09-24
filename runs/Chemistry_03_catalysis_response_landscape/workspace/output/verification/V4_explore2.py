"""V4 step 3: what do the supplied rank blocks in set3 and set4 encode?"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = "/data/user/conversations/conv-4ba01bd5-0b72-45b2-94f6-e776171d414a/project/data/"
s3 = pd.read_csv(D + "observation_set_3.tsv", sep="\t")
s4 = pd.read_csv(D + "observation_set_4.tsv", sep="\t")
ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]


def cells(df, condcol, sercol, respcols):
    rec = []
    for b in range(len(df) // 13):
        sub = df.iloc[b * 13:(b + 1) * 13]
        ser = sub[sercol].dropna().iloc[0]
        for _, row in sub.iterrows():
            rec.append([int(ser), int(row[condcol])] + [float(row[c]) for c in respcols])
    return pd.DataFrame(rec, columns=["series", "condition"] + list(respcols))


c3 = cells(s3, "numeric_feature_2", "numeric_feature_1", ["numeric_feature_3", "numeric_feature_4", "numeric_feature_5"])
c4 = cells(s4, "numeric_feature_3", "numeric_feature_1", ["numeric_feature_4", "numeric_feature_5", "numeric_feature_6"])

R3 = s3[s3.columns[9:22]].values[:8]
R4 = s4[s4.columns[11:24]].values[:8]
print("R3=\n", R3)
print("R4=\n", R4)

# rank candidates from set3 cell table
def rankmat(cellsdf, series_list, col, log=False):
    out = {}
    for cond, g in cellsdf.groupby("condition"):
        v = g.sort_values("series")[col].values
        if log:
            v = np.log(v)
        out[cond] = rankdata(v, method="average") - 1.0
    return out


for label, cm, df_r, ref in [("set3", c3, R3, ["numeric_feature_3", "numeric_feature_4", "numeric_feature_5"]),
                             ("set4", c4, R4, ["numeric_feature_4", "numeric_feature_5", "numeric_feature_6"])]:
    print(f"\n===== {label} =====")
    for col in ref:
        for log in ([False, True] if col not in ("numeric_feature_5", "numeric_feature_6") else [False, True]):
            rm = rankmat(cm, None, col, log=log)
            hits = []
            for j in range(13):
                h = [c for c in range(13) if np.allclose(df_r[:len(rm[0]) + 0, j], rm[c]) if True]
                hits.append(h)
            # above would break; redo properly below
    # proper: match column j against all conditions
    results = {}
    for col in ref:
        for log in [False, True]:
            rm = rankmat(cm, None, col, log=log)
            nser = len(next(iter(rm.values())))
            tab = []
            for j in range(13):
                h = [c for c in range(13) if np.allclose(df_r[:nser, j], rm[c])]
                tab.append(h)
            results[(col, log)] = tab
            nmatch = sum(1 for t in tab if t)
            print(f"  col={col} log={log}: cols with >=1 match: {nmatch}/13")
            print("   ", tab)

# set3 tie inspection for cond2/cond3
for cond in [2, 3]:
    g = c3[c3.condition == cond].sort_values("series")
    print(f"\nset3 cond{cond} response by series:", np.round(g.numeric_feature_5.values, 6))
    print(f"set3 cond{cond} rank vector:", rankdata(g.numeric_feature_5.values) - 1)

# what is set4 row 7?  compare with grand means
print("\nset4 series-present rows per block:", [(b, s4.numeric_feature_1.values[b * 13]) for b in range(7)])
print("set4 block0 f4 (mag low):", np.round(s4.numeric_feature_4.values[:8], 4))
print("set4 block0 f5 (mag high):", np.round(s4.numeric_feature_5.values[:8], 4))
print("set4 block0 resp:", np.round(s4.numeric_feature_6.values[:8], 6))
print("mean of rows0..6 resp:", s4.numeric_feature_6.values[:7].mean(), " row7:", s4.numeric_feature_6.values[7])
print("set4 blocks: is row 7 of each block the 8th row? len per block/13:", len(s4) / 13)