"""Adversarial probe 3: prose-claim arithmetic + raw-string/hex identity checks."""
from __future__ import annotations
import json, struct
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
D = P / "data"
O = {}

uni = pd.read_csv(D / "uniform_alternative_comparison.tsv", sep="\t").sort_values("uniform_velocity_cm_s").reset_index(drop=True)
cand = pd.read_csv(D / "candidate_spectrum_comparison.tsv", sep="\t")
rank = pd.read_csv(D / "ranked_full_parameter_grid.tsv", sep="\t")
rank_ic = rank.delta_information_criterion.to_numpy(float)
cand_ic = cand.information_criterion.to_numpy(float)
MV = cand[cand.candidate_family == "minimum_velocity_scan"]
diag = MV[MV.max_velocity_cm_s == MV.min_velocity_cm_s]

# --- raw-string / hex identity of the 19 matched IC values
uni_raw = {}
for line in (D / "uniform_alternative_comparison.tsv").read_text().strip().split("\n")[1:]:
    a, b = line.split("\t"); uni_raw[float(a)] = b
cand_raw = {}
for line in (D / "candidate_spectrum_comparison.tsv").read_text().strip().split("\n")[1:]:
    p = line.split("\t")
    if p[0] == "minimum_velocity_scan":
        cand_raw[(float(p[1]), float(p[2]))] = (p[3], p[4], p[5])
hex_pairs = []
for _, r in diag.iterrows():
    vmin = float(r.min_velocity_cm_s)
    if vmin in uni_raw:
        us, cs = uni_raw[vmin], cand_raw[(float(r.max_velocity_cm_s), vmin)][2]
        hex_pairs.append({"v": vmin, "uni_str": us, "cand_str": cs, "str_equal": us == cs,
                          "hex_equal": struct.pack(">d", float(us)) == struct.pack(">d", float(cs)),
                          "f_uni": 1210.0, "w_uni": 300.0})
O["hex_pairs"] = hex_pairs
O["n_str_equal"] = int(sum(h["str_equal"] for h in hex_pairs))
O["n_hex_equal"] = int(sum(h["hex_equal"] for h in hex_pairs))
O["n_str_differ_but_float_equal"] = int(sum(1 for h in hex_pairs
                                           if not h["str_equal"] and h["hex_equal"]))

# --- top-29 vmax composition
r = rank.sort_values("candidate_rank").reset_index(drop=True)
top29 = r.head(29)
O["top29_rows_vmax50"] = [[int(x.candidate_rank), float(x.max_velocity_cm_s),
                           float(x.min_velocity_cm_s), float(x.delta_information_criterion)]
                          for _, x in top29[top29.max_velocity_cm_s == 50].iterrows()]
O["top29_vmax_counts"] = {str(float(k)): int(x) for k, x in top29.max_velocity_cm_s.value_counts().items()}
O["first_rank_with_vmax50"] = int(r[r.max_velocity_cm_s == 50].candidate_rank.min())
O["worst_rank_with_vmax50_in_top29"] = int(top29[top29.max_velocity_cm_s == 50].candidate_rank.max())

# --- top-250 family attribution (18 FA / 11 MV claim)
def key(df, cols, nd=6):
    a = np.round(df[cols].to_numpy(float), nd)
    return [tuple(x) for x in a]
KC = ["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "peak_width_microhz"]
KR = ["max_velocity_cm_s", "min_velocity_cm_s", "peak_frequency_microhz", "frequency_width_microhz"]
fam = {k: f for k, f in zip(key(cand, KC), cand.candidate_family.values)}
attrib = [fam.get(k) for k in key(rank, KR)]
from collections import Counter
O["top250_family_counts"] = {str(k): int(v) for k, v in Counter(attrib).items()}

# --- prose claim: "a third of the family has IC > 4000"
O["frac_cand_gt_4000"] = float(np.mean(cand_ic > 4000))
O["frac_cand_gt_4000_exact"] = f"{int(np.sum(cand_ic>4000))}/{len(cand_ic)}"

# --- prose claim: family margin 6.8 "smaller than a single grid step (rank1->rank3 = 5.7)"
O["rank_gap_1_3"] = float(sorted(rank_ic)[2] - sorted(rank_ic)[0])
O["rank_gap_1_4"] = float(sorted(rank_ic)[3] - sorted(rank_ic)[0])
O["rank_gap_3_4"] = float(sorted(rank_ic)[3] - sorted(rank_ic)[2])
O["margin_6p8_smaller_than_5p7"] = bool(6.8 < 5.7)

# --- prose claim: delta 100 "larger than the entire span of the top-250 ranked set"
O["rank_span"] = float(rank_ic.max() - rank_ic.min())
O["100_gt_span"] = bool(100.0 > rank_ic.max() - rank_ic.min())

# --- prose claim: curvature "factor 1.2 costs ~2-4 IC units"
for A in (1470.4938770120555, 1903.582530368582, 3047.372210798297):
    O[f"factor1.2_cost_A{A:.1f}"] = float(A * np.log10(1.2) ** 2)
O["factor1.2_cost_A2000_formula_in_prose"] = float(2000 * (0.2 ** 2))
O["factor_for_2IC_A2000"] = float(10 ** np.sqrt(2 / 2000))
O["factor_for_4IC_A2000"] = float(10 ** np.sqrt(4 / 2000))

# --- slope-inversion split (below/above 1 cm/s) independent
v = uni.uniform_velocity_cm_s.to_numpy(float); y = uni.information_criterion.to_numpy(float)
x = np.log10(v); sl = np.diff(y) / np.diff(x)
inv_idx = [j for j in range(len(sl) - 1) if sl[j + 1] < sl[j]]
vmid = np.sqrt(v[:-1] * v[1:])
O["inv_segments"] = [[float(v[j + 1]), float(v[j + 2]), float(np.sqrt(v[j + 1] * v[j + 2]))]
                     for j in inv_idx]
O["n_inv_below_1"] = int(sum(1 for j in inv_idx if vmid[j + 1] <= 1.0))
O["n_inv_above_1"] = int(sum(1 for j in inv_idx if vmid[j + 1] > 1.0))

# --- prose claim: "preferred scale stays in 0.03-0.1 cm/s band for uniform/vmin axis"
O["global_best_vmin"] = float(cand.loc[cand_ic.argmin(), "min_velocity_cm_s"])
O["vmin_in_003_01_band"] = bool(0.03 <= O["global_best_vmin"] <= 0.1)

# --- does the ranked table's 100-best reach rank 162?
O["n_le_100"] = int(np.sum(rank_ic <= 100))
O["ic_at_rank162"] = float(sorted(rank_ic)[161])
O["ic_at_rank163"] = float(sorted(rank_ic)[162])

# --- "n_candidate_rows_within_10_of_best" = 3 check, and which rows
O["cand_rows_within_10"] = [[str(rr.candidate_family), float(rr.max_velocity_cm_s),
                             float(rr.min_velocity_cm_s), float(rr.peak_frequency_microhz),
                             float(rr.information_criterion)]
                            for _, rr in cand[cand_ic <= cand_ic.min() + 10].iterrows()]

# --- any p-values / CI / tests anywhere in the track JSON (string scan, case-insensitive)
blob = (P / "output/results/ic_comparison.json").read_text().lower()
O["mentions"] = {t: int(blob.count(t)) for t in
                 ["p-value", "p_value", "p =", "significan", "confidence interval", "95%",
                  "standard error", "std err", "surrogate", "null", "random", "shuffle", "permut"]}
# where do 'null' / 'significan' occur (context)
import re
for t in ["null", "significan", "p-value"]:
    for m in re.finditer(t, blob):
        O.setdefault(f"ctx_{t}", []).append(blob[max(0, m.start() - 90):m.end() + 90])

print(json.dumps(O, indent=2, default=str))
Path(P / "output/results/_verify_ic_adv3_scratch.json").write_text(json.dumps(O, indent=2, default=str), encoding="utf-8")