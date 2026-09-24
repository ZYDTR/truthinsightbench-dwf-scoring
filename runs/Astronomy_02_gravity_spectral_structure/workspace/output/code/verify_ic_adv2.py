"""Adversarial probe 2: prose assertions, independence, and alternative parsing."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
D = P / "data"
O = {}

uni = pd.read_csv(D / "uniform_alternative_comparison.tsv", sep="\t").sort_values("uniform_velocity_cm_s").reset_index(drop=True)
cand = pd.read_csv(D / "candidate_spectrum_comparison.tsv", sep="\t")
rank = pd.read_csv(D / "ranked_full_parameter_grid.tsv", sep="\t")
v = uni.uniform_velocity_cm_s.to_numpy(float); ic = uni.information_criterion.to_numpy(float)
cand_ic = cand.information_criterion.to_numpy(float)
rank_ic = rank.delta_information_criterion.to_numpy(float)

# 1) prose-claim arithmetic: "a factor 1.2 in velocity costs ~2-4 IC units" with A in 1.47e3..3.05e3
for A in (1470.4938770120555, 3047.372210798297):
    O[f"curv_A{A:.1f}_dIC_at_factor1.2"] = float(A * (np.log10(1.2)) ** 2)
# what factor costs 2 / 4 IC units?
for A in (1470.4938770120555, 3047.372210798297):
    O[f"curv_A{A:.1f}_factor_for_dIC2"] = float(10 ** np.sqrt(2.0 / A))
    O[f"curv_A{A:.1f}_factor_for_dIC4"] = float(10 ** np.sqrt(4.0 / A))
# actual |dIC| to the neighbouring sampled points of the minimum
O["dIC_min_to_0.05"] = float(ic[8] - ic[7]); O["dIC_min_to_0.2"] = float(ic[9] - ic[8])

# 2) q7 claim "larger than the entire span of the top-250 ranked set"
O["rank_span"] = float(rank_ic.max() - rank_ic.min())

# 3) q6 stability claim "top-29 ranked rows all use vmax in {20,30}"
top29 = rank.sort_values("candidate_rank").head(29)
O["top29_vmax_values"] = sorted(float(x) for x in top29.max_velocity_cm_s.unique())
O["top29_vmax_in_20_30"] = bool(set(top29.max_velocity_cm_s) <= {20.0, 30.0})
top10 = rank.sort_values("candidate_rank").head(10)
O["top10_vmax_values"] = sorted(float(x) for x in top10.max_velocity_cm_s.unique())
r4 = rank[rank_ic <= rank_ic.min() + 10]
O["rank_within10_rows"] = [[float(r.max_velocity_cm_s), float(r.min_velocity_cm_s),
                            float(r.peak_frequency_microhz), float(r.frequency_width_microhz),
                            float(r.delta_information_criterion)] for _, r in r4.iterrows()]
O["rank_within10_vmax_set"] = sorted(float(x) for x in r4.max_velocity_cm_s.unique())
O["rank_within10_vmax_subset_20_30"] = bool(set(r4.max_velocity_cm_s) <= {20.0, 30.0})

# 4) stability prose: does the PREFERRED vmin for the winning candidate sit in 0.03-0.1?
O["global_best_cand_vmin"] = float(cand.loc[cand_ic.argmin(), "min_velocity_cm_s"])
O["rank1_vmin"] = float(rank.loc[rank_ic.argmin(), "min_velocity_cm_s"])
O["rank1_vmax"] = float(rank.loc[rank_ic.argmin(), "max_velocity_cm_s"])
fa = cand[cand.candidate_family == "frequency_amplitude_scan"]
O["FA_vmin_all_zero"] = bool((fa.min_velocity_cm_s == 0).all())
O["frac_cand_ic_gt_4000"] = float(np.mean(cand_ic > 4000))
O["n_cand_ic_gt_4000"] = int(np.sum(cand_ic > 4000))

# 5) top-250 is TRUNCATED, not a cartesian grid: coverage per (vmax,vmin) pair
pv = rank.groupby(["max_velocity_cm_s", "min_velocity_cm_s"]).size()
O["rank_pair_counts"] = {f"{a}|{b}": int(n) for (a, b), n in pv.items()}
O["rank_pair_full_cell"] = 10 * 4
O["rank_n_pairs_reaching_40"] = int(np.sum(pv >= 40))
# how many rows would a complete product grid have at these axes
O["rank_cartesian"] = 7 * 8 * 10 * 4
# marginal coverage by f and w
O["rank_w_counts"] = {str(float(k)): int(x) for k, x in rank.frequency_width_microhz.value_counts().items()}
O["rank_f_counts"] = {str(float(k)): int(x) for k, x in rank.peak_frequency_microhz.value_counts().items()}

# 6) does the ranked table contain any row equal to the candidate-table minimum tuple?
O["rank_has_cand_min_row"] = bool(len(rank[(rank.max_velocity_cm_s == 20) &
                                            (rank.min_velocity_cm_s == 0.0) &
                                            (rank.peak_frequency_microhz == 1010) &
                                            (rank.frequency_width_microhz == 300)]))

# 7) alternative parse via numpy (no pandas) for the three load-bearing numbers
arr = np.genfromtxt(D / "uniform_alternative_comparison.tsv", delimiter="\t", names=True)
O["np_uni_n"] = int(arr.shape[0])
O["np_uni_min_ic"] = float(arr["information_criterion"].min())
O["np_uni_argmin_v"] = float(arr["uniform_velocity_cm_s"][int(arr["information_criterion"].argmin())])
c = np.genfromtxt(D / "candidate_spectrum_comparison.tsv", delimiter="\t", names=True, dtype=None, encoding="utf-8")
O["np_cand_n"] = int(c.shape[0])
O["np_cand_min_ic"] = float(c["information_criterion"].min())
r = np.genfromtxt(D / "ranked_full_parameter_grid.tsv", delimiter="\t", names=True)
O["np_rank_n"] = int(r.shape[0])
O["np_rank_min_delta"] = float(r["delta_information_criterion"].min())
# raw-text check of the uniform IC vs diagonal IC values
dtxt = (D / "uniform_alternative_comparison.tsv").read_text().strip().split("\n")[1:]
O["raw_uni_pairs"] = [tuple(x.split("\t")) for x in dtxt]

# 8) count any inferential statistic in the track JSON
J = json.loads((P / "output/results/ic_comparison.json").read_text())
blob = json.dumps(J).lower()
for tok in ["p_value", "p-value", "p =", "p=", "stderr", "std_err", "confidence", "ci_",
            "significan", "chi2", "t-test", "ttest", "z-score", "zscore"]:
    O[f"token_{tok}"] = int(blob.count(tok))
# list all keys mentioning variance/uncertainty
def walk(o, path=""):
    out = []
    if isinstance(o, dict):
        for k, val in o.items():
            out += walk(val, f"{path}.{k}")
    elif isinstance(o, list):
        for i, val in enumerate(o):
            out += walk(val, f"{path}[{i}]")
    else:
        out.append(path)
    return out
paths = walk(J)
O["n_json_leaf_paths"] = len(paths)
O["paths_with_uncertainty"] = [p for p in paths if any(t in p.lower() for t in
                                                       ["uncert", "error", "pval", "p_val", "conf", "se", "sd", "std"])]

print(json.dumps(O, indent=2, default=str))
Path(P / "output/results/_verify_ic_adv2_scratch.json").write_text(json.dumps(O, indent=2, default=str), encoding="utf-8")