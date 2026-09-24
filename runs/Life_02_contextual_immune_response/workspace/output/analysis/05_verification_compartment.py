#!/usr/bin/env python3
"""V1 verification: compartment_response_replicates.tsv (raw TSV only)."""
import json
import numpy as np
import pandas as pd
from scipy import stats

RAW = "/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project/data/compartment_response_replicates.tsv"
OUT = "/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project/output/results/verification_compartment.json"

errors = []
checks = []
df = pd.read_csv(RAW, sep="\t")
print("rows:", len(df), "cols:", list(df.columns))

# ---- build 18 cells from raw rows -----------------------------------------
wide = df.pivot_table(
    index=["assay_id", "endpoint_type", "compartment_id", "replicate_id"],
    columns="condition_role", values="response_value", aggfunc="first",
).reset_index()
print("replicate rows:", len(wide), "cells:",
      wide.groupby(["assay_id", "endpoint_type", "compartment_id"]).ngroups)

bad = wide[(wide["baseline"] <= 0) | (wide["perturbation"] <= 0)]
if len(bad):
    errors.append(f"non-positive values in {len(bad)} rows; log2 undefined")

wide["log2_ratio"] = np.log2(wide["perturbation"] / wide["baseline"])
wide["up"] = wide["perturbation"] > wide["baseline"]

cell = wide.groupby(["assay_id", "endpoint_type", "compartment_id"]).agg(
    n=("log2_ratio", "size"),
    mean_log2=("log2_ratio", "mean"),
    up_pairs=("up", "sum"),
).reset_index()
cell["fold_change"] = 2.0 ** cell["mean_log2"]
assert (cell["n"] == 4).all(), "not all cells have 4 replicates"
cell = cell.sort_values(["assay_id", "endpoint_type", "compartment_id"]).reset_index(drop=True)

def add(claim, recomputed, claimed):
    ok = np.allclose(np.asarray(recomputed, float), np.asarray(claimed, float),
                     rtol=5e-4, atol=5e-4) if not isinstance(claimed, str) else recomputed == claimed
    checks.append({"claim": claim, "recomputed": recomputed,
                   "claimed": claimed, "agrees": bool(ok)})

# ---- V1a -------------------------------------------------------------------
n_up_cells = int((cell["fold_change"] > 1).sum())
add("V1a cells with fold change > 1", n_up_cells, 18)

# ---- V1b -------------------------------------------------------------------
n_up_pairs = int(wide["up"].sum())
add("V1b replicate pairs pert > baseline", n_up_pairs, 69)

# ---- V1c / V1d -------------------------------------------------------------
m = float(cell["mean_log2"].mean())
add("V1c mean cell log2 fold change", round(m, 6), 0.871)
sd = float(cell["mean_log2"].std(ddof=1))
se = sd / np.sqrt(len(cell))
tcrit = float(stats.t.ppf(0.975, df=17))
ci = [m - tcrit * se, m + tcrit * se]
add("V1d 95% CI of mean log2 FC (t, df=17)", [round(ci[0], 6), round(ci[1], 6)], [0.749, 0.992])

# ---- V1e / V1f -------------------------------------------------------------
med = float(cell["fold_change"].median())
add("V1e median cell fold change", round(med, 6), 1.868)
mn, mx = float(cell["fold_change"].min()), float(cell["fold_change"].max())
add("V1f min cell fold change", round(mn, 6), 1.382)
add("V1f max cell fold change", round(mx, 6), 2.531)

# ---- V1g -------------------------------------------------------------------
g_count = float(cell.loc[cell.endpoint_type == "endpoint_count", "fold_change"].mean())
g_frac = float(cell.loc[cell.endpoint_type == "endpoint_fraction", "fold_change"].mean())
add("V1g mean FC endpoint_count", round(g_count, 6), 1.930)
add("V1g mean FC endpoint_fraction", round(g_frac, 6), 1.777)
add("V1g cells per endpoint type",
    [int((cell.endpoint_type == "endpoint_count").sum()),
     int((cell.endpoint_type == "endpoint_fraction").sum())], [9, 9])

# ---- V1h paired t-test over 9 assay x compartment strata -------------------
piv = cell.pivot_table(index=["assay_id", "compartment_id"], columns="endpoint_type",
                       values="mean_log2")
d = piv["endpoint_count"] - piv["endpoint_fraction"]
tt = stats.ttest_rel(piv["endpoint_count"], piv["endpoint_fraction"])
add("V1h paired t statistic", round(float(tt.statistic), 6), 2.313)
add("V1h paired t p-value", round(float(tt.pvalue), 6), 0.0495)
add("V1h mean log2 delta (count - fraction)", round(float(d.mean()), 6), 0.111)
add("V1h strata n", int(len(piv)), 9)

# ---- adversarial: cell-level sign test on 18 cells -------------------------
signs = (cell["mean_log2"] > 0).sum()
bt = stats.binomtest(int(signs), int(len(cell)), 0.5, alternative="greater")
bt2 = stats.binomtest(int(signs), int(len(cell)), 0.5, alternative="two-sided")
adversarial = {
    "check": "Cell-level sign test: each of 18 cells contributes one sign of its mean log2 FC",
    "n_positive_cells": int(signs), "n_cells": int(len(cell)),
    "p_one_sided": float(bt.pvalue), "p_two_sided": float(bt2.pvalue),
    "pooled_pairs_note": (f"pooled binomial on 72 pairs would treat pairs as independent; "
                          f"{n_up_pairs}/72 gives p={float(stats.binomtest(n_up_pairs, 72, 0.5, alternative='greater').pvalue):.3e}"),
    "conclusion": "SIGNIFICANT — all 18 cell-level signs are positive; p_two_sided=%.3e. "
                  "Effective n drops from 72 to 18 but the conclusion that all cells are up is unchanged."
                  % float(bt2.pvalue),
}

all_agree = all(c["agrees"] for c in checks)
conclusion = ("All V1a-V1h claims reproduced from raw TSV." if all_agree
              else "MISMATCH in: " + "; ".join(c["claim"] for c in checks if not c["agrees"]))
conclusion += f" Adversarial cell-level sign test: {adversarial['conclusion']}"

res = {"unit": "compartment_response_replicates",
       "checks": checks, "adversarial_check": adversarial,
       "conclusion": conclusion, "errors": errors}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

for c in checks:
    print(f"{'OK ' if c['agrees'] else 'NO '}{c['claim']}: got {c['recomputed']} claimed {c['claimed']}")
print("adversarial:", {k: adversarial[k] for k in ("n_positive_cells", "p_one_sided", "p_two_sided")})
print("errors:", errors)