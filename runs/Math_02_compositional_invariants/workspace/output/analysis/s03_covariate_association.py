"""Stage 3 - covariate-associated structure and its confounds.

Tests whether the continuous covariate and coarse group carry composition-level
signal, and separates that signal from the library-size confound using partial
rank correlation, within-group stratification, leave-one-group-out and rarefaction.
Writes output/results/covariate_association.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_composition import (  # noqa: E402
    OUT,
    clr,
    counts_matrix,
    library_size,
    load_covariates,
    partial_spearman,
    rarefy,
    richness,
    save_json,
)

P, samples, features = counts_matrix()
lib = library_size(P).astype(float)
rich = richness(P).astype(float)
Z = clr(P, 0.5)
Zc = Z - Z.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
scores = U[:, :5] * S[:5]

frame = pd.DataFrame(
    {"lib": lib, "rich": rich, **{f"PC{i+1}": scores[:, i] for i in range(5)}},
    index=samples,
)
cv = load_covariates().set_index("sample_id")
tab = cv.join(frame, how="inner")
res = {"n_joined": int(len(tab)), "n_covariate_rows": int(len(cv))}

# --------------------------------------------------------------------------
# 1. marginal rank associations
# --------------------------------------------------------------------------
marginal = {}
for col in ["lib", "rich", "PC1", "PC2", "PC3"]:
    r = stats.spearmanr(tab.continuous_covariate, tab[col])
    marginal[col] = {"spearman": float(r.statistic), "p": float(r.pvalue)}
res["marginal_spearman_vs_covariate"] = marginal

# --------------------------------------------------------------------------
# 2. the richness signal survives library-size control
# --------------------------------------------------------------------------
pr, pp = partial_spearman(tab.rich.values, tab.continuous_covariate.values, tab.lib.values)
pl, plp = partial_spearman(tab.lib.values, tab.continuous_covariate.values, tab.rich.values)
res["partial_rank_correlation"] = {
    "rich_vs_cov_given_lib": {"rho": pr, "p": pp, "n": int(len(tab))},
    "lib_vs_cov_given_rich": {"rho": pl, "p": plp, "n": int(len(tab))},
    "raw_rich_vs_cov": float(stats.spearmanr(tab.rich, tab.continuous_covariate).statistic),
    "raw_lib_vs_cov": float(stats.spearmanr(tab.lib, tab.continuous_covariate).statistic),
    "raw_rich_vs_lib": float(stats.spearmanr(tab.rich, tab.lib).statistic),
    "interpretation": (
        "Richness and library size move in opposite directions with the covariate, and "
        "each keeps its own sign after controlling the other. The richness effect is "
        "therefore not a restatement of the depth effect."
    ),
}

# --------------------------------------------------------------------------
# 3. group structure: between- vs within-group
# --------------------------------------------------------------------------
groups = tab.groupby("coarse_group_id")
per_group = {}
for g, sub in groups:
    r_raw = stats.spearmanr(sub.rich, sub.continuous_covariate)
    r_par, p_par = partial_spearman(sub.rich.values, sub.continuous_covariate.values, sub.lib.values)
    per_group[g] = {
        "n": int(len(sub)),
        "rich_median": float(sub.rich.median()),
        "lib_median": float(sub.lib.median()),
        "covariate_median": float(sub.continuous_covariate.median()),
        "raw_rich_vs_cov": float(r_raw.statistic),
        "raw_p": float(r_raw.pvalue),
        "partial_rich_vs_cov_given_lib": r_par,
        "partial_p": p_par,
    }
res["per_group"] = per_group

between_cov = groups.continuous_covariate.transform("mean")
between_rich = groups.rich.transform("mean")
res["variance_shares"] = {
    "covariate_between_group_share": float(between_cov.var() / tab.continuous_covariate.var()),
    "richness_between_group_share": float(between_rich.var() / tab.rich.var()),
    "library_between_group_share": float(groups.lib.transform("mean").var() / tab.lib.var()),
    "interpretation": (
        "The coarse group label explains only a small share of covariate variance and a "
        "modest share of richness variance, so it is a coarse stratifier rather than a "
        "proxy that could produce the association by itself."
    ),
}

kruskal = {}
for col in ["lib", "rich", "PC1"]:
    hs = [s.values for _, s in groups[col]]
    h, p = stats.kruskal(*hs)
    kruskal[col] = {"H": float(h), "p": float(p)}
res["kruskal_by_group"] = kruskal

# --------------------------------------------------------------------------
# 4. robustness: leave-one-group-out
# --------------------------------------------------------------------------
loo = {}
for drop in ["group_A", "group_B", "group_C"]:
    sub = tab[tab.coarse_group_id != drop]
    r = stats.spearmanr(sub.rich, sub.continuous_covariate)
    rp, _ = partial_spearman(sub.rich.values, sub.continuous_covariate.values, sub.lib.values)
    loo[drop] = {
        "n": int(len(sub)),
        "raw_rho": float(r.statistic),
        "partial_rho_given_lib": rp,
        "directional_agreement": bool(r.statistic > 0 and rp > 0),
    }
res["leave_one_group_out"] = loo

# --------------------------------------------------------------------------
# 5. robustness: equal-effort rarefaction removes the depth confound
# --------------------------------------------------------------------------
rng = np.random.default_rng(4242)
# Align the 529-sample count matrix to the 489 samples that carry covariates.
joined_pos = pd.Index(samples).get_indexer(tab.index)
assert (joined_pos >= 0).all(), "covariate sample not present in the count matrix"
rare_rows = []
for rep in range(10):
    sub = rarefy(P, 3000, rng)
    rr = (sub > 0).sum(axis=1)[joined_pos]
    rare_rows.append(
        {
            "rich_vs_cov": float(stats.spearmanr(rr, tab.continuous_covariate.values).statistic),
            "rich_vs_raw_rich": float(stats.spearmanr(rr, tab.rich.values).statistic),
            "rich_vs_lib": float(stats.spearmanr(rr, tab.lib.values).statistic),
        }
    )
rare_df = pd.DataFrame(rare_rows)
res["rarefaction_robustness"] = {
    "depth": 3000,
    "n_replicates": 10,
    "rich_vs_cov_mean": float(rare_df.rich_vs_cov.mean()),
    "rich_vs_cov_sd": float(rare_df.rich_vs_cov.std()),
    "rich_vs_cov_min": float(rare_df.rich_vs_cov.min()),
    "rich_vs_rawrich_mean": float(rare_df.rich_vs_raw_rich.mean()),
    "rich_vs_lib_mean": float(rare_df.rich_vs_lib.mean()),
    "interpretation": (
        "At a common rarefied depth of 3000 counts the covariate-richness association "
        "persists with the same sign and similar magnitude, so it is not produced by "
        "unequal library size."
    ),
}

# --------------------------------------------------------------------------
# 6. baseline explanations that fail
# --------------------------------------------------------------------------
# 6a. is the covariate just a proxy for group membership?
only_group = pd.get_dummies(tab.coarse_group_id, drop_first=True).astype(float)
res["baselines"] = {}
res["baselines"]["group_label_only"] = {
    "spearman_group_code_vs_rich": float(
        stats.spearmanr(tab.coarse_group_id.map({"group_A": 0, "group_B": 1, "group_C": 2}), tab.rich).statistic
    ),
    "note": "A coarse 3-level label has limited rank information relative to the continuous covariate.",
}

# 6b. does the covariate predict the group mean better than chance? permutation
rng = np.random.default_rng(99)
obs = stats.spearmanr(tab.rich, tab.continuous_covariate).statistic
perm = np.array(
    [abs(stats.spearmanr(tab.rich, rng.permutation(tab.continuous_covariate.values)).statistic) for _ in range(2000)]
)
res["baselines"]["permutation_null"] = {
    "observed_rho": float(obs),
    "n_permutations": 2000,
    "permutation_p": float((perm >= abs(obs)).mean()),
    "null_abs_rho_p95": float(np.percentile(perm, 95)),
}

# 6c. outlier trimming
trimmed = tab[(tab.rich.between(tab.rich.quantile(0.01), tab.rich.quantile(0.99)))]
res["baselines"]["outlier_trim_1_99"] = {
    "n_after_trim": int(len(trimmed)),
    "raw_rho": float(stats.spearmanr(trimmed.rich, trimmed.continuous_covariate).statistic),
    "partial_rho_given_lib": partial_spearman(
        trimmed.rich.values, trimmed.continuous_covariate.values, trimmed.lib.values
    )[0],
}

save_json(res, OUT / "results" / "covariate_association.json")
print("stage 3 ok")
print("  n joined                 :", res["n_joined"])
print("  raw rich~cov rho         :", round(marginal["rich"]["spearman"], 4), "p=%.2e" % marginal["rich"]["p"])
print("  partial rich~cov | lib   :", round(pr, 4), "p=%.2e" % pp)
print("  raw lib~cov rho          :", round(marginal["lib"]["spearman"], 4))
print("  rarefied rich~cov        :", round(res["rarefaction_robustness"]["rich_vs_cov_mean"], 4))
print("  LOOG directions          :", {k: v["directional_agreement"] for k, v in loo.items()})