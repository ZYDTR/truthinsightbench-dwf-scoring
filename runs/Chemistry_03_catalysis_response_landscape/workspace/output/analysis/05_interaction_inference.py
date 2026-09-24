#!/usr/bin/env python3
"""Step 05 - Correct inference for interactions in the unreplicated factorial.

A single-replicate factorial has no pure-error term, so a permutation test that
reshuffles cells is NOT a valid null for interaction SS (it changes the cell
variances rather than testing sparsity of interactions).  This step instead uses
the two standard tools for this design:

  1. Tukey's one-degree-of-freedom non-additivity test (a model-based reference
     distribution for a single non-additive degree of freedom), and
  2. the Lenth / Daniel-style pseudo-standard-error approach, in which the
     interaction contrasts themselves estimate the error scale, so that
     "significant" means "larger than the bulk of the other contrasts".

The step also records the previous invalid permutation attempt so the failed
test is preserved rather than silently dropped.

Outputs: evidence/05_interaction_inference.json
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
RNG = np.random.default_rng(11)

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t").rename(
    columns={"numeric_feature_2": "A", "numeric_feature_3": "B", "numeric_feature_4": "C"})
FAC = ["A", "B", "C"]
OUT: dict = {}


def ss_of(df, y, cols):
    g = df[y].mean()
    m = df.groupby(cols)[y].transform("mean")
    return float(((m - g) ** 2).sum())


def decompose(df, y):
    tot = ss_of(df, y, FAC)
    S1 = sum(ss_of(df, y, [f]) for f in FAC)
    S2 = sum(ss_of(df, y, list(c)) for c in itertools.combinations(FAC, 2))
    return {"ss_total": tot, "ss_main": S1, "ss_2way": S2 - 2 * S1, "ss_3way": tot - S2 + S1}


# ------------------------------------------------- Tukey 1-df non-additivity
res = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    d = decompose(d1, resp)
    y = d1[resp].to_numpy(float)
    # fit additive model on the log/identity scale, form the non-additivity regressor
    # z = fitted_additive^2 - ... (Tukey's single df for two-way; generalised here to
    # the additive model on the three main effects)
    grand = y.mean()
    fit = np.full(len(d1), grand)
    for f in FAC:
        mm = d1.groupby(f)[resp].mean()
        fit = fit + (d1[f].map(mm).to_numpy() - grand)
    resid = y - fit
    z = fit ** 2
    # remove the part of z explained by the additive model (project out)
    Zadd = np.full(len(d1), z.mean())
    for f in FAC:
        mm = d1.groupby(f).apply(lambda g: z[g.index].mean(), include_groups=False)
        Zadd = Zadd + (d1[f].map(mm).to_numpy() - z.mean())
    zt = z - Zadd
    if (zt ** 2).sum() > 0:
        beta = (zt @ resid) / (zt ** 2).sum()
        ss_nonadd = float((beta * zt) ** 2 .sum()) if False else float(beta ** 2 * (zt ** 2).sum())
        df_resid = len(d1) - (1 + sum(d1[f].nunique() - 1 for f in FAC))
        ss_resid = float((resid ** 2).sum()) - ss_nonadd
        ms_resid = ss_resid / df_resid
        F = (ss_nonadd / 1) / ms_resid
        p = float(1 - stats.f.cdf(F, 1, df_resid))
    else:
        ss_nonadd = F = p = float("nan")
    # bootstrap CI on the 2-way / 3-way eta^2 by resampling whole factor levels
    bs2, bs3 = [], []
    for _ in range(300):
        sub = d1.copy()
        # resample rows within each (A,B,C) is degenerate (1 row/cell); instead
        # resample the rows with replacement (whole-observation bootstrap)
        idx = RNG.integers(0, len(d1), len(d1))
        b = decompose(d1.iloc[idx], resp)
        bs2.append(b["ss_2way"] / b["ss_total"])
        bs3.append(b["ss_3way"] / b["ss_total"])
    res[resp] = {
        "decomposition": d,
        "eta2": {k: d[k] / d["ss_total"] for k in ["ss_main", "ss_2way", "ss_3way"]},
        "tukey_nonadditivity": {
            "ss_nonadditivity": ss_nonadd, "df_resid": int(df_resid),
            "ms_resid": float(ms_resid) if ss_nonadd == ss_nonadd else float("nan"),
            "F": F, "p": p,
            "note": "1-df test for departure from additivity using the residual after "
                    "the three main effects; the remaining 1410 residual df are used as error.",
        },
        "bootstrap_eta2": {"eta2_2way_ci95": [float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))],
                           "eta2_3way_ci95": [float(np.percentile(bs3, 2.5)), float(np.percentile(bs3, 97.5))],
                           "n_boot": 300},
    }
OUT["tukey_and_bootstrap"] = res

# ------------------------------------------------- Lenth-style contrast scan
# Build the full set of 2-way interaction contrasts (each of 1 df) and compare
# their magnitudes against a robust pseudo-SD of the contrast distribution.
def contrast_matrix(df, term):
    blocks = []
    for f in term:
        lv = sorted(df[f].unique())
        k = len(lv)
        blk = np.zeros((len(df), k - 1))
        for j in range(k - 1):
            blk[:, j] = np.where(df[f].to_numpy() == lv[j], 1.0,
                                 np.where(df[f].to_numpy() == lv[-1], -1.0, 0.0))
        blocks.append(blk)
    p = blocks[0]
    for b in blocks[1:]:
        p = (p[:, :, None] * b[:, None, :]).reshape(len(df), -1)
    return p


def lenth(values: np.ndarray, label: str) -> dict:
    s = 1.5 * np.median(np.abs(values))
    if s == 0:
        return {"label": label, "pseudo_sd": 0.0, "n_flagged": 0}
    margin = 2.5 * s
    return {
        "label": label, "n": int(len(values)),
        "pseudo_sd_s0": float(s), "margin_ME_2.5s": float(margin),
        "median_abs_effect": float(np.median(np.abs(values))),
        "iqr_abs_effect": float(np.subtract(*np.percentile(np.abs(values), [75, 25]))),
        "n_flagged_beyond_margin": int((np.abs(values) > margin).sum()),
        "max_abs_effect": float(np.abs(values).max()),
        "ratio_max_to_pseudo_sd": float(np.abs(values).max() / s),
        "p_hat_lenth": float(2 * (1 - stats.norm.cdf(np.abs(values).max() / (3.75 * s / np.sqrt(1))))
                             ) if s > 0 else float("nan"),
    }


lenth_res = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    y = d1[resp].to_numpy(float)
    grand = y.mean()
    fit = np.full(len(d1), grand)
    for f in FAC:
        mm = d1.groupby(f)[resp].mean()
        fit = fit + (d1[f].map(mm).to_numpy() - grand)
    main_fit = fit - grand
    lenth_res[resp] = {}
    for term in [("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C")]:
        X = contrast_matrix(d1, term)
        # per-contrast effect: normalise so each contrast is a mean difference
        eff = (X.T @ (y - fit)) / (X ** 2).sum(axis=0)
        eff = eff / np.std(eff) * np.std(eff)  # keep raw scale
        lenth_res[resp]["+".join(term)] = lenth(eff, "+".join(term))
    # also scan raw cell deviations as the unstructured alternative
    cell_dev = y - d1.groupby(FAC)[resp].transform("mean").to_numpy()
    lenth_res[resp]["cell_residual"] = lenth(cell_dev - cell_dev.mean(), "cell_residual")
OUT["lenth_scan"] = lenth_res

# ---------------------------------------------- preserve the failed null test
OUT["failed_test_preserved"] = {
    "test": "within-(A,B)-cell permutation null for the 3-way interaction SS",
    "why_invalid": (
        "Reshuffling the response within each (A,B) block does not preserve the "
        "distribution of the cell variances, so the induced 3-way SS is far LARGER "
        "(f6: null mean 378.2 vs observed 103.8) than the observed value for any "
        "data of this shape. The test therefore has no power to reject non-additivity "
        "and is not used as evidence. It is retained here because the contract "
        "requires reporting failed tests that bear on a submitted claim."
    ),
    "observed_3way_ss": {"numeric_feature_6": 103.82638359029579,
                         "numeric_feature_7": 6.016842008627037},
    "null_mean_3way_ss": {"numeric_feature_6": 378.1840570824998,
                          "numeric_feature_7": 133.69172197489172},
    "verdict": "invalid; superseded by Tukey 1-df and Lenth-style contrast scans",
}

# ------------------------------------- direct sparsity check of 3-way coefficients
sparsity = {}
for resp in ["numeric_feature_6", "numeric_feature_7"]:
    y = d1[resp].to_numpy(float)
    grand = y.mean()
    fit = np.full(len(d1), grand)
    for f in FAC:
        mm = d1.groupby(f)[resp].mean()
        fit = fit + (d1[f].map(mm).to_numpy() - grand)
    for a, b in itertools.combinations(FAC, 2):
        mm = d1.groupby([a, b])[resp].mean()
        fa = d1.groupby(a)[resp].mean()
        fb = d1.groupby(b)[resp].mean()
        inter = (mm - fa.reindex(mm.index.get_level_values(a)).to_numpy()
                 - fb.reindex(mm.index.get_level_values(b)).to_numpy() + grand)
        sparsity[f"{resp}:{a}x{b}"] = {
            "n_cells": int(len(inter)),
            "sd_of_cell_interactions": float(np.std(inter, ddof=1)),
            "max_abs_cell_interaction": float(np.abs(inter).max()),
            "median_abs_cell_interaction": float(np.median(np.abs(inter))),
            "ratio_max_to_median": float(np.abs(inter).max() / max(np.median(np.abs(inter)), 1e-12)),
            "n_cells_beyond_2x_median": int((np.abs(inter) > 2 * np.median(np.abs(inter))).sum()),
        }
OUT["cell_interaction_sparsity"] = sparsity

with (EV / "05_interaction_inference.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str))