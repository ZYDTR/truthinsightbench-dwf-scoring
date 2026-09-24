#!/usr/bin/env python3
"""Step 09 - Confound check on the transfer claims.

The condition-level profiles of panel 1's responses correlate both with the
magnitude columns and with the response column of panels 3/4.  This step asks
whether the apparent transfer of panel 1's response to panel 4 survives after
removing the shared magnitude component (partial correlation / regression).

Outputs: evidence/09_transfer_confound.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EV = ROOT / "output" / "evidence"
RNG = np.random.default_rng(7)

d1 = pd.read_csv(DATA / "observation_set_1.tsv", sep="\t")
d3 = pd.read_csv(DATA / "observation_set_3.tsv", sep="\t")
d4 = pd.read_csv(DATA / "observation_set_4.tsv", sep="\t")
OUT: dict = {}


def cond(df, cond_col, col):
    return df.groupby(cond_col)[col].mean().reindex(range(13)).to_numpy()


P = {
    "p1_f6": d1.groupby("numeric_feature_3")["numeric_feature_6"].mean().to_numpy(),
    "p1_f7": d1.groupby("numeric_feature_3")["numeric_feature_7"].mean().to_numpy(),
    "p3_A": cond(d3, "numeric_feature_2", "numeric_feature_3"),
    "p3_B": cond(d3, "numeric_feature_2", "numeric_feature_4"),
    "p3_R": cond(d3, "numeric_feature_2", "numeric_feature_5"),
    "p4_A": cond(d4, "numeric_feature_3", "numeric_feature_4"),
    "p4_B": cond(d4, "numeric_feature_3", "numeric_feature_5"),
    "p4_R": cond(d4, "numeric_feature_3", "numeric_feature_6"),
}
OUT["condition_profiles"] = {k: [float(v) for v in vv] for k, vv in P.items()}

# use log-magnitude, which is the scale on which the panels agree best
L = {k: np.log(v) for k, v in P.items() if k.endswith(("_A", "_B"))}
L.update({k: v for k, v in P.items() if k.endswith(("_f6", "_f7", "_R"))})


def partial(x, y, z):
    """Spearman partial correlation of x,y controlling z (rank-based)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        sl, ic = np.polyfit(b, a, 1)
        return a - (sl * b + ic)
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])


pairs = [("p1_f6", "p4_R", "p4_B"), ("p1_f6", "p4_R", "p4_A"),
         ("p1_f6", "p3_R", "p3_B"), ("p1_f6", "p3_R", "p3_A"),
         ("p1_f7", "p4_R", "p4_B"), ("p3_R", "p4_R", "p4_B"),
         ("p1_f6", "p4_B", "p4_A"), ("p1_f6", "p3_B", "p3_A"),
         ("p1_f7", "p3_B", "p3_A"), ("p3_B", "p4_B", "p4_A")]
pc = {}
for x, y, z in pairs:
    xs = L.get(x, P[x]); ys = L.get(y, P[y]); zs = L.get(z, P[z])
    pc[f"{x}~{y}|{z}"] = {
        "raw_spearman": float(stats.spearmanr(xs, ys).statistic),
        "partial_controlling": z,
        "partial_spearman": partial(xs, ys, zs),
        "n_levels": 13,
    }
OUT["partial_correlations"] = pc

# rank-based permutation p for the partial correlations
NPERM = 20000
for k, v in pc.items():
    x, y, z = k.replace("|", "~").split("~")
    xs, ys, zs = L.get(x, P[x]), L.get(y, P[y]), L.get(z, P[z])
    obs = v["partial_spearman"]
    null = np.empty(NPERM)
    for i in range(NPERM):
        null[i] = partial(xs, ys[RNG.permutation(13)], zs)
    v["partial_perm_p_two_sided"] = float((np.abs(null) >= abs(obs)).mean())
    v["partial_null_abs_p95"] = float(np.percentile(np.abs(null), 95))

# multi-predictor regression: does p1_f6 add anything beyond both p4 magnitudes?
reg = {}
for target, preds in [("p4_R", ["p4_A", "p4_B"]), ("p4_R", ["p4_A", "p4_B", "p1_f6"]),
                      ("p4_R", ["p1_f6"]), ("p3_R", ["p3_A", "p3_B", "p1_f6"]),
                      ("p4_B", ["p1_f6"]), ("p3_B", ["p1_f6"]),
                      ("p4_R", ["p1_f7"]), ("p3_R", ["p1_f7"])]:
    yv = L.get(target, P[target])
    X = np.column_stack([L.get(p, P[p]) for p in preds])
    Xz = (X - X.mean(0)) / X.std(0)
    yz = (yv - yv.mean()) / yv.std()
    A = np.column_stack([Xz, np.ones(len(yz))])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    pred = A @ beta
    r2 = float(1 - ((yz - pred) ** 2).sum() / ((yz - yz.mean()) ** 2).sum())
    # adjusted R2
    n, k = len(yz), len(preds)
    adj = float(1 - (1 - r2) * (n - 1) / (n - k - 1))
    # standardised coefficients with a permutation p on the extra term
    reg["+".join(preds) + " -> " + target] = {
        "r2": r2, "adj_r2": adj, "n": n, "k": k,
        "std_beta": {p: float(b) for p, b in zip(preds, beta[:-1])},
        "intercept": float(beta[-1]),
    }
OUT["regressions"] = reg

# within-panel: does the condition-mean magnitude track the condition-mean response?
wp = {}
for tag, A, B, R in [("panel_3", "p3_A", "p3_B", "p3_R"), ("panel_4", "p4_A", "p4_B", "p4_R")]:
    wp[tag] = {
        "spearman_B_vs_R": float(stats.spearmanr(P[B], P[R]).statistic),
        "spearman_A_vs_R": float(stats.spearmanr(P[A], P[R]).statistic),
        "spearman_logB_vs_R": float(stats.spearmanr(np.log(P[B]), P[R]).statistic),
        "response_range_across_conditions": float(P[R].max() - P[R].min()),
        "magnitude_B_range_across_conditions": float(P[B].max() - P[B].min()),
        "magnitude_A_range_across_conditions": float(P[A].max() - P[A].min()),
    }
    # within-condition dispersion of the response across the series
    df = d3 if tag == "panel_3" else d4
    ccol = "numeric_feature_2" if tag == "panel_3" else "numeric_feature_3"
    rcol = "numeric_feature_5" if tag == "panel_3" else "numeric_feature_6"
    wp[tag]["within_condition_sd_of_R"] = float(
        df.groupby(ccol)[rcol].std(ddof=1).mean())
    bcol = "numeric_feature_4" if tag == "panel_3" else "numeric_feature_5"
    wp[tag]["within_condition_sd_of_B"] = float(
        df.groupby(ccol)[bcol].std(ddof=1).mean())
OUT["within_panel_condition_relations"] = wp

with (EV / "09_transfer_confound.json").open("w") as fh:
    json.dump(OUT, fh, indent=2, default=str)

print("PARTIAL CORRELATIONS (Spearman, controlling the stated magnitude):")
for k, v in pc.items():
    print(f"  {k:26s} raw={v['raw_spearman']:+.4f}  partial={v['partial_spearman']:+.4f}"
          f"  perm_p={v.get('partial_perm_p_two_sided', float('nan')):.4f}")
print("\nREGRESSIONS:")
for k, v in reg.items():
    print(f"  {k:32s} R2={v['r2']:.4f} adjR2={v['adj_r2']:.4f} "
          f"betas={ {a: round(b,3) for a,b in v['std_beta'].items()} }")
print("\nWITHIN-PANEL CONDITION RELATIONS:")
print(json.dumps(wp, indent=1))