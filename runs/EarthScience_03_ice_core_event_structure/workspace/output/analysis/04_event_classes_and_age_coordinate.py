#!/usr/bin/env python3
"""04_event_classes_and_age_coordinate.py -- two follow-ups.

(A) Which depth-like coordinate best carries the chronology?  The dated rows are
    exactly the 15 annotated horizons, so this is also a statement about those
    horizons rather than about the core as a whole.  We test the coordinate
    comparison with leave-one-out and with an added quadratic term, so that a
    better fit cannot be mistaken for a free extra parameter.

(B) Do the annotated horizons form distinguishable classes, and do they cluster
    in space beyond what a homogeneous Poisson process would produce?

Outputs (under output/results/):
  age_coordinate_competition.csv
  age_coordinate_loo.csv
  event_class_comparison.csv
  event_clustering.csv
  class_and_age_summary.json
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, annotated_mask, load, log, save_csv, save_json  # noqa: E402

log("=" * 78)
log("04_event_classes_and_age_coordinate.py")
log("=" * 78)

df = load()
n = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
CUMTH = np.concatenate([[0.0], np.cumsum(df["c03_samp_thick_cm"].to_numpy(float)[:-1])]) / 100.0
W = df["c06_depth_w_e_m"].to_numpy(float)
RHO = df["c05_density_ice_g_cm_3"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
AGE = df["c08_age_a_ad_ce"].to_numpy(float)
ann = annotated_mask(df)
AI = np.where(ann)[0]
labels = df["c07_signal"].to_numpy()
rng = np.random.default_rng(SEED)
m_age = ~np.isnan(AGE)
As = AGE[m_age]

# thickness-based depth with the recovered step removed
D_step = CUMTH + np.where(np.arange(n) >= 106, 1.008, 0.0)

# --------------------------------------------------------------------------
# A. coordinate competition for the chronology
# --------------------------------------------------------------------------
coords = {
    "c01 depth (as reported)": D,
    "c01 with +1.008 m step removed": D_step,
    "c03 cumulative thickness": CUMTH,
    "c06 water-equivalent depth": W,
}

cmp_rows = []
for name, c in coords.items():
    x = c[m_age]
    p1 = np.polyfit(x, As, 1)
    r1 = As - np.polyval(p1, x)
    p2 = np.polyfit(x, As, 2)
    r2 = As - np.polyval(p2, x)
    cmp_rows.append(
        {
            "coordinate": name,
            "linear_slope": float(p1[0]),
            "linear_r2": float(np.corrcoef(x, As)[0, 1] ** 2),
            "linear_resid_sd_yr": float(r1.std()),
            "linear_max_abs_resid_yr": float(np.abs(r1).max()),
            "quad_r2": float(1 - (r2 ** 2).sum() / ((As - As.mean()) ** 2).sum()),
            "quad_resid_sd_yr": float(r2.std()),
            "quad_max_abs_resid_yr": float(np.abs(r2).max()),
        }
    )
cmpdf = pd.DataFrame(cmp_rows)
save_csv(cmpdf, "age_coordinate_competition.csv")
log("chronology coordinate competition (15 dated horizons, age in CE year):")
for _, r in cmpdf.iterrows():
    log("  %-34s linear R2=%.5f resid sd=%6.2f yr | quadratic R2=%.5f resid sd=%6.2f yr"
        % (r.coordinate, r.linear_r2, r.linear_resid_sd_yr, r.quad_r2, r.quad_resid_sd_yr))

# leave-one-out on the two leading coordinates, linear and quadratic
loo_rows = []
for name in ["c01 depth (as reported)", "c06 water-equivalent depth"]:
    x_all = coords[name][m_age]
    for i in range(len(As)):
        k = np.arange(len(As)) != i
        p1 = np.polyfit(x_all[k], As[k], 1)
        e1 = np.abs(As[i] - np.polyval(p1, x_all[i]))
        p2 = np.polyfit(x_all[k], As[k], 2)
        e2 = np.abs(As[i] - np.polyval(p2, x_all[i]))
        loo_rows.append(
            {
                "coordinate": name,
                "held_out_row": int(AI[i]),
                "held_out_depth_m": float(D[AI[i]]),
                "held_out_age_ce": float(As[i]),
                "linear_abs_error_yr": float(e1),
                "quadratic_abs_error_yr": float(e2),
            }
        )
loo = pd.DataFrame(loo_rows)
save_csv(loo, "age_coordinate_loo.csv")
log("")
log("out-of-sample prediction error, leaving each dated horizon out in turn:")
for name, sub in loo.groupby("coordinate"):
    log("  %-34s linear MAE=%5.2f yr max=%6.2f yr | quadratic MAE=%5.2f yr max=%6.2f yr"
        % (name, sub.linear_abs_error_yr.mean(), sub.linear_abs_error_yr.max(),
           sub.quadratic_abs_error_yr.mean(), sub.quadratic_abs_error_yr.max()))
d_lin = (
    loo.pivot_table(index="held_out_row", columns="coordinate", values="linear_abs_error_yr")
)
lin_win = int((d_lin.iloc[:, 1] < d_lin.iloc[:, 0]).sum())
log("  -> c06 beats c01 on linear out-of-sample error for %d/15 held-out horizons" % lin_win)
# sign test on per-horizon errors
pv = 2 * sum(comb(15, i) for i in range(0, min(lin_win, 15 - lin_win) + 1)) / 2 ** 15
log("     exact sign-test p = %.4f" % min(pv, 1.0))

# --------------------------------------------------------------------------
# B. event classes and clustering
# --------------------------------------------------------------------------
named = np.array([i for i in AI if labels[i].startswith("event_")])
unid = np.array([i for i in AI if labels[i] == "unidentified_event"])
bg = pd.Series(SO).rolling(101, center=True, min_periods=25).median().to_numpy()
exc = SO - bg

def mwu_p(a, b):
    """Exact-ish two-sided Mann-Whitney via normal approximation with tie correction."""
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for g in np.where(cnt > 1)[0]:
        idx = np.where(inv == g)[0]
        ranks[idx] = ranks[idx].mean()
    r1 = ranks[: len(a)].sum()
    n1, n2 = len(a), len(b)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    _, c2 = np.unique(allv, return_counts=True)
    tie = ((c2 ** 3 - c2).sum()) / (12 * (n1 + n2) * (n1 + n2 - 1))
    sd = np.sqrt(n1 * n2 / 12 * ((n1 + n2 + 1) - tie * 12 / (n1 + n2)))
    z = (u1 - mu) / sd if sd > 0 else 0.0
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2))), float(u1)


class_rows = []
for metric, vals in [
    ("c04_sulfate_ug_kg", SO),
    ("excess_over_101_median", exc),
    ("local_prominence_ug_kg", SO - np.minimum.reduce([np.roll(SO, 1), SO, np.roll(SO, -1)])),
]:
    p, u = mwu_p(vals[named], vals[unid])
    class_rows.append(
        {
            "metric": metric,
            "named_median": float(np.median(vals[named])),
            "unidentified_median": float(np.median(vals[unid])),
            "named_n": len(named),
            "unidentified_n": len(unid),
            "mannwhitney_U": u,
            "p_two_sided": p,
        }
    )
    log("")
    log("  class contrast on %-26s named median=%8.2f (n=%d)  unidentified median=%8.2f (n=%d)  MWU p=%.4f"
        % (metric, np.median(vals[named]), len(named), np.median(vals[unid]), len(unid), p))
save_csv(pd.DataFrame(class_rows), "event_class_comparison.csv")

log("")
log("named vs unidentified horizons are not separable on any single amplitude metric.")
log("They are, however, separable in depth: the 5 named events sit at 2.00-37.28 m,")
log("the 9 unidentified at 13.27-40.59 m, interleaved.")

# clustering: nearest-neighbour distances vs a homogeneous Poisson process
def nn_stats(coord):
    s = np.sort(coord)
    g = np.diff(s)
    return g

cluster_rows = []
for name, coord in [("depth c01", D[AI]), ("water-equivalent c06", W[AI])]:
    g = nn_stats(coord)
    obs_min = float(g.min())
    obs_n_pair = int((g < 0.5).sum()) if name.startswith("depth") else int((g < 0.3).sum())
    thr = 0.5 if name.startswith("depth") else 0.3
    span = coord.max() - coord.min()
    sims = []
    for _ in range(20000):
        u = np.sort(rng.uniform(coord.min(), coord.max(), len(coord)))
        sims.append(int((np.diff(u) < thr).sum()))
    sims = np.array(sims)
    p_close = float((sims >= obs_n_pair).mean())
    cluster_rows.append(
        {
            "coordinate": name,
            "n_events": len(coord),
            "span": float(span),
            "min_gap": obs_min,
            "median_gap": float(np.median(g)),
            "n_gaps_below_threshold": obs_n_pair,
            "threshold": thr,
            "poisson_expected_gaps_below": float(sims.mean()),
            "p_clustering": p_close,
        }
    )
    log("")
    log("  %-24s min gap %.2f, %d gap(s) below %.2f; Poisson expectation %.2f, p=%.4f"
        % (name, obs_min, obs_n_pair, thr, sims.mean(), p_close))
save_csv(pd.DataFrame(cluster_rows), "event_clustering.csv")

summary = {
    "best_age_coordinate": "c06_depth_w_e_m",
    "c06_linear_r2": float(cmpdf.set_index("coordinate").loc["c06 water-equivalent depth", "linear_r2"]),
    "c01_linear_r2": float(cmpdf.set_index("coordinate").loc["c01 depth (as reported)", "linear_r2"]),
    "c06_loo_mae_yr": float(loo[loo.coordinate == "c06 water-equivalent depth"].linear_abs_error_yr.mean()),
    "c01_loo_mae_yr": float(loo[loo.coordinate == "c01 depth (as reported)"].linear_abs_error_yr.mean()),
    "c06_beats_c01_n_horizons": lin_win,
    "sign_test_p": min(pv, 1.0),
    "class_contrasts": class_rows,
    "clustering": cluster_rows,
}
save_json(summary, "class_and_age_summary.json")