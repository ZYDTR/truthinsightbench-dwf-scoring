#!/usr/bin/env python3
"""Second independent script: adversarial attack on Draft Finding 1.

Focus:
  * alignment/derivation direction of c01 vs c03 (off-by-one, rounding, drift)
  * what model actually yields a residual near the claimed ~0.016 m
  * c06 recursion vs product
  * exact count of distinct c05 (claim says 261)
  * hidden algebraic dependence of c04 / c07 on geometry
  * figures
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = "/data/user/conversations/conv-f07721cb-d3dc-42e9-83c7-7e986203115b/project"
OUT = os.path.join(PROJ, "output", "verification", "figures")
raw = pd.read_csv(os.path.join(PROJ, "data", "observations.tsv"), sep="\t")
names = list(raw.columns)
c = {f"c{i+1:02d}": raw[names[i]] for i in range(9)}
N = len(raw)
out = {}


def rec(k, v):
    out[k] = v
    print(k, "=", v)


c01 = c["c01"].values.astype(float)
c02 = c["c02"].values.astype(float)
c03 = c["c03"].values.astype(float)
c04 = c["c04"].values.astype(float)
c05 = c["c05"].values.astype(float)
c06 = c["c06"].values.astype(float)

# --------------------------------------------------------------- alignment tests
cum = np.cumsum(c03) / 100.0
cum_shift = np.concatenate([[0.0], np.cumsum(c03[:-1]) / 100.0])  # c01[i] = sum c03[0..i-1]
stepv = np.zeros(N); stepv[106:] = 1.0

r_aligned = c01 - cum
r_shift = c01 - cum_shift
rec("shift_align_resid_min_max", [float(r_shift.min()), float(r_shift.max())])
rec("shift_align_resid_absmax_excl_step", float(np.abs(np.delete(r_shift, np.arange(106, N))).max()))
rec("direct_align_resid_absmax_excl_step", float(np.abs(np.delete(r_aligned, np.arange(106, N))).max()))

# residual per segment for both alignments
for nm, rv in (("direct", r_aligned), ("shift", r_shift)):
    pre = rv[:106]; post = rv[106:]
    rec(f"{nm}_pre_minmax", [float(pre.min()), float(pre.max())])
    rec(f"{nm}_post_minmax", [float(post.min()), float(post.max())])

# best single step with shift alignment, allowing baseline
def fit_model(align, knot, allow_baseline):
    H = np.zeros(N); H[knot + 1:] = 1.0
    t = (c01 - align)
    cols = [H] if not allow_baseline else [np.ones(N), H]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    model = align + A @ coef
    res = c01 - model
    return coef, float(np.abs(res).max()), res


rec("direct_no_base_k105", fit_model(cum, 105, False)[:2])
rec("shift_no_base_k105", fit_model(cum_shift, 105, False)[:2])
rec("shift_with_base_k105", fit_model(cum_shift, 105, True)[:2])
rec("direct_with_base_k105", fit_model(cum, 105, True)[:2])

# grid over knot for both alignments, with baseline, report best max-resid and params
grid = {}
for nm, align in (("direct", cum), ("shift", cum_shift)):
    for k in range(90, 120):
        coef, mx, _ = fit_model(align, k, True)
        grid.setdefault(nm, {})[k] = (mx, [float(x) for x in coef])
best_shift = min(grid["shift"], key=lambda k: grid["shift"][k][0])
best_direct = min(grid["direct"], key=lambda k: grid["direct"][k][0])
rec("best_model_shift", {"knot": best_shift, "max_res": grid["shift"][best_shift][0],
                         "coefs(base,step)": grid["shift"][best_shift][1]})
rec("best_model_direct", {"knot": best_direct, "max_res": grid["direct"][best_direct][0],
                          "coefs(base,step)": grid["direct"][best_direct][1]})

# residual with the exact claim model: cum + 1.008*H(k=105), no baseline
claim_res = c01 - (cum + 1.008 * stepv)
rec("claim_model_max_abs_resid", float(np.abs(claim_res).max()))
rec("claim_model_resid_min_max", [float(claim_res.min()), float(claim_res.max())])
claim_res_s = c01 - (cum_shift + 1.008 * stepv)
rec("claim_model_shiftalign_max_abs_resid", float(np.abs(claim_res_s).max()))

# which model gives ~0.016?
for nm, align in (("direct", cum), ("shift", cum_shift)):
    best = grid[nm]
    kbest = min(best, key=lambda k: best[k][0])
    rec(f"grid_{nm}_best_max_res", best[kbest][0])

# rounding model: c01 = round(align + b + s*H, 2)
def round_model(align, knot, b, s):
    H = np.zeros(N); H[knot + 1:] = 1.0
    return np.round(align + b + s * H, 2)


for nm, align, kk, bb, ss in (("direct", cum, 105, 0.0, 1.008),
                              ("shift", cum_shift, 105, 0.0, 1.008)):
    rm = round_model(align, kk, bb, ss)
    rec(f"roundmodel_{nm}_b{bb}_s{ss}_exact_frac",
        float(np.mean(np.abs(rm - c01) < 1e-9)))
    rec(f"roundmodel_{nm}_b{bb}_s{ss}_within1cm_frac",
        float(np.mean(np.abs(rm - c01) <= 0.01)))

# exact-decimal residual check (avoid float noise): compare 100*c01 against integers
rec("c01_times100_is_integer", bool(np.all(np.abs(c01 * 100 - np.round(c01 * 100)) < 1e-9)))
rec("c03_times100_is_integer", bool(np.all(np.abs(c03 * 100 - np.round(c03 * 100)) < 1e-9)))

# --------------------------------------------------------------- c03 block structure
blocks = []
start = 0
for i in range(1, N):
    if c03[i] != c03[i - 1]:
        blocks.append((start, i - 1, float(c03[start])))
        start = i
blocks.append((start, N - 1, float(c03[start])))
rec("n_c03_blocks", len(blocks))
rec("c03_block_summary", [{"i0": a, "i1": b, "c03": v, "n": b - a + 1} for a, b, v in blocks])

# within-block mean of the observed c01 diffs vs c03/100
rows = []
for a, b, v in blocks:
    if b > a:
        segment = c01[a:b + 1]
        mean_diff = (segment[-1] - segment[0]) / (b - a) * 100
    else:
        mean_diff = float("nan")
    rows.append({"i0": a, "i1": b, "n": b - a + 1, "c03": v, "mean_c01_diff_cm": mean_diff,
                 "gap_cm": None if np.isnan(mean_diff) else mean_diff - v})
rec("block_mean_diff_table", rows)
bl = pd.DataFrame(rows)
rec("block_mean_diff_max_abs_gap_cm", float(np.nanmax(np.abs(bl["gap_cm"].values))))

# does the big step coincide with a c03 block boundary?
rec("blocks_containing_index_105_106",
    [r for r in rows if r["i0"] <= 105 <= r["i1"] or r["i0"] <= 106 <= r["i1"]])

# c07 at the step row and neighbours
rec("c07_around_step", [str(x) for x in c["c07"].values[100:112]])
rec("c03_around_step", [float(x) for x in c03[100:112]])
rec("c01_around_step", [float(x) for x in c01[100:112]])

# --------------------------------------------------------------- c06 recursion vs product
prod = c01 * c05
rec("d_max_abs_resid_product", float(np.abs(c06 - prod).max()))
dprod = c06 - prod
rec("d_resid_by_c01_corr", float(np.corrcoef(dprod, c01)[0, 1]))
# recursion candidates
dc01 = np.diff(c01)
recurs = {
    "rect_right": lambda i: c06[i - 1] + dc01[i - 1] * c05[i],
    "rect_left": lambda i: c06[i - 1] + dc01[i - 1] * c05[i - 1],
    "trapz": lambda i: c06[i - 1] + dc01[i - 1] * (c05[i] + c05[i - 1]) / 2.0,
}
for nm, f in recurs.items():
    pred = np.empty(N - 1)
    for i in range(1, N):
        pred[i - 1] = f(i)
    r = c06[1:] - pred
    rec(f"recursion_{nm}_max_abs_resid", float(np.abs(r).max()))
    rec(f"recursion_{nm}_within_0.005_frac", float(np.mean(np.abs(r) <= 0.005)))
    rec(f"recursion_{nm}_within_0.01_frac", float(np.mean(np.abs(r) <= 0.01)))
    rec(f"recursion_{nm}_within_0.05_frac", float(np.mean(np.abs(r) <= 0.05)))

# is c06 on a 0.01 grid?
rec("c06_on_0.01_grid_maxdev", float(np.abs(c06 * 100 - np.round(c06 * 100)).max()))
rec("c06_unique", int(pd.Series(c06).nunique()))

# c06 minus product: smooth in c01?
rec("dprod_sign_frac_positive", float(np.mean(dprod > 0)))
# product using rounded-to-3 density already is c05; try quadratic density
A = np.vstack([c01 ** 2, c01, np.ones(N)]).T
coefq, *_ = np.linalg.lstsq(A, c05, rcond=None)
qf = A @ coefq
rec("c06_vs_c01_quadfit_absmax", float(np.abs(c06 - c01 * qf).max()))
rec("c06_vs_c01_round_quadfit2dp_absmax", float(np.abs(c06 - np.round(c01 * qf, 2)).max()))
rec("c06_vs_c01_round_quadfit2dp_frac_exact", float(np.mean(np.abs(c06 - np.round(c01 * qf, 2)) < 1e-9)))
# density back-solved from c06: dhat = c06/c01
dhat = np.where(c01 > 0, c06 / np.where(c01 > 0, c01, np.nan), np.nan)
ok = np.isfinite(dhat)
rec("dhat_vs_c05_absmax", float(np.nanmax(np.abs(dhat - c05))))
rec("dhat_vs_c05_q25_q50_q75",
    [float(x) for x in np.nanpercentile(np.abs(dhat - c05), [25, 50, 75, 90, 99])])
rec("dhat_vs_c05_frac_within_0.0005", float(np.nanmean(np.abs(dhat - c05) <= 0.0005)))
rec("dhat_vs_c05_frac_within_0.001", float(np.nanmean(np.abs(dhat - c05) <= 0.001)))
rec("dhat_vs_c05_frac_within_0.005", float(np.nanmean(np.abs(dhat - c05) <= 0.005)))
# is dhat smooth (i.e. c06 computed from an unrounded density)?
rec("dhat_on_0.001_grid_frac", float(np.nanmean(np.abs(dhat * 1000 - np.round(dhat * 1000)) < 1e-6)))
# monotone / smoothness of dhat vs c05: dhat should be near c05 if product; check corr with c01
rec("dhat_minus_c05_corr_with_c01", float(np.corrcoef((dhat - c05)[ok], c01[ok])[0, 1]))

# --------------------------------------------------------------- distinct c05 exact
u5 = np.unique(c05)
rec("c05_n_distinct", int(len(u5)))
rec("c05_n_distinct_rounded3", int(len(np.unique(np.round(c05, 3)))))
rec("c05_n_distinct_as_int_millis", int(len(np.unique(np.round(c05 * 1000).astype(int)))))
rec("c05_missing_millis_in_range",
    sorted(set(range(int(round(u5.min() * 1000)), int(round(u5.max() * 1000)) + 1)) -
           set(np.round(u5 * 1000).astype(int))))

# --------------------------------------------------------------- attack on c04 / c07
say_pairs = {}
# c04 vs c05 (both low-ish cardinality): if c04 were a function of geometry it would show
g = pd.DataFrame({"x": c05, "y": c04}).dropna().groupby("x")["y"].nunique()
rec("c04_per_c05_n_x_with_multiple", int((g > 1).sum()))
rec("c04_per_c05_n_x", int(len(g)))
# c04 duplicates: same c04 at different rows
vc = pd.Series(c04).dropna().value_counts()
rec("c04_exact_duplicate_values", int((vc > 1).sum()))
dupvals = vc[vc > 1].index.values
info = []
for v in dupvals:
    idx = np.where(c04 == v)[0]
    info.append({"c04": float(v), "idxs": [int(i) for i in idx],
                 "c01": [float(c01[i]) for i in idx], "c05": [float(c05[i]) for i in idx],
                 "c07": [str(c["c07"].values[i]) for i in idx]})
rec("c04_duplicate_detail", info)
# c07 labels vs geometry
lab = c["c07"].values.astype(str)
mask = lab != "nan"
rec("c07_label_rows", [{"idx": int(i), "c01": float(c01[i]), "c03": float(c03[i]),
                        "c05": float(c05[i]), "c06": float(c06[i]), "c04": None if pd.isna(c04[i]) else float(c04[i]),
                        "label": str(lab[i])} for i in np.where(mask)[0]])
# is c07 derivable from c08/c09 (the dated rows)?
rec("c07_unique_labels", dict(Counter([str(x) for x in lab[mask]])))

# --------------------------------------------------------------- figures
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(c01, r_aligned, lw=0.8, label="direct cumsum(c03)/100")
ax.plot(c01, r_shift, lw=0.8, label="shifted cumsum(sum of preceding c03)")
ax.axhline(1.008, color="r", ls="--", lw=1, label="+1.008 step level")
ax.axhline(0.0, color="k", ls=":", lw=1)
ax.set_xlabel("c01 depth top (m)"); ax.set_ylabel("residual (m)")
ax.set_title("c01 - cumsum(c03)/100: residual drift and the step")
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.plot(c01, c05, ".", ms=2)
ax.set_xlabel("c01 (m)"); ax.set_ylabel("c05 density")
ax.set_title("c05 vs c01: monotone ramp (262 distinct values)")

ax = axes[1, 0]
ax.plot(c01, dprod, ".", ms=2)
ax.set_xlabel("c01 (m)"); ax.set_ylabel("c06 - c01*c05 (m)")
ax.set_title("residual of claim (d): c06 = c01*c05")

ax = axes[1, 1]
ax.plot(c01[ok], (dhat - c05)[ok], ".", ms=2)
ax.set_xlabel("c01 (m)"); ax.set_ylabel("c06/c01 - c05")
ax.set_title("reverse-direction check: density back-solved from c06")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "verify_draft1_figure.png"), dpi=130)

with open(os.path.join(OUT, "verify_draft1_attack.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("WROTE attack json + figure")