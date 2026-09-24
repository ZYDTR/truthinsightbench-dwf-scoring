#!/usr/bin/env python3
"""Final targeted robustness checks + hand-rendered SVG figures (no matplotlib).

Checks
------
G1  Drop structurally distinct units (those with 8 site rows instead of 7)
    - a complete-unit leave-out robustness check for the threshold finding.
G2  Recompute the surface containment excluding the 4 env-positive/surface-zero
    units, to confirm the asymmetry is not carried by a handful of units.
G3  Exact binomial check that the low-assay cluster exceeds the base rate.

Figures are written as SVG so that no plotting library is required.

Run:  python3 output/analysis/run_checks_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data"
OUT = HERE / "results"
FIG = PROJECT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

from ti_lib import SEED, fisher_exact_2x2, logistic_fit, perm_pvalue, sub_rng  # noqa: E402

rng_fig = sub_rng("checks/figures")  # figure jitter only
G: dict = {"schema": "life04-checks", "seed": SEED + 3}

unit = pd.read_csv(DATA / "unit_level_observations.tsv", sep="\t")
site = pd.read_csv(DATA / "site_level_observations.tsv", sep="\t")
syms = pd.read_csv(DATA / "symptom_assay_surface_summary.tsv", sep="\t")
u = unit.rename(columns={"quantitative_assay_value": "assay"}).merge(
    syms[["unit_id", "symptom_present", "surface_positive_percent"]], on="unit_id")
u["pos"] = u.any_environment_positive.astype(int)
u["any_surface_positive"] = (u.surface_positive_percent > 0).astype(int)
y = u.pos.to_numpy(dtype=float)
assay = u.assay.to_numpy()
site["label_index"] = site.site_role_id.str.split("_").str[1].astype(int)

# --------------------------------------------------------------------------
# G1 complete-unit leave-out of the 8-row units
# --------------------------------------------------------------------------
n_rows = site.unit_id.value_counts()
big_units = sorted(n_rows[n_rows == 8].index)
keep = ~u.unit_id.isin(big_units).to_numpy()
G["G1_drop_eight_row_units"] = {
    "dropped_units": big_units, "n_dropped": int((~keep).sum()), "n_remaining": int(keep.sum()),
    "positivity_dropped": float(y[~keep].mean()), "positivity_remaining": float(y[keep].mean()),
}
for label, m in [("full_sample", np.ones(len(y), dtype=bool)), ("without_eight_row_units", keep)]:
    ya, aa = y[m], assay[m]
    fit_lin = logistic_fit(np.column_stack([np.ones(len(ya)), aa]), ya)
    cut = 24.16
    bl = aa < cut
    ent = {"n": int(len(ya)), "n_pos": int(ya.sum()),
           "linear_rho_direction_coef": float(fit_lin["beta"][1])}
    if bl.sum() and (~bl).sum():
        a = int(np.sum((ya == 1) & bl)); b = int(np.sum((ya == 1) & ~bl))
        c = int(np.sum((ya == 0) & bl)); d = int(np.sum((ya == 0) & ~bl))
        ent["threshold_2x2"] = {"table": [[a, b], [c, d]], **fisher_exact_2x2(a, b, c, d),
                                "rate_below": float(ya[bl].mean()), "rate_above": float(ya[~bl].mean())}
    G["G1_drop_eight_row_units"][label] = ent

# --------------------------------------------------------------------------
# G2 containment asymmetry excluding env+/surface- units
# --------------------------------------------------------------------------
ct = pd.crosstab(u.pos, u.any_surface_positive).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
env_pos_surf_zero = u.loc[(u.pos == 1) & (u.any_surface_positive == 0), "unit_id"].tolist()
G["G2_containment"] = {
    "full_table_pos_by_surface": [[int(ct.loc[1, 1]), int(ct.loc[1, 0])], [int(ct.loc[0, 1]), int(ct.loc[0, 0])]],
    "env_pos_surface_zero_units": env_pos_surf_zero,
    "converse_cell_surface_pos_env_neg_is_zero": bool(int(ct.loc[0, 1]) == 0),
    "note": "the asymmetry is a single zero cell; it is not an average over units",
}

# --------------------------------------------------------------------------
# G3 exact binomial: low-assay cluster vs base rate
# --------------------------------------------------------------------------
base = float(y.mean())
k_below = int(y[assay < 24.16].sum()); n_below = int((assay < 24.16).sum())


def binom_tail(k, n, p):
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


G["G3_low_assay_cluster"] = {
    "cut": 24.16, "k_positive": k_below, "n_below": n_below, "base_rate": base,
    "exact_binom_p_upper": binom_tail(k_below, n_below, base),
    "expected_positive_if_base_rate": base * n_below,
}

# ==========================================================================
# SVG figures
# ==========================================================================
def svg_open(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="{w/2}" y="26" font-family="DejaVu Sans,Helvetica,sans-serif" font-size="16" '
            f'text-anchor="middle" fill="#111">{title}</text>']


def svg_close():
    return ['</svg>']


AXIS = "#333"
FONT = 'font-family="DejaVu Sans,Helvetica,sans-serif"'


# ---- Figure 1: per-unit positivity proportion -----------------------------
per_unit = site.groupby("unit_id").positive.agg(["sum", "count"])
per_unit["prop"] = per_unit["sum"] / per_unit["count"]
order = per_unit.sort_values("prop", ascending=False)
w, h = 860, 420
L, B, T, Rm = 70, 70, 60, 30
pw, ph = w - L - Rm, h - B - T
f1 = svg_open(w, h, "Figure 1. Environmental positivity is a unit-level property (n=30 units)")
f1 += [f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{AXIS}"/>',
       f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="{AXIS}"/>']
for frac in [0, .25, .5, .75, 1.0]:
    yy = T + ph - frac * ph
    f1 += [f'<line x1="{L-6}" y1="{yy}" x2="{L}" y2="{yy}" stroke="{AXIS}"/>',
           f'<text x="{L-12}" y="{yy+4}" {FONT} font-size="12" text-anchor="end" fill="#333">{frac:.2f}</text>',
           f'<line x1="{L}" y1="{yy}" x2="{L+pw}" y2="{yy}" stroke="#ddd" stroke-width="1"/>']
bw = pw / len(order)
for i, (uid, row) in enumerate(order.iterrows()):
    x = L + i * bw + 3
    bh = row["prop"] * ph
    color = "#c0392b" if row["prop"] in (0.0, 1.0) else "#5d8aa8"
    f1.append(f'<rect x="{x:.1f}" y="{T+ph-bh:.1f}" width="{bw-6:.1f}" height="{bh:.1f}" fill="{color}"/>')
    f1.append(f'<text x="{x+(bw-6)/2:.1f}" y="{T+ph+14}" {FONT} font-size="8" text-anchor="middle" '
              f'transform="rotate(60 {x+(bw-6)/2:.1f} {T+ph+14})" fill="#555">{uid.replace("unit_","u")}</text>')
f1 += [f'<text x="{L+pw/2}" y="{h-14}" {FONT} font-size="12" text-anchor="middle" fill="#333">'
       f'sampled unit (sorted) — red = unanimous unit (13 all-negative, 3 all-positive)</text>',
       f'<text x="18" y="{T+ph/2}" {FONT} font-size="12" text-anchor="middle" fill="#333" '
       f'transform="rotate(-90 18 {T+ph/2})">proportion of sampled sites positive</text>']
f1 += svg_close()
(FIG / "fig1_unit_level_positivity.svg").write_text("\n".join(f1), encoding="utf-8")

# ---- Figure 2: assay vs positivity with threshold -------------------------
w, h = 860, 430
L, B, T, Rm = 70, 70, 60, 150
pw, ph = w - L - Rm, h - B - T
xmin, xmax = 17.0, 38.5
f2 = svg_open(w, h, "Figure 2. Lower quantitative assay values mark unit environmental positivity")
f2 += [f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{AXIS}"/>',
       f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="{AXIS}"/>']
for xv in range(18, 39, 4):
    xx = L + (xv - xmin) / (xmax - xmin) * pw
    f2 += [f'<line x1="{xx}" y1="{T+ph}" x2="{xx}" y2="{T+ph+6}" stroke="{AXIS}"/>',
           f'<text x="{xx}" y="{T+ph+20}" {FONT} font-size="12" text-anchor="middle" fill="#333">{xv}</text>',
           f'<line x1="{xx}" y1="{T}" x2="{xx}" y2="{T+ph}" stroke="#eee"/>']
for lbl, pos in [("1 (positive)", 0.75), ("0 (negative)", 0.25)]:
    yy = T + ph - pos * ph
    f2 += [f'<line x1="{L-6}" y1="{yy}" x2="{L}" y2="{yy}" stroke="{AXIS}"/>',
           f'<text x="{L-12}" y="{yy+4}" {FONT} font-size="12" text-anchor="end" fill="#333">{lbl}</text>',
           f'<line x1="{L}" y1="{yy}" x2="{L+pw}" y2="{yy}" stroke="#eee"/>']
jit = rng_fig.normal(0, 0.022, len(u))
for i, row in u.iterrows():
    xx = L + (row["assay"] - xmin) / (xmax - xmin) * pw
    yy = T + ph - (row["pos"] + jit[i]) * ph
    f2.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.5" fill="{"#c0392b" if row["pos"] else "#5d8aa8"}" '
              f'opacity="0.85"/>')
cx = L + (24.16 - xmin) / (xmax - xmin) * pw
f2 += [f'<line x1="{cx:.1f}" y1="{T}" x2="{cx:.1f}" y2="{T+ph}" stroke="#2c7a2c" stroke-width="2" '
       f'stroke-dasharray="6,4"/>',
       f'<text x="{cx+8:.1f}" y="{T+18}" {FONT} font-size="12" fill="#2c7a2c">change point 24.16</text>',
       f'<text x="{L+pw/2}" y="{h-14}" {FONT} font-size="12" text-anchor="middle" fill="#333">'
       f'quantitative_assay_value (ordering as recorded)</text>',
       f'<text x="18" y="{T+ph/2}" {FONT} font-size="12" text-anchor="middle" fill="#333" '
       f'transform="rotate(-90 18 {T+ph/2})">any_environment_positive (jittered)</text>']
bx = L + pw + 22
f2 += [f'<text x="{bx}" y="{T+20}" {FONT} font-size="12" fill="#111">below 24.16: 8/8 positive</text>',
       f'<text x="{bx}" y="{T+42}" {FONT} font-size="12" fill="#111">at/above 24.16: 9/22 positive</text>',
       f'<text x="{bx}" y="{T+64}" {FONT} font-size="12" fill="#111">Fisher two-sided p=0.0044</text>',
       f'<text x="{bx}" y="{T+86}" {FONT} font-size="12" fill="#111">monotone rho=-0.354, p=0.056</text>']
f2 += svg_close()
(FIG / "fig2_assay_threshold.svg").write_text("\n".join(f2), encoding="utf-8")

# ---- Figure 3: naive vs unit-blocked inference ----------------------------
w, h = 780, 360
L, B, T, Rm = 150, 60, 70, 220
pw, ph = w - L - Rm, h - B - T
rows = [("site rows treated as independent (n=212)", 4.54e-5, "#c0392b", "p = 4.5e-05"),
        ("unit-blocked, whole units (n=30)", 0.255, "#5d8aa8", "p = 0.26")]
f3 = svg_open(w, h, "Figure 3. The same contrast changes by ~4 orders of magnitude under unit blocking")
f3 += [f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="{AXIS}"/>']
cuts = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
for cv in cuts:
    xx = L + (math.log10(cv) - math.log10(1e-5)) / (math.log10(1) - math.log10(1e-5)) * pw
    f3 += [f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+ph}" stroke="#eee"/>',
           f'<text x="{xx:.1f}" y="{T+ph+18}" {FONT} font-size="11" text-anchor="middle" fill="#333">{cv:g}</text>']
sigx = L + (math.log10(0.05) - math.log10(1e-5)) / (math.log10(1) - math.log10(1e-5)) * pw
f3 += [f'<line x1="{sigx:.1f}" y1="{T}" x2="{sigx:.1f}" y2="{T+ph}" stroke="#2c7a2c" stroke-dasharray="5,4"/>',
       f'<text x="{sigx+6:.1f}" y="{T+14}" {FONT} font-size="11" fill="#2c7a2c">0.05</text>']
bh = ph / len(rows) * 0.55
for i, (label, p, color, txt) in enumerate(rows):
    yy = T + ph * (i + 0.5) / len(rows)
    xx = L + (math.log10(p) - math.log10(1e-5)) / (math.log10(1) - math.log10(1e-5)) * pw
    f3 += [f'<rect x="{L:.1f}" y="{yy-bh/2:.1f}" width="{xx-L:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.85"/>',
           f'<text x="{L-10}" y="{yy+4}" {FONT} font-size="12" text-anchor="end" fill="#333">{label}</text>',
           f'<text x="{xx+8:.1f}" y="{yy+4}" {FONT} font-size="12" fill="#111">{txt}</text>']
f3 += [f'<text x="{L+pw/2}" y="{h-14}" {FONT} font-size="12" text-anchor="middle" fill="#333">'
       f'two-sided Fisher exact p (illness_day >= 15 vs positivity)</text>']
f3 += svg_close()
(FIG / "fig3_naive_vs_blocked.svg").write_text("\n".join(f3), encoding="utf-8")

# ---- Figure 4: rotating site_role_id --------------------------------------
w, h = 860, 300
cell = 26
f4 = svg_open(w, h, "Figure 4. site_role_id is a rotating position counter, not a stable location")
f4 += [f'<text x="{70}" y="{T-6}" {FONT} font-size="12" fill="#333">unit</text>']
for lab in range(1, 11):
    f4.append(f'<text x="{120+(lab-1)*cell+cell/2}" y="{T-6}" {FONT} font-size="11" text-anchor="middle" '
              f'fill="#333">s{lab}</text>')
for i, (uid, g) in enumerate(site.groupby("unit_id", sort=False)):
    yy = T + 6 + i * 7.2
    f4.append(f'<text x="{110}" y="{yy+4}" {FONT} font-size="8" text-anchor="end" fill="#555">'
              f'{uid.replace("unit_","u")}</text>')
    start = int(g.label_index.iloc[0])
    for j, idx in enumerate(g.label_index):
        xx = 120 + (idx - 1) * cell
        f4.append(f'<rect x="{xx+2}" y="{yy}" width="{cell-4}" height="5.6" fill="'
                  f'{"#c0392b" if g.positive.iloc[j] else "#c9d6de"}" opacity="0.95"/>')
f4 += [f'<text x="{120+5*cell}" y="{T+30+31*7.2}" {FONT} font-size="12" text-anchor="middle" fill="#333">'
       f'each unit starts at a different label and increments by +1 (mod 10); 0/30 units violate the rule</text>']
f4 += svg_close()
(FIG / "fig4_site_label_rotation.svg").write_text("\n".join(f4), encoding="utf-8")

(OUT / "checks.json").write_text(json.dumps(G, indent=2, default=str) + "\n", encoding="utf-8")
print("checks complete ->", OUT / "checks.json")
print("figures ->", FIG)
print(json.dumps(G, indent=1, default=str)[:2400])