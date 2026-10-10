#!/usr/bin/env python3
"""Generate an SVG figure (pure Python, no matplotlib) summarizing the three
registered findings for Information_01_dynamics_generalization.

Reads the machine-readable result artifacts written by analysis.py and emits
output/results/figure.svg.
"""
from __future__ import annotations
import json, os
from pathlib import Path

OUT = Path(__file__).resolve().parent
RES = OUT / "results"

def load(name):
    return json.loads((RES / name).read_text())

def f1(): return load("finding1_within_domain_graph_invariance.json")
def f2(): return load("finding2_boundary_distribution_degradation.json")
def f3(): return load("finding3_ood_paired_state_shift.json")

# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------
def bar(x, y, w, h, fill, stroke="000000", sw=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def esc(s):
    import xml.sax.saxutils as _sx
    return _sx.escape(str(s))

def txt(x, y, s, size=11, anchor="middle", fill="#000"):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-family="Arial,Helvetica,sans-serif" text-anchor="{anchor}" fill="{fill}">{esc(s)}</text>'

def fmt(v):
    return f"{v:.3f}"

W = 1150
# three stacked panels, each H=230, gaps
PH = 240
GAP = 20
TOP = 40
H = TOP + 3*PH + GAP*2 + 120
CW = W - 2*40
CX0 = 40

# ---- panel 1: within-domain mean error across panels (grouped by graph type) ----
p1 = f1()
panels = p1["panels"]
order = ["training_graph_uniform","equivalent_graph_uniform","equivalent_graph_beta_5_2",
         "higher_degree_graph_uniform","higher_degree_graph_beta_5_2","equivalent_graph_beta_5_2"]
# sort by graph then distribution for readability
def sortkey(k):
    base = k.split("_uniform")[0].split("_beta_5_2")[0].replace("equivalent_graph","equivalent").replace("higher_degree_graph","higher").replace("training_graph","training")
    return base
order = sorted(order, key=sortkey)
ymin, ymax = 0.4, 0.6
def s1(y): return TOP + 30 + (ymax - y)/(ymax - ymin)*(PH-120)
bw = CW/len(order)*0.6; gap = CW/len(order)
g1 = []
g1.append(txt(CX0+CW/2, TOP+15, "Finding 1: within-domain mean relative error by (graph type, state distribution)", 13, "middle", "#000"))
g1.append(txt(CX0+CW/2, TOP+32, "All panels within ~0.48-0.52 % (graph-type invariance)", 11, "middle", "#333"))
for i,k in enumerate(order):
    v = panels[k]["mean"]
    x = CX0 + i*gap + gap/2 - bw/2
    y = s1(v)
    h = s1(0.4) - y
    g1.append(bar(x, y, bw, h, "#4C9ABF"))
    # x label (short)
    lbl = k.replace("_uniform","").replace("_beta_5_2","").replace("equivalent_graph","eq").replace("higher_degree_graph","hi").replace("training_graph","trn")
    g1.append(txt(CX0 + i*gap + gap/2, s1(0.4)+12, lbl, 10))
    g1.append(txt(x+bw/2, s1(0.4)-6, fmt(v), 10))
g1.append(txt(CX0, s1(ymax), "0.60%", 10, "left"))
g1.append(txt(CX0, s1(ymin), "0.40%", 10, "left"))

# ---- panel 2: boundary distribution degradation ----
p2 = f2()
graphs = sorted(p2["graphs"].keys())
# 3 groups, 3 bars each (uniform, beta_5_2, boundary)
colors = {"uniform":"#86BC8F","beta_5_2":"#FFC07A","boundary":"#D95D39"}
ymin2, ymax2 = 0.4, 1.0
def s2(y): return TOP+PH+GAP+30 + (ymax2 - y)/(ymax2-ymin2)*(PH-130)
nbar = 3
bw2 = (CW/len(graphs))/nbar*0.62
g2 = []
g2.append(txt(CX0+CW/2, TOP+PH+GAP+15, "Finding 2: within-domain mean relative error by state distribution (per graph)", 13, "middle", "#000"))
g2.append(txt(CX0+CW/2, TOP+PH+GAP+32, "Boundary beta(0.5,0.5) ~1.9x higher than uniform/beta(5,2)", 11, "middle", "#333"))
for gi,gn in enumerate(graphs):
    d = p2["graphs"][gn]
    gx = CX0 + gi*(CW/len(graphs)) + (CW/len(graphs))/2
    for bi,dk in enumerate(["uniform","beta_5_2","boundary"]):
        v = d["uniform"]["mean"] if dk=="uniform" else (d["beta_5_2"]["mean"] if dk=="beta_5_2" else d["boundary"]["mean"])
        x = gx - (nbar/2)*(CW/len(graphs)) + bi*(CW/len(graphs))/nbar + (CW/len(graphs))/nbar/2 - bw2/2
        y = s2(v); h = s2(ymin2) - y
        g2.append(bar(x, y, bw2, h, colors[dk]))
        g2.append(txt(CX0 + gi*(CW/len(graphs)) + (CW/len(graphs))/2, s2(ymin2)+14, gn.replace("_graph","").replace("equivalent_novel","eq-novel").replace("higher_degree","hi-degree"), 10))
g2.append(txt(CX0, s2(ymax2), "1.0%", 10, "left"))
g2.append(txt(CX0, s2(ymin2), "0.4%", 10, "left"))

# ---- panel 3: OOD paired state shift ----
p3 = f3()
graphs3 = sorted(p3["graphs"].keys())
ymin3, ymax3 = 0.0, 2.1
def s3(y): return TOP+2*PH+2*GAP+30 + (ymax3 - y)/(ymax3-ymin3)*(PH-130)
bw3 = (CW/len(graphs3))/2*0.6
g3 = []
g3.append(txt(CX0+CW/2, TOP+2*PH+2*GAP+15, "Finding 3: paired state-shift (U(0,1) -> shifted) OOD mean relative error per graph", 13, "middle", "#000"))
g3.append(txt(CX0+CW/2, TOP+2*PH+2*GAP+32, "Out-of-domain error ~3.4-3.7x in-domain on all three graphs (p<1e-165)", 11, "middle", "#333"))
for gi,gn in enumerate(graphs3):
    d = p3["graphs"][gn]
    gx = CX0 + gi*(CW/len(graphs3)) + (CW/len(graphs3))/2
    for bi,dk in enumerate(["in_domain","out_of_domain"]):
        v = d["in_domain"]["mean"] if dk=="in_domain" else d["out_of_domain"]["mean"]
        x = gx - (2/2)*(CW/len(graphs3)) + bi*(CW/len(graphs3))/2 + (CW/len(graphs3))/2 - bw3/2
        y = s3(v); h = s3(0.0) - y
        col = "#3B7A57" if dk=="in_domain" else "#C0392B"
        g3.append(bar(x, y, bw3, h, col))
        g3.append(txt(x+bw3/2, y-5, f"{fmt(v)}%", 10))
    g3.append(txt(gx, s3(0.0)+14, gn.replace("_graph","").replace("equivalent_novel","eq-novel").replace("training","trn"), 10))
g3.append(txt(CX0, s3(ymax3), "2.1%", 10, "left"))
g3.append(txt(CX0, s3(ymin3), "0.0%", 10, "left"))

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial,Helvetica,sans-serif">')
svg.append('<rect width="100%" height="100%" fill="white"/>')
svg.append('<text x="40" y="25" font-size="16" font-weight="bold">Cross-Graph / Cross-State Generalization of a Persisted Network-Dynamics Model</text>')
svg.append('<text x="40" y="38" font-size="11" fill="#333">MAK checkpoint (single model, 100-node ER graphs, n=1000 state draws per panel). Percent relative error.</text>')
svg.append("".join(g1))
svg.append("".join(g2))
svg.append("".join(g3))
svg.append('</svg>')
(RES / "figure.svg").write_text("\n".join(svg), encoding="utf-8")
print("wrote", (RES / "figure.svg"), "bytes:", (RES / "figure.svg").stat().st_size)
