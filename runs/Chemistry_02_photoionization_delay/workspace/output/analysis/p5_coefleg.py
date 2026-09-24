#!/usr/bin/env python3
"""
p5_coefleg -- Angular-coefficient (CoefLeg) blocks: structure, relation to the
dense experiment panels, and predictability from the experiment dipole-matrix
(DME) blocks.

Deliverables (all under ROOT/output/):
  analysis/p5_coefleg.py            this script
  evidence/p5_coefleg.json          machine-readable results
  evidence/p5_coefleg.log           stdout log
  evidence/p5_coefleg.png           figure
  evidence/p5_coefleg_summary.csv   per-energy per-family summary table

Re-runnable from scratch; reads only ROOT/data/*.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import eval_legendre, lpmv

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
OUT_ANALYSIS = os.path.join(ROOT, "output", "analysis")
OUT_EVIDENCE = os.path.join(ROOT, "output", "evidence")
os.makedirs(OUT_ANALYSIS, exist_ok=True)
os.makedirs(OUT_EVIDENCE, exist_ok=True)

JSON_PATH = os.path.join(OUT_EVIDENCE, "p5_coefleg.json")
LOG_PATH = os.path.join(OUT_EVIDENCE, "p5_coefleg.log")
PNG_PATH = os.path.join(OUT_EVIDENCE, "p5_coefleg.png")
CSV_PATH = os.path.join(OUT_EVIDENCE, "p5_coefleg_summary.csv")

T0 = time.time()
FAILURES = []


class Tee:
    """Duplicate stdout into the log file."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


sys.stdout = Tee(LOG_PATH)


def say(*a):
    print(*a)
    sys.stdout.flush()


# ----------------------------------------------------------------------------
# 0. Load
# ----------------------------------------------------------------------------
say("=" * 78)
say("p5_coefleg: CoefLeg blocks -- structure, dense-panel reconstruction, DME link")
say("=" * 78)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DATA_FILES = [
    "angular_coefficients_datasets.tsv",
    "angle_energy_observations_datasets.tsv",
    "dipole_matrix_elements_datasets.tsv",
    "coordinate_definitions.json",
]
SHA = {}
for fn in DATA_FILES:
    p = os.path.join(DATA, fn)
    SHA[fn] = sha256(p)
    say("sha256 %s  %s" % (SHA[fn], fn))

coord = json.load(open(os.path.join(DATA, "coordinate_definitions.json")))
say("coordinate_definitions: angle step %s deg, hbar_as_eV=%s"
    % (coord["angle_coordinate"]["step_deg"], coord["phase_delay_conversion"]["hbar_as_eV"]))

cf = pd.read_csv(os.path.join(DATA, "angular_coefficients_datasets.tsv"), sep="\t")
obs = pd.read_csv(os.path.join(DATA, "angle_energy_observations_datasets.tsv"), sep="\t")
dme = pd.read_csv(os.path.join(DATA, "dipole_matrix_elements_datasets.tsv"), sep="\t")

COEF_NAME = sorted(cf["dataset_path"].unique())
CE_EV = [float(n.replace("CoefLeg_", "")[:-2]) for n in COEF_NAME]
COEF_BY_E = dict(zip(CE_EV, COEF_NAME))
say("\nCoefLeg groups (%d):" % len(CE_EV))
say("  energies eV: %s" % CE_EV)

B = {}
for E in CE_EV:
    M = (cf[cf["dataset_path"] == COEF_BY_E[E]]
         .pivot_table(index="index_0", columns="index_1", values="value").to_numpy())
    B[E] = M
    assert M.shape == (9, 11), (E, M.shape)

MAG = (obs[obs["dataset_path"] == "experiment/magnitude"]
       .pivot_table(index="index_0", columns="index_1", values="value").to_numpy())
PHA = (obs[obs["dataset_path"] == "experiment/phase"]
       .pivot_table(index="index_0", columns="index_1", values="value").to_numpy())
E_EXP = (obs[obs["dataset_path"] == "experiment/photon_energy_eV"]
         .sort_values("index_0")["value"].to_numpy())
THETA = np.arange(181.0)
CTH = np.cos(np.deg2rad(THETA))

# exact-match membership on the dense 0.1 eV experiment grid
ON_GRID = [E for E in CE_EV if np.any(np.abs(E_EXP - E) < 1e-9)]
OFF_GRID = [E for E in CE_EV if E not in ON_GRID]
say("CoefLeg energies exactly on the dense 0.1 eV grid (%d): %s" % (len(ON_GRID), ON_GRID))
say("CoefLeg energies NOT on the grid (%d): %s" % (len(OFF_GRID), OFF_GRID))


def mag_row(E):
    """Dense experiment magnitude row for the nearest 0.1 eV grid energy."""
    if E in ON_GRID:
        i = int(np.argmin(np.abs(E_EXP - E)))
    else:
        i = int(np.argmin(np.abs(E_EXP - E)))  # nearest, flagged as off-grid
    return i, MAG[i]


FAMILIES = ["F00", "F20", "F21", "F22", "F11"]
FCOL = {"F00": 1, "F20": 3, "F21": 5, "F22": 7, "F11": 9}   # value columns
ECOL = {"F00": 2, "F20": 4, "F21": 6, "F22": 8, "F11": 10}  # error columns

# ----------------------------------------------------------------------------
# (a) Structural characterisation
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(a) STRUCTURAL CHARACTERISATION")
say("-" * 78)

stack = np.stack([B[E] for E in CE_EV])          # (13, 9, 11)
say("stacked shape %s" % (stack.shape,))

# exact zero pattern: entries that are exactly 0.0 in EVERY energy block
zero_every = {}
for c in range(11):
    zc = [k for k in range(9) if np.all(stack[:, k, c] == 0.0)]
    zero_every[c] = zc
say("exact-zero entries (identical across all 13 energies), by column:")
for c in range(11):
    say("   col %2d (order rows %s)" % (c, zero_every[c]))

for fam in FAMILIES:
    say("   %s: value always zero at orders %s ; error always zero at orders %s"
        % (fam, zero_every[FCOL[fam]], zero_every[ECOL[fam]]))

f00_0 = sorted(set(round(float(B[E][0, 1]), 6) for E in CE_EV))
say("F00 at order 0 takes values %s  (i.e. pinned to 1.0 +- 4e-5)" % f00_0)

# error-column scaling
say("\nerror-column scaling (errF20/errF00 is order independent?):")
try:
    for E in CE_EV:
        b = B[E]
        ratio = b[:, ECOL["F20"]] / b[:, ECOL["F00"]]
        say("   E=%6.2f  mean=%.5f  spread=%.2e"
            % (E, float(ratio.mean()), float(np.max(ratio) - np.min(ratio))))
    # is errF00 itself proportional to sqrt(2k+1)?
    for E in CE_EV:
        b = B[E]
        pred = b[0, ECOL["F00"]] * np.sqrt(2 * np.arange(9) + 1)
        say("   E=%6.2f  errF00 vs errF00(0)*sqrt(2k+1): max rel dev = %.4g"
            % (E, float(np.max(np.abs(b[:, ECOL["F00"]] - pred) / pred))))
except Exception as exc:
    FAILURES.append("errF00 scaling check failed: %r" % (exc,))
    say("   FAILED errF00 scaling check: %r" % (exc,))

say("\nfamily magnitude (sum over the 9 order rows):")
fam_mag = {}
for E in CE_EV:
    b = B[E]
    fam_mag[E] = {
        "F00_sum": float(b[:, 1].sum()),
        "F20_sum": float(b[:, 3].sum()),
        "F21_abssum": float(np.abs(b[:, 5]).sum()),
        "F22_abssum": float(np.abs(b[:, 7]).sum()),
        "F11_abssum": float(np.abs(b[:, 9]).sum()),
    }
    say("   E=%6.2f  F00=%8.4f  F20=%8.4f  |F21|=%7.4f  |F22|=%7.4f  |F11|=%7.4f"
        % (E, fam_mag[E]["F00_sum"], fam_mag[E]["F20_sum"], fam_mag[E]["F21_abssum"],
           fam_mag[E]["F22_abssum"], fam_mag[E]["F11_abssum"]))

# energy trend of the order-0 anisotropy proxy
say("\norder-0 family values across energy:")
for E in CE_EV:
    say("   E=%6.2f  F00(0)=%.5f  F20(0)=%9.5f  F21(0)=%8.5f  F22(0)=%8.5f  F11(0)=%8.5f"
        % (E, B[E][0, 1], B[E][0, 3], B[E][0, 5], B[E][0, 7], B[E][0, 9]))

# mean error profile per family/order
say("\nmean error magnitude per order (averaged over 13 energies):")
err_prof = {}
for fam in FAMILIES:
    prof = np.mean([B[E][:, ECOL[fam]] for E in CE_EV], axis=0)
    err_prof[fam] = [float(x) for x in prof]
    say("   err%s: %s" % (fam, np.round(prof, 5)))

# S/N of the order-0 coefficient
say("\norder-0 signal-to-noise per family (per energy):")
for E in CE_EV:
    sn = {f: (float(abs(B[E][0, FCOL[f]]) / B[E][0, ECOL[f]])) for f in FAMILIES if B[E][0, ECOL[f]] > 0}
    say("   E=%6.2f  %s" % (E, {k: round(v, 2) for k, v in sn.items()}))

# ----------------------------------------------------------------------------
# (b) Reconstruction of the angle-resolved intensity from CoefLeg
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(b) RECONSTRUCTION OF I(theta) FROM THE CoefLeg BLOCK")
say("-" * 78)


def Plm(L, M, CS=True, normed=True, sq2=True):
    """Associated Legendre P_L^M(cos theta) with optional Condon-Shortley phase,
    optional Y_LM normalisation, optional sqrt(2) for the m>0 'real harmonic' factor."""
    if M > L or L < 0:
        return np.zeros_like(CTH)
    v = lpmv(M, L, CTH)
    if (not CS) and M > 0:
        v = v * ((-1) ** M)
    if normed:
        v = v * math.sqrt((2 * L + 1) / (4.0 * math.pi)
                          * math.factorial(L - M) / math.factorial(L + M))
    if M > 0 and sq2:
        v = v * math.sqrt(2.0)
    return v


def aff_r2(rec, y):
    """R2 of the best affine (a + b*rec) fit -- removes the unknown normalisation."""
    A = np.stack([np.ones_like(rec), rec], 1)
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    p = A @ c
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def best_scale(rec, y):
    return float((rec @ y) / (rec @ rec))


def fixed_rms(rec, y):
    return float(np.sqrt(((rec - y) ** 2).mean()) / np.abs(y).mean())


# --- Two readings of the block ------------------------------------------------
# Reading A ("family = harmonic"): the 5 families are 5 fixed harmonics
#   (L,M) = (0,0),(2,0),(2,1),(2,2),(1,1); the 9 order rows are 9 terms.
READ_A = [(1, 0, 0), (3, 2, 0), (5, 2, 1), (7, 2, 2), (9, 1, 1)]  # (col, L, M)
# Reading B ("order = degree"): the order index k IS the Legendre/associated
#   order; the trailing digit of the family name gives m.
READ_B = [(1, 0), (3, 0), (5, 1), (7, 2), (9, 1)]  # (col, m) with L = k

REC = {}
for CS in (True, False):
    for normed in (True, False):
        for sq2 in (True, False):
            tag = "CS%d_norm%d_sq2%d" % (int(CS), int(normed), int(sq2))
            recA, recB = {}, {}
            for E in CE_EV:
                ya = sum(B[E][k, c] * Plm(L, M, CS, normed, sq2) for c, L, M in READ_A for k in range(9))
                yb = sum(B[E][k, c] * Plm(k, m, CS, normed, sq2) for c, m in READ_B for k in range(9))
                recA[E] = ya
                recB[E] = yb
            REC[tag] = {"A": recA, "B": recB}

say("Convention grid: Condon-Shortley phase on/off x Y-normalised/unnormalised x sqrt(2) m>0 factor on/off")
say("Reading A: I(theta)=sum_{k=0..8} [F00_k Y00 + F20_k ReY20 + F21_k ReY21 + F22_k ReY22 + F11_k ReY11]")
say("Reading B: I(theta)=sum_{k=0..8} [F00_k Y_k0 + F20_k Y_k0 + F21_k ReY_k1 + F22_k ReY_k2 + F11_k ReY_k1]")

say("\nReading A, affine-R2 (best scale+offset per energy) and corr:")
say("%-22s %8s %8s %8s %8s   %8s %8s" % ("convention", "meanR2", "minR2", "maxR2", "meanCorr", "meanFixRelRMS", "scale"))
readA_stats = {}
for tag, d in REC.items():
    r2s, cors, rels, scs = [], [], [], []
    for E in CE_EV:
        i, y = mag_row(E)
        r = d["A"][E]
        r2s.append(aff_r2(r, y))
        cors.append(float(np.corrcoef(r, y)[0, 1]))
        rels.append(fixed_rms(r, y))
        scs.append(best_scale(r, y))
    readA_stats[tag] = {"mean_affineR2": float(np.mean(r2s)), "min": float(np.min(r2s)),
                        "max": float(np.max(r2s)), "mean_corr": float(np.mean(cors)),
                        "mean_fixed_relRMS": float(np.mean(rels)),
                        "mean_best_scale": float(np.mean(scs)),
                        "per_energy_affineR2": [float(x) for x in r2s]}
    say("%-22s %8.4f %8.4f %8.4f %8.4f   %12.4f %8.4f"
        % (tag, np.mean(r2s), np.min(r2s), np.max(r2s), np.mean(cors), np.mean(rels), np.mean(scs)))

say("\nReading B, affine-R2 (best scale+offset per energy) and corr:")
readB_stats = {}
for tag, d in REC.items():
    r2s, cors, rels, scs = [], [], [], []
    for E in CE_EV:
        i, y = mag_row(E)
        r = d["B"][E]
        r2s.append(aff_r2(r, y))
        cors.append(float(np.corrcoef(r, y)[0, 1]))
        rels.append(fixed_rms(r, y))
        scs.append(best_scale(r, y))
    readB_stats[tag] = {"mean_affineR2": float(np.mean(r2s)), "min": float(np.min(r2s)),
                        "max": float(np.max(r2s)), "mean_corr": float(np.mean(cors)),
                        "mean_fixed_relRMS": float(np.mean(rels)),
                        "mean_best_scale": float(np.mean(scs)),
                        "per_energy_affineR2": [float(x) for x in r2s]}
    say("%-22s %8.4f %8.4f %8.4f %8.4f   %12.4f %8.4f"
        % (tag, np.mean(r2s), np.min(r2s), np.max(r2s), np.mean(cors), np.mean(rels), np.mean(scs)))

# --- baseline/null: the dense panel's own 9-term Legendre fit -----------------
say("\nBaseline / ceiling: dense magnitude panel fitted by its OWN 9 Legendre terms P_0..P_8")
base_r2, base_rel = [], []
PL9 = np.stack([eval_legendre(k, CTH) for k in range(9)], 1)
for E in CE_EV:
    i, y = mag_row(E)
    c, _, _, _ = np.linalg.lstsq(PL9, y, rcond=None)
    p = PL9 @ c
    base_r2.append(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    base_rel.append(float(np.sqrt(((y - p) ** 2).mean()) / np.abs(y).mean()))
say("   dense panel self-fit: mean R2=%.5f  mean relRMS=%.5f"
    % (np.mean(base_r2), np.mean(base_rel)))

# null: shuffled dense panel vs the best reconstruction
rng = np.random.default_rng(20240919)
best_tag = max(readB_stats, key=lambda t: readB_stats[t]["mean_affineR2"])
i0, y0 = mag_row(CE_EV[0])
sh = [aff_r2(REC[best_tag]["B"][CE_EV[0]], rng.permutation(y0)) for _ in range(500)]
sh = np.array(sh)
say("   null (500 label shuffles of the 22.5 eV panel vs reading-B record %s): "
    "mean=%.4f 95pct=%.4f  true=%.4f"
    % (best_tag, sh.mean(), np.quantile(sh, 0.95),
       aff_r2(REC[best_tag]["B"][CE_EV[0]], y0)))

# --- also compare against magnitude^2 ----------------------------------------
say("\nCorrelation of the reading-B reconstruction with magnitude and with magnitude^2:")
mag2_cmp = []
for E in CE_EV:
    i, y = mag_row(E)
    r = REC[best_tag]["B"][E]
    sc_m = best_scale(r, y)
    sc_s = best_scale(r, y ** 2)
    rel_m = float(np.sqrt(((sc_m * r - y) ** 2).mean()) / np.abs(y).mean())
    rel_s = float(np.sqrt(((sc_s * r - y ** 2) ** 2).mean()) / np.abs(y ** 2).mean())
    mag2_cmp.append({"E": E, "corr_mag": float(np.corrcoef(r, y)[0, 1]),
                     "corr_mag2": float(np.corrcoef(r, y ** 2)[0, 1]),
                     "relRMS_vs_mag_best_scale": rel_m, "relRMS_vs_mag2_best_scale": rel_s})
    say("   E=%6.2f  corr(rec,mag)=%+.4f  corr(rec,mag^2)=%+.4f  best-scale relRMS: mag=%.4f  mag^2=%.4f"
        % (E, mag2_cmp[-1]["corr_mag"], mag2_cmp[-1]["corr_mag2"], rel_m, rel_s))

# --- row-0 (1 + beta P2) reading ---------------------------------------------
say("\nRow-0-only reconstruction I = F00(0) + F20(0)*P2(cos theta) vs dense magnitude:")
row0 = []
for E in CE_EV:
    i, y = mag_row(E)
    rec = B[E][0, 1] + B[E][0, 3] * eval_legendre(2, CTH)
    row0.append({"E": E, "F00_0": float(B[E][0, 1]), "F20_0": float(B[E][0, 3]),
                 "corr": float(np.corrcoef(rec, y)[0, 1]),
                 "affineR2": aff_r2(rec, y), "best_scale": best_scale(rec, y)})
    say("   E=%6.2f  F00(0)=%.4f F20(0)=%8.4f  corr=%+.4f  affineR2=%.4f  best_scale=%.3f"
        % (E, B[E][0, 1], B[E][0, 3], row0[-1]["corr"], row0[-1]["affineR2"], row0[-1]["best_scale"]))

# --- the two independent Legendre series present in the dense panel -----------
say("\nDense-panel Legendre content (c_L from a 9-term fit), and comparison with")
say("the CoefLeg columns under the 'order = degree' reading B:")
leg_coef = {}
for E in CE_EV:
    i, y = mag_row(E)
    c, _, _, _ = np.linalg.lstsq(PL9, y, rcond=None)
    leg_coef[E] = c
say("   E=%6.2f  magLeg c_0..c_8 = %s" % (CE_EV[0], np.round(leg_coef[CE_EV[0]], 4)))
say("   E=%6.2f  magLeg c_0..c_8 = %s" % (CE_EV[-1], np.round(leg_coef[CE_EV[-1]], 4)))
cors_by_order = {}
for k in range(9):
    v = np.array([leg_coef[E][k] for E in CE_EV])
    if np.std(v) == 0:
        cors_by_order[k] = None
        continue
    cors_by_order[k] = {f: float(np.corrcoef(np.array([B[E][k, FCOL[f]] for E in CE_EV]), v)[0, 1])
                        for f in FAMILIES}
say("   corr(CoefLeg col at order k, magLegend c_k) across the 13 energies:")
for k in range(9):
    say("     k=%d  %s" % (k, {f: (None if cors_by_order[k] is None else round(cors_by_order[k][f], 3))
                               for f in FAMILIES}))

# beta from a 2-term (1+P2) fit of the dense panel, vs F20(0)/F00(0)
say("\nAnisotropy cross-check: beta of a 1+beta*P2 fit to the dense panel")
say("   vs F20(0)/F00(0) from the CoefLeg row 0:")
beta_dense = []
for E in CE_EV:
    i, y = mag_row(E)
    A2 = np.stack([np.ones_like(CTH), eval_legendre(2, CTH)], 1)
    c, _, _, _ = np.linalg.lstsq(A2, y, rcond=None)
    beta_dense.append(float(c[1] / c[0]))
    say("   E=%6.2f  beta_dense=%+8.4f   F20(0)/F00(0)=%8.4f   F20(0)=%8.4f"
        % (E, beta_dense[-1], B[E][0, 3] / B[E][0, 1], B[E][0, 3]))
r_beta = float(np.corrcoef(np.array([B[E][0, 3] / B[E][0, 1] for E in CE_EV]), beta_dense)[0, 1])
say("   corr(F20(0)/F00(0), beta_dense) = %.4f" % r_beta)

# ----------------------------------------------------------------------------
# (c) Predictability from the experiment DME block
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(c) PREDICTABILITY OF THE CoefLeg FAMILIES FROM THE EXPERIMENT DME BLOCK")
say("-" * 78)

D_BY_E = {}
for E in CE_EV:
    key = "experiment/DME_%seV" % repr(E)
    try:
        M = (dme[dme["dataset_path"] == key]
             .pivot_table(index="index_0", columns="index_1", values="value").to_numpy())
        assert M.shape == (7, 6), (key, M.shape)
        D_BY_E[E] = M
    except Exception as exc:  # pragma: no cover
        FAILURES.append("DME load failed for %s: %r" % (key, exc))
        D_BY_E[E] = None

d0 = D_BY_E[CE_EV[0]]
say("DME rows: (l,m) = %s" % [(int(d0[r, 0]), int(d0[r, 1])) for r in range(7)])
say("The experiment DME block contains only m in {0,1}; the l=1,2,3 channels for m=0 and m=1.")

DIC_LABELS = []
for r in range(7):
    for g in ("Re", "Im"):
        DIC_LABELS.append("d[l%d,m%d].%s" % (int(d0[r, 0]), int(d0[r, 1]), g))
PROD_LABELS = []
for i in range(14):
    for j in range(i, 14):
        PROD_LABELS.append(("%s^2" % DIC_LABELS[i]) if i == j
                           else ("%s*%s" % (DIC_LABELS[i], DIC_LABELS[j])))


def dme_dictionary(E):
    d = D_BY_E[E]
    z = np.concatenate([[d[r, 2], d[r, 3]] for r in range(7)])  # Re,Im for all 7 rows
    return np.array([z[i] * z[j] for i in range(14) for j in range(i, 14)])


DIC = np.array([dme_dictionary(E) for E in CE_EV])  # 13 x 105
say("candidate dictionary: all 105 products of one DME Re/Im entry with another, per energy")

fam_res = {}
for fam in FAMILIES:
    Y = np.array([[B[E][k, FCOL[fam]] for k in range(9)] for E in CE_EV])  # 13 x 9
    # ---- in-sample: 105 features x 13 energies, 9 outputs. Heavily over-parameterised.
    c, _, rank, _ = np.linalg.lstsq(DIC, Y, rcond=None)
    p = DIC @ c
    r2_in = float(1 - ((Y - p) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    # ---- honest LOO: greedy forward selection of 1 and 2 features, + intercept
    n = 13
    loo = {}
    for nf in (1, 2):
        pred = np.zeros_like(Y)
        for k in range(n):
            tr = np.array([j for j in range(n) if j != k])
            Xtr, Ytr = DIC[tr], Y[tr]
            sel = []
            for _ in range(nf):
                b = None
                for j in range(105):
                    if j in sel:
                        continue
                    xs = np.column_stack([Xtr[:, s] for s in sel + [j]] + [np.ones(len(tr))])
                    cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
                    ess = float(((Ytr - xs @ cc) ** 2).sum())
                    if b is None or ess < b[0]:
                        b = (ess, j)
                sel.append(b[1])
            xs = np.column_stack([Xtr[:, s] for s in sel] + [np.ones(len(tr))])
            cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
            pred[k] = np.array([DIC[k, s] for s in sel] + [1.0]) @ cc
        loo[nf] = float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    # ---- strongest single feature over all 13 energies
    flatY = Y.ravel()
    Xrep = np.repeat(DIC, 9, axis=0)
    best = None
    for j in range(105):
        x = Xrep[:, j]
        if np.std(x) == 0:
            continue
        r = float(np.corrcoef(x, flatY)[0, 1])
        if best is None or abs(r) > abs(best[1]):
            best = (j, r)
    # ---- permutation null for that max|r| (shuffle the energy label of the target)
    nullmax = np.zeros(200)
    for t in range(200):
        yp = rng.permutation(flatY)
        rp = np.array([np.nan if np.std(Xrep[:, j]) == 0
                       else abs(np.corrcoef(Xrep[:, j], yp)[0, 1]) for j in range(105)])
        nullmax[t] = np.nanmax(rp)
    p_perm = float((nullmax >= abs(best[1])).mean())
    # ---- permutation null for the greedy-LOO R2 (the load-bearing number)
    loo_null = np.zeros(200)
    for t in range(200):
        Yp = Y[rng.permutation(n)]
        pred = np.zeros_like(Yp)
        for k in range(n):
            tr = np.array([j for j in range(n) if j != k])
            Xtr, Ytr = DIC[tr], Yp[tr]
            sel = []
            for _ in range(2):
                b = None
                for j in range(105):
                    if j in sel:
                        continue
                    xs = np.column_stack([Xtr[:, s] for s in sel + [j]] + [np.ones(len(tr))])
                    cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
                    ess = float(((Ytr - xs @ cc) ** 2).sum())
                    if b is None or ess < b[0]:
                        b = (ess, j)
                sel.append(b[1])
            xs = np.column_stack([Xtr[:, s] for s in sel] + [np.ones(len(tr))])
            cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
            pred[k] = np.array([DIC[k, s] for s in sel] + [1.0]) @ cc
        loo_null[t] = float(1 - ((Yp - pred) ** 2).sum() / ((Yp - Yp.mean(0)) ** 2).sum())
    p_loo_2 = float((loo_null >= loo[2]).mean())
    fam_res[fam] = {
        "in_sample_R2_full_dict": r2_in,
        "LOO_R2_1feature": loo[1],
        "LOO_R2_2feature": loo[2],
        "LOO_R2_2feature_perm_null_mean": float(loo_null.mean()),
        "LOO_R2_2feature_perm_null_95pct": float(np.quantile(loo_null, 0.95)),
        "LOO_R2_2feature_perm_p": p_loo_2,
        "best_single_feature": PROD_LABELS[best[0]],
        "best_single_r": float(best[1]),
        "best_single_r2": float(best[1] ** 2),
        "perm_p_for_max_r": p_perm,
        "perm_null95_max_abs_r": float(np.quantile(nullmax, 0.95)),
        "in_sample_rank": int(rank),
    }
    say("  %s: in-sample R2(full 105-feature dict, rank %d)=%.4f [SATURATED: rank = n_energies] | "
        "LOO R2(1 feat)=%.4f (2 feat)=%.4f | LOO2 perm-null mean=%.3f 95pct=%.3f p=%.3f | "
        "best single: %s r=%.3f R2=%.3f perm-p=%.3f (null95 max|r|=%.3f)"
        % (fam, rank, r2_in, loo[1], loo[2], loo_null.mean(), np.quantile(loo_null, 0.95),
           p_loo_2, PROD_LABELS[best[0]], best[1], best[1] ** 2, p_perm,
           np.quantile(nullmax, 0.95)))

say("\nBaseline for (c): predicting a family with its 13-energy mean gives R2 = 0 by")
say("construction; the LOO numbers above are the honest generalisation estimate.")
say("The full-dictionary in-sample R2 is saturated (13 energies, rank-13 design) and is")
say("reported only to show that; the greedy-LOO R2 against its own permutation null is")
say("the load-bearing statistic.")

# ---- CRITICAL competing baseline: predict each family from ENERGY ALONE ---------
# If a smooth trend in photon energy reproduces the family as well as the DME
# dictionary does, the dictionary adds nothing beyond the energy dependence.
say("\nCOMPETING BASELINE for (c): predict each family value at energy E by linear")
say("interpolation in photon energy from the other 12 energies (no DME at all).")
Earr = np.array(CE_EV)
energy_base = {}
for fam in FAMILIES:
    Y = np.array([[B[E][k, FCOL[fam]] for k in range(9)] for E in CE_EV])
    pred = np.zeros_like(Y)
    for k in range(13):
        tr = np.array([j for j in range(13) if j != k])
        Xtr = np.column_stack([np.ones(len(tr)), Earr[tr]])
        cc, _, _, _ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)
        pred[k] = np.array([1.0, Earr[k]]) @ cc
    r2_e = float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    # permutation null for the energy-only model
    ne = np.zeros(200)
    for t in range(200):
        Yp = Y[rng.permutation(13)]
        pr = np.zeros_like(Yp)
        for k in range(13):
            tr = np.array([j for j in range(13) if j != k])
            Xtr = np.column_stack([np.ones(len(tr)), Earr[tr]])
            cc, _, _, _ = np.linalg.lstsq(Xtr, Yp[tr], rcond=None)
            pr[k] = np.array([1.0, Earr[k]]) @ cc
        ne[t] = float(1 - ((Yp - pr) ** 2).sum() / ((Yp - Yp.mean(0)) ** 2).sum())
    energy_base[fam] = {"LOO_R2_energy_only": r2_e,
                        "perm_null_mean": float(ne.mean()),
                        "perm_null_95pct": float(np.quantile(ne, 0.95))}
    say("   %s: energy-only LOO R2=%.4f (perm-null mean=%.3f, 95pct=%.3f)   "
        "vs DME-dict 2-feature LOO R2=%.4f"
        % (fam, r2_e, ne.mean(), np.quantile(ne, 0.95), fam_res[fam]["LOO_R2_2feature"]))

# ----------------------------------------------------------------------------
# (d) Alternative reading: order index = Legendre degree, per family
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(d) ALTERNATIVE READING: order index 0..8 IS the Legendre degree")
say("    I(theta) = sum_k F_family(k) * P_k(cos theta), one family at a time")
say("-" * 78)

Pk = np.stack([eval_legendre(k, CTH) for k in range(9)], 1)
readD = {}
for fam in FAMILIES:
    rows = []
    for E in CE_EV:
        i, y = mag_row(E)
        rec = Pk @ B[E][:, FCOL[fam]]
        rows.append({"E": E, "corr": float(np.corrcoef(rec, y)[0, 1]),
                     "affineR2": aff_r2(rec, y), "best_scale": best_scale(rec, y)})
    readD[fam] = rows
    say("  %s: corr  mean=%+.3f min=%+.3f max=%+.3f | affineR2 mean=%+.3f"
        % (fam, np.mean([r["corr"] for r in rows]), np.min([r["corr"] for r in rows]),
           np.max([r["corr"] for r in rows]), np.mean([r["affineR2"] for r in rows])))
    say("      per-energy affineR2: %s" % np.round([r["affineR2"] for r in rows], 3))

# 5-parameter combination (each family gets its own scale) = upper bound for (d)
say("\n  upper bound for (d): 5 independent scales (one per family), per energy:")
comb = []
for E in CE_EV:
    i, y = mag_row(E)
    X = np.stack([Pk @ B[E][:, FCOL[f]] for f in FAMILIES], 1)
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ c
    comb.append(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
say("      R2: %s   mean=%.4f" % (np.round(comb, 4), np.mean(comb)))

# combined single-Legendre-series reading (C): c_k = sum of families
say("\n  reading C: c_k = F00(k)+F20(k)+F21(k)+F22(k)+F11(k), I = sum_k c_k P_k:")
readC = []
for E in CE_EV:
    i, y = mag_row(E)
    cvec = sum(B[E][:, FCOL[f]] for f in FAMILIES)
    rec = Pk @ cvec
    readC.append(aff_r2(rec, y))
say("      affineR2: %s  mean=%.4f" % (np.round(readC, 4), np.mean(readC)))
say("  reading C2: c_k = F00(k)+F20(k) only:")
readC2 = []
for E in CE_EV:
    i, y = mag_row(E)
    cvec = B[E][:, FCOL["F00"]] + B[E][:, FCOL["F20"]]
    rec = Pk @ cvec
    readC2.append(aff_r2(rec, y))
say("      affineR2: %s  mean=%.4f" % (np.round(readC2, 4), np.mean(readC2)))

# degree-of-freedom-matched family-subset comparison (every model gets 2 params)
say("\n  family-subset comparison, each with a free scale+offset (2 params):")
subset_cmp = {}
for key, famset in [("F00 only", ["F00"]), ("F20 only", ["F20"]), ("F21 only", ["F21"]),
                    ("F22 only", ["F22"]), ("F11 only", ["F11"]),
                    ("F00+F20", ["F00", "F20"]), ("F00+F20+F21", ["F00", "F20", "F21"]),
                    ("all5", FAMILIES)]:
    rr = []
    for E in CE_EV:
        i, y = mag_row(E)
        cvec = sum(B[E][:, FCOL[f]] for f in famset)
        rr.append(aff_r2(Pk @ cvec, y))
    subset_cmp[key] = float(np.mean(rr))
    say("     %-14s mean affineR2=%.4f  min=%.4f  max=%.4f"
        % (key, np.mean(rr), np.min(rr), np.max(rr)))

# correlation of the F00+F20 sum with the dense panel's own Legendre coefficients
say("\n  corr(F00 column+F20 column, dense panel's own Legendre coeff vector) per energy:")
c2_corr = []
for E in CE_EV:
    i, y = mag_row(E)
    cm, _, _, _ = np.linalg.lstsq(Pk, y, rcond=None)
    c2 = B[E][:, FCOL["F00"]] + B[E][:, FCOL["F20"]]
    c2_corr.append(float(np.corrcoef(c2, cm)[0, 1]))
say("     %s  mean=%.4f" % (np.round(c2_corr, 3), np.mean(c2_corr)))

# ----------------------------------------------------------------------------
# (e) Robustness: only the 7 energies that lie exactly on the 0.1 eV grid
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(e) ROBUSTNESS: restrict to the energies exactly on the dense 0.1 eV grid")
say("-" * 78)
say("on-grid energies: %s" % ON_GRID)
say("off-grid energies: %s" % OFF_GRID)


def summarize_subset(subset, key):
    out = {}
    for tag, d in REC.items():
        A_r2 = np.mean([aff_r2(d["A"][E], mag_row(E)[1]) for E in subset])
        B_r2 = np.mean([aff_r2(d["B"][E], mag_row(E)[1]) for E in subset])
        out[tag] = {"readingA_mean_affineR2": float(A_r2), "readingB_mean_affineR2": float(B_r2)}
    return out


rob_full = summarize_subset(CE_EV, "all13")
rob_on = summarize_subset(ON_GRID, "on_grid7")
SAY_TAGS = sorted(rob_full.keys())
say("\nreading-B affine R2, all 13 energies vs the 7 on-grid energies only:")
for tag in SAY_TAGS:
    say("   %-16s all13=%.4f   on_grid7=%.4f"
        % (tag, rob_full[tag]["readingB_mean_affineR2"], rob_on[tag]["readingB_mean_affineR2"]))
say("\nreading-A affine R2, all 13 energies vs the 7 on-grid energies only:")
for tag in SAY_TAGS:
    say("   %-16s all13=%.4f   on_grid7=%.4f"
        % (tag, rob_full[tag]["readingA_mean_affineR2"], rob_on[tag]["readingA_mean_affineR2"]))

# (e) robustness for (c): LOO on the 7 on-grid energies only (ridge-free, 2 features)
say("\n(c) robustness on the 7 on-grid energies: greedy 1-2 feature LOO")
rob_c = {}
for fam in FAMILIES:
    Yall = np.array([[B[E][k, FCOL[fam]] for k in range(9)] for E in CE_EV])
    DICall = DIC
    keep = [CE_EV.index(E) for E in ON_GRID]
    Y = Yall[keep]
    Dg = DICall[keep]
    n = len(keep)
    loo = {}
    for nf in (1, 2):
        pred = np.zeros_like(Y)
        for k in range(n):
            tr = np.array([j for j in range(n) if j != k])
            Xtr, Ytr = Dg[tr], Y[tr]
            sel = []
            for _ in range(nf):
                b = None
                for j in range(105):
                    if j in sel:
                        continue
                    xs = np.column_stack([Xtr[:, s] for s in sel + [j]] + [np.ones(len(tr))])
                    cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
                    ess = float(((Ytr - xs @ cc) ** 2).sum())
                    if b is None or ess < b[0]:
                        b = (ess, j)
                sel.append(b[1])
            xs = np.column_stack([Xtr[:, s] for s in sel] + [np.ones(len(tr))])
            cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
            pred[k] = np.array([Dg[k, s] for s in sel] + [1.0]) @ cc
        loo[nf] = float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    # permutation null for the on-grid7 LOO R2 (2 features)
    null7 = np.zeros(200)
    for t in range(200):
        Yp = Y[rng.permutation(n)]
        pred = np.zeros_like(Yp)
        for k in range(n):
            tr = np.array([j for j in range(n) if j != k])
            Xtr, Ytr = Dg[tr], Yp[tr]
            sel = []
            for _ in range(2):
                b = None
                for j in range(105):
                    if j in sel:
                        continue
                    xs = np.column_stack([Xtr[:, s] for s in sel + [j]] + [np.ones(len(tr))])
                    cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
                    ess = float(((Ytr - xs @ cc) ** 2).sum())
                    if b is None or ess < b[0]:
                        b = (ess, j)
                sel.append(b[1])
            xs = np.column_stack([Xtr[:, s] for s in sel] + [np.ones(len(tr))])
            cc, _, _, _ = np.linalg.lstsq(xs, Ytr, rcond=None)
            pred[k] = np.array([Dg[k, s] for s in sel] + [1.0]) @ cc
        null7[t] = float(1 - ((Yp - pred) ** 2).sum() / ((Yp - Yp.mean(0)) ** 2).sum())
    loo["perm_null_mean_2feat"] = float(null7.mean())
    loo["perm_null_95pct_2feat"] = float(np.quantile(null7, 0.95))
    loo["perm_p_2feat"] = float((null7 >= loo[2]).mean())
    rob_c[fam] = loo
    say("   %s: on-grid7 LOO R2 1feat=%.4f 2feat=%.4f | perm-null mean=%.3f 95pct=%.3f p=%.3f"
        % (fam, loo[1], loo[2], loo["perm_null_mean_2feat"],
           loo["perm_null_95pct_2feat"], loo["perm_p_2feat"]))

# leave-one-energy-out on the reading-B conclusion (all 13)
say("\nLeave-one-out stability of the reading-B affine R2 (best convention %s):" % best_tag)
loo_r2 = []
for j, Eout in enumerate(CE_EV):
    sub = [E for E in CE_EV if E != Eout]
    m = np.mean([aff_r2(REC[best_tag]["B"][E], mag_row(E)[1]) for E in sub])
    loo_r2.append(m)
    say("   drop E=%6.2f -> mean affineR2 = %.4f" % (Eout, m))
say("   spread = %.4f" % (np.max(loo_r2) - np.min(loo_r2)))

# ----------------------------------------------------------------------------
# (f) Summary table
# ----------------------------------------------------------------------------
say("\n" + "-" * 78)
say("(f) WRITING SUMMARY TABLE -> %s" % CSV_PATH)
say("-" * 78)

best_B = REC[best_tag]["B"]
rows = []
for E in CE_EV:
    i, y = mag_row(E)
    b = B[E]
    recB = best_B[E]
    for fi, fam in enumerate(FAMILIES):
        rows.append({
            "energy_eV": E,
            "on_dense_grid": bool(E in ON_GRID),
            "family": fam,
            "order0_value": float(b[0, FCOL[fam]]),
            "order0_err": float(b[0, ECOL[fam]]),
            "sum_value_over_orders": float(b[:, FCOL[fam]].sum()),
            "abssum_value_over_orders": float(np.abs(b[:, FCOL[fam]]).sum()),
            "max_abs_value": float(np.abs(b[:, FCOL[fam]]).max()),
            "mean_err_over_orders": float(b[:, ECOL[fam]].mean()),
            "n_exact_zero_orders": int((b[:, FCOL[fam]] == 0).sum()),
            "legendre_series_corr_with_dense_mag": float(
                np.corrcoef(Pk @ b[:, FCOL[fam]], y)[0, 1]),
            "legendre_series_affineR2_with_dense_mag": aff_r2(Pk @ b[:, FCOL[fam]], y),
            "readingB_rec_affineR2": aff_r2(recB, y),
            "readingB_rec_best_scale": best_scale(recB, y),
        })

with open(CSV_PATH, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
say("wrote %d rows" % len(rows))

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# panel 1: F00 vs F20 order-0 across energy
a = ax[0, 0]
a.plot(CE_EV, [B[E][0, 1] for E in CE_EV], "o-", label="F00(order 0)")
a.plot(CE_EV, [B[E][0, 3] for E in CE_EV], "s-", label="F20(order 0)")
a.axhline(0, color="k", lw=0.6)
a.set_xlabel("photon energy (eV)")
a.set_ylabel("coefficient value")
a.set_title("(a) CoefLeg order-0 families vs energy")
a.legend(fontsize=8)

# panel 2: family magnitude vs energy
a = ax[0, 1]
for fam, key in [("F00", "F00_sum"), ("F20", "F20_sum"), ("F21", "F21_abssum"),
                 ("F22", "F22_abssum"), ("F11", "F11_abssum")]:
    a.plot(CE_EV, [fam_mag[E][key] for E in CE_EV], "o-", ms=3, label=fam)
a.set_xlabel("photon energy (eV)")
a.set_ylabel("sum over order rows")
a.set_yscale("log")
a.set_title("(a) family magnitude vs energy")
a.legend(fontsize=8)

# panel 3: reconstruction examples at 22.5 eV
a = ax[0, 2]
i, y = mag_row(CE_EV[0])
a.plot(THETA, y, "k-", lw=2, label="dense exp magnitude 22.5 eV")
a.plot(THETA, REC[best_tag]["B"][CE_EV[0]] * best_scale(REC[best_tag]["B"][CE_EV[0]], y),
       "--", label="reading B (%s)" % best_tag)
best_A_tag = max(readA_stats, key=lambda t: readA_stats[t]["mean_affineR2"])
a.plot(THETA, REC[best_A_tag]["A"][CE_EV[0]] *
       best_scale(REC[best_A_tag]["A"][CE_EV[0]], y), ":", label="reading A (%s)" % best_A_tag)
a.set_xlabel("emission angle (deg)")
a.set_ylabel("magnitude (scaled)")
a.set_title("(b) reconstruction vs dense panel, 22.5 eV")
a.legend(fontsize=7)

# panel 4: affine R2 per energy reading A vs B
a = ax[1, 0]
a.plot(CE_EV, readA_stats[best_tag]["per_energy_affineR2"], "s-", label="reading A")
a.plot(CE_EV, readB_stats[best_tag]["per_energy_affineR2"], "o-", label="reading B")
for tag, mk in [("CS1_norm1_sq21", "^"), ("CS0_norm0_sq20", "v")]:
    a.plot(CE_EV, readB_stats[tag]["per_energy_affineR2"], mk + "--", ms=4,
           label="B %s" % tag, alpha=0.7)
a.plot(CE_EV, base_r2, "k:", lw=2, label="dense panel own 9-term fit (ceiling)")
a.axhline(0, color="k", lw=0.5)
a.set_xlabel("photon energy (eV)")
a.set_ylabel("affine R$^2$")
a.set_title("(b) reconstruction quality per energy")
a.legend(fontsize=7)

# panel 5: (d) per-family Legendre-series affine R2
a = ax[1, 1]
for fam in FAMILIES:
    a.plot(CE_EV, [r["affineR2"] for r in readD[fam]], "o-", ms=3, label=fam)
a.plot(CE_EV, comb, "k^-", ms=4, label="5 free scales (upper bound)")
a.axhline(0, color="k", lw=0.5)
a.set_xlabel("photon energy (eV)")
a.set_ylabel("affine R$^2$")
a.set_title("(d) family-as-Legendre-series vs dense panel")
a.legend(fontsize=7)

# panel 6: (c) predictability
a = ax[1, 2]
xs = np.arange(len(FAMILIES))
a.bar(xs - 0.2, [fam_res[f]["LOO_R2_1feature"] for f in FAMILIES], 0.4,
      label="LOO R$^2$, 1 feature", color="tab:blue")
a.bar(xs + 0.2, [fam_res[f]["LOO_R2_2feature"] for f in FAMILIES], 0.4,
      label="LOO R$^2$, 2 features", color="tab:orange")
a.plot(xs, [fam_res[f]["LOO_R2_2feature_perm_null_95pct"] for f in FAMILIES], "k_", ms=14,
       label="permutation null 95th pct")
a.plot(xs, [fam_res[f]["in_sample_R2_full_dict"] for f in FAMILIES], "r*", ms=12,
       label="in-sample R$^2$, 105 features (saturated)")
a.axhline(0, color="k", lw=0.6)
a.set_xticks(xs)
a.set_xticklabels(FAMILIES)
a.set_ylabel("R$^2$ (9 orders pooled)")
a.set_title("(c) CoefLeg vs DME-product dictionary")
a.legend(fontsize=7)

fig.suptitle("p5_coefleg: CoefLeg blocks, dense-panel reconstruction, DME predictability", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(PNG_PATH, dpi=130)
say("\nwrote figure %s" % PNG_PATH)

# ----------------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------------
out = {
    "package": "p5_coefleg",
    "script": os.path.abspath(__file__),
    "data_sha256": SHA,
    "coefleg_energies_eV": CE_EV,
    "energies_on_dense_0p1eV_grid": ON_GRID,
    "energies_off_dense_grid": OFF_GRID,
    "structure": {
        "shape_per_block": [9, 11],
        "columns": ["order", "F00", "errF00", "F20", "errF20", "F21", "errF21",
                    "F22", "errF22", "F11", "errF11"],
        "exact_zero_entry_orders": {str(k): v for k, v in zero_every.items()},
        "F00_order0_values": f00_0,
        "errF20_over_errF00_mean": float(np.mean([
            (B[E][:, ECOL["F20"]] / B[E][:, ECOL["F00"]]).mean() for E in CE_EV])),
        "mean_err_per_family_per_order": err_prof,
        "family_magnitude_by_energy": fam_mag,
        "order0_SN_by_energy": {
            E: {f: float(abs(B[E][0, FCOL[f]]) / B[E][0, ECOL[f]]) for f in FAMILIES
                if B[E][0, ECOL[f]] > 0} for E in CE_EV},
    },
    "reconstruction_b": {
        "description": "I(theta)=sum_k [F00_k Y_k0 + F20_k Y_k0 + F21_k ReY_k1 + F22_k ReY_k2 + F11_k ReY_k1]",
        "conventions": readB_stats,
        "best_convention": best_tag,
    },
    "reconstruction_a": {
        "description": "I(theta)=sum_k [F00_k Y00 + F20_k ReY20 + F21_k ReY21 + F22_k ReY22 + F11_k ReY11]",
        "conventions": readA_stats,
    },
    "reconstruction_b_vs_mag_and_mag2": mag2_cmp,
    "baselines": {
        "dense_panel_own_9term_legendre_fit_mean_R2": float(np.mean(base_r2)),
        "dense_panel_own_9term_legendre_fit_mean_relRMS": float(np.mean(base_rel)),
        "shuffled_panel_affineR2_mean": float(sh.mean()),
        "shuffled_panel_affineR2_95pct": float(np.quantile(sh, 0.95)),
        "shuffled_panel_affineR2_true_22p5eV": aff_r2(REC[best_tag]["B"][CE_EV[0]], mag_row(CE_EV[0])[1]),
        "constant_predictor_R2_for_families": 0.0,
    },
    "row0_beta_comparison": {
        "per_energy": row0,
        "corr_F20overF00_with_dense_beta": r_beta,
        "beta_dense_1plusP2": beta_dense,
    },
    "legendre_order_reading_d": {
        f: {"mean_corr": float(np.mean([r["corr"] for r in readD[f]])),
            "mean_affineR2": float(np.mean([r["affineR2"] for r in readD[f]])),
            "per_energy_affineR2": [r["affineR2"] for r in readD[f]]}
        for f in FAMILIES},
    "reading_d_5scale_upper_bound": {"per_energy_R2": comb, "mean": float(np.mean(comb))},
    "reading_C_sum_of_families": {"per_energy_affineR2": readC, "mean": float(np.mean(readC))},
    "reading_C2_F00_plus_F20": {"per_energy_affineR2": readC2, "mean": float(np.mean(readC2)),
                                "corr_with_dense_own_legendre_coeffs": c2_corr,
                                "corr_with_dense_own_legendre_coeffs_mean": float(np.mean(c2_corr))},
    "reading_d_family_subset_comparison_2params": subset_cmp,
    "dme_predictability_c": {
        "dictionary_size": int(DIC.shape[1]),
        "dictionary_definition": "all products z_i*z_j (i<=j) of the 14 DME Re/Im entries",
        "per_family": fam_res,
        "competing_energy_only_baseline": energy_base,
    },
    "robustness_e": {
        "all13_readingA_mean_affineR2": {t: rob_full[t]["readingA_mean_affineR2"] for t in rob_full},
        "all13_readingB_mean_affineR2": {t: rob_full[t]["readingB_mean_affineR2"] for t in rob_full},
        "ongrid7_readingA_mean_affineR2": {t: rob_on[t]["readingA_mean_affineR2"] for t in rob_on},
        "ongrid7_readingB_mean_affineR2": {t: rob_on[t]["readingB_mean_affineR2"] for t in rob_on},
        "ongrid7_dme_LOO_R2": rob_c,
        "leave_one_energy_out_readingB_affineR2": {"per_drop": loo_r2,
                                                   "spread": float(np.max(loo_r2) - np.min(loo_r2))},
    },
    "artifacts": {"script": os.path.abspath(__file__), "json": JSON_PATH, "log": LOG_PATH,
                  "figure": PNG_PATH, "summary_csv": CSV_PATH},
    "failures": FAILURES,
    "runtime_s": None,
}
out["runtime_s"] = time.time() - T0
with open(JSON_PATH, "w") as fh:
    json.dump(out, fh, indent=2, default=float)
say("wrote %s" % JSON_PATH)
say("runtime %.1f s" % out["runtime_s"])
say("=" * 78)
sys.stdout.flush()