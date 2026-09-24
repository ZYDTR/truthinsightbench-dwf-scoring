#!/usr/bin/env python3
"""
p6_exptheory: direct comparison of the experiment-like and theory-like
angle-resolved panels (magnitude, phase, delay) of the photoionization
dataset.

Inputs (read-only):  <ROOT>/data/*.tsv, <ROOT>/data/coordinate_definitions.json
Outputs:             <ROOT>/output/evidence/p6_exptheory.json
                     <ROOT>/output/evidence/p6_comparison_table.csv
                     <ROOT>/output/evidence/p6_exptheory.png
(stdout is captured by the harness into p6_exptheory.log)

Re-runnable from scratch; only data/ and its own outputs are touched.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numpy.polynomial import legendre as leg  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
OUTDIR = os.path.join(ROOT, "output", "evidence")
os.makedirs(OUTDIR, exist_ok=True)

F_ANG = os.path.join(DATA, "angle_energy_observations_datasets.tsv")
F_COEF = os.path.join(DATA, "angular_coefficients_datasets.tsv")
F_DEF = os.path.join(DATA, "coordinate_definitions.json")

JSON_PATH = os.path.join(OUTDIR, "p6_exptheory.json")
CSV_PATH = os.path.join(OUTDIR, "p6_comparison_table.csv")
PNG_PATH = os.path.join(OUTDIR, "p6_exptheory.png")

HBAR_AS_EV = 658.2119569          # from data/coordinate_definitions.json
MATCH_TOL_MEV = 5.0               # "within a few meV"
ANGLES = np.arange(0, 181, dtype=float)
TH_RAD = np.deg2rad(ANGLES)
NPTS_1DEG = 181
LMAX = 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel(df, name):
    """Long-format group -> 2-D array [index_0, index_1]."""
    sub = df[df["dataset_path"] == name]
    return (sub.pivot_table(index="index_0", columns="index_1", values="value")
            .to_numpy(dtype=float))


def load_energy(df, name):
    """Long-format 1-D energy group -> ascending array."""
    sub = df[df["dataset_path"] == name].sort_values("index_0")
    return sub["value"].to_numpy(dtype=float)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def circ_mean(x):
    return np.arctan2(np.sin(x).mean(), np.cos(x).mean())


def complex_legendre(F, lmax=LMAX):
    """Complex Legendre coefficients c_L = (2L+1)/2 int F P_L(cos th) sin th dth."""
    out = {}
    for l in range(lmax + 1):
        P = leg.legval(np.cos(TH_RAD), [0] * l + [1])
        out[l] = (2 * l + 1) / 2.0 * np.trapezoid(F * P * np.sin(TH_RAD), TH_RAD)
    return out


def legendre_reconstruct(coef, lmax=LMAX):
    rec = np.zeros_like(TH_RAD, dtype=complex)
    for l in range(lmax + 1):
        rec += coef[l] * leg.legval(np.cos(TH_RAD), [0] * l + [1])
    return rec


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def shift_null(exp_curve, th_curve, n_shift=180):
    """Max |Pearson r| over circular angular shifts of the theory curve (null)."""
    vals = []
    for k in range(1, n_shift + 1):
        r = pearson(exp_curve, np.roll(th_curve, k))
        if np.isfinite(r):
            vals.append(abs(r))
    vals = np.asarray(vals)
    return float(vals.max()), float(np.percentile(vals, 95))


# --------------------------------------------------------------------------
# 0. provenance
# --------------------------------------------------------------------------
print("=" * 78)
print("p6_exptheory : experiment-like vs theory-like panel comparison")
print("=" * 78)
hashes = {}
for p in (F_ANG, F_COEF, F_DEF):
    h = sha256(p)
    hashes[os.path.relpath(p, ROOT)] = h
    print(f"sha256 {os.path.relpath(p, ROOT):45s} {h}")

with open(F_DEF) as fh:
    coord = json.load(fh)
meas_energies = coord["experimental_grid_provenance"]["measured_energy_values_eV"]
print(f"hbar_as_eV={HBAR_AS_EV}  measured DME energies: {meas_energies}")

# --------------------------------------------------------------------------
# 1. load panels
# --------------------------------------------------------------------------
df = pd.read_csv(F_ANG, sep="\t")
EE = load_energy(df, "experiment/photon_energy_eV")
EED = load_energy(df, "experiment/photon_energy_eV_del")
TE = load_energy(df, "theory/photon_energy_eV")
TED = load_energy(df, "theory/photon_energy_eV_del")
EMAG = load_panel(df, "experiment/magnitude")
EPHA = load_panel(df, "experiment/phase")
EDEL = load_panel(df, "experiment/delay")
TMAG = load_panel(df, "theory/magnitude")
TPHA = load_panel(df, "theory/phase")
TDEL = load_panel(df, "theory/delay")

assert EMAG.shape == (276, NPTS_1DEG) and TMAG.shape == (99, NPTS_1DEG)
assert EDEL.shape == (275, NPTS_1DEG) and TDEL.shape == (98, NPTS_1DEG)

# unwrapped (angle-continuous) phase panels
EPHA_U = np.unwrap(EPHA, axis=1)
TPHA_U = np.unwrap(TPHA, axis=1)

# cross-check: provided delay panels == forward-difference derivative of phase
d_exp = np.diff(EPHA, axis=0) * HBAR_AS_EV / np.diff(EE)[:, None]
d_th = np.diff(TPHA, axis=0) * HBAR_AS_EV / np.diff(TE)[:, None]
r_delcheck_exp = pearson(EDEL.ravel(), d_exp.ravel())
r_delcheck_th = pearson(TDEL.ravel(), d_th.ravel())
print("\n[consistency] delay vs d(phase)/dE * hbar  (Pearson over all cells)")
print(f"  experiment r={r_delcheck_exp:.6f} rms={np.sqrt(np.mean((EDEL - d_exp) ** 2)):.3f} as")
print(f"  theory     r={r_delcheck_th:.6f} rms={np.sqrt(np.mean((TDEL - d_th) ** 2)):.3f} as")
print(f"  experiment delay energy array == EE[1:]: max|diff|={np.abs(EED - EE[1:]).max():.2e} eV")
print(f"  theory     delay energy array ~ TE[1:]: max|diff|={np.abs(TED - TE[1:]).max():.2e} eV")

# phase-panel wrap statistics (stored phases are wrapped into (-pi, pi])
wrap_exp = int(np.sum(np.abs(np.diff(EPHA, axis=1)) > np.pi))
wrap_th = int(np.sum(np.abs(np.diff(TPHA, axis=1)) > np.pi))
print(f"\n[phase storage] per-angle 2*pi discontinuities: experiment {wrap_exp}, theory {wrap_th}")
print(f"[phase storage] post-unwrap max |angular jump|: experiment "
      f"{np.abs(np.diff(EPHA_U, axis=1)).max():.3f} rad, theory "
      f"{np.abs(np.diff(TPHA_U, axis=1)).max():.3f} rad")

# rows where the phase step is so close to pi that the unwrap direction is ambiguous
amb_th_rows = [int(i) for i in np.where(np.abs(np.diff(TPHA_U, axis=1)).max(axis=1) > 1.0)[0]]
amb_exp_rows = [int(i) for i in np.where(np.abs(np.diff(EPHA_U, axis=1)).max(axis=1) > 1.0)[0]]
print(f"[phase storage] theory rows with post-unwrap |jump|>1 rad: {amb_th_rows} "
      f"(energies {np.round(TE[amb_th_rows], 3).tolist()})")
print(f"[phase storage] experiment rows with post-unwrap |jump|>1 rad: {len(amb_exp_rows)} rows "
      f"(out of {EPHA.shape[0]})")

# --------------------------------------------------------------------------
# 2. (a) matched energy pairs -- theory exact grid points only
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(a) matched energy pairs (exact theory grid points, tol=%.1f meV)" % MATCH_TOL_MEV)
print("=" * 78)
pairs = []
unmatched = []
print(f"{'E_meas':>7} {'theory(E)':>10} {'dE [meV]':>10} {'idx_th':>6} "
      f"{'exp node':>9} {'exp node off [meV]':>18} {'theory delay E':>14}")
for m in meas_energies:
    j = int(np.argmin(np.abs(TE - m)))
    d_mev = (TE[j] - m) * 1e3
    k = int(np.argmin(np.abs(EE - m)))
    off = (EE[k] - m) * 1e3
    jd = int(np.argmin(np.abs(TED - m)))
    if abs(d_mev) <= MATCH_TOL_MEV:
        pairs.append(dict(e_meas=float(m), e_theory=float(TE[j]), dE_meV=float(d_mev),
                          idx_th=j, e_exp_node=float(EE[k]), exp_node_off_meV=float(off),
                          idx_exp_node=k, e_theory_del=float(TED[jd]), idx_th_del=jd))
        print(f"{m:7.2f} {TE[j]:10.4f} {d_mev:10.4f} {j:6d} {EE[k]:9.2f} {off:18.2f} {TED[jd]:14.4f}")
    else:
        unmatched.append(dict(e_meas=float(m), e_theory_nearest=float(TE[j]),
                              dE_meV=float(d_mev), e_theory_del_nearest=float(TED[jd])))
n5 = len(pairs)
print(f"\n-> {n5} of the 13 measured energies have a theory grid point within "
      f"{MATCH_TOL_MEV:.0f} meV ({n5} matched pairs).")
for u in unmatched:
    print(f"-> UNMATCHED {u['e_meas']:.2f} eV : nearest theory phase-grid point "
          f"{u['e_theory_nearest']:.4f} eV ({u['dE_meV']:+.2f} meV)")
print("   claim in the brief was 12/13 matched (all except 48.4 eV); observed 11/13 "
      "(22.5 eV is also unmatched: nearest theory point 749 meV away).")
diffs = np.array([p["dE_meV"] for p in pairs])
print(f"   matched dE: min={diffs.min():.4f} meV max={diffs.max():.4f} meV "
      f"median={np.median(diffs):.4f} meV ; all identical to -1.000 meV within 1e-9? "
      f"{bool(np.all(np.abs(diffs + 1.0) < 1e-9))}")
ecov = (TE.min(), TE.max())
print(f"   theory panel energy coverage: {ecov[0]:.3f}-{ecov[1]:.3f} eV; "
      f"experiment panel: {EE.min():.1f}-{EE.max():.1f} eV "
      f"(theory covers only {TE.max() - TE.min():.2f} eV of the 27.5 eV experiment span)")

# --------------------------------------------------------------------------
# 3. (b,c,e) per-energy comparisons
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(b,c,e) per-energy angle-resolved comparison")
print("=" * 78)


def interp_row(panel, e_grid, e_target):
    """Linear interpolation of a panel row in energy (clipped at the edges)."""
    i = int(np.searchsorted(e_grid, e_target))
    if i == 0:
        return panel[0].copy()
    if i >= len(e_grid):
        return panel[-1].copy()
    w = (e_target - e_grid[i - 1]) / (e_grid[i] - e_grid[i - 1])
    return (1 - w) * panel[i - 1] + w * panel[i]


rows = []
for pr in pairs:
    m = pr["e_meas"]
    e_th = pr["e_theory"]
    j = pr["idx_th"]

    # experiment rows interpolated to the exact theory energies (declared)
    e_mag = interp_row(EMAG, EE, e_th)
    e_pha_raw = interp_row(EPHA, EE, e_th)
    e_pha = interp_row(EPHA_U, EE, e_th)              # unwrapped panel, interpolated
    t_mag = TMAG[j]
    t_pha = TPHA_U[j]

    jd = pr["idx_th_del"]
    e_del_t = TED[jd]
    e_del = interp_row(EDEL, EED, e_del_t)
    t_del = TDEL[jd]

    # --- magnitude ---
    r_mag = pearson(e_mag, t_mag)
    rms_mag = float(np.sqrt(np.mean((e_mag - t_mag) ** 2)))
    nrms_mag = rms_mag / float(e_mag.std())
    nrms_mag_th = rms_mag / float(t_mag.std())

    # --- phase ---
    r_pha_u = pearson(e_pha, t_pha)
    r_pha_raw = pearson(e_pha_raw, TPHA[j])
    rms_pha_rawdiff = float(np.sqrt(np.mean(wrap(e_pha - t_pha) ** 2)))  # mod 2pi "before"
    delta = e_pha - t_pha
    # best single constant offset (mod 2pi): grid search, verified against circ mean
    cs = np.linspace(-np.pi, np.pi, 20001)
    cost = np.array([np.mean(wrap(delta - c) ** 2) for c in cs])
    c_star = float(cs[int(np.argmin(cost))])
    c_circ = float(circ_mean(delta))
    rms_pha_before = float(np.sqrt(np.mean(wrap(delta) ** 2)))
    rms_pha_after = float(np.sqrt(cost.min()))
    nrms_pha_after = rms_pha_after / float(e_pha.std())
    delta_minus = wrap(delta - c_star)
    # is the phase residual global or confined to angular sectors?
    frac_big = float(np.mean(np.abs(delta_minus) > 1.0))
    big_angles = np.where(np.abs(delta_minus) > 1.0)[0]
    # branch test: a pi-flip of a contiguous half-range would be a wrap artifact;
    # a genuine mismatch leaves a localized compact sector
    if big_angles.size:
        span = int(big_angles.max() - big_angles.min() + 1)
        compact = bool(span <= big_angles.size + 5)
    else:
        span = 0
        compact = True

    # --- delay ---
    r_del = pearson(e_del, t_del)
    rms_del = float(np.sqrt(np.mean((e_del - t_del) ** 2)))
    med_del = float(np.median(np.abs(e_del - t_del)))
    nrm_del = rms_del / float(e_del.std())
    ddiff = e_del - t_del
    ang_max = int(np.argmax(np.abs(ddiff)))

    # --- permutation/shift nulls ---
    null_mag = shift_null(e_mag, t_mag)
    null_pha = shift_null(e_pha, t_pha)
    null_del = shift_null(e_del, t_del)
    # second, more decisive null: keep the experiment row fixed but substitute a
    # MISMATCHED theory energy row (all other 98 theory rows); a genuine match
    # must beat this "energy-scramble" baseline.
    def scramble_null(expc, panel, exclude):
        vals = []
        for jj in range(panel.shape[0]):
            if jj == exclude:
                continue
            v = pearson(expc, panel[jj])
            if np.isfinite(v):
                vals.append(v)
        vals = np.asarray(vals)
        return float(vals.max()), float(np.percentile(np.abs(vals), 95))
    scr_mag = scramble_null(e_mag, TMAG, j)
    scr_del = scramble_null(e_del, TDEL, jd)
    scr_pha = scramble_null(e_pha, TPHA_U, j)

    # RMS-based energy-scramble null (the decisive baseline: is the METCHED theory
    # energy row a better match than unrelated theory energies under the same metric?)
    def scramble_rms(expc, panel, exclude, metric):
        vals = []
        for jj in range(panel.shape[0]):
            if jj == exclude:
                continue
            vals.append(metric(expc, panel[jj]))
        vals = np.asarray(vals)
        return float(np.percentile(vals, 5)), float(vals.mean()), int(np.sum(vals <= 0) * 0)

    def mag_nrms(ec, tc):
        return np.sqrt(np.mean((ec - tc) ** 2)) / ec.std()

    def pha_resid(ec, tc):
        d = wrap(ec - tc)
        c = circ_mean(d)
        return np.sqrt(np.mean(wrap(d - c) ** 2))

    def del_rms(ec, tc):
        return np.sqrt(np.mean((ec - tc) ** 2))

    m_p5, m_mean, _ = scramble_rms(e_mag, TMAG, j, mag_nrms)
    p_p5, p_mean, _ = scramble_rms(e_pha, TPHA_U, j, pha_resid)
    d_p5, d_mean, _ = scramble_rms(e_del, TDEL, jd, del_rms)

    row = dict(
        e_meas=m, e_theory=e_th, dE_meV=pr["dE_meV"], idx_th=j, idx_th_del=jd,
        e_exp_node=pr["e_exp_node"], exp_node_off_meV=pr["exp_node_off_meV"],
        e_delay_theory=e_del_t,
        mag_r=r_mag, mag_rms=rms_mag, mag_nrms_exp=nrms_mag, mag_nrms_theory=nrms_mag_th,
        mag_null_maxabsr=null_mag[0],
        mag_scramble_null_max_r=scr_mag[0], mag_scramble_null_p95abs=scr_mag[1],
        mag_beats_scramble_null=bool(r_mag > scr_mag[0]),
        mag_scramble_rms_null_p5=m_p5,
        mag_better_than_scramble_rms_null_p5=bool(nrms_mag < m_p5),
        phase_r_unwrapped=r_pha_u, phase_r_stored=r_pha_raw,
        phase_rms_before_rad=rms_pha_before, phase_rms_after_rad=rms_pha_after,
        phase_nrms_after=nrms_pha_after,
        phase_frac_angles_resid_gt1rad=frac_big,
        phase_resid_big_angle_min=(int(big_angles.min()) if big_angles.size else -1),
        phase_resid_big_angle_max=(int(big_angles.max()) if big_angles.size else -1),
        phase_resid_sector_is_compact=compact,
        phase_offset_rad=c_star, phase_offset_deg=float(np.rad2deg(c_star)),
        phase_offset_circmean_rad=c_circ,
        phase_offset_gridsearch_agrees=bool(abs(wrap(c_star - c_circ)) < 1e-3),
        phase_null_maxabsr=null_pha[0],
        phase_scramble_null_max_r=scr_pha[0], phase_scramble_null_p95abs=scr_pha[1],
        phase_beats_scramble_null=bool(r_pha_u > scr_pha[0]),
        phase_scramble_rms_null_p5=p_p5,
        phase_better_than_scramble_rms_null_p5=bool(rms_pha_after < p_p5),
        delay_scramble_null_max_r=scr_del[0], delay_scramble_null_p95abs=scr_del[1],
        delay_beats_scramble_null=bool(r_del > scr_del[0]),
        delay_scramble_rms_null_p5=d_p5,
        delay_better_than_scramble_rms_null_p5=bool(rms_del < d_p5),
        delay_r=r_del, delay_rms_as=rms_del, delay_medabs_as=med_del,
        delay_nrms=nrm_del, delay_argmax_deg=ang_max, delay_null_maxabsr=null_del[0],
        exp_phase_std_rad=float(e_pha.std()), th_phase_std_rad=float(t_pha.std()),
        exp_delay_std_as=float(e_del.std()), th_delay_std_as=float(t_del.std()),
    )
    rows.append(row)

hdr = (f"{'E[eV]':>7} | {'mag_r':>6} {'magRMS':>7} {'magNRMS':>8} | "
       f"{'pha_r':>6} {'phaR':>7} {'bef':>6} {'aft':>6} {'off_deg':>8} | "
       f"{'del_r':>6} {'delRMS':>8} {'delMed':>7} {'argmax':>6}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['e_meas']:7.2f} | {r['mag_r']:6.3f} {r['mag_rms']:7.4f} {r['mag_nrms_exp']:8.3f} | "
          f"{r['phase_r_unwrapped']:6.3f} {r['phase_rms_before_rad']:7.3f} "
          f"{r['phase_rms_before_rad']:6.3f} {r['phase_rms_after_rad']:6.3f} "
          f"{r['phase_offset_deg']:8.2f} | "
          f"{r['delay_r']:6.3f} {r['delay_rms_as']:8.1f} {r['delay_medabs_as']:7.1f} "
          f"{r['delay_argmax_deg']:6d}")
print("\nnull baselines (is the matched theory energy special?)")
print(f"{'E[eV]':>7} | {'mag_r':>6} {'scr_null':>8} {'beats':>5} | {'pha_r':>6} {'scr_null':>8} {'beats':>5} | "
      f"{'del_r':>6} {'scr_null':>8} {'beats':>5}")
for r in rows:
    print(f"{r['e_meas']:7.2f} | {r['mag_r']:6.3f} {r['mag_scramble_null_max_r']:8.3f} "
          f"{str(r['mag_beats_scramble_null'])[:5]:>5} | {r['phase_r_unwrapped']:6.3f} "
          f"{r['phase_scramble_null_max_r']:8.3f} {str(r['phase_beats_scramble_null'])[:5]:>5} | "
          f"{r['delay_r']:6.3f} {r['delay_scramble_null_max_r']:8.3f} "
          f"{str(r['delay_beats_scramble_null'])[:5]:>5}")

print("\nRMS-based energy-scramble null (matched theory row vs 5th percentile of all other "
      "theory rows, same metric)")
print(f"{'E[eV]':>7} | {'magNRMS':>8} {'null_p5':>8} {'better':>7} | {'phaResid':>8} {'null_p5':>8} "
      f"{'better':>7} | {'delRMS':>8} {'null_p5':>8} {'better':>7}")
for r in rows:
    print(f"{r['e_meas']:7.2f} | {r['mag_nrms_exp']:8.3f} {r['mag_scramble_rms_null_p5']:8.3f} "
          f"{str(r['mag_better_than_scramble_rms_null_p5'])[:6]:>7} | "
          f"{r['phase_rms_after_rad']:8.3f} {r['phase_scramble_rms_null_p5']:8.3f} "
          f"{str(r['phase_better_than_scramble_rms_null_p5'])[:6]:>7} | "
          f"{r['delay_rms_as']:8.1f} {r['delay_scramble_rms_null_p5']:8.1f} "
          f"{str(r['delay_better_than_scramble_rms_null_p5'])[:6]:>7}")

print("\n'phaR' column is redundant with 'bef' (mod-2pi RMS of the phase difference, rad).")
print("every phase panel is stored wrapped; 'phase_r_stored' (Pearson on stored values) is "
      "reported in the JSON/CSV as a documented artifact, not as agreement.")

# --------------------------------------------------------------------------
# 4. (d) angular expansion: complex Legendre components
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(d) complex Legendre decomposition  F(theta) = M(theta) exp(i*phase_unwrapped)")
print("=" * 78)
for r in rows:
    j = int(np.argmin(np.abs(TE - r["e_theory"])))
    e_pha = interp_row(EPHA_U, EE, r["e_theory"])
    e_mag = interp_row(EMAG, EE, r["e_theory"])
    Fexp = e_mag * np.exp(1j * e_pha)
    Fth = TMAG[j] * np.exp(1j * TPHA_U[j])
    ce = complex_legendre(Fexp)
    ct = complex_legendre(Fth)
    rese = np.sqrt(np.mean(np.abs(Fexp - legendre_reconstruct(ce)) ** 2)) / np.sqrt(np.mean(np.abs(Fexp) ** 2))
    rest = np.sqrt(np.mean(np.abs(Fth - legendre_reconstruct(ct)) ** 2)) / np.sqrt(np.mean(np.abs(Fth) ** 2))
    r["legendre_rel_resid_exp"] = float(rese)
    r["legendre_rel_resid_theory"] = float(rest)
    for l in range(LMAX + 1):
        r[f"c{l}_exp_abs"] = float(np.abs(ce[l]))
        r[f"c{l}_th_abs"] = float(np.abs(ct[l]))
        r[f"c{l}_amp_ratio"] = float(np.abs(ce[l]) / np.abs(ct[l])) if abs(ct[l]) > 0 else float("nan")
        r[f"c{l}_phase_diff_deg"] = float(np.rad2deg(wrap(np.angle(ce[l]) - np.angle(ct[l]))))
    r["c0_phase_diff_rel_deg"] = 0.0
    for l in range(1, LMAX + 1):
        r[f"c{l}_phase_diff_rel_deg"] = float(
            np.rad2deg(wrap((np.angle(ce[l]) - np.angle(ct[l])) - (np.angle(ce[0]) - np.angle(ct[0])))))
    r["sum_abs_c_exp"] = float(sum(abs(ce[l]) for l in range(LMAX + 1)))
    r["frac_L0_exp"] = float(abs(ce[0]) / sum(abs(ce[l]) for l in range(LMAX + 1)))
    r["frac_L0_th"] = float(abs(ct[0]) / sum(abs(ct[l]) for l in range(LMAX + 1)))

print(f"{'E[eV]':>7} {'relres_e':>9} {'relres_t':>9} | {'|c0|_e':>7} {'|c0|_t':>7} {'ratio0':>7} "
      f"{'dphi0[deg]':>10} | {'|c2|_e':>7} {'|c2|_t':>7} {'ratio2':>7} {'dphi2':>8} {'dphi2rel':>9}")
print("-" * 120)
for r in rows:
    print(f"{r['e_meas']:7.2f} {r['legendre_rel_resid_exp']:9.5f} {r['legendre_rel_resid_theory']:9.5f} | "
          f"{r['c0_exp_abs']:7.4f} {r['c0_th_abs']:7.4f} {r['c0_amp_ratio']:7.3f} "
          f"{r['c0_phase_diff_deg']:10.2f} | "
          f"{r['c2_exp_abs']:7.4f} {r['c2_th_abs']:7.4f} {r['c2_amp_ratio']:7.3f} "
          f"{r['c2_phase_diff_deg']:8.2f} {r['c2_phase_diff_rel_deg']:9.2f}")
print("relative residual = ||F - sum_{L<=8} c_L P_L|| / ||F|| (trapezoid quadrature)")
print("dphiL = arg(c_L^exp) - arg(c_L^th); dphiL_rel = dphiL - dphi0 (isotropic phase aligned)")

# --------------------------------------------------------------------------
# 5. (e) where along the angle do delays differ most
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(e) angular location of the delay differences")
print("=" * 78)
prof = np.zeros(NPTS_1DEG)
for r in rows:
    jd = int(np.argmin(np.abs(TED - r["e_meas"])))
    d = interp_row(EDEL, EED, TED[jd]) - TDEL[jd]
    prof += np.abs(d)
prof /= len(rows)
wins = [(a, float(prof[a:a + 30].mean())) for a in range(0, 151, 30)]
print("mean |tau_exp - tau_th| [as] averaged over the 11 matched energies")
print("  coarse windows:", ", ".join(f"{a}-{a+30}deg: {v:.1f}" for a, v in wins))
best_win = max(wins, key=lambda t: t[1])
print(f"  largest window: {best_win[0]}-{best_win[0]+30} deg, mean {best_win[1]:.1f} as")
print("  fine profile every 15 deg:", np.round(prof[::15], 1).tolist())
argmax_angles = [r["delay_argmax_deg"] for r in rows]
print(f"  per-energy argmax|dtau| angles: {argmax_angles}")
print(f"  forward (0-40 deg) mean|dtau| {prof[0:41].mean():.1f} as ; "
      f"mid (41-140) {prof[41:141].mean():.1f} as ; backward (141-180) {prof[141:].mean():.1f} as")

# --------------------------------------------------------------------------
# 6. (g) robustness variants
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(g) robustness")
print("=" * 78)
ANG10 = np.arange(0, 181, 10)
i10 = ANG10.astype(int)

for r in rows:
    m = r["e_meas"]
    j = r["idx_th"]
    # nearest-node experiment rows instead of interpolation
    k = int(np.argmin(np.abs(EE - TE[j])))
    e_mag_nn = EMAG[k]
    e_pha_nn = EPHA_U[k]
    t_mag = TMAG[j]
    t_pha = TPHA_U[j]
    r["mag_r_nn"] = pearson(e_mag_nn, t_mag)
    r["mag_nrms_nn"] = float(np.sqrt(np.mean((e_mag_nn - t_mag) ** 2))) / float(e_mag_nn.std())
    d_nn = e_pha_nn - t_pha
    r["phase_rms_after_nn"] = float(np.sqrt(np.mean(wrap(d_nn - circ_mean(d_nn)) ** 2)))
    jd = int(np.argmin(np.abs(TED - m)))
    kd = int(np.argmin(np.abs(EED - TED[jd])))
    d_del_nn = EDEL[kd] - TDEL[jd]
    r["delay_r_nn"] = pearson(EDEL[kd], TDEL[jd])
    r["delay_rms_nn"] = float(np.sqrt(np.mean(d_del_nn ** 2)))

    # (i) after removing the best-fit per-energy constant phase offset from the
    #     theory phase panel; delays rebuilt from the corrected phase panel
    r["phase_rms_after_offset_rad"] = r["phase_rms_after_rad"]
    r["mag_r_after_offset"] = r["mag_r"]          # magnitude untouched
    # offset curve over the theory phase grid, linearly interpolated between
    # matched energies (constant extension outside the matched range)
    e_match = np.array([q["e_theory"] for q in rows])
    off_match = np.array([q["phase_offset_rad"] for q in rows])
    order = np.argsort(e_match)
    off_curve = np.interp(TE, e_match[order], off_match[order])
    TPHA_C = TPHA_U - off_curve[:, None]
    d_th_c = np.diff(TPHA_C, axis=0) * HBAR_AS_EV / np.diff(TE)[:, None]
    jd2 = int(np.argmin(np.abs(TED - m)))
    # (a) direct re-differentiation of the offset-corrected theory phase panel.
    #     This is unreliable where the theory phase has an angular step within
    #     0.3 rad of pi (unwrap direction ambiguous), flagged per energy.
    d_corr = interp_row(EDEL, EED, TED[jd2]) - d_th_c[jd2]
    r["delay_rms_after_offset_rediff_as"] = float(np.sqrt(np.mean(d_corr ** 2)))
    # (b) equivalent, numerically stable route: adding an energy-dependent phase
    #     offset c(E) to the theory phase changes tau by +dc/dE * hbar, so correct
    #     the PROVIDED delay panel by that smooth analytic term instead of
    #     re-differentiating a possibly-mis-unwrapped phase panel.
    dc_dE = np.gradient(off_curve, TE)
    slope_as = float(np.interp(TED[jd2], TE, dc_dE) * HBAR_AS_EV)
    t_del_corr = TDEL[jd2] + slope_as
    d_corr2 = interp_row(EDEL, EED, TED[jd2]) - t_del_corr
    r["delay_rms_after_offset_as"] = float(np.sqrt(np.mean(d_corr2 ** 2)))
    r["delay_r_after_offset"] = pearson(interp_row(EDEL, EED, TED[jd2]), t_del_corr)
    r["delay_offset_slope_as"] = slope_as
    # flag energies where re-differentiating the theory phase panel is unreliable
    # because an angular step sits within 0.3 rad of pi (unwrap direction ambiguous)
    j1 = int(np.argmin(np.abs(TE - m)))
    jump_th = float(np.abs(np.diff(TPHA_U[j1])).max())
    jump_exp = float(np.abs(np.diff(interp_row(EPHA_U, EE, TE[j1]))).max())
    r["max_phase_step_theory_rad"] = jump_th
    r["max_phase_step_exp_rad"] = jump_exp
    r["phase_unwrap_ambiguous"] = bool(jump_th > np.pi - 0.3 or jump_exp > np.pi - 0.3)

    # forward-sector (0-40 deg) and complementary-sector comparison, since the
    # delay differences are concentrated at forward angles
    fwd = slice(0, 41)
    bwd = slice(41, 181)
    e_d = interp_row(EDEL, EED, TED[jd2])
    t_d = TDEL[jd2]
    cb = circ_mean(e_pha[fwd] - t_pha[fwd])
    r["phase_rms_after_offset_0_40deg"] = float(np.sqrt(np.mean(wrap(e_pha[fwd] - t_pha[fwd] - cb) ** 2)))
    r["delay_rms_0_40deg_as"] = float(np.sqrt(np.mean((e_d[fwd] - t_d[fwd]) ** 2)))
    r["delay_rms_41_180deg_as"] = float(np.sqrt(np.mean((e_d[bwd] - t_d[bwd]) ** 2)))
    r["mag_rms_0_40deg"] = float(np.sqrt(np.mean((e_mag[fwd] - t_mag[fwd]) ** 2)))
    r["mag_rms_41_180deg"] = float(np.sqrt(np.mean((e_mag[bwd] - t_mag[bwd]) ** 2)))
    r["mag_nrms_0_40deg"] = (r["mag_rms_0_40deg"] / float(e_mag[fwd].std())
                             if e_mag[fwd].std() > 1e-3 else float("inf"))
    r["mag_nrms_41_180deg"] = (r["mag_rms_41_180deg"] / float(e_mag[bwd].std())
                               if e_mag[bwd].std() > 1e-3 else float("inf"))
    r["mag_backward_norm_unreliable"] = bool(e_mag[bwd].std() <= 1e-3)

    # (ii) isotropic component only
    e_pha = interp_row(EPHA_U, EE, TE[j])
    e_mag = interp_row(EMAG, EE, TE[j])
    ce = complex_legendre(e_mag * np.exp(1j * e_pha))
    ct = complex_legendre(TMAG[j] * np.exp(1j * TPHA_U[j]))
    r["iso_amp_ratio"] = r["c0_amp_ratio"]
    r["iso_phase_diff_deg"] = r["c0_phase_diff_deg"]
    r["iso_amp_ratio_dev"] = abs(r["c0_amp_ratio"] - 1.0)

    # (iii) 10-degree angular subsampling
    e_mag10 = interp_row(EMAG, EE, TE[j])[i10]
    t_mag10 = TMAG[j][i10]
    e_pha10 = interp_row(EPHA_U, EE, TE[j])[i10]
    t_pha10 = TPHA_U[j][i10]
    r["mag_r_10deg"] = pearson(e_mag10, t_mag10)
    r["mag_nrms_10deg"] = float(np.sqrt(np.mean((e_mag10 - t_mag10) ** 2))) / float(e_mag10.std())
    d10 = e_pha10 - t_pha10
    r["phase_rms_after_10deg"] = float(np.sqrt(np.mean(wrap(d10 - circ_mean(d10)) ** 2)))
    e_del10 = interp_row(EDEL, EED, TED[jd])[i10]
    t_del10 = TDEL[jd][i10]
    r["delay_r_10deg"] = pearson(e_del10, t_del10)
    r["delay_rms_10deg"] = float(np.sqrt(np.mean((e_del10 - t_del10) ** 2)))

print(f"{'E[eV]':>7} | {'magNRMS':>7} {'nn':>7} {'10deg':>7} | {'phaRes':>7} {'nn':>7} {'10d':>7} "
      f"{'offset':>7} | {'delRMS':>7} {'nn':>7} {'10d':>7} {'offset':>7} | {'mag_r_nn':>8} {'del_r_nn':>8} "
      f"| {'mag_r':>7} {'phaRes':>7} {'delRMS':>7}")
print("-" * 150)
for r in rows:
    print(f"{r['e_meas']:7.2f} | {r['mag_nrms_exp']:7.3f} {r['mag_nrms_nn']:7.3f} {r['mag_nrms_10deg']:7.3f} | "
          f"{r['phase_rms_after_rad']:7.3f} {r['phase_rms_after_nn']:7.3f} {r['phase_rms_after_10deg']:7.3f} "
          f"{r['phase_rms_after_offset_rad']:7.3f} | "
          f"{r['delay_rms_as']:7.1f} {r['delay_rms_nn']:7.1f} {r['delay_rms_10deg']:7.1f} "
          f"{r['delay_rms_after_offset_as']:7.1f} | {r['mag_r_nn']:8.3f} {r['delay_r_nn']:8.3f} "
          f"| {r['mag_r']:7.3f} {r['phase_rms_after_rad']:7.3f} {r['delay_rms_as']:7.1f}")

# qualitative-stability summary
def trend(x, y):
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


Es = np.array([r["e_meas"] for r in rows])
tr = {
    "mag_nrms_vs_E": trend(Es, [r["mag_nrms_exp"] for r in rows]),
    "mag_r_vs_E": trend(Es, [r["mag_r"] for r in rows]),
    "phase_resid_after_vs_E": trend(Es, [r["phase_rms_after_rad"] for r in rows]),
    "delay_rms_vs_E": trend(Es, [r["delay_rms_as"] for r in rows]),
    "delay_r_vs_E": trend(Es, [r["delay_r"] for r in rows]),
    "phase_offset_vs_E": trend(Es, [r["phase_offset_rad"] for r in rows]),
    "c2_amp_ratio_vs_E": trend(Es, [r["c2_amp_ratio"] for r in rows]),
    "c0_amp_ratio_vs_E": trend(Es, [r["c0_amp_ratio"] for r in rows]),
    "delay_rms_after_offset_vs_E": trend(Es, [r["delay_rms_after_offset_as"] for r in rows]),
    "mag_nrms_nn_vs_E": trend(Es, [r["mag_nrms_nn"] for r in rows]),
    "phase_resid_after_nn_vs_E": trend(Es, [r["phase_rms_after_nn"] for r in rows]),
    "delay_rms_nn_vs_E": trend(Es, [r["delay_rms_nn"] for r in rows]),
    "mag_nrms_10deg_vs_E": trend(Es, [r["mag_nrms_10deg"] for r in rows]),
    "phase_resid_after_10deg_vs_E": trend(Es, [r["phase_rms_after_10deg"] for r in rows]),
    "delay_rms_10deg_vs_E": trend(Es, [r["delay_rms_10deg"] for r in rows]),
    "mag_r_nn_vs_E": trend(Es, [r["mag_r_nn"] for r in rows]),
}
# do the three interpolation/interp variants pick the same best/worst energy?
for key in ("mag_nrms_exp", "mag_nrms_nn", "mag_nrms_10deg"):
    srt = sorted(rows, key=lambda r: r[key])
    print(f"  variant {key:22s}: best {srt[0]['e_meas']:.2f} eV  worst {srt[-1]['e_meas']:.2f} eV")
for key in ("phase_rms_after_rad", "phase_rms_after_nn", "phase_rms_after_10deg"):
    srt = sorted(rows, key=lambda r: r[key])
    print(f"  variant {key:22s}: best {srt[0]['e_meas']:.2f} eV  worst {srt[-1]['e_meas']:.2f} eV")
for key in ("delay_rms_as", "delay_rms_nn", "delay_rms_10deg", "delay_rms_after_offset_as"):
    srt = sorted(rows, key=lambda r: r[key])
    print(f"  variant {key:22s}: best {srt[0]['e_meas']:.2f} eV  worst {srt[-1]['e_meas']:.2f} eV")
print("\nSpearman trend across the 11 matched energies (rho, p):")
for k, v in tr.items():
    print(f"  {k:28s} rho={v[0]:+.3f}  p={v[1]:.4f}")

# --------------------------------------------------------------------------
# 7. (f) best / worst agreeing energies
# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(f) energy regions of agreement / disagreement")
print("=" * 78)
for key, label, better in [("mag_nrms_exp", "magnitude normalized RMS", "low"),
                           ("phase_rms_after_rad", "phase residual after offset [rad]", "low"),
                           ("delay_rms_as", "delay RMS [as]", "low"),
                           ("delay_medabs_as", "delay median|diff| [as]", "low")]:
    srt = sorted(rows, key=lambda r: r[key])
    print(f"{label:38s} best {srt[0]['e_meas']:6.2f} eV ({srt[0][key]:.3f})  "
          f"worst {srt[-1]['e_meas']:6.2f} eV ({srt[-1][key]:.3f})")
# composite rank (sum of ranks over three primary metrics)
for r in rows:
    r["_composite_rank"] = 0.0
for key in ("mag_nrms_exp", "phase_rms_after_rad", "delay_rms_as"):
    srt = sorted(rows, key=lambda r: r[key])
    for rank, r in enumerate(srt):
        r["_composite_rank"] += rank
comp = sorted(rows, key=lambda r: r["_composite_rank"])
print(f"composite (rank sum over mag-NRMS, phase-residual, delay-RMS):")
print(f"  best  {comp[0]['e_meas']:.2f} eV  (ranksum {comp[0]['_composite_rank']:.0f})")
print(f"  worst {comp[-1]['e_meas']:.2f} eV  (ranksum {comp[-1]['_composite_rank']:.0f})")
print("  full order:", [(r["e_meas"], int(r["_composite_rank"])) for r in comp])

# --------------------------------------------------------------------------
# 8. CSV + JSON + figure
# --------------------------------------------------------------------------
for r in rows:
    r.pop("_composite_rank", None)
tab = pd.DataFrame(rows)
col_order = ["e_meas", "e_theory", "dE_meV", "e_exp_node", "exp_node_off_meV", "e_delay_theory",
             "mag_r", "mag_rms", "mag_nrms_exp", "mag_nrms_theory", "mag_null_maxabsr",
             "mag_scramble_null_max_r", "mag_beats_scramble_null",
             "mag_scramble_rms_null_p5", "mag_better_than_scramble_rms_null_p5",
             "phase_r_unwrapped", "phase_r_stored", "phase_rms_before_rad", "phase_rms_after_rad",
             "phase_nrms_after", "phase_offset_rad", "phase_offset_deg",
             "phase_offset_circmean_rad", "phase_offset_gridsearch_agrees", "phase_null_maxabsr",
             "phase_scramble_null_max_r", "phase_beats_scramble_null",
             "phase_scramble_rms_null_p5", "phase_better_than_scramble_rms_null_p5",
             "delay_r", "delay_rms_as", "delay_medabs_as", "delay_nrms", "delay_argmax_deg",
             "delay_null_maxabsr", "delay_scramble_null_max_r", "delay_beats_scramble_null",
             "delay_scramble_rms_null_p5", "delay_better_than_scramble_rms_null_p5",
             "exp_phase_std_rad", "th_phase_std_rad",
             "exp_delay_std_as", "th_delay_std_as"]
col_order += [c for c in tab.columns if c not in col_order]
tab = tab[col_order]
tab.to_csv(CSV_PATH, index=False)
print(f"\nwrote {CSV_PATH} ({len(tab)} rows x {len(tab.columns)} cols)")

# figure -------------------------------------------------------------------
fig = plt.figure(figsize=(16.5, 12))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
E = np.array([r["e_meas"] for r in rows])

ax = fig.add_subplot(gs[0, 0])
ax.plot(E, [r["mag_r"] for r in rows], "o-", color="C0", label="Pearson r")
ax.plot(E, [r["mag_null_maxabsr"] for r in rows], "s--", color="gray",
        label="null: max|r| over 180 shifts")
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("magnitude r")
ax.set_title("(b) magnitude agreement"); ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(E, [r["mag_nrms_exp"] for r in rows], "o-", label="norm. RMS (exp panel)")
ax.plot(E, [r["mag_nrms_nn"] for r in rows], "^:", label="robust: nearest node")
ax.plot(E, [r["mag_nrms_10deg"] for r in rows], "v--", label="robust: 10 deg steps")
ax.plot(E, [r["mag_scramble_rms_null_p5"] for r in rows], "s--", color="gray",
        label="null: 5th pct, mismatched theory energy")
ax.axhline(1.0, color="k", lw=1, ls=":", label="baseline = panel's own s.d.")
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("RMS / s.d.(exp)")
ax.set_yscale("log")
ax.set_title("(b) magnitude normalized RMS vs null"); ax.legend(fontsize=6); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[0, 2])
ax.plot(E, [r["phase_rms_before_rad"] for r in rows], "o-", label="before offset (mod 2$\\pi$)")
ax.plot(E, [r["phase_rms_after_rad"] for r in rows], "s-", label="after best constant offset")
ax.plot(E, [r["phase_scramble_rms_null_p5"] for r in rows], "^--", color="gray",
        label="null: 5th pct, mismatched theory energy")
ax.plot(E, np.abs([r["phase_offset_rad"] for r in rows]), "d:", label="|offset| [rad]")
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("phase RMS [rad]")
ax.set_title("(c) phase residual vs constant offset"); ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[1, 0])
ax.plot(E, [r["delay_r"] for r in rows], "o-", color="C3", label="Pearson r")
ax.plot(E, [r["delay_r_nn"] for r in rows], "^:", color="C3", alpha=.6, label="robust: nearest node")
ax.plot(E, [r["delay_null_maxabsr"] for r in rows], "s--", color="gray", label="null max|r|")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("delay r")
ax.set_title("(e) delay agreement (correlation)"); ax.legend(fontsize=7); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[1, 1])
ax.plot(E, [r["delay_rms_as"] for r in rows], "o-", label="RMS [as]")
ax.plot(E, [r["delay_medabs_as"] for r in rows], "s-", label="median |diff| [as]")
ax.plot(E, [r["delay_rms_10deg"] for r in rows], "v--", label="RMS, 10 deg steps")
ax.plot(E, [r["delay_scramble_rms_null_p5"] for r in rows], "^-.", color="gray",
        label="null: 5th pct, mismatched theory energy")
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("delay difference [as]")
ax.set_yscale("log")
ax.set_title("(e) delay difference magnitude vs null"); ax.legend(fontsize=6); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot(E, [r["c0_amp_ratio"] for r in rows], "o-", label="|c$_0$| ratio exp/th")
ax.plot(E, [r["c2_amp_ratio"] for r in rows], "s-", label="|c$_2$| ratio exp/th")
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.set_xlabel("photon energy [eV]"); ax.set_ylabel("amplitude ratio")
ax.set_title("(d) isotropic / P$_2$ amplitude ratio"); ax.legend(fontsize=7); ax.grid(alpha=.3)

# angular profiles at the best-agreeing energy
best = sorted(rows, key=lambda r: abs(r["phase_rms_after_rad"]))[0]
worst = sorted(rows, key=lambda r: -abs(r["phase_rms_after_rad"]))[0]
for col, (tag, rr) in enumerate([("low-E agree", best), ("high-E disagree", worst)]):
    j = rr["idx_th"]
    jd = rr["idx_th_del"]
    ax = fig.add_subplot(gs[2, col])
    ax.plot(ANGLES, interp_row(EMAG, EE, rr["e_theory"]) / interp_row(EMAG, EE, rr["e_theory"]).mean(),
            color="C0", lw=1.2, label="exp magn. (norm.)")
    ax.plot(ANGLES, TMAG[j] / TMAG[j].mean(), color="C0", ls="--", label="theory magn. (norm.)")
    ax.plot(ANGLES, interp_row(EDEL, EED, TED[jd]) / 100.0, color="C3", lw=1.2,
            label="exp delay /100 as")
    ax.plot(ANGLES, TDEL[jd] / 100.0, color="C3", ls="--", label="theory delay /100 as")
    ax.set_xlabel("emission angle [deg]"); ax.set_ylabel("normalized")
    ax.set_title(f"@ {rr['e_meas']:.2f} eV ({tag})\nphase res. after off = "
                 f"{rr['phase_rms_after_rad']:.2f} rad, delay RMS = {rr['delay_rms_as']:.0f} as",
                 fontsize=9)
    ax.legend(fontsize=6); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[2, 2])
ax.plot(ANGLES, prof, color="C2")
ax.axvspan(0, 40, color="C2", alpha=.12)
ax.set_xlabel("emission angle [deg]"); ax.set_ylabel("mean |$\\tau_{exp}-\\tau_{th}$| [as]")
ax.set_title("(e) delay difference profile (11 matched E)")
ax.grid(alpha=.3)

fig.suptitle("p6_exptheory: experiment-like vs theory-like angle-resolved panels "
             "(11 exact-energy-matched pairs)", fontsize=13)
fig.savefig(PNG_PATH, dpi=130, bbox_inches="tight")
print(f"wrote {PNG_PATH}")

# --------------------------------------------------------------------------
# 9. JSON
# --------------------------------------------------------------------------
# plateau artifacts in the experiment magnitude panel (stored rounded values
# produce exactly repeated neighbours), which break sectorwise normalization
Dmag = np.abs(np.diff(EMAG, axis=1))
flat_frac = float((Dmag < 1e-3).mean())
print(f"\n[panel texture] experiment magnitude: {flat_frac:.3f} of (row,angle) steps are "
      f"exactly flat (|dM|<1e-3) -> coarse rounded/plateau texture in the exp panel")
print(f"[panel texture] theory magnitude: {float((np.abs(np.diff(TMAG, axis=1)) < 1e-3).mean()):.3f}")

summary = {
    "n_matched": n5,
    "exp_magnitude_flat_step_fraction": flat_frac,
    "brief_expected_matched": 12,
    "matched_dE_meV_all_minus_1": bool(np.all(np.abs(diffs + 1.0) < 1e-9)),
    "unmatched": unmatched,
    "theory_energy_coverage_eV": [float(TE.min()), float(TE.max())],
    "experiment_energy_coverage_eV": [float(EE.min()), float(EE.max())],
    "phase_wrap_points_exp": wrap_exp, "phase_wrap_points_th": wrap_th,
    "delay_phase_consistency_r_exp": r_delcheck_exp,
    "delay_phase_consistency_r_th": r_delcheck_th,
    "spearman_trends": {k: {"rho": v[0], "p": v[1]} for k, v in tr.items()},
    "best_phase_agreement_energy_eV": best["e_meas"],
    "best_phase_agreement_residual_rad": best["phase_rms_after_rad"],
    "worst_phase_agreement_energy_eV": worst["e_meas"],
    "worst_phase_agreement_residual_rad": worst["phase_rms_after_rad"],
    "delay_profile_windows_deg_as": {f"{a}-{a+30}": v for a, v in wins},
    "delay_profile_forward_0_40_as": float(prof[0:41].mean()),
    "delay_profile_mid_41_140_as": float(prof[41:141].mean()),
    "delay_profile_back_141_180_as": float(prof[141:].mean()),
    "phase_unwrap_ambiguous_energies_eV": [r["e_meas"] for r in rows if r["phase_unwrap_ambiguous"]],
    "beats_energy_scramble_null": {
        "magnitude": [r["e_meas"] for r in rows if r["mag_beats_scramble_null"]],
        "phase": [r["e_meas"] for r in rows if r["phase_beats_scramble_null"]],
        "delay": [r["e_meas"] for r in rows if r["delay_beats_scramble_null"]],
    },
    "n_beats_energy_scramble_null": {
        "magnitude": int(sum(r["mag_beats_scramble_null"] for r in rows)),
        "phase": int(sum(r["phase_beats_scramble_null"] for r in rows)),
        "delay": int(sum(r["delay_beats_scramble_null"] for r in rows)),
    },
    "n_better_than_scramble_rms_null_p5": {
        "magnitude": int(sum(r["mag_better_than_scramble_rms_null_p5"] for r in rows)),
        "phase": int(sum(r["phase_better_than_scramble_rms_null_p5"] for r in rows)),
        "delay": int(sum(r["delay_better_than_scramble_rms_null_p5"] for r in rows)),
    },
    "better_than_scramble_rms_null_p5": {
        "magnitude": [r["e_meas"] for r in rows if r["mag_better_than_scramble_rms_null_p5"]],
        "phase": [r["e_meas"] for r in rows if r["phase_better_than_scramble_rms_null_p5"]],
        "delay": [r["e_meas"] for r in rows if r["delay_better_than_scramble_rms_null_p5"]],
    },
    "median_over_energies": {
        k: float(np.median([r[k] for r in rows])) for k in
        ("mag_r", "mag_nrms_exp", "phase_r_unwrapped", "phase_rms_before_rad",
         "phase_rms_after_rad", "phase_offset_rad", "delay_r", "delay_rms_as",
         "delay_rms_0_40deg_as", "delay_rms_41_180deg_as", "c0_amp_ratio", "c2_amp_ratio",
         "delay_rms_10deg", "mag_nrms_10deg", "phase_rms_after_10deg")
    },
}
result = {
    "package": "p6_exptheory",
    "data_sha256": hashes,
    "hbar_as_eV": HBAR_AS_EV,
    "matched_pairs": pairs,
    "per_energy": rows,
    "summary": summary,
    "notes": [
        "experiment panel values were linearly interpolated in energy to the exact theory "
        "grid energies (experiment grid step 0.1 eV); nearest-node values are reported as "
        "the robustness variant (columns *_nn).",
        "phases are stored wrapped in (-pi,pi]; all phase statistics use angle-unwrapped "
        "panels, and 'phase_r_stored' (Pearson on stored wrapped values) is reported as a "
        "documented artifact of the wrap discontinuities.",
        "the experiment-like dense grid is documented as a derived spline grid from 13 "
        "measured DME energy blocks, so its cells are not independent measurements.",
        "delay panels equal the forward energy-difference of the phase panels times "
        "hbar_as_eV, labelled with the upper energy of each interval.",
    ],
}
with open(JSON_PATH, "w") as fh:
    json.dump(result, fh, indent=1, default=float)
print(f"wrote {JSON_PATH}")
print("\ndone.")