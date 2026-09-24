#!/usr/bin/env python3
"""p3_phase_delay_curves

Phase-to-delay conversion in the experiment and theory panels (angle-energy
observations) plus the three standalone curves (0.5_Wigner_delay,
angle_averaged_delay, eigenphase_sum).

Deliverables (all under ROOT/output/):
  evidence/p3_phase_delay_curves.json   machine-readable results
  evidence/p3_phase_delay_curves.log    full stdout/stderr of this run
  evidence/p3_phase_delay_curves.png    figure
  evidence/p3_aligned_curves.csv        aligned curve table

Re-runnable from scratch; only reads ROOT/data/.
"""

import hashlib
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = "/data/user/conversations/conv-9e234f5e-4d56-4e60-8340-10f7a5d416d0/project"
DATA = os.path.join(ROOT, "data")
OUT_EV = os.path.join(ROOT, "output", "evidence")
OUT_AN = os.path.join(ROOT, "output", "analysis")
LOG_PATH = os.path.join(OUT_EV, "p3_phase_delay_curves.log")
JSON_PATH = os.path.join(OUT_EV, "p3_phase_delay_curves.json")
PNG_PATH = os.path.join(OUT_EV, "p3_phase_delay_curves.png")
CSV_PATH = os.path.join(OUT_EV, "p3_aligned_curves.csv")

os.makedirs(OUT_EV, exist_ok=True)
os.makedirs(OUT_AN, exist_ok=True)

# ---------------------------------------------------------------------------
# stdout/stderr tee into the log file (full duplication of stdout)
# ---------------------------------------------------------------------------
_log_fh = open(LOG_PATH, "w")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)

RESULT = {"package": "p3_phase_delay_curves", "failures": [], "sections": {}}


def log(*a):
    print(*a)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def try_section(name, fn):
    log("\n" + "=" * 78 + "\nSECTION %s\n" % name + "=" * 78)
    try:
        out = fn()
        RESULT["sections"][name] = out
        log("[%s] done" % name)
        return out
    except Exception:
        tb = traceback.format_exc()
        log("[%s] FAILED\n%s" % (name, tb))
        RESULT["failures"].append({"section": name, "error": tb.strip().splitlines()[-1], "traceback": tb})
        return None


def stats(pred, obs, label=""):
    """abs-difference summary of obs vs pred (same shape)."""
    d = np.abs(obs - pred)
    out = {
        "label": label,
        "median_abs_as": float(np.median(d)),
        "p95_abs_as": float(np.percentile(d, 95)),
        "max_abs_as": float(d.max()),
        "rms_as": float(np.sqrt(np.mean(d ** 2))),
        "median_ratio_obs_over_pred": float(np.median(obs / pred)),
    }
    # cells where |pred| is non-trivial, so the ratio is meaningful
    m = np.abs(pred) > 1.0
    if m.sum() > 10:
        out["median_ratio_pred_gt_1as"] = float(np.median(obs[m] / pred[m]))
        out["frac_cells_abs_pred_gt_1as"] = float(np.mean(m))
        c = float(np.sum(obs[m] * pred[m]) / np.sum(pred[m] ** 2))
        out["lsq_scale_on_pred_gt_1as"] = c
    return out


def locpoly_deriv(y, x, xout, deg=2, half=2):
    """Local polynomial derivative of columns of y (m x K) on grid x evaluated at xout."""
    out = np.empty((len(xout), y.shape[1]))
    n = len(x)
    for i, xe in enumerate(xout):
        j = int(np.argmin(np.abs(x - xe)))
        lo = max(0, j - half)
        hi = min(n - 1, j + half)
        idx = np.arange(lo, hi + 1)
        if len(idx) < deg + 1:
            idx = np.arange(max(0, hi - deg), hi + 1)
        c = np.polyfit(x[idx], y[idx, :], deg)
        d = np.zeros(y.shape[1])
        for k in range(deg):
            d = d + (deg - k) * c[k] * xe ** (deg - k - 1)
        out[i, :] = d
    return out


# ---------------------------------------------------------------------------
# load coordinate definitions + all data used (shas recorded)
# ---------------------------------------------------------------------------
FILES = {}
log("### loading data")
coord_path = os.path.join(DATA, "coordinate_definitions.json")
with open(coord_path) as f:
    coord = json.load(f)
FILES[os.path.relpath(coord_path, DATA)] = sha256(coord_path)
HBAR = float(coord["phase_delay_conversion"]["hbar_as_eV"])
log("hbar_as_eV =", HBAR, " (from coordinate_definitions.json)")

obs_path = os.path.join(DATA, "angle_energy_observations_datasets.tsv")
FILES[os.path.relpath(obs_path, DATA)] = sha256(obs_path)
dfobs = pd.read_csv(obs_path, sep="\t")

cur_path = os.path.join(DATA, "phase_and_delay_curves_datasets.tsv")
FILES[os.path.relpath(cur_path, DATA)] = sha256(cur_path)
dfcur = pd.read_csv(cur_path, sep="\t")

RESULT["data_files_sha256"] = FILES
RESULT["hbar_as_eV"] = HBAR
RESULT["hbar_source"] = os.path.relpath(coord_path, DATA)


def panel2d(ds):
    d = dfobs[dfobs.dataset_path == ds]
    return d.pivot_table(index="index_0", columns="index_1", values="value").to_numpy()


def panel1d(ds):
    d = dfobs[dfobs.dataset_path == ds].sort_values("index_0")
    return d.value.to_numpy()


def curve(name):
    d = dfcur[dfcur.dataset_path == name].sort_values(["index_1", "index_0"])
    e = d[d.index_1 == 0].sort_values("index_0").value.to_numpy()
    v = d[d.index_1 == 1].sort_values("index_0").value.to_numpy()
    return e, v


ph_e = panel2d("experiment/phase")          # 276 x 181
dl_e = panel2d("experiment/delay")          # 275 x 181
E_e = panel1d("experiment/photon_energy_eV")     # 276
Ed_e = panel1d("experiment/photon_energy_eV_del")  # 275

ph_t = panel2d("theory/phase")              # 99 x 181
dl_t = panel2d("theory/delay")              # 98 x 181
E_t = panel1d("theory/photon_energy_eV")    # 99
Ed_t = panel1d("theory/photon_energy_eV_del")      # 98

Ewig, W = curve("0.5_Wigner_delay")
Eaa, aa = curve("angle_averaged_delay")
Eeig, Seig = curve("eigenphase_sum")

log("experiment: phase %s delay %s  E %d Ed %d" % (ph_e.shape, dl_e.shape, E_e.size, Ed_e.size))
log("theory    : phase %s delay %s  E %d Ed %d" % (ph_t.shape, dl_t.shape, E_t.size, Ed_t.size))
log("Wigner %d pts %.2f..%.2f | angle_averaged %d pts %.4f..%.4f | eigenphase %d pts %.2f..%.2f"
    % (Ewig.size, Ewig[0], Ewig[-1], Eaa.size, Eaa[0], Eaa[-1], Eeig.size, Eeig[0], Eeig[-1]))
log("grid checks: max|Ed_e-E_e[1:]| = %.3e ; max|Ed_t-E_t[1:]| = %.3e"
    % (np.abs(Ed_e - E_e[1:]).max(), np.abs(Ed_t - E_t[1:]).max()))

# ===========================================================================
# (a) experiment panel: delay == hbar * dphase/dE (2-pt forward difference)
# ===========================================================================
def part_a():
    h = np.diff(E_e)                       # 0.1 eV uniform
    assert np.allclose(h, 0.1)
    dphi = np.diff(ph_e, axis=0)           # 275 x 181, between E[i], E[i+1] = Ed[i]
    pred = HBAR * dphi / h[:, None]
    out = {"scheme": "2-point forward difference on the 0.1 eV grid, evaluated at Ed[i]=E[i+1]"}
    out.update(stats(pred, dl_e, "experiment delay vs hbar*dphase/dE"))
    res = dl_e - pred
    out["R2_vs_zero"] = float(1 - (res ** 2).sum() / ((dl_e - dl_e.mean()) ** 2).sum())
    out["frac_cells_abs_diff_lt_1as"] = float(np.mean(np.abs(res) < 1.0))
    out["frac_rows_with_max_abs_diff_gt_1as"] = float(np.mean(np.abs(res).max(axis=1) > 1.0))
    out["delay_median_abs_as"] = float(np.median(np.abs(dl_e)))
    out["delay_std_as"] = float(dl_e.std())
    out["delay_range_as"] = [float(dl_e.min()), float(dl_e.max())]
    out["n_cells"] = int(dl_e.size)

    # ---- nulls -----------------------------------------------------------
    const_med = float(np.median(np.abs(dl_e - np.median(dl_e))))
    const_rms = float(np.sqrt(np.mean((dl_e - np.median(dl_e)) ** 2)))
    rng = np.random.default_rng(0)
    perm = dl_e[:, rng.permutation(dl_e.shape[1])]
    perm_d = np.abs(perm - pred)
    out["null_constant"] = {"median_abs_as": const_med, "rms_as": const_rms}
    out["null_angle_permuted_delay"] = {
        "median_abs_as": float(np.median(perm_d)),
        "p95_abs_as": float(np.percentile(perm_d, 95)),
        "rms_as": float(np.sqrt(np.mean(perm_d ** 2))),
    }
    out["null_rejected"] = bool(out["median_abs_as"] < 0.01 * const_med)
    out["verdict"] = (
        "null rejected: median |delay - hbar*dphi/dE| = %.4f as vs %.2f as for the constant null "
        "(factor %.0f smaller) and %.2f as for the angle-permuted null; median ratio %.7f"
        % (out["median_abs_as"], const_med, const_med / out["median_abs_as"],
           out["null_angle_permuted_delay"]["median_abs_as"], out["median_ratio_obs_over_pred"]))

    # ---- robustness ------------------------------------------------------
    fl = np.abs(res)
    thr = np.percentile(fl, 99)
    out["robust_trim_top1pct"] = {
        "median_abs_as": float(np.median(fl[fl <= thr])),
        "max_abs_as": float(fl[fl <= thr].max()),
        "threshold_as": float(thr),
    }
    kink = ((Ed_e >= 43.0) & (Ed_e <= 44.0)) | ((Ed_e >= 48.6) & (Ed_e <= 49.6))
    keep = ~kink
    out["robust_exclude_43-44_48.6-49.6eV"] = {
        "rows_excluded": int(kink.sum()),
        "median_abs_as": float(np.median(fl[keep])),
        "p95_abs_as": float(np.percentile(fl[keep], 95)),
        "max_abs_as": float(fl[keep].max()),
    }
    per_ang = np.median(fl, axis=0)
    out["per_angle_median_abs_as"] = {
        "min": float(per_ang.min()), "median": float(np.median(per_ang)), "max": float(per_ang.max())
    }
    # exact pairing check: is Ed the only convention that works?
    shifts = {}
    for s in (-2, -1, 0, 1, 2):
        if s < 0:
            a, b = dl_e[-s:], HBAR * dphi[:s] / 0.1
        elif s > 0:
            a, b = dl_e[:-s], HBAR * dphi[s:] / 0.1
        else:
            a, b = dl_e, HBAR * dphi / 0.1
        shifts[str(s)] = float(np.median(np.abs(a - b)))
    out["pairing_shift_scan_median_abs_as"] = shifts

    log(json.dumps(out, indent=2))
    return out


try_section("a_experiment_fd", part_a)

# ===========================================================================
# (b) improved derivatives in the experiment panel
# ===========================================================================
def part_b():
    h = 0.1
    Emid = 0.5 * (E_e[:-1] + E_e[1:])
    schemes = {}

    fd = HBAR * np.diff(ph_e, axis=0) / h
    schemes["2pt_forward_at_Ed"] = stats(fd, dl_e, "2pt forward")

    q5_mid = HBAR * locpoly_deriv(ph_e, E_e, Emid, deg=2, half=2)
    schemes["local_quadratic_win5_at_midpoint"] = stats(q5_mid, dl_e, "quad5 midpoint")

    q5_ed = HBAR * locpoly_deriv(ph_e, E_e, Ed_e, deg=2, half=2)
    schemes["local_quadratic_win5_at_Ed"] = stats(q5_ed, dl_e, "quad5 Ed")

    cs = CubicSpline(E_e, ph_e, axis=0)
    schemes["cubic_spline_at_midpoint"] = stats(HBAR * cs(Emid, 1), dl_e, "spline midpoint")
    schemes["cubic_spline_at_Ed"] = stats(HBAR * cs(Ed_e, 1), dl_e, "spline Ed")

    # 5-point 4th-order stencil (uniform grid): derivative at x_{i+2}=E[i+2], row i=0..271;
    # row i of the delay panel sits at Ed[i]=E[i+1], so it pairs with i+1 -> rows 1..272
    st5 = HBAR * (ph_e[:-4] - 8 * ph_e[1:-3] + 8 * ph_e[3:-1] - ph_e[4:]) / (12 * h)
    schemes["5pt_4thorder_at_E[i+2]_rows1_272"] = stats(st5, dl_e[1:273], "5pt stencil")

    # scipy savgol node derivative (uniform grid), evaluated at Ed[i]=E[i+1]
    try:
        from scipy.signal import savgol_filter
        sg = savgol_filter(ph_e, 5, 2, deriv=1, axis=0, delta=h, mode="interp")
        schemes["savgol_filter_win5_deg2_at_Ed"] = stats(HBAR * sg[1:], dl_e, "savgol at Ed")
    except Exception as e:  # pragma: no cover
        RESULT["failures"].append({"section": "b_savgol", "error": repr(e)})

    best = min(schemes.items(), key=lambda kv: kv[1]["median_abs_as"])
    out = {"schemes": schemes, "best": best[0],
           "best_median_abs_as": best[1]["median_abs_as"],
           "ratio_worst_to_best_median": float(
               max(v["median_abs_as"] for v in schemes.values()) / best[1]["median_abs_as"]),
           "note": ("the 2-point difference that matches the row alignment is already the best "
                    "predictor, so the derivative is NOT improved by a smoother local model; "
                    "quadratic/spline derivatives are worse in median and catastrophically worse in "
                    "max because they inject curvature from the coarse (0.1 eV) sampled phase")}
    # authenticity check: does phase actually have an independent noise floor, or is it a
    # smooth spline of delay through the derivative?  Implied phase-noise eps from the residual:
    res_e = dl_e - fd
    eps = -(0.1 / HBAR) * np.cumsum(res_e, axis=0)
    out["implied_phase_noise_from_residual"] = {
        "std_rad": float(eps.std()), "max_abs_rad": float(np.abs(eps).max()),
        "phase_quantization_5th_decimal_std_rad": float(1e-5 / np.sqrt(12)),
        "ratio_std_to_quantization": float(eps.std() / (1e-5 / np.sqrt(12))),
        "note": "phase is stored to 5 decimals (+-5e-6); residual implies ~10x that, i.e. delay and "
                "phase are separately stored real-valued series, not one derived from the other verbatim",
    }
    # robustness: quad5/spline residual after excluding the two kink regions
    kink = ((Ed_e >= 43.0) & (Ed_e <= 44.0)) | ((Ed_e >= 48.6) & (Ed_e <= 49.6))
    keep = ~kink
    out["robust_exclude_kinks"] = {}
    for nm, pr in (("local_quadratic_win5_at_midpoint", q5_mid),
                   ("cubic_spline_at_midpoint", HBAR * cs(Emid, 1)),
                   ("2pt_forward_at_Ed", fd)):
        d = np.abs(dl_e - pr)[keep]
        out["robust_exclude_kinks"][nm] = {
            "median_abs_as": float(np.median(d)),
            "p95_abs_as": float(np.percentile(d, 95)),
            "max_abs_as": float(d.max()),
        }
    log(json.dumps(out, indent=2))
    return out


try_section("b_improved_derivative_experiment", part_b)

# ===========================================================================
# (c) theory panel, non-uniform energy grid
# ===========================================================================
def part_c():
    h = np.diff(E_t)
    Emid = 0.5 * (E_t[:-1] + E_t[1:])
    pred = HBAR * np.diff(ph_t, axis=0) / h[:, None]
    out = {"scheme": "2-point forward difference on the non-uniform theory grid, evaluated at Ed_t[i]"}
    out.update(stats(pred, dl_t, "theory delay vs hbar*dphase/dE"))
    res = dl_t - pred
    out["R2_vs_zero"] = float(1 - (res ** 2).sum() / ((dl_t - dl_t.mean()) ** 2).sum())
    out["frac_cells_abs_diff_lt_1as"] = float(np.mean(np.abs(res) < 1.0))
    out["delay_median_abs_as"] = float(np.median(np.abs(dl_t)))
    out["delay_range_as"] = [float(dl_t.min()), float(dl_t.max())]
    out["n_cells"] = int(dl_t.size)
    i, j = np.unravel_index(np.argmax(np.abs(res)), res.shape)
    out["worst_cell"] = {"Ed_eV": float(Ed_t[i]), "angle_deg": int(j), "abs_diff_as": float(abs(res[i, j]))}
    out["energy_grid"] = {"min_eV": float(E_t.min()), "max_eV": float(E_t.max()),
                          "mean_step_eV": float(h.mean()), "min_step_eV": float(h.min()),
                          "max_step_eV": float(h.max())}

    const_med = float(np.median(np.abs(dl_t - np.median(dl_t))))
    const_rms = float(np.sqrt(np.mean((dl_t - np.median(dl_t)) ** 2)))
    rng = np.random.default_rng(1)
    perm = dl_t[:, rng.permutation(dl_t.shape[1])]
    out["null_constant"] = {"median_abs_as": const_med, "rms_as": const_rms}
    out["null_angle_permuted_delay"] = {
        "median_abs_as": float(np.median(np.abs(perm - pred))),
        "rms_as": float(np.sqrt(np.mean((perm - pred) ** 2))),
    }
    out["null_rejected"] = bool(out["median_abs_as"] < 0.01 * const_med)
    out["verdict"] = (
        "null rejected on the non-uniform grid: median |delay - hbar*dphi/dE| = %.4f as vs %.2f as "
        "for the constant null (factor %.0f smaller) and %.2f as for the angle-permuted null"
        % (out["median_abs_as"], const_med, const_med / out["median_abs_as"],
           out["null_angle_permuted_delay"]["median_abs_as"]))

    # derivative variants on the non-uniform grid
    vari = {}
    vari["2pt_forward_at_Ed"] = stats(pred, dl_t, "2pt")
    vari["local_quadratic_win5_at_Ed"] = stats(HBAR * locpoly_deriv(ph_t, E_t, Ed_t, 2, 2), dl_t, "quad5")
    vari["local_quadratic_win5_at_midpoint"] = stats(HBAR * locpoly_deriv(ph_t, E_t, Emid, 2, 2), dl_t, "quad5 mid")
    cs = CubicSpline(E_t, ph_t, axis=0)
    vari["cubic_spline_at_Ed"] = stats(HBAR * cs(Ed_t, 1), dl_t, "spline")
    vari["cubic_spline_at_midpoint"] = stats(HBAR * cs(Emid, 1), dl_t, "spline mid")
    # 5-point 4th order on non-uniform-ish grid, evaluated by interpolation, interior only
    x5 = E_t[2:-2]
    c5 = (ph_t[:-4] - 8 * ph_t[1:-3] + 8 * ph_t[3:-1] - ph_t[4:]) / 12.0
    d5 = np.column_stack([np.interp(Ed_t, x5, c5[:, k]) for k in range(c5.shape[1])])
    m = (Ed_t > E_t[5]) & (Ed_t < E_t[-6])
    d5r = np.abs(dl_t - HBAR * d5)[m]
    vari["5pt_4thorder_interior_only"] = {
        "label": "5pt 4th order, interior",
        "median_abs_as": float(np.median(d5r)), "p95_abs_as": float(np.percentile(d5r, 95)),
        "max_abs_as": float(d5r.max()), "rms_as": float(np.sqrt(np.mean(d5r ** 2))), "n_cells": int(d5r.size),
    }
    m2 = np.broadcast_to(m[:, None], res.shape)
    vari["2pt_forward_interior_only"] = {
        "label": "2pt forward, same interior cells",
        "median_abs_as": float(np.median(np.abs(res)[m2])),
        "p95_abs_as": float(np.percentile(np.abs(res)[m2], 95)),
        "max_abs_as": float(np.abs(res)[m2].max()),
        "rms_as": float(np.sqrt(np.mean(res[m2] ** 2))),
    }
    out["derivative_variants"] = vari
    out["best_scheme"] = min(
        {k: v for k, v in vari.items() if "median_abs_as" in v},
        key=lambda k: vari[k]["median_abs_as"])
    log(json.dumps(out, indent=2))
    return out


try_section("c_theory_fd", part_c)

# ===========================================================================
# (d) 0.5_Wigner_delay vs hbar * d(eigenphase_sum)/dE, factor 1 and 2
# ===========================================================================
def part_d():
    out = {"relation": "0.5_Wigner_delay (as) ?= F * hbar * d(eigenphase_sum)/dE, F in {1,2}"}
    g_node = np.gradient(Seig, Eeig)                 # finite difference on the 0.5 eV grid
    csS = CubicSpline(Eeig, Seig)
    g_node_spl = csS(Eeig, 1)                        # cubic-spline derivative on the same grid
    g_full = np.interp(Ewig, Eeig, g_node)           # FD derivative interpolated to the 0.1 eV grid
    g_full_spl = csS(Ewig, 1)                        # spline derivative evaluated on the 0.1 eV grid
    gn_full = np.interp(Ewig, Eeig, g_node_spl)

    # grid alignment for the node-based comparison
    m_nodes = np.isin(np.round(Ewig, 10), np.round(Eeig, 10))
    Wn = W[m_nodes]
    out["n_wigner_points"] = int(Ewig.size)
    out["n_eigenphase_nodes"] = int(Eeig.size)
    out["wigner_std_as"] = float(W.std())
    out["null_constant_wigner"] = {"rms_as": float(np.sqrt(np.mean((W - W.mean()) ** 2))),
                                   "median_abs_as": float(np.median(np.abs(W - np.median(W))))}

    variants = {}
    for F in (1, 2):
        p_full = F * HBAR * g_full
        p_full_spl = F * HBAR * g_full_spl
        p_node = F * HBAR * g_node
        p_node_spl = F * HBAR * g_node_spl
        variants["factor_%d_fd_on_0.5eV_interp_to_0.1eV" % F] = stats(p_full, W, "F=%d FD" % F)
        variants["factor_%d_spline_deriv_on_0.1eV" % F] = stats(p_full_spl, W, "F=%d spline" % F)
        variants["factor_%d_fd_nodes_only" % F] = stats(p_node, Wn, "F=%d FD nodes" % F)
        variants["factor_%d_spline_nodes_only" % F] = stats(p_node_spl, Wn, "F=%d spline nodes" % F)
    out["variants"] = variants

    # best-fit multiplicative factor
    fs = np.linspace(0.5, 1.5, 2001)
    meds = np.array([np.median(np.abs(W - f * HBAR * g_full)) for f in fs])
    i = int(np.argmin(meds))
    out["best_fit_factor"] = {"factor": float(fs[i]), "median_abs_as": float(meds[i])}
    ok = fs[meds < 2.0]
    out["factor_range_with_median_lt_2as"] = [float(ok.min()), float(ok.max())]
    f2_med = float(np.median(np.abs(W - 2 * HBAR * g_full)))
    out["factor_2_excluded"] = bool(f2_med > 5.0)
    out["support"] = ("factor 1 supported: median |diff| %.3f as vs %.3f as for factor 2, "
                      "median ratio %.4f (factor 1) vs %.4f (factor 2)"
                      % (variants["factor_1_fd_on_0.5eV_interp_to_0.1eV"]["median_abs_as"],
                         variants["factor_2_fd_on_0.5eV_interp_to_0.1eV"]["median_abs_as"],
                         variants["factor_1_fd_on_0.5eV_interp_to_0.1eV"]["median_ratio_obs_over_pred"],
                         variants["factor_2_fd_on_0.5eV_interp_to_0.1eV"]["median_ratio_obs_over_pred"]))
    # context: is the eigenphase sum curve the phase of either panel?
    ph_e_mean = ph_e.mean(axis=1)
    pe = np.interp(Eeig, E_e, ph_e_mean)
    ph_t_mean = ph_t.mean(axis=1)
    pt = np.interp(Eeig, E_t, ph_t_mean)
    out["context_corr_eigenphase_vs_panel_mean_phase"] = {
        "experiment": float(np.corrcoef(Seig, pe)[0, 1]),
        "theory": float(np.corrcoef(Seig, pt)[0, 1]),
        "median_abs_eigenphase_vs_experiment_mean_phase_rad": float(np.median(np.abs(Seig - pe))),
    }
    log(json.dumps(out, indent=2))
    return out


try_section("d_eigenphase_vs_wigner", part_d)

# ===========================================================================
# (e) is angle_averaged_delay the angular average of the panel delay?
# ===========================================================================
def offset_scan(Ed, panel, label, IE_grid=None):
    """Scan photon energy = eKE + IE; residual of interpolated angle-average vs the curve."""
    if IE_grid is None:
        IE_grid = np.arange(12.0, 22.5001, 0.01)
    rows = []
    for IE in IE_grid:
        ph = Eaa + IE
        m = (ph >= Ed[0]) & (ph <= Ed[-1])
        if m.sum() < 20:
            continue
        p = np.interp(ph[m], Ed, panel)
        r = p - aa[m]
        rows.append((float(IE), float(m.sum()), float(np.sqrt(np.mean(r ** 2))),
                     float(np.median(np.abs(r))), float(np.corrcoef(p, aa[m])[0, 1]),
                     float(np.median(aa[m] / p))))
    rows = np.array(rows)
    ib = int(np.argmin(rows[:, 2]))
    return rows, ib


def residual_at(Ed, panel, IE):
    ph = Eaa + IE
    m = (ph >= Ed[0]) & (ph <= Ed[-1])
    p = np.interp(ph[m], Ed, panel)
    r = p - aa[m]
    const = np.sqrt(np.mean((aa[m] - aa[m].mean()) ** 2))
    lin = np.polyfit(p, aa[m], 1)
    rms_scaled = np.sqrt(np.mean((np.polyval(lin, p) - aa[m]) ** 2))
    return {
        "IE_eV": float(IE),
        "n_points": int(m.sum()),
        "rms_as": float(np.sqrt(np.mean(r ** 2))),
        "median_abs_as": float(np.median(np.abs(r))),
        "corr": float(np.corrcoef(p, aa[m])[0, 1]),
        "median_ratio_aa_over_panel": float(np.median(aa[m] / p)),
        "best_linear_slope_aa_on_panel": float(lin[0]),
        "rms_after_best_scale_offset_as": float(rms_scaled),
        "constant_residual_null_rms_as": float(const),
        "eKE_window_eV": [float(Eaa[m][0]), float(Eaa[m][-1])],
    }


def part_e():
    out = {}
    avg_e = dl_e.mean(axis=1)
    avg_t = dl_t.mean(axis=1)

    rows_e, ib_e = offset_scan(Ed_e, avg_e, "experiment")
    rows_t, ib_t = offset_scan(Ed_t, avg_t, "theory")
    out["experiment"] = {
        "scan_range_eV": [12.0, 22.5], "scan_step_eV": 0.01,
        "best": residual_at(Ed_e, avg_e, rows_e[ib_e, 0]),
        "corr_scan_max": {"IE_eV": float(rows_e[np.argmax(rows_e[:, 4]), 0]),
                          "corr": float(rows_e[:, 4].max())},
        "alternatives": {
            "offset_0": residual_at(Ed_e, avg_e, 0.0),
            "offset_ranges_just_overlap": residual_at(Ed_e, avg_e, Ed_e[0] - Eaa[0]),
        },
    }
    out["theory"] = {
        "scan_range_eV": [12.0, 22.5], "scan_step_eV": 0.01,
        "best": residual_at(Ed_t, avg_t, rows_t[ib_t, 0]),
        "corr_scan_max": {"IE_eV": float(rows_t[np.argmax(rows_t[:, 4]), 0]),
                          "corr": float(rows_t[:, 4].max())},
        "alternatives": {
            "offset_0": residual_at(Ed_t, avg_t, 0.0),
            "offset_ranges_just_overlap": residual_at(Ed_t, avg_t, Ed_t[0] - Eaa[0]),
        },
    }
    out["curve_own_scale"] = {
        "angle_averaged_std_as": float(aa.std()),
        "angle_averaged_range_as": [float(aa.min()), float(aa.max())],
        "angle_averaged_median_abs_consecutive_diff_as": float(np.median(np.abs(np.diff(aa)))),
        "panel_delay_angle_dependence_mean_row_range_as": float(np.mean(dl_e.max(axis=1) - dl_e.min(axis=1))),
    }

    # ---- robustness variants -------------------------------------------------
    rob = {}
    # (i) interpolation basis: cubic spline instead of linear
    def best_with_spline(Ed, panel):
        best = None
        cs = CubicSpline(Ed, panel)
        for IE in np.arange(12.0, 22.5001, 0.01):
            ph = Eaa + IE
            m = (ph >= Ed[0]) & (ph <= Ed[-1])
            if m.sum() < 20:
                continue
            p = cs(ph[m])
            r = np.sqrt(np.mean((p - aa[m]) ** 2))
            if best is None or r < best[1]:
                best = (float(IE), float(r), float(np.corrcoef(p, aa[m])[0, 1]), int(m.sum()))
        return {"IE_eV": best[0], "rms_as": best[1], "corr": best[2], "n_points": best[3]}
    rob["spline_interpolation_experiment"] = best_with_spline(Ed_e, avg_e)
    rob["spline_interpolation_theory"] = best_with_spline(Ed_t, avg_t)
    # (ii) median-abs objective instead of rms
    def best_medabs(Ed, panel):
        best = None
        for IE in np.arange(12.0, 22.5001, 0.01):
            ph = Eaa + IE
            m = (ph >= Ed[0]) & (ph <= Ed[-1])
            if m.sum() < 20:
                continue
            p = np.interp(ph[m], Ed, panel)
            v = np.median(np.abs(p - aa[m]))
            if best is None or v < best[1]:
                best = (float(IE), float(v), int(m.sum()))
        return {"IE_eV": best[0], "median_abs_as": best[1], "n_points": best[2]}
    rob["median_abs_objective_experiment"] = best_medabs(Ed_e, avg_e)
    rob["median_abs_objective_theory"] = best_medabs(Ed_t, avg_t)
    # (iii) angle weightings of the panel delay
    th = np.deg2rad(np.arange(181))
    wsin = np.sin(th)
    sin_avg_e = (dl_e * wsin).sum(axis=1) / wsin.sum()
    sin_avg_t = (dl_t * wsin).sum(axis=1) / wsin.sum()
    mg_e = (0.5 * (panel2d("experiment/magnitude")[:-1] + panel2d("experiment/magnitude")[1:]))
    wt_e = mg_e * wsin
    cs_avg_e = (dl_e * wt_e).sum(axis=1) / wt_e.sum(axis=1)
    mg_t = panel2d("theory/magnitude")[1:]
    wt_t = mg_t * wsin
    cs_avg_t = (dl_t * wt_t).sum(axis=1) / wt_t.sum(axis=1)
    for lab, av, Ed in (("sin_weighted_experiment", sin_avg_e, Ed_e),
                        ("cross_section_weighted_experiment", cs_avg_e, Ed_e),
                        ("sin_weighted_theory", sin_avg_t, Ed_t),
                        ("cross_section_weighted_theory", cs_avg_t, Ed_t)):
        rows, ib = offset_scan(Ed, av, lab)
        rob[lab] = {"best_IE_eV": float(rows[ib, 0]), "rms_as": float(rows[ib, 2]),
                    "corr": float(rows[ib, 4]), "n_points": int(rows[ib, 1])}
    # (iv) fixed eKE window (same points for every IE)
    def fixed_window(Ed, panel, lo, hi):
        f = (Eaa >= lo) & (Eaa <= hi)
        const = float(np.sqrt(np.mean((aa[f] - aa[f].mean()) ** 2)))
        best = None
        for IE in np.arange(12.0, 22.5001, 0.01):
            ph = Eaa[f] + IE
            if ph.min() < Ed[0] or ph.max() > Ed[-1]:
                continue
            p = np.interp(ph, Ed, panel)
            r = np.sqrt(np.mean((p - aa[f]) ** 2))
            if best is None or r < best[1]:
                best = (float(IE), float(r), float(np.corrcoef(p, aa[f])[0, 1]))
        return {"IE_eV": best[0], "rms_as": best[1], "corr": best[2], "fixed_window_eV": [lo, hi],
                "constant_residual_null_rms_as": const, "beats_constant_null": bool(best[1] < 0.5 * const)}
    rob["fixed_window_2_19eV_theory"] = fixed_window(Ed_t, avg_t, 2, 19)
    rob["fixed_window_12_27eV_experiment"] = fixed_window(Ed_e, avg_e, 12, 27)
    # (v) extended scan below the prescribed 12 eV floor (is 12.06 a boundary artefact?)
    ext = {}
    for lo in (2.0, 6.0, 10.0):
        rows, ib = offset_scan(Ed_e, avg_e, "x", IE_grid=np.arange(lo, 22.5001, 0.01))
        ext["floor_%.1f" % lo] = {"best_IE_eV": float(rows[ib, 0]), "rms_as": float(rows[ib, 2]),
                                  "corr": float(rows[ib, 4]), "n_points": int(rows[ib, 1])}
    rob["extended_scan_experiment"] = ext
    # (vi) per-angle best match (experiment): can ANY single angle reproduce the curve?
    per_ang = {}
    for k in range(0, 181, 5):
        b = None
        for IE in np.arange(12.0, 22.5001, 0.02):
            ph = Eaa + IE
            m = (ph >= Ed_e[0]) & (ph <= Ed_e[-1])
            if m.sum() < 30:
                continue
            p = np.interp(ph[m], Ed_e, dl_e[:, k])
            r = np.sqrt(np.mean((p - aa[m]) ** 2))
            if b is None or r < b[1]:
                b = (float(IE), float(r), float(np.corrcoef(p, aa[m])[0, 1]))
        per_ang[str(k)] = {"IE_eV": b[0], "rms_as": b[1], "corr": b[2]}
    rr = np.array([[v["rms_as"], v["corr"]] for v in per_ang.values()])
    rob["per_angle_best_match_experiment"] = {
        "angles_scanned": sorted(int(k) for k in per_ang),
        "min_rms_as": float(rr[:, 0].min()), "median_rms_as": float(np.median(rr[:, 0])),
        "max_rms_as": float(rr[:, 0].max()), "max_corr": float(rr[:, 1].max()),
        "angles_rms_lt_10as": [int(k) for k, v in per_ang.items() if v["rms_as"] < 10.0],
    }
    out["robustness"] = rob

    def verdict(panel_key):
        best = out[panel_key]["best"]
        alt = out[panel_key]["alternatives"]["offset_ranges_just_overlap"]
        own = aa.std()
        return ("best-fit IE %.2f eV: rms %.2f as (corr %.3f) vs curve own scale (std) %.2f as; "
                "range-overlap-fixed IE %.2f eV gives rms %.2f as (corr %.3f); offset 0 gives rms %.2f as; "
                "agreement within own scale: %s"
                % (best["IE_eV"], best["rms_as"], best["corr"], own,
                   alt["IE_eV"], alt["rms_as"], alt["corr"],
                   out[panel_key]["alternatives"]["offset_0"]["rms_as"],
                   "yes" if best["rms_as"] < own else "no"))
    out["verdict_experiment"] = verdict("experiment")
    out["verdict_theory"] = verdict("theory")
    log(json.dumps(out, indent=2, default=str))
    return out


try_section("e_offset_scan", part_e)

# ===========================================================================
# (f) shape comparison of the two standalone curves
# ===========================================================================
def part_f():
    Waa = np.interp(Eaa, Ewig, W)
    d = Waa - aa
    i = int(np.argmax(np.abs(d)))
    out = {
        "corr_wigner_vs_angle_averaged_on_aa_grid": float(np.corrcoef(Waa, aa)[0, 1]),
        "corr_on_full_wigner_grid_overlap": float(np.corrcoef(
            W[(Ewig >= Eaa[0]) & (Ewig <= Eaa[-1])],
            np.interp(Ewig[(Ewig >= Eaa[0]) & (Ewig <= Eaa[-1])], Eaa, aa))[0, 1]),
        "median_abs_diff_as": float(np.median(np.abs(d))),
        "max_abs_diff_as": float(np.abs(d).max()),
        "max_abs_diff_eKE_eV": float(Eaa[i]),
        "sign_of_diff_at_max": "wigner_above" if d[i] > 0 else "angle_averaged_above",
        "band_corr": {
            "eKE_0.9_5": float(np.corrcoef(Waa[Eaa < 5], aa[Eaa < 5])[0, 1]),
            "eKE_5_10": float(np.corrcoef(Waa[(Eaa >= 5) & (Eaa < 10)], aa[(Eaa >= 5) & (Eaa < 10)])[0, 1]),
            "eKE_10_20": float(np.corrcoef(Waa[(Eaa >= 10) & (Eaa < 20)], aa[(Eaa >= 10) & (Eaa < 20)])[0, 1]),
            "eKE_20_27.9": float(np.corrcoef(Waa[Eaa >= 20], aa[Eaa >= 20])[0, 1]),
        },
        "where_diverge_most": ("largest gap %.1f as at eKE %.1f eV (%s); median gap %.1f as"
                               % (np.abs(d).max(), Eaa[i], "wigner above" if d[i] > 0 else "angle-averaged above",
                                  np.median(np.abs(d)))),
        "note": "both curves are compressed onto the 200-point angle_averaged eKE grid (0.9-27.899 eV).",
    }
    log(json.dumps(out, indent=2))
    return out


try_section("f_shape_comparison", part_f)

# ===========================================================================
# (g) aligned-curve CSV
# ===========================================================================
def part_g():
    g = Ewig.copy()  # 0.5..30.0 step 0.1 (296 pts)
    aa_on = np.where((g >= Eaa[0]) & (g <= Eaa[-1]), np.interp(g, Eaa, aa), np.nan)
    # best offsets from section e (falls back to scanned defaults)
    se = RESULT["sections"].get("e_offset_scan") or {}
    IE_e = se.get("experiment", {}).get("best", {}).get("IE_eV", 12.06)
    IE_t = se.get("theory", {}).get("best", {}).get("IE_eV", 21.86)
    avg_e = dl_e.mean(axis=1)
    avg_t = dl_t.mean(axis=1)

    def panel_on(Ed, panel, IE):
        ph = g + IE
        good = (ph >= Ed[0]) & (ph <= Ed[-1])
        p = np.full(g.size, np.nan)
        p[good] = np.interp(ph[good], Ed, panel)
        return p

    g_fd = np.interp(Ewig, Eeig, np.gradient(Seig, Eeig))
    csS = CubicSpline(Eeig, Seig)
    df = pd.DataFrame({
        "eKE_eV": g,
        "eigenphase_sum_rad_interp": np.interp(g, Eeig, Seig),
        "wigner_delay_as": W,
        "wigner_pred_factor1_as": HBAR * g_fd,
        "wigner_pred_factor2_as": 2 * HBAR * g_fd,
        "wigner_pred_factor1_spline_as": HBAR * csS(g, 1),
        "angle_averaged_delay_as": aa_on,
        "angle_averaged_vs_panel_exp_IE%.2f_eV" % IE_e: panel_on(Ed_e, avg_e, IE_e),
        "angle_averaged_vs_panel_theory_IE%.2f_eV" % IE_t: panel_on(Ed_t, avg_t, IE_t),
        "wigner_deriv_fd_as_per_eV": g_fd,
        "wigner_deriv_spline_as_per_eV": csS(g, 1),
    })
    df.to_csv(CSV_PATH, index=False)
    log("wrote %s (%d rows, %d cols)" % (CSV_PATH, len(df), df.shape[1]))
    log(df.head(8).to_string(index=False))
    return {"csv_path": CSV_PATH, "rows": int(len(df)), "columns": list(df.columns),
            "IE_experiment_eV": IE_e, "IE_theory_eV": IE_t}


try_section("g_aligned_csv", part_g)

# ===========================================================================
# figure
# ===========================================================================
def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (1) experiment: delay vs phase derivative at two angles
    a1 = ax[0, 0]
    h = 0.1
    fd = HBAR * np.diff(ph_e, axis=0) / h
    for ang, col in ((30, "C0"), (90, "C1"), (150, "C2")):
        a1.plot(Ed_e, dl_e[:, ang], "-", color=col, lw=1.2, label="delay %d deg" % ang)
        a1.plot(Ed_e, fd[:, ang], "--", color=col, lw=1.0, alpha=0.8,
                label=r"$\hbar\,d\phi/dE$ %d deg" % ang)
    a1.set_xlabel("photon energy (eV)")
    a1.set_ylabel("delay (as)")
    a1.set_title("(a) experiment panel: delay vs 2-pt $\\hbar\\,d\\phi/dE$")
    a1.legend(fontsize=7, ncol=2)
    a1.set_ylim(-300, 300)

    # (2) residual distributions of derivative schemes
    a2 = ax[0, 1]
    Emid = 0.5 * (E_e[:-1] + E_e[1:])
    cs_ph = CubicSpline(E_e, ph_e, axis=0)
    bins = np.linspace(-2.5, 2.5, 120)
    a2.hist(np.clip((dl_e - fd).ravel(), -2.5, 2.5), bins=bins, histtype="step", lw=1.5,
            label="2pt FD, med %.3f as" % np.median(np.abs(dl_e - fd)))
    q5 = HBAR * locpoly_deriv(ph_e, E_e, Emid, 2, 2)
    a2.hist(np.clip((dl_e - q5).ravel(), -2.5, 2.5), bins=bins, histtype="step", lw=1.5,
            label="local quad win5, med %.3f as" % np.median(np.abs(dl_e - q5)))
    sp = HBAR * cs_ph(Emid, 1)
    a2.hist(np.clip((dl_e - sp).ravel(), -2.5, 2.5), bins=bins, histtype="step", lw=1.5,
            label="cubic spline, med %.3f as" % np.median(np.abs(dl_e - sp)))
    a2.set_yscale("log")
    a2.set_xlabel("delay - prediction (as), clipped to +/-2.5")
    a2.set_ylabel("count")
    a2.set_title("(b) experiment residuals by derivative scheme")
    a2.legend(fontsize=8)

    # (3) eigenphase -> Wigner delay factor
    a3 = ax[1, 0]
    g_fd = np.interp(Ewig, Eeig, np.gradient(Seig, Eeig))
    csS = CubicSpline(Eeig, Seig)
    a3.plot(Ewig, W, "k-", lw=1.6, label="0.5_Wigner_delay")
    a3.plot(Ewig, HBAR * g_fd, "C0--", lw=1.2, label=r"$1\times\hbar\,dS/dE$ (FD)")
    a3.plot(Ewig, HBAR * csS(Ewig, 1), "C2:", lw=1.2, label=r"$1\times\hbar\,dS/dE$ (spline)")
    a3.plot(Ewig, 2 * HBAR * g_fd, "C3-.", lw=1.0, alpha=0.8, label=r"$2\times\hbar\,dS/dE$ (FD)")
    a3.set_xlabel("eKE (eV)")
    a3.set_ylabel("delay (as)")
    a3.set_title("(d) eigenphase-sum factor test")
    a3.legend(fontsize=8)
    a3.set_ylim(-120, 300)

    # (4) angle_averaged curve vs best-offset panel interpolation
    a4 = ax[1, 1]
    a4.plot(Eaa, aa, "k-", lw=1.6, label="angle_averaged_delay")
    se = RESULT["sections"].get("e_offset_scan") or {}
    if se:
        IE_e = se["experiment"]["best"]["IE_eV"]
        IE_t = se["theory"]["best"]["IE_eV"]
        pe = np.interp(Eaa + IE_e, Ed_e, dl_e.mean(axis=1))
        pt = np.interp(Eaa + IE_t, Ed_t, dl_t.mean(axis=1))
        a4.plot(Eaa, pe, "C3--", lw=1.1,
                label="exp panel mean, IE=%.2f eV (rms %.1f as)" % (IE_e, se["experiment"]["best"]["rms_as"]))
        a4.plot(Eaa, pt, "C0-.", lw=1.1,
                label="theory panel mean, IE=%.2f eV (rms %.1f as)" % (IE_t, se["theory"]["best"]["rms_as"]))
    a4.set_xlabel("eKE (eV)")
    a4.set_ylabel("delay (as)")
    a4.set_title("(e) is angle_averaged_delay a panel angular average?")
    a4.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130)
    log("wrote %s" % PNG_PATH)
    return PNG_PATH


try:
    make_figure()
except Exception:
    tb = traceback.format_exc()
    log("figure FAILED\n" + tb)
    RESULT["failures"].append({"section": "figure", "error": tb.strip().splitlines()[-1]})

# ---------------------------------------------------------------------------
RESULT["artifact_paths"] = {"script": os.path.join(OUT_AN, "p3_phase_delay_curves.py"),
                            "json": JSON_PATH, "log": LOG_PATH, "figure": PNG_PATH,
                            "csv": CSV_PATH}
with open(JSON_PATH, "w") as f:
    json.dump(RESULT, f, indent=2, default=str)
log("\nwrote %s" % JSON_PATH)
log("FAILURES: %d" % len(RESULT["failures"]))
for fl in RESULT["failures"]:
    log("  - %s: %s" % (fl["section"], fl["error"]))
_log_fh.flush()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_log_fh.close()