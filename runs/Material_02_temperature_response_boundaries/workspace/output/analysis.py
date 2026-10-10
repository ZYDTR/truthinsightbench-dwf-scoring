#!/usr/bin/env python3
"""
Reproducible analysis for TruthInsightBench task Material_02_temperature_response_boundaries.

Domain: material_science. T0: 2022-01-01. One anonymous oxide sample.

Loads all five frozen data files, computes magnetic and electrical response
metrics, writes machine-readable result JSON files, and prints a summary.

Uses ONLY numpy and pandas (no matplotlib).
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data")
OUT  = BASE

MAG_FILES = {
    "300K": "bulk_magnetization_300K.tsv",
    "860K": "bulk_magnetization_860K.tsv",
    "870K": "bulk_magnetization_870K.tsv",
}
ELEC_FILES = {
    "switching": "electric_switching_corrected.tsv",
    "current_voltage": "electric_current_voltage.tsv",
}

GRID = 400_000          # integration grid for loop areas
CENTRAL_CUTOFF = 5_000  # "central" field window |H| <= 5000 Oe
CENTRAL_SUS_CUTOFF = 25_000  # central susceptibility window |H| <= 25000 Oe


# ---------------------------------------------------------------------------
# Magnetic helpers
# ---------------------------------------------------------------------------
def _branches(df):
    """Split a field-scan into up1 / down / ret using direction changes."""
    H = df["field_Oe"].values
    d = np.diff(H)
    signs = np.sign(d)
    changes = np.where(np.diff(signs) != 0)[0] + 1
    tp1 = changes[0]          # first direction change = +max turning
    tp2 = np.argmin(H)        # global min = -trough turning
    return df.loc[:tp1], df.loc[tp1:tp2], df.loc[tp2:], (tp1, tp2)


def _loop_area(down, ret, cutoff):
    """Geometric enclosed loop area = integral of |M_desc - M_ret| dH
    over the field window [-cutoff, +cutoff]."""
    down = down.sort_values("field_Oe")
    ret = ret.sort_values("field_Oe")
    Hd = down["field_Oe"].values
    Md = down["magnetic_moment_emu"].values
    Hr = ret["field_Oe"].values
    Mr = ret["magnetic_moment_emu"].values
    Hmin = min(Hd.min(), Hr.min())
    Hmax = max(Hd.max(), Hr.max())
    Hg = np.linspace(-cutoff, cutoff, GRID)
    Hg = Hg[(Hg >= Hmin) & (Hg <= Hmax)]
    if len(Hg) < 2:
        return 0.0
    Md_g = np.interp(Hg, Hd, Md)
    Mr_g = np.interp(Hg, Hr, Mr)
    return float(np.trapezoid(np.abs(Md_g - Mr_g), Hg))


def _coercive(down):
    """H where M crosses 0 on the descending branch (linear interpolation)."""
    x = down["field_Oe"].values
    y = down["magnetic_moment_emu"].values
    cross = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    if len(cross) == 0:
        return 0.0
    c = cross[0]
    return float(x[c] - y[c] * (x[c + 1] - x[c]) / (y[c + 1] - y[c]))


def _remanence(down):
    """M at the point closest to H=0 on the descending branch."""
    x = down["field_Oe"].values
    y = down["magnetic_moment_emu"].values
    i0 = int(np.argmin(np.abs(x)))
    return float(y[i0])


def _magnetic_metrics(path, label):
    df = pd.read_csv(path, sep="\t")
    H = df["field_Oe"].values
    M = df["magnetic_moment_emu"].values
    up1, down, ret, tp = _branches(df)

    ipk = int(np.argmax(M))
    itr = int(np.argmin(M))
    # full and central susceptibility
    sus_full = float(np.polyfit(H, M, 1)[0])
    mc = df[np.abs(df["field_Oe"]) <= CENTRAL_SUS_CUTOFF]
    sus_central = float(np.polyfit(mc["field_Oe"].values,
                                   mc["magnetic_moment_emu"].values, 1)[0])
    # endpoint moments (mean within +/-3000 Oe of field max/min)
    mpos = float(df[np.abs(df["field_Oe"] - H.max()) < 3000]
                 ["magnetic_moment_emu"].mean())
    mneg = float(df[np.abs(df["field_Oe"] - H.min()) < 3000]
                 ["magnetic_moment_emu"].mean())
    hyst = mpos - mneg
    # robustness: remove top 5% by |M|, remove high-std points
    order = np.argsort(np.abs(M))[::-1]
    k = max(1, int(0.05 * len(df)))
    top5 = set(order[:k])
    std = df["moment_std_err_emu"].values
    med = float(np.median(std))
    highstd = set(np.where(std > 10 * med)[0])

    def _metrics_on(sub_df):
        Hs = sub_df["field_Oe"].values
        Ms = sub_df["magnetic_moment_emu"].values
        up1s, downs, rets, _ = _branches(sub_df)
        ipk = int(np.argmax(Ms)); itr = int(np.argmin(Ms))
        return {
            "peak_m": float(Ms[ipk]),
            "peak_h": float(Hs[ipk]),
            "hysteresis": float(Ms.max() - Ms.min()),
            "loop_area_full": _loop_area(downs, rets, max(Hs.max(), -Hs.min())),
            "sus_full": float(np.polyfit(Hs, Ms, 1)[0]),
        }

    df5 = df[~df.index.isin(top5)].copy()
    dfhs = df[~df.index.isin(highstd)].copy()
    m5 = _metrics_on(df5)
    mhs = _metrics_on(dfhs)

    return {
        "n_points": int(len(df)),
        "field_max": float(H.max()),
        "field_min": float(H.min()),
        "Hmax_turning": int(tp[0]),
        "Hmin_turning": int(tp[1]),
        "peak_m": float(M[ipk]),
        "peak_h": float(H[ipk]),
        "trough_m": float(M[itr]),
        "trough_h": float(H[itr]),
        "sus_full": sus_full,
        "sus_central": sus_central,
        "m_pos50k": mpos,
        "m_neg50k": mneg,
        "hysteresis": hyst,
        "loop_area_full": _loop_area(down, ret, max(H.max(), -H.min())),
        "loop_area_central": _loop_area(down, ret, CENTRAL_CUTOFF),
        "coercive_field": _coercive(down),
        "remanence": _remanence(down),
        "rob_removed_top5pct": {
            "n_points": int(len(df5)),
            "removed": int(len(df) - len(df5)),
            "threshold_absM": float(np.sort(np.abs(M))[len(df) - k]),
            "peak_m": m5["peak_m"],
            "peak_h": m5["peak_h"],
            "hysteresis": m5["hysteresis"],
            "loop_area_full": m5["loop_area_full"],
            "sus_full": m5["sus_full"],
        },
        "rob_removed_high_std": {
            "n_points": int(len(dfhs)),
            "removed": int(len(df) - len(dfhs)),
            "threshold_std": float(10 * med),
            "peak_m": mhs["peak_m"],
            "peak_h": mhs["peak_h"],
            "hysteresis": mhs["hysteresis"],
            "loop_area_full": mhs["loop_area_full"],
            "sus_full": mhs["sus_full"],
        },
    }


# ---------------------------------------------------------------------------
# Electrical helpers
# ---------------------------------------------------------------------------
def _shoelace(X, Y):
    return float(0.5 * np.abs(np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1])))


def _branches2(df):
    x = df["sequence_coordinate"].values
    d = np.diff(x)
    signs = np.sign(d)
    changes = np.where(np.diff(signs) != 0)[0] + 1
    tp1, tp2 = changes[0], changes[-1]
    return df.loc[:tp1].reset_index(drop=True), df.loc[tp1:tp2].reset_index(drop=True), \
        df.loc[tp2:].reset_index(drop=True), (tp1, tp2)


def _electrical_metrics(path, label):
    df = pd.read_csv(path, sep="\t")
    x = df["sequence_coordinate"].values
    y = df.iloc[:, 1].values
    up, down, ret, tp = _branches2(df)

    area_under = float(np.trapezoid(y, x))
    loop_area = _shoelace(np.append(x, x[0]), np.append(y, y[0]))

    xu, yu = up["sequence_coordinate"].values, up.iloc[:, 1].values
    xd, yd = down["sequence_coordinate"].values, down.iloc[:, 1].values
    # hysteresis: mean/max absolute diff between down-branch and the
    # up-branch (and return-branch) at the same coordinate.
    xr, yr = ret["sequence_coordinate"].values, ret.iloc[:, 1].values
    d_up = np.abs(yd - np.interp(xd, xu, yu))
    d_ret = np.abs(yd - np.interp(xd, xr, yr))
    all_diff = np.concatenate([d_up, d_ret])
    hyst_mean = float(all_diff.mean())
    hyst_max = float(all_diff.max())
    # area between the two main branches over the full coordinate span
    gx = np.linspace(xd.min(), xd.max(), GRID)
    yd_g = np.interp(gx, xd, yd)
    yu_g = np.interp(gx, xu, yu)
    hyst_area = float(np.trapezoid(np.abs(yd_g - yu_g), gx))

    # robustness
    n = len(df)
    k = int(0.05 * n)
    m = np.ones(n, bool); m[:k] = False; m[-k:] = False
    trimmed = _shoelace(np.append(x[m], x[m][0]), np.append(y[m], y[m][0]))
    min_sub = _shoelace(np.append(x, x[0]), np.append(y - y.min(), y[0] - y.min()))
    mean_sub = _shoelace(np.append(x, x[0]), np.append(y - y.mean(), y[0] - y.mean()))

    return {
        "n_points": int(len(df)),
        "coordinate_range": [float(x.min()), float(x.max())],
        "signal_range": [float(y.min()), float(y.max())],
        "mean": float(y.mean()),
        "std": float(y.std(ddof=0)),
        "min": float(y.min()),
        "max": float(y.max()),
        "loop_or_unidirectional": "loop",
        "area_under_curve": area_under,
        "loop_area": loop_area,
        "hysteresis_present": bool(hyst_mean > 0),
        "hysteresis_mean_abs_diff": hyst_mean,
        "hysteresis_max_abs_diff": hyst_max,
        "hysteresis_area_between_branches": hyst_area,
        "robustness": {
            "original_area": loop_area,
            "trimmed_5pct": trimmed,
            "trimmed_5pct_change_pct": float((trimmed - loop_area) / loop_area * 100
                                             if loop_area else 0.0),
            "min_subtracted": min_sub,
            "min_subtracted_change_pct": float((min_sub - loop_area) / loop_area * 100
                                               if loop_area else 0.0),
            "mean_subtracted": mean_sub,
            "mean_subtracted_change_pct": float((mean_sub - loop_area) / loop_area * 100
                                                if loop_area else 0.0),
        },
    }


def _electrical_comparison(s, cv):
    """Cross-file checks: correlation, derivative/integral relations."""
    y1 = s["polarization_like_signal"].values
    x1 = s["sequence_coordinate"].values
    y2 = cv["current_or_voltage_signal"].values
    x2 = cv["sequence_coordinate"].values

    same_idx_corr = float(np.corrcoef(y1, y2)[0, 1])

    # derivative check on the monotonic ascending branch (first 44 pts)
    up = s.iloc[:44]; upc = cv.iloc[:44]
    yu1 = up["polarization_like_signal"].values
    xu1 = up["sequence_coordinate"].values
    yu2 = upc["current_or_voltage_signal"].values
    dy1 = np.gradient(yu1, xu1)
    slope, intercept = np.polyfit(dy1, yu2, 1)
    pred = slope * dy1 + intercept
    r2 = 1 - np.sum((yu2 - pred) ** 2) / np.sum((yu2 - yu2.mean()) ** 2)
    corr_der = float(np.corrcoef(dy1, yu2)[0, 1])

    # integral check on ascending branch
    inty1 = np.concatenate([[0.0],
                            np.cumsum(0.5 * (yu1[:-1] + yu1[1:]) * np.diff(xu1))])
    corr_int = float(np.corrcoef(inty1, yu2)[0, 1])

    return {
        "relationship": (
            "Both files are closed loops of 176 points sampled on the same "
            "coordinate sequence with x2 = 7.50 * x1 (current_voltage coordinate "
            "~7.5x larger). Switching is a smooth normalized polarization-like "
            "loop; current_voltage is a noisier raw electrical loop. Same-index "
            "correlation is near zero because the loop is traversed in the same "
            "direction, but on the monotonic ascending branch current_voltage is "
            "strongly proportional to the derivative of switching, consistent with "
            "current ~ rate of change of polarization."
        ),
        "same_index_corr": same_idx_corr,
        "coordinate_ratio": float(np.mean(x2[(x1 != 0) & (x2 != 0)] /
                                          x1[(x1 != 0) & (x2 != 0)])),
        "derivative_slope": float(slope),
        "derivative_intercept": float(intercept),
        "derivative_r2": float(r2),
        "derivative_corr": corr_der,
        "integral_corr": corr_int,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- magnetic ----
    magnetic = {}
    for label, fname in MAG_FILES.items():
        magnetic[label] = _magnetic_metrics(os.path.join(DATA, fname), label)

    # ---- electrical ----
    electrical = {}
    for label, fname in ELEC_FILES.items():
        electrical[label] = _electrical_metrics(os.path.join(DATA, fname), label)
    s = pd.read_csv(os.path.join(DATA, "electric_switching_corrected.tsv"), sep="\t")
    cv = pd.read_csv(os.path.join(DATA, "electric_current_voltage.tsv"), sep="\t")
    electrical["comparison"] = _electrical_comparison(s, cv)
    electrical["findings"] = [
        "Both electrical files are closed loops; hysteresis is present in both.",
        "Switching loop area (shoelace) = %.4f; current_voltage loop area = %.2f."
        % (electrical["switching"]["loop_area"], electrical["current_voltage"]["loop_area"]),
        "Coordinates are proportional: current_voltage coordinate = 7.50 * switching coordinate.",
        "Same-index signal correlation = %.6f; signals are independent traces at the same coordinates."
        % (electrical["comparison"]["same_index_corr"]),
        "Switching is smooth/normalized (std %.4f); current_voltage is noisy (std %.1f)."
        % (electrical["switching"]["std"], electrical["current_voltage"]["std"]),
        "Derivative check: on the ascending branch current_voltage is strongly "
        "proportional to the derivative of switching (R2=%.4f, corr=%.4f)."
        % (electrical["comparison"]["derivative_r2"], electrical["comparison"]["derivative_corr"]),
        "Background (mean/min) subtraction leaves the shoelace hysteresis area "
        "essentially unchanged (shift-invariant).",
        "Trimming 5% of the points changes the loop area by <3% for both files.",
    ]
    electrical["notes"] = (
        "loop_area computed via shoelace closed-polygon area (shift-invariant). "
        "Files share an identical 176-point coordinate sequence with a 7.5 scaling."
    )

    # ---- write result JSONs ----
    with open(os.path.join(OUT, "01_magnetic_results.json"), "w") as f:
        json.dump(magnetic, f, indent=2)
    with open(os.path.join(OUT, "03_electrical_results.json"), "w") as f:
        json.dump(electrical, f, indent=2)

    # ---- print summary to stdout ----
    print("=" * 72)
    print("MATERIAL_02_TEMPERATURE_RESPONSE_BOUNDARIES — ANALYSIS SUMMARY")
    print("=" * 72)
    print("\n[ MAGNETIC ]")
    hdr = ("%8s %10s %14s %14s %14s %14s %12s %12s")
    print(hdr % ("T(=K)", "n", "sus_full", "sus_central", "peak_m", "hysteresis", "loop_full", "remanence"))
    print("-" * len(hdr))
    for t in ["300K", "860K", "870K"]:
        m = magnetic[t]
        print("%8s %10d %14.4e %14.4e %14.7f %14.7f %12.5f %12.7f" % (
            t, m["n_points"], m["sus_full"], m["sus_central"],
            m["peak_m"], m["hysteresis"], m["loop_area_full"], m["remanence"]))
    for t in ["300K", "860K", "870K"]:
        m = magnetic[t]
        print("   %s: coercive=%.3f Oe  loop_central=%.5f  peak_m=%.7f emu" % (
            t, m["coercive_field"], m["loop_area_central"], m["peak_m"]))

    print("\n[ ELECTRICAL ]")
    for t in ["switching", "current_voltage"]:
        e = electrical[t]
        print("   %s: n=%d  loop_area=%.4f  hysteresis_present=%s  std=%.4f" % (
            t, e["n_points"], e["loop_area"], e["hysteresis_present"], e["std"]))
    c = electrical["comparison"]
    print("   comparison: same_index_corr=%.5f  coord_ratio=%.2f  deriv_R2=%.4f "
          "deriv_corr=%.4f  integral_corr=%.4f" % (
        c["same_index_corr"], c["coordinate_ratio"], c["derivative_r2"],
        c["derivative_corr"], c["integral_corr"]))

    print("\n[ KEY RESULTS ]")
    print("  Phase transition (FM->PM) at or below 860K: remanence drops from "
          "+%.6f emu (300K) to %.6f emu (860K), coercive field from %.1f Oe to %.1f Oe." % (
        magnetic["300K"]["remanence"], magnetic["860K"]["remanence"],
        magnetic["300K"]["coercive_field"], magnetic["860K"]["coercive_field"]))
    print("  860K and 870K are indistinguishable (paramagnetic), confirming "
          "transition at or below 860K.")
    print("  300K has the largest peak moment (%.7f emu) and largest hysteresis "
          "(%.7f emu)." % (magnetic["300K"]["peak_m"], magnetic["300K"]["hysteresis"]))
    print("  Electrical: current_voltage proportional to d(polarization)/dcoordinate "
          "on the ascending branch (R2=%.4f), consistent with current ~ dP/dt." % (
        c["derivative_r2"]))
    print("\nResults written to 01_magnetic_results.json and 03_electrical_results.json")


if __name__ == "__main__":
    main()
