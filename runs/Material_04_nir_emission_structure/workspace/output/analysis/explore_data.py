#!/usr/bin/env python3
"""Profile the full dataset and compute per-spectrum peak metrics."""
import os
import sys
import numpy as np
import pandas as pd

PROJ = "/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project"
DATA = os.path.join(PROJ, "data")
OUT_ANALYSIS = os.path.join(PROJ, "output", "analysis")
OUT_EVIDENCE = os.path.join(PROJ, "output", "evidence")
OUT_FIGURES = os.path.join(PROJ, "output", "figures")

AGG = os.path.join(DATA, "aggregate_state_emission_spectra.tsv")
DIL = os.path.join(DATA, "dilute_state_emission_spectra.tsv")
LON = os.path.join(DATA, "long_wavelength_response.tsv")

os.makedirs(OUT_ANALYSIS, exist_ok=True)
os.makedirs(OUT_EVIDENCE, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

print("=== DIRS CREATED ===")
print(OUT_ANALYSIS)
print(OUT_EVIDENCE)
print(OUT_FIGURES)
print()

# ---------- Load ----------
agg = pd.read_csv(AGG, sep="\t")
dil = pd.read_csv(DIL, sep="\t")
lon = pd.read_csv(LON, sep="\t")

# ---------- Inventory ----------
def describe(name, df, has_state):
    print(f"--- {name} ---")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"unique design_series ({len(df['design_series'].unique())}): {sorted(df['design_series'].unique().tolist())}")
    if has_state:
        print(f"unique aggregate_state_id ({len(df['aggregate_state_id'].unique())}): {sorted(df['aggregate_state_id'].unique().tolist())}")
    if "wavelength_nm" in df.columns:
        print(f"wavelength_nm: min={df['wavelength_nm'].min()} max={df['wavelength_nm'].max()} nunique={df['wavelength_nm'].nunique()}")
    else:
        print("wavelength_nm: (column not present)")
    print(f"rows per (series,state):")
    if has_state:
        counts = df.groupby(["design_series", "aggregate_state_id"]).size()
        for (s, st), n in counts.items():
            print(f"  {s},{st}: {n}")
    else:
        counts = df.groupby("design_series").size()
        for s, n in counts.items():
            print(f"  {s}: {n}")
    miss = int(df.isna().sum().sum())
    print(f"total missingness count: {miss}")
    print()

describe("AGGREGATE: " + os.path.basename(AGG), agg, True)
describe("DILUTE: " + os.path.basename(DIL), dil, False)
describe("LONG RESPONSE: " + os.path.basename(LON), lon, False)

# ---------- Peak / FWHM helpers ----------
_trapz = getattr(np, "trapezoid", None) or np.trapz

def fwhm_nm(wl, ie):
    """Full width at half max, linear interpolation between wavelength points."""
    imax = float(ie.max())
    if imax <= 0:
        return float("nan")
    half = imax / 2.0
    peak_idx = int(np.argmax(ie))
    below = ie < half
    if not below.any() or below.all():
        return float("nan")
    left_b = below[:peak_idx]
    right_b = below[peak_idx + 1:]
    if not left_b.any() or not right_b.any():
        return float("nan")
    li = int(np.nonzero(left_b)[0][-1])
    ri = peak_idx + 1 + int(np.nonzero(right_b)[0][0])
    dl = wl[li + 1] - wl[li]
    if dl == 0:
        return float("nan")
    frac_l = (half - ie[li]) / (ie[li + 1] - ie[li])
    x_left = wl[li] + frac_l * dl
    dr = wl[ri + 1] - wl[ri]
    if dr == 0:
        return float("nan")
    frac_r = (ie[ri] - half) / (ie[ri] - ie[ri + 1])
    x_right = wl[ri] + frac_r * dr
    return float(x_right - x_left)

def spectrum_row(g, extra_cols=None):
    g = g.sort_values("wavelength_nm")
    wl = g["wavelength_nm"].to_numpy(dtype=float)
    ie = g["relative_emission_au"].to_numpy(dtype=float)
    row = {
        "peak_wavelength_nm": float(wl[np.argmax(ie)]),
        "max_intensity": float(ie.max()),
        "fwhm_nm": fwhm_nm(wl, ie),
        "integrated_intensity": float(_trapz(ie, wl)),
    }
    if extra_cols:
        for c in extra_cols:
            row[c] = g[c].to_numpy(dtype=float)
    return row

# ---------- Aggregate per (series, state) ----------
agg_rows = []
for (series, state) in sorted(agg.groupby(["design_series", "aggregate_state_id"]).groups.keys()):
    g = agg[(agg["design_series"] == series) & (agg["aggregate_state_id"] == state)]
    r = spectrum_row(g)
    r["design_series"] = series
    r["aggregate_state_id"] = state
    r["composition_fraction"] = float(g["composition_fraction"].iloc[0])
    in_low = (g["wavelength_nm"] >= 900.0) & (g["wavelength_nm"] <= 1100.0)
    in_high = (g["wavelength_nm"] >= 1300.0) & (g["wavelength_nm"] <= 1600.0)
    mean_low = float(g.loc[in_low, "relative_emission_au"].mean()) if in_low.any() else float("nan")
    mean_high = float(g.loc[in_high, "relative_emission_au"].mean()) if in_high.any() else float("nan")
    r["band_ratio"] = float("nan" if (np.isnan(mean_low) or np.isnan(mean_high) or mean_high == 0) else mean_low / mean_high)
    agg_rows.append(r)
agg_sum = pd.DataFrame(agg_rows)
agg_sum.to_csv(os.path.join(OUT_EVIDENCE, "spectral_summary.tsv"), sep="\t", index=False)
print("=== AGGREGATE SPECTRAL SUMMARY (saved to spectral_summary.tsv) ===")
print(agg_sum.to_string(index=False))
print()

# ---------- Dilute per series ----------
dil_rows = []
for series in sorted(dil["design_series"].unique()):
    g = dil[dil["design_series"] == series]
    r = spectrum_row(g)
    r["design_series"] = series
    dil_rows.append(r)
dil_sum = pd.DataFrame(dil_rows)
dil_sum.to_csv(os.path.join(OUT_EVIDENCE, "dilute_summary.tsv"), sep="\t", index=False)
print("=== DILUTE SPECTRAL SUMMARY (saved to dilute_summary.tsv) ===")
print(dil_sum.to_string(index=False))
print()

# ---------- Long wavelength response ----------
print("=== LONG WAVELENGTH RESPONSE (per design_series) ===")
for series, g in lon.groupby("design_series", sort=True):
    resp = g["long_wavelength_response_au"].to_numpy(dtype=float)
    comp = g["composition_fraction"].to_numpy(dtype=float)
    print(f"{series}: response min={resp.min():.2f} max={resp.max():.2f} mean={resp.mean():.2f} | composition min={comp.min():.4f} max={comp.max():.4f}")
print()

# ---------- Matplotlib check ----------
try:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt
    print("matplotlib available:", matplotlib.__version__)
    series_list = sorted(agg["design_series"].unique())
    n = len(series_list)
    fig, axes = plt.subplots(nrows=n, ncols=1, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, series in zip(axes, series_list):
        for state in agg[agg["design_series"] == series]["aggregate_state_id"].unique():
            g = agg[(agg["design_series"] == series) & (agg["aggregate_state_id"] == state)].sort_values("wavelength_nm")
            ax.plot(g["wavelength_nm"], g["relative_emission_au"], label=f"{state}")
        ax.set_xlabel("wavelength_nm")
        ax.set_ylabel("relative_emission_au")
        ax.set_title(series)
        ax.legend()
    plt.suptitle("Aggregate emission spectra (one subplot per series)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGURES, "aggregate_spectra.png"))
    print(f"saved figure to {os.path.join(OUT_FIGURES, 'aggregate_spectra.png')}")
    plt.close(fig)
except ImportError as e:
    err_text = f"ModuleNotFoundError: {e}"
    print("ERROR: matplotlib unavailable, skipping figure:", err_text, file=sys.stderr)
except Exception as e:
    err_text = type(e).__name__ + ": " + str(e)
    print("ERROR plotting:", err_text, file=sys.stderr)
    sys.exit(0)

print("DONE")
