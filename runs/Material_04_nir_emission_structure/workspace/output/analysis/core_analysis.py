#!/usr/bin/env python3
"""Core quantitative relationships: composition effect, state effect, dilute vs aggregate."""
import numpy as np
import pandas as pd

BASE = "/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project"
DATA = BASE + "/data/long_wavelength_response.tsv"
SPECTRAL = BASE + "/output/evidence/spectral_summary.tsv"
DILUTE = BASE + "/output/evidence/dilute_summary.tsv"
OUT = BASE + "/output/evidence"

long_df = pd.read_csv(DATA, sep="\t")
spec_df = pd.read_csv(SPECTRAL, sep="\t")
dilute_df = pd.read_csv(DILUTE, sep="\t")

def ols_fit(x, y):
    """OLS y ~ x: return (slope, intercept, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x.mean()
    ym = y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = ym - slope * xm
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2

def _average_ranks(z):
    """Competition ranks with ties averaged (0-based)."""
    z = np.asarray(z, dtype=float)
    n = z.size
    order = np.argsort(z, kind="mergesort")
    r = np.empty(n)
    r[order] = np.arange(1, n + 1)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and z[order[j + 1]] == z[order[i]]:
            j += 1
        r[order[i:j + 1]] = r[order[i]] + (r[order[j]] - r[order[i]]) / 2.0
        i = j + 1
    return r

def spearman_rho(x, y):
    rx = _average_ranks(np.asarray(x, dtype=float))
    ry = _average_ranks(np.asarray(y, dtype=float))
    xm = rx.mean()
    ym = ry.mean()
    sxx = float(((rx - xm) ** 2).sum())
    syy = float(((ry - ym) ** 2).sum())
    sxy = float(((rx - xm) * (ry - ym)).sum())
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / np.sqrt(sxx * syy)

def sign_changes(values):
    """Count sign changes in consecutive differences, ignoring zero diffs."""
    diffs = np.asarray(values, dtype=float)[1:] - np.asarray(values, dtype=float)[:-1]
    signs = [np.sign(d) for d in diffs if d != 0]
    n = 0
    for a, b in zip(signs, signs[1:]):
        if a * b < 0:
            n += 1
    return n

rows = []
print("=" * 70)
print("1) COMPOSITION EFFECT ON LONG-WAVELENGTH RESPONSE (per design_series)")
print("=" * 70)
for series in long_df["design_series"].unique():
    sub = long_df[long_df["design_series"] == series].sort_values("composition_fraction")
    x = sub["composition_fraction"].to_numpy(float)
    y = sub["long_wavelength_response_au"].to_numpy(float)
    slope, intercept, r2 = ols_fit(x, y)
    quad = float(np.polyfit(x, y, 2)[0])
    rho = spearman_rho(x, y)
    nsc = sign_changes(y)
    row = {
        "design_series": series,
        "n_points": len(sub),
        "composition_min": round(float(x.min()), 4),
        "composition_max": round(float(x.max()), 4),
        "response_min": round(float(y.min()), 4),
        "response_max": round(float(y.max()), 4),
        "ols_slope_au_per_frac": round(slope, 6),
        "ols_intercept_au": round(intercept, 6),
        "ols_r2": round(r2, 6),
        "spearman_rho": round(rho, 6),
        "n_sign_changes_first_diffs": nsc,
        "quadratic_curvature_coeff": round(quad, 6),
    }
    rows.append(row)
    print(f"series={series} n={len(sub)} comp[{row['composition_min']},{row['composition_max']}] "
          f"resp[{row['response_min']},{row['response_max']}] "
          f"slope={row['ols_slope_au_per_frac']:.4f} intercept={row['ols_intercept_au']:.4f} "
          f"R2={row['ols_r2']:.6f} spearman={row['spearman_rho']:.6f} "
          f"sign_changes={row['n_sign_changes_first_diffs']} "
          f"quad_coeff={row['quadratic_curvature_coeff']:.4f}")

comp_eff = pd.DataFrame(rows)
comp_eff.to_csv(OUT + "/composition_effects.tsv", sep="\t", index=False)
print(f"saved {OUT}/composition_effects.tsv ({len(comp_eff)} rows)")

print()
print("=" * 70)
print("2) STATE EFFECT ON AGGREGATE SPECTRA (per design_series)")
print("=" * 70)
state_by_state_rows = []
for series in spec_df["design_series"].unique():
    sub = spec_df[spec_df["design_series"] == series].sort_values("aggregate_state_id")
    s1 = sub[sub["aggregate_state_id"] == "state_1"].iloc[0]
    s5 = sub[sub["aggregate_state_id"] == "state_5"].iloc[0]
    p1 = float(s1["peak_wavelength_nm"])
    p5 = float(s5["peak_wavelength_nm"])
    f1 = float(s1["fwhm_nm"])
    f5 = float(s5["fwhm_nm"])
    i1 = float(s1["max_intensity"])
    i5 = float(s5["max_intensity"])
    peak_shift = p5 - p1
    fwhm_change = float(np.nan if (np.isnan(f1) or np.isnan(f5)) else f5 - f1)
    intensity_change = i5 / i1
    state_by_state_rows.extend(
        [{
            "design_series": series,
            "aggregate_state_id": r[1]["aggregate_state_id"],
            "peak_wavelength_nm": r[1]["peak_wavelength_nm"],
            "fwhm_nm": r[1]["fwhm_nm"],
            "max_intensity": r[1]["max_intensity"],
        } for r in sub.iterrows()]
    )
    print(f"series={series}: peak_state1={p1} peak_state5={p5} "
          f"peak_shift={peak_shift:+.1f} nm | fwhm_state1={f1} fwhm_state5={f5} "
          f"fwhm_change={fwhm_change if not np.isnan(fwhm_change) else np.nan} nm | "
          f"intensity_state1={i1} intensity_state5={i5} intensity_ratio_5_1={intensity_change:.6f}")

state_by_state = pd.DataFrame(state_by_state_rows)
state_by_state.to_csv(OUT + "/state_by_state.tsv", sep="\t", index=False)
print(f"saved {OUT}/state_by_state.tsv ({len(state_by_state)} rows)")

st_rows = []
for series in spec_df["design_series"].unique():
    sub = spec_df[spec_df["design_series"] == series].sort_values("aggregate_state_id")
    s1 = sub[sub["aggregate_state_id"] == "state_1"].iloc[0]
    s5 = sub[sub["aggregate_state_id"] == "state_5"].iloc[0]
    peak_shift = float(s5["peak_wavelength_nm"] - s1["peak_wavelength_nm"])
    f1, f5 = s1["fwhm_nm"], s5["fwhm_nm"]
    fwhm_change = float(np.nan if (np.isnan(f1) or np.isnan(f5)) else f5 - f1)
    i1, i5 = float(s1["max_intensity"]), float(s5["max_intensity"])
    intensity_change = float(i5 / i1)
    st_rows.append({
        "design_series": series,
        "state_1_peak_wavelength_nm": s1["peak_wavelength_nm"],
        "state_5_peak_wavelength_nm": s5["peak_wavelength_nm"],
        "peak_shift_state5_minus_state1_nm": peak_shift,
        "state_1_fwhm_nm": f1,
        "state_5_fwhm_nm": f5,
        "fwhm_change_state5_minus_state1_nm": fwhm_change,
        "state_1_max_intensity": s1["max_intensity"],
        "state_5_max_intensity": s5["max_intensity"],
        "intensity_ratio_state5_over_state1": intensity_change,
    })
state_eff = pd.DataFrame(st_rows)
state_eff.to_csv(OUT + "/state_effects.tsv", sep="\t", index=False)
print(f"saved {OUT}/state_effects.tsv ({len(state_eff)} rows)")

print()
print("=" * 70)
print("3) DILUTE vs AGGREGATE PEAK WAVELENGTHS")
print("=" * 70)
dv_rows = []
for series in dilute_df["design_series"].unique():
    d = dilute_df[dilute_df["design_series"] == series].iloc[0]
    dpeak = float(d["peak_wavelength_nm"])
    sub = spec_df[spec_df["design_series"] == series].sort_values("aggregate_state_id")
    for _, r in sub.iterrows():
        apeak = float(r["peak_wavelength_nm"])
        dv_rows.append({
            "design_series": series,
            "aggregate_state_id": r["aggregate_state_id"],
            "dilute_peak_wavelength_nm": dpeak,
            "aggregate_state_peak_wavelength_nm": apeak,
            "delta_nm_dilute_minus_aggregate": round(dpeak - apeak, 4),
        })
        print(f"series={series} {r['aggregate_state_id']}: dilute={dpeak} aggregate={apeak} "
              f"delta(dilute-aggregate)={dpeak - apeak:+.1f} nm")

dv = pd.DataFrame(dv_rows)
dv.to_csv(OUT + "/dilute_vs_aggregate.tsv", sep="\t", index=False)
print(f"saved {OUT}/dilute_vs_aggregate.tsv ({len(dv)} rows)")

print()
print("=" * 70)
print("FULL SUMMARY")
print("=" * 70)
print("COMPOSITION EFFECTS (response ~ composition_fraction):")
print(comp_eff.to_string(index=False))
print()
print("STATE EFFECTS:")
print(state_eff.to_string(index=False))
print()
print("DILUTE vs AGGREGATE (all states):")
print(dv.to_string(index=False))
print()
print("DONE. All artifacts written under", OUT)
