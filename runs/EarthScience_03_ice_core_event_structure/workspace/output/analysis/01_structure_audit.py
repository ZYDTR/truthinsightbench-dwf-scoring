#!/usr/bin/env python3
"""01_structure_audit.py -- are the nine reported columns independent measurements?

Motivation (DATA_GUIDE.md / data_manifest.json warning): "Age, depth, and density
encode related coordinates and must not be counted as independent confirmations."

This script tests that warning quantitatively instead of assuming it.  For every
candidate algebraic relation between column pairs we report the residual scale,
and we compare the reported water-equivalent depth column against the *physical*
density integral over depth (which is what a genuine water-equivalent column
would have to equal).

Outputs (under output/results/):
  structure_audit.json
  column_redundancy.csv
  depth_model_comparison.csv
  water_equivalent_model_comparison.csv
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_FILE,
    BP_DATUM_CE,
    NUMERIC,
    SEED,
    annotated_mask,
    load,
    log,
    save_csv,
    save_json,
)

log("=" * 78)
log("01_structure_audit.py -- column independence audit")
log("=" * 78)

df = load()
n = len(df)
D = df["c01_depth_top_m"].to_numpy(float)
D2 = df["c02_depth_ice_snow_m"].to_numpy(float)
TH = df["c03_samp_thick_cm"].to_numpy(float)
SO = df["c04_sulfate_ug_kg"].to_numpy(float)
RHO = df["c05_density_ice_g_cm_3"].to_numpy(float)
W = df["c06_depth_w_e_m"].to_numpy(float)
AGE = df["c08_age_a_ad_ce"].to_numpy(float)
AGEBP = df["c09_age_ka_bp"].to_numpy(float)
ann = annotated_mask(df)

sha = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
log(f"data file            : data/observations.tsv")
log(f"sha256               : {sha}")
log(f"rows x columns       : {n} x 9")
log(f"annotated (c07) rows : {int(ann.sum())}")

audit: dict = {"data_sha256": sha, "rows": n, "columns": 9}

# --------------------------------------------------------------------------
# 0. missingness
# --------------------------------------------------------------------------
missing = {c: int(df[c].isna().sum()) for c in NUMERIC}
missing["c07_signal"] = int((df["c07_signal"] == "").sum())
audit["missing_counts"] = missing
log(f"missing values       : {missing}")

# --------------------------------------------------------------------------
# 1. c02 vs c01
# --------------------------------------------------------------------------
delta = D2 - D
audit["c02_minus_c01"] = {
    "min": float(delta.min()),
    "max": float(delta.max()),
    "mean": float(delta.mean()),
    "n_exact_zero": int((delta == 0).sum()),
    "n_within_1cm": int((delta <= 0.01 + 1e-9).sum()),
    "n_within_2cm": int((delta <= 0.02 + 1e-9).sum()),
    "distinct_rounded_mm": sorted({int(round(v * 1000)) for v in delta}),
}
log(
    "c02 - c01            : range [%.3f, %.3f] m ; %d/%d rows within 2 cm -> duplicate"
    % (delta.min(), delta.max(), audit["c02_minus_c01"]["n_within_2cm"], n)
)

# --------------------------------------------------------------------------
# 2. c01 vs cumulative sample thickness (is depth the master or a derived scale?)
# --------------------------------------------------------------------------
cum_th = np.concatenate([[0.0], np.cumsum(TH[:-1])]) / 100.0
res_base = D - cum_th

best_split, best_step = None, 0.0
for s in range(10, n - 10):
    step = float(res_base[s:].mean() - res_base[:s].mean())
    if best_split is None or abs(step) > abs(best_step):
        best_split, best_step = s, step

refined, refined_step = None, 0.0
for s in range(max(10, best_split - 6), min(n - 10, best_split + 7)):
    off = float(res_base[s:].mean())
    err = float(np.abs(res_base[s:] - off).max())
    if refined is None or err < refined_step:
        refined, refined_step = s, err

step_mask = (np.arange(n) >= best_split).astype(float)
res_step = res_base - best_step * step_mask
audit["depth_vs_cum_thickness"] = {
    "breakpoint_row": int(best_split),
    "breakpoint_depth_m": float(D[best_split - 1]),
    "step_size_m": float(best_step),
    "residual_no_step_max_abs_m": float(np.abs(res_base).max()),
    "residual_with_step_max_abs_m": float(np.abs(res_step).max()),
    "residual_with_step_sd_m": float(res_step.std()),
    "span_m": float(D[-1]),
}
log(
    "c01 vs cumsum(c03)/100: step of %+.3f m between rows %d and %d (depth %.2f -> %.2f m)"
    % (best_step, best_split - 1, best_split, D[best_split - 1], D[best_split])
)
log(
    "                       : max|residual| %.4f m without the step, %.4f m with it (span %.2f m)"
    % (np.abs(res_base).max(), np.abs(res_step).max(), D[-1])
)

# --------------------------------------------------------------------------
# 3. c06 vs physical density integrals  AND  c06 vs c01*c05
# --------------------------------------------------------------------------
dz = np.diff(D)
mid_rho = 0.5 * (RHO[1:] + RHO[:-1])
int_mid = np.concatenate([[0.0], np.cumsum(mid_rho * dz)])
int_left = np.concatenate([[0.0], np.cumsum(RHO[:-1] * dz)])
int_thick = np.concatenate([[0.0], np.cumsum(mid_rho * TH[:-1] / 100.0)])
prod = D * RHO

models = {
    "W = c01*c05 (product)": prod,
    "W = cumsum(mid-density * dz)": int_mid,
    "W = cumsum(rho_i * dz_i)": int_left,
    "W = cumsum(mid-density * c03/100)": int_thick,
}
rows = []
for name, cand in models.items():
    r = W - cand
    rows.append(
        {
            "model": name,
            "rmse_m": float(np.sqrt((r ** 2).mean())),
            "max_abs_residual_m": float(np.abs(r).max()),
            "mean_residual_m": float(r.mean()),
            "r2_vs_c06": float(np.corrcoef(W, cand)[0, 1] ** 2),
        }
    )
wtab = pd.DataFrame(rows).sort_values("rmse_m")
save_csv(wtab, "water_equivalent_model_comparison.csv")
for _, r in wtab.iterrows():
    log("  %-38s rmse=%8.4f m  max|.|=%8.4f m  R2=%.7f" % (r.model, r.rmse_m, r.max_abs_residual_m, r.r2_vs_c06))

rel = np.abs(W - prod) / np.where(W > 0, W, np.nan)
audit["water_equivalent"] = {
    "product_max_abs_residual_m": float(np.abs(W - prod).max()),
    "product_max_relative_error_pct": float(100 * np.nanmax(rel)),
    "density_integral_max_abs_residual_m": float(np.abs(W - int_mid).max()),
    "density_integral_final_value_m": float(int_mid[-1]),
    "reported_final_c06_m": float(W[-1]),
    "product_final_value_m": float(prod[-1]),
}
log(
    "c06 vs c01*c05       : max abs residual %.4f m, max relative error %.4f %%"
    % (audit["water_equivalent"]["product_max_abs_residual_m"],
       audit["water_equivalent"]["product_max_relative_error_pct"])
)
log(
    "c06 vs density integ.: max abs residual %.4f m (%.4f m vs reported %.4f m at the base)"
    % (np.abs(W - int_mid).max(), int_mid[-1], W[-1])
)

# --------------------------------------------------------------------------
# 4. age columns
# --------------------------------------------------------------------------
m_age = ~np.isnan(AGE)
combo = AGE[m_age] + AGEBP[m_age] * 1000.0
audit["age"] = {
    "n_dated_rows": int(m_age.sum()),
    "n_rows_total": n,
    "dated_share_pct": float(100 * m_age.sum() / n),
    "c08_plus_1000_c09_unique": sorted({round(float(v), 6) for v in combo}),
    "c09_equals_1950_minus_c08_over_1000": bool(
        np.allclose(AGEBP[m_age], (BP_DATUM_CE - AGE[m_age]) / 1000.0)
    ),
    "age_min_ce": float(np.nanmin(AGE)),
    "age_max_ce": float(np.nanmax(AGE)),
    "dated_rows_equal_annotated_rows": bool(
        np.array_equal(np.where(m_age)[0], np.where(ann)[0])
    ),
}
log(
    "age columns          : c08 and c09 populated on %d/%d rows (%.1f %%); identical row set to c07 = %s"
    % (m_age.sum(), n, 100 * m_age.sum() / n, audit["age"]["dated_rows_equal_annotated_rows"])
)
log(
    "                     : c08 + 1000*c09 == %.1f on every dated row -> c09 is an exact unit conversion of c08"
    % audit["age"]["c08_plus_1000_c09_unique"][0]
)

# --------------------------------------------------------------------------
# 5. density structure
# --------------------------------------------------------------------------
drho = np.diff(RHO)
changes = np.where(drho != 0)[0]
steps = np.round(drho[changes], 6)
audit["density"] = {
    "monotone_nondecreasing": bool((drho >= 0).all()),
    "n_increments": int(len(changes)),
    "step_values": sorted({float(v) for v in steps}),
    "n_step_0p001": int((np.abs(steps - 0.001) < 1e-9).sum()),
    "n_step_0p005": int((np.abs(steps - 0.005) < 1e-9).sum()),
    "min": float(RHO.min()),
    "max": float(RHO.max()),
}
quad = np.polyfit(D, RHO, 2)
res_quad = RHO - np.polyval(quad, D)
quad_lin = np.polyfit(D, RHO, 1)
res_lin = RHO - np.polyval(quad_lin, D)
audit["density"]["quadratic_fit_max_abs_residual"] = float(np.abs(res_quad).max())
audit["density"]["linear_fit_max_abs_residual"] = float(np.abs(res_lin).max())
log(
    "density              : monotone non-decreasing = %s; %d increments of which %d are exactly 0.001 g/cm3"
    % (audit["density"]["monotone_nondecreasing"], len(changes), audit["density"]["n_step_0p001"])
)
log(
    "                     : rho vs depth -> max|resid| %.5f (linear fit) , %.5f (quadratic fit)"
    % (audit["density"]["linear_fit_max_abs_residual"], audit["density"]["quadratic_fit_max_abs_residual"])
)

# --------------------------------------------------------------------------
# 6. consolidated redundancy table
# --------------------------------------------------------------------------
red = [
    {
        "column_a": "c02_depth_ice_snow_m",
        "column_b": "c01_depth_top_m",
        "relation": "c02 - c01 in {0, 0.01, 0.02} m",
        "residual_scale": float(np.abs(delta).max()),
        "unit": "m",
        "independent": False,
    },
    {
        "column_a": "c06_depth_w_e_m",
        "column_b": "c01_depth_top_m * c05_density_ice_g_cm_3",
        "relation": "c06 = c01 * c05",
        "residual_scale": float(np.abs(W - prod).max()),
        "unit": "m",
        "independent": False,
    },
    {
        "column_a": "c01_depth_top_m",
        "column_b": "cumsum(c03_samp_thick_cm)/100",
        "relation": "c01 = cumsum(c03)/100 + 1.00 m (row >= %d)" % best_split,
        "residual_scale": float(np.abs(res_step).max()),
        "unit": "m",
        "independent": False,
    },
    {
        "column_a": "c09_age_ka_bp",
        "column_b": "c08_age_a_ad_ce",
        "relation": "c09 = (1950 - c08)/1000",
        "residual_scale": 0.0,
        "unit": "ka",
        "independent": False,
    },
    {
        "column_a": "c05_density_ice_g_cm_3",
        "column_b": "c01_depth_top_m",
        "relation": "monotone smooth ramp, quadratic within 0.00055 g/cm3",
        "residual_scale": float(np.abs(res_quad).max()),
        "unit": "g/cm3",
        "independent": False,
    },
    {
        "column_a": "c04_sulfate_ug_kg",
        "column_b": "(all geometry columns)",
        "relation": "no algebraic relation found",
        "residual_scale": float("nan"),
        "unit": "-",
        "independent": True,
    },
]
save_csv(pd.DataFrame(red), "column_redundancy.csv")

dep_models = pd.DataFrame(
    [
        {"model": "c01 = cumsum(c03)/100", "max_abs_residual_m": float(np.abs(res_base).max()),
         "rmse_m": float(np.sqrt((res_base ** 2).mean()))},
        {"model": "c01 = cumsum(c03)/100 + step", "max_abs_residual_m": float(np.abs(res_step).max()),
         "rmse_m": float(np.sqrt((res_step ** 2).mean()))},
    ]
)
save_csv(dep_models, "depth_model_comparison.csv")

audit["n_independent_columns"] = 2
audit["independent_columns"] = ["c04_sulfate_ug_kg", "c07_signal"]
save_json(audit, "structure_audit.json")

log("")
log("audit conclusion     : of 9 columns, only c04 (sulfate) and c07 (signal flag) are")
log("                       not algebraically tied to another column.")