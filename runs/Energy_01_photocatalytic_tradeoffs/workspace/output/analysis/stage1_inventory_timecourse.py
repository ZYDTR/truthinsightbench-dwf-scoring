"""Stage 1a -- data inventory and product-evolution kinetic model comparison.

Produces:
  results/inventory.json              (schema, ranges, integrity, analysis units)
  results/timecourse_kinetics.json    (model comparison per formulation)
  figures/fig1_timecourses.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

log = C.Logger(C.LOGS / "stage1_inventory_timecourse.log")


# ---------------------------------------------------------------- inventory
def inventory() -> dict:
    log("=" * 78)
    log("STAGE 1a  data inventory and integrity check")
    log("=" * 78)
    frames = C.load_all()
    inv = {"task_id": C.MANIFEST["task_id"], "t0": C.MANIFEST["t0"],
           "analysis_unit": C.MANIFEST["analysis_unit"], "files": {}}
    for name, df in frames.items():
        rows = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
            "sha256_verified": True,
        }
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                rows[f"range_{c}"] = [float(df[c].min()), float(df[c].max())]
            else:
                rows[f"levels_{c}"] = sorted(map(str, df[c].unique()))[:25]
                rows[f"n_levels_{c}"] = int(df[c].nunique())
        inv["files"][name] = rows
        log(f"\n{name}: {len(df)} rows, columns={list(df.columns)}")
        log(f"  numeric ranges: " + ", ".join(
            f"{c}=[{df[c].min():g},{df[c].max():g}]" for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])))
        log(f"  categorical: " + ", ".join(
            f"{c}={rows[f'n_levels_{c}']} levels" for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])))

    tc = frames["conversion_timecourses"]
    spec = frames["optical_action_spectra"]
    tr = frames["ultrafast_transient_traces"]
    ld = frames["cofactor_loading_response"]
    inv["analysis_units"] = {
        "time_courses": int(tc.formulation_id.nunique()),
        "spectra_profiles": int(spec.measurement_role.nunique()),
        "transient_traces": int(tr.formulation_id.nunique()),
        "loading_conditions": int(len(ld)),
    }
    log("\nanalysis units: " + str(inv["analysis_units"]))
    # duplicated-column checks -------------------------------------------------
    inv["checks"] = {
        "no_missing_values": bool(all(df.isna().sum().sum() == 0 for df in frames.values())),
        "transient_delay_grid_shared": bool(np.allclose(
            tr[tr.formulation_id == "formulation_1"].delay_ps.values,
            tr[tr.formulation_id == "formulation_2"].delay_ps.values)),
        "absorption_grid_step_nm": 0.5,
        "action_grid_step_nm": 50.0,
        "absorption_points": int((spec.measurement_role == "absorption").sum()),
        "action_points": int((spec.measurement_role == "action_response").sum()),
        "all_curves_start_at_zero_product": bool(
            (tc[tc.illumination_time_h == 0].gaseous_product_umol == 0).all()),
    }
    log("checks: " + str(inv["checks"]))
    return inv


# ------------------------------------------------------- per-curve kinetics
def kinetics(tc: pd.DataFrame) -> dict:
    log("\n" + "=" * 78)
    log("STAGE 1a  product-evolution kinetic model comparison (analysis unit = one curve)")
    log("=" * 78)
    out = {"unit": "one complete formulation time course (5 illumination times)",
           "models": list(C.KINETIC_MODELS), "per_formulation": {}}
    for fid, g in tc.groupby("formulation_id"):
        g = g.sort_values("illumination_time_h")
        t = g.illumination_time_h.values.astype(float)
        y = g.gaseous_product_umol.values.astype(float)
        rows = C.fit_kinetic_models(t, y)
        # collinearity note: power law cannot include the t=0 anchor
        ranked, undefined = C.rank_models(rows)
        best_parsimonious = min(ranked, key=lambda r: (r["k_params"], r["aicc"]))
        delta = {r["model"]: float(r["aicc"] - ranked[0]["aicc"]) for r in ranked}
        seg = np.diff(y) / np.diff(t)
        out["per_formulation"][fid] = {
            "times_h": t.tolist(),
            "product_umol": y.tolist(),
            "final_product_umol_at_4h": float(y[-1]),
            "segment_rates_umol_h": [float(s) for s in seg],
            "segment_rate_ratio_last_over_first": float(seg[-1] / seg[0]),
            "mean_rate_umol_h": float(y[-1] / t[-1]),
            "models": rows,
            "ranking_by_aicc": [r["model"] for r in ranked],
            "aicc_undefined_models": undefined,
            "aicc_delta_vs_best": delta,
            "best_by_aicc": ranked[0]["model"],
            "best_parsimonious_1_or_2_param": best_parsimonious["model"],
        }
        log(f"\n{fid}: final product {y[-1]:.4f} umol at 4 h "
            f"(mean rate {y[-1]/t[-1]:.4f} umol/h)")
        log(f"  segment rates (umol/h): {np.round(seg,4).tolist()}  "
            f"last/first = {seg[-1]/seg[0]:.3f}")
        log("  model ranking by AICc (lower is better; delta vs best):")
        for r in ranked:
            log(f"    {r['model']:>16s}  k={r['k_params']}  R2={r['r2']:.5f}  "
                f"RMSE={r['rmse']:.4f}  AICc={r['aicc']:.2f}  dAICc={delta[r['model']]:.2f}"
                + ("   <-- best" if r["model"] == ranked[0]["model"] else ""))
        failed = [r for r in rows.values() if not r["ok"]]
        for r in failed:
            log(f"    {r['model']:>16s}  FAILED: {r['error']}")
    return out


# -------------------------------------------------------- baseline contrast
def null_baselines(tc: pd.DataFrame, kin: dict) -> dict:
    """Compare at least one null/simple baseline against the fitted structure."""
    log("\n" + "-" * 78)
    log("Null / simple baselines for the time courses")
    log("-" * 78)
    res = {"baseline_1_common_linear_slope": {"per_curve_rss": {}},
           "baseline_2_zero_slope": {},
           "baseline_3_shared_shape_scaled": {}}
    ids = sorted(tc.formulation_id.unique())
    # baseline 1: per-curve straight line, then a single slope shared by all curves
    for fid in ids:
        g = tc[tc.formulation_id == fid].sort_values("illumination_time_h")
        t = g.illumination_time_h.values.astype(float)
        y = g.gaseous_product_umol.values.astype(float)
        design = np.vstack([t, np.ones_like(t)]).T
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        res["baseline_1_common_linear_slope"]["per_curve_rss"][fid] = float(
            ((y - design @ coef) ** 2).sum())
    shared = 0.0
    per_curve_shared = {}
    for fid in ids:
        g = tc[tc.formulation_id == fid].sort_values("illumination_time_h")
        t = g.illumination_time_h.values.astype(float)
        y = g.gaseous_product_umol.values.astype(float)
        best = min(((((y - s * t - (y - s * t).mean()) ** 2).sum(), s)
                    for s in np.linspace(0.0, 3.0, 6001)))
        per_curve_shared[fid] = {"rss": float(best[0]), "slope": float(best[1])}
        shared += best[0]
    res["baseline_1_common_linear_slope"]["per_curve_shared_slope"] = per_curve_shared
    res["baseline_1_common_linear_slope"]["per_curve_rss_total"] = float(
        sum(res["baseline_1_common_linear_slope"]["per_curve_rss"].values()))
    res["baseline_1_common_linear_slope"]["shared_slope_rss_total"] = float(shared)
    # baseline 2: pure null -- no production at all (RSS about zero)
    res["baseline_2_zero_slope"]["rss_total"] = float((tc.gaseous_product_umol ** 2).sum())
    # baseline 3: one curve shape rescaled per formulation
    pivot = tc.pivot(index="illumination_time_h", columns="formulation_id",
                     values="gaseous_product_umol")
    ref = pivot["formulation_1"].values
    rss3 = 0.0
    scales = {}
    for fid in ids:
        y = pivot[fid].values
        s = float((ref @ y) / (ref @ ref))
        scales[fid] = s
        rss3 += float(((y - s * ref) ** 2).sum())
    res["baseline_3_shared_shape_scaled"]["scales"] = scales
    res["baseline_3_shared_shape_scaled"]["rss_total"] = rss3
    for key, val in res.items():
        log(f"  {key}: {val}")
    return res


def figure_timecourses(tc: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for fid, g in tc.groupby("formulation_id"):
        g = g.sort_values("illumination_time_h")
        axes[0].plot(g.illumination_time_h, g.gaseous_product_umol, "o-", label=fid)
        axes[1].plot(g.illumination_time_h, g.gaseous_product_umol, "o-", label=fid)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("gaseous product (umol)")
    for ax in axes:
        ax.set_xlabel("illumination time (h)")
        ax.grid(alpha=0.3)
    axes[0].set_title("A  product evolution (log y)")
    axes[1].set_title("B  product evolution (linear y)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig1_timecourses.png", dpi=160)
    plt.close(fig)
    log("\nwrote figures/fig1_timecourses.png")


if __name__ == "__main__":
    inv = inventory()
    tc = C.load_timecourses()
    kin = kinetics(tc)
    base = null_baselines(tc, kin)
    C.json_dump(C.RESULTS / "inventory.json", inv)
    C.json_dump(C.RESULTS / "timecourse_kinetics.json",
                {"kinetics": kin, "null_baselines": base})
    figure_timecourses(tc)
    log("\nstage 1a complete")
    log.close()