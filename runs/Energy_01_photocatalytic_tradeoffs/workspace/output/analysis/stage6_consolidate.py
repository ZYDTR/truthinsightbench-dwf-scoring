"""Stage 6 -- consolidate the executed evidence into a registration summary,
a flat evidence table, and the native artifact registry used by the report.

This stage reads ONLY the JSON artifacts written by stages 1-5 plus the frozen
workspace data; it introduces no new numbers of its own beyond assembling them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

log = C.Logger(C.LOGS / "stage6_consolidate.log")


def load(name: str) -> dict:
    return json.loads((C.RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    inv = load("inventory.json")
    kin = load("timecourse_kinetics.json")
    spec = load("spectra_analysis.json")
    trans = load("transient_analysis.json")
    loadcm = load("loading_crossmodal.json")
    rob = load("robustness.json")

    log("=" * 78)
    log("STAGE 6  evidence consolidation and artifact registry")
    log("=" * 78)

    # ---------------- flat per-unit evidence table ------------------------
    rows = []
    for fid, v in kin["kinetics"]["per_formulation"].items():
        rows.append({
            "analysis_unit": fid, "modality": "conversion_timecourses",
            "n_points": len(v["times_h"]),
            "final_product_4h_umol": v["final_product_umol_at_4h"],
            "mean_rate_umol_h": v["mean_rate_umol_h"],
            "best_aicc_model": v["best_by_aicc"],
            "aicc_delta_runner_up": sorted(v["aicc_delta_vs_best"].values())[1],
            "power_law_exponent_b": v["models"]["power_law"]["params"]["exponent_b"],
            "segment_rate_ratio_last_over_first": v["segment_rate_ratio_last_over_first"],
        })
    for fid, v in trans["trace_metrics"]["per_formulation"].items():
        late = v["window_statistics"]["delay_ge_30.0ps"]
        rows.append({
            "analysis_unit": fid, "modality": "ultrafast_transient_traces",
            "n_points": v["n_points"],
            "peak_delay_ps": v["peak_delay_ps"], "peak_value": v["peak_value"],
            "late_ge30ps_mean": late["mean"],
            "late_ge30ps_ci_low": late["block_bootstrap"]["ci_low"],
            "late_ge30ps_ci_high": late["block_bootstrap"]["ci_high"],
            "late_ge30ps_fraction_positive": late["fraction_positive"],
            "best_aicc_model": trans["decay_models"]["per_formulation"][fid]["ranking_by_aicc"][0],
        })
    for role in spec["descriptors"]["absorption_grid_nm"] and ["absorption", "action_response"]:
        pass
    rows.append({
        "analysis_unit": "absorption_profile", "modality": "optical_action_spectra",
        "n_points": inv["checks"]["absorption_points"],
        "peak_nm": spec["descriptors"]["absorption_global_max"]["nm"],
        "peak_value": spec["descriptors"]["absorption_global_max"]["value"],
        "half_max_red_edge_nm": spec["descriptors"]["absorption_half_max_crossing_nm_400_800"],
    })
    rows.append({
        "analysis_unit": "action_profile", "modality": "optical_action_spectra",
        "n_points": inv["checks"]["action_points"],
        "half_max_red_edge_nm": spec["descriptors"]["action_half_max_crossing_nm"],
        "absorption_limited_rmse": spec["models"]["models"]["M1_action_proportional_to_A"]["rmse"],
        "best_aicc_model": spec["models"]["ranking_by_aicc"][0],
    })
    for i, (L, R) in enumerate(zip(loadcm["loading_analysis"]["loading"],
                                   loadcm["loading_analysis"]["rate"])):
        rows.append({
            "analysis_unit": f"loading_{L:g}pct", "modality": "cofactor_loading_response",
            "n_points": 1, "loading_percent": L, "rate_umol_h": R,
        })
    flat = pd.DataFrame(rows)
    flat.to_csv(C.RESULTS / "evidence_table.csv", index=False)
    log(f"\nwrote results/evidence_table.csv ({len(flat)} rows)")

    # ---------------- per-curve shared-shape identifiability ----------------
    # The raw-RSS baseline comparison in stage 1 is dominated by formulation_1
    # because the curves differ in magnitude by up to ~55x, so it cannot say
    # whether one shared shape competes with a formulation-specific shape.
    # This recomputes the comparison WITHIN each curve, in that curve's own
    # units, which is the only scale on which the two hypotheses are comparable.
    tc = C.load_timecourses()
    piv = tc.pivot(index="illumination_time_h", columns="formulation_id",
                   values="gaseous_product_umol")
    t = piv.index.values.astype(float)
    n5 = len(t)
    ref = piv["formulation_1"].values

    def _aicc5(rss: float, k: int):
        kk = k + 1
        aic = n5 * np.log(max(rss, 1e-300) / n5) + 2 * kk
        return float(aic + (2 * kk * (kk + 1)) / (n5 - kk - 1)) if n5 - kk - 1 > 0 else None

    shape_rows = []
    for fid in piv.columns:
        y = piv[fid].values.astype(float)
        scale = float((ref @ y) / (ref @ ref))
        rss_shared = float(((y - scale * ref) ** 2).sum())
        lin = np.polyfit(t, y, 1)
        rss_line = float(((y - np.polyval(lin, t)) ** 2).sum())
        pw = C.fit_kinetic_models(t, y)["power_law"]
        shape_rows.append({
            "formulation_id": fid,
            "scale_vs_formulation_1_shape": scale,
            "rss_shared_formulation_1_shape": rss_shared,
            "aicc_shared_shape_k1": _aicc5(rss_shared, 1),
            "rss_own_straight_line": rss_line,
            "aicc_own_line_k2": _aicc5(rss_line, 2),
            "rss_own_power_law": pw["rss"],
            "aicc_own_power_law_k2": _aicc5(pw["rss"], 2),
            "power_law_exponent_b": pw["params"]["exponent_b"],
            "own_power_law_beats_shared_shape_by_aicc": bool(
                _aicc5(pw["rss"], 2) is not None
                and _aicc5(rss_shared, 1) is not None
                and _aicc5(pw["rss"], 2) < _aicc5(rss_shared, 1)),
        })
    shape_pd = pd.DataFrame(shape_rows)
    shape_pd.to_csv(C.RESULTS / "shared_shape_check.csv", index=False)
    log("\nper-curve shared-shape vs own-shape comparison (within-curve units):")
    log(shape_pd[["formulation_id", "power_law_exponent_b",
                  "aicc_shared_shape_k1", "aicc_own_power_law_k2",
                  "own_power_law_beats_shared_shape_by_aicc"]].to_string(index=False))
    log("NOTE: at n = 5 the AICc small-sample correction is large (6 for k=2, 24 for k=3 "
        "in the k_total convention), so a 1-parameter shared shape can out-score a "
        "2-parameter own-shape fit even when the own shape tracks the curve better. "
        "RSS within the same curve is therefore reported alongside AICc.")

    # ---------------- headline numbers per finding ------------------------
    f1 = {
        "power_law_exponent_b": {fid: v["models"]["power_law"]["params"]["exponent_b"]
                                 for fid, v in kin["kinetics"]["per_formulation"].items()},
        "segment_rate_ratio": {fid: v["segment_rate_ratio_last_over_first"]
                               for fid, v in kin["kinetics"]["per_formulation"].items()},
        "final_yield_umol": {fid: v["final_product_umol_at_4h"]
                             for fid, v in kin["kinetics"]["per_formulation"].items()},
        "aicc_winner": {fid: v["best_by_aicc"]
                        for fid, v in kin["kinetics"]["per_formulation"].items()},
        "yield_span_max_over_min": (max(v["final_product_umol_at_4h"] for v in kin["kinetics"]["per_formulation"].values())
                                    / min(v["final_product_umol_at_4h"] for v in kin["kinetics"]["per_formulation"].values())),
        "baselines": kin["null_baselines"],
        "loo_timepoint_stability": {fid: v["winner_stability"]
                                    for fid, v in loadcm["timecourse_holdout"]["leave_one_timepoint_out"].items()},
        "loo_curve_winners": {k: v["winner"] for k, v in loadcm["timecourse_holdout"]["leave_one_curve_out"].items()},
    }
    hold = spec["window_holdout"]["holdouts"]
    f2 = {
        "model_ranking": spec["models"]["ranking_by_aicc"],
        "aicc_deltas": spec["models"]["aicc_delta_vs_best"],
        "m1_rmse": spec["models"]["models"]["M1_action_proportional_to_A"]["rmse"],
        "window_holdout_fit_400_500": hold["fit_400_500_predict_550_650"],
        "long_span_pairs": [p for p in rob["R1_action_spectrum"]["window_sensitivity_pairs"]
                            if p["probe_nm"] >= 600],
        "short_span_pairs": [p for p in rob["R1_action_spectrum"]["window_sensitivity_pairs"]
                             if p["probe_nm"] <= 500],
        "red_end_supported": rob["R1_action_spectrum"]["red_end_claim_supported"],
        "blue_end_supported": rob["R1_action_spectrum"]["blue_end_claim_supported"],
        "saturating_null": rob["R1_action_spectrum"]["saturating_absorption_null"],
        "efficiency_ratio": spec["ratio"]["q_relative"],
    }
    c = trans["cross_trace_contrast"]
    f3 = {
        "f1_late_mean": c["formulation_1_mean"], "f1_late_ci": c["formulation_1_ci"],
        "f1_fraction_positive": c["formulation_1_fraction_positive"],
        "f2_late_mean": c["formulation_2_mean"], "f2_late_ci": c["formulation_2_ci"],
        "f2_fraction_positive": c["formulation_2_fraction_positive"],
        "difference": c["difference_mean"], "difference_ci": c["difference_ci"],
        "fisher_p": c["fisher_exact_positivity"]["p"],
        "model_winners": {fid: v["ranking_by_aicc"][0]
                          for fid, v in trans["decay_models"]["per_formulation"].items()},
        "asymptotic_offsets": {fid: trans["decay_models"]["per_formulation"][fid]["models"]["rise_then_decay"]["params"]["c"]
                               for fid in trans["decay_models"]["per_formulation"]},
        "threshold_sign_stability": {fid: v["sign_stable_across_thresholds"]
                                     for fid, v in trans["artifact_robustness"].items()},
        "shared_shape_null": rob["R2_shared_shape_null"],
        "holdout": trans["window_holdout"]["per_formulation"],
        "significance": trans["significance"],
    }
    la = loadcm["loading_analysis"]
    f4 = {
        "loading": la["loading"], "rate": la["rate"],
        "peak_loading_percent": la["peak_loading_percent"],
        "rate_ratio_8pct_to_0.2pct": la["rate_ratio_8pct_to_0.2pct"],
        "monotone_test": la["monotone_test"],
        "peak_shape": la["peak_shape"],
        "leave_one_out": la["leave_one_condition_out"],
        "inhibitor_fit": rob["R3_loading_optimum"]["inhibitor_fit"],
        "bootstrap_peak": rob["R3_loading_optimum"]["bootstrap_peak"],
        "plateau_alt": rob["R3_loading_optimum"]["alternative_plateau_description"],
        "model_ranking": la["ranking_by_aicc"],
    }
    f5 = {
        "shared_ids": loadcm["cross_modal"]["shared_ids"],
        "n_shared": loadcm["cross_modal"]["association"]["n_shared_formulations"],
        "reason": loadcm["cross_modal"]["association"]["reason"],
        "ordinal_contrast": loadcm["cross_modal"]["association"]["ordinal_contrast"],
        "coverage": loadcm["cross_modal"]["coverage"],
        "analysis_units": inv["analysis_units"],
    }
    summary = {
        "task_id": C.MANIFEST["task_id"], "t0": C.MANIFEST["t0"],
        "analysis_unit": C.MANIFEST["analysis_unit"],
        "inventory_checks": inv["checks"],
        "finding_1_timecourse_kinetics": f1,
        "finding_2_action_spectrum": f2,
        "finding_3_transient_sign_contrast": f3,
        "finding_4_loading_optimum": f4,
        "finding_5_crossmodal_boundary": f5,
    }
    C.json_dump(C.RESULTS / "registration_summary.json", summary)

    log("\n--- headline numbers ---")
    log(f"F1 power-law exponent b: {f1['power_law_exponent_b']}")
    log(f"F1 segment-rate ratio   : {f1['segment_rate_ratio']}")
    log(f"F1 final yields (umol)  : {f1['final_yield_umol']}  span x{f1['yield_span_max_over_min']:.1f}")
    log(f"F1 AICc winners         : {f1['aicc_winner']}")
    log(f"F2 model ranking        : {f2['model_ranking']}")
    log(f"F2 red-end supported    : {f2['red_end_supported']}   blue-end supported: {f2['blue_end_supported']}")
    log(f"F2 650 nm holdout       : observed {f2['window_holdout_fit_400_500']['held_out_observed'][-1]} vs "
        f"predicted {f2['window_holdout_fit_400_500']['held_out_predicted_absorption_limited'][-1]}")
    log(f"F3 f1 late {f3['f1_late_mean']:+.4f} {f3['f1_late_ci']}, f2 late {f3['f2_late_mean']:+.4f} {f3['f2_late_ci']}")
    log(f"F3 difference {f3['difference']:+.4f} CI {f3['difference_ci']}")
    log(f"F4 peak loading {f4['peak_loading_percent']}%; monotone exact p = "
        f"{f4['monotone_test']['exact_permutation_p_two_sided']}")
    log(f"F5 shared formulations {f5['shared_ids']}; orders agree = "
        f"{f5['ordinal_contrast']['orders_agree']}")

    # ---------------- native artifact registry ---------------------------
    registry_files = [
        ("F001", "analysis/common.py", "shared loaders, kinetic model bank, metrics, bootstrap"),
        ("F002", "analysis/stage1_inventory_timecourse.py", "inventory + time-course kinetic model comparison"),
        ("F003", "analysis/stage2_spectra.py", "action-vs-absorption model comparison and window holdout"),
        ("F004", "analysis/stage3_transients.py", "rise/decay model comparison, windowed stats, cross-trace contrast"),
        ("F005", "analysis/stage4_loading_crossmodal.py", "loading models, cross-modal synthesis, curve holdouts"),
        ("F006", "analysis/stage5_robustness.py", "unit-free action ratios, shared-shape null, loading bootstrap"),
        ("F007", "analysis/stage6_consolidate.py", "evidence consolidation and registry assembly"),
        ("F010", "results/inventory.json", "schema, ranges, integrity, analysis-unit counts"),
        ("F011", "results/timecourse_kinetics.json", "per-formulation kinetic fits and null baselines"),
        ("F012", "results/spectra_analysis.json", "action/absorption model comparison and holdouts"),
        ("F013", "results/transient_analysis.json", "transient decay models, window stats, contrast"),
        ("F014", "results/loading_crossmodal.json", "loading response and cross-modal coverage"),
        ("F015", "results/robustness.json", "R1/R2/R3 robustness battery"),
        ("F016", "results/registration_summary.json", "consolidated headline numbers per finding"),
        ("F017", "results/evidence_table.csv", "flat per-analysis-unit evidence table"),
        ("F020", "figures/fig1_timecourses.png", "product-evolution curves"),
        ("F021", "figures/fig2_spectra.png", "absorption, action, and efficiency ratio"),
        ("F022", "figures/fig3_transients.png", "transient traces and rise/decay fits"),
        ("F023", "figures/fig4_loading.png", "cofactor-loading response"),
        ("F024", "figures/fig5_crossmodal.png", "cross-modal contrast for shared formulations"),
        # ---- independent verification and alternative-explanation battery ----
        ("F030", "verification/recomputed_kinetics.json", "V1 independent recomputation of the kinetic fits"),
        ("F031", "verification/final_result.json", "V2 independent recomputation of action vs absorption"),
        ("F032", "verification/transient_verification.json", "V3 independent recomputation of the transient late window"),
        ("F033", "verification/cofactor_loading_recomputed.json", "V4 independent recomputation of the loading scan"),
        ("F034", "verification/audit_results.json", "V5 independent integrity and consistency audit of all four files"),
        ("F035", "verification/transient_artifact_test.json", "A1 transient sign-contrast artifact tests (H1 offset / H2 drift / H3 decay)"),
        ("F036", "verification/h1h2h3_optics_test.json", "A2 action-spectrum alternative explanations incl. the transmittance bound"),
        ("F037", "verification/cofactor_shape_identifiability.json", "A3 loading-shape identifiability and AICc degeneracy at n = 4"),
        ("F038", "verification/mechanism_fits_results.json", "A4 kinetic-shape identifiability and 8 h falsification predictions"),
        ("F039", "verification/cofactor_shape_identifiability_agentB.json", "A3 second independent loading-shape run"),
        # ---- post-hoc audit of the reported findings ----
        ("F040", "analysis/stage7_audit_fixes.py", "stage 7: persisted audit-requested checks"),
        ("F041", "results/audit_fixes.json", "normalisation conventions, transmittance bound grid, loading leave-one-out restatement"),
        ("F042", "analysis/workflow_audit_findings.js", "native workflow auditing each finding against its artifacts"),
        ("F043", "audit/audit_finding1_check.py", "auditor scratch check for Finding 1"),
        ("F044", "audit/audit_finding2.py", "auditor scratch check for Finding 2"),
        ("F045", "audit/audit_finding4_check.py", "auditor scratch check for Finding 4"),
    ]
    registry = {
        "schema": "truthinsightbench-native-artifact-registry",
        "task_id": C.MANIFEST["task_id"],
        "note": ("Native artifact identifiers used by output/Result.md so that every "
                 "registered finding resolves to real executed analysis source and "
                 "machine-readable result evidence."),
        "files": [],
    }
    for ident, rel, desc in registry_files:
        path = C.OUT / rel
        if not path.exists():
            raise RuntimeError(f"registry target missing: {rel}")
        registry["files"].append({
            "id": ident, "relative_path": rel, "description": desc,
            "exists": True, "bytes": path.stat().st_size,
        })
    C.json_dump(C.OUT / "file_registry.json", registry)
    log(f"\nwrote file_registry.json with {len(registry['files'])} resolved artifacts")
    log("\nstage 6 complete")
    log.close()


if __name__ == "__main__":
    main()