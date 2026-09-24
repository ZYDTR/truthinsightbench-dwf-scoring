"""Stage 7 -- consolidate the registered findings into one machine-readable table.

This stage does not compute new science.  It reads the stage JSONs that the
executed pipeline wrote and the independently produced verification JSONs, and
emits a single `results/findings_registry.json` that maps each registered
finding to (a) the headline numbers, (b) the analysis source, (c) the result
artifact, and (d) the verification artifact that independently recomputed it.

Keeping this in code (rather than only in the report) makes the
finding-to-evidence traceability machine-checkable and keeps the report's
numbers tied to files that actually exist.

Run:  python3 analysis/stage7_consolidate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, OUTPUT, save_json  # noqa: E402

VERIFICATION = OUTPUT / "verification"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> None:
    s1 = load(RESULTS / "stage1_inventory.json")
    s2 = load(RESULTS / "stage2_boundaries.json")
    s3 = load(RESULTS / "stage3_crossfile.json")
    s4 = load(RESULTS / "stage4_robustness.json")
    s5 = load(RESULTS / "stage5_uv_mechanism.json")
    s6 = load(RESULTS / "stage6_statistical_power.json")

    v_d1 = load(VERIFICATION / "D1_metallicity_crossover.json")
    v_d2 = load(VERIFICATION / "D2_temperature_reversal.json")
    v_d3 = load(VERIFICATION / "D3_crossfile_relation.json")
    v_d4 = load(VERIFICATION / "D4_uv_boundary.json")
    g1 = load(VERIFICATION / "GAP1_calibration_circularity.json")
    g2 = load(VERIFICATION / "GAP2_mechanism.json")
    g3 = load(VERIFICATION / "GAP3_artefact_nulls.json")
    g4 = load(VERIFICATION / "GAP4_interaction_and_uncertainty.json")

    def fail(name):
        raise SystemExit(f"missing required artifact: {name}")

    for name, obj in (("stage1", s1), ("stage2", s2), ("stage3", s3),
                      ("stage4", s4), ("stage5", s5), ("stage6", s6)):
        if obj is None:
            fail(name)
    for name, obj in (("D1", v_d1), ("D2", v_d2), ("D3", v_d3), ("D4", v_d4),
                      ("GAP1", g1), ("GAP2", g2), ("GAP3", g3), ("GAP4", g4)):
        if obj is None:
            fail(name)

    # ---- D1: composition response boundary --------------------------------
    cross = {t: s2["metallicity_response"][t]["first_crossing_nm"] for t in ("5300", "5800", "6300")}
    d1 = {
        "id": "D1",
        "title": "The composition response reverses sign at a shared optical boundary",
        "analysis_unit": "one complete temperature-by-wavelength spectrum",
        "headline": {
            "crossover_wavelength_nm_by_temperature": cross,
            "slope_at_180nm_by_temperature": {
                t: s2["metallicity_response"][t]["slope_at_180nm"] for t in cross},
            "median_slope_longward_500nm_by_temperature": {
                t: s2["metallicity_response"][t]["median_slope_longward_500nm"] for t in cross},
            "median_r2_observed": {t: s6["r2_vs_permutation_null"][t]["median_r2_observed"]
                                   for t in cross},
            "median_r2_null": {t: s6["r2_vs_permutation_null"][t]["median_r2_null"]
                               for t in cross},
            "p_value_r2_vs_permutation_null": {
                t: s6["r2_vs_permutation_null"][t]["p_value_report"]
                for t in cross},
            "p_value_is_floor_limited": {
                t: s6["r2_vs_permutation_null"][t]["p_value_is_floor_limited"]
                for t in cross},
            "cross_slice_profile_correlation": s6["cross_slice_profile_reproduction"],
            "bootstrap_ci95_nm": {
                "5300": g4["level_bootstrap_metallicity_crossover"]["T5300"]["ci95_percentile_nm"],
                "5800": g4["level_bootstrap_metallicity_crossover"]["T5800"]["ci95_percentile_nm"],
                "6300": g4["level_bootstrap_metallicity_crossover"]["T6300"]["ci95_percentile_nm"],
            },
            "paired_difference_5300_minus_6300_nm": {
                "median": g4["paired_bootstrap_differences_metallicity_crossover"]
                          ["d_5300_minus_6300"]["summary"]["median"],
                "ci95": [g4["paired_bootstrap_differences_metallicity_crossover"]
                         ["d_5300_minus_6300"]["summary"]["p2_5"],
                         g4["paired_bootstrap_differences_metallicity_crossover"]
                         ["d_5300_minus_6300"]["summary"]["p97_5"]],
            },
            "interaction_F": g4["joint_model"]["local_window_slope"]["nested_F_interaction"]["F"],
            "interaction_p": g4["joint_model"]["local_window_slope"]["nested_F_interaction"]["p_value"],
            "interaction_coef_nm": g4["joint_model"]["local_window_slope"]["model_with_interaction"]
                                   ["coef"]["T_x_M_interaction_nm"],
            "interaction_ci95_hc1_nm": g4["joint_model"]["local_window_slope"]["model_with_interaction"]
                                        ["ci95_hc1"]["T_x_M_interaction_nm"],
        },
        "robustness": {
            "leave_one_temperature_out_crossover_nm": s4["R1_leave_one_temperature_out"],
            "rank_based_alternative": s4["R3_rank_based_alternative"],
            "metallicity_window": s4["R4_metallicity_window"],
            "jackknife_max_shift_nm": g3["verdicts"]["claim_i_metallicity_crossover"]
                                      ["monotone_preserving_null"]
                                      ["jackknife_max_abs_shift_nm"],
            "sub_ladder_ci_nm": g3["verdicts"]["claim_i_metallicity_crossover"]
                                ["monotone_preserving_null"]
                                ["sub_ladder_k16_q2.5_97.5_nm"],
            "uniform_5nm_resampling_max_shift_nm": g3["verdicts"]
                                                   ["claim_i_metallicity_crossover"]
                                                   ["wavelength_null_N2"]
                                                   ["max_abs_shift_nm"],
            "decimation_max_shift_nm": g3["verdicts"]["claim_i_metallicity_crossover"]
                                       ["wavelength_null_N3"]["max_abs_shift_nm"],
            "slice_separation_nm": g3["verdicts"]["claim_i_metallicity_crossover"]
                                   ["slice_separation_nm"],
            "permutation_null_verdict": (
                "the plain label-permutation null is NON-INFORMATIVE and is NOT "
                "used as a p-value: the composition climatology is rank-1 "
                "(PC1 >= 98.9% of level-space variance), so permuted labels "
                "reproduce a ~0.9-correlated slope profile about half the time. "
                "The informative checks are invariance to label rescaling, "
                "single-rung jackknife (<=0.12 nm at 5300/5800 K), random "
                "monotone 16-of-31 sub-ladders (median within 0.2 nm), and "
                "wavelength resampling"),
        },
        "counterevidence": [
            "the T=6300 K slice has 3 sign changes, not 1, so 'the' crossover is "
            "definition-dependent (first neg->pos = 409.9 nm; a persistence rule "
            "gives 434.4 nm; a local-window rule gives 421 nm)",
            "the label-permutation null produces a neg->pos crossing in 80-100% of "
            "permutations, so the crossing location alone does not discriminate",
            "flux values carry only 3 significant figures, bounding any resolvable "
            "deviation to ~0.2-0.5%",
        ],
        "falsification_test": (
            "Predict dlogF/dM at 300 nm and at 700 nm for a temperature slice not in "
            "this file (e.g. 5000 K or 7000 K). The finding requires the 300 nm slope "
            "to stay negative and the 700 nm slope to stay positive."),
        "evidence": {
            "analysis": ["output/analysis/stage2_boundaries.py",
                         "output/analysis/stage4_robustness.py",
                         "output/analysis/stage6_statistical_power.py"],
            "results": ["output/results/stage2_boundaries.json",
                        "output/results/stage4_robustness.json",
                        "output/results/stage6_statistical_power.json"],
            "verification": ["output/verification/D1_metallicity_crossover.json",
                             "output/verification/GAP3_artefact_nulls.json",
                             "output/verification/GAP4_interaction_and_uncertainty.json"],
            "figure": ["output/figures/fig1_wavelength_boundaries.png"],
        },
    }

    # ---- D2: temperature response reversal --------------------------------
    d2 = {
        "id": "D2",
        "title": "The temperature response reverses sign in the red, and the crossing moves",
        "analysis_unit": "one complete temperature-by-wavelength spectrum",
        "headline": {
            "ratio_6300_over_5300_at_probe_wavelengths": v_d2["primary_numbers"]
                                                         ["ratio_at_probes_6300_over_5300"],
            "probe_wavelengths_nm": v_d2["primary_numbers"]
                                    ["ratio_at_probes_wavelengths_used_nm"],
            "crossing_nm_by_metallicity": v_d2["primary_numbers"]
                                          ["crossing_wavelength_nm_6300_over_5300"],
            "crossing_monotonicity": v_d2["primary_numbers"]["crossing_monotonicity"],
            "long_wavelength_deficit": v_d2["primary_numbers"]["long_wavelength_deficit"],
            "crossing_span_nm": v_d2["robustness"]["uniform_ladder_Z_le_0.5"]["span_nm"],
            "adjacent_pair_crossovers_nm": s2["adjacent_temperature_crossovers"],
        },
        "robustness": {
            "decimation_shift_by_case": g3["verdicts"]["claim_ii_temperature_crossover"]
                                        ["wavelength_null_N2"]["shift_summary_nm"]
                                        ["6300_over_5300"]["decimate_shifts_by_case"],
            "r850_below_1_under_all_nulls": g3["verdicts"]["claim_ii_temperature_crossover"]
                                            ["wavelength_null_N2"]["shift_summary_nm"]
                                            ["6300_over_5300"]["r850_all_below_1_under_all_nulls"],
            "decimation_max_shift_nm": g3["verdicts"]["claim_ii_temperature_crossover"]
                                       ["wavelength_null_N2"]["shift_summary_nm"]
                                       ["6300_over_5300"]["decimate_max_abs_shift_nm"],
            "null_verdict": g3["verdicts"]["claim_ii_temperature_crossover"]["verdict"],
            "crossing_quantisation_note": (
                "many crossings land exactly on 5-nm grid nodes; jitter is of the "
                "order of the grid spacing"),
        },
        "counterevidence": [
            "the redward trend is NOT strictly monotone: backward steps occur, "
            "largest ~1.6 nm (at the lowest index), within the sampling-plus-"
            "quantisation jitter; an independent refutation therefore judged the "
            "metallicity dependence NOT separable from the quantisation noise "
            "floor (see output/verification/D2_refutation.json)",
            "the '0.86-0.88 at 850 nm' band is left at both ends by some levels "
            "(observed range 0.8559-0.8856)",
            "850 nm is the last grid point: the file cannot show whether the "
            "deficit steepens, flattens or turns over beyond it",
            "'order 100x at 180 nm' holds as a median (183x); the ladder spans "
            "85-279x and 7/35 levels fall below 100x",
            "a fitted blackbody pair cannot reproduce the reversal at all: every "
            "one of the 35 fitted Planck-pair curves stays above 1 across the "
            "whole band, so the reversal requires atmospheric structure",
        ],
        "falsification_test": (
            "Extend the wavelength axis redward of 850 nm for one temperature pair. "
            "The finding requires F(6300)/F(5300) to remain below 1 and the crossing "
            "to keep moving redward with increasing metallicity index."),
        "evidence": {
            "analysis": ["output/analysis/stage2_boundaries.py",
                         "output/analysis/stage4_robustness.py"],
            "results": ["output/results/stage2_boundaries.json",
                        "output/results/stage4_robustness.json"],
            "verification": ["output/verification/D2_temperature_reversal.json",
                             "output/verification/GAP3_artefact_nulls.json"],
            "figure": ["output/figures/fig1_wavelength_boundaries.png"],
        },
    }

    # ---- D3: cross-file photon-to-energy correspondence -------------------
    d3 = {
        "id": "D3",
        "title": "The normalized series is a photon-to-energy transform of the 5800 K slice",
        "analysis_unit": "one complete ordered response series",
        "headline": {
            "level_to_metallicity_assignment": v_d3["adjacent_metallicity_discrimination"]
                                           ["claimed_set"],
            "per_level_best_fit": v_d3["search_all_105_pairs"]["per_level_best"],
            "shared_temperature_K": 5800.0,
            "proportionality_constant": v_d3["constant_of_proportionality_T5800_band330_850"]
                                           ["c_final"],
            "constant_relative_spread": v_d3["constant_of_proportionality_T5800_band330_850"]
                                        ["relative_spread_over_5x121_points"],
            "pooled_median_rel_dev_5800": v_d3["temperature_uniqueness_pooled_single_constant"]
                                          ["5800.0"]["pooled_med_rel_dev"],
            "pooled_median_rel_dev_5300": v_d3["temperature_uniqueness_pooled_single_constant"]
                                          ["5300.0"]["pooled_med_rel_dev"],
            "pooled_median_rel_dev_6300": v_d3["temperature_uniqueness_pooled_single_constant"]
                                          ["6300.0"]["pooled_med_rel_dev"],
            "transform_discrimination_median_rel_dev": {
                "level_times_wavelength": v_d3["transform_discrimination"]
                                          ["fitted_with_claimed_M"]["level*wl"]["median"],
                "level": v_d3["transform_discrimination"]["fitted_with_claimed_M"]
                         ["level"]["median"],
                "level_over_wavelength": v_d3["transform_discrimination"]
                                         ["fitted_with_claimed_M"]["level/wl"]["median"],
            },
            "separation_vs_other_temperature_refit_C": s3["photon_to_energy_null"]
                                                       ["refit_C"]
                                                       ["separation_factor_vs_other_temperature"],
            "separation_vs_next_curve_fixed_C": s3["photon_to_energy_null"]
                                                   ["fixed_C"]["separation_factor_vs_next"],
            "separation_vs_next_curve_refit_C": s3["photon_to_energy_null"]
                                                   ["refit_C"]["separation_factor_vs_next"],
            "extrapolation_verdict": g1["verdict"]["is_global_proportionality"],
            "out_of_band_mean_rel_dev": g1["verdict"]
                                        ["fit_500_700_extrapolated_to_330_500"]
                                        ["mean_abs_rel_dev"],
            "quantization_floor_mean_rel_dev": g1["quantization_floor"]
                                               ["residual_is_explained_by_rounding_5800K_330_850"]
                                               ["expected_mean_abs_for_uniform_rounding"],
            "frac_points_within_rounding_half_step": g1["quantization_floor"]
                                                     ["residual_is_explained_by_rounding_5800K_330_850"]
                                                     ["frac_points_within_half_step"],
        },
        "robustness": {
            "registration": "the two files' wavelength axes differ (175 vs 176 points) "
                            "and are index-shifted past index 66; the match was computed "
                            "after explicit realignment",
            "calibration_not_circular": (
                "fitting the constant on 500-700 nm only, or 400-500 nm only, or "
                "750-850 nm only, reproduces the held-out bands to the same "
                "magnitude as inside the fitting window"),
            "additional_independent_checks": [
                "peak positions: the normalized levels peak at 469.9-479.9 nm, "
                "matching the peak of F/wavelength (480 nm at 5300/5800 K), not the "
                "peak of F (575-605 nm) -- an independently predicted consequence",
                "ratio consistency: level_k/level_1 reverses sign at 443-447 nm, "
                "matching the grid's own T=5800 K composition crossover at 445.8 nm, "
                "with no wavelength factor involved",
            ],
        },
        "counterevidence": [
            "the search is discrete and the winning metallicity for level 3 is "
            "nearly degenerate (-0.05 vs 0.00) at the flux quantisation floor",
            "the residual, though small, is at the 3-significant-figure rounding "
            "floor, so no finer agreement can be established from this file",
            "the correspondence is a property of the realigned data product; the "
            "raw row-index alignment does not match",
        ],
        "falsification_test": (
            "Obtain the same response series computed at a fourth effective "
            "temperature or for an unlisted metallicity index. The finding requires "
            "the same transform with only the constant changing per level."),
        "evidence": {
            "analysis": ["output/analysis/stage3_crossfile.py"],
            "results": ["output/results/stage3_crossfile.json"],
            "verification": ["output/verification/D3_crossfile_relation.json",
                             "output/verification/GAP1_calibration_circularity.json"],
            "figure": ["output/figures/fig2_crossfile_relation.png"],
        },
    }

    # ---- D4: UV breakdown is a sampling artefact ---------------------------
    d4 = {
        "id": "D4",
        "title": "The relation breaks below ~330 nm by a discrete, composition-blind factor",
        "analysis_unit": "one complete ordered response series",
        "headline": {
            "plateau_value_299_322nm": s5["exact_plateaus"]["1nm_region_299_322"]
                                       ["measured_mean"],
            "plateau_predicted": 5.0,
            "plateau_rel_error": s5["exact_plateaus"]["1nm_region_299_322"]["rel_error"],
            "narrow_value_322_327nm": s5["exact_plateaus"]["2nm_region_322_327"]
                                      ["measured_mean"],
            "narrow_predicted": 2.5,
            "optical_value_330_850nm": s5["exact_plateaus"]["optical_330_850"]
                                       ["measured_mean"],
            "best_model": s5["best_model"],
            "model_comparison_mean_abs_rel_error": s5["alternative_models"],
            "transition_alignment": s5["transition_alignment"],
            "composition_independence_rel_spread": s5["composition_independence"]
                                                   ["rel_spread_across_levels_plateau"],
            "metallicity_sensitivity_at_300nm": s5["composition_independence"]
                                                ["metallicity_sensitivity_at_300nm_log_slope"],
            "region_accuracy": s5["prediction_accuracy"],
            "functional_form_identified": g2["mechanism"]["functional_form_identified"],
            "mechanism_identified": g2["mechanism"]["mechanism_identified"],
            "free_numerator_nm": g2["free_N"]["N_hat_nm"],
            "free_numerator_combined_sd_nm": g2["free_N"]["N_hat_combined_sd_nm"],
            "free_numerator_distinguishable_from_5nm": g2["free_N"]["distinguishable_from_5nm"],
            "free_exponent_p": g2["model_comparison_uv"]["models"]["d_step_power_free_p"]["params"]["p"],
            "step_robust_mean": g2["normalisation_test"]["mean"],
            "step_robust_sd": g2["normalisation_test"]["sd"],
            "node_count_model_error_step_robust": g2["mechanism"]["literal_node_count_model"]["mean_abs_rel_error_step_robust"],
            "width_model_error_step_robust": g2["mechanism"]["width_model"]["mean_abs_rel_error_step_robust"],
        },
        "robustness": {
            "sampling_convention_variants": v_d4["robustness_to_optical_window"],
            "grid_sampling_test": v_d4["grid_sampling_step_change_nodes"],
            "axis_alignment": v_d4["axis_alignment"],
            "mechanism_test": v_d4["mechanism_test_factor_equals_5_over_local_grid_step"],
            "factor_ladder_summary": v_d4["factor_ladder_summary"],
            "registration_negative_control": v_d4["registration_negative_control"],
            "physics_test": v_d4["physics_test"],
            "registration_shift_is_not_the_cause": (
                "the plateau factor stays at 5.00 and the far-UV mean stays at 2.05 "
                "under index-shifted, index-unshifted, linear and nearest-neighbour "
                "resampling, so the shift is not what produces the deviation"),
        },
        "counterevidence": [
            "the mechanism's numerator (5 nm) equals the optical grid step, so the "
            "data cannot by itself distinguish 'factor = step ratio' from any other "
            "quantity proportional to 1/step (see the GAP2 identification test)",
            "the narrow 322-327 nm segment is defined by only 3 points",
            "the far-UV ramp is least accurate (mean relative error ~1.8%)",
        ],
        "falsification_test": (
            "Re-derive the response series from a flux grid with a uniform "
            "wavelength ladder. The artefact explanation requires the sub-330 nm "
            "deviation to vanish; a physical explanation requires it to persist."),
        "evidence": {
            "analysis": ["output/analysis/stage5_uv_mechanism.py",
                         "output/analysis/stage3_crossfile.py"],
            "results": ["output/results/stage5_uv_mechanism.json",
                        "output/results/stage3_crossfile.json"],
            "verification": ["output/verification/D4_uv_boundary.json"],
            "figure": ["output/figures/fig2_crossfile_relation.png"],
        },
    }

    registry = {
        "dataset_integrity": {
            "hashes_match_manifest": all(v["match"] for v in s1["integrity"].values()),
            "grid_complete": s1["flux_grid"]["complete_grid"],
            "rows": s1["flux_grid"]["rows"],
            "temperature_slices_K": s1["flux_grid"]["temperature_slices_K"],
            "n_metallicity_levels": len(s1["flux_grid"]["metallicity_levels"]),
            "n_wavelengths": s1["flux_grid"]["n_wavelengths"],
            "missing_or_nonpositive_flux": s1["flux_grid"]["n_nonpositive_flux"],
        },
        "analysis_unit_note": (
            "the analysis unit is one complete temperature-by-wavelength spectrum or "
            "one complete ordered response series; neighbouring wavelengths are "
            "correlated scan points, and wavelength-spacing-aware resampling "
            "confirmed the boundaries are insensitive to that correlation"),
        "findings": [d1, d2, d3, d4],
    }

    path = save_json("findings_registry.json", registry)
    print(f"wrote {path}")
    print(f"registered findings: {[f['id'] for f in registry['findings']]}")
    print(f"dataset integrity OK: {registry['dataset_integrity']['hashes_match_manifest']}")


if __name__ == "__main__":
    main()