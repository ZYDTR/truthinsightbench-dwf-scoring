#!/usr/bin/env python3
"""06_build_registry.py -- build the native finding-to-evidence registry.

Emits ``output/file_registry.json``, which maps short artifact IDs (F001, F002,
...) to real files inside ``output/``.  The authoritative report references
those IDs next to each registered finding, so every claim resolves to concrete
analysis source and machine-readable result evidence.

It also re-verifies that the artifacts the report cites actually exist and are
non-empty, which is what makes the registry trustworthy rather than decorative.

Run this last, after every other script has produced its artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT, PROJECT, RESULTS, log, save_json  # noqa: E402

log("=" * 78)
log("06_build_registry.py -- native evidence registry")
log("=" * 78)

# id -> (relative path under output/, role, description)
REGISTRY = [
    # --- executed analysis source -------------------------------------------
    ("F001", "analysis/common.py", "code", "Shared loader for data/observations.tsv; fixes the 1950 CE datum recovered from c08+c09."),
    ("F002", "analysis/01_structure_audit.py", "code", "Column-independence audit: duplicate/dependency tests and the depth, water-equivalent and age model comparisons."),
    ("F003", "analysis/02_event_detection.py", "code", "144-variant detector grid, annotated-horizon excess ranks, permutation enrichment test, block leave-one-out."),
    ("F004", "analysis/03_long_record_relationships.py", "code", "Moving-block bootstrap of the sulfate trend, age-depth model comparison and leave-one-out, event-rate stability."),
    ("F005", "analysis/04_event_classes_and_age_coordinate.py", "code", "Chronology coordinate competition with out-of-sample validation; label-class contrasts and spatial clustering."),
    ("F006", "analysis/05_robustness.py", "code", "Adversarial checks: integer-cm depth arithmetic, quadrature alternatives, block-length sensitivity, local-maximum calibration, monotone-transform competition."),
    ("F007", "analysis/06_build_registry.py", "code", "Builds and re-verifies this registry."),
    # --- machine-readable results -------------------------------------------
    ("F010", "results/structure_audit.json", "result", "Missingness, c02/c01 identity, the 1.008 m depth step, c06=c01*c05, age-column conversion, density grid."),
    ("F011", "results/column_redundancy.csv", "result", "Per-column-pair dependency table with residual scales and an independence flag."),
    ("F012", "results/water_equivalent_model_comparison.csv", "result", "RMSE / max residual / R2 for the product model versus four density-integral quadratures."),
    ("F013", "results/depth_model_comparison.csv", "result", "c01 vs cumulative c03 with and without the inserted step."),
    ("F014", "results/detector_grid.csv", "result", "All 144 (background, window, threshold) variants with detected counts, recall of the 15 horizons and false positives."),
    ("F015", "results/annotated_event_excess.csv", "result", "The 15 annotated horizons with background, excess, global rank and local-maximum status."),
    ("F016", "results/event_detection_summary.json", "result", "Detector-grid spread, local-maximum ceiling, enrichment permutation p-value."),
    ("F017", "results/block_leave_one_out_detector.csv", "result", "Recall after dropping each contiguous block at 5, 10 and 20 blocks."),
    ("F018", "results/trend_bootstrap.csv", "result", "Sulfate-vs-geometry correlations with moving-block bootstrap intervals and the i.i.d. contrast."),
    ("F019", "results/long_record_summary.json", "result", "Autocorrelation, effective sample size, shallow/deep contrast and age-fit summary."),
    ("F020", "results/age_depth_model_comparison.csv", "result", "Age-depth fits for four depth-like coordinates."),
    ("F021", "results/age_depth_leave_one_out.csv", "result", "Leave-one-out fits over the 15 dated horizons."),
    ("F022", "results/event_rate_stability.csv", "result", "Exact binomial split tests for event rate in depth and water-equivalent depth."),
    ("F023", "results/event_spacing.csv", "result", "Gap statistics and coefficients of variation against the Poisson expectation."),
    ("F024", "results/age_coordinate_competition.csv", "result", "Linear and quadratic fits per coordinate."),
    ("F025", "results/age_coordinate_loo.csv", "result", "Held-out prediction error per dated horizon for c01 and c06."),
    ("F026", "results/event_class_comparison.csv", "result", "Named vs unidentified label contrasts on three amplitude metrics."),
    ("F027", "results/event_clustering.csv", "result", "Nearest-neighbour gap statistics against a homogeneous Poisson null."),
    ("F028", "results/class_and_age_summary.json", "result", "Condensed summary of the class and coordinate analyses."),
    ("F029", "results/robustness_integer_depth.csv", "result", "Distribution of the integer-centimetre deviation between depth increments and sample thickness."),
    ("F030", "results/robustness_breakpoint_grid.csv", "result", "Every candidate breakpoint/offset pair for the depth cascade."),
    ("F031", "results/quadrature_comparison.csv", "result", "Final values of all candidate water-equivalent constructions."),
    ("F032", "results/trend_block_length_sensitivity.csv", "result", "Trend intervals at block lengths 25-400 samples."),
    ("F033", "results/trend_leave_one_block_out.csv", "result", "Trend correlation after dropping each of 20 contiguous blocks."),
    ("F034", "results/localmax_null_calibration.csv", "result", "Local-maximum count for the annotated set versus random subsets."),
    ("F035", "results/monotone_transform_competition.csv", "result", "In-sample and out-of-sample age fit quality for six candidate transforms."),
    ("F036", "results/robustness_summary.json", "result", "Machine-readable digest of every robustness check."),
    ("F037", "results/run_log.txt", "result", "Verbatim stdout of every analysis script, in execution order."),
    # --- independent adversarial verification (separate agents, own code) ----
    ("F050", "verification/figures/verify_draft1_attack.json", "result", "Independent column-dependency attack: exact duplicate/dependency search, breakpoint models, reversed derivation, density-vs-depth determinism."),
    ("F051", "verification/figures/v10_draft1_independent.json", "result", "Second independent redundancy audit; establishes that equal c01 always implies equal c05."),
    ("F052", "verification/figures/f2_wed_verification.json", "result", "Adversarial test of the water-equivalent gap: six quadratures, an ice-equivalent variant and residual-scale comparison."),
    ("F053", "verification/figures/f2_wed_final_table.json", "result", "Final water-equivalent candidate table with gaps expressed against both the reported value and the core mean."),
    ("F054", "verification/figures/verify_draft3_sulfate_trend.json", "result", "Independent serial-correlation-aware trend tests: Bartlett/Geyer effective sample size, AR(1) GLS, Newey-West, and minimum detectable slope."),
    ("F055", "verification/figures/indep_finding4_summary.json", "result", "Independent 720-variant detector grid confirming the recall ceiling and the single non-local-maximum horizon."),
    ("F056", "verification/figures/indep_detector_grid.csv", "result", "The independent 720-variant detector grid in full."),
    ("F057", "verification/figures/f5_summary.json", "result", "Independent label-class contrasts with exact Mann-Whitney p-values, AUCs, confidence intervals and power analysis."),
    ("F058", "verification/figures/ver6_report.txt", "result", "Independent chronology report: step attribution, per-horizon leave-one-out errors, density-as-cubic mechanism, and the c01^1.25 counterexample."),
    ("F059", "verification/figures/ver6_subset_robustness.txt", "result", "Subset robustness of the chronology: all 15, the 9 unidentified and the 5 named horizons, plus block bootstrap of the MAE difference."),
    ("F060", "verification/figures/g7_indep_results.json", "result", "Third independent redundancy audit including the segmented step fit and the forward-reconstruction fractions."),
    # --- figures -------------------------------------------------------------
    ("F070", "figures/fig01_record_overview.png", "figure", "Four-panel record overview: sulfate, density, water-equivalent depth and sample thickness against depth, with the annotated horizons and the depth step marked."),
    ("F071", "figures/fig02_event_definition_sensitivity.png", "figure", "Annotated horizons by label class, the detector-count distribution and the recall-versus-count relation."),
    ("F072", "figures/fig03_chronology_coordinates.png", "figure", "Age against c01 and c06 with fit lines, leave-one-out predictions and residual curvature."),
    ("F073", "figures/fig05_water_equivalent_gap.png", "figure", "Reported c06 against the density integral and their residual structure."),
    ("F074", "figures/fig06_class_contrast_power.png", "figure", "Power curve showing how large a class difference the 5-vs-9 comparison could have detected."),
    ("F075", "figures/fig07_independent_redundancy.png", "figure", "Independent redundancy audit figure."),
    ("F076", "figures/fig08_age_fit_competition.png", "figure", "Age fit competition across candidate depth coordinates."),
    ("F077", "figures/fig09_age_loo_errors.png", "figure", "Per-horizon leave-one-out prediction errors for the competing coordinates."),
]


def main() -> int:
    files = []
    missing = []
    for aid, rel, role, desc in REGISTRY:
        path = OUTPUT / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
            continue
        files.append(
            {
                "id": aid,
                "relative_path": rel,
                "role": role,
                "description": desc,
                "bytes": path.stat().st_size,
            }
        )
    if missing:
        log("MISSING ARTIFACTS (registry not written):")
        for m in missing:
            log("  - %s" % m)
        return 1

    payload = {
        "schema": "truthinsightbench-native-evidence-registry",
        "task_id": "EarthScience_03_ice_core_event_structure",
        "note": (
            "Artifact IDs F001.. are referenced from the registered findings in "
            "output/Result.md so that each finding resolves to executed analysis "
            "source and machine-readable result evidence."
        ),
        "files": files,
    }
    path = OUTPUT / "file_registry.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log("registry written: output/file_registry.json (%d artifacts)" % len(files))
    log("")
    log("code artifacts      : %d" % sum(1 for f in files if f["role"] == "code"))
    log("result artifacts    : %d" % sum(1 for f in files if f["role"] == "result"))
    log("figure artifacts    : %d" % sum(1 for f in files if f["role"] == "figure"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())