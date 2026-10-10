# Sensor Conditions and Fusion Phenotypes in a Single-Vesicle System

- **Task:** `Life_01_dual_sensor_fusion`
- **T0:** 2024-01-01
- **Independent analysis unit:** one complete run. Time bins and the workbook *Mean/S.D.* columns within a run are NOT independent replicates; every mean/SD below is recomputed from the raw run columns with pandas (`ddof=1`).
- **Evidence boundary:** only the bundled `data/` tables are used as direct evidence. No external data, figures, or precomputed results were downloaded, substituted, or merged.

## Methods

A serial native-Workflow pipeline (`analysis/serial_workflow.js`) delegated five mutually independent data groups to sub-agents, each returning via the real `StructuredOutput` tool (journal `result` entries are the receipts). Groups A–D re-parsed the block-structured (spreadsheet-style) CSVs into run-level tidy frames and recomputed statistics; Group R re-parsed the raw CSVs again to run leave-one-run-out, alternative-definition, alternative-model, and threshold-sensitivity checks. All analysis scripts and machine-readable result artifacts live under `output/`.

## Finding 1: Early fusion (first-two-frames) declines with SensorB dose while total fusion at 5 s stays flat

**Object and scope.** Total fusion at 5 s and early fusion (fraction fusing in the first two frames) measured across independent SensorB doses in the fused-sensor panels (`source_table_01` block B; corroborated by `source_table_03`, `source_table_04`, `source_table_08`).

**Relationship / direction / boundary.** Non-monotonic-in-direction is ruled out: early fusion *monotonically declines* as SensorB concentration rises (higher `1:X` ratio = lower X = higher concentration: 1:2000 → 1:800 → 1:400 → 1:200), while total fusion is essentially flat. This is a *condition-dependent decoupling*: dose reprograms the timing, not the yield.

**Core quantitative result (5 runs per condition, ddof=1).**
| Condition | Total fusion @5 s (mean±SD) | First-two-frames fusion (mean±SD) |
|---|---|---|
| None | 88.83±1.996 | 64.21±2.584 |
| 1:2000 SensorB | 87.57±3.102 | 50.29±10.10 |
| 1:800 SensorB | 88.45±4.530 | 21.44±5.553 |
| 1:400 SensorB | 87.87±3.106 | 15.82±7.395 |
| 1:200 SensorB | 89.94±2.912 | 8.33±6.689 |

Early fusion falls from 64.2% (None) to 8.3% (1:200 SensorB), a drop of ≈55.9 pp; total fusion remains within 87.6–89.9% across every condition. The SensorA-only condition (1: 200 SensorA) shows no early-fusion suppression (65.5%), so the effect is specific to SensorB.

**Analysis + result evidence.**
- Analysis source: `output/results/parse_A.py`
- Result artifact: `output/results/fusion_dose_stats.json` (run-level `runs[]`, per-condition mean/SD, and the cross-check)
- Robustness: `output/results/robustness.py` → `output/results/robustness_results.json` (key `F1`)

**Robustness.** Leave-one-run-out: for the high-dose conditions (1:200/1:400/1:800 SensorB), every leave-one-run-out mean stays < 25% (max loo = 22.96%), so the "early fusion suppressed" claim survives dropping any single run. Alternative early window (block C, bins ≤ 0.294 s) preserves the same decreasing direction (per-condition sums: None 160.5 → 1:200 SensorB 20.8). Total fusion flatness holds (range 87.6–89.9). Cross-check of table_01 block B against the derived file: 60 values, max_abs_diff = 0.0, passed.

**Counter-evidence / limitation / alternative.** The lowest-dose point (1:2000 SensorB) has high variance (SD 10.1) and one run (44.2%) exceeds 25%, so "all runs < 25%" fails at that single concentration; the robust claim is therefore restricted to high doses. Alternative explanation (dose reduces *total* fusion) is refuted by the flat total-fusion curve.

**Falsifiable observation.** If early fusion were dose-independent, first-two-frames fusion would stay ≈64% across all SensorB doses; the observed ≈56 pp decline to 8.3% falsifies that.

## Finding 2: Competitor concentration monotonically suppresses bound target

**Object and scope.** Molecular binding of a competitor to the target complex across 0–60 uM (`source_table_05.csv`, 4 independent runs per concentration). A contrasting ligand/CPX-FusionComplex curve (`source_table_11.csv`) is reported separately to avoid conflation.

**Relationship / direction / boundary.** Monotonic-decreasing: bound target falls monotonically with competitor concentration (no adjacent increase), well described by a natural-log line (or a power law). This is the *monotonic* explanation for the competitor curve.

**Core quantitative result (4 runs per concentration, ddof=1).** Natural-log fit to bound vs. concentration: slope = −0.05186, intercept = 2.1342, r² = 0.9203, n = 9; half-concentration (linear interpolation of the raw curve) = 4.24 uM. The curve decays from ≈8.45 uM at 0 uM to ≈0.38 uM at 60 uM.

**Analysis + result evidence.**
- Analysis source: `output/results/parse_C_robust.py` (core extraction: `output/results/parse_C.py`)
- Result artifact: `output/results/binding_robustness.json` (core: `output/results/binding_competition_stats.json`)
- Robustness: same `binding_robustness.json` + `output/results/robustness_results.json` (key `F2`)

**Robustness.** Monotonic-decreasing = true. Alternative power-law (log–log) fit is even better (exponent = −0.8719, r² = 0.9721). Leave-one-concentration-out: dropping the highest concentration (60 uM) keeps the slope negative (−0.054995) and the fitted half-concentration changes only +12.0% (<20%); the interpolated half-concentration is unchanged (0%). Dropping the lowest (0 uM) keeps the slope negative but shifts the *fit-derived* half-concentration by −85% (the 0-uM point anchors the intercept), while the curve-interpolated half-concentration is unchanged — so the monotonic decay and the ~4.2 uM half-concentration are stable, and the fit-derived half-concentration is robust to the interior points. Preserved means/SDs re-matched to ~4e-7.

**Counter-evidence / limitation / alternative.** The contrasting ligand curve (`source_table_11.csv`) is *not* monotonic-decreasing: it rises with concentration (natural-log slope +0.0744, r² = 0.9005) and is non-monotonic (largest adjacent increase = 5.088 uM between 20→25 uM; power-law exponent +0.7995). This is a ligand-saturation-type relationship and is distinct from the competitor suppression; it confirms the two tables measure different relationships and must not be averaged.

**Falsifiable observation.** If competitor did not suppress binding, bound target would be flat or rising across 0–60 uM; the observed monotonic fall from ~8.5 to ~0.4 uM falsifies that.

## Finding 3: 0.5 uM Ca²⁺ basal promotes early fusion, in a condition-heterogeneous way

**Object and scope.** Basal-Ca²⁺ (none vs. 0.5 uM) effect on early fusion and total fusion across three sensor configurations (`source_table_07` block A, 5 runs no-Ca²⁺ / 3 runs 0.5 uM per condition). Related kinetic context from survival curves (`source_table_02`).

**Relationship / direction / boundary.** Condition-heterogeneous positive effect: 0.5 uM Ca²⁺ basal *increases* early fusion in all three conditions, but the magnitude is condition-dependent — negligible in bare SensorAWT and large only when 1:200 SensorB is co-present. Total fusion is essentially unchanged.

**Core quantitative result (ddof=1 from run columns).**
| Condition | Early-fusion no-Ca²⁺ (mean±SD) | Early-fusion 0.5 uM (mean±SD) | Δ early | Total-fusion change |
|---|---|---|---|---|
| SensorAWT | 64.22±2.586 | 64.63±6.045 | +0.415 | −6.10 |
| SensorAWT + 1:200 SensorB | 8.33±6.689 | 32.45±3.995 | +24.12 | −1.09 |
| SensorADA + 1:200 SensorB | 9.19±1.954 | 32.39±1.106 | +23.19 | −0.94 |

Early fusion jumps from ~8–9% to ~32% when Ca²⁺ basal is present *with* SensorB, but barely changes (+0.4) for bare SensorAWT (which already has high early fusion, 64%). Total fusion is largely unchanged (changes −6.1 to −0.9; the −6.1 for bare SensorAWT is within its large SD of 13.7).

**Analysis + result evidence.**
- Analysis source: `output/results/parse_B.py`
- Result artifact: `output/results/survival_kinetics_stats.json` (keys `table07`; survival context in `survival_curves`)
- Robustness: `output/results/robustness.py` → `output/results/robustness_results.json` (key `F3`)

**Robustness.** Per-condition Δ(early 0.5 uM − early no-Ca²⁺) is positive for all three conditions (mean Δ = 15.91 pp; max |total-fusion change| = 6.10 pp). The direction (Ca²⁺ basal promotes early fusion) is preserved across all conditions.

**Counter-evidence / limitation / alternative.** The bare-SensorAWT case (+0.4 pp, large SD) is a weak/no effect — the finding is therefore stated as condition-heterogeneous, not universal. Alternative explanation (Ca²⁺ drives *total* fusion) is refuted by the flat total-fusion changes. Kinetic context: higher SensorB concentration preserves vesicles (survival at 3 s: None ~11.5%, 1:800 ~38.9%, 1:200 ~93.6%), consistent with a SensorB-gated, Ca²⁺-triggered early-fusion phenotype rather than a general loss of integrity.

**Falsifiable observation.** If Ca²⁺ basal were inert to early fusion, Δ early would be ≈0 in all three conditions; the +23–24 pp increases in the two SensorB conditions falsify that.

## Finding 4: Docking is a null effect of sensor condition

**Object and scope.** Number of docked vesicles across sensor conditions (`source_table_09` block A, independent runs).

**Relationship / direction / boundary.** Null: no detectable dependence of docking on sensor condition or concentration; all condition means fall within ≈0.2 of the reference.

**Core quantitative result (ddof=1, runs only).**
| Condition | Docked (mean±SD) | Δ vs None |
|---|---|---|
| None (ref) | 2.52±0.198 | — |
| SensorA (1:200) | 2.324±0.102 | −0.196 |
| SensorB (1:2000) | 2.44±0.181 | −0.080 |
| SensorB (1:800) | 2.364±0.121 | −0.156 |
| SensorB (1:400) | 2.642±0.150 | +0.122 |
| SensorB (1:200) | 2.522±0.157 | +0.002 |

Maximum absolute difference from the None reference = 0.196; all |diff| < 0.5, so the null declaration holds.

**Analysis + result evidence.**
- Analysis source: `output/results/robustness.py` (the only script that re-parses `source_table_09`)
- Result artifact: `output/results/robustness_results.json` (key `F4`)

**Robustness.** `declared_null = true`; threshold all|diff|<0.5 satisfied with margin (max 0.196). The null is not a weak positive or negative: the largest single deviation (−0.196) is small relative to the run-to-run SDs (0.10–0.20).

**Counter-evidence / limitation / alternative.** Because the run-to-run SDs are comparable in magnitude to the observed deviations, this is a *null within measurement precision*, not proof of zero effect; a larger n could resolve a small effect. Alternative (sensor strongly changes docking) is not supported by any condition.

**Falsifiable observation.** If sensor changed docking, some condition mean would deviate from the None reference by a larger-than-noise margin (here, |diff| > 0.5 at the pre-set threshold); none do.

## Execution and process notes

The analysis ran as a single serial native Workflow `wf_6ae5c696-45e` (task `w3yks7g75`), with exactly one child started and awaited at a time (no `parallel()`/`pipeline()`/concurrency). All five groups completed with zero errors and with real `StructuredOutput` receipts in the journal. Two groups hit the platform's ~180 s no-progress stall and were auto-retried on the *same* journal key; each interruption/retry is recorded below as execution evidence.

| Group | Journal key (full) | Child ID (attempt) | Interval since last progress (s) | Completed? |
|---|---|---|---|---|
| A-fusion-dose | `v2:08cea42f7a504abd8120e3f6b1e613654e69ecd2d5ca9e3f72553914b00c7ec9` | `a0cc8da592747b3e7` (1) | 871 | no — stalled |
| A-fusion-dose (retry) | same key above | `a4f3cde109dad3711` (2) | — | **yes** |
| D-timeseries-fluorescence | `v2:d05a83c1dda3a632b299135a2bdb3b537380d7458ff1d490fd78afea62e13726` | `a1cc1b0a838fef3b0` (1) | 1577 | no — stalled |
| D-timeseries-fluorescence (retry) | same key above | `a8ed82efe1ee701dc` (2) | 180 | no — stalled |
| D-timeseries-fluorescence (retry) | same key above | `a3a882660fcc55128` (3) | — | **yes** |
| B-survival-kinetics | `v2:e1ac49323d9fd9d8c60065daacc79a08f22b0a97afed1b7b9c1c47cf32afe4d5` | `a4165e76137ead571` (1) | — | **yes** (no stall) |
| C-binding-robustness | `v2:ca2656f9ba7f80b0520f04957c25ce364b0a7d1700121f64440d175674f47d6f` | `a61cd19163f6ff2c1` (1) | — | **yes** (no stall) |
| R-robustness | `v2:2911650313c8aa4cc29f5f0d44dbf7d301596b6109335f88f5524fdd75d93d3d` | `ab37bcdab41848f13` (1) | — | **yes** (no stall) |

The prior concurrent Workflow `wf_03c1bc83-620` was stopped before the serial run; its artifacts were preserved and not overwritten. No old journal/log was deleted.

## Reproduction

1. **Parse + statistics.** Run each group script (each re-parses the raw CSVs and writes its result JSON):
   - `output/results/parse_A.py` → `output/results/fusion_dose_stats.json` (Finding 1)
   - `output/results/parse_B.py` → `output/results/survival_kinetics_stats.json` (Finding 3; survival context)
   - `output/results/parse_C.py` → `output/results/binding_competition_stats.json`; `output/results/parse_C_robust.py` → `output/results/binding_robustness.json` (Finding 2)
   - `output/results/parse_D.py` → `output/results/timeseries_stats.json` (context: time-series, fluorescence, clamp)
2. **Robustness / cross-validation.** `output/results/robustness.py` → `output/results/robustness_results.json` (keys `F1`–`F4`; leave-one-run-out, alternative window/model, monotonicity, null threshold).
3. **Orchestration.** `analysis/serial_workflow.js` runs the five groups serially (one child, awaited, then next).
4. **Validate.** `python3 validate_output.py output` — should report `passed: true`.

All means/SDs are recomputed from run columns with `ddof=1`; the workbook Mean/S.D. columns are used only as a cross-check (max deviation ≈0.003, i.e. rounding).
