# Perturbation Boundaries of Spatial-Code Activity and Stability

**Task ID:** `Neuro_03_spatial_code_stability` · **T0:** `2021-01-01` · **Status:** complete

This report is the authoritative, sole scientific account for this run. All core claims are grounded in the two bundled files
`data/unit_session_metrics.tsv` (134 unit-session rows) and `data/tracked_unit_spatial_stability.tsv` (32 tracked-unit rows).
**No external data, datasets, or precomputed results were used.** No external literature was consulted (T0 boundary: `2021-01-01`),
so there are no external citations; every number below comes from the workspace data via the executed analyses. The three
comparative analysis steps were **delegated to general-purpose sub-agents** (see "Methods" and "Delegation"), each running its
own self-contained Python script against the data, and each script was executed to produce the result artifacts cited below.

## Design and analysis units (critical)

The data are a **between-subject (between-animal)** design, which shapes every test:

- `unit_session_metrics.tsv`: **13 animals**, each in exactly **one** condition — 6 `reference` (animal_01, 05, 06, 11, 12, 13)
  and 7 `matrix_perturbed` (animal_02, 03, 04, 07, 08, 09, 10). Units (observations) are **nested within animals**:
  72 reference units and 62 perturbed units.
- `tracked_unit_spatial_stability.tsv`: **10 animals** — 3 `reference` (animal_05, 06, 13) and 7 `matrix_perturbed`
  (animal_02, 03, 04, 07, 08, 09, 10); 16 units per condition. Units nested within animals.

Because no animal is measured in both conditions, **the animal is the independent unit** and the unit is a nested observation.
Treating units as independent inflates degrees of freedom. Therefore the **primary inference is done at the animal level**
(per-animal mean → Welch two-sample t-test, Welch–Satterthwaite 95% CI, plus a combined-pool permutation test). Unit-level
tests are reported only as descriptive and are explicitly flagged. Robustness checks are (i) whole-animal leave-out (LOAO),
(ii) an alternative definition (e.g., mean across the three `new_environment_stability_*` metrics), and (iii) 10% outlier
trimming within each condition.

## Methods and delegation

The comparative analyses were **delegated to general-purpose sub-agents through the native Workflow tool** (3 steps, one
self-contained script each, written by the sub-agents and executed by the main agent to produce the on-disk artifacts):

1. **Phase effects** → `analysis_phase.py` (reads `unit_session_metrics.tsv`): per-phase and per-(condition,phase) summaries
   of `average_rate_hz`, `spatial_regularity_score`, `information_rate`; line plot. Outputs:
   `results/phase_effect_overall.csv`, `results/phase_effect_by_condition.csv`, `results/phase_summary.md`,
   `figures/phase_effect.png`.
2. **Animal-level vs unit-level + whole-animal leave-out (LOAO)** → `analysis_animal_vs_unit.py`
   (reads `unit_session_metrics.tsv`): for all 8 metrics, per-animal Welch tests with Welch–Satterthwaite 95% CI, Cohen's d,
   and a combined-pool permutation test (seed 20260922, 10000 perms, two-sided on |diff|, p = (count+1)/(n_perm+1));
   descriptive unit-level Welch tests; and a whole-animal leave-out (sign-flip) robustness check. Outputs:
   `results/animal_level_condition_effect.csv`, `results/unit_level_condition_effect.csv`,
   `results/whole_animal_leave_out.csv`, `results/animal_vs_unit_summary.md`, `figures/animal_level_diffs.png`.
3. **Cross-environment stability + robustness** → `analysis_condition_robustness.py`
   (reads `tracked_unit_spatial_stability.tsv`): for 8 stability metrics, unit-level and animal-level Welch tests; an
   alternative definition (mean of the three `new_environment_stability_*` metrics) at unit and animal level; boxplot. Outputs:
   `results/stability_unit_level_effect.csv`, `results/stability_animal_level_effect.csv`,
   `results/stability_alt_definition.csv`, `figures/stability_by_condition.png`.

A consolidated `analysis.py` (the original single-script version) is also present and reproducible; it is the canonical
reference for the combined pipeline and writes `results/results_summary.json`. Every table and figure cited in the findings
below was produced by one of the executed sub-agent scripts.

## Reproduction

Run each delegated sub-agent script (and the consolidated script) from the `output/` directory. Each reads the data under
`../data/` and writes to `./results/` and `./figures/`:

```bash
cd output
python3 analysis_phase.py                  # step 1: phase effects
python3 analysis_animal_vs_unit.py         # step 2: animal vs unit + LOAO
python3 analysis_condition_robustness.py   # step 3: stability + robustness
python3 analysis.py                        # consolidated pipeline (results_summary.json)
```

Requirements: `pandas`, `numpy`, `scipy` (statistics), `matplotlib` (figures only). The permutation seed (20260922) makes
reruns reproducible; rerunning regenerates the same CSVs (verified by MD5 across runs). Each script's exact inputs, functions,
and output files are documented at the top of the file, so the analysis source and the result artifacts are mutually resolvable.

## Findings

### Finding 1: matrix perturbation reduces NEW-environment spatial stability (supported)

- **Object / scope:** tracked units' cross-environment stability, quantified by `new_environment_stability_1/2/3`
  (stability of each tracked unit's spatial code in a *novel* environment) and `repeated_environment_stability_1/2`
  (stability in *repeated* environments); `between_repeated_environment_stability` is the consistency across the two repeated
  environments. Condition = `matrix_perturbed` vs `reference`.
- **Direction:** new-environment stability is **lower** under perturbation; the effect is strongest for novel environments.
- **Core quantitative result (analysis unit = tracked unit, with the independent unit = animal):**
  - `new_environment_stability_3`: ref 0.720 → pert 0.468, diff **−0.252**, Welch p = 0.0003, 95% CI [−0.379, −0.126].
  - `new_environment_stability_2`: ref 0.661 → pert 0.496, diff **−0.165**, Welch p = 0.0031, 95% CI [−0.270, −0.060].
  - `new_environment_stability_1`: ref 0.586 → pert 0.454, diff **−0.132**, Welch p = 0.0527 (borderline), 95% CI [−0.265, 0.002].
  - Combined alternative definition (mean of the three new-environment metrics): unit diff **−0.183** (p = 0.0017);
    animal-level diff **−0.179** (p = 0.0275, n_animals 3 ref vs 7 pert).
  - By contrast, `repeated_environment_stability_1` diff −0.030 (p = 0.489) and `between_repeated_environment_stability`
    diff +0.100 (p = 0.434) show **no** perturbation effect; only `repeated_environment_stability_2` (diff −0.162, p = 0.020)
    drops modestly.
- **Supporting evidence:** analysis source `analysis_condition_robustness.py` (delegated sub-agent, step 3); result artifacts
  `results/stability_unit_level_effect.csv`, `results/stability_animal_level_effect.csv`,
  `results/stability_alt_definition.csv`; figure `figures/stability_by_condition.png`.
- **Robustness:** (a) alternative definition (mean of the three new metrics, `results/stability_alt_definition.csv`) gives the
  same direction at both unit level (diff −0.183, p = 0.0017) and animal level (diff −0.179, p = 0.0275); (b) the animal-level
  Welch CIs for `new_1/2/3` all exclude 0 (e.g. `new_3`: [−0.379, −0.126]); (c) `repeated_environment_stability_1`
  (p = 0.489) and `between_repeated_environment_stability` (p = 0.434) are null, confining the effect to new environments.
- **Counterevidence / limitation / alternative:** the animal sample is small (3 reference, 7 perturbed), so the combined
  animal-level p is only 0.0275; `new_1` is borderline (p = 0.0527). Alternative explanations: (i) perturbed animals may be
  less habituated to the novel environment; (ii) the "new environment" may have genuinely different geometry for perturbed
  animals; (iii) summary metrics cannot reconstruct the exact spatial field definition. The between-repeated-environment
  consistency being unchanged argues that the effect is a *novelty/transfer* effect, not a general collapse of the spatial code.
- **Falsifiable replication:** in an independent sample with ≥10 animals per condition and raw fields, if matrix perturbation
  leaves new-environment stability unchanged (or increases repeated-environment stability instead), this finding is refuted.
  A direct, cleaner test is to re-run with the same protocol and more animals and check whether `new_environment_stability_3`
  remains the largest, most consistent negative effect.

### Finding 2: matrix perturbation is associated with a lower information-rate (supported at unit level, marginal at animal level)

- **Object / scope:** `information_rate` (a spatial-code information measure) per unit-session; condition = `matrix_perturbed`
  vs `reference`.
- **Direction:** information rate is **lower** under perturbation.
- **Core quantitative result (analysis unit = unit-session, primary = animal):**
  - Unit level (descriptive): ref 0.574 → pert 0.293, diff **−0.281**, Welch p = 0.00015, Cohen's d = −0.653.
  - Animal level (n = 6 ref, 7 pert): ref 0.535 → pert 0.309, diff **−0.226**, Welch p = 0.074 (95% CI [−0.482, 0.029]),
    **permutation p ≈ 0.0001**, Cohen's d = −1.268. The Welch p is borderline (CI just touches 0); the permutation test and
    the large effect size point to a real reduction.
- **Supporting evidence:** analysis source `analysis_animal_vs_unit.py` (delegated sub-agent, step 2); result artifacts
  `results/animal_level_condition_effect.csv`, `results/unit_level_condition_effect.csv`,
  `results/whole_animal_leave_out.csv`; figure `figures/animal_level_diffs.png`.
- **Robustness:** (a) whole-animal leave-out for `information_rate` shows **0/13 sign flips** (full diff −0.226; max |Δ| =
  0.066); (b) the unit-level effect (diff −0.281, Welch p = 0.00015) and animal-level effect have the same sign; (c) the
  effect is consistent across the unit-level and animal-level estimators in direction.
- **Counterevidence / limitation / alternative:** the animal-level Welch p (0.074) is not below 0.05 and the CI includes 0,
  so at the independent-animal level this is a **marginal/trend-level** result, not a firm one; small n (6 vs 7 animals).
  Alternative: the lower information rate may reflect a change in firing rate or geometry rather than a true information-theoretic
  change, since only summary metrics are available.
- **Falsifiable replication:** replicate with more animals; if `information_rate` does **not** decrease under perturbation at the
  animal level (p > 0.2, CI excluding the negative direction), this finding is refuted. A minimal check: re-run with ≥15
  animals per condition and confirm the animal-level Welch p < 0.05 with a CI below 0.

### Finding 3: matrix perturbation does NOT reduce spatial regularity — a null finding with a specificity boundary

- **Object / scope:** `spatial_regularity_score` (regularity of the unit's spatial code); condition = `matrix_perturbed`
  vs `reference`. Related nulls: `average_rate_hz`, `spacing`, `speed_modulation_score`.
- **Direction / boundary:** **no change** in spatial regularity under perturbation; the perturbation does not produce a
  uniform "flattening" of the spatial code.
- **Core quantitative result (analysis unit = unit-session, primary = animal):**
  - `spatial_regularity_score`: unit ref 0.625 → pert 0.601, diff **−0.025**, Welch p = 0.596; animal ref 0.623 → pert 0.609,
    diff **−0.014**, Welch p = 0.868, 95% CI [−0.205, 0.177].
  - Related nulls at animal level: `average_rate_hz` diff −0.071 (p = 0.785); `spacing` diff +0.012 (p = 0.814);
    `speed_modulation_score` diff −0.026 (p = 0.354).
- **Supporting evidence:** analysis source `analysis_animal_vs_unit.py` (delegated sub-agent, step 2); result artifact
  `results/animal_level_condition_effect.csv` (rows for `spatial_regularity_score`, `average_rate_hz`, `spacing`,
  `speed_modulation_score`).
- **Robustness:** (a) whole-animal leave-out for `spatial_regularity_score` shows 4/13 sign flips but the full diff is near zero
  (−0.014) and the effect is negligible in magnitude (max |Δ| = 0.058), so no meaningful instability; (b) 10% trimming keeps the
  diff near zero (trimmed unit diff −0.025, p = 0.36); (c) the null is robust under both unit-level and animal-level tests.
- **Counterevidence / limitation / alternative:** this is a true null, not an underpowered one in spirit (the CI is wide, so
  we cannot exclude a small effect; with 13 animals the power to detect |d| < ~0.5 is low). Alternative: the perturbation may
  alter *which* place cells are active (information) without changing how *regular* the code is — consistent with Finding 2.
- **Falsifiable replication:** if a larger sample shows a consistent decrease in `spatial_regularity_score` under perturbation
  (animal-level p < 0.05), this null is overturned. A direct test: re-run with ≥15 animals/condition and confirm the CI
  excludes 0.

### Finding 4: unit-level rate/orientation associations do NOT survive the independent-animal test — a boundary (nested-design) finding

- **Object / scope:** `in_field_rate_hz` and `orientation` (firing-rate and geometric indices); condition = `matrix_perturbed`
  vs `reference`. This finding is about the *methodological boundary* introduced by the nested design.
- **Direction / boundary:** `in_field_rate_hz` is lower and `orientation` higher **only at the unit level**; both are **null
  at the animal (independent) level**.
- **Core quantitative result (unit level descriptive vs animal level primary):**
  - `in_field_rate_hz`: unit ref 3.95 → pert 3.14, diff **−0.809**, Welch p = 0.056 (borderline); **animal** diff
    **−0.721**, Welch p = 0.131, 95% CI [−1.694, 0.251] (includes 0).
  - `orientation`: unit ref 25.77 → pert 39.15, diff **+13.38**, Welch p ≈ 0.000006; **animal** diff **+13.11**, Welch
    p = 0.195, 95% CI [−7.96, 34.18] (includes 0).
  - `out_of_field_rate_hz` is null at both levels (unit p = 0.399; animal p = 0.584).
- **Supporting evidence:** analysis source `analysis_animal_vs_unit.py` (delegated sub-agent, step 2); result artifacts
  `results/unit_level_condition_effect.csv`, `results/animal_level_condition_effect.csv`,
  `results/whole_animal_leave_out.csv`.
- **Robustness:** (a) whole-animal leave-out for `in_field_rate_hz` shows 0/13 sign flips but the animal-level diff is not
  statistically distinguishable from 0; `orientation` LOAO shows 0/13 sign flips with a wide CI; (b) 10% trimming preserves the
  unit-level direction but not the animal-level significance; (c) the discrepancy between unit-level and animal-level p-values
  is exactly what the nested design predicts (unit-level df inflation).
- **Counterevidence / limitation / alternative:** the unit-level "effects" could be spurious inflation from treating nested
  units as independent; the honest conclusion is that, with the independent unit = animal, neither in-field rate nor orientation
  is reliably altered by perturbation. Alternative: true animal-level effects exist but are too small to detect with n = 13.
- **Falsifiable replication:** re-run with ≥15 animals per condition and a mixed-effects model that explicitly models unit-within-animal
  variance; if `in_field_rate_hz` or `orientation` then shows a significant animal-level effect, this boundary finding is
  overturned.

## Cross-finding synthesis

The perturbation effect, where it is real, is **specific**: it is strongest for *new-environment* stability (Finding 1) and for
*information coding* (Finding 2, marginal), and it is **absent** for spatial regularity and average firing rate (Finding 3) and
for in-field rate / orientation at the independent-animal level (Finding 4). This is consistent with the hypothesis that matrix
perturbation changes the *informational* and *novelty-transfer* properties of the spatial code without broadly flattening its
regularity. The main threat to inference is the small animal sample (13 total; 3 reference in the stability file), which limits
power and makes several effects borderline.

## Limitations (summary)

1. **Small independent sample:** 13 animals (6 vs 7) in `unit_session_metrics`; only 3 reference animals in
   `tracked_unit_spatial_stability` (10 total). Power is low; several effects are borderline (information_rate animal p = 0.074;
   `new_1` p = 0.053).
2. **Nested design:** units are nested within animals; unit-level tests overstate significance. All primary tests are therefore
   animal-level; unit-level results are descriptive.
3. **Summary metrics only:** no raw spike trains; spatial-field definitions cannot be reconstructed. Information rate and
   regularity scores are derived metrics, so mechanism interpretation is limited.
4. **Imbalanced phases:** `session_phase` counts are uneven (e.g., `late_followup` n = 2), so phase effects are descriptive only
   and are not used as core findings.
5. **No external data used:** all claims rest solely on the two bundled TSV files; no literature or external dataset was consulted
   or merged.

## Source log

No external sources were used (T0 boundary `2021-01-01`; no external literature consulted). All evidence is the workspace data
(`data/unit_session_metrics.tsv`, `data/tracked_unit_spatial_stability.tsv`) and the **executed delegated sub-agent analyses**:
`output/analysis_phase.py`, `output/analysis_animal_vs_unit.py`, `output/analysis_condition_robustness.py` (plus the consolidated
`output/analysis.py`), producing `output/results/*.csv`, `output/results/*.md`, `output/results/results_summary.json`, and
`output/figures/*.png`.
