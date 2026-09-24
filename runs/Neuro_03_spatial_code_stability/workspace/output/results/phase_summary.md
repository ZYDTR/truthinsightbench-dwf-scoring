# Session-phase effects on activity and spatial metrics (Neuro_03_spatial_code_stability)

## Design verification
- Distinct animals: **13**
- Reference animals: **6** (animal_01, animal_05, animal_06, animal_11, animal_12, animal_13)
- Matrix-perturbed animals: **7** (animal_02, animal_03, animal_04, animal_07, animal_08, animal_09, animal_10)
- Animals appearing in BOTH conditions: **0** (none)

Because no animal is in both conditions, the animal is the independent analysis unit and
unit-session rows are nested observations. **session_phase is imbalanced across conditions
and animals** (e.g., `after_followup` n=60 vs `late_followup` n=2), so this is a
**descriptive** analysis of per-phase and per-(condition,phase) means; it is not a
primary inferential test.

## Method
- Per-phase summary: n = number of observation rows (unit-level, nested) in each phase;
  means of `average_rate_hz`, `spatial_regularity_score`, `information_rate`.
- Per-(condition, phase) summary: same metrics, split by condition.
- Source script: `analysis_phase.py`; tables in `results/phase_effect_overall.csv` and
  `results/phase_effect_by_condition.csv`; figure in `figures/phase_effect.png`.

## Per-phase means (overall)
| session_phase | n | mean rate (Hz) | mean regularity | mean info rate |
|---|---|---|---|---|
| before_perturbation | 34 | 1.8116 | 0.6675 | 0.3549 |
| early_followup | 29 | 1.6555 | 0.5865 | 0.3249 |
| mid_followup | 9 | 1.9586 | 0.5498 | 0.1827 |
| late_followup | 2 | 1.5012 | 0.4857 | 0.2827 |
| after_followup | 60 | 2.2639 | 0.6108 | 0.5967 |

Interpretation: rates and information rate are highest in `after_followup` and
`before_perturbation`, lowest in `late_followup`; regularity is highest in
`before_perturbation`. `late_followup` has only n=2, so its values are unstable.

## Per-(condition, phase) means
| condition | phase | n | mean rate (Hz) | mean regularity | mean info rate |
|---|---|---|---|---|---|
| reference | before_perturbation | 14 | 1.5280 | 0.7367 | 0.5272 |
| matrix_perturbed | before_perturbation | 20 | 2.0102 | 0.6191 | 0.2342 |
| matrix_perturbed | early_followup | 21 | 1.8092 | 0.5848 | 0.2985 |
| matrix_perturbed | mid_followup | 7 | 2.0576 | 0.6756 | 0.1975 |
| matrix_perturbed | late_followup | 2 | 1.5012 | 0.4857 | 0.2827 |
| matrix_perturbed | after_followup | 12 | 1.9868 | 0.5734 | 0.4383 |
| reference | after_followup | 48 | 2.3331 | 0.6202 | 0.6363 |
| reference | early_followup | 8 | 1.2522 | 0.5911 | 0.3944 |
| reference | mid_followup | 2 | 1.6119 | 0.1096 | 0.1304 |

Interpretation: reference animals carry most of the `after_followup` load (n=48) and
show the highest `after_followup` information rate (0.6363), whereas perturbed animals
show a consistently lower information rate in the early/mid follow-up phases
(0.2342 / 0.1975) but recover by `after_followup` (0.4383). Regularity is higher
for reference in `before_perturbation` (0.7367 vs 0.6191). Because phase counts are
heavily imbalanced and nested, these are descriptive patterns, not condition-inferential
effects.
