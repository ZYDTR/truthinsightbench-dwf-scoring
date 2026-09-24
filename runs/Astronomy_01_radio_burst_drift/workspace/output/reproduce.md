# Reproduction Guide

Companion to `output/Result.md` (the sole authority for the registered findings).
This document records how to re-execute the analysis and which artifact supports
which finding.

- **Task:** `Astronomy_01_radio_burst_drift`, T0 = `2022-10-13`
- **Analysis unit:** one complete dynamic-spectrum event (n = 5), or one complete
  source-polarization series (n = 6 series / 12 series × regime). Pixels and
  adjacent directional coordinates are nested repeats.
- **Environment:** Python 3.11.2; NumPy 2.4.6; pandas 3.0.5; SciPy 1.17.1;
  Matplotlib 3.11.2; openpyxl 3.0.9. No network access is required and no
  external data is read.

## 1. Data boundary

All evidence comes from the nine files under `data/`. Their SHA-256 digests are
recomputed and compared against `data_manifest.json` by
`output/analysis/s01_inventory.py`, which writes
`output/results/s01_file_hashes.json`. Run that script first; if a digest
differs, every downstream number is invalidated.

## 2. Execution order

Each stage is standalone and writes machine-readable results plus a stdout log.

```bash
cd <project root>

# Stage 1 - inventory, hashing, sheet census
python3 output/analysis/s01_inventory.py

# Stage 2 - workbook parsing and provenance mapping
python3 output/analysis/s03_map_and_explore.py
python3 output/analysis/s15_reconcile_coverage.py
python3 output/analysis/s16_coverage_matrix.py

# Stage 3 - drift re-derivation and its robustness tests
python3 output/analysis/s10_drift_rederive.py
python3 output/analysis/s17_robustness_scan.py
python3 output/analysis/s19_surrogate_null.py
python3 output/analysis/s20_symmetric_decisive.py
python3 output/analysis/s22_structure_addendum.py

# Stage 4 - distributions and conditional comparisons
python3 output/analysis/driftpop_s01_binning_structure.py
python3 output/analysis/s05_rb_binning.py
python3 output/analysis/polc_s01_channel_shape.py

# Stage 5 - final consolidation of every cited number
python3 output/analysis/s23_final_consolidation.py

# Structural validation of the delivery contract
python3 validate_output.py output
```

Stage 5 (`s23_final_consolidation.py`) recomputes the small set of numbers the
report cites, so that each reported value has one reproducible source. If any
quoted number in `output/Result.md` disagrees with `output/results/s23_key_numbers.csv`,
the CSV is authoritative.

## 3. Memory discipline (required)

The first execution attempt was killed by the host (exit code −9, SIGKILL) after
six concurrent sub-agents each loaded the 32 MB workbook with openpyxl in
default mode. All working code must:

- open the workbook as `openpyxl.load_workbook(path, read_only=True, data_only=True)`;
- keep at most two 425 × 425 grids resident at once, looping event-by-event and
  releasing each before loading the next;
- avoid loading more than one large array per process.

The scripts listed above already follow this discipline. `output/analysis/s11_parse_workbook.py`
and `output/analysis/s14_images_effective.py` produce the extracted small
artifacts (`s11_blocks.csv`, `s14_images.npz`) that later stages reuse instead of
re-parsing the workbook.

## 4. Finding-to-evidence map

The registry `output/file_registry.json` maps each `Fnnn` identifier below to a
real file. Paths are also given directly so they resolve without the registry.

| Finding | Claim | Analysis source | Result evidence | Figures |
|---|---|---|---|---|
| **D1** | Reported drift rates not recoverable from the pixel arrays | `output/analysis/s10_drift_rederive.py` (F001); `s20_symmetric_decisive.py` (F003); `s19_surrogate_null.py` (F005); `s17_robustness_scan.py` (F007); `s22_structure_addendum.py` (F009) | `output/results/s10_drift_rederived_per_event.csv` (F002); `s20_symmetric_summary.json` (F004); `s19_surrogate_summary.json` (F006); `s17_robustness_summary.json` (F008); `s22_mask_diffuseness.csv` (F010) | `figures/s20_symmetric_decisive.png` (F034); `figures/s17_slope_scan_null.png` (F035) |
| **D2** | "fast"/"slow" split is a binning and provenance boundary | `output/analysis/s05_rb_binning.py` (F011); `driftpop_s01_binning_structure.py` (F013); `s15_reconcile_coverage.py` (F015); `chpol_s02_block_decomposed.py` (F041) | `output/results/s05_binning_grid_sensitivity.csv` (F012); `driftpop_s01_binning_structure.csv` (F014); `s15_slow_reconstruction.csv` (F016); `s16_normalized_provenance.csv` (F020); `polc_s02_slow_subblocks.csv` | — |
| **D3** | Normalized views are a lossy, non-injective projection | `output/analysis/s16_coverage_matrix.py` (F018); `covx_s02_absent_families.py` (F024); `s15_reconcile_coverage.py` (F015) | `output/results/covx_coverage_final.json` (F025); `s16_coverage_summary.json` (F019); `s15_orientation_reconstruction.csv` (F017); `covx_coarse_shape_similarity.csv` (F026); `covx_question_availability.csv` (F044) | `figures/covx_coverage_boundary.png` |
| **D4** | Channel shape differences real but count-dominated; direction flips by group | `output/analysis/polc_s01_channel_shape.py` (F021); `chpol_s01_channel_contrast.py` (F041) | `output/results/polc_s03_channel_contrast.csv` (F022); `polc_s04_group2_fast_test.json` (F023); `chpol_structure_check.csv` (F041) | `figures/polc_fig1_channel_profiles.png` (F036); `figures/polc_fig3_group2_fast.png` |
| **D5** | Nine unrepresented image sheets unresolvable | `output/analysis/s14_images_effective.py` (F029); `covx_s02_absent_families.py` (F024) | `output/results/s14_event_match.csv` (F030); `covx_image_model_tests.csv` (F027); `covx_unrepresented_images.csv` (F028); `s14_pairwise.csv` | `figures/covx_unrepresented_images_montage.png` (F037) |
| all | Final recomputation of cited numbers; robustness and covariate checks | `output/analysis/s23_final_consolidation.py` (F031); `s21_consolidate.py` | `output/results/s23_final_consolidation.json` (F032); `s23_key_numbers.csv` (F033); `s04_fit_crosscheck.csv` (F040); `s21_consolidated_leave_one_out.csv` (F042); `s04_orientation_series_stats.csv` (F043); `s01_file_hashes.json` (F039) | — |

## 5. Stochastic elements

| Step | Setting |
|---|---|
| Surrogate null (`s19`) | `seed = 20221013`, 25 surrogates, threshold 1.05, 401 slopes |
| Symmetric decisive scan (`s20`) | 401 slopes over [−40, +40] MHz/s, thresholds 1.01/1.05/1.20, injection amplitudes 0.01–0.25 |
| Channel Poisson bootstrap (`polc`) | seeded in `polc_s01_channel_shape.py`; reports counting-noise intervals only |
| Monoslope null (`s17`) | 401-slope grid, thresholds 1.00+ε … 1.20 |

Bootstrap intervals in D4 quantify **counting noise only**; the histogram bins are
nested repeats and are never treated as independent observations.

## 6. Known incomplete step

The mixture-model selection with multiple restarts plus a parametric bootstrap
false-preference test (`output/analysis/popfin_s01_model_selection.py`) was drafted
but did not finish before the run was consolidated. It is the named primary
falsification test for D2. The model comparisons that did complete on the original
bin grids are in `output/results/s05_binning_grid_sensitivity.csv` and
`output/results/s04_fit_crosscheck.csv`, and are reported in D2 with their
instability (degenerate component for `ps3`/`ps4`) explicitly flagged.

## 7. Validation

```bash
python3 validate_output.py output
```

Expect `"passed": true`. The validator checks delivery roles and traceability
only; it does not score scientific content.