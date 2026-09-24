# Reproduction guide

Task `Astronomy_02_gravity_spectral_structure` (T0 = 2022-01-01).
Authoritative scientific account: `output/Result.md`.

## Environment

```
Python 3.11.2
numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, matplotlib 3.11.2 (Agg backend)
```

`scipy` and `matplotlib` were installed into the user site during this run
(`pip install scipy matplotlib`). No other dependency was added, and the environment needed no
network access for the analysis itself.

## Data integrity check

All five task tables are read read-only from `data/` and were hashed against `data_manifest.json`:

```bash
cd /data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project
sha256sum data/*.tsv
```

Expected:

```
63a5790704c3faf4cffc414b0e2a579e585d1c375090ea217f99c5dcc5ff7bbe  data/candidate_spectrum_comparison.tsv
30baf2dfaa9432d571385c53ec895de5de83aca61b85281843e52e12a4dc7fcb  data/mode_frequency_grid.tsv
584915428982b43472d2a1f3d2fec0c82c52c3ff3ba046877f8a4d87ff88e067  data/pass_residual_traces.tsv
bd21877cf7d72a6cb52cb81687db9f47ec91093819317251998bf91fd0f7df2d  data/ranked_full_parameter_grid.tsv
c913ed4d4f6405956fbe86c376e1662287572ad55c1eee61d9290890a0402fce  data/uniform_alternative_comparison.tsv
```

## Full re-run

```bash
cd /data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project

# 1. Core analysis tracks (runs track1..track5 in order via subprocess)
python3 output/code/run_all.py

# 2. Additional tracks written after the harness was assembled
python3 output/code/track6_localization.py
python3 output/code/track7_matched_null_adjudication.py

# 3. Semantic delivery validation
python3 validate_output.py output
```

`run_all.py` prints each command it issues and each script's exit code, and returns non-zero if any
child fails. Observed end-to-end result: **6/6 scripts exited 0 in 869.5 s**
(`output/results/_run_all.log`, `output/results/track5_run_all_summary.json`).

Note: `run_all.py` currently covers `track1..track5` only; `track6` and `track7` were produced after
the harness was written and must be invoked separately as shown above.

## Individual scripts

| Script | Purpose | Runtime | Writes |
|---|---|---|---|
| `output/code/common.py` | Shared helpers (loaders, Lomb-Scargle, AR(1) surrogates, blocked stats) | — | imported by all tracks |
| `output/code/track1_model_comparison.py` | Whole-pass RMS, blocked paired tests, bootstrap, LOO, robustness, cross-validation | 134.1 s | `output/results/model_comparison.json` |
| `output/code/track1_followup_checks.py` | Follow-up consistency checks for track 1 | 110.7 s | `output/results/track1_followup.log` |
| `output/code/track2_spectral_tests.py` | Per-pass periodograms, AR(1) band null, comb test, phase coherence, slopes | 552.8 s | `output/results/spectral_tests.json`, `spectral_tests_arrays.npz` |
| `output/code/track3_grid_structure.py` | Axis inventory, truncation, top-cell separation, degeneracy, separability, mode alignment, resolution | 43.3 s | `output/results/grid_structure.json` |
| `output/code/track4_ic_comparison.py` | Uniform-vs-diagonal identity, IC minima, ΔIC offset check, boundary sensitivity | 3.8 s | `output/results/ic_comparison.json` |
| `output/code/track5_figures.py` | Five evidence figures + `figures_manifest.json` + figure annotations | 24.8 s | `output/figures/fig1..fig5`, `figures_manifest.json`, `results/track5_figure_annotations.json` |
| `output/code/track6_localization.py` | Centre/edge localization, time-shuffle null, envelope-shape fits, robustness | — | `output/results/localization.json`, `localization_summary.md` |
| `output/code/track7_matched_null_adjudication.py` | Matched-order AR(1) null for the stacked peaks (corrects the mismatched null) | — | stdout / `/tmp/adj.out` (see script) |
| `output/code/run_all.py` | Executable harness | 869.5 s | `results/_run_all.log`, `results/track5_run_all_summary.json` |

Verification scripts (independent re-derivation, not part of the analysis path):
`output/code/verify_model_comparison.py`, `verify_grid.py`, `verify_ic_comparison.py`,
`verify_spectral_recompute.py`, `verify_spectral_detrend_band.py`,
`verify_spectral_circshift_coverage.py`, `verify_spectral_stacknull_bias.py`.

## Random seeds

All stochastic steps are seeded and the seed is recorded in the corresponding result JSON under a
`seed` key and in the script source.

| Seed | Used for |
|---|---|
| `20220101` | Whole-pass bootstrap (track 1), spectral AR(1) surrogates (track 2), localisation shuffle null default |
| `2022` | Window/whole-pass perturbation surrogates |
| `7` | Mode-alignment uniform chance null (200,000 draws) |
| `3` | Two-sinusoid separation simulation |
| `5` | Centre/edge within-pass time-shuffle null |
| `4242` | Matched-null adjudication (track 7) |

## Reproduction wobble

`track2_spectral_tests.py` records its own wall-clock time in `runtime_sec`. Re-running it under a
different host load changes that single field and nothing else: a sandbox re-run reproduced every
substantive number in `spectral_tests.json` bit-for-bit, with `runtime_sec` the only difference
(`output/results/verify_spectral.json`). The runtimes tabulated above come from the recorded
`run_all` log, not from the current on-disk copies.

## Key intermediate validation points

Reproducing the registered findings should check these in order:

1. `model_comparison.json → item2_blocked_paired_tests_over_22_passes`: all three pairs must show
   22/22 wins and Wilcoxon p = 4.768e-07.
2. `model_comparison.json → item3_whole_pass_bootstrap_median_ratio → raw`: median B/A must be
   0.54130 with CI [0.45999, 0.62074].
3. `localization.json → item7_verdict`: median A−B centre/edge ratio 1.955; blocked p 3.338e-06.
4. `ic_comparison.json → q1_uniform_is_mv_diagonal`: `bit_identical` true, `max_abs_difference` 0.0.
5. `spectral_tests.json → sec7_verdict`: stacked peaks at 489 / 2474 / 1706 µHz, all outside the
   860–1310 µHz candidate band.
6. `grid_structure.json → 5_mode_grid_alignment → observed → median_offset_uHz`: 5.76, versus a null
   median of 8.08.
7. `grid_structure.json → 2_top_cell_separation → counts_within_delta_ic`: `{"1": 1, "2": 1, "5": 2,
   "10": 4}`.

## Scope statement

The workspace `data/` directory is the complete dataset for this task. No external scientific dataset,
precomputed task result, or external literature was downloaded, substituted, or merged. No source
outside the workspace contributed to any registered finding.