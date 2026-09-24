# Reproduction

Companion reproduction document for `output/Result.md`
(task `Energy_03_interface_response_boundaries`).

## 1. Requirements

- Python 3.11 (developed on 3.11.2)
- `numpy` 2.4.6, `pandas` 3.0.5 (present in the initial environment)
- `matplotlib` 3.11.2 — needed **only** by `make_figures.py`; it was not present
  initially and was installed during the run. Every scientific number in the
  report is produced by scripts that do **not** import matplotlib.
- No `scipy`, `statsmodels`, or `sklearn` is required or used. All inferential
  statistics come from `output/native/analysis/tib_stats.py`, a dependency-free
  module written for this task.

## 2. Inputs

Read directly from the workspace, never copied or edited:

| File | SHA-256 (must match `data_manifest.json`) |
|---|---|
| `data/anonymous_condition_response_curves.tsv` | `b3668160fac559700404181f686acec1394cfcdaf2ef34d38a75834556f89726` |
| `data/independent_condition_check_curves.tsv` | `1a79f8b8bf11fe116930ddc9e4b3a50baefe441235f41cf0bd8a1f81ae335237` |

Both hashes are re-verified programmatically at the start of every run
(`run_analysis.py` stage A and `run_verification.py`); a mismatch raises and
aborts. No external dataset is downloaded, substituted, or merged.

## 3. Execution order

Run from the workspace root. Each script is standalone, deterministic, and
writes its own artifact, so any subset can be re-run in any order.

```bash
python3 output/native/analysis/run_analysis.py      # A integrity · B ordering · C exponents · D model competition
python3 output/native/analysis/run_stage_e.py       # logistic light-off fits · shared-shape models
python3 output/native/analysis/run_stage_f.py       # forward holdout · panel integrity · crossing census
python3 output/native/analysis/run_stage_g.py       # multi-cut holdouts · whole-curve holdout · alt. sigmoids
python3 output/native/analysis/run_stage_h.py       # decisive shared-width vs per-curve-width test
python3 output/native/analysis/run_stage_i.py       # block tests · single-ordering-parameter · result tables
python3 output/native/analysis/run_stage_j.py       # curvature robustness
python3 output/native/analysis/make_figures.py      # figures + figure-backing CSVs  (needs matplotlib)
python3 output/native/analysis/run_verification.py  # recompute every cited number -> results/results.json
python3 validate_output.py output                   # semantic delivery validation
```

`run_stage_f.py` and `run_stage_g.py` import helpers from `run_stage_e.py`;
all other scripts import only `tib_stats.py`. Run the `run_stage_e.py` file at
least once before them, or rely on the `sys.path` insert each script performs
(it imports the module, it does not re-execute `main()`).

## 4. Determinism

There is **no randomness anywhere in the pipeline**. No RNG is seeded, because
none is used:

- permutation / sign tests use **complete enumeration** of the assignment space
  (`tib_stats.permutation_p`, and the inline binomial sums), not Monte Carlo;
- optimisers are deterministic numerical-gradient descent with backtracking
  (`run_stage_h.numer_gd`) or deterministic Gauss-Newton
  (`run_stage_e.fit_logistic`);
- grid searches iterate fixed `linspace` grids.

Re-running any script reproduces byte-identical numerical results. Verify with:

```bash
python3 output/native/analysis/run_verification.py
sha256sum output/native/results/results.json
```

## 5. Reproduction of each registered finding

| Finding | Scripts | Primary artifacts |
|---|---|---|
| 1 — log-response curvature | `run_stage_j.py`, `run_verification.py` | `results/J_curvature_robustness.json`, `results/results.json` → `D1_curvature`, `results/fig3_data.csv`, `figures/fig3_log_response_concavity.png` |
| 2 — apparent exponent spread | `run_analysis.py` (stage C), `run_stage_h.py`, `run_verification.py` | `results/C_growth_and_exponents.json`, `results/H_shape_test.json`, `results/results.json` → `D2_relationship`, `results/curve_parameters.csv` |
| 3 — ordering boundary | `run_analysis.py` (stage B), `run_stage_g.py`, `run_verification.py` | `results/B_ordering_crossings.json`, `results/G3_separation_boundary.json`, `results/results.json` → `D3_ordering`, `results/response_matrix_with_ranks.csv`, `figures/fig2_rank_ordering.png` |
| 4 — saturation boundary | `run_stage_e.py`, `run_stage_g.py`, `run_verification.py` | `results/E_sigmoid_model.json`, `results/F5_saturation_boundary.json`, `results/G5_alternative_sigmoids.json`, `results/G4_outlier_sensitivity.json` |
| 5 — panel duplication / block null | `run_stage_f.py`, `run_stage_i.py`, `run_verification.py` | `results/F2_cross_panel_integrity.json`, `results/I_consolidation.json`, `results/results.json` → `D4_panel_correspondence`, `D5_block_null` |

`results/results.json` is the consolidated verification artifact: it recomputes
every number cited in `Result.md` from the raw TSVs in a single pass and is the
fastest way to check the report against the data.

## 6. Design constraints enforced by the code

- **Analysis unit = one complete condition curve.** No script ever splits a
  curve into individual points for training/testing.
- **Holdouts are whole-curve or contiguous-setting only.** Contiguous-setting
  cut points are 240/260/280/300 K (`run_stage_g.g1_multicut`); whole-curve
  holdouts are `run_stage_g.g2_whole_curve_loo`.
- **No random curve-point splitting anywhere**, per the task data guide.
- Zero-valued responses are excluded from log-scale models and the count of
  excluded points is written into the artifacts
  (`results/D_model_competition.json` → `n_zero_response_points_excluded_from_log_models`).

## 7. Known retained failures

Reported rather than hidden; full tracebacks are in `output/native/logs/`.

1. `run_analysis.py`, first execution: nested-model F test returned **negative**
   F (F = −8.56) because the compared models were not genuinely nested. The
   error is preserved in `logs/run_analysis.log`; the comparison was replaced by
   the correct restricted-model test in `run_stage_h.py`.
2. `run_stage_e.py`, first execution: `np.clip` broadcast error (2-element vs
   3-element parameter bounds when the ceiling was fixed). Traceback in
   `logs/run_stage_e.log`; bounds corrected.
3. A `pip install --quiet matplotlib` reported success while `import matplotlib`
   still failed. The non-quiet retry succeeded; only that retry produced the
   figures.
4. Crossref lookups for several candidate pre-`T0` review DOIs returned no
   parsable record; those sources were dropped rather than cited from memory.
   See `output/native/provenance/external_sources.md`.

## 8. Environment notes for re-running

`matplotlib` writes font-cache data to the platform home directory on first
use. If the run is repeated in a fresh container, the first
`make_figures.py` invocation may emit a font-cache advisory on stderr; it does
not affect output. All other scripts are pure-stdlib + numpy/pandas.