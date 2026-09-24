# Reproduction — Robust Regions and Failure Boundaries in Replicated Input-Output Surfaces

Task `Information_03_signal_surface_invariants`, T0 = 2022-01-01.
This document reproduces every registered finding in `output/Result.md` from the frozen task data.

## 1. Environment

- Python 3.11.2
- numpy 2.4.6, pandas 3.0.5, matplotlib 3.11.2
- **scipy is not installed.** Rank-based statistics (Spearman, permutation nulls) are implemented
  directly with `pandas.Series.rank()` and `numpy.random.default_rng`. Any reproduction using
  `scipy.stats` may differ in tie handling; the project code is the reference.
- No network access is required or used. Nothing is downloaded; the workspace `data/` directory is the
  complete dataset.

## 2. Inputs and integrity

All scripts call `common.load_panels()`, which parses each TSV with `sep="\t"` and records its SHA-256.
Expected values (these match `data_manifest.json` and `workspace_receipt.json`):

| File | Bytes | SHA-256 |
|---|---|---|
| `data/replicated_response_panel.tsv` | 242606 | `410e51b559eb4d3416f131c2eb4d8f75995ff00d3b76af7b47a0ee8b1220533d` |
| `data/independent_response_panel.tsv` | 121008 | `f85f73bbbf15535425752c6ca2ae65e523c8dd8c06a2d5de11bf1986aaa884a1` |
| `data/independent_fit_error.tsv` | 564 | `57b739c708fb384f7a661d85e62b67f6c5f333cab170214d6f972ca067d2564a` |

Declared designs, asserted at load: replicated 7×7×8×7×3 = 8232 rows; independent 7×7×7×4×3 = 4116
rows; fit-error 7×3 = 21 rows. No nulls; no duplicate keys; no imputation.

## 3. Exact commands and expected outputs

Run each from `output/analysis/` (so that `import common` resolves). Each script is standalone and prints
its own progress; logs are written to `output/logs/`.

```bash
cd output/analysis

python3 step1_surface_geometry.py     # -> results/t01..t08, logs/step1.log
python3 step2_holdout_validity.py     # -> results/t09,      logs/step2.log
python3 step3_independent_panel.py    # -> results/t10..t19, logs/step3.log
python3 step4_figures.py              # -> results/figures/*.png, logs/step4.log
python3 step5_robustness.py           # -> results/t20,      logs/step5.log
python3 step6_verification.py         # -> results/t21,      logs/step6.log   (~4 min)
python3 step7_domain_boundary.py      # -> results/t22,      logs/step7.log
```

Then, from the project root:

```bash
python3 validate_output.py output
```

### Expected headline values per script

| Script | Quantity | Expected | Tolerances |
|---|---|---|---|
| step1 | log-scale orthogonal variance shares (channel / a1+a2+bin / 2-way / 3-way / replicate error) | 58.92% / 11.75% / 18.91% / 6.00% / 3.39% | exact (balanced design) |
| step1 | rank-1 R², 49×7 and 392×7 log surfaces | 0.9550, 0.9556 (rank-2 0.9902, 0.9829) | ±5e-5 |
| step1 | a1-slope ratio chan6/chan0 | 29.2 (0.0166 → 0.4846) | ±0.05 |
| step1 | bin-0 excess over quadratic trend, bins 1–7 | +0.9482 (×2.58), positive in 49/49 cells | ±5e-5 |
| step2 | rank-1 holdout R², full `a1` rows | mean 0.7950, worst 0.4033 | ±5e-5 |
| step2 | rank-1 holdout R², full `a2` columns | mean 0.9636, worst 0.9382 | ±5e-5 |
| step3 | cell-level cross-panel Pearson, 1372 shared cells | +0.0168 (Spearman +0.0324) | ±5e-5 |
| step3 | cell-level CV sharing, 1372 shared cells | +0.3822 (p < 5e-5); +0.1993 after `(a1,a2)` means removed | ±5e-5 |
| step3 | independent bin-0 excess over its own trend | −0.4539 (×0.64) | ±5e-5 |
| step3 | structural transfer of the replicated model to the independent panel | R² = −0.2768 (in-sample 0.5751) | ±1e-3 |
| step5 | rank-1 R² under 7 normalisations | log 0.9550; per-channel z 0.5681; linear 0.9987 | ±5e-5 |
| step5 | rank-1 null simulation, 500 surfaces | mean 0.5589 ± 0.0202, 0/500 ≥ observed, p = 0.002 | ±5e-3 |
| step5 | bootstrap rank-1 R² (4000 resamples of complete grid conditions) | 0.9556, 95% CI [0.9387, 0.9682] | Monte-Carlo noise ~±0.002 |
| step6 | six headline numbers, recomputed without `common.py` | all MATCH, `n_discrepancies = 0`, `errors = []` | 5e-5 |
| step7 | replicated log-SS share outside the shared region | 0.8127 (channels 4–6 = 0.675 of channel-marginal SS) | ±5e-4 |

### Reproducibility notes

- Permutation and bootstrap results are seeded (`numpy.random.default_rng` with fixed seeds per section),
  so re-runs reproduce them exactly. Two permutation results reported at slightly different p-values
  (`t08` = 0.02770 vs `t20` = 0.02685) come from different seeds with an **identical observed statistic**
  (matching to 1e-15); this is expected and is not a discrepancy.
- `step5_robustness.py` is deterministic: its numeric payload hash is stable across runs.
- `step6_verification.py` deliberately shares no code with `common.py`; it recomputes cell means, SVD
  rank-R² by two independent formulas, OLS slopes, Pearson, average-rank Spearman, polynomial fits and
  permutation nulls from scratch.
- Runtime: step1 ≈ seconds, step3 ≈ seconds, step5 ≈ 20 s, step6 ≈ 4 min on the reference machine.

## 4. Mapping from findings to the commands that produce them

| Finding | Commands |
|---|---|
| Finding 1 (single latent gain axis) | `step1_surface_geometry.py`, `step2_holdout_validity.py`, `step5_robustness.py` (null simulation, leave-one-out) |
| Finding 2 (magnitude-scale boundary) | `step5_robustness.py` (normalisation battery), `step6_verification.py` (`A_rank1_r2.variants_not_the_claim`) |
| Finding 3 (cross-panel level disagreement) | `step3_independent_panel.py`, `step7_domain_boundary.py`, `step6_verification.py` (`C_...`, `R1_...`) |
| Finding 4 (shared multiplicative noise scale) | `step3_independent_panel.py`, `step6_verification.py` (`D_...`, `R2_...`), `t15` for the fit-error null |
| Finding 5 (bin-0 boundary not reproduced) | `step1_surface_geometry.py`, `step3_independent_panel.py`, `step6_verification.py` (`E_...`, `R3_...`) |

## 5. Known failure modes preserved for the record

These were encountered during the run, are retained rather than hidden, and do not affect the final
numbers:

1. `step6_verification.py:617` raised `TypeError: cannot unpack non-iterable NoneType object` in the first
   version of the bin-0 trend check. Repaired; the final run reports `errors: []`.
2. An early route to a balanced-design interaction SS was abandoned (`/usr/bin/timeout` with
   `numpy.linalg.lstsq` on the full 4-way interaction design exceeded 2 minutes / was killed). The
   orthogonal balanced-design closed form in `step1` replaced it and is exact.
3. `step3_independent_panel.py` initially raised `KeyError: 'a1'` because the overview loop grouped by an
   abbreviation instead of the column name; the fix changed no comparison, merge key or R² baseline.
4. The channel-amplification *ratio* bootstrap is reported as unusable rather than as a defect: channel 0's
   slope sits at the noise floor, so its 95% CI spans zero. The robust substitutes (slope ordering,
   chan6/chan5 peak ratio) are given in `t20_robustness.json` and in the report.
5. The independent fit-error matrix is underpowered at n = 7 axis levels; its test is reported as a null,
   not as evidence against the shared noise axis.