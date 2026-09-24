# Reproduction Guide

Task: `Energy_04_biohybrid_operating_tradeoffs`
Companion to the authoritative report `output/Result.md`.

## 1. Requirements

- Python 3.11
- `numpy` 2.4.6, `pandas` 3.0.5, `matplotlib` 3.11.2
- `scipy` is **not** required. It was unavailable in the original run
  environment, so every statistic (OLS, Pearson and Spearman correlation,
  permutation tests, bootstrap resampling) is implemented directly with
  `numpy`/`pandas`. See `output/results/findings.json` -> `_meta`.

## 2. Inputs

Read-only, from `data/`:

| File | Rows x Cols | sha256 (must match `data_manifest.json`) |
|---|---|---|
| `data/time_response_curves.tsv` | 1367 x 7 | `50c44917c225aba66a15569aa57be74b9242d1baee1c6afeb4c8bf8652dd622d` |
| `data/perturbation_time_response.tsv` | 4 x 7 | `ce6bbd0b5c7ffc78d6669f96a5e228d75524ed3344d5c40aec7a93b3067cf43d` |
| `data/dose_response.tsv` | 3 x 3 | `9b3b2857de0973e0fd0e6af27f47bfd68056841553addf54561aa650b8a8488e` |

The script re-verifies these hashes at run time and records the outcome in
`output/results/findings.json` -> `data_integrity`. No external data is
downloaded, substituted, or merged.

## 3. Run

From the project root:

```
python3 output/analysis/analysis.py
```

Deterministic: a single fixed seed (`RNG_SEED = 20230919`) governs all
permutation and bootstrap draws, so intervals reproduce exactly.

## 4. Outputs

- `output/results/findings.json` — complete machine-readable result tree, keyed
  by finding (`finding1_regimes`, `finding2_path_dependence`,
  `finding3_coupling`, `finding4_fitted_columns`,
  `finding5_dose_and_perturbation`, plus `fano_by_regime`,
  `tail_decline_order_null`, `data_integrity`, `_meta`).
- `output/results/table_regimes.tsv` — the three regimes with ranges and
  time-value support.
- `output/results/table_matched_time_branches.tsv` — ascending vs descending
  branch values at matched `time`.
- `output/results/table_fit_vs_raw.tsv` — raw vs fitted distances and
  descending-sort comparison.
- `output/results/table_headline_numbers.tsv` — one row per headline number.
- `output/figures/fig1_regime_structure.png` — regimes in row order vs the
  labelled `time` axis, plus the sweep in both keyings.
- `output/figures/fig2_hysteresis.png` — ascending vs descending branch on a
  common log-time grid.
- `output/figures/fig3_coupling_by_regime.png` — pairwise correlations of
  levels vs first differences, per regime.
- `output/figures/fig4_raw_vs_fitted.png` — raw vs fitted channels.
- `output/figures/fig5_dose_perturbation.png` — dose scan and perturbation
  trajectories.

## 5. Code map

All analysis lives in `output/analysis/analysis.py`:

| Function | Produces |
|---|---|
| `part0_load` | hash verification, shapes, missingness |
| `part1_regimes` | Finding 1 (regime segmentation, boundary sensitivity, leave-one-regime-out) |
| `part2_pathdependence` | Finding 2 (matched-time branches, log-time loop areas, jackknife, apex location) |
| `part3_coupling` | Finding 3 (level / first-difference / rank / detrended coupling, Fano factors, order-shuffle null) |
| `part4_fitted` | Finding 4 (raw-vs-fit association, window scan, monotonicity permutation test, smoother-axis test, double-counting) |
| `part5_dose` | Finding 5 (dose models + leave-one-setting-out, perturbation trajectory models, bootstrap from supplied SDs) |
| `make_figures` | the five PNG figures |
| `write_tables` | the four TSV tables |

## 6. Determinism and known failure retained

The block-bootstrap variant of the tail-decline null was found to be invalid
(concatenating randomly ordered 25-row blocks destroys the trend by
construction). It is retained in the output as a documented failure under
`tail_decline_order_null.block_bootstrap_note` and was replaced by a plain
row-shuffle null plus a segment-persistence check. See `Result.md` section 5.

## 7. Independent verification

`output/verification/` holds a second, independent pass over the same data,
produced by eight verifier agents run through the native workflow tool. Each
verifier wrote its own scripts and recomputed from the raw TSVs **without
executing `analysis.py`**, and additionally read the PNG figures back at pixel
level to check they represent the data faithfully.

To inspect the verification without re-running the workflow:

```
python3 output/verification/verifier_artifact_consistency.py   # 632 claims, 4 tables + findings.json
python3 output/verification/verifier_figures.py                # figure fidelity + PNG pixel read-back
python3 output/verification/verifier_finding1.py               # per-finding recomputation
python3 output/verification/verifier_finding3.py
python3 output/verification/verifier_finding5.py
```

Verifier outputs are preserved unedited alongside the scripts
(`*_results.json`, `verifier_figures_out.json`, `verifier_full_output.txt`).

Verification found three discrepancies, all corrected in `Result.md` and
`analysis.py` (see `Result.md` section 6): a leave-one-regime-out text/value
pairing error, an SD-count error in Finding 5, and a fig-4a title that
overstated the plotted range. It found no discrepancy that changed a registered
conclusion.