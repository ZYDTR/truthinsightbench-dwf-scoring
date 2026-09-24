# Reproduction Guide

Task ID: `EarthScience_01_glacial_fluvial_erosion`

Every number in `output/Result.md` is produced by the scripts listed here. All scripts read only
`data/` inside this workspace and write only inside `output/`. No network access is required or used.

## Environment

```
python3 --version        # 3.11.2
pip install --user scipy statsmodels matplotlib
```

Preinstalled: `pandas 3.0.5`, `numpy 2.4.6`. Installed during this run: `scipy 1.17.1`,
`statsmodels 0.15.0`, `matplotlib 3.11.2`.

## Execution order

Run from the project root. The first step builds the shared frame every later step consumes.

```bash
# 1. Shared consolidated frame (reads data/raw/*.tsv, writes output/analysis/common/)
python3 output/analysis/common/prepare_data.py

# 2. Cross-cutting core analysis: coverage, model comparison, stratum slopes, figures
python3 output/analysis/w0_main/main_analysis.py

# 3. Falsification tests: thickness diagnostic, within-study slopes, common support, compiler effect
python3 output/analysis/w0_main/decisive_tests.py

# 4. Coupling audit: unit identity, thickness collapse, bottom-up reconstruction from raw columns
python3 output/analysis/w1_coupling/coupling_audit.py

# 5. Variance decomposition and explanatory-power ranking
python3 output/analysis/w2_variance/w2_variance.py

# 6. Method-stratified scaling with bootstrap, Theil-Sen, leave-one-compilation-out, thresholds
python3 output/analysis/w3_scaling/w3_scaling.py

# 7. Process-category contrast, method- and scale-matched
python3 output/analysis/w4_process/w4_process_analysis.py

# 8. Data inventory, hashes, missingness, independence structure
python3 output/analysis/w5_audit/audit_data.py

# 9. De-biased scaling from records with independent clocks
python3 output/analysis/w6_decoupled/w6_decoupled.py
```

Scripts 2, 3, 4 and 6 perform bootstrap resampling and are CPU-bound; on a loaded machine each
may take several minutes. All use fixed seeds, so repeated runs reproduce the same numbers.

## Determinism and seeds

| Script | Randomness | Seed |
|--------|-----------|------|
| `w0_main/main_analysis.py` | 2000-draw row bootstrap for stratum slope CIs; 2000-draw bootstrap for matched comparisons | `np.random.default_rng(20240919)` |
| `w0_main/decisive_tests.py` | none (deterministic statistics only) | — |
| `w1_coupling/coupling_audit.py` | none | — |
| `w2_variance/w2_variance.py` | none (closed-form statistics) | — |
| `w3_scaling/w3_scaling.py` | 2000-draw row and cluster bootstraps | fixed seed, recorded in `w3_scaling/results.json` |
| `w4_process/w4_process_analysis.py` | bootstrap CIs on matched differences | fixed seed, recorded in `w4_process/results.json` |
| `w6_decoupled/w6_decoupled.py` | see script header | see script header |

## Analysis-unit and filter conventions

- **Analysis unit:** one published erosion-rate estimate per row. Rows sharing a `Reference`,
  `Compilation`, `Locality` or coordinate pair are *not* independent replicates; every
  inferential result is reported with its cluster count as well as its row count, and the large
  strata are additionally bootstrapped by `Reference` cluster in `w3_scaling`.
- **Log space:** rates and intervals span orders of magnitude, so all regressions are fitted on
  `log10(Erosion rate)` against `log10(Time interval)`. Slopes are reported in dex per dex.
- **Filter rule (explicit):** `Erosion rate > 0` and `Time interval > 0`. This removes 4
  non-positive rates and every row with a missing interval. Per-stratum counts of what was
  removed are in `w5_audit` and `w1_coupling`; nothing is removed silently.
- **No winsorization is applied to the headline results.** Winsorized and trimmed variants are
  computed as robustness checks in `w3_scaling/table_outlier_treatments.csv` and reported as such.

## Output map

| Role | Location |
|------|----------|
| Authoritative findings | `output/Result.md` |
| Reproduction trace | this file, plus the `# Reproducibility` section of `output/Result.md` |
| Executed analysis source | `output/analysis/common/`, `output/analysis/w0_main/`, `w1_coupling/`, `w2_variance/`, `w3_scaling/`, `w4_process/`, `w5_audit/`, `w6_decoupled/` |
| Inspectable result evidence | `results.json` in each workstream directory; `table_*.csv` and `fig_*.png` alongside them |
| Provenance | `output/provenance/DATA_SOURCE_LOG.md` |

## Known gaps and honest status

- `output/analysis/w1_coupling/` was originally assigned to a parallel sub-agent that died on a
  transport error (`API Error: Connection closed mid-response`). Rather than lose the workstream,
  the coupling audit was re-implemented and executed by the lead agent as
  `w1_coupling/coupling_audit.py`; its results are first-class evidence in Finding 1. The real
  error is preserved here and in `output/analysis/FAILURES.md`.
- `data/raw/beryllium.tsv` (70 rows) is a dissolved/colloidal beryllium analyte table with no
  erosion-rate column. It is inventoried and then excluded from the erosion analysis; see
  `w5_audit`.
- The auxiliary per-location files (`slope_*.tsv`, `prior compilations/CHELSEA/precipitation_*.tsv`,
  `lithology_*.tsv`) are **not** row-aligned to the two main tables. Their alignment is diagnosed
  in `w5_audit`.
