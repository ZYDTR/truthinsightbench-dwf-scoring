# Reproduction

Task `Math_02_compositional_invariants`. Every number in `Result.md` is produced by
the scripts below from `data/` alone, with no network access and no external
scientific data. Results are written as JSON to `results/` and read back by
`analysis/s08_findings_registry.py`, so the findings registry is derived from
executed output rather than transcribed by hand.

## Environment

- Python 3.11.2; `pandas` 3.0.5, `numpy` 2.4.6, `scipy` 1.17.1,
  `scikit-learn` 1.9.1, `matplotlib` 3.11.2.
- `scipy`, `scikit-learn` and `matplotlib` were absent from the base image and
  were installed with `pip install scipy scikit-learn matplotlib`.
- Random seeds are fixed per stage (`s02`: 20240101/7, `s03`: 4242/99,
  `s05`: 20240101, `s06`: 20240920, `s07`: 808/555). Reruns reproduce to the
  precision reported.

## Order of execution

Run from the workspace root. Each stage is independent given `data/`, but the
registry stage requires the others.

```bash
python3 output/analysis/s01_inventory.py             # -> results/inventory.json, results/integrity.json
python3 output/analysis/s02_compositional_structure.py  # -> results/compositional_structure.json
python3 output/analysis/s03_covariate_association.py    # -> results/covariate_association.json
python3 output/analysis/s04_bias_scenarios.py           # -> results/bias_scenarios.json
python3 output/analysis/s05_figures.py                  # -> figures/F1..F5, results/figure_manifest.json
python3 output/analysis/s06_holdouts.py                 # -> results/holdouts.json
python3 output/analysis/s07_applicability_boundary.py   # -> results/applicability_boundary.json
python3 output/analysis/s08_findings_registry.py        # -> results/findings_registry.json, file_registry.json
```

All eight stages were executed for this report; console transcripts are kept in
`logs/s01.log` … `logs/s08.log`.

## What each stage does

| Stage | File | Role |
|---|---|---|
| s01 | `analysis/s01_inventory.py` | File inventory, row/column counts, id spaces, missingness, SHA-256 verification against `data_manifest.json`, and the closure/scale check on `relative_count`. |
| s02 | `analysis/s02_compositional_structure.py` | Depth scaling of richness, Taylor law on raw and closed scales, rarefaction curve, pairwise log-ratio variance with a half-sample split, CLR spectrum with pseudocount sensitivity, occupancy-abundance coupling. |
| s03 | `analysis/s03_covariate_association.py` | Rank associations of the covariate with depth, richness and CLR axes; partial rank correlation controlling depth; per-group estimates; variance shares; Kruskal-Wallis; leave-one-group-out; equal-effort rarefaction; permutation null; 1–99% trimming. |
| s04 | `analysis/s04_bias_scenarios.py` | Condition-ordering predicate and its exceptions; zero structure; paired scenario contrast on the 400 shared metric ids; condition coupling with and without zeros; bootstrap and permutation inference; exact rational structure; variance decomposition; feature↔metric id correspondence test. |
| s05 | `analysis/s05_figures.py` | The five supporting figures. |
| s06 | `analysis/s06_holdouts.py` | Complete-sample and group-stratified holdouts, feature holdout, scenario holdout with out-of-sample transfer, explicit comparison of the raw/transformed/ratio/zero-indicator explanation families, cross-validated monotonicity, covariate value domain. |
| s07 | `analysis/s07_applicability_boundary.py` | Median split, bootstrap CI on the segment difference, changepoint scan, per-group boundary tests (group-relative and global threshold), rarefied and common-feature robustness. |
| s08 | `analysis/s08_findings_registry.py` | Consolidates stages 1–7 into `results/findings_registry.json` and the native file registry `file_registry.json`. |

## Data integrity

`analysis/s01_inventory.py` recomputes SHA-256 for each file under `data/` and
compares against `data_manifest.json`. All three files match; see
`results/integrity.json` (`all_match: true`).

## Traceability

`results/findings_registry.json` holds the five registered findings with their
core numbers, robustness blocks, counterevidence and falsification proposals.
`file_registry.json` maps `ARTnnnn` handles to every file under `output/`. The
`ART` prefix is deliberately distinct from the finding ids `F001`–`F005` so a
file handle cannot be mistaken for a finding.

## Reset

The pipeline is idempotent: deleting `output/results/`, `output/figures/` and
`output/logs/` and rerunning the eight commands regenerates every artifact. No
stage mutates `data/`.