# Reproduction Document — State- and Composition-Dependent Long-Wavelength Emission Structure

This document is self-contained: it states the exact commands, the expected inputs,
and a mapping from each registered finding (in `output/Result.md`) to its analysis
script and machine-readable evidence TSV. No external data are used. Only `numpy`
and `pandas` are required (no `pip install`, no network access).

## Environment and layout

- Project root: `/data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project`
- Inputs (read-only, in `data/`):
  - `data/aggregate_state_emission_spectra.tsv` — 12,770 rows; 5 aggregate states × 4 design series.
  - `data/dilute_state_emission_spectra.tsv` — 2,253 rows; one dilute spectrum per real series.
  - `data/long_wavelength_response.tsv` — 20 rows; 5 composition fractions per series.
- Outputs (machine-readable evidence, in `output/evidence/`), produced by the scripts below.

## Exact commands (run from the project root)

```bash
cd /data/user/conversations/conv-8a8fd561-97b4-4af7-9671-12640fdbd642/project

# Core quantitative relationships (composition effect, state effect, dilute vs aggregate)
python3 output/analysis/core_analysis.py

# Robustness checks (FWHM threshold, peak definition, leave-one-series-out, band validation)
python3 output/analysis/robustness.py
```

Optional, also executed (figures were skipped because `matplotlib` is unavailable):

```bash
python3 output/analysis/explore_data.py      # exploratory data description
python3 output/analysis/make_figures.py      # attempts figures; writes figure_availability.tsv
```

Both core scripts print a per-row summary to stdout and write/overwrite the TSVs
listed below. Re-running is idempotent (it overwrites the evidence files).

## Evidence files produced

| TSV | Written by | Contents |
|---|---|---|
| `output/evidence/composition_effects.tsv` | core_analysis.py | per-series OLS slope/intercept/R², Spearman, sign-changes, quadratic curvature of response ~ composition |
| `output/evidence/state_effects.tsv` | core_analysis.py | per-series state_1→state_5 peak shift, FWHM change, max-intensity ratio |
| `output/evidence/state_by_state.tsv` | core_analysis.py | per-series, per-state peak/FWHM/max-intensity |
| `output/evidence/spectral_summary.tsv` | core_analysis.py (reads, pre-existing) | per-state peak, max, FWHM, integrated intensity, band_ratio |
| `output/evidence/dilute_vs_aggregate.tsv` | core_analysis.py | per-series dilute peak vs each aggregate-state peak, delta in nm |
| `output/evidence/dilute_summary.tsv` | core_analysis.py (reads, pre-existing) | per-series dilute peak/FWHM/integrated intensity |
| `output/evidence/robustness_peak.tsv` | robustness.py | argmax vs Gaussian-fit peak shift per state |
| `output/evidence/robustness_band.tsv` | robustness.py | full-band vs 900–1250 nm band peak/FWHM comparison per state |
| `output/evidence/robustness_fwhm.tsv` | robustness.py | FWHM at 50/30/20% of max intensity per state |
| `output/evidence/robustness_leaveout.tsv` | robustness.py | leave-one-series-out on the state-effect peak shift |
| `output/evidence/figure_availability.tsv` | make_figures.py | figure skip record (matplotlib unavailable) |

## Finding-to-evidence traceability map

| Finding (from Result.md) | Analysis script(s) | Evidence TSV(s) |
|---|---|---|
| Finding 1 — composition effect on long-wavelength response (positive, monotonic, near-linear) | `output/analysis/core_analysis.py` (section 1) | `output/evidence/composition_effects.tsv` |
| Finding 2 — state_1→state_5 peak shifts blue, intensity drops, (where defined) FWHM broadens | `output/analysis/core_analysis.py` (sections 2–3) | `output/evidence/state_effects.tsv`, `output/evidence/state_by_state.tsv`, `output/evidence/spectral_summary.tsv`; robustness: `output/evidence/robustness_leaveout.tsv`, `output/evidence/robustness_peak.tsv`, `output/evidence/robustness_band.tsv`, `output/evidence/robustness_fwhm.tsv` |
| Finding 3 — dilute-vs-aggregate offset is not consistent (conditional conflict) | `output/analysis/core_analysis.py` (section 3) | `output/evidence/dilute_vs_aggregate.tsv`, `output/evidence/dilute_summary.tsv` |
| Finding 4 — FWHM/peak-position metrics are method- and band-sensitive (applicability boundary) | `output/analysis/robustness.py` (checks 1, 2, 4) | `output/evidence/robustness_fwhm.tsv`, `output/evidence/robustness_peak.tsv`, `output/evidence/robustness_band.tsv`, `output/evidence/figure_availability.tsv` |

## Notes on interpretation

- All quoted numbers are taken from the TSVs above; re-running the scripts reproduces
  them verbatim (the scripts print the same values to stdout).
- `matplotlib` was unavailable in the analysis environment, so `figure_1_emission_structure`
  (`aggregate_spectra.png`) and `figure_2_response_composition` were skipped; this
  is recorded in `figure_availability.tsv` and is a reporting gap, not a data gap.
- FWHM is undefined (NaN) for 7 of 20 aggregate states (all 5 reference states,
  series_B state_5, series_C state_5); FWHM-based claims therefore apply only to
  series_A.
- The dilute-vs-aggregate comparison is defined only for series_A/B/C (the reference
  series has no dilute spectrum).
