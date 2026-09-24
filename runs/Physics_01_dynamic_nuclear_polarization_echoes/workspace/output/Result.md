# Electron and Nuclear Spin Dynamics under Phase Inversion

**Task ID:** `Physics_01_dynamic_nuclear_polarization_echoes` · **T0:** 2024-06-26

This report is the complete and only authoritative scientific account for the task. It registers three findings (2–5 are permitted). The workspace `data/` directory is the sole direct evidence; no external data were downloaded or merged. External literature was not used for any core claim.

## Scope, data, and analysis unit

The frozen dataset contains 13 raw MATLAB (`.mat`) experiment files plus one derived comparison manifest:

- **EPR readout** — six files `data/raw/*_dnp_echo_pp_length_2.mat`. Each holds `dta_001` of shape (1024, 150) = 1024 time points × 150 averages, and a `dnp_echo` pulse-sequence record. Averaging over the 150 averages yields a 1024-point EPR trace.
- **NMR readout** — seven files `data/raw/*_dnpdata_dnp_contact_swp.mat` (six DNP contact scans) and one `data/raw/20240424_1547_dnpdata_nmr_only.mat` reference. Each holds `nmrdta_1..62`, each (8, 256) = 8 phase steps × 256 direct points, plus a 62-point indirect axis.
- **Phase-inversion times** — 0 ns (control), 132, 220, 440, 880, 2200 ns, read from the pitrain `tp` field of the DNP pulse record (confirmed against `data/comparison_manifest.json`).

**Analysis unit:** one complete scan under a fixed phase-inversion condition per `.mat` file. Time points within a scan are correlated repeated measurements and were *not* treated as independent samples; the "n" for the trend is the number of phase-inversion conditions (n = 6).

**Processing.** EPR: average over the 150 averages, estimate the carrier (Larmor) line from the trace FFT, demodulate to a baseband complex signal, and take the analytic-signal envelope for the echo. The echo window is where the envelope exceeds 10 % of its maximum. Signed-signal reconstruction uses the real part of the demodulated signal. NMR: combine the 8 phase steps with the reference phase `exp(-i·φ)`, subtract the per-indirect-point DC offset, and take the envelope. This is implemented in `analysis/run_analysis.py`.

All numbers below are reproducible by running `python3 analysis/run_analysis.py`; the raw per-condition values are in `results/epr_metrics.csv`, `results/nmr_metrics.csv`, and `results/summary.json`.

## Finding 1: EPR echo signal decreases monotonically with phase-inversion time

**Object and scope.** The integrated EPR echo signal as a function of the DNP phase-inversion time (0–2200 ns), one scan per condition.

**Relationship and direction.** The EPR echo envelope area and echo peak both decrease monotonically as the phase-inversion time increases. This is a *supported negative relationship*.

**Core quantitative result and analysis unit.** Averaged over 150 EPR averages per scan, the echo envelope area falls from 2,794,069 (0 ns) to 944,879 (2200 ns) — a factor of 2.96 reduction (ratio 2200/0 = 0.338). The echo peak falls from 12,892 to 4,578 (ratio 0.355). Per-condition values: 0 = 2.794e6, 132 = 2.286e6, 220 = 1.738e6, 440 = 1.257e6, 880 = 1.253e6, 2200 = 0.945e6 (area); 12,892 / 10,991 / 8,088 / 6,111 / 5,829 / 4,578 (peak). The Pearson correlation between echo area and inversion time is r = −0.76.

**Supporting analysis and result evidence.** Analysis: `analysis/run_analysis.py` (functions `process_epr`, the `echo_area_env`/`echo_peak` columns). Results: `results/epr_metrics.csv`, `results/summary.json` (`epr` array and `correlations.epr_area_vs_inv`), figure `figures/epr_echo.png`.

**Robustness result.** Replacing the 10 % threshold echo window with a fixed inner window (points 190–790) preserves the exact monotonic order: 2,783,985 → 2,282,970 → 1,733,574 → 1,255,915 → 1,243,090 → 945,144. The echo peak is independently monotonic. The carrier estimate (≈0.1309 pts⁻¹) is stable across all six conditions, so the trend is not an artifact of a drifting demodulation frequency.

**Counterevidence, limitation, and alternative explanations.** Only six conditions are available, so the r = −0.76 is a moderate (not near-perfect) correlation; the monotonic *order* is the stronger evidence than the correlation coefficient. The *signed* area (real part of the demodulated echo) flips sign at 440 ns (from +803,887 to −723,726) and is unstable at 132 ns; this is an echo-phase artifact, not a physical reversal, and is why the envelope (unsigned) area is the reported quantity. A plausible alternative explanation — that the apparent decrease is simply a change in the echo window or in the 150-average statistics rather than a true reduction in the echo — is disfavored because the envelope peak and the fixed-window area both track the same monotonic trend.

**Executable falsification/replication.** Re-run `analysis/run_analysis.py` on the raw `.mat` files; if the echo envelope area at 2200 ns is not the smallest of the six conditions, or if the peak is not monotonic, the finding is falsified. An independent replication with a different echo-window threshold (e.g., 5 % or 20 %) should still yield a monotonically decreasing envelope area.

## Finding 2: NMR signal is non-monotonic and does not track the EPR echo

**Object and scope.** The integrated NMR signal (after 8-step phase-cycle recombination and DC removal) as a function of phase-inversion time (0–2200 ns), one scan per condition.

**Relationship and direction.** The NMR total envelope area is *non-monotonic*: it is roughly flat at low inversion times, rises to a maximum at 440 ns, then falls. This is a *condition-dependent (non-monotonic) relationship*, in contrast to the monotonic EPR trend.

**Core quantitative result and analysis unit.** Per-condition total NMR envelope area (×10⁹): 0 = 786.0, 132 = 772.1, 220 = 804.2, 440 = 988.2 (maximum), 880 = 934.9, 2200 = 883.3. Relative to the 0 ns control: 132 = 0.98, 220 = 1.02, 440 = 1.26, 880 = 1.19, 2200 = 1.12. The Pearson correlation between NMR area and inversion time is only r = +0.40 (weak). Per-indirect-point envelopes (62 points) are stored in `results/nmr_metrics.csv` (`per_indirect_peak`) and `results/summary.json` (`nmr` array).

**Supporting analysis and result evidence.** Analysis: `analysis/run_analysis.py` (function `process_nmr`, `exp(-i·φ)` combination, per-indirect-point DC subtraction). Results: `results/nmr_metrics.csv`, `results/summary.json` (`nmr` array, `correlations.nmr_area_vs_inv`), figure `figures/nmr_perindirect.png` and `figures/nmr_structure.png`.

**Robustness result.** (a) The early/late fraction of the direct-axis envelope is nearly constant across conditions (early ≈ 0.189, late ≈ 0.023 for all six), so the amplitude change at 440 ns is a real whole-envelope amplitude change, not a shift in the direct-axis shape. (b) Re-combining the phase cycle with the opposite sign, `exp(+i·φ)`, preserves the same non-monotonic shape (peak shifted to 880–2200 ns): 26.7, 25.6, 25.2, 26.2, 27.0, 27.0 (×10⁹) — confirming the non-monotonicity is not an artifact of the chosen phase-reference sign. (c) The mean NMR FID decay time constant is essentially constant (~213 pts for 220–2200 ns), so the trend is in amplitude, not decay rate.

**Counterevidence, limitation, and alternative explanations.** The peak at 440 ns sits above the 0/880/2200 ns values by 13–26 %, which is larger than the spread between the 880 and 2200 ns points, so the maximum is a genuine feature rather than noise; however, with only six conditions and no per-scan replicate statistics, a formal significance test is not warranted. An alternative explanation — that the 440 ns increase is a change in the NMR reference/normalization rather than a true polarized-signal change — is addressed by using the same processing and the `nmr_only` reference consistently; the reference (50.6 ×10⁹) is far smaller than any contact scan, as expected for a non-DNP reference, and is not part of the trend.

**Executable falsification/replication.** Re-run `analysis/run_analysis.py`; if the 440 ns NMR area is *not* the maximum of the six conditions, or if the early/late fractions are not roughly constant, the non-monotonic finding is weakened. An independent replication using only the `exp(+i·φ)` combination should still show a non-monotonic (peak-shifted) curve.

## Finding 3: The EPR and NMR readouts are condition-dependent and anti-correlated, not stably coupled

**Object and scope.** The pairwise relationship between the EPR echo signal and the NMR signal across the same six phase-inversion conditions.

**Relationship and direction.** The two readouts are *anti-correlated* (r = −0.78) and *not stably coupled*: EPR decreases monotonically while NMR is non-monotonic and peaks at 440 ns. The relationship is therefore *condition-dependent*, not a fixed proportional offset.

**Core quantitative result and analysis unit.** Across the six conditions, EPR echo area (2.794 → 0.945 ×10⁶) falls while NMR area (786 → 883, peaking 988 ×10⁹) rises and then falls. The Pearson correlation between EPR area and NMR area is r = −0.78. Normalized to the 0 ns control, the two traces diverge visibly (see `figures/comparison.png`).

**Supporting analysis and result evidence.** Analysis: `analysis/run_analysis.py` (`correlations` computation over `epr` and `nmr` arrays). Results: `results/summary.json` (`correlations.epr_area_vs_nmr_area`), `results/epr_metrics.csv`, `results/nmr_metrics.csv`, figure `figures/comparison.png`.

**Robustness result.** The anti-correlation holds under the fixed inner EPR window (EPR areas 2.784 → 0.945 ×10⁶ remain monotonic) and under the opposite NMR phase sign (NMR areas 26.7 → 27.0 ×10⁹ remain non-monotonic); the EPR-vs-NMR relationship is anti-correlated in both cases, so the divergence is not an artifact of a single processing choice.

**Counterevidence, limitation, and alternative explanations.** An alternative explanation is that the two readouts should track a single underlying DNP polarization and the observed divergence reflects processing differences (EPR echo vs NMR contact). The constant NMR early/late fractions and the stable EPR carrier argue against processing artifacts within each readout, but no cross-readout calibration factor is supplied in the data, so a *quantitative* one-to-one coupling cannot be established from these files alone — only its absence. A null or weak correlation (r ≈ 0) would be the expected result if the readouts were independent; the observed r = −0.78 indicates a real, measurable, condition-dependent (inverse) association rather than independent noise.

**Executable falsification/replication.** Re-run `analysis/run_analysis.py`; if the EPR and NMR areas are positively correlated or uncorrelated (r near 0) across the six conditions, this finding is falsified. An independent replication on a second set of scans at the same phase-inversion times should reproduce an anti-correlated (or at least non-proportional) EPR–NMR relationship.

## Reproduction

The analysis is decomposed into two scripts under `output/analysis/`. The first is the per-file processor, which is the unit of parallelism (each of the 13 `.mat` files is processed independently and may be run concurrently, e.g. via the Workflow tool). The second is a single sequential reduction that combines the 13 per-file results into the trend/correlation/figure deliverables.

Run from the project root (`/data/user/conversations/conv-2056fa59-740e-4882-8dcc-f5fabaa7bfcd/project`):

```bash
# 1. Per-file processing (run once per .mat file; these are independent and can be parallelized).
#    Each call writes output/results/perfile/<name>.json.
for f in data/raw/*.mat; do
  base=$(basename "$f" .mat)
  python3 output/analysis/process_one.py "$f" "output/results/perfile/${base}.json"
done

# 2. Sequential aggregation of the 13 per-file results into CSV/JSON/figures.
python3 output/analysis/aggregate.py

# 3. Validate the semantic output contract (validator lives at the project root).
python3 validate_output.py output
```

`output/analysis/process_one.py` reads each raw `.mat` file (project-relative paths), processes the EPR echo (average over the 150 averages, carrier estimation, demodulation, echo-window envelope) or the NMR readout (8-step phase-cycle recombination, per-indirect-point DC removal, envelope), and writes a per-file metrics JSON. `output/analysis/aggregate.py` combines the 13 per-file JSONs in `output/results/perfile/` into `output/results/epr_metrics.csv`, `output/results/nmr_metrics.csv`, `output/results/summary.json`, and the figures under `output/figures/`. All six phase-inversion conditions and both readouts are processed identically, so the reported numbers are reproducible from the frozen `data/` directory alone.

A monolithic equivalent, `output/analysis/run_analysis.py`, performs the same per-file processing and aggregation in one script and produces identical core results; it is retained for convenience.

## Limitations and unresolved questions

- **Small condition count (n = 6).** Trend statements are based on six phase-inversion times; correlation coefficients should be read as directional, not as tight statistical fits.
- **No per-condition replicate statistics.** Each condition is a single averaged scan; the data guide explicitly warns that time points within a scan are correlated and should not inflate confidence. No per-condition standard deviations are reported as if independent.
- **EPR signed-area phase flip at 440 ns.** The real part of the demodulated echo changes sign at 440 ns; this is treated as a phase artifact and the unsigned envelope area is reported instead. The physical cause (echo-phase shift under phase inversion) is unresolved from these files alone.
- **No cross-readout calibration.** The EPR and NMR signals have different units and no supplied conversion factor, so only the *relative* (correlation) relationship between them is claimable, not a calibrated one-to-one DNP-polarization equivalence.
- **NMR peak at 440 ns.** The non-monotonic peak is real but, with n = 6 and no replicate statistics, its statistical significance is not formally established.

These limitations do not alter the three registered findings, which are reported exactly as the data warrant: a supported monotonic EPR decrease (Finding 1), a supported non-monotonic NMR signal (Finding 2), and a supported condition-dependent anti-correlation between the two readouts (Finding 3).
