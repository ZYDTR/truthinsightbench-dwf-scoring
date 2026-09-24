# Frequency- and Time-Domain Boundaries of Controlled-Dynamics Forecasts

**Task ID:** `Physics_03_controlled_dynamics_forecast` · **T0:** 2021-04-01 · **Report date:** 2026-09-23

This report is the sole authoritative account of the registered findings. It quantifies agreement between paired *reference* and *forecast* channels of a controlled-dynamics model, **separately** in the frequency domain and the full time-domain trajectory, and tests whether the two evidence channels agree, conflict, or shift by condition.

## Scope, data, and analysis unit

- **Data (the only direct evidence for core claims):** six tabular files under `data/normalized_views/` — three frequency-response scans (`control_a_frequency_response.tsv`, `control_b_frequency_response.tsv`, `signed_control_frequency_response.tsv`) and three complete trajectories (`trajectory_condition_01.tsv`, `condition_02.tsv`, `condition_03.tsv`, each 15,000 points) — plus a large auxiliary workbook `data/source_data/all_source_data_deidentified.xlsx` (65 sheets).
- **Independent analysis unit (per `data_manifest.json`):** one complete ordered control scan or one complete trajectory. Adjacent control points / time points are *correlated* observations, so each scan and each trajectory is the unit of comparison — not individual points.
- **Paired channels are not independent experiments** (manifest warning): `reference_*` and `forecast_*` are outputs of the same run, so agreement metrics below measure the *forecast's* fidelity to the reference, not an A/B experiment.
- **Data integrity (verified, not assumed):** `output/integrity_check.py` recomputes SHA-256 and byte-size for all 8 data files (all match `data_manifest.json`) and cross-checks each normalized-view TSV against its source workbook sheet cell-by-cell — all six TSVs match losslessly (0 mismatched cells; see `output/integrity_check.json`).

## Methods

The four analysis scripts below were *executed* (not merely written) with `python3`; their outputs are the machine-readable result artifacts cited in each finding.

| Purpose | Script (executed) | Result artifact (machine-readable) |
|---|---|---|
| Frequency-domain agreement | `analysis_frequency.py` | `frequency_results.json` |
| Time-domain trajectory agreement | `analysis_trajectory.py` | `trajectory_results.json` |
| Cross-channel + robustness | `analysis_robustness.py` | `robustness_results.json` |
| Reference-vs-forecast figures | `figures.py` | `figures/freq_reference_vs_forecast.svg`, `figures/traj_reference_vs_forecast.svg`, `figures/error_bar.svg` |

Metrics: exact matches (`|ref − fc| < 1e-6`), mean/median/max absolute error, RMSE, mean relative error, bias, Pearson correlation, and normalized RMSE (RMSE / std of reference). Frequency values live on a coarse grid (multiples of 1/32 of a frequency unit), so "exact" agreement means agreement to the grid bin.

## Finding 1: Frequency-domain agreement is high and near-exact on the coarse frequency grid across all three controls

- **Object and scope:** the three frequency-response scans; agreement measured as reference vs forecast frequency.
- **Relationship / direction:** strong positive agreement; forecast reproduces the reference frequency bin in most cases, with small residual errors.
- **Core quantitative result (analysis unit = one complete scan):**
  - `control_a`: 21 pairs, **16/21 exact (76.2%)**, MAE 0.0465, max abs error 0.1953, RMSE 0.0953, mean relative error 0.305%, bias +0.0465.
  - `control_b`: 16 pairs, **8/16 exact (50.0%)**, MAE 0.0977, max abs error 0.1953, RMSE 0.1381, mean relative error 0.713%, bias +0.0732.
  - `signed_control`: 10 pairs, **3/10 exact (30.0%)**, MAE 0.0146, max abs error 0.0366, RMSE 0.0189, mean relative error 0.491%, bias −0.0049.
- **Supporting evidence:** `analysis_frequency.py` → `frequency_results.json`; visual: `figures/freq_reference_vs_forecast.svg`.
- **Robustness:** exact-match counts are threshold-robust (identical at 1e-6 and 1e-3 for all three; see `robustness_results.json`). Errors are bounded by ~6 grid bins (0.1953) and relative errors are ≤ 0.713%.
- **Counterevidence / limitation / alternatives:** agreement is *not* perfect (30–50% of points miss the exact bin in two of three controls), and the differing points cluster where the forecast lags the reference (e.g., `control_a` indices 10–18, `control_b` indices 0,5–9,12,14). An alternative explanation is that the frequency estimator quantizes to a coarse grid, so small sub-bin offsets round to a neighbouring bin; the data alone cannot distinguish quantization from genuine model error.
- **Falsifiable replication:** re-run the same forecast on an independent control scan with a *finer* frequency grid; if exact-match fraction stays ≈76% while relative error grows, the effect is quantization; if exact-match fraction falls, the frequency estimator is unstable.

## Finding 2: Time-domain agreement is moderate and degrades monotonically across the three trajectory conditions, with all trajectories sharing an identical initial condition

- **Object and scope:** the three complete trajectories (15,000 points each, `t` from 0 to 37.4975); agreement measured as reference_response vs forecast_response.
- **Relationship / direction:** positive agreement that *degrades with condition index*; all three trajectories start from the same initial response (first point identical in all three).
- **Core quantitative result (analysis unit = one complete trajectory):**
  - `condition_01`: RMSE 0.3067, max abs error 1.2490, Pearson r 0.9595, normalized RMSE 0.2819.
  - `condition_02`: RMSE 0.6086, max abs error 2.6922, Pearson r 0.8595, normalized RMSE 0.5115.
  - `condition_03`: RMSE 1.0908, max abs error 4.7560, Pearson r 0.6687, normalized RMSE 0.7437.
- **Supporting evidence:** `analysis_trajectory.py` → `trajectory_results.json`; visual: `figures/traj_reference_vs_forecast.svg`.
- **Robustness:** recomputing each trajectory directly from the raw TSV (independent of the cached JSON) reproduces identical RMSE / max-abs-error values, and the per-condition ranking is invariant to leave-one-condition-out (see `robustness_results.json`, `leave_one_condition_out`).
- **Counterevidence / limitation / alternatives:** agreement is *correlation*, not identity — Pearson 0.67 for `condition_03` means the forecast tracks the *trend* but not the *amplitude* (last-point forecast −0.307 vs reference −2.442 for `condition_03`). An alternative reading is that the forecast is a stable attractor/phase-locked response that diverges in amplitude from the reference as the driven trajectory grows, rather than a random walk.
- **Falsifiable replication:** run the forecast on a *held-out* condition (a new drive/parameter setting, not a random time-point split — per the objective) and check whether RMSE grows with condition index as here; if RMSE does not grow, the degradation is condition-specific rather than intrinsic to the horizon.

## Finding 3: The frequency and time channels CONFLICT — high frequency accuracy coexists with low-to-moderate full-trajectory accuracy (boundary finding)

- **Object and scope:** the two evidence channels as a pair (frequency scans vs complete trajectories).
- **Relationship / direction:** *conflict / boundary*, not agreement. The frequency channel shows near-exact, small-error agreement (Finding 1), while the time channel shows substantial, condition-dependent divergence (Finding 2).
- **Core quantitative result:** frequency MAE spans 0.0146–0.0977 with many exact matches; time RMSE spans 0.3067–1.0908 with max absolute errors up to 4.7560 and Pearson 0.67–0.96. Because the two channels use different units (frequency vs response), the physically meaningful statement is a *contrast in agreement character*: the forecast nails the spectral feature but does not faithfully reproduce the full waveform.
- **Supporting evidence:** `analysis_robustness.py` → `robustness_results.json` (see `cross_channel` block, `verdict: "CONFLICT"`); visual: `figures/error_bar.svg` (frequency MAE bars vs trajectory RMSE bars on one axis).
- **Robustness:** the contrast is robust — it is not an artefact of a single condition: the worst time condition (`condition_03`, RMSE 1.09) is still far less accurate in the time domain than any of the three frequency scans (max frequency MAE 0.0977).
- **Counterevidence / limitation / alternatives:** the two channels measure different things (a single spectral feature vs the whole trajectory), so "conflict" here is a *boundary* statement, not a contradiction. The manifest explicitly warns that "good frequency prediction does not by itself establish full-trajectory accuracy or a correct microscopic mechanism"; this finding is the direct data-level realization of that warning. It does *not* establish the microscopic mechanism, and a mechanism claim is out of scope.
- **Falsifiable replication:** produce a forecast that is *frequency-accurate but trajectory-accurate* (RMSE comparable to frequency MAE) on a held-out condition; if such a forecast exists for the same model class, the boundary is not intrinsic to this model.

## Finding 4: Robustness — the condition ranking is stable, exact frequency matches are threshold-robust, and `signed_control` is the most frequency-accurate control

- **Object and scope:** robustness of the above conclusions to re-sampling (leave-one-condition-out) and to the exact-match threshold.
- **Relationship / direction:** the conclusions are *stable*, not fragile.
- **Core quantitative result:**
  - **Leave-one-condition-out (time domain):** global ranking by RMSE and by max-abs-error is `condition_01 < condition_02 < condition_03` in every leave-one-out subset; no ranking change (see `robustness_results.json`, `leave_one_condition_out`, `per_condition_ranking_changes: false`).
  - **Threshold sensitivity (frequency):** exact-match counts are unchanged at 1e-6 vs 1e-3 for all three controls (`control_a` 16, `control_b` 8, `signed_control` 3). At 1e-1, `signed_control` rises 3→10 (all 10 within 0.1, consistent with its small MAE 0.0146), while `control_a`/`control_b` remain 16/8.
  - **Control comparison:** `signed_control` mean relative error 0.00491 is *below* the average of `control_a` (0.00305) and `control_b` (0.00713), i.e. 0.00509 — the signed-control scan is the single most frequency-accurate of the three.
- **Supporting evidence:** `analysis_robustness.py` → `robustness_results.json`.
- **Robustness:** these checks recompute from raw TSVs (not the cached JSON), so they are independent of the first-pass computation.
- **Counterevidence / limitation / alternatives:** the threshold 1e-6 is arbitrary; the 1e-1 jump for `signed_control` shows that "exactness" is threshold-dependent for the best-scan. The leave-one-out here is over *conditions*, not over *time points* — the objective explicitly forbids random time-point splits as a substitute for held-out conditions, and none were used.
- **Falsifiable replication:** vary the exact-match threshold across an ordered grid (e.g., 1e-2, 1e-3, 1e-4) and confirm the exact-match *ordering* of controls is invariant; if the ordering flips, the "most accurate control" claim is threshold-dependent.

## Reproduction

To reproduce this analysis:
1. `python3 integrity_check.py` → verifies data hashes/sizes and TSV↔sheet fidelity (`integrity_check.json`).
2. `python3 analysis_frequency.py` → `frequency_results.json`.
3. `python3 analysis_trajectory.py` → `trajectory_results.json`.
4. `python3 analysis_robustness.py` → `robustness_results.json`.
5. `python3 figures.py` → `figures/*.svg`.

All scripts use only `pandas`/`numpy` (and the stdlib for figures); run from the `output/` directory.

## Limitations

- **Paired, not independent:** reference/forecast are outputs of the same run; agreement metrics are a single-model fidelity check, not an independent comparison.
- **Unit mismatch:** frequency and response live in different units; the cross-channel "conflict" is a contrast in agreement character, not a unit-normalized error ratio.
- **De-identified labels:** controls/conditions are anonymous (`control_a/b`, `signed_control`, `condition_01/02/03`); the report makes no claim about their physical identity.
- **No mechanism claim:** the data establish accuracy boundaries, not a microscopic mechanism (per the manifest warning).

## Data provenance

All data come from the benchmark workspace `data/` (65-sheet workbook + 6 TSV views). SHA-256 and byte-size of all 8 files match `data_manifest.json`; each normalized-view TSV matches its source workbook sheet losslessly. No external dataset was downloaded, merged, or used as direct evidence.
