# Reproduction Guide

**Task:** `Energy_02_flow_battery_operating_boundary`
**Authoritative report:** `output/Result.md`
**T0:** `2022-01-01`

This document is the reproduction trace for every registered finding in
`output/Result.md`. It states the exact command sequence, the environment, the
determinism guarantees, and the complete output inventory.

## 0. Post-verification corrections — re-run required

The analysis was independently re-derived by six `general-purpose` subagents
(see `output/verification/` and §7 of `Result.md`). Nineteen discrepancies were
reported, all confirmed against the raw data, and the affected code and prose
were corrected. **The corrected scripts supersede the original artifacts**; the
command sequence below regenerates the corrected set.

Corrections that live in code rather than prose:

| File | Correction |
|---|---|
| `run_analysis.py` | Dense 0.05-pp SoC sweep added (`ordering_dense_sweep.csv`) because the original 14-probe grid could not locate the ordering onset; SoC scan window widened 35–66 % → 25–76 %; genuine power-law fit computed and labelled separately from the log-rate polynomial; energy-weighted ratio now computed over an explicit 2–96 % window instead of being a hard-coded string; affine pair count reported at three tolerances with the tolerance and grid disclosed; sign-change bracketing corrected; peak-to-500 direction corrected; unitless ratio relabelled |
| `robustness_checks.py` | **Bug fix:** odd/even cycling variants were re-indexed as 1…n after subsampling, which exactly doubled their fitted slopes; each variant now carries its true `cycle_index`. SoC optimum scan window widened 48–52 % → 40–60 %. `CYC_IDX` defined (a `NameError` introduced during correction had left stale artifacts on disk before it was caught) |
| `verify_claims.py` | VERIFY 1 form labels corrected to match the code (the top-scoring entry used a mismatched-abscissa construction); two new blocks added — VERIFY 8 (permutation-tautology control against 1500 random monotone pairs) and VERIFY 9 (residual-share ranking) |

Two analysis bugs found and fixed during this pass are worth recording because
both would have silently inflated a robustness claim:

- The odd/even cycling slopes were reported as −0.2389/−0.2410 per cycle; the
  true values are **−0.1195/−0.1205**. The bug made the drift look 2× steeper
  than it is.
- The RMSE ≤ 1.0 pp SoC band was reported as 48.0–52.0 % because the scan window
  was ±2 pp around the probe; the true band is **43.23–55.06 %**.

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.11.2 |
| numpy | 2.4.6 |
| scipy | 1.17.1 (installed during the run; used only as an availability check, not for any reported number) |
| matplotlib | 3.11.2 (installed during the run; produces all figures) |
| pandas | 3.0.5 (present; not required by the analysis) |
| Platform | Linux (aarch64) |

Only `numpy` and `matplotlib` are imported by the analysis scripts. `scipy`,
`pandas`, `statsmodels`, and `sklearn` were checked for availability;
`statsmodels` and `sklearn` are absent and no reported result depends on them.
Every statistical procedure in this run is implemented directly on `numpy`
(ordinary least squares via `numpy.linalg.lstsq`, permutation tests by explicit
enumeration, bootstrap and block-bootstrap by explicit resampling). No fit,
test, or figure relies on a library that failed to load.

## 2. Inputs

Inputs are read from `data/` exactly as supplied and are never modified. Their
`sha256` digests, as recorded in `data_manifest.json` and re-verified at the
start of this run, are:

| File | Bytes | sha256 |
|---|---|---|
| `data/rate_dependent_voltage_curves.tsv` | 39477 | `ad25c508e67a6e6655e8ad26627138ce43f4b14a7c2d106586e2f45473595942` |
| `data/rate_tradeoff_summaries.tsv` | 580 | `8109ae000e3f6e91842d1b6d9f90ddcddc3cb737a668b74dce07f763158233b5` |
| `data/long_cycle_capacity.tsv` | 8837 | `f6febffe04971b6b0e280361937419153f40c57f71b255a18f65729215f9e853` |

Row counts: 852 + 16 + 600. No network access is required or used at any step.
No external dataset, precomputed result, or literature source enters the
analysis. Nothing in this reproduction depends on a source outside `data/`.

## 3. Command sequence

Run from the project root (the directory containing `data/` and `output/`):

```bash
# Step 1 - primary analysis (Sections 0-7)
python3 output/analysis/run_analysis.py

# Step 2 - adversarial verification of each claim
python3 output/analysis/verify_claims.py

# Step 3 - leave-out and alternative-definition robustness
python3 output/analysis/robustness_checks.py

# Step 4 - semantic delivery validation
python3 validate_output.py output
```

Each script is self-contained, reads only `data/`, writes only under `output/`,
and creates its own output directories if absent. Total runtime is under
5 minutes on a single core.

## 4. Determinism

All randomness is seeded:

| Script | Seed | Purpose |
|---|---|---|
| `run_analysis.py` | `20220101` | block bootstrap, moving-block permutation null |
| `verify_claims.py` | `7` | trend-preserving block-residual nulls |
| `robustness_checks.py` | `31337` | bootstrap intervals, resampling variants |

Re-running any script reproduces the same numbers bit-for-bit. Two design
choices are deliberate and worth recording:

- **The trend-preserving null replaces the naive shuffle.** An early version of
  the cycling change-point test shuffled cycle indices, which destroys the
  dominant trend and returns a spuriously small p-value. The reported null
  instead resamples the *residuals of the single-segment fit* in moving blocks
  of 25 and re-fits the two-segment model, preserving both the trend and the
  serial structure. This is why the reported change-point p-value is 0.345
  (not significant) rather than the naive 0.000.
- **The outlier variant preserves time order.** An early robustness variant
  trimmed the sorted capacity values, which scrambles the trend and produced a
  meaningless *positive* slope (+0.126). It was replaced by a residual-based
  exclusion (`|residual| > 2 sd`) that keeps every retained cycle in its
  original position.

Both corrections move results in the conservative direction and are visible in
the committed scripts.

## 5. Expected console output

`run_analysis.py` prints one block per section. The lines that carry registered
numbers are:

```
  offset charge_like s6->s7: +0.041520 V  spread=4.44e-16  exact=True
  strictly increasing charge V : True
  strictly decreasing dischg V : True
  strictly increasing gap      : True
  gap 0.1199 -> 0.6894 V (x5.750); discharge branch supplies 0.3780 V of the 0.5695 V growth
  coarse grid: all three orderings hold at 11/14 SoC probes
  dense sweep (1939 probes, 0.1-97.0 %): all three orderings hold at 1549/1939
    charge ordering  first fails at SoC never
    discharge ordering first fails at SoC 0.112 (fails up to SoC 97.012)
    gap ordering     first fails at SoC never
    discharge pair 1->2: violates from SoC 77.862 to 97.012
    discharge pair 2->3: violates from SoC 0.112 to 97.012
    discharge pair 5->6: violates from SoC 0.112 to 0.362
  efficiency = 100*Vdis/Vch @ SoC50 : RMSE=0.1837 pp, max|res|=0.2632 pp
  power/rate = Vdis           @ SoC50 : RMSE=0.00268 V, max|res|=0.00358 V
  exhaustive permutation (5040): identity assignment rank 1 (eff) and 1 (voltage)
  peak power density 330.00 at rate 300.0; dP/drate flips between the chord
    200.0->300.0 (+0.4970) and the chord 300.0->500.0 (-0.4000), i.e. at the pivot rate 300.0
  P(500)/rate = 0.5000 V, below the lowest measured discharge voltage (0.9915 V)
  genuine power law on the first 7 points: exponent 0.9249, log-space R2 = 0.998865
  affine family test: 13/42 pairs need |a-1|>0.05 with R2>0.97
    (tol=0 gives 18, tol=0.01 gives 16; 18 pairs reach R2>0.97 at any scale)
  energy-weighted ratio (window 2-96 %, explicit): RMSE=2.8664 pp
  best by AIC: quadratic_in_log10 (delta AIC vs linear-in-log10 = 31.55)
  loss (100-E) ~ rate^0.548 in log space, R2=0.99792
  whole-series slope = -0.11999 units/cycle, 95% block-bootstrap CI [-0.13461, -0.11169]
  2-segment best change point at cycle 286 : RSS 44893 -> 38068 (ratio 0.8480)
  trend-preserving block-residual null for the change point (400 resamples, block=25): p = 0.34500
  retention (cycle 600 vs cycle 1) = 91.48%
  lag-1 autocorrelation of first differences = -0.5346
```

`robustness_checks.py` now prints the corrected per-cycle variant slopes:

```
  drop odd cycles                   : n=300 slope=-0.12052 CI=[-0.14729,-0.11080]
  drop even cycles                  : n=300 slope=-0.11945 CI=[-0.13858,-0.11211]
  ...
  quadratic coefficient: point estimate -12.3061, bootstrap 95% CI [-15.8200, -8.8454]
  SoC values with RMSE_eff <= 1.0 pp: 43.25 to 55.05
```

`verify_claims.py` additionally reports the two audit blocks added after
verification:

```
VERIFY 8 -- is the 5040-permutation identity rank actually evidential?
  identity is the minimum-MSE pairing in 1500/1500 random INDEPENDENT monotone pairs
  -> rank 1/5040 is guaranteed by monotonicity, not evidence of shared construction
  informative statistic: identity RMSE 0.1837 pp vs runner-up 1.0627 pp (5.8x worse)

VERIFY 9 -- residual share: which rate drives the efficiency model misfit?
  linear-in-log10 : largest |residual| at rate 200.0 (-3.274 pp); rate 80 is rank 5/7
  linear-in-rate  : largest |residual| at rate 10.0 (+2.171 pp); rate 80 is rank 4/7
  -> the rate-80 value is NOT the largest contributor for either linear model
```

`verify_claims.py` prints seven verification blocks and ends with
`verification_summary.json written`. `robustness_checks.py` prints five blocks
(A–E) and ends with `robustness_summary.json written`.

## 6. Output inventory

### Analysis sources (`output/analysis/`)

| File | Role |
|---|---|
| `run_analysis.py` | primary analysis; produces every number in the report |
| `verify_claims.py` | independent adversarial re-derivation; alternative forms, hold-out, nulls |
| `robustness_checks.py` | leave-one-unit-out, alternative definitions, bootstrap intervals |

### Machine-readable result evidence (`output/results/`)

| File | Supports |
|---|---|
| `curve_inventory.csv` | structural inventory of all 14 curves |
| `grid_sharing.csv` | which curves share abscissae (42 pairs) |
| `offset_tests.csv` | the exact +0.041520 V charge 6→7 offset |
| `curve_metrics.csv` | Finding 1: per-setting voltages, gaps, ratios |
| `ordering_robustness_soc.csv` | Finding 1: 14 alternative SoC probes |
| `ordering_dense_sweep.csv` | Finding 1: 1 939-probe 0.05-pp sweep locating the ordering onset |
| `affine_family_tests.csv` | Finding 1: rejection of the pure additive family |
| `soc_reduction_scan.csv` | Finding 2: SoC scan locating the reduction point |
| `summary_linkage.csv` | Finding 2: the full efficiency/power linkage table |
| `linkage_permutation.json` | Finding 2: exhaustive 5040-permutation test |
| `linkage_leave_one_out.csv` | Finding 2: leave-one-rate-out linkage |
| `power_turnover.csv` | Finding 3: power series, implied voltages, dP/drate |
| `power_turnover.json` | Finding 3: turnover summary and window check |
| `efficiency_models.json` | Finding 4: model comparison, power-law fit |
| `efficiency_local_slopes.csv` | Finding 4: interval-by-interval slopes |
| `efficiency_leave_one_out.csv` | Finding 4: leave-one-rate-out efficiency |
| `cycling_segments.csv` | Finding 5: 12 windows with bootstrap slope intervals |
| `cycling_analysis.json` | Finding 5: slope, change point, nulls, autocorrelation |
| `cycling_resolution.csv` | Finding 5: 1/11 grid and the cycle-50/51 break |
| `boundary_checks.json` | Section 6 negative and boundary checks |
| `key_results.json` | consolidated key numbers for all sections |
| `verification_summary.json` | all nine verification blocks (incl. the permutation-tautology control and residual-share audit) |
| `verify_efficiency_forms.tsv` | alternative efficiency reductions compared |
| `verify_power_window.tsv` | Finding 3: implied voltage vs measured window |
| `../verification/verdict_*.json` | the six subagents' verbatim adversarial verdicts |
| `../verification/raw_*.md` | first-launch verdicts returned as prose (harness refused the schema tool for that role) |
| `robustness_summary.json` | all five robustness blocks (A–E) |
| `robustness_leave_one_setting_out.csv` | Finding 1: 21 leave-one-setting-out configs |
| `robustness_soc_optimum_fine.csv` | Finding 2: fine SoC optimum scan |
| `robustness_linkage_leave_one_rate_out.csv` | Finding 2: leave-one-rate-out detail |
| `robustness_cycling_variants.csv` | Finding 5: 7 cyclic-section variants |
| `robustness_power_peak_definitions.csv` | Finding 3: alternative peak definitions |
| `file_registry.json` | artifact-ID registry mapping F0xx → real files |

### Figures (`output/figures/`)

| File | Shows |
|---|---|
| `fig1_rate_voltage_family.png` | the 14 curves, charge and discharge panels |
| `fig2_ordering_and_gap.png` | branch voltages vs label order; gap growth; ratio |
| `fig3_summary_linkage_and_power.png` | efficiency linkage; power turnover; implied voltage vs window |
| `fig4_cycling.png` | the 600-cycle series with fits and the alternation histogram |

## 7. Semantic delivery mapping

| Contract role | Artifact |
|---|---|
| Authoritative findings | `output/Result.md` |
| Executed analysis source | `output/analysis/run_analysis.py`, `output/analysis/verify_claims.py`, `output/analysis/robustness_checks.py` |
| Inspectable result evidence | `output/results/*.csv`, `output/results/*.json`, `output/figures/*.png` |
| Reproduction trace | this document (`output/reproduce.md`) |
| Finding-to-evidence traceability | the index at the end of `output/Result.md` and `output/file_registry.json` |
| Independent verification | `output/verification/` (six subagent verdicts, two returning "refuted") |

## 8. Known gaps and honest limitations of this run

- **No unrounded source values.** The efficiency and power rows are published to
  2 decimal places; the efficiency residuals of Finding 2 (max 0.263 pp) are at
  the same order as plausible intermediate rounding and cannot be reduced
  further from the supplied data.
- **The SoC probe of 50 % is inferred, not supplied.** It was located by a scan
  over this same data (`robustness_soc_optimum_fine.csv`), so the reported
  permutation p-value for Finding 2 is mildly optimistic and is stated as an
  upper bound.
- **Scipy was not used for any reported figure.** It was installed mid-run and
  is listed for completeness; all reported statistics come from explicit
  `numpy` implementations.
- **No external verification was possible or attempted.** The task forbids
  merging external data, and no literature source first public strictly before
  `T0` was needed for background, methods, or alternative explanations.
  Consequently no external citation log is provided, because no external source
  contributed to any claim.