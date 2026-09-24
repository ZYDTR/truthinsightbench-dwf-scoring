# w3_scaling — method-stratified erosion-rate / timescale scaling

Script: `w3_scaling.py` (run with `python3 w3_scaling.py`, exit 0, ~5 min dominated by
the 2000-resample bootstraps; deterministic seed 20260919).
Data: `data/raw/glacial_erosion_Earth.tsv` (427 rows) and
`data/raw/nonglacial_erosion_Earth.tsv` (6288 rows). No external data used.

Row filter for every fit: `Erosion rate > 0 AND Time interval > 0` (non-null). This passes
427/427 glacial and 6128/6288 nonglacial rows. Model: OLS `logER ~ logTI`, slopes in
dex per dex. Tables in `tables/`, figures in `figures/`, all numbers in `results.json`.

## Headline result: two of the five "large" strata are not identifiable, and the
## flagship -1 result is a definitional artefact

### 1. nonglacial / Cosmogenic detrital (n=3470, slope -0.9950): the -1 is nearly an identity

The pooled slope is -0.995 (se 0.0008, R^2 = 0.9979) and every robustness check returns
-1.0 to 3 decimal places. But this is not independent evidence of a scaling law:

- Within each Reference, `ER * TI` is nearly constant. Across the 155 References with
  >= 3 rows, the median within-Reference max/min ratio of `ER*TI` is **1.05**; 127/155 have
  ratio < 1.10. That is the signature of `ER = Exhumation / TI` computed from a single
  per-site exhumation estimate, so `ER` and `TI` are not independently measured.
- The stratum is effectively **one compilation** (OCTOPUS 3457 rows; n=13 for 'This study'),
  and the sole fittable compilation gives slope -0.9952.
- The between-Reference share of variance in logTI is 0.80, so 80% of the x-variation is
  a level difference between References, not within-site time structure.
- The within-Reference fixed-effects slope (-1.0057, se 0.0008) is the same, because
  demeaning removes only the level, and `ER*TI = const` survives demeaning.

Verdict: the -1 here reproduces the arithmetic of dividing a per-site total by a per-site
exposure age. It should **not** be reported as an empirical erosion-versus-timescale
scaling. The mechanical-coupling reading ("slope -1") is *consistent* with the number but
is not what the regression demonstrates. The tell is that R^2 = 0.998 coexists with a
formal "inconsistent with -1" verdict (`d_from_minus1_se` = 6.36 at n=3470): the fit is
precise to 8e-4 dex yet still 6 sigma from the identity it supposedly measures. The
residuals are nonzero only from per-sample scatter in the underlying exhumation estimates.

### 2. nonglacial / Volumetric (n=2583, slope +0.117): slope is NOT identified within Reference

- Only **6 distinct Time-interval values** exist in 2583 rows (TI = 5 yr for 59%,
  TI = 0.00274 yr = 1 day for 38%). TI is set by the compilation (Chen et al. 2022 uses
  TI = 5; GloRiSe uses TI = 1 day), not measured per site.
- **0 of 64 References** can support a per-Reference slope (all have < 3 distinct TI values).
- Within-Reference fixed effects: slope **0.0025** (se 0.221, p = 0.99), R^2_within ~ 5e-8;
  only 3 References (178 rows, 6.9% of the data) have any within-Reference x-variation at all.
- The pooled +0.117 is **entirely a between-compilation comparison**: TI=5 rows (Chen) have
  median ER 0.094 and TI=1-day rows (GloRiSe) have median ER 0.041. Within compilation,
  the GloRiSe-only slope is **-0.083** (se 0.030, p = 0.006) — the *opposite sign* to pooled.

Verdict: the pooled +0.117 is a between-dataset composition contrast (two source datasets
with different assumed integration windows), not a timescale scaling. Any statement of the
form "nonglacial volumetric erosion increases with timescale" is unsupported.

### 3. Glacial / Thermochronometric (n=68, slope -0.652): sign is robust, magnitude is not

Same disease as case 2 but mild: TI = 5 yr for 69% of rows, so the between-Reference share
of logTI variance is 0.996, and only **1 Reference** (6 rows, 11% of data) has
within-Reference x-variation; the FE slope is -0.79 but with se 0.77 (p = 0.31). The
per-compilation slope is -0.654 (identical to pooled) because 66/68 rows are one
compilation. Pooled significance and magnitude are therefore driven by cross-site level
differences, not by within-site time evolution.

### 4. Glacial / Volumetric (n=233, slope -0.0729): the one genuinely identified but weak case

- Most heterogeneous stratum: 98 References, 107 distinct TI values, max single-TI share 6%,
  11 compilations.
- Pooled slope -0.0729 (se 0.0253, classical CI [-0.123, -0.023], p = 0.0044, R^2 = 0.035).
- Within-Reference FE slope is **-0.120** (se 0.076, p = 0.11) and loses significance.
- The pooled significance is **not HC3-fragile** and **not LOCO-fragile** (LOCO range
  [-0.081, -0.055], all negative, sign never flips).
- One Compilation drop does flip significance once ('Hallet et al., 1996' -> slope -0.055),
  but 62 rows of that compilation is the largest single contributor; sign never flips.

Verdict: a real, weak negative relation; CI excludes both 0 and -1 by a wide margin.

### 5. Glacial / Cosmogenic surface (n=99, slope -0.342): least robust stratum

- Only 2 compilations, 19 References, 68 distinct TI.
- Pooled -0.342 (classical CI [-0.490, -0.195], p = 1.2e-5).
- Within-Reference FE slope collapses to **-0.081** (se 0.151, p = 0.59), i.e. the relation
  lives almost entirely between sites/References.
- **Leave-one-compilation-out flips the sign**: dropping 'This study' (67 rows) leaves
  Balter-Kennedy et al. 2021 (32 rows) with slope **+0.167** (the two compilations disagree).
- Cluster bootstrap by Reference widens the CI to [-0.816, -0.115].
- Winsorizing or trimming the log-space outcome shrinks the magnitude by 10-35%.

Verdict: not robust. Pooled significance reflects the dominant compilation's internal
structure, and the two compilations do not agree in sign.

## Task-by-task summary

1. **All (Process, Methodology) strata with n >= 5** (9 strata) fitted; full n / slope / se /
   CI / R^2 / p in `tables/strata_main_ols.csv` and `results.json`. Range of pooled slopes:
   -0.995 (nongl. cosmogenic detrital) to +0.117 (nongl. volumetric). 7 negative, 1 positive,
   1 weak negative non-significant (nonglacial Relief n=7).
2. **HC3**: only Glacial/Relief changes significance (classical p = 0.0030 -> HC3 p = 0.196,
   se 0.106 -> 0.274, driven by n=18 and 3 compilations). All other strata keep their
   conclusions. The largest HC3 stretch is Nonglacial/Cosmogenic surface (se 0.078 -> 0.198,
   2.5x) but it stays significant. Full columns `se_hc3`, `p_hc3`, `hc3_changes_significance`.
3. **Large strata** (`tables/large_strata_robustness.csv`): Theil-Sen reproduces OLS within
   0.004 for nongl. cosmogenic detrital. Row bootstrap CIs are very tight for n=3470/2583.
   **Cluster-by-Reference bootstrap is 1.6x to 6.2x wider** than the row bootstrap and is
   the honest error bar: nongl. Volumetric row-boot CI width 0.036 vs cluster-boot 0.223
   (6.2x, CI [+0.024, +0.246]); nongl. Cosmogenic detrital 0.0026 vs 0.0068 (2.6x,
   [-0.9985, -0.9918]); Glacial/Cosmogenic surface 0.285 vs 0.701 (2.5x, [-0.816, -0.115]);
   Glacial/Thermochronometric 0.421 vs 0.853 (2.0x, [-1.018, -0.165], which now *does* cover
   -1). This last point is the clearest case where the row-level analysis and the
   cluster-level analysis disagree about the -1 hypothesis.
4. **Leave-one-compilation-out** (`tables/leave_one_compilation_out_summary.csv`): sign flips
   only for Glacial/Cosmogenic surface (drop 'This study' -> +0.167) and Glacial/Relief
   (drop 'This study' -> +0.149), both n < 100 with 2-3 compilations. The two nonglacial
   giants cannot be meaningfully tested: nongl. Cosmogenic detrital has essentially one
   compilation, nongl. Volumetric has 3 and dropping Chen et al. 2022 (1521 of 2583 rows)
   gives -0.036 (sign flip, significance loss) — this is the composition effect of finding 2.
5. **Per-Reference slopes** (`tables/per_reference_slopes.csv`,
   `per_reference_fittability.csv`): in nongl. Cosmogenic detrital all 10 largest References
   are fittable and all 10 slopes lie in [-1.047, -0.980] (sd 0.019), i.e. uniformly -1.
   In nongl. Volumetric **none of the 10 largest References (indeed 0/64) is fittable** — a
   single Time-interval value per Reference makes the within-Reference slope undefined, which
   is itself the proof that the pooled slope is a between-group contrast.
6. **Threshold sensitivity** (`tables/threshold_sensitivity.csv`): nongl. Cosmogenic detrital
   is rock-stable (all variants within 0.3% of -0.995). nongl. Volumetric is not: dropping
   the top/bottom 5% of ER moves the slope 0.117 -> 0.078 (-33%), and restricting to
   TI >= 1 yr (n=1594) moves it to **+0.354** (a 3x change), because that filter deletes
   the entire 1-day GloRiSe population and leaves a different composition. TI >= 10 yr is
   degenerate (n=4).
7. **Outlier treatment** (`tables/outlier_treatments.csv`): winsorizing/trimming logER moves
   nongl. Cosmogenic detrital by at most 8% (5/95 winsorization) and nongl. Volumetric by up
   to 33% (5/95 trim). Glacial strata shrink 5-53% under 5/95 treatment; the weakest
   (Glacial/Cosmogenic surface) erodes from p = 1.2e-5 to p = 0.0034 under 5/95 trimming
   (slope -0.342 -> -0.222), surviving alpha = 0.05 with a 35% smaller magnitude.
8. **Consistency with -1 / 0** both classical and HC3, in
   `tables/consistency_classification.csv`: only Nonglacial/Cosmogenic surface (n=64,
   CI [-1.138, -0.825] classical) and Glacial/Cosmogenic detrital (n=9) have CIs covering -1.
   Nonglacial/Cosmogenic detrital (n=3470) formally excludes -1 only because se = 0.0008 —
   see the tautology finding above; the distinction is statistically real but scientifically
   vacuous. Glacial/Volumetric, Cosmogenic surface and Thermochronometric all exclude both
   -1 and 0 (intermediate negative values). Nonglacial/Volumetric excludes both (positive).
   Nonglacial/Relief (n=7) is indeterminate.

## Exhumation identity check (independent verification of the main agent's claim)

Verified with graded tolerances on a ratio `ET = ER * TI` versus `Exhumation * 1000` (mm):

- Glacial: 81 rows populated, all 81 usable. Only **11** match to rel. diff < 1e-12,
  20 to < 1e-6, 28 to < 1e-3, 48 to < 1e-2. Median rel. diff 2.9e-3, p95 ratio 7.6.
  The identity holds *approximately*, not exactly: per-compilation, 'Balter-Kennedy et al.,
  2021' has 24/32 and 'This study' 23/41 within 1%, but 'Burbank et al., 2000' (4 rows) and
  'Faleide et al., 1996' (3 rows) match 0/4 and 0/3.
- Nonglacial: 11 rows populated, **0 match at any tolerance**; the median ratio is 1000 and
  values are consistent with `Exhumation (m)` being stored as km (or ER in m/yr), i.e. a
  ~1000x unit mismatch rather than the mm-vs-m identity.
- So the main agent's claim is **partly confirmed**: the identity is real for a subset of
  glacial rows (roughly half within 1%, and concentrated in two compilations) but it is not
  exact for the whole glacial set and fails completely for the nonglacial set.

## What failed / unresolved

- `scipy.stats.theilslopes(method='hip')` raises `ValueError: method must be either 'joint'
  or 'separate'` in scipy 1.17.1; the script uses `method='joint'` and records the method
  used. This is a real error from execution, not a hypothetical.
- `TI >= 10 yr` for nonglacial Volumetric leaves n=4 and is degenerate (noted in the table).
- Per-Reference slopes are structurally undefined for nonglacial Volumetric (0/64 fittable)
  and near-undefined for Glacial/Thermochronometric (1/18); this is a data-design limit, not
  a numerical one, and it blocks the requested task-5 analysis for that stratum.
- Compilation and Reference are heavily confounded with the Time-interval value itself
  (compilations are effectively time-window datasets), so no regression on this file can
  cleanly separate "timescale" from "dataset". Any causal reading of these slopes requires
  the underlying per-site age and denudation data, which is not in the workspace.
- Runtime: the full script is ~5 min, dominated by 4 cluster bootstraps x 2000 resamples on
  n=3470/2583 plus 2 more on Glacial strata; not a failure, just slow.