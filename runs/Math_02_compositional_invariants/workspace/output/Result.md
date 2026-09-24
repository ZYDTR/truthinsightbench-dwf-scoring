# Compositional Invariants and Bias Boundaries in Sparse Count Systems

Task `Math_02_compositional_invariants` · T0 `2020-01-01` · Analysis unit: one
complete sample composition, or one simulated bias scenario cell.

This report is the authoritative account. All evidence is from `data/`. No
external scientific data, precomputed result or source-identifying material was
used, and no external literature was consulted, so no external citation applies.
Code is in `analysis/`, machine-readable results in `results/`, figures in
`figures/`, execution transcripts in `logs/`.

## Summary

Three files describe two loosely coupled systems: a sparse 529 × 600 composition
matrix with per-sample covariates, and a 1000-cell grid of simulated bias
scenario metrics. The composition side is dominated by sampling depth rather than
by any compact algebraic invariant, but it carries a robust covariate signal that
turns out to have an upper applicability boundary. The bias side is dominated by
the difference between its two conditions, and the apparent metric-level coupling
between them is largely an artefact of exactly-zero cells. Two of the five
registered findings are negative or boundary results.

## Data as observed

| File | Rows | Unit | Key structure |
|---|---|---|---|
| `data/anonymous_feature_counts.tsv` | 82,435 | sample × feature | 529 samples, 600 features, 26.0% of the grid present |
| `data/anonymous_sample_covariates.tsv` | 490 | sample | 489 join to the count matrix; 1 has no counts |
| `data/bias_scenario_metrics.tsv` | 2,000 | scenario × metric × condition | 600 metrics in `small_sample`, 400 in `high_variance`, both under `condition_2` and `condition_3` |

**The `relative_count` column is not relative.** Every stored value and every row
sum is an exact positive integer; no row sums to 1 or 100, and row sums span
14,806 to 1,295,174 (87.5×, CV 0.68). The closure constraint is therefore a
property of how the quantities were produced, not of the stored numbers. Library
size is a free, sample-level nuisance parameter and is modelled explicitly
throughout. All three files match their `data_manifest.json` SHA-256
(`results/integrity.json`).

The bias metrics are exact rationals over small denominators, and the
denominator range differs by scenario: at most 400 in `small_sample`, up to 800
in `high_variance` with 82.5% divisible by 4 (`results/bias_scenarios.json` →
`rational_structure`). The metric value is a ratio over a scenario-specific
number of replicates.

Two structural traps were found and avoided. First, the 600 feature ids and 600
metric ids share the numeric space 1–600, but metric values are uncorrelated with
the feature of the same index (Spearman −0.089 to +0.026) and the `condition_2`
zero rate does not follow that feature's prevalence. **The shared id space is a
naming coincidence, not a join key**, and no row-level linkage between the two
tables is supported. Second, feature rows in the count file are shuffled rather
than blocked, so row order carries no batch information.

---

## Finding 1 — No low-dimensional compositional invariant; the structure is depth-dominated

**Object and scope.** The 529 × 600 composition matrix. **Relationship.** A null
result with a boundary: no stable low-dimensional invariant, no proportionality
law, and no composition-level summary that is independent of sampling depth.

The leading CLR axis carries only **21.9%** of variance, and 50% of variance
requires **7** axes. This is not an artefact of zero replacement: across
pseudocounts 0.1, 0.5, 1.0 and 2.0 the leading share stays near 0.22 and the
50%-variance count stays at 6–7 (`results/compositional_structure.json` →
`clr_pca.pseudocount_sensitivity`).

Proportionality fails just as clearly. Across 1,003 feature pairs the median
pairwise log-ratio variance is **4.09**, with only **6.6%** of pairs below the
conventional threshold of 1. Median joint presence is **0.130** — the pairs are
mostly defined by *which* samples contain both members, not by a stable ratio
between them. Re-estimating on a random half of the samples moves the median
log-ratio variance by a factor of **0.998** (median ratio, so re-estimation on half
the samples leaves it essentially unchanged), confirming the spread is a property of
the population rather than of the sample size.

Depth dominates the observable summaries. Richness scales with library size on
log-log axes with slope **0.071** — a tenfold deeper library yields only about
1.18× more features. Rarefied richness rises steadily with depth and **never
saturates**: at depth 1,000 the mean is **31.0** features, at 100,000 it is
117.0, still climbing toward the raw mean of **155.8**. Observed richness is
therefore a depth statistic, not a property of the sample.

**Robustness.** Pseudocount sensitivity (above) and the half-sample log-ratio
split (above). The Taylor-law estimate across features on the *closed* scale is
**not estimable by construction** — every closed row sums to 1, so the
across-feature mean is pinned at 1/600 for every sample and the x-axis is
constant. This is reported rather than silently dropped (`taylor_across_samples_closed.degenerate`).

**Counterevidence.** Occupancy and abundance are strongly coupled (Spearman
**+0.902**), so a one-dimensional abundance gradient does organise *prevalence*.
What fails is the stronger claim of a compact invariant in the composition
itself. The result is also a boundary on method, not a claim that the system is
featureless: 26.0% grid density over 600 features is simply too sparse, relative
to depth, for proportionality-based methods to have leverage.

**Falsification.** Fit the same CLR spectrum on an independently generated
composition matrix from the same system. A leading axis carrying a majority of
variance, or a majority of pairs with log-ratio variance below 1, would refute
the high-dimensionality claim.

*Evidence:* `analysis/s02_compositional_structure.py` →
`results/compositional_structure.json`; `figures/F1_scale_dependence.png`.

---

## Finding 2 — Richness tracks the covariate and depth with opposite signs, and neither explains the other

**Object and scope.** The 489 samples carrying both counts and covariates.
**Relationship.** Positive association with the covariate and negative with
depth, each surviving control of the other.

Richness and the continuous covariate correlate at Spearman **+0.460**
(p = 5.9 × 10⁻²⁷), while library size correlates with the same covariate at
**−0.283** (p = 1.7 × 10⁻¹⁰). The opposed signs mean the naive first
interpretation — that richer libraries mechanically yield more observed features,
so the covariate is a depth proxy — is backwards: the covariate is associated
with *deeper* libraries and *higher* richness simultaneously.

Partial rank correlation confirms neither term absorbs the other: controlling
depth, richness–covariate is **+0.532** (p = 4.4 × 10⁻³⁷); controlling richness,
depth–covariate is **−0.405** (p = 1.0 × 10⁻²⁰).

**Robustness.** The result was attacked four ways, all agreeing:

- *Equal-effort rarefaction.* Thinning every sample to a common depth of 3,000
  counts — which removes the depth confound by construction — leaves the
  association at **+0.562** (SD 0.010 over 10 replicates).
- *Leave-one-group-out.* Dropping `group_A`, `group_B` or `group_C` in turn gives
  **+0.537**, **+0.308**, **+0.752**; all three retain the sign and remain
  positive after depth control.
- *Sample holdout.* Disjoint sample halves give the same sign in **100%** of 200
  random splits; group-stratified splits agree in 100% of 200.
- *Feature holdout and alternatives.* Disjoint 300-feature halves give +0.456 and
  +0.453. A permutation null over 2,000 shuffles puts the observed value far
  outside the null — no shuffle reached |ρ| ≥ 0.460, and the null's 95th
  percentile is 0.090. Trimming to the 1–99% richness range leaves partial
  ρ = +0.511.

**Counterevidence and boundary.** The relation is **not proportional**. A
monotone fit earns a genuine out-of-sample gain over a straight line — 5-fold
cross-validated R² of **0.431** versus **0.133** — and an 8-bin step function
matches it at 0.409. The trend is concave, rising over the lower covariate range
and then flattening. Finding 4 establishes exactly where.

**Falsification.** Collect an independent sample set spanning the same covariate
range and measure richness at equal sequencing depth. A null or reversed rank
correlation below the covariate median would refute this.

*Evidence:* `analysis/s03_covariate_association.py` →
`results/covariate_association.json`; `analysis/s06_holdouts.py` →
`results/holdouts.json`; `figures/F2_covariate_association.png`.

---

## Finding 3 — `condition_3 > condition_2` is a near-invariant whose exceptions are structured by scenario

**Object and scope.** All 1,000 scenario × metric cells. **Relationship.** A
near-invariant ordering with a quantified, non-uniform exception rate.

`condition_3` exceeds `condition_2` in **995 of 1,000** cells (99.5%). The
exceptions are not scattered noise. Four of the five violations fall in
`high_variance`, the scenario with the higher `condition_2` zero rate:

| Scenario | n | c3 > c2 | Violations | min c3 | max c2 |
|---|---|---|---|---|---|
| `small_sample` | 600 | 99.83% | 1 | 0.402 | 0.663 |
| `high_variance` | 400 | 99.00% | 4 | 0.148 | 0.464 |

The two conditions' ranges **overlap** (global min c3 = 0.148 < global max
c2 = 0.663), so the ordering is an empirical regularity with a measured
exception rate, not a strict algebraic law. None of the five violations involves
a zero in either condition; all are interior cells on the wrong side of the line.

The scenario manipulation acts on the two conditions asymmetrically, which is
what makes the ordering scenario-sensitive. On the 400 shared metric ids, the
paired contrast for `condition_3` is strong and near-universal — mean 0.566
(`high_variance`) vs 0.757 (`small_sample`), paired Cliff's delta **−0.690**,
Wilcoxon p = 3.8 × 10⁻⁴⁴, with `condition_3` lower in 84.5% of pairs. For
`condition_2` the same contrast is weak and marginal — Cliff's delta **−0.188**,
Wilcoxon p = 0.73.

**Robustness.** The paired design uses only the 400 metric ids present in both
scenarios, so the comparison is within-metric. Variance decomposition attributes
**81.5%** of variance to condition identity, versus 1.8% to scenario and 3.9% to
metric identity alone. The metric × condition interaction explains 88.3%. The
rational structure of the values is itself scenario-dependent: `small_sample`
denominators cap at 400 while `high_variance` reaches 800 — corroborating that
the two conditions are not interchangeable readouts of one quantity.

**Counterevidence.** The scenario holdout is decisive and negative: a linear
`condition_2 → condition_3` calibration fitted in one scenario transfers to the
other with **out-of-sample R² of −1.60** (small→high) and **−2.68** (high→small),
with large mean bias. The relation must **not** be pooled across scenarios, and
the in-sample fits (−0.05 and −0.13) are near-worthless even on their own terms.

**Falsification.** Re-run the bias simulation with a third condition or a denser
metric grid. A materially higher violation rate, or violations located away from
the `condition_2` upper tail rather than concentrated in the high-zero-rate
scenario, would refute the near-invariant reading.

*Evidence:* `analysis/s04_bias_scenarios.py` → `results/bias_scenarios.json`;
`analysis/s06_holdouts.py` → `results/holdouts.json`;
`figures/F3_bias_conditions.png`.

---

## Finding 4 — The covariate–richness association has an upper applicability boundary

**Object and scope.** The 489 joined samples, split at the covariate median
(15.0). **Relationship.** Conditional: the association is present below the
boundary and absent above it.

Below the covariate median, ρ(covariate, richness) = **+0.619** (n = 271). Above
it, ρ = **−0.010** (n = 218). The gap is **+0.629**, with a 4,000-replicate
bootstrap 95% CI of **[+0.466, +0.790]** that excludes zero.

The split is not an artefact of choosing the median. Three independent scans agree:

- *Threshold scan.* Over 61 candidate thresholds between the 25th and 85th
  covariate percentiles, a large contiguous region separates a strongly positive
  lower segment from a flat upper segment; 33 of the 39 admissible thresholds
  satisfy ρ_below > 0.4 *and* ρ_above < 0.1. The best-separating threshold is 23.0
  (ρ +0.557 below, −0.213 above).
- *Nested quantile scan.* Cutting at the 10th, 20th, 25th, 33rd, 40th, 50th, 60th
  and 75th percentiles in turn, the lower segment's ρ stays between **+0.566 and
  +0.795** while the remainder never exceeds **+0.263** and is near zero or
  negative at every cut from the 25th percentile upward (+0.263/+0.078/−0.012/
  +0.039/+0.145/−0.010/−0.157/+0.052).
- *Alternative quantile cuts.* Four further cuts (0.4/0.5/0.6/0.7) reproduce the
  collapse (ρ_below +0.795/+0.619/+0.566/+0.574; ρ_above +0.145/−0.010/−0.157/−0.160).

The narrowest cut on this list is also the sharpest: below a covariate of 2.0
(25% of samples) ρ = +0.582, while above it the association is already gone
(ρ = −0.012). The association is carried by the low end of the covariate range,
not by the range as a whole.

**Robustness.** Under equal-effort rarefaction to depth 3,000 the gap persists in
**20 of 20** replicates (ρ_below +0.706 vs ρ_above +0.129, mean gap +0.577).
Restricting richness to the 118 features present in at least half of all samples
— removing rare-feature noise entirely — strengthens the contrast: ρ_below
**+0.730**, ρ_above **−0.005**.

**Counterevidence, and its resolution.** A *group-relative* median split does not
reproduce the boundary in `group_B`. This initially looked like a genuine
counterexample. It is not: `group_B` is a low-covariate stratum whose own median
covariate is **1.53**, far below the global median of 15.0, so its "above" half
sits entirely inside the region where the association is strongest. Holding the
global threshold fixed resolves it — `group_B` has 60 samples below and 23 above,
with ρ 0.698 vs 0.578. All three groups then show the same qualitative pattern
(ρ_below 0.728/0.698/0.579 vs ρ_above −0.003/0.578/0.029). The boundary is a
property of the covariate scale, **not** of the coarse group label; the
between-group share of covariate variance is only 8.8%.

A two-level step alone also under-explains the pattern: the lower segment retains
a strong *within-segment* trend (ρ +0.619) that a pure label effect cannot
reproduce, even though the level shift itself is large (mean richness 140.2 below
vs 170.0 above; Mann-Whitney p = 3.4 × 10⁻¹³; Cliff's delta +0.383 over 59,078
sample pairs).

**Falsification.** Sample the covariate above 15 at higher density with
equal-depth sequencing. A rank correlation above the boundary comparable to the
lower segment would refute the plateau.

*Evidence:* `analysis/s07_applicability_boundary.py` →
`results/applicability_boundary.json`; `figures/F5_applicability_boundary.png`.

---

## Finding 5 — The condition coupling is zero-driven, not a monotone dose–response

**Object and scope.** The 400 metric ids shared by both scenarios. **Relationship.**
A confound result: the apparent `condition_2`–`condition_3` association is carried
by exactly-zero cells rather than by a dose-response.

Pooled over shared metric ids the two conditions correlate at Spearman **+0.458**
in `high_variance` (p = 3.4 × 10⁻²², permutation p < 0.0003) but only **+0.102**
in `small_sample` (p = 0.041, permutation p = 0.049). The scenario contrast is
robust: Δρ = **+0.356**, bootstrap 95% CI **[+0.232, +0.473]**, P(Δρ > 0) = 1.000.

That difference is almost entirely a zero-rate effect. `condition_2` is exactly
zero in **46.5%** of `high_variance` cells but only **19.5%** of `small_sample`
cells. **Excluding zero cells collapses the association in both scenarios** — to
**+0.034** in `high_variance` and **+0.065** in `small_sample`. The scenario with
the higher zero rate is the scenario with the stronger apparent coupling.

Tested directly against the alternative explanation families the objective names,
the zero/non-zero indicator is the best-supported description. Over the same 400
shared metric ids used for the correlations above, its point-biserial correlation
with `condition_3` in `high_variance` is **+0.478** (Mann-Whitney p = 3.2 × 10⁻²³),
against a rank (Spearman) specification of **+0.459** — the indicator matches or
beats the best continuous form. The ratio specification is not even defined on
46.5% of `high_variance` and 19.5% of `small_sample` cells, and where it is
defined it runs strongly *negative* (−0.989 in `high_variance`). A log transform
of `condition_2` is rank-identical to the raw values and therefore adds nothing.
Mean `condition_3` is 0.488 where `condition_2` = 0 versus 0.633 where it is
positive — a **level shift**, not a graded response.

**Robustness.** Bootstrap CI (above); permutation nulls over 4,000 shuffles
(both scenarios); the scenario holdout from Finding 3, where the transfer R² is
negative in both directions, confirming that whatever couples the conditions
does not travel between scenarios. The feature↔metric correspondence test shows
the coupling cannot be attributed to the shared id space.

**Counterevidence.** A residual **+0.034** positive rank correlation survives
zero exclusion in `high_variance` (not significant, p = 0.62). It is not a
metric-level dose-response: the same linear fit transfers across scenarios with
out-of-sample R² of −1.60 and −2.68, so the residual is the scenario contrast
itself rather than a transferable relationship. The honest statement is that
zero-exclusion removes *most* of the association, not all of it.

**Falsification.** Run a scenario that raises the `condition_2` zero rate while
holding the conditional (non-zero) distribution fixed. If the pooled correlation
rises, the zero-rate mechanism is confirmed; if instead the non-zero
dose-response steepens, this finding is refuted.

*Evidence:* `analysis/s04_bias_scenarios.py` → `results/bias_scenarios.json`;
`analysis/s06_holdouts.py` → `results/holdouts.json`;
`figures/F4_zero_driven_coupling.png`.

---

## Cross-cutting limitations

1. **No row-level linkage between the two tables exists.** The composition and
   bias tables were analysed as two separate systems. Any statement joining them
   would rest on the shared id space, which the data refute as a join key.
2. **All covariate findings are associational.** The covariate is unlabelled and
   its measurement scale is unknown; it mixes fractional values (0.03, 0.25) with
   large integers over a ~2,800-fold range, and is treated as ordinal throughout.
   No causal or temporal ordering is identified.
3. **Richness is depth-confounded by nature.** Every richness result is reported
   with its rarefied counterpart, because observed richness cannot be compared
   across samples without equal-effort correction.
4. **Feature-level splits are not composition-level validation.** The data guide
   warns that random feature-row validation leaks composition identity. The
   feature holdout in Finding 2 is therefore used only to show the signal is
   distributed across features; all primary validation uses sample-level or
   scenario-level holdouts.
5. **The two scenarios are a two-level factor.** Scenario effects are estimated
   as a contrast between exactly two simulated regimes; no dose-response in
   scenario severity is estimable.
6. **One covariate row has no counts** (490 rows, 489 joined), and 40 samples have
   counts but no covariate. Missingness is structural, not imputed.

## Summary of registered findings

| # | Claim | Direction | Core statistic | Status |
|---|---|---|---|---|
| 1 | Compositional invariants | null / boundary | CLR PC1 = 21.9%; median log-ratio var = 4.09 | Supported null |
| 2 | Richness vs covariate vs depth | positive / negative, independent | ρ = +0.460; partial +0.532; rarefied +0.562 | Supported |
| 3 | `c3 > c2` ordering | near-invariant with structured exceptions | 995/1000; 4 of 5 exceptions in high_variance | Supported with boundary |
| 4 | Covariate–richness boundary | conditional | ρ +0.619 below vs −0.010 above; CI [+0.466, +0.790] | Supported |
| 5 | Condition coupling | confounded, zero-driven | +0.458 → +0.034 after zero exclusion | Supported |

## Reproducibility

See `reproduce.md`. Eight scripts under `analysis/` regenerate every number,
table and figure from `data/`; console transcripts are in `logs/`. The findings
registry `results/findings_registry.json` is derived from the stage JSON outputs,
not hand-transcribed. `file_registry.json` maps `ARTnnnn` handles to every output
file; the `ART` prefix is deliberately distinct from the finding ids `F001`–`F005`.