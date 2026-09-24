# w4_process — process-category effect conditional on scale and method

Script: `w4_process_analysis.py` (self-contained, deterministic seed 20240919, run with `python3`).
All numbers below are from real execution against `data/raw/glacial_erosion_Earth.tsv` (427 rows)
and `data/raw/nonglacial_erosion_Earth.tsv` (6288 rows). No external data.

Sign convention: **positive = Glacial rate higher than Nonglacial, in dex** (log10 mm/yr).
Two different quantities carry this same sign with different meanings, so read carefully:

- **matched / paired / "diff" quantities** = mean(logER_glacial - logER_nonglacial), positive =
  glacial faster. Sections 2, 3, 6 (matched part), 8.
- **`process_coef_dex` / "adjusted" OLS terms** = the coefficient on
  `C(Process, Treatment("Glacial"))[T.Nonglacial]`, i.e. **Nonglacial minus Glacial**, so the sign
  is *opposite* in meaning. Sections 4, 6 (the -1.12 dex figure). The reported -1.44 dex means
  nonglacial higher by 1.44 dex under that model, which is the sign flip discussed in §4 (do not
  interpret it as a real nonglacial-faster process effect).
- The marginal "difference" +0.965 dex (glacial faster, 9.2x) is a matched-convention quantity.

---

## Headline

A marginal "glacial erosion is ~9x faster" claim is real but **not robust to scale or method
conditioning**. The marginal gap is 0.965 dex (9.2x, p=5.7e-74). Method- and scale-matched
estimates fall to 0.46–0.63 dex, the fully adjusted regression flips sign to -1.12 dex, and the
only well-supported scale x method cells give 0.17–2.13 dex depending on which cells are admitted.
**The direction reverses in 3 of 18 strata after cluster-robust adjustment (nonglacial faster).**

## 1. Marginal comparison (n=427 glacial, 6284 nonglacial with rate>0)

| | n | mean | median | sd | q1 | q3 | IQR |
|---|---|---|---|---|---|---|---|
| Glacial | 427 | -0.296 | -0.260 | 0.885 | -0.824 | +0.230 | 1.054 |
| Nonglacial | 6284 | -1.261 | -1.212 | 0.895 | -1.792 | -0.673 | 1.120 |

- Welch t = 21.79, **p = 5.7e-74**, mean diff **+0.965 dex** (9.2x), Cohen's d = 1.079.
- Mann-Whitney U = 2,112,101, p = 5.2e-88, rank-biserial -0.574.
- Median diff +0.953 dex (9.0x).
- Cluster-robust SEs (by Reference): 0.089 (glacial), 0.053 (nonglacial) — far larger than the
  iid SE implied by the t-test, so treat the p-values as optimistic but the effect as real.

## 2. Method-matched comparison

| Methodology | n_glac | n_nonglac | mean diff (dex) | Cohen's d | MWU p |
|---|---|---|---|---|---|
| Cosmogenic detrital | 9 | 3470 | +0.394 | 0.52 | 0.127 |
| Cosmogenic surface | 99 | 220 | **+2.992** | 4.58 | 8.4e-46 |
| Relief | 18 | 7 | **-0.999** | -0.92 | 0.006 |
| Thermochronometric | 68 | 4 | -0.482 | -0.76 | 0.150 |
| Volumetric | 233 | 2583 | +0.981 | 1.21 | 8.8e-51 |

Two strata are **negative** (nonglacial faster) even before matching.

Per-TYPE logER (mean / median):
- Glacial: Alpine (164) -0.237 / -0.125; Alpine tidewater (61) +0.343 / +0.602;
  Continental (127) -0.577 / -0.502; High-latitude (71) -0.618 / -0.530;
  Outlet ice stream (4) +2.154 / +2.692.
- Nonglacial: Fluvial (6064) -1.176 / -1.179; Subaerial (224) **-3.591 / -3.578**.

The nonglacial reference is itself heterogeneous: Subaerial is a separate population 2.4 dex
below Fluvial, and it happens to be the entire nonglacial Cosmogenic-surface arm (224 rows, all
`Type=Subaerial`). That single fact drives the largest apparent contrast (+2.99 dex).

## 3. Scale-matched comparison (k=5 nearest nonglacial neighbours in logTI, within Methodology)

| Methodology | n | mean diff | median diff | ratio | cluster-robust SE | p (iid) | p (cluster) | % rows glacial faster |
|---|---|---|---|---|---|---|---|---|
| Cosmogenic detrital | 9 | +0.582 | +0.815 | 3.8x | 0.242 | 0.024 | 0.016 | 89% |
| Cosmogenic surface | 99 | +1.867 | +2.067 | 73.7x | 0.208 | 3e-43 | 2.3e-19 | 99% |
| Relief | 18 | **-1.019** | -1.355 | 0.10x | 0.288 | 0.0022 | 0.00041 | 11% |
| Thermochronometric | 68 | **-0.482** | -0.246 | 0.33x | 0.253 | 4.6e-08 | 0.056 | 26% |
| Volumetric | 233 | **+0.252** | +0.329 | 1.8x | 0.139 | 2.2e-04 | 0.070 | 61% |
| **Pooled (all 427) — headline** | 427 | **+0.463** | +0.374 | 2.9x | 0.186 | 1.2e-13 | **0.013** | 63% |

Robustness variants (all pooled over methods unless noted): k=1 nearest neighbour +0.716 dex
(p=1.2e-20); reverse anchoring on nonglacial rows gives +0.782 to +1.049 dex inside the large
strata. 2-D matching on logTI+logArea *shrinks* Cosmogenic detrital to +0.241 (p=0.48, null) and
Volumetric *grows* to +0.663, so the per-stratum magnitudes are not stable under the choice of
matching keys even though the pooled sign is.

**The matched Volumetric contrast (+0.25 dex, 1.8x) is only marginally significant once clustered
(p=0.071) and is ~4x smaller than the unmatched Volumetric value (+0.98 dex).** Scale matching
removes most of the apparent Volumetric effect.

Within-TI-decile (pooled, no NN weighting) is the clearest conditioning result:

| logTI decile (yr) | nG | nN | diff (dex) | p |
|---|---|---|---|---|
| 0.003–5 (5) | 69 | 2579 | +1.072 | 2e-20 |
| 0.7–500 | 143 | 487 | **-0.125** | 0.138 |
| 497–2300 | 10 | 645 | **-0.104** | 0.675 |
| 6829–16k | 31 | 624 | **-0.072** | 0.632 |
| 16k–36k | 39 | 617 | +0.924 | 8e-12 |
| 36k–152k | 12 | 643 | +0.725 | 7e-04 |
| >152k | 123 | 533 | +2.066 | 2e-59 |

**In the three middle deciles, spanning the bulk of the overlap window, the process difference is
indistinguishable from zero (-0.07 to -0.13 dex, all p>0.13).** The huge positive values live in
the cells where nonglacial has essentially no support (very short windows, where nonglacial
Volumetric rates are ~1e-2 mm/yr) and in the very long-window cell.

## 4. Functional-form sensitivity — why part 3 and part 4 appear to conflict

Within-methodology linear logTI-adjusted contrasts have the **opposite sign** to matched contrasts
in 4 of 5 strata:

| Methodology | marginal | linear-adjusted | quad-adjusted | kNN5 matched | glacial vs nonglacial median logTI gap |
|---|---|---|---|---|---|
| Cosmogenic detrital | +0.394 | **-0.652** | -0.660 | +0.582 | -0.13 |
| Cosmogenic surface | +2.891 | **-1.619** | -0.965 | +1.867 | -2.45 |
| Relief | -0.999 | **+1.448** | +1.470 | -1.019 | -0.65 |
| Thermochronometric | -0.482 | **+0.691** | +0.657 | -0.482 | -0.45 |
| Volumetric | +0.981 | **-0.752** | -0.968 | +0.252 | +0.95 |

The cause is **not** a real process effect: it is non-overlapping or non-parallel logTI support.
- Cosmogenic surface: glacial logTI 5-95% is [1.94, 4.52]; nonglacial is [5.38, 6.90] — **zero
  overlap at the 5-95% level**. A linear adjustment extrapolates across a 2.4-dex gap that
  contains no data. Restricted to the (nonexistent) common support the stratum is not analysable.
- Volumetric: nonglacial logTI is bimodal (median 0.70 dex; 38% of all 6288 nonglacial rows sit
  at exactly 0.699) while glacial spans -1.25 to 6.41; only 24% of glacial Volumetric rows sit
  inside the nonglacial 5-95% range.
- Cosmogenic detrital: 3470 nonglacial rows but only 9 glacial, and the two overlap well
  (0.89 of the nonglacial rows are inside the glacial TI range); here the linear adjustment
  (-0.652) still contradicts the matched value (+0.582).
- A single pooled linear slope on logTI (part 4) is therefore a **mis-specified common adjustment**
  and must not be read as the process effect.

Note the Volumetric regression adjusted contrast changes sign between the marginal (+0.981) and
the linear-adjusted (-0.752) forms even though within-method matching gives a *positive* +0.252.
A quadratic in logTI does not fix it (-0.968). This is the signature of a leverage problem, not
of a process effect: 971 nonglacial rows at the single logTI value 0.003 yr (log=-2.56) sit far
outside the glacial Volumetric support (5th pct = -1.25 dex) and dominate the fitted slope.

## 5. Stratified regression (part 4), all rows complete on logER/logTI/Method/Process, n=6555

- `logER ~ Process + logTI + Methodology`: process coef (Nonglacial - Glacial) **-1.440 dex,
  se 0.048, p=4.7e-182**, R²=0.178; logTI slope -0.157 dex/dex (se 0.007, p=2.7e-97).
  (The interaction model's process term is the same -1.441 with se 0.067, so its CI is wider;
  cluster-robust SEs by Reference are ~2-3x the iid ones, see `results.json`.)
- Adding `Process:logTI`: interaction **+0.0007, se 0.0199, p=0.973** (95% CI -0.038 to +0.040);
  ΔR² from the interaction = 1.4e-07. **The process effect does not vary with timescale in log
  space once method and a single logTI slope are controlled.** Centering logTI leaves the process
  term at -1.440 (se 0.049).
- Full 3-way `Process*Methodology*logTI`: R²=0.606; the process main effect at the reference level
  is -0.422 (se 0.589, p=0.474) and the only significant process x method terms are
  Cosmogenic surface (+2.416, p=0.015) and the Cosmogenic-surface-specific TI slope (-0.585,
  p=0.0022).
- Incremental R²: method alone 0.042; logTI alone 0.004; process alone 0.069; logTI+method 0.067;
  +process 0.178.

**Part 4 vs part 3 sign conflict is fully explained by functional form** (see §4). The regression
answer (-1.44 dex nonglacial-faster) is an artefact of forcing one linear logTI slope across
disjoint support; the matched answer (+0.46 dex glacial-faster) is the better-identified estimate.

## 6. Confounding — how much is composition?

- TI distributions: glacial median 1270 yr (IQR 18–5.0e5), nonglacial median 1201 yr
  (IQR 5–1.5e4). Medians nearly identical (ratio 1.06) but **logTI overlap coefficient is only
  0.327**; 56% of nonglacial rows fall in the glacial logTI IQR, 50% of glacial in the nonglacial
  IQR. Glacial logTI IQR (4.43) is wider than nonglacial (3.48). MWU on logTI p=1.3e-14.
- Method composition differs sharply: glacial is 55% Volumetric / 23% Cosmogenic surface /
  16% Thermochronometric; nonglacial is 55% Cosmogenic detrital / 41% Volumetric.
  All 5 methods contain both processes, but 3 strata have <10 nonglacial rows.
- Kitagawa-Oaxaca-Blinder decomposition of the 0.965 dex marginal gap by Methodology:
  **within-method process component +1.095 dex; method-composition component -0.191 dex**
  (sum +0.904, matches). Composition moves the gap the *wrong* way and is small in absolute terms.
  Adding TI deciles as a second stratifying variable makes composition dominate instead
  (within +0.145, composition +0.874 over 23 cells) — evidence that the two decompositions are
  not mutually consistent and that the composition/within split is not identified without
  stronger assumptions.
- Cascade of estimates (dex): marginal +0.965 → TI-reweighted +0.911 → method-matched +0.463 →
  common-support matched +0.631 → KOB within-method +1.095 → fully adjusted OLS **-1.120**.
  The estimate is not stable across defensible adjustment strategies; the spread
  (roughly 0.5 to 1.1 dex for the "process survives" family, versus -1.1 for OLS) is the honest
  uncertainty, far wider than any single CI.

## 7. Alternative explanations — do controls kill the effect?

Matched-pair imbalance (glacial minus its nonglacial neighbours, n=321 pairs with all controls):

| control | mean pair imbalance | paired-t p |
|---|---|---|
| log10 Area | -0.230 | 0.082 |
| log10 Slope | +0.140 | 0.00065 |
| log10 Precipitation | +0.421 | 2.4e-51 |
| abs Latitude (deg) | +10.55 | 9.2e-10 |

Glacial rows sit at systematically higher precipitation and much higher latitude than their
timescale-matched nonglacial neighbours. Adjusting pair differences for all four:
**intercept +0.941 dex (se 0.093, p=4.6e-21, 95% CI +0.758 to +1.124)** — the process effect
*survives and even grows* (raw pair mean +0.758 dex). dlogS and dabsLat are the significant
coefficients (-0.257, p=0.029; -0.028, p=1.4e-26); dlogP is +0.331, p=0.043; dlogA is -0.045,
p=0.077. Note these are the *pooled across strata* matched control slopes in
`pooled_joint_control_model`; the per-stratum values in `matched_control_deltas.csv`
(`matched_control_delta_regressions`) are computed separately and the dlogP slope there is
+1.022, so the precipitation coefficient is stratum-dependent and not a stable quantity.

Fully adjusted OLS on 6450 complete-case rows (Process + logTI + Method + logA + logS + logP +
absLat) gives **-1.120 dex (se 0.051 iid / 0.175 clustered by Reference, p=1.7e-103 / 1.4e-10)**
— a sign flip, and it is *stronger*, not weaker, on the common-support subset (-1.470 dex).
**This sign flip is not credible as a process effect**: it is an out-of-support extrapolation
that forces common Area/Slope/Precipitation/Latitude slopes across two samples with very
different latitude and precipitation support. Report it, but do not believe it.

**Summary for part 6: the matched process difference survives every within-pair control
(+0.76 to +0.94 dex); the pooled OLS "reverses" only because it extrapolates.**

## 8. Reversals and nulls (18 strata; 7 reversed, 9 null)

3 strata reverse with cluster-robust support (nonglacial FASTER than glacial):

| Stratum | n | mean diff (dex) | ratio | p (iid) | p (cluster) |
|---|---|---|---|---|---|
| Relief / Alpine | 6 | **-0.716** | 0.19x | 0.0037 | 4.9e-07 |
| Relief / Continental | 10 | **-1.767** | 0.017x | 4.2e-07 | 3.7e-41 |
| Thermochronometric / Continental | 5 | **-1.615** | 0.024x | 0.0017 | 1.2e-11 |

Reversals that do **not** survive cluster adjustment:
- Thermochronometric / Alpine: matched -0.429 dex, iid p=8.4e-07 but **clustered p=0.083** (only
  13 glacial Reference clusters; the effect is carried by one or two references). Note the
  *simple-slope regression* within that stratum points the other way (+0.691 dex, cluster
  p=0.0014) — with only 4 nonglacial Thermochronometric rows, that stratum cannot identify a
  process effect at all.
- Volumetric / Continental: -0.110 dex, p=0.304 (null); Volumetric / Alpine tidewater +0.230 dex,
  p=0.173 (null); Volumetric / High-latitude +0.038 dex, p=0.756 (null).
- Burbank et al., 2000 compilation: -0.271 dex, p=0.234 (n=4 glacial / 11 nonglacial).

Decisive caveat on the Relief stratum: the matched comparison is **degenerate** — only 7 nonglacial
Relief rows exist, so with k=5 every glacial row is matched against nearly the same few rows
(plus 7 nonglacial Relief rows fail the metre-identity check in §9). The Relief reversals are as
likely to reflect a 7-row reference sample as a real process difference.

Also degenerate: Thermochronometric k=4 with only 4 nonglacial rows total.

## 9. Verification of the exhumation unit claim — NOT exactly reproduced

The main agent's claim needs qualification.

- 92 rows have Exhumation + rate + interval (81 glacial, 11 nonglacial — matches known fact).
- Using the **proper** conversion `rate(mm/yr) * interval(yr) / 1000 == Exhumation(m)`:
  28/92 within 0.1%, **48/92 within 1%** (median ratio 1.002004).
- Using the **raw** product `rate * interval == Exhumation`: **0/92**. So the column is not stored
  as mm.
- **7/92 rows have the column equal to the raw unconverted product** (ratio in [900,1100]) — a
  genuine mm-as-m unit error, and **all 7 are nonglacial** (7 of the 11 nonglacial rows, all
  Relief). Median ratio over the whole set is 1.002, but the tail reaches 1e4.
- The remaining rows satisfy neither identity because rate and interval were independently rounded
  before deposition, so the product does not reproduce the stored value to any tolerance.

**Conclusion: `ER * TI = 1000 * Exhumation` holds for a bare majority (52%) and only to ~1%;
it is not an exact identity, and one nonglacial stratum (Relief) is systematically mm-contaminated.
Any analysis that treats Exhumation as an independent measurement must drop the 7 failed rows.**

## 10. Independent check of the known ER~TI stratum slopes

| Process | Methodology | n | slope (dex/dex) | se | R² |
|---|---|---|---|---|---|
| Nonglacial | Cosmogenic detrital | 3470 | **-0.995** | 0.001 | **0.998** |
| Nonglacial | Volumetric | 2583 | **+0.117** | 0.010 | 0.053 |
| Nonglacial | Cosmogenic surface | 64 | -0.982 | 0.078 | 0.718 |
| Glacial | Volumetric | 233 | -0.073 | 0.025 | 0.035 |
| Glacial | Thermochronometric | 68 | -0.652 | 0.114 | 0.331 |
| Glacial | Cosmogenic surface | 99 | -0.342 | 0.074 | 0.179 |
| Glacial | Relief | 18 | -0.370 | 0.106 | 0.432 |

Confirmed: the nonglacial Cosmogenic-detrital slope is -0.995 with R²=0.998 (a near-perfect
-1 law, i.e. thickness ≈ constant) and nonglacial Volumetric is +0.117 with high R² relative to
its small slope. Note the **glacial** Volumetric slope (-0.073) is not the same as the nonglacial
one (+0.117), so "Volumetric" is not a homogeneous measurement class across processes.

## What failed / is unresolved

- **Failed and fixed during development**: an early version of the pooled control regression
  mislabelled its intercept and used a wrong design matrix; a stale run also hit a 590 s timeout
  caused by non-vectorised bootstrap loops plus a slow matplotlib import in this sandbox
  (~20 s per import). Both are fixed; the delivered script runs clean in ~4 minutes.
- **Not resolvable from this data**: the process effect is **not identified** at the population
  level. Glacial and nonglacial samples do not share latitude, precipitation, or timescale
  support; only 2 methodology x TI-decile cells have >=3 glacial and >=20 nonglacial rows. Every
  estimate is a local, support-dependent quantity.
- **Rows are not independent**: glacial has 154 references / 427 rows (max 32 rows in one
  reference); nonglacial has 265 references / 6288 rows (max 512). Clustered SEs are 2-4x the iid
  SEs. All p-values here are optimistic; the clustered numbers are reported alongside.
- **Unresolved**: whether the residual +0.46 dex matched effect is process (glacial erosion is
  genuinely faster) or a measurement artefact (e.g. glacial rates inferred from sediment yields
  integrate a different spatial footprint than nonglacial cosmogenic/thermochronometric rates).
  The data contain no instrument or basin-size control that would separate these.