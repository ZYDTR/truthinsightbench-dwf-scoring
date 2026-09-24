# Structural Domains and Model Boundaries of Organic Bond Dissociation Enthalpies

**Task** `Chemistry_01_bond_enthalpy_domain` · **T0** 2019-10-18 · **Analysis unit (primary)** one experimental bond observation (543 rows)

This report is the sole authoritative account of the registered findings. It characterizes the complete
computational bond-dissociation-enthalpy (BDE) database, the three estimation channels evaluated on the same
543 experimental observations, and the error and failure boundaries of each channel — including the regions
where the channels are better, equivalent, worse, and **not distinguishable at this sample size**.

All numbers below were produced by executed code in `output/analysis/` and are stored as machine-readable
artifacts in `output/results/`. No external scientific data were downloaded, substituted, or merged; `data/`
is the complete evidence base, and no external source of any kind was consulted for any registered finding.

Twelve independent adversarial verifications were run against the headline claims; all twelve completed and
returned **`confirmed`** on the numbers (`output/results/verify_*.json`, cited per finding). Verification did not
merely echo the workstreams — it **refuted one framing** (Finding 3, which is reported here in its bounded form),
**found one sign-convention defect and one code defect** in supporting workstream scripts (both corrected, and
the erroneous derived number excluded), and **identified a collinearity that qualifies Finding 2's interpretation**
(HAC > 10 is perfectly confounded with database extrapolation). Those corrections are reflected throughout.

---

## 1. Data inventory and analysis units

Three files, all sha256-verified against `data_manifest.json` (`output/results/ws5_part1_raw.json`):

| File | Content | Rows | Unit |
|---|---|---|---|
| `data/source_data/bde_db_full.csv.gz` | complete row-level BDE database | 484,907 | one stored bond row |
| `data/normalized_views/experimental_benchmark.tsv` | deterministic benchmark view | 543 | one experimental bond observation |
| `data/source_data/experimental_source_tables.xlsx` (Sheet_001) | experimental source workbook | 543 | same 543 observations |

The three analysis units are **distinct and must not be interchanged** (Finding 1):

| Unit | Count |
|---|---|
| raw database rows | 484,907 |
| unique molecules | 42,577 |
| unique (molecule, fragment1, fragment2) pairs | 290,664 |
| unique BDE values | 144,147 |
| benchmark bond observations | 543 |
| benchmark unique parent molecules | 419 |
| benchmark unique unordered fragment pairs | 539 |

Raw rows exceed unique fragment pairs by a factor of **1.6683**: a molecule's symmetry-equivalent bonds are
stored as separate rows, and `bde` and `bond_type` are constant inside a (molecule, fragment1, fragment2) key
(0 keys with varying `bde`; `output/results/f1_coverage_boundary.json`). Benchmark repeats are far rarer: 86 of
419 parent molecules contribute more than one observation, and only 4 fragment pairs appear twice.

**Cross-file agreement is exact.** The TSV and the workbook match 543/543 on every shared column
(`output/results/ws5_part1_raw.json`), and `observed_value` equals `experimental_bde_kcal_mol` for all rows.

**Bond-class labeling was validated, not assumed.** A labeler that derives the bond class from the two
fragments' radical centres using sanitised RDKit parsing reproduces the database's own `bond_type` column on
**484,907/484,907 rows (100%)** and on **331/331** benchmark rows that have a database counterpart
(`output/analysis/common.py`; independently reproduced on a 20,000-row sample by `verify_8`). This matters
because the benchmark view itself carries no bond-type column.

---

## 2. Registered findings

Each finding states its object and scope, the relationship or boundary, the core quantitative result and its
analysis unit, the supporting analysis and result evidence, a robustness result, counterevidence and
limitations, and a falsification test.

---

### Finding 1 — The database is a bounded chemical space (C/N/O only, ≤10 heavy atoms) and 27.6% of the benchmark lies outside it; within the domain it is near-exhaustive

**Object and scope.** The complete database (`bde_db_full.csv.gz`) and the 543-observation benchmark view.

**Relationship and boundary.** The database's structural domain is closed and small: its **heavy-atom** element
set is exactly **{C, N, O}**, and its maximum parent heavy-atom count (HAC) is **10** across all 42,577 unique
molecules, with **0 molecules above 10** (`output/results/f1_coverage_boundary.json`). The domain statement is
about heavy atoms specifically: fragments also include the bare hydrogen radical `[H]` (361,147 fragment cells),
which contains no heavy atom by construction.

The ceiling is a **dataset boundary, not an isolated maximum**: 16,373 of 42,577 molecules (38.5%) sit exactly at
HAC 10 and 69.9% at HAC ≥ 9 (`verify_2`). The 150 beyond-ceiling observations are carried by 141 unique
molecules, **none of which appears anywhere in the database**.

Coverage of the benchmark by that domain, on the bond-observation unit:

| Criterion | Observations | Share |
|---|---|---|
| exact (molecule, fragment1, fragment2) row present in database | 331 / 543 | 60.96% |
| parent molecule present anywhere in database | 369 / 543 | 67.96% |
| parent HAC **beyond** the database ceiling of 10 | 150 / 543 | **27.62%** |
| parent containing an element never seen in the database | 0 / 543 | 0% |

The benchmark's HAC reaches 38 against the database's 10, so the 150 out-of-domain observations are excluded by
the database's size ceiling, not by element composition.

**Within** its domain the database is close to complete for this benchmark: of the 369 observations whose parent
is present, **331 (89.70%)** also have their exact bond row stored. The residual 38 (7.0% of the benchmark) are
cases where the parent is in the database but that particular bond was not enumerated.

**Core quantitative result.** Database: 484,907 rows / 42,577 molecules / 290,664 unique fragment pairs
(redundancy 1.6683). Domain: elements {C, N, O}, HAC 1–10. Coverage: 331/543 exact, 369/543 parent-present,
150/543 beyond the ceiling. Unit: bond observation.

A second boundary compounds the first: **six of the database's nine bond classes have zero benchmark
coverage.** The benchmark contains only C-H (302 observations), C-C (240) and H-O (1). Missing entirely: C-N,
C-O, H-N, N-N, N-O, O-O — which together are 79,088 unique database pairs.

**Supporting evidence.** Execution: `output/analysis/finding1_coverage.py` → `output/results/f1_coverage_boundary.json`;
independent workstream `output/results/ws1_db_coverage.json`; figure `output/figures/summary_findings.png` (panel F1).
Verification: `output/results/verify_2_coverage.json`, `verify_1_db_scale.json`, `verify_8_bondclass_coverage.json`
— all **confirmed**.

**Robustness.** The ceiling conclusion is insensitive to the threshold because the database side is 0 above 10
for every threshold ≥ 10 (`output/results/ws6_robustness.json`, check 5, verdict ROBUST). An alternative size
definition (molecular weight) gives the same picture: database max MW 152.1, with 138 benchmark observations above
it. Leave-one-molecule-out does not move the counts. Verification independently confirmed the class counts
(C-H 302 / C-C 240 / H-O 1) and all nine database class counts summing to 484,907.

**Counterevidence and limitations.** HAC is computed from the stored SMILES, so a different protonation or
tautomer choice would shift a given compound. Coverage is a statement about overlap with *this* database, not
about the chemical literature. The single H-O benchmark observation is not a database row at all (its parent is
absent), so its label rests on the fragment-based labeler rather than on database provenance.

**Falsification test.** Enumerate every bond of the 150 out-of-domain benchmark parents and query the database
for any stored bond of a molecule with HAC > 10 or containing S/P/F/Cl/Br/I. Finding a single such row would
refute the domain ceiling. Independently: recompute the benchmark's bond-class census with a different
(atom-pair-environment) classifier; if more than three classes appear, the coverage statement fails.

---

### Finding 2 — The quantum-vs-learned ranking is not global: it reverses between the small-molecule and large/extrapolated regions, and neither channel dominates

**Object and scope.** The three estimation channels evaluated on the same 543 experimental observations, split by
parent molecule size.

**Relationship and direction.** The overall paired comparison is genuinely ambiguous, and the ambiguity is
structural, not noise. On the full 543 observations, quantum's MAE (2.0978) is better than learned's (2.4107) by
0.3129 kcal/mol, and the paired t-test on the per-observation absolute-error difference is significant
(p = 5.37 × 10⁻⁵, 95% CI [−0.4706, −0.1682]) — yet the **Wilcoxon signed-rank test on that same difference gives
p = 0.553** and the sign test gives 246 quantum-better vs 284 learned-better (p = 0.108). A split verdict of
exactly this shape indicates a conditional effect.

Splitting by parent heavy-atom count resolves it. The quantum-minus-learned absolute-error gap **changes sign**:

| Size band | n | quantum MAE | learned MAE | gap (q − l) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| HAC ≤ 5 | 113 | 1.676 | **1.578** | **+0.098** | 0.0084 | 0.031 |
| HAC 6–9 | 237 | 1.902 | 1.922 | −0.020 | 0.734 | 0.051 |
| HAC ≥ 10 | 193 | **2.585** | 3.499 | **−0.913** | 5.66 × 10⁻⁶ | 2.90 × 10⁻⁴ |

**Core quantitative result.** Interaction contrast: gap = +0.098 at HAC ≤ 5 versus −0.913 at HAC ≥ 10, Welch
t = 5.084, **p = 8.31 × 10⁻⁷**, Mann-Whitney p = 4.86 × 10⁻⁴, Cliff's δ = 0.245. Unit: bond observation.
The small-molecule side is a **null** on its own (paired t = −0.580, p = 0.564) while the large-molecule side is
well supported (paired t = −4.669, p = 5.66 × 10⁻⁶). So the defensible statement is: **quantum is decisively
better on the large/extrapolated region, and there is no detectable difference on small molecules** — rather than
a symmetric reversal.

**Supporting evidence.** Execution: `output/analysis/size_interaction_test.py` →
`output/results/size_interaction_test.json`; workstreams `output/results/ws2_channel_accuracy.json` (paired
comparisons, unique-molecule unit) and `output/results/ws6_robustness.json` (three of nine channel-comparison
claims rated not testable); figure `output/figures/summary_findings.png` (panel F2). Verification:
`output/results/verify_9_size_interaction.json` and `verify_4_paired.json` — both **confirmed**, including the
t/Wilcoxon divergence and the n's.

**Robustness.** The reversal direction holds under all five alternative loss definitions (MAE, RMSE, median
absolute error, 90th-percentile absolute error, fraction within 2 and within 5 kcal/mol) and is unchanged by
leave-one-molecule-out (max |ΔMAE| 0.026 from any single molecule), by removing the 4 repeated fragment pairs,
and by the molecule-aggregated unit (419 unique molecules: gap −0.416, p = 2.9 × 10⁻⁵). WS6 nonetheless rates
the bare "quantum beats learned" claim **FRAGILE** precisely because the magnitude is small.

**Counterevidence and limitations — the size reading is confounded and must be qualified.** Because the database's
HAC maximum is exactly 10, **HAC > 10 is perfectly collinear with extrapolation beyond the database domain**: all
150 benchmark observations with parent HAC > 10 are out-of-database. `verify_9` localized where the effect actually
comes from — within the C-H HAC ≥ 10 slice, 124 of 146 rows are HAC > 10 and only 22 are HAC = 10, and that
**HAC = 10 cell alone (n = 22) shows a gap of only +0.056** (learned 2.495 vs quantum 2.439). The headline
+1.044 slice edge is therefore carried by the out-of-domain HAC > 10 rows. The divergence is **equally well
described as database-domain extrapolation as by molecular size, and the two cannot be separated with these
data.** It is also not purely a membership flag: within the out-of-database C-H stratum the edge still rises with
HAC (6–9: +0.406; 10–13: +0.364; 14+: +1.687).

Stratifying on provenance confirms the entanglement: the size contrast **does not survive within the in-database
stratum** (gap +0.079 at n = 103 small vs +0.070 at n = 35 large, Welch p = 0.944), and in the held-out stratum
the small side has only n = 10. The sign-crossing at HAC ≤ 5 is also statistically a **null** when tested as an
edge of its own (paired t = −0.580, p = 0.564; bootstrap CI [−0.119, 0.058] includes 0), so "learned is better on
small molecules" should be read as "no detectable difference", while "quantum is better on large molecules" is
well supported. Bond observations are not independent (419 molecules carry 543 rows), so nominal p-values are
optimistic. The experimental reference carries ~2.8 kcal/mol of heterogeneity (Finding 3), so gaps below that
scale should not be treated as decisive.

**Falsification test.** Pre-register the HAC ≤ 5 / HAC ≥ 10 split, obtain ≥ 100 held-out observations in each
band for molecules with HAC in 11–20 where the database has no coverage, and test whether the learned channel
again degrades relative to quantum. A failure to reproduce the reversal would refute the finding; a reversal
already present at the *same* size but differing in provenance would instead point to a provenance mechanism.

---

### Finding 3 — The benchmark's own experimental heterogeneity is ~2.8 kcal/mol, so the calibrated channels' errors are of the same order as the reference noise; the reference channel is not

**Object and scope.** All three channels, measured against the 543 experimental values, interpreted against the
heterogeneity of those experimental values themselves.

**Relationship and boundary.** The experimental values are heterogeneous literature values, **not** independent
repeats from one laboratory. Two independent methods estimate the reference's own spread σ_ref:

- **Method A (matched near-duplicate pairs).** Pairs of observations whose quantum estimates agree within
  0.1 kcal/mol are near-identical bonds; their experimental spread is reference heterogeneity that does not
  assume channel/reference independence. From 1,002 such pairs: sd(ΔE) = 4.0215 → **σ_ref = 2.8437**
  (row bootstrap CI 2.02–3.08; molecule-cluster bootstrap widens it, `verify_5`).
- **Method B (moment decomposition).** σ_ref² = [V(E−Q) + V(E−L) − V(Q−L)]/2 = 8.0261 → **σ_ref = 2.8330**.

The two methods agree to **0.011 kcal/mol**. The implied oracle MAE — what a perfect unbiased estimator would
still show against these noisy references — is **≈2.27 kcal/mol**.

**Core quantitative result.** σ_ref ≈ 2.83 (two methods, agreeing to 0.011). Oracle MAE floor 2.27 kcal/mol.
Ratio of each channel's MAE to that floor: quantum **0.925**, learned **1.062**, reference **2.418**. Unit: bond
observation.

**Robustness — and an important refutation.** `verify_6` confirmed every number but **refuted the framing** that
the quantum channel is provably "at or below the noise floor". The matched-pair ΔE distribution is strongly
non-Gaussian (skew 0.81, excess kurtosis 3.17, Shapiro p = 1.2 × 10⁻²⁰), so the Gaussian MAE conversion is the
least conservative floor. Recomputing under five scale definitions (`output/results/headline_noise_floor.json`,
`floor_definition_sensitivity`):

| Channel | Gaussian | empirical heavy-tailed | robust MAD | robust IQR | median \|ΔE\| |
|---|---|---|---|---|---|
| quantum | 0.925 | 1.069 | 1.320 | 1.320 | 1.561 |
| learned | 1.062 | 1.229 | 1.517 | 1.517 | 1.794 |
| **reference** | **2.418** | **2.797** | **3.452** | **3.452** | **4.083** |

The reference channel's excess is **robust across all five definitions** (ratio 2.42–4.08). The quantum
channel's "at/below floor" reading is **not robust** (0.925–1.561). The defensible statement is the bounded
one: **the calibrated channels' errors are of the same order as the benchmark's own reference heterogeneity**,
which is why they cannot be cleanly ranked against each other on a 543-row benchmark whose reference noise is
comparable to the effect. Method B independently implies a positive quantum-vs-truth σ of 1.02 and learned-vs-truth
σ of 2.00, so the channels are not identical to a perfect estimator either.

**Supporting evidence.** Execution: `output/analysis/headline_noise_floor.py` →
`output/results/headline_noise_floor.json`; supporting `output/results/ws5_provenance.json`,
`ws5_decomposition.json`, `ws5_floor_empirical.json`; figure `output/figures/summary_findings.png` (panel F3).
Verification: `output/results/verify_5_floor.json`, `verify_6_floor_implication.json` — both **confirmed**, with
the framing corrected as above. `verify_11` also confirmed the related heterogeneity numbers and found that a
naive pooled-variance formula in a workstream script produced a spurious 165.3 kcal/mol figure (a variance-squared
slip); the correct pooled within-molecule SD is **9.965** — that erroneous value is excluded from this report.

**Counterevidence and limitations.** Method B assumes the reference error is uncorrelated with the channels'
errors; since the quantum channel is demonstrably derived from a database the experiments correlate with
(Finding 4), Method B is an upper bound. Method A is biased upward because |ΔQ| ≤ 0.1 pairs are chemically
near-identical but not identical, so residual true-BDE variation is included; the true floor may be lower still.
The 1,002 pairs are not independent draws (468 rows, 372 molecules), and the 543 experimental values contain only
284 distinct numbers with the top value reused 9 times, so the effective sample behind the interval is smaller
than nominal. The *source* of the heterogeneity is unresolvable here: the files carry no method, temperature,
phase, or solvent fields.

**Falsification test.** Obtain genuine replicate measurements of the same bond in the same phase from one
laboratory and compare their spread against 2.83 kcal/mol. If same-lab replicates show materially *less* spread
than the matched-pair estimate, the reference heterogeneity is smaller than 2.83 and the calibrated channels
are further from the floor than reported here; if they show comparable spread, the floor stands.

---

### Finding 4 — The reference channel is genuinely worse (2.4–4.1× the noise floor) and its signed error is bimodal with opposite signs per bond class

**Object and scope.** The `reference_estimate_bde_kcal_mol` channel against the 543 experimental observations.

**Relationship and direction.** Unlike the two calibrated channels, the reference channel's error cannot be
explained by reference heterogeneity. Its MAE is **5.4860 kcal/mol** — **2.42× the Gaussian floor and 4.08× under
the most conservative floor definition** (Finding 3), so a real defect is required.

| Channel | MAE | RMSE | median AE | bias | max AE | within 5 kcal/mol |
|---|---|---|---|---|---|---|
| quantum | 2.0978 | 3.0595 | 1.400 | +0.554 | 15.74 | 91.0% |
| learned | 2.4107 | 3.6278 | 1.610 | +1.079 | 25.21 | 86.9% |
| **reference** | **5.4860** | **6.7596** | **4.900** | +1.378 | **33.40** | **52.1%** |

Paired against both calibrated channels the gap is unambiguous and survives every test:
reference-vs-quantum mean difference **+3.388** (95% CI [3.053, 3.742], Wilcoxon p = 1.9 × 10⁻⁵⁹);
reference-vs-learned **+3.075** (CI [2.718, 3.448], p = 1.4 × 10⁻⁵¹). In contrast to the quantum/learned pair,
**every** test agrees here.

**A structural signature.** The reference estimate takes **a single value for each of the 539 distinct
(fragment_a, fragment_b) environments — 0 environments carry more than one reference value** — whereas both
calibrated channels vary continuously within environments. It behaves as a coarse environment-level lookup, not
a per-observation estimator.

**Bimodality.** Its signed error flips sign between the two populated bond classes:

| Bond class | n | reference bias | reference MAE | quantum bias | learned bias |
|---|---|---|---|---|---|
| C-H | 302 | **+5.103** | 5.711 | +0.110 | +0.930 |
| C-C | 240 | **−3.303** | 5.225 | +1.107 | +1.251 |

The calibrated channels show no such flip. A single global bias correction therefore cannot repair the reference
channel, and any single summary of its "worst cases" mixes two opposing failure mechanisms. In the extreme tail
this is visible as two families: large **positive** errors on strained small rings
(`bond_observation_0049` C1=CC1 +16.9; `bond_observation_0072` C1C2CC12 +21.5) and large **negative** errors on the
triarylmethane/tetraphenylmethane family (`bond_observation_0508` −33.4; `bond_observation_0520` −31.0;
`bond_observation_0509` −24.7) (`output/results/ws4_failures.json`).

**Core quantitative result.** MAE 5.4860, ratio to floor 2.42–4.08; bias +5.103 (C-H, n=302) vs −3.303 (C-C,
n=240); 0/539 environments with more than one reference value. Unit: bond observation.

**Supporting evidence.** Execution: `output/analysis/finding4_reference_channel.py` →
`output/results/f4_reference_channel.json`; workstreams `output/results/ws3_bondclass.json`,
`ws4_failures.json`, `ws5_provenance.json`; figures `output/figures/summary_findings.png` (panel F3) and
`output/figures/error_vs_bond_strength.png`. Verification: `output/results/verify_3_channel_accuracy.json`,
`verify_12_reference_structure.json` — both **confirmed**.

**Robustness.** The conclusion is ROBUST under all five alternative loss definitions, under leave-one-molecule-out
(max |ΔMAE| 0.0515), under per-bond-class leave-out, under own-worst-5% trimming, and under the molecule-aggregated
unit (419 molecules: +3.190, p = 5.9 × 10⁻⁴³) (`output/results/ws6_robustness.json`).

**Counterevidence and limitations.** Only C-H and C-C are populated well enough to compare, so the sign flip is
established on **two** classes, not the full nine-class taxonomy. WS4 notes the reference channel's weak-bond
direction is **unresolved** (7 positive vs 11 negative for exp < 70, p = 0.48). The reference values carry no
method/phase/solvent metadata, so the *mechanism* behind the coarse lookup cannot be identified — only its
statistical signature. Its error also correlates only weakly with the calibrated channels' errors
(r = 0.214 and 0.196 on absolute errors), so it fails on largely different rows: of the worst-25 rows for each
channel, quantum and reference share only 4.

**Falsification test.** Recompute the reference channel's per-class bias on an independent experimental set
covering the six unpopulated bond classes. If the bias sign is constant across classes there, the bimodality is
a property of this benchmark rather than of the channel; if it flips again, the environment-lookup mechanism is
supported.

---

### Finding 5 — The weak-bond overshoot of both calibrated channels is real but bounded at roughly half the reference noise; the direction of the extreme failures is unresolved

**Object and scope.** Both calibrated channels in the weak-bond region, and the direction of their extreme errors.

**Relationship and boundary.** Both calibrated channels overshoot weak bonds when the experimental value is used
as the selection variable:

| Experimental band | n | quantum bias | learned bias |
|---|---|---|---|
| [70, 80) | 129 | **+1.347** (93 over / 36 under, p = 5.4 × 10⁻⁷) | **+2.601** (107 / 22, p = 1.3 × 10⁻¹⁴) |
| [80, 90) | 238 | +0.451 (p = 9.0 × 10⁻⁵) | +0.711 (p = 2.4 × 10⁻⁷) |
| [90, 100) | 89 | −0.317 (p = 0.011) | −0.166 (p = 0.033) |
| [110, 160] | 31 | −0.880 (p = 0.029) | −1.065 (p = 0.003) |

The pattern is a systematic **overestimate in the weak-bond region and underestimate at the strong-bond end**.

**The competing explanation was tested, not assumed away.** Conditioning on low experimental values is exactly
the selection that regression to the mean manufactures even for an unbiased estimator. Re-conditioning on the
*estimate* instead of the experiment — which does not select on reference noise — shrinks the effect but does not
eliminate it: in the [70, 80) prediction band the learned channel still overshoots by **+1.185** (n = 118,
t = −3.56, p = 5.4 × 10⁻⁴; parent-clustered bootstrap CI [0.544, 1.841]) and quantum by **+0.584** (n = 124,
p = 0.042). So the overshoot is **partly** a selection artefact and **partly** a residual bias.

**Core quantitative result.** Learned overshoot at [70,80): +2.601 unconditioned, +1.185 conditioned on the
prediction (−54%), against a reference noise floor of ~2.27 kcal/mol. Quantitatively, the residual +1.185 is
**about half the noise floor**, so it is real but comparable to the benchmark's own reference uncertainty.

**Direction of the extreme failures is unresolved.** WS4's sign tests do not establish the direction of the
worst cases: quantum 9 overshoots vs 8 undershoots among rows with any channel error > 15 (p = 1.0); learned's
worst-25 is one-sided (24/25 overshoot, binomial p = 1.6 × 10⁻⁶) but quantum's is not (17 vs 8, p = 0.11). The
reference channel's weak-bond direction is also unresolved (Finding 4).

**Supporting evidence.** Execution: `output/analysis/weak_bond_overshoot_test.py` →
`output/results/weak_bond_overshoot_test.json`; workstream `output/results/ws4_failures.json` (tail census and
data_supported / suggestive_only / unresolved ledger); figure `output/figures/error_vs_bond_strength.png`.
Verification: `output/results/verify_10_weakbond.json` — **confirmed** on all four numbers and all counts, while
flagging a sign-convention defect: the conditioned values were quoted as `mean(exp − pred)` while the
unconditioned ones were quoted as `mean(pred − exp)`. Under a single convention the overshoot shrinks from
+2.601 to +1.185 **without changing sign**. The corrected convention is used in this report.

**Robustness.** The magnitude threshold is not sensitive: shifting the band to [69.5, 80.5) gives +1.160 (n = 123).
The overshoot persists under the molecule-clustered bootstrap. WS4's descriptor tests are reported with
Benjamini-Hochberg correction across nine descriptors.

**Counterevidence and limitations.** The weak-bond region is thin — only 4 observations below 60 kcal/mol and 18
below 70 — so the extreme tail of this effect is very poorly sampled. Weak-bond and large-molecule effects are
confounded (median HAC 12.5 in the weak bin vs 8.0 overall). The channels' failure sets are largely disjoint:
only **3 rows** have all three channels wrong by more than 10 kcal/mol, and only 18 by more than 5, so there is
no single "hard row" family driving all channels.

**Falsification test.** Obtain experimental BDEs for weak bonds (exp < 70 kcal/mol) in molecules small enough to
sit inside the database domain, then test whether the calibrated channels still overshoot when conditioned on
the prediction. A shift to zero or to undershoot would refute the residual-bias reading and leave the pure
selection explanation standing.

---

## 3. Cross-cutting result: the channels fail on largely independent rows

Only **3 of 543** observations have all three channels wrong by more than 10 kcal/mol, and only 18 by more than 5,
while **61** have at least one channel wrong by more than 10. Pairwise overlap of each channel's worst-25 rows:
quantum∩reference 4, learned∩reference 4, learned∩quantum 12. The reference channel's absolute errors correlate
with the calibrated channels' at only r = 0.214 and 0.196 (vs r = 0.754 between quantum and learned). **The three
channels are not redundant failure-wise**, and no single benchmark subset exercises all of them.

## 4. Explicitly unresolved at this sample size

Reported as unresolved rather than asserted, in line with the required research actions:

- **The overshoot direction of the quantum channel's extreme tail** (9 vs 8 among >15 kcal/mol errors, p = 1.0).
- **The reference channel's weak-bond direction** (7 positive vs 11 negative below 70 kcal/mol, p = 0.48).
- **Any quantum channel steric/polycyclic failure family**: the apparent penalty disappears after stratifying on
  database membership (Δ = +0.047, p = 0.52 within non-database rows). The learned channel's crowded-parent
  penalty does survive that stratification (MAE 2.71 → 4.23, n = 90 vs 122, MWU p = 1.3 × 10⁻⁴), but 93 of 95
  crowded rows are also out-of-database, so the unstratified contrast is confounded.
- **The residual 38 observations** where the parent is in the database but the exact bond is not: too few to
  characterize.
- **The source of the reference heterogeneity**: unresolvable from this workspace, which carries no method,
  temperature, phase, or solvent fields.

## 5. Units, missingness, and what is *not* claimed

- Every statistic above names its unit. Row-level, molecule-level and fragment-pair-level statistics differ:
  quantum MAE is 2.0978 on 543 observations but 2.2020 on 419 molecule means.
- There is **no missingness** in either file (0 nulls in all 543 × 8 benchmark cells; 0 unparsable molecules in
  484,907 database rows).
- No external data were used. No claim here rests on literature; the `T0` boundary of 2019-10-18 was not
  crossed because no external source was consulted for any registered finding.
- This benchmark cannot resolve differences below roughly 2–3 kcal/mol between the two calibrated channels, and
  the report does not claim that it can.

## 6. Reproduction

Full instructions, environment notes, and the exact command sequence are in **`output/reproduce.md`**. Analysis
sources are under `output/analysis/` (`common.py` plus one script per finding), machine-readable results under
`output/results/`, and figures under `output/figures/`. The finding-to-evidence mapping with per-file hashes is
`output/results/evidence_index.json`.

**Environment notes (real, not worked around silently).** `pandas.read_excel` is unusable in this environment
(pandas 3.0.5 requires openpyxl ≥ 3.1.5; 3.0.9 is installed), so the workbook is read directly via `openpyxl` in
`common.load_xlsx()`. `pyarrow`/`fastparquet` are absent, so the analysis-table cache is CSV. A `"\x00"`-joined
string key silently truncates under pandas 3.0's `str` dtype and undercounted unique fragment pairs by 28; tuple
keys are used instead and the corrected count (290,664) matches the independent workstream. Both a
`scipy`/`sklearn`/`matplotlib`/`rdkit` install and the parallel verification workflows ran after failures were
observed and preserved.