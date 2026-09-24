# Unit- and Site-Level Environmental Transfer Boundaries

**Task:** `Life_04_environmental_transfer_boundary` · **T0:** 2020-01-01 · **Report language:** English

**Authoritative report.** This file is the sole authority for the registered findings. All analysis
and result evidence lives under `output/`; the mapping from each finding to its code and result
artifacts is given below and also enumerated in `output/file_registry.json`.

---

## 1. Scope, data, and analysis unit

The frozen payload is three tab-delimited files (5,328 bytes total, SHA-256 verified against
`data_manifest.json` inside the run):

| File | Shape | Content |
|---|---|---|
| `data/unit_level_observations.tsv` | 30 × 5 | one row per sampled unit: `illness_day`, `stay_day`, `quantitative_assay_value`, `any_environment_positive` |
| `data/site_level_observations.tsv` | 212 × 3 | one row per sampled site: `site_role_id`, `positive` |
| `data/symptom_assay_surface_summary.tsv` | 30 × 5 | one row per sampled unit: `symptom_present`, `surface_positive_percent`, `quantitative_assay_value` |

There are **30 independent analysis units**. The 212 site rows are repeated observations nested
inside those 30 units (6–8 per unit), and the unit-level files are one row per unit. There are no
missing cells in any file; the `illness_day` and `quantitative_assay_value` columns agree exactly
between the two unit-level files (both assertions pass in `results.json → payload`). No external
data were downloaded, and no source was used as evidence for any core claim — accordingly no
external citation or `T0`-dated source log is required for this report.

**The unit is the inferential subject throughout.** The dataset's own guide states that unit-level
blocking is required; Section 3 shows quantitatively what happens when it is omitted.

---

## 2. Registered findings

Five findings are registered. Two are substantive relationships, two are boundaries/conflicts, and
one is an essential negative structural result. Directions are reported as the data show them,
including where they contradict the intuitive expectation.

### Finding 1 — Environmental positivity is a property of the sampled unit, not of the sampled site; site-level labels carry no recoverable location signal

**Object and scope.** The 212 site-level observations nested in 30 units.

**Relationship / boundary.** Positivity is determined at the unit level. Treating the 212 rows as
212 independent subjects is not a conservative approximation — it is a different and wrong model of
the data.

**Core quantitative result.** 16 of 30 units are *unanimous* — 13 units are entirely negative
(0/7 or 0/6 or 0/8 positive) and 3 units are entirely positive (7/7, 8/8, 7/7). The per-unit
positivity proportions take only 8 distinct values and cluster at the extremes
(`0.0` ×13, `1.0` ×3). The one-way ANOVA intraclass correlation of `positive` within unit is
**ICC = 0.488**; with a mean cluster size of 7.07 and adjusted k₀ = 7.066, the design effect is
**Deff = 1 + (k₀−1)·ICC = 3.96**, so the 212 site rows carry the information of only **≈53.5
independent observations**. Unit-level positive counts are strongly overdispersed relative to a
Binomial with the pooled rate p̂ = 0.358 (observed variance 6.12 vs Binomial 1.37, ratio **4.46**,
permutation p = 5.0e-05, 20,000 replicates).

**Counterintuitive corollary (a negative result).** `site_role_id` is **not** a stable location
identifier. Within every one of the 30 units the label increments by exactly +1 modulo 10
(0/30 violations), i.e. it is a rotating position counter whose starting value differs per unit
(units start at labels 1, 9, 7, 5, 1, 7, 4, 1, 8, 5, 2, 9, 6, 3, 10, …). Consequently a given
`site_role_id` denotes a different physical position in different units. A naive reading would find
that label `site_6` is positive in 42.9% of its rows versus 18.2% for `site_2` and call that a
persistent contaminated location. Under the correct null — shuffling positivity *within* each unit,
which preserves every unit's positive count — the observed maximum spread across labels (0.247) is
unremarkable (**p = 0.377**, 200,000 replicates). The same holds after correcting each label to a
rotation-invariant position-within-unit index (max spread 0.571, **p = 0.349**). **No site-level
structure beyond the unit total is detectable, and the label scheme would prevent identifying it
even if it existed.**

**Supporting analysis and result evidence.** Analysis source `analysis/run_analysis.py`
(§0b, §1) and `analysis/run_sensitivity.py` (§A), artifact IDs
`F001`, `F003`; result evidence `F006` (`analysis/results/results.json → within_unit_positivity`,
`design_effect_sites`, `overdispersion_positive_counts`, `site_label_structure`), `F007`
(`sensitivity.json → A_label_rotation`), `F010`, `F015`, `F016`, and figure `F017`
(`figures/fig1_unit_level_positivity.svg`), `F020` (`figures/fig4_site_label_rotation.svg`).

**Robustness.** The ICC/design-effect calculation is an ANOVA estimator that requires no
distributional assumption; the overdispersion test is a 20,000-replicate parametric bootstrap
whose null preserves each unit's site count; the label tests use two independent definitions of
"site" (raw label, rotation-corrected position) and both return non-significant spreads.

**Counterevidence, limitations, alternatives.** The design effect is large but the sample is small
(30 units), so the ICC has wide uncertainty that bootstrapping over only 30 clusters could not
narrow. The all-or-nothing pattern is what one expects from a within-unit shared exposure, but a
batch/shipment effect, a shared swabbing operator, or a per-unit laboratory run would produce the
identical signature; the frozen fields cannot separate them. The claim is *not* that sites are
homogeneous in truth — only that this dataset, at this sampling resolution, carries no site-level
resolution.

**Falsification / replication test (executable).** Re-sample the same units and record the
physical location of each swab (a field this payload deliberately omits). If positivity tracks
location, then within-swab-pair agreement or a location main effect should be demonstrable after
adjusting for the unit; if it does not, unit-level random effects will absorb essentially all
variance and the ICC will stay near 0.5. A parallel falsifier requiring no new data: if
`site_role_id` were a stable location, the per-unit starting label would be constant across units;
it is not (30/30 units start elsewhere or wrap), which the reproduction script re-checks directly.

---

### Finding 2 — Positivity concentrates at low quantitative assay values, but as a threshold step rather than a graded dose–response

**Object and scope.** The 30 units, relating `quantitative_assay_value` (range 18.02–37.70) to
`any_environment_positive`.

**Relationship / direction.** Units with the *lowest* recorded assay values are the ones that are
environmentally positive, and the relationship is better described as a step than as a slope.

**Core quantitative result.** Unit-level positivity splits at **assay = 24.16**: all **8/8** units
below the cut are positive (rate 1.00) versus **9/22** at or above it (rate 0.409); exact Fisher
two-sided **p = 0.0044** (one-sided p = 0.0042), odds ratio not estimable (a zero cell). The
low-assay cluster is 8/8 against a base rate of 17/30 = 0.567 (exact binomial upper tail
**p = 0.011**, expected 4.5 positives at the base rate). By contrast the *monotone* summary is
weaker: Spearman ρ = **−0.354** (two-sided permutation p = **0.056**, 200,000 replicates; AUC =
P(assay_positive < assay_negative) = **0.706**, bootstrap 95% CI [0.502, 0.878]), and the
tertile/quartile gradients are flat above the cut (Q1 8/8 = 1.00, Q2 2/7 = 0.29, Q3 3/7 = 0.43,
Q4 4/8 = 0.50; top-vs-bottom quartile Fisher p = 0.077). A changepoint grid over 27 candidate cuts
prefers 24.16 on likelihood (Δlog-likelihood vs the intercept-only null = 5.64; AIC 33.8 vs 43.1
for the null and 40.5 for the linear model). Because the best cut was *chosen*, its improvement was
re-tested with a permutation null that re-optimises the cut in every replicate: observed 5.64 against
a 95th percentile of −0.95, **p = 5.0e-05** (20,000 replicates).

**Supporting analysis and result evidence.** Analysis source `analysis/run_analysis.py` (§2c) and
`analysis/run_supplementary.py` (§1–2), plus `analysis/run_sensitivity.py` (§D); artifact IDs
`F001`, `F004`, `F003`. Result evidence `F006` (`results.json → F_assay`), `F008`
(`supplementary.json → changepoint_grid`, `best_cut_2x2`, `low_assay_cluster_composition`), `F007`
(`sensitivity.json → D_assay_robustness`), figure `F018`
(`figures/fig2_assay_threshold.svg`).

**Robustness.** (a) *Complete-unit leave-out*: dropping the four structurally distinct units that
have 8 site rows instead of 7 leaves 26 units (14 positive) and preserves the threshold finding
(6/6 below versus 8/20 at or above, **p = 0.017**). (b) *Threshold-form sensitivity*: the
conclusion is stated for a step, and the weaker monotone alternative is reported rather than
suppressed. (c) *Alternative models*: Firth-penalised logistic and a weak-ridge logistic agree in
sign and stay finite under separation; leave-one-unit-out coefficient stability is tight (assay
coefficient −0.132 full sample, LOO range [−0.155, −0.121], SD 0.011). (d) *Multiplicity*: across
the five candidate relationships the assay's raw two-sided p = 0.056 becomes Holm-adjusted 0.223,
while the surface measure's 5.0e-05 becomes 2.5e-04. (e) *Model-free anchor*: the permutation null
for the assay logistic coefficient gives a two-sided p = 0.041.

**Counterevidence, limitations, alternatives.** Three qualifications materially bound this finding.
First, the direction of the assay scale is **not documented**: the field is described only as
`quantitative_assay_value`, so "lower is more positive" is an empirical description of this payload,
not a claim about cycle thresholds or concentrations. Second, the effect is **not independent of
time** — assay correlates with `illness_day` (Spearman ρ = 0.501) and all 8 low-assay units also
have illness days ≤ 8; after adjusting for `symptom_present` and `illness_day` simultaneously the
coefficient attenuates slightly (−0.132 → −0.129) and its permutation two-sided p weakens to
**0.061**, i.e. the assay effect does not clearly survive full adjustment in a 30-unit sample.
Third, non-monotonicity plus a saturated cell means the step is estimated from few units: the cut
is a *maximum-likelihood* location on a grid, and its exact placement (24.16 vs neighbouring cuts)
is not resolvable at n = 30. A plain measurement-scale reading — that the swabs of positive units
were simply read out at systematically lower values on whatever instrument scale this field records
— remains compatible with every number above.

**Falsification / replication test (executable).** Pre-specify the cut (e.g. the median of a fresh
batch) and test the below/above contrast in new units; a true threshold should reproduce a
saturated low-value block. Conversely, if the association is an artefact of time since illness
onset, adding units with *early* illness days and *high* assay values would erase the step. Both
are decidable with the existing schema and require no new fields.

---

### Finding 3 — The apparent 4-fold protective effect of later illness day is a design artefact; no threshold-free day or stay-day association survives blocking

**Object and scope.** The 30 units, relating `illness_day` (2–30) and `stay_day` (1–20) to
`any_environment_positive`.

**Relationship / boundary.** The *threshold* version of this relationship is an artefact of unit
non-independence. After blocking, neither day variable carries a reliable association with
positivity, and such direction as exists is the reverse of the intuitive one.

**Core quantitative result.** If the 212 site rows are treated as independent, the 2 × 2 table
(rows positive/negative, columns `illness_day ≥ 15`/`< 15`) is [[6, 62], [50, 94]]. Positivity
among late rows is 6/56 = **10.7%** versus 62/156 = **39.7%** among early rows (risk ratio 0.270);
read by outcome, 6/68 = 8.8% of positive rows are late versus 50/144 = 34.7% of negative rows
(ratio 0.254, 95% CI 0.115–0.563). Fisher two-sided **p = 4.5e-05**. The identical contrast on the
30 whole units gives the table [[4, 13], [6, 7]]: positivity 4/10 = **40.0%** among late units versus
13/20 = **65.0%** among early units (RR 0.615), or by outcome 4/17 = 23.5% versus 6/13 = 46.2%
(ratio 0.510, 95% CI 0.180–1.441); **p = 0.255**. The apparent significance moves by roughly four
orders of magnitude purely from the choice of analysis unit (Figure 3). Threshold-free, the direction is **opposite** to the
naive expectation: positives occur on *earlier* illness days (median 5.0 vs 13.0; Spearman
ρ = −0.254, AUC = 0.647 for "positive unit has the lower day"), but this is not significant
(two-sided permutation p = 0.170; exact null over all **C(30,14) = 119,759,850** label assignments
gives p = 0.915 for the upper tail and the same two-sided 0.170, agreeing with the Monte-Carlo
estimate to five decimals). `stay_day` is flat: ρ = −0.012, p = 0.949. Scanning all 28 admissible
illness-day thresholds, **no cut reaches p < 0.05** (minimum p = 0.063). Unit-level logistic fits
agree: the day coefficient is −0.070 (cluster-robust SE 0.052, p = 0.179; Firth −0.063), with
ΔMcFadden R² = 0.055 over the null and a leave-one-out increment *worse* than the null
(−0.43, permutation p = 0.208).

**Supporting analysis and result evidence.** Analysis source `analysis/run_analysis.py` (§1, §2a,
§2b, §3, §4) and `analysis/run_sensitivity.py` (§E); artifact IDs `F001`, `F003`. Result evidence
`F006` (`results.json → naive_vs_blocked`, `F_day`, `models_unit_level`), `F007`
(`sensitivity.json → E_day_threshold_sensitivity`), `F011`
(`analysis/results/day_threshold_sensitivity.csv`), `F012`
(`analysis/results/leave_one_unit_out.csv`), figure `F019` (`figures/fig3_naive_vs_blocked.svg`).

**Robustness.** Whole-unit holdouts: across 30 leave-one-unit-out refits the day model beat the
null on held-out log-likelihood in 20/30 cases with a *negative* mean increment (−0.014); over 435
leave-two-units-out refits 58% (mean +0.031); over 200 random 5-unit blocks 67% (mean +0.023). The
contiguous-block holdout is the least favourable (3/6, mean +0.054). All are far from a stable
advantage. Firth and ridge specifications agree in sign, and the threshold sweep is exhaustive
rather than illustrative.

**Counterevidence, limitations, alternatives.** The day association does have a coherent
*alternative* reading that the data cannot exclude: the point estimate is consistently negative and
strongly driven by the low-day range, since 13 of the 17 positive units have `illness_day` ≤ 9 while
the four late positive units sit isolated at days 17, 17, 19 and 30 (positive-unit range 2–30,
negative-unit range 3–29 — so the ranges overlap and the signal is a concentration, not a
separation). That concentration is equally consistent with a genuine early-illness window of
environmental contamination as with the null. Against that:
`stay_day` — the other time coordinate, correlated with `illness_day` at ρ = 0.788 — shows nothing
(ρ = −0.012), and stay/illness days are in any case not interchangeable (median ratio 0.67). The
`illness_day ≥ 15` comparison is also not a fair test of a monotone effect, since it was chosen from
a grid; the honest statement is that no threshold in the grid is significant and that the
threshold-free estimate is imprecise (30 units, 17 positive).

**Falsification / replication test (executable).** Enrol units with illness days spanning the
upper range (≥ 15) at the same sampling intensity and recompute the blocked contrast. A real
early-window effect should produce a monotone blocked trend across strata; an artefact of the
present sample should produce a flat, non-significant blocked result — which is what the current
28-threshold sweep already shows.

---

### Finding 4 — Symptom status is uninformative about unit environmental positivity, while surface positivity is a strict subset of environmental positivity

**Object and scope.** The 30 units, relating `symptom_present` (23 present / 7 absent) and
`surface_positive_percent` (a 0–100% field in 20-point steps) to `any_environment_positive`.

**Relationship / conflict.** Two symptom-linked signals behave in opposite ways: the binary symptom
flag carries no information, and the surface measure is perfectly *contained* in the environmental
outcome.

**Core quantitative result.** Symptom status is **uninformative, with a non-significant trend
toward lower positivity among symptomatic units**: 12/17 symptomatic units are positive (70.6%)
versus 11/13 asymptomatic (84.6%), Fisher two-sided **p = 0.427**, RR 0.834 (95% CI 0.568–1.225);
unit-level Spearman ρ = −0.164. This is a **null result**: the confidence interval spans 1 and the
direction is not stable in sign once blocking is applied. The same null holds unconditionally —
positivity was recorded in 5 of the 7 asymptomatic units (units `unit_03`, `unit_07`, `unit_11`,
`unit_12`, `unit_29`), so symptoms are neither necessary nor sufficient for environmental
positivity. By contrast the surface measure is **strictly nested**: 13 units are
environment-positive *and* surface-positive, 4 are environment-positive with
`surface_positive_percent = 0`, 0 are surface-positive with an environment-negative result, and 13
are negative on both. The converse cell is **empty** (0 vs 13), and the percentage itself tracks
positivity (ρ = 0.727, p = 5.0e-05; mean 43.5% among positive units vs 0.0% among negative units —
all 17 negative units have exactly 0%).

**Supporting analysis and result evidence.** Analysis source `analysis/run_analysis.py` (§2d) and
`analysis/run_checks_figures.py` (§G2); artifact IDs `F001`, `F005`. Result evidence `F006`
(`results.json → F_symptom`), `F009` (`checks.json → G2_containment`,
`G1_drop_eight_row_units.full_sample`), `F007` (`sensitivity.json → B_surface_nesting`,
`F_containment_permutation`).

**Robustness.** The containment asymmetry rests on a single zero cell (0/13), so it was checked
directly rather than averaged: the four environment-positive/surface-zero units (`unit_07`,
`unit_13`, `unit_23`, `unit_29`) are enumerated in `checks.json → G2_containment`, and the
converse zero is exact. A unit-level permutation of positivity (20,000 replicates) puts the
13-unit overlap well above random overlap (null mean 7.36, p = 5.0e-05), but that test is *not*
the evidence for nesting — because both indicators describe the same unit, the permutation only
reproduces the marginals, and it is reported here to show that the overlap statistic alone would
have been misleading. The symptom null is reported with its effect size and CI rather than as
"no difference".

**Counterevidence, limitations, alternatives.** The containment result has two readings that the
frozen fields **cannot distinguish**, and this is the principal boundary on the finding. If the
5-swab surface panel is a *subset* of the 6–8 environmental site rows for the same unit, the
containment is definitional (a nested-sampling artefact) and adds nothing. If the panels are
disjoint, it is a substantive sensitivity statement: the environmental panel detects units that the
coarser, 5-sample surface panel misses. The payload has no field identifying which site rows are
surface samples, and `site_role_id` — being a rotation counter (Finding 1) — carries no role
semantics, so the two readings are **not separable**. The defensible, data-supported part is the
**direction of the asymmetry**: environmental positivity is the more inclusive indicator in all 30
units, and the reverse (surface-positive with environment-negative) never occurs. A second
limitation: `surface_positive_percent` takes only the six values {0, 20, 40, 60, 80, 100}, implying
a denominator of 5, and that denominator is never stated; the percentages are therefore coarser than
the environmental panel by construction.

**Falsification / replication test (executable).** Publish the sample-to-site crosswalk (or sample
surface and non-surface sites as separate labelled panels). If the surface swabs are disjoint and the
0/13 containment reproduces, the sensitivity boundary is confirmed; if the surface panel turns out
to be a labelled subset of the environmental rows, the finding must be demoted to a definitional
one. This test is decisive and requires only an additional field, not additional subjects.

---

### Finding 5 — The clustering constraint changes conclusions, not just their precision: it is the failure boundary of this dataset

**Object and scope.** The whole frozen payload as an inferential design (30 units, 212 site rows,
6–8 sites per unit).

**Relationship / boundary.** Unit blocking is a *boundary condition* on admissible conclusions
rather than a technical adjustment. Any analysis of this payload that treats sites or days as
independent subjects will report significant environmental relationships that the blocked design
does not support.

**Core quantitative result.** The same contrast (`illness_day ≥ 15` vs positivity) yields
**p = 4.5e-05** at the site level and **p = 0.255** at the unit level (Finding 3), a change of
about four orders of magnitude with no change in the underlying data. The mechanism is quantified:
ICC = 0.488, Deff = 3.96, effective n ≈ 53.5 rather than 212 (Finding 1). Positivity counts per
unit are 4.46× more variable than Binomial (p = 5.0e-05). A second constraint applies to the
*site-level* view of the assay: because positivity is constant within a unit while the assay varies
within it, any site-level assay coefficient is a within-unit artefact, and the site-level logistic
fit is reported with that warning attached (`results.json → F_assay.site_level_assay_logistic`).
After Holm adjustment across the five candidate relationships, only the surface measure survives
(adjusted p = 2.5e-04); the assay (0.223) and the day variables (0.509, 0.949) do not.

**Supporting analysis and result evidence.** Analysis source `analysis/run_analysis.py` (§1) and
`analysis/run_sensitivity.py` (§C); artifact IDs `F001`, `F003`. Result evidence `F006`
(`results.json → naive_vs_blocked`, `design_effect_sites`, `overdispersion_positive_counts`), `F007`
(`sensitivity.json → C_multiplicity`), `F011`, figure `F019`
(`figures/fig3_naive_vs_blocked.svg`).

**Robustness.** The blocked/naive pair is computed from the same rows and the same cut, so the
contrast isolates the unit-of-analysis choice; the design effect is corroborated independently by
the variance-ratio overdispersion test (4.46, p = 5.0e-05) and by the ICC, which use different
estimators. The threshold sweep shows the inflation is not specific to the 15-day cut — every one
of the 28 cuts is non-significant once blocked while the site-level framing makes the same
comparisons look decisive.

**Counterevidence, limitations, alternatives.** The blocked tests are underpowered by construction:
30 units with 17 positive events cannot exclude moderate associations, and several of the null
results above (day, symptom) have confidence intervals that include meaningful effects in either
direction. "Not supported" is therefore the correct verdict for those relationships, not
"refuted". The direction of the surviving assay finding is likewise bounded by the undetermined
scale of `quantitative_assay_value` (Finding 2) and by the fact that the observational data cannot
establish a transmission route or direction — a warning the payload itself carries.

**Falsification / replication test (executable).** Analyse this payload both ways in a fresh run
(documenting site-level p = 4.5e-05 and unit-level p = 0.255) as a calibration check on any
pipeline; a pipeline that reports the site-level value as its headline has demonstrably crossed the
blocking boundary. More generally, re-run any candidate relationship with a cluster-robust or
unit-blocked estimator and a whole-unit holdout: associations that survive both are candidates for
substance; those that appear only unblocked are artefacts of the nesting.

---

## 3. What is directly observed, inferred, and unresolved

**Directly observed (no model).** The 30×5 and 212×3 counts: 16/30 unanimous units, 17/30 positive
units, 3 all-positive units, the exact per-unit site counts (6–8), the zero missing cells, the
absence of values outside the observed variable ranges, the six distinct values of
`surface_positive_percent`, the emptiness of the surface-positive/environment-negative cell, and
the 30/30 compliance of `site_role_id` with the +1-mod-10 rotation.

**Data-supported inference (model- or null-dependent).** The ICC, design effect, and overdispersion
quantities (Finding 1, Finding 5); the Fisher/permutation p-values for every contrast; the
changepoint location and its search-adjusted significance (Finding 2); the failure of day and
symptom associations to survive blocking (Findings 3 and 4).

**Mechanism interpretations (not established by these data).** That low assay values reflect high
viral load; that unit-level clustering reflects a shared room, care group, or swabbing operator;
that the surface/environment containment reflects relative sampling intensity. Each is consistent
with the data; none is identified by them.

**Unresolved.** The measurement scale and direction of `quantitative_assay_value`; the denominator
and sampling frame behind `surface_positive_percent`; which site rows, if any, are surface samples;
what `site_role_id` numbered in the source record; the true illness-day window within which
environmental positivity is elevated. These are recorded as open questions rather than closed with
plausible narratives.

## 4. Failed, conflicting, and negative results retained

- The `illness_day ≥ 15` site-level contrast (p = 4.5e-05) is **retained in the report as a
  counterexample**, not discarded, because it is the concrete demonstration of Finding 5.
- The illness-day relationship is a **non-replication**, not a small effect: its threshold-free
  estimate does not survive blocking, and its most favourable framing (site level) was shown to be
  invalid.
- The assay relationship **conflicts with itself across specifications**: significant as a
  threshold (p = 0.0044), non-significant as a monotone trend (p = 0.056), and weakened under
  adjustment (p = 0.061). Both sides are reported.
- The symptom relationship is a **null with a CI spanning the null**, retained as such.
- Two first-draft permutation loops exceeded the 600 s command timeout and were killed; the
  vectorised replacements were validated by assertion against the original fitted statistics before
  their 20,000-replicate p-values were used (see `output/reproduce.md` §6).
- No test in this run was left unrun, and no reported number is a plan or an estimate of what a run
  would produce.

## 5. Limitations of the payload as evidence

The dataset is small (30 units, 212 rows, 5,328 bytes) and clearly a de-identified projection:
unit and site identity have been removed, `site_role_id` has been replaced by a rotation counter,
and no field documents the assay scale or the surface denominator. The observational design cannot
establish transmission routes or directions. Several quantities that would test the mechanism
interpretations above — swab location, sampling date, the surface crosswalk, the instrument scale —
are absent by construction. Conclusions are therefore bounded to association, containment, and the
behaviour of the design under blocking.

## 6. Reproducibility

Full instructions, environment, command order, method-to-file map, embedded verification
assertions, and the handling of the two killed runs are in `output/reproduce.md`. Every registered
finding above resolves to executed analysis source and to a machine-readable result artifact, as
enumerated in the finding sections and in `output/file_registry.json`. Semantic delivery was
checked with `python3 validate_output.py output`.