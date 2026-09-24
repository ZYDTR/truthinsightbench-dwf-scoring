# Stage-Indexed Interaction Structure Across Development

**Task:** `Life_05_developmental_interaction_structure` · **T0:** 2022-01-01
**Data:** `data/stage_indexed_interactions.tsv` — the complete frozen task dataset, sha256
`67ab6ace870c985522f905801f0eed680818a0c73225ca113057ba12fdea992f`
**Report language:** English. All numbers below were produced by executed code in `output/analysis/`.

---

## 1. Summary

The frozen table describes 1105 anonymous `(interaction_id, cell_pair_id)` combinations observed across
18 ordered stages. Two structural facts dominate it:

1. **Identity, not stage, is the primary organiser of the measured strength.** Which interaction family a
   record belongs to explains 83.6 % of the variance in `interaction_strength_like_value`; the analysis
   unit explains 90.1 %. The ordered stage axis explains about 1.8 % of total variance. The table is
   predominantly a catalogue of persistent pair-specific levels with a comparatively small developmental
   modulation riding on top.
2. **There is nevertheless a real, directionally consistent developmental shift.** Within-unit-centred
   strength rises across the ordered stages at about +0.042 per stage (Pearson r = 0.92 across the 18
   stage means), and adding that stage trend improves out-of-sample prediction for interaction families
   that were entirely excluded when the trend was estimated (MAE 0.468 → 0.428, −8.7 %, permutation
   p ≤ 0.005).

Three further results qualify how these two facts should be used. The stage shift **is not uniform across
families**: the aggregate gain is clear but per-family outcomes split almost evenly, so the trend is a
distributional property, not a family-specific law. The `association_pvalue` column is **a coarse report
tier, not a graded evidence readout**: only the floor/non-floor distinction carries information, and that
distinction is highly labile for the same unit across stages. And a large fraction of individual units are
**genuinely unstable** — 210 of 579 taxonomised units reverse the sign of their strength across stages.

The two registered conclusions that report a *supported relationship* are Findings 1 and 2. Finding 3
reports a **null / boundary** result about the evidence column. Findings 4 and 5 report **conditional
conflicts and applicability boundaries** that materially restrict how Findings 1 and 2 may be used. The
data do not support any claim that these association values identify direct biological communication or
causality, and none is made.

---

## 2. Data and analysis unit

| Property | Value |
|---|---|
| Raw rows | 5990 (5 columns, no missing cells) |
| Exact replicated rows collapsed | 70 rows in 40 groups, max multiplicity 8 — replicates are byte-identical in both value columns |
| Analysis records | 5920 (`output/results/analysis_table.tsv`) |
| Stages | 18, `stage_index` 0…17, ordered |
| Interaction families | 285 (`interaction_id`) |
| Cell pairs | 9 (`cell_pair_id`) |
| Analysis units | 1105 (`unit_id` = interaction × cell pair) |
| Unit panel | median 3 stages per unit; 14 units in all 18 stages; 387 units in a single stage; 486 units have internal gaps |

The declared analysis unit is one interaction-cell-pair combination followed across ordered stages, with
repeated stages for the same pair treated as dependent. Consequently every stage-level quantity in this
report is estimated either with **within-unit centring** (each unit acts as its own control) or with
**family-clustered resampling**; the 5920 records are never treated as independent.

**A composition trap that changes the headline.** The naive record-weighted per-stage mean and a
family-equal-weighted curve disagree sharply at the start of the series (B0 vs B1 in Fig. 1A; e.g. stage 0:
−0.698 vs −0.864; stage 12: 0.650 vs 0.310). Families differ enormously in mean strength (SD of family
means 1.43) and in size (median 10, max 130), and family mean strength correlates +0.58 with the number of
stages a family occupies. Any stage summary weighted by raw record counts therefore partly reports *which
families happen to be present*, not *how strong interactions are at that stage*. This is why the report
carries five estimators rather than one.

---

## Finding 1 — Interaction strength rises across ordered stages, and the trend transports to held-out families

**Object and scope.** All 1105 `(interaction_id, cell_pair_id)` units, 18 ordered stages. The measured
quantity is `interaction_strength_like_value`; the claim is about its *within-unit* developmental profile.

**Relationship and direction.** Within-unit-centred strength increases with stage order. The per-stage
curve is +0.415, +0.295, +0.252, +0.273, +0.462, +0.448, +0.634, +0.745, +0.749, +0.694, +0.861, +0.859,
+0.741, +0.874, +0.853, +0.880, +0.873, +1.010 for stages 0…17.

**Core quantitative result (analysis unit = one unit followed across stages).**

- Within-unit OLS slope = **+0.0418 per stage**; family-clustered bootstrap (B = 1000, resampling the 285
  families) 95 % CI **[+0.0293, +0.0544]**.
- Stage-curve Pearson r = **+0.922**, Spearman ρ = +0.922 over the 18 stage means.
- Per-unit slopes (579 units with ≥ 3 stages): mean **+0.0529**, median +0.0554; **72.7 % positive**;
  t(578) = 9.27, p = 3.5 × 10⁻¹⁹; Wilcoxon p = 8.3 × 10⁻³²; sign test 421 vs 158, p = 1.4 × 10⁻²⁸.
- **Out-of-sample transport.** Under leave-one-family-out holdout (5533 records, 214 families, 718 units),
  predicting each held-out record as its unit's leave-one-out mean **plus** a stage offset δ[s] estimated
  only from *other* families reduces MAE from 0.4681 to 0.4275 (**−8.7 %**) and RMSE by 9.3 %.

**Supporting analysis and result evidence.**
`output/analysis/02_stage_trend.py` → `output/results/stage_trend.json`;
`output/analysis/04_trajectories_holdout.py` → `output/results/trajectories_holdout.json`;
`output/analysis/05_structure_decomposition.py` → `output/results/structure_decomposition.json`;
`output/analysis/13_holdout_replication.py` → `output/results/holdout_replication.json`;
figure `output/figures/F1_stage_trend_and_distribution.png`, `output/figures/F4_holdout_transport.png`.

**Robustness.**

- *Composition baselines.* All four controls agree in sign and significance: record-weighted (r = 0.79,
  slope +0.078), family-equal (r = 0.71, +0.061), post-stratified (r = 0.88, +0.076), within-unit centred
  (r = 0.92, +0.042). The effect is not an artefact of which families appear when.
- *Eleven alternative definitions* (`output/analysis/12_robustness_variants.py`): the trend keeps its sign
  and significance under excluding stage 0, per-stage medians, 1/99 % winsorising, 3-SD trimming,
  restricting to units with ≥ 6 stages, floor-p-only records, non-floor-p-only records, inverse-family-size
  weighting and inverse-unit-coverage weighting — **10 of 11 variants**.
- *Family-clustered vs record-level inference.* All CIs and tests above resample or block on family; the
  record-level p-values in the JSON are reported only as descriptive companions.
- *Permutation control.* Re-running the identical holdout pipeline with `stage_index` permuted within each
  unit yields a null improvement of **−0.0011 ± 0.0004**, against the observed +0.0406; the observed value
  exceeds all 200 permutations (p < 0.005). The same control using 500 permutations gives p = 0.002.
- *Cell-pair holdout* (a second, non-nested holdout unit): MAE 0.4681 → 0.4271, −8.7 %.

**Counterevidence, limitation, alternative explanations.**

- **This is a boundary, not a footnote.** Transport is real *in aggregate* but **not uniform per family**:
  102 of 214 held-out families improved and 112 worsened (two-sided sign test 102 vs 112, p = 0.5385).
  Per-family MAE
  gain correlates +0.34 with family size and +0.40 with the number of stages a family occupies, so the
  aggregate effect is carried mainly by the better-covered families. See Finding 4.
- **The single most fragile variant:** restricting to the 14 units observed in *all* 18 stages drops the
  slope from +0.0418 to **+0.0258** (r = 0.404). The well-covered panel is tiny and mostly late.
- **Shape is only approximately linear.** A quadratic term adds little (R² 0.850 → 0.872) but the curve is
  not a straight line: stage 0 is an outlier low point (rank 4 of 18 under within-unit centring). The
  series is better described as a rise concentrated in stages 0 → 7 that then plateaus than as steady
  per-stage accumulation. Treating +0.042/stage as a linear developmental rate is a simplification this
  data does not license.
- **Stage order is ordinal only.** `stage_index` is an anonymous ordered integer; nothing in the table
  establishes that consecutive indices are equally spaced in developmental time, so a per-stage *rate* is
  not a rate in any physical unit. The sign and ordering of the trend are the defensible claims.
- **Alternative explanation not excluded:** the shift could reflect a change in measurement conditions or
  in the composition of *units* rather than a change in the underlying biology. The holdout rules out
  family identity and stage occupancy as explanations; it cannot separate a biological trajectory from a
  stage-correlated technical drift, because no technical covariate exists in the frozen table.

**Executable falsification.** Take the 14 units observed in all 18 stages, drop the largest family, and
test whether the within-unit slope remains positive with a family-clustered CI excluding zero. If the
trend were a genuine developmental property it should survive on a fully balanced panel; the present
estimate does not (slope +0.0258, r = 0.404), so this test is the sharpest available discriminator.

---

## Finding 2 — Interaction identity, not stage, carries most of the structure

**Object and scope.** All 5920 deduplicated records and all 1105 unit means.

**Relationship and direction.** Grouping by interaction family or by analysis unit explains far more of
`interaction_strength_like_value` than the stage axis does. The data behave as a table of persistent
pair-specific levels.

**Core quantitative result.**

| Grouping | Share of record-level variance explained | Permutation result |
|---|---|---|
| `interaction_id` (285 families) | **0.836** (unit level 0.880) | observed between-family variance 2.055 vs shuffle null 0.722 ± 0.087, z = 15.2, p ≤ 0.002 |
| `unit_id` (1105 units) | **0.901** | — |
| `cell_pair_id` (9 pairs) | 0.040 | observed 0.178 vs null 0.0056 ± 0.0032, z = 53.8, p ≤ 0.002 |
| `stage_index` (18 stages) | **0.018** | — |

Record-level partition: 90.1 % of variance is *between* units, 9.9 % within units; the stage axis accounts
for 1.8 % of the total, which is 18.6 % of the within-unit slice.

**Supporting analysis and result evidence.**
`output/analysis/01_data_audit.py` → `output/results/data_audit.json`;
`output/analysis/05_structure_decomposition.py` → `output/results/structure_decomposition.json`;
`output/analysis/10_block_structure.py` → `output/results/block_structure.json`;
figure `output/figures/F1_stage_trend_and_distribution.png` (panel B shows the bimodal marginal
distribution created by the mixture of family levels).

**Robustness.**

- The family signal is **not an artefact of unequal stage occupancy**: after adjusting for stage
  composition, the between-family variance retains **85 %** of its raw value (1.749 vs 2.055) and remains
  far outside its own shuffle null (z = 13.8, p ≤ 0.002).
- The family effect is **not driven by family size**: label shuffling that preserves the exact size of
  every family still collapses between-family variance to 0.722.
- **Split-half reliability of family means is r = 0.834** (early vs late stages), so family levels are a
  reproducible property of the family, not stage noise.
- Cell-pair identity adds a **real but small** increment beyond family: 1.8 % of total unit-level variance,
  i.e. 14.6 % of the residual left after family, against a shuffle null of 25.7 ± 1.7 (z = 14.3, p ≤ 0.002).
  Pair effects are only moderately reproducible across halves (r = 0.557).
- A degenerate variant was detected and discarded rather than reported: grouping by
  `(interaction_id, cell_pair_id)` *is* `unit_id`, so it explains 100 % of unit-mean variance **by
  construction**. That figure is not evidence and is not used.

**Counterevidence, limitation, alternative explanations.**

- The 0.836 figure is inflated relative to a random-effects ICC estimated from mean squares (**0.829**),
  because the η² form is a fixed-effects quantity on unequal group sizes. The substantive conclusion is
  unchanged, but the exact percentage is definition-dependent; this report quotes both.
- **Family mean strength rises with the number of stages a family occupies** (Spearman +0.58). This is a
  genuine confound between "how strong is this family" and "how well was it sampled", and it is exactly why
  the family-equal and post-stratified curves, not the record-weighted curve, are the primary stage
  estimators.
- The identifier blocks are anonymous. Whether "families" correspond to ligand–receptor pairs, cell types
  or any other biological grouping cannot be determined from this dataset, and no such interpretation is
  offered.

**Executable falsification.** Hold out families and cell pairs jointly: if instead of 214 held-out families
the 9 cell pairs were held out *as well as* the family, a purely pair-driven artefact would lose its
predictive advantage. The present pair-level holdout retains the full −8.7 % (MAE 0.4681 → 0.4271), which
argues against a pure pair artefact but does not exclude a family-level one.

---

## Finding 3 — The association-evidence column is a coarse floor/non-floor tier, not a graded strength readout (null / boundary result)

**Object and scope.** `association_pvalue` across all 5920 records; specifically whether it indexes
`interaction_strength_like_value`.

**Relationship and direction.** The column takes only **six discrete levels** (0.0009, 0.01, 0.02, 0.03,
0.04, 0.05) and **87.6 % of records sit at the floor 0.0009**. Only the floor/non-floor distinction carries
information; the five non-floor levels are **not ordered** by strength.

**Core quantitative result.**

| Framing | Spearman ρ(p, strength) | p |
|---|---|---|
| Raw, all records | −0.285 | 7 × 10⁻¹¹¹ |
| Within-unit centred | −0.134 | 3 × 10⁻²⁵ |
| Within-family centred | −0.220 | 1 × 10⁻⁶⁵ |
| **Non-floor levels only** (n = 736) | **−0.041** | **0.272** |
| Non-floor levels only, within-unit | **+0.007** | **0.844** |

Among the five non-floor levels the mean strength spans only 0.28 units (vs a global SD of 1.75) and is
non-monotone (p = 0.04 has the highest mean, p = 0.05 the lowest); Kruskal–Wallis p = 0.097. The full
six-level tier explains 0.018 of within-unit strength variance, of which essentially all comes from the
floor split (0.016); the non-floor levels themselves explain 0.008.

The floor/non-floor contrast *is* real but small: paired within-unit difference **+0.371 strength units**
(floor higher in 223 of 336 units, Wilcoxon p = 2.4 × 10⁻¹⁵), within-unit AUC 0.618, Cohen's d ≈ 0.38.
The column also does **not** track absolute magnitude: |strength − unit mean| is flat across tiers
(0.385 floor vs 0.366 non-floor; Spearman with |deviation| = −0.078).

**Supporting analysis and result evidence.**
`output/analysis/03_evidence_coupling.py` → `output/results/evidence_coupling.json`;
independent replication in `output/analysis/11_evidence_tier.py` → `output/results/evidence_tier.json`;
figure `output/figures/F2_evidence_tier.png`.

**Robustness.** The raw ρ of −0.285 is the *most* flattering framing and is the one a naive analysis would
quote; it is mostly between-unit composition, and it collapses to −0.041 (p = 0.272) as soon as the five
non-floor tiers are examined on their own. A unit-blocked permutation test (2000 permutations, shuffling
the floor label *within* each unit) confirms the floor contrast survives unit blocking
(observed +0.211 centred units, p < 0.0005), while the same test applied to non-floor ordering does not
produce a significant effect. The independent stream reproduced every one of these figures from a separate
code path.

**Counterevidence and alternative explanations.**

- The column is **not degenerate or constant**: 22 of 718 multi-stage units are always non-floor, and
  360 are always floor, so the non-floor levels are genuinely used. The claim is that they are
  *uninformative about strength*, not that they are absent.
- The floor value 0.0009 looks like a censoring floor (a "most significant" report bucket) rather than a
  computed p-value. If it is a floor, then the true p-distribution is left-censored and the floor/non-floor
  contrast understates the real evidence difference. This cannot be verified from the frozen table.
- An alternative reading is that the non-floor levels encode a *different* quantity (e.g. a per-stage
  detection threshold rather than a per-pair association), which the table does not distinguish. This is
  compatible with the observed non-monotonicity.

**Executable falsification.** Recompute the same six statistics on a table where the non-floor levels are
replaced by a single pooled "non-floor" flag. If the five non-floor levels carried ordering information,
collapsing them would destroy a significant relationship; in these data it changes nothing (ρ = −0.041 →
p = 0.272 either way).

---

## Finding 4 — A large share of individual units is transient, and the stage trend's transport is heterogeneous across families (conditional conflict / applicability boundary)

**Object and scope.** The 579 units observed in ≥ 3 stages (taxonomy) and the 214 held-out families
(transport).

**Relationship and direction.** Two conditional restrictions on Finding 1. First, a substantial minority of
units do not follow the aggregate trend at all. Second, the aggregate out-of-sample gain is **not** a
per-family guarantee.

**Core quantitative result.**

*Trajectory taxonomy* (thresholds: sign-reversing if min < 0 < max; else emerging/declining if |slope| >
0.15; else persistent if SD ≤ 0.50; else unresolved):

| Class | Units | Share |
|---|---|---|
| Persistent level | 239 | 41.3 % |
| **Transient, sign-reversing** | **210** | **36.3 %** |
| Unresolved flat | 66 | 11.4 % |
| Emerging | 49 | 8.5 % |
| Declining | 15 | 2.6 % |

*Heterogeneous transport.* Leave-one-family-out MAE gain is **positive for 102 families and negative for
112** (sign test 102 vs 112, p = 0.5385) even though the pooled gain is +8.7 % and beats every permutation. Per-family
gain correlates +0.34 with family record count and +0.40 with the number of stages a family occupies.

**Supporting analysis and result evidence.**
`output/analysis/04_trajectories_holdout.py` → `output/results/trajectories_holdout.json`;
`output/analysis/05_structure_decomposition.py` → `output/results/structure_decomposition.json`
(`partB_heterogeneity`); `output/results/family_holdout_annotated.tsv`;
figure `output/figures/F3_trajectory_taxonomy.png`, `output/figures/F4_holdout_transport.png`.

**Robustness.**

- The **transient count is completely threshold-invariant**: it is exactly 210 across the entire 12-point
  threshold grid (slope thresholds 0.10/0.15/0.20/0.30 × flat-SD thresholds 0.30/0.50/0.75). This is
  because it is defined by a sign reversal, which no threshold choice can affect.
- The **emerging/declining** counts are the threshold-sensitive ones (49/15 at slope 0.15, falling to
  4/4 at 0.30), and the persistent/unresolved split moves a great deal (persistent 104 → 309 across the
  grid). Those three class sizes should be read as *illustrative*, not as estimates; the report registers
  the taxonomy as evidence about transience, not about those categories.
- Sign instability is corroborated at unit level: of 718 multi-stage units, **491 (68.4 %) hold a constant
  sign** across all their stages and **227 (31.6 %) reverse sign** — an independent construction agreeing
  with the 36.3 % transient figure on the 579-unit subset.

**Counterevidence, limitation, alternative explanations.**

- "Transient" here means *sign reversal of a value whose own reliability is unmeasured*. The table gives
  one value per unit-stage with no replicate, so there is no way to separate genuine biological
  reorganisation from measurement noise. A unit whose true level is near zero will cross zero readily.
- Consequently the taxonomy **cannot** be read as "36 % of interactions genuinely switch on and off". The
  defensible claim is narrower: *for a large fraction of units, the sign of the measured strength is not a
  stable property*.
- The heterogeneity result cuts against over-reading Finding 1. The correct statement is that the stage
  trend is a **distributional** property of the table that improves average prediction, not a law that
  holds for any given family. Reporting the pooled −8.7 % without the 102/112 split would overstate it.
- Per-family gains for small families are estimated from very few records (families with 2–5 records appear
  at both extremes of the gain distribution), so the per-family correlation with size is partly a
  precision artefact.

**Executable falsification.** Repeat the taxonomony using a bootstrap over stages within each unit: if the
transient class were driven by noise, its size would fall sharply when units whose sign is unstable only
under resampling are reclassified; if it were real reorganisation, the class would persist. The relevant
recomputation is `stage_trend.json → unit_level.n_sign_reversing` versus the per-unit slope distribution.

---

## Finding 5 — The evidence verdict is labile for the same unit across stages, even where strength is stable

**Object and scope.** The 718 units observed at ≥ 2 stages; comparing the stability of the *strength*
measurement against the stability of the *evidence tier*.

**Relationship and direction.** For the same analysis unit, `interaction_strength_like_value` is a highly
reproducible property while the floor/non-floor evidence verdict is not, even though both are reported for
the same unit-stage.

**Core quantitative result.**

| Property, same units across stages | Split-half agreement |
|---|---|
| Strength (early vs late unit means) | **r = 0.908** (restricted to ≥ 2 obs per half: 0.931) |
| Evidence verdict (floor fraction ≥ 0.5) | agreement 0.875, Cohen's **κ = 0.122** |

Of the 718 units observed at ≥ 2 stages, **360 (50.1 %)** are always at the p floor, **22 (3.1 %)** are
always non-floor, and **336 (46.8 %)** show a mixed verdict across their stages, counted over each unit's
*observed* stages. (The main body of `03_evidence_coupling.py` reported 8 / 22 / 688 because it treated
unobserved stages as non-floor in the all-18-column pivot; that is a bug. The correct counts match the
independent stream `11_evidence_tier.py` and this recomputation.) Consecutive-stage transitions:
P(floor next | floor now) = 0.931, P(floor next | non-floor now) = 0.708 — a non-floor verdict reverts to
the floor about 71 % of the time. Split-half reliability of the floor fraction r = 0.372 versus r = 0.924
for strength.

**Supporting analysis and result evidence.**
`output/analysis/03_evidence_coupling.py` → `output/results/evidence_coupling.json`;
independent replication in `output/analysis/11_evidence_tier.py` → `output/results/evidence_tier.json`
(`section4_lability`); figure `output/figures/F2_evidence_tier.png`.

**Robustness.**

- Unit length is controlled: comparing against an i.i.d. benchmark that preserves each unit's observed
  number of stages, the *expected* counts would be 316 always-floor and 399 mixed against the observed
  360 and 336, so real within-unit clustering exists (dispersion 2.25). The verdict is not pure
  stage-to-stage noise, but its persistence is still far below that of strength.
- The result holds under both the majority-vote and floor-fraction definitions of a unit's verdict.
- The stage-share of the floor is itself non-monotone (0.662 at stage 0 rising to 0.956 at stage 9, range
  0.294), so the instrument's own operating point drifts across the developmental series.

**Counterevidence, limitation, alternative explanations.**

- This is **not** simply a restatement of Finding 3. Finding 3 says the non-floor tiers are unordered
  *cross-sectionally*; this finding says the verdict is *temporally* unstable within a unit. A column could
  in principle be unordered yet stable, or ordered yet labile; here it is both unordered and labile.
- The most likely mechanical explanation is a **threshold-crossing artefact**: if strength and the
  evidence tier are monotonically related with a threshold near each unit's mean level, small fluctuations
  in strength flip the verdict while leaving strength itself highly correlated. That account is consistent
  with every number here and is the more parsimonious reading than any biological one. It cannot be tested
  directly because the table has no replicate measurements.
- The 22 always-non-floor units (3.1 %) are a small cell; any statistic conditioning on them is unstable
  and is not used.

**Executable falsification.** If the tier were a stable property, then for units whose strength split-half
correlation exceeds 0.9 the verdict agreement should also be high. Test this conditional relationship; a
flat agreement across the strength-reliability range would confirm the threshold-crossing account.

---

## 8. Cross-cutting limitations

1. **No causality, no mechanism.** `association_pvalue` and `interaction_strength_like_value` are
   association-shaped quantities on anonymous identifiers. Nothing here identifies direct biological
   communication, signalling direction, or causal effect, and no such claim is made anywhere in this report.
2. **Anonymous identifiers.** `interaction_id`, `cell_pair_id` and `stage_index` carry no biological
   labels. Stage indices are ordered but not metrically spaced; identifiers are not convertible to cell
   types, genes, or ligands. Any interpretation in biological terms is outside what this dataset supports.
3. **Uneven, gappy panels.** Median 3 stages per unit, 486 units with internal gaps, only 14 units complete
   across all 18 stages, and 387 units in a single stage. All panel-based estimates are therefore weighted
   towards better-covered units, and the balanced-panel estimate (Finding 1 robustness) is materially weaker.
4. **No replicate measurements.** With one value per unit-stage, measurement noise and true variation are
   not separable. This bounds Findings 4 and 5 in particular.
5. **Some class sizes are threshold artefacts.** The persistent/unresolved/emerging/declining counts in
   Finding 4 move substantially with the threshold; only the sign-reversing count (210) is
   threshold-invariant.
6. **Composition confounding is intrinsic.** Family identity correlates with panel coverage (+0.58); no
   weighting scheme removes this completely, which is why the conclusion is stated as an ordering result
   rather than a rate estimate.
7. **Failed and repaired steps are recorded, not hidden.** A loop-based holdout implementation timed out and
   was replaced with a vectorised one (`output/reproduce.md` §6). A first attempt at the permutation control
   reused residuals centred on the true stage labels, which made permutations degenerate; the centreing is
   recomputed from the permuted labels.

## 9. External sources

None. No external literature, dataset, or precomputed result was used, downloaded, or merged. All evidence
is from `data/stage_indexed_interactions.tsv`. Because no external source was used, no `T0` source-date
verification was required.

## 10. Reproduction

See `output/reproduce.md` for environment, run order, method definitions and the recorded failures.

```bash
python3 output/analysis/01_data_audit.py
python3 output/analysis/02_stage_trend.py
python3 output/analysis/03_evidence_coupling.py
python3 output/analysis/04_trajectories_holdout.py
python3 output/analysis/05_structure_decomposition.py
python3 output/analysis/06_figures.py
python3 validate_output.py output
```

### Finding-to-evidence index

| Finding | Analysis source | Result evidence |
|---|---|---|
| 1 — stage trend and transport | `output/analysis/02_stage_trend.py`, `output/analysis/05_structure_decomposition.py`, `output/analysis/12_robustness_variants.py` | `output/results/stage_trend.json`, `output/results/structure_decomposition.json`, `output/results/robustness_variants.json`, `output/figures/F1_stage_trend_and_distribution.png` |
| 2 — identity dominates stage | `output/analysis/01_data_audit.py`, `output/analysis/10_block_structure.py` | `output/results/data_audit.json`, `output/results/block_structure.json`, `output/results/structure_decomposition.json` |
| 3 — evidence column is a coarse tier | `output/analysis/03_evidence_coupling.py`, `output/analysis/11_evidence_tier.py` | `output/results/evidence_coupling.json`, `output/results/evidence_tier.json`, `output/figures/F2_evidence_tier.png` |
| 4 — transience and heterogeneous transport | `output/analysis/04_trajectories_holdout.py`, `output/analysis/13_holdout_replication.py` | `output/results/trajectories_holdout.json`, `output/results/holdout_replication.json`, `output/results/family_holdout_annotated.tsv`, `output/figures/F3_trajectory_taxonomy.png`, `output/figures/F4_holdout_transport.png` |
| 5 — evidence verdict lability | `output/analysis/03_evidence_coupling.py`, `output/analysis/11_evidence_tier.py` | `output/results/evidence_coupling.json`, `output/results/evidence_tier.json` |