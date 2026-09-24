# Reproduction Guide

Task: `Life_05_developmental_interaction_structure` — Stage-Indexed Interaction Structure Across Development
Time boundary `T0`: 2022-01-01. No external scientific data was used; all evidence comes from `data/`.

## 1. Environment

- Python 3.11.2
- `numpy` 2.4.6, `pandas` 3.0.5, `scipy` 1.17.1, `matplotlib` 3.11.2
- No `statsmodels`, no `scikit-learn`. All models (fixed-effects centring, permutation tests,
  bootstrap) are implemented directly on `numpy`/`scipy`. No other Python package is required.

`scipy` and `matplotlib` were installed with `pip` during this run; everything else was present.

## 2. Input and integrity

Single input file: `data/stage_indexed_interactions.tsv`
sha256 `67ab6ace870c985522f905801f0eed680818a0c73225ca113057ba12fdea992f` (verified against
`data_manifest.json` by `output/analysis/01_data_audit.py`; the script aborts on mismatch).

Raw: 5990 rows, 5 columns, no missing cells. The frozen table contains 70 exact replicated rows
spanning 40 `(stage_index, interaction_id, cell_pair_id)` groups (max multiplicity 8). Every
replicate carries an identical `association_pvalue` and an identical
`interaction_strength_like_value`, so replicates add no information and are collapsed to one
record per group, giving the 5920-row analysis table `output/results/analysis_table.tsv`
(1105 units, 18 stages, 285 interaction families, 9 cell pairs).

## 3. Run order

Run from the project root. Each script prints its JSON result and writes it to `output/results/`.

```bash
python3 output/analysis/01_data_audit.py            # integrity + schema + variance structure -> data_audit.json
python3 output/analysis/02_stage_trend.py           # stage trend under 5 estimators         -> stage_trend.json
python3 output/analysis/03_evidence_coupling.py     # p-tier vs strength, tier lability      -> evidence_coupling.json
python3 output/analysis/04_trajectories_holdout.py  # taxonomy + family holdout              -> trajectories_holdout.json
python3 output/analysis/05_structure_decomposition.py  # variance partition + permutation control -> structure_decomposition.json
python3 output/analysis/06_figures.py               # figures F1-F4                          -> output/figures/*.png
```

Delegated independent streams (same input, independent implementations):

```bash
python3 output/analysis/10_block_structure.py            # family/unit permutation structure -> block_structure.json
python3 output/analysis/11_evidence_tier.py              # evidence-tier instrument profile -> evidence_tier.json
python3 output/analysis/12_robustness_variants.py        # 11 alternative trend definitions -> robustness_variants.json
python3 output/analysis/13_holdout_replication.py        # independent holdout replication   -> holdout_replication.json
python3 output/analysis/independent_rederr.py            # full independent re-derivation of every headline number from the RAW tsv -> independent_rederr.json
```

All randomness is seeded (`numpy.random.default_rng(20220920)` in the permutation and bootstrap
steps; the shuffle control in `05` uses the same seed), so every number below is reproducible
exactly.

## 4. Definition of the analysis unit

The declared analysis unit is one `(interaction_id, cell_pair_id)` combination followed across
ordered stages; repeated stages for the same pair are dependent. This is implemented as
`unit_id = interaction_id + "|" + cell_pair_id` in `analysis_table.tsv`. Because stages repeat
within a unit, every stage-level claim in the report is estimated either with **within-unit
centring** (subtract each unit's own mean so each unit acts as its own control) or with
**family-clustered resampling**, never by treating the 5920 records as independent.

## 5. Key method definitions

| Quantity | Definition | Where |
|---|---|---|
| B0 record-weighted curve | per-stage mean of all records; the naive baseline | `02_stage_trend.py` |
| B1 family-equal curve | per-stage mean over family means, then across families | `02_stage_trend.py` |
| B3 post-stratified curve | records reweighted so each family's within-stage share equals its table-wide share | `02_stage_trend.py` |
| B4 within-unit centred curve | per-stage mean of `strength − unit mean`, shifted by the grand mean | `02_stage_trend.py` |
| within-unit slope | OLS slope of `strength − unit mean` on `stage_index` | `02_stage_trend.py` |
| family-clustered bootstrap | resample the 285 families with replacement, B = 1000 | `02_stage_trend.py` |
| ICC | share of record-level variance carried by group means (population variance) | `01_data_audit.py` |
| unit-blocked permutation | shuffle the p-floor label **within each unit** so the unit's record count is preserved | `03_evidence_coupling.py` |
| trajectory classes | sign-reversing if `min < 0 < max`; else emerging/declining if `|slope| > 0.15`; else persistent if `SD ≤ 0.50`; else unresolved | `04_trajectories_holdout.py` |
| leave-one-family-out holdout | `null = unit leave-one-out mean`; `trend = null + δ[s]`, where `δ[s]` is estimated only from families other than the held-out one | `04`, replicated vectorised in `05` |
| stage-shuffle control | same pipeline with `stage_index` permuted within each unit | `05`, independently in `13` |

## 6. Failures encountered and repaired

These are recorded rather than hidden; each was found and fixed, and the numbers in
`Result.md` come from the repaired code paths.

1. **Loop-based holdout timed out.** The first version of the leave-one-family-out test in
   `04_trajectories_holdout.py` computed the per-record stage offset inside a Python loop over all
   5920 records and did not finish inside the timeout; the process was killed. It was replaced in
   `05_structure_decomposition.py` by a vectorised `numpy.add.at` family-by-stage grid
   (`stage_sum[cidx] − grid_sum[fam_codes, cidx]`), which produces the same estimator in under a
   second. The `04` script retains its own loop-based implementation so the two are genuinely
   independent code paths; the two agree (MAE improvement 0.0391 vs 0.0406, both ≈ 8.4–8.7 %).

2. **Degenerate permutation control in the delegated replication.** `13_holdout_replication.py`
   computed its residuals once from the *true* stage labels and then reused them while shuffling the
   stage labels. Because the permuted offsets therefore retained a non-zero stage mean, every
   permutation predicted far worse than the null and the control reported a spurious mean
   improvement of −0.90 instead of ≈ 0. Fixed by recomputing the centring from the permuted labels
   inside the permutation loop. The repaired script now reports a null improvement of
   −0.0011 ± 0.0004 against an observed +0.0406, matching `05_structure_decomposition.py` and its own
   500-permutation run independently.

3. **Two different per-family gain definitions.** `family_holdout_annotated.tsv` (written by `05`)
   scores each family by the *mean of its per-record absolute-error gains*, giving 103 improved /
   111 worsened. `holdout_replication.json` (written by `13`) scores each family by
   `MAE_null − MAE_trend` *within* that family, giving 102 / 112. Both are legitimate; the second is
   the one quoted in `Result.md` and drawn in `F4_holdout_transport.png`, and the figure reads its
   data directly from `holdout_replication.json` so the two cannot drift apart. The difference
   (one family) does not affect any conclusion.

4. **A degenerate variance share.** `05_structure_decomposition.py` originally reported that
   grouping by `(interaction_id, cell_pair_id)` "explains 100 %" of unit-mean variance. That grouping
   *is* `unit_id`, so the figure is true by construction and is not evidence. It is now reported with
   an explicit degenerate note and excluded from the conclusions.

5. **A verdict flag contradicting its own statistics.** `10_block_structure.py` gated the
   "family identity organises strength" verdict on an arbitrary 3× fold-change rule. The observed
   ratio is 2.85× but sits 15.2 null standard deviations above the shuffle mean, so the flag returned
   `false` while every other statistic in the same artifact supported `true`. The gate now uses the
   permutation evidence directly (p and z), and returns `true`.

   6. **Stale sign-test text in the report.** `Result.md` originally stated the per-family
   heterogeneity as "102 vs 111 excluding one floating-point tie, p = 0.58". The authoritative
   artifacts (`holdout_replication.json.per_group_summary` and
   `leave_one_family_out_per_family`, 214 families) show **102 improved / 112 worse, 0 ties**, with a
   two-sided binomial sign test `binomtest(102, 214, 0.5)` = **0.5385040002283777**. The report text
   was corrected to "102 vs 112, p = 0.5385" in both Finding 1 and Finding 4. This is a text
   correction to match the artifacts, not a change to any computed number.

   7. **Independent re-derivation roll-up bug.** `independent_rederr.py` set
   `all_headline_numbers_reproduce` by iterating only over top-level `bool` summary entries, which
   (a) skipped the nested per-check dicts (`F1_holdout_match`, `F2_family_share_match`,
   `F3_floor_share_match`, `F5_transition_match`) and (b) let the *intentionally* `False`
   negative-control flag `F5_lability_buggy_would_match` (which asserts the buggy 8/22/688
   NaN-pivot path does **not** reproduce the correct 360/22/336) force the whole flag to `False`.
   The roll-up now flattens every positive match flag (including the nested dicts) and excludes
   the negative-control flag. With this fix, `independent_rederr.json.all_headline_numbers_reproduce`
   is `true` while `F5_lability_buggy_would_match` correctly remains `false`. No tolerance was
   loosened and no value was hard-coded.

## 7. Deliverables

| Role | Path |
|---|---|
| Authoritative report | `output/Result.md` |
| Executed analysis source | `output/analysis/*.py` |
| Machine-readable results | `output/results/*.json`, `output/results/*.tsv` |
| Figures | `output/figures/F1..F4*.png` |
| Semantic validation receipt | `output/validation_receipt.json` |

Validate with:

```bash
python3 validate_output.py output
```