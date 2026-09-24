export const meta = {
  name: 'bde-domain-workstreams',
  description: 'Parallel workstreams for bond-dissociation-enthalpy domain/boundary analysis',
  phases: [
    { title: 'Workstreams', detail: 'independent analysis dimensions fan out in parallel' },
    { title: 'Verify', detail: 'adversarial recomputation of each numeric claim' },
  ],
}

const ROOT = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'

const PREAMBLE = `
You are one workstream of a parallel scientific analysis of a bond-dissociation-enthalpy (BDE) benchmark.

WORKSPACE: ${ROOT}
- Data (read-only, the ONLY evidence): data/normalized_views/experimental_benchmark.tsv (543 rows),
  data/source_data/bde_db_full.csv.gz (484907 rows), data/source_data/experimental_source_tables.xlsx (Sheet_001, 543 rows).
- Shared helper module ALREADY WRITTEN AND VALIDATED: output/analysis/common.py
  Use it:
      import sys; sys.path.insert(0, '${ROOT}/output/analysis')
      import common
      df = common.build_analysis_table()   # 543 rows, benchmark joined to DB, with bond_class, db_bde, in_db_row, hac_parent, train_flag, per-channel errors
      db = common.load_db(); bench = common.load_benchmark(); xl = common.load_xlsx()
  It also provides: bond_class(), radical_center(), heavy_atom_count(), paired_bootstrap_ci(), paired_t_test(),
  wilcoxon(), cliffs_delta(), mae(), rmse(), save_json().
- Installed and importable: pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl==3.0.9, rdkit.
  NOTE: pandas.read_excel FAILS (needs openpyxl>=3.1.5) -> use common.load_xlsx() which reads via openpyxl directly.
  NOTE: matplotlib has no display; use matplotlib.use('Agg').
- NO NETWORK ACCESS. Do not attempt downloads. Use only workspace data.
- Write your machine-readable result artifact(s) to output/results/ with the exact filename given below,
  and any figure to output/figures/. Use common.save_json(obj, name) if convenient.
- RUN PYTHON CODE. A plan or a described method is NOT evidence. Every number you report must come from
  an executed script whose command you actually ran. If a library fails, keep the real error and use a simpler executable method.
- Report real numbers exactly as computed, including null/negative/ambiguous results. Never round away a failure.
- The experimental reference values come from heterogeneous literature and are NOT independent repeats from one lab. Treat 543 rows
  as 543 bond observations; unique molecules and unique fragment pairs are DIFFERENT analysis units. Say which unit each statistic uses.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['workstream', 'artifacts', 'key_numbers', 'narrative', 'caveats', 'commands_run'],
  properties: {
    workstream: { type: 'string' },
    artifacts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['path', 'description'],
        properties: { path: { type: 'string' }, description: { type: 'string' } },
      },
    },
    key_numbers: {
      type: 'array',
      description: 'The concrete numeric findings, each with the analysis unit it uses',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['claim', 'value', 'unit', 'n'],
        properties: {
          claim: { type: 'string' },
          value: { type: 'string' },
          unit: { type: 'string', description: 'analysis unit: bond observation / unique molecule / fragment pair / database row' },
          n: { type: 'string' },
        },
      },
    },
    narrative: { type: 'string', description: 'interpretation of these numbers in plain prose' },
    caveats: { type: 'array', items: { type: 'string' } },
    commands_run: { type: 'array', items: { type: 'string' } },
  },
}

const WORKSTREAMS = [
  {
    key: 'WS1-db-coverage',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS1 - Database inventory, structural domain and coverage boundary.

Answer with executed numbers:
1. Inventory of the complete database (bde_db_full.csv.gz): raw rows, unique molecules, unique (molecule,fragment1,fragment2)
   fragment-pair units, unique bde values. Show that raw rows != molecules != fragment pairs, with the counts.
2. Bond-class census: table of the 9 bond_type classes with counts on BOTH raw-row and unique-fragment-pair units.
   Compute the redundancy factor (raw rows / unique pairs) per class and explain which class drives row inflation.
3. Structural domain of the database: element set present, heavy-atom-count (HAC) range of the molecules, distribution of
   molecule size, how many molecules at each HAC. Use rdkit via common.heavy_atom_count / common.element_set.
   Compute ORDINARY-least-squares BDE summary (mean/sd/min/median/max) per bond class on unique-pair units.
4. Coverage boundary: how many of the 543 benchmark observations fall INSIDE vs OUTSIDE the database structural domain?
   Use (a) exact (molecule,fragment1,fragment2) row present, (b) parent molecule present, (c) parent HAC beyond the
   database HAC maximum, (d) elements present in the benchmark parent that never occur in the database.
   Report the counts and the HAC cutoff that defines the boundary.
5. Is the database exhaustive within its own domain? Within the benchmark rows whose parent molecule exists in the database,
   how many have an exact matching fragment pair? Report the split.

WRITE ARTIFACT: output/results/ws1_db_coverage.json containing every table above as records.
Also write a figure output/figures/ws1_domain.png (matplotlib Agg) with the bond-class census and the HAC distributions.
Report back the key numbers.`,
  },
  {
    key: 'WS2-channel-accuracy',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS2 - Head-to-head accuracy of the three estimation channels on the same experimental observations.

The benchmark has three estimate columns for the same 543 experimental values:
  quantum_estimate_bde_kcal_mol, learned_estimate_bde_kcal_mol, reference_estimate_bde_kcal_mol.
Answer with executed numbers:
1. Per channel: MAE, RMSE, median absolute error, mean signed bias, and the full error quantile table
   (min, 5%, 25%, 50%, 75%, 95%, max of the SIGNED error pred-experiment).
2. Calibration: regress each estimate on the experimental value. Report slope, intercept, Pearson r, R^2.
   State whether each channel is compressed (slope<1) or amplified, and what that implies about its error at the ends of the range.
3. PAIRED comparison of quantum vs learned (same 543 observations): mean and median of the per-observation
   |error| difference, a paired t-test and a Wilcoxon signed-rank test on the absolute-error difference, a bootstrap
   95% CI for the mean difference in absolute error, and Cliff's delta. State clearly whether one is better than the
   other or whether they are NOT DISTINGUISHABLE at this sample size. A null result is a valid and important answer.
4. Do the same paired comparison for quantum vs reference and learned vs reference.
5. Split the 543 observations by the workbook column in_reference_training_set (True=331, False=212; it exactly equals the
   subset with an exact database row). Report each channel's MAE on the training subset and on the held-out subset,
   and quantify the degradation. This is the closest thing to an out-of-sample test available in this task.
6. Correlation between the three channels' estimates, and correlation between their absolute errors. Does reference
   fail where the other two also fail, or independently?

WRITE ARTIFACT: output/results/ws2_channel_accuracy.json with all tables.
Also write output/figures/ws2_accuracy.png (error distributions + calibration).
Report back the key numbers.`,
  },
  {
    key: 'WS3-bondclass',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS3 - Error structure by chemical bond class and by structural environment.

common.build_analysis_table() gives a validated bond_class label (C-H, C-C, C-N, C-O, H-N, H-O, N-N, N-O, O-O) derived
from the fragment radical centres using sanitised RDKit parsing; it reproduces the database bond_type on all 331 matched
rows. Answer with executed numbers:
1. Error by bond class for each of the three channels: n, MAE, mean signed bias, median absolute error, per class.
   Include classes with small n and say plainly when n is too small to distinguish anything.
2. Is the dominant determinant of error the bond class, or simply the molecule size? Cross-tabulate MAE by bond class
   AND by parent HAC bin (e.g. <=5, 6-9, >=10). Report whether the size trend survives within a fixed bond class,
   or whether size is confounded with class.
3. Bound-character / environment effects you CAN measure from the data alone (no external data):
   - radical centre aromaticity and ring membership (from common.radical_center) -> error by that flag;
   - whether the parent contains a benzene ring, whether the parent contains N or O;
   - experimental BDE magnitude bin -> error (is error heteroscedastic? does each channel fail in the weak-bond region?).
4. Identify the region (bond class x size x magnitude) where each channel is worst, with n and MAE, and the region
   where each is best. Give a concrete quantitative statement of the applicability boundary.
5. Report honestly where the sample is too thin to support a claim (n < ~10) rather than asserting a class effect.

WRITE ARTIFACT: output/results/ws3_bondclass.json with all tables.
Also write output/figures/ws3_bondclass.png (per-class MAE with n annotated).
Report back the key numbers.`,
  },
  {
    key: 'WS4-failures',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS4 - Failure modes, outliers and the hard boundary of each channel.

Answer with executed numbers:
1. Tail census: for each channel, count and list observations with |error| > 5, > 10, > 15 kcal/mol.
   Give the observation_ids and the parent SMILES for the worst cases.
2. Are the extreme failures CONCENTRATED in a structurally identifiable family? Inspect the worst cases and characterise
   them with descriptors you can compute from the workspace only (HAC, ring count, aromatic ring count, element set,
   number of rotatable bonds via rdkit, radical centre aromatic/ring flag, bond class, whether the two fragments are
   identical). Run a concrete test of whether the failure set differs from the rest of the benchmark on those descriptors
   (report the group means and a rank-based p-value). State whether the characterisation is supported or only suggestive.
3. Steric / crowded case family: build a flag for parents where the two fragment radical centres are both in rings or
   the molecule has fused/bridged polycyclic character (e.g. ring count >= 3). Measure each channel's MAE with and
   without that flag. Report n for both groups.
4. The weak-bond region: the experimental range goes down to 15.2 kcal/mol. Report how many observations are below
   60, below 70, below 80 kcal/mol, and each channel's MAE/bias in each region. Quantify the divergence of the channels
   as the bond gets weaker (does the estimate-expt gap widen?).
5. Direction of failure: for the worst cases, do the two calibrated channels overshoot (estimate too HIGH) systematically?
   Test the sign of the signed error in the weak-bond region.
6. Explicitly state which of these characterisations are DATA-SUPPORTED and which remain UNRESOLVED at this sample size.

WRITE ARTIFACT: output/results/ws4_failures.json with the tail census, descriptor test, and region table.
Also write output/figures/ws4_failures.png (signed error vs experimental BDE, per channel, with the worst cases marked).
Report back the key numbers.`,
  },
  {
    key: 'WS5-provenance',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS5 - Provenance audit and experimental-reference heterogeneity.

This workstream asks what the data files actually ARE, and how much of the measured error is attributable to the
experimental reference values rather than to the estimators. Answer with executed numbers:
1. Cross-file consistency: verify the three files agree. Confirm the TSV and the xlsx Sheet_001 are row-for-row identical
   on molecule/fragments/observed/estimates (report the exact match counts for each of the 8 columns). Report the sha256 of
   each source file (common.sha256). Report the single extra workbook column in_reference_training_set and its exact
   correspondence with the exact-database-row subset.
2. Provenance of the quantum channel: on the 331 rows that have an exact database row, compare the database bde to the
   quantum estimate. Report the distribution of the difference (quantiles, max |diff|, fraction within 0.01/0.1/0.5 kcal/mol).
   Conclude, with the numbers, whether the quantum channel is the database's own value reproduced for this subset.
   Do the same comparison for the learned channel and report whether it agrees or disagrees.
3. Reference-channel characterisation: is reference_estimate a per-fragment-pair lookup (does it take a single value per
   (fragment_a,fragment_b) pair)? Report the number of distinct reference values vs distinct fragment pairs, and how many
   fragment pairs carry more than one distinct reference value. Then report the reference channel's error on the 331
   database-matched rows vs the 212 unmatched rows.
4. HETEROGENEITY BOUND: this is the core question. The experimental values are literature values, not repeats from one lab.
   Estimate how large the experimental reference uncertainty plausibly is using ONLY workspace evidence:
   (a) group observations sharing the same parent molecule and report the spread of their experimental BDE;
   (b) group observations sharing the same fragment pair and report the spread;
   (c) compare the channels' disagreement (quantum vs learned, quantum vs reference) against those spreads.
   Then state a defensible numeric floor: below roughly how many kcal/mol of MAE can this benchmark NOT distinguish two
   channels, given the reference-value heterogeneity it contains? Give the number and the reasoning.
5. State which conclusions from (2)-(4) are direct observation vs inference.

WRITE ARTIFACT: output/results/ws5_provenance.json with all tables and the numeric floor.
Also write output/figures/ws5_provenance.png (database-vs-quantum difference histogram; per-molecule experimental spread).
Report back the key numbers.`,
  },
  {
    key: 'WS6-robustness',
    prompt: `${PREAMBLE}
YOUR WORKSTREAM: WS6 - Robustness suite. This workstream exists to try to BREAK the other workstreams' claims.

Answer with executed numbers, for each check stating pass / fail / partially:
1. LEAVE-ONE-MOLECULE-OUT: group the 543 observations by parent molecule (419 unique; 86 molecules contribute >1 row).
   Recompute every channel's MAE after removing all rows belonging to each molecule in turn (or, if that is slow, the
   jackknife influence: MAE with molecule i removed, summarised over i). Report the max change in MAE and whether any
   single molecule drives the ranking of the channels. Also report MAE on the largest cluster-free subset (one row per
   molecule) so the statistics use independent-ish units.
2. ALTERNATIVE DEFINITION of the error: recompute the channel comparison using (a) RMSE, (b) median absolute error,
   (c) 90th-percentile absolute error, (d) fraction of observations within 2 kcal/mol, (e) fraction within 5 kcal/mol.
   State whether the quantum-vs-learned and quantum-vs-reference conclusions are stable across all five definitions.
3. OUTLIER TREATMENT: recompute with (a) the worst 5% of observations per channel removed, (b) the single worst
   observation removed, (c) 1%-99% winsorisation of the estimates. Report whether the ranking flips.
4. BATCH / CLASS LEAVE-OUT: recompute the channel comparison with each bond class removed in turn. Report which class,
   if any, is responsible for the overall ranking.
5. THRESHOLD SENSITIVITY: for the claim that the database covers only small molecules (HAC <= 10 in the database vs larger
   in the benchmark), recompute the coverage boundary at HAC thresholds 8, 9, 10, 11, 12 and report how the conclusion
   moves. Also test the boundary using an alternative size definition (molecular weight via rdkit Descriptors.MolWt).
6. PAIRED-UNIT SENSITIVITY: remove the 4 fragment-pair groups that appear more than once so that every observation is a
   unique fragment pair, and recompute the headline comparison. Report whether the conclusion holds.
7. Give an explicit verdict list: for each candidate claim you tested, is it ROBUST, FRAGILE, or NOT TESTABLE.

WRITE ARTIFACT: output/results/ws6_robustness.json with every check's numbers and the verdict list.
Also write output/figures/ws6_robustness.png (leave-one-molecule-out influence + alternative-loss comparison).
Report back the key numbers and the verdict list.`,
  },
]

phase('Workstreams')

const results = await parallel(
  WORKSTREAMS.map((ws) => () =>
    agent(ws.prompt, {
      label: ws.key,
      phase: 'Workstreams',
      schema: SCHEMA,
      agentType: 'general-purpose',
    })
  )
)

const ok = results.filter(Boolean)
log(`${ok.length}/${WORKSTREAMS.length} workstreams completed`)

// Barrier is genuine: the verifiers below must be pointed at the claims that
// were actually produced, which requires all workstreams' numbers at once.
const claims = ok.flatMap((r) =>
  (r.key_numbers || []).slice(0, 4).map((k) => ({ ws: r.workstream, claim: k.claim, value: k.value, unit: k.unit, n: k.n }))
)

log(`${claims.length} numeric claims collected for adversarial verification`)

phase('Verify')

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['claim', 'reproduced', 'value_found', 'method', 'verdict', 'notes'],
  properties: {
    claim: { type: 'string' },
    reproduced: { type: 'boolean' },
    value_found: { type: 'string' },
    method: { type: 'string', description: 'the exact independent computation performed' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'different_number', 'not_verifiable'] },
    notes: { type: 'string' },
  },
}

const verdicts = await parallel(
  claims.map((c, i) => () =>
    agent(
      `${PREAMBLE}

YOUR ROLE: INDEPENDENT ADVERSARIAL VERIFIER. Workstream "${c.ws}" reported this claim:

  CLAIM: ${c.claim}
  REPORTED VALUE: ${c.value}
  UNIT: ${c.unit}   n: ${c.n}

Your job is to REFUTE it. Do NOT reuse the workstream's script - write your OWN minimal independent computation from the
raw files (read the TSV/gz/xlsx directly with pandas/openpyxl; you may use common.py helpers but not the workstream's artifact).
If your independent number differs, say so with both numbers. If the claim is about the analysis unit, check that the unit
stated is actually the unit used (raw database row vs unique molecule vs unique fragment pair vs bond observation).
Pay special attention to: (a) whether a "unique pair" count was actually computed on unique pairs, (b) whether an
n-weighted mean was reported as an unweighted one or vice versa, (c) whether NaN/inf handling changed a count,
(d) whether a trend is confounded. Default to "refuted" or "different_number" if you cannot reproduce it.
Report the exact command you ran in the method field.`,
      { label: `verify-${i + 1}`, phase: 'Verify', schema: VERDICT, agentType: 'general-purpose' }
    )
  )
)

const good = verdicts.filter(Boolean)
const tally = good.reduce((acc, v) => {
  acc[v.verdict] = (acc[v.verdict] || 0) + 1
  return acc
}, {})

log(`verification tally: ${JSON.stringify(tally)}`)

return {
  workstreams: ok,
  claims_verified: good,
  tally,
  failed_workstreams: WORKSTREAMS.length - ok.length,
  failed_claims: claims.length - good.length,
}