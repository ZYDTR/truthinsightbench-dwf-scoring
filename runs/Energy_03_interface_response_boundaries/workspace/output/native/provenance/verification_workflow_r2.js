export const meta = {
  name: 'tib-independent-verification-r2',
  description: 'Re-verify findings 3-5 plus provenance, validator, and determinism without schema-constrained output',
  phases: [
    { title: 'VerifyR2', detail: '3 finding verifiers + provenance auditor + validator auditor + determinism auditor' },
  ],
}

// Round-1 lesson: this runtime rejects the structured-output tool with
// "StructuredOutput is not allowed for this Conversation role", which caused
// every schema-constrained agent to fail its final step and retry. No schema
// option is used here; agents return findings as plain text.

const ROOT = '/data/user/conversations/conv-67a2d991-196f-410c-8cc5-6286dad6c56d/project'
const OUT = ROOT + '/output'
const RES = OUT + '/native/results'

const BAN = 'STRICT: do NOT modify, create, move, or delete any file in the workspace unless this task explicitly tells you to. Read and compute only; scratch files go in /tmp. Run every command yourself and never trust a number you did not compute. Report EVERY discrepancy, however small. If everything matches, say so explicitly and list what you checked, so your report doubles as evidence of coverage.'

const FORMAT = [
  '',
  'Return your answer as plain text in exactly this shape (no JSON, no tool call):',
  'VERDICT: <confirmed | confirmed_with_minor_issues | refuted>',
  'CLAIMS_CHECKED: <integer>',
  'CLAIMS_CORRECT: <integer>',
  'DISCREPANCIES:',
  '  - report_says: <...> | artifact_says: <...> | recomputed: <...> | severity: <material|minor|cosmetic>',
  '  (or the single line "  none")',
  'SUMMARY: <short paragraph>',
].join('\n')

const SPECS = [
  {
    key: 'F3',
    title: 'ordering boundary at the endpoints',
    claims: [
      'over the full 120 to 320 K window, 12 of the 28 main-panel curve pairs reverse their relative order at least once',
      'Spearman(rank at 120 K, rank at 320 K) = 0.310 with p = 0.51 for the main panel',
      'over the restricted 140 to 300 K window only 2 of 28 pairs reverse and Spearman(rank at 140 K, rank at 300 K) = 0.976 with p = 0.0043',
      'the 120 to 140 K step has Spearman = 0.595 with p = 0.186; every subsequent adjacent step is at least 0.952 and five of the nine later steps are exactly 1.000',
      'check panel: 1 of 6 pairs reverse over the full window and 0 of 6 over 140 to 300 K',
      'at 120 K, 5 of the 8 main curves have response below 0.6 while the maximum at that setting is 1.097',
      'condition_block1_3 is non-monotone in raw response; it participates in 4 of the 12 reversing pairs, and the pair condition_block1_3 vs condition_block2_4 reverses three times',
      'the exact sign test for the endpoint reversal count gives p = 0.83, i.e. the reversal count is NOT statistically significant on its own',
    ],
  },
  {
    key: 'F4',
    title: 'saturation boundary lies at or beyond the design maximum',
    claims: [
      'the maximum response anywhere in either panel is 93.01',
      'with a free ceiling, logistic ymax estimates span 51.99 to 130.55 with mean 81.83 and SD 24.93, and the ymax standard error averages 9.86',
      'exactly 1 of 8 main curves has fitted T50 above the maximum setting of 320 K, namely condition_block2_1 at 327.2 K',
      'fixing the ceiling at 100 changes fitted T50 by at most 10 K for 7 of 8 main curves, with max absolute change 14.7 K, and costs at most 0.023 in log-space R squared',
      'comparing logistic, Gompertz and Hill by BIC on log y: logistic wins 7 of 8 main curves and 3 of 4 check curves, with mean R squared 0.994 versus 0.976 (Gompertz) versus 0.889 (Hill)',
      'forward holdout (train T <= 280 K, test 300 and 320 K): the saturating logistic beats an unsaturated exponential in 12 of 12 curves, main 8 of 8 with sign-test p = 0.0039, check 4 of 4; it beats a last-value-persistence baseline in 11 of 12',
      'mean forward-holdout RMSE in log space: 0.166 for the saturating form versus 1.063 for the exponential, main panel',
      'dropping the 120 K onset setting changes fitted T50 by mean absolute 7.06 K and max absolute 14.70 K',
    ],
  },
  {
    key: 'F5',
    title: 'check panel partly duplicates the main panel; block labels carry no signal',
    claims: [
      'check condition_block1_4 is identical to main condition_block2_1 at all 11 settings, maximum absolute difference exactly 0.0',
      'check condition_block1_5 is identical to main condition_block2_2 at all 11 settings, maximum absolute difference exactly 0.0',
      'across all 12 curves there are 24 exact within-setting value coincidences beyond those two whole-curve duplicates',
      'exactly 2 check curves are genuinely new: condition_block1_1 and condition_block1_2',
      'those 2 genuinely new curves are not constant multiples of any main curve',
      'complete 4-of-8 enumeration of fitted parameters gives exact two-sided p = 0.743 for T50, 0.571 for w, and 0.086 for the ceiling, over 70 assignments',
      'the block interaction term in the pooled log-response model is not significant: F = 0.260 with df 1 and 77, p = 0.611, and the fitted block-slope ratio is 0.934',
      'with 4 curves per block the minimum achievable two-sided exact p-value is 0.029',
    ],
  },
]

phase('VerifyR2')
log('Round 2: 3 finding verifiers + provenance, validator, and determinism auditors (no schema-constrained output).')

const results = await parallel([
  ...SPECS.map((s) => () => agent([
    'You are an independent adversarial verifier for a scientific report. Task Energy_03_interface_response_boundaries.',
    '',
    'Workspace root: ' + ROOT,
    'Report: ' + OUT + '/Result.md   (read the section registering ' + s.key + ')',
    'Raw data, the ONLY direct evidence: ' + ROOT + '/data/anonymous_condition_response_curves.tsv and ' + ROOT + '/data/independent_condition_check_curves.tsv',
    'Artifacts: ' + RES + '/*.json and ' + RES + '/*.csv',
    '',
    'Finding under test: ' + s.key + ' - ' + s.title,
    '',
    'For every claim below, recompute the value from the raw TSVs with your own python3 code (numpy and pandas exist; scipy does NOT, so implement any statistic you need yourself). Do not import the workspace analysis modules and treat their output as truth. Compare three things per claim: what the report says, what the cited artifact says, what you recomputed. Any disagreement is a discrepancy.',
    '',
    'CLAIMS:',
    ...s.claims.map((c, i) => '  ' + (i + 1) + '. ' + c),
    '',
    'Be adversarial: look for claims that are true only under the exact original code path, boundaries stated too strongly, and numbers that do not appear in any artifact.',
    BAN,
    FORMAT,
  ].join('\n'), { label: 'verify:' + s.key, phase: 'VerifyR2' })),

  () => agent([
    'You are an independent auditor. Task Energy_03_interface_response_boundaries.',
    'Workspace root: ' + ROOT + '   Output tree: ' + OUT,
    '',
    'Audit PROVENANCE AND LOG INTEGRITY:',
    '1. Inventory every file under ' + OUT + '; confirm there are no symbolic links and no non-regular files.',
    '2. Every analysis script under ' + OUT + '/native/analysis/ should have a console transcript under ' + OUT + '/native/logs/. List which have one and which do not.',
    '3. ' + OUT + '/reproduce.md claims four method failures are retained in those logs. Verify each is really present and quote the actual log text: (a) a nested-model F test returning a NEGATIVE F of about -8.56, (b) an np.clip broadcast error in the sigmoid fitter, (c) a pip install reporting success while the import still failed, (d) Crossref DOI lookups returning no parsable record. Say for each whether it was found and whether the description matches.',
    '4. Read ' + OUT + '/native/provenance/external_sources.md. Confirm each source has a DOI or stable identifier plus a first-public date, and that every date is strictly before T0 = 2022-01-01. Flag any violation, and any source cited in Result.md but absent from the log.',
    '5. Confirm the SHA-256 hashes recorded for the two data files in ' + RES + '/results.json match the real files and the values in ' + ROOT + '/data_manifest.json.',
    '6. ' + OUT + '/Result.md section 3 lists five corrections C1-C5 from a previous audit. Verify C1, C2 and C5 are now correctly stated in the current report text by recomputing them.',
    BAN,
    FORMAT,
  ].join('\n'), { label: 'audit:provenance-logs', phase: 'VerifyR2' }),

  () => agent([
    'You are an independent auditor. Task Energy_03_interface_response_boundaries.',
    'Workspace root: ' + ROOT + '   Output tree: ' + OUT,
    '',
    'Re-run the public semantic validator and audit it:',
    '1. Run: cd ' + ROOT + ' && python3 validate_output.py output --receipt /tmp/val_audit.json . Report the real exit code and the full errors and warnings arrays.',
    '2. Run it a second time and check the receipt is identical apart from path fields. Report determinism.',
    '3. Independently confirm the validator is truthful using your own code: exactly one root-level Markdown file matching Result.md case-insensitively and non-empty; registered finding labels consecutive from 1 with a count between 2 and 5; at least one code-suffix file under the output tree; at least one machine-readable result artifact or figure; and each registered finding section naming an analysis file and a result artifact that really exist on disk.',
    '4. Try to DEFEAT the validator: find any way the report could pass while misrepresenting the evidence. Report whatever you find.',
    '5. Confirm no symbolic links and no zero-byte files exist under ' + OUT + '.',
    BAN,
    FORMAT,
  ].join('\n'), { label: 'audit:validator', phase: 'VerifyR2' }),

  () => agent([
    'You are an independent verifier checking REPRODUCIBILITY. Task Energy_03_interface_response_boundaries.',
    'Workspace root: ' + ROOT,
    '',
    'NOTE: this task REQUIRES you to re-run the analysis scripts, which WILL rewrite files under output/native/results and output/native/figures. That is expected and authorised for this task ONLY. Do not edit anything under output/native/analysis, do not touch data/, Result.md, or reproduce.md.',
    '',
    'The claim under test is that the pipeline is deterministic and re-runnable from scratch. Verify end to end:',
    '1. Record SHA-256 of every file under ' + OUT + '/native/results/ and ' + OUT + '/native/figures/ into a /tmp list.',
    '2. Run these nine scripts SEQUENTIALLY from the workspace root, capturing exit codes: run_analysis.py, run_stage_e.py, run_stage_f.py, run_stage_g.py, run_stage_h.py, run_stage_i.py, run_stage_j.py, make_figures.py, run_verification.py (all under output/native/analysis/). Report any non-zero exit and its last output lines. run_stage_h.py is by far the slowest and can take several minutes; give it a generous timeout rather than killing it.',
    '3. Re-hash the same files and diff against the step 1 list. Report EVERY changed, added, or removed file. PNGs may legitimately differ if the plotting library embeds a timestamp; if a PNG differs, say so explicitly and state whether the JSON and CSV artifacts are byte-identical.',
    '4. Re-run: cd ' + ROOT + ' && python3 validate_output.py output . Report the exit code and whether it still passes.',
    '5. Confirm the no-randomness claim from the CODE, not just from the observed hash match: search the analysis sources under ' + OUT + '/native/analysis/ for any use of random number generation and report what you find.',
    BAN,
    FORMAT,
  ].join('\n'), { label: 'audit:determinism', phase: 'VerifyR2' }),
])

return {
  F3: results[0],
  F4: results[1],
  F5: results[2],
  provenance: results[3],
  validator: results[4],
  determinism: results[5],
  agents_launched: 6,
  agents_returned: results.filter(Boolean).length,
}