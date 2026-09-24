export const meta = {
  name: 'bde-verify-claims',
  description: 'Adversarially verify the headline numeric claims of the BDE domain analysis',
  phases: [
    { title: 'Verify', detail: 'independent recomputation of each headline claim' },
  ],
}

const ROOT = '/data/user/conversations/conv-75e3fc1d-c4c9-41b5-800b-05f4632eec99/project'

// The workflow harness could not collect StructuredOutput from these agents in
// the first run, so verdicts are written to disk and read back afterwards.
const PREAMBLE = `
You are an INDEPENDENT ADVERSARIAL VERIFIER for a bond-dissociation-enthalpy (BDE) analysis.

WORKSPACE: ${ROOT}
DATA (read-only, the ONLY evidence): data/normalized_views/experimental_benchmark.tsv (543 rows),
  data/source_data/bde_db_full.csv.gz (484907 rows), data/source_data/experimental_source_tables.xlsx.
Helpers: output/analysis/common.py provides build_analysis_table(), load_db(), load_benchmark(), load_xlsx(),
  radical_center(), bond_class(), heavy_atom_count(), save_json(), mae(), rmse().
Installed: pandas numpy scipy scikit-learn matplotlib rdkit openpyxl.
NOTE: pandas.read_excel FAILS (openpyxl 3.0.9 < 3.1.5) - use common.load_xlsx().
NO NETWORK. Write your verdict JSON to the exact path given at the end. RUN REAL CODE.

YOUR JOB IS TO REFUTE. Recompute from the raw files with your OWN script. Do not import or re-run the
original workstream's script. If your number differs, report BOTH numbers. Check the analysis unit
(raw DB row vs unique molecule vs unique fragment pair vs bond observation) matches what is claimed.
Default to "refuted" or "different_number" when you cannot reproduce a value.
`

const CLAIMS = [
  {
    file: 'verify_1_db_scale.json',
    claim: `The database has 484907 raw rows but only 290664 unique (molecule,fragment1,fragment2) units and
42577 unique molecules; raw-row counts are ~1.67x inflated relative to the pair unit, and rows sharing a
(molecule,fragment1,fragment2) key never disagree on bde or bond_type.`,
    value: '484907 rows / 290664 unique (mol,f1,f2) / 42577 molecules / redundancy 1.6683 / 0 keys with varying bde',
  },
  {
    file: 'verify_2_coverage.json',
    claim: `The database domain is capped at heavy-atom count 10 (max over 42577 unique molecules) and contains only
C, N and O. Exactly 331 of the 543 benchmark observations have an exact (molecule,fragment1,fragment2) row in the
database; 369/543 have their parent molecule present; 150/543 (27.6%) lie beyond the database HAC maximum of 10.`,
    value: 'HAC max 10; elements {C,N,O}; 331 exact rows; 369 parent present; 150 beyond HAC 10',
  },
  {
    file: 'verify_3_channel_accuracy.json',
    claim: `On the 543 bond observations, MAE against experiment is quantum 2.0978, learned 2.4107, reference 5.4860
kcal/mol; RMSE 3.0595 / 3.6278 / 6.7596. The sample mean of the experimental values is 86.9446.`,
    value: 'quantum MAE 2.0978 / learned 2.4107 / reference 5.4860',
  },
  {
    file: 'verify_4_paired.json',
    claim: `Paired on the same 543 observations, quantum beats learned by only 0.3129 kcal/mol mean |error| difference
(paired t p=5.37e-05) but the Wilcoxon signed-rank test on that same difference gives p=0.553, and a sign test gives
246 quantum-better vs 284 learned-better (p=0.108). quantum vs reference is a large, unambiguous gap of 3.3882 kcal/mol
with both tests at p<1e-50.`,
    value: 'q-vs-l diff -0.3129, t p=5.4e-05, Wilcoxon p=0.553, sign 246/284 p=0.108; q-vs-r diff -3.3882',
  },
  {
    file: 'verify_5_floor.json',
    claim: `Two independent methods estimate the experimental reference's own heterogeneity sigma at about 2.83-2.84
kcal/mol: (A) matched near-duplicate pairs with |quantum difference| <= 0.1 give sigma 2.8437 from 1002 pairs;
(B) the moment decomposition sigma^2 = [V(E-Q)+V(E-L)-V(Q-L)]/2 gives 2.8330. The implied oracle MAE an unbiased
perfect estimator would show against these references is therefore about 2.26 kcal/mol.`,
    value: 'method A sigma_ref 2.8437 (n_pairs=1002); method B sigma_ref 2.8330; oracle MAE ~2.26',
  },
  {
    file: 'verify_6_floor_implication.json',
    claim: `Because the oracle MAE floor is ~2.26 kcal/mol, the quantum channel's measured MAE of 2.0978 is AT OR BELOW
that floor (ratio 0.925) and the learned channel's 2.4107 is only 1.06x the floor; the reference channel's 5.4860 is
2.42x the floor. So quantum and learned cannot be separated from a perfect estimator at this sample size, while
reference genuinely is worse.`,
    value: 'quantum ratio 0.925; learned ratio 1.062; reference ratio 2.418',
  },
  {
    file: 'verify_7_provenance.json',
    claim: `On the 331 exact-database-row observations the quantum estimate is essentially the database's own bde value:
median |db - quantum| = 0.0195, 92.4% of rows agree within 0.5 kcal/mol, but 0 rows are bit-identical and the max gap
is 6.9985 kcal/mol. The learned estimate is farther from the database: only 5.1% within 0.01 and median |db - learned|
= 0.1291, i.e. 6.6x the quantum median gap.`,
    value: 'median |db-q| 0.0195, 92.4% <=0.5, max 6.9985; median |db-l| 0.1291',
  },
  {
    file: 'verify_8_bondclass_coverage.json',
    claim: `The 543 benchmark observations cover only 3 bond classes: C-H 302, C-C 240 and H-O 1, with the H-O cell
having n=1. The database taxonomy has 9 classes (C-H 321949, C-C 71166, H-N 27314, C-N 24707, C-O 24354, H-O 11884,
N-O 1634, N-N 1597, O-O 302 rows). Therefore 6 of the database's 9 bond classes have no benchmark coverage.`,
    value: 'benchmark C-H 302 / C-C 240 / H-O 1; database has 9 classes',
  },
  {
    file: 'verify_9_size_interaction.json',
    claim: `The two calibrated channels diverge with molecule size: on observations whose parent has HAC >= 10 the learned
channel's MAE (3.2458, n=146 for C-H) substantially exceeds quantum's, whereas at HAC <= 5 learned is slightly BETTER
than quantum (C-H: learned 1.7304 vs quantum 1.7571, n=52). This cross-over is the reason the overall 0.31 kcal/mol
edge is fragile.`,
    value: 'C-H HAC>=10: learned 3.2458 vs quantum 2.2016; C-H HAC<=5: learned 1.7304 vs quantum 1.7571',
  },
  {
    file: 'verify_10_weakbond.json',
    claim: `Both calibrated channels overshoot weak bonds: for experimental values in [70,80) kcal/mol the learned
channel's mean signed error is +2.601 kcal/mol (n=129, 107 overshoots vs 22 undershoots) and quantum's is +1.347
(n=129, 93 vs 36). However the overshoot largely survives conditioning on the estimate rather than the experiment,
where it shrinks to about -1.19 (learned, n=118) and -0.58 (quantum, n=124) in the [70,80) prediction band.`,
    value: 'learned bias +2.601 in exp[70,80); conditioning on prediction gives -1.185 (n=118)',
  },
  {
    file: 'verify_11_heterogeneity.json',
    claim: `The experimental values are not independent repeats: 419 unique parent molecules carry 543 observations,
86 molecules contribute more than one row, and within those molecules the experimental spread has mean 13.42 and
median 11.90 kcal/mol (max 47.9). The correct pooled within-molecule SD is about 9.97 kcal/mol, NOT the ~165 figure
that a naive computation of the pooled variance produces.`,
    value: '86 multi-row molecules, mean spread 13.42, pooled within-SD 9.97 (not 165)',
  },
  {
    file: 'verify_12_reference_structure.json',
    claim: `The reference channel behaves like a coarse per-fragment-pair lookup rather than a per-observation estimator:
it takes a single value for each of the 539 distinct (fragment_a,fragment_b) pairs, i.e. 0 pairs carry more than one
reference value, and its error is bimodal by bond class (C-H bias +5.10, C-C bias -3.30).`,
    value: '539 fragment pairs, 0 with >1 reference value; C-H bias +5.1029, C-C bias -3.3033',
  },
]

phase('Verify')

const results = await parallel(
  CLAIMS.map((c, i) => () =>
    agent(
      `${PREAMBLE}

CLAIM TO REFUTE (number ${i + 1}):

  ${c.claim}

  REPORTED VALUE: ${c.value}

Recompute this from the raw data with your own code. For claim 5 specifically, implement BOTH methods
independently and say whether they agree. For claim 11, additionally check whether a naive pooled-variance
formula could have produced the wrong ~165 number, and report the correct value.

Write your verdict as a JSON object to: ${ROOT}/output/results/${c.file}
with EXACTLY these keys:
  {"claim": "<one-line restatement>", "reproduced": true|false, "value_found": "<your independent number(s)>",
   "unit": "<analysis unit you used>", "method": "<the exact script/command you ran>",
   "verdict": "confirmed"|"refuted"|"different_number"|"not_verifiable", "notes": "<discrepancies, confounds, caveats>"}

Then reply with a one-paragraph summary of your verdict.`,
      { label: `verify-${i + 1}`, phase: 'Verify', agentType: 'general-purpose' }
    )
  )
)

log(`verification finished: ${results.filter(Boolean).length}/${CLAIMS.length} agents returned`)

return { expected: CLAIMS.length, returned: results.filter(Boolean).length }