export const meta = {
  name: 'audit-registered-findings',
  description: 'Audit each registered finding claim-by-claim against its cited artifacts, plus a completeness critique',
  phases: [
    { title: 'Audit', detail: 'one agent per registered finding, checking every quoted number against the cited files' },
    { title: 'Critique', detail: 'completeness and overclaim critique over the whole report' },
  ],
}

const P = args.project
const OUT = P + '/output'
const DATA = P + '/data'

const RULES = `
You are auditing a finished scientific report. Workspace root: ${P}
The report under audit: ${OUT}/Result.md
Frozen source data (read-only): ${DATA}
Artifacts you may read: everything under ${OUT}/ (results/, verification/, figures/, analysis/).
Write ONLY into ${OUT}/audit/ (create it if needed).

RULES
- Your job is to CHECK, not to re-analyse from scratch and not to rewrite the report.
- Verify quoted numbers by reading the cited artifact files (JSON/CSV) and, where a number is not
  directly stored, by recomputing it from ${DATA} with python3 (numpy/pandas/scipy available).
- Report a discrepancy if a number in the report does not match the artifact it cites, if the
  artifact cited does not actually contain the support claimed, or if the report's interpretation
  is stronger than the evidence permits. Quote both the report's value and the value you found.
- Do NOT invent problems. If a claim checks out, say so plainly. A clean audit is a valid result.
- Return your answer as PLAIN TEXT in your final message. Do NOT attempt to call any
  structured-output tool; it is not available in this environment.
- Keep runtime under 3 minutes. Write any scratch script into ${OUT}/audit/.
`

phase('Audit')

const findings = [
  {
    n: 1,
    title: 'Product evolution separates into two opposed kinetic-shape groups, but the winning functional form is not identifiable at five points per curve',
    focus: `Check specifically: the final yields 9.48 / 4.13 / 0.636 / 0.173 umol and the 54.8x span;
the four segment-rate sets; the last/first ratios 0.933, 0.802, 1.540, 1.581; the power-law
exponents 0.984, 0.906, 1.249, 1.230; the claim that AICc is UNDEFINED for every model when n = 4
(verify the arithmetic: with n=4 and 2 fitted parameters plus variance, is n-k-1 <= 0?); the
leave-one-time-point stability figures 3/5, 4/5, 0/5, 2/5; the normalised RMS differences 0.0083
(3 vs 4) and 0.0676 (1 vs 3); and the 8 h predictions 18.83 / 19.06 for formulation 1 and
7.71 / 8.26 / 25.10 for formulation 2.
Files: results/timecourse_kinetics.json, results/loading_crossmodal.json,
results/shared_shape_check.csv, results/evidence_table.csv,
verification/mechanism_fits_results.json, verification/recomputed_kinetics.json.`,
  },
  {
    n: 2,
    title: 'The action spectrum is not absorption-limited at the red end',
    focus: `Check specifically: the free-scale absorption fit R2 = 0.9226 and RMSE 8.87; the
over-prediction factor 4.32x at 650 nm (18.13 predicted vs 4.2 observed); the residuals
+1.3, +4.9, +4.3, -1.2, -15.2, -13.9; the ratio statistics 0.228 (400->650) and 0.218 (450->650);
the holdout fitted on <=500 nm with s = 94.23 predicting 55.13 / 34.30 / 18.94 at 550 / 600 / 650 nm
against observed 51.6 / 17.6 / 4.2; the efficiency ratios 0.955, 1.000, 0.993, 0.919, 0.504, 0.218;
the claim that exactly NINE pairs with probe >= 600 nm all give a statistic below 1 with maximum
0.548, and that the three pairs with probe <= 500 nm give 1.047, 1.040, 0.993; the claim that the
action grid is 6 points at ONE decimal place while absorption is at THREE; and the analytical
inequality claim that 1-10^(-k*A) is concave so no transmittance conversion can drive the 450->650
ratio below 0.2334. Verify that inequality claim numerically over a wide k grid.
Files: results/spectra_analysis.json, results/robustness.json,
verification/final_result.json, verification/h1h2h3_optics_test.json, verification/audit_results.json.`,
  },
  {
    n: 3,
    title: 'The two transient traces differ in the sign of their long-delay asymptote',
    focus: `Check specifically: the peak values 0.999 at 1.136 ps and 1.001 at 2.242 ps; the late-window
(>= 30 ps, 17 points) means -0.2502 and +0.1882 with 0/17 and 16/17 positives; the bootstrap CIs
[-0.3405, -0.1562] and [0.1233, 0.2477]; the difference -0.4384 with CI [-0.5404, -0.3188]; the
first zero crossing at ~29.6 ps; the final values -0.376 and +0.140. Then check the three artifact
verdicts against the artifact: H1 refuted with dAICc +63.5 full trace and +7.9 late-only and
F p = 6e-14 and offset ratio 0.452; H2 refuted with detrended late means -0.088 and +0.284 and
slopes p = 7e-7 and p = 0.007; H3 supported with c1 = -0.371 CI [-0.546, -0.196] and c2 = +0.176
CI [0.090, 0.261] and dAICc +28.4 / +4.2. Also check the shared-shape-time-rescaled null R2 = 0.5916
and the rise-then-decay versus monoexponential R2 comparison.
Files: results/transient_analysis.json, verification/transient_artifact_test.json,
verification/transient_verification.json.`,
  },
  {
    n: 4,
    title: 'The cofactor-loading scan has an interior peak but its shape and optimum are not identifiable',
    focus: `Check specifically: the rates 1081 / 1217 / 470 / 268; the interior peak at 1.0 %; the
61.4 % drop from 1.0 to 3.0 %; the rate at 8.0 % being 24.8 % of the 0.2 % value; Spearman
rho = -0.800 with asymptotic p = 0.200 and EXACT permutation p = 0.333 over 24 orderings (verify
8 of 24 orderings are at least as extreme); the leave-one-out results (rho -1.0 for dropping 0.2 %
or 1.0 %, rho -0.5 with p = 0.667 for dropping 3.0 % or 8.0 %); the substrate-inhibition fit
R2 = 0.9901, RMSE 39.8, V = 17486, K = 2.599, Ki = 0.0923, optimum sqrt(K*Ki) = 0.490 %; the
bootstrap optimum CI [0.453, 0.525] and 3660/5000 successful resamples; the low-loading pair
differing by 12.6 %; and the CORRECTION that AICc is undefined for all 3-parameter models at n = 4
(verify the report's corrected stage-4 artifact now reports aicc_undefined_models and ranks by BIC).
Files: results/loading_crossmodal.json, results/evidence_table.csv,
verification/cofactor_loading_recomputed.json, verification/cofactor_shape_identifiability.json,
verification/cofactor_shape_identifiability_agentB.json.`,
  },
  {
    n: 5,
    title: 'The cross-modal question is not answerable, and the one contrast that can be formed reverses',
    focus: `Check specifically: that ONLY formulation_1 and formulation_2 appear in both
conversion_timecourses.tsv and ultrafast_transient_traces.tsv (verify directly from the two raw
files); that optical_action_spectra.tsv and cofactor_loading_response.tsv contain NO formulation
identifier (verify the column names of the raw files); the yield ratio 2.30x between formulation_1
and formulation_2; the opposite ordering claim (orders_agree = false); the scale ratio claim that
loading rates 268-1217 umol/h exceed time-course rates 0.043-2.37 umol/h by roughly two orders of
magnitude (compute the actual ratio); and the claim that the three orderings (yield 1>2>3>4,
transient 2>1, loading 1.0>0.2>3.0>8.0) share no data-justified common labelling.
Files: results/loading_crossmodal.json, results/registration_summary.json, results/evidence_table.csv.`,
  },
]

const auditResults = await parallel(
  findings.map((f) => () =>
    agent(
      `${RULES}
Audit REGISTERED FINDING ${f.n} of the report:
"${f.title}"

${f.focus}

Also audit the finding's own stated limitations: does the report claim less than or more than the
evidence supports? Is any counterevidence it admits actually decisive, or is it understated?

Report as plain text: (a) every number you checked and whether it matched, (b) any mismatch with
both values quoted, (c) any claim you judge unsupported by its cited artifact, (d) an overall
verdict of CLEAN / MINOR_ISSUES / MATERIAL_PROBLEM for this finding.`,
      { label: `audit:F${f.n}`, phase: 'Audit' }
    )
  )
)

log(`Audit phase complete: ${auditResults.filter(Boolean).length}/${findings.length} agents returned`)

phase('Critique')

const critique = await agent(
  `${RULES}
You are the completeness and overclaim critic for the whole report ${OUT}/Result.md.
You did NOT write it. Read it fully, plus enough of results/ and verification/ to judge it.

Answer these concretely, with the file you are judging named in each point:
 1. Does every registered finding actually follow from the evidence cited for it? Name any that
    does not.
 2. Does the report satisfy the task objective: "Analyze how product evolution, optical/action
    response, ultrafast transients, and cofactor loading relate across the supplied anonymous
    formulations. Compare monotone, peaked, formulation-specific, cross-modal, and no-stable-relation
    explanations, and test their sensitivity to complete curve or spectral-window holdouts."
    In particular: were COMPLETE CURVE holdouts and SPECTRAL-WINDOW holdouts both really executed,
    and for which modalities? Name any modality where the holdout was not done.
 3. Is any claim overstated relative to the number of INDEPENDENT analysis units (4 curves, 2
    spectra, 2 traces, 4 loading conditions)? Quote the worst offender if any.
 4. Does anything in the report describe a method that was not actually run, or a result not
    actually observed? Check the "Reported failures" section against the artifacts.
 5. What is genuinely still unresolved or missing, that a reader would want flagged?
Return plain text with a short numbered list plus a one-line overall verdict.`,
  { label: 'critic:report', phase: 'Critique' }
)

return { audits: auditResults, critique }