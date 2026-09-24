export const meta = {
  name: 'verify-and-alternatives',
  description: 'Independently recompute key numbers, then test alternative explanations for each candidate relationship',
  phases: [
    { title: 'Verify', detail: 'independent recomputation of the reported numbers from raw TSV' },
    { title: 'Alternatives', detail: 'alternative-explanation and identifiability tests' },
    { title: 'Critique', detail: 'completeness critic over all phase outputs' },
  ],
}

const P = args.project
const DATA = args.data
const OUT = args.out

const GROUND_RULES = `
You are one of several independent agents working on the SAME frozen task dataset.
Workspace root: ${P}
Frozen task data (read-only, the ONLY direct evidence): ${DATA}
Your own output directory (create files ONLY here): ${OUT}

STRICT RULES
- Do NOT modify anything under ${DATA} or anywhere else outside your own output directory.
- Do NOT download external data, and do NOT search the internet or literature.
- Do NOT assume a relationship exists. A refuted, null, or unresolved outcome is a fully
  acceptable and often preferred result. Never tune a test toward a desired answer.
- Independently implement your own computation from the raw TSV files. Do not read or reuse
  any other file under ${P}/output/ (no results/*.json, no analysis/*.py, no report). The whole
  point is that your numbers must be reproducible without seeing the values you are checking.
- The independent analysis unit is ONE complete curve / spectrum / trace / loading condition.
  Neighbouring points along a curve are NOT independent replicates. Say so when it matters.
- numpy, pandas, scipy and matplotlib are installed for python3. Use /usr/bin/env python3.
- Keep total runtime under about 3 minutes per agent. Performance warning learned the hard way
  in this workspace: a Python for-loop calling scipy.optimize.curve_fit 20000 times is far too
  slow; use scipy.optimize.least_squares(method='lm'), vectorise, or cut the resample count.
- Write your machine-readable output as JSON to a file in your output directory, and report that
  file's path back in the result_file field.
`

phase('Verify')

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    target: { type: 'string', description: 'the quantity that was recomputed' },
    recomputed_values: { type: 'object', description: 'the values you computed, as numbers' },
    agrees_with_reported: { type: 'boolean', description: 'true only if your recomputation matches the reported values you were given' },
    discrepancies: { type: 'array', items: { type: 'string' }, description: 'each concrete numerical or qualitative mismatch, empty if none' },
    method_notes: { type: 'string', description: 'exactly how you computed it, including any judgement calls' },
    result_file: { type: 'string', description: 'absolute path to the JSON you wrote' },
  },
  required: ['target', 'recomputed_values', 'agrees_with_reported', 'discrepancies', 'method_notes', 'result_file'],
}

const VERIFY_TASKS = [
  {
    key: 'V1_timecourse_kinetics',
    prompt: `${GROUND_RULES}
TARGET: product-evolution kinetics in conversion_timecourses.tsv (columns formulation_id,
illumination_time_h, gaseous_product_umol; 4 formulations x 5 times).

Independently compute, for EACH of the four formulations:
 - the final product at t = 4 h and the mean rate over 4 h,
 - the four successive segment rates (umol/h) and the ratio last/first segment rate,
 - a fit of y = a * t^b by least squares on ALL five time points (including t = 0), reporting a and b,
 - fits of a straight line, a quadratic, and a saturating form y = A*(1-exp(-k*t)), each with R2 and RMSE,
 - an AICc for each of those four models using n = 5 and k_params = (number of fitted parameters + 1),
   where AICc = n*ln(RSS/n) + 2k + 2k(k+1)/(n-k-1).

The values I have reported elsewhere in this workspace and want checked are:
 - power-law exponent b: formulation_1 0.9844, formulation_2 0.9063, formulation_3 1.2489, formulation_4 1.2303
 - segment-rate ratio last/first: 0.9333, 0.8017, 1.5403, 1.5805
 - final product at 4 h (umol): 9.48, 4.13, 0.636, 0.173
 - best model by AICc: formulation_1 saturation, formulation_2 power_law, formulation_3 power_law, formulation_4 power_law

Report whether each matches. Set agrees_with_reported to false if ANY value fails to reproduce, and
list every mismatch in discrepancies with your value and the reported value.`,
  },
  {
    key: 'V2_action_vs_absorption',
    prompt: `${GROUND_RULES}
TARGET: the relation between the action-response spectrum and the absorption spectrum in
optical_action_spectra.tsv (columns measurement_role, wavelength_nm, response_au_or_percent).
measurement_role has two levels: 'absorption' (901 points, 350-800 nm at 0.5 nm) and
'action_response' (6 points at 400, 450, 500, 550, 600, 650 nm).

Independently compute:
 - the absorption value at each of the six action wavelengths (they coincide exactly with the
   0.5 nm grid, so no interpolation should be needed),
 - the ratio statistic Rstat = (action(probe)/action(ref)) / (absorption(probe)/absorption(ref))
   for the pair ref=400 probe=650 and the pair ref=450 probe=650,
 - the same ratio statistic for every one of the 15 possible probe/reference pairs on the action grid,
 - the best single-scale absorption-limited prediction action ~ s*absorption: report s, RMSE and R2,
 - a spectral-window holdout: fit s using only the action points with wavelength <= 500 nm, then
   report the predicted vs observed action at 550, 600 and 650 nm and the relative errors.

The values I have reported elsewhere and want checked are:
 - absorption at the action wavelengths: 400 nm 0.887, 450 nm 0.861, 500 nm 0.85, 550 nm 0.585,
   600 nm 0.364, 650 nm 0.201
 - ratio statistic (400->650) 0.228, (450->650) 0.218
 - best single-scale fit s = 94.2-ish with RMSE 8.872 and R2 0.92255
 - holdout fitted on <= 500 nm: predicted action at 650 nm = 18.94 versus observed 4.2

Report whether each matches. Set agrees_with_reported to false if ANY value fails to reproduce.`,
  },
  {
    key: 'V3_transient_late_window',
    prompt: `${GROUND_RULES}
TARGET: the long-delay behaviour of the two ultrafast transient traces in
ultrafast_transient_traces.tsv (columns formulation_id, delay_ps, normalized_transient_signal;
formulation_1 and formulation_2, 98 delays each from 0.000322 ps to 76.8937 ps).

Independently compute, for EACH formulation:
 - the peak value and the delay at which it occurs,
 - the mean and median of normalized_transient_signal for every delay >= 30 ps, the number of
   such points, and the fraction of them that are positive,
 - a moving-block bootstrap 95% confidence interval for that >= 30 ps mean. Use block length
   max(2, int(sqrt(n))) and at least 5000 resamples, with a fixed RNG seed, and state the seed.
 - the difference between the two formulations' >= 30 ps means and a bootstrap 95% CI for the
   difference (resample each trace independently, then subtract).

The values I have reported elsewhere and want checked are:
 - formulation_1 >= 30 ps mean -0.2502, 95% CI [-0.3405, -0.1562], fraction positive 0.00 (17 points)
 - formulation_2 >= 30 ps mean +0.1882, 95% CI [+0.1233, +0.2477], fraction positive 0.94 (17 points)
 - difference formulation_1 - formulation_2 = -0.4384, 95% CI [-0.5404, -0.3188]
 - peak: formulation_1 at 1.13634 ps (value 0.99928), formulation_2 at 2.24169 ps (value 1.00077)

Because your bootstrap seed will differ from mine, treat the means and fractions as exact targets
and the interval endpoints as agreeing if they are within about 0.03 of mine. Note explicitly in
method_notes which of the two kinds of comparison you applied.`,
  },
  {
    key: 'V4_loading_response',
    prompt: `${GROUND_RULES}
TARGET: the cofactor-loading scan in cofactor_loading_response.tsv (columns
cofactor_loading_percent, product_evolution_rate_umol_h; four conditions: 0.2, 1.0, 3.0, 8.0).

Independently compute:
 - the loading setting with the maximum rate, and the rate values themselves,
 - the rate at 8% as a percentage of the rate at 0.2%,
 - the Spearman rank correlation between loading and rate, with its p-value, AND an exact
   two-sided permutation p-value obtained by enumerating all 24 orderings of the four rates,
 - for each of the four leave-one-condition-out subsets, the Spearman rho and whether the
   remaining three rates are strictly decreasing in loading,
 - a fit of the substrate-inhibition form rate = V*L/(K + L + L^2/Ki), reporting V, K, Ki, R2 and
   the loading at which that function peaks (analytically sqrt(K*Ki)).

The values I have reported elsewhere and want checked are:
 - peak at loading = 1.0% with rate 1217.0 umol/h; rates 1081, 1217, 470, 268
 - rate at 8% is 24.8% of the rate at 0.2%
 - Spearman rho = -0.800, p = 0.2000, exact permutation p = 0.3333 (one third, since 8 of 24
   orderings are at least as extreme)
 - leave-out 0.2%: rho = -1.0 strictly decreasing; leave-out 1.0%: rho = -1.0 strictly decreasing;
   leave-out 3.0%: rho = -0.5 not strictly decreasing; leave-out 8.0%: rho = -0.5 not strictly
   decreasing
 - substrate-inhibition fit peak at sqrt(K*Ki) = 0.490% loading with R2 = 0.99007

Note: the four loading values span a wide range on a linear axis. Also report what the exact
permutation p-value implies about how much a monotone-decline claim can be supported with n = 4.
Set agrees_with_reported to false if ANY value fails to reproduce.`,
  },
  {
    key: 'V5_integrity_audit',
    prompt: `${GROUND_RULES}
TARGET: an independent integrity and internal-consistency audit of the four frozen data files.
This is a verification task, not a discovery task.

Check and report:
 - the sha256 of each of the four files, and compare against the sha256 recorded in
   ${P}/data_manifest.json (read only that one file; it is task metadata, not another agent's output),
 - row and column counts, exact column names, and the number of missing values per column,
 - whether the absorption grid is exactly regular at 0.5 nm with no gaps or duplicates,
 - whether both transient traces share an identical delay grid,
 - the numerical precision / rounding structure of each numeric column: report how many significant
   figures the values appear to carry and give concrete examples. State plainly whether the data
   look rounded or full-precision.
 - any duplicate rows, and any exact arithmetic relations between columns or rows that would make
   a relationship tautological (for example a column that is a fixed multiple of another),
 - whether the transient traces are exactly normalised (does either trace's maximum equal exactly
   1.0, and what is the largest value in each),
 - whether any file contains internal contradictions (for example an action value implied to be
   above 100 percent, or negative absorption in a region where it should be non-negative).

Report concrete numbers. If you find something that looks like a data-quality problem, say so
plainly; if everything is consistent, say that plainly. Do not speculate about the origin of the
data beyond what the numbers themselves show.`,
  },
]

const verifyResults = await parallel(
  VERIFY_TASKS.map((task) => () =>
    agent(task.prompt, { label: `verify:${task.key}`, phase: 'Verify', schema: VERIFY_SCHEMA })
      .then((r) => ({ key: task.key, ...r }))
  )
)

log(`Verification phase complete: ${verifyResults.filter(Boolean).length}/${VERIFY_TASKS.length} agents returned`)

phase('Alternatives')

const ALT_SCHEMA = {
  type: 'object',
  properties: {
    hypothesis: { type: 'string', description: 'the alternative explanation that was tested' },
    test_description: { type: 'string', description: 'the concrete test that was executed' },
    outcome: { type: 'string', enum: ['refuted', 'supported', 'unresolved'], description: 'the verdict on the alternative explanation' },
    key_numbers: { type: 'object', description: 'the decisive numbers from the test' },
    interpretation: { type: 'string', description: 'what the outcome means for the original candidate relationship' },
    result_file: { type: 'string', description: 'absolute path to the JSON you wrote' },
    code_file: { type: 'string', description: 'absolute path to the script you executed' },
  },
  required: ['hypothesis', 'test_description', 'outcome', 'key_numbers', 'interpretation', 'result_file', 'code_file'],
}

const ALT_TASKS = [
  {
    key: 'A1_transient_sign_artifact',
    prompt: `${GROUND_RULES}
CONTEXT: In ultrafast_transient_traces.tsv the two traces have opposite long-delay behaviour.
formulation_1 decays through zero and settles negative (mean -0.2502 for delay >= 30 ps, 0 of 17
points positive), while formulation_2 settles clearly positive (+0.1882, 16 of 17 points positive).
The traces are labelled "normalized_transient_signal", so a negative asymptotic value is unusual
for a pure excited-state absorption or bleach decay.

HYPOTHESES TO TEST, one at a time, each with its own executable fit:
 H1 CONSTANT OFFSET / BASELINE DIFFERENCE: f1 and f2 differ by a constant offset, so the sign
    contrast is a level difference rather than a shape difference. Test by fitting a model in which
    f2 is forced to share one decay shape with f1 after allowing a free additive offset, and compare
    its fit against a model that allows an independent shape for each trace.
 H2 LINEAR DRIFT: the late-time sign is produced by a slow monotone drift (an instrumental or
    thermal drift) rather than by the decay. Test by fitting a straight line to each trace's late
    window and checking whether the decay model's residual still shows the sign.
 H3 THE ZERO CROSSING IS THE DECAY ITSELF: the late sign follows naturally from a decay that
    overshoots below the pre-rise baseline. Test by asking whether the pre-rise baseline of each
    trace (say t < 0.6 ps) plus the fitted decay reproduces the observed late sign and magnitude.

For each hypothesis report the fitted parameters, R2 and RMSE of the competing fits, and a clear
verdict. Then state, in interpretation, which hypothesis best explains the observed contrast and
whether that means the f1-versus-f2 contrast is a genuine kinetic difference or an artefact.
A verdict of unresolved is acceptable and must be used if the data cannot separate the hypotheses.`,
  },
  {
    key: 'A2_optical_mismatch_mechanism',
    prompt: `${GROUND_RULES}
CONTEXT: In optical_action_spectra.tsv the action response tracks absorption closely at the blue
end but falls much faster than absorption at the red end. Concretely, over 450->650 nm the action
drops to 5.08% of its 450 nm value while absorption only drops to 23.34%; the best single-scale
absorption-limited prediction over-predicts the 650 nm action by roughly 4.5x. An earlier test
found that a saturating function of absorbance, action ~ a*A/(b+A), does NOT rescue the
absorption-limited null (its RMSE is no better than the linear-in-A fit).

TEST THESE ALTERNATIVE EXPLANATIONS with executable fits:
 H1 SHARP INSTRUMENT-LIKE CUT-OFF: the absorption fall-off between 500 and 800 nm is a sharp
    filter/cut-off edge rather than a smooth physical band edge. Test by fitting the absorption in
    500-800 nm with (i) an exponential, (ii) a logistic, and (iii) a piecewise-linear kink at a
    free wavelength, and compare AICc and residual structure.
 H2 OPTICAL-DENSITY SATURATION VIA TRANSMITTANCE: the measured "absorption" is really an optical
    density that should be converted to absorptance A -> 1 - 10^(-OD) before comparing with action.
    Test by treating the absorption values as OD with an unknown scale factor, converting to
    absorptance with that scale optimised over a grid, and re-running the absorption-limited fit.
    Report the best scale, its RMSE, and whether it beats the plain linear-in-A fit.
 H3 A SECOND SPECTRAL COMPONENT: the red-end deficit comes from a distinct longer-wavelength
    absorber whose contribution to the action is smaller than its contribution to the absorption.
    Test by fitting action = s*A + c*exp(-(lambda-l0)/w) with a free centre, width and amplitude,
    and report whether a single extra red component explains the mismatch.

For each, report the decisive numbers and a verdict of refuted / supported / unresolved. In
interpretation, state which alternative (including "none of these") best accounts for the
observed action-versus-absorption mismatch. Do not claim a specific charge-transfer mechanism.`,
  },
  {
    key: 'A3_loading_identifiability',
    prompt: `${GROUND_RULES}
CONTEXT: cofactor_loading_response.tsv has only four loading settings (0.2, 1.0, 3.0, 8.0 percent)
with rates 1081, 1217, 470 and 268 umol/h. The observed ordering is a strict interior peak: rising
from 0.2 to 1.0 percent and then falling. The exact two-sided permutation p-value for a monotone
association with n = 4 is 1/3, so a monotone decline cannot be established at conventional
significance. The task guide warns that "the loading scan has few settings, so an apparent optimum
must be treated as a bounded observation rather than a universal optimum".

YOUR JOB is to quantify what these four points can and cannot identify. Executable work:
 - Fit a family of candidate response shapes, each with three or fewer fitted parameters: monotone
   power law, monotone exponential decay, plateu-then-linear-decline (a kink model), saturating
   rise without inhibition, substrate-inhibition V*L/(K+L+L^2/Ki), and a simple "flat up to a
   break then decline" two-regime model. For each, report parameters, R2, RMSE and AICc.
 - Report every model whose AICc is within 2 of the best model, i.e. the set of statistically
   indistinguishable descriptions. This set is the honest answer to "what shape does this scan have".
 - Report explicitly whether the peak LOCATION (0.2% versus 1.0%) is identifiable from these four
   points: test this by fitting an interpolating model that puts the maximum at 0.2% versus one
   that puts it at 1.0% and comparing their fit quality.
 - Test the axis-sensitivity alternative: if the loading axis were actually logarithmic inside the
   recording software (i.e. the recorded numbers are already log-scaled), would the shape become
   monotone? Report what the data allow you to conclude about this, including "cannot be
   distinguished from the data alone".
 - Do NOT rely on AICc alone with n = 4; state its limitation in your interpretation.

Outcome should be 'unresolved' if the four points cannot separate the candidate shapes, and
'supported' only if one shape clearly wins by more than 2 AICc units over every rival.`,
  },
  {
    key: 'A4_timecourse_shape_identifiability',
    prompt: `${GROUND_RULES}
CONTEXT: In conversion_timecourses.tsv, formulation_1 and formulation_2 rise nearly linearly with
segment-rate ratios last/first of 0.933 and 0.802, while formulation_3 and formulation_4 are
super-linear with ratios of 1.540 and 1.581, i.e. their production rate roughly increases over the
4 h window. With only five time points per curve, several different mechanisms can produce an
apparently increasing rate.

TEST THESE SPECIFIC MECHANISMS with executable fits on the five points of each curve:
 M1 INDUCTION / LAG then constant rate: y = r*max(t - t_lag, 0), free r and lag.
 M2 AUTOCATALYTIC or product-coupled growth: y' = k*y, giving y = y0*exp(k*t), fitted with a free
    y0 so that the t = 0 anchor is not forced to zero.
 M3 A PURE POWER LAW: y = a*t^b with b > 1.
 M4 A QUADRATIC: y = a*t^2 + b*t + c.
 M5 A SINGLE SHARED SHAPE rescaled per curve: do formulations 3 and 4 have the same curve shape as
    each other, and do 1 and 2 share a shape, once each curve is scaled by its own final value?
    Test by comparing the normalised curves pointwise and by fitting a shared-shape model with one
    free scale per curve, reporting the residual.

Then answer the identifiability question that actually matters: with five points per curve, how
many of M1-M5 land within 2 AICc units of the best model? That set, not a single winner, is the
honest description of the kinetic shape. Report it per formulation.

Finally, give one executable falsification: state what each leading candidate predicts at t = 8 h
(a time not present in the data) and how far apart those predictions are, so a future experiment
could discriminate them. Report those predicted values explicitly.

Use AICc with n = 5 and be explicit in interpretation about its small-sample limitation. An outcome
of 'unresolved' is correct whenever the points cannot separate the mechanisms.`,
  },
]

const altResults = await parallel(
  ALT_TASKS.map((task) => () =>
    agent(task.prompt, { label: `alt:${task.key}`, phase: 'Alternatives', schema: ALT_SCHEMA })
      .then((r) => ({ key: task.key, ...r }))
  )
)

log(`Alternatives phase complete: ${altResults.filter(Boolean).length}/${ALT_TASKS.length} agents returned`)

phase('Critique')

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    unresolved_items: { type: 'array', items: { type: 'string' }, description: 'questions that remain genuinely unresolved after all the work above' },
    unverified_claims: { type: 'array', items: { type: 'string' }, description: 'any claim in the material above that still lacks direct support' },
    missing_analyses: { type: 'array', items: { type: 'string' }, description: 'analyses the task objective implies but that nobody ran' },
    contradictions: { type: 'array', items: { type: 'string' }, description: 'places where two agents disagree with each other or with the reported values' },
    strongest_evidence: { type: 'array', items: { type: 'string' }, description: 'the few results that are most securely supported' },
    overclaim_risks: { type: 'array', items: { type: 'string' }, description: 'places where a conclusion would outrun what n permits' },
  },
  required: ['unresolved_items', 'unverified_claims', 'missing_analyses', 'contradictions', 'strongest_evidence', 'overclaim_risks'],
}

const critic = await agent(
  `${GROUND_RULES}
You are the completeness critic for this task. You did NOT run any of the analyses below; your job
is to read their reported outputs and say what is still missing, unresolved, or overclaimed.

TASK OBJECTIVE: "Analyze how product evolution, optical/action response, ultrafast transients, and
cofactor loading relate across the supplied anonymous formulations. Compare monotone, peaked,
formulation-specific, cross-modal, and no-stable-relation explanations, and test their sensitivity
to complete curve or spectral-window holdouts."

DATA AVAILABLE: four files under ${DATA} -- conversion_timecourses.tsv (4 formulations x 5 times),
optical_action_spectra.tsv (1 absorption profile of 901 points plus 1 action profile of 6 points),
ultrafast_transient_traces.tsv (2 formulations x 98 delays), cofactor_loading_response.tsv (4
loading conditions). No formulation identifier links the optical spectra or the loading scan to the
named formulations in the other two files.

VERIFICATION RESULTS (independent recomputation attempts):
${JSON.stringify(verifyResults.filter(Boolean).map((r) => ({ key: r.key, agrees: r.agrees_with_reported, discrepancies: r.discrepancies, notes: r.method_notes })), null, 2)}

ALTERNATIVE-EXPLANATION RESULTS:
${JSON.stringify(altResults.filter(Boolean).map((r) => ({ key: r.key, hypothesis: r.hypothesis, outcome: r.outcome, numbers: r.key_numbers })), null, 2)}

Read the JSON result files those agents wrote under ${OUT} if you need more detail. Do not write
any analysis of your own; just enumerate the gaps. Be concrete and skeptical. In particular, check
whether the objective's mention of "cross-modal" relations is actually supportable given that only
two formulations appear in both the time-course and transient files, and whether "complete curve
holdouts" was really done for every modality.`,
  { label: 'critic:completeness', phase: 'Critique', schema: CRITIC_SCHEMA }
)

return {
  verification: verifyResults.filter(Boolean),
  alternatives: altResults.filter(Boolean),
  critic,
}