export const meta = {
  name: 'round2-transient-artifact-and-critique',
  description: 'Test whether the transient sign contrast is an artifact, and critique the assembled evidence',
  phases: [
    { title: 'Artifact', detail: 'transient sign-contrast artifact and alternative-explanation tests' },
    { title: 'Critique', detail: 'completeness and overclaim critic over everything produced so far' },
  ],
}

const P = args.project
const DATA = args.data
const OUT = args.out

phase('Artifact')

const artifactAgent = agent(
  `You are working on a frozen scientific task dataset. Workspace root: ${P}
Frozen data (read-only, the ONLY direct evidence): ${DATA}
Write your outputs ONLY into: ${OUT}

STRICT RULES
- Do not modify anything outside ${OUT}. Do not download external data or search the internet.
- Do not read ${P}/output/results/ or ${P}/output/analysis/Result.md or output/analysis/*.py.
  Compute everything yourself from the raw TSV.
- numpy/pandas/scipy/matplotlib are installed for python3.
- Write a script AND run it; save a JSON with your numbers. Keep runtime under 3 minutes:
  do NOT loop scipy.optimize.curve_fit thousands of times (that is far too slow here);
  use scipy.optimize.least_squares(method='lm') or vectorise.
- A verdict of 'unresolved' or 'refuted' is fully acceptable and often the right answer.
- IMPORTANT: return your answer as PLAIN TEXT in your final message (do not attempt to call any
  structured-output tool; it is not available in this environment). Put the full JSON content
  in the text as well so it is not lost.

CONTEXT
File: ultrafast_transient_traces.tsv (columns formulation_id, delay_ps,
normalized_transient_signal). Two formulations, 98 delays each from 0.000322 ps to 76.8937 ps.
Both traces rise to about +1.0 around 1-2 ps and then decay.
Measured long-delay behaviour (delay >= 30 ps, 17 points):
  formulation_1: mean -0.2502, 0 of 17 points positive, final value -0.376
  formulation_2: mean +0.1882, 16 of 17 points positive, final value +0.140
These means are separated by -0.4384 with a bootstrap 95% CI of about [-0.54, -0.32], and a
Fisher exact test on the positivity counts gives p < 1e-4.
The traces are labelled "normalized_transient_signal", so a negative asymptote is unusual.

YOUR JOB: decide whether the formulation_1 versus formulation_2 long-delay sign contrast is a
genuine kinetic difference or an artefact. Test each of these with an executable fit:

 H1 CONSTANT OFFSET: the two traces differ by an additive baseline offset, so the "sign contrast"
    is a level difference rather than a shape difference. Fit: force f2 to share one decay shape
    with f1 plus a free additive offset, and compare against the model with an independent shape
    for each trace. Report both res, RMSE and AICc-style comparison. If the shared-shape-plus-offset
    model fits essentially as well, H1 is SUPPORTED and the contrast is largely a level artefact.

 H2 SLOW LINEAR DRIFT: the late sign comes from a slow monotone drift rather than the decay.
    Fit a straight line to the late window (>= 30 ps) of each trace; report slope, its
    significance, and whether removing the fitted line changes the sign of the late mean.

 H3 THE DECAY ITSELF: the late sign follows naturally from a decay that settles below or above
    the pre-rise baseline. For each trace, fit a rise-then-decay model with a free asymptotic
    offset c over all 98 points, report c and its sign, and compare the fitted c with the observed
    late-window mean. If the fitted offset already has the observed sign and magnitude, H3 is
    SUPPORTED and the contrast is a genuine property of the fitted kinetics.

Then give a one-line overall verdict: is the f1-versus-f2 long-delay sign contrast best explained
by H1, H2, H3, or none/unresolved? State which specific numbers drove that choice.

Save your script as ${OUT}/transient_artifact_test.py and your JSON as
${OUT}/transient_artifact_test.json.`,
  { label: 'artifact:transient-sign', phase: 'Artifact' }
)

const [artifactResult] = await parallel([() => artifactAgent])

log('Artifact test complete')

phase('Critique')

const critiqueAgent = agent(
  `You are the completeness critic for a scientific task. Workspace root: ${P}
Frozen data: ${DATA}. Analysis artifacts produced so far: ${P}/output/verification/ and
${P}/output/results/. You may read those. Do NOT write analysis of your own; only enumerate gaps.
Return your answer as PLAIN TEXT in your final message (no structured-output tool is available).

TASK OBJECTIVE: "Analyze how product evolution, optical/action response, ultrafast transients, and
cofactor loading relate across the supplied anonymous formulations. Compare monotone, peaked,
formulation-specific, cross-modal, and no-stable-relation explanations, and test their sensitivity
to complete curve or spectral-window holdouts."

DATA: conversion_timecourses.tsv (4 formulations x 5 times), optical_action_spectra.tsv (1
absorption profile of 901 points, 1 action profile of 6 points), ultrafast_transient_traces.tsv
(2 formulations x 98 delays), cofactor_loading_response.tsv (4 loading conditions). No formulation
identifier links the optical spectra or the loading scan to the named formulations in the other files.

WHAT WAS FOUND (summaries; read the JSON files for detail):
- Time courses: yields span 0.173 to 9.48 umol at 4 h (54.8x). Curve shapes differ: formulations 1
  and 2 have segment-rate ratios last/first of 0.93 and 0.80 (near-linear / slightly decelerating);
  formulations 3 and 4 have 1.54 and 1.58 (accelerating). Power-law exponent b is 0.984, 0.906,
  1.249, 1.230 respectively.
- Action vs absorption: the action spectrum tracks absorption at the blue end but falls far faster
  at the red end. Over 450->650 nm action drops to 5.1% while absorption drops only to 23.3%; the
  absorption-limited null over-predicts the 650 nm action by about 4.3x. A spectral-window holdout
  (fit on <=500 nm) predicts 18.94 at 650 nm vs 4.2 observed.
- Transients: opposite long-delay signs between the two formulations (details in
  verification/transient_verification.json and transient_artifact_test.json).
- Loading: 4 points (0.2, 1.0, 3.0, 8.0 percent) with rates 1081, 1217, 470, 268. Interior peak at
  1%. Exact two-sided permutation p for a monotone association is 1/3 (n=4). With n=4, AICc is
  formally undefined for any 3-parameter model because n-k-1 = 0.

Read ${P}/output/verification/*.json for the alternative-explanation results (optics H1/H2/H3,
cofactor shape identifiability, time-course mechanism fits).

Produce a concrete, skeptical list covering:
 1. unresolved_items - what remains genuinely unresolved.
 2. unverified_claims - any claim above that still lacks direct support.
 3. missing_analyses - analyses the objective implies that nobody ran. In particular judge whether
    the objective's "cross-modal" requirement is supportable at all given only TWO formulations
    appear in both the time-course and the transient files, and whether "complete curve holdouts"
    was really done for every modality (spectra, transients, loading, time courses).
 4. contradictions - anywhere two artifacts disagree with each other.
 5. strongest_evidence - the few results that are most securely supported.
 6. overclaim_risks - where a conclusion would outrun what the number of independent units permits.
Be specific and cite the file whose content you are judging.`,
  { label: 'critic:completeness', phase: 'Critique' }
)

const [critiqueResult] = await parallel([() => critiqueAgent])

return {
  artifact: artifactResult,
  critique: critiqueResult,
}