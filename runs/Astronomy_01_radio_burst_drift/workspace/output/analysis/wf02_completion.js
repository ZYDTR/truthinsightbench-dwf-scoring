export const meta = {
  name: 'radio-burst-drift-completion',
  description: 'Complete the remaining analysis dimensions and adversarially verify the finding set',
  phases: [
    { title: 'Complete', detail: 'finish mixture fits, channel contrast, robustness, spectra tests' },
    { title: 'Verify', detail: 'adversarial independent re-computation of the central claims' },
  ],
}

const ROOT = '/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project'

// ---------------------------------------------------------------- MEMORY DISCIPLINE
// A previous run of 6 concurrent agents was SIGKILLed (exit -9) by the host OOM
// killer. Each agent had loaded the 32 MB workbook with openpyxl default mode,
// which inflates to >1 GB. The rules below are BINDING.
const MEMORY = `
=== MEMORY DISCIPLINE (BINDING - a previous run died of OOM, exit code -9) ===
The worker has 1 CPU and 4 GB RAM, shared with other agents.
- NEVER call openpyxl.load_workbook() with read_only=False / data_only=False.
  ALWAYS: openpyxl.load_workbook(path, read_only=True, data_only=True).
- NEVER load more than 2 of the 425x425 dynamic-spectrum grids into memory at once.
- NEVER paste a large data file into your context. Compute summaries in code.
- Prefer the already-extracted small artifacts in ${ROOT}/output/results/ over
  re-parsing the workbook.
- Do not spawn parallel python processes yourself. One process at a time.
- If a computation needs the 5 grids, loop event-by-event and free each before
  loading the next (del + gc.collect()).
- Keep every written artifact well under 20 MB.
`

const BRIEF = `
PROJECT ROOT: ${ROOT}
Absolute paths only, under that root. Read ONLY files under the root.
NEVER use web search or external data. Workspace data are the sole direct evidence.
${MEMORY}

TASK: TruthInsightBench Astronomy_01_radio_burst_drift,
"Drift Distributions and Conditional Differences in Discrete Radio Bursts", T0=2022-10-13.

ANALYSIS UNIT (binding): one complete dynamic-spectrum event, or one complete
source-polarization series. Pixels and adjacent directional coordinates are NESTED
REPEATS, not independent samples. There are only 5 complete events. Never present a
pixel count as an event-level sample size.

=== VERIFIED FACTS ALREADY ESTABLISHED (reuse; do not redo the expensive parts) ===
F1. data/normalized_views/event_<i>_dynamic_spectrum.tsv, i=1..5:
    425 rows x 426 cols. col0=frequency_MHz descending 16.9708 -> 8.0017
    (df=-0.0211536 MHz); other cols 'time_s_<t>' ascending, dt=0.0025905 s.
    Pixels in [1.0, 1.5]; 49.8% of pixels are exactly 1.0; values are capped at 1.5.
F2. reported drift rates (dynamic_spectrum_event_metadata.tsv):
    e1=-12.7777  e2=-12.4520  e3=-3.3614  e4=-16.8673  e5=-4.0560 MHz/s.
F3. output/results/s10_drift_rederived_per_event.csv: SIX independent pixel-based
    estimators ALL disagree with the reported rates, differences 3.2 to 3670 MHz/s,
    relative error 0.92-35. An earlier centroid/PCA estimator gave +0.44 MHz/s for
    event_1 (reported -12.78), i.e. even the SIGN differs.
F4. output/results/s11_image_bounds.csv + output/results/s14_image_stats.csv:
    there are 14 image sheets. Sheet_003..007 equal normalized event_1..5 EXACTLY
    (max_abs_diff = 0.0). Sheet_008, 015, 016, 017, 018, 021, 022, 036, 037 are
    additional 425x425 arrays, NOT in the normalized views. Several share Bounds
    rows; pairwise Pearson r between arbitrary image sheets is ~0.00-0.06 and
    mean|difference| ~0.13, i.e. they are mutually uncorrelated, not repeats.
F5. drift_population_series.tsv: 948 rows, 6 series (3 anonymous groups x 2 channels),
    each split into drift_regime 'fast' and 'slow'.
      fast: 146 bins total, 21-26 per series, bin width ~1.0-1.04 MHz/s,
            centers 1.93..27.81 MHz/s, 26554 events total.
      slow: 802 bins total, 127-139 per series, bin width ~0.10 MHz/s,
            centers 0.15..11.74 MHz/s, 6683 events total.
    The two regimes use DIFFERENT bin widths and their ranges OVERLAP. 608 of 948
    bins have event_count == 0.
    Per-series totals: ps1 10560/892, ps2 14492/1303, ps3 357/803, ps4 52/392,
    ps5 429/1092, ps6 664/2201  (fast/slow).
F6. PROVENANCE SOLVED (output/results/s15_*.csv, s16_*.csv, s04_structure_check.csv):
    - Each 'fast' series equals exactly ONE workbook block:
      ps1->Sheet_025, ps2->Sheet_027, ps3->Sheet_029, ps4->Sheet_031,
      ps5->Sheet_033, ps6->Sheet_035  (all the '(-35<s<-10)' blocks).
    - Each 'slow' series equals the CONCATENATION of TWO workbook blocks from one
      sheet: ps1->Sheet_024, ps2->Sheet_026, ps3->Sheet_028, ps4->Sheet_030,
      ps5->Sheet_032, ps6->Sheet_034, each labelled '(-10<s<0)' + '(0<s<20)'.
      Reconstruction is EXACT (max |dx| ~5e-14, |dy| = 0) for all six.
    - Each normalized orientation series equals the concatenation of the three
      workbook blocks of one (group, channel) from Sheet_009..014, after DROPPING
      x=-34, x=-9 and x=1. Dropping x=-9 discards real counts (e.g. 212 for
      orientation_series_1). Reconstruction exact (max |dy| = 0) for all six.
    - The normalized orientation coordinate grid has 52 of the 55 raw points; it
      OMITS x=-34, -9, 1. Coordinates -33,-32,-31,-28,-1,0,2,11 are zero in ALL six
      series (a mix of coverage gaps and measured zeros).
F7. output/results/s05_binning_grid_sensitivity.csv: on COMMON re-binned grids
    (linear 0.10/0.25/0.50/1.00 and log), a two-component mixture is preferred over
    one component by both AIC and BIC for nearly every series; but the fitted
    two-component solution is unstable (for ps3 and ps4 a component collapses to a
    median of ~1e-06 MHz/s, an edge/degenerate solution). ps4 fast has only 52
    events and 21 bins. Treat these fits as requiring scrutiny.
F8. output/results/s04_fit_crosscheck.csv: single log-normal fits were reproduced by
    two independent optimisers to ~1e-9, so the single-component fits are reliable.
F9. output/results/s16_coverage_summary.json: 101 workbook blocks total;
    6 exact, 30 aggregate, 14 image, 1 empty, 50 ABSENT from the normalized views.
    Absent families include Sheet_001/002 (33.6k-row angle series), Sheet_019/020/
    023/038 (radial-distance profiles), Sheet_039..044 (coarse 1.75 MHz/s drift
    histograms), Sheet_046/047 ('group_b C' variants).

=== OUTPUT RULES ===
- analysis code -> ${ROOT}/output/analysis/  (prefix with your tag)
- machine-readable results -> ${ROOT}/output/results/  (csv/json)
- figures -> ${ROOT}/output/results/figures/  (png, Agg backend, dpi>=120)
- stdout log -> ${ROOT}/output/logs/
Every number must come from code you actually executed. Report null results and
failures. State the analysis unit for every number. Record real error text.
`

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    status: { type: 'string' },
    code_files: { type: 'array', items: { type: 'string' } },
    result_files: { type: 'array', items: { type: 'string' } },
    figure_files: { type: 'array', items: { type: 'string' } },
    key_numbers: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          value: { type: 'string' },
          unit: { type: 'string' },
          analysis_unit: { type: 'string' },
        },
        required: ['name', 'value'],
      },
    },
    claims: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string' },
          support: { type: 'string' },
          counterevidence: { type: 'string' },
          alternative_explanation: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'moderate', 'low', 'unresolved'] },
        },
        required: ['claim', 'support'],
      },
    },
    falsification_test: { type: 'string' },
    errors: { type: 'array', items: { type: 'string' } },
  },
  required: ['dimension', 'status', 'code_files', 'result_files', 'claims'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    target: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'partially_confirmed', 'refuted', 'unverifiable'] },
    method: { type: 'string' },
    evidence: { type: 'string' },
    code_file: { type: 'string' },
    result_file: { type: 'string' },
    residual_uncertainty: { type: 'string' },
  },
  required: ['target', 'verdict', 'method', 'evidence'],
}

// ---------------------------------------------------------------- Phase 1
phase('Complete')

const dims = [
  {
    key: 'spectra_robust',
    label: 'C1:spectra-threshold-LOO',
    prompt: `${BRIEF}

YOUR DIMENSION: Robustness of the dynamic-spectrum results, and the decisive test
of whether the reported drift rates are recoverable from the pixels at all.

Required work:
1. For each of the 5 events, recompute a pixel-based drift rate using AT LEAST four
   signal thresholds (background 1.0; thresholds 1.01, 1.02, 1.05, 1.10, 1.20) with
   a fixed simple estimator (intensity-weighted mean frequency per time column, then
   a robust Theil-Sen slope vs time). Report the rate per event per threshold in a
   table. State the analysis unit (per event, n=5).
2. FULL leave-one-out: for every event-level summary statistic you compute, recompute
   with each event removed in turn (5 folds) and report the full spread (min/max/range).
   Flag any statistic that changes sign or inverts.
3. THE DECISIVE TEST: test whether a straight ridge with the REPORTED slope is
   present in the pixel data at all. For each event, compute the mean image intensity
   along the line f(t) = f0 + m*(t - t0) for the reported slope m, and compare it to
   the distribution of the same quantity for a large set of alternative slopes
   (e.g. 400 slopes spanning -40..+40 MHz/s). Report the percentile of the reported
   slope in that null distribution. If the reported slope is not an extreme value,
   the reported rate is NOT recovered from the pixels. Do this at 3 thresholds.
4. Component structure: for each event count the connected components of the
   thresholded mask, their sizes, and the number of time columns showing multiple
   separated runs of signal. Report whether each event is a single ridge or a
   multi-component / diffuse field. Report the fraction of the grid above threshold.
5. Figures: (a) per-event background-subtracted image with the reported-slope line
   overlaid and the fitted ridge; (b) the slope-scan null distribution with the
   reported slope marked; (c) threshold-sensitivity curves.
6. Save all tables as csv+json.

Report as claims, in the data's own terms: whether the reported drift rates are
reproducible from the pixels (support OR refute - say which), whether the images
contain a coherent drifting ridge, and how the answer moves with threshold.`,
  },
  {
    key: 'mixture_finish',
    label: 'C2:mixture-identification',
    prompt: `${BRIEF}

YOUR DIMENSION: Finish the population-structure question with a proper, honest
model comparison and an explicit statement of what the sample can support.

Required work:
1. For each of the 6 series separately for 'fast' and 'slow', fit on the ORIGINAL
   bin grid: (a) one log-normal, (b) a two-component log-normal mixture, (c) a single
   power law in drift, (d) a single exponential. Use a POISSON likelihood on the
   binned counts, maximised with scipy.optimize, with multiple random restarts from a
   fixed seed (this is the step the earlier run did not finish - no results file
   exists). Report parameters, log-likelihood, AIC, BIC, and the fitted component
   weights and medians for every model and series.
2. GUARD AGAINST DEGENERATE SOLUTIONS: reject or explicitly flag any mixture fit
   where a component weight -> 0, a component median runs to the grid edge, or a
   sigma -> 0. The earlier common-grid run produced median ~1e-06 MHz/s for ps3 and
   ps4, which is a degenerate edge solution. Report how often this happens.
3. Report the effective sample size honestly: the number of independent analysis
   units (6 series; and for event-level questions only 5 events), the total counts,
   and the CONCENTRATION - what share of each series' events sits in its top 1, 3 and
   5 bins. Report the Gini or Herfindahl index of the bin-count distribution.
4. State whether the data can distinguish one population from several, or whether
   the answer is 'insufficient data' for specific series. Note explicitly that
   ps4-fast has only 52 events and 21 bins.
5. Test support: use a parametric bootstrap - simulate counts from the best
   single-component fit, refit both models, and report how often the two-component
   model is spuriously preferred by BIC. This directly tests whether the apparent
   extra component is real or noise. This is the key falsification step.
6. Figures: per-series fitted models overlaid on the counts; a bootstrap
   false-preference rate bar chart.
7. Save all fit tables as csv+json.

Report as claims: which model wins per series on the ORIGINAL grid, the bootstrap
false-preference rate, and for which series the answer is genuinely unresolved.`,
  },
  {
    key: 'channel_finish',
    label: 'C3:channel-paired-shape',
    prompt: `${BRIEF}

YOUR DIMENSION: Finish the polarization-channel contrast with an explicit,
uncertainty-aware shape comparison.

Required work:
1. For each anonymous group and each regime, compare channel_l vs channel_r using the
   ORIGINAL bin grids. Because the two channels of a group are the SAME source
   observed in two polarization channels, treat them as a PAIRED comparison and say so.
2. Compute with bin counts as weights: weighted median, weighted IQR, mode bin,
   fraction of events above 10 MHz/s, and the total-count ratio. Report per group,
   per regime, per channel. State total events per cell:
     group_1 fast 10560 vs 14492 ; group_1 slow 892 vs 1303
     group_2 fast 357 vs 52     ; group_2 slow 803 vs 392
     group_3 fast 429 vs 664    ; group_3 slow 1092 vs 2201
3. Uncertainty: run a parametric bootstrap that resamples bin counts from a Poisson
   model of the observed counts (>=2000 replicates) and report 95% intervals on the
   weighted median difference between channels. State clearly that bins are NOT
   independent samples and that the bootstrap therefore captures counting noise only.
4. Channel-ratio sanity: for group_2 fast the two channels differ by a factor ~6.9
   (357 vs 52). Test whether that is compatible with a single underlying distribution
   given the counts (e.g. a two-sample comparison on the binned profiles with a
   Poisson/multinomial model). Report the result.
5. Also compare the paired workbook blocks that differ ONLY by channel and check
   whether the coarse workbook histograms (Sheet_039..044) agree with the fine
   blocks. Report disagreements.
6. Figures: paired channel profiles per group and regime; bootstrap interval panel.
7. Save tables as csv+json.

Report as claims: whether channel materially changes the drift distribution; for
which groups/regimes the difference is resolvable; and where the counts are too small.`,
  },
  {
    key: 'coverage_supp',
    label: 'C4:coverage-and-absent-families',
    prompt: `${BRIEF}

YOUR DIMENSION: The coverage boundary and the 50 workbook blocks that the normalized
views leave out. The workspace claims data/ is the COMPLETE dataset, so what the
normalized views omit is a first-class scientific result.

Required work:
1. Using the ALREADY-EXTRACTED small artifacts, do NOT re-parse the workbook unless
   necessary. Available: output/results/s11_blocks.csv (all non-image blocks),
   output/results/s11_image_bounds.csv, output/results/s16_coverage_block.csv,
   output/results/s16_coverage_summary.json, output/results/s14_images.npz
   (14 grids, 9.5 MB compressed - load at most 2 at a time), output/results/s15_*.csv.
2. Characterise the ABSENT families and state what scientific content they carry:
   - Sheet_001/Sheet_002: 33644-row 'cml angle' vs 'io phase angle'/'group_b phase
     angle' series, 6 blocks each. Report the x range, the y range, the number of
     points, and whether the two sheets describe the same x grid with different y.
     Compute the correlation between the Sheet_001 and Sheet_002 y columns.
   - Sheet_019/020/023/038: radial-distance / SNR profile families. Report the SNR
     value in each (row 3), the block sizes, and what differs between them.
   - Sheet_039..044: coarse drift histograms with 1.75 MHz/s bins. Report how they
     compare with the fine blocks for the same group+channel: do totals agree, does
     the coarse histogram reproduce the fine one after rebinning?
   - Sheet_046/047: 'group_b C' variants. Established: they correlate r>0.985 with
     the corresponding base blocks but have totals 0.81-0.97 of the base. Report
     what this means and whether they are a subsample, a smoothed version, or an
     error bar variant. Test the ratio structure numerically.
   - Sheet_008, 015..018, 021, 022, 036, 037: the 9 image sheets not in the
     normalized views. Report each one's Bounds row (tmin, tmax, fmin, fmax,
     drift-rate) and its pixel statistics, and classify each as: additional
     independent event, derivative/reprocessed image, or unclassifiable. Use numeric
     evidence only (value range, correlation with the 5 known events, whether it is
     a difference of two known images, sum/mean statistics).
3. Build the definitive coverage table: every workbook block -> representation in
   normalized views (exact / aggregate / absent). Count blocks and rows per class.
   This is the authoritative coverage boundary.
4. State clearly which scientific questions in the research objective CANNOT be
   answered from the normalized views alone and require the workbook.
5. Figures: coverage summary bar chart; the Sheet_001 vs Sheet_002 comparison;
   a montage of the 9 unrepresented image sheets rendered small (downsample each to
   <=64x64 before plotting to stay inside the memory budget).
6. Save tables as csv+json.

Report as claims about coverage and about what the unrepresented families contain.`,
  },
]

const finderResults = await parallel(
  dims.map((d) => () =>
    agent(d.prompt, { label: d.label, phase: 'Complete', schema: FINDING_SCHEMA }).then((r) =>
      r ? { ...r, _key: d.key } : null
    )
  )
)

const ok = finderResults.filter(Boolean)
log(`completion agents returned: ${ok.length}/${dims.length}`)

// ---------------------------------------------------------------- Phase 2
phase('Verify')

const allClaims = ok
  .flatMap((r) => (r.claims || []).map((c) => `[${r.dimension}] ${c.claim}\n  SUPPORT: ${c.support}\n  COUNTER: ${c.counterevidence || '-'}`))
  .join('\n\n')
const allNumbers = ok
  .flatMap((r) => (r.key_numbers || []).map((k) => `[${r.dimension}] ${k.name} = ${k.value} ${k.unit || ''}`))
  .join('\n')

const verifyTargets = [
  {
    label: 'V1:reported-drift-recoverable',
    prompt: `Independently determine whether the reported drift rates
(e1=-12.7777 e2=-12.4520 e3=-3.3614 e4=-16.8673 e5=-4.0560 MHz/s) are recoverable
from the pixel data. Use a method NO other agent used: e.g. a Hough-style transform,
a 2-D autocorrelation / structure-tensor orientation estimate, or a cross-correlation
lag between consecutive time frames converted to a slope. You must reach a verdict by
computation. Explicitly consider and rule out the possibility that a sign convention
or an axis-transposition explains the discrepancy (test the transposed grid too).
Default to 'refuted' if you cannot recover the reported rates.`,
  },
  {
    label: 'V2:two-population-real',
    prompt: `Adversarially test whether the 'fast' vs 'slow' structure in
drift_population_series.tsv is a genuine property of the drift distribution or a
pure artifact of the two regimes using different bin widths, given that the two
regimes are EXACTLY a concatenation of workbook blocks ('fast' = the single
'(-35<s<-10)' block; 'slow' = the '(-10<s<0)' + '(0<s<20)' blocks). Test whether the
regime boundary corresponds to a real gap in the drift distribution or merely to a
binning/labeling boundary. In particular: test whether the gap between the top of the
slow range (~10-11.7 MHz/s) and the bottom of the fast range (~1.9-7.8 MHz/s) is
POPULATED once you use a common grid, i.e. whether the two regimes actually overlap
in drift-rate space. Report the overlap numerically and reach a verdict on whether
'two populations' is supported, refuted, or unresolved.`,
  },
  {
    label: 'V3:provenance-claims',
    prompt: `Independently re-derive, from the raw files, the three central provenance
claims and report exact residuals:
(1) each 'fast' series equals exactly one workbook block, and each 'slow' series
    equals the concatenation of two blocks from one sheet;
(2) each normalized orientation series equals the concatenation of three workbook
    blocks after dropping x=-34, x=-9, x=1, and that dropping x=-9 discards real counts;
(3) the normalized orientation coordinate grid omits exactly x=-34, -9, 1 and
    contains 52 of the 55 raw points.
Write your own extraction code (read_only=True streaming for the workbook) and report
max|dx| and max|dy| for each reconstruction. If any claim fails, say so exactly.`,
  },
]

const verdicts = await parallel(
  verifyTargets.map((v) => () =>
    agent(
      `${BRIEF}

YOUR ROLE: adversarial verifier. You must EXECUTE code that re-computes the quantity
independently. Never accept a claim on plausibility. Write code to
${ROOT}/output/analysis/ and results to ${ROOT}/output/results/ with a 'vN_' prefix.

CLAIMS FROM PARALLEL AGENTS (UNVERIFIED - your job is to check them):
${allClaims}

NUMBERS THEY REPORTED:
${allNumbers}

YOUR TARGET:
${v.prompt}`,
      { label: v.label, phase: 'Verify', schema: VERDICT_SCHEMA }
    )
  )
)

return { completion_results: ok, verification: verdicts.filter(Boolean) }