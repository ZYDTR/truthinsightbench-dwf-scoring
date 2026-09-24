export const meta = {
  name: 'radio-burst-drift-parallel-analysis',
  description: 'Parallel multi-dimension analysis of radio-burst drift distributions, then adversarial verification',
  phases: [
    { title: 'Spectra', detail: 're-derive drift rate from the 5 complete dynamic spectra' },
    { title: 'Distributions', detail: 'mixture vs single-population and channel-paired tests' },
    { title: 'Provenance', detail: 'reconcile the 47-sheet workbook against the normalized views' },
    { title: 'Orientation', detail: 'source-group and channel conditional structure' },
    { title: 'Robustness', detail: 'leave-one-out, binning, threshold and count-model falsification' },
    { title: 'Verify', detail: 'adversarial re-computation of the assembled claims' },
  ],
}

const ROOT = '/data/user/conversations/conv-02a7c52b-6f08-461b-b10b-65392ab66dba/project'

const BRIEF = `
PROJECT ROOT: ${ROOT}
Use absolute paths under that root. You may ONLY read files under the root.
NEVER use web search or external data. Workspace data are the sole direct evidence.

TASK: TruthInsightBench Astronomy_01_radio_burst_drift,
"Drift Distributions and Conditional Differences in Discrete Radio Bursts", T0=2022-10-13.

ANALYSIS UNIT (binding): one complete dynamic-spectrum event, or one complete
source-polarization series. Pixels and adjacent directional coordinates are NESTED
REPEATS, not independent samples. There are only 5 complete events. Never report a
pixel count as an event-level sample size.

DATA ALREADY ESTABLISHED (verified in this session; re-verify what you rely on):
- data/normalized_views/dynamic_spectrum_event_metadata.tsv : 5 rows (event_1..event_5)
  columns time_min_s time_max_s frequency_min_MHz frequency_max_MHz
  reported_event_drift_rate_MHz_s frequency_bins(=425) time_bins(=425)
  reported drift rates: e1=-12.7777, e2=-12.4520, e3=-3.3614, e4=-16.8673, e5=-4.0560 MHz/s
  all share fmin=8.0017, fmax=16.9708 MHz; span ~1.098 s
- data/normalized_views/event_<i>_dynamic_spectrum.tsv : 425 rows x 426 cols.
  col0 = frequency_MHz (DESCENDING from 16.9708 to 8.0017, df=-0.021154 MHz);
  remaining cols are "time_s_<t>" ordered ascending, dt=0.002591 s.
  Pixel values in [1.0, 1.5]; background is exactly 1.0; signal is >1.0.
- data/normalized_views/drift_population_series.tsv : 948 rows,
  cols population_series_id anonymous_source_group polarization_channel drift_regime
  distribution_bin_center event_count. 6 series = 3 anonymous groups x 2 channels.
  Each series is split into drift_regime='fast' and 'slow'.
  CRITICAL STRUCTURE ALREADY MEASURED:
    fast: 146 bins total across all series, 21-26 bins/series, centers 1.929..27.811 MHz/s
    slow: 802 bins total, 127-139 bins/series, centers 0.146..11.735 MHz/s
    => the two regimes use DIFFERENT bin widths (~1.0 vs ~0.10 MHz/s) and OVERLAP in range.
    608 of 948 bins have event_count==0.
  Per-series event totals: ps1 fast=10560 slow=892; ps2 fast=14492 slow=1303;
    ps3 fast=357 slow=803; ps4 fast=52 slow=392; ps5 fast=429 slow=1092; ps6 fast=664 slow=2201.
- data/normalized_views/orientation_count_series.tsv : 312 rows, 6 series x 52 coords,
  cols orientation_series_id relative_coordinate event_count. coords -33..20 step 1.
  Series event totals: 11240, 15551, 1151, 444, 1503, 2843.
- data/source_data/all_source_data_deidentified.xlsx : 47 sheets, deidentified.
  Layout convention: col A = sheet name; col B = block label; row 3 holds headers
  ('x','y' or 'tmin(s)','tmax(s)','fmin(MHz)','fmax(MHz)','Drift-rate(MHZ/s)');
  blocks are separated by blank rows; a new col-B string starts a new block.
  Sheet families:
    Sheet_001,Sheet_002 : 33644 rows x 4, headers 'cml angle','<x> phase angle',
      6 blocks each, labels include group_a/group_b/group_c/Gan(channel_l),
      Gan(channel_l), Ganymed(channel_l), (channel_r). Two sheets, differ in the
      second column name and in the y values (Sheet_001 col3 vs Sheet_002 col3).
    Sheet_003..Sheet_008 : 430x427 image sheets. Sheet_003..Sheet_007 correspond
      exactly to normalized event_1..event_5. Sheet_008 has its own Bounds row
      (tmin=23429.39358, tmax=23430.49194, drift=-12.77765) and is NOT in the
      normalized views.
    Sheet_015..Sheet_018, Sheet_021, Sheet_022, Sheet_036, Sheet_037 : further 430x427
      image sheets with their own Bounds rows (some with negative coordinate bounds).
    Sheet_009..Sheet_014 : 62x4, three orientation blocks per sheet
      (labels '<group> (-35<s<-10)' , '<group>(-10<s<0)' , '<group>(0<s<20)'),
      x integer -34..-10, -9..0, 1..20; y integer counts.
    Sheet_019, Sheet_020, Sheet_023, Sheet_038 : radial-distance style profiles
      ('RD_box_of_ones','RD_raw','RD_corr','RD_corr_centered','Fit','params'),
      with an 'SNR' value in row 3 (e.g. 12.296175, 2.386328, 7.361634).
    Sheet_024..Sheet_035 : drift-rate distributions, two blocks per sheet
      ('(-10<s<0)' and '(0<s<20)'), fine x bins ~0.10 MHz/s.
    Sheet_039..Sheet_044 : 67x4, three drift-rate blocks per sheet with
      x = 0,1.75,3.5,...,33.25 (1.75 MHz/s bins).
    Sheet_045 : empty (1x1).
    Sheet_046, Sheet_047 : 72x4, three blocks labelled 'group_b C(...)'.
  EARLIER MAPPING RESULT: the 'fast' regime of each population_series matches
  EXACTLY one workbook block: ps1->Sheet_025, ps2->Sheet_027, ps3->Sheet_029,
  ps4->Sheet_031, ps5->Sheet_033, ps6->Sheet_035 (all '(-35<s<-10)' blocks).
  The 'slow' regime matched NO single workbook block.

ARTIFACTS YOU MUST PRODUCE (write real files):
- analysis code you actually ran  -> ${ROOT}/output/analysis/s<NN>_<name>.py
- machine-readable results (csv/json/npz) -> ${ROOT}/output/results/
- figures (png) -> ${ROOT}/output/results/figures/
- a stdout log -> ${ROOT}/output/logs/
Prefix every filename with your dimension tag. Keep code deterministic.
Use python3 with numpy/pandas/scipy/matplotlib (all installed). openpyxl available.
For figures use matplotlib with Agg backend and write PNG dpi>=120.

RULES:
- Every number you report must come from code you actually executed.
- Report null results, conflicts, and failures. Do not manufacture a relationship.
- State the analysis unit for every number.
- Record real error text if something fails; continue with a simpler executable method.
`

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    status: { type: 'string', description: 'complete | partial | failed, with reason' },
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
          support: { type: 'string', description: 'the executed computation and numbers backing it' },
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
    method: { type: 'string', description: 'what you executed to check it' },
    evidence: { type: 'string', description: 'actual numbers from your re-computation' },
    code_file: { type: 'string' },
    result_file: { type: 'string' },
    residual_uncertainty: { type: 'string' },
  },
  required: ['target', 'verdict', 'method', 'evidence'],
}

// ---------------------------------------------------------------- Phase 1
phase('Spectra')

const dims = [
  {
    key: 'spectra',
    label: 'D1:dynamic-spectra',
    prompt: `${BRIEF}

YOUR DIMENSION: Re-derive the drift rate of each of the 5 complete dynamic spectra
from the pixel data itself, independently of the reported value.

Required work:
1. Load all five event_<i>_dynamic_spectrum.tsv grids.
2. Characterise the signal: background is 1.0; quantify signal pixels (>1.0+eps),
   the fraction of the grid occupied, and the per-channel time profile.
3. Re-derive a drift rate (MHz/s) per event with AT LEAST TWO independent methods,
   e.g. (a) per-time-column intensity-weighted mean frequency then a robust linear
   fit of frequency vs time; (b) an intensity-weighted 2-D second-moment / principal
   axis of the signal mask; (c) optionally a Radon / slope-scan maximising signal
   containment. Because frequency DECREASES with row index, take care with sign.
4. Compare each re-derived rate to reported_event_drift_rate_MHz_s. Report
   differences and a relative error. State clearly whether the reported rates are
   reproducible from the pixels.
5. Test for sub-structure WITHIN a single event: is the emission a single ridge or
   are there multiple resolved components (e.g. look at the time profile of the
   frequency centroid, and the count of connected components in the signal mask)?
6. Produce figures: for each event a background-subtracted dynamic spectrum with the
   fitted ridge overlaid, plus one figure comparing re-derived vs reported rates.
7. Save a per-event table of re-derived drift rates (csv + json).

Report as claims: whether reported drift rates are reproducible; how many distinct
emission components each event shows; whether any event is better described as
multi-component. Include the numeric analysis unit (per event, n=5).

Falsification test to state: what pixel-level result would show the reported rates
are NOT reproducible from these spectra.`,
  },
  {
    key: 'mixture',
    label: 'D2:drift-population-mixture',
    prompt: `${BRIEF}

YOUR DIMENSION: The central question of the task - is the drift-rate distribution
one population, several populations, a continuous distribution, or insufficient data?

Required work:
1. Load drift_population_series.tsv. Work per population_series_id (6 series).
2. Establish the binning structure precisely and state it numerically: bin width,
   bin edges, range, and the overlap between the 'fast' and 'slow' regimes. Report
   how many bins are empty in each regime.
3. Fit competing models to each series' event_count vs distribution_bin_center:
   (a) single log-normal / single Gaussian in log10(drift);
   (b) two-component mixture in log10(drift);
   (c) single power-law or exponential.
   Use maximum likelihood on the BINNED counts with a Poisson likelihood, and also
   a multinomial likelihood with the bin probabilities; report AIC and BIC for each
   model. Implement the fits yourself with scipy.optimize (scipy.stats has
   lognorm/norm); do not rely on any package that is not installed.
4. CRITICAL ALTERNATIVE EXPLANATION: the 'fast' regime uses ~1.0 MHz/s bins and the
   'slow' regime ~0.10 MHz/s bins, and their ranges OVERLAP. Test whether the
   apparent two-population structure is an artifact of DIFFERENT BINNING rather than
   a genuine bimodality. Do this by re-binning both regimes onto a COMMON grid
   (e.g. 0.25 MHz/s bins over 0-28 MHz/s, and a log-spaced grid) and re-testing for
   bimodality (dip statistic style reasoning or mixture AIC/BIC on the common grid).
   Report the result honestly even if it destroys the two-population reading.
5. Report effective sample size: the number of independent analysis units, and the
   total event counts per regime. Note that counts are dominated by a few bins.
6. Figures: per-series histogram with fitted models overlaid; one figure showing the
   common-grid re-binning before/after.
7. Save fit tables as csv+json.

Report as claims: which model wins per series, whether the two-regime split survives
common-grid re-binning, and what the data can and cannot support given the sample.`,
  },
  {
    key: 'channel',
    label: 'D3:polarization-channel',
    prompt: `${BRIEF}

YOUR DIMENSION: Does the polarization channel alter the conclusion? This is an
explicit sub-question of the research objective.

Required work:
1. Using drift_population_series.tsv, contrast channel_l vs channel_r WITHIN each
   anonymous group, separately for each drift_regime.
2. Because the two channels of a group are the SAME physical source observed in two
   polarization channels, treat them as a PAIRED comparison. Compute the total event
   counts per channel and the ratio; note that the counts are very unequal
   (e.g. group_1 fast: 10560 vs 14492; group_2 fast: 357 vs 52; group_3 fast: 429 vs 664).
3. Test whether the channel changes the SHAPE of the distribution: fit the
   single log-normal model to each channel separately and compare fitted medians and
   sigmas; also compute distribution-free summaries (weighted median, weighted IQR,
   mode bin, fraction of events above 10 MHz/s) with the bin counts as weights.
4. Quantify uncertainty properly for the counts: use a Poisson or multinomial
   bootstrap over bins to get confidence intervals on the weighted medians and on
   channel ratios. State clearly that bins are not independent samples.
5. Test the same question with the workbook source blocks that are paired by channel
   within a group (Sheet_009 vs Sheet_010, Sheet_011 vs Sheet_012,
   Sheet_013 vs Sheet_014 for orientation; Sheet_025 vs Sheet_027, Sheet_029 vs
   Sheet_031, Sheet_033 vs Sheet_035 for the fast drift regime). Sheets 046/047 are
   'group_b C' variants - report what they are and whether they agree with the
   corresponding non-C blocks.
6. Figures: paired channel comparison per group and regime.
7. Save comparison tables as csv+json.

Report as claims: whether channel materially changes the drift distribution, in which
groups and regimes, and where the data are insufficient to decide.`,
  },
  {
    key: 'provenance',
    label: 'D4:workbook-reconciliation',
    prompt: `${BRIEF}

YOUR DIMENSION: Provenance and coverage. The workspace says data/ is the COMPLETE
task dataset and that all 47 source-data sheets are present. Verify what the
normalized views actually contain relative to the workbook, and what is NOT
represented.

Required work:
1. Programmatically parse ALL 47 sheets of all_source_data_deidentified.xlsx into a
   tidy structure: (sheet, block_label, x, y) for the non-image sheets, and the
   header/Bounds rows plus the numeric 425x425 block for the image sheets.
2. For each of the 14 image sheets (Sheet_003..008, 015..018, 021, 022, 036, 037),
   extract its Bounds row (tmin, tmax, fmin, fmax, drift-rate) and its pixel grid.
   Determine which image sheet corresponds to which normalized event by comparing the
   grid to event_<i>_dynamic_spectrum.tsv numerically (exact or up to a transpose).
   Report an exact-match table. Confirm or refute that Sheet_003..007 == events 1..5.
   Report what Sheet_008 and the pairs 015/016, 017/018, 021/022, 036/037 are.
3. Determine whether Sheet_015..018/021/022/036/037 are ADDITIONAL events, repeated
   renderings of the same events, or processed/derivative images (e.g. different
   rescaling, residual, or model images). Use numeric evidence: correlations,
   identical Bounds rows, value ranges, and pixel differences. Several pairs share
   identical Bounds rows - report exactly which pairs share bounds and what differs
   numerically between them.
4. Reconcile the drift distributions: for the 'slow' regime series that matched no
   workbook block, test whether it is an AGGREGATE of the fine-binned workbook sheets
   Sheet_024/026/028/030/032/034 (and/or Sheet_039..044). Attempt an explicit
   reconstruction: sum or concatenate candidate workbook blocks on the appropriate
   grid and compare to the normalized slow series numerically. Report whether an exact
   reconstruction is achieved, and if not, the closest match and its residual.
5. Same for the orientation series: orientation_count_series.tsv has 6 series of 52
   points over -33..20, while workbook orientation blocks have 25/10/20 points over
   -34..-10 / -9..0 / 1..20. Test whether a concatenation of the three blocks for a
   given (group, channel) reproduces one normalized orientation series. Note the
   normalized series include coordinate 0 and start at -33, so account for that.
6. Produce a machine-readable coverage table: every workbook block -> its
   representation in the normalized views (exact / aggregate / absent / derivative).
   Explicitly LIST every workbook block that has NO normalized counterpart, with its
   size and total counts. This is the coverage boundary.
7. Figures: the Bounds-row comparison and a reconciliation residual plot.

Report as claims: which sheets are additional vs duplicate events; whether the slow
regime is a re-binning of the fine-binned sheets; and the exact coverage boundary
(what the normalized views leave out).`,
  },
  {
    key: 'orientation',
    label: 'D5:orientation-and-groups',
    prompt: `${BRIEF}

YOUR DIMENSION: The directional/orientation count series and the source-group
conditional differences.

Required work:
1. Load orientation_count_series.tsv (6 series x 52 relative coordinates, -33..20).
   Establish what varies ACROSS the 6 series by comparing them numerically: are they
   distinct profiles or rescalings of one another? Compute pairwise correlations and
   the ratio of totals. Report the totals: 11240, 15551, 1151, 444, 1503, 2843.
2. Characterise each series' shape: peak coordinate(s), the fraction of zero bins,
   the centre of mass, and the number of local maxima. Several coordinates are zero
   for every series (e.g. -33, -32, -31, -28) - report exactly which coordinates are
   structurally zero across all six series; these are coverage gaps, not measurements.
3. Contrast the three anonymous source groups using BOTH the orientation series and
   the workbook orientation blocks (Sheet_009..014, and the 'group_b C' blocks in
   Sheet_046/047). Since the normalized orientation series could not be matched to a
   single workbook block in earlier work, reconstruct the mapping: test whether each
   normalized series equals the concatenation of the three workbook blocks
   (blocks at x=-34..-10, x=-9..0, x=1..20) belonging to one group+channel, allowing
   for the normalized series starting at -33 and including 0.
4. Quantify group differences with a distribution-free statistic that respects the
   nesting (distances are nested repeats within a series): e.g. a chi-square or
   total-variation distance between series normalised to their totals, plus a
   bootstrap that resamples at the SERIES level (only 6 series - state this limit
   explicitly). Do NOT treat the 52 coordinates as independent replicates.
5. State explicitly how many independent units support any group-level claim
   (report: 6 series, or 3 groups, depending on the comparison), and whether the
   data can support a group-level difference at all.
6. Figures: the 6 orientation profiles overlaid; a group-contrast panel.
7. Save the mapping table and comparison statistics as csv+json.

Report as claims: whether the six series are distinguishable; whether source group
changes the orientation profile; and the honest limit imposed by n=6 series.`,
  },
  {
    key: 'robustness',
    label: 'D6:robustness-falsification',
    prompt: `${BRIEF}

YOUR DIMENSION: A dedicated falsification and robustness harness that an independent
reviewer would demand. You must actively try to BREAK the other dimensions'
likely readings, not confirm them.

Required work:
1. BINNING SENSITIVITY: take the drift-population counts and re-bin onto at least
   four alternative grids (linear 0.1, 0.25, 0.5, 1.0 MHz/s over 0..28, and a
   log-spaced grid). For each, report the number of occupied bins, the mode location,
   and whether a two-component mixture beats a single component by AIC/BIC. Report
   how the answer changes with the grid. This tests whether any claimed
   multi-population result is grid-dependent.
2. THRESHOLD SENSITIVITY: for the dynamic spectra, recompute per-event drift rates
   using several signal thresholds (e.g. 1.02, 1.05, 1.1, 1.2 times background) and
   report how the re-derived rate moves. This gives an error bar driven by method
   choice, not by statistics.
3. COMPLETE-UNIT LEAVE-ONE-OUT: for every event-level statistic derived from the 5
   dynamic spectra, recompute with each event removed in turn (n=5 folds) and report
   the full spread. Any claim that flips sign or inverts under a single-event
   deletion must be reported as NOT robust.
4. COUNT-MODEL CHECK: test whether the binned event counts are consistent with
   Poisson noise around a smooth distribution, or are over-dispersed. Compute the
   variance-to-mean ratio and a dispersion test per series. Report whether
   over-dispersion (if present) is explained by the binning alone.
5. SAMPLE-SIZE REALITY CHECK: compute and state the effective event-level sample
   size for each claim class: 5 events for dynamic spectra, 6 series for
   orientation, 6 series for drift populations, and the total event counts with their
   concentration (e.g. what share of events sits in the top 3 bins). Quantify the
   concentration numerically.
6. OUTLIER TREATMENT: identify the most extreme bins and report each series' results
   with and without the top 1% of bins by count.
7. Figures: binning-sensitivity panel; threshold-sensitivity panel; leave-one-out
   spread panel.
8. Save every table as csv+json.

Report as claims about ROBUSTNESS ONLY: which candidate conclusions survive and which
do not. Be adversarial and specific. A finding that does not survive must be stated.`,
  },
]

const finderResults = await parallel(
  dims.map((d) => () =>
    agent(d.prompt, { label: d.label, phase: 'Spectra', schema: FINDING_SCHEMA }).then((r) =>
      r ? { ...r, _key: d.key } : null
    )
  )
)

const ok = finderResults.filter(Boolean)
log(`dimension agents returned: ${ok.length}/${dims.length}`)

// ---------------------------------------------------------------- Phase 2
phase('Verify')

const allNumbers = ok
  .flatMap((r) => (r.key_numbers || []).map((k) => `[${r.dimension}] ${k.name} = ${k.value} ${k.unit || ''} (${k.analysis_unit || 'unit n/s'})`))
  .join('\n')
const allClaims = ok
  .flatMap((r) => (r.claims || []).map((c) => `[${r.dimension}] CLAIM: ${c.claim}\n  SUPPORT: ${c.support}\n  COUNTER: ${c.counterevidence || '-'}\n  ALT: ${c.alternative_explanation || '-'}`))
  .join('\n\n')

const verifyTargets = [
  {
    label: 'V1:drift-rate-reproducibility',
    prompt: `Adversarially re-compute, from scratch, whether the reported drift rates in
dynamic_spectrum_event_metadata.tsv are reproducible from the pixel data in
data/normalized_views/event_<i>_dynamic_spectrum.tsv. Write your own independent
estimator (do not reuse any file another agent wrote). Report the per-event
re-derived rate, the difference from the reported value, and whether the sign
convention in the source data is consistent. Default to verdict 'refuted' if your
numbers disagree with the claim by more than the claimed uncertainty.`,
  },
  {
    label: 'V2:bimodality-vs-binning',
    prompt: `Adversarially test the claim that the drift-rate distributions are a genuine
two-population (fast/slow) structure rather than an artifact of the two regimes
using different bin widths. Independently re-bin drift_population_series.tsv onto at
least three common grids that BOTH regimes can populate, and test bimodality
(dip-style reasoning, or a two-component vs one-component Poisson-likelihood mixture
compared by BIC). Your job is to try to REFUTE the two-population reading. If the
two-population structure disappears on a common grid, say so plainly.`,
  },
  {
    label: 'V3:slow-regime-reconstruction',
    prompt: `Adversarially verify or refute the claim that the 'slow' drift-regime series in
data/normalized_views/drift_population_series.tsv is a re-binning/aggregate of the
fine-binned workbook sheets (Sheet_024/026/028/030/032/034) rather than an
independent measurement. Attempt an EXPLICIT numeric reconstruction from the
workbook and report the residual. If the reconstruction fails, report the closest
candidate and the actual residual value. Also verify whether the normalized
orientation series can be reconstructed from the workbook orientation blocks.`,
  },
]

const verdicts = await parallel(
  verifyTargets.map((v) => () =>
    agent(
      `${BRIEF}

YOUR ROLE: adversarial verifier for one specific target. You must EXECUTE code that
re-computes the quantity independently. Do not accept any claim on plausibility.
Write your code to ${ROOT}/output/analysis/ and results to ${ROOT}/output/results/
with a 'v_<n>_' prefix.

CLAIM CONTEXT ASSEMBLED FROM PARALLEL AGENTS (unverified - your job is to check):
${allClaims}

NUMBERS CLAIMED BY PARALLEL AGENTS:
${allNumbers}

YOUR TARGET:
${v.prompt}`,
      { label: v.label, phase: 'Verify', schema: VERDICT_SCHEMA }
    )
  )
)

return {
  finder_results: ok,
  verification: verdicts.filter(Boolean),
}