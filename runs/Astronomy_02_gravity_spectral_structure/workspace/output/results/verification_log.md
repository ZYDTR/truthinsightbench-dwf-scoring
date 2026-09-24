
---

## Adversarial verification — TRACK 5 "Evidence figures and the reproduction harness" (verifier: independent parallel track)

Verdict: **confirmed** (severity minor). Artifacts exist: true. Script runs: true.

**What was checked, with real execution.** All five PNGs exist under output/figures/ and parse as valid
non-interlaced PNGs (IHDR signature present, IEND present, 8-bit RGBA): fig1 3200x1920 / 1935225 B,
fig2 2320x1184 / 202690 B, fig3 2240x1472 / 481309 B, fig4 2640x1056 / 254273 B, fig5 2400x1600 / 322619 B.
Every figures_manifest.json byte count matches the real on-disk size exactly, and no extra or missing PNG
is present in the directory. The tEXt chunk of every file names "Matplotlib version 3.11.2", the same
version recorded in the manifest. `python3 -m py_compile` succeeds on all six track scripts, both verify
scripts, run_all.py and common.py (10/10, real exit 0). Pixel content is genuinely non-blank: non-white
fractions are fig1 0.124, fig2 0.052, fig3 0.587, fig4 0.363, fig5 0.241.

**The decisive test against the "stale figure written by a script that never ran" failure mode.** I deleted
all five PNGs and figures_manifest.json, then ran /usr/bin/python3 output/code/track5_figures.py in
isolation. It exited 0 and regenerated all five PNGs with **bit-for-bit identical SHA-256 digests** to the
pre-deletion files (7e015283…, 286ea07a…, 57a8878d…, b50e064b…, dd6b808a…) and identical byte sizes; the
regenerated manifest again matched disk for all five. I also probed the harness itself with an injected
child that exits non-zero: run_all.py recorded returncode 3, n_failed=1 and returned exit 1, so it is not a
vacuous always-zero wrapper. That probe's summary overwrite was undone and track5_run_all_summary.json was
restored to its original content (md5 0b61b1d9992089c4c5eeb87c9055e31e, n_failed=0). The recorded log shows
6/6 exit codes 0 and is internally consistent with that summary.

**Independent re-derivation of the load-bearing numbers (own code, not the track's script).** Recomputed
from the raw TSVs: whole-pass RMS means A/B/C = 0.0336917535 / 0.0202727021 / 0.0252685530 mm/s (identical
to the annotations to the last digit, abs diff 0.0); A-B paired difference 0.0134190514, CI
[0.0103668225, 0.0168200567], 22/22 passes > 0 (identical); A-C and C-B CIs identical; pass-averaged LS
peaks 489 / 2474 / 1706 uHz identical; band-mean 860-1310 uHz power A 0.0490241344, B 0.0039089660,
C 0.0553375303 with identical bootstrap CIs; grid max delta-IC 137.2 at rank 250, 250 unique cells of the
7x8x10x4 = 2240 lattice, 11.16% coverage; smallest top-20 margin 0.1 at ranks 14-15; f=860 first enters at
rank 148; FA min IC 43.9 at (f=1010, vmax=20), second min 47.2, uniform min 1522.1 at v=0.10 (interior,
not a boundary value), MV min 50.7. No asserted number was found that no code output reproduces.

**Blocking / unit-of-analysis audit.** Every interval in track5 is a bootstrap over the 22 whole passes,
never over the 1698 samples; there is no p-value anywhere, and fig5 panel (b) states on the canvas that the
grid rows carry no per-row standard error. No figure claims the ranked table is a complete cartesian grid —
on the contrary, fig5 panel (a) titles it "250 of 2240 factorial cells kept", and the script itself notes
the table is a truncated top-250 at ic = 137.2. No physical-uniqueness claim is made from the relative IC
comparison. No alignment/coincidence claim is made at all (modes_in_band = 18 is a bare count); I ran the
chance null anyway and it would not have supported a coincidence (6 of 18 modes within 5 uHz of a comb line
vs chance mean 3.6, p = 0.133).

**Defects reported (all minor).** (1) fig2: the paired-difference text uses ax2.text(0, -0.235, va='top') —
its top edge is 23.5% of the axes height below the bottom of the right axes, so the last rows of the three
difference lines run flush to the image edge (ink present on the final row y=1183; the C-B line loses one
descender row). The text is legible in pixel crops but is not confined to the canvas. (2) fig1's pass_14
zoom is captioned "largest relative A-B separation": true by mean|A-B|/mean|A| (0.9917, rank 1/22) but false
by relative RMS separation, where pass_13 (0.7501) exceeds pass_14 (0.7391). (3) fig3 panel 2 calls
1010-1160 uHz the ranked-grid peak frequencies, though the ranked grid also carries f=860 in 33/250 rows;
the ridge claim itself is correct (all top-20 rows lie in 1010-1160). None of these affect any headline
number, hence severity minor and verdict confirmed.

Cleanup note: a background re-run of track5_figures.py overwrote the two timing scalars (total_elapsed_s
646.04, ls_spectra_elapsed_s 80.71, reflecting load-average ~50 on the host). I restored the submitted
manifest and annotations to their original timing values (17.96 and 1.54, byte sizes 2286 and 4016 as
recorded) so the submitted artifacts are byte-identical to what track5 produced. My own temporary crops
under output/results/_verifier_crop*.png were removed.

---

## Adversarial verification — TRACK 4 "Information-criterion comparison against the supplied uniform alternative" (verifier: independent parallel track)

Verdict: **partially_confirmed** (severity minor). Artifacts exist: true. Script runs: true.

**Artifacts.** output/results/ic_comparison.json (39159 B) and output/code/track4_ic_comparison.py (43460 B)
both exist and are non-empty. I re-ran the script in isolation: exit 0, and the regenerated
ic_comparison.json is byte-identical to the submitted file (md5 2d09431c90eec05b7c9c8ace74fa9b80, cmp
clean), so the JSON on disk is what the submitted script produces.

**Independent re-derivation (own code; the track's script was never imported for any number).** I rebuilt
all four load-bearing results from the raw TSVs with pandas, then cross-checked the three headline numbers
again with numpy.genfromtxt and with raw text. Every number matched to the last digit (abs diff 0.0):
uniform table is the vmax==vmin diagonal of minimum_velocity_scan for 19/24 rows (max |diff| exactly 0.0;
all 19 IC pairs identical both as raw strings and as IEEE-754 hex bit patterns), the 5 non-matching rows
(v = 0.001/0.002/0.003/0.005/0.01 cm/s) lie strictly below the diagonal floor of 0.02 cm/s and their IC
values occur nowhere in the 507-row candidate table; uniform minimum IC 1522.1 at v = 0.1 cm/s (interior,
index 8/24), curve strictly unimodal (1 sign change; 8 down-steps, 15 up-steps), local 3-point quadratic
A = 3047.372210798297 IC/dex^2 with vertex 0.085927 cm/s and vertex IC 1508.878 (below the sampled
minimum by 13.22); delta_information_criterion is the absolute IC with zero offset (26 strict float64 key
overlaps, 29 after rounding to 1e-6, max signed offset 0.0, max absolute offset 1.42e-14 on one row,
0 rows > 1e-9); ranked minimum delta_IC = 0.0 at rank 1 (vmax=30, vmin=0.05, f=1010, w=275) versus
candidate-table best 43.9 (FA, vmax=20, vmin=0, f=1010, w=300) and MV best 50.7; uniform-minus-candidate
1478.2, ratio 34.672; variant restrictions (a), (b), (c) and the leave-one-slice-out argmin sets all
reproduced (29 candidate rows recoverable in the top-250, 18 FA / 11 MV, 0 ranked rows with vmax==vmin).

**Independence / unit-of-analysis audit — the specific error categories asked for.** (a) No p-value or CI
treats time samples or grid rows as replicates: the JSON contains no p-value, standard error, confidence
interval, surrogate or permutation statistic at all (string scan of all 751 leaf paths); uncertainty is
handled only by deterministic whole-slice leave-one-out, which is not an inferential test. (b) No blocked
statistic exists to be mis-blocked - this track is a deterministic IC-ordering exercise and says so. (c) No
claim that the ranked grid is a complete cartesian grid: the JSON never uses "cartesian"/"factorial"/"all
possible" and repeatedly calls it a truncated top-250 (I measured 250 of 7x8x10x4 = 2240 cells = 11.16%,
maximum 21 of 40 rows per (vmax, vmin) cell, so it is a truncated ranking, not a product grid). (d) No
physical-uniqueness claim: the comparison is explicitly labelled relative, nested and partly circular
("nested comparison, not an independent model comparison"). (f) No alignment/coincidence claim is made.
(g) There are no surrogates in this track, so the "null result caused by a signal-preserving surrogate" bug
cannot apply.

**The central claim survives.** The uniform alternative is not an independent alternative: it is exactly
the diagonal of the same minimum_velocity_scan family at f = 1210 uHz and w = 300 uHz, bit-identical at
every shared velocity. The track's own nested/circular warning and its "marginal 6.8-unit FA-vs-MV margin"
reading are both supported by the recomputation.

**Defects found (all in narrative prose, none in the central claim, hence severity minor).** Three prose
assertions are not reproduced by any code output and two are outright false arithmetic: (1) the claim that
the 6.8-unit family margin is "smaller than a single grid step (rank1 -> rank3 = 5.7)" - 6.8 > 5.7 and
rank1->rank3 is two steps; (2) "Delta ~100 is larger than the entire span of the top-250 ranked set" -
the span is 0.0-137.2, and the same JSON's delta_100 block says "comparable to the whole spread (0-137.2)",
so the two blocks contradict each other; (3) "a factor 1.2 in velocity costs ~2-4 IC units" - with the
reported curvature A = 1.47e3-3.05e3 IC/dex^2 the cost is 9.2-19.1 IC units, and the paragraph's own
formula with its own A = 2e3 example gives 80, not 2-4; 2-4 IC units corresponds to a factor ~1.06-1.13,
which weakens the paragraph's "resolved to within a factor ~1.2-1.3" conclusion. (4) "the top-29 ranked
rows all use vmax in {20, 30}" - rank 27 is vmax = 50 (it is true for the top 19 and for the 4 rows within
10 units). (5) "a third of the family has IC > 4000" - it is 127/507 = 0.2505. (6) "relative_offset_vs_best"
is division by min(best, 1.0), not by the best; mislabelled but harmless. (7) The smoothness text calls the
largest absolute slope jump (1834.7 IC/dex, at 0.1->0.2, i.e. at the minimum) an "inversion" and localises
it at 0.002-0.003; that jump is a recovery, and the largest genuine inversion is -1762.0 at 0.002-0.003.
None of these change the diagonal-identity finding, the uniform minimum, the offset-zero finding, or the
nested-comparison conclusion, so the verdict is partially_confirmed rather than refuted; the headline
numbers are confirmed exactly, but the track's machine-readable JSON contains several load-bearing-in-
narrative statements that are false as written and should be corrected.

---

## Adversarial verification — TRACK 3 "Structure, degeneracy and resolution boundary of the candidate parameter grids" (verifier: independent parallel track)

Verdict: **partially_confirmed** (severity minor). Artifacts exist: true. Script runs: true
(`python3 -m py_compile output/code/track3_grid_structure.py` exits 0; `_run_all.log` line 54 records
`RESULT track3_grid_structure.py -> ok in 43.3 s`, exit 0, and grid_structure.json is 28463 bytes of
valid JSON with 9 top-level keys).

**Method.** I re-implemented the track's quantities from the raw TSVs in
`output/code/verify_grid_fast.py` and `output/code/verify_grid.py` and did NOT import or execute
track3_grid_structure.py. The stored JSON was read only as a claim source.

**Every load-bearing number reproduces.** Taking the three headline claims first:
(1) truncation — the table has candidate_rank exactly 1..250, IC monotone non-increasing in rank,
250/250 cells inside the declared 7x8x10x4 = 2240 lattice, cutoff 137.2; the cross-file ledger gives
105 lattice cells known independently from candidate_spectrum_comparison.tsv, 28 with IC < 137.2 all
present, 0 spurious, smallest absent 139.9, and 29 rows whose IC matches the candidate file exactly
(max abs offset 1.4e-14). (2) top-cell separation — cells within delta-IC < 1/2/5/10/20 are
1/1/2/4/9 and rank1-to-rank2/5/10 margins are 2.0/11.9/21.0, all reproduced to the last digit; the
hard-coded neighbour-step costs 14.1/12.8, 25.3/10.1, 20.6/1.1 are exactly right (max |diff| 7e-15).
(3) degeneracy — the fa family is a complete 16x19 factorial (304 rows), the ridge fit is
f_star = 709.2858 * vmax^0.1213664 uHz with R2 = 0.97288891 (residual max 0.055875 dex), the
vmax>=0.5 variant gives 0.1353756/672.6809/0.9881479, fa full-vs-additive R2 = 0.688525/0.203313,
and the mv regressions give 0.000625/0.580008/0.758510/0.780816. The ranked-grid fits reproduce too:
R2/RMSE 0.023554/33.382276 against IC sd 33.872715, f^2+w^2 0.174397, richer basis 0.383151,
interactions 0.138210, leave-one-vmax-out 0.018716-0.067330. Mode alignment reproduces exactly:
observed median nearest-mode offset 5.76 uHz, null medians 8.089/7.236/7.335 uHz and one-sided
p = 0.178/0.271/0.290 on my own seeds and 200k/100k draws, family-wise p 0.497, in-band spacing
20.69 uHz. Resolution: 22 passes, 19x79 / 2x72 / 1x60 min, 61-80 samples (sum 1698), 1/T =
210.970464-277.777778 uHz, 250/325/50 uHz = 1.1850/1.5405/0.2370 x 1/T, and my own two-sinusoid
simulation (equivalent windowed LS grid, own seed, 100 trials) gives 0.18/0.25/0.59/0.76/0.90/1.00
resolved at 50/100/211/250/275/300 uHz versus the JSON's 0.14/0.31/0.56/0.75/0.88/1.00, i.e. within
Monte-Carlo noise except for a mild under-power at 50-100 uHz. The per-pass band medians
883.0/1207.0/1111.5 uHz reproduce with identical IQR, leave-one-out and bootstrap intervals.

**Problems that are real (all minor).** (i) The JSON field `loglog_fit_vmax_ge_0.1` is mislabelled:
it carries n = 16 (the vmax>=0.1 subset) but slope 0.009894 / R2 0.003257, which is the ALL-19 fit;
the genuine vmax>=0.1 fit is -0.082530 / R2 0.204143. The section-3 verdict and section-7 conclusion
quote -0.082 / 0.204, which is numerically correct for vmax>=0.1, so the prose is right and the
machine-readable field is the defective one; a downstream consumer keyed on the field name would get
the wrong fit. (ii) The section-6 verdict and section-7 conclusion say the two-sinusoid test resolves
"55% of trials at 250 uHz but 95-100% at 275-325", while the JSON's own field (200 trials) says 0.75
at 250 and 0.88 at 275. The 0.55/0.95 values are stale, from an earlier 20-trial explore run
(`_t3e10.txt`); my own 100-trial rerun gives 0.76/0.90 and confirms the JSON field, not the prose.
(iii) Three literal "%%" tokens survive unformatted in the stored JSON strings (sections 1, 6, 7),
confirming the prose was hand-written rather than produced from the printed values. (iv) The whole
`single_grid_step_cost_at_optimum` and `interior_curvature_at_vmax30` blocks are hard-coded literals
in the script — they happen to be exactly correct, but they are not computed there. (v) "residual
scatter corresponds to +-1 grid step" overstates: recomputed ridge residuals are max 73.64 uHz
(1.47 x the 50 uHz step) and rms 43.17 uHz. (vi) "Adding interactions nearly doubles R2 (0.024 ->
0.138)" is wrong as written: 0.0236 -> 0.1382 is a 5.9x increase. (vii) The ledger's single
"false negative" is a boundary tie: that cell (minimum_velocity_scan vmax=20, vmin=0.02, f=1210,
w=300, IC 137.2) IS in the table at candidate_rank 250 with IC 137.2; it is counted FN only because
the stated admission rule is the strict `ic < 137.2`. (viii) The unobserved region is larger than the
"remaining 1990 cells" framing: the union of both supplied files evidences only 326 of the 2240
lattice cells, so 1914 cells (85.4%) could not be checked even in principle.

**Specific error modes I looked for and did NOT find.** No adjacent time samples or grid rows are
treated as replicates — pass-level dispersion uses n = 22 whole passes (bootstrap and leave-one-pass-
out), and grid separation is reported as a margin ladder precisely because no replicate IC exists.
No p-value or CI is blocked at the wrong level: the only p-values are the mode-alignment nulls, which
draw independent uniform frequencies (a legitimate chance null) rather than time samples or grid rows.
The track never claims a complete cartesian grid — it explicitly labels the table a top-250
truncation whose rank-1 cell is best only among the 250 survivors. No physical uniqueness is claimed
from the relative IC comparison; the relative-only caveat list explicitly disowns continuous optima
and treats the 2240-cell step sizes as arbitrary. The alignment claim is tested against three proper
chance nulls (uniform over the mode range, uniform over the candidate band, and a rigid 50 uHz comb
with free phase) and I reproduced every null median and p. Finally, the null result is not a bug: the
"not closer than chance" conclusion is corroborated by the in-band mode spacing being 2.4x denser
than the 50 uHz comb, which mechanically makes small nearest-mode offsets unsurprising, and the null
draws are independent of the observed modes.

**Consequence for the track's stated conclusions.** The five headline conclusions (truncated ranking,
non-unique rank-1 cell, f_star ~ 710 vmax^0.121, no mode-grid locking, single-pass 1/T = 211 uHz
against a 250 uHz narrowest width) all stand on recomputed evidence. The prose of sections 3 and 6
and the section-7 summary need two number corrections (the -0.082/0.204 attribution and the
55%/95% resolution fractions), the JSON needs the `loglog_fit_vmax_ge_0.1` field corrected and the
`%%` tokens rendered, and the "nearly doubles" and "+-1 grid step" phrasings should be fixed; none
of these changes a conclusion.

---

## Adversarial verification — TRACK 2 "Reproducible spectral structure in the residual traces" (independent parallel verifier)

Verdict: **partially_confirmed** (severity major). Artifacts exist: true. Script runs: true.

Both artifacts are non-empty: spectral_tests.json is 232195 B, track2_spectral_tests.py is 51970 B
(1166 lines). I re-ran the track script in a sandbox copy (/tmp/verify_t2, data symlinked, so the
real results directory was not overwritten). It exited 0 in 1127 s and a full recursive numeric diff
against the on-disk JSON returned exactly one difference: /runtime_sec 508.14 -> 1127.08. Every
substantive number is bit-for-bit reproducible from the shipped code and data.

**Independent re-derivation (my own Lomb-Scargle, validated at 1.6e-16 against scipy; my own AR(1)
generator).** Median per-pass 860-1310 uHz band percentile vs AR(1): A claimed 0.471 / recomputed
0.4755; B claimed 0.0085 / recomputed 0.0075; C claimed 0.7105 / recomputed 0.711 -- all within
Monte-Carlo noise of 1000 surrogates. Stacked global peaks A 489 uHz (0.200109082358112), B 2474 uHz
(0.040382537522277), C 1706 uHz (0.066821155020203) reproduce to <=5e-17. Rayleigh R at 1010/1210 uHz
reproduces to the last digit for all three models; median spectral slopes reproduce to the last digit.
The core null result is real and survives an entirely independent pipeline.

**The material defect: the sec4b AR(1) stacked-peak nulls evaluate a different statistic than the
observed one.** The observation is max over frequency of the 22-pass average spectrum. The null is
built as the 22-pass average of each surrogate pass's own maximum power. By Jensen the latter is
systematically larger. I reproduced the track's null exactly with my own code (A q50 0.24297 vs
reported 0.24247, q95 0.26717 vs 0.26988), then built the correctly matched null: max-of-pass-mean
q50 0.0412, q95 0.0965, q99 0.1053 for A, roughly 1.7x below the reported null. The effect on the
quoted p-values is decisive: reported p_stacked_global_peak_vs_ar1 A=0.99667, B=1.0, C=1.0 become
A<0.004 (0/300), B<0.004 (0/300), C=0.0267 (8/300); with the track's own surrogate convention the
matched values are A<0.007 and B<0.007. The same mismatch drives the band-limited variant:
p_vs_matched_ar1_null A=1.0 and C=0.994 become 0.185 and 0.065. The null for the circular-shift
("phase_randomised") stacked peak does use the matched statistic and its p (A 0.00333) reproduces.

**A mislabeled null.** The circular-shift null is described as periodogram-preserving. With the
irregular 1-minute sampling held fixed a circular shift is a time translation, not a phase
randomisation; I measured the normalized LS power change it induces: median max-relative change 0.275
over the 22 passes, worst pass 0.810, >0.4 for 12 of 22 passes. I confirmed this is not an artefact of
my implementation: scipy.signal.lombscargle gives the same, and the effect persists on a synthetic
exactly-uniform 1-minute grid (max-relative change 0.37-0.92 depending on AR coefficient). The null
therefore does not stay on the observed periodogram, contrary to its purpose statement.

**A stated-verdict clause is false for model B.** The verdict that "pass-level 860-1310 uHz band power
is consistent with an AR(1) red-noise null in all three models" fails for B: median percentile 0.0085,
0 of 22 passes above the median, two-sided sign test p=4.8e-7, and the out-of-sample split-half check
agrees (held-out ratio median 0.211 vs null percentile 0.0085, p=4.8e-7). B shows a reproducible
band-power deficit relative to AR(1), i.e. the AR(1) null is rejected for B too -- in the opposite
direction. The track computed these numbers and did not flag them. For A the band test is genuinely
null-consistent (median 0.471, 0/22 above q95), so the headline "NO reproducible localized line at
860-1310 uHz" holds for A and B but for B it is a deficit, not a null.

**The pass-peak "support" chance expectation is invalid.** expected_passes_by_chance_fixed=3.315
assumes uniformly distributed pass-peak frequencies. Under a proper red-noise full-pipeline null the
expectation is 8.29 per 22 passes for A (2.5-97.5% interval 4-12), 4.62 (2-8) for B, 5.30 (2-9) for C.
Moreover A's 16/22 support is explained by a broadband low-frequency excess rather than a line: the
stacked peak is only 1.16x the local median within one resolution element (2.87x the median 700 uHz
away), and A's per-pass peaks are broadly scattered (median 557 uHz, IQR 432-656) with the 489 uHz
stacked peak matching a low-frequency envelope shared across passes. The AR(1) surrogates do not
reproduce this envelope, which is why the matched AR(1) null flags A and B.

**Reproducibility and boundary issues.** (i) sec4b phase_randomised_stacked_null reports top-level
"n_realisations": 500 while the executed loop and every per-model field say 300 -- a consumer reading
the top-level field is misled by a factor 1.67 in null resolution. (ii) Model C's "1/22 above the 99th
percentile" and "3/22 above q95" are boundary-fragile: pass_03 model C sits at percentile 0.949-0.957
across four independent surrogate seeds, so the q95 count oscillates between 3 and 4 of 22; the counts
need a Monte-Carlo tolerance, though the no-line conclusion is unaffected.

**Error modes I checked and did not find.** No p-value or CI anywhere treats time samples or grid rows
as independent -- every sign test, Wilcoxon, KS and percentile in the JSON is aggregated at the 22-pass
level, and the phase test's uniform-phase null is a legitimate chance null (its R values, p=0.187/0.679/
0.139 at 1010 uHz and 0.288/0.965/0.274 at 1210 uHz, reproduce exactly). No cartesian-grid claim, no
physical-uniqueness claim from information criteria, and no asserted number lacking a code path. The
band/percentile null result is not a bug: my independent implementation reproduces it, AR(1) surrogates
are drawn independently of the observed series, and the model C marginal excess (16/22 above median,
Wilcoxon p=0.025) is the only borderline signal, consistent with the track's own partial wording.

**Bottom line.** The load-bearing descriptive numbers and the headline "no reproducible localized
860-1310 uHz line" are confirmed for model A (null-consistent) and for model B (significant deficit).
But the track's stacked-peak AR(1) p-values, its band-limited stacked p-values, one clause of its
stated verdict, and its pass-peak chance expectation are wrong or misleading as shipped, and one null
is mislabeled. These change the interpretation of the strongest stacked excess (models A and B were
declared red-noise-consistent at p~1.0; a matched null makes them significant), so the verdict is
partially confirmed with major severity.
