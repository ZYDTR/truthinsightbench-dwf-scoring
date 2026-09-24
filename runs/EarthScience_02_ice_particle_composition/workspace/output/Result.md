# Depth-Resolved Particle Abundance and Composition in Ice Samples

**Task** `EarthScience_02_ice_particle_composition` · **T0** 2018-01-01 · **Payload** bundled

This report is the sole authoritative account of the registered findings. Every number below
was produced by executing code in this workspace; the producing script and result artifact are
named next to each result. Reproduction instructions are in `output/reproduce.md`.

**Scope of the evidence.** The only direct evidence is the two workspace tables
`data/particle_composition_profiles.tsv` (30 rows × 83 columns) and
`data/sample_environment_profiles.tsv` (30 rows × 15 columns), whose SHA-256 hashes match
`data_manifest.json` (`output/results/01_data_audit.json`). No external dataset, no
precomputed task result, and no external literature was used. No external source is cited
anywhere in this report, for the reason recorded in `output/reproduce.md`
("Failed / unavailable steps"): the planned literature check was rejected upstream before
returning any source, and the contract requires a verified first-public date strictly before
T0 for every source used. Consequently every mechanistic statement below is explicitly
labelled as *interpretation*, not as literature-backed fact.

**Analysis unit.** One sampled depth interval. The 30 intervals fall into 5 collection groups
(`group_01` n=5, `group_02` n=7, `group_03` n=7, `group_04` n=5, `group_05` n=6). Intervals
within a group are repeated observations of one ice station, not independent sites; all
inference therefore either uses within-group contrasts or clusters standard errors at the
group level (`G-1 = 4` degrees of freedom). Latitude, longitude and `ice_regime` are constant
within every group, so they cannot be separated from station identity and are not used as
explanatory variables. Only `temperature_C` for `sample_023` and `sample_024` is missing.

---

## Finding 1 — The abundance–depth relationship is a broad but statistically unresolved tendency, not an established decline

**Object and scope.** Total particle concentration (`particle_total_per_L`) versus depth
(`depth_mid_m`), compared *within* collection groups, over all 30 depth intervals.

**Relationship and direction.** In 4 of 5 groups the concentration is lower in deeper
intervals, so the *direction* is consistently negative, but the magnitude is not
distinguishable from zero once the repeated-measures structure and the small number of
independent groups are respected.

**Core quantitative result (analysis unit: depth interval; clustered in 5 groups).**
Pooled across groups, Spearman ρ(depth, log₁₀ total) = **−0.338** (p = 0.067, n = 30). The
within-group mean ρ is **−0.287** (per group: +0.100, −0.857, −0.321, −0.100, −0.257). The
within-group slope from a group-demeaned regression is **−0.826 log₁₀(per L) per metre** with
cluster-robust SE 0.570, **p = 0.221**, 95 % CI [−2.41, +0.76]. Deepest-versus-shallowest
interval pairs give a mean log₁₀ ratio of **−0.635** (t = −0.955, p = 0.394, 4 of 5 groups
decreasing).

**Supporting analysis and result evidence.**
`output/analysis/06_robustness.py` → `output/results/06_robustness.json` (`F1_depth_abundance`);
`output/analysis/03_candidates.py` → `output/results/03_candidates.json` (`C1_total_vs_depth`);
independent re-derivation in `output/results/verify_F1.json` (produced by
`output/results/verify_F1.py`, `verify_F1_part2.py`, `verify_F1_part3.py`) and
`output/results/10_figures_adversarial.json`. Figure: `output/figures/fig1_depth_abundance.png`
with plot data `output/results/fig1_depth_abundance.csv`.

**Robustness checks (all executed).**

- *Whole-group leave-one-out* (5 refits, cluster-robust): slope ranges −1.089 to −0.206. It is
  significant only when `group_02` is retained; dropping `group_02` removes the effect
  (slope −0.206, p = 0.533).
- *Alternative exposure — relative depth* (`depth_mid / max depth` of the group): slope
  −0.946, p = 0.283. *Alternative exposure — depth from the ice surface* (`depth_max_m`):
  −0.855, p = 0.214. *Alternative form — log–log power law:* slope −0.751, p = 0.091, R² = 0.099.
- *Exact group sign-flip null* (all 2⁵ = 32 sign configurations of the group means):
  observed −0.826, **p = 0.3125** two-sided. This null makes no distributional assumption
  and is the sharpest available test; it does not reject.
- *Within-group depth permutation* (20 000 draws, preserving group sizes and the observed
  values): the independent verifier reports **p = 0.033** two-sided (one-sided 0.017) and the
  main pipeline reports p = 0.029 for the demeaned correlation. This test is **more
  permissive** than the sign-flip null because it conditions on the observed values within
  each group; the disagreement between the two is reported rather than resolved in favour of
  the significant one. A permutation that drops `group_02` gives p = 0.580, confirming that
  the permissive result depends on that group.
- *Alternative scale*: on the raw (untransformed) concentration the within-group permutation
  gives p = 0.017 two-sided, i.e. the tendency is not an artefact of the log transform.
- *Alternative estimator*: Theil–Sen within-group slopes (+0.417, −2.118, −1.013, −0.141,
  −0.232) reproduce the same pattern.
- *Cluster bootstrap* (40 000 draws, resampling groups): 99.97 % of draws negative,
  interval [−1.54, −0.236], but with only 5 clusters the bootstrap is lumpy and the
  effective number of distinct groups per draw averages 3.36, so it is reported as
  indicative only.

**Counterevidence, limitations and alternative explanations.**

- `group_01` is the single sign exception and is also the only `Landfast ice` group; it is
  also the shallowest profile (0.14–1.04 m). Because regime is perfectly nested in group,
  "pack ice loses particles with depth, landfast ice does not" and "these five stations
  differ" are observationally indistinguishable here.
- `group_02` alone supplies 70 % of the within-group covariance numerator and 49 % of the
  denominator; the pooled within-group result is therefore dominated by one station whose
  depth profile happens to span 0.09–1.53 m and whose top interval carries the extreme
  75 143 per L.
- The pooled test is *anti-conservative*: treating 30 intervals as independent gives
  p = 0.0155 (classical) versus p = 0.221 (cluster-robust). The pooled classical
  p ≤ 0.05 is an artefact of ignoring the group structure.
- Depth is monotone in the sampling order and confounded with every other depth-correlated
  quantity (temperature, chlorophyll a, salinity), so a depth effect cannot be isolated
  from a general "bottom of the core is different" contrast.
- *Interpretation (not evidence):* a downward tendency is physically plausible if particles
  are scavenged or incorporated near the ice surface, but this dataset cannot distinguish
  that from station-level differences.

**Executable new observation that could falsify or validate this finding.** Re-sample ≥ 20
independent ice stations (not 5), each with ≥ 5 contiguous depth intervals spanning
≥ 1.5 m, and pre-register the within-station slope of log₁₀(total) on depth. The finding
predicts a negative mean slope with a between-station SD no larger than the observed
~1.13 log₁₀ units per metre. If ≥ 10 of 20 stations show a positive slope, or if the
sign-flip null on the enlarged sample remains non-significant, the tendency should be
downgraded from a broad structure to station-specific behaviour.

---

## Finding 2 — Within a station, particle concentration falls steeply as chlorophyll a rises; this survives removal of the deepest interval and is not explained by depth alone

**Object and scope.** Total particle concentration versus chlorophyll a
(`chlorophyll_a_ug_L`), compared within collection groups, all 30 depth intervals.

**Relationship and direction.** Negative and steep: higher chlorophyll a is associated with
markedly lower total particle concentration inside the same ice station.

**Core quantitative result (analysis unit: depth interval; clustered in 5 groups).**
Within-group mean Spearman ρ = **−0.496** (per group −0.975, −0.919, −0.487, +0.100, −0.200;
4 of 5 negative). The group-demeaned slope is **−1.143 log₁₀(per L) per log₁₀(µg/L)**,
cluster-robust SE 0.329, **p = 0.025**, 95 % CI [−2.06, −0.23], within-group R² = 0.486.
In original units a ten-fold increase in chlorophyll a is associated with total concentration
falling by a factor of **10^(−1.143) ≈ 0.072**, i.e. about a 93 % reduction. A two-stage
estimate across the 5 group-level slopes gives a mean of −1.042 (t = −2.20, df = 4,
p = 0.092); the cluster jackknife gives −1.107 (SE 0.441, t = −2.51, p = 0.066).

**Supporting analysis and result evidence.**
`output/analysis/06_robustness.py` → `output/results/06_robustness.json`
(`F2_chlorophyll_abundance`); `output/analysis/03_candidates.py` →
`output/results/03_candidates.json` (`C2_chl_vs_total`); independent adversarial verification
with its own code in `output/verify/verify_F2.json` and `output/verify/verify_F2_adversarial.json`
(driven by `output/verify/verify_F2.py`, `verify_F2_adversarial.py`,
`verify_F2_cluster_inference.py`); further checks in `output/results/10_figures_adversarial.json`.
Figure: `output/figures/fig2_chlorophyll_abundance.png` with plot data
`output/results/fig2_chlorophyll_abundance.csv`.

**Robustness checks (all executed).**

- *Deepest-interval removal.* Dropping the deepest interval of every group **strengthens** the
  result: slope −1.782, SE 0.265, **p = 0.0026**, n = 25. Dropping the two deepest of each
  group: −1.555, p = 0.102 (n = 20); three deepest: −2.067, p = 0.036 (n = 15).
- *Decisive confound test.* Chlorophyll a and depth are themselves strongly related within
  group (mean within-group ρ = +0.579; within-group R² of log chl on depth = 0.549), so a
  pure "surface-versus-bottom" explanation was tested directly. Adding a separate
  deepest-interval indicator to the model leaves the chlorophyll coefficient essentially
  unchanged: β(log chl) = **−1.727, p = 0.0019** (deepest-interval coefficient +1.005,
  p = 0.018). The association is therefore **not** reducible to a bottom-of-core contrast.
- *Depth control.* With depth (and depth²) also in the model: β(log chl) = −1.571, p = 0.024.
  With depth tertile dummies: β(log chl) = −1.773, p = 0.0017 while the deep-tertile dummy is
  also significant (+1.212, p = 0.008) — chlorophyll a and the deep interval each carry
  independent signal.
- *Whole-group leave-one-out*: slope ranges −1.271 to −0.682; it remains negative in all five
  refits but is no longer significant at 5 % in any single refit (largest p = 0.110), which
  is expected with only 4 remaining clusters.
- *Alternative covariate*: log POC alone gives −1.125 (p = 0.348) and log PON −0.891
  (p = 0.346), i.e. chlorophyll a is the only covariate of the set that individually reaches
  significance; in a joint model with depth and log POC the chlorophyll coefficient is
  −1.701 (p = 0.014) and log POC is +1.267 (p = 0.013).
- *Alternative scale/estimator*: rank-based within-group slope −0.554 (p = 0.036);
  within-group Kendall τ mean −0.437 (per group −0.949, −0.781, −0.390, 0.000, −0.067).
- *Exact group sign-flip null*: p = 0.125 two-sided — not significant. A within-group
  permutation null gives p = 0.00045.

**Counterevidence, limitations and alternative explanations.**

- The five independent groups are the real replication, and the sign-flip null over those
  five groups does not reject (p = 0.125). The evidence is therefore strong on the
  interval level and weak-but-consistent on the group level.
- Two groups drive nearly all of it: `group_02` alone contributes 70 % of the within-group
  covariance numerator. Excluding the three samples with the highest chlorophyll a
  (`sample_012` 1.74, `sample_017` 4.19, `sample_024` 6.64 µg/L) leaves slope −1.421
  (p = 0.064, n = 27): the direction is unchanged but the significance is lost.
- Chlorophyll a is coarsely quantised (23 distinct values of 30, 12 at or below 0.1 µg/L,
  minimum step 0.01) and the three largest values carry more than twice the total variance
  (variance ratio 2.16), so the covariate's dynamic range is set by 3 of 30 samples.
- `sample_024` is simultaneously the lowest-abundance sample (33 per L) and the
  highest-chlorophyll sample (6.64 µg/L), and it also has the coarsest composition
  resolution (1 particle = 5.88 percentage points, Finding 4). Excluding it alone leaves
  slope −1.238 (p = 0.035).
- *Interpretation (not evidence):* the sign pattern is what one would expect if the deepest,
  chlorophyll-rich interval is a biological mat or bottom community in which particle
  numbers are counted differently, or if abundant algal cells displace or mask
  microplastic-sized particles in the analysis. Both are hypotheses this dataset cannot
  discriminate; the dataset contains no measurement that separates "particles are absent"
  from "particles are not counted".

**Executable new observation that could falsify or validate this finding.** Sample ≥ 15
independent ice stations with ≥ 6 depth intervals each, measuring chlorophyll a by a method
with a continuous calibration (e.g. HPLC pigments or in-situ fluorescence) rather than the
0.01 µg/L-rounded value used here, and count particles in fixed volumes. The finding predicts
a within-station slope near −1.1 log₁₀ units per decade of chlorophyll a that persists after
excluding the deepest interval of each station. A mean slope ≥ 0, or a slope that vanishes
once the deepest interval is excluded, would falsify it.

---

## Finding 3 — The polyethylene share rises, and the varnish share falls, with particle abundance in the low-abundance range; the relationship is not detectable at high abundance

**Object and scope.** Polymer assemblage composition — the polyethylene share
(`polymer_PE_percent`) and the varnish share (`polymer_varnish_percent`) — versus total
particle concentration, within collection groups, all 30 intervals. **The claim is scoped to
the low-abundance part of this dataset** (below roughly 1 000 particles per L), for the
reason given under "Threshold sensitivity" below.

**Relationship and direction.** Positive for polyethylene, negative for varnish: the
assemblage becomes more polyethylene-dominated and less varnish-dominated as total particle
concentration *falls* toward the low-abundance end of the observed range.

**Core quantitative result (analysis unit: depth interval; clustered in 5 groups).**
Pooled Spearman ρ(log₁₀ total, PE %) = **+0.647** (p = 1.1 × 10⁻⁴, n = 30); within-group mean
ρ = **+0.537** (per group +0.300, +0.883, +0.857, +0.900, −0.257). The group-demeaned slope
is **+29.8 percentage points of PE per decade of concentration**, cluster-robust SE 6.13,
**p = 0.008**, 95 % CI [+12.8, +46.9], within-group R² = 0.571. The two-stage estimate over
group-level slopes gives +33.0 (t = 2.82, df = 4, p = 0.048).

**Supporting analysis and result evidence.**
`output/analysis/06_robustness.py` → `output/results/06_robustness.json`
(`F3_PE_share_abundance`); `output/analysis/03_candidates.py` →
`output/results/03_candidates.json` (`C4_PE_vs_abundance`); counting-null test in
`output/analysis/05_quantisation_null.py` → `output/results/05_quantisation.json`
(`counting_null_PE_share`); independent adversarial verification in
`output/verify_F3/verify_F3.json` (driven by `output/verify_F3/verify_F3_adversarial.py`,
run log `output/verify_F3/run.log`).

**Robustness checks (all executed).**

- *Whole-group leave-one-out* (5 refits): slope 26.0 → 34.7, never leaving the
  positive range; significant (p < 0.05) when `group_01`, `group_03`, `group_04` or
  `group_05` is dropped, and p = 0.096 when `group_02` is dropped.
- *Alternative outcomes*: varnish percent −17.4 pp per decade (**p = 0.032**); the
  exclusive PE share `PE/(PE+varnish)` +28.7 pp per decade (**p = 0.004**).
- *Alternative exposure*: pooled ρ(log₁₀ total, log₁₀ PE count) = +0.834 (p = 1.0 × 10⁻⁸).
  **This one is not independent evidence**: `log₁₀(PE count) = log₁₀(total) + log₁₀(PE %/100)`
  holds exactly, so the reported +1.17 slope is the mechanical 1.0 plus the PE-share trend
  (test of slope = 1: p = 0.61). It is listed here only to record that it was checked and
  found to add nothing.
- *Zero/pseudocount sensitivity of the log-ratio*: the log-ratio slope is 3.45 (p = 0.176),
  2.80 (p = 0.043), 2.19 (p = 0.011) for pseudocounts 10⁻⁹, 10⁻³ and 0.5 respectively, and
  3.92 (p = 0.013) when the single PE = 0 sample is dropped. Direction is robust; the
  cluster p-value is not, which is reported as fragility rather than a significant result.
- *Covariate specificity*: among within-group covariates, log₁₀ total has the highest
  within-group R² against PE % (0.571), ahead of log chl (0.303), POC (0.138), C:N (0.108),
  depth (0.032) and salinity (0.009), and adding depth leaves the log-total coefficient at
  +33.1 (p = 0.009) with depth itself not significant (+14.0, p = 0.247).
- *Resolution restriction*: excluding the single coarsest-resolution sample
  (`sample_024`, 1 particle = 5.88 percentage points) gives +30.7 (p = 0.007), so the
  result is not carried by the worst-resolved sample.
- *Counting-noise null* (the strongest alternative explanation tested): a multinomial
  simulation that draws counts at each sample's own reconstructed tally (Finding 4) from its
  group-mean polymer composition and computes the same within-group statistic yields a null
  mean of **−0.002** (SD 0.220, 95 % range [−0.423, +0.426]) against the observed +0.537,
  giving **p = 0.011** (4 000 simulations). Pure counting noise therefore does not generate
  this pattern.
- *Joint control*: in a within-group model with depth and log chlorophyll a, the
  log-total coefficient on PE % is +26.2 (p = 0.010) while depth (+29.1, p = 0.100) and
  chlorophyll a (−24.5, p = 0.122) are not significant, i.e. abundance carries the signal
  among these three collinear predictors.
- *Depth partial permutation* (20 000 draws, depth held fixed within group): partial
  r(log total, PE % | depth) = +0.761, **p = 0.0002**, versus a null range of
  [−0.389, +0.427]. Permuting depth instead leaves the observed value inside the null
  range (p = 0.211), confirming the signal is in abundance rather than depth.
- **Threshold sensitivity — the decisive qualification.** Restricting to samples with
  total > 1 000 per L (n = 15, 4 groups) reduces the within-group slope from +29.85 to
  **+11.47** (SE 13.01, p = 0.443), the pooled ρ from +0.647 to +0.472 (p = 0.076) and the
  mean within-group ρ from +0.537 to **−0.078**. At a > 3 000 per L threshold (n = 12,
  3 groups) the slope is +1.95 (p = 0.849). The abundance–composition relationship is
  therefore **carried by the low-abundance half of the dataset**, exactly the half whose
  percentages are coarsely resolved (Finding 4). This is reported as material counterevidence,
  not as a footnote.

**Counterevidence, limitations and alternative explanations.**

- **The threshold test above is the strongest counterevidence.** Within the well-resolved,
  high-abundance subset the effect is not detectable. Two readings are possible and the
  data do not separate them: (i) the relationship is a low-abundance phenomenon, or
  (ii) the relationship is an artefact of coarse resolution at low abundance. The
  counting-null simulation (p = 0.011) argues *against* a pure counting-noise explanation,
  and the low-tally samples being varnish-dominated rather than PE-dominated argues against
  the "default class" artefact, so reading (i) is better supported. But the low band and the
  coarse-resolution band are the *same samples*, so the two readings are not separable in
  this dataset. That is why the registered conclusion is scoped to the low-abundance range
  and why no causal direction is claimed.
- The low-band robustness is real but thin: in the low band (≤ 1 000 per L, n = 15, 3–4
  groups per fit) the within-group slope is +48.9 (SE 18.9, p = 0.081), and the group
  coefficients are +30.8 (low band) versus +17.1 (high band), a difference that is **not**
  significant (Δ = −13.7, p = 0.035 only on a nominal test that the same verifier rates as
  unreliable at G = 5).
- `group_05` looks like an exception (ρ = −0.257 over 6 samples), but the adversarial check
  shows this is **one data point, not a regime**: dropping the single sample `sample_025`
  (the group's most abundant sample and the only sample anywhere with PE = 0) flips the
  group's slope from −1.16 to **+12.97** and makes the group_05 interaction significant in
  the *opposite* direction (Δ = +10.88, p = 0.0003). `group_05`'s PE also tracks depth
  (ρ = +0.89) rather than abundance, so its apparent exception status is not a station-level
  property. Removing `group_05` entirely raises the pooled ρ from +0.647 to +0.845
  (p = 2.1 × 10⁻⁷).
- **The "varnish falls" half is tautological.** Within-group corr(PE, varnish) = −0.81 and
  each explains 65 % of the other's within-group variance, so the varnish decline is
  arithmetically the mirror of the PE rise. Registering them as one finding (as done here)
  is not a stylistic choice; treating them as two independent conclusions would double-count
  a single compositional axis.
- **Small-`G` inference is anti-conservative.** The nominal cluster-robust p = 0.008 is not
  design-based: the exact group sign-flip gives p = 0.152 two-sided (0.091 one-sided) and a
  wild cluster bootstrap p = 0.128 (0.065 one-sided). The smallest attainable one-sided
  p with 5 groups is 0.031. Between-group slope heterogeneity is significant
  (Cochran Q p = 0.0025, τ² = 329), so the five group slopes do not estimate one common
  effect; a random-effects mean is +27.7. The interval-level significance should therefore
  be read as a description of the pooled within-group pattern, not as a population claim.
- **The outcome is bounded.** PE % saturates near 100 % in the richest samples (median
  99.7 % above 10 000 per L), so a constant "pp per decade" model is misspecified at the top
  of the range and the pooled ρ is partly generated by the few samples where PE ≈ 99–100 %.
- Finally, the "PE is the classifier's default class when few particles are analysed"
  artefact was tested and **refuted in direction**: low-tally samples have *low* PE %
  (mean 23.6 %) and *high* varnish % (41.6 %), the opposite of what that artefact predicts.
- PE and varnish are near-exclusive complements (pooled ρ(PE, varnish) = **−0.880**, and the
  two together are 26.6–99.9 % of all polymer counts), so "PE rises" and "varnish falls"
  are *one* compositional fact, not two independent findings. Registering them as a single
  conclusion is deliberate.
- The polymer percentages are **coupled representations of the same counted particles** as
  the total concentration (manifest warning): `count = percent/100 × particle_total_per_L`
  holds to floating point for all 39 class columns. The relationship is therefore a
  statement about *relative* composition only; it cannot be restated as an absolute
  polyethylene concentration increase without circularity.
- *Interpretation (not evidence):* a compositional shift toward polyethylene at high
  particle abundance would be consistent with a secondary microplastic source dominating
  the high-abundance samples while varnish-like paint/coating particles dominate
  low-abundance ones. The dataset contains no source attribution, so this remains a
  hypothesis.

**Executable new observation that could falsify or validate this finding.** Collect
≥ 15 independent stations, count ≥ 500 particles per station by FTIR/Raman so that
compositional percentages resolve to ≤ 0.2 percentage points, and pre-register two regressions
within station: PE share on log₁₀ total over the **full** abundance range, and over the
**high-abundance** range only (> 1 000 per L). The finding predicts a positive slope of
roughly +30 pp per decade in the full range that becomes flat at high abundance. A positive
slope that persists at high abundance with fine resolution would falsify the low-abundance
scoping and reinstate a general relationship; a full-range slope near zero at fine resolution
would falsify the finding entirely.

---

## Finding 4 — Composition percentages are counts over a per-sample tally, giving a detection floor that rises sharply as abundance falls

**Object and scope.** Metrological structure of `particle_composition_profiles.tsv`:
the relationship between the reported percentages and the underlying integer particle tally,
across all 30 samples.

**Relationship and boundary.** The percentage columns are integer counts divided by the
sample's total counted particles *N* (within each class family), so a class can be reported
only if at least one particle of it was counted. The smallest representable value is
therefore 100/*N* percentage points. Because the table reports 2 decimals, *N* can be
partially inverted.

**Core quantitative result (analysis unit: sample).**

- The count columns are exactly derived: `count − percent/100 × particle_total_per_L` has
  maximum absolute residual **3.96 per L** and median residual **0.00** over all 39 class
  columns, and the polymer and size families each reproduce their own percentages with a
  family denominator that agrees with the reported total to within 1.3 %. The count columns
  therefore carry **no information beyond the percentages**.
- The percentage columns sum to 100.00 ± 0.02 within each family in every sample.
- Inverting the rounding: a smallest consistent tally *N* can be found for all 30 samples,
  and in **15 of 30 samples the polymer classification and the size classification yield the
  identical smallest consistent *N*** (`output/results/07_resolution_spectrum.json` →
  `tally_bounds_summary.n_both_families_share_smallest_N`, list of the 15 sample IDs in
  `agreeing_samples`). Those *N* span **17** (`sample_024`) to **2 436** (`sample_021`).
- For those 15 samples the implied one-particle concentration `particle_total_per_L / N`
  lies between **0.48 and 1.94 per L** — a narrow range consistent with a roughly constant
  counted volume, which is the physical reason the two independent classifications agree.
- Because true *N* ≥ the smallest consistent *N*, the resolution limit quoted here is an
  **upper bound**: the real floor is at least as coarse as 100/*N*.
- Consequence, in the samples with the coarsest resolution: `sample_024` (33 per L, *N* = 17)
  has a floor of **5.88 percentage points per particle**; `sample_011` (41 per L, *N* = 64)
  1.56 pp; `sample_020` (90 per L, *N* = 76) 1.32 pp.
- The number of size classes that can be reported at all tracks abundance:
  ρ(nonzero size bins, log₁₀ total) = **+0.69** (p = 2.3 × 10⁻⁵), from 3 nonzero bins in the
  poorest sample to 10 in the richest.

**Supporting analysis and result evidence.**
`output/analysis/07_resolution_and_spectrum.py` →
`output/results/07_resolution_spectrum.json` (`tally_bounds`, `tally_bounds_summary`,
`derived_count_check`); `output/analysis/01_data_audit.py` →
`output/results/01_data_audit.json` (`polymer_count_equals_percent_x_total`,
`size_count_equals_percent_x_total`, `polymer_percent_sum`, `size_percent_sum`);
earlier reconstruction restricted to both-family agreement in
`output/analysis/05_quantisation_null.py` → `output/results/05_quantisation.json`;
independent adversarial verification in `output/results/verify_F4_maincheck.json`
(chance-agreement null, uniqueness, bound direction, falsification check: see below). Figure:
`output/figures/fig4_quantisation.png` with plot data `output/results/fig4_quantisation.csv`.

**Robustness checks (all executed).**

- *Two independent classifications.* The polymer family (17 classes) and the size family
  (22 bins) are different partitions of the same counted particles. Requiring both to admit
  the same smallest *N* is a strong test; 15/30 pass it.
- **Chance-agreement simulation (the decisive test of that agreement).** Two competing null
  models were simulated 400 times each, generating integer count vectors from a
  Dirichlet-multinomial, rounding to 2 decimals exactly as the source table does, and running
  the same search (`output/results/verify_F4_maincheck.json`):
  if the two families share the true tally, the smallest consistent *N* agrees in
  **400/400** simulations; if the two families have independent tallies drawn from the same
  range, agreement occurs in **2/400** (0.5 %). The observed 15/30 (50 %) therefore lies far
  from the independent-tally null and is consistent with a genuinely shared tally. This is
  strong evidence that the reconstructed tallies are real, not an artefact of the search.
- *Uniqueness (a real limitation).* The set of consistent *N* is not a single value: for
  *N* ≤ 3 000 the solution-set size has median 4 (up to ~190) per family. *N* is therefore
  **bounded, not identified**, and every quoted quantum is the sharpest member of a set.
- *Two-decimal consistency.* Every nonzero reported percentage is reproduced by
  integer/N to within 0.0051 percentage points, i.e. within the table's own rounding, and the
  implied counts sum to *N* (not merely ≤ *N*) in the agreeing samples.
- *Independence of the two families.* A Mantel test on the sample-to-sample distance
  matrices of the polymer and size compositions gives r = −0.024 (p = 0.94, 20 000
  permutations), i.e. the two classifications carry essentially **no shared** compositional
  information. Their agreement on *N* is therefore not a trivial consequence of one
  determining the other.
- *Direction of the bound.* Stated explicitly as an upper bound on the resolution limit; the
  step-5 field named `both_families_share_smallest_N` was used rather than the earlier,
  mis-ordered "common minimum", and the earlier ill-posed quantum search in
  `output/analysis/04_resolution.py` is superseded.

**Counterevidence, limitations and alternative explanations.**

- The reconstruction is not unique (see the uniqueness check above): the quoted *N* is the
  smallest of a median-4-element solution set, so the quoted resolution is a best case.
- In 15 of 30 samples the two families do **not** share a single smallest *N*, so for half
  the dataset the tally is only bounded, not determined.
- **One falsification-check violation, retained.** For 23 samples the reconstructed *N* is
  available and the model requires the smallest reported nonzero percentage to be at least
  100/*N*. This holds in 22 of 23. `sample_025` violates it marginally: its smallest nonzero
  class is **0.03 %** whereas 100/*N* = **0.0343 %**. The violation is in the direction the
  bound permits — it means the true *N* for `sample_025` is slightly *larger* than the
  smallest consistent value, which is exactly what "smallest consistent *N*" means as a
  lower bound. It is recorded because it is a check the model could have failed.
- The constant-quantum-per-litre interpretation (0.48–1.94 per L across the 15 agreeing
  samples) is an inference from a small, non-random subset: the samples where the two
  classifications happen to agree.
- The reconstructed tallies are internally consistent with the data but are not measurements;
  the tables contain no column that states how many particles were counted.

**Executable new observation that could falsify or validate this finding.** Obtain the
counting protocol or the raw per-particle identification lists for these samples and compare
the reported integer counts with the reconstructed *N*. The finding predicts that the raw
tally equals 100/per-percentage-point resolution for the 15 agreeing samples and that no
reported class corresponds to fewer than one particle. A raw tally inconsistent with the
reconstruction (for example a fixed count of 1 000 particles for every sample) would
falsify it.

---

## Finding 5 — The apparent steepening of the particle size spectrum at low abundance is substantially a counting-resolution artefact, and the residual effect is unresolved

**Object and scope.** The particle size distribution — the 22 `particle_size_*_percent`
bins — versus total particle concentration. The scope statement is a *boundary*: this
dataset can support the shape of the spectrum, but not a claim that the spectrum steepens as
abundance falls.

**Relationship and boundary.** Fitting log₁₀(percent) against log₁₀(bin midpoint) gives a
per-sample slope that is strongly correlated with abundance — but the correlation is produced
mainly by the detection floor of Finding 4, because scarce samples can only report their
coarsest bins. Restricted to bins every sample can report, the relationship disappears, and
the restricted test is too coarse to exclude a moderate real effect.

**Core quantitative result (analysis unit: sample, 30 samples).**

- Over the nine finest bins (11 µm … 200–225 µm) the per-sample slope averages
  **−2.31** (median −2.29, SD 0.74, range −4.09 to −0.60), and it correlates with log₁₀ total
  at ρ = **−0.880** (p = 1.5 × 10⁻¹⁰); within-group mean ρ = −0.737, sign-consistent in 5 of
  5 groups. The group-demeaned slope is −0.670 (SE 0.049, p = 1.7 × 10⁻⁴).
- **Restricted to the three bins that are nonzero in essentially every sample**
  (11 µm, 11–25 µm, 25–50 µm; all three nonzero in 29/30 samples), the association
  **vanishes**: ρ = **+0.067** (p = 0.72), within-group mean ρ = +0.137, group-demeaned
  slope −0.093 (SE 0.268, p = 0.746). Restricting further to the 27 samples with
  *N* ≥ 100 counted particles leaves ρ = +0.021 (p = 0.92).
- Intermediate range (bins 1–4, i.e. up to 75–100 µm): ρ = −0.580 (p = 7.8 × 10⁻⁴) —
  much weaker than the 9-bin value.
- The fitted slope is predicted by how many bins the sample could report:
  ρ(slope, number of nonzero bins) = **−0.501** (p = 0.005), and ρ(slope, log₁₀ *N*) = −0.840
  (p = 6.5 × 10⁻⁹). In a within-group model containing both the number of nonzero bins and
  log₁₀ total, the bin count remains significant (+0.183, p = 0.015) *and* abundance remains
  significant (−0.910, p = 4.6 × 10⁻⁴).
- **Counting-null simulations — two independent runs that disagree, both retained.**
  - *Run 1* (`output/analysis/07_resolution_spectrum.py` → `spectrum_counting_null`,
    3 000 draws): null within-group mean ρ = **−0.430** against the observed −0.737,
    p = 0.020 ⇒ resolution explains ~58 %.
  - *Run 2* (`output/verify_F5/verify_F5.py` → `output/verify_F5/verify_F5.json`, seed 777,
    3 000 draws, percentages rounded to 2 decimals like the source table): null within-group
    mean ρ = **−0.667** against the observed −0.737, **p = 0.300**, i.e. resolution explains
    ~96 % of the within-group association; for the pooled (non-grouped) statistic the null
    explains ~13 % (p = 0.028), and for bins 1–4 it explains ~88 % within group.
  - The two runs differ in whether the sampled percentages are re-rounded to 2 decimals
    before refitting, and they bracket the answer (**58 % to 96 % explained**). The honest
    statement is that resolution explains *most* of the within-group size-spectrum/abundance
    association and that the exact share is not pinned down by this dataset. The run that
    reproduces the table's own rounding (run 2) is the more faithful null and gives the
    larger artefact share; the finding is registered on that basis.
  - An earlier, still weaker attempt (`output/analysis/05_quantisation_null.py`, archived in
    `output/results/05_quantisation.json` → `counting_null_PE_share`) mis-derived the tallies
    and is superseded by both runs above. It is retained in the archive rather than deleted.
- **Power analysis (the reason the finding is "unresolved" and not "artefact only").**
  A synthetic abundance-dependent tilt was injected into the 3-bin sub-composition and the
  test re-run 1 000 times per grid point (`verify_F5.json` → `power_analysis`). Detecting a
  real effect at the 5 % level with 80 % power requires a slope change of roughly
  **2.7 log-units across the observed abundance range**; the observed 9-bin slope range is
  about 3.5 log-units, so only a very large real effect would have been detectable, and the
  **within-group** 3-bin test has essentially **zero power** even at a 6.7-log-unit tilt
  (power 0.20 at the largest simulated effect). The 3-bin restriction therefore cannot
  demonstrate that no real effect exists; it can only demonstrate that the 9-bin effect is
  not recoverable from the universally resolved bins. This is what makes the registered
  conclusion a boundary-plus-unresolved statement rather than a clean null.
- **Which predictor wins.** With the number of nonzero bins and abundance both in a
  within-group model, both retain significance (bin count +0.212, p = 0.015; log₁₀ total
  −0.901, p = 0.00097), so the resolution proxy and abundance carry partly separable
  information rather than one merely proxying the other.
- **Core-bin constraint check**: the three universal bins carry 82–100 % of the particles
  (mean 97.1 %), so their proportions are compositional and leave little free variation.
  This constrains the *level* of the 3-bin slope but does not by itself destroy power
  (the two log-ratios remain free); it is listed as a limitation, not as the explanation.
- The shape of the *universally reported* part of the spectrum is nonetheless a robust
  empirical result: the mean percentages are **67.4 %** (11 µm), **22.6 %** (11–25 µm) and
  **7.1 %** (25–50 µm), and this strictly decreasing ordering holds in **30 of 30 samples**,
  with a log–log slope over those three bins of −1.91.

**Supporting analysis and result evidence.**
`output/analysis/07_resolution_and_spectrum.py` →
`output/results/07_resolution_spectrum.json` (`spectrum`, `spectrum_by_bin_range`,
`spectrum_curvature`, `universal_bins_shape`, `spectrum_counting_null`);
`output/analysis/06_robustness.py` → `output/results/06_robustness.json`
(`F5_size_spectrum_boundary`); `output/analysis/03_candidates.py` →
`output/results/03_candidates.json` (`C6_size_spectrum`); independent adversarial
verification in `output/verify_F5/verify_F5.json` (driven by `output/verify_F5/verify_F5.py`).
Figure: `output/figures/fig3_size_spectrum.png`
with plot data `output/results/fig3_size_spectrum.csv`.

**Robustness checks (all executed).**

- *Bin-range sensitivity*: 1–9 → ρ = −0.880; 1–4 → −0.580; 1–3 → +0.067; 2–5 → −0.838;
  3–7 → −0.775. The dependence of the answer on how many bins are used is itself the
  boundary evidence.
- *Resolution-matched restriction*: the 3-bin and high-*N* fits above.
- *Curvature test*: fitting a quadratic in log₁₀(midpoint) gives a mean curvature coefficient
  of −0.427 with 22/30 samples negative, and that coefficient also correlates with abundance
  (ρ = −0.614, p = 3.1 × 10⁻⁴) — a second, independent form of the same resolution-bound
  pattern.
- *Null simulation* with the sample's own tally, as above.
- *Shape invariants*: the monotone decreasing ordering of the three universal bins holds in
  every sample, and the within-group correlation of the 25–75 µm share with abundance is
  ≈ 0 (pooled ρ = −0.196, p = 0.30; within-group mean ρ = −0.031), i.e. the coarse fraction's
  relative weight shows no abundance trend once resolution is matched.

**Counterevidence, limitations and alternative explanations.**

- **The restriction is demonstrably underpowered** (see the power analysis above): the
  within-group 3-bin test has ~0 power even against a very large injected effect. The
  restricted test therefore cannot distinguish "no real effect" from "real but moderate
  effect that three dominant bins cannot detect". This is why the finding is registered as
  an *unresolved joint outcome* (artefact-dominated, residual effect bounded) rather than
  as a clean null.
- The two independent counting-null runs disagree on the artefact share (58 % vs 96 %), so
  "most of the effect is resolution" is supported while the specific share is not.
- The counting-null simulation uses the group-mean size distribution as the generator, which
  assumes the underlying spectrum is the same for all samples in a group; if the true
  spectrum really does vary with abundance within a group, the null is conservative and the
  true artefact share is smaller.
- The 9-bin slope is not a physical power-law exponent: bins are 25 µm wide at the coarse end
  and the count distribution's upper tail is truncated by the detection floor, so the
  quantity being correlated is partly a shape statistic and partly a resolution statistic.
- Bins above 375 µm are never nonzero in any sample (5 of 22 bins are structurally empty),
  and only one sample reports anything in 475–500 µm; the spectrum's coarse tail is simply
  not sampled in this dataset.
- *Interpretation (not evidence):* a genuine steepening with falling abundance is plausible if
  low-abundance samples are enriched in large, rare particles, but this dataset cannot
  separate that from the count floor.

**Executable new observation that could falsify or validate this finding.** Count fixed,
larger volumes — or image a fixed area — so that at least 500 particles are identified in
*every* sample, including the low-abundance ones (33–150 per L here), and re-fit the 9-bin
slope. If the abundance–slope correlation persists at full resolution with equal bin
coverage, the artefact explanation is falsified and the steepening is real. If it vanishes,
the boundary is confirmed. A pre-registered power calculation should accompany the design,
because the present restricted test cannot exclude slopes within roughly ±0.5 log-units per
decade.

---

## Cross-cutting structure: what varies between stations and what does not

A variance partition (one-way, group as factor; `output/results/08_variance_partition.json`)
separates broad structure from station-specific structure:

| variable | between-group η² | F(4,25) | p |
|---|---|---|---|
| log₁₀ POC | **0.785** | 22.9 | 4.8 × 10⁻⁸ |
| salinity | **0.636** | 10.9 | 2.9 × 10⁻⁵ |
| log₁₀ chlorophyll a | **0.444** | 5.0 | 0.004 |
| C:N ratio | 0.400 | 4.2 | 0.010 |
| PE percent | 0.330 | 3.1 | 0.034 |
| varnish percent | 0.199 | 1.6 | 0.216 |
| log₁₀ total per L | 0.158 | 1.17 | 0.349 |
| **share of particles > 25 µm** | **0.089** | 0.61 | 0.657 |
| **11 µm bin share** | **0.046** | 0.30 | 0.866 |

The chemical and biomass covariates are strongly station-determined, while the size
distribution is nearly invariant across stations — the size spectrum is a *broad* property of
this dataset, whereas chlorophyll a, salinity and POC are *station* properties. This is the
direct evidence for the objective's distinction between broad structure and group-specific
anomalies, and it is consistent with Finding 5's boundary: size composition carries little
station-level signal to correlate with, so any abundance–spectrum claim has to be made at the
interval level, exactly where the resolution floor bites.

At the cluster level (n = 5 independent groups), ρ(log₁₀ total, log₁₀ chlorophyll a) = −0.900
(p = 0.037) reproduces Finding 2's direction on the independent-unit scale; ρ(log₁₀ total,
PE %) = +0.600 (p = 0.285) and ρ(latitude, PE %) = −0.800 (p = 0.104) are consistent in sign
with Finding 3 but not significant with only 5 units.

**Group-specific anomalies.** A within-group |z| > 2 scan flags: `sample_024` (group_03) for
chlorophyll a (+2.22) and POC (+2.20); `sample_012` (group_02) for varnish percent (+2.20);
`sample_011` (group_02) for particles > 25 µm (+2.16) and the 11 µm share (−2.15);
`sample_006` (group_02) for temperature (+2.08). `group_02` contributes 3 of the 5 flagged
anomalies. `sample_011`, `sample_012` and `sample_024` are also the three lowest-abundance
samples (41, 67, 33 per L) and therefore the coarsest-resolved — the anomaly scan and the
resolution analysis (Finding 4) identify the **same** samples, which is expected if
compositional anomalies are partly a small-count phenomenon.

---

## Limitations that apply to the whole report

1. **Five independent units.** No matter how many intervals are analysed, the number of
   independent ice stations is 5. Every cluster-robust p-value here has 4 degrees of
   freedom, and the exact group sign-flip null (32 configurations, minimum attainable
   two-sided p = 0.0625) cannot reach 5 % significance. Interval-level p-values in this
   report should be read as descriptions of within-station patterns, not as
   population-level inference.
2. **One cluster dominates.** `group_02` supplies ~70 % of the within-group covariance for
   Findings 2 and 3 and is the only group whose depth trend reaches significance on its own.
   All three interval-level findings are materially dependent on it; the leave-one-out
   results are reported next to every finding for exactly this reason.
3. **Regime is not testable.** `ice_regime` has two levels and zero within-group variance, so
   "pack ice versus landfast ice" cannot be separated from "these particular stations".
4. **Coupled representations.** Percentages and total concentration describe the same counted
   particles; composition findings are statements about relative composition only
   (`DATA_GUIDE.md` warning, verified numerically in Finding 4).
5. **No causal identification.** Depth, chlorophyll a, temperature, salinity and POC all vary
   systematically with position in the core. The within-group design removes station
   differences but cannot remove the depth ordering itself, so every reported association is
   an adjusted correlation, not an effect.
6. **Variance-estimator disagreement.** For the un-demeaned pooled fits, the hand-rolled
   cluster-robust covariance (CR1 with
   `G/(G−1) × (n−1)/(n−k)` scaling) and `statsmodels`' cluster covariance differ by a factor
   of ~4 in the zero-slope case, so their p-values differ (e.g. 0.194 versus 0.132 for the
   intercept of `log₁₀ total ~ log₁₀ total`). The report quotes the `statsmodels`-compatible
   p-value where the two disagree and always reports the exact sign-flip and permutation
   nulls, which do not depend on a variance estimator at all. The hand-rolled Spearman
   implementation was verified to agree with `scipy.stats.spearmanr` to <10⁻¹⁵ in p.
7. **External literature was not used.** As recorded in `output/reproduce.md`, the literature
   check failed upstream (HTTP 400 content-inspection rejection) before returning any source,
   and the contract forbids using a source whose first-public date cannot be verified as
   strictly before T0. No mechanism stated in this report is attributed to a published
   source.

---

## Reproduction

See `output/reproduce.md` for the environment, the exact run order, the analysis
conventions, the cross-validation of the hand-rolled statistics, and the list of failed or
superseded steps that are retained rather than hidden. Semantic delivery is checked with:

```
python3 validate_output.py output
```