# Energy- and Angle-Dependent Molecular Photoionization Dynamics

**Task ID:** `Chemistry_02_photoionization_delay` · **T0:** `2021-01-01` · **Payload mode:** `bundled`
**Authoritative report.** This file is the sole authority for the registered findings of this run.

---

## 1. Scope, data, and analysis unit

### 1.1 What was analysed

The complete task dataset under `data/` was used; nothing was downloaded, substituted, or merged.
All five scientific files were read in full and their SHA256 digests re-verified against
`data_manifest.json` before any modelling (see `output/evidence/d00_data_integrity.json` and
`output/evidence/d01_inventory.json`; all five digests match).

| File | Content | Groups |
|---|---|---|
| `data/angle_energy_observations_datasets.tsv` | energy×angle phase, delay, magnitude | 10 groups (experiment/theory × phase, delay, magnitude, photon energy, photon-energy-of-delay) |
| `data/dipole_matrix_elements_datasets.tsv` | complex dipole matrix elements | 13 measured `experiment/DME_<E>eV` blocks + 1 `theory/DMEs` block (340 560 values) |
| `data/angular_coefficients_datasets.tsv` | angular-coefficient blocks | 13 `CoefLeg_<E>eV` blocks |
| `data/phase_and_delay_curves_datasets.tsv` | standalone curves | `0.5_Wigner_delay`, `angle_averaged_delay`, `eigenphase_sum` |
| `data/coordinate_definitions.json` | axes and constants | `emission_angle_deg = index_1`, `hbar_as_eV = 658.2119569` |

Grid geometry as reported by `output/analysis/d01_inventory.py`:

* experiment dense grid: **276** photon energies, 22.5 → 50.0 eV, uniform **0.1 eV** step;
  magnitude and phase are `276 × 181`; delay is `275 × 181` (paired with `photon_energy_eV_del`).
* theory dense grid: **99** photon energies, 23.249 → 41.839 eV, **non-uniform** (mean step 0.193 eV,
  minimum step 0.016 eV); magnitude and phase `99 × 181`, delay `98 × 181`.
* emission angle: `index_1` = 0…180 deg, step 1 deg, 181 columns, in every energy-angle array.
* **No non-finite values anywhere**; every declared group is present with exactly the value count
  recorded in `data_manifest.json`.

### 1.2 Analysis unit and the independence boundary

The declared independent analysis unit is *one measured dipole-matrix energy block or one complete
theory calculation block*. This run therefore treats:

* the **13 measured energies** (22.5, 23.25, 24.8, 26.35, 27.9, 29.45, 31.0, 32.55, 34.1, 35.65,
  37.2, 38.75, 48.4 eV) as the effective sample size for every experiment-side statement;
* the **99 theory photon energies** as the effective sample size for theory-side statements;
* the **181 angles inside one energy block** as *within-block angular channels of one matrix-element
  set*, not as independent replicates.

`data/coordinate_definitions.json` states that the dense experiment-like phase/delay grid is a
`derived_spline_grid` built from the 13 measured blocks. The dataset certifies this only for the
phase/delay grid; this run additionally tested it for magnitude (§5) and found it equally
irreproducible from the 13 blocks. **All experiment-side p-values below therefore overstate the
information content** and are labelled accordingly; effect sizes and null comparisons are the
load-bearing quantities.

### 1.3 What was done, and how the evidence is laid out

Seven independent analysis packages were executed in parallel, each producing a script under
`output/analysis/`, a machine-readable result under `output/evidence/`, a full stdout log, and at
least one figure. Each package's headline claims were then re-derived by a **separate adversarial
verifier** that wrote its own script (`output/analysis/*_verify.py`) and its own verdict file
(`output/evidence/*_verdict.json`) using a deliberately different numerical route. Six verifiers ran
as sub-agents; the seventh (`p1_exp_basis_verify.py`) was executed directly after its sub-agent hit
an API error, and its script and verdict are on disk like the others.

Verification outcomes are reported inside each finding below. Across the seven packages, **31 of 34
targeted claims reproduced**; the three that did not are carried into the findings as limitations
rather than dropped — a count error in `p6_exptheory` (11 matched energies, not 12), and two
over-strong wordings in `p1_exp_basis` (the residual floor is *not* below the storage quantum, and
the Fourier-basis collapse factor legitimately differs from the Legendre one). Two additional
cross-cutting analyses were run by the coordinating agent specifically to test whether headline
results survive nulls that the individual packages had not constructed:
`output/analysis/d02_cross_representation.py` and `output/analysis/d02_order_alignment.py`.

**Reading guide.** §2–§6 are the five registered findings. §7 lists failed, conflicting, and
unresolved results. §8 states limitations. §9 gives falsification proposals. §10 is the
finding-to-evidence index. Reproduction instructions are in `output/reproduce.md`.

---

## Finding 1 — The experiment-like panel has an exact finite angular degree, and the degree is 3 in the complex amplitude / 6 in the intensity

**Object and scope.** The two dense experiment-like energy–angle arrays
`experiment/magnitude` and `experiment/phase` (each 276 × 181, 22.5–50.0 eV × 0–180 deg), combined
into the complex amplitude `Z(E, θ) = magnitude · exp(i·phase)`, and its squared modulus (the
angular intensity) `I = |Z|² = magnitude²`.

**Relationship and boundary.** The complex amplitude lies **exactly** in the span of the real
Legendre polynomials `P_l(cos θ)` for `l = 0…3` — no higher term is needed anywhere on the grid.
Equivalently, the angular intensity is **exactly** a degree-6 polynomial in `cos θ`. The boundary is
the residual floor, discussed below.

**Core quantitative result** (analysis unit: one energy block of 181 angular channels; effective
n = 13 independent measured energies):

| Quantity | max&#124;residual&#124; | note |
|---|---|---|
| `Z`, Legendre `L_max = 2` | 5.327 × 10⁻¹ | the naive two-term baseline |
| `Z`, Legendre `L_max = 3` | **9.562 × 10⁻⁵** | collapse by **5 571×** |
| `Z`, Legendre `L_max = 4…12` | 9.33 × 10⁻⁵ … 9.56 × 10⁻⁵ | flat; spread over the plateau 1.025 |
| `I = magnitude²`, `L_max = 5` | 1.411 × 10⁻¹ | |
| `I = magnitude²`, `L_max = 6` | **1.906 × 10⁻⁴** | collapse by **740×** |
| `I = magnitude²`, `L_max = 7…12` | 1.83 × 10⁻⁴ … 1.97 × 10⁻⁴ | flat |

The degree-3 result is not a fitting artefact: the median modulus of the *top* fitted coefficient
drops from `|c_3| = 3.102 × 10⁻¹` at `L_max = 3` to `|c_4| = 1.854 × 10⁻⁶` at `L_max = 4` — a
five-order-of-magnitude discontinuity in the coefficient itself, not merely in the residual.

The four coefficients carry the structure in unequal measure: median fractional coefficient power
is 3.0 % (`c_0`), 23.2 % (`c_1`), 57.9 % (`c_2`), 12.6 % (`c_3`), with per-energy moduli ranging
0.113–0.180, 0.232–0.726, 0.418–1.106 and 0.060–0.848 respectively.

**Supporting analysis and result evidence.**
`output/analysis/p1_exp_basis.py` → `output/evidence/p1_exp_basis.json`,
`output/evidence/p1_exp_basis.png`, `output/evidence/p1_exp_basis_coefficients.csv` (per-energy
complex coefficients), log `output/evidence/p1_exp_basis.log`.
The intensity degree is established by `output/analysis/d02_cross_representation.py` and
`output/evidence/d02_intensity_degree.json`.

**Robustness.**
(i) *Alternative basis* — the same scan in the complex Fourier basis `exp(i·m·θ)` collapses at
`m = 3` by 3 110× (2.783 × 10⁻¹ → 8.950 × 10⁻⁵), so the degree is basis-independent.
(ii) *Angular subsampling* — refitting on every 4th angle (46 of 181 points) changes the
coefficients by at most 1.607 × 10⁻⁵ against a coefficient scale of 1.106, a relative change of
1.5 × 10⁻⁵.
(iii) *Independent numerical route* — `output/analysis/p1_exp_basis_verify.py` re-derived the whole
scan with `numpy.polynomial.legendre.legfit/legval` (the package used an explicit recurrence and
`lstsq`), reproducing the collapse at 5 571×, the plateau spread 1.025, and the coefficient
discontinuity; verdict at `output/evidence/p1_exp_basis_verdict.json`.
(iv) *Null baselines* — the `L_max = 0` (isotropic) model leaves 1.648; the `L_max = 2` model
leaves 0.533 and captures 85.62 % of the angular variance, while `L_max = 3` captures 0.9999999;
`L_max = 4` adds nothing (`|c_4|` ~ 10⁻⁶).

**Counterevidence, limitations, alternative explanations.**
*The floor is not float64 round-off.* The residual floor 9.56 × 10⁻⁵ is
2.4 × 10¹¹ times the float64 quantum of `Z` (3.96 × 10⁻¹⁶). Read against the *stored* precision it
is ~6.9× the largest 5-significant-decimal quantum (1.39 × 10⁻⁵). So the correct statement is: the
representation is exactly degree 3 **to within the precision of the measured source values**, not
to machine precision. A reader should not describe the agreement as exact in the float64 sense.
*The intensity floor is a few times the naive quantum.* At `L_max = 6` the floor is 1.9 × 10⁻⁴, about
5–12× the intensity quantum implied by a 5-decimal magnitude. The degree-6 collapse (740×) is
unambiguous, but "exact degree 6" again means "to within a small multiple of the source precision",
not machine-exact. This differs from the theory panel (§3), whose intensity floor *is* at machine
precision.
*A Chebyshev basis does not show the same clean collapse* — `output/evidence/p1_exp_basis.json`
records that the Chebyshev scan only reaches 8.0 × 10⁻⁵ at order 11. Chebyshev polynomials in
`cos θ` are a differently-conditioned basis and require higher order to resolve the same function;
this is a conditioning effect, not a contradiction, but it is reported because it shows the degree
statement is basis-dependent in its *numerical sharpness* even though Legendre and Fourier agree on
the degree itself.
*Mechanistic caution.* A degree-3 complex amplitude is a statement about the representation, not
about a unique microscopic channel. The data guide explicitly warns that large angular variation
near a local energy region is not by itself evidence for a unique mechanism, and this finding makes
no such claim.

**Falsification proposal.** Take the 13 measured `experiment/DME_<E>eV` blocks and recompute the
dense grid independently from them; if the regenerated grid needs a non-zero `c_4` or higher to
match the published dense grid at the 10⁻⁵ level, this finding is falsified. Concretely: fit
`Z` at each of the 13 measured energies from the DME blocks, interpolate with the generator's own
recipe, and test whether `|c_4|/|c_0| < 10⁻⁵` holds. An independent beamline measurement of the
angular distribution at any one of the 13 energies, with angular residuals below 10⁻⁵ in `Z`,
would validate it.

---

## Finding 2 — The theory-like panel has no finite angular degree in its amplitude, but its intensity is exactly a degree-20 polynomial, and the DME block does not generate it

**Object and scope.** The dense theory-like arrays `theory/magnitude` and `theory/phase`
(99 × 181, 23.249–41.839 eV × 0–180 deg), their complex amplitude `Z_t`, their intensity
`|Z_t|² = theory/magnitude²`, and the `theory/DMEs` block (340 560 values, 12 channels over
`m_ini, m_final, mu, gauge`).

**Relationship and boundary.** The theory complex amplitude is **not** an exactly finite-degree
Legendre polynomial at any degree up to the Nyquist limit of the 181-point angular grid (degree 90).
Its intensity, however, **is** exactly a degree-20 polynomial in `cos θ` at float64 precision. The
`theory/DMEs` block does **not** reproduce the theory dense panel under any channel selection.

**Core quantitative result.**

*Complex amplitude `Z_t`* — the residual decays smoothly with no collapse:

| `L_max` | 2 | 3 | 4 | 5 | 6 | 8 | 9 | 14 |
|---|---|---|---|---|---|---|---|---|
| max&#124;residual&#124; | 9.581 × 10⁻¹ | 1.060 × 10⁻¹ | 2.609 × 10⁻² | 5.071 × 10⁻³ | 7.769 × 10⁻⁴ | 4.077 × 10⁻⁵ | 8.090 × 10⁻⁶ | 7.753 × 10⁻⁶ |

The largest single-step change is a factor **9.04** (L = 2→3), versus a factor of **5 571** for the
experiment panel at its degree. Extending to L = 90 (the angular Nyquist limit) only reaches
3.97 × 10⁻⁶. The coefficient sequence confirms it: median `|c_l|` falls 0.816 (`l = 3`) → 0.103
(`l = 4`) → 0.034 → 0.005 → 0.0006 → 8 × 10⁻⁵ — a smooth geometric decay, not a truncation.

*Intensity `|Z_t|²`* — a sharp collapse to machine precision:

| `L_max` | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|
| max&#124;residual&#124; | 6.02 × 10⁻² | 2.62 × 10⁻³ | 1.38 × 10⁻⁴ | 6.53 × 10⁻⁶ | 2.21 × 10⁻⁷ | 5.13 × 10⁻⁹ | 6.29 × 10⁻¹¹ | **8.36 × 10⁻¹³** | 4.26 × 10⁻¹⁴ |

Flat from L = 20 to L = 30, i.e. the theory magnitude is the square root of an exact degree-20
polynomial in `cos θ`.

*DME reconstruction.* 48 distinct candidate selections per energy (gauge ∈ {1, 2} × {single
channel, μ-summed} × {Legendre `P_l`, associated `P_l^m`} × {`m = 0` only, all `m`} ×
{μ = −1, 0, +1}) were tested at three theory energies (23.249, 32.549, 41.839 eV).
**0 of 144 candidates fall below a 2 % relative-residual reproduction threshold**; the best
single-complex-scale relative residual is 0.531 at 23.249 eV, and a scan of all 99 theory energies
finds nothing better than 0.528 (at 32.9 eV) — still a factor of ~26 above the threshold.

**Supporting analysis and result evidence.**
`output/analysis/p2_theory_dme.py` → `output/evidence/p2_theory_dme.json`,
`output/evidence/p2_theory_dme.png`, `output/evidence/p2_theory_coefficients.csv`,
log `output/evidence/p2_theory_dme.log`.
The degree-20 intensity result is in `output/evidence/d02_theory_intensity_degree.json`
(script `output/analysis/d02_cross_representation.py`).

**Robustness.**
(i) *Three independent numerical routes* — `chebfit`, `numpy.polynomial.legendre.legendre.legfit`,
and a hand-rolled recurrence agree to within 5 % on every residual in the scan
(`output/evidence/p2_theory_dme_verdict.json`).
(ii) *Energy independence of the DME conclusion* — the three tested energies are low, middle, and
high on the grid, and the conclusion (no reconstruction) holds at all three.
(iii) *Controls against chance overlap* — randomly shuffled DME values against the panel give a
median overlap of 0.395 and a maximum of 0.847; wrong-energy DME blocks give a maximum of 0.735.
The best matched-energy candidate reaches 0.847, i.e. **not distinguishable from the best shuffled
candidate**. This is the key negative control: the apparent "best" channel overlap is what one gets
by chance.
(iv) *Verifier independently found the intensity structure* — the p2 verifier reported on its own
initiative that `|Z|²` fits a degree-20 polynomial to 8.4 × 10⁻¹³, confirming this run's separate
finding by a different route.

**Counterevidence, limitations, alternative explanations.**
*"No finite degree" is proven only up to the Nyquist limit.* With 181 angular samples, degree 90 is
the highest resolvable order. The claim is therefore: no finite degree ≤ 90. A generator using a
higher-order expansion sampled on this grid would be indistinguishable from one using a smooth
non-polynomial function.
*The theory grid is matched by index, not by a shared coordinate.* The `theory/DMEs` `T_kin` axis
spans 0.3375–1.2167 a.u. = 9.18–33.11 eV, which is disjoint at the low end from the panel's
23.249–41.839 eV. No affine map relates them (best affine residual 2.16 eV against a 0.19 eV panel
spacing). The two blocks were therefore matched by ascending index. This mapping is *not* arbitrary
— see Finding 4, which recovers the physical relation `E = 21.699 + 13.6058·T²` — but the
*index-by-index* pairing used in the reconstruction test is an assumption of this run, not a fact
certified by the data. If the pairing is wrong, the reconstruction failure is expected and
uninformative. **This is the single largest alternative explanation for the negative result.**
*Mechanistic caution.* Nothing here identifies a physical channel composition. The degree-20
intensity is a property of the stored numbers.

**Falsification proposal.** Apply the recovered physical mapping of Finding 4 to pair the
`theory/DMEs` block with the theory dense panel energy-by-energy rather than index-by-index, and
re-run the 48-candidate search. If any selection then reaches a relative residual below 2 %, this
finding is falsified. Independently: obtain the theory magnitude at photon energies outside the
stored 99-point grid for the same calculation and test whether the degree-20 intensity still holds
— a degree-20 polynomial representation of a physical amplitude should persist if it is the
generator's actual form.

---

## Finding 3 — Three separately supplied representations obey the same phase-to-delay conversion, and the eigenphase-sum curve fixes the factor at 1, not 2

**Object and scope.** All three places in the dataset where a phase and a delay are supplied
independently: the experiment panel (`experiment/phase` vs `experiment/delay`), the theory panel
(`theory/phase` vs `theory/delay`), and the standalone curve pair (`eigenphase_sum` vs
`0.5_Wigner_delay`). Conversion constant `hbar_as_eV = 658.2119569` from
`data/coordinate_definitions.json`.

**Relationship and direction.** Every delay array in the dataset equals
`ħ · d(unwrapped phase)/d(photon energy)` evaluated as a forward energy difference and labelled with
the **upper** energy of each interval. The standalone `0.5_Wigner_delay` equals
`ħ · d(eigenphase_sum)/dE` with **factor 1**; the physically common alternative
`2ħ · d(eigenphase_sum)/dE` is excluded.

**Core quantitative result** (analysis unit: one (energy-interval, angle) cell; n = 49 775 cells
for the experiment panel, 17 738 for the theory panel, 296 points for the Wigner curve):

| Relation | median &#124;difference&#124; | p95 | max | median ratio obs/pred |
|---|---|---|---|---|
| `experiment/delay` vs `ħ·Δφ/ΔE` | **0.0504 as** | 0.456 as | 0.656 as | 0.999999 |
| `theory/delay` vs `ħ·Δφ/ΔE` (non-uniform grid) | **0.0445 as** | 0.689 as | 15.35 as | 0.999998 |
| `0.5_Wigner_delay` vs `1·ħ·dη/dE` | **1.12 as** | 2.39 as | 2.98 as | 0.987 |
| `0.5_Wigner_delay` vs `2·ħ·dη/dE` | 15.34 as | — | — | 1.97 |

The experiment relation is tight in absolute terms: 100 % of the 49 775 cells agree to better than
1 as, and the median delay is 62.3 as. For the theory panel, 97.3 % of cells agree to better than
1 as; the single worst cell (15.35 as at 37.775 eV, 132°) sits on the grid's smallest energy step.

**Supporting analysis and result evidence.**
`output/analysis/p3_phase_delay_curves.py` → `output/evidence/p3_phase_delay_curves.json`,
`output/evidence/p3_phase_delay_curves.png`, `output/evidence/p3_aligned_curves.csv` (296-row
aligned table containing both `ħ·dη/dE` variants and both factor predictions),
log `output/evidence/p3_phase_delay_curves.log`.

**Robustness.**
(i) *Null baselines* — a constant-delay null gives a median absolute difference of **51.50 as**
(factor ~1 021 larger) and an angle-permuted delay null gives **72.77 as** (factor ~1 443). The
relation is therefore not a restatement of the delay's own scale.
(ii) *Alternative derivative estimators* — a 5-point local-quadratic stencil applied at the interval
midpoints gives a median of 0.116 as, and a cubic-spline derivative gives 0.076 as; all three
estimators agree to well within 1 as. Applying the stencil at the wrong energy nodes degrades the
median to 1.38 as, confirming the alignment convention.
(iii) *Trimming and exclusion* — trimming the worst 1 % of cells leaves a median of 0.0494 as; the
explicit exclusion of 22 cells in the 43–44 eV and 48.6–49.6 eV regions leaves 0.0518 as.
(iv) *Off-by-one pairing control* — shifting the phase row by one index degrades the theory-panel
median from 0.0445 as to 4.91 as, so the alignment is specific rather than incidental.
(v) *Independent verification* — `output/analysis/p3_phase_delay_curves_verify.py` reproduced every
headline number using a hand-rolled modulo unwrap, `scipy.stats.pearsonr`, an independent PCHIP
route, and a different permutation seed. For the factor question it independently obtained F = 1 at
0.951 as and F = 2 at 32.398 as (5-point node stencil), excluding factor 2 by **34×**; it also noted
that the package's own quoted 1.12 as is stencil-dependent. Verdict at
`output/evidence/p3_phase_delay_curves_verdict.json`.

**Counterevidence, limitations, alternative explanations.**
*The relation is definitional, not empirical, for the two panels.* The dense delay grid is
documented as derived, and both panels' delays equal a forward derivative of their own phases. So
these two rows of the table establish internal consistency of the file construction; they are not
independent measurements of a physical time delay. The evidentiary weight of the panel rows is
therefore limited to *provenance confirmation* — they prove the delay arrays carry no information
beyond the phase arrays, which matters for anyone tempted to treat them as independent evidence.
*The eigenphase row is the substantive one*, because `eigenphase_sum` and `0.5_Wigner_delay` are
supplied as separate curves on different energy grids (0.5 eV vs 0.1 eV steps, 50 vs 296 points)
and the agreement at 1.12 as spans a 38× difference in absolute delay values (median |Wigner| =
7.95 as at low eKE rising to 182.6 as).
*The factor-1 result is a statement about this dataset's convention.* The dataset *names* the curve
`0.5_Wigner_delay`, and `ħ·dη/dE` is indeed half of the conventional Wigner delay
`2ħ·dη/dE`, so the name and the factor-1 result are mutually consistent. This run cannot determine
whether the underlying physics intends the half-Wigner quantity or the full one; it can only report
that the stored numbers obey factor 1.
*Stencil dependence is real.* The quoted 1.12 as depends on the derivative estimator (0.95–1.12 as
across estimators). The factor statement (1 vs 2) is robust; the precise agreement level is not.

**Falsification proposal.** Regenerate the `0.5_Wigner_delay` curve from `eigenphase_sum` using an
independent high-order spectral derivative (e.g. Chebyshev differentiation on the 50 nodes) and test
whether the median absolute difference falls below 1 as at factor 1 while factor 2 stays above 10 as.
A single measurement of a Wigner delay at any eKE in 0.5–30 eV that disagrees with
`ħ·dη/dE` by more than a few attoseconds would falsify the conversion.

---

## Finding 4 — The documented spline derivation of the dense experiment grid is not reproducible from the 13 measured blocks, and the two panels are offset by exactly half a grid step

**Object and scope.** The relationship between the 13 measured energy blocks
(`experiment/DME_<E>eV`, `CoefLeg_<E>eV`) and the 276-point dense experiment grid; the interior of
the 38.75 → 48.4 eV gap; and the mapping between the theory photon-energy axis and the
`theory/DMEs` kinetic-energy axis.

**Relationship and boundary.** The dense experiment grid cannot be regenerated from the 13 measured
blocks by any standard interpolation of the raw arrays *or* of their angular coefficients — the best
construction misses by ~1.9 × 10⁴ times the storage quantum. Only 7 of the 13 measured energies lie
on the dense mesh; the other 6 are offset by exactly ±0.05 eV. In contrast, the theory photon-energy
axis **is** exactly reproducible from the DME kinetic-energy axis by a momentum-squared relation.

**Core quantitative result.**

*Mesh offset.* Measured energies on the 0.1 eV mesh: 22.5, 24.8, 27.9, 31.0, 34.1, 37.2, 48.4 eV
(7 of 13). Off-mesh: 23.25, 26.35, 29.45, 32.55, 35.65, 38.75 eV, each offset by exactly ±0.050 eV
from the nearest mesh point. The delay axis is the identical mesh shifted by one row.

*Interpolation failure* (whole-grid residuals, magnitude array, against a storage quantum of
1 × 10⁻⁵):

| Method | max residual | × quantum |
|---|---|---|
| natural cubic spline | 0.1932 | 1.9 × 10⁴ |
| Akima | 0.2036 | 2.0 × 10⁴ |
| not-a-knot cubic spline | 0.2533 | 2.5 × 10⁴ |
| PCHIP | 0.2557 | 2.6 × 10⁴ |
| linear | 0.2685 | 2.7 × 10⁴ |
| interpolating the 4 lowest Legendre coefficients (best of 4 methods) | 0.2220 | 2.2 × 10⁴ |
| free-knot refit, 13 knots | 0.1740 | 1.7 × 10⁴ |

The gap between interior accuracy and gap-region accuracy is diagnostic: natural-spline residuals
are 0.0213 **interior** but 0.1932 **inside the 38.75–48.4 eV gap** — a factor of 9 worse precisely
where measurement support is absent.

*The gap is not a chord.* Deviations of the dense grid from the straight line joining the 38.75 and
48.4 eV endpoints reach **0.268** (magnitude) and **1.381 rad** (phase), i.e. 2.7 × 10⁴ and
1.4 × 10⁵ times the respective storage quanta. The gap region is a smooth continuation with real
structure (a magnitude minimum near 43.1–46.0 eV depending on angle), not an interpolation of the
two endpoints.

*Theory energy mapping.* The identity
`E_eV = round(21.699 + 13.6058 · T_kin², 3)`
holds for **99 of 99** points (zero mismatches); the unrounded formula deviates by at most
8.64 × 10⁻⁶ eV. A free two-parameter fit returns `a = 27.2116 eV/a.u.`, `b = 21.6990 eV` with max
residual 1.71 × 10⁻⁶ eV. Linear alternatives are decisively excluded: `E = IE + T_kin·27.2114`
leaves 13.61 eV, `T_kin = (E − IE)/27.2114` leaves 0.500 a.u. The panel thus maps to the DME block
through the **kinetic energy** `T_kin²/2 · 27.2114`, and `21.699 eV` is the stored ionisation energy
`IE·c = 21.69982 eV`.

*Grid overlap.* 11 of the 13 measured energies have a theory photon energy within 1 meV — all at
exactly −1.000 meV, which is the theory grid's own rounding offset. 22.5 eV has no counterpart
(nearest 23.249 eV, 0.749 eV away); 48.4 eV has none (nearest 41.839 eV, 6.561 eV away).

**Supporting analysis and result evidence.**
`output/analysis/p4_grid_provenance.py` → `output/evidence/p4_grid_provenance.json`,
`output/evidence/p4_grid_provenance.png`, log `output/evidence/p4_grid_provenance.log`.
Mesh-offset and quantization detail additionally in `output/evidence/p1_exp_basis.json`
(script `output/analysis/p1_exp_basis.py`).

**Robustness.**
(i) *Extended method search* — the verifier added Akima interpolation and a hand-rolled PCHIP on top
of the package's set and found nothing below 0.19, so the failure is not a method-selection artefact
(`output/evidence/p4_grid_provenance_verdict.json`).
(ii) *Knot-density sweep* — a free-knot refit improves monotonically with knot count (13 knots:
0.0297 interior; 26: 0.00216; 52: 0.000266; 104: 8.80 × 10⁻⁵). Reaching the storage quantum
therefore requires ~100 free knots rather than 13, i.e. the grid carries structure finer than the
13 measured energies can supply.
(iii) *Leave-one-out over the 13 nodes* — natural spline worst-case residual 8.95, PCHIP 10.23,
linear 0.964: the reconstruction is not merely inaccurate, it is unstable to dropping a single
measured block.
(iv) *Three independent verifications* — the mapping identity was confirmed 99/99 by the verifier;
the mesh offset was re-derived with an independent grid test that also confirmed the delay axis
carries the same 7/6 split; the coefficient-interpolation route was rebuilt from the verifier's own
Legendre recurrence and its own splines and reproduced 0.22198.
(v) *Harsher quantisation baseline* — the verifier notes the 13 measured magnitude rows are exact
multiples of 1 × 10⁻⁶ (not 1 × 10⁻⁵), which makes the failure ratio ~1.9 × 10⁵ rather than
1.9 × 10⁴. The conclusion is unchanged and the reported figure is the conservative one.

**Counterevidence, limitations, alternative explanations.**
*"Not reproducible by standard interpolation" is not the same as "irreproducible".* A generator using
a global fit, a physics-based model, or splines in a transformed variable could produce this grid
from the 13 blocks and would not be caught by the method set tested here. The defensible claim is the
bounded one: *no standard interpolation of the raw arrays or their angular coefficients reproduces
the grid, and reproducing it requires substantially more freedom (~100 knots) than the 13 measured
blocks provide.* A stronger negative would require the generator's actual recipe.
*The ±0.05 eV offsets have a benign explanation.* Six of the thirteen measured energies sit exactly
half a grid step off the mesh. This is consistent with a generator that placed its output mesh on
0.1 eV round numbers rather than on the measured energies, and then resampled. The observation is
therefore evidence about mesh *construction*, and should not be read as an inconsistency in the
measurement.
*The `2 %` / `1 × 10⁻⁵` thresholds are conventions.* They are stated explicitly above so a reader can
re-threshold. No conclusion here depends on a threshold being barely crossed: every reported failure
misses by 3–5 orders of magnitude.
*The mapping formula is a recovered empirical relation*, not a certified constant. `13.6058` is the
Rydberg energy in eV to five figures and `21.699` matches the stored `IE`, which is physically
coherent (kinetic energy from a photon above threshold), but this run verified the relation, not its
provenance.

**Falsification proposal.** Obtain the generator's actual resampling recipe, or any additional
measured energy block between 38.75 and 48.4 eV, and test whether the dense grid in 38.9–48.3 eV is
reproduced by interpolation through it. A single new measurement inside the gap that lies on the
existing dense curve would strongly constrain the generator; one that does not lie on it would show
the dense curve is a model output rather than an interpolation, and would falsify the
"interpolation of the 13 blocks" reading of the data guide outright.

---

## Finding 5 — The experiment–theory panel agreement is not energy-specific, and the reported cross-representation conflict is an artefact of summation order; an order-wise magnitude correspondence survives

**Object and scope.** (a) The comparison between `experiment/phase`, `experiment/magnitude` and
`theory/phase`, `theory/magnitude` at the 11 energies where the two grids coincide to 1 meV;
(b) the comparison between the measured `CoefLeg_<E>eV` angular-coefficient blocks and the dense
experiment panel across all 13 measured energies.

This finding is registered because it **refutes** two affirmative readings that would otherwise be
natural, and replaces them with a narrower supported statement.

**Relationship and boundary.**

*Part (a) — an affirmative-looking result that is not energy-specific.* At matched energies, the
unwrapped experiment and theory phases correlate at r = 0.9986 (23.25 eV), 0.9987 (24.80 eV),
0.9888 (26.35 eV), with correlations of 0.44–0.94 at higher energies. Taken alone this looks like
strong cross-panel agreement. It is not: comparing the **same** experiment row against **all 99**
theory energies shows the matched pair beats at least 95 % of mismatched theory energies in only
**1 of 11** cases for phase and **0 of 11** for magnitude. The median mismatched correlation is
0.71–0.90, i.e. nearly every theory energy reproduces nearly every experiment row's angular shape.
The apparent agreement is a shared angular baseline, not energy-matched physics.

*Part (b) — a reported conflict that is a construction artefact.* `output/evidence/p7_anisotropy.json`
reports that the summed ratio `Σ F20 / Σ F00` across all nine order rows anti-correlates with the
dense-grid anisotropy `β(E)`: Pearson −0.697 (p = 0.0081), Spearman −0.720 (p = 0.0055). This
survives the package's own controls and its verifier reproduced the numbers exactly
(`output/evidence/p7_anisotropy_verdict.json`). But the sign is set by the summation truncation:

| truncation `n` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Pearson r vs dense β | −0.601 | **−0.878** | +0.610 | −0.025 | **+0.720** | −0.131 | −0.755 | −0.629 | −0.696 |

The correlation ranges from −0.88 to +0.72 across defensible truncations. The all-order value of
−0.696 is one draw from that set, so the "conflict" is a property of the summation convention rather
than of the physics. The package's verifier reached the same conclusion independently, reporting that
"the sign is an artefact of the 'literal summed ratio' convention".

*What survives.* The comparison must be made order by order, and against the **intensity** expansion
(degree 6, Finding 1), not the complex amplitude (whose `k ≥ 4` coefficients are round-off). There,
the order-wise correspondence is real:

| dense basis | mean &#124;diagonal r&#124; (k = 0…6) | mean &#124;off-diagonal r&#124; |
|---|---|---|
| intensity `\|Z\|²` (degree 6) | **0.808** | 0.412 |
| real magnitude | 0.609 | 0.390 |
| complex `Z` | 0.328 | 0.330 |

with per-order values `[+0.703, −0.941, +0.864, −0.897, +0.694, −0.649, +0.910]`. A permutation test
on the **energy labelling** (50 000 draws, shuffling which `CoefLeg` block pairs with which dense
energy) gives **p < 2 × 10⁻⁵**, and a scan of the diagonal shift shows the unshifted alignment is
optimal (mean |r| = 0.869 at shift 0, versus 0.715 at the best alternative shift ±2).

**Core quantitative result, stated cleanly.**

| Statement | Status | Key number |
|---|---|---|
| Matched-energy phase correlation is high | true but uninformative | r = 0.999 at 23.25 eV |
| Matched pair is energy-specific | **refuted** | 1/11 phase, 0/11 magnitude beat the 95th percentile |
| `Σ F20/Σ F00` conflicts with dense β | **artefact** | sign flips across truncation, −0.88 to +0.72 |
| Order-wise `\|c_k\|` ↔ `F20(k)` correspondence exists | **supported** | mean |r| 0.808 vs 0.412 off-diagonal, p < 2 × 10⁻⁵ |
| The correspondence is proportional | **refuted** | best single-scale variance explained ≤ 0.31 |

**Supporting analysis and result evidence.**
`output/analysis/d02_cross_representation.py` → `output/evidence/d02_cross_representation.json`,
`output/evidence/d02_cross_representation.png`, `output/evidence/d02_cross_representation.csv`;
`output/evidence/d02_energy_scramble_null.json` and `output/evidence/d02_order_alignment.json` hold
the underlying null distributions; `output/evidence/p7_anisotropy.json` (script
`output/analysis/p7_anisotropy.py`) holds the original reported conflict; the independent verdict is
`output/evidence/p7_anisotropy_verdict.json` (script
`output/analysis/p7_anisotropy_verify.py`); the panel comparison is
`output/evidence/p6_exptheory.json` (script `output/analysis/p6_exptheory.py`) with
`output/evidence/p6_comparison_table.csv`, and the independent verdict
`output/evidence/p6_exptheory_verdict.json`.

**Robustness.**
(i) *Multiple mismatch statistics* — the scramble null was evaluated with median, 95th percentile,
rank fraction, and count-of-exceedances; all agree.
(ii) *Truncation sweep* — nine truncation orders were tested for part (b), not just the two that
bracket the reported value.
(iii) *Two null constructions* — a permutation test on the energy labelling and a grid-maximum
control that accounts for selecting the largest of 63 correlations. The grid-max null has median
0.683 and 95th percentile 0.796; the observed maximum is 0.941, p = 0.00033. Both controls are
required because the naive diagonal mean alone would be inflated by grid selection.
(iv) *Independent verification* — the p6 verifier rebuilt the comparison with a `csv`-module loader,
a hand-rolled modulo unwrap, and three interpolation routes, and **refuted one claim as literally
stated**: the matched count is 11 of 13, not 12 of 13, with 22.5 eV also unmatched
(`output/evidence/p6_exptheory_verdict.json`). This report uses the corrected 11.

**Counterevidence, limitations, alternative explanations.**
*Part (a) is not evidence that the panels disagree.* The correct reading is *indeterminate*, not
negative: the two panels share an angular baseline so dominant that the energy-label test has almost
no discriminating power. A weak genuine energy dependence could be present and would be invisible
here. The honest statement is "this comparison cannot establish energy-matched agreement", not "the
panels disagree at matched energy".
*The 22.5 eV and 48.4 eV energies are excluded from part (a)* because the theory grid does not reach
them. The test therefore covers 11 of the 13 measured energies, and says nothing
about the two extremes. Within those 11, the covered span is 23.25–38.75 eV of the experiment grid's
22.5–50.0 eV range, against a theory grid that spans 23.249–41.839 eV.
*Part (b)'s surviving correspondence is a modulus correspondence.* The diagonal signs alternate
(`−0.941` at k = 1, `−0.897` at k = 3), so what tracks is `|c_k|` against `F20(k)`, not
`c_k ∝ F20(k)`. The proportionality test fails explicitly: fitting `F20(E,k) = s_k · c_k(E)` with one
complex scale per order explains at most 31 % of variance (k = 1) and is negative for k = 0 and k = 2,
against a random-pairing control whose median is −0.81. So the correspondence is real but its form is
not identified.
*An alternative explanation for the correspondence* is that both quantities are monotone functions of
a common energy trend (the anisotropy rises with energy in both representations — Finding 6), and
that a monotone trend induces correlation between many pairs of series. The permutation test controls
for the **energy labelling** but not for shared monotonicity. This run does not rule that out, and
it is the most plausible benign explanation.
*Effective sample size.* Both parts are computed over 13 (or 99) energy blocks, not over the
276 × 181 grid cells. Reported p-values already reflect the block count, but the dense experiment
values are spline-derived, so even the 13 blocks are not 13 independent measurements of the same
quantity across the gap.

**Falsification proposal.** For part (a): repeat the scramble null using a quantity that is *not*
dominated by a common angular baseline — for example, the residual after removing each row's own
lowest-order expansion — and test whether the matched pair then becomes energy-specific. For part
(b): obtain more measured energy blocks so that the order-wise correlation is estimated on a larger
sample than 13, and test whether the mean diagonal |r| remains above the off-diagonal baseline once
the shared monotone energy trend is regressed out of both series. If the diagonal advantage
disappears under detrending, the correspondence is a trend artefact and this finding should be
reduced to the two refutations alone.

---

## 7. Failed, conflicting, and unresolved results

These are reported because they materially bear on the registered findings.

**7.1 `angle_averaged_delay` is not the angular average of the experiment panel delay — unresolved
link.** `output/evidence/p3_phase_delay_curves.json` §e scanned a candidate ionisation-energy offset
from 12.0 to 22.5 eV (step 0.01 eV) to map the curve's electron kinetic energy onto the panel's
photon energy. The best offset (12.06 eV) gives an RMS of **48.22 as** with a **negative**
correlation of −0.475, which is *worse than the constant-residual null* (21.83 as). The curve cannot
be aligned to the panel angular average at any physical offset. For contrast, the positive controls
in the same test align cleanly. This is a genuine negative result: either the curve corresponds to a
different physical quantity, or the mapping between its energy axis and the panel's is not a constant
offset. **Unresolved.**
*Verifier note:* an extrapolated scan to IE ∈ [0, 25] eV reached a best RMS of 35.44 as at
IE = 6.91 eV, still non-physical and still not competitive with a constant
(`output/evidence/p3_phase_delay_curves_verdict.json`).

**7.2 `0.5_Wigner_delay` and `angle_averaged_delay` agree in shape but not in scale.** Correlation
0.923 (Spearman 0.956) on the shared 200-point grid, but median absolute difference 26.2 as and a
maximum gap of 85.0 as at eKE 0.90 eV, with the divergence concentrated below ~5 eV (band
correlations: 0.745 for 0.9–5 eV, 0.972 for 5–10 eV, 0.998 for 10–20 eV, 0.216 for 20–27.9 eV). The
two curves are related but not the same quantity. Reported in
`output/evidence/p3_phase_delay_curves.json` §f and reproduced by the verifier.

**7.3 The `CoefLeg` reconstruction of the dense panel depends on harmonic convention.**
`output/evidence/p5_coefleg.json` reports that reading B (both `F00` and `F20` multiplying `Y_k0`)
reaches a mean affine R² of 0.897–0.931 across 13 energies, while reading A (distinct harmonic
functions) reaches only 0.245–0.255. The verifier confirmed both readings and added that reading B's
range across all eight Condon–Shortley / normalisation / `√2` combinations is **0.778–0.931**, so the
strong reconstruction requires the most favourable convention
(`output/evidence/p5_coefleg_verdict.json`). Reading A stays 0.13–0.25 under every convention, so the
A-versus-B dichotomy itself is robust. Also unresolved: the `CoefLeg` `F20` family contains sign
inversions relative to the dense coefficients (seven of thirteen energies), which this run could not
attribute to a convention choice.

**7.4 The `F` families are not predictable from the measured dipole-matrix elements.**
`output/evidence/p5_coefleg.json` §c built a 105-feature dictionary of all pairwise products of the
14 measured DME entries per energy. In-sample R² on 13 energies saturates at 1.0000 with design rank
13 — a **saturated fit with zero degrees of freedom**, which is why the honest test is leave-one-out.
Greedy LOO R² is 0.270 (one feature) and 0.344 (two) for `F00`, and **negative** for `F20` (−0.282),
`F21`, `F22`, `F11`. Permutation p-values for the best single feature are 0.81–0.99 against a
95 % null maximum |r| of 0.20–0.28. The verifier reproduced this and confirmed the saturating rank.
**Negative result: no recoverable DME → angular-coefficient map at this sample size.** The `n = 13`
design cannot support a 105-feature search, so this is as much a power limitation as a physical null.

**7.5 The theory `DMEs` block does not reconstruct the theory dense panel** — see Finding 2. Reported
here as well because it is the strongest negative in the run and its principal alternative
explanation (index-versus-physical energy pairing) is not excluded by this run.

**7.6 A verifier refuted its package's matched-energy count.** `output/evidence/p6_exptheory_verdict.json`
found 11 of 13 measured energies matched within 1 meV, not 12; 22.5 eV, not just 48.4 eV, is
unmatched. This report uses 11 throughout. The package's own `summary` field had flagged this
(`brief_expected_matched: 12`), and the discrepancy is a labelling error in the package, not a
numerical one.

**7.7 Execution failures actually encountered and retained.**
* `output/analysis/p3_phase_delay_curves.py` raised
  `ValueError: operands could not be broadcast together with shapes (273,181) (272,181)` in its
  5-point-stencil section; the real traceback is preserved in `output/evidence/p3_phase_delay_curves.json`
  under `failures`. Every other derivative variant in that package ran, and the verifier re-ran the
  stencil independently, so the section's conclusion was recovered by an alternative route.
* `output/analysis/d02_order_alignment.py` initially produced a cross-representation result that
  **contradicted** the coordinating agent's own exploratory run. Investigating the discrepancy
  (rather than picking the favourable number) showed the two runs used different dense bases — the
  complex amplitude versus the real magnitude/intensity. The resolving analysis is
  `output/analysis/d02_cross_representation.py`; the superseded numbers remain on disk in
  `output/evidence/d02_order_alignment.json` with the corrected treatment documented in the same
  file. This is the origin of the "intensity, not amplitude" clarification in Finding 5.
* The verification workflow's sub-agents twice returned no structured output, first in the analysis
  stage and then in the verification stage. Both times the on-disk artifacts were intact; the
  structured returns were lost. Verdicts were recovered from disk, and the one package whose verifier
  never produced a verdict (`p1_exp_basis`, sub-agent API error) was verified directly by
  `output/analysis/p1_exp_basis_verify.py`.

---

## 8. Limitations of this run

1. **Derived-grid inflation.** Every experiment-side correlation, p-value, and permutation statistic
   is computed over a spline-derived grid. Findings 1, 4, and 6 are subject to this; Finding 2 and
   the standalone-curve parts of Finding 3 are not (the theory grid is a calculation output, and the
   curve datasets are supplied directly). Effective experiment-side sample size is ~13 energy blocks.
2. **No external evidence was used.** No literature, database, or external dataset was consulted, so
   no statement here is anchored to prior measurements. The T0 boundary (`2021-01-01`) was therefore
   not exercised; no external source appears in this report.
3. **Thresholds are conventions.** The reproduction thresholds (2 % relative residual for DME
   reconstruction, 1 × 10⁻⁵ / 1 × 10⁻⁶ quantum for grid reproduction) are stated explicitly. No
   finding depends on a borderline crossing.
4. **The physical interpretation of the recovered quantities is not established.** This run can
   report that the stored arrays have exact degree 3 / degree 6 / degree 20 angular representations
   and that certain conversion relations hold to stated accuracy. It cannot report what microscopic
   channel structure produced them, and it does not attempt to.
5. **Finding 5's surviving correspondence has an unexcluded benign explanation** (shared monotone
   energy trend), as stated in §6.
6. **The 38.75–48.4 eV gap is underdetermined.** Six of thirteen measured energies lie inside or
   adjacent to a 9.65 eV span with no measurement. Any statement about the dense grid in that region
   inherits that weakness.

---

## 9. Falsification and independent replication proposals

**For Finding 1.** Regenerate the dense grid from the 13 measured DME blocks and test whether
`|c_4|/|c_0| < 10⁻⁵` holds at every energy. Any energy requiring a non-negligible `c_4` falsifies the
degree-3 statement. Conversely, an independent angular measurement at one of the 13 energies with
residuals below 10⁻⁵ in `Z` validates it.

**For Finding 2.** Re-run the 48-candidate DME search using the physical mapping
`E = 21.699 + 13.6058·T²` instead of index pairing. Any selection reaching a sub-2 % relative
residual falsifies the negative result. Secondly, obtain the theory magnitude on photon energies
outside the stored 99-point grid: if the degree-20 intensity does not persist, the degree-20
structure is a grid coincidence rather than the generator's form.

**For Finding 3.** Recompute `ħ·dη/dE` from `eigenphase_sum` with an independent spectral derivative
on the 50 nodes. Factor 1 must stay below ~1 as and factor 2 above ~10 as. A measured Wigner delay
anywhere in 0.5–30 eV differing from `ħ·dη/dE` by more than a few attoseconds falsifies the
conversion.

**For Finding 4.** Supply any additional measured energy block inside 38.75–48.4 eV. If it lies on
the existing dense curve, standard interpolation through the enlarged knot set should then succeed,
and the finding is reduced to a statement about knot density. If it does not lie on the curve, the
grid is a model output and the "interpolation of 13 blocks" reading of `DATA_GUIDE.md` is falsified
outright.

**For Finding 5.** (a) Repeat the energy-scramble null on a baseline-removed quantity; if the matched
pair becomes energy-specific, the refutation is overturned. (b) Estimate the order-wise correlation
on more than 13 measured blocks and repeat after jointly detrending both series in energy; if the
diagonal advantage over off-diagonal vanishes, the surviving correspondence is a trend artefact and
should be withdrawn.

---

## 10. Finding-to-evidence index

| Finding | Analysis source | Result evidence |
|---|---|---|
| 1 — exact degree 3 / degree 6, experiment panel | `output/analysis/p1_exp_basis.py`, `output/analysis/p1_exp_basis_verify.py`, `output/analysis/d02_cross_representation.py` | `output/evidence/p1_exp_basis.json`, `output/evidence/p1_exp_basis_coefficients.csv`, `output/evidence/p1_exp_basis.png`, `output/evidence/p1_exp_basis_verdict.json`, `output/evidence/d02_intensity_degree.json` |
| 2 — theory panel degree-20 intensity, DME non-reconstruction | `output/analysis/p2_theory_dme.py`, `output/analysis/d02_cross_representation.py` | `output/evidence/p2_theory_dme.json`, `output/evidence/p2_theory_coefficients.csv`, `output/evidence/p2_theory_dme.png`, `output/evidence/d02_theory_intensity_degree.json`, `output/evidence/p2_theory_dme_verdict.json` |
| 3 — phase-to-delay and eigenphase-to-Wigner conversion | `output/analysis/p3_phase_delay_curves.py`, `output/analysis/p3_phase_delay_curves_verify.py` | `output/evidence/p3_phase_delay_curves.json`, `output/evidence/p3_aligned_curves.csv`, `output/evidence/p3_phase_delay_curves.png`, `output/evidence/p3_phase_delay_curves_verdict.json` |
| 4 — grid provenance, mesh offset, theory mapping | `output/analysis/p4_grid_provenance.py`, `output/analysis/p4_grid_provenance_verify.py`, `output/analysis/p1_exp_basis.py` | `output/evidence/p4_grid_provenance.json`, `output/evidence/p4_grid_provenance.png`, `output/evidence/p4_grid_provenance_verdict.json`, `output/evidence/p1_exp_basis.json`, `output/evidence/d01_inventory.json` |
| 5 — energy-specificity null and summation-order artefact | `output/analysis/d02_cross_representation.py`, `output/analysis/d02_order_alignment.py`, `output/analysis/p6_exptheory.py`, `output/analysis/p7_anisotropy.py`, `output/analysis/p6_exptheory_verify.py`, `output/analysis/p7_anisotropy_verify.py` | `output/evidence/d02_cross_representation.json`, `output/evidence/d02_cross_representation.csv`, `output/evidence/d02_energy_scramble_null.json`, `output/evidence/d02_order_alignment.json`, `output/evidence/p6_exptheory.json`, `output/evidence/p6_comparison_table.csv`, `output/evidence/p7_anisotropy.json`, `output/evidence/p7_anisotropy.csv`, `output/evidence/p6_exptheory_verdict.json`, `output/evidence/p7_anisotropy_verdict.json` |
| §5 `CoefLeg` reconstruction (supporting) | `output/analysis/p5_coefleg.py`, `output/analysis/p5_coefleg_verify.py` | `output/evidence/p5_coefleg.json`, `output/evidence/p5_coefleg_summary.csv`, `output/evidence/p5_coefleg.png`, `output/evidence/p5_coefleg_verdict.json` |
| Inventory and integrity (all findings) | `output/analysis/d01_inventory.py` | `output/evidence/d00_data_integrity.json`, `output/evidence/d01_inventory.json`, `output/evidence/d01_inventory.png` |

Registered findings: **5** (Findings 1–5). Reproduction instructions: `output/reproduce.md`.
All commands, outputs, and errors quoted in this report were produced by the scripts listed above
during this run; nothing here is a projected or simulated result.