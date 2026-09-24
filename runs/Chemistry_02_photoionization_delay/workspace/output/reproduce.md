# Reproducing this run

Task: `Chemistry_02_photoionization_delay` — Energy- and Angle-Dependent Molecular Photoionization
Dynamics. Authoritative report: `output/Result.md`.

## Environment

Python 3.11 with `numpy`, `pandas`, `scipy`, `matplotlib`. No other scientific packages are
required. (`h5py`, `sklearn`, `statsmodels`, `pyarrow` are absent from this environment and are not
used.) `scipy` and `matplotlib` were installed during the run from the configured package index;
the two earlier inventory scripts (`d00`, `d01`) run with `numpy`/`pandas` only.

## Inputs

Everything the analysis reads is under `data/` in the workspace root, which was treated as immutable:

| File | SHA256 (verified against `data_manifest.json` in `output/evidence/d00_data_integrity.json`) |
|---|---|
| `data/angle_energy_observations_datasets.tsv` | `55a1d9621c0e386fb44a63764a97cf81491d2bd7af0209c74e212329172df396` |
| `data/angular_coefficients_datasets.tsv` | `a47e1e06130ec2a0dc8e3a4b2b45233828b37e6ff31101dbefebfd8b9d0ac411` |
| `data/coordinate_definitions.json` | `349bbd36f9de8dd1805b280c0f08270e3d1c75992efd76aa8effb603adbec23b` |
| `data/dipole_matrix_elements_datasets.tsv` | `df51186c5de242e7b6d7aa51b30f2b75411bbea139507a3061273c23f2c01762` |
| `data/phase_and_delay_curves_datasets.tsv` | `10b077ae335ce79515d03341dbf924a4143b78c310676c89dc8e56af5d6c044c` |

No external data, literature, or precomputed result was used. Each analysis script hardcodes the
workspace root as `ROOT` at the top of the file.

## How to reproduce

Every script is self-contained and re-runnable from a clean state. It reads only `data/`, writes to
`output/evidence/`, and prints its full log to stdout. Run from the workspace root:

```bash
cd <workspace root>

# inventory and data integrity (no scipy needed)
python3 output/analysis/d01_inventory.py   > output/evidence/d01_inventory.log

# the seven analysis packages
python3 output/analysis/p1_exp_basis.py            > output/evidence/p1_exp_basis.log
python3 output/analysis/p2_theory_dme.py           > output/evidence/p2_theory_dme.log
python3 output/analysis/p3_phase_delay_curves.py   > output/evidence/p3_phase_delay_curves.log
python3 output/analysis/p4_grid_provenance.py      > output/evidence/p4_grid_provenance.log
python3 output/analysis/p5_coefleg.py              > output/evidence/p5_coefleg.log
python3 output/analysis/p6_exptheory.py            > output/evidence/p6_exptheory.log
python3 output/analysis/p7_anisotropy.py           > output/evidence/p7_anisotropy.log

# cross-cutting nulls and the cross-representation re-analysis
python3 output/analysis/d02_order_alignment.py     > output/evidence/d02_order_alignment.log
python3 output/analysis/d02_cross_representation.py > output/evidence/d02_cross_representation.log

# independent adversarial verification (one per package)
for p in p2_theory_dme p3_phase_delay_curves p4_grid_provenance \
         p5_coefleg p6_exptheory p7_anisotropy; do
  python3 output/analysis/${p}_verify.py > output/evidence/${p}_verify.log
done
python3 output/analysis/p1_exp_basis_verify.py     > output/evidence/p1_exp_basis_verify.log
```

Each run overwrites its own JSON, log, and figure in `output/evidence/`. Nothing in the pipeline
depends on a previously generated artifact other than the `data/` inputs, so any script can be run
individually.

## Method summary, in the order the report presents it

1. **Integrity and inventory.** SHA256 of all five scientific files against `data_manifest.json`;
   per-group shape, axis grids, units, finite-value counts, and the declared analysis unit.
2. **Angular basis structure.** For each energy row of the dense experiment and theory panels, a
   complex least-squares fit of `Z = magnitude·exp(i·phase)` and of `I = magnitude²` onto real
   Legendre polynomials `P_l(cos θ)`, scanning `L_max` from 0 to the Nyquist limit, with the
   residual collapse point taken as the angular degree. Cross-checked in a complex Fourier basis and
   in Chebyshev polynomials.
3. **Phase-to-delay conversion.** Forward energy differences of the phase panels multiplied by
   `hbar_as_eV = 658.2119569`, compared against every delay array; local-quadratic and cubic-spline
   derivative variants; constant-delay and angle-permuted nulls; an off-by-one pairing control.
4. **Standalone curves.** `ħ·d(eigenphase_sum)/dE` at factor 1 and factor 2 against
   `0.5_Wigner_delay`; an ionisation-energy offset scan (12.0–22.5 eV, step 0.01 eV) to test whether
   `angle_averaged_delay` is the angular average of the experiment delay panel.
5. **Grid provenance.** Which measured energies lie on the 0.1 eV mesh; whether the dense grid is
   reproducible from the 13 measured rows by natural/not-a-knot cubic splines, PCHIP, Akima, or
   linear interpolation, applied both to the raw arrays and to their Legendre coefficients; a
   knot-density sweep; the interior of the 38.75–48.4 eV gap; and the mapping between the theory
   photon-energy axis and the `theory/DMEs` kinetic-energy axis.
6. **Cross-representation comparison.** Order-wise correlation between the dense expansion
   coefficients and the `CoefLeg` `F` families, with a permutation test on the energy labelling and
   a grid-maximum null; a truncation sweep of the summed `ΣF20/ΣF00` ratio; a proportionality test
   against random and shift-by-one controls.
7. **Panel comparison and its null.** Experiment versus theory at the 11 energies where the grids
   coincide, including a scramble null in which each experiment row is compared against all 99
   theory energies and the matched pair is ranked inside that distribution.
8. **Anisotropy trend.** `magnitude = A(1 + β·P₂(cos θ))` at every dense energy and the full
   expansion `β(E)`, with permutation nulls, leave-one-out influence, and a split-sample check.

## Verification design

Each package's headline claims were re-derived by a separate verifier that wrote an independent
script using a deliberately different route — different library routine, different
parameterisation, different subsetting, or a harsher version of the same test. Verdicts are in
`output/evidence/*_verdict.json` and their logs in `output/evidence/*_verify.log`. Verifiers were
instructed to prefer refutation over confirmation when the evidence was unclear. Two claims were
refuted or re-scoped as a result, and both are reported as such in `Result.md`:

* `output/evidence/p6_exptheory_verdict.json` refuted the matched-energy count (11 of 13, not 12),
  and `Result.md` uses 11.
* `output/evidence/p7_anisotropy_verdict.json` showed that the reported `ΣF20/ΣF00` anti-correlation
  is an artefact of the summation convention; `Result.md` reports it as such and replaces it with
  the order-wise comparison.

## Known execution issues, retained

* `output/analysis/p3_phase_delay_curves.py` raised a broadcast `ValueError` in its 5-point-stencil
  section. The real traceback is preserved under `failures` in
  `output/evidence/p3_phase_delay_curves.json`; the other derivative variants in that package ran,
  and the verifier covered the stencil independently.
* `output/analysis/d02_order_alignment.py` initially produced a result that contradicted the
  coordinating agent's own exploratory run. The discrepancy was traced to the choice of dense basis
  (complex amplitude versus real magnitude/intensity). `output/analysis/d02_cross_representation.py`
  resolves it; the superseded numbers remain in `output/evidence/d02_order_alignment.json` and are
  documented in `Result.md` §7.7.
* The multi-agent verification workflow twice lost its sub-agents' structured return values while
  leaving their on-disk artifacts intact. Verdicts were recovered from disk, and the one package
  whose verifier raised an API error (`p1_exp_basis_verify.py`) was executed directly by the
  coordinating agent. Its script and verdict are on disk like the others.

## Semantic validation

```bash
python3 validate_output.py output
```

The validator checks delivery structure and finding-to-evidence traceability only; it is not
scientific evidence. Its receipt is printed to stdout and is not stored under `output/`.