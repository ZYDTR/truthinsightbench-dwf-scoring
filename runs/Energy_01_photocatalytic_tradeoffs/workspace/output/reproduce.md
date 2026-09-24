# Reproduction guide

Task: `Energy_01_photocatalytic_tradeoffs` (T0 = 2022-01-01).

Everything reported in `Result.md` is produced from the four frozen files in `data/` by the
scripts in `analysis/`, plus the independently executed verification and
alternative-explanation battery in `verification/`. No external data were downloaded,
substituted, or merged.

## 1. Environment

Python 3.11 with `numpy`, `pandas`, `scipy` and `matplotlib`. In this workspace `scipy` and
`matplotlib` were absent at session start and were installed with:

```
python3 -m pip install --user scipy matplotlib
```

The install is part of the executed record, not an assumption: the first two attempts timed
out at 120 s and 400 s respectively against the configured package mirror, and only a
background run with a 590 s budget completed (`scipy 1.17.1`, `matplotlib 3.11.2`).

## 2. Data integrity (run first)

`analysis/stage1_inventory_timecourse.py` re-hashes all four files and refuses to proceed on a
mismatch. All four sha256 values agree with `data_manifest.json`. An independent audit of the
same files (including grid regularity, rounding structure and non-tautology checks) is in
`verification/` (artifact `V5`).

## 3. Pipeline order

Run from inside `output/analysis/` — the scripts import `common.py` from their own directory
and resolve `data/` and `output/` relative to the script location, so the working directory
does not otherwise matter.

```
python3 stage1_inventory_timecourse.py    # inventory, integrity, kinetic model bank, fig1
python3 stage2_spectra.py                 # action-vs-absorption models, window holdout, fig2
python3 stage3_transients.py              # rise/decay models, windowed stats, contrast, fig3
python3 stage4_loading_crossmodal.py      # loading models, cross-modal coverage, fig4+fig5
python3 stage5_robustness.py              # R1/R2/R3 robustness battery
python3 stage6_consolidate.py             # evidence_table.csv + registration_summary.json + registry
```

Each script writes a plain-text transcript to `logs/` and machine-readable JSON to `results/`.

After the audit stage, run:

```
python3 stage7_audit_fixes.py             # persists two audit-requested checks
```

All randomness (bootstraps, permutation tests) is seeded with the literal value `20220101` in
`common.py` and the stage scripts. Only `results/transient_analysis.json` embeds the seed in its
own JSON (14 `seed` fields); the other seeded artifacts are reproducible from the source but do not
carry the value in the output file.

`stage2`–`stage4` each take well under a minute. `stage3` and `stage5` contain the only
non-trivial runtimes; both were tuned down after a first version was measured as too slow
(a Python loop over 20 000 `curve_fit` calls did not finish in 600 s and was replaced by
`scipy.optimize.least_squares` with 5 000 resamples, which completes in about 3 minutes).
Those first failures are preserved in `logs/` rather than hidden.

## 4. Independent verification and alternatives (subagent workflow)

```
Workflow({ scriptPath: "output/analysis/workflow_verify_alternatives.js", args: {...} })
```

This native workflow fan-out is itself part of the executed evidence trail. It runs five
independent recomputation agents, four alternative-explanation agents and one completeness
critic. Each writes its own script and JSON into `verification/`. The agents were explicitly
forbidden from reading `output/results/`, `output/analysis/` or the report, so their numbers
are independent of the pipeline above; they were given my reported values only to check
against.

## 5. Validation

```
python3 validate_output.py output
```

Validates the five semantic delivery roles and the finding-to-evidence traceability. It does
not score the science.

## 6. What is deliberately *not* here

- No formulation identity, catalyst name, or mechanism assignment is claimed anywhere. The
  dataset is identity-neutral and `data/` contains no formulation-level labels linking the
  optical spectra or the loading scan to the named formulations.
- No literature was searched or cited for the core claims. Every quantitative statement in
  `Result.md` traces to a file under `results/`, `figures/` or `verification/` produced in
  this run.
- Where the data cannot separate explanations (the loading-scan shape, the kinetic-shape
  family, the cross-modal question) the report says so rather than choosing a winner.