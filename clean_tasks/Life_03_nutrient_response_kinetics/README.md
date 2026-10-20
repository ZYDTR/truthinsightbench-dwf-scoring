# Life_03: clean task input

This package contains the complete **participant-visible task** for *Nutrient-Response Kinetics Across System States and Tissues*, TruthInsightBench V1.0. The original task's seven files are preserved byte-for-byte. Three public runner files and a generated workspace receipt make it ready to copy into a fresh participant environment. No scientific run or new grading was performed for this publication.

“Complete” refers to the benchmark's declared task boundary. It does not claim that every variable or file in the underlying scientific source is included.

## Contents

```text
workspace/
├── ResearchObjective.md
├── DATA_GUIDE.md
├── data_manifest.json
├── data/
│   ├── nutrient_response_timecourse.tsv
│   ├── circulating_input_measurement.tsv
│   ├── tissue_uptake_measurement.tsv
│   └── baseline_system_phenotype.tsv
├── PublicTaskContract.md
├── LaunchPrompt.md
├── validate_output.py
└── workspace_receipt.json
```

The four TSV files contain 144 data rows and 7,685 bytes in total. `data_manifest.json` records each file's size and SHA-256. The task's literature cutoff is `2020-01-01`; the public contract governs permitted sources and evidence.

`publication_manifest.json` binds all 11 workspace files to their hashes and records which public source files were copied. The task-specific input bytes match the registered V1.0 task and the historical archive's input files.

## Start a separate, clean run

The surrounding repository includes historical scientific answers and evaluator assets. Only give the participant the copied `workspace/` contents, in a separate environment without access to this archive. Keep this README and publication metadata on the operator side.

From a parent directory where both destination names below are unused:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/ZYDTR/truthinsightbench-dwf-scoring.git tib-inputs
git -C tib-inputs sparse-checkout set clean_tasks/Life_03_nutrient_response_kinetics
python3 -c "import shutil; shutil.copytree('tib-inputs/clean_tasks/Life_03_nutrient_response_kinetics/workspace', 'nutrient-response-run')"
mkdir -p nutrient-response-run/output
```

`copytree` refuses to overwrite an existing run directory. Git does not retain the empty `output/` directory, so create it after copying. Mount only `nutrient-response-run/` into the participant's sandbox, then use the verbatim contents of `LaunchPrompt.md` as the initial task message. No model, agent session, credentials, or execution environment is included in this input package.

After the participant produces its deliverables, run from that workspace:

```bash
python3 validate_output.py output
```

The public validator checks delivery structure and traceability; it does not grade scientific quality. Dependencies for scientific analysis depend on the chosen methods and must be supplied by the execution environment.

## Attribution

Source: [TruthInsightBench task](https://github.com/TruthInsight-stack/TruthInsightBench/tree/main/tasks/Life_03_nutrient_response_kinetics). This package preserves the locally frozen V1.0 bytes; the upstream `main` branch may subsequently change. See the repository's retained [Apache-2.0 license](../../TruthInsightBench/LICENSE) and [third-party notices](../../TruthInsightBench/provenance/THIRD_PARTY_NOTICES.md). The task manifest declares the source-data workbook's CC BY license boundary. No provenance lookup or source reconstruction is required to solve the task.
