# Verification strategy

- Unit tests check snapshot selection and mixed-model labeling in the no-overwrite export adapter.
- Integration validation runs `scripts/validate_exports.py` against all 27 run roots using the unmodified benchmark evaluator in dry-run mode. This checks hashes, task registry identity, output submission format, and cohort assembly without a judge call.
- Full scientific scoring is opt-in because it needs an authorized external judge endpoint and pinned judge model. No score is claimed from a dry-run.
- Before pushing, inspect the repository scope, scan for credentials and unexpected private runtime data, confirm every exported task has a true source release and validation receipt, and verify GitHub repository visibility is public.
