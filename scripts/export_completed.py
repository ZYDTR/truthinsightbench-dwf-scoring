#!/usr/bin/env python3
"""Export released DWF runs as immutable TruthInsightBench scoring inputs.

This is a compatibility adapter, not a replay of the platform execution. It
uses frozen collect snapshots and refuses to overwrite an existing export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIB = ROOT / "TruthInsightBench"
sys.path.insert(0, str(TIB / "agents"))
from workspace_integrity import sha256_file, tree_snapshot, validate_workspace  # noqa: E402


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def released_sources(source_root: Path) -> list[Path]:
    result: list[Path] = []
    for run in sorted((source_root / "runs").iterdir()):
        manifest_path = run / "run.json"
        release_path = run / "private" / "release.json"
        if not manifest_path.is_file() or not release_path.is_file():
            continue
        if load_json(manifest_path).get("status") == "RELEASED":
            result.append(run)
    return result


def latest_complete_snapshot(run: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for candidate in (run / "snapshots").iterdir():
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file() or not (candidate / "output").is_dir():
            continue
        manifest = load_json(manifest_path)
        candidates.append((str(manifest.get("at", "")), candidate))
    if not candidates:
        raise ValueError(f"no complete frozen snapshot: {run.name}")
    return max(candidates)[1]


def model_counts(snapshot: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in sorted((snapshot / "sdk").rglob("*.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    message = json.loads(line).get("message", {})
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                model = message.get("model")
                if isinstance(model, str) and model and model != "<synthetic>":
                    counts[model] += 1
    if not counts:
        raise ValueError(f"no real assistant model in SDK JSONL: {snapshot}")
    return counts


def model_label(counts: Counter[str]) -> str:
    names = sorted(counts)
    return names[0] if len(names) == 1 else "mixed:" + "+".join(names)


def export_one(run: Path, destination: Path, agent_id: str) -> dict:
    task_id = run.name
    manifest = load_json(run / "run.json")
    release = load_json(run / "private" / "release.json")
    validation = load_json(run / "latest-validation.json")
    release_task_id = release.get("task_id", release.get("run_id"))
    if manifest.get("task_id") != task_id or release_task_id != task_id:
        raise ValueError(f"task identity mismatch: {task_id}")
    if manifest.get("status") != "RELEASED" or not validation.get("passed"):
        raise ValueError(f"run not released and validated: {task_id}")
    if not (run / "reviews").is_dir():
        raise ValueError(f"missing review: {task_id}")
    snapshot = latest_complete_snapshot(run)
    frozen_output = snapshot / "output"
    source_output = tree_snapshot(frozen_output)
    source_input = validate_workspace(run / "workspace", expected_task_id=task_id)
    if source_output["unsafe_entries"] or source_output["file_count"] == 0:
        raise ValueError(f"unsafe or empty frozen output: {task_id}")
    if not source_input["passed"]:
        raise ValueError(f"source input integrity failed: {task_id}: {source_input['errors']}")
    seeded_output = run / "workspace" / "output"
    if seeded_output.exists() and (not seeded_output.is_dir() or any(seeded_output.iterdir())):
        raise ValueError(f"seeded workspace output is not empty: {task_id}")

    counts = model_counts(snapshot)
    target = destination / task_id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite scoring export: {target}")
    target.mkdir(parents=True)
    workspace = target / "workspace"
    shutil.copytree(run / "workspace", workspace)
    shutil.copytree(frozen_output, workspace / "output", dirs_exist_ok=True)
    copied_input = validate_workspace(workspace, expected_task_id=task_id)
    copied_output = tree_snapshot(workspace / "output")
    if not copied_input["passed"] or copied_input["input_tree_sha256"] != source_input["input_tree_sha256"]:
        raise ValueError(f"copied input mismatch: {task_id}")
    if copied_output != source_output:
        raise ValueError(f"copied output mismatch: {task_id}")

    # The validator receipt contains absolute local paths and is not a scorer
    # input. Keep it outside this publishable archive.
    with tempfile.TemporaryDirectory(prefix="tib-submission-check-") as directory:
        submission_receipt = Path(directory) / "submission_validation.json"
        check = subprocess.run(
            [sys.executable, "-B", str(TIB / "evaluation" / "submission_format" / "validate_submission.py"),
             str(workspace / "output"), "--receipt", str(submission_receipt)],
            text=True, capture_output=True,
        )
        if check.returncode:
            raise ValueError(f"submission validation failed for {task_id}: {check.stdout} {check.stderr}")
        checked = load_json(submission_receipt)
    if not checked.get("passed"):
        raise ValueError(f"submission validation did not pass: {task_id}")

    receipt = {
        "schema": "truthinsightbench-agent-run-receipt",
        "release": "V1.0",
        "agent_id": agent_id,
        "task_id": task_id,
        "model": model_label(counts),
        "status": "completed",
        "platform_exit_code": 0,
        "platform_exit_code_basis": "DWF adapter: released run plus passing input/output validation; not a captured process exit code",
        "output_validation_passed": True,
        "workspace_input_integrity_passed": True,
        "workspace": "workspace",
        "output_path": "workspace/output",
        "workspace_input_tree_sha256": copied_input["input_tree_sha256"],
        "workspace_input_file_count": copied_input["input_file_count"],
        "workspace_input_bytes": copied_input["input_bytes"],
        "output_tree_sha256": copied_output["sha256"],
        "output_file_count": copied_output["file_count"],
        "output_bytes": copied_output["bytes"],
    }
    (target / "run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evidence = {
        "task_id": task_id,
        "source_run_status": manifest["status"],
        "source_snapshot": snapshot.name,
        "source_run_json_sha256": sha256_file(run / "run.json"),
        "source_release_json_sha256": sha256_file(run / "private" / "release.json"),
        "source_latest_validation_sha256": sha256_file(run / "latest-validation.json"),
        "source_snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "assistant_model_messages": dict(sorted(counts.items())),
        "release_checks": {
            key: release.get(key)
            for key in ("idle_confirmed", "jsonl_result_event", "output_nonempty", "worker_stopped_by_release")
        },
        "registered_findings": checked.get("registered_findings", {}).get("count"),
        "input_sha256": copied_input["input_tree_sha256"],
        "output_sha256": copied_output["sha256"],
    }
    (target / "export_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=27)
    parser.add_argument("--agent-id", default="sap-dwf-202609-mixed")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    sources = released_sources(source_root)
    if len(sources) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} released runs, found {len(sources)}")
    destination = ROOT / "runs"
    if any(destination.iterdir()):
        raise SystemExit(f"scoring export directory is not empty: {destination}")
    evidence = [export_one(run, destination, args.agent_id) for run in sources]
    manifest = {
        "schema": "tib-dwf-scoring-export",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(evidence),
        "registered_finding_count": sum(item["registered_findings"] for item in evidence),
        "tasks": [item["task_id"] for item in evidence],
        "scoring_assets_sha256": tree_snapshot(TIB)["sha256"],
        "models": sorted({model for item in evidence for model in item["assistant_model_messages"]}),
    }
    (ROOT / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"task_count": manifest["task_count"],
                      "registered_finding_count": manifest["registered_finding_count"],
                      "models": manifest["models"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
