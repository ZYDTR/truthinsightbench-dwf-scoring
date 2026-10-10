#!/usr/bin/env python3
"""Append a versioned extension without changing existing scoring submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from export_completed import (
    ROOT, TIB, export_one, latest_complete_snapshot, load_json,
    sha256_file, tree_snapshot,
)


BASE_COMMIT = "ace8ab84c407201a089a45ff7e572092c6f27f87"


def protected_hashes(root: Path, tasks: list[str]) -> dict[str, str]:
    """Bind every pre-existing submission and the original scoring assets."""
    result = {}
    for directory in [root / "TruthInsightBench", *(root / "runs" / t for t in tasks)]:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("protected archive contains a symlink")
            if path.is_file():
                result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def select_released_source(source_root: Path, task_id: str) -> Path:
    """Resolve canonical or independent attempt identities; reject ambiguity."""
    canonical = source_root / "runs" / task_id
    candidates = []
    for path in [canonical, *sorted((canonical / "attempts").glob("*"))]:
        if not (path / "run.json").is_file() or not (path / "private/release.json").is_file():
            continue
        run = load_json(path / "run.json")
        if run.get("status") != "RELEASED":
            continue
        release = load_json(path / "private/release.json")
        validation = load_json(path / "latest-validation.json")
        if run.get("task_id") != task_id or release.get("task_id", release.get("run_id")) != task_id:
            raise ValueError(f"released source identity mismatch: {task_id}")
        if not validation.get("passed"):
            raise ValueError(f"released source validation failed: {task_id}")
        latest_complete_snapshot(path)
        candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"expected one released source for {task_id}, found {len(candidates)}")
    return candidates[0]


def source_view(source: Path, task_id: str, directory: Path) -> Path:
    """Give the frozen adapter a logical task name without editing the Run."""
    view = directory / task_id
    view.mkdir()
    for name in ["run.json", "latest-validation.json", "workspace", "snapshots", "reviews"]:
        (view / name).symlink_to(source / name)
    (view / "private").mkdir()
    (view / "private/release.json").symlink_to(source / "private/release.json")
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--expected-base-count", type=int, default=27)
    parser.add_argument("--version", default="20261010-40")
    args = parser.parse_args()
    if not args.version or Path(args.version).name != args.version:
        raise SystemExit("version must be a directory basename")
    manifest_path = ROOT / "export_manifest.json"
    old_bytes = manifest_path.read_bytes()
    old = load_json(manifest_path)
    existing = old["tasks"]
    if len(existing) != args.expected_base_count or len(set(existing)) != len(existing):
        raise SystemExit("unexpected base task count or duplicate tasks")
    registered = [r["task_id"] for r in load_json(args.source_root / "config/tasks.json")["bindings"]]
    if len(registered) != args.expected_count or len(set(registered)) != len(registered):
        raise SystemExit("unexpected registry count or duplicate tasks")
    if not set(existing).issubset(registered):
        raise SystemExit("base archive has tasks outside this registry")
    added = sorted(set(registered) - set(existing))
    version_root = ROOT / "versions" / args.version
    base_version_root = ROOT / "versions/20260924-27"
    if version_root.exists() or base_version_root.exists():
        raise SystemExit("refusing to overwrite version receipts")
    for task in added:
        if (ROOT / "runs" / task).exists():
            raise SystemExit(f"refusing to overwrite scoring export: {task}")
    sources = {task: select_released_source(args.source_root.resolve(), task) for task in added}
    before = protected_hashes(ROOT, existing)
    # Compare protected bytes with the actual published Git baseline, not only
    # with an in-memory pre-export scan.
    published = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", BASE_COMMIT], cwd=ROOT,
    ).decode().split("\0")
    protected_paths = {p for p in published if p.startswith("TruthInsightBench/") or
                       any(p.startswith(f"runs/{t}/") for t in existing)}
    if not protected_paths.issubset(before):
        raise SystemExit("published protected files are missing")
    local_only = set(before) - protected_paths
    for name in local_only:
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", name], cwd=ROOT)
        if ignored.returncode != 0:
            raise SystemExit(f"unexpected non-published protected file: {name}")
    published_before = {name: before[name] for name in sorted(protected_paths)}
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as batch:
        for name, digest in published_before.items():
            batch.stdin.write(f"{BASE_COMMIT}:{name}\n".encode())
            batch.stdin.flush()
            header = batch.stdout.readline().decode().split()
            if len(header) != 3 or header[1] != "blob":
                raise SystemExit(f"missing published blob: {name}")
            remaining = int(header[2])
            hasher = hashlib.sha256()
            while remaining:
                block = batch.stdout.read(min(remaining, 1024 * 1024))
                if not block:
                    raise SystemExit("published blob stream ended early")
                hasher.update(block)
                remaining -= len(block)
            if batch.stdout.read(1) != b"\n" or hasher.hexdigest() != digest:
                raise SystemExit(f"protected file differs from published baseline: {name}")
        batch.stdin.close()
        if batch.wait() != 0:
            raise SystemExit("published blob verification failed")
    assets = tree_snapshot(TIB)
    if assets["unsafe_entries"]:
        raise SystemExit("original scoring asset tree has unsafe entries")
    # The historical top-level checksum does not match its published Git tree. All
    # published asset bytes were verified above; retain the historical manifest
    # verbatim and bind the new manifest to the actual unchanged asset tree.
    # All new submissions are prepared outside the archive before publication.
    with tempfile.TemporaryDirectory(prefix="tib-scoring-extension-") as temporary:
        staging = Path(temporary)
        views = staging / "views"
        views.mkdir()
        outputs = staging / "runs"
        outputs.mkdir()
        evidence = []
        for task in added:
            view = source_view(sources[task], task, views)
            item = export_one(view, outputs, "sap-dwf-202610-bonsai")
            item["export_version"] = args.version
            item["source_attempt"] = sources[task].name
            item["source_release_at"] = load_json(sources[task] / "private/release.json").get("at")
            (outputs / task / "export_evidence.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            evidence.append(item)
        if protected_hashes(ROOT, existing) != before:
            raise SystemExit("protected files changed during extension")
        for task in added:
            shutil.copytree(outputs / task, ROOT / "runs" / task)
    if protected_hashes(ROOT, existing) != before:
        raise SystemExit("protected files changed after extension")
    base_version_root.mkdir(parents=True)
    (base_version_root / "export_manifest.json").write_bytes(old_bytes)
    version_root.mkdir(parents=True)
    extension = {
        "schema": "tib-dwf-scoring-extension.v1",
        "version": args.version,
        "extended_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": BASE_COMMIT,
        "base_manifest_sha256": sha256_file(base_version_root / "export_manifest.json"),
        "base_task_count": len(existing),
        "added_task_count": len(added),
        "added_tasks": added,
        "added_registered_finding_count": sum(e["registered_findings"] for e in evidence),
        "total_task_count": len(registered),
        "protected_file_count": len(published_before),
        "protected_files_sha256": published_before,
        "preexisting_ignored_local_files_preserved_not_published": len(local_only),
        "old_27_and_original_scoring_assets_unchanged": True,
        "historical_manifest_scoring_assets_sha256": old["scoring_assets_sha256"],
        "published_unchanged_scoring_assets_sha256": assets["sha256"],
        "scoring_asset_file_count": assets["file_count"],
        "historical_asset_checksum_matches_published_tree": old["scoring_assets_sha256"] == assets["sha256"],
        "raw_sdk_logs_exported": False,
    }
    (version_root / "extension_manifest.json").write_text(
        json.dumps(extension, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    full = {
        **old,
        "task_count": len(registered),
        "registered_finding_count": old["registered_finding_count"] + extension["added_registered_finding_count"],
        "tasks": sorted(registered),
        "models": sorted(set(old["models"]) | {m for e in evidence for m in e["assistant_model_messages"]}),
        "export_version": args.version,
        "scoring_assets_sha256": assets["sha256"],
        "extended_at": extension["extended_at"],
        "base_manifest": "versions/20260924-27/export_manifest.json",
        "extension_manifest": f"versions/{args.version}/extension_manifest.json",
    }
    manifest_path.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"version": args.version, "base": len(existing), "added": len(added),
                      "total": len(registered), "findings": full["registered_finding_count"],
                      "protected_files": len(published_before), "protected_unchanged": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
