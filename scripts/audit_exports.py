#!/usr/bin/env python3
"""Read-only public-package privacy and frozen-base audit; never print secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_ENV_KEYS = {
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY",
    "CCSDK_OPENAI_COMPAT_API_KEY", "CCSDK_ANTHROPIC_API_KEY",
    "CCSDK_CLAUDE_API_KEY", "CCSDK_LITELLM_API_KEY", "LITELLM_MASTER_KEY",
    "SECRET_KEY", "JWT_SECRET", "POSTGRES_PASSWORD", "DATABASE_PASSWORD",
    "DATABASE_URL", "REDIS_PASSWORD", "MINIO_SECRET_KEY",
}
TOKEN_FIELDS = {"token", "access_token", "refresh_token", "api_key"}
PATTERNS = {
    "workstation_path": re.compile(rb"/Users/[A-Za-z0-9._-]+/|[A-Z]:\\Users\\"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----\r?\n"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "api_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
    "credential_url": re.compile(rb"(?:postgres(?:ql)?|redis|https?)://[^\s/:@]+:[^\s/@]+@"),
}


def known_secrets(source_root: Path | None) -> set[bytes]:
    """Read only exact credential keys and token fields into memory."""
    values: set[bytes] = set()
    if source_root is None:
        return values
    env = source_root / ".local/credentials.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key in SECRET_ENV_KEYS:
                value = value.strip().strip("\"'")
                if len(value) >= 8:
                    values.add(value.encode())
    for path in (source_root / "runs").glob("**/private/token.json"):
        data = json.loads(path.read_text())
        for field in TOKEN_FIELDS:
            value = data.get(field)
            if isinstance(value, str) and len(value) >= 8:
                values.add(value.encode())
    return values


def looks_like_sdk_log(data: bytes) -> bool:
    """Distinguish a scientific verification note from serialized SDK events."""
    for line in data.splitlines():
        if not line.lstrip().startswith(b"{"):
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict) and (
            (row.get("type") in {"assistant", "user", "system", "result", "queue-operation"}
             and any(k in row for k in {"sessionId", "uuid", "parentUuid", "message"}))
            or (row.get("type") in {"started", "result"} and "key" in row and "agentId" in row)
            or (row.get("type") in {"response_item", "event_msg", "turn_context"} and "payload" in row)
        ):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional private harness for exact-value checks")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "export_manifest.json").read_text())
    extension = json.loads((ROOT / manifest["extension_manifest"]).read_text())
    policy = json.loads((ROOT / "versions/20261010-40/privacy_scope.json").read_text())
    if any(item["issue"] not in {"raw_log_file", "serialized_runtime_log", "workstation_path"}
           for item in policy["preserved_legacy_exceptions"]):
        raise SystemExit("legacy policy cannot waive credential or integrity findings")
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode().split("\0")
    paths = sorted(set(p for p in paths if p))
    secrets = known_secrets(args.source_root)
    issues = []
    scientific_notes = []
    largest = 0
    byte_count = 0
    for name in paths:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            issues.append({"path": name, "issue": "unsafe_or_missing_file"})
            continue
        data = path.read_bytes()
        largest = max(largest, len(data))
        byte_count += len(data)
        parts = Path(name).parts
        if any(p in {".local", "sdk", "rounds", "state", "private"} for p in parts):
            issues.append({"path": name, "issue": "runtime_or_private_directory"})
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            issues.append({"path": name, "issue": "raw_log_file"})
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                issues.append({"path": name, "issue": label})
        if any(secret in data for secret in secrets):
            issues.append({"path": name, "issue": "known_credential_value"})
        if path.suffix.lower() in {".md", ".txt", ".json", ".log", ".jsonl", ".ndjson"}:
            if looks_like_sdk_log(data):
                issues.append({"path": name, "issue": "serialized_runtime_log"})
            elif path.name.endswith(".jsonl.md"):
                scientific_notes.append(name)
    protected = extension["protected_files_sha256"]
    changed = [name for name, digest in protected.items()
               if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        issues.extend({"path": name, "issue": "protected_base_changed"} for name in changed)
    if len(manifest["tasks"]) != 40 or len(set(manifest["tasks"])) != 40:
        issues.append({"path": "export_manifest.json", "issue": "not_40_unique_tasks"})
    legacy_exceptions = []
    actionable = []
    allowed = {(item["path"], item["issue"]): item for item in policy["preserved_legacy_exceptions"]}
    for item in issues:
        exception = allowed.get((item["path"], item["issue"]))
        path = ROOT / item["path"]
        if exception and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == protected.get(item["path"]):
            legacy_exceptions.append(item)
        else:
            actionable.append(item)
    result = {
        "schema": "tib-public-archive-audit.v2", "passed": not actionable,
        "scope": "Strict privacy for new 13 exports; original published 27 and scoring tree preserved with explicit user-accepted legacy exceptions",
        "task_count": manifest["task_count"], "file_count": len(paths), "bytes": byte_count,
        "largest_file_bytes": largest, "known_credential_values_checked": len(secrets),
        "protected_files_checked": len(protected), "protected_files_unchanged": not changed,
        "raw_runtime_logs_detected": sum(i["issue"] in {"raw_log_file", "serialized_runtime_log"} for i in issues),
        "raw_runtime_log_files_detected": len({i["path"] for i in issues
             if i["issue"] in {"raw_log_file", "serialized_runtime_log"}}),
        "new_exports_raw_runtime_logs_detected": sum(i["issue"] in {"raw_log_file", "serialized_runtime_log"} and
             any(i["path"].startswith("runs/" + t + "/") for t in extension["added_tasks"]) for i in issues),
        "legacy_exceptions": legacy_exceptions,
        "full_repository_has_no_legacy_logs_or_paths": not legacy_exceptions,
        "scientific_verification_notes_named_jsonl_md": scientific_notes,
        "issues": actionable,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
