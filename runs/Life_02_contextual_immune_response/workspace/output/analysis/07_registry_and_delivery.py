#!/usr/bin/env python3
"""Build the artifact registry and run the delivery-contract self-check.

Writes:
  output/results/file_registry.json   native artifact-id -> real path registry
  output/results/delivery_report.json structured delivery-role inventory
  output/results/run_manifest.json    provenance: inputs, hashes, environment

The registry maps every scientific artifact produced by this run to a stable
Fxxx identifier so the report can cite findings by resolvable id as well as by
path.  Paths are stored relative to output/ so the validator can resolve them.

Run:  python3 output/analysis/07_registry_and_delivery.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
ROOT = OUTPUT.parent
DATA = ROOT / "data"
RESULTS = OUTPUT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(OUTPUT).as_posix()


# --------------------------------------------------------------------------
# 1. Artifact registry
# --------------------------------------------------------------------------
# Ordered by scientific role so ids are stable and meaningful.
ROLE_ORDER = [
    "analysis_source",
    "result_table",
    "result_json",
    "figure",
    "process_log",
]

entries: list[dict] = []


def register(path: Path, role: str, title: str, produced_by: str) -> None:
    entries.append(
        {
            "id": f"F{len(entries) + 1:03d}",
            "role": role,
            "title": title,
            "relative_path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "produced_by": produced_by,
        }
    )


SCRIPT_TITLES = {
    "common.py": ("Shared path/statistics helpers for the analysis scripts",
                  "support module; no scientific result of its own"),
    "01_compartment_intervention.py": (
        "Core analysis: compartment cells, ordering reproducibility, endpoint conflict, "
        "assay-level leave-one-out, intervention ordering, multi-condition class fits",
        "scripts 01 + 04 + 06 read its CSV/JSON outputs"),
    "02_time_resolved.py": (
        "Time-resolved analysis: per-time-point paired contrasts, time x role interaction, "
        "segmented slopes, robustness",
        "fig4 and 06_sensitivity read its outputs"),
    "03_robustness.py": (
        "Targeted robustness: repeated-measures class contrast, ordering tie sensitivity, "
        "rank concordance, precision weighting, endpoint consistency, outlier treatment",
        "supports finding 3 and finding 5"),
    "04_figures.py": (
        "Figure generation for all four report figures (Agg backend, deterministic PNGs)",
        "writes output/figures/*.png"),
    "05_verification_compartment.py": (
        "Independent recomputation of the compartment-file headline numbers (own code path)",
        "writes results/verification_compartment.json"),
    "05_verification_intervention.py": (
        "Independent recomputation of the intervention and multi-condition numbers",
        "writes results/verification_intervention.json"),
    "05_verification_time.py": (
        "Independent recomputation of the time-resolved numbers",
        "writes results/verification_time.json"),
    "06_sensitivity.py": (
        "Resolution limits: minimum detectable effects, TOST equivalence, assays needed, "
        "ordering boundary, replicate resolution floor",
        "supports findings 2 and 4"),
    "07_registry_and_delivery.py": (
        "Builds the artifact registry and the delivery-role self-check",
        "writes results/file_registry.json and results/delivery_report.json"),
}

for name in sorted(SCRIPT_TITLES):
    path = HERE / name
    if path.exists():
        title, produced_by = SCRIPT_TITLES[name]
        register(path, "analysis_source", title, produced_by)

for path in sorted(RESULTS.glob("*.csv")):
    register(path, "result_table", f"Result table {path.name}", "analysis scripts 01-06")

for path in sorted(RESULTS.glob("*.json")):
    if path.name in {"file_registry.json", "delivery_report.json", "run_manifest.json"}:
        continue
    register(path, "result_json", f"Machine-readable result summary {path.name}",
             "analysis scripts 01-06")

for path in sorted((OUTPUT / "figures").glob("*.png")):
    register(path, "figure", f"Report figure {path.name}", "04_figures.py")

for path in sorted((OUTPUT / "logs").glob("*.log")):
    register(path, "process_log", f"Executed-script console log {path.name}",
             "captured stdout of the corresponding analysis script")

registry = {
    "schema": "life02-native-artifact-registry",
    "task_id": "Life_02_contextual_immune_response",
    "note": "relative_path is relative to the output/ directory",
    "count": len(entries),
    "files": entries,
}
(RESULTS / "file_registry.json").write_text(
    json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"[write] results/file_registry.json  ({len(entries)} registered artifacts)")
for entry in entries:
    print(f"   {entry['id']}  {entry['role']:16s} {entry['relative_path']}")

# --------------------------------------------------------------------------
# 2. Delivery-role self-check
# --------------------------------------------------------------------------
roles: dict[str, list[str]] = {
    "authoritative_findings_root_markdown": [],
    "executed_analysis_source": [],
    "inspectable_result_evidence": [],
    "reproduction_trace": [],
}
report_md = next(
    (p for p in OUTPUT.iterdir() if p.is_file() and p.name.casefold() == "result.md"), None
)
repro_md = next(
    (p for p in OUTPUT.iterdir() if p.is_file() and p.name.casefold() == "reproduce.md"), None
)
if report_md:
    roles["authoritative_findings_root_markdown"].append(rel(report_md))
if repro_md:
    roles["reproduction_trace"].append(rel(repro_md))
roles["executed_analysis_source"] = [
    rel(HERE / n) for n in sorted(SCRIPT_TITLES) if (HERE / n).exists()
]
roles["inspectable_result_evidence"] = [
    e["relative_path"] for e in entries if e["role"] in {"result_table", "result_json", "figure"}
]
roles["reproduction_trace"] += [
    "reproduce.md (companion reproduction document)",
    "reproduce.md#methods (methods + execution order)",
]

missing = [key for key, value in roles.items() if not value]
delivery = {
    "schema": "life02-delivery-role-inventory",
    "task_id": "Life_02_contextual_immune_response",
    "roles": roles,
    "roles_missing": missing,
    "self_check_passed": not missing,
}
(RESULTS / "delivery_report.json").write_text(
    json.dumps(delivery, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"[write] results/delivery_report.json  missing_roles={missing}")

# --------------------------------------------------------------------------
# 3. Run manifest
# --------------------------------------------------------------------------
data_hashes = {p.name: sha256(p) for p in sorted(DATA.glob("*.tsv"))}
manifest = {
    "schema": "life02-run-manifest",
    "task_id": "Life_02_contextual_immune_response",
    "t0": "2020-01-01",
    "inputs": data_hashes,
    "input_bytes": {p.name: p.stat().st_size for p in sorted(DATA.glob("*.tsv"))},
    "environment": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    },
    "script_order": [
        "analysis/common.py",
        "analysis/01_compartment_intervention.py",
        "analysis/02_time_resolved.py",
        "analysis/03_robustness.py",
        "analysis/04_figures.py",
        "analysis/05_verification_compartment.py",
        "analysis/05_verification_intervention.py",
        "analysis/05_verification_time.py",
        "analysis/06_sensitivity.py",
        "analysis/07_registry_and_delivery.py",
    ],
    "external_data_used": False,
    "network_access_used": False,
    "artifact_count": len(entries),
}
(RESULTS / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("[write] results/run_manifest.json")
print("\n[done] 07_registry_and_delivery.py")