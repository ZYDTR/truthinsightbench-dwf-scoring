"""Build the finding-to-evidence traceability index.

Emits output/results/evidence_index.json, which pins every registered finding
to the exact analysis source and result artifact that supports it, with hashes.
This is the machine-readable companion to the traceability table in Result.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FINDINGS = [
    {
        "finding": 1,
        "short_label": "Database domain ceiling and benchmark coverage gap",
        "analysis_sources": [
            "analysis/common.py",
            "analysis/finding1_coverage.py",
        ],
        "result_evidence": [
            "results/f1_coverage_boundary.json",
            "results/ws1_db_coverage.json",
            "results/ws4_failures.json",
        ],
        "figures": ["figures/summary_findings.png"],
        "verification": ["results/verify_2_coverage.json", "results/verify_1_db_scale.json",
                         "results/verify_8_bondclass_coverage.json"],
    },
    {
        "finding": 2,
        "short_label": "Channel ranking reverses between small-molecule and large/extrapolated regions",
        "analysis_sources": [
            "analysis/common.py",
            "analysis/size_interaction_test.py",
        ],
        "result_evidence": [
            "results/size_interaction_test.json",
            "results/ws2_channel_accuracy.json",
            "results/ws6_robustness.json",
        ],
        "figures": ["figures/summary_findings.png"],
        "verification": ["results/verify_9_size_interaction.json",
                         "results/verify_4_paired.json"],
    },
    {
        "finding": 3,
        "short_label": "Experimental reference noise floor",
        "analysis_sources": [
            "analysis/common.py",
            "analysis/headline_noise_floor.py",
        ],
        "result_evidence": [
            "results/headline_noise_floor.json",
            "results/ws5_decomposition.json",
            "results/ws5_provenance.json",
        ],
        "figures": ["figures/summary_findings.png"],
        "verification": ["results/verify_5_floor.json", "results/verify_6_floor_implication.json"],
    },
    {
        "finding": 4,
        "short_label": "Reference channel is genuinely worse, with bimodal signed error",
        "analysis_sources": [
            "analysis/common.py",
            "analysis/finding4_reference_channel.py",
        ],
        "result_evidence": [
            "results/f4_reference_channel.json",
            "results/ws3_bondclass.json",
            "results/ws4_failures.json",
        ],
        "figures": ["figures/summary_findings.png", "figures/error_vs_bond_strength.png"],
        "verification": ["results/verify_3_channel_accuracy.json",
                         "results/verify_12_reference_structure.json"],
    },
    {
        "finding": 5,
        "short_label": "Weak-bond overshoot: bounded residual bias, direction unresolved",
        "analysis_sources": [
            "analysis/common.py",
            "analysis/weak_bond_overshoot_test.py",
        ],
        "result_evidence": [
            "results/weak_bond_overshoot_test.json",
            "results/ws4_failures.json",
        ],
        "figures": ["figures/summary_findings.png", "figures/error_vs_bond_strength.png"],
        "verification": ["results/verify_10_weakbond.json"],
    },
]


def main() -> int:
    out = common.OUT
    index = []
    missing = []
    for f in FINDINGS:
        entry = {
            "finding": f["finding"],
            "short_label": f["short_label"],
            "analysis_sources": [],
            "result_evidence": [],
            "figures": [],
            "verification": [],
        }
        for key in ("analysis_sources", "result_evidence", "figures", "verification"):
            for rel in f[key]:
                p = out / rel
                if p.is_file():
                    entry[key].append(
                        {"path": f"output/{rel}", "bytes": p.stat().st_size,
                         "sha256": common.sha256(p)[:16]}
                    )
                else:
                    missing.append(f"F{f['finding']}: output/{rel}")
                    entry[key].append({"path": f"output/{rel}", "missing": True})
        index.append(entry)

    payload = {
        "schema": "bde-domain-finding-to-evidence",
        "authoritative_report": "output/Result.md",
        "note": (
            "Every registered finding resolves to executed analysis sources and "
            "machine-readable result artifacts under output/. Paths are relative "
            "to the project root. Hashes are the first 16 hex characters of sha256."
        ),
        "findings": index,
        "unresolved_references": missing,
    }
    common.save_json(payload, "evidence_index.json")
    for e in index:
        ok = all("missing" not in x for k in ("analysis_sources", "result_evidence")
                 for x in e[k])
        print(f"  F{e['finding']} {e['short_label'][:52]:54s} all_refs_present={ok}")
    print(f"unresolved references: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())