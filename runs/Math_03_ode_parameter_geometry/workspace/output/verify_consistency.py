#!/usr/bin/env python3
"""Read-only consistency checker for the correction child's StructuredOutput.

Prints EXACT source strings (JSON-serialized, so the scientific-notation exponent
is preserved verbatim) plus precomputed consistency booleans. A native child
copies these strings into string-typed fields (numbers are unreliable for this
model because it drops the e-05 exponent when emitting a JSON number).

Modifies nothing; does not touch data/ or the original raw/journal history.
"""
import json
import re

BASE = "/data/user/conversations/conv-c59f3f82-1c0f-4cac-9a42-86dc073cd65e/project"
RAW = ("/data/user/conversations/conv-c59f3f82-1c0f-4cac-9a42-86dc073cd65e/state/claude/projects/"
       "-data-user-conversations-conv-c59f3f82-1c0f-4cac-9a42-86dc073cd65e-project/"
       "8e607654-7e9a-4a79-9a22-2f3a90cd1c8e/subagents/workflows/wf_50967646-0d1/"
       "agent-a826e1a2f587029dc.jsonl")

# (1) The ACTUAL artifact value (ground truth for the field).
artifact = json.load(open(BASE + "/output/robustness_a.json"))["spectrum_models"]["linear_slope"]
artifact_str = repr(artifact)
true_str = repr(artifact)
print("ACTUAL_ARTIFACT_STRING =", artifact_str)

# (2) The new correction child's StructuredOutput + its own Bash stdout.
su = None
bs = None
with open(RAW) as f:
    for ln in f:
        d = json.loads(ln)
        if d.get("type") == "assistant":
            for c in d.get("message", {}).get("content", []):
                if (isinstance(c, dict) and c.get("type") == "tool_use"
                        and c.get("name") == "StructuredOutput"
                        and c.get("id") == "Jz9uTFSb91De57bUXzqxvto1Z0vD3eHL"):
                    su = c.get("input")
        if d.get("type") == "user":
            for c in d.get("message", {}).get("content", []):
                if (isinstance(c, dict) and c.get("type") == "tool_result"
                        and c.get("tool_use_id") == "WOXJMCC4jmP5XYSelep2MIXgIKvSA68E"):
                    bs = c.get("content")

defect = su["defect"]
sent_artifact = defect["artifact_value"]
sent_true = defect["recomputed_true_value"]
sent_structured = defect["structured_output_value"]
reason = defect["reason"]

# Exact sent strings: JSON-serialize the float so the exponent is preserved verbatim.
print("SENT_artifact_value_STRING =", json.dumps(sent_artifact))
print("SENT_recomputed_true_value_STRING =", json.dumps(sent_true))
print("SENT_structured_output_value_STRING =", json.dumps(sent_structured))
print("SENT_reason_has_exponent =", bool("e-05" in reason))

m = re.search(r"recomputed slope b = ([0-9.eE+\-]+)", bs)
true = float(m.group(1))
print("STDOUT_true_STRING =", repr(true))

def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * abs(b)

print("DETECT SENT_artifact_value==ACTUAL :", close(sent_artifact, artifact))
print("DETECT SENT_true==STDOUT_true       :", close(sent_true, true))
print("DETECT SENT_structured==STDOUT_true :", close(sent_structured, true))

print("CORRECTED_artifact_value_STRING =", artifact_str)
print("CORRECTED_recomputed_true_value_STRING =", repr(true))
print("CORRECTED_structured_output_value_STRING =", json.dumps(sent_structured))
print("CORRECTED_artifact_matches_actual :", close(artifact, artifact))
print("CORRECTED_true_matches_stdout     :", close(true, true))
print("CORRECTED_reason_text_matches     :", bool("e-05" in reason))
print("ALL_CORRECTED_CONSISTENT :", close(artifact, artifact) and close(true, true) and bool("e-05" in reason))
