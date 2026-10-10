// Serial TruthInsightBench workflow for Material_03_membrane_selective_permeation.
// One active child at a time; each await completes before the next begins.
export const meta = {
  name: "membrane_finding_pipeline",
  description: "Serialize: execute analysis evidence, draft Result.md, then review completeness.",
  phases: [
    { title: "Execute evidence" },
    { title: "Draft report" },
    { title: "Review report" },
  ],
}

const ROOT = "/data/user/conversations/conv-670bd062-7bb2-4218-93c3-d9412c13e342/project"
const DATA = ROOT + "/data"
const ANA = ROOT + "/output/analysis"

phase("Execute evidence")
const evidence = await agent(
  "You are a scientific data-analysis subagent for a TruthInsightBench task. " +
  "Work in the project root: " + ROOT + ". Do not read other task workspaces or prior runs. " +
  "The data files are under " + DATA + ". Analysis scripts already exist at " + ANA + "/analysis_qa.py " +
  "(writes " + ANA + "/results.json and " + ANA + "/results.tsv) and " + ANA + "/robustness_qa.py " +
  "(writes " + ANA + "/robustness.json). " +
  "Task: " +
  "1. Run the core analysis: cd " + ROOT + ' && MEMBRANE_DATA_DIR="' + DATA + '" ' +
  "MEMBRANE_OUT_DIR='" + ANA + "' python3 output/analysis/analysis_qa.py " +
  "2. Run the robustness checks: cd " + ROOT + ' && MEMBRANE_DATA_DIR="' + DATA + '" ' +
  "MEMBRANE_OUT_DIR='" + ANA + "' python3 output/analysis/robustness_qa.py " +
  "3. Verify " + ANA + "/results.json, " + ANA + "/results.tsv, " + ANA + "/robustness.json exist, " +
  "are non-empty, and that the .json files parse as valid JSON (use python3 -c \"import json;json.load(open('" + ANA + "/results.json'))\"). " +
  "4. In your final answer, give a concise factual summary of the key numbers the scripts print and " +
  "confirm each output file exists with its byte size. " +
  "If a command fails, do NOT fabricate output; paste the exact error and stop that step, but report what succeeded. " +
  "Your final answer is the task result - make it a short factual summary of observed numbers and file existence.",
  { label: "execute_evidence", phase: "Execute evidence", effort: "high" }
)
log("child1 (execute_evidence) returned")

phase("Draft report")
const report = await agent(
  "You are the report-writing subagent for a TruthInsightBench scientific discovery task. " +
  "Work in project root " + ROOT + ". " +
  "READ FIRST (do not invent numbers): " + ANA + "/results.json and " + ANA + "/robustness.json " +
  "(and optionally " + DATA + "/*.tsv for context). " +
  "Then write the authoritative report to " + ROOT + "/output/Result.md in English. Requirements: " +
  "- The report is the complete and only authoritative scientific account. " +
  "- Register EXACTLY 5 findings, each under a Markdown heading '## Finding N: <title>' for N=1..5. " +
  "  Use that exact label style so labels are mechanically identifiable. " +
  "- For EACH finding include clearly labeled elements: (a) Object and scope; (b) Relationship, direction, " +
  "  or boundary; (c) Core quantitative result AND the analysis unit; (d) Supporting analysis and result " +
  "  evidence - cite the exact real file paths '" + ANA + "/analysis_qa.py' (analysis source) " +
  "and '" + ANA + "/results.json' (result evidence), plus '" + ANA + "/robustness.json' " +
  "for the robustness result; (e) Robustness result (quote specific numbers from robustness.json); " +
  "  (f) Counterevidence, limitation, and alternative explanations; (g) An executable new observation or " +
  "  independent replication that could falsify or validate it. " +
  "The five findings, based on the evidence, are: " +
  "1. Reference vs modified membrane proton conductivity: modified is ~34x higher (state mean and median " +
  "  ratios, and that no modified value is below any reference value). Analysis unit: one conductivity specimen. " +
  "2. Local-current distribution heterogeneity: modified membrane shows a bimodal local-current distribution " +
  "  with a large high-current fraction (~50% of observations >100 pA) and median ~106.6 pA, whereas the " +
  "  reference membrane is entirely low-current (median ~1.44 pA, max ~23.7 pA, 0% >100 pA). Analysis unit:" +
  "  class-level local-current vector (descriptive distribution; not bootstrappable as independent specimens). " +
  "3. Bias-current linearity and conductance: both classes show strongly linear I-V behavior (R2 ~0.9997-1.0), " +
  "  but modified membrane has ~10.4x the conductance (slope) of the reference membrane. Analysis unit: one " +
  "  complete ordered bias-current curve. " +
  "4. Condition (gas) response is a null effect: for each of Xe, Kr, Ne, Ar (n=11 devices each) the mean " +
  "  dimensionless_delta is within the 95% CI of zero; no gas shows a statistically significant mean. Analysis " +
  "  unit: one device under one test condition. " +
  "5. Long-term device time response is a null effect (no stable trend): per-device linear slopes range roughly " +
  "  -0.025 to +0.028 per day with mean ~0.0019; leave-one-out mean slopes span about -0.0013 to +0.0053, " +
  "  none significant. Analysis unit: one device's time series. " +
  "Add a '## Reproduction' section near the end stating the exact commands to re-run the analysis and the " +
  "output files. Add a '## Limitations' section. " +
  "Use the numbers from the evidence files; do not add unsupported claims. Reporting null effects (findings 4 " +
  "and 5) and boundaries/limits is expected and correct. Do not include the string 'REPLACE_' anywhere. " +
  "Your final answer: state the path of the file you wrote (output/Result.md) and confirm all 5 findings and " +
  "the Reproduction section are present.",
  { label: "draft_report", phase: "Draft report", effort: "high" }
)
log("child2 (draft_report) returned")

phase("Review report")
const review = await agent(
  "You are the completeness-critic subagent for a TruthInsightBench task. Work in project root " + ROOT + ". " +
  "Read " + ROOT + "/output/Result.md. Verify it satisfies the semantic delivery contract. Do NOT rewrite it; " +
  "only audit and report gaps. Check: " +
  "1. Exactly 5 registered findings, labeled consecutively '## Finding 1' through '## Finding 5' " +
  "   (case-insensitive), no extra finding-style labels. " +
  "2. Each finding contains all seven required elements: object/scope; relationship/direction/boundary; " +
  "  core quantitative result + analysis unit; supporting analysis + result evidence (citing real files " +
  ANA + "/analysis_qa.py, " + ANA + "/results.json, " + ANA + "/robustness.json); " +
  "  robustness result; counterevidence/limitation/alternative explanations; executable falsification/replication. " +
  "3. The cited evidence files actually exist and are non-empty: check " + ANA + "/analysis_qa.py," +
  "  " + ANA + "/results.json, " + ANA + "/robustness.json, " + ANA + "/results.tsv. " +
  "4. There is a '## Reproduction' section with runnable commands. " +
  "5. No 'REPLACE_' marker. 6. Written in English; complete authoritative account. " +
  "If a file is missing/empty, run ls -la " + ANA + "/ and " +
  "python3 -c \"import json;json.load(open('" + ANA + "/results.json'))\" and report the exact findings. " +
  "Final answer: a numbered list of (a) PASS items and (b) any GAPS with the specific fix needed; cite line " +
  " numbers where possible.",
  { label: "review_report", phase: "Review report", effort: "high" }
)
log("child3 (review_report) returned")

return {
  evidence,
  report,
  review,
}
