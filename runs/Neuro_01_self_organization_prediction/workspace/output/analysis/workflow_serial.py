export const meta = {
  name: "neuro01_serial_execution",
  description: "Run remaining conditions serially, then cross-condition aggregation and verification for Neuro_01.",
  phases: [
    { title: "Phase 1: run remaining conditions serially" },
    { title: "Phase 2: cross-condition aggregation" },
    { title: "Phase 3: final artifact verification" }
  ]
}

// Serial execution: one child at a time, await each before the next.
// No parallel(), no concurrent pipeline, no Promise.all.

const WORKDIR = "/data/user/conversations/conv-5884203e-e071-45de-91eb-fc6c20f5e9d8/project/output/analysis";
const RESULTS = "/data/user/conversations/conv-5884203e-e071-45de-91eb-fc6c20f5e9d8/project/output/results";
const PY = "python3";

// apv and mix0 were already run and verified by the main agent.
const CONDITIONS = ["bic", "ctrl", "dzp", "mix50"];

const CONDITION_SCHEMA = {
  type: "object",
  properties: {
    condition: { type: "string" },
    success: { type: "boolean" },
    returncode: { type: "integer" },
    n_trials: { type: "integer" },
    r2: {
      type: "object",
      description: "map from model name to {r2_overall, r2_median_trial, n_trials, n_trials_above_0.05}"
    },
    files_written: { type: "array", items: { type: "string" } },
    errors: { type: "string" }
  },
  required: ["condition", "success", "returncode", "r2", "files_written"]
};

const CROSS_SCHEMA = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    returncode: { type: "integer" },
    cross_files: { type: "array", items: { type: "string" } },
    errors: { type: "string" }
  },
  required: ["success", "returncode", "cross_files"]
};

const VERIFY_SCHEMA = {
  type: "object",
  properties: {
    success: { type: "boolean" },
    conditions_present: { type: "array", items: { type: "string" } },
    summary_count: { type: "integer" },
    robustness_count: { type: "integer" },
    cross_count: { type: "integer" },
    missing: { type: "array", items: { type: "string" } },
    errors: { type: "string" }
  },
  required: ["success", "conditions_present", "summary_count", "robustness_count", "cross_count", "missing"]
};

// ---------- Phase 1: run each condition serially ----------
phase("Phase 1: run remaining conditions serially");

const conditionResults = [];
for (const cond of CONDITIONS) {
  log("=== child: run_condition " + cond + " (serial) ===");
  const prompt = [
    "You are a general-purpose scientific-execution child for task Neuro_01_self_organization_prediction.",
    "Run EXACTLY this command from the analysis directory and report its results. Do not modify the code.",
    "",
    "cd " + WORKDIR + " && " + PY + " run_condition.py " + cond,
    "",
    "After it completes, read the file " + RESULTS + "/" + cond + "_summary.csv and " +
    RESULTS + "/" + cond + "_robustness.csv (they should exist if success).",
    "",
    "Return a JSON object with the schema. 'r2' must be a map over the model names found in the summary CSV " +
    "(at minimum persist_r, s, o, s_o, r_s_o, s_self, s_o_s) each to an object with r2_overall, r2_median_trial, " +
    "n_trials, n_trials_above_0.05. 'files_written' must list the four CSVs and the JSON summary for this condition. " +
    "If the command fails, set success=false, returncode to the exit code, and put the real error text in 'errors'.",
  ].join("\n");
  const res = await agent(prompt, {
    label: "run_" + cond,
    phase: "Phase 1: run remaining conditions serially",
    schema: CONDITION_SCHEMA,
    effort: "low"
  });
  if (res) {
    conditionResults.push(res);
    log("  " + cond + " -> success=" + res.success + " rc=" + res.returncode +
        " r2(o)=" + (res.r2 && res.r2.o ? res.r2.o.r2_overall : "?"));
  } else {
    log("  " + cond + " -> child returned null (failed/skipped)");
    conditionResults.push({ condition: cond, success: false, returncode: -1, r2: {}, files_written: [] });
  }
}

// ---------- Phase 2: cross-condition aggregation ----------
phase("Phase 2: cross-condition aggregation");
log("=== child: cross_condition.py (serial) ===");
const crossPrompt = [
  "You are a general-purpose scientific-execution child for task Neuro_01_self_organization_prediction.",
  "Run EXACTLY this command from the analysis directory and report its results. Do not modify the code.",
  "",
  "cd " + WORKDIR + " && " + PY + " cross_condition.py",
  "",
  "After it completes, list the files it created in " + RESULTS + " that start with 'cross_condition_'.",
  "",
  "Return a JSON object with the schema. 'cross_files' must list the absolute paths of the cross-condition CSVs. " +
  "If the command fails, set success=false, returncode to the exit code, and put the real error text in 'errors'.",
].join("\n");
const cross = await agent(crossPrompt, {
  label: "cross",
  phase: "Phase 2: cross-condition aggregation",
  schema: CROSS_SCHEMA,
  effort: "low"
});
if (cross) {
  log("  cross -> success=" + cross.success + " rc=" + cross.returncode);
} else {
  log("  cross -> child returned null (failed/skipped)");
  cross = { success: false, returncode: -1, cross_files: [] };
}

// ---------- Phase 3: final artifact verification ----------
phase("Phase 3: final artifact verification");
log("=== child: verify all artifacts present ===");
const verifyPrompt = [
  "You are a general-purpose scientific-execution child for task Neuro_01_self_organization_prediction.",
  "Verify the results directory. Run these commands and report the counts. Do not modify any file.",
  "",
  "cd " + WORKDIR + " && echo '--- summary files ---' && ls -1 " + RESULTS + "/*_summary.csv 2>/dev/null | wc -l && " +
  "ls -1 " + RESULTS + "/*_summary.csv 2>/dev/null | sed 's/.*_summary.csv//' | sort",
  "",
  "echo '--- robustness files ---' && ls -1 " + RESULTS + "/*_robustness.csv 2>/dev/null | wc -l",
  "",
  "echo '--- cross files ---' && ls -1 " + RESULTS + "/cross_condition_*.csv 2>/dev/null | wc -l",
  "",
  "echo '--- expected conditions (apv,bic,ctrl,dzp,mix0,mix50) ---' && for c in apv bic ctrl dzp mix0 mix50; do " +
  "  if ls " + RESULTS + "/" + c + "_summary.csv >/dev/null 2>&1; then echo present: $c; else echo MISSING: $c; fi; done",
  "",
  "Return a JSON object with the schema. 'conditions_present' must list every condition whose summary CSV exists. " +
  "'missing' must list any of apv bic ctrl dzp mix0 mix50 that does NOT have a summary CSV. 'summary_count', " +
  "'robustness_count', 'cross_count' are the line counts from the commands above. set success=true only if all six " +
  "conditions are present and all three counts are >= 1.",
].join("\n");
const verify = await agent(verifyPrompt, {
  label: "verify",
  phase: "Phase 3: final artifact verification",
  schema: VERIFY_SCHEMA,
  effort: "low"
});

// ---------- Emit final summary ----------
log("=== WORKFLOW COMPLETE ===");
const summary = {
  conditions: conditionResults,
  cross: cross,
  verify: verify || { success: false, conditions_present: [], summary_count: 0, robustness_count: 0, cross_count: 0, missing: [] }
};
log("verify.success=" + (verify ? verify.success : false) +
    "  conditions_present=" + (verify ? verify.conditions_present.join(",") : "none") +
    "  missing=" + (verify ? verify.missing.join(",") || "none" : "unknown"));
return summary;
