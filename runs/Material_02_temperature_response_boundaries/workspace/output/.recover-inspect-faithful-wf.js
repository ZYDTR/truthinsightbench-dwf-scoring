export const meta = {
  name: 'material-02-inspect-faithful-receipt',
  description: 'Serial single-child faithful Inspect receipt (scientific-notation ranges as strings) for Material_02',
  phases: [
    { title: 'Inspect-Faithful' },
  ],
}

const PROJECT = '/data/user/conversations/conv-d8a7104a-9926-4308-b4ac-cb93bc8e2977/project'
const OUT = PROJECT + '/output'
const DATA_DIR = PROJECT + '/data'

phase('Inspect-Faithful')

const prompt = `
You are producing the FINAL, faithful Inspect-phase receipt for TruthInsightBench
task Material_02_temperature_response_boundaries.

WHY THIS STEP: Two prior Inspect children each produced a schema-valid StructuredOutput,
but in BOTH, the ranges for the very small column "moment_std_err_emu" had their
scientific-notation exponents dropped when the child hand-wrote the JSON tool call
(e.g. 2.96323770988519e-08 became 2.96323770988519; 2.09e-08 became 2.09;
3.37e-08 became 3.37), so on the 300K file the reported min was greater than its max.
This is a TRANSCRIPTION error in the model's hand-written JSON tool-call input, NOT a
serialization error (the raw data, the child's own Python/Bash output, and a Python
json.dump file all retain full precision; only the hand-written StructuredOutput JSON
drops the exponents). Your job is to produce a StructuredOutput in which the
moment_std_err_emu ranges are EXACT, full-precision, and keep their scientific
notation.

HOW TO GUARANTEE EXACTNESS (do all of these):
- The frozen data files under ${DATA_DIR}/ are the only source of truth. Read them
  read-only with pandas (pd.read_csv(path, sep='\\t')).
- Compute every value in Python. For "moment_std_err_emu" (values ~1e-8 to 1e-5),
  convert each min/max to a STRING using Python's repr() of the float, so the
  scientific notation is preserved exactly. Example:
      f"{float_val!r}"  ->  '2.96323770988519e-08'
    Do NOT use str(round(...)) or any formatting that strips the exponent.
- In the StructuredOutput you return, put these EXACT STRINGS for the
  moment_std_err_emu "min" and "max". The Inspect schema declares "ranges" as a
  free object ({"type":"object"}) with no property constraints, so string values
  are fully schema-valid. For ALL OTHER columns keep min/max as numbers, exactly as
  before.
- Do NOT invent or hard-code any number. Every value must come from the data.
- DO NOT overwrite any existing file. You may (re)write the machine file
  ${OUT}/00_inspection_receipt.json (your own file, created in this continuation)
  and APPEND a clearly-marked section to ${OUT}/00_inspection.md. Preserve all
  existing content.

STEPS:
1. For each of the five files
  (bulk_magnetization_300K.tsv, bulk_magnetization_860K.tsv,
   bulk_magnetization_870K.tsv, electric_switching_corrected.tsv,
   electric_current_voltage.tsv), compute: filename, rows, columns, ranges (one
   {min,max} per numeric column), missing_values, field_direction ("ascending",
   "descending", or "loop").
2. For the three bulk files, the moment_std_err_emu min/max must be full-precision
   STRINGS with scientific notation (e.g. "2.96323770988519e-08"). Verify in
   Python that for every range min < max, and print the exact moment_std_err_emu
   strings to stdout.
3. Write the machine receipt to ${OUT}/00_inspection_receipt.json (numbers with
   full precision via json.dump). This file is the authoritative numeric receipt.
4. Return a JSON object via StructuredOutput that matches the machine receipt,
   EXCEPT: in the returned StructuredOutput, the moment_std_err_emu "min" and
   "max" are the full-precision STRINGS (e.g. "2.96323770988519e-08"), copied
   character-for-character from what you printed in step 2. Keep every other
   field identical to the machine receipt.
5. The "notes" field must state: this is the final faithful Inspect receipt; prior
   receipts (original Inspect child a52a2a4d544fda753 server-error; recovery
   wf_6979c8db-8a5; consistency wf_746eb039-d28) dropped scientific-notation
   exponents in the moment_std_err_emu ranges when hand-writing the StructuredOutput
   JSON tool-call input (a transcription error, not a serialization error); this
   receipt stores those ranges as full-precision strings to preserve the exact
   values, and the numeric machine receipt is 00_inspection_receipt.json; all prior
   receipts and journals are preserved as history.

Return schema (IDENTICAL to the original Inspect schema; ranges is a free object so
strings are allowed):
{
  "files": [
    {
      "filename": string,
      "rows": number,
      "columns": [string],
      "ranges": object,
      "missing_values": number,
      "field_direction": string
    }
  ],
  "notes": string
}

Only touch the five data files (read-only), (re)write ${OUT}/00_inspection_receipt.json,
and append to ${OUT}/00_inspection.md. Do not modify any other file.
`

const receiptResult = await agent(prompt, {
  label: 'inspect-faithful-receipt',
  phase: 'Inspect-Faithful',
  schema: {
    type: 'object',
    properties: {
      files: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            filename: { type: 'string' },
            rows: { type: 'number' },
            columns: { type: 'array', items: { type: 'string' } },
            ranges: { type: 'object' },
            missing_values: { type: 'number' },
            field_direction: { type: 'string' },
          },
        },
      },
      notes: { type: 'string' },
    },
    required: ['files'],
  },
})
log('Inspect faithful receipt complete')

return {
  inspection: receiptResult,
}
