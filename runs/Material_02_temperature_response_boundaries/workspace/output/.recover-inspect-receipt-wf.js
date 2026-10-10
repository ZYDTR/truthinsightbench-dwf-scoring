export const meta = {
  name: 'material-02-inspect-receipt-consistency',
  description: 'Serial single-child consistency-corrected Inspect receipt for Material_02 (faithful ranges)',
  phases: [
    { title: 'Inspect-Receipt' },
  ],
}

const PROJECT = '/data/user/conversations/conv-d8a7104a-9926-4308-b4ac-cb93bc8e2977/project'
const OUT = PROJECT + '/output'
const DATA_DIR = PROJECT + '/data'

phase('Inspect-Receipt')

const prompt = `
You are completing a CONSISTENCY-CORRECTION of the Inspect-phase receipt for
TruthInsightBench task Material_02_temperature_response_boundaries.

CONTEXT (do not repeat, just act on it): A prior Inspect recovery child produced
a schema-conformant StructuredOutput, but in that receipt the ranges for the
very small column "moment_std_err_emu" had their scientific-notation exponents
dropped during transcription (e.g. a value of 2.96323770988519e-08 was written
as 2.96323770988519; 2.09e-08 as 2.09; 3.37e-08 as 3.37), so on at least one
file the reported min was greater than its max. The underlying data computation
was correct; only the receipt's numbers were transcribed incorrectly. Your job
is to produce a NEW faithful receipt whose ranges exactly match the frozen raw
data, retaining full float precision (including scientific notation).

STRICT RULES:
- The frozen data files under ${DATA_DIR}/ are the only source of truth. Read them
  read-only with pandas (pd.read_csv(path, sep='\\t')).
- Do NOT overwrite any existing file. You may CREATE a new file
  ${OUT}/00_inspection_receipt.json and APPEND a clearly-marked section to
  ${OUT}/00_inspection.md. Preserve everything that already exists.
- Do not invent or hard-code any number. Compute every value from the data.
- The moment_std_err_emu values are very small (order 1e-8 to 1e-5). You MUST
  retain scientific notation (e-08, e-06, e-05, etc.). Do NOT drop the exponents.
  A value like 2.96323770988519e-08 must stay 2.96323770988519e-08, never become
  2.96323770988519.

STEPS:
1. For each of the five files
  (bulk_magnetization_300K.tsv, bulk_magnetization_860K.tsv,
   bulk_magnetization_870K.tsv, electric_switching_corrected.tsv,
   electric_current_voltage.tsv), compute:
   - filename (base name)
   - rows (data rows excluding header)
   - columns (header names, in order)
   - ranges: for each numeric column, {min, max} computed from the data
   - missing_values (total NaN count across columns)
   - field_direction: the sweep/coordinate axis (field_Oe in bulk files,
     sequence_coordinate in electrical files) is one of "ascending",
     "descending", or "loop" (closed hysteresis/switching loop).
2. Build the receipt dict and WRITE it to ${OUT}/00_inspection_receipt.json using
   Python's json.dump (this preserves full float precision, e.g. e-08). This file
   is the authoritative machine receipt.
3. Run a VERIFICATION step in Python that:
   - reads back the file,
   - asserts that for EVERY range, min < max,
   - for the three bulk files specifically, prints the exact moment_std_err_emu
     min and max with full precision,
   - prints "VERIFIED: all ranges min<max" only if every assertion passes.
   If any assertion fails, fix the computation before continuing.
4. Print the full receipt JSON to stdout (so the correct numbers with exponents
   are present in the tool result).
5. Return a JSON object via the StructuredOutput tool that EXACTLY mirrors the
   files array you wrote to ${OUT}/00_inspection_receipt.json. When writing the
   ranges, COPY the numbers VERBATIM from the printed JSON above, INCLUDING the
   e-notation. Double-check that min < max for every range, especially the small
   moment_std_err_emu values. The notes field should state: this is a consistency-
   corrected Inspect receipt; a prior receipt (workflow wf_6979c8db-8a5) dropped
   scientific-notation exponents in the moment_std_err_emu ranges during
   transcription, so some min>max; this receipt is faithful to the frozen raw data
   (ranges copied with full precision); the prior receipt and all journals are
   preserved as history.

Return schema (use StructuredOutput):
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
Only touch the five data files (read-only), create ${OUT}/00_inspection_receipt.json,
and append to ${OUT}/00_inspection.md. Do not modify any other file.
`

const receiptResult = await agent(prompt, {
  label: 'inspect-receipt-consistency',
  phase: 'Inspect-Receipt',
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
log('Inspect receipt consistency correction complete')

return {
  inspection: receiptResult,
}
