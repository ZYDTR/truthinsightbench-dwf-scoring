export const meta = {
  name: 'material-02-inspect-recovery',
  description: 'Serial single-child recovery of the Inspect phase StructuredOutput receipt for Material_02',
  phases: [
    { title: 'Inspect-Recovery' },
  ],
}

const PROJECT = '/data/user/conversations/conv-d8a7104a-9926-4308-b4ac-cb93bc8e2977/project'
const OUT = PROJECT + '/output'
const DATA_DIR = PROJECT + '/data'

phase('Inspect-Recovery')

const inspectPrompt = `
You are recovering the INSPECT phase of a TruthInsightBench magnetic-material
analysis (task Material_02_temperature_response_boundaries). The original
Inspect child failed mid-response with a Server error and never produced its
schema-conformant StructuredOutput receipt. Your job is to redo ONLY the data
inspection so that a valid StructuredOutput receipt exists.

Data files to inspect (tab-separated TSV) under ${DATA_DIR}/:
- bulk_magnetization_300K.tsv
- bulk_magnetization_860K.tsv
- bulk_magnetization_870K.tsv
- electric_switching_corrected.tsv
- electric_current_voltage.tsv

For EACH file, determine and report:
- filename (base name)
- rows: number of data rows EXCLUDING the header
- columns: array of the header column names, in order
- ranges: an object mapping each numeric column to {min, max}
- missing_values: total number of missing/NaN values across all columns
- field_direction: one of "ascending", "descending", or "loop" for the sweep/
  coordinate axis (the independent axis: field_Oe in the bulk files,
  sequence_coordinate in the electrical files). "loop" means the axis ascends
  to a peak then descends then ascends again, forming a closed hysteresis/
  switching loop.

Steps:
1. Read each file with pandas (pd.read_csv(path, sep='\\t')).
2. Compute row count, column list, per-column min/max, and total missing values.
3. Determine the sweep/coordinate direction by looking at the sign changes of
   the independent axis.
4. Write your findings to ${OUT}/00_inspection.md: do NOT delete or overwrite the
   existing content. APPEND a clearly-marked section at the end titled
   "## Recovery receipt (Inspect StructuredOutput)" that states (a) this is a
   recovery of the Inspect receipt that previously failed with a server error,
   (b) the per-file rows/columns/direction you computed, and (c) a timestamp.
   Preserve the original report above it unchanged.
5. Return a JSON object using the StructuredOutput tool that conforms EXACTLY
   to the schema below. Do not invent numbers; compute them from the frozen
   data. The returned JSON is the authoritative receipt.

The schema you must return (use StructuredOutput to emit it):
{
  "type": "object",
  "properties": {
    "files": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "filename": { "type": "string" },
          "rows": { "type": "number" },
          "columns": { "type": "array", "items": { "type": "string" } },
          "ranges": { "type": "object" },
          "missing_values": { "type": "number" },
          "field_direction": { "type": "string" }
        }
      }
    },
    "notes": { "type": "string" }
  },
  "required": ["files"]
}

Notes field: a brief prose note confirming the recovery, e.g. the original
Inspect child failed with a Server error and this receipt was produced by a
serial recovery child; state the per-file direction findings.

Only touch the five data files (read-only) and ${OUT}/00_inspection.md (append
only). Do not modify any other file.
`

const inspectResult = await agent(inspectPrompt, {
  label: 'inspect-data-recovery',
  phase: 'Inspect-Recovery',
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
log('Inspect recovery complete')

return {
  inspection: inspectResult,
}
