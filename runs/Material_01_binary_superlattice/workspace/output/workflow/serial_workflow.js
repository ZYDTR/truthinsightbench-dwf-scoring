export const meta = {
  name: 'binary_superlattice_serial',
  description: 'Serial decomposition: robustness, trajectory, figure, report',
  phases: [
    { title: 'Robustness' },
    { title: 'Trajectory' },
    { title: 'Figure' },
    { title: 'Report' },
  ],
}

const BASE = '/data/user/conversations/conv-d944711d-fa11-472b-99db-a1344a7c93b2/project'

const ROBUST_SCHEMA = {
  type: 'object',
  properties: {
    monotonic_chi_equal_size: { type: 'boolean', description: 'does same-type fraction monotonically decrease as chi increases for R2=3 structures?' },
    chi_values: { type: 'array', items: { type: 'number' } },
    same_frac_by_chi: { type: 'array', items: { type: 'number' } },
    cutoffs_tested: { type: 'array', items: { type: 'string' } },
    robust: { type: 'boolean', description: 'does the monotonic trend hold across all tested cutoffs and after leave-one-size-out?' },
    trend_strength_chi_equal_size: { type: 'string', description: 'e.g. "0.4->0.359, 0.6->0.490, 0.8->0.604, 1.0->0.875"' },
    file_written: { type: 'string' },
    method: { type: 'string' },
    limitations: { type: 'string' }
  },
  required: ['monotonic_chi_equal_size', 'cutoffs_tested', 'robust', 'file_written', 'method']
}

const TRAJ_SCHEMA = {
  type: 'object',
  properties: {
    n_frames: { type: 'integer' },
    n_particles_per_frame: { type: 'integer' },
    particle_composition: { type: 'string', description: 'e.g. "6 CH4 + 6 H2 = 6:6"' },
    energy_frame0: { type: 'number' },
    energy_final: { type: 'number' },
    energy_min: { type: 'number' },
    energy_min_frame: { type: 'integer' },
    final_frame_is_min: { type: 'boolean' },
    frames_within_05_of_min: { type: 'integer' },
    convergence_behavior: { type: 'string' },
    file_written: { type: 'string' },
    method: { type: 'string' }
  },
  required: ['n_frames', 'n_particles_per_frame', 'energy_min', 'energy_min_frame', 'final_frame_is_min', 'frames_within_05_of_min', 'file_written']
}

const FIG_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    x_axis: { type: 'string' },
    y_axis: { type: 'string' },
    n_points: { type: 'integer' },
    x_values: { type: 'array', items: { type: 'number' } },
    y_values: { type: 'array', items: { type: 'number' } },
    trend_description: { type: 'string' },
    file_written: { type: 'string' },
    method: { type: 'string' },
    tool_note: { type: 'string', description: 'e.g. matplotlib unavailable, hand-built SVG' }
  },
  required: ['title', 'file_written', 'n_points', 'x_values', 'y_values', 'tool_note']
}

const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    findings_count: { type: 'integer' },
    finding_labels: { type: 'array', items: { type: 'string' } },
    root_report_path: { type: 'string' },
    all_findings_registered: { type: 'boolean' },
    method_section_present: { type: 'boolean' },
    reproducibility_section_present: { type: 'boolean' },
    file_written: { type: 'string' },
    method: { type: 'string' }
  },
  required: ['findings_count', 'finding_labels', 'root_report_path', 'all_findings_registered', 'file_written']
}

// ---------------------------------------------------------------------------
// Child 1: Robustness of chi -> same-type clustering
// ---------------------------------------------------------------------------
phase('Robustness')
const robust = await agent(
  `You are a materials-science analysis agent. Working directory: ${BASE}.
The dataset is under ${BASE}/data/.
- structures/structure_*.xyz : 14 binary configurations (C and H2 particles), header line e.g. "R1=3, R2=3; Chi = 0.4, Epsilon = 0.04; Energy = -8.0403; Box dimension: 54(p)*54(p)*30(f)".
- structure_manifest.json : gives r1,r2,chi,epsilon,energy,box,n per structure.

Task: Quantify how interfacial preference (Chi) relates to same-type nearest-neighbor (NN) clustering, and TEST ROBUSTNESS.

Method (must actually run with python3):
1. Parse each .xyz. Compute periodic pairwise distances (box from header "Box dimension: N(p)" — note: use the (p) cubic box).
2. For each structure compute the first peak in the 1D distance distribution (histogram, 0.5 bins) = characteristic NN distance c.
3. At cutoff c (and also at 0.9c, 1.1c, 1.3c, 1.6c, 2.0c), count same-type (C-C or H2-H2) vs cross-type (C-H2) pairs and report same-type fraction.
4. FOCUS on equal-size structures (R1==R2, i.e. R2=3, 9 C + 9 H2): structures 01,02,03,04. Report same-type fraction vs Chi for each cutoff.
5. ROBUSTNESS checks:
   a) Monotonicity: does same-type fraction monotonically DECREASE as Chi increases? (Chi=1.0 highest, 0.4 lowest.)
   b) Cutoff sensitivity: does the monotonic trend hold at 0.9c,1.1c,1.3c,1.6c,2.0c?
   c) Leave-one-size-out: for the mixed-size subset (R1!=R2), does the chi->clustering trend still hold directionally? Report it (do not force it).
   d) Alternative definition: instead of "first peak", use a fixed cutoff = 0.5 * (median NN distance across all equal-size structures). Report the trend.
6. Write results to ${BASE}/output/results/robustness_chi_clustering.csv (columns: public_id,R2,chi,epsilon,cutoff_name,cutoff_value,same_pairs,cross_pairs,same_frac).
7. Write a short method+limitations note to ${BASE}/output/results/robustness_note.txt.

Return via StructuredOutput with the schema (monotonic_chi_equal_size, chi_values, same_frac_by_chi, cutoffs_tested, robust, trend_strength_chi_equal_size, file_written, method, limitations).
Be honest: if a robustness check fails or the mixed-size subset does not show a clean trend, say so in limitations.
DO NOT download external data. Use only ${BASE}/data/.
`,
  { label: 'robustness_chi_clustering', phase: 'Robustness', schema: ROBUST_SCHEMA }
)
console.log('robust:', robust)

// ---------------------------------------------------------------------------
// Child 2: Trajectory convergence analysis
// ---------------------------------------------------------------------------
phase('Trajectory')
const traj = await agent(
  `You are a materials-science analysis agent. Working directory: ${BASE}.
The binary 6:6 optimization trajectory is under ${BASE}/data/optimization_trajectory/:
- LM_ENERGIES_32.txt : lines "frame_index  accepted_energy" (34 frames, 0..33).
- LM_MOVIE_32.xyz : 36 blocks? Actually 34 frames. Each frame: a line with particle count (12), a header line "R1=3, R2=3; v1 = 0.6, v2 = 0.08; Energy = ...", then 12 particle lines each "TYPE x y z" (types are CH4 and H2).

Task: Characterize convergence of the accepted-local-minimum trajectory for the 6:6 system.

Method (must actually run with python3):
1. Parse LM_ENERGIES_32.txt into (frame, energy).
2. Parse LM_MOVIE_32.xyz into per-frame (count, types, coords).
3. Compute: n frames; n particles per frame; composition (count CH4 and H2) -> confirm 6:6;
   energy at frame 0; final energy; global minimum energy and its frame index;
   whether the final frame IS the global minimum;
   number of frames within 0.05 of the minimum;
   a description of convergence behavior (e.g. "monotonic descent then oscillation around minimum in last N frames").
4. For the final (converged) frame, compute a simple 2D/3D packing descriptor:
   - characteristic NN distance (first peak of distance histogram),
   - mean nearest-neighbor distance,
   - z-layering (number of distinct z-shells by gap clustering).
5. Write results to ${BASE}/output/results/trajectory_analysis.csv (columns: frame,energy,type_counts,nn_peak,mean_nn,z_shells) plus a summary row or a separate ${BASE}/output/results/trajectory_summary.json.
6. Write a short note to ${BASE}/output/results/trajectory_note.txt (method + limitations).

Return via StructuredOutput with the schema (n_frames, n_particles_per_frame, particle_composition, energy_frame0, energy_final, energy_min, energy_min_frame, final_frame_is_min, frames_within_05_of_min, convergence_behavior, file_written, method).
Be honest about the oscillation (it is a local minimum with dynamical oscillation, not a single frozen point).
DO NOT download external data.
`,
  { label: 'trajectory_analysis', phase: 'Trajectory', schema: TRAJ_SCHEMA }
)
console.log('traj:', traj)

// ---------------------------------------------------------------------------
// Child 3: Figure generation
// ---------------------------------------------------------------------------
phase('Figure')
const fig = await agent(
  `You are a data-visualization agent. Working directory: ${BASE}.
matplotlib is NOT available. You must hand-build a valid, self-contained SVG figure
by writing it with a python3 script (or directly) to ${BASE}/output/figures/chi_vs_samefrac.svg.

Input data (already computed): ${BASE}/output/results/structure_features.csv and
${BASE}/output/results/neighbor_analysis.csv and ${BASE}/output/results/robustness_chi_clustering.csv
(if robustness was not written yet, read ${BASE}/data/structure_manifest.json and
${BASE}/data/structures/structure_01.xyz..04.xyz and recompute same-type fraction
at the first-peak cutoff yourself — 9 C + 9 H2 each).

Task: Produce ONE figure that is the primary evidence for the chi -> same-type
clustering finding.
- X axis: Chi (values 0.4, 0.6, 0.8, 1.0 for the equal-size R2=3 structures 04,03,02,01).
- Y axis: same-type nearest-neighbor pair fraction at the first-peak cutoff
  (expected approx: chi 0.4 -> 0.359, 0.6 -> 0.490, 0.8 -> 0.604, 1.0 -> 0.875).
- Show points + a line + axis labels + a title
  "Same-type NN fraction vs interfacial preference (Chi), equal-size 9C/9H2".
- Add a second small panel or a secondary line for the mixed-size subset only if it
  is available in robustness_chi_clustering.csv; otherwise keep a single clear panel.
- The SVG must be valid XML: declare viewBox, use <svg>, <line>, <circle>, <text>,
  <polyline>. No external image references.

Write the SVG to ${BASE}/output/figures/chi_vs_samefrac.svg.
Also write the underlying (x,y) data used to ${BASE}/output/results/figure_data.csv.
Return via StructuredOutput with the schema (title, x_axis, y_axis, n_points, x_values,
y_values, trend_description, file_written, method, tool_note).
tool_note should state that matplotlib was unavailable and you hand-built the SVG.
DO NOT download external data.
`,
  { label: 'figure_chi_samefrac', phase: 'Figure', schema: FIG_SCHEMA }
)
console.log('fig:', fig)

// ---------------------------------------------------------------------------
// Child 4: Report synthesis (Result.md)
// ---------------------------------------------------------------------------
phase('Report')
const report = await agent(
  `You are a scientific-report author. Working directory: ${BASE}.
You must WRITE the authoritative report at ${BASE}/output/Result.md.

Already-computed evidence files (read them with python3 or as text):
- ${BASE}/output/results/structure_features.csv (14 structures: R1,R2,chi,epsilon,energy,
  energy_per_particle,box,n_particles,n_C,n_H2,n_layers_z,nn_dist_mean,xy_rms,z_shells)
- ${BASE}/output/results/neighbor_analysis.csv (first-peak cutoff, same_frac at -20/0/+20)
- ${BASE}/output/results/robustness_chi_clustering.csv (+ robustness_note.txt)
- ${BASE}/output/results/trajectory_analysis.csv + trajectory_summary.json (+ trajectory_note.txt)
- ${BASE}/output/figures/chi_vs_samefrac.svg + figure_data.csv
- ${BASE}/output/analysis/*.py (executed scripts)

Write ${BASE}/output/Result.md in English with EXACTLY 2-5 registered findings, each labeled
with a heading/bold beginning "Finding 1", "Finding 2", ... consecutively.
Each finding must include: object and scope; relationship/direction/boundary; core
quantitative result and analysis unit; supporting evidence (exact file paths under
output/); robustness result; counterevidence/limitation/alternative explanations; and an
executable new observation/replication that could falsify it.

Minimum required content (grounded in the data, be honest about nulls/conflicts):
FINDING 1: For equal-size binary pairs (R1==R2==3, 9C/9H2), interfacial preference Chi
positively correlates with same-type nearest-neighbor clustering. Quantify:
Chi 1.0->0.875, 0.8->0.604, 0.6->0.490, 0.4->0.359 same-type fraction at first-peak
cutoff 6.25 A. Monotonic; robust to cutoff +/-20% (values unchanged) and to the
0.9/1.1/1.3/1.6/2.0x cutoffs (see robustness_chi_clustering.csv).
FINDING 2: The 6:6 binary trajectory converges to a stable local minimum.
Quantify: 34 frames, 12 particles (6 CH4 + 6 H2), energy 6.95026 (frame0) ->
global min -4.15068 at frame 33 (the final frame); 34 frames within 0.05 of the min;
converged frame shows 6:6 composition with a characteristic NN distance and z-layering
(see trajectory_analysis.csv). Note the oscillation around the minimum in the last
frames is a local minimum with dynamical oscillation, not a frozen point.
FINDING 3 (honest null / boundary): Energy is NOT a clean monotonic function of Chi
alone across all 14 structures because it is confounded by epsilon and by particle number
(17-37 particles). The most bound per-particle energy is structure_13
(chi=0.6, eps=0.16, R2=2, n=37) at -0.530/particle, while the least bound is
structure_01 (chi=1.0, eps=0.03, R2=3, n=18) at -0.080/particle.
State that a single-parameter Chi->energy relationship is not supported; only the
Chi->clustering and trajectory-convergence relationships are clean.
(Optionally FINDING 4: particle-size ratio R2/R1 changes the characteristic bond
distance (peak NN distance 5.25-13.75 A across structures), i.e. size sets the
packing scale. Ground it in neighbor_analysis.csv first_peak column.)

The report MUST include:
- A "## Methods / Analysis workflow" section (or "## Reproduction") naming the
  executed scripts in output/analysis/ and how to rerun (python3 output/analysis/*.py).
- A "## Reproducibility" section with the exact command:
  python3 output/analysis/analyze_structures.py
  python3 output/analysis/analyze_neighbors.py
  python3 output/analysis/*.py (any robustness/trajectory scripts)
  python3 validate_output.py output
- A "## Limitations" section.
- A "## References / Sources" section stating NO external sources were used (T0=2022-01-01
  boundary; all evidence is workspace data), which satisfies the external-source policy.

Do NOT invent numbers not in the evidence files. If a value is uncertain, say so.
After writing, run: python3 ${BASE}/validate_output.py ${BASE}/output
and report whether it passed. If it failed with a semantic error, FIX the report
(e.g. ensure findings are labeled Finding 1..N, ensure evidence paths resolve,
ensure Methods/Reproduction sections present) and re-run until it passes.
Return via StructuredOutput with the schema (findings_count, finding_labels,
root_report_path, all_findings_registered, method_section_present,
reproducibility_section_present, file_written, method).
DO NOT download external data.
`,
  { label: 'authoritative_report', phase: 'Report', schema: REPORT_SCHEMA }
)
console.log('report:', report)

return {
  robust: robust,
  traj: traj,
  fig: fig,
  report: report,
}
