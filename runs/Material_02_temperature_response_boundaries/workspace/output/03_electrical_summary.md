# Electrical Switching Data Analysis

## File 1: electric_switching_corrected.tsv (polarization-like signal)

- Points: 176
- Coordinate range: [-23.4667, 23.4667]
- Signal range: [-0.188460, 0.199729]
- Mean: 0.012150, Std: 0.156748, Min: -0.188460, Max: 0.199729
- Shape: loop
- Area under curve (open path): -9.128641
- Loop area (shoelace): 9.208509
- Hysteresis present: True
- Hysteresis mean abs diff (up vs down): 0.103072, max: 0.380433

## File 2: electric_current_voltage.tsv (current/voltage signal)

- Points: 176
- Coordinate range: [-176.0000, 176.0000]
- Signal range: [-184.5601, 165.2140]
- Mean: -0.9187, Std: 78.2661, Min: -184.5601, Max: 165.2140
- Shape: loop
- Area under curve (open path): 35673.6853
- Loop area (shoelace): 35703.6301
- Hysteresis present: True
- Hysteresis mean abs diff (up vs down): 61.9890, max: 176.4436

## Comparison

- Same-index correlation: -0.0011
- Coordinate scaling (current_voltage / switching): 7.5000
- Derivative/integral: same-index corr=-0.0011; on up-branch: y2 = 6279.5*d(y1)/dx + -1.6, R2=0.9976, corr(y2, d(y1)/dx)=0.9988 [strong derivative relation]; corr(y2, int(y1))=-0.9172 [weak/anti integral relation]; y2/y1 not constant (mean=-276.1, std=5531.1); loop-area integral ratio switching/current_voltage=0.000258
- Relationship: Both files are closed loops of 176 points sampled on the same coordinate sequence with x2 = 7.50 * x1 (current_voltage coordinate ~7.5x larger). Switching is a smooth normalized polarization-like loop (signal ~-0.188..0.200); current_voltage is a noisier raw electrical loop (signal ~-184.6..165.2). Same-index correlation is near zero (-0.0011) because the loop is traversed in the same direction, but on the monotonic ascending branch current_voltage is strongly proportional to the derivative of switching (y2 = 6279 * d(y1)/dx + -1.6, R2=0.9976), consistent with current ~ rate of change of polarization.

## Robustness

### switching
- Original loop area: 9.2085
- Trimmed 5%: 9.1709 (change -0.41%)
- Min-subtracted: 9.2085 (change +0.0000%)
- Mean-subtracted: 9.2085 (change +0.0000%)

### current_voltage
- Original loop area: 35703.6301
- Trimmed 5%: 36490.0503 (change +2.20%)
- Min-subtracted: 35703.6301 (change -0.0000%)
- Mean-subtracted: 35703.6301 (change -0.0000%)

## Findings

- Both electrical files are closed loops (coordinate ascends to a peak, sweeps negative, returns to ~0); hysteresis present in both.
- Switching loop area (shoelace) = 9.2085; current_voltage loop area = 35703.6301.
- Coordinates are proportional: current_voltage coordinate = 7.50 * switching coordinate (exact 7.50).
- Same-index signal correlation = -0.0011; signals are independent traces at the same coordinates, not linearly related.
- Switching is smooth/normalized (std 0.1567); current_voltage is noisy (std 78.3).
- Derivative/integral check: on the monotonic ascending branch, current_voltage is strongly proportional to the derivative of switching (y2 = 6279 * d(y1)/dx, R2=0.9976, corr=0.9988), consistent with current ~ rate of change of polarization; the integral relationship is weak (corr=-0.917).
- Background (mean/min) subtraction leaves the shoelace hysteresis area essentially unchanged (shift-invariant), confirming the loop-area measure is robust to DC offsets.
- Trimming 5% of the points changes the loop area by <3% for both files, indicating the hysteresis area is stable to endpoint effects.

## Notes

loop_area computed via shoelace closed-polygon area (shift-invariant). hysteresis_mean/max_abs_diff from up-branch vs down-branch (and return vs down-branch) interpolation. Files share identical 176-point coordinate sequence with a 7.5 scaling between them.
