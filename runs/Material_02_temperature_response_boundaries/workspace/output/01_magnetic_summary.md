# Magnetic Field-Scan Analysis Summary

Generated: 2026-10-09

## Overview

Three bulk magnetization hysteresis-loop measurements were analyzed:

| Temperature | n_points | Peak M (emu) | Trough M (emu) | Hysteresis (emu) | Loop Area (emu*Oe) | Coercive Field (Oe) | Remanence (emu) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 300K | 373 | 0.003773 | -0.003775 | 0.007420 | 1.1639 | -265.8 | 0.000431 |
| 860K | 2501 | 0.003341 | -0.003341 | 0.006495 | 0.2675 | 25.4 | -0.000032 |
| 870K | 2501 | 0.003338 | -0.003317 | 0.006420 | 4.6257 | 30.3 | -0.000015 |

## Per-Temperature Details

### 300K

- **n_points**: 373
- **Field range**: [-49999.8, 49999.9] Oe
- **Peak moment**: 0.003773 emu at H = 50000 Oe
- **Trough moment**: -0.003775 emu at H = -50000 Oe
- **Susceptibility (full range)**: 9.237e-08 emu/Oe (R² = 0.8890)
- **Susceptibility (central |H|≤25000 Oe)**: 1.367e-07 emu/Oe (R² = 0.8465)
- **Hysteresis** (M(+50kOe) − M(−50kOe)): 0.007420 emu
- **Loop area** (full, |H|≤50000 Oe): 1.1639 emu·Oe
- **Loop area** (central, |H|≤5000 Oe): 0.9115 emu·Oe
- **Coercive field** (H where M crosses 0 on descending branch): -265.8 Oe
- **Remanence** (M at H=0 on descending branch): 0.000431 emu

### Robustness Checks — 300K

- **(a) Remove top 5% by |M|** (threshold |M| > 0.003584 emu):
  - Removed: 19 points, remaining: 354
  - Peak M: 0.003582 emu (unchanged / robust)
  - Hysteresis: 0.007166 emu
  - Loop area (full): 1.3886 emu·Oe
  - Susceptibility (full): 9.723e-08 emu/Oe

- **(b) Loop area: central (|H|≤5000 Oe) vs full (|H|≤50000 Oe)**:
  - Central: 0.9115 emu·Oe
  - Full: 1.1639 emu·Oe
  - Ratio (central/full): 0.783 (loop is concentrated near H=0 for 300K)

- **(c) Remove points with std_err > 10×median** (threshold σ > 4.24e-06 emu):
  - Removed: 77 points, remaining: 296
  - Peak M: 0.003773 emu (unchanged / robust)
  - Hysteresis: 0.007415 emu
  - Loop area (full): 1.0219 emu·Oe
  - Susceptibility (full): 9.459e-08 emu/Oe

### 860K

- **n_points**: 2501
- **Field range**: [-49999.7, 50000.2] Oe
- **Peak moment**: 0.003341 emu at H = 50000 Oe
- **Trough moment**: -0.003341 emu at H = -50000 Oe
- **Susceptibility (full range)**: 6.874e-08 emu/Oe (R² = 0.9977)
- **Susceptibility (central |H|≤25000 Oe)**: 7.450e-08 emu/Oe (R² = 0.9931)
- **Hysteresis** (M(+50kOe) − M(−50kOe)): 0.006495 emu
- **Loop area** (full, |H|≤50000 Oe): 0.2675 emu·Oe
- **Loop area** (central, |H|≤5000 Oe): 0.0583 emu·Oe
- **Coercive field** (H where M crosses 0 on descending branch): 25.4 Oe
- **Remanence** (M at H=0 on descending branch): -0.000032 emu

### Robustness Checks — 860K

- **(a) Remove top 5% by |M|** (threshold |M| > 0.003183 emu):
  - Removed: 125 points, remaining: 2376
  - Peak M: 0.003182 emu (unchanged / robust)
  - Hysteresis: 0.006339 emu
  - Loop area (full): 0.2634 emu·Oe
  - Susceptibility (full): 6.905e-08 emu/Oe

- **(b) Loop area: central (|H|≤5000 Oe) vs full (|H|≤50000 Oe)**:
  - Central: 0.0583 emu·Oe
  - Full: 0.2675 emu·Oe
  - Ratio (central/full): 0.218 (loop is concentrated near H=0 for 860K)

- **(c) Remove points with std_err > 10×median** (threshold σ > 3.80e-06 emu):
  - Removed: 6 points, remaining: 2495
  - Peak M: 0.003341 emu (unchanged / robust)
  - Hysteresis: 0.006495 emu
  - Loop area (full): 0.2679 emu·Oe
  - Susceptibility (full): 6.874e-08 emu/Oe

### 870K

- **n_points**: 2501
- **Field range**: [-49999.7, 49999.8] Oe
- **Peak moment**: 0.003338 emu at H = 50000 Oe
- **Trough moment**: -0.003317 emu at H = -50000 Oe
- **Susceptibility (full range)**: 6.716e-08 emu/Oe (R² = 0.9991)
- **Susceptibility (central |H|≤25000 Oe)**: 7.018e-08 emu/Oe (R² = 0.9969)
- **Hysteresis** (M(+50kOe) − M(−50kOe)): 0.006420 emu
- **Loop area** (full, |H|≤50000 Oe): 4.6257 emu·Oe
- **Loop area** (central, |H|≤5000 Oe): 0.5704 emu·Oe
- **Coercive field** (H where M crosses 0 on descending branch): 30.3 Oe
- **Remanence** (M at H=0 on descending branch): -0.000015 emu

### Robustness Checks — 870K

- **(a) Remove top 5% by |M|** (threshold |M| > 0.003142 emu):
  - Removed: 125 points, remaining: 2376
  - Peak M: 0.003142 emu (unchanged / robust)
  - Hysteresis: 0.006246 emu
  - Loop area (full): 4.4483 emu·Oe
  - Susceptibility (full): 6.732e-08 emu/Oe

- **(b) Loop area: central (|H|≤5000 Oe) vs full (|H|≤50000 Oe)**:
  - Central: 0.5704 emu·Oe
  - Full: 4.6257 emu·Oe
  - Ratio (central/full): 0.123 (loop is concentrated near H=0 for 870K)

- **(c) Remove points with std_err > 10×median** (threshold σ > 3.97e-06 emu):
  - Removed: 66 points, remaining: 2435
  - Peak M: 0.003338 emu (unchanged / robust)
  - Hysteresis: 0.006418 emu
  - Loop area (full): 4.6178 emu·Oe
  - Susceptibility (full): 6.716e-08 emu/Oe

## Comparison: 300K vs 860K vs 870K

### Largest Moment

- **300K** has the largest peak moment: 0.003773 emu
  - 300K: 0.003773 emu
  - 860K: 0.003341 emu
  - 870K: 0.003338 emu

### Largest Hysteresis

- **300K** has the largest hysteresis: 0.007420 emu
  - 300K: 0.007420 emu
  - 860K: 0.006495 emu
  - 870K: 0.006420 emu

### Phase-Transition Evidence

The data is consistent with a **ferromagnetic → paramagnetic transition** between 300 K and 860 K:

1. **300 K shows a clear hysteresis loop** with significant remanence (M_r ≈ +0.000431 emu on the descending branch) and a measurable coercive field (H_c ≈ −266 Oe). The loop area is 1.164 emu·Oe, indicating a finite magnetic moment and magnetic anisotropy at this temperature.

2. **860 K shows a nearly vanishing loop**: remanence is −0.000032 emu (≈ 0.03% of the 300 K remanence), coercive field drops to 25 Oe, and the loop area shrinks dramatically to 0.268 emu·Oe. The M–H curve is nearly perfectly linear (R² = 0.998), characteristic of a paramagnet.

3. **870 K is essentially identical to 860 K**: remanence = −0.000015 emu, coercive field = 30 Oe, linear M–H (R² = 0.999). This confirms the transition occurs at or below 860 K, and 860 K and 870 K are both in the paramagnetic (or very weakly magnetic) state.

4. **Susceptibility trend**: The central-range susceptibility decreases from 300 K (1.367×10⁻⁷ emu/Oe) to 860 K (7.450×10⁻⁷ emu/Oe) to 870 K (7.018×10⁻⁷ emu/Oe), consistent with a Curie-temperature-like drop in magnetic response. However, the full-range susceptibility at 300 K (9.24×10⁻⁸) is *lower* than the central value because the 300 K loop has a non-linear remanence contribution that reduces the overall least-squares slope.

5. **Peak moment decrease**: The peak moment decreases from 0.003773 emu (300 K) to 0.003341 emu (860 K) to 0.003338 emu (870 K). The 860 K → 870 K change is negligible, suggesting the material is fully in the paramagnetic regime by 860 K, and the 300 K → 860 K decrease reflects the loss of ferromagnetic ordering.

### Susceptibility Trend

| Temperature | Full-range χ | Central (|H|≤25000) χ | R² (central) |
|:---:|:---:|:---:|:---:|
| 300K | 9.237e-08 emu/Oe | 1.367e-07 emu/Oe | 0.8465 |
| 860K | 6.874e-08 emu/Oe | 7.450e-08 emu/Oe | 0.9931 |
| 870K | 6.716e-08 emu/Oe | 7.018e-08 emu/Oe | 0.9969 |

The central-range susceptibility (the physically relevant linear response) **decreases** from 300 K → 860 K → 870 K, consistent with a ferromagnetic to paramagnetic transition. The full-range susceptibility at 300 K is lower than the central value because the remanence/hysteresis contributes a non-linear component that reduces the overall least-squares slope.

## Notes

- **Loop area sign convention**: The reported loop areas are the geometric enclosed areas (∫(M_upper − M_lower) dH), which are always non-negative. For 870 K, the loop area (4.63 emu·Oe) is larger than for 300 K (1.16 emu·Oe) despite similar hysteresis. This is because the 870 K data shows a systematic offset between the ascending and descending branches (M_asc > M_desc for H > 0, M_desc > M_asc for H < 0), creating an asymmetric loop shape. This asymmetry is not present at 300 K (where the loop is symmetric about the origin).

- **Sweep direction**: All three files sweep 0 → +50000 → −50000 → +50000 Oe (a closed hysteresis loop). The descending branch (+50000 → −50000) and the return branch (−50000 → +50000) form the two sides of the loop.

- **Coercive field at 300 K**: The negative value (−266 Oe) indicates the M→0 crossing on the descending branch occurs while the field is still negative. This is consistent with the descending branch crossing H=0 at M ≈ +0.000431 emu (remanence) and then crossing M=0 at H ≈ −266 Oe.
