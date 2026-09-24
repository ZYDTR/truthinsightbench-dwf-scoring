# Animal-level vs unit-level condition effects (Neuro_03_spatial_code_stability)

## Design verification
- Distinct animals: **13**
- Reference animals: **6** (animal_01, animal_05, animal_06, animal_11, animal_12, animal_13)
- Matrix-perturbed animals: **7** (animal_02, animal_03, animal_04, animal_07, animal_08, animal_09, animal_10)
- Animals appearing in BOTH conditions: **0** (none)
No animal is in both conditions; animals are the independent analysis unit and unit-session rows are nested observations. Primary tests use per-animal means followed by two-sample tests across animals. Unit-level tests are descriptive only (nested) and must not be treated as independent samples.

## Method
- Animal-level (primary): per-animal mean within each condition, then Welch two-sample t-test (unequal variance), Welch-Satterthwaite 95% CI, Cohen's d (pooled SD), and a combined-pool permutation test (seed 20260922, 10000 permutations, two-sided on |diff|, p = (count+1)/(n_perm+1)).
- Unit-level (descriptive, nested): Welch t-test over all observation rows.
- Robustness: whole-animal leave-out (LOAO); for each animal left out, the animal-level mean difference is recomputed and flagged if its sign flips relative to the full-sample diff.

## Results

### average_rate_hz
- Ref animal mean: 1.8642 (n=6 animals); perturbed animal mean: 1.7928 (n=7 animals).
- Animal-level diff (pert - ref): -0.0714; Welch p = 0.7852; permutation p = 0.5988; Cohen's d = -0.149; 95% CI [-0.6416, 0.4988].
- Unit-level (nested, descriptive) diff: -0.1099; Welch p = 0.686; Cohen's d = -0.069.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 3/13 animals flipped the sign of the animal-level diff when removed (full diff = -0.0714).

### out_of_field_rate_hz
- Ref animal mean: 1.0352 (n=6 animals); perturbed animal mean: 1.1424 (n=7 animals).
- Animal-level diff (pert - ref): 0.1073; Welch p = 0.5839; permutation p = 0.5633; Cohen's d = 0.298; 95% CI [-0.3279, 0.5424].
- Unit-level (nested, descriptive) diff: 0.1718; Welch p = 0.399; Cohen's d = 0.147.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 1/13 animals flipped the sign of the animal-level diff when removed (full diff = 0.1073).

### in_field_rate_hz
- Ref animal mean: 3.7612 (n=6 animals); perturbed animal mean: 3.0397 (n=7 animals).
- Animal-level diff (pert - ref): -0.7215; Welch p = 0.1308; permutation p = 0.005599; Cohen's d = -0.893; 95% CI [-1.6942, 0.2512].
- Unit-level (nested, descriptive) diff: -0.8090; Welch p = 0.05631; Cohen's d = -0.325.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 0/13 animals flipped the sign of the animal-level diff when removed (full diff = -0.7215).

### spatial_regularity_score
- Ref animal mean: 0.6230 (n=6 animals); perturbed animal mean: 0.6091 (n=7 animals).
- Animal-level diff (pert - ref): -0.0139; Welch p = 0.8676; permutation p = 0.7049; Cohen's d = -0.102; 95% CI [-0.2047, 0.1769].
- Unit-level (nested, descriptive) diff: -0.0247; Welch p = 0.5962; Cohen's d = -0.091.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 4/13 animals flipped the sign of the animal-level diff when removed (full diff = -0.0139).

### information_rate
- Ref animal mean: 0.5352 (n=6 animals); perturbed animal mean: 0.3087 (n=7 animals).
- Animal-level diff (pert - ref): -0.2264; Welch p = 0.07378; permutation p = 9.999e-05; Cohen's d = -1.268; 95% CI [-0.4822, 0.0293].
- Unit-level (nested, descriptive) diff: -0.2812; Welch p = 0.0001459; Cohen's d = -0.653.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 0/13 animals flipped the sign of the animal-level diff when removed (full diff = -0.2264).

### spacing
- Ref animal mean: 0.4309 (n=6 animals); perturbed animal mean: 0.4428 (n=7 animals).
- Animal-level diff (pert - ref): 0.0119; Welch p = 0.8144; permutation p = 0.7835; Cohen's d = 0.131; 95% CI [-0.0971, 0.1209].
- Unit-level (nested, descriptive) diff: -0.0019; Welch p = 0.904; Cohen's d = -0.022.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: False).
- Whole-animal leave-out: 3/13 animals flipped the sign of the animal-level diff when removed (full diff = 0.0119).

### orientation
- Ref animal mean: 24.7647 (n=6 animals); perturbed animal mean: 37.8780 (n=7 animals).
- Animal-level diff (pert - ref): 13.1133; Welch p = 0.1947; permutation p = 0.0442; Cohen's d = 0.789; 95% CI [-7.9570, 34.1836].
- Unit-level (nested, descriptive) diff: 13.3827; Welch p = 6.134e-06; Cohen's d = 0.807.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 0/13 animals flipped the sign of the animal-level diff when removed (full diff = 13.1133).

### speed_modulation_score
- Ref animal mean: 0.0871 (n=6 animals); perturbed animal mean: 0.0614 (n=7 animals).
- Animal-level diff (pert - ref): -0.0257; Welch p = 0.3537; permutation p = 0.212; Cohen's d = -0.522; 95% CI [-0.0844, 0.0330].
- Unit-level (nested, descriptive) diff: -0.0160; Welch p = 0.203; Cohen's d = -0.229.
- Unit-level effect SURVIVES at the animal level: **False** (animal-level p < 0.05: False; sign matches unit-level: True).
- Whole-animal leave-out: 0/13 animals flipped the sign of the animal-level diff when removed (full diff = -0.0257).
