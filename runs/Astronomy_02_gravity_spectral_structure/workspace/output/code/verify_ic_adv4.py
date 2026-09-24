"""Adversarial probe 4: slope-jump decomposition, second differences, JSON language audit."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

P = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project")
D = P / "data"
O = {}
uni = pd.read_csv(D / "uniform_alternative_comparison.tsv", sep="\t").sort_values("uniform_velocity_cm_s")
v = uni.uniform_velocity_cm_s.to_numpy(float); y = uni.information_criterion.to_numpy(float)
x = np.log10(v)
sl = np.diff(y) / np.diff(x)
dsl = np.diff(sl)
i_max_abs = int(np.argmax(np.abs(dsl)))
O["max_abs_slope_change"] = float(np.abs(dsl).max())
O["max_abs_slope_change_at_segment"] = [float(v[i_max_abs + 1]), float(v[i_max_abs + 2])]
O["max_slope_change_signed"] = float(dsl[i_max_abs])
O["largest_true_inversion_drop"] = float(dsl.min())
O["largest_true_inversion_at"] = [float(v[int(np.argmin(dsl)) + 1]), float(v[int(np.argmin(dsl)) + 2])]
O["second_diff_sign_changes"] = int(np.sum(np.diff(np.sign(np.diff(y, 2))) != 0))

# is np.diff(slopes) at the minimum a POSITIVE jump (recovery) rather than an inversion?
O["slope_at_seg_below_min"] = float(sl[7]); O["slope_at_seg_above_min"] = float(sl[8])
O["jump_at_min"] = float(sl[8] - sl[7])

# language audit of the track JSON
J = json.loads((P / "output/results/ic_comparison.json").read_text())
blob = json.dumps(J)
O["lang_counts"] = {t: int(blob.lower().count(t)) for t in
                    ["complete cartesian", "cartesian", "factorial", "complete grid",
                     "all possible", "lattice", "independently", "independent", "physical",
                     "uniqueness", "unique", "exact"]}

def strings(o, path=""):
    if isinstance(o, dict):
        for k, val in o.items():
            yield from strings(val, f"{path}.{k}")
    elif isinstance(o, list):
        for i, val in enumerate(o):
            yield from strings(val, f"{path}[{i}]")
    elif isinstance(o, str):
        yield path, o

O["strings_mentioning_unique"] = [(p, s) for p, s in strings(J) if "uniq" in s.lower() or "physical" in s.lower()]
O["strings_mentioning_cartesian"] = [(p, s) for p, s in strings(J) if "cartesian" in s.lower() or "factorial" in s.lower()]
O["strings_mentioning_grid_limited"] = [(p, s) for p, s in strings(J) if "grid-limited" in s.lower()]
O["top29_claim_location"] = [(p, s) for p, s in strings(J) if "top-29" in s]
O["delta100_claim_location"] = [(p, s) for p, s in strings(J) if "entire span" in s]
O["curvature_interpretation"] = [(p, s) for p, s in strings(J) if "factor 1.2" in s]
O["smoothness_claim"] = [(p, s) for p, s in strings(J) if "largest" in s and "IC/dex" in s]

print(json.dumps(O, indent=2, default=str))
Path(P / "output/results/_verify_ic_adv4_scratch.json").write_text(json.dumps(O, indent=2, default=str), encoding="utf-8")