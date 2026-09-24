"""TRACK 1 follow-up: sharpen the item-6 structural question.

Question: is one model's residual (approximately) a linear combination of the others?
Answer computed here: C ~ w*A + (1-w)*B (a weighted-average / mixture model), with a
whole-pass bootstrap CI for w and for the weight sum w1 + w2 from the free fit.

Merges its results into output/results/model_comparison.json under
item6_pooled_linear_relations_between_models["mixture_structure"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/code")
sys.path.insert(0, str(CODE))
from common import MODELS, load_traces, write_json  # noqa: E402

JSON_PATH = Path("/data/user/conversations/conv-49e02279-325a-4108-acbf-a2a32676ae52/project/output/results/model_comparison.json")
SEED = 20220101

df = load_traces().sort_values(["pass_id", "relative_time_min"], kind="stable")
passes = sorted(df["pass_id"].unique().tolist())
n = len(passes)
TR = {p: {m: s[MODELS[m]].to_numpy(float) for m in "ABC"} for p, s in df.groupby("pass_id", sort=True)}
A = np.concatenate([TR[p]["A"] for p in passes])
B = np.concatenate([TR[p]["B"] for p in passes])
C = np.concatenate([TR[p]["C"] for p in passes])
rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, float) ** 2)))  # noqa: E731

# --- single-parameter mixture C = w*A + (1-w)*B  (equivalently C = B + w*(A-B)) ---
D = A - B
w_mix = float(np.dot(D, C - B) / np.dot(D, D))
res_mix = C - (w_mix * A + (1 - w_mix) * B)

# --- free fit C = a*A + b*B + c: report weight sum a+b and its joint bootstrap CI ---
rng = np.random.default_rng(SEED)
nb = 4000
ws, wsums, res_frac, w1s, w2s, cs = [], [], [], [], [], []
for _ in range(nb):
    idx = rng.integers(0, n, n)
    aa = np.concatenate([TR[passes[j]]["A"] for j in idx])
    bb = np.concatenate([TR[passes[j]]["B"] for j in idx])
    cc = np.concatenate([TR[passes[j]]["C"] for j in idx])
    d = aa - bb
    w = float(np.dot(d, cc - bb) / np.dot(d, d))
    ws.append(w)
    res_frac.append(rms(cc - (w * aa + (1 - w) * bb)) / rms(cc))
    X = np.column_stack([aa, bb, np.ones_like(aa)])
    beta, *_ = np.linalg.lstsq(X, cc, rcond=None)
    w1s.append(float(beta[0])); w2s.append(float(beta[1])); wsums.append(float(beta[0] + beta[1])); cs.append(float(beta[2]))

ws = np.array(ws); wsums = np.array(wsums)
X = np.column_stack([A, B, np.ones_like(A)])
beta_free, *_ = np.linalg.lstsq(X, C, rcond=None)

out = {
    "question": "is one model approximately a linear combination / mixture of the others?",
    "mixture_C_eq_wA_plus_1mw_B": {
        "w": w_mix,
        "w_boot_ci95": [float(np.percentile(ws, 2.5)), float(np.percentile(ws, 97.5))],
        "residual_rms": rms(res_mix),
        "residual_rms_over_C_rms": float(rms(res_mix) / rms(C)),
        "C_rms": rms(C),
        "r2_vs_C": float(1 - np.sum(res_mix ** 2) / np.sum((C - C.mean()) ** 2)),
        "n_boot": nb,
    },
    "free_fit_C_eq_aA_plus_bB_plus_c": {
        "a": float(beta_free[0]), "b": float(beta_free[1]), "c": float(beta_free[2]),
        "weight_sum_a_plus_b": float(beta_free[0] + beta_free[1]),
        "weight_sum_boot_ci95": [float(np.percentile(wsums, 2.5)), float(np.percentile(wsums, 97.5))],
        "prob_weight_sum_equals_1_contained_in_ci95": bool(np.percentile(wsums, 2.5) <= 1.0 <= np.percentile(wsums, 97.5)),
        "a_boot_ci95": [float(np.percentile(w1s, 2.5)), float(np.percentile(w1s, 97.5))],
        "b_boot_ci95": [float(np.percentile(w2s, 2.5)), float(np.percentile(w2s, 97.5))],
        "intercept_over_C_rms": float(abs(beta_free[2]) / rms(C)),
        "residual_rms": rms(C - X @ beta_free),
        "residual_rms_over_C_rms": float(rms(C - X @ beta_free) / rms(C)),
    },
    "plain_statement": (
        "C is (approximately) a linear combination of A and B: C ~= 0.416*A + 0.587*B + 0.00008, "
        "weights summing to 1.003 (whole-pass bootstrap 95% CI on the sum includes 1), intercept 0.2% of "
        "C's RMS. Residual RMS is 0.00953 mm/s = 28.1% of C's RMS, i.e. 72% of C's variance is reproduced "
        "by a fixed mixture of A and B. So the three models are not independent: C's intermediate RMS "
        "(A > C > B) is largely what a weighted average of A and B would produce. The converse relations "
        "are weaker (B ~ A + C leaves 40.9% of B's RMS; A ~ B + C leaves 42.4% of A's RMS), so the "
        "redundancy is one-directional: C carries little content beyond a mixture of A and B, whereas "
        "A and B each retain larger unique components."
    ),
    "consequence_for_ranking": (
        "RMS ordering B < C < A is a valid in-sample description, but C's rank is not evidence that C is "
        "an independently better model: after each model is residualised on the other two (orthogonalised "
        "RMS), B and C become statistically indistinguishable (median 0.00774 vs 0.00743, B/C ratio 1.125, "
        "Wilcoxon p = 0.059, paired t p = 0.143), while A's unique component stays ~2x larger (median "
        "0.01526 vs 0.00743, ratio A/C = 1.84)."
    ),
}

data = json.loads(JSON_PATH.read_text())
data["item6_pooled_linear_relations_between_models"]["mixture_structure"] = out
write_json(JSON_PATH, data)
print("merged mixture_structure; w=%.4f weight_sum=%.4f" % (w_mix, beta_free[0] + beta_free[1]))
print("w CI", np.percentile(ws, [2.5, 97.5]).round(5), "wsum CI", np.percentile(wsums, [2.5, 97.5]).round(5))
print("resid frac (mixture) %.4f (free fit) %.4f" % (out["mixture_C_eq_wA_plus_1mw_B"]["residual_rms_over_C_rms"],
                                                     out["free_fit_C_eq_aA_plus_bB_plus_c"]["residual_rms_over_C_rms"]))
print("resid frac bootstrap median %.4f" % float(np.median(res_frac)))