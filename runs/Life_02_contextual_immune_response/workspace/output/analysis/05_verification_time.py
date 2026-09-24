#!/usr/bin/env python3
"""V5a-V5d verification: paired differences and time trend in time_resolved_response.tsv."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

RAW = "/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project/data/time_resolved_response.tsv"
OUT = "/data/user/conversations/conv-beae8ffb-0726-4214-9c89-748e1330ccf7/project/output/results/verification_time.json"

df = pd.read_csv(RAW, sep="\t")
errors = []

# ---- pivot: one row per (time, replicate), columns = condition_role ----
wide = df.pivot_table(index=["time_units", "replicate_id"],
                      columns="condition_role",
                      values="normalized_response_percent").reset_index()
wide["diff"] = wide["perturbation"] - wide["baseline"]
print("n pairs per time point:\n", wide.groupby("time_units").size().to_string())

# ---- V5a: per-time paired t-test on 4 differences ----
pt = {}
paired_sds = {}
for t, g in wide.groupby("time_units"):
    d = g["diff"].to_numpy()
    res = stats.ttest_1samp(d, 0.0)
    pt[float(t)] = dict(n=len(d), mean_diff=float(d.mean()),
                        sd=float(d.std(ddof=1)), t=float(res.statistic), p=float(res.pvalue))
    paired_sds[float(t)] = float(d.std(ddof=1))
ps = np.array([v["p"] for v in pt.values()])
print("\nper-time-point paired t-tests:")
for t, v in pt.items():
    print(f"  t={t:5.1f} n={v['n']} mean_diff={v['mean_diff']:+.6f} sd={v['sd']:.6f} "
          f"t={v['t']:+.6f} p={v['p']:.6f}")
min_p = float(ps.min())
min_t = float(np.array(list(pt.keys()))[ps.argmin()])
print("min p =", min_p, "at time", min_t)
claim_v5a = dict(min_p=min_p, min_p_time=min_t, all_p_ge_0_05=bool(np.all(ps >= 0.05)),
                 sig_at_0_05=[float(t) for t, v in pt.items() if v["p"] < 0.05])

# ---- V5b: largest |mean difference| ----
sds_all = np.array(list(paired_sds.values()))
max_t = max(pt, key=lambda t: abs(pt[t]["mean_diff"]))
print("\nlargest |mean diff| at time", max_t, "=", pt[max_t]["mean_diff"], "pp; t=", pt[max_t]["t"])
print("paired SD range:", sds_all.min(), "-", sds_all.max())

# ---- V5c: OLS slope per condition_role, all 28 obs each ----
slopes = {}
for role, g in df.groupby("condition_role"):
    X = sm.add_constant(g["time_units"].to_numpy(float))
    fit = sm.OLS(g["normalized_response_percent"].to_numpy(float), X).fit()
    slopes[role] = dict(slope=float(fit.params[1]), p=float(fit.pvalues[1]),
                        n=int(fit.nobs), r2=float(fit.rsquared))
    print(f"\nOLS {role}: slope={fit.params[1]:.6f} p={fit.pvalues[1]:.6g} n={int(fit.nobs)}")

# ---- V5d: mixed model time x condition interaction, random intercept for replicate_id ----
inter = None
try:
    form = "normalized_response_percent ~ time_units * C(condition_role)"
    md = smf.mixedlm(form, df, groups=df["replicate_id"])
    mf = md.fit(reml=True)
    print("\nmixed model summary:")
    print(mf.summary())
    keys = [k for k in mf.params.index if ":" in k]
    inter = dict(param=keys, value={k: float(mf.params[k]) for k in keys},
                 p={k: float(mf.pvalues[k]) for k in keys})
    print("interaction params:", inter)
except Exception as e:
    errors.append(f"mixedlm failed: {type(e).__name__}: {e}")
    print("\nMIXED MODEL FAILED REAL ERROR:", type(e).__name__, e)

# ---- assemble JSON ----
def chk(claim, recomputed, agrees, expression):
    return dict(claim=claim, recomputed=recomputed, agrees=bool(agrees), expression=expression)

per_time_out = {str(t): dict(n=v["n"], mean_diff_pp=round(v["mean_diff"], 6),
                             paired_sd=round(v["sd"], 6), t=round(v["t"], 6), p=round(v["p"], 6))
                for t, v in pt.items()}
checks = [
    chk("V5a: no time point significant at 0.05; all 7 p>=0.05; min p=0.506",
        dict(per_time=per_time_out, min_p=round(min_p, 6), min_p_time=min_t,
             all_p_ge_0_05=bool(np.all(ps >= 0.05))),
        bool(np.all(ps >= 0.05)) and abs(min_p - 0.506) < 0.02,
        "stats.ttest_1samp(perturbation-baseline diff per time_units, popmean=0)"),
    chk("V5b: largest |mean diff| at t=21 with -3.057 pp",
        dict(time=min_t if min_t == max_t else max_t, mean_diff_pp=round(pt[max_t]["mean_diff"], 6),
             t_stat=round(pt[max_t]["t"], 6)),
        max_t == 21.0 and abs(pt[max_t]["mean_diff"] - (-3.057)) < 0.01,
        "wide.groupby('time_units')['diff'].mean(); abs() argmax"),
]
for role, s in slopes.items():
    claimed = 0.8390 if role == "baseline" else 0.6945
    checks.append(chk(f"V5c: OLS slope {role} = {claimed}, p<0.001",
                      dict(slope=round(s["slope"], 6), p=s["p"]),
                      abs(s["slope"] - claimed) < 0.001 and s["p"] < 0.001,
                      f"sm.OLS(normalized_response_percent ~ time_units) within condition_role={role}, n=28"))
if inter:
    pv = list(inter["p"].values())[0]
    checks.append(chk("V5d: time x condition_role interaction p = 0.4304",
                      dict(p=round(pv, 6), params=inter["param"]),
                      abs(pv - 0.4304) < 0.001,
                      "smf.mixedlm('normalized_response_percent ~ time_units * C(condition_role)', groups=replicate_id).fit()"))
else:
    checks.append(chk("V5d: time x condition_role interaction p = 0.4304", None, False,
                      "mixedlm failed; see errors"))

conclusion = ("V5a: all 7 paired t-tests non-significant at 0.05. " if np.all(ps >= 0.05)
              else "V5a: at least one time point significant at 0.05. ")
conclusion += (f"min p={min_p:.4f} at time {min_t}. Largest |mean diff| = {pt[max_t]['mean_diff']:.4f} pp "
               f"at time {max_t} (t={pt[max_t]['t']:.3f}). ")
conclusion += ("Slopes: " + "; ".join(f"{r} {s['slope']:.4f} (p={s['p']:.2g})"
                                       for r, s in slopes.items()) + ". ")
if inter:
    conclusion += f"Interaction p={list(inter['p'].values())[0]:.4f}. "
conclusion += (f"Paired SDs range {sds_all.min():.4f}-{sds_all.max():.4f} pp, so a 3 pp difference is near "
               f"the noise floor; non-significance reflects low power (n=4).")

payload = dict(unit="time", checks=checks,
               paired_sds={str(k): round(v, 6) for k, v in paired_sds.items()},
               conclusion=conclusion, errors=errors)
with open(OUT, "w") as f:
    json.dump(payload, f, indent=2)
print("\nwrote", OUT)
print(json.dumps(payload, indent=2))