"""Scratch re-verification for Audit of REGISTERED FINDING 1."""
import json, numpy as np, os
ROOT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project"
R = os.path.join(ROOT, "output")
raw = {}
with open(os.path.join(ROOT, "data/conversion_timecourses.tsv")) as fh:
    fh.readline()
    for line in fh:
        fid, t, y = line.split()
        raw.setdefault(fid, ([], []))
        raw[fid][0].append(float(t)); raw[fid][1].append(float(y))
ylast = {f: raw[f][1][-1] for f in raw}
print("finals:", ylast, "span:", max(ylast.values())/min(ylast.values()))
for f in raw:
    t = np.array(raw[f][0]); y = np.array(raw[f][1])
    seg = np.diff(y)/np.diff(t)
    print(f, "seg", [round(s,4) for s in seg], "ratio", round(seg[-1]/seg[0],6))

# AICc arithmetic, k_total convention (fitted + variance), n=4
for fitted in (1,2,3):
    k = fitted+1
    print("n=4 fitted=%d k_total=%d n-k-1=%d AICc_defined=%s" % (fitted,k,4-k-1,4-k-1>0))
# n=5 corrections
for k in (2,3):
    print("n=5 k_total=%d corr=%.1f" % (k, 2*k*(k+1)/(5-k-1)))

# normalised RMS differences, both conventions
def rms(a, b, excl_t0):
    an, bn = a/a[-1], b/b[-1]
    d = an-bn
    if excl_t0: d = d[1:]
    return float(np.sqrt(np.mean(d**2)))
for pair in [("formulation_3","formulation_4"), ("formulation_1","formulation_3"), ("formulation_1","formulation_2")]:
    a = np.array(raw[pair[0]][1]); b = np.array(raw[pair[1]][1])
    print(pair, "incl t0:", round(rms(a,b,False),6), "excl t0:", round(rms(a,b,True),6))

# artifact-stored values
d = json.load(open(os.path.join(R, "verification/mechanism_fits_results.json")))
for g in ("group_1_2","group_3_4"):
    print(g, "stored normalised_rms_diff:", d["_group_shared_shape"][g]["normalised_rms_diff"])
kin = json.load(open(os.path.join(R, "results/timecourse_kinetics.json")))
print("runner-up deltas:", {f: sorted(v["aicc_delta_vs_best"].values())[1] for f,v in kin["kinetics"]["per_formulation"].items()})
print("null baselines:", kin["null_baselines"]["baseline_3_shared_shape_scaled"]["rss_total"],
      kin["null_baselines"]["baseline_1_common_linear_slope"]["per_curve_rss_total"])
print("F1 M2_exp rss:", d["formulation_1"]["fits"]["M2_exp"]["rss"],
      "M3_power rss:", d["formulation_1"]["fits"]["M3_power"]["rss"])
print("within2 sets:", {f: d[f]["models_within_2_aicc"] for f in d if f.startswith("formulation")})
print("8h:", json.dumps(d["_predictions_at_8h"], indent=1)[:1200])
