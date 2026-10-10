#!/usr/bin/env python3
"""Master analysis for Math_03_ode_parameter_geometry (ground truth)."""
import json, os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT  = os.path.dirname(os.path.abspath(__file__))
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pt = pd.read_csv(os.path.join(DATA, "parameter_trajectories.tsv"), sep="\t",
                 dtype={"parameter_index":"int64","solver_family_id":"int64","progress_index":"int64"})
sv = pd.read_csv(os.path.join(DATA, "singular_value_spectrum.tsv"), sep="\t",
                 dtype={"mode_index":"int64"}).sort_values("mode_index").reset_index(drop=True)
ms = pd.read_csv(os.path.join(DATA, "measurement_simulation_pairs.tsv"), sep="\t")

m  = sv["mode_index"].to_numpy()
ls = sv["log_singular_value"].to_numpy()
n  = len(ls)
svv = np.exp(ls)

# --- spectrum fits ---
A  = np.vstack([np.ones_like(m), m]).T
c, *_ = np.linalg.lstsq(A, ls, rcond=None)
pred = c[0] + c[1]*m
r2_lin = 1 - np.sum((ls-pred)**2)/np.sum((ls-ls.mean())**2)
A2 = np.vstack([np.ones_like(m), np.log1p(m)]).T
c2, *_ = np.linalg.lstsq(A2, ls, rcond=None)
pred2 = c2[0] + c2[1]*np.log1p(m)
r2_loglog = 1 - np.sum((ls-pred2)**2)/np.sum((ls-ls.mean())**2)

def effdim(t): return int(np.sum(ls > t))
energy = svv**2
cum = np.cumsum(energy)/energy.sum()
k95, k99 = int(np.searchsorted(cum, 0.95)+1), int(np.searchsorted(cum, 0.99)+1)
early, late = int(n*0.05), int(n*0.8)
def slope(a,b):
    c,*_ = np.linalg.lstsq(np.vstack([np.ones(len(a)),a]).T,b,rcond=None); return float(c[1])
slope_early, slope_late = slope(m[:early], ls[:early]), slope(m[late:], ls[late:])

# --- trajectories ---
piv = pt.pivot(index="parameter_index", columns=("solver_family_id","progress_index"),
               values="normalised_distance_like_value").sort_index()
t = pd.DataFrame(index=piv.index)
for f in (0,1):
    for d in (0,1,2):
        t[f"f{f}d{d}"] = piv[(f,d)]
t["param"] = t.index
svf = sv.set_index("mode_index")["log_singular_value"]
t["log_sv"] = t["param"].map(svf)
for f in (0,1):
    t[f"tot{f}"]     = t[f"f{f}d2"] - t[f"f{f}d0"]
    t[f"slope{f}"]   = (t[f"f{f}d2"] - t[f"f{f}d0"])/2.0
    t[f"conv{f}"]    = t[f"tot{f}"] < 0
    t[f"div{f}"]     = t[f"tot{f}"] > 0
    t[f"flat{f}"]    = t[f"tot{f}"].abs() < 0.01

def fam(f):
    return {"mean_total":float(t[f"tot{f}"].mean()), "median_total":float(t[f"tot{f}"].median()),
            "frac_conv":float(t[f"conv{f}"].mean()), "frac_div":float(t[f"div{f}"].mean()),
            "frac_flat":float(t[f"flat{f}"].mean()),
            "mean_d0":float(t[f"f{f}d0"].mean()), "mean_d2":float(t[f"f{f}d2"].mean()),
            "std_total":float(t[f"tot{f}"].std())}

c0, c1 = float(np.corrcoef(t["slope0"], t["log_sv"])[0,1]), float(np.corrcoef(t["slope1"], t["log_sv"])[0,1])
ct0, ct1 = float(np.corrcoef(t["tot0"], t["log_sv"])[0,1]), float(np.corrcoef(t["tot1"], t["log_sv"])[0,1])
all_s = np.concatenate([t["slope0"].values, t["slope1"].values])
all_sv = np.concatenate([t["log_sv"].values, t["log_sv"].values])
corr_pool = float(np.corrcoef(all_s, all_sv)[0,1])
X = np.column_stack([np.ones_like(all_sv), all_sv])
reg = np.linalg.lstsq(X, np.concatenate([t["tot0"].values, t["tot1"].values]), rcond=None)[0]
# leave-one-family-out robustness: recompute pooled correlation excluding one family
def pooled_corr(drop):
    if drop==0: s, v = t["slope1"].values, t["log_sv"].values
    else: s, v = t["slope0"].values, t["log_sv"].values
    return float(np.corrcoef(s, v)[0,1])
lof0 = pooled_corr(1)  # without family 0 -> family1 only
lof1 = pooled_corr(0)  # without family 1 -> family0 only

# --- measurement ---
err = ms["measurement_like_value"] - ms["simulation_like_value"]
rel = (ms["measurement_like_value"] - ms["simulation_like_value"]) / ms["simulation_like_value"]
def winsor(e, p=(5,95)):
    lo, hi = np.percentile(e, p); return e.clip(lo, hi)
res = {
    "n_pairs": int(len(ms)),
    "err_mean": float(err.mean()), "err_median": float(err.median()), "err_std": float(err.std()),
    "err_abs_mean": float(np.abs(err).mean()), "err_abs_std": float(np.abs(err).std()),
    "err_iqr": float(np.percentile(err,75)-np.percentile(err,25)),
    "err_p05": float(np.percentile(err,5)), "err_p95": float(np.percentile(err,95)),
    "err_winsor_mean": float(winsor(err).mean()), "err_winsor_std": float(winsor(err).std()),
    "rel_mean": float(np.nanmean(rel)), "rel_median": float(np.nanmedian(rel)), "rel_std": float(np.nanstd(rel)),
    "sim_mean": float(ms["simulation_like_value"].mean()), "sim_std": float(ms["simulation_like_value"].std()),
    "meas_mean": float(ms["measurement_like_value"].mean()),
    "corr_meas_sim": float(np.corrcoef(ms["measurement_like_value"], ms["simulation_like_value"])[0,1]),
    "frac_err_abs_lt_0.1": float((np.abs(err)<0.1).mean()),
    "frac_err_abs_lt_0.5": float((np.abs(err)<0.5).mean()),
    "frac_rel_abs_lt_0.5": float((np.abs(rel)<0.5).mean()),
}
t_stat = float(np.mean(err) / (np.std(err)/np.sqrt(len(err))))
from math import erf
p_two = float(1 - erf(t_stat/np.sqrt(2)))
p_one = float(0.5*(1 - erf(t_stat/np.sqrt(2))))
res["t_stat"] = t_stat; res["p_two_sided"] = p_two; res["p_one_sided"] = p_one

# --- solver-specific diff (bootstrap) ---
rng = np.random.default_rng(42); N = len(t); boot=[]
for _ in range(3000):
    idx = rng.integers(0, N, N)
    boot.append(t.loc[idx,"tot0"].mean()-t.loc[idx,"tot1"].mean())
boot = np.array(boot)
diff = float(t["tot0"].mean()-t["tot1"].mean())
ci = [float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))]
p_sign_test = float(np.mean(boot>0) if diff>0 else np.mean(boot<0))
perparam = t["tot0"]-t["tot1"]
p_sign_perparam = float(np.mean(perparam>0))

# --- write outputs ---
results = {
  "meta": {"n_parameters": int(pt.parameter_index.nunique()), "n_families": int(pt.solver_family_id.nunique()),
           "n_progress": int(pt.progress_index.nunique()), "n_modes": int(sv.mode_index.nunique()),
           "n_pairs": int(len(ms)), "log_sv_base": "natural", "join": "parameter_index==mode_index"},
  "spectrum": {"log_sv_min": float(ls.min()), "log_sv_max": float(ls.max()),
     "span_ln": float(ls[0]-ls[-1]), "span_orders10": float((ls[0]-ls[-1])/np.log(10)),
     "linear_slope": float(c[1]), "linear_R2": float(r2_lin), "loglog_slope": float(c2[1]),
     "loglog_R2": float(r2_loglog), "effdim": {str(t): effdim(t) for t in (0.0,-5.0,-10.0,-20.0,-50.0,-80.0)},
     "energy_k95": k95, "energy_k99": k99, "slope_early5": float(slope_early), "slope_late20": float(slope_late)},
  "trajectories": {"family0": fam(0), "family1": fam(1),
     "corr_slope_sv_fam0": c0, "corr_slope_sv_fam1": c1, "corr_tot_sv_fam0": ct0, "corr_tot_sv_fam1": ct1,
     "corr_pool_slope_sv": corr_pool, "reg_total_on_sv_slope": float(reg[1]),
     "lof0_without_fam0": lof0, "lof1_without_fam1": lof1},
  "measurement_simulation": res,
  "solver_specific": {"diff_mean_tot_fam0_minus_fam1": diff, "bootstrap_ci95": ci,
     "sign_test_p_one_sided": p_sign_test, "frac_param_fam0_bigger": float(np.mean(perparam>0))},
}
with open(os.path.join(OUT,"results.json"),"w") as f: json.dump(results, f, indent=2, default=float)

sv.to_csv(os.path.join(OUT,"result_tables_spectrum.tsv"), sep="\t", index=False)
t.to_csv(os.path.join(OUT,"result_tables_trajectories.tsv"), sep="\t", index=False)
pd.DataFrame({"observation_index": ms["observation_index"], "error": err.to_numpy(),
              "rel_error": rel.to_numpy()}).to_csv(os.path.join(OUT,"result_tables_pairs.tsv"), sep="\t", index=False)
pd.DataFrame([{"item":"n_parameters","value":int(pt.parameter_index.nunique())},
   {"item":"n_modes","value":int(sv.mode_index.nunique())},{"item":"n_pairs","value":int(len(ms))},
   {"item":"spectrum_span_orders10","value":round(float((ls[0]-ls[-1])/np.log(10)),3)},
   {"item":"spectrum_linear_R2","value":round(float(r2_lin),6)},{"item":"spectrum_linear_slope_per_mode","value":round(float(c[1]),6)},
   {"item":"effdim_threshold0","value":effdim(0.0)},{"item":"energy_k95","value":k95},
   {"item":"fam0_frac_conv","value":round(float(t["conv0"].mean()),4)},{"item":"fam1_frac_conv","value":round(float(t["conv1"].mean()),4)},
   {"item":"corr_slope_sv_fam0","value":round(c0,4)},{"item":"corr_slope_sv_fam1","value":round(c1,4)},
   {"item":"err_mean","value":round(float(err.mean()),4)},{"item":"err_abs_mean","value":round(float(np.abs(err).mean()),4)},
   {"item":"t_stat","value":round(t_stat,4)},{"item":"p_two_sided","value":round(p_two,4)},
   {"item":"fam0_minus_fam1_mean_total","value":round(diff,5)}]).to_csv(os.path.join(OUT,"result_tables_summary.tsv"), sep="\t", index=False)

# --- figures ---
fig, ax = plt.subplots(figsize=(8,4))
ax.loglog(m+1, svv, "k.", ms=1, label="SV")
ax.loglog(m+1, np.exp(c[0]+c[1]*m), "-", label="linear fit")
ax.loglog(m+1, np.exp(c2[0]+c[1]*np.log1p(m)), "--", label="power-law fit")
ax.set_xlabel("mode index"); ax.set_ylabel("singular value"); ax.legend(); ax.set_title("Singular-value spectrum")
fig.tight_layout(); fig.savefig(os.path.join(FIG,"spectrum.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(t["param"][:500], t["log_sv"][:500], s=2, label="log SV")
ax2 = ax.twinx()
ax2.scatter(t["param"][:500], t["slope0"][:500], s=2, color="tab:blue", alpha=0.4, label="slope fam0")
ax2.scatter(t["param"][500:1000], t["slope1"][500:1000], s=2, color="tab:red", alpha=0.4, label="slope fam1")
ax.set_xlabel("parameter index"); ax.set_ylabel("log singular value"); ax2.set_ylabel("trajectory slope")
ax.set_title("Parameter index vs log SV & trajectory slope"); fig.tight_layout()
fig.savefig(os.path.join(FIG,"trajectory.png"), dpi=120); plt.close(fig)

bins = np.linspace(ls.min(), ls.max(), 20)
conv0b, conv1b, xs = [], [], []
for i in range(19):
    mask = (t["log_sv"]>=bins[i]) & (t["log_sv"]<bins[i+1])
    xs.append(0.5*(bins[i]+bins[i+1]))
    if mask.sum()>0:
        conv0b.append(t.loc[mask,"conv0"].mean()); conv1b.append(t.loc[mask,"conv1"].mean())
    else:
        conv0b.append(np.nan); conv1b.append(np.nan)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(xs, conv0b, "o-", label="fam0 conv frac"); ax.plot(xs, conv1b, "s-", label="fam1 conv frac")
ax.set_xlabel("log singular value"); ax.set_ylabel("fraction convergent"); ax.legend()
ax.set_title("Convergence fraction vs log SV"); fig.tight_layout()
fig.savefig(os.path.join(FIG,"convergence_vs_sv.png"), dpi=120); plt.close(fig)

print("WROTE", OUT)
print(json.dumps(results, indent=2, default=float))
