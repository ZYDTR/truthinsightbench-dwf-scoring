import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
Rj = Rz[:-lag]
nch = Yr.shape[1]
ss_tot_total = float(np.sum((Yr - Yr.mean(axis=0))**2))
print("n=%d nch=%d SS_tot_total=%.4f" % (n, nch, ss_tot_total))

def joint_r2(X, Y):
    """Joint OLS: X (n,p) shared features -> Y (n,nch). Returns total R2 and per-channel mean R2."""
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    res = Y - Xd @ beta
    ss_res_total = float(np.sum(res*res))
    ss_res_ch = np.sum(res*res, axis=0)
    ss_tot_ch = np.sum((Y - Y.mean(axis=0))**2)
    r2_ch = 1 - ss_res_ch/ss_tot_ch
    return 1 - ss_res_total/ss_tot_total, float(np.nanmean(r2_ch)), ss_res_total

# joint models (correct, shared coefficients, all channels)
r2_s_tot, r2_s_ch, ss_s = joint_r2(Xs, Yr)
r2_o_tot, r2_o_ch, ss_o = joint_r2(Xo, Yr)
r2_so_tot, r2_so_ch, ss_so = joint_r2(np.column_stack([Xs,Xo]), Yr)
r2_rso_tot, r2_rso_ch, ss_rso = joint_r2(np.column_stack([Rj, Xs, Xo]), Yr)
print("joint total R2:  s=%.4f o=%.4f s_o=%.4f r_s_o=%.4f" % (r2_s_tot, r2_o_tot, r2_so_tot, r2_rso_tot))
print("joint ch-mean R2: s=%.4f o=%.4f s_o=%.4f r_s_o=%.4f" % (r2_s_ch, r2_o_ch, r2_so_ch, r2_rso_ch))
print("SS_res: s=%.4f o=%.4f s_o=%.4f r_s_o=%.4f" % (ss_s, ss_o, ss_so, ss_rso))
print("monotonic total: s_o>=o? %s  r_s_o>=s_o? %s  r_s_o>=s? %s" % (
    r2_so_tot>=r2_o_tot, r2_rso_tot>=r2_so_tot, r2_rso_tot>=r2_s_tot))

# Now the FWL per-channel path (what the code does), aggregated as the code does:
# per-channel r2_rso = 1 - SS_res(e_ch ~ [1,rj]) / SS_tot_Yr_ch
#   where e = Yr - proj[1,s,o]Yr (joint fit residual)
def fwl_rso():
    Xso = np.column_stack([Xs, Xo])
    Xd_so = np.column_stack([np.ones(n), Xso])
    beta_so, *_ = np.linalg.lstsq(Xd_so, Yr, rcond=None)
    e = Yr - Xd_so @ beta_so   # joint residual
    r2_list = []
    for j in range(nch):
        Xj = np.column_stack([np.ones(n), Rj[:, j]])
        beta_j, *_ = np.linalg.lstsq(Xj, e[:, j], rcond=None)
        ss_res_j = float(np.sum((e[:,j] - Xj @ beta_j)**2))
        ss_tot_j = float(np.sum((Yr[:,j] - Yr[:,j].mean())**2))
        r2_list.append(1 - ss_res_j/ss_tot_j)
    return float(np.nanmean(r2_list)), e
fwl_rso_mean, e = fwl_rso()
# per-channel SS_res of [1,rj,s,o] from FWL = SS_res(e_ch~[1,rj])
print("FWL r_s_o ch-mean R2 = %.4f  (join total R2 of r_s_o = %.4f)" % (fwl_rso_mean, r2_rso_tot))
print("FWL r_s_o >= s_o ch-mean? %s   FWL r_s_o >= s ch-mean? %s" % (
    fwl_rso_mean >= r2_so_ch, fwl_rso_mean >= r2_s_ch))

# Compare total SSR of FWL path vs joint lstsq for r_s_o:
# SS_res total of [1,rj,s,o] = SS_res total of [1,s,o] + SS_res total of [1,rj] on e (FWL theorem, per-channel additive)
ss_res_fwl_total = float(np.sum(e*e, axis=0))  # placeholder; compute properly
# per channel
ss_res_fwl_perch = []
for j in range(nch):
    Xj = np.column_stack([np.ones(n), Rj[:, j]])
    beta_j, *_ = np.linalg.lstsq(Xj, e[:, j], rcond=None)
    ss_res_fwl_perch.append(float(np.sum((e[:,j] - Xj @ beta_j)**2)))
ss_res_fwl_total = float(np.sum(ss_res_fwl_perch))
print("\nFWL total SS_res of [1,rj,s,o] = %.4f   joint lstsq total SS_res = %.4f" % (ss_res_fwl_total, ss_rso))
print("FWL identity (SS_res equal)? %s (diff=%.3e)" % (np.isclose(ss_res_fwl_total, ss_rso, rtol=1e-6, atol=1e-2), abs(ss_res_fwl_total-ss_rso)))
