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
Xd_so = np.column_stack([np.ones(n), Xs, Xo])  # (n, 35)

# joint fit of [1,s,o]
beta_joint, *_ = np.linalg.lstsq(Xd_so, Yr, rcond=None)
e_joint = Yr - Xd_so @ beta_joint  # (n, 64)

print("=== joint [1,s,o] residual ===")
print("cond(Xd_so)=%e" % np.linalg.cond(Xd_so))
print("mean(e_joint per ch) should be ~0:", np.abs(e_joint.mean(axis=0)).max())

# per-channel projection of channel 0 onto [1,s,o]
y0 = Yr[:, 0]
beta_pc, *_ = np.linalg.lstsq(Xd_so, y0, rcond=None)
proj_pc = Xd_so @ beta_pc
e_pc = y0 - proj_pc

print("joint e[:,0] vs per-channel e: max diff = %e" % np.max(np.abs(e_joint[:,0] - e_pc)))
print("SS_res joint ch0 = %.4f, per-ch ch0 = %.4f, diff = %.4f" % (
    float(np.sum(e_joint[:,0]**2)), float(np.sum(e_pc**2)), float(np.sum(e_joint[:,0]**2 - e_pc**2))))

# Now FWL per-channel: SS_res(y_j ~ [1,rj,s,o]) vs SS_res(e_j ~ [1,rj])
j = 0
Xr_j = np.column_stack([np.ones(n), Rj[:, j], Xs, Xo])  # 36 cols
beta_rso, *_ = np.linalg.lstsq(Xr_j, y0, rcond=None)
ss_res_rso = float(np.sum((y0 - Xr_j @ beta_rso)**2))

Xr2 = np.column_stack([np.ones(n), Rj[:, j]])
beta_e, *_ = np.linalg.lstsq(Xr2, e_joint[:, j], rcond=None)
ss_res_fwl = float(np.sum((e_joint[:, j] - Xr2 @ beta_e)**2))

ss_tot = float(np.sum((y0 - y0.mean())**2))
print("\n=== FWL identity for channel 0 ===")
print("SS_res(y~[1,rj,s,o]) direct = %.4f  R2=%.6f" % (ss_res_rso, 1-ss_res_rso/ss_tot))
print("SS_res(e~[1,rj])   FWL      = %.4f  R2=%.6f" % (ss_res_fwl, 1-ss_res_fwl/ss_tot))
print("identical? %s (diff=%.3e)" % (np.isclose(ss_res_rso, ss_res_fwl, rtol=1e-6, atol=1e-4), abs(ss_res_rso-ss_res_fwl)))

# aggregate over all channels to reproduce summary
ss_res_rso_arr = np.zeros(nch); ss_res_fwl_arr = np.zeros(nch); ss_tot_arr = np.sum((Yr-Yr.mean(axis=0))**2)
print("\n=== aggregating over 64 channels (direct vs FWL) ===")
for jj in range(nch):
    Xr_j = np.column_stack([np.ones(n), Rj[:, jj], Xs, Xo])
    b, *_ = np.linalg.lstsq(Xr_j, Yr[:, jj], rcond=None)
    ss_res_rso_arr[jj] = np.sum((Yr[:,jj] - Xr_j @ b)**2)
    Xr2 = np.column_stack([np.ones(n), Rj[:, jj]])
    b2, *_ = np.linalg.lstsq(Xr2, e_joint[:, jj], rcond=None)
    ss_res_fwl_arr[jj] = np.sum((e_joint[:,jj] - Xr2 @ b2)**2)
r2_rso_direct = 1 - ss_res_rso_arr/ss_tot_arr
r2_rso_fwl   = 1 - ss_res_fwl_arr/ss_tot_arr
print("mean R2 direct (r_s_o) = %.6f" % float(np.nanmean(r2_rso_direct)))
print("mean R2 FWL    (r_s_o) = %.6f" % float(np.nanmean(r2_rso_fwl)))
print("max |diff| = %.4f" % float(np.max(np.abs(r2_rso_direct - r2_rso_fwl))))
print("mean SS_res diff = %.4f (of ~%.4f)" % (float(np.nanmean(np.abs(ss_res_rso_arr-ss_res_fwl_arr))), float(np.nanmean(ss_res_rso_arr))))
