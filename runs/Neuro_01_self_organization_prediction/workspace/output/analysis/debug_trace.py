import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
j = 0
y = Yr[:, j]
rj = Rz[:-lag, j]
Xso = np.column_stack([Xs, Xo])
Xd_so = np.column_stack([np.ones(n), Xso])  # [1, s, o]
Xd_rso = np.column_stack([np.ones(n), rj, Xso])  # [1, rj, s, o]

ss_tot = float(np.sum((y - y.mean())**2))
print("n=%d  SS_tot(y)=%.4f  mean(y)=%.3f std(y)=%.3f" % (n, ss_tot, y.mean(), y.std()))

# [1,s,o] model
b_so, *_ = np.linalg.lstsq(Xd_so, y, rcond=None)
res_so = y - Xd_so @ b_so
ss_res_so = float(np.sum(res_so*res_so))
r2_so = 1 - ss_res_so/ss_tot
print("[1,s,o]   ss_res=%.4f  R2=%.6f" % (ss_res_so, r2_so))

# [1,rj,s,o] model (direct)
b_rso, *_ = np.linalg.lstsq(Xd_rso, y, rcond=None)
res_rso = y - Xd_rso @ b_rso
ss_res_rso = float(np.sum(res_rso*res_rso))
r2_rso = 1 - ss_res_rso/ss_tot
print("[1,rj,s,o] ss_res=%.4f  R2=%.6f  (r2>=r2_so? %s)" % (ss_res_rso, r2_rso, r2_rso>=r2_so))
print("  ss_res_rso <= ss_res_so? %s  (FWL identity)" % (ss_res_rso <= ss_res_so))

# FWL: e = y - proj[1,s,o] y = res_so
e = res_so
Xd_rj = np.column_stack([np.ones(n), rj])
b_rj_e, *_ = np.linalg.lstsq(Xd_rj, e, rcond=None)
res_fwl = e - Xd_rj @ b_rj_e
ss_res_fwl = float(np.sum(res_fwl*res_fwl))
r2_fwl = 1 - ss_res_fwl/ss_tot
print("FWL [1,rj] on e:  ss_res=%.4f  R2=%.6f" % (ss_res_fwl, r2_fwl))
print("  ss_res_fwl == ss_res_rso? %s (diff=%.3e)" % (np.isclose(ss_res_fwl, ss_res_rso, rtol=1e-6, atol=1e-4), abs(ss_res_fwl-ss_res_rso)))
print("  ss_res_fwl <= ss_res_so? %s" % (ss_res_fwl <= ss_res_so))
