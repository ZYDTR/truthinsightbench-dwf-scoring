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
ss_tot = np.sum((Yr - Yr.mean(axis=0))**2)
Xd_so = np.column_stack([np.ones(n), Xs, Xo])

# joint [1,s,o] residual
beta_so, *_ = np.linalg.lstsq(Xd_so, Yr, rcond=None)
e = Yr - Xd_so @ beta_so
ss_so = np.sum(e*e, axis=0)
r2_so = 1 - ss_so/ss_tot

# joint [1,s,o,r] per-channel residual = e_j - P_[1,r_j] e_j  (FWL, shared s,o)
r2_rso = np.zeros(nch)
for j in range(nch):
    Xj = np.column_stack([np.ones(n), Rj[:, j]])
    b, *_ = np.linalg.lstsq(Xj, e[:, j], rcond=None)
    ss_res = np.sum((e[:,j] - Xj @ b)**2)
    r2_rso[j] = 1 - ss_res/ss_tot[j]

print("s_o  per-ch R2: mean=%.4f min=%.4f max=%.4f std=%.4f" % (r2_so.mean(), r2_so.min(), r2_so.max(), r2_so.std()))
print("r_s_o per-ch R2: mean=%.4f min=%.4f max=%.4f std=%.4f" % (r2_rso.mean(), r2_rso.min(), r2_rso.max(), r2_rso.std()))
print("s_o > 0.5: %.2f%% of channels" % (100*r2_so.sum(r2_so>0.5)/nch))
print("r_s_o > 0.5: %.2f%% of channels" % (100*r2_rso.sum(r2_rso>0.5)/nch))
print("r_s_o > 0.9: %.2f%% of channels" % (100*r2_rso.sum(r2_rso>0.9)/nch))
print()
print("Top 10 r_s_o per-ch R2:", np.sort(r2_rso)[-10:])
print("Bottom 10 r_s_o per-ch R2:", np.sort(r2_rso)[:10])
print("Top 10 s_o per-ch R2:", np.sort(r2_so)[-10:])
