import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Oz_l = Oz[:-lag]; Yr = Rz[lag:]
print("shapes: Oz_l", Oz_l.shape, "Yr", Yr.shape)

# o-alone: model [1, o] -> Yr  (batch, 33 cols)
Xd = np.column_stack([np.ones(n), Oz_l])
print("Xd shape", Xd.shape, "cond(Xd)=%e" % np.linalg.cond(Xd))

# 1) QR
Q, R = np.linalg.qr(Xd)
R1 = R[:33,:33]
bproj = Q[:, :33] @ np.linalg.solve(R1, Q[:, :33].T @ Yr)
ssr_qr = np.sum((Yr - bproj)**2, axis=0)
ss_tot = np.sum((Yr - Yr.mean(axis=0))**2)
r2_qr = 1 - ssr_qr/ss_tot
print("QR   o-alone: SSRmean=%.4f  R2mean=%.6f  R2min=%.6f  R2max=%.6f" %
      (ssr_qr.mean(), r2_qr.mean(), r2_qr.min(), r2_qr.max()))

# 2) lstsq
beta, *_ = np.linalg.lstsq(Xd, Yr, rcond=None)
ssr_ls = np.sum((Yr - Xd @ beta)**2, axis=0)
r2_ls = 1 - ssr_ls/ss_tot
print("lstsq o-alone: SSRmean=%.4f  R2mean=%.6f  R2min=%.6f  R2max=%.6f" %
      (ssr_ls.mean(), r2_ls.mean(), r2_ls.min(), r2_ls.max()))

# 3) Gram solve (small)
XtX = Xd.T @ Xd; XtY = Xd.T @ Yr
print("Gram cond=%e" % np.linalg.cond(XtX))
beta_g, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
ssr_g = np.sum((Yr - Xd @ beta_g)**2, axis=0)
r2_g = 1 - ssr_g/ss_tot
print("Gram o-alone: SSRmean=%.4f  R2mean=%.6f  R2min=%.6f  R2max=%.6f" %
      (ssr_g.mean(), r2_g.mean(), ssr_g.mean(), r2_g.min(), r2_g.max()))

# 4) simple regression via np.poly? no. Check correlation of one channel.
# compute R2 via correlation for channel 0: r2 = corr(Yr[:,0], Oz_l[:,0])**2
c0 = np.corrcoef(Yr[:,0], Oz_l[:,0])[0,1]
print("channel0 corr(o,r)=%.6f -> R2=%.6f" % (c0, c0**2))
