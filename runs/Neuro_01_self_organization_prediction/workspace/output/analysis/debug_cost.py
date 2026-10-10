import numpy as np, time
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG

t = load_conditions(conditions=["bic"])["bic"][0]
N = t["s"].shape[0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
n = Yr.shape[0]
Xso = np.column_stack([Xs, Xo])  # (n,34)
Xd = np.column_stack([np.ones(n), Xso])  # (n,35)

# measure lstsq on full design matrix (robust reference)
t0 = time.time(); beta = np.linalg.lstsq(Xd, Yr, rcond=None)[0]; t1 = time.time()
print("lstsq full X (n x 35) -> Yr: %.2fs" % (t1-t0))

# measure Gram solve
XtX = Xd.T @ Xd; XtY = Xd.T @ Yr
t0 = time.time(); betag = np.linalg.solve(XtX, XtY); t1 = time.time()
print("Gram solve (35x35): %.4fs" % (t1-t0))
print("cond(Gram):", np.linalg.cond(XtX))
print("cond(Xd):", np.linalg.cond(Xd))
# compare projections
proj_full = Xd @ beta
proj_gram = Xd @ betag
print("max |proj_full - proj_gram|:", np.max(np.abs(proj_full - proj_gram)))
print("residual diff norm:", np.linalg.norm((Yr-proj_full) - (Yr-proj_gram)))
