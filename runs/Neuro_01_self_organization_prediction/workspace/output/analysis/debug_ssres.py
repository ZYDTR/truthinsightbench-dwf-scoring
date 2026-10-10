import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG

t = load_conditions(conditions=["bic"])["bic"][0]
N = t["s"].shape[0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
n = Yr.shape[0]; j = 0; y = Yr[:, j]

Xso = np.column_stack([Xs, Xo])
Xd = np.column_stack([np.ones(n), Xso])
beta = np.linalg.lstsq(Xd, Yr, rcond=None)[0]
Y = Yr[:, j:j+1]

res = Y - Xd @ beta
ss_res_direct = float(np.sum(res*res))

YTY = Y.T @ Y
XtY = Xd.T @ Y
XtX = Xd.T @ Xd
ss_res_formula = float(np.diag(YTY)[0]
              - 2.0 * np.diag(XtY.T @ beta)[0]
              + np.diag(beta.T @ XtX @ beta)[0])

print("direct ss_res     = %.6f" % ss_res_direct)
print("formula ss_res    = %.6f" % ss_res_formula)
print("diff               = %.4f" % (ss_res_formula - ss_res_direct))
print("relative diff      = %.3e" % ((ss_res_formula - ss_res_direct)/ss_res_direct))
