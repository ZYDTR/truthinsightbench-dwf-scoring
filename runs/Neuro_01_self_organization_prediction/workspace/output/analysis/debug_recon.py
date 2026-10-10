import numpy as np
from data_loader import load_conditions
from prediction import analyze_trial, _standardize_per_trial, LAG, _r2_and_mse
t = load_conditions(conditions=["bic"])["bic"][0]
res = analyze_trial(t, lag=1)
so = res["s_o"]["r2"]
print("analyze s_o: mean=%.6f min=%.4f max=%.4f" % (so.mean(), so.min(), so.max()))

Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
n = Sz.shape[0] - LAG
Xs = Sz[:-LAG]; Xo = Oz[:-LAG]; Yr = Rz[LAG:]
ss_tot = np.sum((Yr - Yr.mean(axis=0, keepdims=True))**2, axis=0)
print("ss_tot per ch: mean=%.4f" % ss_tot.mean())

# Path A: exact code path _r2_and_mse([Xs,Xo], Yr)
r2_a, mse_a = _r2_and_mse(np.column_stack([Xs, Xo]), Yr)
print("Path A (_r2_and_mse [s,o]): mean=%.6f" % float(np.nanmean(r2_a)))

# Path B: independent lstsq
X = np.column_stack([np.ones(n), Xs, Xo])
b, *_ = np.linalg.lstsq(X, Yr, rcond=None)
ss_res_b = np.sum((Yr - X @ b)**2, axis=0)
r2_b = 1 - ss_res_b/ss_tot
print("Path B (lstsq [1,s,o]): mean=%.6f" % float(np.nanmean(r2_b)))

# now the code-path beta (Gram) vs lstsq beta
Xr, keep = Xs, Xo
# replicate _r2_and_mse internally
Xd = np.column_stack([np.ones(n), Xs, Xo])
XtX = Xd.T @ Xd; XtY = Xd.T @ Yr
try: b_gram = np.linalg.solve(XtX, XtY)
except: b_gram, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
print("Gram beta vs lstq beta: max diff = %e" % np.max(np.abs(b_gram - b)))
print("Gram proj vs lstq proj: max diff = %e" % np.max(np.abs(Xd @ b_gram - X @ b)))
ss_res_a = np.sum((Yr - Xd @ b_gram)**2, axis=0)
print("ss_res: pathA(mean)=%.4f  pathB(mean)=%.4f  diff=%.4f" % (ss_res_a.mean(), ss_res_b.mean(), abs(ss_res_a.mean()-ss_res_b.mean())))
