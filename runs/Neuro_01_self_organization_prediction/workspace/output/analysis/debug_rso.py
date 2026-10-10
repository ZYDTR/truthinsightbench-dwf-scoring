import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, _residuals, _r2_and_mse, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
j = 0
y = Yr[:, j]; rj = Rz[:-lag, j]
ss_tot = float(np.sum((y - y.mean())**2))

def gram_r2(X, y):
    # Gram solve (as in _r2_and_mse), residual-based ss_res
    Xr, keep = _standardize_per_trial.__globals__['__builtins__'] and None, None
    import prediction as pred
    Xrd, k = pred._drop_constant_cols(X)
    n2 = X.shape[0]
    Xd = np.column_stack([np.ones(n2), Xrd])
    XtY = Xd.T @ y[:, None]; XtX = Xd.T @ Xd
    try: beta = np.linalg.solve(XtX, XtY)
    except: beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
    ss_res = float(np.sum((y[:,None] - Xd @ beta)**2))
    return 1.0 - ss_res/ss_tot

# s_o model [1,s,o]
r2_so_gram = gram_r2(np.column_stack([Xs, Xo]), y)
# full [1,rj,s,o] via Gram solve (36 cols)
r2_rso_gram = gram_r2(np.column_stack([rj, Xs, Xo]), y)
# FWL path (as in analyze_trial)
e = _residuals(np.column_stack([Xs, Xo]), Yr)
r2_rso_fwl, _ = _r2_and_mse(rj[:, None], e[:, j][:, None], ss_tot=ss_tot)
r2_rso_fwl = float(r2_rso_fwl[0])

print("SS_tot(y) = %.4f" % ss_tot)
print("  s_o   [1,s,o]      Gram  R2 = %.6f" % r2_so_gram)
print("  r_s_o [1,rj,s,o]   Gram  R2 = %.6f" % r2_rso_gram)
print("  r_s_o [1,rj,s,o]   FWL   R2 = %.6f" % r2_rso_fwl)
print("  monotonic r_s_o_gram >= s_o_gram? %s" % (r2_rso_gram >= r2_so_gram - 1e-9))
print("  monotonic r_s_o_fwl  >= s_o_gram? %s" % (r2_rso_fwl >= r2_so_gram - 1e-9))
