import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
j = 0
y = Yr[:, j]; rj = Rz[:-lag, j]
ss_tot = float(np.sum((y - y.mean())**2))

def qr_ssr(A, b):
    Q, R = np.linalg.qr(A)
    R1 = R[:A.shape[1], :A.shape[1]]
    bproj = Q[:, :A.shape[1]] @ np.linalg.solve(R1, Q[:, :A.shape[1]].T @ b)
    return float(np.sum((b - bproj)**2)), bproj

X1 = np.column_stack([np.ones(n), Xs, Xo])   # [1, s, o]
ssr_y_x1, proj_x1 = qr_ssr(X1, y)
e = y - proj_x1
print("QR SSR(y~[1,s,o]) = %.6f   mean(e)=%.3e" % (ssr_y_x1, e.mean()))

X2_full = np.column_stack([np.ones(n), rj])
ssr_e_x2, _ = qr_ssr(X2_full, e)
print("QR SSR(e~[1,rj])   = %.6f" % ssr_e_x2)

Xfull = np.column_stack([np.ones(n), Xs, Xo, rj])
ssr_y_full, _ = qr_ssr(Xfull, y)
print("QR SSR(y~[1,s,o,rj])= %.6f" % ssr_y_full)

print("FWL holds? SSR(e~X2)==SSR(y~X1,X2): %s" % np.isclose(ssr_e_x2, ssr_y_full, rtol=1e-8, atol=1e-4))
print("  SSR(e~X2)=%.6f   SSR(y~X1,X2)=%.6f   diff=%.4f" % (ssr_e_x2, ssr_y_full, abs(ssr_e_x2-ssr_y_full)))
print("R2: [s,o]=%.6f  [s,o,rj]=%.6f" % (1-ssr_y_x1/ss_tot, 1-ssr_y_full/ss_tot))
