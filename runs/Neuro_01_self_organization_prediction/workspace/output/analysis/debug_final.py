import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
import time
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]; Rj = Rz[:-lag]
nch = Yr.shape[1]
ss_tot = np.sum((Yr - Yr.mean(axis=0, keepdims=True))**2, axis=0)  # (64,)
print("n=%d nch=%d ss_tot[0]=%.4f" % (n, nch, ss_tot[0]))

def per_ch_gram_r2(Xcols):
    """Per-channel OLS: for each j, fit [1, *Xcols_j] -> y_j. Xcols is list of
    (n,) per-channel arrays or shared (n,k). Returns mean R2."""
    r2 = np.zeros(nch)
    for j in range(nch):
        cols = [np.ones(n), Rj[:, j]] + [Xs] + [Xo] if Xcols == 'rso' else []
        X = np.column_stack([np.ones(n), Rj[:, j], Xs, Xo])  # (n,36)
        y = Yr[:, j]
        G = X.T @ X  # (36,36)
        h = X.T @ y  # (36,)
        try: b = np.linalg.solve(G, h)
        except: b, *_ = np.linalg.lstsq(G, h, rcond=None)
        ss_res = float(np.sum((y - X @ b)**2))
        r2[j] = 1.0 - ss_res/ss_tot[j]
    return float(np.nanmean(r2)), r2

t0 = time.time()
m_rso, r2_rso = per_ch_gram_r2('rso')
print("per-channel r_s_o [1,rj,s,o] -> yj : mean R2=%.6f  (time %.2fs)" % (m_rso, time.time()-t0))
print("  min=%.4f max=%.4f  >0.95: %.2f%%  >0.5: %.2f%%" % (r2_rso.min(), r2_rso.max(),
      100*r2_rso.sum(r2_rso>0.95)/nch, 100*r2_rso.sum(r2_rso>0.5)/nch))
