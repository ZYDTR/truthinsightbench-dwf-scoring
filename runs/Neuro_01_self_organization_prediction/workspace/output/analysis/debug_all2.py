import numpy as np, time
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG
t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]; Rj = Rz[:-lag]
nch = Yr.shape[1]
ss_tot = np.sum((Yr - Yr.mean(axis=0, keepdims=True))**2, axis=0)

def gram_solve(G, h):
    try:
        return np.linalg.solve(G, h)
    except:
        return np.linalg.lstsq(G, h, rcond=None)[0]

def per_ch_r2(Xcols):
    r2 = np.zeros(nch)
    for j in range(nch):
        X = np.column_stack(Xcols)
        y = Yr[:, j]
        G = X.T @ X; h = X.T @ y; b = gram_solve(G, h)
        ss_res = float(np.sum((y - X @ b)**2)); r2[j] = 1.0 - ss_res/ss_tot[j]
    return float(np.nanmean(r2)), r2

t0 = time.time()
m_sopc, r2_sopc = per_ch_r2([np.ones(n), Xs, Xo])
r2_rso = np.zeros(nch)
for j in range(nch):
    X = np.column_stack([np.ones(n), Rj[:, j], Xs, Xo])
    y = Yr[:, j]; G = X.T @ X; h = X.T @ y; b = gram_solve(G, h)
    ss_res = float(np.sum((y - X @ b)**2)); r2_rso[j] = 1.0 - ss_res/ss_tot[j]
m_rsopc = float(np.nanmean(r2_rso))
t1 = time.time()
print("per-ch s_o  [1,s,o]    -> yj : mean R2=%.6f  (time %.2fs)" % (m_sopc, t1 - t0))
print("  min=%.4f max=%.4f  >0.5: %.2f%%" % (r2_sopc.min(), r2_sopc.max(), 100.0*(r2_sopc>0.5).sum()/nch))
print("per-ch r_s_o[1,rj,s,o] -> yj : mean R2=%.6f  (time %.2fs)" % (m_rsopc, time.time()-t0))
print("  min=%.4f max=%.4f  >0.5: %.2f%%" % (r2_rso.min(), r2_rso.max(), 100.0*(r2_rso>0.5).sum()/nch))
print("monotonic per-ch: r_s_o>=s_o? %s" % (m_rsopc >= m_sopc - 1e-9))
Xd = np.column_stack([np.ones(n), Xs, Xo])
b, *_ = np.linalg.lstsq(Xd, Yr, rcond=None)
ss_res = np.sum((Yr - Xd @ b)**2, axis=0); r2_jso = 1 - ss_res/ss_tot
m_jso = float(np.nanmean(r2_jso))
print("joint s_o  [1,s,o] -> Yr : mean R2=%.6f" % m_jso)
print("per-ch s_o >= joint s_o? %s" % (m_sopc >= m_jso - 1e-9))
