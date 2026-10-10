import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, LAG

t = load_conditions(conditions=["bic"])["bic"][0]
Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
lag = LAG
n = Sz.shape[0] - lag
Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]; Rj = Rz[:-lag]
nch = Yr.shape[1]
ss_tot = np.sum((Yr - Yr.mean(axis=0, keepdims=True))**2, axis=0)  # (64,)
print("n=%d nch=%d  ss_tot per ch: mean=%.4f min=%.4f max=%.4f" % (
    n, nch, ss_tot.mean(), ss_tot.min(), ss_tot.max()))

def per_ch_r2(Xcols, which):
    """Xcols: list of (n,) or (n,k) arrays. Fit per-channel [1, *Xcols] -> y_j.
    Returns mean R2 and the per-channel r2 array."""
    r2 = np.zeros(nch)
    for j in range(nch):
        X = np.column_stack([np.ones(n), *[Xj for Xj in Xcols]])  # (n, p)
        y = Yr[:, j]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_res = float(np.sum((y - X @ beta)**2))
        r2[j] = 1.0 - ss_res/ss_tot[j]
    return float(np.nanmean(r2)), r2

m_s, r2_s = per_ch_r2([Xs], "s")
m_o, r2_o = per_ch_r2([Xo], "o")
m_so, r2_so = per_ch_r2([Xs, Xo], "s_o")
m_rso, r2_rso = per_ch_r2([Rj, Xs, Xo], "r_s_o")

print("\nPer-channel lstsq R2 (mean over 64 ch):")
print("  s       = %.6f  (min=%.4f max=%.4f)" % (m_s, r2_s.min(), r2_s.max()))
print("  o       = %.6f  (min=%.4f max=%.4f)" % (m_o, r2_o.min(), r2_o.max()))
print("  s_o     = %.6f  (min=%.4f max=%.4f)" % (m_so, r2_so.min(), r2_so.max()))
print("  r_s_o   = %.6f  (min=%.4f max=%.4f)" % (m_rso, r2_rso.min(), r2_rso.max()))
print("\nmonotonic: s_o>=o? %s   r_s_o>=s_o? %s   r_s_o>=s? %s" % (
    m_so>=m_o-1e-9, m_rso>=m_so-1e-9, m_rso>=m_s-1e-9))
print("max r_s_o ch:", r2_rso.max(), "min r_s_o ch:", r2_rso.min())
