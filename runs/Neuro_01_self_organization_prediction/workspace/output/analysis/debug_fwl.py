import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, _residuals, _r2_and_mse, LAG

for cond in ["bic", "apv", "mix0"]:
    trials = load_conditions(conditions=[cond])
    t = trials[cond][0]
    N = t["s"].shape[0]
    Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
    lag = LAG
    Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
    n = Yr.shape[0]
    j = 0
    y = Yr[:, j]

    X_full = np.column_stack([np.ones(n), Xs, Xo])
    beta_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    ss_res_full_direct = float((y - X_full @ beta_full) @ (y - X_full @ beta_full))

    X_rso = np.column_stack([np.ones(n), Rz[:-lag, j], Xs, Xo])
    beta_rso, *_ = np.linalg.lstsq(X_rso, y, rcond=None)
    ss_res_rso_direct = float((y - X_rso @ beta_rso) @ (y - X_rso @ beta_rso))

    e_so = _residuals(np.column_stack([Xs, Xo]), Yr)
    Xj = Rz[:-lag, j]
    Xd = np.column_stack([np.ones(n), Xj])
    beta_fwl, *_ = np.linalg.lstsq(Xd, e_so[:, j], rcond=None)
    ss_res_fwl = float((e_so[:, j] - Xd @ beta_fwl) @ (e_so[:, j] - Xd @ beta_fwl))

    ss_tot = float(((y - y.mean())**2).sum())
    r2j, msej = _r2_and_mse(Xj, e_so[:, j][:, None], ss_tot=ss_tot)

    print("=== cond=%s trial0 channel0 ===" % cond)
    print("  Direct R2 [s,o]      = %.6f" % (1-ss_res_full_direct/ss_tot))
    print("  Direct R2 [r,s,o]    = %.6f  (ss_res=%.6f)" % (1-ss_res_rso_direct/ss_tot, ss_res_rso_direct))
    print("  FWL    R2 [r,s,o]    = %.6f  (ss_res=%.6f)" % (1-ss_res_fwl/ss_tot, ss_res_fwl))
    print("  Code _r2_and_mse R2  = %.6f" % float(r2j[0]))
    print("  FWL ss_res == direct?  %s   (diff=%.3e)" % (np.isclose(ss_res_fwl, ss_res_rso_direct, rtol=1e-6, atol=1e-4), abs(ss_res_fwl-ss_res_rso_direct)))
    print("  FWL ss_res > direct? (BUG if true) %s" % (ss_res_fwl > ss_res_rso_direct))
