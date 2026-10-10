import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, _residuals, _r2_and_mse, LAG

def direct_r2(X, Y):
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    ss_tot = np.sum((Y - Y.mean(axis=0))**2)
    res = Y - Xd @ beta
    ss_res = np.sum(res*res, axis=0)
    return float(np.nanmean(1.0 - ss_res/ss_tot))

conds = ["apv","bic","mix0","ctrl","dzp","mix50"]
for cond in conds:
    trials = load_conditions(conditions=[cond])
    t = trials[cond][0]
    N = t["s"].shape[0]
    Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
    lag = LAG
    Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]
    rows = []
    for name, X, Y in [("s", Xs, Yr), ("o", Xo, Yr), ("s_o", np.column_stack([Xs,Xo]), Yr),
                        ("s_self", Xs, Sz[lag:]), ("s_o_s", np.column_stack([Xs,Xo]), Sz[lag:])]:
        a = float(np.nanmean(_r2_and_mse(X, Y)[0]))
        b = direct_r2(X, Y)
        rows.append((name, a, b, abs(a-b)))
    # r_s_o
    e = _residuals(np.column_stack([Xs,Xo]), Yr)
    Xj = Rz[:-lag, :]
    r2_fwl_rso = []
    for j in range(64):
        r2c, _ = _r2_and_mse(Xj[:,j][:,None], e[:,j][:,None], ss_tot=np.sum((Yr[:,j]-Yr[:,j].mean())**2))
        r2_fwl_rso.append(float(r2c[0]))
    r2_fwl_rso = float(np.nanmean(r2_fwl_rso))
    r2_dir_rso = []
    for j in range(64):
        Xr = np.column_stack([np.ones(len(Yr)), Xj[:,j], Xs, Xo])
        beta, *_ = np.linalg.lstsq(Xr, Yr[:,j], rcond=None)
        resv = Yr[:,j] - Xr @ beta
        ss_res = float(resv@resv); ss_tot = float(((Yr[:,j]-Yr[:,j].mean())**2).sum())
        r2_dir_rso.append(1.0 - ss_res/ss_tot)
    r2_dir_rso = float(np.nanmean(r2_dir_rso))
    rows.append(("r_s_o", r2_fwl_rso, r2_dir_rso, abs(r2_fwl_rso - r2_dir_rso)))
    print("=== %s ===" % cond)
    for name,a,b,d in rows:
        print("  %-8s fwl=%.4f  direct=%.4f  diff=%.4f" % (name,a,b,d))
