import numpy as np
from data_loader import load_conditions
from prediction import _standardize_per_trial, _residuals, LAG

def r2_gram_residual(X, Y):
    # NEW: Gram solve for beta (accurate), but ss_res from residual directly.
    Xr, keep = None, None
    import prediction as pred
    Xr, keep = pred._drop_constant_cols(X)
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), Xr])
    XtY = Xd.T @ Y
    XtX = Xd.T @ Xd
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
    res = Y - Xd @ beta
    ss_res = np.sum(res*res, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0))**2)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res/ss_tot, np.nan)
    return float(np.nanmean(r2)), res

def r2_full(X, Y):
    # reference: full lstsq
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    res = Y - Xd @ beta
    ss_res = np.sum(res*res, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0))**2)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res/ss_tot, np.nan)
    return float(np.nanmean(r2)), res

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
        a, ra = r2_gram_residual(X, Y)
        b, rb = r2_full(X, Y)
        resdiff = float(np.linalg.norm(ra - rb)) / max(1.0, float(np.linalg.norm(ra)))
        rows.append((name, a, b, abs(a-b), resdiff))
    # r_s_o
    e = _residuals(np.column_stack([Xs,Xo]), Yr)
    Xj = Rz[:-lag, :]
    # new residual-based r2 for r_s_o per channel
    r2_new = []
    for j in range(64):
        Xd = np.column_stack([np.ones(len(Yr)), Xj[:,j]])
        XtY = Xd.T @ e[:, j][:, None]
        XtX = Xd.T @ Xd
        try:
            beta = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
        res = e[:, j][:, None] - Xd @ beta
        ss_res = float(np.sum(res*res))
        ss_tot = float(((Yr[:,j]-Yr[:,j].mean())**2).sum())
        r2_new.append(1.0 - ss_res/ss_tot)
    a_rso = float(np.nanmean(r2_new))
    # reference full r_s_o
    r2_ref_rso = []
    for j in range(64):
        Xr = np.column_stack([np.ones(len(Yr)), Xj[:,j], Xs, Xo])
        beta, *_ = np.linalg.lstsq(Xr, Yr[:,j], rcond=None)
        resv = Yr[:,j] - Xr @ beta
        ss_res = float(resv@resv); ss_tot = float(((Yr[:,j]-Yr[:,j].mean())**2).sum())
        r2_ref_rso.append(1.0 - ss_res/ss_tot)
    b_rso = float(np.nanmean(r2_ref_rso))
    rows.append(("r_s_o", a_rso, b_rso, abs(a_rso-b_rso), None))
    print("=== %s ===" % cond)
    for name,a,b,d,rd in rows:
        print("  %-8s fixed=%.4f  ref=%.4f  diff=%.4f" % (name,a,b,d))
