import numpy as np
from data_loader import load_conditions
from prediction import analyze_trial, _standardize_per_trial, LAG, _r2_and_mse

def ref_r2(X, Y):
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    ss_res = np.sum((Y - Xd @ beta)**2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True))**2, axis=0)
    r2 = 1 - ss_res/ss_tot
    return float(np.nanmean(r2)), r2

for cond in ["bic", "mix0", "mix50"]:
    t = load_conditions(conditions=[cond])[cond][0]
    Sz = _standardize_per_trial(t["s"]); Oz = _standardize_per_trial(t["o"]); Rz = _standardize_per_trial(t["r"])
    lag = LAG
    Xs = Sz[:-lag]; Xo = Oz[:-lag]; Yr = Rz[lag:]; Ys = Sz[lag:]
    print("=== %s (trial 0) ===" % cond)
    for name, X, Y in [("s", Xs, Yr), ("o", Xo, Yr), ("s_o", np.column_stack([Xs,Xo]), Yr),
                        ("s_self", Xs, Ys), ("s_o_s", np.column_stack([Xs,Xo]), Ys)]:
        ref = ref_r2(X, Y)
        code, _ = _r2_and_mse(X, Y)
        print("  %-8s ref=%.6f  code=%.6f  match=%s" % (name, ref[0], float(np.nanmean(code)), abs(ref[0]-float(np.nanmean(code)))<1e-5))
    pr_ref = 0.0
    for j in range(Yr.shape[1]):
        X = np.column_stack([np.ones(len(Yr)), Rz[:-lag,j]])
        beta,*_ = np.linalg.lstsq(X, Yr[:,j], rcond=None)
        ssr = float(np.sum((Yr[:,j]-X@beta)**2)); sst = float(np.sum((Yr[:,j]-Yr[:,j].mean())**2))
        pr_ref += 1 - ssr/sst
    pr_ref /= Yr.shape[1]
    code_pr = analyze_trial(t, lag=1)["persist_r"]["r2"]
    print("  %-8s ref=%.6f  code=%.6f  match=%s" % ("persist_r", pr_ref, float(np.nanmean(code_pr)), abs(pr_ref-float(np.nanmean(code_pr)))<1e-5))
