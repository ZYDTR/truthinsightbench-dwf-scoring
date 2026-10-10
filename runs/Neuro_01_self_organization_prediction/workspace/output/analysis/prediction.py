"""Core linear predictive-power analysis for Neuro_01_self_organization_prediction.

Analysis unit = ONE complete trial (25,600 time points are NOT independent
replicates). Within each trial we fit OLS of the response at the NEXT time
point on a chosen feature set evaluated at the current time point (lag=1).
Per-trial fits are descriptive (full time series); results are aggregated
across the independent trials of each condition.

Standardization (explicit modelling choice, per manifest warning):
  Every feature and target channel is standardized within its trial (zero mean,
  unit variance) before fitting, so R^2 = fraction of intra-trial variance
  explained. R^2 is invariant to a per-channel linear transform of the target,
  so per-channel standardization of the target does not change R^2; we
  standardize features per-trial so that all feature sets are on a common
  scale and the intercept is well-conditioned.

Feature sets (predict at lag 1 from time t):
  zero        : predict 0                      (baseline)
  mean        : predict the trial mean         (intercept only)
  persist_r   : r(t,j) -> r(t+1, j)           (response persistence)
  s           : s(t) -> r(t+1, j)             (hidden sources)
  o           : o(t) -> r(t+1, j)             (sensory stimulus)
  s_o         : [s(t), o(t)] -> r(t+1, j)
  r_s_o       : [r(t,j), s(t), o(t)] -> r(t+1, j)
  s_self      : s(t) -> s(t+1)                (latent self-organization)
  s_o_s       : [s(t), o(t)] -> s(t+1)        (stimulus-driven latent dynamics)

Shared-feature models (features identical across all target channels) are fit
with a single normal-equation solve per feature set. Per-channel models
(persist_r, r_s_o) are fit channel by channel.
"""
import numpy as np

LAG = 1


def _standardize_per_trial(arr):
    """Standardize a (T, C) array along axis=0 (per trial)."""
    m = arr.mean(axis=0, keepdims=True)
    s = arr.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return (arr - m) / s


def _drop_constant_cols(X):
    """Drop zero-variance (constant) columns from X. A constant column
    is linearly dependent on the intercept and adds nothing to the fit,
    so dropping it does not change the OLS column space, R^2, or MSE.
    Returns (X_reduced, keep_mask)."""
    var = np.var(X, axis=0)
    keep = var > 1e-14
    if keep.all():
        return X, keep
    return X[:, keep], keep


def _r2_and_mse(X, Y, ss_tot=None):
    """OLS R^2 and MSE of Y on X using normal equations (fast, exact to
    machine precision for well-conditioned X).
    X: (T, p) feature matrix WITHOUT intercept. Y: (T, nch) target.
    Returns (r2 (nch,), mse (nch,)).
    Uses a built-in intercept by column-stacking ones.

    Fast path: solve the small (p+1) x (p+1) Gram system XtX beta = XtY.
    The Gram matrix is small (p+1 <= ~68) even when T is huge, and it is
    rank-deficient exactly when the feature columns are collinear (e.g.
    the 32 stimulus channels are near-linearly dependent in the mix0/mix50
    conditions). On rank deficiency we fall back to lstsq on the SAME small
    Gram matrix, which returns the minimum-norm OLS solution in ~0.01s
    and gives the identical column-space (hence identical R^2) to a full
    lstsq on the large T x (p+1) design matrix. Constant feature columns are
    dropped first (they are rank-dependent on the intercept).

    ss_tot: if provided (a per-channel (nch,) vector), use it as the
    denominator for R^2 (enables the Frisch-Waugh-Lovell trick where we fit
    on a residual but measure R^2 against the ORIGINAL target's per-channel
    total sum of squares). Otherwise use the per-channel total sum of squares
    of Y (each channel's R^2 uses its own intra-trial variance)."""
    Xr, keep = _drop_constant_cols(X)
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), Xr])
    XtY = Xd.T @ Y
    XtX = Xd.T @ Xd
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        # Rank deficiency: minimum-norm OLS via lstsq on the small Gram.
        # Same column space as the full design matrix -> same R^2.
        beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
    # ss_res computed from the residual directly. The alternative Gram
    # formula  ss_res = diag(YTY) - 2 diag(XtY^T beta) + diag(beta^T XtX beta)
    # suffers catastrophic cancellation on ill-conditioned / rank-deficient
    # designs (e.g. the near-collinear 32 stimulus channels in mix0/mix50),
    # inflating ss_res and corrupting R^2 (observed up to ~100% error).
    # Computing ||Y - Xd beta||^2 is numerically stable and gives the same
    # OLS residual, so it is exact.
    ss_res = np.sum((Y - Xd @ beta) ** 2, axis=0)
    if ss_tot is None:
        # Per-channel total sum of squares: R^2 for channel j uses channel j's
        # own intra-trial variance as the denominator. (A scalar sum over all
        # channels would overstate R^2 by ~n_channels; per-channel is the
        # standard within-channel R^2.)
        ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    mse = ss_res / n
    return r2, mse


def _residuals(X, Y):
    """Return e = Y - proj_{[1, X]}Y, the OLS residual, (T, nch)."""
    Xr, keep = _drop_constant_cols(X)
    Xd = np.column_stack([np.ones(X.shape[0]), Xr])
    XtY = Xd.T @ Y
    XtX = Xd.T @ Xd
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
    return Y - Xd @ beta


def _r2_channel(x, y):
    """Univariate OLS R^2 and MSE of y on x (both (T,))."""
    r2, mse = _r2_and_mse(x[:, None], y[:, None])
    return float(r2[0]), float(mse[0])


def analyze_trial(trial, lag=1, models=None):
    """Analyze one trial. Returns dict model -> {'r2': array over channels,
    'mse': array over channels}. Shared-feature models report one value per
    target channel; s_self / s_o_s report per latent-source channel.

    Indexing: features at time t, target at time t+lag, for t = 0..N-1-lag.
    We standardize each stream over its FULL length (N rows) and then slice,
    so x (length N-lag) and y (length N-lag) are aligned row-for-row.
    """
    if models is None:
        models = ("zero", "mean", "persist_r", "s", "o", "s_o",
                  "r_s_o", "s_self", "s_o_s")
    s = trial["s"]
    o = trial["o"]
    r = trial["r"]
    N = s.shape[0]
    if N - lag <= 0:
        raise ValueError("lag too large")

    # Standardize each stream within the trial (over the full length).
    Sz = _standardize_per_trial(s)
    Oz = _standardize_per_trial(o)
    Rz = _standardize_per_trial(r)

    # Feature window (t) and target window (t+lag), aligned.
    Xs = Sz[:-lag, :]      # (N-lag, 2)
    Xo = Oz[:-lag, :]      # (N-lag, 32)
    Yr = Rz[lag:, :]       # (N-lag, 64)

    res = {}

    # Shared-feature models: one solve per feature set.
    for name, X, Y in (
        ("s", Xs, Yr),
        ("o", Xo, Yr),
        ("s_o", np.column_stack([Xs, Xo]), Yr),
        ("s_self", Xs, Sz[lag:, :]),     # s(t) -> s(t+lag)
        ("s_o_s", np.column_stack([Xs, Xo]), Sz[lag:, :]),
    ):
        r2, mse = _r2_and_mse(X, Y)
        res[name] = {"r2": r2, "mse": mse}

    # per-channel models: persist_r and r_s_o (per-channel because r(t,j)
    # is a channel-specific feature).
    nch = Rz.shape[1]
    r2_pr = np.empty(nch); mse_pr = np.empty(nch)
    r2_rso = np.empty(nch); mse_rso = np.empty(nch)
    n = Yr.shape[0]

    # r_s_o uses [r(t,j), s, o] -> r(t+lag, j).  This is the expensive per-
    # channel model (35 columns).  Use the Frisch-Waugh-Lovell theorem:
    #   SS_res(Y ~ [1, r_j, s, o]) = SS_res(e ~ [1, r_j])
    # where e = Y - proj_{[1, s, o]}Y.  So fit [1, s, o] on Y ONCE (shared
    # across channels, one batch solve), then fit [1, r_j] on e per channel
    # (only 2 columns, cheap).  This is exactly the full OLS fit, with a
    # much smaller per-channel solve.  R^2 uses the ORIGINAL Y's SS_tot.
    e_so = _residuals(np.column_stack([Xs, Xo]), Yr)   # (n, nch)
    ss_tot_Y = np.sum((Yr - Yr.mean(axis=0, keepdims=True)) ** 2, axis=0)
    for j in range(nch):
        # persist_r: r(t,j) -> r(t+lag, j)  (1 feature, univariate lstsq)
        r2_p, mse_p = _r2_channel(Rz[:-lag, j], Yr[:, j])
        r2_pr[j] = r2_p; mse_pr[j] = mse_p
        # r_s_o via FWL: fit [1, r_j] on e_so, but R^2 must use the ORIGINAL
        # Yr's total sum of squares (FWL preserves SS_res, and R^2 = 1 -
        # SS_res/SS_tot_original).
        Xj = Rz[:-lag, j]
        r2j, msej = _r2_and_mse(Xj, e_so[:, j][:, None], ss_tot=ss_tot_Y)
        r2_rso[j] = r2j[0]; mse_rso[j] = msej[0]
    res["persist_r"] = {"r2": r2_pr, "mse": mse_pr}
    res["r_s_o"] = {"r2": r2_rso, "mse": mse_rso}

    # baselines (computed on the target window Yr)
    nch = Rz.shape[1]
    n = Yr.shape[0]
    res["zero"] = {"r2": np.zeros(nch), "mse": np.zeros(nch)}
    for j in range(nch):
        y = Yr[:, j]
        ss_res = float(np.sum(y * y))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        res["zero"]["r2"][j] = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        res["zero"]["mse"][j] = ss_res / n
    res["mean"] = {"r2": np.zeros(nch), "mse": np.zeros(nch)}
    for j in range(nch):
        y = Yr[:, j]
        res["mean"]["mse"][j] = float(np.sum((y - y.mean()) ** 2) / n)

    return res


def aggregate(trial_results):
    """trial_results: list of per-trial result dicts (same models).
    Returns dict model -> summary stats."""
    models = set()
    for tr in trial_results:
        models.update(tr.keys())
    out = {}
    for name in models:
        r2_list = np.array([tr[name]["r2"] for tr in trial_results])  # (n_trials, nch)
        mse_list = np.array([tr[name]["mse"] for tr in trial_results])  # (n_trials, nch)
        n_trials = len(trial_results)
        # r2_list is a 2-D float array (nch is uniform within a model).
        per_trial = np.nanmean(r2_list, axis=1)
        out[name] = {
            "n_trials": int(n_trials),
            "r2_overall": float(np.nanmean(r2_list)),
            "r2_per_trial": per_trial.tolist(),
            "r2_median_trial": float(np.nanmedian(per_trial)),
            "r2_channel_mean": np.nanmean(r2_list, axis=0).tolist(),
            "n_channels": int(r2_list.shape[1]),
            "mse_overall": float(np.nanmean(mse_list)),
            "n_trials_above_0.05": int(np.sum(per_trial > 0.05)),
            "n_trials_above_0.5": int(np.sum(per_trial > 0.5)),
        }
    return out


if __name__ == "__main__":
    import time
    from data_loader import load_all
    trials = load_all()
    apv = [t for t in trials if t["condition"] == "apv"]
    t0 = time.time()
    tres = [analyze_trial(t, lag=1) for t in apv[:3]]
    print("3 apv trials in %.2fs" % (time.time() - t0))
    agg = aggregate(tres)
    for name in sorted(agg):
        a = agg[name]
        print("%-9s n=%d  r2_overall=%.4f  median_trial=%.4f  n>0.05=%d/%d" % (
            name, a["n_trials"], a["r2_overall"], a["r2_median_trial"],
            a["n_trials_above_0.05"], a["n_trials"]))
