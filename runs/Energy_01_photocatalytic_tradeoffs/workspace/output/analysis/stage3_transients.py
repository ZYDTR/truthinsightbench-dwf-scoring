"""Stage 3 -- ultrafast transient traces for formulation_1 and formulation_2.

Analysis unit: one complete transient trace.  The 98 delay points inside a trace
form a serial sweep, not independent replicates, so inference uses:
  (i)   whole-trace rise/decay model comparison (OLS on all 98 points);
  (ii)  moving-block bootstrap CIs for windowed means (serially correlated data);
  (iii) delay-window holdouts (fit early, predict the held-out late window);
  (iv)  artifact-window and threshold sensitivity checks.

The traces are not pure decays: both rise from ~0 around 0.8-1.2 ps and only
then decay.  Models therefore include an explicit excitation rise term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common as C

log = C.Logger(C.LOGS / "stage3_transients.log")


# --------------------------------------------------------------------------
# candidate rise/decay models; every model is integrated over the instrument
# response implicitly by leaving the rise constant free
# --------------------------------------------------------------------------
def m_flat(t, c):
    return np.full_like(t, c)


def m_mono_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def m_rise_decay(t, A, tau_rise, tau_decay, c):
    return A * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise)) + c


def m_rise_biexp_decay(t, A, f, tau1, tau2, tau_rise, c):
    return A * (f * np.exp(-t / tau1) + (1 - f) * np.exp(-t / tau2) - np.exp(-t / tau_rise)) + c


def m_rise_stretched(t, A, tau_d, beta, tau_rise, c):
    return A * (np.exp(-np.power(np.maximum(t, 1e-9) / tau_d, beta))
                - np.exp(-t / tau_rise)) + c


def m_sum_two_decays(t, a, tau1, b, tau2, c):
    return a * np.exp(-t / tau1) + b * np.exp(-t / tau2) + c


MODELS = {
    "flat_constant": (m_flat, 1, None),
    "monoexponential_decay": (m_mono_decay, 3, None),
    "sum_two_decays": (m_sum_two_decays, 5, None),
    "rise_then_decay": (m_rise_decay, 4, None),
    "rise_then_biexponential_decay": (m_rise_biexp_decay, 6, None),
    "rise_then_stretched_decay": (m_rise_stretched, 5, None),
}


def _starts(name: str, t: np.ndarray, y: np.ndarray) -> list[list[float]]:
    amp = float(np.max(np.abs(y)))
    starts = []
    if name == "flat_constant":
        return [[float(y.mean())]]
    if name == "monoexponential_decay":
        return [[amp, tau, c] for tau in (1.0, 5.0, 20.0, 100.0)
                for c in (0.0, float(y[-5:].mean()))]
    if name == "sum_two_decays":
        return [[amp, 1.0, amp, 30.0, c] for c in (-0.2, 0.0, float(y[-5:].mean()))]
    if name == "rise_then_decay":
        return [[amp, tr, td, c] for tr in (0.2, 0.5, 1.0)
                for td in (2.0, 10.0, 50.0, 200.0)
                for c in (0.0, float(y[-5:].mean()), -0.2)]
    if name == "rise_then_biexponential_decay":
        return [[amp, f, 2.0, 30.0, 0.5, c] for f in (0.3, 0.7)
                for c in (0.0, float(y[-5:].mean()))]
    if name == "rise_then_stretched_decay":
        return [[amp, 20.0, beta, 0.5, c] for beta in (0.4, 0.7, 1.0)
                for c in (0.0, float(y[-5:].mean()))]
    return starts


def fit_model(name: str, t: np.ndarray, y: np.ndarray) -> dict:
    fn, k, _ = MODELS[name]
    n = len(t)
    if name == "flat_constant":
        pred = fn(t, y.mean())
        return {"ok": True, "params": {"c": float(y.mean())}, "pred": pred,
                "k_params": k, "rss": float(((y - pred) ** 2).sum())}
    best = None
    for p0 in _starts(name, t, y):
        try:
            popt, _ = optimize.curve_fit(fn, t, y, p0=p0, maxfev=400000)
        except Exception:
            continue
        if not np.all(np.isfinite(popt)):
            continue
        rss = float(((y - fn(t, *popt)) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, popt)
    if best is None:
        return {"ok": False, "error": f"{name}: every starting point failed"}
    params = {pn: float(pv) for pn, pv in zip(_param_names(name), best[1])}
    pred = fn(t, *best[1])
    if not np.all(np.isfinite(pred)):
        return {"ok": False, "error": f"{name}: non-finite prediction"}
    return {"ok": True, "params": params, "k_params": k, "rss": best[0], "pred": pred}


def _param_names(name: str) -> list[str]:
    return {
        "flat_constant": ["c"],
        "monoexponential_decay": ["a", "tau_ps", "c"],
        "sum_two_decays": ["a", "tau1_ps", "b", "tau2_ps", "c"],
        "rise_then_decay": ["A", "tau_rise_ps", "tau_decay_ps", "c"],
        "rise_then_biexponential_decay": ["A", "f_fast", "tau1_ps", "tau2_ps", "tau_rise_ps", "c"],
        "rise_then_stretched_decay": ["A", "tau_decay_ps", "beta", "tau_rise_ps", "c"],
    }[name]


def decay_model_comparison(tr: pd.DataFrame) -> dict:
    log("=" * 78)
    log("STAGE 3  ultrafast transient traces (unit = one complete trace)")
    log("=" * 78)
    log("\n" + "-" * 78)
    log("Rise/decay model comparison per trace (OLS on all 98 delay points)")
    log("-" * 78)
    out = {"per_formulation": {}}
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        t = g.delay_ps.values.astype(float)
        y = g.normalized_transient_signal.values.astype(float)
        n = len(t)
        sst = float(((y - y.mean()) ** 2).sum())
        rows = {}
        residuals = {}
        for name in MODELS:
            fit = fit_model(name, t, y)
            if not fit["ok"]:
                rows[name] = {"model": name, "ok": False, "error": fit["error"]}
                log(f"  {name}: FAILED -- {fit['error']}")
                continue
            kk = fit["k_params"] + 1
            aicc = n * np.log(max(fit["rss"], 1e-300) / n) + 2 * kk + \
                (2 * kk * (kk + 1)) / max(n - kk - 1, 1)
            rows[name] = {
                "model": name, "ok": True, "k_params": fit["k_params"],
                "rss": fit["rss"], "r2": 1.0 - fit["rss"] / sst,
                "rmse": float(np.sqrt(fit["rss"] / n)), "aicc": float(aicc),
                "params": fit["params"],
            }
            residuals[name] = y - fit["pred"]
        ranked = sorted([r for r in rows.values() if r.get("ok")], key=lambda r: r["aicc"])
        out["per_formulation"][fid] = {
            "models": rows,
            "ranking_by_aicc": [r["model"] for r in ranked],
            "aicc_delta_vs_best": {r["model"]: float(r["aicc"] - ranked[0]["aicc"]) for r in ranked},
        }
        log(f"\n{fid}:")
        for r in ranked:
            note = ""
            if r["model"] == "rise_then_decay":
                note = (f"  [tau_rise={r['params']['tau_rise_ps']:.3f} ps, "
                        f"tau_decay={r['params']['tau_decay_ps']:.2f} ps, "
                        f"late offset c={r['params']['c']:+.4f}]")
            log(f"  {r['model']:>32s} k={r['k_params']} R2={r['r2']:.4f} RMSE={r['rmse']:.4f} "
                f"AICc={r['aicc']:>8.2f} dAICc={r['aicc']-ranked[0]['aicc']:.2f}{note}")
    return out


def trace_metrics(tr: pd.DataFrame) -> dict:
    log("\n" + "-" * 78)
    log("Windowed trace statistics with moving-block bootstrap CIs")
    log("-" * 78)
    out = {"per_formulation": {}}
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        t = g.delay_ps.values.astype(float)
        y = g.normalized_transient_signal.values.astype(float)
        t_peak = float(t[int(np.argmax(y))])
        y_peak = float(y.max())
        rise = {}
        for frac in (0.1, 0.5, 0.9):
            idx = np.where((t <= t_peak) & (y >= frac * y_peak))[0]
            rise[f"t_at_{int(frac*100)}pct_of_peak_ps"] = float(t[idx[0]]) if idx.size else None
        windows = {}
        for lo in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            w = C.signal_window_stats(t, y, lo)
            bs = C.block_bootstrap_mean(t, y, t >= lo)
            w["block_bootstrap"] = bs
            windows[f"delay_ge_{lo}ps"] = w
        noise = C.readout_noise(y, t)
        lo = t >= 30.0
        se = noise / np.sqrt(lo.sum())
        out["per_formulation"][fid] = {
            "n_points": int(len(t)),
            "delay_range_ps": [float(t.min()), float(t.max())],
            "peak_delay_ps": t_peak, "peak_value": y_peak,
            "early_pre_rise_mean_first_6_ps": float(y[t <= 0.6].mean()),
            "rise_metrics": rise,
            "window_statistics": windows,
            "readout_noise_mad": noise,
            "late_window_ge30ps_mean_over_noise_se": float(y[lo].mean() / se) if se > 0 else None,
            "first_zero_crossing_after_2ps": C._first_zero_crossing(t, y, 2.0),
            "final_three_values": y[-3:].tolist(),
        }
        log(f"\n{fid}: peak {y_peak:.3f} at {t_peak:.3f} ps; pre-rise mean {out['per_formulation'][fid]['early_pre_rise_mean_first_6_ps']:+.4f}")
        log(f"  rise metrics: {rise}")
        for key, w in windows.items():
            bs = w["block_bootstrap"]
            log(f"  {key:>18s}: n={w['n_points']:3d} mean={w['mean']:+.4f} "
                f"median={w['median']:+.4f} frac_pos={w['fraction_positive']:.2f} "
                f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}] "
                f"excludes_zero={bs['excludes_zero']}")
        log(f"  readout noise (MAD of successive differences) = {noise:.4f}; "
            f"late-window mean / noise-SE = {y[lo].mean()/se:+.2f}")
    return out


def window_holdout(tr: pd.DataFrame) -> dict:
    """Fit the rise/decay model on early delays only, then predict the held-out
    long-delay window.  The test asks whether the early signal alone predicts the
    sign of the late response."""
    log("\n" + "-" * 78)
    log("Delay-window holdout: fit rise_then_decay on early delays, predict late")
    log("-" * 78)
    out = {"per_formulation": {}}
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        t = g.delay_ps.values.astype(float)
        y = g.normalized_transient_signal.values.astype(float)
        rows = {}
        for fit_cut in (5.0, 10.0, 20.0, 30.0):
            fit_m, held_m = t <= fit_cut, t > fit_cut
            if held_m.sum() < 3:
                continue
            fn = m_rise_decay
            best = None
            for p0 in _starts("rise_then_decay", t[fit_m], y[fit_m]):
                try:
                    popt, _ = optimize.curve_fit(fn, t[fit_m], y[fit_m], p0=p0, maxfev=400000)
                except Exception:
                    continue
                if not np.all(np.isfinite(popt)):
                    continue
                rss = float(((y[fit_m] - fn(t[fit_m], *popt)) ** 2).sum())
                if best is None or rss < best[0]:
                    best = (rss, popt)
            if best is None:
                rows[f"fit_le_{fit_cut}ps"] = {"ok": False,
                                               "error": "rise_then_decay failed on early window"}
                continue
            pred_early = fn(t[fit_m], *best[1])
            pred_held = fn(t[held_m], *best[1])
            obs = y[held_m]
            rows[f"fit_le_{fit_cut}ps"] = {
                "ok": True, "fit_window_n": int(fit_m.sum()), "held_window_n": int(held_m.sum()),
                "params_from_early_fit": {pn: float(pv) for pn, pv in
                                          zip(("A", "tau_rise_ps", "tau_decay_ps", "c"), best[1])},
                "r2_early_fit": float(1 - ((y[fit_m] - pred_early) ** 2).sum() /
                                      ((y[fit_m] - y[fit_m].mean()) ** 2).sum()),
                "held_out_observed_mean": float(obs.mean()),
                "held_out_predicted_mean": float(pred_held.mean()),
                "model_extrapolated_negative_late": bool(pred_held.mean() < 0),
                "observed_negative_late": bool(obs.mean() < 0),
                "sign_agreement": bool(np.sign(obs.mean()) == np.sign(pred_held.mean())),
                "rmse_held_out": float(np.sqrt(((obs - pred_held) ** 2).mean())),
                "bias_held_out": float((pred_held - obs).mean()),
            }
            r = rows[f"fit_le_{fit_cut}ps"]
            log(f"  {fid} fit<={fit_cut:4.0f} ps (R2={r['r2_early_fit']:.4f}): "
                f"extrapolated late mean {r['held_out_predicted_mean']:+.4f} vs observed "
                f"{r['held_out_observed_mean']:+.4f}  sign_ok={r['sign_agreement']}")
        out["per_formulation"][fid] = rows
    return out


def artifact_window_robustness(tr: pd.DataFrame) -> dict:
    """Exclude the coherent-artifact region (t < 0.5 ps, before the rise) and
    re-check the late-time conclusions with several late-window thresholds."""
    log("\n" + "-" * 78)
    log("Robustness: artifact-window exclusion and late-window threshold sweep")
    log("-" * 78)
    out = {}
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        t = g.delay_ps.values.astype(float)
        y = g.normalized_transient_signal.values.astype(float)
        m = t >= 0.5
        sweep = {}
        for lo in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0):
            sel_all = t >= lo
            sel_clean = (t >= lo) & m
            sweep[f"ge_{lo:g}ps"] = {
                "n": int(sel_all.sum()),
                "mean_all_points": float(y[sel_all].mean()),
                "mean_excluding_lt_0.5ps": float(y[sel_clean].mean()),
                "sign": "negative" if y[sel_clean].mean() < 0 else "positive",
                "fraction_positive_excluding_lt_0.5ps": float((y[sel_clean] > 0).mean()),
            }
        out[fid] = {
            "n_excluded_before_0.5ps": int((~m).sum()),
            "peak_delay_ps_after_exclusion": float(t[m][int(np.argmax(y[m]))]),
            "late_window_threshold_sweep": sweep,
            "sign_stable_across_thresholds": len({v["sign"] for v in sweep.values()}) == 1,
        }
        log(f"  {fid}: peak delay after excluding t<0.5 ps = "
            f"{out[fid]['peak_delay_ps_after_exclusion']:.3f} ps; "
            f"late-window sign stable across 15-50 ps thresholds: "
            f"{out[fid]['sign_stable_across_thresholds']}")
        for key, v in sweep.items():
            log(f"     {key:>10s}: mean={v['mean_excluding_lt_0.5ps']:+.4f} "
                f"frac_pos={v['fraction_positive_excluding_lt_0.5ps']:.2f} sign={v['sign']}")
    return out


def block_significance(t: np.ndarray, y: np.ndarray, seed: int = 20220101) -> dict:
    """Sign-test style statistic for a serially correlated trace: the fraction of
    points above zero compared with a sign-randomised null that preserves the
    autocorrelation by circularly shifting the trace."""
    pos = float((y > 0).mean())
    n = len(y)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n, size=5000)
    null = np.array([(np.roll(y, s) > 0).mean() for s in shifts])
    p = float((np.abs(null - 0.5) >= abs(pos - 0.5)).mean())
    return {"fraction_positive": pos, "n": n, "circular_shift_p": min(1.0, (p * 5000 + 1) / 5001),
            "null_mean": float(null.mean())}


def cross_trace_contrast(tr: pd.DataFrame) -> dict:
    log("\n" + "-" * 78)
    log("Cross-trace contrast at long delay (>=30 ps)")
    log("-" * 78)
    g1 = tr[tr.formulation_id == "formulation_1"].sort_values("delay_ps")
    g2 = tr[tr.formulation_id == "formulation_2"].sort_values("delay_ps")
    t = g1.delay_ps.values.astype(float)
    y1 = g1.normalized_transient_signal.values.astype(float)
    y2 = g2.normalized_transient_signal.values.astype(float)
    sel = t >= 30.0
    bs1 = C.block_bootstrap_mean(t, y1, sel)
    bs2 = C.block_bootstrap_mean(t, y2, sel)
    rng = np.random.default_rng(20220101)
    seg1, seg2 = y1[sel], y2[sel]

    def boot(seg, n_boot=20000):
        block = max(2, int(np.sqrt(len(seg))))
        n_blocks = int(np.ceil(len(seg) / block))
        starts = np.arange(len(seg) - block + 1)
        pick = rng.choice(starts, size=(n_boot, n_blocks), replace=True)
        flat = (pick[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :len(seg)]
        return seg[flat].mean(axis=1)

    b1, b2 = boot(seg1), boot(seg2)
    diff = b1 - b2
    table = [[int((seg1 > 0).sum()), int((seg1 <= 0).sum())],
             [int((seg2 > 0).sum()), int((seg2 <= 0).sum())]]
    odds, p = stats.fisher_exact(table)
    # circular-shift null for the difference of means (preserves autocorrelation)
    n = len(seg1)
    shift_null = np.array([seg2[np.roll(np.arange(n), s)].mean() for s in range(1, n)])
    res = {
        "window_ps": [30.0, float(t.max())], "n_points": int(sel.sum()),
        "formulation_1_mean": float(seg1.mean()), "formulation_1_ci": [float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))],
        "formulation_2_mean": float(seg2.mean()), "formulation_2_ci": [float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))],
        "difference_mean": float(seg1.mean() - seg2.mean()),
        "difference_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "difference_ci_excludes_zero": bool(np.percentile(diff, 97.5) < 0 or np.percentile(diff, 2.5) > 0),
        "formulation_1_fraction_positive": float((seg1 > 0).mean()),
        "formulation_2_fraction_positive": float((seg2 > 0).mean()),
        "fisher_exact_positivity": {"table": table, "odds_ratio": float(odds), "p": float(p)},
    }
    log(f"  f1 mean(>=30 ps) = {res['formulation_1_mean']:+.4f} "
        f"95% CI [{res['formulation_1_ci'][0]:+.4f}, {res['formulation_1_ci'][1]:+.4f}] "
        f"frac_pos={res['formulation_1_fraction_positive']:.2f}")
    log(f"  f2 mean(>=30 ps) = {res['formulation_2_mean']:+.4f} "
        f"95% CI [{res['formulation_2_ci'][0]:+.4f}, {res['formulation_2_ci'][1]:+.4f}] "
        f"frac_pos={res['formulation_2_fraction_positive']:.2f}")
    log(f"  difference f1-f2 = {res['difference_mean']:+.4f} "
        f"95% CI [{res['difference_ci'][0]:+.4f}, {res['difference_ci'][1]:+.4f}] "
        f"excludes zero: {res['difference_ci_excludes_zero']}")
    log(f"  Fisher exact on positivity: table={table} p={p:.4f}")
    return res


def figure_transients(tr: pd.DataFrame, models: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        t = g.delay_ps.values.astype(float)
        y = g.normalized_transient_signal.values.astype(float)
        axes[0].plot(t, y, "o-", ms=3, label=fid)
        axes[1].plot(t, y, "o-", ms=3, label=fid)
        best = models["per_formulation"][fid]["ranking_by_aicc"][0]
        p = models["per_formulation"][fid]["models"][best]["params"]
        ts = np.linspace(t.min(), t.max(), 600)
        axes[2].plot(ts, m_rise_decay(ts, p.get("A", 1.0), p.get("tau_rise_ps", 0.5),
                                      p.get("tau_decay_ps", 10.0), p.get("c", 0.0)),
                     "-", label=f"{fid} {best}")
        axes[2].plot(t, y, "o", ms=2, alpha=0.35)
    axes[0].set_xscale("log")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("delay (ps)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalised transient signal")
    axes[1].axvline(30, color="grey", ls="--", lw=0.8)
    axes[1].set_ylabel("normalised transient signal")
    axes[0].set_title("A  full delay range (log x)")
    axes[1].set_title("B  linear view (dashed = 30 ps)")
    axes[2].set_title("C  rise_then_decay fit")
    axes[0].legend(fontsize=8)
    axes[2].set_xscale("log")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig3_transients.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    tr = C.load_transients()
    models = decay_model_comparison(tr)
    metrics = trace_metrics(tr)
    hold = window_holdout(tr)
    artifact = artifact_window_robustness(tr)
    contrast = cross_trace_contrast(tr)
    signs = {}
    for fid, g in tr.groupby("formulation_id"):
        g = g.sort_values("delay_ps")
        signs[fid] = block_significance(g.delay_ps.values.astype(float),
                                        g.normalized_transient_signal.values.astype(float))
    C.json_dump(C.RESULTS / "transient_analysis.json",
                {"decay_models": models, "trace_metrics": metrics, "window_holdout": hold,
                 "artifact_robustness": artifact, "cross_trace_contrast": contrast,
                 "significance": signs})
    figure_transients(tr, models)
    log("\nwrote figures/fig3_transients.png")
    log("\nstage 3 complete")
    log.close()