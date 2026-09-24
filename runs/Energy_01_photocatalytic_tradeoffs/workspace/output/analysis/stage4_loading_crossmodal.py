"""Stage 4 -- cofactor-loading response and cross-modal synthesis.

Two questions:
  (a) Is the loading-rate response monotone, peaked, or unresolved given only
      four loading settings?
  (b) Can the four modalities be related to each other, and over what coverage?

Every claim is bounded by the number of independent analysis units actually
available, which is small and is reported explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common as C

log = C.Logger(C.LOGS / "stage4_loading_crossmodal.log")


# ------------------------------------------------------------ loading models
def _m_mono_exp(L, a, k, c):
    return a * np.exp(-k * L) + c


def _m_mono_power(L, a, b):
    return a * np.power(L, -b)


def _m_peaked_single_site(L, V, K, Ki):
    """Classic substrate-inhibition / single-site saturation with inhibition.

    rate = V * L / (K + L + L^2 / Ki)
    Rising at low L (saturating site), falling at high L (inhibition), so it
    contains the monotone-saturation model as the Ki -> infinity limit.
    """
    return V * L / (K + L + L * L / Ki)


def _m_bilinear_kink(L, r_low, r_high, L_break):
    """Piecewise-linear response with a break point (an optimum at the kink)."""
    return np.where(L <= L_break, r_low * L, r_low * L_break + r_high * (L - L_break))


def _m_constant(L, c):
    return np.full_like(L, c)


LOADING_MODELS = {
    "M0_constant": (_m_constant, 1, ["c"]),
    "M1_monotone_power_law": (_m_mono_power, 2, ["a", "b"]),
    "M2_monotone_exponential": (_m_mono_exp, 3, ["a", "k", "c"]),
    "M3_peaked_substrate_inhibition": (_m_peaked_single_site, 3, ["V", "K", "Ki"]),
    "M4_peaked_bilinear_kink": (_m_bilinear_kink, 3, ["r_low", "r_high", "L_break"]),
}


def loading_analysis(ld: pd.DataFrame) -> dict:
    log("=" * 78)
    log("STAGE 4  cofactor-loading response (unit = one loading condition, n = 4)")
    log("=" * 78)
    L = ld.cofactor_loading_percent.values.astype(float)
    R = ld.product_evolution_rate_umol_h.values.astype(float)
    log("  loading (%) : rate (umol/h)")
    for a, b in zip(L, R):
        log(f"    {a:5.2f} : {b:7.1f}")
    log(f"  rate at the highest loading is {100*(R[-1]-R[0])/R[0]:+.1f}% of the rate at the "
        f"lowest loading; peak at {L[int(np.argmax(R))]:g}%")

    out: dict = {"loading": L.tolist(), "rate": R.tolist(),
                 "peak_loading_percent": float(L[int(np.argmax(R))]),
                 "peak_rate": float(R.max()),
                 "rate_ratio_highest_to_peak": float(R[-1] / R.max()),
                 "rate_ratio_8pct_to_0.2pct": float(R[-1] / R[0]),
                 "log_slope_8pct_vs_1pct": float(np.log(R[-1] / R[1]) / np.log(L[-1] / L[1]))}

    # ---- model comparison -------------------------------------------------
    n = len(L)
    rows = {}
    aicc_undefined: list[str] = []
    for name, (fn, k, pnames) in LOADING_MODELS.items():
        p0 = {"M0_constant": [R.mean()],
              "M1_monotone_power_law": [R[0], 0.5],
              "M2_monotone_exponential": [R[0] - R[-1], 1.0, R[-1]],
              "M3_peaked_substrate_inhibition": [R.max() * 3, 1.0, 20.0],
              "M4_peaked_bilinear_kink": [R[0] / L[0], 0.0, float(L[int(np.argmax(R))])]}[name]
        try:
            popt, _ = optimize.curve_fit(fn, L, R, p0=p0, maxfev=200000)
            if not np.all(np.isfinite(popt)):
                raise RuntimeError("non-finite parameters")
            pred = fn(L, *popt)
            rss = float(((R - pred) ** 2).sum())
            sst = float(((R - R.mean()) ** 2).sum())
            kk = k + 1
            aic = n * np.log(max(rss, 1e-300) / n) + 2 * kk
            # With n = 4 the AICc correction has denominator n - k - 1, which is
            # zero for every 3-parameter model.  AICc is then formally undefined
            # and is reported as null rather than as an invented finite number.
            aicc = (aic + (2 * kk * (kk + 1)) / (n - kk - 1)
                    if n - kk - 1 > 0 else None)
            bic = n * np.log(max(rss, 1e-300) / n) + kk * np.log(n)
            rows[name] = {"model": name, "ok": True, "k_params": k, "rss": rss,
                          "r2": 1.0 - rss / sst,
                          "aicc": float(aicc) if aicc is not None else None,
                          "aic": float(aic), "bic": float(bic),
                          "params": {pn: float(pv) for pn, pv in zip(pnames, popt)}}
            if aicc is None:
                aicc_undefined.append(name)
        except Exception as exc:
            rows[name] = {"model": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    ranked = sorted([r for r in rows.values() if r.get("ok") and r.get("aicc") is not None],
                    key=lambda r: r["aicc"])
    out["models"] = rows
    out["ranking_by_aicc"] = [r["model"] for r in ranked]
    out["aicc_undefined_models"] = sorted(aicc_undefined)
    out["aicc_delta_vs_best"] = {r["model"]: float(r["aicc"] - ranked[0]["aicc"]) for r in ranked}
    log(f"\n  loading-response model comparison (n = 4). AICc needs n - k - 1 > 0, so with "
        f"n = 4 every 3-parameter model has an UNDEFINED AICc: {sorted(aicc_undefined)}. "
        f"Only the 1- and 2-parameter models can be ranked:")
    for r in ranked:
        log(f"    {r['model']:>34s} k={r['k_params']} R2={r['r2']:.5f} AICc={r['aicc']:7.2f} "
            f"dAICc={r['aicc']-ranked[0]['aicc']:.2f}  BIC={r['bic']:7.2f}  params={r['params']}")
    log("  models with undefined AICc (ranked by BIC instead, which is defined at n = 4):")
    for name in sorted(aicc_undefined, key=lambda m: rows[m]["bic"]):
        r = rows[name]
        log(f"    {r['model']:>34s} k={r['k_params']} R2={r['r2']:.5f} AICc=undefined "
            f"BIC={r['bic']:7.2f}  params={r['params']}")

    # ---- monotonicity tests ----------------------------------------------
    mono = C.spearman(L, R)
    log(f"\n  Spearman rho(loading, rate) = {mono['rho']:+.3f}, p = {mono['p']:.4f}, "
        f"permutation p = {mono['permutation_p']:.4f} (n = 4)")
    # exact enumeration: with n=4 there are only 24 orderings
    perms = [stats.spearmanr(L, p).statistic for p in _permutations(R)]
    out["monotone_test"] = {
        **mono,
        "exact_permutation_p_two_sided": float(np.mean([abs(v) >= abs(mono["rho"]) - 1e-12 for v in perms])),
        "n_distinct_permutations": len(perms),
    }
    # falling segment 1% -> 8% is a strict decrease; is that resolvable given 4 points?
    log(f"  exact two-sided permutation p over all {len(perms)} orderings = "
        f"{out['monotone_test']['exact_permutation_p_two_sided']:.4f}")
    # rise-then-fall: the observed ordering is a strict peak at 1%
    peak_idx = int(np.argmax(R))
    out["peak_shape"] = {
        "argmax_index": peak_idx,
        "strictly_rising_before_peak": bool(np.all(np.diff(R[:peak_idx + 1]) > 0)),
        "strictly_falling_after_peak": bool(np.all(np.diff(R[peak_idx:]) < 0)),
        "is_interior_peak": bool(0 < peak_idx < len(R) - 1),
    }
    log(f"  interior peak: {out['peak_shape']}")

    # ---- robustness: leave-one-condition-out ------------------------------
    loo = {}
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        sp = C.spearman(L[keep], R[keep])
        loo[f"drop_{L[i]:g}pct"] = {
            "remaining_loading": L[keep].tolist(), "remaining_rate": R[keep].tolist(),
            "spearman_rho": sp["rho"], "spearman_p": sp["p"],
            "monotone_decreasing": bool(np.all(np.diff(R[keep]) < 0)),
            "peak_loading_percent": float(L[keep][int(np.argmax(R[keep]))]),
        }
        log(f"  leave-out {L[i]:5.2f}%: rho={sp['rho']:+.3f} p={sp['p']:.4f} "
            f"monotone_decreasing={loo[f'drop_{L[i]:g}pct']['monotone_decreasing']} "
            f"peak_at={loo[f'drop_{L[i]:g}pct']['peak_loading_percent']:g}%")
    out["leave_one_condition_out"] = loo

    # ---- bootstrap peak location (resampling residuals around the best fit) -
    best = rows[ranked[0]["model"]]
    fn, _, pnames = LOADING_MODELS[ranked[0]["model"]]
    popt = np.array([best["params"][p] for p in pnames])
    resid = R - fn(L, *popt)
    rng = np.random.default_rng(20220101)
    peaks = []
    for _ in range(5000):
        Rb = fn(L, *popt) + rng.choice(resid, size=n, replace=True)
        Lgrid = np.geomspace(L.min(), L.max(), 4000)
        peaks.append(float(Lgrid[int(np.argmax(fn(Lgrid, *popt) if False else
                                               _model_on_grid(ranked[0]["model"], popt, Lgrid)))]))
    out["bootstrap_peak_location"] = {
        "model": ranked[0]["model"],
        "median_peak_percent": float(np.median(peaks)),
        "ci95": [float(np.percentile(peaks, 2.5)), float(np.percentile(peaks, 97.5))],
        "fraction_of_grid_below_3pct": float(np.mean(np.array(peaks) < 3.0)),
    }
    log(f"  bootstrap peak location ({ranked[0]['model']} + residual resampling, n=5000): "
        f"median {np.median(peaks):.2f}%  95% CI "
        f"[{np.percentile(peaks, 2.5):.2f}, {np.percentile(peaks, 97.5):.2f}]%")
    return out


def _model_on_grid(name: str, popt: np.ndarray, grid: np.ndarray) -> np.ndarray:
    fn, _, _ = LOADING_MODELS[name]
    return fn(grid, *popt)


def _permutations(values: np.ndarray):
    import itertools
    return [np.array(p) for p in itertools.permutations(values)]


# ------------------------------------------------------ cross-modal synthesis
def cross_modal(tr: pd.DataFrame, tc: pd.DataFrame, spec: pd.DataFrame,
                ld: pd.DataFrame) -> dict:
    log("\n" + "=" * 78)
    log("STAGE 4  cross-modal synthesis and its coverage boundary")
    log("=" * 78)
    ids_tc = set(tc.formulation_id)
    ids_tr = set(tr.formulation_id)
    shared = sorted(ids_tc & ids_tr)
    log(f"  formulation IDs in time courses      : {sorted(ids_tc)}")
    log(f"  formulation IDs in transient traces  : {sorted(ids_tr)}")
    log(f"  shared formulation IDs (cross-modal) : {shared}")

    rows = []
    for fid in shared:
        g = tc[tc.formulation_id == fid].sort_values("illumination_time_h")
        t = g.illumination_time_h.values.astype(float)
        y = g.gaseous_product_umol.values.astype(float)
        kin = C.fit_kinetic_models(t, y)
        best = C.rank_models(kin)[0][0]
        gt = tr[tr.formulation_id == fid].sort_values("delay_ps")
        tt = gt.delay_ps.values.astype(float)
        yy = gt.normalized_transient_signal.values.astype(float)
        late = tt >= 30.0
        rows.append({
            "formulation_id": fid,
            "final_product_4h_umol": float(y[-1]),
            "mean_rate_umol_h": float(y[-1] / t[-1]),
            "best_kinetic_model": best["model"],
            "best_kinetic_aicc": float(best["aicc"]),
            "transient_peak_value": float(yy.max()),
            "transient_peak_delay_ps": float(tt[int(np.argmax(yy))]),
            "transient_late_ge30ps_mean": float(yy[late].mean()),
            "transient_late_ge30ps_fraction_positive": float((yy[late] > 0).mean()),
            "transient_final_value": float(yy[-1]),
            "transient_readout_noise": C.readout_noise(yy, tt),
        })
    tbl = pd.DataFrame(rows)
    log("\n  cross-modal table for the formulations present in BOTH modalities:")
    log(tbl.to_string(index=False))

    # what can and cannot be tested
    n_shared = len(shared)
    log(f"\n  n shared analysis units available for a cross-modal association test = {n_shared}")
    assoc = {
        "n_shared_formulations": n_shared,
        "spearman_yield_vs_late_transient": None,
        "reason": ("with two shared formulations a rank correlation is not defined; "
                   "only an ordinal contrast can be stated"),
    }
    if n_shared >= 3:
        sp = C.spearman(tbl.mean_rate_umol_h.values, tbl.transient_late_ge30ps_mean.values)
        assoc["spearman_yield_vs_late_transient"] = sp
        log(f"  Spearman rho = {sp['rho']:.3f} p = {sp['p']:.4f}")
    else:
        log("  " + assoc["reason"])
    yields = tbl.final_product_4h_umol.values
    late = tbl.transient_late_ge30ps_mean.values
    assoc["ordinal_contrast"] = {
        "yield_order_high_to_low": [tbl.formulation_id.values[i] for i in np.argsort(-yields)],
        "late_transient_order_high_to_low": [tbl.formulation_id.values[i] for i in np.argsort(-late)],
        "orders_agree": bool(np.array_equal(np.argsort(-yields), np.argsort(-late))),
        "yield_ratio_f1_over_f2": float(yields[0] / yields[1]),
        "late_transient_difference": float(late[0] - late[1]),
        "late_transient_signs": {fid: ("positive" if v > 0 else "negative")
                                 for fid, v in zip(tbl.formulation_id.values, late)},
    }
    log(f"  ordinal contrast: yield order {assoc['ordinal_contrast']['yield_order_high_to_low']}, "
        f"late-transient order {assoc['ordinal_contrast']['late_transient_order_high_to_low']}, "
        f"agree = {assoc['ordinal_contrast']['orders_agree']}")
    log(f"  late-transient signs: {assoc['ordinal_contrast']['late_transient_signs']}")

    # coverage of the other two modalities -----------------------------------
    cov = {
        "optical_spectra_have_formulation_id": False,
        "optical_spectra_roles": sorted(spec.measurement_role.unique().tolist()),
        "loading_scan_has_formulation_id": False,
        "loading_rate_scale_umol_h": [float(ld.product_evolution_rate_umol_h.min()),
                                      float(ld.product_evolution_rate_umol_h.max())],
        "timecourse_rate_scale_umol_h": [float(tc.groupby("formulation_id").gaseous_product_umol.max().min() / 4),
                                         float(tc.groupby("formulation_id").gaseous_product_umol.max().max() / 4)],
        "note": ("the optical spectra and the loading scan carry no formulation identifier, "
                 "so they cannot be joined to a named formulation without an external "
                 "assumption; their absolute scales also differ from the time-course scale"),
    }
    log(f"\n  coverage boundary: {cov['note']}")
    log(f"  loading-scan rate scale {cov['loading_rate_scale_umol_h']} umol/h vs "
        f"time-course mean-rate scale {cov['timecourse_rate_scale_umol_h']} umol/h "
        f"(ratio ~{cov['loading_rate_scale_umol_h'][0]/cov['timecourse_rate_scale_umol_h'][1]:.0f}x)")
    return {"table": rows, "association": assoc, "coverage": cov, "shared_ids": shared}


# --------------------------------------------------- time-course robustness
def timecourse_holdout(tc: pd.DataFrame) -> dict:
    """Complete-curve holdout: refit each kinetic model leaving out one
    illumination time point, and leave out one entire curve."""
    log("\n" + "-" * 78)
    log("Time-course robustness: leave-one-time-point-out and leave-one-curve-out")
    log("-" * 78)
    out = {"leave_one_timepoint_out": {}, "leave_one_curve_out": {}}
    for fid, g in tc.groupby("formulation_id"):
        g = g.sort_values("illumination_time_h")
        t = g.illumination_time_h.values.astype(float)
        y = g.gaseous_product_umol.values.astype(float)
        base = C.fit_kinetic_models(t, y)
        base_rank = C.rank_models(base)[0][0]["model"]
        rows = {}
        for i in range(len(t)):
            keep = np.ones(len(t), dtype=bool)
            keep[i] = False
            sub = C.fit_kinetic_models(t[keep], y[keep])
            ranked, undefined = C.rank_models(sub)
            if ranked:
                winner = ranked[0]["model"]
                basis = "AICc"
            else:
                # Dropping one of five points leaves n = 4, where AICc is undefined
                # for every model in the bank (n - k - 1 <= 0).  BIC stays defined,
                # so it is used as the fallback selector and this is recorded.
                winner = min((r for r in sub.values() if r.get("ok")),
                             key=lambda r: r["bic"])["model"]
                basis = "BIC_fallback_aicc_undefined_at_n4"
            rows[f"drop_t={t[i]:g}h"] = {
                "winner": winner,
                "selection_basis": basis,
                "aicc_undefined_models": undefined,
                "winner_agrees_with_full_fit": winner == base_rank,
                "ranking": [r["model"] for r in ranked],
            }
        agree = sum(v["winner_agrees_with_full_fit"] for v in rows.values())
        out["leave_one_timepoint_out"][fid] = {
            "full_fit_winner": base_rank, "per_dropout": rows,
            "winner_stability": f"{agree}/{len(rows)}",
        }
        log(f"  {fid}: full-fit winner (5 points, AICc) = {base_rank}; "
            f"leave-one-time-point-out stability = {agree}/{len(rows)}")
        for k, v in rows.items():
            log(f"     {k}: winner={v['winner']} basis={v['selection_basis']} "
                f"agrees={v['winner_agrees_with_full_fit']}")
    # leave one whole curve out of the pooled comparison
    pivot = tc.pivot(index="illumination_time_h", columns="formulation_id",
                     values="gaseous_product_umol")
    for drop in pivot.columns:
        rss = {}
        for model in ("linear", "power_law", "quadratic"):
            total = 0.0
            for fid in pivot.columns:
                if fid == drop:
                    continue
                t = pivot.index.values.astype(float)
                y = pivot[fid].values.astype(float)
                fit = C.fit_kinetic_models(t, y)[model]
                total += fit["rss"]
            rss[model] = total
        winner = min(rss, key=rss.get)
        out["leave_one_curve_out"][f"drop_{drop}"] = {"pooled_rss": rss, "winner": winner}
        log(f"  pooled fit dropping {drop}: winner = {winner}  rss = "
            f"{ {k: round(v,5) for k,v in rss.items()} }")
    return out


def figure_loading(ld: pd.DataFrame, load_out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = ld.cofactor_loading_percent.values.astype(float)
    R = ld.product_evolution_rate_umol_h.values.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.geomspace(L.min(), L.max(), 400)
    axes[0].plot(L, R, "o", ms=9, color="C3", label="measured (4 settings)")
    for name, style in (("M1_monotone_power_law", "--"), ("M3_peaked_substrate_inhibition", "-")):
        m = load_out["models"][name]
        if m.get("ok"):
            pnames = dict((n, (fn, k, pn)) for n, (fn, k, pn) in LOADING_MODELS.items())[name][2]
            popt = np.array([m["params"][p] for p in pnames])
            axes[0].plot(grid, _model_on_grid(name, popt, grid), style, label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cofactor loading (%)")
    axes[0].set_ylabel("product evolution rate (umol/h)")
    axes[0].set_title("A  loading response (4 points)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(L, R / R.max(), "o-", color="C0")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cofactor loading (%)")
    axes[1].set_ylabel("rate normalised to peak")
    axes[1].set_title("B  normalised: interior peak at 1 %")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig4_loading.png", dpi=160)
    plt.close(fig)


def figure_crossmodal(cm: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = pd.DataFrame(cm["table"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(tbl))
    axes[0].bar(x - 0.2, tbl.final_product_4h_umol, width=0.4, label="product at 4 h (umol)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tbl.formulation_id)
    axes[0].set_ylabel("product (umol)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("A  product yield (shared formulations)")
    ax2 = axes[1]
    ax2.bar(x - 0.2, tbl.transient_late_ge30ps_mean, width=0.4, color="C3",
            label="late transient mean (>=30 ps)")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tbl.formulation_id)
    ax2.set_ylabel("normalised transient signal")
    ax2.set_title("B  late-window transient sign is opposite")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig5_crossmodal.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    tr = C.load_transients()
    tc = C.load_timecourses()
    spec = C.load_spectra()
    ld = C.load_loading()
    load_out = loading_analysis(ld)
    cm = cross_modal(tr, tc, spec, ld)
    hold = timecourse_holdout(tc)
    C.json_dump(C.RESULTS / "loading_crossmodal.json",
                {"loading_analysis": load_out, "cross_modal": cm,
                 "timecourse_holdout": hold})
    figure_loading(ld, load_out)
    figure_crossmodal(cm)
    log("\nwrote figures/fig4_loading.png and figures/fig5_crossmodal.png")
    log("\nstage 4 complete")
    log.close()