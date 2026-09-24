"""Stage 2 -- optical absorption vs action response.

Question: is the action (photoresponse) spectrum explainable by the measured
absorption spectrum alone (absorption-limited null), or does the action fall
off faster than absorption (an efficiency/onset limitation)?

All comparisons are made on the 6 action wavelengths, which are the only
wavelengths at which both quantities were measured.  The absorption profile is
treated as ONE analysis unit; the 901 absorption points are not independent
replicates of the action points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import common as C

log = C.Logger(C.LOGS / "stage2_spectra.log")


def spectral_descriptors(spec: pd.DataFrame) -> dict:
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    a_nm = abs_.wavelength_nm.values.astype(float)
    a_v = abs_.response_au_or_percent.values.astype(float)
    c_nm = act.wavelength_nm.values.astype(float)
    c_v = act.response_au_or_percent.values.astype(float)

    vis = a_nm >= 400.0
    peaks = C.local_maxima(a_nm, a_v, min_prominence=0.004)
    slow = C.slow_band_metrics(a_nm, a_v, 425.0, 800.0)

    # half-maximum "edges" for each profile, defined on the shared 400-800 region
    act_half = _half_max_crossing(c_nm, c_v)
    abs_half_vis = _half_max_crossing(a_nm[vis], a_v[vis])

    return {
        "absorption_grid_nm": [float(a_nm.min()), float(a_nm.max()), 0.5],
        "action_grid_nm": c_nm.tolist(),
        "action_values_percent": c_v.tolist(),
        "absorption_values_at_action_nm": [float(np.interp(w, a_nm, a_v)) for w in c_nm],
        "absorption_global_max": {"nm": float(a_nm[np.argmax(a_v)]), "value": float(a_v.max())},
        "absorption_visible_max_400_800": {
            "nm": float(a_nm[vis][np.argmax(a_v[vis])]), "value": float(a_v[vis].max())},
        "absorption_half_max_crossing_nm_400_800": abs_half_vis,
        "action_half_max_crossing_nm": act_half,
        "local_maxima_prominence_ge_0.004": peaks,
        "n_local_maxima": len(peaks),
        "slow_band_425_800": slow,
        "absorption_at_800nm": float(a_v[a_nm == 800.0][0]),
        "absorption_negative_min": float(a_v.min()),
        "monotone_decreasing_400_800": bool(np.all(np.diff(a_v[vis]) <= 1e-9) is False),
    }


def _half_max_crossing(nm: np.ndarray, val: np.ndarray) -> float | None:
    """Longest wavelength where a profile is still >= half of its own maximum."""
    half = 0.5 * val.max()
    idx = np.where(val >= half)[0]
    if idx.size == 0:
        return None
    i = int(idx.max())
    if i >= len(nm) - 1:
        return float(nm[i])
    # linear interpolation to the crossing between i and i+1
    x0, x1 = nm[i], nm[i + 1]
    y0, y1 = val[i], val[i + 1]
    if y0 == y1:
        return float(x0)
    return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))


def model_comparison(spec: pd.DataFrame) -> dict:
    """Absorption-limited null vs alternative shapes on the 6 action points.

    Models (all evaluated on identical data, n = 6):
      M0  constant action (1 param)                        -- pure null
      M1  action = s * A(lambda) (1 param)                 -- absorption-limited
      M2  action = s * A(lambda)^p (2 params)              -- sub/super-linear in A
      M3  action = step at a cut-off wavelength (1 param)  -- onset-limited
      M4  action = s * A(lambda) * exp(-(lambda-l0)/w)     -- absorption + roll-off
    """
    log("\n" + "=" * 78)
    log("STAGE 2  action-vs-absorption model comparison (n = 6 shared wavelengths)")
    log("=" * 78)
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    a_nm = abs_.wavelength_nm.values.astype(float)
    a_v = abs_.response_au_or_percent.values.astype(float)
    c_nm = act.wavelength_nm.values.astype(float)
    c_v = act.response_au_or_percent.values.astype(float)
    template = np.interp(c_nm, a_nm, a_v)

    models: dict[str, dict] = {}

    def score(name: str, pred: np.ndarray, k: int) -> None:
        rss = float(((c_v - pred) ** 2).sum())
        sst = float(((c_v - c_v.mean()) ** 2).sum())
        kk = k + 1
        aicc = len(c_v) * np.log(max(rss, 1e-300) / len(c_v)) + 2 * kk + \
            (2 * kk * (kk + 1)) / max(len(c_v) - kk - 1, 1)
        models[name] = {"model": name, "k_params": kk, "rss": rss,
                        "r2": 1.0 - rss / sst, "rmse": float(np.sqrt(rss / len(c_v))),
                        "aicc": float(aicc)}

    # M0 constant
    score("M0_constant_action", np.full_like(c_v, c_v.mean()), 1)
    # M1 absorption-limited
    s1 = float(template @ c_v / (template @ template))
    score("M1_action_proportional_to_A", s1 * template, 1)
    # M2 power law in A, with p profiled on a grid
    best2 = None
    for p in np.arange(0.05, 6.0, 0.001):
        basis = np.power(template, p)
        s = float(basis @ c_v / (basis @ basis))
        rss = float(((c_v - s * basis) ** 2).sum())
        if best2 is None or rss < best2[0]:
            best2 = (rss, p, s)
    score("M2_action_proportional_to_A_power_p", best2[2] * np.power(template, best2[1]), 2)
    models["M2_action_proportional_to_A_power_p"]["p"] = float(best2[1])
    # M3 onset step
    best3 = None
    for cut in np.arange(c_nm.min(), c_nm.max(), 0.5):
        basis = (c_nm <= cut).astype(float)
        if basis.sum() == 0:
            continue
        s = float(basis @ c_v / (basis @ basis))
        rss = float(((c_v - s * basis) ** 2).sum())
        if best3 is None or rss < best3[0]:
            best3 = (rss, cut, s)
    score("M3_onset_step", best3[2] * (c_nm <= best3[1]).astype(float), 1)
    models["M3_onset_step"]["cutoff_nm"] = float(best3[1])
    # M4 absorption * exponential roll-off
    best4 = None
    for l0 in np.arange(450.0, 700.0, 2.5):
        for w in np.arange(5.0, 300.0, 2.5):
            basis = template * np.exp(-np.maximum(c_nm - l0, 0.0) / w)
            denom = float(basis @ basis)
            if denom <= 0:
                continue
            s = float(basis @ c_v / denom)
            rss = float(((c_v - s * basis) ** 2).sum())
            if best4 is None or rss < best4[0]:
                best4 = (rss, l0, w, s)
    basis4 = template * np.exp(-np.maximum(c_nm - best4[1], 0.0) / best4[2])
    score("M4_absorption_times_exp_rolloff", best4[3] * basis4, 3)
    models["M4_absorption_times_exp_rolloff"].update({"l0_nm": float(best4[1]), "w_nm": float(best4[2])})

    ranked = sorted(models.values(), key=lambda r: r["aicc"])
    d_aicc = {r["model"]: float(r["aicc"] - ranked[0]["aicc"]) for r in ranked}
    for r in ranked:
        log(f"  {r['model']:>38s}  k={r['k_params']}  R2={r['r2']:.5f}  RMSE={r['rmse']:.3f}  "
            f"AICc={r['aicc']:.2f}  dAICc={d_aicc[r['model']]:.2f}"
            + ("   <-- best" if r["model"] == ranked[0]["model"] else ""))
    log(f"  M2 best exponent p = {best2[1]:.3f};  M3 best cut-off = {best3[1]:.1f} nm; "
        f"M4 l0 = {best4[1]:.1f} nm, w = {best4[2]:.1f} nm")

    return {
        "models": models,
        "ranking_by_aicc": [r["model"] for r in ranked],
        "aicc_delta_vs_best": d_aicc,
        "template_absorption_at_action_nm": template.tolist(),
        "best_single_scale_absorption_limited": float(s1),
    }


def efficiency_ratio_test(spec: pd.DataFrame) -> dict:
    """Relative quantum efficiency q(lambda) = action / absorption, normalised.

    If the action were strictly absorption-limited, q would be flat.  A
    significant downward trend of q with wavelength is direct evidence of an
    additional efficiency loss beyond absorptance.
    """
    log("\n" + "-" * 78)
    log("Relative efficiency ratio q(lambda) = action / absorption")
    log("-" * 78)
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    template = np.interp(act.wavelength_nm.values, abs_.wavelength_nm.values,
                         abs_.response_au_or_percent.values)
    q_raw = act.response_au_or_percent.values / template
    q = q_raw / q_raw.max()
    sp = C.spearman(act.wavelength_nm.values, q)
    log("  wavelength : q (relative efficiency)")
    for w, qq in zip(act.wavelength_nm.values, q):
        log(f"    {w:6.0f} nm : {qq:.3f}")
    log(f"  Spearman rho(q vs wavelength) = {sp['rho']:.3f}, p = {sp['p']:.4f}, "
        f"permutation p = {sp['permutation_p']:.4f}")
    lo = q[act.wavelength_nm.values <= 550]
    hi = q[act.wavelength_nm.values >= 600]
    return {
        "q_relative": q.tolist(),
        "spearman_q_vs_wavelength": sp,
        "q_mean_400_550nm": float(lo.mean()),
        "q_mean_600_650nm": float(hi.mean()),
        "q_ratio_short_to_long": float(lo.mean() / hi.mean()),
    }


def window_holdout(spec: pd.DataFrame) -> dict:
    """Spectral-window holdout: fit the absorption-limited scale on the short
    wavelengths only, then predict the long wavelengths that were held out."""
    log("\n" + "-" * 78)
    log("Spectral-window holdout for the absorption-limited null")
    log("-" * 78)
    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    template = np.interp(act.wavelength_nm.values, abs_.wavelength_nm.values,
                         abs_.response_au_or_percent.values)
    w = act.wavelength_nm.values
    y = act.response_au_or_percent.values
    out = {"holdouts": {}}
    for name, mask_fit in {
        "fit_400_500_predict_550_650": w <= 500,
        "fit_400_550_predict_600_650": w <= 550,
        "fit_450_600_predict_400_650": (w >= 450) & (w <= 600),
    }.items():
        s = float(template[mask_fit] @ y[mask_fit] / (template[mask_fit] @ template[mask_fit]))
        pred = s * template
        held = ~mask_fit
        err = y[held] - pred[held]
        rel = err / y[held]
        out["holdouts"][name] = {
            "scale_fitted_on_window": s,
            "held_out_nm": w[held].tolist(),
            "held_out_observed": y[held].tolist(),
            "held_out_predicted_absorption_limited": pred[held].tolist(),
            "held_out_relative_error": rel.tolist(),
            "max_abs_relative_error": float(np.max(np.abs(rel))) if held.sum() else None,
        }
        log(f"  {name}: scale={s:.1f}")
        for wl, o, p, r in zip(w[held], y[held], pred[held], rel):
            log(f"     {wl:6.0f} nm observed={o:6.2f} absorption-limited-pred={p:6.2f} "
                f"rel.err={r:+.3f}")
    return out


def figure_spectra(spec: pd.DataFrame, out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abs_ = spec[spec.measurement_role == "absorption"].sort_values("wavelength_nm")
    act = spec[spec.measurement_role == "action_response"].sort_values("wavelength_nm")
    template = np.interp(act.wavelength_nm.values, abs_.wavelength_nm.values,
                         abs_.response_au_or_percent.values)
    models = out["models"]["models"]
    s1 = out["models"]["best_single_scale_absorption_limited"]
    p2 = models["M2_action_proportional_to_A_power_p"]["p"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(abs_.wavelength_nm, abs_.response_au_or_percent, lw=1.2, color="C0")
    axes[0].set_xlabel("wavelength (nm)")
    axes[0].set_ylabel("absorption (au)", color="C0")
    axes[0].set_title("A  absorption profile (901 points)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(act.wavelength_nm, act.response_au_or_percent, "o-", color="C3",
                 label="action response (measured)")
    xs = np.linspace(400, 650, 400)
    tpl_s = np.interp(xs, abs_.wavelength_nm, abs_.response_au_or_percent)
    axes[1].plot(xs, s1 * tpl_s, "--", color="C0",
                 label=f"absorption-limited (1 scale = {s1:.0f})")
    axes[1].plot(xs, models["M2_action_proportional_to_A_power_p"]["params_out"] if False else
                 np.nan * xs, alpha=0)
    s2 = float(template @ act.response_au_or_percent.values / (template @ template))
    # refit power model on the dense grid for the curve
    bp = None
    for p in np.arange(0.05, 6.0, 0.01):
        basis = np.power(template, p)
        s = float(basis @ act.response_au_or_percent.values / (basis @ basis))
        rss = float(((act.response_au_or_percent.values - s * basis) ** 2).sum())
        if bp is None or rss < bp[0]:
            bp = (rss, p, s)
    axes[1].plot(xs, bp[2] * np.power(tpl_s, bp[1]), "-", color="C2",
                 label=f"power-in-A (p = {bp[1]:.2f})")
    axes[1].set_xlabel("wavelength (nm)")
    axes[1].set_ylabel("action response (%)")
    axes[1].set_title("B  action spectrum vs models")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    q = out["ratio"]["q_relative"]
    axes[2].plot(act.wavelength_nm, q, "o-", color="C4")
    axes[2].set_ylim(0, 1.1)
    axes[2].set_xlabel("wavelength (nm)")
    axes[2].set_ylabel("relative efficiency q (normalised)")
    axes[2].set_title("C  q = action/absorption (flat if absorption-limited)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig2_spectra.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    spec = C.load_spectra()
    desc = spectral_descriptors(spec)
    models = model_comparison(spec)
    ratio = efficiency_ratio_test(spec)
    hold = window_holdout(spec)
    payload = {"descriptors": desc, "models": models, "ratio": ratio, "window_holdout": hold}
    C.json_dump(C.RESULTS / "spectra_analysis.json", payload)

    log("\nspectral descriptors:")
    for key in ("absorption_global_max", "absorption_visible_max_400_800",
                "absorption_half_max_crossing_nm_400_800", "action_half_max_crossing_nm",
                "n_local_maxima", "absorption_at_800nm", "slow_band_425_800"):
        log(f"  {key}: {desc[key]}")

    figure_spectra(spec, payload)
    log("\nwrote figures/fig2_spectra.png")
    log("\nstage 2 complete")
    log.close()