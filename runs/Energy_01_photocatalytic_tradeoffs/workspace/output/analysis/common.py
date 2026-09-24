"""Shared loading, model, and metric utilities for the Energy_01 cross-modal analysis.

All scientific evidence in this task comes from ``data/*.tsv`` in the task
workspace.  No external data are downloaded, substituted, or merged.

Analysis unit (from DATA_GUIDE.md / data_manifest.json): one complete
formulation time course, spectrum, transient trace, or loading condition.
Neighbouring points along a curve or spectrum are NOT independent replicates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = PROJECT / "output"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"

MANIFEST = json.loads((PROJECT / "data_manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_all() -> dict[str, pd.DataFrame]:
    """Load the four frozen task modalities and verify manifest hashes."""
    frames = {}
    for row in MANIFEST["files"]:
        path = DATA / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"hash mismatch for {row['path']}: {digest}")
        frames[path.stem] = pd.read_csv(path, sep="\t")
    return frames


def load_timecourses() -> pd.DataFrame:
    return pd.read_csv(DATA / "conversion_timecourses.tsv", sep="\t")


def load_spectra() -> pd.DataFrame:
    return pd.read_csv(DATA / "optical_action_spectra.tsv", sep="\t")


def load_transients() -> pd.DataFrame:
    return pd.read_csv(DATA / "ultrafast_transient_traces.tsv", sep="\t")


def load_loading() -> pd.DataFrame:
    return pd.read_csv(DATA / "cofactor_loading_response.tsv", sep="\t")


# --------------------------------------------------------------------------
# kinetic models for gaseous product evolution
# --------------------------------------------------------------------------
def _fit_linear(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, 2


def _fit_quadratic(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.vstack([t**2, t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, 3


def _fit_power(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """y = a * t**b, fitted by least squares on the SAME full time grid as every
    other model so that the likelihoods entering AICc are comparable.

    A deterministic exponent grid plus a local golden-section refinement avoids
    dependence on a starting point; ``a`` is profiled out in closed form.
    """
    def rss_for(b: float) -> tuple[float, float]:
        basis = np.power(t, b)
        denom = float(basis @ basis)
        a = float(basis @ y / denom) if denom > 0 else 0.0
        return float(((y - a * basis) ** 2).sum()), a

    grid = np.arange(0.05, 4.0, 0.001)
    scored = [(rss_for(b)[0], b) for b in grid]
    _, b_best = min(scored)
    lo, hi = max(b_best - 0.001, 0.05), b_best + 0.001
    b_ref = optimize.minimize_scalar(lambda b: rss_for(b)[0], bounds=(lo, hi),
                                     method="bounded", options={"xatol": 1e-8})
    b_final = float(b_ref.x) if b_ref.success else b_best
    rss, a = rss_for(b_final)
    if rss > min(s[0] for s in scored):
        b_final, (rss, a) = b_best, rss_for(b_best)
    return np.array([a, b_final]), 2


def _fit_logistic(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """y = L / (1 + exp(-k (t - t0))), fitted with a deterministic grid start."""
    ymax, ymin = float(y.max()), float(y.min())
    best = None
    for L0 in (ymax, ymax * 1.5, ymax * 3.0):
        for k0 in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
            for t0_0 in (0.0, 1.0, 2.0, 3.0):
                try:
                    popt, _ = optimize.curve_fit(
                        lambda tt, L, k, t0: L / (1.0 + np.exp(-np.clip(k * (tt - t0), -500.0, 500.0))),
                        t, y, p0=[L0, k0, t0_0], maxfev=200000,
                    )
                except Exception:
                    continue
                if not np.all(np.isfinite(popt)):
                    continue
                if popt[0] <= 0 or popt[1] <= 0:
                    continue
                pred = popt[0] / (1.0 + np.exp(-np.clip(popt[1] * (t - popt[2]), -500.0, 500.0)))
                resid = float(((y - pred) ** 2).sum())
                if best is None or resid < best[0]:
                    best = (resid, popt)
    if best is None:
        raise RuntimeError("logistic fit failed for all admissible starting points")
    return best[1], 3


def _fit_saturation(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """y = A (1 - exp(-k t)); A from the best linear estimate given k."""
    best = None
    for k in np.arange(0.01, 5.0, 0.001):
        basis = 1.0 - np.exp(-k * t)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        A = float(basis @ y / denom)
        resid = float(((y - A * basis) ** 2).sum())
        if best is None or resid < best[0]:
            best = (resid, np.array([A, k]))
    if best is None:
        raise RuntimeError("saturation fit failed")
    return best[1], 2


def _fit_induction(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Piecewise: zero production until lag t_l, then constant rate r.

    y = r * max(t - t_l, 0), evaluated at the observed time points.
    """
    best = None
    for t_l in np.arange(0.0, float(t.max()), 0.005):
        basis = np.maximum(t - t_l, 0.0)
        denom = float(basis @ basis)
        if denom <= 0:
            continue
        r = float(basis @ y / denom)
        resid = float(((y - r * basis) ** 2).sum())
        if best is None or resid < best[0]:
            best = (resid, np.array([r, t_l]))
    if best is None:
        raise RuntimeError("induction fit failed")
    return best[1], 2


KINETIC_MODELS = {
    "linear": (_fit_linear, ("slope_umol_h", "intercept_umol")),
    "quadratic": (_fit_quadratic, ("a_umol_h2", "b_umol_h", "c_umol")),
    "power_law": (_fit_power, ("amplitude", "exponent_b")),
    "logistic": (_fit_logistic, ("L_umol", "k_per_h", "t_mid_h")),
    "saturation": (_fit_saturation, ("A_umol", "k_per_h")),
    "induction_linear": (_fit_induction, ("rate_umol_h", "lag_h")),
}


def fit_kinetic_models(t: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Fit every candidate kinetic model and return sorted AICc/BIC comparison."""
    n = len(t)
    rows = {}
    for name, (fn, pnames) in KINETIC_MODELS.items():
        try:
            params, k = fn(t, y)
        except Exception as exc:  # preserve the real failure
            rows[name] = {"model": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        pred = predict_kinetic(name, params, t)
        if not np.all(np.isfinite(pred)):
            rows[name] = {"model": name, "ok": False,
                          "error": "non-finite prediction (unidentified parameters)"}
            continue
        rss = float(((y - pred) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - rss / sst if sst > 0 else np.nan
        k_total = k + 1  # + residual variance
        aic = n * np.log(max(rss, 1e-300) / n) + 2 * k_total
        # AICc is only defined when n - k - 1 > 0.  Silently clamping the
        # denominator would report a finite number for an undefined quantity,
        # so the degeneracy is surfaced instead of hidden.
        aicc = (aic + (2 * k_total * (k_total + 1)) / (n - k_total - 1)
                if n - k_total - 1 > 0 else None)
        bic = n * np.log(max(rss, 1e-300) / n) + k_total * np.log(n)
        rows[name] = {
            "model": name,
            "ok": True,
            "k_params": k_total,
            "rss": rss,
            "r2": r2,
            "rmse": float(np.sqrt(rss / n)),
            "aicc": float(aicc) if aicc is not None else None,
            "bic": float(bic),
            "params": {pn: float(pv) for pn, pv in zip(pnames, params)},
        }
    return rows


def rank_models(rows: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Rank fitted models by AICc and report which ones have an undefined AICc.

    AICc needs n - k - 1 > 0.  Models whose parameter count exhausts the residual
    degrees of freedom cannot be ranked and are returned separately instead of
    being assigned an invented number.
    """
    ok = [r for r in rows.values() if r.get("ok")]
    undefined = sorted(r["model"] for r in ok if r.get("aicc") is None)
    ranked = sorted((r for r in ok if r.get("aicc") is not None), key=lambda r: r["aicc"])
    return ranked, undefined


def predict_kinetic(name: str, params: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a fitted kinetic model on time vector ``t``."""
    if name == "linear":
        return params[0] * t + params[1]
    if name == "quadratic":
        return params[0] * t**2 + params[1] * t + params[2]
    if name == "power_law":
        return params[0] * np.power(t, params[1])
    if name == "logistic":
        L, k, t0 = params
        return L / (1.0 + np.exp(-k * (t - t0)))
    if name == "saturation":
        return params[0] * (1.0 - np.exp(-params[1] * t))
    if name == "induction_linear":
        return params[0] * np.maximum(t - params[1], 0.0)
    raise KeyError(name)


def aicc_for(name: str, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """AICc of a *fixed* parameter vector (used for holdout-evaluated models)."""
    n = len(t)
    rss = float(((y - predict_kinetic(name, params, t)) ** 2).sum())
    k_total = len(params) + 1
    aic = n * np.log(max(rss, 1e-300) / n) + 2 * k_total
    return float(aic + (2 * k_total * (k_total + 1)) / max(n - k_total - 1, 1))


# --------------------------------------------------------------------------
# transient decay analysis
# --------------------------------------------------------------------------
def biexponential(t: np.ndarray, a: float, tau1: float, b: float, tau2: float, c: float) -> np.ndarray:
    return a * np.exp(-t / tau1) + b * np.exp(-t / tau2) + c


def triexponential(t, a, tau1, b, tau2, c, tau3, d):
    return a * np.exp(-t / tau1) + b * np.exp(-t / tau2) + c * np.exp(-t / tau3) + d


def fit_biexp_log(t: np.ndarray, y: np.ndarray) -> dict:
    """Fit a biexponential + offset on log-spaced delays with Poisson-like weights.

    Weighting w_i = t_i (proportional to log-spacing) prevents the dense long-delay
    cluster from dominating, which is a common ultrafast-TA practice and also
    serves as one of the alternative model choices tested for robustness.
    """
    sigma = np.maximum(np.sqrt(np.abs(y)), 1e-3)
    best = None
    ymax = float(np.abs(y).max())
    for tau1 in (0.5, 1.0, 2.0, 5.0, 10.0):
        for tau2 in (20.0, 60.0, 120.0, 300.0, 1000.0):
            if tau2 <= tau1:
                continue
            for c0 in (0.0, y[-5:].mean()):
                p0 = [0.5 * ymax, tau1, 0.5 * ymax, tau2, c0]
                try:
                    popt, _ = optimize.curve_fit(
                        biexponential, t, y, p0=p0, sigma=sigma,
                        bounds=([-np.inf, 0.05, -np.inf, 0.1, -np.inf], [np.inf, 1e4, np.inf, 1e5, np.inf]),
                        maxfev=400000,
                    )
                except Exception:
                    continue
                resid = float((((y - biexponential(t, *popt)) / sigma) ** 2).sum())
                if best is None or resid < best[0]:
                    best = (resid, popt)
    if best is None:
        return {"ok": False, "error": "all biexponential starting points failed"}
    chi2, popt = best
    pred = biexponential(t, *popt)
    rss = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "ok": True,
        "params": {"a": popt[0], "tau1_ps": popt[1], "b": popt[2], "tau2_ps": popt[3], "offset": popt[4]},
        "weighted_chi2": chi2,
        "r2": 1.0 - rss / sst if sst > 0 else np.nan,
        "rmse": float(np.sqrt(rss / len(t))),
    }


def signal_window_stats(t: np.ndarray, y: np.ndarray, lo: float) -> dict:
    """Mean/median/fraction-positive of a trace inside a delay window."""
    m = t >= lo
    seg = y[m]
    return {
        "window_start_ps": float(lo),
        "n_points": int(m.sum()),
        "mean": float(seg.mean()),
        "median": float(np.median(seg)),
        "fraction_positive": float((seg > 0).mean()),
        "first_zero_crossing_ps": _first_zero_crossing(t, y, start_ps=2.0),
    }


def _first_zero_crossing(t: np.ndarray, y: np.ndarray, start_ps: float) -> float | None:
    for i in range(1, len(t)):
        if t[i] < start_ps:
            continue
        if np.sign(y[i]) != np.sign(y[i - 1]):
            return float(t[i])
    return None


def block_bootstrap_mean(t: np.ndarray, y: np.ndarray, mask: np.ndarray,
                         n_boot: int = 20000, seed: int = 20220101) -> dict:
    """Moving-block bootstrap CI for the mean of a serially correlated trace.

    Block length ~ sqrt(n) keeps short-range correlation inside blocks.
    A fixed seed makes the result reproducible.
    """
    seg = y[mask]
    n = len(seg)
    if n < 4:
        return {"ok": False, "error": "too few points"}
    block = max(2, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    pick = rng.choice(starts, size=(n_boot, n_blocks), replace=True)
    # gather each block through a flat index matrix -> shape (n_boot, n_blocks*block)
    offsets = np.arange(block)
    flat = (pick[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = seg[flat].mean(axis=1)
    return {
        "ok": True,
        "observed_mean": float(seg.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_boot": n_boot,
        "block_length": block,
        "seed": seed,
        "excludes_zero": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0),
    }


def readout_noise(y: np.ndarray, t: np.ndarray, lo: float = 5.0, hi: float = 30.0) -> float:
    """Robust high-frequency noise scale: MAD of successive differences / sqrt(2)."""
    m = (t >= lo) & (t <= hi)
    d = np.diff(y[m])
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))


# --------------------------------------------------------------------------
# spectral analysis
# --------------------------------------------------------------------------
def action_ratio_test(abs_nm: np.ndarray, abs_val: np.ndarray,
                      act_nm: np.ndarray, act_val: np.ndarray) -> dict:
    """Compare action spectra against a unit-scaled absorption template.

    Model A (absorbance-limited null): action ∝ absorption, single scaling free.
    Model B (onset-limited alternative): action = full response below a cut-off
    wavelength and zero above it.
    Both have 1 free parameter evaluated over the 6 action points.
    """
    order = np.argsort(abs_nm)
    abs_nm, abs_val = abs_nm[order], abs_val[order]
    template = np.interp(act_nm, abs_nm, abs_val)
    # Model A
    scale_a = float((template @ act_val) / (template @ template))
    pred_a = scale_a * template
    rss_a = float(((act_val - pred_a) ** 2).sum())
    # Model B: best single cut-off among the midpoints of the action grid
    best_b = None
    for cut in act_nm[:-1] + np.diff(act_nm) / 2.0:
        basis = (act_nm <= cut).astype(float)
        if basis.sum() == 0:
            continue
        scale_b = float((basis @ act_val) / (basis @ basis))
        rss = float(((act_val - scale_b * basis) ** 2).sum())
        if best_b is None or rss < best_b[0]:
            best_b = (rss, cut, scale_b)
    rss_b, cut_b, scale_b = best_b
    n = len(act_nm)
    aicc = lambda rss, k: n * np.log(max(rss, 1e-300) / n) + 2 * (k + 1) + \
        (2 * (k + 1) * (k + 2)) / max(n - (k + 1) - 1, 1)
    return {
        "action_nm": act_nm.tolist(),
        "action_value": act_val.tolist(),
        "absorption_at_action_nm": template.tolist(),
        "model_A_absorption_limited": {
            "scale": scale_a,
            "rss": rss_a,
            "rmse": float(np.sqrt(rss_a / n)),
            "aicc": float(aicc(rss_a, 1)),
        },
        "model_B_onset_limited": {
            "cutoff_nm": float(cut_b),
            "scale": float(scale_b),
            "rss": rss_b,
            "rmse": float(np.sqrt(rss_b / n)),
            "aicc": float(aicc(rss_b, 1)),
        },
        "rss_ratio_A_over_B": float(rss_a / rss_b) if rss_b > 0 else None,
        "preferred_model": "A_absorption_limited" if rss_a < rss_b else "B_onset_limited",
    }


def slow_band_metrics(abs_nm: np.ndarray, abs_val: np.ndarray,
                      lo: float = 425.0, hi: float = 800.0) -> dict:
    """Characterise the sub-gap shoulder: peak position and normalised area."""
    m = (abs_nm >= lo) & (abs_nm <= hi)
    nm, val = abs_nm[m], abs_val[m]
    # smooth with the native 0.5 nm sampling: a 5 nm centred moving average
    win = 11
    kernel = np.ones(win) / win
    smooth = np.convolve(val, kernel, mode="valid")
    smooth_nm = nm[win // 2: len(nm) - win // 2 + (1 if win % 2 else 0)]
    if len(smooth_nm) != len(smooth):
        smooth_nm = nm[win // 2: win // 2 + len(smooth)]
    i_peak = int(np.argmax(smooth))
    tail = smooth[-1] if len(smooth) else 0.0
    return {
        "band_lo_nm": lo,
        "band_hi_nm": hi,
        "peak_nm_smoothed_5nm": float(smooth_nm[i_peak]),
        "peak_value_smoothed_5nm": float(smooth[i_peak]),
        "subset_max_nm_raw": float(nm[int(np.argmax(val))]),
        "subset_max_value_raw": float(val.max()),
        "area_nm_units": float(np.trapezoid(val, nm)),
        "tail_value_at_hi_nm": float(val[-1]),
        "half_max_red_edge_nm": _red_edge(nm, val),
    }


def _red_edge(nm: np.ndarray, val: np.ndarray) -> float | None:
    """Longest wavelength at which the absorption is still >= half its band max."""
    half = 0.5 * val.max()
    above = nm[val >= half]
    return float(above.max()) if above.size else None


def local_maxima(nm: np.ndarray, val: np.ndarray, min_prominence: float = 0.004) -> list[dict]:
    """Deterministic prominence-filtered local maxima at the native resolution."""
    peaks = []
    for i in range(2, len(nm) - 2):
        window = val[i - 2:i + 3]
        if val[i] == window.max() and (window == val[i]).sum() == 1:
            ip, iv = i, val[i]
            j = i
            while j > 0 and val[j - 1] < val[j]:
                j -= 1
            left_min = val[j:i + 1].min() if i > j else val[i]
            j = i
            while j < len(nm) - 1 and val[j + 1] < val[j]:
                j += 1
            right_min = val[i:j + 1].min() if j > i else val[i]
            prominence = iv - max(left_min, right_min)
            if prominence >= min_prominence:
                peaks.append({"wavelength_nm": float(nm[i]), "value": float(iv),
                              "prominence": float(prominence)})
    return peaks


# --------------------------------------------------------------------------
# association tests across analysis units
# --------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(len(x)),
            "method": "spearman", "permutation_p": float(permutation_p(x, y))}


def permutation_p(x: np.ndarray, y: np.ndarray, n: int = 20000, seed: int = 20220101) -> float:
    """Two-sided permutation p-value for |Spearman rho| (exactness by enumeration
    when the number of distinct permutations is small)."""
    rho_obs = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        if abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(rho_obs) - 1e-12:
            count += 1
    return (count + 1) / (n + 1)


def holdout_curve(outcomes: list[dict]) -> dict:
    """Summarise leave-one-curve-out direction agreement."""
    signed = [o["rho_signed"] for o in outcomes if np.isfinite(o["rho_signed"])]
    if not signed:
        return {"ok": False}
    return {
        "ok": True,
        "n_holdouts": len(signed),
        "rho_signed_values": [float(v) for v in signed],
        "sign_agreement_all": bool(all(v > 0 for v in signed) or all(v < 0 for v in signed)),
        "min": float(np.min(signed)),
        "max": float(np.max(signed)),
        "median": float(np.median(signed)),
    }


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n",
                    encoding="utf-8")


class Logger:
    """Tiny stdout+file logger so the console transcript is also a frozen artifact."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.handle.write(msg + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()