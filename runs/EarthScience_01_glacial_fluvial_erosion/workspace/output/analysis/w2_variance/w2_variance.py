#!/usr/bin/env python3
"""
w2_variance: variance decomposition and explanatory-power ranking.

Ranks candidate explanations for log10(Erosion rate) on the pooled
glacial+nonglacial erosion-rate table, tests the no-stable-relation baseline,
quantifies clustering / effective sample size, and checks for a Simpson-style
sign reversal of the pooled ER~TI slope once method is controlled.

Estimation uses explicit design-matrix OLS (normal equations) so that the
417-level C(Reference) model is tractable; R^2 / AIC / BIC reproduce
statsmodels OLS exactly (asserted in-script).

All inputs are the frozen workspace TSVs. Deterministic, no network.
Run from project root:  python3 output/analysis/w2_variance/w2_variance.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

BASE = "/data/user/conversations/conv-ab8a0c4c-4c0f-46e9-96bb-16e130938d6a/project"
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "output", "analysis", "w2_variance")
os.makedirs(OUT, exist_ok=True)
FMT = "%.10g"


def log(msg):
    print(msg)
    sys.stdout.flush()


def num(s):
    return pd.to_numeric(s.str.strip().replace({"": np.nan}), errors="coerce")


# ---------------------------------------------------------------- load ------
g = pd.read_csv(os.path.join(RAW, "glacial_erosion_Earth.tsv"), sep="\t", dtype=str)
n = pd.read_csv(os.path.join(RAW, "nonglacial_erosion_Earth.tsv"), sep="\t", dtype=str)
g["Process"] = "Glacial"
n["Process"] = "Nonglacial"
pool = pd.concat([g, n], ignore_index=True, sort=False)

pool["ER"] = num(pool["Erosion rate (mm/yr)"])
pool["TI"] = num(pool["Time interval (yr)"])
pool["A"] = num(pool["Area (km2)"])
pool["S"] = num(pool["Slope (m/km)"])
pool["P"] = num(pool["Precipitation (mm/yr)"])
pool["Lat"] = num(pool["Latitude"])
pool["Lon"] = num(pool["Longitude"])
pool["Ex"] = num(pool["Exhumation (m)"])

# conventions: logER over rate>0, logTI over interval>0, logX = log10(X)
pool["logER"] = np.log10(pool["ER"].where(pool["ER"] > 0))
pool["logTI"] = np.log10(pool["TI"].where(pool["TI"] > 0))
pool["logA"] = np.log10(pool["A"].where(pool["A"] > 0))
pool["logS"] = np.log10(pool["S"].where(pool["S"] > 0))
pool["logP"] = np.log10(pool["P"].where(pool["P"] > 0))
pool["ET"] = pool["ER"] * pool["TI"]

# --------------------------------------------------------- unit identity ----
ex_mask = pool["Ex"].notna() & pool["ER"].notna() & pool["TI"].notna()
exsub = pool.loc[ex_mask].copy()
exsub["ET"] = exsub["ER"] * exsub["TI"]
exsub["EXm"] = exsub["Ex"] * 1000.0
for tname, tol in [("rtol_1e-6", 1e-6), ("rtol_1e-3", 1e-3)]:
    exsub[tname] = np.isclose(exsub["ET"], exsub["EXm"], rtol=tol, atol=1e-9)
exsub["rel_err"] = (exsub["ET"] - exsub["EXm"]).abs() / exsub["ET"].replace(0, np.nan)
ident_glac = exsub["Process"] == "Glacial"
ident_nong = exsub["Process"] == "Nonglacial"
identity = {
    "rows_with_exhumation_and_ER_and_TI": int(len(exsub)),
    "glacial_rows": int(ident_glac.sum()),
    "nonglacial_rows": int(ident_nong.sum()),
    "ERxTI_eq_Exhumation_x1000_rtol1e-6": int(exsub["rtol_1e-6"].sum()),
    "ERxTI_eq_Exhumation_x1000_rtol1e-3": int(exsub["rtol_1e-3"].sum()),
    "glacial_rtol1e-3": int((ident_glac & exsub["rtol_1e-3"]).sum()),
    "nonglacial_rtol1e-3": int((ident_nong & exsub["rtol_1e-3"]).sum()),
    "max_relative_error_glacial": float(exsub.loc[ident_glac, "rel_err"].max()),
    "median_relative_error_all": float(exsub["rel_err"].median()),
    "median_relative_error_glacial": float(exsub.loc[ident_glac, "rel_err"].median()),
    "median_relative_error_nonglacial": float(exsub.loc[ident_nong, "rel_err"].median()),
    "unit_identity": "ER(mm/yr)*TI(yr) = mm of erosion; Exhumation(m)*1000 = mm",
    "implied_mechanism": "a near-1:1 identity means measured thickness / time span, i.e. mechanical ER~TI coupling",
}

# ------------------------------------------------------------- samples ------
m_er = pool["logER"].notna()
m_ti = m_er & pool["logTI"].notna()
mx = ["logTI", "logS", "logP", "logA", "Lat"]
m_common = m_ti & pool[mx].notna().all(axis=1)
SAMPLE_MASKS = {"full_logER": m_er, "logTI_subset": m_ti, "common_nested": m_common}
samples = {k: int(v.sum()) for k, v in SAMPLE_MASKS.items()}
log("=== SAMPLES ===")
log(json.dumps(samples))

# ----------------------------------------------------- fast OLS machinery ---
REQUIRED = {
    "null": [],
    "proc": ["Process"],
    "method": ["Methodology"],
    "ti": ["logTI"],
    "comp": ["Compilation"],
    "ref": ["Reference"],
    "ti_plus_method": ["logTI", "Methodology"],
    "ti_x_method": ["logTI", "Methodology"],
    "ti_plus_proc": ["logTI", "Process"],
    "method_plus_proc": ["Methodology", "Process"],
    "method_x_proc": ["Methodology", "Process"],
    "ti_plus_method_proc": ["logTI", "Methodology", "Process"],
    "ti_x_method_proc": ["logTI", "Methodology", "Process"],
    "geology": ["logS", "logP", "logA", "Lat"],
    "ti_plus_geology": ["logTI", "logS", "logP", "logA", "Lat"],
    "geology_plus_method": ["logS", "logP", "logA", "Lat", "Methodology"],
    "cells_plus_ti": ["logTI", "Methodology", "Process"],
    "combined": ["logTI", "logS", "logP", "logA", "Methodology"],
    "combined_plus_lat": ["logTI", "logS", "logP", "logA", "Lat", "Methodology"],
}
FORMULAS = {
    "null": "logER ~ 1",
    "proc": "logER ~ C(Process)",
    "method": "logER ~ C(Methodology)",
    "ti": "logER ~ logTI",
    "comp": "logER ~ C(Compilation)",
    "ref": "logER ~ C(Reference)",
    "ti_plus_method": "logER ~ logTI + C(Methodology)",
    "ti_x_method": "logER ~ logTI * C(Methodology)",
    "ti_plus_proc": "logER ~ logTI + C(Process)",
    "method_plus_proc": "logER ~ C(Methodology) + C(Process)",
    "method_x_proc": "logER ~ C(Methodology) + C(Process) + C(Methodology):C(Process)",
    "ti_plus_method_proc": "logER ~ logTI + C(Methodology) + C(Process)",
    "ti_x_method_proc": "logER ~ logTI * C(Methodology) + C(Process)",
    "geology": "logER ~ logS + logP + logA + Latitude",
    "ti_plus_geology": "logER ~ logTI + logS + logP + logA + Latitude",
    "geology_plus_method": "logER ~ logS + logP + logA + Latitude + C(Methodology)",
    "cells_plus_ti": "logER ~ logTI + C(Methodology):C(Process) (saturated cells + logTI)",
    "combined": "logER ~ logTI + logS + logP + logA + C(Methodology)",
    "combined_plus_lat": "logER ~ logTI + logS + logP + logA + Latitude + C(Methodology)",
}


def cells(df):
    return pd.get_dummies((df["Methodology"].astype(str) + "|" + df["Process"].astype(str))
                          .astype("category"), drop_first=False, dtype=float).to_numpy(float)


def dummies(df, col, drop_first=True):
    return pd.get_dummies(df[col].astype("category"), drop_first=drop_first, dtype=float).to_numpy(float)


def build(name, df):
    """Explicit design matrix for a named model on df (rows already complete)."""
    y = df["logER"].to_numpy(float)
    one = np.ones((len(df), 1))
    if name == "null":
        return y, one
    if name == "proc":
        return y, dummies(df, "Process", drop_first=False)
    if name == "method":
        return y, dummies(df, "Methodology", drop_first=False)
    if name == "comp":
        return y, dummies(df, "Compilation", drop_first=False)
    if name == "ref":
        return y, dummies(df, "Reference", drop_first=False)
    if name == "ti":
        return y, np.column_stack([one, df["logTI"].to_numpy(float)])
    if name == "ti_plus_method":
        return y, np.column_stack([one, df["logTI"].to_numpy(float), dummies(df, "Methodology")])
    if name == "ti_x_method":
        X = np.column_stack([one, df["logTI"].to_numpy(float), dummies(df, "Methodology")])
        D = dummies(df, "Methodology")
        return y, np.column_stack([X, D * df["logTI"].to_numpy(float)[:, None]])
    if name == "ti_plus_proc":
        return y, np.column_stack([one, df["logTI"].to_numpy(float), dummies(df, "Process")])
    if name == "method_plus_proc":
        return y, np.column_stack([one, dummies(df, "Methodology"), dummies(df, "Process")])
    if name == "method_x_proc":
        return y, cells(df)
    if name == "ti_plus_method_proc":
        return y, np.column_stack([one, df["logTI"].to_numpy(float),
                                   dummies(df, "Methodology"), dummies(df, "Process")])
    if name == "ti_x_method_proc":
        X = np.column_stack([one, df["logTI"].to_numpy(float),
                             dummies(df, "Methodology"), dummies(df, "Process")])
        D = dummies(df, "Methodology")
        return y, np.column_stack([X, D * df["logTI"].to_numpy(float)[:, None]])
    if name == "geology":
        return y, np.column_stack([one] + [df[c].to_numpy(float) for c in
                                           ["logS", "logP", "logA", "Lat"]])
    if name == "ti_plus_geology":
        return y, np.column_stack([one, df["logTI"].to_numpy(float)] +
                                  [df[c].to_numpy(float) for c in ["logS", "logP", "logA", "Lat"]])
    if name == "geology_plus_method":
        return y, np.column_stack([one] + [df[c].to_numpy(float) for c in
                                           ["logS", "logP", "logA", "Lat"]] +
                                  [dummies(df, "Methodology")])
    if name == "cells_plus_ti":
        return y, np.column_stack([cells(df), df["logTI"].to_numpy(float)])
    if name == "combined":
        return y, np.column_stack([one, df["logTI"].to_numpy(float)] +
                                  [df[c].to_numpy(float) for c in ["logS", "logP", "logA"]] +
                                  [dummies(df, "Methodology")])
    if name == "combined_plus_lat":
        return y, np.column_stack([one, df["logTI"].to_numpy(float)] +
                                  [df[c].to_numpy(float) for c in
                                   ["logS", "logP", "logA", "Lat"]] +
                                  [dummies(df, "Methodology")])
    raise KeyError(name)


def fit_yX(y, X):
    """OLS via normal equations; mirrors statsmodels gaussian OLS statistics."""
    y = np.asarray(y, float)
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    ssr = float(resid @ resid)
    n_ = len(y)
    k = int(np.linalg.matrix_rank(X))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    adj = 1.0 - (1.0 - r2) * (n_ - 1) / (n_ - k) if n_ - k > 0 else np.nan
    llf = -n_ / 2.0 * (np.log(2 * np.pi) + np.log(ssr / n_) + 1.0)
    return {"n": n_, "k_params": k, "ssr": ssr, "sst": sst, "r2": float(r2),
            "adj_r2": float(adj), "aic": float(-2 * llf + 2 * k),
            "bic": float(-2 * llf + np.log(n_) * k), "llf": float(llf),
            "df_resid": n_ - k, "beta": beta, "X": X,
            "sigma": float(np.sqrt(ssr / (n_ - k)))}


def fit_model(name, df):
    """Available-case fit: drop rows with any NaN in the required columns."""
    req = REQUIRED[name]
    d = df.dropna(subset=req) if req else df
    y, X = build(name, d)
    return fit_yX(y, X), len(d)


def partial_F(full, red):
    d1 = full["k_params"] - red["k_params"]
    d2 = full["df_resid"]
    if d1 <= 0 or d2 <= 0:
        return {"error": "not a valid nested pair", "delta_k": d1, "df2": d2,
                "delta_r2": float(full["r2"] - red["r2"])}
    F = ((red["ssr"] - full["ssr"]) / d1) / (full["ssr"] / d2)
    return {"F": float(F), "df1": d1, "df2": d2, "p": float(stats.f.sf(F, d1, d2)),
            "delta_r2": float(full["r2"] - red["r2"])}


def fit_formula_sm(formula, df):
    r = smf.ols(formula, data=df).fit()
    return {"n": int(r.nobs), "k_params": int(len(r.params)), "ssr": float(r.ssr),
            "sst": float(r.centered_tss), "r2": float(r.rsquared),
            "adj_r2": float(r.rsquared_adj), "aic": float(r.aic), "bic": float(r.bic),
            "llf": float(r.llf), "df_resid": int(r.df_resid), "res": r}


def slope_of(fit, j):
    """OLS coefficient j with classical SE / CI / p."""
    X = fit["X"]
    XtX_inv = np.linalg.pinv(X.T @ X)
    s2 = fit["ssr"] / fit["df_resid"]
    se = float(np.sqrt(max(s2 * XtX_inv[j, j], 0.0)))
    est = float(fit["beta"][j])
    tcrit = float(stats.t.ppf(0.975, fit["df_resid"]))
    t = est / se if se > 0 else np.nan
    return {"estimate": est, "se": se, "ci95": [est - tcrit * se, est + tcrit * se],
            "t": float(t), "p": float(2 * stats.t.sf(abs(t), fit["df_resid"]))}


# ---- self-checks -----------------------------------------------------------
_chk = pool.loc[m_common]
_hand = fit_yX(*build("ti_plus_method_proc", _chk))
_sm = fit_formula_sm("logER ~ logTI + C(Methodology) + C(Process)", _chk)
log("self-check ti+method+proc: hand r2=%.12f sm r2=%.12f | aic %.6f/%.6f | bic %.6f/%.6f | k %d/%d"
    % (_hand["r2"], _sm["r2"], _hand["aic"], _sm["aic"], _hand["bic"], _sm["bic"],
       _hand["k_params"], _sm["k_params"]))
assert abs(_hand["r2"] - _sm["r2"]) < 1e-10 and abs(_hand["aic"] - _sm["aic"]) < 1e-6
assert abs(_hand["bic"] - _sm["bic"]) < 1e-6 and _hand["k_params"] == _sm["k_params"]

_chk2 = pool.loc[m_ti]
_hand2 = fit_yX(*build("ti_x_method", _chk2))
_sm2 = fit_formula_sm("logER ~ logTI * C(Methodology)", _chk2)
log("self-check ti*method: hand r2=%.12f sm r2=%.12f | aic %.6f/%.6f | k %d/%d"
    % (_hand2["r2"], _sm2["r2"], _hand2["aic"], _sm2["aic"], _hand2["k_params"], _sm2["k_params"]))
assert abs(_hand2["r2"] - _sm2["r2"]) < 1e-10 and abs(_hand2["aic"] - _sm2["aic"]) < 1e-6

_yr = pool.loc[m_er, "logER"]
_grp = pool.loc[m_er, "Reference"]
_stat = fit_yX(_yr.to_numpy(float), dummies(pool.loc[m_er], "Reference", drop_first=False))
_ssr_anova = float(((_yr - _grp.map(_yr.groupby(_grp).mean())) ** 2).sum())
log("self-check C(Reference): hand ssr=%.6f anova ssr=%.6f" % (_stat["ssr"], _ssr_anova))
assert abs(_stat["ssr"] - _ssr_anova) < 1e-6
assert _stat["k_params"] == pool.loc[m_er, "Reference"].nunique()

_hand3 = fit_yX(*build("combined", _chk))
_sm3 = fit_formula_sm("logER ~ logTI + logS + logP + logA + C(Methodology)", _chk)
log("self-check combined: hand r2=%.12f sm r2=%.12f | k %d/%d"
    % (_hand3["r2"], _sm3["r2"], _hand3["k_params"], _sm3["k_params"]))
assert abs(_hand3["r2"] - _sm3["r2"]) < 1e-10 and _hand3["k_params"] == _sm3["k_params"]

# --------------------------------------------------------- model fitting ----
MAIN_MODELS = ["null", "proc", "method", "ti", "comp", "ref", "ti_plus_method",
               "ti_x_method", "ti_plus_method_proc", "ti_x_method_proc",
               "method_x_proc", "geology", "combined"]

results = {"samples": samples, "unit_identity": identity, "model_tables": {}}
fitted, tables = {}, {}
for sname, mask in SAMPLE_MASKS.items():
    df_s = pool.loc[mask]
    rows = []
    for lab in MAIN_MODELS:
        f, n_used = fit_model(lab, df_s)
        fitted[(sname, lab)] = f
        rows.append({"sample": sname, "model": lab, "formula": FORMULAS[lab],
                     "n": f["n"], "k_params": f["k_params"], "r2": f["r2"],
                     "adj_r2": f["adj_r2"], "aic": f["aic"], "bic": f["bic"],
                     "ssr": f["ssr"], "sigma": f["sigma"]})
        log("  fit %-13s %-20s n=%-5d k=%-4d r2=%.4f aic=%.1f bic=%.1f"
            % (sname, lab, f["n"], f["k_params"], f["r2"], f["aic"], f["bic"]))
    tables[sname] = pd.DataFrame(rows)
    results["model_tables"][sname] = rows
log("=== model fitting done ===")


def G(s, lab):
    return fitted[(s, lab)]


# ------------------------------------------------- interaction / nested F ---
tests = {}
for sname in SAMPLE_MASKS:
    _df_sm = pool.loc[SAMPLE_MASKS[sname]]
    _mpp = fit_model("method_plus_proc", _df_sm)[0]      # not in the stored catalogue
    tests["%s|ti_x_method_vs_ti_plus_method (equal slope across methods)" % sname] = partial_F(
        G(sname, "ti_x_method"), G(sname, "ti_plus_method"))
    tests["%s|method_x_proc_vs_method_plus_proc (proc effect varies by method)" % sname] = partial_F(
        G(sname, "method_x_proc"), _mpp)
    tests["%s|ti_x_method_proc_vs_method_plus_proc" % sname] = partial_F(
        G(sname, "ti_x_method_proc"), _mpp)
    for lab, nm in [("ti", "logTI"), ("method", "C(Methodology)"), ("proc", "C(Process)"),
                    ("geology", "logS+logP+logA+Lat"), ("comp", "C(Compilation)"),
                    ("ref", "C(Reference)"), ("ti_plus_method", "logTI+C(Method)"),
                    ("ti_plus_method_proc", "logTI+C(Meth)+C(Proc)"),
                    ("combined", "logTI+geology+C(Meth)")]:
        d = G(sname, lab)
        tests["%s|%s_vs_null" % (sname, lab)] = partial_F(d, G(sname, "null"))
        tests["%s|%s_overall_F" % (sname, lab)] = {
            "F": float((d["sst"] - d["ssr"]) / (d["k_params"] - 1) / (d["ssr"] / d["df_resid"])),
            "df1": d["k_params"] - 1, "df2": d["df_resid"],
            "p": float(stats.f.sf((d["sst"] - d["ssr"]) / (d["k_params"] - 1) /
                                  (d["ssr"] / d["df_resid"]), d["k_params"] - 1, d["df_resid"])),
            "r2": d["r2"]}
results["interaction_and_nested_tests"] = tests

# ------------------------------------------- incremental R^2 (common set) ---
dfc = pool.loc[m_common]


def P(full_lab, red_lab, df=dfc):
    return partial_F(*[fit_model(l, df)[0] for l in (full_lab, red_lab)])


inc = {}
# timescale last
inc["common_nested|logTI added last after method"] = P("ti_plus_method", "method")
inc["common_nested|logTI added last after method+proc"] = P("ti_plus_method_proc", "method_plus_proc")
inc["common_nested|logTI added last after method+proc+method:proc (saturated cells)"] = P(
    "cells_plus_ti", "method_x_proc")
inc["common_nested|logTI added last after method*proc interaction"] = P(
    "ti_x_method_proc", "method_x_proc")
inc["common_nested|logTI*method added last after method+proc"] = P(
    "ti_x_method_proc", "method_plus_proc")
inc["common_nested|logTI added last after geology"] = P("ti_plus_geology", "geology")
inc["common_nested|logTI added last after geology(S,P,A)+method"] = P("combined", "geology_plus_method")
inc["common_nested|logTI added last after geology(S,P,A,Lat)+method"] = P(
    "combined_plus_lat", "geology_plus_method")
# Latitude added last after ti+method+geology (nested within combined)
inc["common_nested|Latitude added last after ti+method+geology"] = P("combined_plus_lat", "combined")
# process last
inc["common_nested|process added last after method"] = P("method_plus_proc", "method")
inc["common_nested|process added last after method+logTI"] = P("ti_plus_method_proc", "ti_plus_method")
inc["common_nested|process added last after method*logTI"] = P("ti_x_method_proc", "ti_x_method")
# method last
inc["common_nested|method added last after logTI"] = P("ti_plus_method", "ti")
inc["common_nested|method added last after logTI+proc"] = P("ti_plus_method_proc", "ti_plus_proc")
inc["common_nested|method added last after logTI+geology"] = P("combined", "ti_plus_geology")
# geology last
inc["common_nested|geology added last after logTI+method"] = P("combined", "ti_plus_method")
results["incremental_r2"] = inc

# sequential R^2 ladder on the common sample
ladder, prev = [], None
for lab, f in [("null", "logER ~ 1"),
               ("method", "logER ~ C(Methodology)"),
               ("method+ti", "logER ~ C(Methodology) + logTI"),
               ("method+ti+proc", "logER ~ C(Methodology) + logTI + C(Process)"),
               ("method*ti+proc", "logER ~ C(Methodology) * logTI + C(Process)"),
               ("ti+method+geology", "logER ~ logTI + C(Methodology) + logS + logP + logA")]:
    r = fit_formula_sm(f, dfc)
    ladder.append({"step": lab, "formula": f, "n": r["n"], "k_params": r["k_params"],
                   "r2": r["r2"], "adj_r2": r["adj_r2"], "aic": r["aic"], "bic": r["bic"],
                   "delta_r2_vs_previous": None if prev is None else float(r["r2"] - prev)})
    prev = r["r2"]
results["sequential_ladder_common_nested"] = ladder

# ----------------------------------- no-stable-relation baseline tests ------
dft = pool.loc[m_ti].copy()
y_t = dft["logER"].to_numpy(float)
x_t = dft["logTI"].to_numpy(float)
marg = fit_yX(y_t, np.column_stack([np.ones(len(dft)), x_t]))
sl = slope_of(marg, 1)
t0 = sl["estimate"] / sl["se"]
t1 = (sl["estimate"] + 1.0) / sl["se"]
baseline = {
    "pooled_logER_on_logTI": {
        "n": marg["n"], "slope_dex_per_dex": sl["estimate"], "se": sl["se"],
        "ci95": sl["ci95"], "t": sl["t"], "p_slope_eq_0": sl["p"],
        "r2": marg["r2"], "adj_r2": marg["adj_r2"], "intercept": float(marg["beta"][0]),
        "p_slope_eq_minus1_fixed_thickness": float(2 * stats.t.sf(abs(t1), marg["df_resid"])),
        "note": ("pooled slope tests whether a single stable ER~TI relation exists; "
                 "slope=-1 is the ER*TI=const mechanical-coupling reference"),
    },
}
strat, strat_p = [], []
for meth, sub in dft.groupby("Methodology"):
    f = fit_yX(sub["logER"].to_numpy(float),
               np.column_stack([np.ones(len(sub)), sub["logTI"].to_numpy(float)]))
    s = slope_of(f, 1)
    strat.append({"methodology": meth, "n": int(len(sub)), "slope": s["estimate"],
                  "se": s["se"], "ci95": s["ci95"], "p": s["p"], "r2": f["r2"]})
for proc, sub in dft.groupby("Process"):
    f = fit_yX(sub["logER"].to_numpy(float),
               np.column_stack([np.ones(len(sub)), sub["logTI"].to_numpy(float)]))
    s = slope_of(f, 1)
    strat_p.append({"process": proc, "n": int(len(sub)), "slope": s["estimate"],
                    "se": s["se"], "ci95": s["ci95"], "p": s["p"], "r2": f["r2"]})
baseline["method_stratified_slopes"] = strat
baseline["process_stratified_slopes"] = strat_p
baseline["F_equal_slopes_across_methods"] = partial_F(G("logTI_subset", "ti_x_method"),
                                                      G("logTI_subset", "ti_plus_method"))
baseline["F_equal_slopes_across_processes"] = partial_F(
    fit_model("ti_x_method_proc", dft)[0], fit_model("ti_plus_proc", dft)[0])
# sensitivity: drop Chen et al., 2022 (zero within-compilation logTI variance)
sens = {}
dft_nd = dft[dft["Compilation"] != "Chen et al., 2022"]
if len(dft_nd) > 100:
    fnd = fit_yX(dft_nd["logER"].to_numpy(float),
                 np.column_stack([np.ones(len(dft_nd)), dft_nd["logTI"].to_numpy(float)]))
    snd = slope_of(fnd, 1)
    sens["pooled_slope_excluding_Chen2022"] = {
        "n": fnd["n"], "slope": snd["estimate"], "se": snd["se"], "ci95": snd["ci95"],
        "p": snd["p"], "r2": fnd["r2"]}
    fnd2 = fit_model("ti_x_method", dft_nd)[0]
    snd2 = slope_of(fnd2, 1)
    sens["conditional_slope_excluding_Chen2022"] = {
        "n": fnd2["n"], "slope": snd2["estimate"], "se": snd2["se"],
        "ci95": snd2["ci95"], "p": snd2["p"], "r2": fnd2["r2"]}
baseline["sensitivity_excluding_degenerate_compilation"] = sens
results["no_stable_relation_baseline"] = baseline

# --------------------------------------- mechanical coupling / strata detail
d_all = pool.loc[m_er]
# ER ~ TI slope near -1 within a stratum is the signature of ER = thickness/TI.
coupling = {
    "rationale": ("If erosion rate is computed as thickness/interval, logER = log(thickness) - logTI, "
                  "so the within-stratum slope is -1 by construction (mathematical coupling). "
                  "A slope near 0 implies a scale-independent rate; intermediate values mix both."),
    "slope_minus1_reference": -1.0,
}
# process x method strata slopes (each cell with n>=10); degenerate logTI strata flagged
cell_rows = []
for (proc, meth), sub in dft.groupby(["Process", "Methodology"]):
    if len(sub) < 10:
        cell_rows.append({"process": proc, "methodology": meth, "n": int(len(sub)),
                          "note": "n<10, not fitted"})
        continue
    if float(sub["logTI"].var()) < 1e-12:
        cell_rows.append({"process": proc, "methodology": meth, "n": int(len(sub)),
                          "logTI_var": float(sub["logTI"].var()),
                          "note": "logTI has zero variance in stratum; slope not identified"})
        continue
    f = fit_yX(sub["logER"].to_numpy(float),
               np.column_stack([np.ones(len(sub)), sub["logTI"].to_numpy(float)]))
    s = slope_of(f, 1)
    cell_rows.append({"process": proc, "methodology": meth, "n": int(len(sub)),
                      "slope": s["estimate"], "se": s["se"], "ci95": s["ci95"],
                      "p": s["p"], "r2": f["r2"]})
coupling["process_x_method_strata_slopes"] = cell_rows
# compilation-level slopes (major compilations); degenerate logTI compilations flagged
comp_rows = []
for comp, sub in dft.groupby("Compilation"):
    if len(sub) < 30:
        continue
    if float(sub["logTI"].var()) < 1e-12:
        comp_rows.append({"compilation": comp, "n": int(len(sub)),
                          "logTI_var": float(sub["logTI"].var()),
                          "note": "logTI has zero variance in compilation; slope not identified"})
        continue
    f = fit_yX(sub["logER"].to_numpy(float),
               np.column_stack([np.ones(len(sub)), sub["logTI"].to_numpy(float)]))
    s = slope_of(f, 1)
    comp_rows.append({"compilation": comp, "n": int(len(sub)), "slope": s["estimate"],
                      "se": s["se"], "ci95": s["ci95"], "p": s["p"], "r2": f["r2"]})
coupling["compilation_slopes_n_ge30"] = comp_rows
# data-quality flags: zero-variance logTI compilations (logTI not identified within them)
_zv = [{"compilation": c, "n": int(len(s)),
        "share_of_logER_rows": float(len(s) / len(d_all))}
       for c, s in d_all.groupby("Compilation")
       if s["logTI"].notna().sum() > 1 and float(s["logTI"].var()) < 1e-12]
coupling["zero_variance_logTI_compilations"] = _zv
# share of rows where ER*TI is (near) equal to a stored thickness-like quantity
coupling["exhumation_identity_share_rtol1e-3"] = float(
    identity["ERxTI_eq_Exhumation_x1000_rtol1e-3"] / max(identity["rows_with_exhumation_and_ER_and_TI"], 1))
results["mechanical_coupling_diagnostics"] = coupling
pd.DataFrame(cell_rows).to_csv(os.path.join(OUT, "process_x_method_strata_slopes.csv"),
                               index=False, float_format=FMT)
pd.DataFrame(comp_rows).to_csv(os.path.join(OUT, "compilation_slopes.csv"),
                               index=False, float_format=FMT)

# --------------------------------------------- Simpson / conditional slope --
fcond = fit_model("ti_plus_method", dft)[0]
Xc = fcond["X"]
modc = sm.OLS(y_t, Xc).fit(cov_type="cluster", cov_kwds={"groups": dft["Reference"].to_numpy()})
modm = sm.OLS(y_t, marg["X"]).fit(cov_type="cluster", cov_kwds={"groups": dft["Reference"].to_numpy()})
sc = slope_of(fcond, 1)
simpson = {
    "n": int(len(dft)),
    "marginal_pooled_logTI_slope": {
        "estimate": sl["estimate"], "se_classical": sl["se"], "ci95_classical": sl["ci95"],
        "p_classical": sl["p"], "se_clustered_by_Reference": float(modm.bse[1]),
        "ci95_clustered": [float(x) for x in modm.conf_int()[1]],
        "p_clustered": float(modm.pvalues[1]), "r2": marg["r2"]},
    "conditional_on_method_logTI_slope": {
        "estimate": sc["estimate"], "se_classical": sc["se"], "ci95_classical": sc["ci95"],
        "p_classical": sc["p"], "se_clustered_by_Reference": float(modc.bse[1]),
        "ci95_clustered": [float(x) for x in modc.conf_int()[1]],
        "p_clustered": float(modc.pvalues[1]), "r2": fcond["r2"]},
    "sign_flip": bool(np.sign(sl["estimate"]) != np.sign(sc["estimate"])),
    "r2_gain_from_conditioning": float(fcond["r2"] - marg["r2"]),
    "method_means_logER": {str(k): float(v) for k, v in
                           dft.groupby("Methodology")["logER"].mean().items()},
    "method_means_logTI": {str(k): float(v) for k, v in
                           dft.groupby("Methodology")["logTI"].mean().items()},
    "note": "Simpson check: within-method slopes vs the pooled slope",
}
results["simpson_reversal_check"] = simpson

# --------------------------------------------------------- clustering -------
def icc_oneway(values, groups):
    d_ = pd.DataFrame({"y": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    grand = d_["y"].mean()
    grp = d_.groupby("g")["y"]
    sizes, means = grp.size(), grp.mean()
    k, N = len(sizes), int(sizes.sum())
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(((d_["y"] - d_["g"].map(means)) ** 2).sum())
    sst = float(((d_["y"] - grand) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - float((sizes ** 2).sum()) / N) / (k - 1)
    mbar = N / k
    icc = max((msb - msw) / (msb + (n0 - 1) * msw), 0.0)
    deff_mbar = 1 + (mbar - 1) * icc
    deff_n0 = 1 + (n0 - 1) * icc
    return {"n": N, "n_groups": k, "mean_cluster_size": float(mbar),
            "n0_effective_cluster_size": float(n0), "msb": float(msb), "msw": float(msw),
            "icc": float(icc), "between_cluster_r2": float(ssb / sst),
            "design_effect_mbar": float(deff_mbar), "design_effect_n0": float(deff_n0),
            "effective_n_mbar": float(N / deff_mbar), "effective_n_n0": float(N / deff_n0),
            "se_inflation_factor": float(np.sqrt(deff_mbar))}


clustering = {
    "rows_logER": int(len(d_all)),
    "unique_Reference": int(d_all["Reference"].nunique()),
    "unique_Compilation": int(d_all["Compilation"].nunique()),
    "unique_Locality": int(d_all["Locality"].nunique()),
    "unique_LatLon_pairs": int(d_all.dropna(subset=["Lat", "Lon"]).drop_duplicates(["Lat", "Lon"]).shape[0]),
    "unique_LatLon_pairs_round3": int(d_all.dropna(subset=["Lat", "Lon"])
                                      .assign(la=lambda d: d["Lat"].round(3),
                                              lo=lambda d: d["Lon"].round(3))
                                      .drop_duplicates(["la", "lo"]).shape[0]),
    "unique_Reference_logTI_subset": int(dft["Reference"].nunique()),
    "unique_Locality_logTI_subset": int(dft["Locality"].nunique()),
    "icc_logER_by_Reference": icc_oneway(d_all["logER"], d_all["Reference"]),
    "icc_logER_by_Compilation": icc_oneway(d_all["logER"], d_all["Compilation"]),
    "icc_logER_by_Locality": icc_oneway(d_all["logER"], d_all["Locality"]),
    "icc_resid_of_pooled_logTI_model_by_Reference": icc_oneway(
        y_t - (marg["beta"][0] + marg["beta"][1] * x_t), dft["Reference"]),
    "icc_resid_of_pooled_logTI_model_by_Locality": icc_oneway(
        y_t - (marg["beta"][0] + marg["beta"][1] * x_t), dft["Locality"]),
    "icc_resid_of_method_conditional_model_by_Reference": icc_oneway(
        y_t - Xc @ fcond["beta"], dft["Reference"]),
}
# independent cross-check of the ICC via ML (statsmodels MixedLM, REML/ML)
icc_ml = {}
try:
    _ml_df = d_all.copy()
    _ml_df["ref"] = _ml_df["Reference"].astype("category")
    for method in ["reml", "ml"]:
        m = smf.mixedlm("logER ~ 1", _ml_df, groups=_ml_df["ref"]).fit(reml=(method == "reml"))
        vb = float(m.cov_re.iloc[0, 0])
        vw = float(m.scale)
        icc_ml[method] = {"var_between": vb, "var_within": vw, "icc": vb / (vb + vw),
                          "converged": bool(m.converged), "n_groups": int(_ml_df["ref"].nunique())}
except Exception as e:
    icc_ml["error"] = f"{type(e).__name__}: {e}"
clustering["icc_ml_cross_check_logER_by_Reference"] = icc_ml

rc = d_all["Reference"].value_counts()
clustering["reference_size_distribution"] = {
    "max": int(rc.max()), "singletons": int((rc == 1).sum()),
    "mean": float(rc.mean()), "p50": float(rc.median()), "p90": float(rc.quantile(0.9)),
    "top10": {str(k): int(v) for k, v in rc.head(10).items()},
}
lc = d_all["Locality"].value_counts()
clustering["locality_size_distribution"] = {
    "max": int(lc.max()), "singletons": int((lc == 1).sum()), "p50": float(lc.median()),
}
results["clustering"] = clustering

# ------------------------------------------------------- model CSV export ---
csv_rows = []
for sname, tab in tables.items():
    for _, r in tab.iterrows():
        csv_rows.append({"sample": sname, "model": r["model"], "n": r["n"],
                         "k_params": r["k_params"], "r2": r["r2"], "adj_r2": r["adj_r2"],
                         "aic": r["aic"], "bic": r["bic"]})
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(os.path.join(OUT, "model_comparison_all.csv"), index=False, float_format=FMT)
for sname, fn in [("common_nested", "model_comparison_primary.csv"),
                  ("full_logER", "model_comparison_full_logER.csv"),
                  ("logTI_subset", "model_comparison_logTI_subset.csv")]:
    csv_df[csv_df["sample"] == sname][["model", "n", "k_params", "r2", "adj_r2", "aic", "bic"]] \
        .sort_values("r2", ascending=False).to_csv(os.path.join(OUT, fn), index=False, float_format=FMT)
pd.DataFrame(strat).to_csv(os.path.join(OUT, "method_stratified_slopes.csv"), index=False, float_format=FMT)
pd.DataFrame(strat_p).to_csv(os.path.join(OUT, "process_stratified_slopes.csv"), index=False, float_format=FMT)
pd.DataFrame([{"test": k, **{kk: vv for kk, vv in v.items()}} for k, v in inc.items()]) \
    .to_csv(os.path.join(OUT, "incremental_r2.csv"), index=False, float_format=FMT)

# -------------------------------------------------------------- figures -----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
tab = tables["common_nested"].sort_values("r2")
axes[0].barh(tab["model"], tab["r2"], color="#4c72b0")
axes[0].set_xlabel("R^2 (common sample, n=%d)" % samples["common_nested"])
axes[0].set_title("Explanatory power of competing models (log10 ER)")
axes[0].tick_params(axis="y", labelsize=8)
sl_df = pd.DataFrame(strat)
axes[1].errorbar(sl_df["slope"], np.arange(len(sl_df)),
                 xerr=[sl_df["slope"] - sl_df["ci95"].apply(lambda x: x[0]),
                       sl_df["ci95"].apply(lambda x: x[1]) - sl_df["slope"]],
                 fmt="o", color="#c44e52", capsize=4)
axes[1].axvline(0, color="k", lw=0.9, ls=":")
axes[1].axvline(-1, color="g", lw=0.9, ls="--")
axes[1].set_yticks(np.arange(len(sl_df)))
axes[1].set_yticklabels(["%s (n=%d)" % (m, nn) for m, nn in zip(sl_df["methodology"], sl_df["n"])],
                        fontsize=8)
axes[1].set_xlabel("log-log ER~TI slope (dex/dex)")
axes[1].set_title("Method-stratified timescale slopes\n(dotted: 0; dashed: -1 fixed-thickness)")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "w2_variance_summary.png"), dpi=140)
plt.close(fig)

fig2, ax = plt.subplots(figsize=(6, 4))
rc_sorted = np.sort(rc.values)[::-1]
ax.plot(np.arange(1, len(rc_sorted) + 1), rc_sorted, color="#4c72b0")
ax.set_yscale("log")
ax.set_xlabel("Reference rank"); ax.set_ylabel("rows per Reference (log scale)")
ax.set_title("Cluster sizes by Reference (pooled logER rows, n=%d)" % len(d_all))
plt.tight_layout()
fig2.savefig(os.path.join(OUT, "reference_cluster_sizes.png"), dpi=140)
plt.close(fig2)

# -------------------------------------------------------------- JSON --------
def native(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: native(v) for k, v in o.items() if k not in ("beta", "X", "res")}
    if isinstance(o, list):
        return [native(v) for v in o]
    return o


results["filter_rules"] = {
    "logER": "log10(Erosion rate (mm/yr)) where rate>0; rate<=0 or blank dropped",
    "logTI": "log10(Time interval (yr)) where interval>0",
    "logA/logS/logP": "log10 of Area (km2)/Slope (m/km)/Precipitation (mm/yr) where value>0",
    "Latitude": "raw numeric Latitude in degrees (not logged, no abs)",
    "Process": "row-set label: Glacial vs Nonglacial (source file)",
    "full_logER": "all rows with logER non-missing (n=6711); each model uses its own available-case n",
    "logTI_subset": "rows with logER and logTI non-missing (n=6555)",
    "common_nested": "rows with logER, logTI, logS, logP, logA, Latitude, Methodology, Process all non-missing (n=6450); used for AIC/BIC, F-tests, incremental R^2",
    "aic_bic": "Gaussian OLS llf = -n/2*(log(2pi)+log(SSR/n)+1); AIC=-2llf+2k; BIC=-2llf+log(n)*k; comparable only within the same sample",
    "estimator": "explicit design-matrix OLS (normal equations); asserted against statsmodels and one-way ANOVA in-script",
    "cluster_se": "statsmodels cov_type='cluster' with groups=Reference (CR0-style)",
    "icc": "one-way random-effects ANOVA estimator with n0 (Kish) correction: ICC=(MSB-MSW)/(MSB+(n0-1)MSW), design effect 1+(mbar-1)*ICC",
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(native(results), f, indent=2)

# ------------------------------------------------------------- printout -----
prim = csv_df[csv_df["sample"] == "common_nested"][
    ["model", "n", "k_params", "r2", "adj_r2", "aic", "bic"]].sort_values("r2", ascending=False)
full_tab = csv_df[csv_df["sample"] == "full_logER"][
    ["model", "n", "k_params", "r2", "adj_r2", "aic", "bic"]].sort_values("r2", ascending=False)
ti_tab = csv_df[csv_df["sample"] == "logTI_subset"][
    ["model", "n", "k_params", "r2", "adj_r2", "aic", "bic"]].sort_values("r2", ascending=False)
log("\n=== COMMON-SAMPLE MODEL TABLE (ranked by R2) ===")
log(prim.to_string(index=False))
log("\n=== full_logER MODEL TABLE (ranked by R2; n varies per model) ===")
log(full_tab.to_string(index=False))
log("\n=== logTI_SUBSET MODEL TABLE (ranked by R2) ===")
log(ti_tab.to_string(index=False))
log("\n=== INTERACTION / NESTED TESTS ===")
log(json.dumps(native(tests), indent=2))
log("\n=== INCREMENTAL R2 ===")
log(json.dumps(native(inc), indent=2))
log("\n=== SEQUENTIAL LADDER (common sample) ===")
log(pd.DataFrame(ladder).to_string(index=False))
log("\n=== BASELINE ===")
log(json.dumps(native(baseline), indent=2))
log("\n=== SIMPSON ===")
log(json.dumps(native(simpson), indent=2))
log("\n=== CLUSTERING ===")
log(json.dumps(native(clustering), indent=2))
log("wrote: " + OUT)