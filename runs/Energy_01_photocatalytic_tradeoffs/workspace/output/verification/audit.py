#!/usr/bin/env python3
"""Independent integrity / internal-consistency audit of the four frozen TSVs.

Reads ONLY /data/user/.../project/data/*.tsv and data_manifest.json.
Writes JSON to the verification output directory.
"""
import hashlib
import itertools
import json
import math
import os
import re
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd

DATA = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data"
MANIFEST = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/data_manifest.json"
OUT = "/data/user/conversations/conv-f64cb198-aadc-4b6c-a8d3-7f7608a4e4ea/project/output/verification/audit_results.json"

FILES = [
    "cofactor_loading_response.tsv",
    "conversion_timecourses.tsv",
    "optical_action_spectra.tsv",
    "ultrafast_transient_traces.tsv",
]

manifest = json.load(open(MANIFEST))
report = {"files": {}, "checks": {}, "discrepancies": [], "notes": []}

# ---------------------------------------------------------------- per-file basics
raw = {}
for fn in FILES:
    path = os.path.join(DATA, fn)
    b = open(path, "rb").read()
    raw[fn] = b
    sha = hashlib.sha256(b).hexdigest()
    man = [f for f in manifest["files"] if f["path"] == fn][0]
    df = pd.read_csv(path, sep="\t", keep_default_na=False, na_filter=False)
    # explicit missing-value scan on the raw text
    lines = b.decode("utf-8").split("\r\n")
    trailing_empty = lines[-1] == ""
    body = lines[:-1] if trailing_empty else lines
    header = body[0].split("\t")
    ncols = len(header)
    bad_field_rows = []
    empty_fields = Counter()
    for i, ln in enumerate(body[1:], start=2):
        parts = ln.split("\t")
        if len(parts) != ncols:
            bad_field_rows.append((i, len(parts)))
        for c, p in zip(header, parts):
            if p.strip() == "" or p.strip().lower() in {"nan", "na", "null", "none"}:
                empty_fields[c] += 1
    report["files"][fn] = {
        "sha256_recomputed": sha,
        "sha256_manifest": man["sha256"],
        "sha256_match": sha == man["sha256"],
        "bytes_recomputed": len(b),
        "bytes_manifest": man["bytes"],
        "bytes_match": len(b) == man["bytes"],
        "n_rows_data": len(df),
        "n_rows_file_lines": len(body) - 1,
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "missing_values_per_column": {c: int(df[c].astype(str).str.strip().eq("").sum()) for c in df.columns},
        "empty_or_na_fields_raw_scan": dict(empty_fields),
        "rows_with_wrong_field_count": bad_field_rows,
        "line_terminator": "CRLF" if b.count(b"\r\n") == len(body) else "mixed/other",
        "trailing_newline": trailing_empty,
        "n_duplicate_rows_exact": int(df.duplicated().sum()),
        "n_fully_unique_rows": int(len(df) - df.duplicated().sum()),
    }
    if sha != man["sha256"]:
        report["discrepancies"].append(f"{fn}: sha256 mismatch ({sha} vs manifest {man['sha256']})")
    if len(b) != man["bytes"]:
        report["discrepancies"].append(f"{fn}: byte count mismatch ({len(b)} vs manifest {man['bytes']})")

# ---------------------------------------------------------------- precision helpers
def sig_figs(tok):
    t = tok.strip().lstrip("+-")
    if t == "":
        return None
    if "e" in t.lower():
        t = re.split("[eE]", t)[0]
    if "." in t:
        ip, fp = t.split(".")
        digits = (ip + fp).lstrip("0")
        digits = digits.rstrip("0") if False else digits
        # trailing zeros after a decimal point are significant
        return len(digits) if digits != "" else None
    else:
        d = t.lstrip("0")
        return len(d) if d != "" else None


def dp(tok):
    t = tok.strip().lstrip("+-")
    if "." in t:
        return len(t.split(".")[1])
    return 0


def precision_profile(series_str, is_integer_like=False):
    toks = [t for t in series_str]
    sig = Counter(sig_figs(t) for t in toks)
    dec = Counter(dp(t) for t in toks)
    # round-trip: is the value exactly representable at k decimals?
    rt = {}
    vals = [float(t) for t in toks]
    for k in range(0, 10):
        ok = all(abs(float(t) - round(float(t), k)) < 1e-12 for t in toks)
        if ok:
            rt[k] = True
    return {
        "sig_fig_histogram": {str(k): v for k, v in sorted(sig.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "decimal_places_histogram": {str(k): v for k, v in sorted(dec.items())},
        "exactly_representable_at_k_decimals": [k for k in sorted(rt)],
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def col_tokens(df, c):
    return [str(v) for v in df[c].astype(str)]


# ---------------------------------------------------------------- 1. conversion_timecourses
fn = "conversion_timecourses.tsv"
df = pd.read_csv(os.path.join(DATA, fn), sep="\t")
f = report["files"][fn]
f["formulations"] = sorted(df["formulation_id"].unique().tolist())
f["rows_per_formulation"] = df.groupby("formulation_id").size().to_dict()
f["time_grids"] = {k: v["illumination_time_h"].tolist() for k, v in df.groupby("formulation_id")}
f["time_grid_identical_across_formulations"] = len({tuple(v) for v in f["time_grids"].values()}) == 1
f["precision"] = {
    "illumination_time_h": precision_profile(col_tokens(df, "illumination_time_h")),
    "gaseous_product_umol": precision_profile(col_tokens(df, "gaseous_product_umol")),
}
f["negative_product_values"] = int((df["gaseous_product_umol"] < 0).sum())
# implied average rate over each interval, and cumulative ratio y/t
f["product_per_hour_ratio"] = {
    k: [round(y / t, 6) if t else None for t, y in zip(v["illumination_time_h"], v["gaseous_product_umol"])]
    for k, v in df.groupby("formulation_id")
}
f["max_ratio_spread_within_formulation"] = {
    k: max(r for r in v if r is not None) - min(r for r in v if r is not None)
    for k, v in f["product_per_hour_ratio"].items()
}
f["is_exact_proportional_to_time"] = {
    k: bool(np.allclose(v["gaseous_product_umol"], v["gaseous_product_umol"].iloc[1] / v["illumination_time_h"].iloc[1] * v["illumination_time_h"], atol=1e-12))
    for k, v in df.groupby("formulation_id")
}
# exact-ratio tests between rows of the same formulation (t=1 vs t=2 etc.)
exact_rels = []
for k, v in df.groupby("formulation_id"):
    y = v["gaseous_product_umol"].tolist()
    t = v["illumination_time_h"].tolist()
    for i in range(len(y)):
        for j in range(len(y)):
            if i < j and y[i] != 0:
                pass
    # first-two-point doubling
    exact_rels.append({"formulation": k, "y(t2)==2*y(t1) exactly": bool(Fraction(y[2]) == 2 * Fraction(y[1])) if len(y) > 2 else None})
f["exact_relations"] = exact_rels

# ---------------------------------------------------------------- 2. cofactor_loading_response
fn = "cofactor_loading_response.tsv"
df = pd.read_csv(os.path.join(DATA, fn), sep="\t")
f = report["files"][fn]
f["precision"] = {
    "cofactor_loading_percent": precision_profile(col_tokens(df, "cofactor_loading_percent")),
    "product_evolution_rate_umol_h": precision_profile(col_tokens(df, "product_evolution_rate_umol_h")),
}
f["loading_values"] = df["cofactor_loading_percent"].tolist()
f["any_loading_over_100_percent"] = bool((df["cofactor_loading_percent"] > 100).any())
f["any_negative_rate"] = bool((df["product_evolution_rate_umol_h"] < 0).any())
f["argmax_loading"] = float(df.loc[df["product_evolution_rate_umol_h"].idxmax(), "cofactor_loading_percent"])
f["max_rate"] = float(df["product_evolution_rate_umol_h"].max())

# ---------------------------------------------------------------- 3. optical_action_spectra
fn = "optical_action_spectra.tsv"
df = pd.read_csv(os.path.join(DATA, fn), sep="\t")
f = report["files"][fn]
f["roles"] = df["measurement_role"].value_counts().to_dict()
abs_df = df[df["measurement_role"] == "absorption"].sort_values("wavelength_nm").reset_index(drop=True)
act_df = df[df["measurement_role"] == "action_response"].sort_values("wavelength_nm").reset_index(drop=True)
wl = abs_df["wavelength_nm"].to_numpy()
d = np.diff(wl)
f["absorption_grid"] = {
    "n_points": int(len(wl)),
    "min_nm": float(wl.min()),
    "max_nm": float(wl.max()),
    "expected_n_for_0.5_step_inclusive": int(round((wl.max() - wl.min()) / 0.5)) + 1,
    "all_diffs_equal_0.5": bool(np.allclose(d, 0.5, atol=1e-12)),
    "min_diff": float(d.min()),
    "max_diff": float(d.max()),
    "n_duplicate_wavelengths": int(len(wl) - len(set(wl))),
    "unique_step_values_rounded": sorted({round(float(x), 9) for x in d}),
    "strictly_increasing": bool(np.all(d > 0)),
}
f["absorption_is_exactly_regular_0p5nm"] = bool(
    len(wl) == int(round((wl.max() - wl.min()) / 0.5)) + 1
    and np.all(d == 0.5)
    and len(set(wl)) == len(wl)
)
f["action_wavelengths"] = act_df["wavelength_nm"].tolist()
f["action_values"] = act_df["response_au_or_percent"].tolist()
f["action_on_absorption_grid"] = [bool(w in set(wl)) for w in act_df["wavelength_nm"]]
f["any_action_over_100"] = bool((act_df["response_au_or_percent"] > 100).any())
f["any_action_negative"] = bool((act_df["response_au_or_percent"] < 0).any())
f["absorption_min"] = float(abs_df["response_au_or_percent"].min())
f["absorption_max"] = float(abs_df["response_au_or_percent"].max())
f["absorption_min_wavelength"] = float(abs_df.loc[abs_df["response_au_or_percent"].idxmin(), "wavelength_nm"])
f["absorption_max_wavelength"] = float(abs_df.loc[abs_df["response_au_or_percent"].idxmax(), "wavelength_nm"])
f["n_negative_absorption"] = int((abs_df["response_au_or_percent"] < 0).sum())
f["n_absorption_above_1"] = int((abs_df["response_au_or_percent"] > 1.0).sum())
f["absorption_at_action_wavelengths"] = {
    float(w): None for w in act_df["wavelength_nm"]
}
for w in act_df["wavelength_nm"]:
    m = abs_df[abs_df["wavelength_nm"] == w]
    f["absorption_at_action_wavelengths"][float(w)] = float(m["response_au_or_percent"].iloc[0]) if len(m) else None
f["action_vs_absorption_100x_check"] = {
    float(w): {"action": float(a), "100*absorption_at_same_wl": (100 * f["absorption_at_action_wavelengths"][float(w)]) if f["absorption_at_action_wavelengths"][float(w)] is not None else None}
    for w, a in zip(act_df["wavelength_nm"], act_df["response_au_or_percent"])
}
f["action_vs_100x_abs_max_abs_diff"] = max(
    abs(v["action"] - v["100*absorption_at_same_wl"])
    for v in f["action_vs_absorption_100x_check"].values()
    if v["100*absorption_at_same_wl"] is not None
)
f["precision"] = {
    "wavelength_nm": precision_profile(col_tokens(df, "wavelength_nm")),
    "response_au_or_percent": precision_profile(col_tokens(df, "response_au_or_percent")),
}
f["absorption_only_precision_response"] = precision_profile(col_tokens(abs_df, "response_au_or_percent"))
f["action_only_precision_response"] = precision_profile(col_tokens(act_df, "response_au_or_percent"))
# file ordering: is it sorted by wavelength with action rows interleaved?
f["file_is_wavelength_sorted"] = bool(df["wavelength_nm"].is_monotonic_increasing)
f["n_out_of_order_pairs"] = int((np.diff(df["wavelength_nm"].to_numpy()) < 0).sum())
# duplicate (role,wavelength) pairs
f["n_duplicate_role_wavelength_pairs"] = int(df.duplicated(subset=["measurement_role", "wavelength_nm"]).sum())

# ---------------------------------------------------------------- 4. ultrafast_transient_traces
fn = "ultrafast_transient_traces.tsv"
df = pd.read_csv(os.path.join(DATA, fn), sep="\t")
f = report["files"][fn]
f["formulations"] = sorted(df["formulation_id"].unique().tolist())
g = {k: v.sort_values("delay_ps").reset_index(drop=True) for k, v in df.groupby("formulation_id")}
f["n_points_per_formulation"] = {k: int(len(v)) for k, v in g.items()}
grids = {k: tuple(v["delay_ps"].tolist()) for k, v in g.items()}
f["delay_grids_identical"] = len(set(grids.values())) == 1
if len(set(grids.values())) != 1:
    report["discrepancies"].append("transient delay grids differ between formulations")
    # find differences
    a = np.array(sorted(grids[list(grids)[0]]))
    b = np.array(sorted(grids[list(grids)[1]]))
    f["delay_grid_max_abs_diff"] = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
f["delay_duplicates_within_formulation"] = {k: int(len(v) - v["delay_ps"].nunique()) for k, v in g.items()}
f["delay_strictly_increasing"] = {k: bool(np.all(np.diff(v["delay_ps"]) > 0)) for k, v in g.items()}
for k, v in g.items():
    dl = v["delay_ps"].to_numpy()
    dd = np.diff(dl)
    f.setdefault("delay_grid_shape", {})[k] = {
        "min": float(dl.min()), "max": float(dl.max()), "n": int(len(dl)),
        "min_gap": float(dd.min()), "max_gap": float(dd.max()),
        "mean_gap": float(dd.mean()),
        "gap_ratio_max_over_min": float(dd.max() / dd.min()),
        "is_uniform": bool(np.allclose(dd, dd[0], rtol=0, atol=1e-12)),
    }
# maxima / normalisation
f["trace_max"] = {k: float(v["normalized_transient_signal"].max()) for k, v in g.items()}
f["trace_min"] = {k: float(v["normalized_transient_signal"].min()) for k, v in g.items()}
f["trace_argmax_delay"] = {k: float(v.loc[v["normalized_transient_signal"].idxmax(), "delay_ps"]) for k, v in g.items()}
f["any_trace_max_exactly_1"] = {k: bool(v["normalized_transient_signal"].max() == 1.0) for k, v in g.items()}
f["n_abs_signal_above_1"] = {k: int((v["normalized_transient_signal"] > 1.0).sum()) for k, v in g.items()}
f["precision"] = {
    "delay_ps": precision_profile(col_tokens(df, "delay_ps")),
    "normalized_transient_signal": precision_profile(col_tokens(df, "normalized_transient_signal")),
}
f["delay_precision_by_formulation"] = {k: precision_profile(col_tokens(v, "delay_ps")) for k, v in g.items()}
# exact relation between the two traces at matched delays
if len(set(grids.values())) == 1:
    a = g["formulation_1"]["normalized_transient_signal"].to_numpy()
    b = g["formulation_2"]["normalized_transient_signal"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b != 0, a / b, np.nan)
    f["f1_over_f2_ratio_std"] = float(np.nanstd(ratio))
    f["f1_over_f2_ratio_mean"] = float(np.nanmean(ratio))
    f["diff_std"] = float(np.std(a - b))
    f["diff_mean"] = float(np.mean(a - b))
    f["corr_f1_f2"] = float(np.corrcoef(a, b)[0, 1])
# duplicate rows / duplicate (formulation, delay)
f["n_duplicate_formulation_delay_pairs"] = int(df.duplicated(subset=["formulation_id", "delay_ps"]).sum())
f["file_interleaving_first_8_formulations"] = df["formulation_id"].head(8).tolist()

# ---------------------------------------------------------------- global exact-multiple scans
def exact_multiple_scan(name, cols):
    """cols: dict name -> list of Fraction"""
    out = []
    names = list(cols)
    for a, b in itertools.combinations(names, 2):
        va, vb = cols[a], cols[b]
        if len(va) != len(vb) or len(va) == 0:
            continue
        ratios = set()
        ok = True
        for x, y in zip(va, vb):
            if y == 0:
                ok = False
                break
            ratios.add(x / y)
        if ok and len(ratios) == 1:
            out.append({"file": name, "col_a": a, "col_b": b, "exact_ratio_a_over_b": str(list(ratios)[0])})
    return out


exact_relations_found = []
cdf = pd.read_csv(os.path.join(DATA, "conversion_timecourses.tsv"), sep="\t")
for k, v in cdf.groupby("formulation_id"):
    y = [Fraction(str(x)) for x in v["gaseous_product_umol"]]
    t = [Fraction(str(x)) for x in v["illumination_time_h"]]
    for i in range(1, len(y)):
        exact_relations_found.append({"formulation": k, "t": float(t[i]), "y_over_t": str(y[i] / t[i])})
report["checks"]["conversion_ratio_exact_fractions"] = exact_relations_found
report["checks"]["conversion_y_over_t_all_distinct"] = len({d["y_over_t"] for d in exact_relations_found}) == len(exact_relations_found)

spectra = pd.read_csv(os.path.join(DATA, "optical_action_spectra.tsv"), sep="\t")
sabs = spectra[spectra["measurement_role"] == "absorption"].sort_values("wavelength_nm")
exact_relations_found += exact_multiple_scan(
    "optical_action_spectra(absorption)",
    {"wavelength_nm": [Fraction(str(x)) for x in sabs["wavelength_nm"]],
     "response": [Fraction(str(x)) for x in sabs["response_au_or_percent"]]},
)
report["checks"]["exact_multiple_relations_found"] = exact_relations_found

# is the absorption value sequence an exact linear function of index? (would be tautological)
w = sabs["wavelength_nm"].to_numpy()
r = sabs["response_au_or_percent"].to_numpy()
lin = np.polyfit(w, r, 1)
report["checks"]["absorption_linear_fit_slope"] = float(lin[0])
report["checks"]["absorption_linear_fit_max_resid"] = float(np.max(np.abs(np.polyval(lin, w) - r)))
report["checks"]["absorption_constant_per_wavelength_count"] = int(len(np.unique(r)))

# monotonic checks on spectra edges
report["checks"]["absorption_at_350"] = float(sabs[sabs["wavelength_nm"] == 350.0]["response_au_or_percent"].iloc[0])
report["checks"]["absorption_at_800"] = float(sabs[sabs["wavelength_nm"] == 800.0]["response_au_or_percent"].iloc[0])
tail = sabs[sabs["wavelength_nm"] >= 700.0]["response_au_or_percent"]
report["checks"]["absorption_700_800_max"] = float(tail.max())
report["checks"]["absorption_700_800_min"] = float(tail.min())

# ---------------------------------------------------------------- all-file duplicate row scan
all_dups = {}
for fn in FILES:
    path = os.path.join(DATA, fn)
    df = pd.read_csv(path, sep="\t")
    d = df.duplicated().sum()
    if d:
        all_dups[fn] = {"n": int(d), "examples": df[df.duplicated(keep=False)].head(8).to_dict("records")}
report["checks"]["duplicate_rows_any_file"] = all_dups

# ---------------------------------------------------------------- assemble
n_disc = 0
# assert manifest consistency again explicitly
for fn in FILES:
    fd = report["files"][fn]
    if not fd["sha256_match"]:
        n_disc += 1
report["checks"]["n_sha256_mismatches"] = n_disc
report["checks"]["n_bytes_mismatches"] = sum(1 for fn in FILES if not report["files"][fn]["bytes_match"])

with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2, default=str)

print(json.dumps({
    "sha_ok": {fn: report["files"][fn]["sha256_match"] for fn in FILES},
    "rows": {fn: report["files"][fn]["n_rows_data"] for fn in FILES},
    "cols": {fn: report["files"][fn]["n_cols"] for fn in FILES},
    "dup_rows": {fn: report["files"][fn]["n_duplicate_rows_exact"] for fn in FILES},
    "abs_regular": report["files"]["optical_action_spectra.tsv"]["absorption_is_exactly_regular_0p5nm"],
    "abs_n": report["files"]["optical_action_spectra.tsv"]["absorption_grid"],
    "delay_identical": report["files"]["ultrafast_transient_traces.tsv"]["delay_grids_identical"],
    "trace_max": report["files"]["ultrafast_transient_traces.tsv"]["trace_max"],
    "action_max": report["files"]["optical_action_spectra.tsv"]["action_values"],
    "abs_min_max": (report["files"]["optical_action_spectra.tsv"]["absorption_min"], report["files"]["optical_action_spectra.tsv"]["absorption_max"]),
    "exact_rels": report["checks"]["exact_multiple_relations_found"],
}, indent=2, default=str))