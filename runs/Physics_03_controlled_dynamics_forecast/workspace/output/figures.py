#!/usr/bin/env python3
"""Figure generator: pure Python standard library, emits raw SVG.

Reads:
  - output/frequency_results.json
  - output/trajectory_results.json
  - data/normalized_views/*.tsv

Writes to output/figures/:
  - freq_reference_vs_forecast.svg : one panel per freq condition
    (control_a, control_b, signed_control), reference vs forecast scatter
    with the y=x identity line.
  - traj_reference_vs_forecast.svg : three time-series panels
    (condition_01/02/03) of reference_response and forecast_response vs time.
  - error_bar.svg : bar chart of freq mean_abs_error per condition and
    traj rmse per condition.
"""

import csv
import json
import os
from xml.sax.saxutils import escape

BASE = "/data/user/conversations/conv-ef9804c2-78a8-407c-8722-19f66e9fbe2d/project"
DATA = os.path.join(BASE, "data", "normalized_views")
OUT = os.path.join(BASE, "output", "figures")

os.makedirs(OUT, exist_ok=True)

FREQ_RESULTS = json.load(open(os.path.join(BASE, "output", "frequency_results.json")))
TRAJ_RESULTS = json.load(open(os.path.join(BASE, "output", "trajectory_results.json")))

FREQ_ORDER = ["control_a", "control_b", "signed_control"]
FREQ_TSV = {
    "control_a": "control_a_frequency_response.tsv",
    "control_b": "control_b_frequency_response.tsv",
    "signed_control": "signed_control_frequency_response.tsv",
}
TRAJ_ORDER = ["condition_01", "condition_02", "condition_03"]
TRAJ_TSV = {
    "condition_01": "trajectory_condition_01.tsv",
    "condition_02": "trajectory_condition_02.tsv",
    "condition_03": "trajectory_condition_03.tsv",
}

BG = "#FFFFFF"
GRID = "#E6E6E6"
AXIS = "#555555"
REF_BLUE = "#2C7BB6"
FC_ORANGE = "#E67E22"
IDENTITY = "#C0392B"
FONT = "Helvetica, Arial, sans-serif"


def fnum(v, d=2):
    return (("%." + str(d) + "f") % v) if d else ("%.0f" % v)


def write_svg(filename, parts):
    path = os.path.join(OUT, filename)
    with open(path, "w") as fh:
        fh.write("\n".join(parts) + "\n")
    return path, os.path.getsize(path)


def svg_open(title, w, h):
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d" font-family="%s">' % (w, h, w, h, FONT),
        '<title>%s</title>' % escape(title),
        '<rect width="%d" height="%d" fill="%s"/>' % (w, h, BG),
        '<text x="%d" y="%d" text-anchor="middle" font-size="18" '
        'font-weight="bold" fill="#000000">%s</text>' % (w // 2, 26, escape(title)),
    ]


def svg_close(parts):
    parts.append("</svg>")
    return parts


def load_freq_pairs(cond):
    path = os.path.join(DATA, FREQ_TSV[cond])
    rows = []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            rows.append((float(row["reference_frequency"]), float(row["forecast_frequency"])))
    return rows


def load_traj(cond):
    path = os.path.join(DATA, TRAJ_TSV[cond])
    times, ref, fc = [], [], []
    with open(path, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            times.append(float(row["time"]))
            ref.append(float(row["reference_response"]))
            fc.append(float(row["forecast_response"]))
    return times, ref, fc


# ---------------------------------------------------------------------------
# Figure 1: frequency reference vs forecast, one panel per condition + y=x line
# ---------------------------------------------------------------------------
def freq_panel(x, y, w, h, cond, res):
    pairs = load_freq_pairs(cond)
    refs = [p[0] for p in pairs]
    fcs = [p[1] for p in pairs]
    L, B, T, R = 70, 45, 40, 15
    pw = w - L - R
    ph = h - T - B
    parts = []
    parts.append('<text x="%d" y="%d" font-size="14" font-weight="bold" fill="#000000">'
                 '%s: MAE=%s, n=%d, RMSE=%s</text>'
                 % (x + 10, y + T - 14, escape(cond), fnum(res["mean_abs_error"]),
                    res["n_pairs"], fnum(res["rmse"])))
    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#FFFFFF" '
                 'stroke="%s" stroke-width="1"/>' % (x + L, y + T, pw, ph, AXIS))
    ng = 5
    for i in range(ng + 1):
        gx = x + L + pw * i / ng
        gy = y + T + ph * i / ng
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5"/>'
                     % (gx, y + T, gx, y + T + ph, GRID))
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5"/>'
                     % (x + L, gy, x + L + pw, gy, GRID))
    allv = refs + fcs
    vmin = min(allv)
    vmax = max(allv)
    pad = max((vmax - vmin) * 0.06, 0.5)
    vmin -= pad
    vmax += pad

    def px(v):
        return x + L + pw * (v - vmin) / (vmax - vmin)

    def py(v):
        return y + T + ph * (1.0 - (v - vmin) / (vmax - vmin))

    # y = x identity line
    parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" '
                 'stroke-width="1.5" stroke-dasharray="6,4"/>'
                 % (px(vmin), py(vmin), px(vmax), py(vmax), IDENTITY))
    parts.append('<text x="%s" y="%s" font-size="10" fill="%s">y = x</text>'
                 % (px(vmax) - 20, py(vmax) - 6, IDENTITY))
    # points: reference on the identity line (blue), forecast (orange)
    for ref, fc in pairs:
        parts.append('<circle cx="%s" cy="%s" r="4" fill="%s" stroke="#FFFFFF" stroke-width="0.5"/>'
                     % (px(ref), py(ref), REF_BLUE))
        parts.append('<circle cx="%s" cy="%s" r="4" fill="%s" stroke="#FFFFFF" stroke-width="0.5"/>'
                     % (px(ref), py(fc), FC_ORANGE))
    # ticks
    for i in range(ng + 1):
        vx = vmin + (vmax - vmin) * i / ng
        vy = vmin + (vmax - vmin) * i / ng
        gx = x + L + pw * i / ng
        gy = y + T + ph * i / ng
        parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="9" fill="#000000">%s</text>'
                     % (gx, y + T + ph + 16, fnum(vx)))
        parts.append('<text x="%s" y="%s" text-anchor="end" font-size="9" fill="#000000">%s</text>'
                     % (x + L - 8, gy + 3, fnum(vy)))
    # axis titles
    parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="12" fill="#000000">'
                 'Reference frequency</text>' % (x + L + pw / 2, y + T + ph + 38))
    parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="12" fill="#000000" '
                 'transform="translate(%s %s) rotate(-90)">Forecast frequency</text>'
                 % (x + L - 42, y + T + ph / 2, x + L - 42, y + T + ph / 2))
    return parts


def build_freq():
    PANEL_W, PANEL_H = 560, 430
    GAP, M = 20, 20
    W = 3 * PANEL_W + 2 * GAP + 2 * M
    H = 64 + PANEL_H + 40
    parts = svg_open("Frequency: Reference vs Forecast", W, H)
    parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="12" fill="#333333">'
                 'One panel per condition (control_a, control_b, signed_control) with y=x identity line</text>'
                 % (W // 2, 44))
    for i, cond in enumerate(FREQ_ORDER):
        x0 = M + i * (PANEL_W + GAP)
        y0 = 64
        parts += freq_panel(x0, y0, PANEL_W, PANEL_H, cond, FREQ_RESULTS[cond])
    return write_svg("freq_reference_vs_forecast.svg", svg_close(parts))


# ---------------------------------------------------------------------------
# Figure 2: trajectory time series, three panels
# ---------------------------------------------------------------------------
def traj_panel(x, y, w, h, cond, res):
    times, refs, fcs = load_traj(cond)
    L, B, T, R = 65, 40, 40, 15
    pw = w - L - R
    ph = h - T - B
    parts = []
    parts.append('<text x="%d" y="%d" font-size="14" font-weight="bold" fill="#000000">'
                 '%s: RMSE=%s, Pearson r=%s</text>'
                 % (x + 10, y + T - 14, escape(cond), fnum(res["rmse"]), fnum(res["pearson_corr"])))
    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#FFFFFF" '
                 'stroke="%s" stroke-width="1"/>' % (x + L, y + T, pw, ph, AXIS))
    ng = 5
    for i in range(ng + 1):
        gx = x + L + pw * i / ng
        gy = y + T + ph * i / ng
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5"/>'
                     % (gx, y + T, gx, y + T + ph, GRID))
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5"/>'
                     % (x + L, gy, x + L + pw, gy, GRID))
    allv = refs + fcs
    vmin = min(allv)
    vmax = max(allv)
    pad = max((vmax - vmin) * 0.05, 0.05)
    vmin -= pad
    vmax += pad
    tmin = min(times)
    tmax = max(times)

    def px(t):
        return x + L + pw * (t - tmin) / (tmax - tmin)

    def py(v):
        return y + T + ph * (1.0 - (v - vmin) / (vmax - vmin))

    # downsample to at most ~1500 points per series to keep SVG manageable
    step = max(1, len(times) // 1500)
    idx = range(0, len(times), step)
    pts_ref = " ".join("%.1f,%.1f" % (px(times[i]), py(refs[i])) for i in idx)
    parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.2"/>'
                 % (pts_ref, REF_BLUE))
    pts_fc = " ".join("%.1f,%.1f" % (px(times[i]), py(fcs[i])) for i in idx)
    parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.2"/>'
                 % (pts_fc, FC_ORANGE))
    # ticks
    for i in range(ng + 1):
        vx = tmin + (tmax - tmin) * i / ng
        vy = vmin + (vmax - vmin) * i / ng
        gx = x + L + pw * i / ng
        gy = y + T + ph * i / ng
        parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="9" fill="#000000">%s</text>'
                     % (gx, y + T + ph + 14, fnum(vx, 1)))
        parts.append('<text x="%s" y="%s" text-anchor="end" font-size="9" fill="#000000">%s</text>'
                     % (x + L - 8, gy + 3, fnum(vy)))
    # axis titles
    parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="12" fill="#000000">Time</text>'
                 % (x + L + pw / 2, y + T + ph + 32))
    parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="12" fill="#000000" '
                 'transform="translate(%s %s) rotate(-90)">Response</text>'
                 % (x + L - 42, y + T + ph / 2, x + L - 42, y + T + ph / 2))
    # legend (top-right)
    lx = x + L + pw - 130
    ly = y + T + 14
    parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="2"/>'
                 % (lx, ly, lx + 24, ly, REF_BLUE))
    parts.append('<text x="%s" y="%s" font-size="10" fill="#000000">Reference</text>'
                 % (lx + 30, ly + 4))
    parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="2"/>'
                 % (lx, ly + 18, lx + 24, ly + 18, FC_ORANGE))
    parts.append('<text x="%s" y="%s" font-size="10" fill="#000000">Forecast</text>'
                 % (lx + 30, ly + 22))
    return parts


def build_traj():
    PANEL_W, PANEL_H = 620, 420
    GAP, M = 20, 20
    W = 3 * PANEL_W + 2 * GAP + 2 * M
    H = 64 + PANEL_H + 40
    parts = svg_open("Trajectory: Reference vs Forecast Response", W, H)
    parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="12" fill="#333333">'
                 'Time series of reference_response and forecast_response (condition_01/02/03)</text>'
                 % (W // 2, 44))
    for i, cond in enumerate(TRAJ_ORDER):
        x0 = M + i * (PANEL_W + GAP)
        y0 = 64
        parts += traj_panel(x0, y0, PANEL_W, PANEL_H, cond, TRAJ_RESULTS[cond])
    return write_svg("traj_reference_vs_forecast.svg", svg_close(parts))


# ---------------------------------------------------------------------------
# Figure 3: bar chart of freq MAE and traj RMSE per condition
# ---------------------------------------------------------------------------
def build_error_bar():
    W, H = 760, 520
    L, B, T, R = 70, 55, 40, 30
    pw = W - L - R
    ph = H - T - B
    cats = FREQ_ORDER + TRAJ_ORDER
    vals = [FREQ_RESULTS[c]["mean_abs_error"] for c in FREQ_ORDER] + \
           [TRAJ_RESULTS[c]["rmse"] for c in TRAJ_ORDER]
    colors = [REF_BLUE] * 3 + [FC_ORANGE] * 3
    vmax = max(vals) * 1.15
    parts = svg_open("Error Metrics", W, H)
    parts.append('<text x="%d" y="%d" text-anchor="middle" font-size="12" fill="#333333">'
                 'Frequency mean_abs_error (blue) and trajectory rmse (orange) per condition</text>'
                 % (W // 2, 44))
    parts.append('<rect x="%d" y="%d" width="%d" height="%d" fill="#FFFFFF" '
                 'stroke="%s" stroke-width="1"/>' % (L, T + 20, pw, ph, AXIS))
    ng = 5
    for i in range(ng + 1):
        gy = T + 20 + ph * i / ng
        vy = vmax * i / ng
        parts.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="0.5"/>'
                     % (L, gy, L + pw, gy, GRID))
        parts.append('<text x="%s" y="%s" text-anchor="end" font-size="10" fill="#000000">%s</text>'
                     % (L - 8, gy + 3, fnum(vy, 2)))
    bw = pw / len(cats) * 0.6
    for i, c in enumerate(cats):
        v = vals[i]
        cx = L + pw * (i + 0.5) / len(cats)
        bh = ph * v / vmax
        parts.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>'
                     % (cx - bw / 2, T + 20 + ph - bh, bw, bh, colors[i]))
        parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="10" fill="#000000">%s</text>'
                     % (cx, T + 20 + ph - bh - 5, fnum(v, 3)))
    for i, c in enumerate(cats):
        cx = L + pw * (i + 0.5) / len(cats)
        parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="11" fill="#000000">%s</text>'
                     % (cx, T + 20 + ph + 16, c))
    parts.append('<text x="%s" y="%s" text-anchor="middle" font-size="12" fill="#000000">Condition</text>'
                 % (L + pw / 2, T + 20 + ph + 42))
    lx = L + pw - 170
    ly = T + 20 + 12
    parts.append('<rect x="%s" y="%s" width="14" height="14" fill="%s"/>' % (lx, ly, REF_BLUE))
    parts.append('<text x="%s" y="%s" font-size="10" fill="#000000">Frequency MAE</text>'
                 % (lx + 20, ly + 10))
    parts.append('<rect x="%s" y="%s" width="14" height="14" fill="%s"/>' % (lx, ly + 22, FC_ORANGE))
    parts.append('<text x="%s" y="%s" font-size="10" fill="#000000">Trajectory RMSE</text>'
                 % (lx + 20, ly + 30))
    return write_svg("error_bar.svg", svg_close(parts))


if __name__ == "__main__":
    results = [build_freq(), build_traj(), build_error_bar()]
    for path, size in results:
        assert os.path.exists(path) and size > 0, "missing or empty figure: %s" % path
        print("OK %s (%d bytes)" % (path, size))
