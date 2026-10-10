"""Parser for the block-structured CSV tables in data/source_tables.

The raw tables are spreadsheet-style (blocks of run columns). This module
extracts *run-level* tidy records so that downstream analysis treats each
independent run as the analysis unit (per the DATA_GUIDE). Workbook "Mean"
and "S.D." columns are recomputed independently and NOT used as evidence.
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "source_tables"


def _read(pathname):
    return pd.read_csv(DATA / pathname, header=None, dtype=str, keep_default_na=False)


def _rows(df, start, stop):
    """Return (row_idx, row) for start <= idx < stop."""
    for idx in range(start, min(stop, len(df))):
        yield idx, df.iloc[idx].tolist()


def _f(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _clean(s):
    return str(s).strip()


def blockB_table01():
    """Table 01 Source block B: run-level total_fusion & fusion_first_two_frames.
    header row 4-5; data rows 6-11 (6 conditions)."""
    df = _read("source_table_01.csv")
    rows = []
    for idx in range(6, 15):
        r = df.iloc[idx].tolist()
        cond = _clean(r[0])
        if not cond:
            continue
        tf_runs = [_f(r[c]) for c in range(1, 6)]
        f2f_runs = [_f(r[c]) for c in range(10, 15)]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 1, "block": "B",
                "condition": cond, "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
    return pd.DataFrame(rows)


def blockC_table01():
    """Table 01 Source block C: per-bin fusion frequency % for 6 conditions.
    conditions start at row 17 (col offsets 1,11,21,31,41,51).
    bins rows 20-61."""
    df = _read("source_table_01.csv")
    cond_offset = [1, 11, 21, 31, 41, 51]
    cond_names = ["None", "1:200 SensorA", "1:2000 SensorB",
                  "1:800 SensorB", "1:400 SensorB", "1:200 SensorB"]
    rows = []
    for idx in range(20, len(df)):
        r = df.iloc[idx].tolist()
        binv = _clean(r[1])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        for off, cond in zip(cond_offset, cond_names):
            runs = [_f(r[off + c]) for c in range(1, 6)]
            # only include if at least one run present and not all-None
            if all(x is None for x in runs):
                continue
            for run_i, v in enumerate(runs, start=1):
                if v is None:
                    continue
                rows.append({
                    "source_table": 1, "block": "C", "condition": cond,
                    "independent_run": run_i, "bin_s": t, "fusion_frequency_pct": v,
                })
    return pd.DataFrame(rows)


def table02():
    """Table 02: survival % vs bin for None, 1:800 SensorB, 1:200 SensorB.
    3 runs each. blocks: A (rows 6-76), B (89-?), C (90-?).
    We take the first block (Source block A) rows 6..76 as the survival curves."""
    df = _read("source_table_02.csv")
    rows = []
    # block A: conditions at col offsets 1 (None), 8 (1:800), 15 (1:200)
    conds = [(1, "None"), (8, "1:800 SensorB"), (15, "1:200 SensorB")]
    for idx in range(6, 77):
        r = df.iloc[idx].tolist()
        binv = _clean(r[1])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        for off, cond in conds:
            runs = [_f(r[off + c]) for c in range(1, 4)]  # run1-3
            if all(x is None for x in runs):
                continue
            for run_i, v in enumerate(runs, start=1):
                if v is None:
                    continue
                rows.append({
                    "source_table": 2, "condition": cond, "independent_run": run_i,
                    "bin_s": t, "survival_pct": v,
                })
    return pd.DataFrame(rows)


def table03():
    """Table 03 TOP PANEL: run-level total_fusion & fusion_first_two_frames for
    None, 1:800 SensorBDA, 1:200 SensorBDA (5 runs each)."""
    df = _read("source_table_03.csv")
    rows = []
    for idx in range(5, 8):
        r = df.iloc[idx].tolist()
        cond = _clean(r[0])
        if not cond:
            continue
        tf_runs = [_f(r[c]) for c in range(1, 6)]
        f2f_runs = [_f(r[c]) for c in range(10, 15)]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 3, "panel": "TOP", "condition": cond,
                "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
    return pd.DataFrame(rows)


def table04():
    """Table 04 TOP PANEL: run-level total fusion & fusion first two frames for
    None, 1:800 SensorB, 1:200 SensorB (5 runs each)."""
    df = _read("source_table_04.csv")
    rows = []
    for idx in range(5, 8):
        r = df.iloc[idx].tolist()
        cond = _clean(r[2])
        if not cond:
            continue
        tf_runs = [_f(r[c]) for c in range(3, 8)]
        f2f_runs = [_f(r[c]) for c in range(12, 17)]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 4, "panel": "TOP", "condition": cond,
                "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
    return pd.DataFrame(rows)


def table05():
    """Table 05: SensorA dose-response for bound target subunit. 4 runs each.
    rows 11-18: SensorA Conc 0..60."""
    df = _read("source_table_05.csv")
    rows = []
    for idx in range(11, 19):
        r = df.iloc[idx].tolist()
        conc = _f(r[1])
        if conc is None:
            continue
        runs = [_f(r[c]) for c in range(2, 6)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 5, "variable": "bound_target_subunit",
                "sensorA_conc_uM": conc, "independent_run": run_i,
                "bound_value_uM": v,
            })
    return pd.DataFrame(rows)


def table07_blockA():
    """Table 07 block A: run-level total fusion & fusion first two frames for
    SensorAWT, SensorAWT+1:200 SensorB, SensorADA+1:200 SensorB.
    No Ca2+basal: 5 runs (cols 1,2,3,5,6). With 0.5uM Ca2+basal: 3 runs.
    Note: run4/run5 columns are at col5/col6 for the No-Ca2+ block (offset 4 in
    the 0-indexed run list)."""
    df = _read("source_table_07.csv")
    rows = []
    for idx in range(7, 10):
        r = df.iloc[idx].tolist()
        cond = _clean(r[0])
        if not cond:
            continue
        # No Ca2+basal: total fusion runs at cols 1,2,3,5,6 ; F2F at cols 10-14
        tf_runs = [_f(r[c]) for c in [1, 2, 3, 5, 6]]
        f2f_runs = [_f(r[c]) for c in [10, 11, 12, 13, 14]]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 7, "block": "A", "ca2_basal": "none",
                "condition": cond, "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
        # 0.5 uM Ca2+basal: total fusion runs at cols 18,19,20 ; F2F at cols 24,25,26
        tf_runs = [_f(r[c]) for c in [18, 19, 20]]
        f2f_runs = [_f(r[c]) for c in [24, 25, 26]]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 7, "block": "A", "ca2_basal": "0.5uM",
                "condition": cond, "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
    return pd.DataFrame(rows)


def table07_blockB():
    """Table 07 block B: per-bin fusion frequency for the three conditions across
    no-Ca2+ and 0.5uM Ca2+ basals.
    Condition 1: SensorAWT  (no-Ca2+ cols 1-6, with-Ca2+ cols 10-14)
    Condition 2: SensorAWT+1:200 SensorB (no-Ca2+ cols 18-22, with-Ca2+ cols 24-28)
    Condition 3: SensorADA+1:200 SensorB (no-Ca2+ cols 32-36, with-Ca2+ cols 38-42)"""
    df = _read("source_table_07.csv")
    rows = []
    conds = [
        ("SensorAWT", [1, 10]),
        ("SensorAWT + 1:200 SensorB", [18, 24]),
        ("SensorADA + 1:200 SensorB", [32, 38]),
    ]
    for idx in range(17, len(df)):
        r = df.iloc[idx].tolist()
        binv = _clean(r[0])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        for cond, (off_nc, off_c) in conds:
            for basal, off in [("none", off_nc), ("0.5uM", off_c)]:
                runs = [_f(r[off + c]) for c in range(1, 6)]
                if all(x is None for x in runs):
                    continue
                for run_i, v in enumerate(runs, start=1):
                    if v is None:
                        continue
                    rows.append({
                        "source_table": 7, "block": "B", "condition": cond,
                        "ca2_basal": basal, "independent_run": run_i,
                        "bin_s": t, "fusion_frequency_pct": v,
                    })
    return pd.DataFrame(rows)


def table08():
    """Table 08 block C: run-level total fusion for None, 1:200 SensorBdoubleTMD,
    1:200 SensorBsingleTMD (5 runs). Also per-bin fusion frequency (block C)."""
    df = _read("source_table_08.csv")
    rows = []
    for idx in range(60, 63):
        r = df.iloc[idx].tolist()
        cond = _clean(r[0])
        if not cond:
            continue
        runs = [_f(r[c]) for c in range(1, 6)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 8, "panel": "C", "condition": cond,
                "independent_run": run_i, "total_fusion_percent": v,
            })
    return pd.DataFrame(rows)


def table09():
    """Table 09: block A docking (5 runs) and block B survival (5 runs)."""
    df = _read("source_table_09.csv")
    rows = []
    for idx in range(7, 13):
        r = df.iloc[idx].tolist()
        cond = _clean(r[1])
        if not cond:
            continue
        runs = [_f(r[c]) for c in range(2, 7)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 9, "block": "A", "condition": cond,
                "independent_run": run_i, "docked_per_5um": v,
            })
    for idx in range(20, 26):
        r = df.iloc[idx].tolist()
        cond = _clean(r[1])
        if not cond:
            continue
        runs = [_f(r[c]) for c in range(2, 7)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 9, "block": "B", "condition": cond,
                "independent_run": run_i, "survival_pct": v,
            })
    return pd.DataFrame(rows)


def table10():
    """Table 10: per-bin Ca Green, Alexa 647N fluorescence, transmittance (3 runs)."""
    df = _read("source_table_10.csv")
    rows = []
    for idx in range(4, len(df)):
        r = df.iloc[idx].tolist()
        binv = _clean(r[1])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        runs_ca = [_f(r[c]) for c in range(2, 5)]
        runs_alex = [_f(r[c]) for c in range(10, 13)]
        runs_trans = [_f(r[c]) for c in range(18, 21)]
        for run_i in range(1, 4):
            v = runs_ca[run_i - 1]
            if v is not None:
                rows.append({
                    "source_table": 10, "independent_run": run_i, "bin_s": t,
                    "ca_green_mean": v,
                })
                v = runs_alex[run_i - 1]
                if v is not None:
                    rows.append({
                        "source_table": 10, "independent_run": run_i, "bin_s": t,
                        "alex_mean": v,
                    })
                v = runs_trans[run_i - 1]
                if v is not None:
                    rows.append({
                        "source_table": 10, "independent_run": run_i, "bin_s": t,
                        "transmittance_mean": v,
                    })
    return pd.DataFrame(rows)


def table11():
    """Table 11: CPX-FusionComplex dose-response for bound target subunit (4 runs).
    rows 9-17: conc 1,2,5,10,15,20,25,30,45."""
    df = _read("source_table_11.csv")
    rows = []
    for idx in range(9, 18):
        r = df.iloc[idx].tolist()
        conc = _f(r[1])
        if conc is None:
            continue
        runs = [_f(r[c]) for c in range(2, 6)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 11, "variable": "bound_target_subunit",
                "cpx_conc_uM": conc, "independent_run": run_i,
                "bound_value_uM": v,
            })
    return pd.DataFrame(rows)


def table12():
    """Table 12: time series of SensorA/SensorB (blocks B and C)."""
    df = _read("source_table_12.csv")
    rows = []
    for idx in range(4, len(df)):
        r = df.iloc[idx].tolist()
        binv = _clean(r[1])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        sa = _f(r[2]); sb = _f(r[3])
        if sa is not None:
            rows.append({"source_table": 12, "block": "B", "bin_s": t,
                         "sensorA": sa, "sensorB": sb})
        ca = _f(r[5]); sa2 = _f(r[6]); sb2 = _f(r[7])
        if ca is not None:
            rows.append({"source_table": 12, "block": "C", "bin_s": t,
                         "calcium_uM": ca, "sensorA": sa2, "sensorB": sb2})
    return pd.DataFrame(rows)


def table13():
    """Table 13: per-configuration time series (6FC/single, 12FC/single, 6FC/dual).
    Each config has a Model Output (Complete Data) and (Select Data & Temporal blurring)
    sub-block with SensorA/SensorB (Scheme 1/2). We extract the Complete Data block.
    Col layout:
      config1 complete: cols 1-4 (time, Ca, S1, S2)
      config1 blurring: cols 7-10
      config2 complete: cols 13-16
      config2 blurring: cols 19-22
      config3 complete: cols 25-28
      config3 blurring: cols 31-34
    (verify from header row 6)"""
    df = _read("source_table_13.csv")
    rows = []
    for idx in range(7, len(df)):
        r = df.iloc[idx].tolist()
        t = _f(r[1])
        if t is None:
            continue
        rows.append({
            "source_table": 13,
            "config_6FC_single": {"time": t, "Ca": _f(r[2]),
                                   "SensorA_S1": _f(r[3]), "SensorB_S1": _f(r[4]),
                                   "SensorB_S2": _f(r[5])},
            "config_12FC_single": {"time": t, "Ca": _f(r[14]),
                                    "SensorA_S1": _f(r[15]), "SensorB_S1": _f(r[16]),
                                    "SensorB_S2": _f(r[17])},
            "config_6FC_dual": {"time": t, "Ca": _f(r[26]),
                                 "SensorA_S1": _f(r[27]), "SensorB_S1": _f(r[28]),
                                 "SensorB_S2": _f(r[29])},
        })
    return pd.DataFrame(rows)


def table14():
    """Table 14: block A docking (4 runs), block B survival (4 runs), block C
    total fusion & fusion first two frames (4 runs) for none/pre-docking/post-docking."""
    df = _read("source_table_14.csv")
    rows = []
    for idx in range(7, 10):
        r = df.iloc[idx].tolist()
        cond = _clean(r[1])
        if not cond:
            continue
        runs = [_f(r[c]) for c in range(2, 6)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 14, "block": "A", "condition": cond,
                "independent_run": run_i, "docked_per_5um": v,
            })
    for idx in range(17, 20):
        r = df.iloc[idx].tolist()
        cond = _clean(r[1])
        if not cond:
            continue
        runs = [_f(r[c]) for c in range(2, 6)]
        for run_i, v in enumerate(runs, start=1):
            if v is None:
                continue
            rows.append({
                "source_table": 14, "block": "B", "condition": cond,
                "independent_run": run_i, "survival_pct": v,
            })
    for idx in range(27, 30):
        r = df.iloc[idx].tolist()
        cond = _clean(r[0])
        if not cond:
            continue
        tf_runs = [_f(r[c]) for c in range(1, 5)]
        f2f_runs = [_f(r[c]) for c in range(9, 13)]
        for run_i, (tf, f2f) in enumerate(zip(tf_runs, f2f_runs), start=1):
            if tf is None:
                continue
            rows.append({
                "source_table": 14, "block": "C", "condition": cond,
                "independent_run": run_i,
                "total_fusion_percent": tf,
                "fusion_in_first_two_frames_percent": f2f,
            })
    return pd.DataFrame(rows)


def table15():
    """Table 15: per-bin normalized fluorescence intensity for Lipids and
    TargetComplexs (None and SensorB 1:200)."""
    df = _read("source_table_15.csv")
    rows = []
    for idx in range(4, len(df)):
        r = df.iloc[idx].tolist()
        binv = _clean(r[1])
        if not binv or not binv.replace(".", "").isdigit():
            continue
        t = _f(binv)
        if t is None:
            continue
        rows.append({
            "source_table": 15, "bin_s": t,
            "lipid_none": _f(r[2]), "lipid_sensorB": _f(r[3]),
            "target_none": _f(r[5]), "target_sensorB": _f(r[6]),
        })
    return pd.DataFrame(rows)


def derived_fusion_dose():
    """Derived navigation file: fusion dose independent runs."""
    return pd.read_csv(DATA.parent / "derived_for_navigation" /
                       "fusion_dose_independent_runs.csv")


def derived_binding_competition():
    """Derived navigation file: binding competition runs."""
    return pd.read_csv(DATA.parent / "derived_for_navigation" /
                       "binding_competition_runs.csv")


def load_all():
    """Load every run-level dataset."""
    return {
        "table01_blockB": blockB_table01(),
        "table01_blockC": blockC_table01(),
        "table02": table02(),
        "table03": table03(),
        "table04": table04(),
        "table05": table05(),
        "table07_blockA": table07_blockA(),
        "table07_blockB": table07_blockB(),
        "table08": table08(),
        "table09": table09(),
        "table10": table10(),
        "table11": table11(),
        "table12": table12(),
        "table13": table13(),
        "table14": table14(),
        "table15": table15(),
        "derived_fusion_dose": derived_fusion_dose(),
        "derived_binding_competition": derived_binding_competition(),
    }


if __name__ == "__main__":
    datasets = load_all()
    for name, df in datasets.items():
        print(f"{name}: shape={df.shape}")
        print(df.head(3).to_string())
        print()
